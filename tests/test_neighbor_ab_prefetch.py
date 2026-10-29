"""CPU-only spawn, checkpoint, and lifecycle tests for paired prefetch."""

import copy
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


class RandomBatches:
    """Exercise all three global CPU RNGs and the real stream audit builder."""

    def __iter__(self):
        return self

    def __next__(self):
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU producer initialized CUDA")
        data = torch.rand(2, 8, 8, 8) + np.random.rand() + random.random()
        return {
            "data": data,
            "target": torch.randint(0, 5, (2, 1, 8, 8), dtype=torch.int16),
            "keys": np.random.choice(["case_a", "case_b", "case_c"], 2),
            "augmentation": [{"gain": float(np.random.uniform(0.75, 1.25))}],
            "crop_bboxes": [{"lower": [0, 1], "upper": [8, 9]}],
        }


class FailingBatches(RandomBatches):
    def __next__(self):
        raise ValueError("intentional producer failure")


class ExitingBatches(RandomBatches):
    def __next__(self):
        os._exit(23)


class HangingBatches(RandomBatches):
    def __next__(self):
        time.sleep(120)
        return super().__next__()


class LargeBatches(RandomBatches):
    def __next__(self):
        batch = super().__next__()
        batch["data"] = batch["data"].repeat(1, 1, 64, 64)
        return batch


class CudaFlagBatches(RandomBatches):
    def __next__(self):
        batch = super().__next__()
        # Simulate a transform reporting CUDA initialization without using a GPU.
        torch.cuda.is_initialized = lambda: True
        return batch


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.ab = importlib.import_module("nnunet_neighbor_ab_trainer")
        self.streams = []

    def tearDown(self):
        for stream in self.streams:
            stream.close()

    def prefetch(self, loader=None, seed=2027, audit_path=None):
        self.assertTrue(hasattr(self.ab, "PairedPrefetchAugmenter"),
                        "The deterministic spawned producer has not been implemented")
        stream = self.ab.PairedPrefetchAugmenter(
            loader or RandomBatches(), seed, audit_path, prefetch=2,
        )
        self.streams.append(stream)
        return stream

    def wait_pending(self, stream, count=2):
        deadline = time.monotonic() + 40
        while stream.pending_batches < count and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(stream.pending_batches, count)

    def assert_batch_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for key in first:
            if isinstance(first[key], torch.Tensor):
                self.assertTrue(torch.equal(first[key], second[key]), key)
            elif isinstance(first[key], np.ndarray):
                np.testing.assert_array_equal(first[key], second[key])
            else:
                self.assertEqual(first[key], second[key], key)

    def assert_state_equal(self, first, second):
        self.assertEqual(set(first), {"rng_state", "batches_seen", "audit_sha256"})
        self.assertEqual(first["batches_seen"], second["batches_seen"])
        self.assertEqual(first["audit_sha256"], second["audit_sha256"])
        self.assertEqual(self.ab.rng_state_sha256(first["rng_state"]),
                         self.ab.rng_state_sha256(second["rng_state"]))

    def test_bytes_audits_and_parent_rng_match_serial_with_owned_batches(self):
        # Early audit commits, parent RNG draws, or mutable IPC tensors break this.
        with tempfile.TemporaryDirectory() as tmp:
            serial_path = Path(tmp) / "serial.jsonl"
            queued_path = Path(tmp) / "queued.jsonl"
            serial = self.ab.PairedSingleThreadedAugmenter(RandomBatches(), 2027, serial_path)
            outside = self.ab.capture_rng_state(include_cuda=False)
            stream = self.prefetch(audit_path=queued_path)
            first = None
            for index in range(10):
                expected = next(serial)
                actual = next(stream)
                self.assert_batch_equal(expected, actual)
                self.assert_state_equal(serial.state_dict(), stream.state_dict())
                if index == 0:
                    first = actual
                    first_copy = copy.deepcopy(actual)
                self.assertEqual(serial_path.read_bytes(), queued_path.read_bytes())
            self.wait_pending(stream)
            self.assert_batch_equal(first, first_copy)
            self.assertEqual(len(queued_path.read_text().splitlines()), 8)
            self.assertEqual(self.ab.rng_state_sha256(outside),
                             self.ab.rng_state_sha256(self.ab.capture_rng_state(False)))
            self.assertFalse(torch.cuda.is_initialized())

    def test_pending_train_val_queues_restore_at_consumed_state(self):
        # Saving producer-ahead state skips data after checkpoint restoration.
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "train.jsonl"
            train = self.prefetch(seed=2027, audit_path=audit_path)
            val = self.prefetch(seed=2028)
            next(train)
            next(val)
            self.wait_pending(train)
            self.wait_pending(val)
            checkpoint = pickle.loads(pickle.dumps({
                "train": train.state_dict(), "val": val.state_dict(),
            }))
            self.assertEqual(checkpoint["train"]["batches_seen"], 1)
            self.assertEqual(len(audit_path.read_text().splitlines()), 1)
            expected_train = next(train)
            expected_val = next(val)
            expected_train_state = train.state_dict()
            outside = self.ab.capture_rng_state(False)
            train.load_state_dict(checkpoint["train"])
            val.load_state_dict(checkpoint["val"])
            self.assertEqual(len(audit_path.read_text().splitlines()), 1)
            self.assert_batch_equal(expected_val, next(val))
            self.assert_batch_equal(expected_train, next(train))
            self.assert_state_equal(expected_train_state, train.state_dict())
            self.assertEqual(self.ab.rng_state_sha256(outside),
                             self.ab.rng_state_sha256(self.ab.capture_rng_state(False)))
            serial = self.ab.PairedSingleThreadedAugmenter(RandomBatches(), 0)
            serial.load_state_dict(train.state_dict())
            self.assert_batch_equal(next(serial), next(train))

    def test_producer_exception_and_premature_death_fail_promptly(self):
        for loader, message in [(FailingBatches(), "intentional producer failure"),
                                (ExitingBatches(), "23")]:
            with self.subTest(loader=type(loader).__name__):
                stream = self.prefetch(loader)
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, message):
                    next(stream)
                self.assertLess(time.monotonic() - started, 40)
                process = stream._process
                stream.close()
                self.assertFalse(process.is_alive())

    def test_producer_rejects_a_loader_that_initializes_cuda(self):
        stream = self.prefetch(CudaFlagBatches())
        with self.assertRaisesRegex(RuntimeError, "initialized CUDA"):
            next(stream)
        self.assertFalse(torch.cuda.is_initialized())

    def test_shutdown_with_full_queue_is_bounded_and_idempotent(self):
        stream = self.prefetch()
        next(stream)
        self.wait_pending(stream)
        process = stream._process
        started = time.monotonic()
        stream.close()
        stream.close()
        stream._finish()
        self.assertLess(time.monotonic() - started, 5)
        self.assertFalse(process.is_alive())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            next(stream)

    def test_shutdown_terminates_a_stuck_loader(self):
        stream = self.prefetch(HangingBatches())
        stream._start()
        process = stream._process
        started = time.monotonic()
        stream.close()
        self.assertLess(time.monotonic() - started, 5)
        self.assertFalse(process.is_alive())

    def test_death_during_large_payload_transfer_does_not_hang_consumer(self):
        # Queue.get(timeout=...) alone cannot time out a partly written payload.
        stream = self.prefetch(LargeBatches())
        next(stream)
        self.wait_pending(stream)
        stream._process.kill()
        stream._process.join(timeout=2)
        finished = threading.Event()
        failures = []

        def receive():
            try:
                next(stream)
            except RuntimeError as error:
                failures.append(error)
            finally:
                finished.set()

        thread = threading.Thread(target=receive, daemon=True)
        thread.start()
        self.assertTrue(finished.wait(timeout=5), "Dead producer left a blocked payload read")
        self.assertEqual(len(failures), 1)

    def test_mode_selection_rejects_unsupported_queue_sizes(self):
        self.assertTrue(hasattr(self.ab, "neighbor_prefetch_depth"))
        with patch.dict(os.environ):
            os.environ.pop("NEIGHBOR_AB_PREFETCH", None)
            self.assertEqual(self.ab.neighbor_prefetch_depth(), 0)
        for value, expected in [("0", 0), ("2", 2)]:
            with patch.dict(os.environ, {"NEIGHBOR_AB_PREFETCH": value}):
                self.assertEqual(self.ab.neighbor_prefetch_depth(), expected)
        for value in ("1", "-1", "3", "yes", "2.0", ""):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"NEIGHBOR_AB_PREFETCH": value}):
                    with self.assertRaisesRegex(ValueError, "NEIGHBOR_AB_PREFETCH"):
                        self.ab.neighbor_prefetch_depth()

    def test_trainer_closes_both_streams_when_training_raises(self):
        # An exception before on_train_end must not leave CPU producers alive.
        trainer = self.ab.nnUNetTrainerNeighborA_500epochs.__new__(
            self.ab.nnUNetTrainerNeighborA_500epochs,
        )
        trainer.dataloader_train = self.prefetch(seed=2027)
        trainer.dataloader_val = self.prefetch(seed=2028)
        next(trainer.dataloader_train)
        next(trainer.dataloader_val)
        processes = [trainer.dataloader_train._process, trainer.dataloader_val._process]
        with patch.object(self.ab.nnUNetTrainerNoDeepSupervision, "run_training",
                          side_effect=ValueError("training failed")):
            with self.assertRaisesRegex(ValueError, "training failed"):
                trainer.run_training()
        for process in processes:
            self.assertFalse(process.is_alive())

    def test_real_trainer_selects_prefetch_and_closes_on_normal_training_end(self):
        from test_neighbor_ab_trainer import NeighborABTests

        ab = self.ab

        class TinyTrainer(ab.nnUNetTrainerNeighborA_500epochs):
            @classmethod
            def build_network_architecture(cls, plans_manager, configuration_manager,
                                           num_input_channels, num_output_channels,
                                           enable_deep_supervision=True):
                return ab.TargetFrameVNet(torch.nn.Conv3d(1, 5, 1), cls.neighbor_mode)

        with tempfile.TemporaryDirectory() as tmp:
            env = {"nnUNet_results": str(Path(tmp) / "results"),
                   "nnUNet_preprocessed": str(Path(tmp) / "preprocessed"),
                   "NEIGHBOR_AB_SEED": "2026", "NEIGHBOR_AB_PREFETCH": "2",
                   "nnUNet_compile": "false"}
            with patch.dict(os.environ, env):
                plans, dataset_json = NeighborABTests.make_dataset(tmp)
                trainer = TinyTrainer(plans, "2d", 0, dataset_json, torch.device("cpu"))
                trainer.on_train_start()
                self.streams.extend([trainer.dataloader_train, trainer.dataloader_val])
                self.assertIsInstance(trainer.dataloader_train, ab.PairedPrefetchAugmenter)
                self.assertIsInstance(trainer.dataloader_val, ab.PairedPrefetchAugmenter)
                next(trainer.dataloader_train)
                next(trainer.dataloader_val)
                processes = [trainer.dataloader_train._process, trainer.dataloader_val._process]
                execution = json.loads((Path(trainer.output_folder) / "neighbor_ab_execution.json").read_text())
                self.assertEqual(execution["prefetch_batches"], 2)
                checkpoint = Path(tmp) / "checkpoint.pth"
                trainer.save_checkpoint(checkpoint)
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                self.assertEqual(saved["neighbor_ab"], trainer.experiment_config())
                self.assertNotIn("prefetch_batches", saved["neighbor_ab"])
                self.assertEqual(saved["neighbor_ab_state"]["train_stream"]["batches_seen"], 1)
                with patch.object(ab.nnUNetTrainerNoDeepSupervision, "on_train_end"):
                    trainer.on_train_end()
                for process in processes:
                    self.assertFalse(process.is_alive())


if __name__ == "__main__":
    unittest.main()
