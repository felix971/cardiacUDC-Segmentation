"""CPU-only checks that migration cannot silently change a frozen experiment."""

import copy
import hashlib
import importlib
import json
import platform
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_neighbor_ab_pair as runner
from nnunet_neighbor_ab_trainer import AUGMENTATION

try:
    continuation = importlib.import_module("prepare_neighbor_continuation")
except ModuleNotFoundError:
    continuation = None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rng(cuda=False):
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch_cpu": torch.get_rng_state()}
    if cuda:
        result["torch_cuda"] = [torch.get_rng_state()]
    return result


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "old" / "seed_2026"
        self.destination = self.root / "new" / "seed_2026"
        self.fold = self.source / runner.DATASET / f"{runner.TRAINERS['A']}__nnUNetPlans__2d/fold_0"
        self.fold.mkdir(parents=True)
        self.checkpoint_path = self.fold / "checkpoint_latest.pth"
        self.config = {
            "schema_version": 1, "seed": 2026, "mode": "A", "target_index": 4,
            "frame_offsets": list(range(-4, 4)), "patch_size": [768, 512], "batch_size": 2,
            "epochs": 500, "updates_per_epoch": 250, "checkpoint_every_epochs": 10,
            "validation_batches_per_epoch": 50, "train_stream_seed": 2027,
            "val_stream_seed": 2028, "compile": False, "deterministic_algorithms": True,
            "initial_state_sha256": "a" * 64, "initial_model_rng_sha256": "b" * 64,
            "optimizer": {"name": "SGD", "lr": 0.01, "momentum": 0.99,
                          "nesterov": True, "weight_decay": 3e-5},
            "scheduler": "polynomial", "augmentation": copy.deepcopy(AUGMENTATION),
            "cross_entropy": "log_softmax_gather_masked_mean", "ignore_label": None,
            "oversample_foreground_percent": 0.33, "probabilistic_oversampling": False,
            "inference_mirroring_axes": None, "cudnn_deterministic": True,
            "cudnn_benchmark": False, "cuda_matmul_tf32": False, "cudnn_tf32": False,
            "cublas_workspace_config": ":4096:8",
        }
        streams = {}
        for name, count in (("train", 2500), ("val", 500)):
            digest = hashlib.sha256(b"").hexdigest()
            records = []
            for index in range(8):
                record = {"batch": index, "keys": ["case"]}
                encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                digest = hashlib.sha256(digest.encode() + encoded).hexdigest()
                record["cumulative_sha256"] = digest
                records.append(json.dumps(record, sort_keys=True) + "\n")
            (self.fold / f"neighbor_ab_{name}_audit.jsonl").write_text("".join(records))
            streams[f"{name}_stream"] = {"rng_state": rng(), "batches_seen": count,
                                          "audit_sha256": "c" * 64}
        streams["model_rng"] = rng(cuda=True)
        self.checkpoint = {
            "current_epoch": 10, "trainer_name": runner.TRAINERS["A"],
            "network_weights": {"weight": torch.tensor([1.0])},
            "optimizer_state": {"state": {}, "param_groups": [{"params": [0]}]},
            "lr_scheduler_state": {"current_step": 10}, "grad_scaler_state": {},
            "logging": {"epoch_start_timestamps": list(range(10))}, "init_args": {},
            "_best_ema": 0.5, "inference_allowed_mirroring_axes": None,
            "neighbor_ab": self.config, "neighbor_ab_state": streams,
        }
        self.status = {"host_label": "ubuntu", "seed": 2026, "order": ["A", "B"],
                       "code_sha256": "d" * 64, "data_sha256": "e" * 64,
                       "split_sha256": "f" * 64, "runs": {}, "status": "running"}
        (self.source / "pair_status.json").write_text(json.dumps(self.status))
        manifest = dict(self.config, hostname="original", source_sha256="1" * 64)
        (self.fold / "neighbor_ab_run.json").write_text(json.dumps(manifest))
        (self.source / "console_A.log").write_text("original log\n")
        self.report_path = self.root / "equivalence_report.json"
        self.write_checkpoint_and_report()

    def write_checkpoint_and_report(self):
        torch.save(self.checkpoint, self.checkpoint_path)
        digest = sha(self.checkpoint_path)
        fullsize_cases = []
        for mode in ("A", "B"):
            for phase in ("train", "val"):
                case = {"mode": mode, "phase": phase, "batches": 8,
                        "batch_shape": [2, 8, 768, 512], "target_shape": [2, 1, 768, 512]}
                for key in ("data_equal", "target_equal", "metadata_equal", "audit_equal",
                            "stream_rng_equal", "model_rng_equal"):
                    case[key] = True
                fullsize_cases.append(case)
        samples = []
        for replicate in range(2):
            order = [("serial", 1.0), ("prefetch", 0.8)]
            if replicate % 2:
                order.reverse()
            for implementation, seconds in order:
                samples.append({"implementation": implementation, "replicate": replicate,
                                "seconds_per_step": [seconds] * 20, "warmup_steps": 2,
                                "peak_allocated_bytes": 100, "peak_reserved_bytes": 200})
        cuda = {"passed": True, "checkpoint_sha256": digest}
        for key in ("updates_equal", "optimizer_equal", "scaler_equal", "weights_equal",
                    "buffers_equal", "model_rng_equal", "stream_rng_equal",
                    "validation_transition", "mode_b_smoke"):
            cuda[key] = True
        self.report = {
            "schema_version": 1, "passed": True, "migration_eligible": True,
            "source_checkpoint": {"path": str(self.checkpoint_path), "sha256": digest,
                                  "seed": 2026, "mode": "A", "current_epoch": 10},
            "baseline_code_sha256": self.status["code_sha256"],
            "current_code_sha256": runner.code_fingerprint(),
            "data_sha256": self.status["data_sha256"], "split_sha256": self.status["split_sha256"],
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": "cuda", "cpu_threads": 8},
            "cpu": {"passed": True, "fullsize_cases": fullsize_cases, "queued_resume": True,
                    "serial_to_prefetch_resume": True, "model_rng_equal": True},
            "cuda": cuda,
            "timing": {"samples": samples, "gain_fraction": 0.25,
                       "minimum_gain_fraction": 0.05, "passed": True},
        }
        self.write_report()

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report))

    def prepare(self):
        self.assertIsNotNone(continuation, "Continuation preparation is not implemented")
        return continuation.prepare_continuation(
            self.source, self.destination, self.checkpoint_path, self.report_path)

    def live_status(self):
        return dict(self.status, code_sha256=runner.code_fingerprint())

    def test_copies_checkpoint_without_rewriting_original_or_old_status(self):
        originals = {path.relative_to(self.source): path.read_bytes()
                     for path in self.source.rglob("*") if path.is_file()}
        record = self.prepare()
        self.assertFalse((self.destination / "pair_status.json").exists())
        self.assertEqual(sha(self.destination / record["checkpoint_relative_path"]), sha(self.checkpoint_path))
        self.assertEqual(record["completed_train_samples"], 5000)
        self.assertEqual(record["completed_val_samples"], 1000)
        for relative, content in originals.items():
            self.assertEqual((self.source / relative).read_bytes(), content)
        evidence = self.destination / "continuation_evidence"
        self.assertEqual((evidence / "source" / "pair_status.json").read_bytes(), originals[Path("pair_status.json")])
        continuation.verify_continuation(self.destination, self.live_status())

    def test_existing_destination_is_rejected_without_modification(self):
        self.destination.mkdir(parents=True)
        marker = self.destination / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(marker.read_text(), "keep")

    def test_wrong_seed_mode_epoch_counts_or_partial_checkpoint_are_rejected(self):
        cases = [("seed", 2027), ("mode", "B"), ("epoch", 11),
                 ("train_count", 2499), ("val_count", 501), ("partial", None)]
        original = copy.deepcopy(self.checkpoint)
        for field, value in cases:
            with self.subTest(field=field):
                self.checkpoint = copy.deepcopy(original)
                if field in ("seed", "mode"):
                    self.checkpoint["neighbor_ab"][field] = value
                elif field == "epoch":
                    self.checkpoint["current_epoch"] = value
                elif field.endswith("count"):
                    self.checkpoint["neighbor_ab_state"][field.split("_")[0] + "_stream"]["batches_seen"] = value
                else:
                    del self.checkpoint["optimizer_state"]
                self.write_checkpoint_and_report()
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.destination.exists())

    def test_config_manifest_mismatch_and_incomplete_audits_are_rejected(self):
        manifest_path = self.fold / "neighbor_ab_run.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["batch_size"] = 3
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.prepare()
        manifest["batch_size"] = 2
        manifest_path.write_text(json.dumps(manifest))
        (self.fold / "neighbor_ab_train_audit.jsonl").write_text("[]\n")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_wrong_proof_hash_code_data_and_cpu_only_certificate_are_rejected(self):
        original = copy.deepcopy(self.report)
        for field in ("checkpoint", "baseline_code_sha256", "current_code_sha256",
                      "data_sha256", "split_sha256", "cpu_only"):
            with self.subTest(field=field):
                self.report = copy.deepcopy(original)
                if field == "checkpoint":
                    self.report["source_checkpoint"]["sha256"] = "0" * 64
                elif field == "cpu_only":
                    self.report["cuda"]["passed"] = False
                    self.report["migration_eligible"] = False
                else:
                    self.report[field] = "0" * 64
                self.write_report()
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.destination.exists())

    def test_runner_rejects_wrong_live_seed_code_data_and_evidence_tampering(self):
        self.prepare()
        for key, wrong in (("seed", 2027), ("code_sha256", "0" * 64),
                           ("data_sha256", "0" * 64), ("split_sha256", "0" * 64)):
            with self.subTest(key=key):
                status = self.live_status()
                status[key] = wrong
                with self.assertRaises(ValueError):
                    continuation.verify_continuation(self.destination, status)
        evidence_log = self.destination / "continuation_evidence/source/console_A.log"
        self.assertEqual(evidence_log.stat().st_mode & 0o222, 0)
        evidence_log.chmod(0o644)
        evidence_log.write_text("changed")
        with self.assertRaises(ValueError):
            continuation.verify_continuation(self.destination, self.live_status())

    def test_first_resume_checks_exact_checkpoint_but_later_resume_allows_progress(self):
        record = self.prepare()
        target = self.destination / record["checkpoint_relative_path"]
        target.write_bytes(b"wrong checkpoint")
        with self.assertRaises(ValueError):
            continuation.verify_continuation(self.destination, self.live_status())
        progressed = copy.deepcopy(self.checkpoint)
        progressed['current_epoch'] = 20
        progressed['neighbor_ab_state']['train_stream']['batches_seen'] = 5000
        progressed['neighbor_ab_state']['val_stream']['batches_seen'] = 1000
        torch.save(progressed, target)
        status = dict(self.live_status(), continuation_sha256=sha(self.destination / "continuation.json"))
        (self.destination / "pair_status.json").write_text(json.dumps(status))
        continuation.verify_continuation(self.destination, self.live_status())

    def test_existing_pair_status_must_bind_the_immutable_continuation_record(self):
        self.prepare()
        status_path = self.destination / "pair_status.json"
        status_path.write_text(json.dumps(self.live_status()))
        with self.assertRaisesRegex(ValueError, "continuation record"):
            continuation.verify_continuation(self.destination, self.live_status())
        status = dict(self.live_status(), continuation_sha256="0" * 64)
        status_path.write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError, "continuation record"):
            continuation.verify_continuation(self.destination, self.live_status())

    def test_fresh_output_needs_no_continuation_and_prefetch_choice_is_explicit(self):
        self.assertIsNotNone(continuation, "Continuation preparation is not implemented")
        self.assertIsNone(continuation.verify_continuation(self.root / "fresh", self.live_status()))
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(runner.execution_configuration()["prefetch"], 0)
        for value in ("0", "2"):
            with patch.dict("os.environ", {"NEIGHBOR_AB_PREFETCH": value}):
                self.assertEqual(runner.execution_configuration()["prefetch"], int(value))
        with patch.dict("os.environ", {"NEIGHBOR_AB_PREFETCH": "1"}):
            with self.assertRaises(ValueError):
                runner.execution_configuration()

    def test_incomplete_semantic_configuration_is_rejected(self):
        del self.checkpoint["neighbor_ab"]["optimizer"]
        self.write_checkpoint_and_report()
        with self.assertRaises(ValueError):
            self.prepare()

    def test_runner_rejects_broken_continuation_before_any_cuda_probe(self):
        data = self.root / "data"
        dataset = data / "preprocessed" / runner.DATASET
        dataset.mkdir(parents=True)
        split = {"train": [f"train_{i}" for i in range(970)],
                 "val": [f"val_{i}" for i in range(218)]}
        (dataset / "splits_final.json").write_text(json.dumps([split]))
        (self.destination / "continuation_evidence").mkdir(parents=True)
        argv = ["run_neighbor_ab_pair.py", "--nnunet-root", str(data),
                "--output-root", str(self.destination.parent), "--seed", "2026",
                "--host-label", "ubuntu", "--order", "A", "B"]
        with patch.object(sys, "argv", argv), patch.object(runner, "verify_registered_modules"), \
                patch.object(runner, "data_fingerprint", return_value="e" * 64), \
                patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA was probed")):
            with self.assertRaisesRegex(ValueError, "without its continuation record"):
                runner.main()


if __name__ == "__main__":
    unittest.main()
