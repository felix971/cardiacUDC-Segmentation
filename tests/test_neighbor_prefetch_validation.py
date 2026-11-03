"""Fail-closed evidence gates for the bounded prefetch experiment."""

import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def valid_report():
    digest = "a" * 64
    cases = []
    for mode in ("A", "B"):
        for phase in ("train", "val"):
            cases.append({"mode": mode, "phase": phase, "batches": 8,
                          "batch_shape": [2, 8, 768, 512], "target_shape": [2, 1, 768, 512],
                          "data_equal": True, "target_equal": True, "metadata_equal": True,
                          "audit_equal": True, "stream_rng_equal": True, "model_rng_equal": True})
    samples = []
    for replicate in range(2):
        order = [("serial", 1.0), ("prefetch", 0.9)]
        if replicate % 2:
            order.reverse()
        for implementation, seconds in order:
            samples.append({"implementation": implementation, "replicate": replicate,
                            "seconds_per_step": [seconds] * 20, "warmup_steps": 3,
                            "peak_allocated_bytes": 10, "peak_reserved_bytes": 20})
    return {
        "schema_version": 1, "passed": True, "migration_eligible": True,
        "source_checkpoint": {"path": "/immutable/checkpoint.pth", "sha256": digest,
                              "seed": 2026, "mode": "A", "current_epoch": 10},
        "baseline_code_sha256": digest, "current_code_sha256": digest,
        "data_sha256": digest, "split_sha256": digest,
        "runtime": {"device": "cuda", "python": "3.12", "torch": "2", "cuda": "12", "cpu_threads": 8},
        "cpu": {"passed": True, "fullsize_cases": cases, "queued_resume": True,
                "serial_to_prefetch_resume": True, "model_rng_equal": True},
        "cuda": {"passed": True, "checkpoint_sha256": digest, "updates_equal": True,
                 "optimizer_equal": True, "scaler_equal": True, "weights_equal": True,
                 "buffers_equal": True, "model_rng_equal": True, "stream_rng_equal": True,
                 "validation_transition": True, "mode_b_smoke": True},
        "timing": {"samples": samples, "gain_fraction": 1 / 0.9 - 1,
                   "minimum_gain_fraction": 0.05, "passed": True},
    }


class ValidationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module_path = SCRIPTS / "validate_neighbor_prefetch.py"

    def validator(self):
        self.assertTrue(self.module_path.is_file(), "The equivalence report gate is missing")
        import validate_neighbor_prefetch
        return validate_neighbor_prefetch

    def test_complete_bound_report_is_accepted(self):
        report = valid_report()
        self.assertIs(self.validator().validate_equivalence_report(report), report)

    def test_missing_or_false_subproof_never_passes(self):
        module = self.validator()
        paths = [("cpu", "queued_resume"), ("cpu", "serial_to_prefetch_resume"),
                 ("cuda", "optimizer_equal"), ("cuda", "model_rng_equal"),
                 ("cuda", "buffers_equal"), ("cuda", "mode_b_smoke"),
                 ("cuda", "validation_transition")]
        for section, field in paths:
            for value in (None, False, "true"):
                with self.subTest(field=field, value=value):
                    report = valid_report()
                    report[section][field] = value
                    with self.assertRaises(ValueError):
                        module.validate_equivalence_report(report)

    def test_proof_for_another_checkpoint_is_rejected(self):
        report = valid_report()
        report["cuda"]["checkpoint_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.validator().validate_equivalence_report(report)

    def test_full_sized_both_modes_both_streams_are_required(self):
        module = self.validator()
        for mutation in ("missing", "duplicate", "too_few", "metadata", "small_patch"):
            report = valid_report()
            cases = report["cpu"]["fullsize_cases"]
            if mutation == "missing":
                cases.pop()
            elif mutation == "duplicate":
                cases[-1] = copy.deepcopy(cases[0])
            elif mutation == "too_few":
                cases[0]["batches"] = 7
            elif mutation == "metadata":
                cases[0]["metadata_equal"] = False
            else:
                cases[0]["batch_shape"] = [2, 8, 16, 16]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                module.validate_equivalence_report(report)

    def test_short_nonfinite_or_unwarmed_timings_are_rejected(self):
        module = self.validator()
        for bad in ([], [1.0] * 19, [math.nan] * 20, [math.inf] * 20, [0.0] * 20, [-1.0] * 20):
            report = valid_report()
            report["timing"]["samples"][0]["seconds_per_step"] = bad
            with self.subTest(bad=bad[:1]), self.assertRaises(ValueError):
                module.validate_equivalence_report(report)
        report = valid_report()
        report["timing"]["samples"][0]["warmup_steps"] = 0
        with self.assertRaises(ValueError):
            module.validate_equivalence_report(report)

    def test_timing_cannot_hide_one_slow_replicate_or_forge_gain(self):
        module = self.validator()
        report = valid_report()
        report["timing"]["samples"][2]["seconds_per_step"] = [1.1] * 20
        with self.assertRaises(ValueError):
            module.validate_equivalence_report(report)
        report = valid_report()
        report["timing"]["gain_fraction"] = 0.99
        with self.assertRaises(ValueError):
            module.validate_equivalence_report(report)

    def test_timing_requires_interleaved_complete_replicates(self):
        module = self.validator()
        for order in ([0, 3, 1, 2], [0, 1], [0, 1, 0, 1]):
            report = valid_report()
            samples = report["timing"]["samples"]
            report["timing"]["samples"] = [samples[index] for index in order]
            with self.subTest(order=order), self.assertRaises(ValueError):
                module.validate_equivalence_report(report)

    def test_cpu_proof_cannot_authorize_migration(self):
        report = valid_report()
        report["runtime"]["device"] = "cpu"
        report["cuda"] = None
        report["timing"] = None
        report["passed"] = False
        report["migration_eligible"] = False
        module = self.validator()
        self.assertIs(module.validate_equivalence_report(report, require_cuda=False), report)
        with self.assertRaises(ValueError):
            module.validate_equivalence_report(report)

    def test_changed_optimizer_or_queued_restore_is_rejected_by_cuda_comparison(self):
        module = self.validator()
        case = {"initial": {"weights": "same"}, "final": {"optimizer": "same"},
                "updates": [{"loss": "same"}], "validation": {"loss": "same"},
                "queued_checkpoint_restore_equal": True}
        module.compare_cuda_cases(case, copy.deepcopy(case))
        changed = copy.deepcopy(case)
        changed["final"]["optimizer"] = "different"
        with self.assertRaises(ValueError):
            module.compare_cuda_cases(case, changed)
        changed = copy.deepcopy(case)
        changed["queued_checkpoint_restore_equal"] = False
        with self.assertRaises(ValueError):
            module.compare_cuda_cases(case, changed)

    def test_checkpoint_with_wrong_data_binding_fails_before_execution(self):
        import json
        module = self.validator()
        self.assertTrue(hasattr(module, "verify_source_binding"), "Checkpoint provenance binding is missing")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pair_status.json").write_text(json.dumps({"seed": 2026, "code_sha256": "a" * 64,
                                                              "data_sha256": "b" * 64, "split_sha256": "a" * 64}))
            fold = root / "fold"
            fold.mkdir()
            checkpoint_path = fold / "checkpoint.pth"
            checkpoint_path.write_bytes(b"checkpoint")
            config = {"seed": 2026, "mode": "A", "updates_per_epoch": 250,
                      "validation_batches_per_epoch": 50}
            (fold / "neighbor_ab_run.json").write_text(json.dumps(config))
            checkpoint = {"neighbor_ab": config, "current_epoch": 10,
                          "neighbor_ab_state": {"train_stream": {"batches_seen": 2500},
                                                "val_stream": {"batches_seen": 500}}}
            report = valid_report()
            with self.assertRaisesRegex(ValueError, "data"):
                module.verify_source_binding(checkpoint_path, checkpoint, report)

    def test_output_cannot_be_created_inside_separately_stored_formal_results(self):
        module = self.validator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal = root / "formal_seed_2026"
            formal.mkdir()
            (formal / "pair_status.json").write_text("{}")
            checkpoint = formal / "checkpoint.pth"
            checkpoint.write_bytes(b"frozen")
            arguments = ["--nnunet-root", str(root / "data"), "--baseline-root", str(root / "code"),
                         "--checkpoint", str(checkpoint), "--output-root", str(formal / "probe")]
            with self.assertRaisesRegex(ValueError, "formal"):
                module.parse_args(arguments)


class RandomLoader:
    def __iter__(self):
        return self

    def __next__(self):
        import numpy as np
        import torch
        return {"data": torch.rand(2, 8, 4, 4) + np.random.rand(),
                "target": torch.randint(0, 5, (2, 1, 4, 4)),
                "keys": np.random.choice(["one", "two", "three"], 2),
                "augmentation": [{"gain": float(np.random.rand())}],
                "crop_bboxes": [{"lower": [0, 0], "upper": [4, 4]}]}


@unittest.skipUnless(importlib.util.find_spec("torch"), "The locked candidate environment is required")
class StreamProofTests(unittest.TestCase):
    def test_real_loader_factories_preserve_parent_rng_across_restore(self):
        import torch
        import nnunet_neighbor_ab_trainer as ab
        from test_neighbor_ab_trainer import NeighborABTests
        import validate_neighbor_prefetch as validation
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plans, dataset = NeighborABTests.make_dataset(tmp)
            # Persisted nnUNetPlans.json lacks this CLI-only constructor input.
            plans.pop("continue_training")
            with patch.dict("os.environ"):
                serial = validation.streams_factory(ab, plans, dataset, "B", root, root / "serial", 2026, 0)
                prefetch = validation.streams_factory(ab, plans, dataset, "B", root, root / "prefetch", 2026, 2)
                proof = validation.prove_streams(serial, prefetch, ab, "B", expected_shape=(2, 8, 16, 16))
            self.assertTrue(proof["model_rng_equal"])
            self.assertTrue(proof["queued_resume"])

    def test_bitwise_hash_distinguishes_signed_zero_and_handles_scalar_tensors(self):
        import torch
        from validate_neighbor_prefetch import tree_sha256
        self.assertNotEqual(tree_sha256(torch.tensor(0.0)), tree_sha256(torch.tensor(-0.0)))
        self.assertNotEqual(tree_sha256({"momentum": torch.tensor([1])}),
                            tree_sha256({"momentum": torch.tensor([2])}))

    def test_queued_train_validation_restore_preserves_consumed_state(self):
        import torch
        import nnunet_neighbor_ab_trainer as ab
        import validate_neighbor_prefetch as validation
        torch.set_num_threads(1)
        self.assertTrue(hasattr(validation, "prove_streams"), "Queued stream proof is missing")

        def serial():
            return tuple(ab.PairedSingleThreadedAugmenter(RandomLoader(), seed) for seed in (11, 12))

        def prefetch():
            return tuple(ab.PairedPrefetchAugmenter(RandomLoader(), seed) for seed in (11, 12))

        proof = validation.prove_streams(serial, prefetch, ab, "A", expected_shape=(2, 8, 4, 4))
        self.assertTrue(proof["queued_resume"])
        self.assertTrue(proof["serial_to_prefetch_resume"])
        self.assertTrue(proof["model_rng_equal"])
        self.assertEqual(proof["pending_batches_at_save"], [2, 2])
        self.assertEqual([case["batches"] for case in proof["fullsize_cases"]], [8, 8])


if __name__ == "__main__":
    unittest.main()
