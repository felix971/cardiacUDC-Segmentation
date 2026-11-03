"""CPU tests for the evidence required before an A/B run is declared complete."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_neighbor_ab_pair as runner


class FinishedRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fold = Path(self.temp.name) / "fold_0"
        self.validation = self.fold / "validation"
        self.validation.mkdir(parents=True)
        self.checkpoint_path = self.fold / "checkpoint_final.pth"
        self.summary_path = self.validation / "summary.json"
        self.expected = []
        cases = []

        for index in range(218):
            identifier = f"Site__patient_{index:03d}__f004"
            self.expected.append(identifier)
            prediction = self.validation / f"{identifier}.nii.gz"
            # The completion check inspects existence and size, not NIfTI contents.
            prediction.write_bytes(b"synthetic prediction")
            metrics = {}
            for label, dice in (("1", 0.7), ("2", 0.8), ("3", 0.8), ("4", 0.9)):
                metrics[label] = {"Dice": dice, "IoU": dice / (2 - dice)}
            cases.append({"prediction_file": str(prediction), "metrics": metrics})

        self.checkpoint = {
            "current_epoch": 500,
            "trainer_name": "nnUNetTrainerNeighborA_500epochs",
            "network_weights": {"weight": torch.tensor([1.0])},
            "neighbor_ab": {
                "seed": 2026,
                "mode": "A",
                "initial_state_sha256": "a" * 64,
                "initial_model_rng_sha256": "b" * 64,
            },
            "neighbor_ab_state": {
                "train_stream": {"batches_seen": 125000, "audit_sha256": "c" * 64},
                "val_stream": {"batches_seen": 25000, "audit_sha256": "d" * 64},
            },
        }
        self.summary = {
            "metric_per_case": cases,
            "foreground_mean": {"Dice": 0.8},
            "mean": {
                "1": {"Dice": 0.7},
                "2": {"Dice": 0.8},
                "3": {"Dice": 0.8},
                "4": {"Dice": 0.9},
            },
        }
        self.write_checkpoint()
        self.write_summary()

    def write_checkpoint(self):
        torch.save(self.checkpoint, self.checkpoint_path)
        os.utime(self.checkpoint_path, (1_700_000_000, 1_700_000_000))

    def write_summary(self):
        self.summary_path.write_text(json.dumps(self.summary))
        os.utime(self.summary_path, (1_700_000_001, 1_700_000_001))

    def verify(self, mode="A", seed=2026):
        return runner.verify_finished(self.fold, self.expected, mode, seed)

    def test_complete_218_case_run_returns_metrics_and_pairing_audits(self):
        for mode in ("A", "B"):
            with self.subTest(mode=mode):
                self.checkpoint["trainer_name"] = f"nnUNetTrainerNeighbor{mode}_500epochs"
                self.checkpoint["neighbor_ab"]["mode"] = mode
                self.write_checkpoint()

                result = self.verify(mode=mode)

                self.assertEqual(result["count"], 218)
                self.assertEqual(result["foreground_mean"]["Dice"], 0.8)
                self.assertEqual(result["mean"]["4"]["Dice"], 0.9)
                self.assertEqual(result["initial_state_sha256"], "a" * 64)
                self.assertEqual(result["initial_model_rng_sha256"], "b" * 64)
                self.assertEqual(result["train_audit_sha256"], "c" * 64)
                self.assertEqual(result["val_audit_sha256"], "d" * 64)

    def test_missing_summary_case_is_rejected_even_if_prediction_exists(self):
        self.summary["metric_per_case"].pop()
        self.write_summary()

        with self.assertRaises(ValueError):
            self.verify()

    def test_extra_summary_case_and_prediction_are_rejected(self):
        extra = copy.deepcopy(self.summary["metric_per_case"][0])
        prediction = self.validation / "unexpected_target.nii.gz"
        prediction.write_bytes(b"extra prediction")
        extra["prediction_file"] = str(prediction)
        self.summary["metric_per_case"].append(extra)
        self.write_summary()

        with self.assertRaises(ValueError):
            self.verify()

    def test_replacing_one_target_is_rejected_despite_218_unique_predictions(self):
        original = Path(self.summary["metric_per_case"][0]["prediction_file"])
        replacement = self.validation / "wrong_target.nii.gz"
        original.rename(replacement)
        self.summary["metric_per_case"][0]["prediction_file"] = str(replacement)
        self.write_summary()

        with self.assertRaises(ValueError):
            self.verify()

    def test_duplicate_case_is_rejected_despite_218_summary_entries(self):
        self.summary["metric_per_case"][1] = copy.deepcopy(self.summary["metric_per_case"][0])
        self.write_summary()

        with self.assertRaises(ValueError):
            self.verify()

    def test_missing_or_empty_prediction_is_rejected(self):
        prediction = self.validation / f"{self.expected[0]}.nii.gz"
        for state in ("missing", "empty"):
            with self.subTest(state=state):
                if state == "missing":
                    prediction.unlink()
                else:
                    prediction.write_bytes(b"")

                with self.assertRaises(ValueError):
                    self.verify()

    def test_unlisted_prediction_is_rejected_even_with_complete_summary(self):
        (self.validation / "unexpected_target.nii.gz").write_bytes(b"extra prediction")

        with self.assertRaises(ValueError):
            self.verify()

    def test_checkpoint_seed_or_mode_must_match_the_requested_run(self):
        for key, wrong in (("seed", 2027), ("mode", "B")):
            with self.subTest(key=key):
                original = self.checkpoint["neighbor_ab"][key]
                self.checkpoint["neighbor_ab"][key] = wrong
                self.write_checkpoint()

                with self.assertRaises(ValueError):
                    self.verify()

                self.checkpoint["neighbor_ab"][key] = original

    def test_partial_checkpoint_cannot_be_declared_complete(self):
        invalid_fields = (
            ("current_epoch", 499),
            ("current_epoch", 501),
            ("trainer_name", "nnUNetTrainerVNet_500epochs"),
            ("network_weights", {}),
        )
        for key, wrong in invalid_fields:
            with self.subTest(key=key, value=wrong):
                original = self.checkpoint[key]
                self.checkpoint[key] = wrong
                self.write_checkpoint()

                with self.assertRaises(ValueError):
                    self.verify()

                self.checkpoint[key] = original

    def test_sampling_stream_counts_must_match_all_500_epochs_exactly(self):
        invalid_counts = (
            ("train_stream", 124999),
            ("train_stream", 125001),
            ("val_stream", 24999),
            ("val_stream", 25001),
        )
        for name, wrong in invalid_counts:
            with self.subTest(stream=name, count=wrong):
                stream = self.checkpoint["neighbor_ab_state"][name]
                original = stream["batches_seen"]
                stream["batches_seen"] = wrong
                self.write_checkpoint()

                with self.assertRaises(ValueError):
                    self.verify()

                stream["batches_seen"] = original

    def test_validation_from_before_final_checkpoint_is_rejected(self):
        os.utime(self.summary_path, (1_699_999_999, 1_699_999_999))

        with self.assertRaises(ValueError):
            self.verify()

    def test_each_case_requires_all_four_foreground_classes(self):
        del self.summary["metric_per_case"][0]["metrics"]["4"]
        self.write_summary()

        with self.assertRaises(ValueError):
            self.verify()

    def test_nonfinite_or_out_of_range_foreground_dice_is_rejected(self):
        for dice in (float("nan"), float("inf"), -0.01, 1.01):
            with self.subTest(dice=dice):
                self.summary["foreground_mean"]["Dice"] = dice
                self.write_summary()

                with self.assertRaises(ValueError):
                    self.verify()

    def test_nonfinite_or_out_of_range_case_dice_is_rejected(self):
        for dice in (float("nan"), float("inf"), -0.01, 1.01):
            with self.subTest(dice=dice):
                self.summary["metric_per_case"][0]["metrics"]["1"]["Dice"] = dice
                self.write_summary()

                with self.assertRaises(ValueError):
                    self.verify()

    def test_class_dice_must_agree_with_the_per_case_results(self):
        self.summary["mean"]["1"]["Dice"] = 0.71
        self.write_summary()

        with self.assertRaises(ValueError):
            self.verify()

    def test_foreground_dice_must_agree_with_the_per_class_results(self):
        self.summary["foreground_mean"]["Dice"] = 0.81
        self.write_summary()

        with self.assertRaises(ValueError):
            self.verify()


class DataFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "preprocessed/Dataset503_CardiacUDCNeighborAB"
        self.cache = self.dataset / "nnUNetPlans_2d_neighbor_ab_fullcanvas_v1"
        self.cache.mkdir(parents=True)
        self.split = {"train": ["train_a", "train_b"], "val": ["validation_a"]}

        for identifier in ("train_a", "train_b", "validation_a"):
            for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
                (self.cache / f"{identifier}{suffix}").write_bytes(b"cached data")
        (self.dataset / "splits_final.json").write_text(json.dumps([self.split]))

    def fingerprint(self):
        return runner.data_fingerprint(self.root, self.split)

    def test_complete_cache_has_stable_fingerprint(self):
        initial = self.fingerprint()

        self.assertEqual(self.fingerprint(), initial)
        self.assertEqual(len(initial), 64)

    def test_missing_data_segmentation_or_properties_id_is_rejected(self):
        for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
            with self.subTest(suffix=suffix):
                missing = self.cache / f"validation_a{suffix}"
                content = missing.read_bytes()
                missing.unlink()

                with self.assertRaises(ValueError):
                    self.fingerprint()

                missing.write_bytes(content)

    def test_extra_data_segmentation_or_properties_id_is_rejected(self):
        for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
            with self.subTest(suffix=suffix):
                extra = self.cache / f"unexpected_case{suffix}"
                extra.write_bytes(b"extra cached data")

                with self.assertRaises(ValueError):
                    self.fingerprint()

                extra.unlink()

    def test_equal_file_counts_do_not_allow_a_replaced_case_id(self):
        for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
            original = self.cache / f"validation_a{suffix}"
            original.rename(self.cache / f"wrong_case{suffix}")

        with self.assertRaises(ValueError):
            self.fingerprint()

    def test_fingerprint_changes_when_cache_or_dataset_metadata_changes(self):
        paths = (
            self.cache / "train_a.b2nd",
            self.cache / "train_a_seg.b2nd",
            self.cache / "train_a.pkl",
            self.dataset / "splits_final.json",
        )
        initial = self.fingerprint()
        for path in paths:
            with self.subTest(file=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b" changed")

                self.assertNotEqual(self.fingerprint(), initial)

                path.write_bytes(original)


class CodeFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = (
            "scripts/trainer.py",
            "scripts/run_neighbor_ab_pair.py",
            "uv.lock",
            "configs/neighbor_ab_base_plans.json",
            "docs/experiments/neighbor-ab-protocol.md",
        )
        for relative in self.paths:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"frozen contents of {relative}\n")
        root_patch = patch.object(runner, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_source_lock_plans_and_protocol_changes_invalidate_provenance(self):
        initial = runner.code_fingerprint()
        self.assertEqual(runner.code_fingerprint(), initial)

        for relative in self.paths:
            with self.subTest(file=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.write_bytes(original + b" changed")

                self.assertNotEqual(runner.code_fingerprint(), initial)

                path.write_bytes(original)

    def test_adding_or_renaming_python_source_invalidates_provenance(self):
        initial = runner.code_fingerprint()
        extra = self.root / "scripts/new_transform.py"
        extra.write_text("new transformation\n")
        self.assertNotEqual(runner.code_fingerprint(), initial)
        extra.unlink()

        original = self.root / "scripts/trainer.py"
        original.rename(self.root / "scripts/replacement_trainer.py")
        self.assertNotEqual(runner.code_fingerprint(), initial)


if __name__ == "__main__":
    unittest.main()
