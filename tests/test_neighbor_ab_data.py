"""Semantic contracts of the target-only neighbor A/B data path."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_neighbor_ab import export_dataset, neighbor_indices
from neighbor_ab_preprocessor import NeighborABPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def make_plans():
    configuration = {
        "architecture": {"network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet", "arch_kwargs": {}, "_kw_requires_import": []},
        "data_identifier": "nnUNetPlans_2d_neighbor_ab_fullcanvas_v1",
        "preprocessor_name": "NeighborABPreprocessor",
        "spacing": [1.0, 1.0],
        "normalization_schemes": ["ZScoreNormalization"] * 8,
        "use_mask_for_norm": [False] * 8,
        "resampling_fn_data": "resample_data_or_seg_to_shape",
        "resampling_fn_seg": "resample_data_or_seg_to_shape",
        "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": False},
        "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": False},
    }
    return PlansManager({
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
        "configurations": {"2d": configuration},
        "label_manager": "LabelManager",
        "foreground_intensity_properties_per_channel": {},
    })


DATASET = {"channel_names": {str(i): "ultrasound" for i in range(8)}, "labels": {"background": 0, "LV": 1, "LA": 2, "RA": 3, "RV": 4, "ignore": 5}}


def preprocess(images, labels, spacing=(999.0, 1.25, 1.25)):
    plans = make_plans()
    preprocessor = NeighborABPreprocessor(verbose=False)
    return preprocessor.run_case_npy(images, labels, {"spacing": spacing}, plans, plans.get_configuration("2d"), DATASET)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class NeighborIndexTests(unittest.TestCase):
    def test_individual_clamping_keeps_target_at_channel_four(self):
        self.assertEqual(neighbor_indices(0, 9), [0, 0, 0, 0, 0, 1, 2, 3])
        self.assertEqual(neighbor_indices(1, 9), [0, 0, 0, 0, 1, 2, 3, 4])
        self.assertEqual(neighbor_indices(7, 9), [3, 4, 5, 6, 7, 8, 8, 8])
        self.assertEqual(neighbor_indices(8, 9), [4, 5, 6, 7, 8, 8, 8, 8])
        self.assertEqual(neighbor_indices(0, 1), [0] * 8)


class FixedCanvasTests(unittest.TestCase):
    def setUp(self):
        self.images = np.zeros((8, 1, 12, 10), dtype=np.float32)
        self.images[4, 0, 4:8, 3:7] = np.arange(16).reshape(4, 4) + 1
        self.labels = np.zeros((1, 1, 12, 10), dtype=np.int16)
        self.labels[0, 0, 4:6, 3:5] = 1
        self.labels[0, 0, 0, 0] = 5

    def test_replacing_seven_neighbors_preserves_final_a_input_bitwise(self):
        replaced = np.random.default_rng(7).normal(100, 20, self.images.shape).astype(np.float32)
        replaced[4] = self.images[4]
        original, _, original_properties = preprocess(self.images, self.labels)
        changed, _, changed_properties = preprocess(replaced, self.labels)
        original_a = np.repeat(original[4:5], 8, axis=0)
        changed_a = np.repeat(changed[4:5], 8, axis=0)
        np.testing.assert_array_equal(original_a, changed_a)
        self.assertEqual(original_properties["bbox_used_for_cropping"], [[0, 1], [0, 12], [0, 10]])
        self.assertEqual(changed_properties["bbox_used_for_cropping"], original_properties["bbox_used_for_cropping"])
        self.assertEqual(original.shape, (8, 1, 15, 12))
        self.assertFalse(np.array_equal(original[0], changed[0]))

    def test_gt_and_missing_gt_cannot_change_preprocessed_images(self):
        changed_labels = np.full_like(self.labels, 5)
        original, _, _ = preprocess(self.images, self.labels)
        changed, _, _ = preprocess(self.images, changed_labels)
        without_labels, _, _ = preprocess(self.images, None)
        np.testing.assert_array_equal(original, changed)
        np.testing.assert_array_equal(original, without_labels)

    def test_normalization_uses_full_frame_and_retains_labels_and_borders(self):
        images, labels, properties = preprocess(self.images, self.labels, spacing=(999, 1, 1))
        expected_target = (self.images[4] - self.images[4].mean()) / self.images[4].std()
        np.testing.assert_array_equal(images[4], expected_target)
        np.testing.assert_array_equal(labels, self.labels)
        self.assertLess(images[4, 0, 0, 0], 0)
        self.assertTrue(np.isfinite(images).all())
        self.assertEqual(tuple(properties["shape_before_cropping"]), (1, 12, 10))
        np.testing.assert_array_equal(self.images[0], np.zeros((1, 12, 10)))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        video = np.zeros((4, 3, 6), dtype=np.uint8)
        for frame in range(6):
            video[:, :, frame] = np.arange(12).reshape(4, 3) + frame * 20
        labels = np.zeros_like(video, dtype=np.uint8)
        labels[0, 0, 0] = 1
        labels[3, 2, 5] = 5
        self.video = video
        self.labels = labels
        image_path = self.root / "train_image.nii.gz"
        label_path = self.root / "train_label.nii.gz"
        nib.save(nib.Nifti1Image(video, np.eye(4)), image_path)
        nib.save(nib.Nifti1Image(labels, np.eye(4)), label_path)
        record = {"record_id": "Site/train", "source_folder": "Site", "image_path": image_path.name, "label_path": label_path.name, "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(), "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest()}
        write_csv(self.root / "data/metadata/cases.csv", [record])
        write_csv(self.root / "data/metadata/label_audit.csv", [{"record_id": "Site/train", "annotated_frames": "[0, 5]", "incomplete_frames": "[]"}])
        write_csv(self.root / "data/splits/baseline_v1.csv", [{"record_id": "Site/train", "split": "train", "label_form": "region", "group_id": "patient1"}, {"record_id": "Missing/test", "split": "test", "label_form": "region", "group_id": "patient2"}])
        baseline_raw = self.root / "data/processed/nnunet/raw/Dataset501_CardiacUDC2D"
        baseline_preprocessed = self.root / "data/processed/nnunet/preprocessed/Dataset501_CardiacUDC2D"
        baseline_raw.mkdir(parents=True)
        baseline_preprocessed.mkdir(parents=True)
        (baseline_raw / "dataset.json").write_text(json.dumps(DATASET))
        self.baseline_splits = baseline_preprocessed / "splits_final.json"
        self.baseline_splits.write_text(json.dumps([{"train": ["Site__train__f000", "Site__train__f005"], "val": []}]))

    def test_export_preserves_target_orientation_labels_ids_and_reads_no_test_files(self):
        output = self.root / "output"
        export_dataset(self.root, output, expected_counts={"train": 2, "val": 0})
        raw = output / "raw/Dataset503_CardiacUDCNeighborAB"
        self.assertFalse((raw / "imagesTs").exists())
        self.assertEqual(len(list((raw / "imagesTr").glob("*.nii.gz"))), 16)
        for channel, frame in enumerate([0, 0, 0, 0, 0, 1, 2, 3]):
            result = nib.load(raw / "imagesTr" / f"Site__train__f000_{channel:04d}.nii.gz")
            np.testing.assert_array_equal(np.asarray(result.dataobj), self.video[:, :, frame].T)
            self.assertEqual(result.header.get_zooms(), (200.0, 200.0))
        label = nib.load(raw / "labelsTr/Site__train__f005.nii.gz")
        np.testing.assert_array_equal(np.asarray(label.dataobj), self.labels[:, :, 5].T)
        actual_splits = json.loads((output / "preprocessed/Dataset503_CardiacUDCNeighborAB/splits_final.json").read_text())
        self.assertEqual(actual_splits, json.loads(self.baseline_splits.read_text()))
        self.assertEqual(json.loads((raw / "dataset.json").read_text())["labels"], DATASET["labels"])
        provenance = json.loads((raw / "source_fingerprints.json").read_text())
        self.assertEqual(len(provenance["records"]), 1)
        self.assertEqual(provenance["counts"], {"train": 2, "val": 0})

    def test_split_mismatch_is_rejected_before_writing_images(self):
        self.baseline_splits.write_text(json.dumps([{"train": ["wrong_target"], "val": []}]))
        output = self.root / "output"
        with self.assertRaisesRegex(ValueError, "Dataset501"):
            export_dataset(self.root, output, expected_counts={"train": 2, "val": 0})
        self.assertFalse((output / "raw/Dataset503_CardiacUDCNeighborAB/imagesTr").exists())

    def test_existing_cache_blocks_overwrite_before_deleting_export(self):
        output = self.root / "output"
        raw = output / "raw/Dataset503_CardiacUDCNeighborAB"
        cache = output / "preprocessed/Dataset503_CardiacUDCNeighborAB/nnUNetPlans_2d_neighbor_ab_fullcanvas_v1"
        raw.mkdir(parents=True)
        cache.mkdir(parents=True)
        sentinel = raw / "preserve_me"
        sentinel.write_text("existing raw data")
        with self.assertRaises(FileExistsError):
            export_dataset(self.root, output, expected_counts={"train": 2, "val": 0}, overwrite=True)
        self.assertEqual(sentinel.read_text(), "existing raw data")

    def test_changed_source_is_rejected_by_actual_fingerprint(self):
        image_path = self.root / "train_image.nii.gz"
        nib.save(nib.Nifti1Image(self.video + 1, np.eye(4)), image_path)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            export_dataset(self.root, self.root / "output", expected_counts={"train": 2, "val": 0})


if __name__ == "__main__":
    unittest.main()
