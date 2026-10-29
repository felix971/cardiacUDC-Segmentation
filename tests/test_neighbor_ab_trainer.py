"""CPU semantic tests: python -m unittest discover -s tests -p test_neighbor_ab_trainer.py."""

import copy
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
torch.set_num_threads(2)


class NeighborABTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((SCRIPTS / "nnunet_neighbor_ab_trainer.py").is_file(),
                        "The neighbor A/B trainer has not been implemented")
        self.ab = importlib.import_module("nnunet_neighbor_ab_trainer")

    def test_a_repeats_augmented_target_and_discards_all_neighbor_content(self):
        # A wrong target index, pre-augmentation repeat or real-frame A path breaks this.
        image = torch.arange(8 * 16 * 16, dtype=torch.float32).reshape(8, 16, 16)
        target = torch.ones(1, 16, 16, dtype=torch.int16)
        changed = image.clone()
        changed[:4] = -9999
        changed[5:] = 9999
        transform = self.ab.NeighborSpatialIntensityTransform(training=True, ignore_label=255)
        self.ab.seed_everything(2026)
        first = transform(image=image, segmentation=target.clone())["image"]
        self.ab.seed_everything(2026)
        second = transform(image=changed, segmentation=target.clone())["image"]
        model = self.ab.TargetFrameVNet(torch.nn.Identity(), mode="A")
        a = model.prepare_volume(first.unsqueeze(0))
        b = model.prepare_volume(second.unsqueeze(0))
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(tuple(a.shape), (1, 1, 8, 16, 16))
        for frame in range(8):
            self.assertTrue(torch.equal(a[:, 0, frame], first[None, 4]))

    def test_real_neighbor_path_and_target_only_gradient(self):
        # Selecting a central slice other than 4 or exposing all temporal labels breaks this.
        image = torch.arange(8.0).reshape(1, 8, 1, 1).expand(1, 8, 16, 16).clone()
        image.requires_grad_(True)
        model = self.ab.TargetFrameVNet(torch.nn.Identity(), mode="B")
        volume = model.prepare_volume(image)
        self.assertTrue(torch.equal(volume[:, 0], image))
        output = model(image)
        self.assertEqual(tuple(output.shape), (1, 1, 16, 16))
        self.assertTrue(torch.equal(output, torch.full_like(output, 4)))
        output.sum().backward()
        self.assertEqual(image.grad[:, 4].sum().item(), 256)
        self.assertEqual(image.grad[:, :4].sum().item(), 0)
        self.assertEqual(image.grad[:, 5:].sum().item(), 0)

    def test_spatial_transform_is_synchronized_and_preserves_ignore(self):
        # Channel-specific geometry or treating ignore as foreground changes this invariant.
        image = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 16, 16).repeat(8, 1, 1)
        target = torch.zeros(1, 16, 16, dtype=torch.int16)
        target[:, 4:12, 4:12] = 255
        transform = self.ab.NeighborSpatialIntensityTransform(training=True, ignore_label=255)
        self.ab.seed_everything(2026)
        for _ in range(20):
            result = transform(image=image.clone(), segmentation=target.clone())
            for frame in range(8):
                self.assertTrue(torch.equal(result["image"][frame], result["image"][4]))
            self.assertLessEqual(set(result["segmentation"].unique().tolist()), {0, 255})
            self.assertIn(255, result["segmentation"].unique().tolist())

    def test_validation_padding_conversion_never_modifies_cached_target(self):
        image = torch.zeros(8, 16, 16)
        target = torch.full((1, 16, 16), -1, dtype=torch.int16)
        transform = self.ab.NeighborSpatialIntensityTransform(training=False, ignore_label=255)
        result = transform(image=image, segmentation=target)
        self.assertTrue(torch.all(target == -1))
        self.assertTrue(torch.all(result["segmentation"] == 255))

    def test_validation_entry_reasserts_determinism_after_nnunet_cli_override(self):
        # nnUNetv2_train changes cuDNN flags after construction/checkpoint loading.
        trainer = self.ab.nnUNetTrainerNeighborA_500epochs.__new__(self.ab.nnUNetTrainerNeighborA_500epochs)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        observed = []

        def inspect_validation_entry(instance, save_probabilities=False):
            observed.append((torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark))

        with patch.object(self.ab.nnUNetTrainerNoDeepSupervision, "perform_actual_validation", inspect_validation_entry):
            trainer.perform_actual_validation()
        self.assertEqual(observed, [(True, False)])

    def test_actual_vnet_has_identical_initial_state_and_only_five_dimensional_dropout(self):
        config = SimpleNamespace(patch_size=[16, 16])
        self.ab.seed_everything(2026)
        a = self.ab.nnUNetTrainerNeighborA_500epochs.build_network_architecture(None, config, 8, 5, False)
        initial_a = self.ab.state_dict_sha256(a)
        del a
        self.ab.seed_everything(2026)
        b = self.ab.nnUNetTrainerNeighborB_500epochs.build_network_architecture(None, config, 8, 5, False)
        self.assertEqual(initial_a, self.ab.state_dict_sha256(b))
        dropout_shapes = []
        for module in b.modules():
            if isinstance(module, torch.nn.Dropout3d):
                module.register_forward_pre_hook(lambda module, args: dropout_shapes.append(tuple(args[0].shape)))
        b.train()
        images = torch.randn(2, 8, 16, 16, requires_grad=True)
        result = b(images)
        self.assertEqual(tuple(result.shape), (2, 5, 16, 16))
        result.square().mean().backward()
        self.assertTrue(dropout_shapes)
        self.assertTrue(all(len(shape) == 5 for shape in dropout_shapes))
        self.assertTrue(all(parameter.grad is not None for parameter in b.parameters()))
        self.assertGreater(images.grad[:, :4].abs().sum().item(), 0)
        self.assertGreater(images.grad[:, 5:].abs().sum().item(), 0)

    def test_isolated_streams_resume_exactly_without_perturbing_model_rng(self):
        class RandomBatches:
            def __iter__(self):
                return self

            def __next__(self):
                data = torch.rand(2, 8, 16, 16) + np.random.rand() + random.random()
                return {"data": data, "target": torch.zeros(2, 1, 16, 16),
                        "keys": np.random.choice(["case_a", "case_b", "case_c"], 2),
                        "augmentation": [{"gain": float(np.random.uniform(0.75, 1.25))}]}

        stream = self.ab.PairedSingleThreadedAugmenter(RandomBatches(), seed=2027)
        self.ab.seed_everything(17)
        expected_model_draw = torch.rand(4)
        self.ab.seed_everything(17)
        next(stream)
        self.assertTrue(torch.equal(expected_model_draw, torch.rand(4)))
        saved = copy.deepcopy(stream.state_dict())
        expected = next(stream)
        resumed = self.ab.PairedSingleThreadedAugmenter(RandomBatches(), seed=99)
        resumed.load_state_dict(saved)
        actual = next(resumed)
        self.assertTrue(torch.equal(actual["data"], expected["data"]))
        self.assertEqual(actual["keys"].tolist(), expected["keys"].tolist())
        self.assertEqual(actual["augmentation"], expected["augmentation"])
        self.assertEqual(resumed.state_dict()["audit_sha256"], stream.state_dict()["audit_sha256"])

    def test_checkpoint_restores_sampling_augmentation_optimizer_and_model_rng(self):
        # A resumed optimizer update must exactly match the uninterrupted next update.
        ab = self.ab

        class TinyTrainer(ab.nnUNetTrainerNeighborA_500epochs):
            @classmethod
            def build_network_architecture(cls, plans_manager, configuration_manager,
                                           num_input_channels, num_output_channels,
                                           enable_deep_supervision=True):
                backbone = torch.nn.Sequential(torch.nn.Conv3d(1, 5, 1), torch.nn.Dropout3d(0.2))
                return ab.TargetFrameVNet(backbone, cls.neighbor_mode)

        with tempfile.TemporaryDirectory() as tmp:
            env = {"nnUNet_results": str(Path(tmp) / "results"),
                   "nnUNet_preprocessed": str(Path(tmp) / "preprocessed"),
                   "NEIGHBOR_AB_SEED": "2026", "nnUNet_compile": "false"}
            with patch.dict(os.environ, env):
                plans, dataset_json = self.make_dataset(tmp)
                trainer = TinyTrainer(copy.deepcopy(plans), "2d", 0, dataset_json, torch.device("cpu"))
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
                trainer.on_train_start()
                self.assertTrue(torch.backends.cudnn.deterministic)
                self.assertFalse(torch.backends.cudnn.benchmark)
                first_batch = next(trainer.dataloader_train)
                trainer.train_step(first_batch)
                checkpoint = str(Path(tmp) / "checkpoint.pth")
                trainer.save_checkpoint(checkpoint)
                expected_batch = next(trainer.dataloader_train)
                expected_loss = trainer.train_step(expected_batch)["loss"]
                expected_state = copy.deepcopy(trainer.network.state_dict())
                expected_rng = torch.rand(5)

                resumed = TinyTrainer(copy.deepcopy(plans), "2d", 0, dataset_json, torch.device("cpu"))
                resumed.load_checkpoint(checkpoint)
                resumed.on_train_start()
                actual_batch = next(resumed.dataloader_train)
                self.assertEqual(actual_batch["keys"].tolist(), expected_batch["keys"].tolist())
                self.assertTrue(torch.equal(actual_batch["data"], expected_batch["data"]))
                self.assertTrue(torch.equal(actual_batch["target"], expected_batch["target"]))
                actual_loss = resumed.train_step(actual_batch)["loss"]
                np.testing.assert_array_equal(actual_loss, expected_loss)
                for name, value in resumed.network.state_dict().items():
                    self.assertTrue(torch.equal(value, expected_state[name]), name)
                self.assertTrue(torch.equal(torch.rand(5), expected_rng))
                stored = torch.load(checkpoint, weights_only=False)
                self.assertEqual(stored["neighbor_ab"]["seed"], 2026)
                self.assertEqual(stored["neighbor_ab"]["mode"], "A")
                self.assertIsNone(stored["inference_allowed_mirroring_axes"])
                self.assertTrue((Path(resumed.output_folder) / "neighbor_ab_run.json").is_file())
                self.assertTrue((Path(resumed.output_folder) / "neighbor_ab_train_audit.jsonl").is_file())

                with patch.dict(os.environ, {"NEIGHBOR_AB_SEED": "2027"}):
                    wrong_seed = TinyTrainer(copy.deepcopy(plans), "2d", 0, dataset_json, torch.device("cpu"))
                    with self.assertRaisesRegex(ValueError, "seed"):
                        wrong_seed.load_checkpoint(checkpoint)

    @staticmethod
    def make_dataset(tmp):
        dataset_name = "Dataset503_TestNeighbor"
        root = Path(tmp) / "preprocessed" / dataset_name
        cache = root / "neighbor_ab_2d"
        cache.mkdir(parents=True)
        for index in range(3):
            image = np.arange(8 * 1 * 20 * 20, dtype=np.float32).reshape(8, 1, 20, 20)
            image = image / 1000 + index
            target = np.zeros((1, 1, 20, 20), dtype=np.int16)
            target[:, :, 5:15, 5:15] = index + 1
            name = f"case_{index}"
            np.savez(cache / f"{name}.npz", data=image, seg=target)
            locations = np.argwhere(target == index + 1)
            with (cache / f"{name}.pkl").open("wb") as handle:
                pickle.dump({"class_locations": {index + 1: locations}}, handle)
        (root / "splits_final.json").write_text(json.dumps([{"train": ["case_0", "case_1"], "val": ["case_2"]}]))
        (root / "dataset_fingerprint.json").write_text("{}")
        plans = {"dataset_name": dataset_name, "plans_name": "NeighborABTestPlans",
                 "continue_training": False,
                 "configurations": {"2d": {"data_identifier": "neighbor_ab_2d",
                                            "batch_size": 2, "patch_size": [16, 16],
                                            "architecture": {},
                                            "batch_dice": True,
                                            "use_mask_for_norm": [False] * 8,
                                            "spacing": [1, 1]}}}
        dataset_json = {"channel_names": {str(i): f"frame_{i}" for i in range(8)},
                        "labels": {"background": 0, "LV": 1, "LA": 2, "RA": 3, "RV": 4},
                        "file_ending": ".nii.gz", "numTraining": 3}
        return plans, dataset_json


if __name__ == "__main__":
    unittest.main()
