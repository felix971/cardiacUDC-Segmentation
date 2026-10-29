"""Paired real-neighbor versus repeated-target VNet3D-T8 experiment.

Dataset503 uses eight 2D image channels in offsets [-4, ..., 3] order. Both
trainers execute the same 3D network and supervise only output frame 4. A repeats
the already augmented target immediately before the 3D forward pass. B retains
the actual frames. This module and nnunet_vnet_trainer.py must be registered in
nnunetv2.training.nnUNetTrainer for both training and predictor restoration.

Set NEIGHBOR_AB_SEED explicitly; use a distinct nnUNet_results root per host/seed.
NEIGHBOR_AB_EXPECTED_INITIAL_HASH optionally enforces the other mode's initial
parameter + BatchNorm state hash before any training. Checkpoints resume at epoch
boundaries and include independent sampling/augmentation and model RNG states.
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import socket

import numpy as np
import torch
from torch.nn import functional as F
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from monai.networks.nets import VNet
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)

try:
    from nnunetv2.training.nnUNetTrainer.nnunet_vnet_trainer import keep_frames_in_shallow_levels
except ModuleNotFoundError:
    from nnunet_vnet_trainer import keep_frames_in_shallow_levels


TARGET_INDEX = 4
FRAME_COUNT = 8
AUGMENTATION = {
    "flip_y_probability": 0.5,
    "flip_x_probability": 0.5,
    "rotation_probability": 0.2,
    "rotation_degrees": [-15.0, 15.0],
    "scale_probability": 0.2,
    "scale_range": [0.85, 1.25],
    "gain_probability": 0.15,
    "gain_range": [0.75, 1.25],
    "offset_probability": 0.15,
    "offset_range": [-0.1, 0.1],
    "image_interpolation": "bilinear",
    "label_interpolation": "nearest",
    "channel_statistics": False,
    "temporal_transforms": False,
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism():
    # nnUNetv2_train overwrites cuDNN flags after construction/loading. Reapply
    # at actual training/validation entry, not just when creating the trainer.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def capture_rng_state(include_cuda=True):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if include_cuda and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def rng_state_sha256(state):
    digest = hashlib.sha256()
    digest.update(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    digest.update(state["torch_cpu"].cpu().numpy().tobytes())
    for value in state.get("torch_cuda", []):
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def tensor_sha256(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def state_dict_sha256(network):
    """Include all parameters AND buffers, notably BatchNorm running statistics."""
    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


class DeterministicCrossEntropyLoss(torch.nn.Module):
    """Unweighted mean CE without PyTorch's nondeterministic CUDA 2D NLL kernel.

    Keep nnU-Net's integer targets and ignore-index semantics. The compound loss
    already skips CE for an entirely ignored batch. Like CrossEntropyLoss itself,
    a direct all-ignored call returns NaN and zero input gradients. FP32 arithmetic
    for half logits matches CrossEntropyLoss's CUDA autocast policy.
    """

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        if target.ndim == logits.ndim:
            if target.shape[1] != 1:
                raise ValueError("Cross entropy requires a single integer target channel")
            target = target[:, 0]
        target = target.long()
        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0)
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        log_probabilities = F.log_softmax(logits, dim=1)
        selected = log_probabilities.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        losses = -selected.masked_fill(~valid, 0)
        return losses.sum() / valid.sum()


class TargetFrameVNet(torch.nn.Module):
    def __init__(self, backbone, mode):
        super().__init__()
        if mode not in ("A", "B"):
            raise ValueError("Neighbor mode must be A or B")
        self.backbone = backbone
        self.mode = mode

    def prepare_volume(self, data):
        if data.ndim != 4 or data.shape[1] != FRAME_COUNT:
            raise ValueError("Expected eight frame channels shaped [N,8,H,W]")
        if self.mode == "A":
            target = data[:, TARGET_INDEX:TARGET_INDEX + 1]
            data = target.repeat(1, FRAME_COUNT, 1, 1)
        return data.unsqueeze(1)

    def forward(self, data):
        volume = self.prepare_volume(data)
        logits = self.backbone(volume)
        return logits[:, :, TARGET_INDEX]


class NeighborSpatialIntensityTransform:
    """CPU-only 2D geometry and scalar intensity draws, shared across all frames.

    Draws never depend on image/label statistics. Sampling/cropping is performed by
    nnU-Net using only the target label. Labels use nearest-neighbor interpolation;
    out-of-canvas and loader padding use the declared ignore label, or background
    if no ignore label exists. There is no temporal interpolation or reversal.
    """

    def __init__(self, training, ignore_label=None):
        self.training = training
        self.ignore_label = ignore_label
        self.audit_parameters = []

    def __call__(self, image, segmentation, **kwargs):
        params = {"flip_y": False, "flip_x": False, "rotation_degrees": 0.0,
                  "scale": 1.0, "gain": 1.0, "offset": 0.0}
        if self.training:
            params["flip_y"] = bool(np.random.random() < AUGMENTATION["flip_y_probability"])
            params["flip_x"] = bool(np.random.random() < AUGMENTATION["flip_x_probability"])
            if np.random.random() < AUGMENTATION["rotation_probability"]:
                params["rotation_degrees"] = float(np.random.uniform(*AUGMENTATION["rotation_degrees"]))
            if np.random.random() < AUGMENTATION["scale_probability"]:
                params["scale"] = float(np.random.uniform(*AUGMENTATION["scale_range"]))
            if np.random.random() < AUGMENTATION["gain_probability"]:
                params["gain"] = float(np.random.uniform(*AUGMENTATION["gain_range"]))
            if np.random.random() < AUGMENTATION["offset_probability"]:
                params["offset"] = float(np.random.uniform(*AUGMENTATION["offset_range"]))

        if params["rotation_degrees"] != 0.0 or params["scale"] != 1.0:
            height, width = image.shape[-2:]
            angle = math.radians(params["rotation_degrees"])
            cosine = math.cos(angle) * params["scale"]
            sine = math.sin(angle) * params["scale"]
            # Correct normalized-coordinate rotation for a rectangular pixel canvas.
            theta = image.new_tensor([[cosine, -sine * height / width, 0],
                                      [sine * width / height, cosine, 0]])
            grid = F.affine_grid(theta[None], (1, FRAME_COUNT, height, width), align_corners=False)
            image = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="zeros",
                                  align_corners=False)[0]
            # Offset labels so grid_sample's zero padding becomes the sentinel -1.
            segmentation = F.grid_sample(segmentation[None].float() + 1, grid, mode="nearest",
                                         padding_mode="zeros", align_corners=False)[0] - 1

        flip_axes = []
        if params["flip_y"]:
            flip_axes.append(-2)
        if params["flip_x"]:
            flip_axes.append(-1)
        if flip_axes:
            image = torch.flip(image, flip_axes)
            segmentation = torch.flip(segmentation, flip_axes)

        image = image * params["gain"] + params["offset"]
        segmentation = segmentation.to(torch.int16)
        padding_label = 0 if self.ignore_label is None else self.ignore_label
        # Loader crops may share read-only memory-mapped labels. Never mutate them.
        segmentation = segmentation.masked_fill(segmentation == -1, padding_label)
        self.audit_parameters.append(params)
        return {"image": image, "segmentation": segmentation}


class AuditedNeighborDataLoader(nnUNetDataLoader):
    def get_bbox(self, *args, **kwargs):
        lower, upper = super().get_bbox(*args, **kwargs)
        self.batch_bboxes.append({"lower": [int(value) for value in lower],
                                  "upper": [int(value) for value in upper]})
        return lower, upper

    def generate_train_batch(self):
        self.batch_bboxes = []
        self.transforms.audit_parameters = []
        batch = super().generate_train_batch()
        batch["augmentation"] = copy.deepcopy(self.transforms.audit_parameters)
        batch["crop_bboxes"] = self.batch_bboxes
        return batch


class PairedSingleThreadedAugmenter(SingleThreadedAugmenter):
    """No prefetch, workers, or hidden pending batches; exactly checkpointable.

    nnU-Net samples using global NumPy RNG. Swap only Python/NumPy/Torch CPU RNG
    while fetching a batch, then restore the caller's state. This makes data draws
    independent of model dropout and isolates validation from the training stream.
    """

    def __init__(self, data_loader, seed, audit_path=None):
        super().__init__(data_loader, None)
        outside = capture_rng_state(include_cuda=False)
        random.seed(seed)
        np.random.seed(seed)
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        self.rng_state = capture_rng_state(include_cuda=False)
        restore_rng_state(outside)
        self.batches_seen = 0
        self.audit_sha256 = hashlib.sha256(b"").hexdigest()
        self.audit_path = Path(audit_path) if audit_path is not None else None

    def __next__(self):
        outside = capture_rng_state(include_cuda=False)
        try:
            restore_rng_state(self.rng_state)
            batch = super().__next__()
            self.rng_state = capture_rng_state(include_cuda=False)
        finally:
            restore_rng_state(outside)

        audit = {
            "batch": self.batches_seen,
            "keys": [str(key) for key in batch["keys"]],
            "crop_bboxes": batch.get("crop_bboxes", []),
            "augmentation": batch.get("augmentation", []),
            "augmented_target_sha256": tensor_sha256(batch["data"][:, TARGET_INDEX]),
            "target_gt_sha256": tensor_sha256(batch["target"]),
            "data_rng_sha256": rng_state_sha256(self.rng_state),
        }
        encoded = json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
        self.audit_sha256 = hashlib.sha256(self.audit_sha256.encode() + encoded).hexdigest()
        audit["cumulative_sha256"] = self.audit_sha256
        if self.audit_path is not None and self.batches_seen < 8:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            # Restarting from a boundary replays any uncheckpointed audit records.
            mode = "w" if self.batches_seen == 0 else "a"
            with self.audit_path.open(mode) as handle:
                handle.write(json.dumps(audit, sort_keys=True) + "\n")
        self.batches_seen += 1
        return batch

    def state_dict(self):
        return {"rng_state": copy.deepcopy(self.rng_state),
                "batches_seen": self.batches_seen, "audit_sha256": self.audit_sha256}

    def load_state_dict(self, state):
        self.rng_state = copy.deepcopy(state["rng_state"])
        self.batches_seen = state["batches_seen"]
        self.audit_sha256 = state["audit_sha256"]
        if self.audit_path is not None and self.audit_path.exists():
            lines = self.audit_path.read_text().splitlines()
            keep = []
            for line in lines:
                if json.loads(line)["batch"] < self.batches_seen:
                    keep.append(line)
            self.audit_path.write_text("".join(line + "\n" for line in keep))


class _nnUNetTrainerNeighborAB(nnUNetTrainerNoDeepSupervision):
    neighbor_mode = None

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        if "NEIGHBOR_AB_SEED" not in os.environ:
            raise ValueError("NEIGHBOR_AB_SEED must be explicitly set")
        self.neighbor_seed = int(os.environ["NEIGHBOR_AB_SEED"])
        if not 0 <= self.neighbor_seed < 2**32 - 2:
            raise ValueError("NEIGHBOR_AB_SEED must be in [0, 2**32 - 3]")
        configure_determinism()
        seed_everything(self.neighbor_seed)
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.is_ddp or self.is_cascaded or self.label_manager.has_regions:
            raise ValueError("Neighbor A/B requires one process, no cascade, and five exclusive classes")
        if self.configuration_manager.batch_size != 2:
            raise ValueError("The paired protocol fixes batch size to 2")
        self.num_epochs = 500
        self.num_iterations_per_epoch = 250
        self.initial_lr = 0.01
        self.weight_decay = 3e-5
        self.save_every = 10
        self.inference_allowed_mirroring_axes = None
        self._pending_resume_state = None
        self.initial_state_sha256 = None

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        patch = configuration_manager.patch_size
        if len(patch) != 2 or any(int(side) % 16 for side in patch):
            raise ValueError("Neighbor A/B requires a 2D patch with sides divisible by 16")
        if num_input_channels != FRAME_COUNT or num_output_channels != 5:
            raise ValueError("Neighbor A/B requires eight input frames and five output classes")
        backbone = VNet(spatial_dims=3, in_channels=1, out_channels=5)
        keep_frames_in_shallow_levels(backbone)
        return TargetFrameVNet(backbone, cls.neighbor_mode)

    def _do_i_compile(self):
        # Keep model RNG draw semantics explicit and identical across fresh/resumed runs.
        return False

    def _build_loss(self):
        loss = super()._build_loss()
        original_ce = loss.ce
        if original_ce.weight is not None or original_ce.reduction != "mean" or original_ce.label_smoothing != 0:
            raise ValueError("Neighbor A/B requires unweighted mean CE without label smoothing")
        loss.ce = DeterministicCrossEntropyLoss(original_ce.ignore_index)
        return loss

    def initialize(self):
        configure_determinism()
        seed_everything(self.neighbor_seed)
        super().initialize()
        self.initial_state_sha256 = state_dict_sha256(self.network)
        self.initial_model_rng_sha256 = rng_state_sha256(capture_rng_state())
        expected = os.environ.get("NEIGHBOR_AB_EXPECTED_INITIAL_HASH")
        if expected and expected != self.initial_state_sha256:
            raise ValueError("Initial parameter/BatchNorm hash differs from the paired run")

    def plot_network_architecture(self):
        # Graph tracing can consume model RNG and change BatchNorm running buffers.
        self.print_to_log_file("Architecture graph skipped for deterministic paired training")

    def set_deep_supervision_enabled(self, enabled):
        pass

    def experiment_config(self):
        return {
            "schema_version": 1, "seed": self.neighbor_seed, "mode": self.neighbor_mode,
            "target_index": TARGET_INDEX, "frame_offsets": list(range(-4, 4)),
            "patch_size": list(self.configuration_manager.patch_size), "batch_size": 2,
            "epochs": self.num_epochs, "updates_per_epoch": self.num_iterations_per_epoch,
            "checkpoint_every_epochs": self.save_every,
            "validation_batches_per_epoch": self.num_val_iterations_per_epoch,
            "optimizer": {"name": "SGD", "lr": self.initial_lr, "momentum": 0.99,
                          "nesterov": True, "weight_decay": self.weight_decay},
            "scheduler": "polynomial", "augmentation": AUGMENTATION,
            "cross_entropy": "log_softmax_gather_masked_mean",
            "ignore_label": self.label_manager.ignore_label,
            "oversample_foreground_percent": self.oversample_foreground_percent,
            "probabilistic_oversampling": self.probabilistic_oversampling,
            "train_stream_seed": self.neighbor_seed + 1, "val_stream_seed": self.neighbor_seed + 2,
            "inference_mirroring_axes": None, "compile": False,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "initial_state_sha256": self.initial_state_sha256,
            "initial_model_rng_sha256": self.initial_model_rng_sha256,
        }

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        dataset_train, dataset_val = self.get_tr_and_val_datasets()
        streams = []
        for name, dataset, training, seed in [
            ("train", dataset_train, True, self.neighbor_seed + 1),
            ("val", dataset_val, False, self.neighbor_seed + 2),
        ]:
            transform = NeighborSpatialIntensityTransform(training, self.label_manager.ignore_label)
            loader = AuditedNeighborDataLoader(
                dataset, self.batch_size, self.configuration_manager.patch_size,
                self.configuration_manager.patch_size, self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=transform,
                probabilistic_oversampling=self.probabilistic_oversampling,
            )
            stream = PairedSingleThreadedAugmenter(
                loader, seed, Path(self.output_folder) / f"neighbor_ab_{name}_audit.jsonl",
            )
            streams.append(stream)
        # Deliberately no warmup next(): every generated batch belongs to an update.
        return tuple(streams)

    def on_train_start(self):
        configure_determinism()
        super().on_train_start()
        if self._pending_resume_state is not None:
            pending = self._pending_resume_state
            self.dataloader_train.load_state_dict(pending["train_stream"])
            self.dataloader_val.load_state_dict(pending["val_stream"])
            restore_rng_state(pending["model_rng"])
            self._pending_resume_state = None
        self.inference_allowed_mirroring_axes = None
        manifest = self.experiment_config()
        manifest["hostname"] = socket.gethostname()
        manifest["torch_version"] = torch.__version__
        manifest["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest_path = Path(self.output_folder) / "neighbor_ab_run.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["seed"] != self.neighbor_seed or previous["mode"] != self.neighbor_mode:
                raise ValueError("Output directory already contains a different seed/mode")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        self.print_to_log_file("Neighbor A/B configuration:", manifest)

    def perform_actual_validation(self, save_probabilities=False):
        configure_determinism()
        self.inference_allowed_mirroring_axes = None
        return super().perform_actual_validation(save_probabilities)

    def save_checkpoint(self, filename):
        if self.disable_checkpointing:
            return
        checkpoint = {
            "network_weights": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "lr_scheduler_state": self.lr_scheduler.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "logging": self.logger.get_checkpoint(), "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch + 1, "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__, "inference_allowed_mirroring_axes": None,
            "neighbor_ab": self.experiment_config(),
            "neighbor_ab_state": {
                "model_rng": capture_rng_state(),
                "train_stream": self.dataloader_train.state_dict(),
                "val_stream": self.dataloader_val.state_dict(),
            },
        }
        temporary = str(filename) + ".tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, filename)

    def load_checkpoint(self, filename_or_checkpoint):
        if not self.was_initialized:
            self.initialize()
        if isinstance(filename_or_checkpoint, (str, Path)):
            checkpoint = torch.load(filename_or_checkpoint, map_location="cpu", weights_only=False)
        else:
            checkpoint = filename_or_checkpoint
        saved_config = checkpoint.get("neighbor_ab")
        if saved_config is None or "neighbor_ab_state" not in checkpoint:
            raise ValueError("Checkpoint has no complete neighbor A/B provenance and RNG state")
        for key, value in self.experiment_config().items():
            if saved_config.get(key) != value:
                raise ValueError(f"Cannot resume with a different neighbor A/B {key}")
        self.network.load_state_dict(checkpoint["network_weights"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])
        if self.grad_scaler is not None and checkpoint["grad_scaler_state"] is not None:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.my_init_kwargs = checkpoint["init_args"]
        self.inference_allowed_mirroring_axes = None
        self._pending_resume_state = copy.deepcopy(checkpoint["neighbor_ab_state"])
        restore_rng_state(self._pending_resume_state["model_rng"])


class nnUNetTrainerNeighborA_500epochs(_nnUNetTrainerNeighborAB):
    neighbor_mode = "A"


class nnUNetTrainerNeighborB_500epochs(_nnUNetTrainerNeighborAB):
    neighbor_mode = "B"
