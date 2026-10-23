"""Turn the annotated frames of the baseline_v1 split into 2D training samples.

One sample is one annotated frame of one record:

    image:  [1, IMAGE_HEIGHT, IMAGE_WIDTH] float32, gray values scaled to 0..1
    target: [IMAGE_HEIGHT, IMAGE_WIDTH]    int64, one class per pixel
            (0 background, 1 LV, 2 LA, 3 RA, 4 RV)

Every sample has the same size, so a DataLoader can stack them into batches.

Only records with label_form "region" in data/splits/baseline_v1.csv are used, and of
those only the annotated frames that contain all four chambers (label_audit.csv). Raw
NIfTI files are only read, never modified.

Run it to check the pipeline on the first samples of a split:

    uv run --locked python scripts/dataset_2d.py --split train

Resizing to tensors (TODO 3 and 4 in to_tensors) is on hold while the 2D baseline runs
on nnU-Net, so this check stops at the first sample for now. export_nnunet.py uses only
build_frame_index and load_frame, which are done.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data/metadata/cases.csv"
AUDIT_PATH = ROOT / "data/metadata/label_audit.csv"
SPLIT_PATH = ROOT / "data/splits/baseline_v1.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/dataset_2d"

# Every frame is resized to this size. 384 x 288 keeps the 4:3 shape of all labeled
# frames (800 x 600, and 1024 x 768 or 640 x 480 in Site_R_52), and both sides can be
# halved five times (384 / 32 = 12, 288 / 32 = 9), as a U-Net does.
IMAGE_HEIGHT = 288
IMAGE_WIDTH = 384

# Filled regions are the training targets; contour labels would have to be filled first.
TARGET_LABEL_FORM = "region"

# 0 background, 1 LV, 2 LA, 3 RA, 4 RV (README, task definition), with the colours
# inspect_case.py uses for them.
CLASS_VALUES = [0, 1, 2, 3, 4]
CLASS_COLORS = ["#000000", "#e45756", "#4c9be8", "#59b873", "#d9ae32"]


def read_rows(path):
    """Return the rows of a CSV file as a list of dicts."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    return rows


def build_frame_index(split):
    """Return the samples of one split as a list of (record, frame_index) pairs.

    record is the cases.csv row of the record, a dict that holds image_path and
    label_path. frame_index is an int, counted from 0 along the last array axis.
    """
    case_of = {}
    for case in read_rows(CASES_PATH):
        case_of[case["record_id"]] = case

    audit_of = {}
    for audit in read_rows(AUDIT_PATH):
        audit_of[audit["record_id"]] = audit

    samples = []
    for split_row in read_rows(SPLIT_PATH):
        if split_row["split"] != split:
            continue

        if split_row["label_form"] != TARGET_LABEL_FORM:
            continue

        # annotated_frames and incomplete_frames are stored as JSON text such as
        # "[6, 7, 26, 27, 47]"; json.loads turns that text into a list of ints.
        record_id = split_row["record_id"]
        audit = audit_of[record_id]
        annotated_frames = json.loads(audit["annotated_frames"])
        incomplete_frames = json.loads(audit["incomplete_frames"])

        # A frame that misses one of the four chambers is left out. Example:
        # Site_G_100/patient40-4 (test) is annotated on [6, 7, 26, 27, 47] and frame 6
        # misses a class, so it gives 4 samples: frames 7, 26, 27, 47.
        for frame_index in annotated_frames:
            if frame_index in incomplete_frames:
                continue

            samples.append((case_of[record_id], frame_index))

    return samples


def load_frame(record, frame_index):
    """Return (image, labels) of one frame as numpy arrays of shape (H0, W0).

    H0 x W0 is the stored size, for example 600 x 800. The NIfTI arrays have shape
    (W0, H0, frames): the first axis runs along the width, so a frame is transposed to
    put rows first, the same way inspect_case.py displays it.
    """
    image_file = nib.load(ROOT / record["image_path"])
    label_file = nib.load(ROOT / record["label_path"])

    # dataobj can be sliced like an array and reads only that one frame, not the whole
    # video. A frame comes out as (W0, H0); .T turns it into (H0, W0), for example
    # (600, 800). image is uint8 (0..255); labels is int8 (0..4).
    image = np.asarray(image_file.dataobj[:, :, frame_index]).T
    labels = np.asarray(label_file.dataobj[:, :, frame_index]).T

    # A value outside 0..4 means the label needs renumbering (label_code_remap.csv),
    # which region labels never do; stop rather than train on a wrong class.
    label_values = np.unique(labels).tolist()
    for value in label_values:
        if value not in CLASS_VALUES:
            raise ValueError(f"{record['record_id']} frame {frame_index}: label value {value}")

    return image, labels


def to_tensors(image, labels):
    """Return (image_tensor, target) at IMAGE_HEIGHT x IMAGE_WIDTH.

    image (H0, W0) uint8 -> image_tensor [1, IMAGE_HEIGHT, IMAGE_WIDTH] float32 in 0..1
    labels (H0, W0) int8 -> target [IMAGE_HEIGHT, IMAGE_WIDTH] int64
    """
    # F.interpolate resizes tensors shaped [batch, channels, height, width]. It refuses
    # int8 and int64 tensors ("not implemented for 'Char'" / "'Long'"), so both arrays
    # become float32 tensors before resizing.
    output_size = (IMAGE_HEIGHT, IMAGE_WIDTH)

    # TODO 3: image_tensor = image as a tensor of shape [1, IMAGE_HEIGHT, IMAGE_WIDTH],
    #         float32, values 0..1. In four steps:
    #         1. turn image into a float32 tensor and scale 0..255 to 0..1;
    #            torch.tensor(image, dtype=torch.float32) copies the array.
    #            (torch.from_numpy would warn: arrays read through nibabel are read-only.)
    #         2. add two leading axes, so the shape is [1, 1, H0, W0]
    #         3. resize to output_size with F.interpolate, mode="bilinear",
    #            align_corners=False
    #         4. drop the leading batch axis again

    # TODO 4: target = labels resized the same way, shape [IMAGE_HEIGHT, IMAGE_WIDTH],
    #         int64, values only 0..4. Use mode="nearest-exact" instead of bilinear:
    #         bilinear would blend class 1 and class 4 into 2.5 along their border, which
    #         is no class at all. Convert to float32 first (see above) and back to int64
    #         (.long()) at the end.

    return image_tensor, target


class CardiacFrames2D(Dataset):
    """The annotated frames of one split, one sample per frame."""

    def __init__(self, split):
        self.split = split
        self.samples = build_frame_index(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record, frame_index = self.samples[index]
        image, labels = load_frame(record, frame_index)
        image_tensor, target = to_tensors(image, labels)

        return image_tensor, target


def save_preview(dataset, sample_count, output_path):
    """Save the first samples as image and overlay, to check them by eye."""
    figure, axes = plt.subplots(sample_count, 2, figsize=(8, 3 * sample_count), squeeze=False)
    colormap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], colormap.N)

    for index in range(sample_count):
        image_tensor, target = dataset[index]
        record, frame_index = dataset.samples[index]

        image = image_tensor[0].numpy()
        target_array = target.numpy()
        masked_target = np.ma.masked_equal(target_array, 0)

        axes[index, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[index, 0].set_title(f"{record['record_id']} frame {frame_index}", fontsize=8)

        axes[index, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[index, 1].imshow(masked_target, cmap=colormap, norm=norm, alpha=0.5)
        axes[index, 1].set_title("target over image", fontsize=8)

        for axis in axes[index]:
            axis.set_axis_off()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=100)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--preview", type=int, default=4, help="Samples in the preview image")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset = CardiacFrames2D(args.split)

    record_ids = set()
    for record, frame_index in dataset.samples:
        record_ids.add(record["record_id"])

    # With baseline_v1: train 970 frames from 194 records, val 218 from 44, test 217 from 44.
    print(f"{args.split}: {len(dataset)} frames from {len(record_ids)} records")

    # One sample: shapes, types and value ranges, and how long loading it takes.
    start = time.perf_counter()
    image_tensor, target = dataset[0]
    seconds = time.perf_counter() - start

    print(f"image  {tuple(image_tensor.shape)} {image_tensor.dtype} "
          f"min {image_tensor.min().item():.3f} max {image_tensor.max().item():.3f}")
    print(f"target {tuple(target.shape)} {target.dtype} values {torch.unique(target).tolist()}")
    print(f"loading one sample took {seconds:.2f} s")

    # A DataLoader stacks samples into a batch; this fails if sizes differ.
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    image_batch, target_batch = next(iter(loader))
    print(f"batch: images {tuple(image_batch.shape)}, targets {tuple(target_batch.shape)}")

    preview_count = min(args.preview, len(dataset))
    preview_path = args.output_dir / f"{args.split}_preview.png"
    save_preview(dataset, preview_count, preview_path)
    print(f"Saved {preview_path}")


if __name__ == "__main__":
    main()
