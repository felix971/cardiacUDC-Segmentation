"""Write the annotated frames of baseline_v1 as an nnU-Net v2 dataset of 2D images.

nnU-Net plans the network, preprocesses and trains on its own, but it needs the data in
its own layout, and it does not know which frames come from the same video. This script
writes one 2D image and one label per frame:

    data/processed/nnunet/raw/Dataset501_CardiacUDC2D/
        imagesTr/<case>_0000.nii.gz   train and val frames
        labelsTr/<case>.nii.gz        their labels, 0..4
        imagesTs/<case>_0000.nii.gz   test frames
        labelsTs/<case>.nii.gz        test labels; nnU-Net never reads them, we use them
                                      to score the test predictions
        dataset.json
        frame_manifest.csv            one row per case: split, record, group and frame
    data/processed/nnunet/preprocessed/Dataset501_CardiacUDC2D/splits_final.json
        fold 0 = the baseline_v1 train and val frames, so frames of one group never sit
        on both sides

Only records with label_form "region" in data/splits/baseline_v1.csv are used, and of
those only the annotated frames that contain all four chambers (label_audit.csv). Raw
NIfTI files are only read, never modified; data/processed/ is ignored by git.

    uv run --locked python scripts/export_nnunet.py
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data/metadata/cases.csv"
AUDIT_PATH = ROOT / "data/metadata/label_audit.csv"
SPLIT_PATH = ROOT / "data/splits/baseline_v1.csv"
NNUNET_DIR = ROOT / "data/processed/nnunet"

# nnU-Net finds a dataset by its number; 501 and up are free for your own datasets.
DATASET_NAME = "Dataset501_CardiacUDC2D"
RAW_DIR = NNUNET_DIR / "raw" / DATASET_NAME
PREPROCESSED_DIR = NNUNET_DIR / "preprocessed" / DATASET_NAME

# Almost every frame is 800 wide. The 8 Site_R_52 records of 1024 x 768 and 640 x 480
# get a larger or smaller pixel size, so nnU-Net resamples them to the 800 x 600 grid,
# as if every frame showed the same field of view. The real pixel size is unknown.
REFERENCE_WIDTH = 800

# Filled regions are the training targets; contour labels would have to be filled first.
TARGET_LABEL_FORM = "region"

# The names nnU-Net reports per class; the numbers are the stored label values
# (README, task definition).
LABELS = {"background": 0, "LV": 1, "LA": 2, "RA": 3, "RV": 4}


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
        if value not in LABELS.values():
            raise ValueError(f"{record['record_id']} frame {frame_index}: label value {value}")

    return image, labels


def case_name(record, frame_index):
    """Return the nnU-Net case name of one frame, e.g. "Site_G_100__patient1-4__f004"."""
    prefix = Path(record["image_path"]).name.removesuffix("_image.nii.gz")
    name = f"{record['source_folder']}__{prefix}__f{frame_index:03d}"

    return name


def save_frame(image, labels, image_path, label_path):
    """Save one frame as a 2D NIfTI image and a 2D NIfTI label.

    image (H0, W0) uint8 and labels (H0, W0) int8 come from load_frame. Each pixel is
    stored as spacing x spacing in size, with spacing = REFERENCE_WIDTH / W0: 1.0 for an
    800-wide frame, 0.78125 for 1024, 1.25 for 640.
    """
    spacing = REFERENCE_WIDTH / image.shape[1]
    affine = np.diag([spacing, spacing, 1.0, 1.0])

    # Nifti1Image wraps an array and its affine; nib.save writes it to disk. Labels are
    # stored as uint8, the integer type nnU-Net expects for label files; 0..4 fit in it
    # unchanged.
    image_file = nib.Nifti1Image(image, affine)
    label_file = nib.Nifti1Image(labels.astype(np.uint8), affine)
    nib.save(image_file, image_path)
    nib.save(label_file, label_path)


def write_dataset_json(training_count):
    """Write dataset.json, which tells nnU-Net the channels, classes and file type."""
    dataset = {
        # The name of the one input channel. Any name other than "CT" makes nnU-Net
        # normalize each image to mean 0 and standard deviation 1.
        "channel_names": {"0": "ultrasound"},
        "labels": LABELS,
        "numTraining": training_count,
        "file_ending": ".nii.gz",
    }

    with (RAW_DIR / "dataset.json").open("w") as stream:
        json.dump(dataset, stream, indent=2)


def write_splits(train_cases, val_cases):
    """Write splits_final.json with one fold: baseline_v1 train against baseline_v1 val.

    Without this file nnU-Net splits imagesTr into 5 random folds by frame, and frames of
    one video would land in both training and validation.
    """
    splits = [{"train": sorted(train_cases), "val": sorted(val_cases)}]

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with (PREPROCESSED_DIR / "splits_final.json").open("w") as stream:
        json.dump(splits, stream, indent=2)


def write_manifest(rows):
    """Write frame_manifest.csv, which maps each case name back to its record and frame.

    nnU-Net reports results per case; with this table a case such as
    "Site_G_100__patient1-4__f004" can be traced to its video, group and split.
    """
    with (RAW_DIR / "frame_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="Delete an earlier export first")
    args = parser.parse_args()

    # Leftover files from an earlier run would become extra training cases.
    if RAW_DIR.exists():
        if not args.overwrite:
            raise SystemExit(f"{RAW_DIR} already exists; rerun with --overwrite to replace it")
        shutil.rmtree(RAW_DIR)

    for folder in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        (RAW_DIR / folder).mkdir(parents=True)

    group_of = {}
    for split_row in read_rows(SPLIT_PATH):
        group_of[split_row["record_id"]] = split_row["group_id"]

    cases_of = {}
    manifest_rows = []
    for split in ["train", "val", "test"]:
        if split == "test":
            image_dir, label_dir = RAW_DIR / "imagesTs", RAW_DIR / "labelsTs"
        else:
            image_dir, label_dir = RAW_DIR / "imagesTr", RAW_DIR / "labelsTr"

        cases_of[split] = []
        samples = build_frame_index(split)
        for count, (record, frame_index) in enumerate(samples, start=1):
            image, labels = load_frame(record, frame_index)
            name = case_name(record, frame_index)
            save_frame(image, labels, image_dir / f"{name}_0000.nii.gz", label_dir / f"{name}.nii.gz")
            cases_of[split].append(name)
            manifest_rows.append({
                "case": name,
                "split": split,
                "record_id": record["record_id"],
                "group_id": group_of[record["record_id"]],
                "frame_index": frame_index,
                "height": image.shape[0],
                "width": image.shape[1],
            })

            if count % 100 == 0 or count == len(samples):
                print(f"{split}: {count}/{len(samples)} frames written")

    write_dataset_json(len(cases_of["train"]) + len(cases_of["val"]))
    write_splits(cases_of["train"], cases_of["val"])
    write_manifest(manifest_rows)

    # With baseline_v1: train 970, val 218, test 217 frames.
    print(f"imagesTr: {len(cases_of['train'])} train + {len(cases_of['val'])} val frames")
    print(f"imagesTs: {len(cases_of['test'])} test frames")
    print(f"Wrote {RAW_DIR} and {PREPROCESSED_DIR / 'splits_final.json'}")


if __name__ == "__main__":
    main()
