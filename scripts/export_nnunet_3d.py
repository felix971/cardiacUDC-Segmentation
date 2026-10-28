"""Write the region videos of baseline_v1 as an nnU-Net v2 dataset of 3D volumes.

The 2D export (export_nnunet.py) turns each annotated frame into its own image. Here a
whole video is one case, an array of height x width x frames, so a 3D network can use
the frames before and after an annotated one:

    data/processed/nnunet/raw/Dataset502_CardiacUDC3D/
        imagesTr/<case>_0000.nii.gz   train and val videos
        labelsTr/<case>.nii.gz        their labels: 0..4 on the annotated frames, 5
                                      ("ignore") on every other frame
        imagesTs/<case>_0000.nii.gz   test videos
        labelsTs/<case>.nii.gz        test labels, built the same way
        dataset.json
        frame_manifest.csv            one row per annotated frame that is scored
    data/processed/nnunet/preprocessed/Dataset502_CardiacUDC3D/splits_final.json
        fold 0 = the baseline_v1 train and val videos

Only about 5 of every 100 frames are annotated. A frame without annotation is all zeros
in the label file, which would read as "all background"; nnU-Net skips pixels marked
with the ignore label in the loss and in its scores, while the network still sees the
image of those frames. Frames that miss a chamber are marked ignore too, as in 2D.

    uv run --locked python scripts/export_nnunet_3d.py
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from export_nnunet import (
    AUDIT_PATH,
    CASES_PATH,
    LABELS,
    NNUNET_DIR,
    REFERENCE_WIDTH,
    ROOT,
    SPLIT_PATH,
    TARGET_LABEL_FORM,
    read_rows,
)


# 502, next to the 2D dataset 501.
DATASET_NAME = "Dataset502_CardiacUDC3D"
RAW_DIR = NNUNET_DIR / "raw" / DATASET_NAME
PREPROCESSED_DIR = NNUNET_DIR / "preprocessed" / DATASET_NAME

# nnU-Net treats the third axis as space and has no notion of time, so the step between
# frames is a choice. With 4 (pixels are about 1 wide), the planned network keeps every
# frame through its first stages and only then halves along time. On videos of this
# size the 3d_fullres patch is 32 frames x 256 x 192 pixels; it is much shorter along
# time than across, so nnU-Net rotates images only within the image plane.
FRAME_SPACING = 4.0

# The ignore label has to be the largest label value.
IGNORE_LABEL = 5
LABELS_3D = dict(LABELS)
LABELS_3D["ignore"] = IGNORE_LABEL


def build_video_index(split):
    """Return the videos of one split as a list of (record, scored_frames) pairs.

    record is the cases.csv row of the video. scored_frames lists the annotated frames
    that contain all four chambers, the same frames export_nnunet.py uses in 2D.
    """
    case_of = {}
    for case in read_rows(CASES_PATH):
        case_of[case["record_id"]] = case

    audit_of = {}
    for audit in read_rows(AUDIT_PATH):
        audit_of[audit["record_id"]] = audit

    videos = []
    for split_row in read_rows(SPLIT_PATH):
        if split_row["split"] != split:
            continue

        if split_row["label_form"] != TARGET_LABEL_FORM:
            continue

        record_id = split_row["record_id"]
        audit = audit_of[record_id]
        annotated_frames = json.loads(audit["annotated_frames"])
        incomplete_frames = json.loads(audit["incomplete_frames"])

        scored_frames = []
        for frame_index in annotated_frames:
            if frame_index in incomplete_frames:
                continue

            scored_frames.append(frame_index)

        videos.append((case_of[record_id], scored_frames))

    return videos


def load_video(record, scored_frames):
    """Return (image, labels) of one video as numpy arrays of shape (H0, W0, T).

    The NIfTI arrays have shape (W0, H0, T); swapping the first two axes puts rows first,
    so frame k of the result, image[:, :, k], is the same picture as in the 2D export.
    image is uint8. labels is uint8: the stored values 0..4 on the scored frames and
    IGNORE_LABEL everywhere else.
    """
    image_file = nib.load(ROOT / record["image_path"])
    label_file = nib.load(ROOT / record["label_path"])

    image = np.asarray(image_file.dataobj).transpose(1, 0, 2)
    stored_labels = np.asarray(label_file.dataobj).transpose(1, 0, 2)

    labels = np.full(stored_labels.shape, IGNORE_LABEL, dtype=np.uint8)
    for frame_index in scored_frames:
        frame_labels = stored_labels[:, :, frame_index]

        # A value outside 0..4 means the label needs renumbering (label_code_remap.csv),
        # which region labels never do; stop rather than train on a wrong class.
        label_values = np.unique(frame_labels).tolist()
        for value in label_values:
            if value not in LABELS.values():
                raise ValueError(f"{record['record_id']} frame {frame_index}: label value {value}")

        labels[:, :, frame_index] = frame_labels

    return image, labels


def case_name(record):
    """Return the nnU-Net case name of one video, e.g. "Site_G_100__patient1-4"."""
    prefix = Path(record["image_path"]).name.removesuffix("_image.nii.gz")
    name = f"{record['source_folder']}__{prefix}"

    return name


def save_video(image, labels, image_path, label_path):
    """Save one video as a 3D NIfTI image and a 3D NIfTI label.

    A pixel is spacing x spacing in size, with spacing = REFERENCE_WIDTH / W0 as in 2D,
    and one frame is FRAME_SPACING long.
    """
    spacing = REFERENCE_WIDTH / image.shape[1]
    affine = np.diag([spacing, spacing, FRAME_SPACING, 1.0])

    image_file = nib.Nifti1Image(image, affine)
    label_file = nib.Nifti1Image(labels, affine)
    nib.save(image_file, image_path)
    nib.save(label_file, label_path)


def write_dataset_json(training_count):
    """Write dataset.json; LABELS_3D adds the ignore label to the four chambers."""
    dataset = {
        "channel_names": {"0": "ultrasound"},
        "labels": LABELS_3D,
        "numTraining": training_count,
        "file_ending": ".nii.gz",
    }

    with (RAW_DIR / "dataset.json").open("w") as stream:
        json.dump(dataset, stream, indent=2)


def write_splits(train_cases, val_cases):
    """Write splits_final.json with one fold: baseline_v1 train against baseline_v1 val."""
    splits = [{"train": sorted(train_cases), "val": sorted(val_cases)}]

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with (PREPROCESSED_DIR / "splits_final.json").open("w") as stream:
        json.dump(splits, stream, indent=2)


def write_manifest(rows):
    """Write frame_manifest.csv, one row per scored frame.

    The 3D results are compared with the 2D ones frame by frame: this table says which
    frames of which video are the scored ones.
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
        frame_count = 0
        videos = build_video_index(split)
        for count, (record, scored_frames) in enumerate(videos, start=1):
            image, labels = load_video(record, scored_frames)
            name = case_name(record)
            save_video(image, labels, image_dir / f"{name}_0000.nii.gz", label_dir / f"{name}.nii.gz")
            cases_of[split].append(name)
            frame_count += len(scored_frames)

            for frame_index in scored_frames:
                manifest_rows.append({
                    "case": name,
                    "split": split,
                    "record_id": record["record_id"],
                    "group_id": group_of[record["record_id"]],
                    "frame_index": frame_index,
                    "frames": image.shape[2],
                    "height": image.shape[0],
                    "width": image.shape[1],
                })

            if count % 20 == 0 or count == len(videos):
                print(f"{split}: {count}/{len(videos)} videos written")

        # With baseline_v1: train 194 videos / 970 frames, val 44 / 218, test 44 / 217.
        print(f"{split}: {len(cases_of[split])} videos, {frame_count} scored frames")

    write_dataset_json(len(cases_of["train"]) + len(cases_of["val"]))
    write_splits(cases_of["train"], cases_of["val"])
    write_manifest(manifest_rows)

    print(f"Wrote {RAW_DIR} and {PREPROCESSED_DIR / 'splits_final.json'}")


if __name__ == "__main__":
    main()
