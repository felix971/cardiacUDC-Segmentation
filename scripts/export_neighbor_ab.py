"""Export Dataset503's shared eight-channel train/val cache for neighbor-content A/B.

Raw videos are (width, height, time). Each exported channel is (height, width),
identical to export_nnunet.py. SimpleITK/nnU-Net subsequently reads (1, width,
height). Offsets are [-4,-3,-2,-1,0,1,2,3], independently clamped; channel 4 is
always the exact target. No test video or test label is opened.

Example (the source Dataset501 and raw videos stay read-only)::

    python scripts/export_neighbor_ab.py --source-root /path/to/original/repo \
        --output-root /path/to/ab/data/processed/nnunet
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

from export_nnunet import REFERENCE_WIDTH, case_name, read_rows


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "Dataset503_CardiacUDCNeighborAB"
OFFSETS = [-4, -3, -2, -1, 0, 1, 2, 3]
TARGET_CHANNEL = 4
DATA_IDENTIFIER = "nnUNetPlans_2d_neighbor_ab_fullcanvas_v1"
EXPECTED_COUNTS = {"train": 970, "val": 218}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def neighbor_indices(target_index, frame_count):
    """Clamp each offset, without moving the eight-frame window at boundaries."""
    if frame_count < 1 or target_index < 0 or target_index >= frame_count:
        raise ValueError("Target index must identify a frame in a nonempty video")
    indices = []
    for offset in OFFSETS:
        index = max(0, min(frame_count - 1, target_index + offset))
        indices.append(index)
    return indices


def target_samples(source_root):
    """Reuse baseline_v1's audited selection without visiting test records."""
    records = {}
    for record in read_rows(source_root / "data/metadata/cases.csv"):
        records[record["record_id"]] = record
    audits = {}
    for audit in read_rows(source_root / "data/metadata/label_audit.csv"):
        audits[audit["record_id"]] = audit

    samples = []
    groups = {"train": set(), "val": set()}
    for row in read_rows(source_root / "data/splits/baseline_v1.csv"):
        split = row["split"]
        if split not in ("train", "val") or row["label_form"] != "region":
            continue
        record = records[row["record_id"]]
        audit = audits[row["record_id"]]
        incomplete = set(json.loads(audit["incomplete_frames"]))
        groups[split].add(row["group_id"])
        for frame_index in json.loads(audit["annotated_frames"]):
            if frame_index in incomplete:
                continue
            samples.append({"record": record, "frame_index": frame_index, "split": split, "group_id": row["group_id"], "case": case_name(record, frame_index)})
    if groups["train"] & groups["val"]:
        raise ValueError("Training and validation groups overlap")
    return samples


def export_dataset(source_root, output_root, expected_counts=None, overwrite=False):
    """Write images/GT plus audited split, offset manifest and source fingerprints.

    The lower-level expected_counts argument supports small fixture tests; CLI
    runs always enforce the frozen 970/218 split and exact Dataset501 target IDs.
    Only one decompressed video/label pair is retained at once, so gzip decoding
    happens once per record rather than once per exported frame or channel.
    """
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if expected_counts is None:
        expected_counts = EXPECTED_COUNTS
    baseline_raw = source_root / "data/processed/nnunet/raw/Dataset501_CardiacUDC2D"
    baseline_splits_path = source_root / "data/processed/nnunet/preprocessed/Dataset501_CardiacUDC2D/splits_final.json"
    baseline_dataset = json.loads((baseline_raw / "dataset.json").read_text())
    baseline_splits = json.loads(baseline_splits_path.read_text())
    samples = target_samples(source_root)
    cases = {"train": [], "val": []}
    for sample in samples:
        cases[sample["split"]].append(sample["case"])
    counts = {}
    for split in ("train", "val"):
        counts[split] = len(cases[split])
        if len(set(cases[split])) != counts[split]:
            raise ValueError(f"Duplicate {split} target IDs")
        cases[split].sort()
        if cases[split] != sorted(baseline_splits[0][split]):
            raise ValueError(f"{split} target IDs differ from Dataset501")
    if counts != expected_counts:
        raise ValueError(f"Expected split counts {expected_counts}, found {counts}")
    if set(cases["train"]) & set(cases["val"]):
        raise ValueError("Training and validation targets overlap")

    raw_dir = output_root / "raw" / DATASET_NAME
    preprocessed_dir = output_root / "preprocessed" / DATASET_NAME
    if preprocessed_dir.exists() and any(preprocessed_dir.iterdir()):
        raise FileExistsError(f"{preprocessed_dir} already contains a split or cache; use a fresh output root")
    if raw_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{raw_dir} exists; use --overwrite to replace this export")
        shutil.rmtree(raw_dir)
    (raw_dir / "imagesTr").mkdir(parents=True)
    (raw_dir / "labelsTr").mkdir()

    allowed_labels = set(baseline_dataset["labels"].values())
    manifest = []
    fingerprints = []
    cached_record_id = None
    image_video = None
    label_video = None
    for count, sample in enumerate(samples, start=1):
        record = sample["record"]
        if record["record_id"] != cached_record_id:
            image_path = source_root / record["image_path"]
            label_path = source_root / record["label_path"]
            image_hash = sha256_file(image_path)
            label_hash = sha256_file(label_path)
            if image_hash != record["image_sha256"] or label_hash != record["label_sha256"]:
                raise ValueError(f"Source SHA256 mismatch for {record['record_id']}")
            image_video = np.asarray(nib.load(image_path).dataobj)
            label_video = np.asarray(nib.load(label_path).dataobj)
            if image_video.ndim != 3 or image_video.shape != label_video.shape:
                raise ValueError(f"Image/label video shape mismatch for {record['record_id']}")
            cached_record_id = record["record_id"]
            fingerprints.append({"record_id": cached_record_id, "image_path": record["image_path"], "label_path": record["label_path"], "image_sha256": image_hash, "label_sha256": label_hash, "shape_width_height_time": list(image_video.shape)})

        target_index = sample["frame_index"]
        indices = neighbor_indices(target_index, image_video.shape[2])
        target = image_video[:, :, target_index].T
        label = label_video[:, :, target_index].T
        if not set(np.unique(label).tolist()).issubset(allowed_labels):
            raise ValueError(f"Undeclared target label in {sample['case']}")
        spacing = REFERENCE_WIDTH / target.shape[1]
        affine = np.diag([spacing, spacing, 1.0, 1.0])
        for channel, frame_index in enumerate(indices):
            image = image_video[:, :, frame_index].T
            image_path = raw_dir / "imagesTr" / f"{sample['case']}_{channel:04d}.nii.gz"
            nib.save(nib.Nifti1Image(image, affine), image_path)
        label_path = raw_dir / "labelsTr" / f"{sample['case']}.nii.gz"
        nib.save(nib.Nifti1Image(label.astype(np.uint8), affine), label_path)
        manifest.append({"case": sample["case"], "split": sample["split"], "record_id": cached_record_id, "group_id": sample["group_id"], "frame_index": target_index, "neighbor_indices": json.dumps(indices), "target_channel": TARGET_CHANNEL, "height": target.shape[0], "width": target.shape[1]})
        if count % 100 == 0 or count == len(samples):
            print(f"Wrote {count}/{len(samples)} train/val targets", flush=True)

    channels = {}
    for channel, offset in enumerate(OFFSETS):
        channels[str(channel)] = f"ultrasound_offset_{offset:+d}"
    dataset = {"channel_names": channels, "labels": baseline_dataset["labels"], "numTraining": len(samples), "file_ending": ".nii.gz"}
    (raw_dir / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    (preprocessed_dir / "splits_final.json").write_text(json.dumps([cases], indent=2) + "\n")
    with (raw_dir / "frame_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    metadata_hashes = {}
    for relative_path in ("data/metadata/cases.csv", "data/metadata/label_audit.csv", "data/splits/baseline_v1.csv"):
        metadata_hashes[relative_path] = sha256_file(source_root / relative_path)
    provenance = {"dataset": DATASET_NAME, "source_root": str(source_root), "counts": counts, "offsets": OFFSETS, "target_channel": TARGET_CHANNEL, "clamping": "each target+offset independently to [0,T-1]", "orientation": "source [W,H,T]; exported [H,W]; nnUNet SimpleITK [1,W,H]", "normalization": "each frame full-canvas image-only z-score before independent spatial resampling", "data_identifier": DATA_IDENTIFIER, "test_data_accessed": False, "metadata_sha256": metadata_hashes, "baseline_splits_sha256": sha256_file(baseline_splits_path), "baseline_dataset_sha256": sha256_file(baseline_raw / "dataset.json"), "records": fingerprints}
    (raw_dir / "source_fingerprints.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed/nnunet")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export_dataset(args.source_root, args.output_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
