"""Inspect one A4C video and its stored labels without altering data."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
# A4C mapping from xmed-lab/GraphEcho, datasets/cardiac_uda.py, view_num == ['4'].
CLASS_NAMES = ["0: Background", "1: LV", "2: LA", "3: RA", "4: RV"]
COLORS = ["#000000", "#e45756", "#4c9be8", "#59b873", "#d9ae32"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", default="Site_G_100/patient1-4")
    parser.add_argument("--frame", type=int, help="Zero-based frame; default: most foreground pixels")
    args = parser.parse_args()

    # Resolve an image/label pair from the existing index.
    with (ROOT / "data/metadata/cases.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    record = next((row for row in records if row["record_id"] == args.record_id), None)
    if record is None:
        parser.error(f"Unknown record ID: {args.record_id}")
    if record["has_label_file"] != "true" or not record["label_path"]:
        parser.error("This record has no label file. Do not invent a background mask.")
    if not args.record_id.endswith("-4"):
        parser.error("Inspection supports the A4C (-4) label convention only.")

    # Validate array alignment and the supported label values.
    image_file = nib.load(ROOT / record["image_path"])
    label_file = nib.load(ROOT / record["label_path"])
    if image_file.shape != label_file.shape or len(image_file.shape) != 3:
        parser.error("Expected matching three-dimensional image/label arrays.")
    if not np.allclose(image_file.affine, label_file.affine):
        parser.error("Image and label affine transforms differ; inspect alignment first.")
    labels = np.asanyarray(label_file.dataobj)
    values, counts = np.unique(labels, return_counts=True)
    if not np.isin(values, np.arange(5)).all():
        parser.error(f"Unexpected labels {values.tolist()}; inspect their meaning before remapping.")

    # Nonzero pixels identify candidate frames, not proven complete masks.
    foreground_counts = np.count_nonzero(labels > 0, axis=(0, 1))
    candidate_frames = np.flatnonzero(foreground_counts)
    if args.frame is None:
        if candidate_frames.size == 0:
            parser.error("No foreground labels found; annotation status is unknown.")
        frame = int(foreground_counts.argmax())
    else:
        frame = args.frame
    if not 0 <= frame < image_file.shape[2]:
        parser.error(f"Frame must be between 0 and {image_file.shape[2] - 1}.")

    image = np.asarray(image_file.dataobj[:, :, frame], dtype=np.float32)
    mask = labels[:, :, frame]
    if not np.isfinite(image).all():
        parser.error("The selected image frame contains non-finite values.")

    # Use the same display transpose for the image and labels.
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 5.5), cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    fig.suptitle(f"{args.record_id} | frame {frame} (zero-based)", fontsize=14)
    axes[0].imshow(image.T, cmap="gray", interpolation="nearest")
    axes[0].set_title("Ultrasound frame")
    label_plot = axes[1].imshow(mask.T, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Stored labels")
    axes[2].imshow(image.T, cmap="gray", interpolation="nearest")
    axes[2].imshow(
        np.ma.masked_equal(mask.T, 0), cmap=cmap, norm=norm,
        interpolation="nearest", alpha=0.5,
    )
    axes[2].set_title("Image + nonzero labels")
    for ax in axes:
        ax.set_axis_off()
    colorbar = fig.colorbar(label_plot, ax=axes, ticks=range(5), shrink=0.65, pad=0.02)
    colorbar.ax.set_yticklabels(CLASS_NAMES)

    output = ROOT / "outputs/inspection" / record["source_folder"] / Path(record["image_path"]).name.removesuffix("_image.nii.gz")
    output.mkdir(parents=True, exist_ok=True)
    preview_path = output / f"frame_{frame:03d}.png"
    fig.savefig(preview_path, dpi=140)
    plt.close(fig)

    report = {
        "record_id": args.record_id,
        "image_path": record["image_path"],
        "label_path": record["label_path"],
        "array_shape": list(image_file.shape),
        "image_storage_dtype": str(image_file.get_data_dtype()),
        "label_storage_dtype": str(label_file.get_data_dtype()),
        "nifti_units": list(image_file.header.get_xyzt_units()),
        "label_value_counts": {str(int(v)): int(n) for v, n in zip(values, counts)},
        "frames_with_foreground": candidate_frames.tolist(),
        "foreground_pixels_per_frame": foreground_counts.tolist(),
        "selected_frame": frame,
        "selected_image_range": [float(image.min()), float(image.max())],
        "selected_frame_class_pixels": {name: int(np.count_nonzero(mask == i)) for i, name in enumerate(CLASS_NAMES)},
        "annotation_completeness": "not_established_by_nonzero_pixel_counts",
        "display_transform": "transpose both spatial arrays; no anatomical reorientation",
        "class_mapping_source": "https://github.com/xmed-lab/GraphEcho/blob/main/datasets/cardiac_uda.py",
        "preview": str(preview_path.relative_to(ROOT)),
    }
    report_path = output / f"frame_{frame:03d}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Record: {args.record_id}")
    print(f"Array shape: {image_file.shape}; axis 2 is treated as video frames")
    print(f"Stored label values: {values.tolist()}")
    print(f"Frames with foreground: {candidate_frames.size}/{image_file.shape[2]}")
    print(f"Zero-based candidate frame indices: {candidate_frames.tolist()}")
    print("Nonzero labels do not prove completeness; all-zero frames remain ambiguous.")
    print(f"Selected frame: {frame}")
    print(f"Preview: {preview_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
