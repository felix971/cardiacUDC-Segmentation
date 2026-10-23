"""Find records that store the same ultrasound recording.

The pixel-by-pixel check needs equal shapes, so only records whose image arrays have the
same shape are compared, every pair within each shape group. A trimmed or resized copy
of a recording has a different shape and is not found. One export of a recording may have
screen text and markers set to 0, so pixels where either side is 0 are ignored: a pair
is the same recording when the pixels that are nonzero on both sides agree.

Writes every compared pair, with its evidence, to data/metadata/same_video_pairs.csv.
Raw NIfTI files are only read, never modified.
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data/metadata/cases.csv"
DEFAULT_OUTPUT_PATH = ROOT / "data/metadata/same_video_pairs.csv"

# Share of both-nonzero pixels that differ. Same recordings measured 0;
# different recordings measured about 0.65 or higher.
SAME_VIDEO_MAX_CONFLICT = 0.01

OUTPUT_COLUMNS = [
    "record_id_a",
    "record_id_b",
    "image_shape",
    "both_nonzero_pixels",
    "conflict_pixels",
    "conflict_fraction",
    "same_video",
]


def group_by_shape(records):
    """Return {image_shape text: [record, ...]} for shapes shared by two or more records."""
    groups = {}

    for record in records:
        shape_text = record["image_shape"]

        if shape_text not in groups:
            groups[shape_text] = []

        groups[shape_text].append(record)

    shared_groups = {}

    for shape_text, group_records in groups.items():
        if len(group_records) >= 2:
            shared_groups[shape_text] = group_records

    return shared_groups


def compare_images(image_a, image_b):
    """Return (both_nonzero_pixels, conflict_pixels) for two arrays of the same shape."""

    # Positions where image_a and image_b are both nonzero. A zero on either side may be
    # screen text that one export blanked out, so it is no evidence either way.
    a_nonzero = image_a != 0
    b_nonzero = image_b != 0
    both_nonzero_mask = a_nonzero & b_nonzero

    # Positions that are nonzero on both sides but differ. Compare with != directly; to
    # measure how much two values differ, cast first, because uint8 subtraction wraps
    # around (80 - 90 gives 246).
    values_differ = image_a != image_b
    conflict_mask = both_nonzero_mask & values_differ

    both_nonzero_pixels = np.count_nonzero(both_nonzero_mask)
    conflict_pixels = np.count_nonzero(conflict_mask)

    return both_nonzero_pixels, conflict_pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV file to write (default: data/metadata/same_video_pairs.csv)",
    )
    args = parser.parse_args()

    with CASES_PATH.open(newline="") as stream:
        records = list(csv.DictReader(stream))

    shape_groups = group_by_shape(records)
    print(f"{len(shape_groups)} image shapes are shared by two or more records")

    rows = []

    for shape_text, group_records in shape_groups.items():
        # Read each image once per group; a group holds at most about a dozen videos.
        images = []

        for record in group_records:
            image_file = nib.load(ROOT / record["image_path"])
            image = np.asarray(image_file.dataobj)
            images.append(image)

        group_size = len(group_records)

        # Visit each unordered pair once: (0, 1), (0, 2), ..., (1, 2), ...
        for index_a in range(group_size):
            for index_b in range(index_a + 1, group_size):
                both_nonzero_pixels, conflict_pixels = compare_images(
                    images[index_a],
                    images[index_b],
                )

                conflict_fraction = conflict_pixels / both_nonzero_pixels
                same_video = conflict_fraction <= SAME_VIDEO_MAX_CONFLICT

                row = {
                    "record_id_a": group_records[index_a]["record_id"],
                    "record_id_b": group_records[index_b]["record_id"],
                    "image_shape": shape_text,
                    "both_nonzero_pixels": both_nonzero_pixels,
                    "conflict_pixels": conflict_pixels,
                    "conflict_fraction": round(conflict_fraction, 4),
                    "same_video": "true" if same_video else "false",
                }
                rows.append(row)

    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"Compared {len(rows)} pairs; wrote {args.output}")

    print("Same recordings:")
    for row in rows:
        if row["same_video"] == "true":
            print(f"  {row['record_id_a']} = {row['record_id_b']}")

    # The smallest conflict among different recordings shows how safe the threshold is.
    smallest_different = None
    for row in rows:
        if row["same_video"] == "false":
            if smallest_different is None or row["conflict_fraction"] < smallest_different["conflict_fraction"]:
                smallest_different = row

    if smallest_different is not None:
        print(
            "Closest different pair: "
            f"{smallest_different['record_id_a']} vs {smallest_different['record_id_b']} "
            f"conflict {smallest_different['conflict_fraction']}"
        )


if __name__ == "__main__":
    main()
