"""Audit every label file: readability, annotated frames, label codes and label form.

Reads data/metadata/cases.csv and writes one row per record that has a label file.
Raw NIfTI files are only read, never modified.
"""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data/metadata/cases.csv"
DEFAULT_OUTPUT_PATH = ROOT / "data/metadata/label_audit.csv"

# A frame counts as annotated only with at least this many nonzero label pixels.
# Stray marks left by the annotation tool have at most 60 pixels; real annotations
# have hundreds to thousands. GraphEcho uses the same threshold of 100.
MIN_ANNOTATION_PIXELS = 100

# Share of foreground pixels whose four neighbours carry the same class.
# Filled regions measure about 0.95 or higher; contour lines about 0.45-0.66.
REGION_MIN_INTERIOR_FRACTION = 0.8

CHAMBER_CLASSES = [1, 2, 3, 4]

OUTPUT_COLUMNS = [
    "record_id",
    "source_folder",
    "decode_ok",
    "total_frames",
    "annotated_count",
    "annotated_frames",
    "stray_frames",
    "label_values",
    "incomplete_frames",
    "interior_fraction",
    "label_form",
]


def measure_interior_fraction(labels, annotated_frames):
    """Return the share of foreground pixels that are surrounded by the same class.

    A pixel inside a filled region has the same class above, below, left and right.
    A pixel on a thin contour line usually does not.
    """
    foreground_total = 0
    interior_total = 0

    for frame_index in annotated_frames:
        frame_labels = labels[:, :, frame_index]
        frame_values = np.unique(frame_labels)

        for class_value in frame_values:
            if class_value == 0:
                continue

            class_mask = frame_labels == class_value

            # Shifted views of the same mask, all with shape (W - 2, H - 2).
            # Position [i, j] in each view looks at pixel [i + 1, j + 1] or one neighbour.
            center = class_mask[1:-1, 1:-1]
            up = class_mask[1:-1, :-2]
            down = class_mask[1:-1, 2:]
            left = class_mask[:-2, 1:-1]
            right = class_mask[2:, 1:-1]

            # TODO 5: interior_mask = pixels where center, up, down, left and right
            #         are all True (combine the five views with &)
            interior_mask = center & up & down & left & right

            foreground_total = foreground_total + np.count_nonzero(class_mask)
            interior_total = interior_total + np.count_nonzero(interior_mask)

    interior_fraction = interior_total / foreground_total
    return interior_fraction


def audit_record(record):
    """Return one output row for a record that has a label file."""
    label_path = ROOT / record["label_path"]
    label_file = nib.load(label_path)

    row = {
        "record_id": record["record_id"],
        "source_folder": record["source_folder"],
    }

    # One known label file is truncated; reading its pixels raises EOFError.
    try:
        labels = np.asarray(label_file.dataobj)
    except EOFError:
        row["decode_ok"] = "false"
        row["label_form"] = "unreadable"
        return row

    total_frames = labels.shape[2]

    # TODO 1: foreground_pixels = number of nonzero label pixels in each frame,
    #         shape (total_frames,)
    foreground_mask = labels > 0
    foreground_pixels = np.count_nonzero(foreground_mask, axis=(0, 1))

    # TODO 2: annotated_frames = frame indices with at least MIN_ANNOTATION_PIXELS
    #         stray_frames = frame indices with more than 0 but fewer than
    #         MIN_ANNOTATION_PIXELS nonzero pixels
    #         (both as 1-D arrays of frame indices)
    is_annotated = foreground_pixels >= MIN_ANNOTATION_PIXELS

    annotated_frames = np.flatnonzero(is_annotated)

    has_any_pixel = foreground_pixels > 0

    below_threshold = foreground_pixels < MIN_ANNOTATION_PIXELS

    is_stray = has_any_pixel & below_threshold

    stray_frames = np.flatnonzero(is_stray)

    # TODO 3: label_values = sorted unique values in the whole label array, as a list
    label_values = np.unique(labels).tolist()
    # TODO 4: incomplete_frames = annotated frames that miss at least one class in
    #         CHAMBER_CLASSES, as a list of frame indices
    incomplete_frames = []
    for frame_index in annotated_frames:
        frame_labels = labels[:, :, frame_index]
        frame_unique_values = np.unique(frame_labels)
        frame_values = frame_unique_values.tolist()

        for class_value in CHAMBER_CLASSES:
            if class_value not in frame_values:
                incomplete_frames.append(int(frame_index))
                break
            
    if annotated_frames.size == 0:
        interior_fraction = None
        label_form = "none"
    else:
        interior_fraction = measure_interior_fraction(labels, annotated_frames)

        if interior_fraction >= REGION_MIN_INTERIOR_FRACTION:
            label_form = "region"
        else:
            label_form = "contour"

    row["decode_ok"] = "true"
    row["total_frames"] = total_frames
    row["annotated_count"] = annotated_frames.size
    row["annotated_frames"] = json.dumps(annotated_frames.tolist())
    row["stray_frames"] = json.dumps(stray_frames.tolist())
    row["label_values"] = json.dumps(label_values)
    row["incomplete_frames"] = json.dumps(incomplete_frames)
    row["label_form"] = label_form

    if interior_fraction is not None:
        row["interior_fraction"] = round(interior_fraction, 4)

    return row


def print_summary(rows):
    """Print counts per folder and the records that need attention."""
    folder_forms = {}

    for row in rows:
        key = (row["source_folder"], row["label_form"])

        if key not in folder_forms:
            folder_forms[key] = 0

        folder_forms[key] = folder_forms[key] + 1

    print("Records per folder and label form:")
    for key in sorted(folder_forms):
        folder, label_form = key
        print(f"  {folder:16s} {label_form:10s} {folder_forms[key]}")

    print("Records that need attention:")
    for row in rows:
        notes = []

        if row["decode_ok"] == "false":
            notes.append("unreadable")
        else:
            if row["annotated_count"] != 5 and row["label_form"] != "contour":
                notes.append(f"{row['annotated_count']} annotated frames")

            if row["incomplete_frames"] != "[]":
                notes.append(f"incomplete frames {row['incomplete_frames']}")

            label_values = json.loads(row["label_values"])
            for value in label_values:
                if value not in [0, 1, 2, 3, 4]:
                    notes.append(f"label values {label_values}")
                    break

        if len(notes) > 0:
            print(f"  {row['record_id']}: {'; '.join(notes)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV file to write (default: data/metadata/label_audit.csv)",
    )
    parser.add_argument(
        "--record-id",
        help="Audit only this record and print its row; no CSV is written",
    )
    args = parser.parse_args()

    with CASES_PATH.open(newline="") as stream:
        records = list(csv.DictReader(stream))

    # Quick check on one record while writing the TODOs.
    if args.record_id is not None:
        for record in records:
            if record["record_id"] == args.record_id:
                row = audit_record(record)
                for column in OUTPUT_COLUMNS:
                    print(f"{column}: {row.get(column, '')}")
                return

        parser.error(f"Unknown record ID: {args.record_id}")

    rows = []

    for record in records:
        # CSV booleans are text; records without a label file are skipped.
        if record["has_label_file"] != "true":
            continue

        row = audit_record(record)
        rows.append(row)
        print(f"{row['record_id']}: {row['label_form']}")

    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.output}")
    print_summary(rows)


if __name__ == "__main__":
    main()
