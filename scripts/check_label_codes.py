"""Check what each stored label code means by comparing two versions of the same recording.

Two records that store the same recording (data/metadata/same_video_pairs.csv) show the
same heart on the same frames, so their labels should mark the same chambers. For each
such pair where both label files are readable, this script takes the frames annotated in
both versions and counts, for every nonzero code in one version, which code the other
version has at the same pixels.

label_all_frame stores contour lines and the other folders store filled regions. A contour
line lies on the region border, so roughly half of its pixels fall on background in the
other version. The meaning of a code is therefore the most frequent NONZERO code it falls on.

Reads cases.csv, label_audit.csv and same_video_pairs.csv; writes one row per pair and code
to data/metadata/label_code_crosstab.csv. Raw NIfTI files are only read, never modified.
"""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data/metadata/cases.csv"
AUDIT_PATH = ROOT / "data/metadata/label_audit.csv"
PAIRS_PATH = ROOT / "data/metadata/same_video_pairs.csv"
DEFAULT_OUTPUT_PATH = ROOT / "data/metadata/label_code_crosstab.csv"

OUTPUT_COLUMNS = [
    "source_record",
    "reference_record",
    "shared_frames",
    "source_code",
    "source_pixels",
    "reference_counts",
    "matched_code",
    "matched_share",
]


def read_rows(path):
    """Return the rows of a CSV file as a list of dicts."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    return rows


def choose_direction(audit_a, audit_b):
    """Return (source, reference): the codes of source are explained by reference.

    A contour version is explained by its region version, because a contour pixel falls
    inside or next to a region, while most region pixels are far from any contour line.
    """
    if audit_a["label_form"] == "contour" and audit_b["label_form"] != "contour":
        return audit_a, audit_b

    if audit_b["label_form"] == "contour" and audit_a["label_form"] != "contour":
        return audit_b, audit_a

    return audit_a, audit_b


def load_frames(record, frames):
    """Return the label array of a record restricted to the given frames, shape (W, H, F)."""
    label_file = nib.load(ROOT / record["label_path"])
    labels = np.asarray(label_file.dataobj)
    frame_labels = labels[:, :, frames]

    return frame_labels


def cross_tabulate(source_labels, reference_labels):
    """Return one dict per nonzero source code with its counts over reference codes."""
    source_values = np.unique(source_labels).tolist()
    reference_values = np.unique(reference_labels).tolist()

    results = []

    for source_code in source_values:
        if source_code == 0:
            continue

        source_mask = source_labels == source_code
        source_pixels = np.count_nonzero(source_mask)

        reference_counts = {}

        for reference_code in reference_values:
            reference_mask = reference_labels == reference_code
            overlap_mask = source_mask & reference_mask
            reference_counts[reference_code] = int(np.count_nonzero(overlap_mask))

        # Background is expected for contour pixels, so the match ignores code 0.
        matched_code = None
        matched_pixels = 0
        nonzero_pixels = 0

        for reference_code, count in reference_counts.items():
            if reference_code == 0:
                continue

            nonzero_pixels = nonzero_pixels + count

            if count > matched_pixels:
                matched_code = reference_code
                matched_pixels = count

        if nonzero_pixels > 0:
            matched_share = round(matched_pixels / nonzero_pixels, 4)
        else:
            matched_share = None

        results.append(
            {
                "source_code": source_code,
                "source_pixels": int(source_pixels),
                "reference_counts": reference_counts,
                "matched_code": matched_code,
                "matched_share": matched_share,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV file to write (default: data/metadata/label_code_crosstab.csv)",
    )
    args = parser.parse_args()

    records = {}
    for record in read_rows(CASES_PATH):
        records[record["record_id"]] = record

    audits = {}
    for audit in read_rows(AUDIT_PATH):
        audits[audit["record_id"]] = audit

    rows = []

    for pair in read_rows(PAIRS_PATH):
        if pair["same_video"] != "true":
            continue

        record_id_a = pair["record_id_a"]
        record_id_b = pair["record_id_b"]

        # Both versions need a readable label file with annotated frames.
        audit_a = audits.get(record_id_a)
        audit_b = audits.get(record_id_b)

        if audit_a is None or audit_b is None:
            print(f"Skip {record_id_a} = {record_id_b}: a label file is missing")
            continue

        if audit_a["decode_ok"] != "true" or audit_b["decode_ok"] != "true":
            print(f"Skip {record_id_a} = {record_id_b}: a label file is unreadable")
            continue

        source_audit, reference_audit = choose_direction(audit_a, audit_b)

        source_frames = json.loads(source_audit["annotated_frames"])
        reference_frames = json.loads(reference_audit["annotated_frames"])

        shared_frames = []
        for frame_index in source_frames:
            if frame_index in reference_frames:
                shared_frames.append(frame_index)

        if len(shared_frames) == 0:
            print(
                f"Skip {source_audit['record_id']} = {reference_audit['record_id']}: "
                "no frame is annotated in both versions"
            )
            continue

        source_labels = load_frames(records[source_audit["record_id"]], shared_frames)
        reference_labels = load_frames(records[reference_audit["record_id"]], shared_frames)

        results = cross_tabulate(source_labels, reference_labels)

        summary = []

        for result in results:
            row = {
                "source_record": source_audit["record_id"],
                "reference_record": reference_audit["record_id"],
                "shared_frames": json.dumps(shared_frames),
                "source_code": result["source_code"],
                "source_pixels": result["source_pixels"],
                "reference_counts": json.dumps(result["reference_counts"]),
                "matched_code": result["matched_code"],
                "matched_share": result["matched_share"],
            }
            rows.append(row)

            summary.append(
                f"{result['source_code']}->{result['matched_code']} ({result['matched_share']})"
            )

        print(
            f"{source_audit['record_id']} vs {reference_audit['record_id']}, "
            f"frames {shared_frames}: {', '.join(summary)}"
        )

    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.output}")

    print("Codes that do not keep their own number:")
    for row in rows:
        if row["source_code"] != row["matched_code"]:
            print(
                f"  {row['source_record']}: {row['source_code']} -> {row['matched_code']} "
                f"(share {row['matched_share']})"
            )


if __name__ == "__main__":
    main()
