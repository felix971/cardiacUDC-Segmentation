"""Assign every record to train, val or test so that related videos stay in one split.

A model must not be tested on a video it was trained on, nor, as far as the file names can
tell, on another video of the same patient. Two relations between records are used:

1. Same recording: two records store the same video (data/metadata/same_video_pairs.csv).
2. Same patient number: for site R the folders clearly share one numbering (Site_R_52 holds
   numbers 1-52, Site_R_73 holds 53-128, and the numbers of Site_R_126 fall in 2-92); for
   site G (Site_G_100 1-100, Site_G_20 1-21, Site_G_29 1-29) it is unknown. Whether one
   number in two folders is one patient is not verified, so records of one site with the
   same number are kept together. Relation 2 cannot find one patient filed under two
   different numbers; relation 1 finds it only when the two records store the same
   recording (Site_R_126/patient-89-4 and patient-90-4).

Records joined by either relation form one group, and each group goes to one split as a
whole. Groups are split separately per site and per number of region-labeled records in
the group (1, 2 or 3), so val and test each get about 15% of the region-labeled records of
each site, not only of its groups. Groups without a region-labeled record cannot be scored,
so they all go to train. This is the in-distribution baseline; a cross-site split would
hold out a site.

Every record of cases.csv is listed with its label_form from label_audit.csv
("no_label_file" when it has no label file), so experiments that also use unlabeled
videos keep the same groups.

Reads cases.csv, label_audit.csv and same_video_pairs.csv. Writes the assignment to
data/splits/baseline_v1.csv, and the settings and the checksums of the inputs, this script
and the output to baseline_v1.json.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data/metadata/cases.csv"
AUDIT_PATH = ROOT / "data/metadata/label_audit.csv"
PAIRS_PATH = ROOT / "data/metadata/same_video_pairs.csv"
DEFAULT_OUTPUT_PATH = ROOT / "data/splits/baseline_v1.csv"

SEED = 20261022
VAL_PERCENT = 15
TEST_PERCENT = 15

# Filled regions are the training targets of the 2D baseline. Contour lines
# (label_all_frame) would have to be filled first, so they do not count here.
TARGET_LABEL_FORM = "region"

# File name of a record without its folder, for example "patient1-4", "patient-5-4" or
# "normal-23-4": a kind, a number, and the view suffix "-4" that every record has.
NAME_PATTERN = re.compile(r"^(patient|normal)-?(\d+)-4$")

OUTPUT_COLUMNS = [
    "record_id",
    "group_id",
    "site",
    "source_folder",
    "label_form",
    "split",
]


def read_rows(path):
    """Return the rows of a CSV file as a list of dicts."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    return rows


def file_sha256(path):
    """Return the SHA256 of a file, to record which input version made the split."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    return digest


def patient_key(case):
    """Return the key that joins records of one site with the same patient number.

    Site_G_100/patient1-4 and Site_G_20/patient-1-4 both give "G/patient-1". A record
    without a site in its folder name (label_all_frame) keeps its own record_id; it joins
    a site only through a same-recording pair.
    """
    record_id = case["record_id"]
    file_name = record_id.split("/")[1]

    match = NAME_PATTERN.match(file_name)
    if match is None:
        raise ValueError(f"Unexpected file name: {record_id}")

    kind = match.group(1)
    number = int(match.group(2))
    site = case["site_from_folder_name"]

    if site == "":
        return record_id

    return f"{site}/{kind}-{number}"


def build_groups(record_ids, links):
    """Return {record_id: group members}, joining the two records of every link.

    Every record starts as its own group. For each link, the two groups are merged; a
    record that appears in several links pulls all of them into one group.
    """
    group_of = {}
    for record_id in record_ids:
        group_of[record_id] = [record_id]

    for record_a, record_b in links:
        group_a = group_of[record_a]
        group_b = group_of[record_b]
        if group_a is group_b:
            continue

        merged = group_a + group_b
        for member in merged:
            group_of[member] = merged

    return group_of


def describe_group(members, case_of, label_form_of):
    """Return (group_id, site, region_count) for the records of one group.

    The group is named after the first of its members' patient keys in sorted() order.
    sorted() compares character codes, and capital letters come before small ones, so
    "R/patient-60" comes before "label_all_frame/patient-60-4": the group of
    Site_R_126/patient-60-4, Site_R_73/patient-60-4 and label_all_frame/patient-60-4 is
    named "R/patient-60". region_count is how many members have a region label.
    """
    member_keys = []
    member_sites = set()
    region_count = 0

    for member in members:
        member_case = case_of[member]
        member_keys.append(patient_key(member_case))

        if member_case["site_from_folder_name"] != "":
            member_sites.add(member_case["site_from_folder_name"])

        if label_form_of[member] == TARGET_LABEL_FORM:
            region_count = region_count + 1

    group_id = sorted(member_keys)[0]

    # Keys carry the site, and every same-recording pair found so far lies inside site R
    # (label_all_frame, which has no site, with Site_R_126, and Site_R_126/patient-89-4
    # with patient-90-4), so a group never spans two sites. Stop if that ever changes.
    if len(member_sites) > 1:
        raise ValueError(f"Group {group_id} joins several sites: {sorted(member_sites)}")

    if len(member_sites) == 1:
        site = member_sites.pop()
    else:
        site = "unknown"

    return group_id, site, region_count


def count_for_share(group_count, percent):
    """Return how many of group_count groups make percent %, rounded half up.

    Integer arithmetic: group_count * percent / 100 plus one half, cut down to a whole
    number. Python's round() would send halves to the even neighbour (round(2.5) == 2).
    """
    return (group_count * percent + 50) // 100


def assign_splits(group_ids, rng):
    """Shuffle the groups of one stratum and return {group_id: split}.

    group_ids is a sorted list. The first VAL_PERCENT % of the shuffled list goes to val,
    the next TEST_PERCENT % to test and the rest to train.
    """
    shuffled = list(group_ids)
    rng.shuffle(shuffled)

    val_count = count_for_share(len(shuffled), VAL_PERCENT)
    test_count = count_for_share(len(shuffled), TEST_PERCENT)

    split_of = {}
    for position, group_id in enumerate(shuffled):
        if position < val_count:
            split = "val"
        elif position < val_count + test_count:
            split = "test"
        else:
            split = "train"

        split_of[group_id] = split

    return split_of


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV file to write; the settings go next to it as .json "
        "(default: data/splits/baseline_v1.csv)",
    )
    args = parser.parse_args()

    cases = read_rows(CASES_PATH)
    audit_rows = read_rows(AUDIT_PATH)
    pair_rows = read_rows(PAIRS_PATH)

    case_of = {}
    for case in cases:
        case_of[case["record_id"]] = case

    # Records without a label file have no audit row.
    label_form_of = {}
    for case in cases:
        label_form_of[case["record_id"]] = "no_label_file"

    for audit in audit_rows:
        label_form_of[audit["record_id"]] = audit["label_form"]

    # Links of relation 2: link every record to the first record with the same patient key.
    records_of_key = {}
    for case in cases:
        key = patient_key(case)
        key_records = records_of_key.setdefault(key, [])
        key_records.append(case["record_id"])

    links = []
    for key_records in records_of_key.values():
        for index in range(1, len(key_records)):
            links.append((key_records[0], key_records[index]))

    # Links of relation 1: same-recording pairs.
    for pair in pair_rows:
        if pair["same_video"] == "true":
            links.append((pair["record_id_a"], pair["record_id_b"]))

    group_of = build_groups(case_of.keys(), links)

    # Name every record's group, and put each group once into its stratum. A stratum is
    # (site, region_count), for example ("G", 1) or ("R", 2).
    group_id_of = {}
    group_site_of = {}
    stratum_groups = {}
    for record_id in sorted(case_of):
        members = group_of[record_id]
        group_id, site, region_count = describe_group(members, case_of, label_form_of)
        group_id_of[record_id] = group_id

        # An earlier member already put this group into its stratum.
        if group_id in group_site_of:
            continue

        group_site_of[group_id] = site

        stratum = (site, region_count)
        groups_in_stratum = stratum_groups.setdefault(stratum, [])
        groups_in_stratum.append(group_id)

    # One random generator for the whole run, used stratum by stratum in sorted order, so
    # the same inputs and seed always give the same split. Reseeding it per stratum would
    # give a different split.
    rng = random.Random(SEED)
    split_of_group = {}
    strata_summary = {}
    for stratum in sorted(stratum_groups):
        site, region_count = stratum
        group_ids = sorted(stratum_groups[stratum])

        # A group without a region label cannot be scored in val or test, so it goes to
        # train, where its videos can still serve as unlabeled or contour data.
        if region_count == 0:
            stratum_splits = {}
            for group_id in group_ids:
                stratum_splits[group_id] = "train"
        else:
            stratum_splits = assign_splits(group_ids, rng)

        split_of_group.update(stratum_splits)

        stratum_counts = {"groups": len(group_ids), "train": 0, "val": 0, "test": 0}
        for split in stratum_splits.values():
            stratum_counts[split] = stratum_counts[split] + 1

        stratum_name = f"{site}, region-labeled records: {region_count}"
        strata_summary[stratum_name] = stratum_counts

    rows = []
    for record_id in sorted(case_of):
        group_id = group_id_of[record_id]
        row = {
            "record_id": record_id,
            "group_id": group_id,
            "site": group_site_of[group_id],
            "source_folder": case_of[record_id]["source_folder"],
            "label_form": label_form_of[record_id],
            "split": split_of_group[group_id],
        }
        rows.append(row)

    # The point of the groups: two records of one recording, or of one site and patient
    # number, must share a split. Check every link.
    split_of_record = {}
    for row in rows:
        split_of_record[row["record_id"]] = row["split"]

    for record_a, record_b in links:
        if split_of_record[record_a] != split_of_record[record_b]:
            raise ValueError(f"{record_a} and {record_b} landed in different splits")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    settings = {
        "seed": SEED,
        "val_percent": VAL_PERCENT,
        "test_percent": TEST_PERCENT,
        "group_basis": [
            "same recording: same_video=true in same_video_pairs.csv",
            "same site and same patient number in the file name (not verified to be one patient)",
        ],
        "stratified_by": "site, then the number of region-labeled records in the group; groups without one all go to train",
        "strata": strata_summary,
        "records": len(rows),
        "groups": len(split_of_group),
        # Only major.minor: the patch version does not change random.shuffle.
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "inputs_sha256": {
            "data/metadata/cases.csv": file_sha256(CASES_PATH),
            "data/metadata/label_audit.csv": file_sha256(AUDIT_PATH),
            "data/metadata/same_video_pairs.csv": file_sha256(PAIRS_PATH),
        },
        "script_sha256": file_sha256(Path(__file__)),
        "output_sha256": file_sha256(args.output),
    }
    settings_path = args.output.with_suffix(".json")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    print(f"Wrote {len(rows)} records in {len(split_of_group)} groups to {args.output}")
    print(f"Wrote settings to {settings_path}")

    print("Groups per stratum:")
    for stratum, counts in strata_summary.items():
        print(
            f"  {stratum:<34} {counts['groups']:>3} groups: "
            f"train {counts['train']}, val {counts['val']}, test {counts['test']}"
        )

    # Region labels are the training targets of the 2D baseline, so count them separately.
    print("Records per folder (region-labeled records in brackets):")
    print(f"  {'folder':<16}{'train':>12}{'val':>12}{'test':>12}")
    folders = []
    for row in rows:
        if row["source_folder"] not in folders:
            folders.append(row["source_folder"])

    for folder in sorted(folders):
        cells = []
        for split in ["train", "val", "test"]:
            all_count = 0
            region_records = 0
            for row in rows:
                if row["source_folder"] == folder and row["split"] == split:
                    all_count = all_count + 1
                    if row["label_form"] == "region":
                        region_records = region_records + 1

            cells.append(f"{all_count} ({region_records})")

        print(f"  {folder:<16}{cells[0]:>12}{cells[1]:>12}{cells[2]:>12}")


if __name__ == "__main__":
    main()
