"""Summarize paired Dice; shuffle replicates are averaged within target frame."""
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
METADATA = ROOT.parent / "seed2026_acceptance_20261031T080322Z/metrics_audit/paired_frame_metrics.csv"
LABELS = ("1", "2", "3", "4")


def read_scores(mode):
    folder = ROOT / mode
    summary = json.loads((folder / "summary.json").read_text())
    scores = {}
    for case in summary["metric_per_case"]:
        key = Path(case["prediction_file"]).name.removesuffix(".nii.gz")
        scores[key] = {label: case["metrics"][label]["Dice"] for label in LABELS}
    if len(scores) != 218 or not all(math.isfinite(v) for row in scores.values() for v in row.values()):
        raise ValueError(f"Incomplete scores in {mode}")
    records = json.loads((folder / "permutations.json").read_text())
    if set(records) != set(scores):
        raise ValueError(f"Permutation manifest does not match scores in {mode}")
    for record in records.values():
        order = record["channel_order"]
        if sorted(order) != list(range(8)) or order[4] != 4:
            raise ValueError(f"Invalid permutation in {mode}")
    observed = mean(mean(row.values()) for row in scores.values())
    if not math.isclose(observed, summary["foreground_mean"]["Dice"], abs_tol=1e-12):
        raise ValueError(f"Macro Dice mismatch in {mode}")
    return scores, observed, sum(record["input_changed"] for record in records.values())


def group_summary(rows, field, critical):
    groups = {}
    for row in rows:
        groups.setdefault(row[field], []).append(row["ordered_minus_shuffled"])
    differences = [mean(values) for values in groups.values()]
    average = mean(differences)
    half_width = critical * stdev(differences) / math.sqrt(len(differences))
    return {
        "groups": len(differences),
        "mean_dice_gap": average,
        "exploratory_95pct_t_interval": [average - half_width, average + half_width],
        "positive_groups": sum(gap > 0 for gap in differences),
    }


def main():
    completion = json.loads((ROOT / "complete.json").read_text())
    ordered, baseline, _ = read_scores("ordered")
    if completion["ordered_pixel_identical"] != 218 or not math.isclose(
        baseline, 0.8579541742528939, abs_tol=1e-12
    ):
        raise ValueError("Ordered baseline failed to reproduce the original")
    repeats = []
    trial_scores = []
    for i in range(1, completion["shuffle_repeats"] + 1):
        scores, dice, changed = read_scores(f"shuffle_{i}")
        if set(scores) != set(ordered):
            raise ValueError("Case set mismatch")
        trial_scores.append(scores)
        repeats.append({"repeat": i, "dice": dice, "ordered_minus_shuffled": baseline - dice,
                        "changed_inputs": changed})

    with METADATA.open() as stream:
        metadata = {row["case_id"]: row for row in csv.DictReader(stream)}
    if set(metadata) != set(ordered):
        raise ValueError("Metadata case set mismatch")
    rows = []
    for key, class_scores in sorted(ordered.items()):
        ordered_score = mean(class_scores.values())
        if not math.isclose(ordered_score, float(metadata[key]["B_Dice"]), abs_tol=1e-12):
            raise ValueError(f"Ordered case Dice mismatch: {key}")
        shuffled_score = mean(mean(scores[key].values()) for scores in trial_scores)
        rows.append({"case_id": key, "record_id": metadata[key]["record_id"],
                     "group_id": metadata[key]["group_id"], "ordered_dice": ordered_score,
                     "shuffled_dice": shuffled_score,
                     "ordered_minus_shuffled": ordered_score - shuffled_score})

    with (ROOT / "paired_dice.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "ordered_dice": baseline,
        "shuffled_mean_dice": mean(row["shuffled_dice"] for row in rows),
        "ordered_minus_shuffled": mean(row["ordered_minus_shuffled"] for row in rows),
        "cases_with_ordered_higher_dice": sum(row["ordered_minus_shuffled"] > 0 for row in rows),
        "cases_with_shuffled_higher_dice": sum(row["ordered_minus_shuffled"] < 0 for row in rows),
        "repeats": repeats,
        "video_equal_weight": group_summary(rows, "record_id", 2.016692199),
        "candidate_group_equal_weight": group_summary(rows, "group_id", 2.048407142),
        "foreground_classes": {
            label: {
                "ordered_dice": mean(ordered[k][label] for k in ordered),
                "shuffled_dice": mean(scores[k][label] for scores in trial_scores for k in ordered),
            }
            for label in LABELS
        },
    }
    (ROOT / "dice_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
