"""Score the 2D and 3D models on the same validation frames.

nnU-Net scores a 2D model frame by frame: each of the 218 validation frames gets its own
Dice, and summary.json averages them. It scores a 3D model video by video: one Dice per
video over all the scored frames of that video, averaged over the 44 videos. The two
averages weight frames differently, so they cannot be compared. This script cuts the 3D
prediction of each video back into its scored frames and scores every frame the way
nnU-Net scores a 2D frame. Then every model has one Dice per chamber on the same frames.

    uv run --locked python scripts/compare_models.py \
        --model-2d U-Net=outputs/nnunet/results/Dataset501_CardiacUDC2D/nnUNetTrainer_500epochs__nnUNetPlans__2d/fold_0/validation/summary.json \
        --model-2d VNet2D=outputs/nnunet/colab/summary.json \
        --model-3d VNet3D=outputs/nnunet/results/Dataset502_CardiacUDC3D/nnUNetTrainerVNet_500epochs__nnUNetPlans__3d_fullres/fold_0/validation

A 2D model is given by its validation summary.json. A 3D model is given by its validation
folder, which holds one predicted label video per case and nnU-Net's summary.json. The
script prints the mean Dice per chamber of each model, the same per source folder, and
the frames with the lowest Dice, and writes the Dice of every frame and model to
outputs/nnunet/compare_val/frame_dice.csv.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from nnunetv2.evaluation.evaluate_predictions import compute_tp_fp_fn_tn
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from export_nnunet import LABELS, NNUNET_DIR, ROOT, read_rows
from export_nnunet_3d import IGNORE_LABEL


MANIFEST_2D_PATH = NNUNET_DIR / "raw/Dataset501_CardiacUDC2D/frame_manifest.csv"
RAW_3D_DIR = NNUNET_DIR / "raw/Dataset502_CardiacUDC3D"
OUTPUT_DIR = ROOT / "outputs/nnunet/compare_val"

CHAMBERS = ["LV", "LA", "RA", "RV"]

# How many of the lowest-scoring frames to print per model.
WORST_COUNT = 5


def parse_model(text):
    """Turn "NAME=PATH" into (name, Path)."""
    name, path = text.split("=", 1)

    return name, Path(path)


def dice_from_counts(tp, fp, fn):
    """Return 2TP / (2TP + FP + FN), or NaN when a chamber is in neither label nor prediction.

    nnU-Net's compute_metrics uses the same rule.
    """
    if tp + fp + fn == 0:
        return np.nan

    return 2 * tp / (2 * tp + fp + fn)


def read_2d_dice(summary_path, frame_of_case):
    """Return {(record_id, frame_index): {chamber: Dice}} from a 2D model's summary.json.

    nnU-Net names each result by its reference file, such as
    ".../Site_G_100__patient1-4__f004.nii.gz"; frame_of_case maps that case name to its
    video and frame.
    """
    with summary_path.open() as stream:
        summary = json.load(stream)

    dice_of_frame = {}
    for result in summary["metric_per_case"]:
        case = Path(result["reference_file"]).name.removesuffix(".nii.gz")
        frame = frame_of_case[case]

        # summary.json stores the labels as text keys: "1" for LV, and so on.
        dice = {}
        for chamber in CHAMBERS:
            label_key = str(LABELS[chamber])
            dice[chamber] = result["metrics"][label_key]["Dice"]

        dice_of_frame[frame] = dice

    return dice_of_frame


def score_3d(validation_dir, frames_of_case):
    """Score a 3D model frame by frame.

    Returns (dice_of_frame, dice_of_video). dice_of_frame is
    {(record_id, frame_index): {chamber: Dice}}, one entry per scored frame.
    dice_of_video is {case: {chamber: Dice}}: the counts of a video's frames added up
    into one Dice, which is how nnU-Net scored the video.

    Labels and predictions are read with SimpleITKIO, the reader nnU-Net used for this
    dataset. It returns an array of shape (1, frames, width, height): SimpleITK puts the
    last NIfTI axis, time, first. So frame k is array[0, k].
    """
    reader = SimpleITKIO()

    dice_of_frame = {}
    dice_of_video = {}
    for case, frames in frames_of_case.items():
        reference, _ = reader.read_seg(str(RAW_3D_DIR / "labelsTr" / f"{case}.nii.gz"))
        prediction, _ = reader.read_seg(str(validation_dir / f"{case}.nii.gz"))

        # TP, FP and FN of each chamber, summed over the scored frames of this video.
        totals = {}
        for chamber in CHAMBERS:
            totals[chamber] = [0, 0, 0]

        for record_id, frame_index in frames:
            reference_frame = reference[0, frame_index]
            prediction_frame = prediction[0, frame_index]

            # A scored frame has no ignore pixels; kept so the counting matches nnU-Net's.
            ignore_mask = reference_frame == IGNORE_LABEL

            dice = {}
            for chamber in CHAMBERS:
                label = LABELS[chamber]
                tp, fp, fn, _ = compute_tp_fp_fn_tn(
                    reference_frame == label,
                    prediction_frame == label,
                    ignore_mask,
                )
                dice[chamber] = dice_from_counts(tp, fp, fn)

                totals[chamber][0] += tp
                totals[chamber][1] += fp
                totals[chamber][2] += fn

            dice_of_frame[(record_id, frame_index)] = dice

        video_dice = {}
        for chamber in CHAMBERS:
            tp, fp, fn = totals[chamber]
            video_dice[chamber] = dice_from_counts(tp, fp, fn)

        dice_of_video[case] = video_dice

    return dice_of_frame, dice_of_video


def check_against_summary(name, validation_dir, dice_of_video):
    """Print how far the per-video Dice rebuilt from frames is from nnU-Net's summary.json.

    nnU-Net scored each video on all its non-ignore pixels, which are exactly the scored
    frames. If the frames were cut along the wrong axis, the two would disagree.
    """
    with (validation_dir / "summary.json").open() as stream:
        summary = json.load(stream)

    largest_difference = 0.0
    for result in summary["metric_per_case"]:
        case = Path(result["reference_file"]).name.removesuffix(".nii.gz")

        for chamber in CHAMBERS:
            label_key = str(LABELS[chamber])
            nnunet_dice = result["metrics"][label_key]["Dice"]
            our_dice = dice_of_video[case][chamber]
            difference = abs(nnunet_dice - our_dice)
            largest_difference = max(largest_difference, difference)

    print(f"{name}: per-video Dice rebuilt from frames vs summary.json, largest difference {largest_difference:.6f}")


def frame_mean(dice):
    """Return the mean Dice of the four chambers of one frame."""
    values = []
    for chamber in CHAMBERS:
        values.append(dice[chamber])

    return float(np.nanmean(values))


def print_chamber_table(dice_of_model, frames):
    """Print the mean Dice per chamber of each model over the given frames."""
    header = f"{'model':<10}"
    for chamber in CHAMBERS:
        header += f"{chamber:>8}"
    header += f"{'mean':>8}"
    print(header)

    for name, dice_of_frame in dice_of_model.items():
        line = f"{name:<10}"

        chamber_means = []
        for chamber in CHAMBERS:
            values = []
            for frame in frames:
                values.append(dice_of_frame[frame][chamber])

            chamber_mean = float(np.nanmean(values))
            chamber_means.append(chamber_mean)
            line += f"{chamber_mean:>8.3f}"

        line += f"{np.mean(chamber_means):>8.3f}"
        print(line)


def print_folder_table(dice_of_model, frames):
    """Print each model's mean Dice (four chambers) per source folder, such as Site_G_100."""
    frames_of_folder = {}
    for frame in frames:
        record_id = frame[0]
        folder = record_id.split("/")[0]
        frames_of_folder.setdefault(folder, []).append(frame)

    header = f"{'folder':<14}{'frames':>7}"
    for name in dice_of_model:
        header += f"{name:>10}"
    print(header)

    for folder in sorted(frames_of_folder):
        folder_frames = frames_of_folder[folder]
        line = f"{folder:<14}{len(folder_frames):>7}"

        for dice_of_frame in dice_of_model.values():
            values = []
            for frame in folder_frames:
                values.append(frame_mean(dice_of_frame[frame]))

            line += f"{np.mean(values):>10.3f}"

        print(line)


def print_worst_frames(dice_of_model, frames):
    """Print the WORST_COUNT frames with the lowest mean Dice for each model."""
    for name, dice_of_frame in dice_of_model.items():
        scored = []
        for frame in frames:
            scored.append((frame_mean(dice_of_frame[frame]), frame))

        scored.sort()

        print(f"{name}:")
        for mean_dice, (record_id, frame_index) in scored[:WORST_COUNT]:
            dice = dice_of_frame[(record_id, frame_index)]

            chamber_parts = []
            for chamber in CHAMBERS:
                chamber_parts.append(f"{chamber} {dice[chamber]:.2f}")
            chamber_text = " ".join(chamber_parts)

            print(f"  {record_id} frame {frame_index}: mean {mean_dice:.3f} ({chamber_text})")


def write_frame_csv(dice_of_model, frames):
    """Write one row per frame and model to OUTPUT_DIR / frame_dice.csv."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "frame_dice.csv"

    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "record_id", "frame_index", *CHAMBERS, "mean"])

        for name, dice_of_frame in dice_of_model.items():
            for record_id, frame_index in frames:
                dice = dice_of_frame[(record_id, frame_index)]

                row = [name, record_id, frame_index]
                for chamber in CHAMBERS:
                    row.append(round(dice[chamber], 4))
                row.append(round(frame_mean(dice), 4))

                writer.writerow(row)

    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-2d", type=parse_model, action="append", default=[], help="NAME=path to a 2D validation summary.json")
    parser.add_argument("--model-3d", type=parse_model, action="append", default=[], help="NAME=path to a 3D validation folder")
    args = parser.parse_args()

    # Case names of the 2D dataset, such as "Site_G_100__patient1-4__f004", mapped to
    # (record_id, frame_index).
    frame_of_case = {}
    for row in read_rows(MANIFEST_2D_PATH):
        if row["split"] == "val":
            frame_of_case[row["case"]] = (row["record_id"], int(row["frame_index"]))

    # Videos of the 3D dataset, each with the list of its scored frames.
    frames_of_case = {}
    for row in read_rows(RAW_3D_DIR / "frame_manifest.csv"):
        if row["split"] == "val":
            frame = (row["record_id"], int(row["frame_index"]))
            frames_of_case.setdefault(row["case"], []).append(frame)

    dice_of_model = {}
    for name, summary_path in args.model_2d:
        dice_of_model[name] = read_2d_dice(summary_path, frame_of_case)

    for name, validation_dir in args.model_3d:
        dice_of_frame, dice_of_video = score_3d(validation_dir, frames_of_case)
        check_against_summary(name, validation_dir, dice_of_video)
        dice_of_model[name] = dice_of_frame

    # Compare only frames that every model scored; with baseline_v1 that is all 218.
    common_frames = set(frame_of_case.values())
    for name, dice_of_frame in dice_of_model.items():
        missing = common_frames - set(dice_of_frame)
        if missing:
            print(f"{name} has no score for {len(missing)} of the validation frames")
        common_frames &= set(dice_of_frame)

    frames = sorted(common_frames)
    print(f"\nMean Dice per chamber over {len(frames)} validation frames")
    print_chamber_table(dice_of_model, frames)

    print("\nMean Dice (four chambers) per source folder")
    print_folder_table(dice_of_model, frames)

    print(f"\nLowest-scoring {WORST_COUNT} frames per model")
    print_worst_frames(dice_of_model, frames)

    print()
    write_frame_csv(dice_of_model, frames)


if __name__ == "__main__":
    main()
