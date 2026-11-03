"""Re-evaluate the frozen seed2026 B checkpoint under frame permutations."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.run.run_training import get_trainer_from_args

TARGET = 4
IDENTITY = list(range(8))
EXPECTED_SHA = "a3cfc20c8803d7c9f2f8cf6c1d10bcd03ab9a7452381c1cc31bd1104aa09dc5c"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parent
    repo = Path.cwd()
    dataset = "Dataset503_CardiacUDCNeighborAB"
    trainer_name = "nnUNetTrainerNeighborB_500epochs"
    original = (repo / "outputs/neighbor_ab/formal/seed_2026" / dataset /
                (trainer_name + "__nnUNetPlans__2d") / "fold_0")
    checkpoint = original / "checkpoint_final.pth"
    if digest(checkpoint) != EXPECTED_SHA:
        raise RuntimeError("Checkpoint differs from the accepted seed2026 checkpoint")
    reference = original / "validation"
    original_summary_sha = digest(reference / "summary.json")
    trainer = get_trainer_from_args("503", "2d", 0, trainer_name, "nnUNetPlans", False,
                                    device=torch.device("cuda"))
    trainer.load_checkpoint(str(checkpoint))
    trainer.disable_checkpointing = True
    _, keys = trainer.do_split()
    keys = sorted(keys)
    if len(keys) != 218 or set(keys) != {p.name[:-7] for p in reference.glob("*.nii.gz")}:
        raise RuntimeError("Wrong validation split")

    original_predict = nnUNetPredictor.predict_sliding_window_return_logits
    for repeat in range(6):
        mode = "ordered" if repeat == 0 else f"shuffle_{repeat}"
        output = root / mode
        output.mkdir()
        trainer.output_folder = str(output)
        records = {}
        print(f"START {mode}", flush=True)

        def predict(predictor, data):
            case = keys[len(records)]
            if data.ndim != 4 or data.shape[0] != 8:
                raise RuntimeError(f"Unexpected input shape: {data.shape}")
            order = IDENTITY.copy()
            changed = False
            if repeat:
                slots = [i for i in IDENTITY if i != TARGET]
                seed = hashlib.sha256(f"seed2026-frame-order-v1:{repeat}:{case}".encode()).digest()
                rng = random.Random(int.from_bytes(seed, "big"))
                shuffled = slots.copy()
                while shuffled == slots:
                    rng.shuffle(shuffled)
                for slot, source in zip(slots, shuffled):
                    order[slot] = source
                permuted = data[order]
                changed = not torch.equal(data, permuted)
                data = permuted
            records[case] = {"channel_order": order, "input_changed": changed}
            return original_predict(predictor, data)

        nnUNetPredictor.predict_sliding_window_return_logits = predict
        try:
            trainer.perform_actual_validation()
        finally:
            nnUNetPredictor.predict_sliding_window_return_logits = original_predict
        validation = output / "validation"
        if set(records) != set(keys) or set(keys) != {p.name[:-7] for p in validation.glob("*.nii.gz")}:
            raise RuntimeError("Validation did not predict every intended case")
        (output / "permutations.json").write_text(json.dumps(records, indent=2) + "\n")
        if repeat == 0:
            for key in keys:
                filename = key + ".nii.gz"
                actual = sitk.GetArrayFromImage(sitk.ReadImage(str(validation / filename)))
                expected = sitk.GetArrayFromImage(sitk.ReadImage(str(reference / filename)))
                if not np.array_equal(actual, expected):
                    raise RuntimeError(f"Ordered baseline pixel mismatch: {key}")
            print("ORDERED_REFERENCE: 218/218 pixel-identical masks", flush=True)
        print(f"DONE {mode}", flush=True)
    if digest(checkpoint) != EXPECTED_SHA or digest(reference / "summary.json") != original_summary_sha:
        raise RuntimeError("Original checkpoint or summary changed")
    (root / "complete.json").write_text(json.dumps({
        "checkpoint_sha256": EXPECTED_SHA, "cases": keys,
        "ordered_pixel_identical": 218, "shuffle_repeats": 5,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
