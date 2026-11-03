"""Write nnUNetPlans_vnet.json: Dataset502's nnU-Net plans plus two 8-frame configurations.

The 3d_fullres patch of Dataset502 is 32 frames x 256 x 192 pixels, about 14 % of a frame
(the median frame is 699 x 517 after cropping). The two new configurations trade frames
for area:

- 3d_t8_768x512: 8 frames x 768 x 512, the patch of the 2D models, almost a whole frame.
- 3d_t8_512x384: 8 frames x 512 x 384, about half a frame and half the memory.

Both inherit everything else from 3d_fullres, including its data_identifier, so they
train on the 3d_fullres data that is already preprocessed. The plans keep the name
nnUNetPlans in nnUNetPlans.json; this copy is named nnUNetPlans_vnet, and nnU-Net puts
that name in the result folder, so the new runs never mix with the old ones.
nnUNetPlans.json itself is not changed.

    uv run --locked python scripts/add_vnet_plans.py
    uv run --locked nnUNetv2_train 502 3d_t8_768x512 0 -tr nnUNetTrainerVNetAniso_500epochs -p nnUNetPlans_vnet

The nnUNet_* variables have to be set as in the README. Run this again after
nnUNetv2_plan_and_preprocess rewrites nnUNetPlans.json.
"""

import json

from export_nnunet_3d import PREPROCESSED_DIR


SOURCE_PLANS_PATH = PREPROCESSED_DIR / "nnUNetPlans.json"
PLANS_NAME = "nnUNetPlans_vnet"
TARGET_PLANS_PATH = PREPROCESSED_DIR / f"{PLANS_NAME}.json"

# [frames, height, width]. nnUNetTrainerVNetAniso_500epochs needs a multiple of 4 frames
# and multiples of 16 for height and width.
PATCH_SIZE_OF_CONFIGURATION = {
    "3d_t8_768x512": [8, 768, 512],
    "3d_t8_512x384": [8, 512, 384],
}

# The same batch size as 3d_fullres.
BATCH_SIZE = 2


def main():
    with SOURCE_PLANS_PATH.open() as stream:
        plans = json.load(stream)

    plans["plans_name"] = PLANS_NAME

    for name, patch_size in PATCH_SIZE_OF_CONFIGURATION.items():
        plans["configurations"][name] = {
            "inherits_from": "3d_fullres",
            "patch_size": patch_size,
            "batch_size": BATCH_SIZE,
        }

    with TARGET_PLANS_PATH.open("w") as stream:
        json.dump(plans, stream, indent=4)

    print(f"Wrote {TARGET_PLANS_PATH}")
    for name, patch_size in PATCH_SIZE_OF_CONFIGURATION.items():
        print(f"  {name}: patch {patch_size}, batch {BATCH_SIZE}")


if __name__ == "__main__":
    main()
