"""Register the experiment modules and its frozen plans in the current environment."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import nnunetv2


ROOT = Path(__file__).resolve().parents[1]
DATASET = 'Dataset503_CardiacUDCNeighborAB'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnunet-root', type=Path, required=True)
    args = parser.parse_args()
    package = Path(nnunetv2.__path__[0])
    destinations = {
        'nnunet_vnet_trainer.py': package / 'training/nnUNetTrainer',
        'nnunet_neighbor_ab_trainer.py': package / 'training/nnUNetTrainer',
        'neighbor_ab_preprocessor.py': package / 'preprocessing/preprocessors',
    }
    for name, folder in destinations.items():
        destination = folder / name
        if destination.is_symlink():
            destination.unlink()
        shutil.copy2(ROOT / 'scripts' / name, destination)

    plans = json.loads((ROOT / 'configs/neighbor_ab_base_plans.json').read_text())
    plans['dataset_name'] = DATASET
    config = plans['configurations']['2d']
    config['preprocessor_name'] = 'NeighborABPreprocessor'
    config['data_identifier'] = 'nnUNetPlans_2d_neighbor_ab_fullcanvas_v1'
    config['batch_size'] = 2
    config['patch_size'] = [768, 512]
    config['normalization_schemes'] = ['ZScoreNormalization'] * 8
    config['use_mask_for_norm'] = [False] * 8
    # Z-score ignores dataset-level statistics. Empty entries ensure no other case
    # or channel contributes to an image's normalization.
    plans['foreground_intensity_properties_per_channel'] = {str(index): {} for index in range(8)}
    destination = args.nnunet_root / 'preprocessed' / DATASET
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'nnUNetPlans.json').write_text(json.dumps(plans, indent=2) + '\n')
    raw_json = args.nnunet_root / 'raw' / DATASET / 'dataset.json'
    shutil.copy2(raw_json, destination / 'dataset.json')
    source_fingerprints = raw_json.parent / 'source_fingerprints.json'
    fingerprint = {
        'experiment': 'neighbor_ab_fullcanvas_v1',
        'planning': 'Frozen Dataset501 spatial spacing and orientation; no data-driven replanning',
        'normalization': 'Independent full-canvas image-only z-score; no dataset foreground statistics',
        'source_fingerprints_sha256': hashlib.sha256(source_fingerprints.read_bytes()).hexdigest(),
    }
    (destination / 'dataset_fingerprint.json').write_text(json.dumps(fingerprint, indent=2) + '\n')
    print(json.dumps({'registered_modules': list(destinations), 'plans': str(destination / 'nnUNetPlans.json')}))


if __name__ == '__main__':
    main()
