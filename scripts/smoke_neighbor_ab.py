"""Bounded acceptance test using actual cached targets, real loss and full-size export."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnunet-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.nnunet_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(nnUNet_raw=str(root / 'raw'), nnUNet_preprocessed=str(root / 'preprocessed'),
                      nnUNet_results=str(output), NEIGHBOR_AB_SEED='2026', nnUNet_compile='false',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import nibabel as nib
    import numpy as np
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.inference.export_prediction import export_prediction_from_logits
    from nnunetv2.training.nnUNetTrainer.nnunet_neighbor_ab_trainer import (
        nnUNetTrainerNeighborA_500epochs, nnUNetTrainerNeighborB_500epochs, state_dict_sha256)
    torch.set_num_threads(8)
    dataset_name = 'Dataset503_CardiacUDCNeighborAB'
    preprocessed = root / 'preprocessed' / dataset_name
    plans = json.loads((preprocessed / 'nnUNetPlans.json').read_text())
    dataset_json = json.loads((preprocessed / 'dataset.json').read_text())
    report = {'kind': 'bounded_acceptance_not_formal_training', 'runs': {}, 'gpu': torch.cuda.get_device_name()}
    for mode, cls in [('A', nnUNetTrainerNeighborA_500epochs), ('B', nnUNetTrainerNeighborB_500epochs)]:
        run_plans = copy.deepcopy(plans)
        run_plans['continue_training'] = False
        trainer = cls(run_plans, '2d', 0, dataset_json, torch.device('cuda'))
        # Simulate the real CLI's override before entering the trainer's lifecycle.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        trainer.on_train_start()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        initial_hash = trainer.initial_state_sha256
        trainer.network.train()
        timings = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(8):
            torch.cuda.synchronize()
            started = time.monotonic()
            batch = next(trainer.dataloader_train)
            volume = trainer.network.prepare_volume(batch['data'])
            if mode == 'A':
                for frame in range(8):
                    assert torch.equal(volume[:, 0, frame], batch['data'][:, 4])
            loss = trainer.train_step(batch)['loss']
            assert np.isfinite(loss).all()
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            timings.append(seconds)
            print(json.dumps({'mode': mode, 'step': step, 'loss': float(loss), 'seconds': seconds}), flush=True)
        assert state_dict_sha256(trainer.network) != initial_hash
        trainer.network.eval()
        with torch.no_grad():
            online = trainer.validation_step(next(trainer.dataloader_val))
        assert np.isfinite(online['loss']).all()
        _, val_dataset = trainer.get_tr_and_val_datasets()
        identifier = sorted(json.loads((preprocessed / 'splits_final.json').read_text())[0]['val'])[0]
        data, seg, _, properties = val_dataset.load_case(identifier)
        predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False,
                                    perform_everything_on_device=True, device=torch.device('cuda'),
                                    verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(trainer.network, trainer.plans_manager, trainer.configuration_manager,
                                        None, dataset_json, cls.__name__, None)
        logits = predictor.predict_sliding_window_return_logits(torch.from_numpy(np.array(data, copy=True))).cpu()
        prediction_base = output / f'{mode}_prediction' / identifier
        prediction_base.parent.mkdir(exist_ok=True)
        export_prediction_from_logits(logits, properties, trainer.configuration_manager, trainer.plans_manager,
                                      dataset_json, str(prediction_base), False, num_threads_torch=2)
        prediction = nib.load(str(prediction_base) + '.nii.gz')
        reference = nib.load(root / 'raw' / dataset_name / 'labelsTr' / f'{identifier}.nii.gz')
        assert prediction.shape == reference.shape
        assert np.allclose(prediction.affine, reference.affine)
        assert set(np.unique(np.asarray(prediction.dataobj))).issubset({0, 1, 2, 3, 4})
        report['runs'][mode] = {
            'initial_state_sha256': initial_hash,
            'train_stream_audit_sha256': trainer.dataloader_train.audit_sha256,
            'val_stream_audit_sha256': trainer.dataloader_val.audit_sha256,
            'seconds_per_update_with_loader': sum(timings[2:]) / len(timings[2:]),
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
            'prediction_case': identifier, 'prediction_shape': list(prediction.shape),
            'prediction_labels': np.unique(np.asarray(prediction.dataobj)).tolist(),
            'last_loss': float(loss), 'deterministic': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        }
        del trainer, predictor, logits, batch, volume
        gc.collect()
        torch.cuda.empty_cache()
    for key in ('initial_state_sha256', 'train_stream_audit_sha256', 'val_stream_audit_sha256'):
        assert report['runs']['A'][key] == report['runs']['B'][key], key
    report['paired_invariants_passed'] = True
    report['training_only_pair_hours_estimate'] = sum(run['seconds_per_update_with_loader'] * 125000 for run in report['runs'].values()) / 3600
    (output / 'acceptance_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
