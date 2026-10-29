"""Run one hardware-matched A/B pair, stopping on any failure and preserving outputs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

import torch


ROOT = Path(__file__).resolve().parents[1]
DATASET = 'Dataset503_CardiacUDCNeighborAB'
TRAINERS = {'A': 'nnUNetTrainerNeighborA_500epochs', 'B': 'nnUNetTrainerNeighborB_500epochs'}


def execution_configuration():
    value = os.environ.get('NEIGHBOR_AB_PREFETCH', '0')
    if value not in ('0', '2'):
        raise ValueError('NEIGHBOR_AB_PREFETCH must be 0 (serial) or 2 (prefetch)')
    return {'prefetch': int(value), 'cpu_threads': 8}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify_finished(fold, expected, mode, seed):
    checkpoint_path = fold / 'checkpoint_final.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['current_epoch'] != 500 or checkpoint['trainer_name'] != TRAINERS[mode] or not checkpoint['network_weights']:
        raise ValueError('Final checkpoint does not match the frozen 500-epoch trainer')
    config = checkpoint['neighbor_ab']
    if config['seed'] != seed or config['mode'] != mode:
        raise ValueError('Checkpoint seed/mode differs from the requested pair')
    streams = checkpoint['neighbor_ab_state']
    if streams['train_stream']['batches_seen'] != 125000 or streams['val_stream']['batches_seen'] != 25000:
        raise ValueError('Checkpoint has an incomplete sampling/augmentation stream')
    summary_path = fold / 'validation/summary.json'
    summary = json.loads(summary_path.read_text())
    if summary_path.stat().st_mtime < checkpoint_path.stat().st_mtime:
        raise ValueError('Validation predates the final checkpoint')
    cases = summary['metric_per_case']
    identifiers = []
    class_dice = {label: [] for label in ('1', '2', '3', '4')}
    for case in cases:
        name = Path(case['prediction_file']).name
        prediction = summary_path.parent / name
        if not name.endswith('.nii.gz') or not prediction.is_file() or prediction.stat().st_size == 0:
            raise ValueError('Missing or empty validation prediction')
        if set(case['metrics']) != {'1', '2', '3', '4'}:
            raise ValueError('Validation must include every foreground class')
        for label, values in class_dice.items():
            value = case['metrics'][label]['Dice']
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid per-case Dice')
            values.append(value)
        identifiers.append(name[:-7])
    if len(identifiers) != 218 or len(set(identifiers)) != 218 or set(identifiers) != set(expected):
        raise ValueError('Validation does not cover exactly the frozen 218 target IDs')
    actual_files = {path.name[:-7] for path in summary_path.parent.glob('*.nii.gz')}
    if actual_files != set(expected):
        raise ValueError('Validation directory contains missing or extra predictions')
    dice = summary['foreground_mean']['Dice']
    if not math.isfinite(dice) or not 0 <= dice <= 1:
        raise ValueError('Validation Dice is invalid')
    means = {}
    for label, values in class_dice.items():
        means[label] = sum(values) / len(values)
        if not math.isclose(means[label], summary['mean'][label]['Dice'], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('Per-class Dice aggregate disagrees with per-case results')
    if not math.isclose(sum(means.values()) / 4, dice, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('Foreground Dice aggregate disagrees with per-case results')
    return {'foreground_mean': summary['foreground_mean'], 'mean': summary['mean'], 'count': len(identifiers),
            'initial_state_sha256': config['initial_state_sha256'],
            'initial_model_rng_sha256': config['initial_model_rng_sha256'],
            'train_audit_sha256': streams['train_stream']['audit_sha256'],
            'val_audit_sha256': streams['val_stream']['audit_sha256']}


def code_fingerprint():
    digest = hashlib.sha256()
    paths = sorted((ROOT / 'scripts').glob('*.py'))
    paths += [ROOT / 'uv.lock', ROOT / 'configs/neighbor_ab_base_plans.json', ROOT / 'docs/experiments/neighbor-ab-protocol.md']
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def data_fingerprint(data_root, split):
    dataset_root = data_root / 'preprocessed' / DATASET
    cache = dataset_root / 'nnUNetPlans_2d_neighbor_ab_fullcanvas_v1'
    expected = set(split['train']) | set(split['val'])
    data_ids = {path.name[:-5] for path in cache.glob('*.b2nd') if not path.name.endswith('_seg.b2nd')}
    seg_ids = {path.name[:-9] for path in cache.glob('*_seg.b2nd')}
    property_ids = {path.stem for path in cache.glob('*.pkl')}
    if data_ids != expected or seg_ids != expected or property_ids != expected:
        raise ValueError('Incomplete or extra preprocessed cases')
    digest = hashlib.sha256()
    for path in sorted(dataset_root.rglob('*')):
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(dataset_root)).encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def verify_registered_modules():
    import nnunetv2
    package = Path(nnunetv2.__path__[0])
    for name, folder in [('nnunet_vnet_trainer.py', 'training/nnUNetTrainer'),
                         ('nnunet_neighbor_ab_trainer.py', 'training/nnUNetTrainer'),
                         ('neighbor_ab_preprocessor.py', 'preprocessing/preprocessors')]:
        if (ROOT / 'scripts' / name).read_bytes() != (package / folder / name).read_bytes():
            raise ValueError(f'Registered module differs from the frozen source: {name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnunet-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--host-label', required=True)
    parser.add_argument('--order', nargs=2, choices=['A', 'B'], required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if set(args.order) != {'A', 'B'}:
        raise ValueError('A pair must contain A and B exactly once')
    execution = execution_configuration()
    data_root = args.nnunet_root.resolve()
    output = args.output_root.resolve() / f'seed_{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    split_path = data_root / 'preprocessed' / DATASET / 'splits_final.json'
    split = json.loads(split_path.read_text())[0]
    if len(split['train']) != 970 or len(split['val']) != 218:
        raise ValueError('Wrong frozen split size')
    environment = os.environ.copy()
    environment.update({
        'nnUNet_raw': str(data_root / 'raw'),
        'nnUNet_preprocessed': str(data_root / 'preprocessed'),
        'nnUNet_results': str(output),
        'NEIGHBOR_AB_SEED': str(args.seed),
        'NEIGHBOR_AB_PREFETCH': str(execution['prefetch']),
        'nnUNet_n_proc_DA': '0', 'nnUNet_compile': 'false',
        'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': str(args.seed),
        'PYTHONUNBUFFERED': '1',
    })
    status_path = output / 'pair_status.json'
    verify_registered_modules()
    status = {'host_label': args.host_label, 'seed': args.seed, 'order': args.order,
              'code_sha256': code_fingerprint(), 'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
              'python': platform.python_version(), 'torch': torch.__version__,
              'execution': execution,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'runs': {}, 'status': 'validating' if args.verify_only else 'running'}
    status['data_sha256'] = data_fingerprint(data_root, split)
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        for key in ('host_label', 'seed', 'order', 'code_sha256', 'split_sha256', 'data_sha256'):
            if previous.get(key) != status[key]:
                raise ValueError(f'Existing pair provenance differs: {key}. Refusing to mix runs.')
        if previous.get('execution', execution) != execution:
            raise ValueError('Existing pair execution mode differs; refusing an unrecorded switch')
    from prepare_neighbor_continuation import verify_continuation
    continuation = verify_continuation(output, status)
    if continuation is not None:
        status['continuation_sha256'] = hashlib.sha256((output / 'continuation.json').read_bytes()).hexdigest()
    if not args.verify_only:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refusing CPU training')
        status['gpu'] = torch.cuda.get_device_name()
        if continuation is not None:
            report_path = output / 'continuation_evidence/equivalence_report.json'
            report = json.loads(report_path.read_text())
            if report['runtime'].get('device_name') != status['gpu']:
                raise ValueError('Continuation GPU differs from the equivalence certificate')
        write_json(status_path, status)
    first_initial_hash = None
    for mode in args.order:
        if code_fingerprint() != status['code_sha256']:
            raise ValueError('Experiment code changed while the pair was running')
        verify_registered_modules()
        fold = output / DATASET / f'{TRAINERS[mode]}__nnUNetPlans__2d/fold_0'
        command = [str(Path(sys.executable).parent / 'nnUNetv2_train'), '503', '2d', '0', '-tr', TRAINERS[mode]]
        final = fold / 'checkpoint_final.pth'
        finished = False
        if final.exists() and (fold / 'validation/summary.json').exists():
            try:
                verify_finished(fold, split['val'], mode, args.seed)
                finished = True
            except (ValueError, KeyError, OSError):
                pass
        if not finished and not args.verify_only:
            if final.exists():
                command.append('--val')
            elif (fold / 'checkpoint_latest.pth').exists():
                command.append('--c')
            if first_initial_hash:
                environment['NEIGHBOR_AB_EXPECTED_INITIAL_HASH'] = first_initial_hash
            print(f'Starting mode {mode}, seed {args.seed}: {command}', flush=True)
            with (output / f'console_{mode}.log').open('a') as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end='', flush=True)
                return_code = process.wait()
            if return_code != 0:
                status.update(status='failed', failed_mode=mode, exit_code=return_code)
                write_json(status_path, status)
                raise RuntimeError(f'Mode {mode} failed; see console_{mode}.log. Remaining runs were not started.')
        verified = verify_finished(fold, split['val'], mode, args.seed)
        run_manifest = json.loads((fold / 'neighbor_ab_run.json').read_text())
        initial_hash = run_manifest['initial_state_sha256']
        if first_initial_hash is not None and initial_hash != first_initial_hash:
            raise ValueError('Paired initial weights/BN states differ')
        first_initial_hash = initial_hash
        status['runs'][mode] = verified
        if not args.verify_only:
            write_json(status_path, status)
    for key in ('initial_state_sha256', 'initial_model_rng_sha256', 'train_audit_sha256', 'val_audit_sha256'):
        if status['runs']['A'][key] != status['runs']['B'][key]:
            raise ValueError(f'Paired run audit mismatch: {key}')
    status['difference_B_minus_A'] = status['runs']['B']['foreground_mean']['Dice'] - status['runs']['A']['foreground_mean']['Dice']
    status['status'] = 'complete'
    status['completed_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(status_path, status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
