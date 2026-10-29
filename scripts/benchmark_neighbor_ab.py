"""Bounded VNet3D-T8 memory/timing probe; never starts formal training."""
import argparse
import json
import time
from pathlib import Path

import torch
from monai.networks.nets import VNet

from nnunet_vnet_trainer import keep_frames_in_shallow_levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch', type=int, nargs=2, default=[768, 512])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    network = VNet(spatial_dims=3, in_channels=1, out_channels=5)
    keep_frames_in_shallow_levels(network)
    network = network.cuda().train()
    optimizer = torch.optim.SGD(network.parameters(), lr=.01, momentum=.99, nesterov=True, weight_decay=3e-5)
    scaler = torch.amp.GradScaler('cuda')
    images = torch.randn(2, 1, 8, *args.patch, device='cuda')
    targets = torch.zeros(2, *args.patch, dtype=torch.long, device='cuda')
    timings = []
    for step in range(5):
        torch.cuda.synchronize()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = network(images)[:, :, 4]
            loss = torch.nn.functional.cross_entropy(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        if step >= 2:
            timings.append(elapsed)
        print(json.dumps({'step': step, 'seconds': elapsed, 'loss': loss.item()}), flush=True)
    report = {
        'kind': 'synthetic_resource_probe_not_formal_training',
        'gpu': torch.cuda.get_device_name(), 'patch': args.patch, 'batch_size': 2,
        'parameters': sum(parameter.numel() for parameter in network.parameters()),
        'seconds_per_step': sum(timings) / len(timings),
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
