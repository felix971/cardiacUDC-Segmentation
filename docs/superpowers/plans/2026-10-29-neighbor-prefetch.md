# Neighbor A/B Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overlap unchanged CPU batch preparation with GPU computation and resume the existing experiment with auditable equivalence.

**Architecture:** One spawned CPU producer per stream, queue depth two, consumed-state checkpoints, serial fallback. Migration uses a separate output root and preserves the original provenance.

**Tech Stack:** Existing Python 3.12, locked PyTorch/nnU-Net/MONAI, multiprocessing spawn, unittest.

**Spec:** docs/superpowers/specs/2026-10-29-neighbor-prefetch-design.md

## Global Constraints

- Preserve scientific experiment_config and all frozen numerical settings.
- Old checkout/environment/output remain unchanged; no extra long training experiments.
- Checkpoint stream state reflects the last consumed batch only.
- CPU workers never initialize CUDA or modify the parent's RNG.
- Keep batch size 2, queue depth 2, CPU threads 8; require measured Ubuntu throughput gain >= 5%.

## Review Focus

- Producer exception or premature death: consumer must fail promptly, not hang (Task 1).
- Pending train/validation queues at checkpoint: next batch and model RNG must match after restore (Tasks 1 and 2).
- Tensor IPC and worker shutdown: no stale shared storage, CUDA context, or orphan process (Task 1).
- Valid-looking migration with wrong seed/code/data/checkpoint: fail before GPU work and leave original intact (Task 3).
- Measurement dominated by startup or inconsistent model RNG: bounded, warmed, interleaved timing and full update equivalence (Task 2).

### Task 1: Prefetched stream and trainer integration

**Files:** scripts/nnunet_neighbor_ab_trainer.py; tests/test_neighbor_ab_prefetch.py.

**Interfaces:** PairedPrefetchAugmenter(data_loader, seed, audit_path=None, prefetch=2), __next__, state_dict/load_state_dict matching PairedSingleThreadedAugmenter, close/_finish. NEIGHBOR_AB_PREFETCH=0 selects serial; 2 selects prefetch. Invalid values fail. Execution metadata is separate from experiment_config. Existing serial output semantics unchanged.

- [ ] Write failing deterministic fake-loader tests for complete batch equality/audit metadata, parent RNG isolation, consumed-state restoration with queued batches, errors/death, repeated close, and invalid configuration.
- [ ] Implement the producer and integration, including bounded cleanup at training end/error. Audit disk writes happen only in the consumer.
- [ ] Run targeted CPU tests then the full existing neighbor suite; commit only owned files.

### Task 2: Isolated full-sized validation and throughput harness

**Files:** scripts/validate_neighbor_prefetch.py; tests/test_neighbor_prefetch_validation.py.

**Interfaces:** CLI --nnunet-root PATH --baseline-root PATH --output-root PATH [--checkpoint PATH] [--cpu-only] [--steps N]. Report equivalence_report.json with baseline/current code fingerprints, batches and metadata equality, RNG/optimizer/update equality, timings and gain threshold. Never mutate baseline or formal outputs; never start long training.

- [ ] Tests reject invalid proof and timing samples and cover resumed queue states.
- [ ] Compare actual original module and current module on full-sized train/validation A/B batches (at least eight each), checkpoint/restored next batches, and parent model RNG.
- [ ] CUDA mode uses the same complete checkpoint and bounded updates (warmup plus interleaved timed serial/prefetch passes), verifies bitwise weights/buffers/optimizer/scaler/model RNG and records throughput/resources. CPU-only mode must not initialize CUDA.
- [ ] Run CPU proof first; root exclusively coordinates stopping the old workload before any GPU proof. Commit owned files before the final proof so code fingerprints stay fixed.

### Task 3: Provenance-preserving continuation

**Files:** scripts/prepare_neighbor_continuation.py; scripts/run_neighbor_ab_pair.py; tests/test_neighbor_continuation.py; docs as needed.

**Interfaces:** preparation CLI --source-root OLD_SEED_ROOT --destination-root NEW_SEED_ROOT --checkpoint PATH --equivalence-report PATH. Validate complete checkpoint and matching provenance, create new destination exclusively, copy checkpoint without editing bytes, preserve old status/manifests/audits/logs under continuation evidence, write continuation.json. Runner validates its new-root continuation before starting/resuming; unchanged runs continue to work.

- [ ] Write red tests for wrong seed/mode/epoch/audit counts/checkpoint hash/code/data proof, existing destination, and original-file preservation.
- [ ] Implement narrow explicit migration and verification, record execution mode without weakening frozen code checks.
- [ ] Run full suite, independent branch review, final CPU/CUDA proof, then root performs durable boundary migration and starts Ubuntu and Colab queues. If equivalence or gain fails, restore the serial workload and launch the unchanged cloud pipeline with the Agg fix.
