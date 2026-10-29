"""Isolated, bounded old-module versus prefetch validation; never formal training.

CPU proof never initializes a network or CUDA. CUDA proof requires an explicit
complete checkpoint and is launched only by the operator after GPU isolation.
"""

import argparse
import copy
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import platform
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DATASET = "Dataset503_CardiacUDCNeighborAB"
BATCH_FIELDS = ("data_equal", "target_equal", "metadata_equal", "audit_equal",
                "stream_rng_equal", "model_rng_equal")
CUDA_FIELDS = ("updates_equal", "optimizer_equal", "scaler_equal", "weights_equal",
               "buffers_equal", "model_rng_equal", "stream_rng_equal",
               "validation_transition", "mode_b_smoke")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def summarize_timings(samples):
    require(isinstance(samples, list) and len(samples) >= 4 and len(samples) % 2 == 0,
            "Timing requires at least two complete interleaved replicates")
    gains = []
    means = {"serial": [], "prefetch": []}
    for index, sample in enumerate(samples):
        order = ("serial", "prefetch") if (index // 2) % 2 == 0 else ("prefetch", "serial")
        implementation = order[index % 2]
        require(sample.get("implementation") == implementation and sample.get("replicate") == index // 2,
                "Timing replicates must counterbalance serial/prefetch order without duplicates")
        seconds = sample.get("seconds_per_step")
        require(isinstance(seconds, list) and len(seconds) >= 20,
                "Each timing sample needs at least 20 measured end-to-end updates")
        require(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in seconds),
                "Timing samples must be positive finite seconds")
        require(type(sample.get("warmup_steps")) is int and sample["warmup_steps"] >= 2,
                "Timing samples require at least two warmup updates")
        allocated = sample.get("peak_allocated_bytes", 0)
        reserved = sample.get("peak_reserved_bytes", 0)
        require(type(allocated) is int and type(reserved) is int and 0 < allocated <= reserved,
                "CUDA memory peaks must be recorded")
        means[implementation].append(statistics.mean(seconds))
        if index % 2 == 1:
            previous = samples[index - 1]
            require(len(previous["seconds_per_step"]) == len(seconds)
                    and previous["warmup_steps"] == sample["warmup_steps"],
                    "Paired timings must execute identical update counts")
            gains.append(means["serial"][-1] / means["prefetch"][-1] - 1)
    gain = min(gains)
    return {"samples": samples, "replicate_gains": gains, "gain_fraction": gain,
            "minimum_gain_fraction": 0.05, "passed": gain >= 0.05,
            "serial_steps_per_second": 1 / statistics.mean(means["serial"]),
            "prefetch_steps_per_second": 1 / statistics.mean(means["prefetch"])}


def validate_equivalence_report(report, require_cuda=True):
    """Fail closed before migration. A CPU report is evidence, never permission."""
    require(isinstance(report, dict) and report.get("schema_version") == 1, "Unknown report schema")
    for field in ("baseline_code_sha256", "current_code_sha256", "data_sha256", "split_sha256"):
        require(valid_digest(report.get(field)), f"Missing or invalid {field}")
    cpu = report.get("cpu", {})
    for field in ("passed", "queued_resume", "serial_to_prefetch_resume", "model_rng_equal"):
        require(cpu.get(field) is True, f"CPU proof failed or missing: {field}")
    cases = cpu.get("fullsize_cases", [])
    require(len(cases) == 4 and {(case.get("mode"), case.get("phase")) for case in cases}
            == {("A", "train"), ("A", "val"), ("B", "train"), ("B", "val")},
            "Full-sized proof must cover both modes and both streams")
    for case in cases:
        require(type(case.get("batches")) is int and case["batches"] >= 8, "Too few full-sized batches")
        require(case.get("batch_shape") == [2, 8, 768, 512]
                and case.get("target_shape") == [2, 1, 768, 512], "Proof used a different batch or patch shape")
        for field in BATCH_FIELDS:
            require(case.get(field) is True, f"Batch proof failed or missing: {field}")
    if not require_cuda:
        return report
    require(report.get("passed") is True and report.get("migration_eligible") is True,
            "Report is not eligible for migration")
    source = report.get("source_checkpoint", {})
    require(isinstance(source, dict) and valid_digest(source.get("sha256"))
            and isinstance(source.get("path"), str) and source.get("mode") in ("A", "B")
            and type(source.get("seed")) is int and type(source.get("current_epoch")) is int,
            "Complete source checkpoint identity is required")
    runtime = report.get("runtime", {})
    require(runtime.get("device") == "cuda" and runtime.get("cpu_threads") == 8
            and all(runtime.get(key) for key in ("python", "torch", "cuda")),
            "CUDA proof must record its runtime and use eight CPU threads")
    cuda = report.get("cuda") or {}
    require(cuda.get("passed") is True, "CUDA proof is missing or failed")
    require(cuda.get("checkpoint_sha256") == source["sha256"], "CUDA proof belongs to another checkpoint")
    for field in CUDA_FIELDS:
        require(cuda.get(field) is True, f"CUDA proof failed or missing: {field}")
    timing = report.get("timing") or {}
    calculated = summarize_timings(timing.get("samples"))
    require(timing.get("minimum_gain_fraction") == 0.05 and calculated["passed"]
            and timing.get("passed") is True, "Repeatable throughput gain must be at least 5%")
    require(type(timing.get("gain_fraction")) in (int, float)
            and math.isclose(timing["gain_fraction"], calculated["gain_fraction"], abs_tol=1e-12),
            "Reported gain differs from measured samples")
    return report


def load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tree_sha256(value):
    """Hash tensor/array bytes including dtype, shape, keys and scalar types."""
    import numpy as np
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor")
            visit(str(item.dtype))
            visit(tuple(item.shape))
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"ndarray")
            visit(str(item.dtype))
            visit(tuple(item.shape))
            if item.dtype.hasobject:
                visit(item.tolist())
            else:
                digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda entry: (type(entry).__name__, repr(entry))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            visit(len(item))
            for entry in item:
                visit(entry)
        else:
            encoded = pickle.dumps(item, protocol=4)
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)

    visit(value)
    return digest.hexdigest()


def close_streams(streams):
    for stream in streams:
        close = getattr(stream, "close", None)
        if close is not None:
            close()


def wait_for_pending(streams, count=2, timeout=60):
    deadline = time.monotonic() + timeout
    while any(stream.pending_batches < count for stream in streams):
        require(time.monotonic() < deadline, "Prefetch queue did not fill before checkpoint proof")
        time.sleep(0.02)


def compare_batches(expected, actual, expected_state, actual_state):
    metadata_expected = {key: value for key, value in expected.items() if key not in ("data", "target")}
    metadata_actual = {key: value for key, value in actual.items() if key not in ("data", "target")}
    checks = {
        "data_equal": tree_sha256(expected["data"]) == tree_sha256(actual["data"]),
        "target_equal": tree_sha256(expected["target"]) == tree_sha256(actual["target"]),
        "metadata_equal": tree_sha256(metadata_expected) == tree_sha256(metadata_actual),
        "audit_equal": expected_state["audit_sha256"] == actual_state["audit_sha256"],
        "stream_rng_equal": tree_sha256(expected_state) == tree_sha256(actual_state),
    }
    return checks


def batch_signature(batch, state):
    return {"data": tree_sha256(batch["data"]), "target": tree_sha256(batch["target"]),
            "metadata": tree_sha256({key: value for key, value in batch.items()
                                     if key not in ("data", "target")}),
            "stream": tree_sha256(state)}


def prove_streams(serial_factory, prefetch_factory, module, mode,
                  expected_shape=(2, 8, 768, 512), starting_states=None):
    """Compare complete batches and round-trip both queued, consumed streams."""
    serial = serial_factory()
    prefetch = prefetch_factory()
    opened = [serial, prefetch]
    outside = module.capture_rng_state(include_cuda=False)
    model_rng = tree_sha256(outside)
    result = {"mode": mode, "fullsize_cases": [], "queued_resume": False,
              "serial_to_prefetch_resume": False, "model_rng_equal": False}
    try:
        if starting_states is not None:
            for streams in (serial, prefetch):
                for stream, state in zip(streams, starting_states):
                    stream.load_state_dict(copy.deepcopy(state))
        for index, phase in enumerate(("train", "val")):
            case = {"mode": mode, "phase": phase, "batches": 8}
            for field in BATCH_FIELDS:
                case[field] = True
            for _ in range(8):
                expected = next(serial[index])
                actual = next(prefetch[index])
                require(tuple(expected["data"].shape) == tuple(expected_shape)
                        and tuple(actual["data"].shape) == tuple(expected_shape),
                        "Data proof did not use the frozen full-sized batch")
                case["batch_shape"] = list(actual["data"].shape)
                case["target_shape"] = list(actual["target"].shape)
                checks = compare_batches(expected, actual, serial[index].state_dict(), prefetch[index].state_dict())
                checks["model_rng_equal"] = model_rng == tree_sha256(module.capture_rng_state(include_cuda=False))
                for field, equal in checks.items():
                    case[field] = case[field] and equal
                require(all(checks.values()), f"Old/new batch mismatch in {mode}/{phase}: {checks}")
            result["fullsize_cases"].append(case)
        wait_for_pending(prefetch)
        result["pending_batches_at_save"] = [stream.pending_batches for stream in prefetch]
        serial_states = [stream.state_dict() for stream in serial]
        consumed_states = [stream.state_dict() for stream in prefetch]
        require(tree_sha256(serial_states) == tree_sha256(consumed_states), "Queued state includes unconsumed work")
        # A real pickle round-trip reproduces checkpoint serialization semantics.
        serialized = pickle.dumps(consumed_states, protocol=4)
        restored_states = pickle.loads(serialized)
        result["serialized_stream_sha256"] = hashlib.sha256(serialized).hexdigest()
        sequence = (0, 1, 0, 1)
        expected_after_save = []
        for index in sequence:
            batch = next(serial[index])
            expected_after_save.append(batch_signature(batch, serial[index].state_dict()))
        for index, expected in zip(sequence, expected_after_save):
            batch = next(prefetch[index])
            require(batch_signature(batch, prefetch[index].state_dict()) == expected,
                    "Queued train/validation transition differs before restoration")
        close_streams(prefetch)
        for states, field in ((restored_states, "queued_resume"), (serial_states, "serial_to_prefetch_resume")):
            resumed = prefetch_factory()
            opened.append(resumed)
            for stream, state in zip(resumed, states):
                stream.load_state_dict(state)
            for index, expected in zip(sequence, expected_after_save):
                batch = next(resumed[index])
                require(batch_signature(batch, resumed[index].state_dict()) == expected,
                        f"Restored next batch differs: {field}")
            close_streams(resumed)
            result[field] = True
        result["model_rng_equal"] = model_rng == tree_sha256(module.capture_rng_state(include_cuda=False))
        require(result["model_rng_equal"], "Stream proof perturbed parent model RNG")
        return result
    finally:
        for streams in opened:
            close_streams(streams)
        module.restore_rng_state(outside)


def set_paths(data_root, output, seed, prefetch):
    os.environ.update(nnUNet_raw=str(data_root / "raw"),
                      nnUNet_preprocessed=str(data_root / "preprocessed"),
                      nnUNet_results=str(output), NEIGHBOR_AB_SEED=str(seed),
                      NEIGHBOR_AB_PREFETCH=str(prefetch), nnUNet_compile="false",
                      nnUNet_n_proc_DA="0", MPLBACKEND="Agg",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    os.environ.pop("NEIGHBOR_AB_EXPECTED_INITIAL_HASH", None)


def load_modules(baseline_root):
    sys.path.insert(0, str(ROOT / "scripts"))
    # Spawn imports this exact canonical name in the child; an ad-hoc alias is
    # safe only for the original, strictly serial implementation.
    current = importlib.import_module("nnunet_neighbor_ab_trainer")
    require(Path(current.__file__).resolve() == ROOT / "scripts/nnunet_neighbor_ab_trainer.py",
            "Candidate trainer import resolved to another checkout")
    baseline = load_source_module("neighbor_prefetch_baseline", baseline_root / "scripts/nnunet_neighbor_ab_trainer.py")
    require(Path(baseline.__file__).resolve() != Path(current.__file__).resolve(),
            "Baseline and candidate must be separate source modules")
    old_runner = load_source_module("neighbor_prefetch_baseline_runner", baseline_root / "scripts/run_neighbor_ab_pair.py")
    current_runner = importlib.import_module("run_neighbor_ab_pair")
    # Both trainers import the registered helper. It must be the same unchanged
    # source for this independent-module comparison to be meaningful.
    import inspect
    helper = Path(inspect.getsourcefile(current.keep_frames_in_shallow_levels))
    for root in (baseline_root, ROOT):
        require(helper.read_bytes() == (root / "scripts/nnunet_vnet_trainer.py").read_bytes(),
                "Registered network helper differs from original or candidate source")
    return baseline, current, old_runner, current_runner


def read_configuration(data_root):
    folder = data_root / "preprocessed" / DATASET
    plans = json.loads((folder / "nnUNetPlans.json").read_text())
    dataset = json.loads((folder / "dataset.json").read_text())
    config = plans["configurations"]["2d"]
    require(config["patch_size"] == [768, 512] and config["batch_size"] == 2,
            "Validation requires the frozen [768,512] patch and batch size 2")
    return plans, dataset


def create_trainer(module, plans, dataset, mode, data_root, output, seed, prefetch, device):
    import torch
    set_paths(data_root, output, seed, prefetch)
    cls = getattr(module, f"nnUNetTrainerNeighbor{mode}_500epochs")
    run_plans = copy.deepcopy(plans)
    run_plans["continue_training"] = False
    return cls(run_plans, "2d", 0, copy.deepcopy(dataset), torch.device(device))


def streams_factory(module, plans, dataset, mode, data_root, output, seed, prefetch):
    def create():
        outside = module.capture_rng_state(include_cuda=False)
        try:
            trainer = create_trainer(module, plans, dataset, mode, data_root, output, seed, prefetch, "cpu")
            trainer.batch_size = trainer.configuration_manager.batch_size
            return trainer.get_dataloaders()
        finally:
            module.restore_rng_state(outside)
    return create


def read_checkpoint(path):
    if path is None:
        return None, None
    import torch
    before = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(before == file_sha256(path), "Source checkpoint changed while it was read")
    fields = ("network_weights", "optimizer_state", "lr_scheduler_state", "grad_scaler_state",
              "logging", "current_epoch", "init_args", "neighbor_ab", "neighbor_ab_state", "_best_ema")
    require(all(field in checkpoint for field in fields), "Source checkpoint is incomplete")
    config = checkpoint["neighbor_ab"]
    states = checkpoint["neighbor_ab_state"]
    require(all(key in states for key in ("model_rng", "train_stream", "val_stream")),
            "Checkpoint lacks complete model/train/validation RNG streams")
    identity = {"path": str(path.resolve()), "sha256": before, "seed": config["seed"],
                "mode": config["mode"], "current_epoch": checkpoint["current_epoch"]}
    return checkpoint, identity


def verify_source_binding(path, checkpoint, report):
    """Reject wrong cached data or source code before initializing any GPU."""
    status_path = None
    for parent in path.resolve().parents:
        candidate = parent / "pair_status.json"
        if candidate.is_file():
            status_path = candidate
            break
    require(status_path is not None, "Checkpoint has no original pair_status.json provenance")
    status = json.loads(status_path.read_text())
    for source_key, report_key in (("code_sha256", "baseline_code_sha256"),
                                   ("data_sha256", "data_sha256"), ("split_sha256", "split_sha256")):
        require(status.get(source_key) == report[report_key], f"Source checkpoint {source_key} differs from proof inputs")
    config = checkpoint["neighbor_ab"]
    require(status.get("seed") == config["seed"], "Checkpoint seed differs from original pair")
    epoch = checkpoint["current_epoch"]
    require(type(epoch) is int and 0 < epoch <= 500, "Checkpoint is not a complete epoch boundary")
    states = checkpoint["neighbor_ab_state"]
    for stream, count in (("train_stream", config["updates_per_epoch"]),
                          ("val_stream", config["validation_batches_per_epoch"])):
        require(states[stream]["batches_seen"] == epoch * count,
                f"Source {stream} count differs from the checkpoint epoch boundary")
    manifest_path = path.parent / "neighbor_ab_run.json"
    require(manifest_path.is_file(), "Checkpoint has no original run manifest")
    manifest = json.loads(manifest_path.read_text())
    for key, value in config.items():
        require(manifest.get(key) == value, f"Source manifest differs from checkpoint: {key}")
    return {"pair_status_path": str(status_path), "pair_status_sha256": file_sha256(status_path),
            "run_manifest_path": str(manifest_path), "run_manifest_sha256": file_sha256(manifest_path)}


def cpu_proof(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    set_paths(args.nnunet_root, args.output_root, args.seed, 2)
    import torch
    torch.set_num_threads(args.cpu_threads)
    baseline, current, old_runner, current_runner = load_modules(args.baseline_root)
    plans, dataset = read_configuration(args.nnunet_root)
    checkpoint, identity = read_checkpoint(args.checkpoint)
    seed = identity["seed"] if identity else args.seed
    split_path = args.nnunet_root / "preprocessed" / DATASET / "splits_final.json"
    split = json.loads(split_path.read_text())[0]
    require(len(split["train"]) == 970 and len(split["val"]) == 218, "Wrong frozen split")
    report = {"schema_version": 1, "kind": "bounded_neighbor_prefetch_equivalence",
              "source_checkpoint": identity,
              "baseline_code_sha256": old_runner.code_fingerprint(),
              "current_code_sha256": current_runner.code_fingerprint(),
              "data_sha256": current_runner.data_fingerprint(args.nnunet_root, split),
              "split_sha256": file_sha256(split_path), "cpu": {}, "cuda": None, "timing": None,
              "passed": False, "equivalence_passed": False, "migration_eligible": False,
              "runtime": {"device": "cpu", "device_name": platform.processor(),
                          "python": platform.python_version(), "torch": torch.__version__,
                          "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads()}}
    if checkpoint is not None:
        report["source_binding"] = verify_source_binding(args.checkpoint, checkpoint, report)
    results = []
    for mode in ("A", "B"):
        old_factory = streams_factory(baseline, plans, dataset, mode, args.nnunet_root,
                                      args.output_root / f"cpu_{mode}_old", seed, 0)
        new_factory = streams_factory(current, plans, dataset, mode, args.nnunet_root,
                                      args.output_root / f"cpu_{mode}_prefetch", seed, 2)
        starting_states = None
        if checkpoint is not None:
            starting_states = [checkpoint["neighbor_ab_state"][name] for name in ("train_stream", "val_stream")]
        results.append(prove_streams(old_factory, new_factory, current, mode, starting_states=starting_states))
    report["cpu"] = {"passed": True, "fullsize_cases": [], "details": results,
                     "queued_resume": True, "serial_to_prefetch_resume": True, "model_rng_equal": True}
    for result in results:
        report["cpu"]["fullsize_cases"].extend(result["fullsize_cases"])
    require(not torch.cuda.is_initialized(), "CPU proof unexpectedly initialized CUDA")
    report["cpu"]["cuda_initialized"] = False
    validate_equivalence_report(report, require_cuda=False)
    return report


def trainer_snapshot(trainer, module):
    weights = dict(trainer.network.named_parameters())
    buffers = dict(trainer.network.named_buffers())
    scaler = trainer.grad_scaler.state_dict() if trainer.grad_scaler is not None else None
    streams = {"train": trainer.dataloader_train.state_dict(), "val": trainer.dataloader_val.state_dict()}
    return {"weights": tree_sha256(weights), "buffers": tree_sha256(buffers),
            "optimizer": tree_sha256(trainer.optimizer.state_dict()), "scaler": tree_sha256(scaler),
            "scheduler": tree_sha256(trainer.lr_scheduler.state_dict()),
            "model_rng": tree_sha256(module.capture_rng_state()), "streams": tree_sha256(streams)}


def finite_output(value):
    import numpy as np
    if isinstance(value, dict):
        return all(finite_output(item) for item in value.values())
    return bool(np.isfinite(value).all())


def measured_update(trainer, validation=False):
    import torch
    torch.cuda.synchronize()
    started = time.perf_counter()
    stream = trainer.dataloader_val if validation else trainer.dataloader_train
    batch = next(stream)
    if validation:
        with torch.no_grad():
            output = trainer.validation_step(batch)
    else:
        output = trainer.train_step(batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    require(finite_output(output), "Nonfinite training or validation output")
    return batch, output, elapsed


def start_telemetry(path):
    handle = path.open("w")
    try:
        process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits", "--loop-ms=500"],
            stdout=handle, stderr=subprocess.DEVNULL,
        )
        return process, handle
    except OSError:
        handle.close()
        return None, None


def stop_telemetry(process, handle):
    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        handle.close()


def run_cuda_case(module, plans, dataset, mode, args, name, prefetch,
                  checkpoint=None, timing=False, replicate=0):
    """Own exactly one model; release all CUDA objects before returning hashes."""
    import torch
    trainer = None
    telemetry = (None, None)
    result = {"name": name, "mode": mode, "implementation": "prefetch" if prefetch else "serial"}
    seed = checkpoint["neighbor_ab"]["seed"] if checkpoint is not None else args.seed
    case_output = args.output_root / name
    case_output.mkdir()
    try:
        trainer = create_trainer(module, plans, dataset, mode, args.nnunet_root,
                                 case_output, seed, prefetch, "cuda")
        if checkpoint is not None:
            trainer.load_checkpoint(checkpoint)
        trainer.on_train_start()
        trainer.on_train_epoch_start()
        result["initial"] = trainer_snapshot(trainer, module)
        torch.cuda.reset_peak_memory_stats()
        if timing:
            telemetry = start_telemetry(case_output / "gpu_telemetry.csv")
            # No batch/state hashing or serialization between timed updates:
            # giving the producer that free CPU interval would bias throughput.
            for _ in range(args.warmup_steps):
                measured_update(trainer)
            start_unix = time.time()
            start_monotonic = time.perf_counter()
            print(json.dumps({"phase": "timing_start", "case": name,
                              "unix_seconds": start_unix, "monotonic_seconds": start_monotonic}), flush=True)
            seconds = []
            for _ in range(args.steps):
                _, _, elapsed = measured_update(trainer)
                seconds.append(elapsed)
            end_monotonic = time.perf_counter()
            end_unix = time.time()
            print(json.dumps({"phase": "timing_end", "case": name,
                              "unix_seconds": end_unix, "monotonic_seconds": end_monotonic}), flush=True)
            result["sample"] = {
                "implementation": result["implementation"], "replicate": replicate,
                "seconds_per_step": seconds, "warmup_steps": args.warmup_steps,
                "start_unix_seconds": start_unix, "end_unix_seconds": end_unix,
                "start_monotonic_seconds": start_monotonic, "end_monotonic_seconds": end_monotonic,
                "wall_seconds": end_monotonic - start_monotonic,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "telemetry_csv": str(case_output / "gpu_telemetry.csv") if telemetry[0] else None,
            }
        else:
            updates = []
            for _ in range(3):
                batch, output, elapsed = measured_update(trainer)
                updates.append({"batch": batch_signature(batch, trainer.dataloader_train.state_dict()),
                                "output": tree_sha256(output), "state": trainer_snapshot(trainer, module)})
            trainer.network.eval()
            batch, output, elapsed = measured_update(trainer, validation=True)
            result["validation_seconds"] = elapsed
            result["validation"] = {"batch": batch_signature(batch, trainer.dataloader_val.state_dict()),
                                    "output": tree_sha256(output), "state": trainer_snapshot(trainer, module)}
            trainer.network.train()
            if prefetch:
                wait_for_pending((trainer.dataloader_train, trainer.dataloader_val))
            saved_state = trainer_snapshot(trainer, module)
            queued_checkpoint = case_output / "queued_checkpoint.pth"
            trainer.save_checkpoint(str(queued_checkpoint))
            batch, output, elapsed = measured_update(trainer)
            updates.append({"batch": batch_signature(batch, trainer.dataloader_train.state_dict()),
                            "output": tree_sha256(output), "state": trainer_snapshot(trainer, module)})
            result["updates"] = updates
            close_streams((trainer.dataloader_train, trainer.dataloader_val))
            trainer = None
            gc.collect()
            torch.cuda.empty_cache()
            trainer = create_trainer(module, plans, dataset, mode, args.nnunet_root,
                                     case_output / "restored", seed, prefetch, "cuda")
            trainer.load_checkpoint(str(queued_checkpoint))
            trainer.on_train_start()
            # This scratch checkpoint is intentionally mid-epoch: retain its
            # restored LR/scheduler and resume the next update directly.
            trainer.network.train()
            require(trainer_snapshot(trainer, module) == saved_state,
                    "Serialized queued checkpoint changed a training state")
            batch, output, elapsed = measured_update(trainer)
            restored_update = {"batch": batch_signature(batch, trainer.dataloader_train.state_dict()),
                               "output": tree_sha256(output), "state": trainer_snapshot(trainer, module)}
            require(restored_update == updates[-1], "Queued checkpoint next update differs after restoration")
            result["queued_checkpoint_restore_equal"] = True
        result["final"] = trainer_snapshot(trainer, module)
        return result
    finally:
        stop_telemetry(*telemetry)
        if trainer is not None:
            close_streams((getattr(trainer, "dataloader_train", None),
                           getattr(trainer, "dataloader_val", None)))
        trainer = None
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def compare_cuda_cases(serial, prefetch):
    require(serial["initial"] == prefetch["initial"], "Old/new initial complete training state differs")
    require(serial["final"] == prefetch["final"], "Old/new final model/optimizer/scaler/RNG state differs")
    if "updates" in serial:
        require(serial["updates"] == prefetch["updates"], "Old/new bounded update trajectory differs")
        require(serial["validation"] == prefetch["validation"], "Train/validation/train transition differs")
        require(serial.get("queued_checkpoint_restore_equal") is True
                and prefetch.get("queued_checkpoint_restore_equal") is True,
                "Serialized queued checkpoint proof is incomplete")


def cuda_proof(args, report):
    set_paths(args.nnunet_root, args.output_root, args.seed, 2)
    import torch
    torch.set_num_threads(8)
    baseline, current, old_runner, current_runner = load_modules(args.baseline_root)
    require(report["baseline_code_sha256"] == old_runner.code_fingerprint()
            and report["current_code_sha256"] == current_runner.code_fingerprint(),
            "Code changed after the CPU proof")
    checkpoint, identity = read_checkpoint(args.checkpoint)
    require(identity == report["source_checkpoint"], "Checkpoint changed after the CPU proof")
    if identity:
        args.seed = identity["seed"]
    require(torch.cuda.is_available(), "CUDA unavailable; refusing CPU training")
    report["runtime"] = {"device": "cuda", "device_name": torch.cuda.get_device_name(),
                         "device_index": torch.cuda.current_device(), "python": platform.python_version(),
                         "torch": torch.__version__, "cuda": torch.version.cuda,
                         "cudnn": torch.backends.cudnn.version(), "cpu_threads": torch.get_num_threads()}
    plans, dataset = read_configuration(args.nnunet_root)
    mode = identity["mode"] if identity else "A"
    cases = []
    for implementation, module, depth in (("serial", baseline, 0), ("prefetch", current, 2)):
        cases.append(run_cuda_case(module, plans, dataset, mode, args,
                                   f"checkpoint_equivalence_{implementation}", depth, checkpoint))
    compare_cuda_cases(*cases)
    smoke = []
    for implementation, module, depth in (("serial", baseline, 0), ("prefetch", current, 2)):
        smoke.append(run_cuda_case(module, plans, dataset, "B", args, f"fresh_B_{implementation}", depth))
    compare_cuda_cases(*smoke)
    samples = []
    timing_states = []
    for replicate in range(2):
        paired = {}
        order = [("serial", baseline, 0), ("prefetch", current, 2)]
        if replicate % 2:
            order.reverse()
        for implementation, module, depth in order:
            result = run_cuda_case(module, plans, dataset, mode, args,
                                   f"timing_{replicate}_{implementation}", depth,
                                   checkpoint, timing=True, replicate=replicate)
            paired[implementation] = result
            samples.append(result["sample"])
        compare_cuda_cases(paired["serial"], paired["prefetch"])
        timing_states.append({"replicate": replicate, "serial": paired["serial"]["final"],
                              "prefetch": paired["prefetch"]["final"]})
    report["cuda"] = {"passed": True, "checkpoint_sha256": identity["sha256"] if identity else None,
                      "cases": cases, "fresh_B_cases": smoke, "timing_final_states": timing_states}
    for field in CUDA_FIELDS:
        report["cuda"][field] = True
    report["timing"] = summarize_timings(samples)
    report["validation_overhead_seconds"] = {
        "serial": cases[0]["validation_seconds"], "prefetch": cases[1]["validation_seconds"],
        "fresh_B_serial": smoke[0]["validation_seconds"], "fresh_B_prefetch": smoke[1]["validation_seconds"],
    }
    report["equivalence_passed"] = True
    report["passed"] = report["timing"]["passed"]
    report["migration_eligible"] = bool(identity and report["passed"])
    if report["migration_eligible"]:
        validate_equivalence_report(report)
    require(report["baseline_code_sha256"] == old_runner.code_fingerprint()
            and report["current_code_sha256"] == current_runner.code_fingerprint(),
            "Code changed during the bounded proof")
    if identity:
        require(file_sha256(args.checkpoint) == identity["sha256"], "Checkpoint changed during the bounded proof")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnunet-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args(argv)
    require(20 <= args.steps <= 50, "Use 20 to 50 timed updates for a bounded proof")
    require(2 <= args.warmup_steps <= 10, "Use two to ten warmup updates")
    require(args.cpu_threads == 8 or args.cpu_only, "CUDA acceptance fixes CPU threads to eight")
    require(args.cpu_threads > 0, "CPU thread count must be positive")
    for name in ("nnunet_root", "baseline_root", "output_root", "checkpoint"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    require(args.baseline_root != ROOT, "Baseline must be a separate original checkout")
    for protected in (args.baseline_root, args.nnunet_root):
        require(not args.output_root.is_relative_to(protected), "Output would mutate a protected baseline/data root")
    if args.checkpoint is not None:
        for parent in args.checkpoint.parents:
            if (parent / "pair_status.json").is_file():
                require(not args.output_root.is_relative_to(parent), "Output would mutate original formal results")
                break
    require(not args.output_root.exists(), "Use a new isolated output root; evidence is never overwritten")
    return args


def write_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    report_path = args.output_root / "equivalence_report.json"
    if args.cpu_only:
        args.output_root.mkdir(parents=True)
        report = cpu_proof(args)
        write_report(report_path, report)
    else:
        # A separate CUDA-hidden process makes CPU-only isolation observable and
        # prevents a trainer constructor's seed routine from initializing CUDA.
        arguments = list(sys.argv[1:] if argv is None else argv)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *arguments, "--cpu-only"], check=True)
        report = json.loads(report_path.read_text())
        try:
            report = cuda_proof(args, report)
        except Exception as error:
            report.update(passed=False, equivalence_passed=False, migration_eligible=False,
                          error=f"{type(error).__name__}: {error}")
            write_report(report_path, report)
            raise
        write_report(report_path, report)
    print(json.dumps({"report": str(report_path), "cpu_passed": report["cpu"]["passed"],
                      "equivalence_passed": report["equivalence_passed"], "passed": report["passed"],
                      "migration_eligible": report["migration_eligible"]}), flush=True)


if __name__ == "__main__":
    main()
