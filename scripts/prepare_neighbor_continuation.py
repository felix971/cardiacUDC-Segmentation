"""Create a separate, checkpoint-bound continuation without editing its source."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil

import torch

import run_neighbor_ab_pair as runner


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require_equal(actual, expected, description):
    if actual != expected:
        raise ValueError(f"Continuation {description} differs from its frozen provenance")


def validate_checkpoint(checkpoint, manifest, seed, mode):
    required = {"network_weights", "optimizer_state", "lr_scheduler_state", "grad_scaler_state",
                "logging", "current_epoch", "init_args", "trainer_name", "_best_ema",
                "neighbor_ab", "neighbor_ab_state", "inference_allowed_mirroring_axes"}
    if not required.issubset(checkpoint):
        raise ValueError("Continuation requires a complete training checkpoint")
    if not checkpoint["network_weights"] or not checkpoint["optimizer_state"].get("param_groups"):
        raise ValueError("Continuation checkpoint has no network or optimizer state")
    require_equal(checkpoint["trainer_name"], runner.TRAINERS[mode], "trainer")
    config = checkpoint["neighbor_ab"]
    for key, value in config.items():
        require_equal(manifest.get(key), value, f"manifest {key}")
    frozen = {"schema_version": 1, "seed": seed, "mode": mode, "target_index": 4,
              "frame_offsets": list(range(-4, 4)), "patch_size": [768, 512], "batch_size": 2,
              "epochs": 500, "updates_per_epoch": 250, "checkpoint_every_epochs": 10,
              "validation_batches_per_epoch": 50, "train_stream_seed": seed + 1,
              "val_stream_seed": seed + 2, "compile": False, "deterministic_algorithms": True,
              "optimizer": {"name": "SGD", "lr": 0.01, "momentum": 0.99,
                            "nesterov": True, "weight_decay": 3e-5},
              "scheduler": "polynomial", "cross_entropy": "log_softmax_gather_masked_mean",
              "ignore_label": None, "oversample_foreground_percent": 0.33,
              "probabilistic_oversampling": False, "inference_mirroring_axes": None,
              "cudnn_deterministic": True, "cudnn_benchmark": False,
              "cuda_matmul_tf32": False, "cudnn_tf32": False,
              "cublas_workspace_config": ":4096:8"}
    from nnunet_neighbor_ab_trainer import AUGMENTATION
    frozen["augmentation"] = AUGMENTATION
    require_equal(set(config), set(frozen) | {"initial_state_sha256", "initial_model_rng_sha256"},
                  "configuration fields")
    for key, value in frozen.items():
        require_equal(config.get(key), value, f"configuration {key}")
    for key in ("initial_state_sha256", "initial_model_rng_sha256"):
        if not isinstance(config.get(key), str) or len(config[key]) != 64:
            raise ValueError(f"Missing initial checkpoint provenance: {key}")
    epoch = checkpoint["current_epoch"]
    if type(epoch) is not int or not 1 <= epoch <= 500:
        raise ValueError("Checkpoint must be a completed epoch boundary in [1, 500]")
    state = checkpoint["neighbor_ab_state"]
    for name, iterations in (("train_stream", 250), ("val_stream", 50)):
        stream = state.get(name, {})
        require_equal(stream.get("batches_seen"), epoch * iterations, f"{name} count")
        if not isinstance(stream.get("audit_sha256"), str) or len(stream["audit_sha256"]) != 64:
            raise ValueError(f"Incomplete {name} audit state")
        if not {"python", "numpy", "torch_cpu"}.issubset(stream.get("rng_state", {})):
            raise ValueError(f"Incomplete {name} RNG state")
    model_rng = state.get("model_rng", {})
    if not {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(model_rng) or not model_rng["torch_cuda"]:
        raise ValueError("Complete model CPU and CUDA RNG state is required")
    return config


def validate_audits(fold, checkpoint):
    for name in ("train", "val"):
        stream = checkpoint["neighbor_ab_state"][f"{name}_stream"]
        records = (fold / f"neighbor_ab_{name}_audit.jsonl").read_text().splitlines()
        if len(records) != min(8, stream["batches_seen"]):
            raise ValueError(f"Incomplete {name} audit records")
        digest = hashlib.sha256(b"").hexdigest()
        for index, line in enumerate(records):
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("batch") != index:
                raise ValueError(f"Invalid {name} audit batch sequence")
            cumulative = record.pop("cumulative_sha256", None)
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            digest = hashlib.sha256(digest.encode() + encoded).hexdigest()
            require_equal(cumulative, digest, f"{name} audit hash chain")
        if stream["batches_seen"] <= 8:
            require_equal(stream["audit_sha256"], digest, f"{name} committed audit")


def validate_report_binding(report, source_status, checkpoint, checkpoint_path, checkpoint_sha256):
    from validate_neighbor_prefetch import validate_equivalence_report

    validate_equivalence_report(report, require_cuda=True)
    if report.get("migration_eligible") is not True:
        raise ValueError("Equivalence certificate is not eligible for checkpoint migration")
    config = checkpoint["neighbor_ab"]
    expected_source = {"path": str(checkpoint_path), "sha256": checkpoint_sha256,
                       "seed": config["seed"], "mode": config["mode"],
                       "current_epoch": checkpoint["current_epoch"]}
    for key, value in expected_source.items():
        require_equal(report["source_checkpoint"].get(key), value, f"certificate checkpoint {key}")
    for source_key, report_key in (("code_sha256", "baseline_code_sha256"),
                                   ("data_sha256", "data_sha256"), ("split_sha256", "split_sha256")):
        require_equal(report.get(report_key), source_status.get(source_key), f"certificate {report_key}")
    require_equal(report.get("current_code_sha256"), runner.code_fingerprint(), "current code")
    runtime = report["runtime"]
    for key, expected in (("python", platform.python_version()), ("torch", torch.__version__),
                          ("cuda", torch.version.cuda), ("cpu_threads", 8)):
        require_equal(runtime.get(key), expected, f"certificate runtime {key}")


def prepare_continuation(source_root, destination_root, checkpoint_path, equivalence_report_path):
    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    report_path = Path(equivalence_report_path).resolve()
    if destination.exists():
        raise FileExistsError(f"Continuation destination already exists: {destination}")
    if source in destination.parents or destination in source.parents:
        raise ValueError("Continuation requires separate source and destination roots")
    if source not in checkpoint_path.parents:
        raise ValueError("Checkpoint must belong to the explicit source seed root")
    source_status = read_json(source / "pair_status.json")
    report = read_json(report_path)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mode = checkpoint.get("neighbor_ab", {}).get("mode")
    if mode not in runner.TRAINERS:
        raise ValueError("Unknown continuation mode")
    seed = source_status["seed"]
    require_equal(source.name, f"seed_{seed}", "source seed directory")
    require_equal(destination.name, f"seed_{seed}", "destination seed directory")
    # This migration is deliberately limited to the running first member of a pair.
    # Do not silently drop any completed partner or start a later member out of order.
    if source_status.get("order") not in (["A", "B"], ["B", "A"]):
        raise ValueError("Source pair order is invalid")
    if source_status["order"][0] != mode or source_status.get("runs"):
        raise ValueError("Only the first, unfinished member of a pair can be migrated")
    fold_relative = Path(runner.DATASET) / f"{runner.TRAINERS[mode]}__nnUNetPlans__2d/fold_0"
    require_equal(checkpoint_path.parent, source / fold_relative, "checkpoint fold")
    other_mode = "B" if mode == "A" else "A"
    other_fold = source / runner.DATASET / f"{runner.TRAINERS[other_mode]}__nnUNetPlans__2d/fold_0"
    if other_fold.exists():
        raise ValueError("Source contains another mode; refusing a partial pair migration")
    manifest = read_json(checkpoint_path.parent / "neighbor_ab_run.json")
    config = validate_checkpoint(checkpoint, manifest, seed, mode)
    validate_audits(checkpoint_path.parent, checkpoint)
    validate_report_binding(report, source_status, checkpoint, checkpoint_path, checkpoint_sha256)
    if (source / "continuation.json").exists():
        raise ValueError("Chained migrations are outside this explicit continuation procedure")

    evidence_sources = []
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".log", ".txt"):
            evidence_sources.append((path, path.relative_to(source), file_sha256(path)))
    destination.mkdir(parents=True, exist_ok=False)
    evidence = destination / "continuation_evidence"
    evidence.mkdir()
    evidence_hashes = {}
    for path, relative, digest in evidence_sources:
        copied = evidence / "source" / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copied)
        require_equal(file_sha256(copied), digest, f"copied evidence {relative}")
        require_equal(file_sha256(path), digest, f"source changed during copy: {relative}")
        evidence_hashes[str(copied.relative_to(destination))] = digest
        # Runtime logs and manifests may evolve; the evidence copies never do.
        if relative != Path("pair_status.json"):
            runtime_copy = destination / relative
            runtime_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(copied, runtime_copy)
    evidence_checkpoint = evidence / "checkpoint.pth"
    shutil.copy2(checkpoint_path, evidence_checkpoint)
    require_equal(file_sha256(evidence_checkpoint), checkpoint_sha256, "copied checkpoint")
    require_equal(file_sha256(checkpoint_path), checkpoint_sha256, "source checkpoint changed during copy")
    evidence_hashes[str(evidence_checkpoint.relative_to(destination))] = checkpoint_sha256
    copied_report = evidence / "equivalence_report.json"
    shutil.copy2(report_path, copied_report)
    require_equal(read_json(copied_report), report, "copied equivalence certificate")
    evidence_hashes[str(copied_report.relative_to(destination))] = file_sha256(copied_report)
    target_name = "checkpoint_final.pth" if checkpoint["current_epoch"] == 500 else "checkpoint_latest.pth"
    target_relative = fold_relative / target_name
    (destination / fold_relative).mkdir(parents=True, exist_ok=True)
    shutil.copy2(evidence_checkpoint, destination / target_relative)
    record = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source), "destination_root": str(destination),
        "source_checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_relative_path": str(target_relative), "seed": seed, "mode": mode,
        "current_epoch": checkpoint["current_epoch"], "experiment_config": config,
        "completed_train_samples": checkpoint["neighbor_ab_state"]["train_stream"]["batches_seen"] * 2,
        "completed_val_samples": checkpoint["neighbor_ab_state"]["val_stream"]["batches_seen"] * 2,
        "baseline_code_sha256": source_status["code_sha256"],
        "current_code_sha256": report["current_code_sha256"],
        "data_sha256": source_status["data_sha256"], "split_sha256": source_status["split_sha256"],
        "evidence_sha256": evidence_hashes,
    }
    runner.write_json(destination / "continuation.json", record)
    for relative in evidence_hashes:
        path = destination / relative
        path.chmod(path.stat().st_mode & ~0o222)
    (destination / "continuation.json").chmod(0o444)
    return record


def verify_continuation(output, live_status):
    output = Path(output).resolve()
    record_path = output / "continuation.json"
    if not record_path.exists():
        if (output / "continuation_evidence").exists():
            raise ValueError("Continuation evidence exists without its continuation record")
        return None
    record = read_json(record_path)
    require_equal(record.get("schema_version"), 1, "record schema")
    require_equal(record.get("destination_root"), str(output), "destination root")
    if (output / "pair_status.json").exists():
        previous = read_json(output / "pair_status.json")
        require_equal(previous.get("continuation_sha256"), file_sha256(record_path), "continuation record")
    for relative, digest in record["evidence_sha256"].items():
        path = (output / relative).resolve()
        if output / "continuation_evidence" not in path.parents:
            raise ValueError("Evidence path escapes the continuation evidence directory")
        require_equal(file_sha256(path), digest, f"evidence {relative}")
    evidence = output / "continuation_evidence"
    source_status = read_json(evidence / "source/pair_status.json")
    report = read_json(evidence / "equivalence_report.json")
    checkpoint = torch.load(evidence / "checkpoint.pth", map_location="cpu", weights_only=False)
    require_equal(file_sha256(evidence / "checkpoint.pth"), record["checkpoint_sha256"], "immutable checkpoint")
    validate_report_binding(report, source_status, checkpoint, Path(record["source_checkpoint"]), record["checkpoint_sha256"])
    for key in ("host_label", "seed", "order", "data_sha256", "split_sha256"):
        require_equal(live_status.get(key), source_status.get(key), f"live {key}")
    require_equal(live_status.get("code_sha256"), report["current_code_sha256"], "live code")
    for key in ("baseline_code_sha256", "current_code_sha256", "data_sha256", "split_sha256"):
        require_equal(record.get(key), report.get(key), f"record {key}")
    for key in ("seed", "mode"):
        require_equal(record.get(key), checkpoint["neighbor_ab"][key], f"record {key}")
    require_equal(record["current_epoch"], checkpoint["current_epoch"], "record epoch")
    require_equal(record["experiment_config"], checkpoint["neighbor_ab"], "record configuration")
    validate_checkpoint(checkpoint, record["experiment_config"], record["seed"], record["mode"])
    for name in ("train", "val"):
        require_equal(record[f"completed_{name}_samples"],
                      checkpoint["neighbor_ab_state"][f"{name}_stream"]["batches_seen"] * 2,
                      f"record {name} samples")
    relative = Path(runner.DATASET) / f"{runner.TRAINERS[record['mode']]}__nnUNetPlans__2d/fold_0"
    target_name = "checkpoint_final.pth" if record["current_epoch"] == 500 else "checkpoint_latest.pth"
    require_equal(record["checkpoint_relative_path"], str(relative / target_name), "checkpoint location")
    target = output / relative / target_name
    if not (output / "pair_status.json").exists():
        require_equal(file_sha256(target), record["checkpoint_sha256"], "initial resume checkpoint")
    else:
        final = output / relative / "checkpoint_final.pth"
        current = torch.load(final if final.exists() else target, map_location="cpu", weights_only=False)
        validate_checkpoint(current, record["experiment_config"], record["seed"], record["mode"])
        require_equal(current["neighbor_ab"], record["experiment_config"], "resumed configuration")
        if current["current_epoch"] < record["current_epoch"]:
            raise ValueError("Continuation checkpoint moved backwards")
        if current["current_epoch"] == record["current_epoch"]:
            require_equal(file_sha256(final if final.exists() else target), record["checkpoint_sha256"], "unchanged boundary checkpoint")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--equivalence-report", type=Path, required=True)
    args = parser.parse_args()
    record = prepare_continuation(args.source_root, args.destination_root, args.checkpoint, args.equivalence_report)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
