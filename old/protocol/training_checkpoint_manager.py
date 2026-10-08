#!/usr/bin/env python3
"""Atomic, hash-verified checkpoint manager for P-CRA-U training.

Use this module from the training loop. Model weights are stored in safetensors;
optimizer/scheduler/scaler/RNG state is stored separately in a trusted local PT
file. Published checkpoint directories are immutable and never auto-deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment error handled at runtime
    np = None

try:
    import torch
    from safetensors.torch import load_file as load_safetensors
    from safetensors.torch import save_file as save_safetensors
except ImportError:  # allow hash-only CLI checks in a non-training environment
    torch = None
    load_safetensors = None
    save_safetensors = None


SCHEMA_VERSION = 1
ALLOWED_SELECTION_SPLITS = {"train", "dev"}
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class CheckpointError(RuntimeError):
    pass


def _require_training_dependencies() -> None:
    if torch is None or np is None or save_safetensors is None or load_safetensors is None:
        raise CheckpointError(
            "Training dependencies are unavailable. Use .conda-roborefer/bin/python "
            "(torch + numpy + safetensors)."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointError(f"Expected JSON object: {path}")
    return value


def _validate_identity(identity: Mapping[str, str]) -> dict[str, str]:
    if not identity:
        raise CheckpointError("identity must contain locked SHA-256 values")
    normalized = dict(identity)
    for key, value in normalized.items():
        if not isinstance(key, str) or not key:
            raise CheckpointError("identity keys must be non-empty strings")
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise CheckpointError(f"identity[{key!r}] is not a lowercase SHA-256 digest")
    return normalized


def _numeric_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in metrics.items():
        number = float(value)
        if not math.isfinite(number):
            raise CheckpointError(f"Metric {key!r} is non-finite: {number}")
        output[str(key)] = number
    return output


def _finite_cpu_state_dict(model: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key, value in model.state_dict().items():
        if not torch.is_tensor(value):
            raise CheckpointError(f"Model state {key!r} is not a tensor")
        tensor = value.detach()
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch.isfinite(tensor).all().item()):
                raise CheckpointError(f"Refusing to save non-finite model tensor: {key}")
        state[key] = tensor.cpu().contiguous().clone()
    if not state:
        raise CheckpointError("Model state_dict is empty")
    return state


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise CheckpointError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_state) != torch.cuda.device_count():
            raise CheckpointError(
                f"CUDA RNG device-count mismatch: checkpoint={len(cuda_state)}, runtime={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_state)


def _move_value_to_device(value: Any, device: Any) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_value_to_device(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(child, device) for child in value)
    return value


def _optimizer_state_to_parameter_devices(optimizer: Any) -> None:
    for parameter, state in optimizer.state.items():
        if hasattr(parameter, "device"):
            optimizer.state[parameter] = _move_value_to_device(state, parameter.device)


def _file_entry(path: Path, base: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(base)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _pointer_name(metric: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", metric.lower()).strip("_")
    if not slug:
        raise CheckpointError("selection metric produces an empty pointer name")
    return f"best_{slug}.json"


def _is_better(candidate: float, incumbent: float, mode: str) -> bool:
    return candidate < incumbent if mode == "min" else candidate > incumbent


def save_checkpoint(
    checkpoint_root: str | Path,
    *,
    run_id: str,
    model: Any,
    optimizer: Any,
    epoch: int,
    global_step: int,
    selection_split: str,
    selection_metric: str,
    selection_mode: str,
    selection_value: float,
    metrics: Mapping[str, float],
    identity: Mapping[str, str],
    scheduler: Any | None = None,
    scaler: Any | None = None,
    sampler_state: Any | None = None,
    extra_state: Any | None = None,
    runtime: Mapping[str, Any] | None = None,
    eligible_for_best: bool = True,
    rank: int = 0,
) -> dict[str, Any]:
    """Save and atomically publish one immutable checkpoint.

    Only rank 0 may call this function. `selection_split` is deliberately
    restricted to train/dev so calibration and test cannot select a model.
    """
    _require_training_dependencies()
    if rank != 0:
        raise CheckpointError("Only distributed rank 0 may write checkpoints")
    if selection_split not in ALLOWED_SELECTION_SPLITS:
        raise CheckpointError(
            f"selection_split={selection_split!r} is forbidden; best checkpoint selection is train/dev only"
        )
    if selection_mode not in {"min", "max"}:
        raise CheckpointError("selection_mode must be 'min' or 'max'")
    if epoch < 0 or global_step < 0:
        raise CheckpointError("epoch/global_step must be non-negative")
    if not run_id or not selection_metric:
        raise CheckpointError("run_id and selection_metric are required")
    selection_value = float(selection_value)
    if not math.isfinite(selection_value):
        raise CheckpointError("selection_value must be finite")
    metrics_clean = _numeric_metrics(metrics)
    identity_clean = _validate_identity(identity)
    model_state = _finite_cpu_state_dict(model)

    root = Path(checkpoint_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_id = f"step_{global_step:09d}"
    final_dir = root / checkpoint_id
    if final_dir.exists():
        raise CheckpointError(f"Immutable checkpoint already exists: {final_dir}")
    temporary = root / f".tmp-{checkpoint_id}-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        model_path = temporary / "model.safetensors"
        state_path = temporary / "training_state.pt"
        metadata_path = temporary / "metadata.json"
        manifest_path = temporary / "manifest.json"

        # No embedded safetensors metadata: provenance is canonical JSON and
        # tensor serialization is not exposed to metadata-key order variation.
        save_safetensors(model_state, str(model_path))
        training_state: dict[str, Any] = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": _capture_rng_state(),
            "sampler": sampler_state,
            "extra": extra_state,
        }
        torch.save(training_state, state_path)
        components = ["model", "optimizer", "rng"]
        if scheduler is not None:
            components.append("scheduler")
        if scaler is not None:
            components.append("scaler")
        if sampler_state is not None:
            components.append("sampler")
        if extra_state is not None:
            components.append("extra")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "run_id": run_id,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection": {
                "split": selection_split,
                "metric": selection_metric,
                "mode": selection_mode,
                "value": selection_value,
                "eligible_for_best": bool(eligible_for_best),
            },
            "metrics": metrics_clean,
            "identity": identity_clean,
            "runtime": dict(runtime or {}),
            "finite_model_tensors": True,
            "state_components": components,
        }
        _atomic_json(metadata_path, metadata)
        for path in [model_path, state_path, metadata_path]:
            _fsync_file(path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "run_id": run_id,
            "files": [_file_entry(path, temporary) for path in [metadata_path, model_path, state_path]],
        }
        _atomic_json(manifest_path, manifest)
        _fsync_file(manifest_path)
        _fsync_directory(temporary)
        os.rename(temporary, final_dir)
        _fsync_directory(root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    verification = verify_checkpoint(final_dir)
    manifest_sha = sha256_file(final_dir / "manifest.json")
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": checkpoint_id,
        "manifest_sha256": manifest_sha,
        "selection": metadata["selection"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(root / "last.json", pointer)
    promoted_best = False
    best_path = root / _pointer_name(selection_metric)
    if eligible_for_best:
        promote = True
        if best_path.exists():
            incumbent = _read_json(best_path)
            old_selection = incumbent.get("selection", {})
            if old_selection.get("metric") != selection_metric or old_selection.get("mode") != selection_mode:
                raise CheckpointError(f"Existing best pointer has incompatible selection rule: {best_path}")
            promote = _is_better(selection_value, float(old_selection["value"]), selection_mode)
        if promote:
            _atomic_json(best_path, pointer)
            promoted_best = True
    return {
        "checkpoint_dir": str(final_dir),
        "checkpoint_id": checkpoint_id,
        "manifest_sha256": manifest_sha,
        "verification_passed": verification["passed"],
        "last_pointer": str(root / "last.json"),
        "best_pointer": str(best_path),
        "promoted_best": promoted_best,
    }


def verify_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    directory = Path(checkpoint_dir).resolve()
    manifest_path = directory / "manifest.json"
    metadata_path = directory / "metadata.json"
    if not directory.is_dir() or not manifest_path.is_file() or not metadata_path.is_file():
        raise CheckpointError(f"Incomplete checkpoint directory: {directory}")
    manifest = _read_json(manifest_path)
    metadata = _read_json(metadata_path)
    missing, size_mismatches, hash_mismatches = [], [], []
    declared = manifest.get("files")
    if not isinstance(declared, list) or not declared:
        raise CheckpointError(f"Manifest has no files: {manifest_path}")
    for item in declared:
        path = directory / item["path"]
        if not path.is_file():
            missing.append(item["path"])
            continue
        if path.stat().st_size != item["bytes"]:
            size_mismatches.append(item["path"])
        if sha256_file(path) != item["sha256"]:
            hash_mismatches.append(item["path"])
    expected_names = {"metadata.json", "model.safetensors", "training_state.pt"}
    declared_names = {item["path"] for item in declared}
    metadata_consistent = (
        metadata.get("schema_version") == SCHEMA_VERSION
        and metadata.get("checkpoint_id") == directory.name == manifest.get("checkpoint_id")
        and metadata.get("run_id") == manifest.get("run_id")
        and metadata.get("selection", {}).get("split") in ALLOWED_SELECTION_SPLITS
        and metadata.get("finite_model_tensors") is True
        and declared_names == expected_names
    )
    passed = not (missing or size_mismatches or hash_mismatches) and metadata_consistent
    result = {
        "checkpoint_dir": str(directory),
        "checkpoint_id": directory.name,
        "manifest_sha256": sha256_file(manifest_path),
        "missing": missing,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "metadata_consistent": metadata_consistent,
        "passed": passed,
    }
    if not passed:
        raise CheckpointError(f"Checkpoint verification failed: {json.dumps(result, sort_keys=True)}")
    return result


def _resolve_pointer(root: Path, pointer: str | Path) -> Path:
    candidate = Path(pointer)
    if candidate.is_dir():
        return candidate.resolve()
    pointer_path = candidate if candidate.suffix == ".json" else root / f"{candidate}.json"
    if not pointer_path.is_absolute():
        pointer_path = (root / pointer_path).resolve() if pointer_path.parent == Path(".") else pointer_path.resolve()
    payload = _read_json(pointer_path)
    directory = (root / payload["checkpoint_path"]).resolve()
    if directory.parent != root.resolve():
        raise CheckpointError(f"Pointer escapes checkpoint root: {pointer_path}")
    if sha256_file(directory / "manifest.json") != payload.get("manifest_sha256"):
        raise CheckpointError(f"Pointer manifest hash mismatch: {pointer_path}")
    return directory


def resume_training(
    checkpoint_root: str | Path,
    *,
    pointer: str | Path,
    model: Any,
    optimizer: Any,
    expected_identity: Mapping[str, str],
    scheduler: Any | None = None,
    scaler: Any | None = None,
    strict_model: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Verify and restore a complete trusted local training checkpoint."""
    _require_training_dependencies()
    root = Path(checkpoint_root).resolve()
    directory = _resolve_pointer(root, pointer)
    verification = verify_checkpoint(directory)
    metadata = _read_json(directory / "metadata.json")
    identity = _validate_identity(expected_identity)
    if metadata["identity"] != identity:
        keys = sorted(set(metadata["identity"]) | set(identity))
        mismatch = [key for key in keys if metadata["identity"].get(key) != identity.get(key)]
        raise CheckpointError(f"Resume identity mismatch: {mismatch}")
    weights = load_safetensors(str(directory / "model.safetensors"), device="cpu")
    incompatible = model.load_state_dict(weights, strict=strict_model)
    # training_state.pt is trusted only after this manager verifies its hash.
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
    if state.get("optimizer") is None:
        raise CheckpointError("Checkpoint lacks optimizer state")
    optimizer.load_state_dict(state["optimizer"])
    _optimizer_state_to_parameter_devices(optimizer)
    if state.get("scheduler") is not None:
        if scheduler is None:
            raise CheckpointError("Checkpoint contains scheduler state but no scheduler was supplied")
        scheduler.load_state_dict(state["scheduler"])
    if state.get("scaler") is not None:
        if scaler is None:
            raise CheckpointError("Checkpoint contains scaler state but no scaler was supplied")
        scaler.load_state_dict(state["scaler"])
    if restore_rng:
        _restore_rng_state(state["rng"])
    return {
        "checkpoint_dir": str(directory),
        "metadata": metadata,
        "verification": verification,
        "incompatible_model_keys": {
            "missing": list(incompatible.missing_keys),
            "unexpected": list(incompatible.unexpected_keys),
        },
        "sampler_state": state.get("sampler"),
        "extra_state": state.get("extra"),
    }


def load_model_for_evaluation(
    checkpoint_dir: str | Path,
    *,
    model: Any,
    expected_identity: Mapping[str, str],
    strict_model: bool = True,
) -> dict[str, Any]:
    """Load only verified safetensors weights; never unpickle training state."""
    _require_training_dependencies()
    directory = Path(checkpoint_dir).resolve()
    verification = verify_checkpoint(directory)
    metadata = _read_json(directory / "metadata.json")
    if metadata["identity"] != _validate_identity(expected_identity):
        raise CheckpointError("Evaluation checkpoint identity mismatch")
    weights = load_safetensors(str(directory / "model.safetensors"), device="cpu")
    incompatible = model.load_state_dict(weights, strict=strict_model)
    return {
        "metadata": metadata,
        "verification": verification,
        "incompatible_model_keys": {
            "missing": list(incompatible.missing_keys),
            "unexpected": list(incompatible.unexpected_keys),
        },
    }


def audit_checkpoint_root(checkpoint_root: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_root).resolve()
    checkpoints = sorted(path for path in root.glob("step_[0-9]*") if path.is_dir()) if root.exists() else []
    valid, invalid = [], []
    for directory in checkpoints:
        try:
            verify_checkpoint(directory)
            valid.append(directory.name)
        except CheckpointError as error:
            invalid.append({"checkpoint_id": directory.name, "error": str(error)})
    pointers = {}
    for path in sorted(root.glob("*.json")) if root.exists() else []:
        try:
            target = _resolve_pointer(root, path)
            pointers[path.name] = {"target": target.name, "valid": True}
        except (CheckpointError, OSError, KeyError) as error:
            pointers[path.name] = {"target": None, "valid": False, "error": str(error)}
    stale_temporary = sorted(path.name for path in root.glob(".tmp-*") if path.is_dir()) if root.exists() else []
    passed = not invalid and not stale_temporary and all(item["valid"] for item in pointers.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_root": str(root),
        "valid_checkpoints": valid,
        "invalid_checkpoints": invalid,
        "pointers": pointers,
        "stale_temporary_directories": stale_temporary,
        "auto_deleted": [],
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("checkpoint_dir")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("checkpoint_root")
    args = parser.parse_args()
    result = verify_checkpoint(args.checkpoint_dir) if args.command == "verify" else audit_checkpoint_root(args.checkpoint_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
