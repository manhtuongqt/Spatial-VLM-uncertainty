#!/usr/bin/env python3
"""Build a resumable, content-addressed, oracle-free pooled feature cache."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

if __package__:
    from .pcra_u_development_common import (
        DevelopmentTrainingError,
        canonical_sha256,
        pool_raw_feature,
        read_json,
        safe_resolve,
        seed_runtime,
        sha256_file,
        write_json,
    )
    from .wp3_feature_hook_smoke import extract_features, forbidden_cache_findings, load_model, model_inventory_sha256
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pcra_u_development_common import (  # type: ignore
        DevelopmentTrainingError,
        canonical_sha256,
        pool_raw_feature,
        read_json,
        safe_resolve,
        seed_runtime,
        sha256_file,
        write_json,
    )
    from wp3_feature_hook_smoke import (  # type: ignore
        extract_features,
        forbidden_cache_findings,
        load_model,
        model_inventory_sha256,
    )


WORKSPACE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "pcra_u_development_train_v1"
FEATURE_MANIFEST = WORKSPACE / "protocol/pcra_u_development_feature_manifest.json"
TRAIN_MANIFEST = WORKSPACE / "protocol/pcra_u_development_train_manifest.json"
CONFIG_PATH = WORKSPACE / "protocol/pcra_u_development_train_config.json"
SCHEMA_PATH = WORKSPACE / "protocol/pcra_u_development_feature_cache.schema.json"
DEFAULT_CACHE_ROOT = WORKSPACE / "results/pcra_u_feature_cache/pcra_u_development_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_schema(index: dict[str, Any]) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ModuleNotFoundError:
        source = """
import json
import sys
from jsonschema import Draft202012Validator
schema = json.load(open(sys.argv[1], encoding='utf-8'))
instance = json.load(sys.stdin)
errors = sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda item: list(item.path))
if errors:
    print(errors[0].message, file=sys.stderr)
    raise SystemExit(2)
"""
        environment = dict(os.environ)
        environment.pop("PYTHONNOUSERSITE", None)
        completed = subprocess.run(
            ["/usr/bin/python3", "-c", source, str(SCHEMA_PATH)],
            input=json.dumps(index, ensure_ascii=False),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            raise DevelopmentTrainingError(
                f"Feature cache schema error: {completed.stderr.strip() or completed.returncode}"
            )
        return
    schema = read_json(SCHEMA_PATH)
    errors = sorted(Draft202012Validator(schema).iter_errors(index), key=lambda item: list(item.path))
    if errors:
        raise DevelopmentTrainingError(f"Feature cache schema error: {errors[0].message}")


def tensor_payload_valid(path: Path) -> bool:
    try:
        tensors = load_safetensors(str(path), device="cpu")
    except Exception:
        return False
    expected = {
        "R0_GRID": [24, 32, 1152],
        "D0_GRID": [24, 32, 1152],
        "R0_THUMB": [1152],
        "D0_THUMB": [1152],
    }
    return set(tensors) == set(expected) and all(
        list(tensors[name].shape) == shape and tensors[name].dtype == torch.float16
        and bool(torch.isfinite(tensors[name].float()).all().item())
        for name, shape in expected.items()
    )


def required_entries(scope: str, feature_manifest: dict[str, Any], train_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = feature_manifest["entries"]
    if scope == "full":
        return entries
    canary_ids = set(train_manifest["canary_family_ids"])
    selected = [entry for entry in entries if entry["family_id"] in canary_ids]
    if len(selected) != 100:
        raise DevelopmentTrainingError(f"Canary cache expects 100 samples, found {len(selected)}")
    return selected


def build(scope: str, cache_root: Path) -> dict[str, Any]:
    feature_manifest = read_json(FEATURE_MANIFEST)
    train_manifest = read_json(TRAIN_MANIFEST)
    config = read_json(CONFIG_PATH)
    if feature_manifest.get("protocol_id") != PROTOCOL_ID or train_manifest.get("protocol_id") != PROTOCOL_ID:
        raise DevelopmentTrainingError("Development manifests are not locked protocol v1")
    if forbidden_cache_findings(feature_manifest):
        raise DevelopmentTrainingError("Feature manifest contains evaluator/oracle-like keys")
    entries = required_entries(scope, feature_manifest, train_manifest)
    model_root = safe_resolve(WORKSPACE, config["feature_input"]["model_root"])
    inventory = model_inventory_sha256(model_root)
    if inventory != train_manifest["protected_inputs"]["model_inventory_sha256"]:
        raise DevelopmentTrainingError("RoboRefer checkpoint inventory changed")
    seed_runtime(int(config["seed"]), strict=False)
    cache_root = cache_root.resolve()
    feature_root = cache_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    index_root = cache_root / "indexes"
    index_root.mkdir(parents=True, exist_ok=True)

    by_key: dict[str, dict[str, Any]] = {}
    sample_to_feature: dict[str, str] = {}
    for entry in entries:
        key = entry["feature_key"]
        sample_to_feature[entry["sample_id"]] = key
        if key in by_key:
            prior = by_key[key]
            if (prior["rgb_sha256"], prior["depth_sha256"]) != (
                entry["rgb_sha256"], entry["depth_sha256"]
            ):
                raise DevelopmentTrainingError(f"Feature-key collision: {key}")
        else:
            by_key[key] = entry

    model = None
    newly_extracted = 0
    reused_verified = 0
    rows: dict[str, Any] = {}
    started = time.perf_counter()
    feature_latencies: list[float] = []
    try:
        for position, key in enumerate(sorted(by_key), start=1):
            entry = by_key[key]
            tensor_path = feature_root / f"{key}.safetensors"
            metadata_path = feature_root / f"{key}.json"
            if tensor_path.is_file() and metadata_path.is_file():
                metadata = read_json(metadata_path)
                if metadata.get("feature_key") != key or metadata.get("sha256") != sha256_file(tensor_path):
                    raise DevelopmentTrainingError(f"Existing cache metadata mismatch: {key}")
                if not tensor_payload_valid(tensor_path):
                    raise DevelopmentTrainingError(f"Existing cache tensor invalid: {key}")
                reused_verified += 1
            else:
                if tensor_path.exists() or metadata_path.exists():
                    raise DevelopmentTrainingError(f"Partial immutable cache pair exists: {key}")
                if model is None:
                    model = load_model(model_root)
                rgb_path = safe_resolve(WORKSPACE, entry["rgb_path"])
                depth_path = safe_resolve(WORKSPACE, entry["depth_path"])
                if sha256_file(rgb_path) != entry["rgb_sha256"] or sha256_file(depth_path) != entry["depth_sha256"]:
                    raise DevelopmentTrainingError(f"Feature input hash changed: {entry['sample_id']}")
                seed_runtime(int(config["seed"]), strict=False)
                extracted = extract_features(model, rgb_path, depth_path)
                r_grid, r_thumb = pool_raw_feature(extracted["r0"])
                d_grid, d_thumb = pool_raw_feature(extracted["d0"])
                payload = {
                    "R0_GRID": r_grid.half().contiguous(),
                    "D0_GRID": d_grid.half().contiguous(),
                    "R0_THUMB": r_thumb.half().contiguous(),
                    "D0_THUMB": d_thumb.half().contiguous(),
                }
                temporary = tensor_path.parent / f".{tensor_path.name}.tmp-{os.getpid()}"
                save_safetensors(payload, str(temporary))
                os.replace(temporary, tensor_path)
                tensor_hash = sha256_file(tensor_path)
                metadata = {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL_ID,
                    "feature_key": key,
                    "rgb_sha256": entry["rgb_sha256"],
                    "depth_sha256": entry["depth_sha256"],
                    "model_inventory_sha256": inventory,
                    "raw_stage": "tower_output_before_projector",
                    "raw_shape": [13, 1024, 1152],
                    "pooling": "locked_3x4_tiles_avgpool4_layernorm_v1",
                    "tensor_shapes": {name: list(value.shape) for name, value in payload.items()},
                    "dtype": "torch.float16",
                    "sha256": tensor_hash,
                    "bytes": tensor_path.stat().st_size,
                    "runtime": {
                        "preprocess_latency_ms": extracted["preprocess_latency_ms"],
                        "feature_latency_ms": extracted["feature_latency_ms"],
                    },
                    "safety": {
                        "evaluator_artifacts_read": False,
                        "oracle_labels_cached": False,
                        "training_performed": False,
                    },
                }
                if forbidden_cache_findings(metadata):
                    raise DevelopmentTrainingError(f"Generated feature metadata is not oracle-free: {key}")
                write_json(metadata_path, metadata)
                newly_extracted += 1
                feature_latencies.append(float(extracted["feature_latency_ms"]))
                del extracted, payload, r_grid, d_grid, r_thumb, d_thumb
            rows[key] = {
                "path": str(tensor_path.relative_to(cache_root)),
                "sha256": sha256_file(tensor_path),
                "bytes": tensor_path.stat().st_size,
                "rgb_sha256": entry["rgb_sha256"],
                "depth_sha256": entry["depth_sha256"],
                "tensor_shapes": {
                    "R0_GRID": [24, 32, 1152],
                    "D0_GRID": [24, 32, 1152],
                    "R0_THUMB": [1152],
                    "D0_THUMB": [1152],
                },
                "dtype": "torch.float16",
            }
            if position % 25 == 0 or position == len(by_key):
                print(f"FEATURE_CACHE {scope} {position}/{len(by_key)} new={newly_extracted} reused={reused_verified}", flush=True)
    finally:
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    index = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scope": scope,
        "status": "COMPLETE",
        "created_at_utc": utc_now(),
        "model_inventory_sha256": inventory,
        "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST),
        "pooling_contract_sha256": canonical_sha256(config["feature_input"]),
        "features": rows,
        "sample_to_feature": sample_to_feature,
        "counts": {
            "samples": len(entries),
            "unique_feature_pairs": len(rows),
            "newly_extracted": newly_extracted,
            "reused_verified": reused_verified,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "median_new_feature_latency_ms": sorted(feature_latencies)[len(feature_latencies) // 2]
            if feature_latencies else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
        },
        "safety": {
            "inference_manifest_only": True,
            "evaluator_artifacts_read": False,
            "training_performed": False,
            "oracle_labels_cached": False,
            "baseline_modified": False,
        },
    }
    validate_schema(index)
    if forbidden_cache_findings(index):
        raise DevelopmentTrainingError("Generated feature index contains evaluator/oracle-like keys")
    index_path = index_root / f"index_{scope}.json"
    if index_path.exists():
        existing = read_json(index_path)
        stable_existing = {k: v for k, v in existing.items() if k not in {"created_at_utc", "runtime", "counts"}}
        stable_new = {k: v for k, v in index.items() if k not in {"created_at_utc", "runtime", "counts"}}
        if stable_existing != stable_new:
            raise DevelopmentTrainingError(f"Refusing to replace non-equivalent index: {index_path}")
    write_json(index_path, index)
    return {"index_path": str(index_path.relative_to(WORKSPACE)), **index["counts"], "status": "COMPLETE"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["canary", "full"], required=True)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT.relative_to(WORKSPACE)))
    args = parser.parse_args()
    result = build(args.scope, safe_resolve(WORKSPACE, args.cache_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
