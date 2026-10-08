#!/usr/bin/env python3
"""Build the separate, resumable and oracle-free Calibration feature cache."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
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
PROTOCOL_ID = "pcra_u_calibration_v1"
FEATURE_MANIFEST = WORKSPACE / "protocol/pcra_u_calibration_feature_manifest.json"
CONFIG_PATH = WORKSPACE / "protocol/pcra_u_calibration_config.json"
SCHEMA_PATH = WORKSPACE / "protocol/pcra_u_calibration_feature_cache.schema.json"
DEFAULT_CACHE_ROOT = WORKSPACE / "results/pcra_u_feature_cache/pcra_u_calibration_v1"
EXPECTED_SHAPES = {
    "R0_GRID": [24, 32, 1152],
    "D0_GRID": [24, 32, 1152],
    "R0_THUMB": [1152],
    "D0_THUMB": [1152],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_schema(index: dict[str, Any]) -> None:
    source = """
import json, sys
from jsonschema import Draft202012Validator
schema=json.load(open(sys.argv[1], encoding='utf-8'))
instance=json.load(sys.stdin)
errors=sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda x:list(x.path))
if errors:
 print(errors[0].message, file=sys.stderr); raise SystemExit(2)
"""
    environment = dict(os.environ)
    environment.pop("PYTHONNOUSERSITE", None)
    completed = subprocess.run(
        ["/usr/bin/python3", "-c", source, str(SCHEMA_PATH)],
        input=json.dumps(index, ensure_ascii=False), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment, check=False,
    )
    if completed.returncode:
        raise DevelopmentTrainingError(f"Calibration feature-cache schema error: {completed.stderr.strip()}")


def tensor_payload_valid(path: Path) -> bool:
    try:
        tensors = load_safetensors(str(path), device="cpu")
    except Exception:
        return False
    return set(tensors) == set(EXPECTED_SHAPES) and all(
        list(tensors[name].shape) == shape
        and tensors[name].dtype == torch.float16
        and bool(torch.isfinite(tensors[name].float()).all().item())
        for name, shape in EXPECTED_SHAPES.items()
    )


def zero_depth_fraction(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[:2] != (480, 640):
        raise DevelopmentTrainingError(f"Invalid observable depth input: {path}")
    zero = image == 0 if image.ndim == 2 else np.all(image == 0, axis=2)
    return float(np.mean(zero))


def build(cache_root: Path) -> dict[str, Any]:
    feature_manifest = read_json(FEATURE_MANIFEST)
    config = read_json(CONFIG_PATH)
    if feature_manifest.get("protocol_id") != PROTOCOL_ID or feature_manifest.get("sample_count") != 1000:
        raise DevelopmentTrainingError("Calibration feature manifest is not locked/complete")
    if forbidden_cache_findings(feature_manifest):
        raise DevelopmentTrainingError("Calibration feature manifest contains evaluator/oracle-like keys")
    entries = feature_manifest["entries"]
    model_root = safe_resolve(WORKSPACE, config["feature_input"]["model_root"])
    inventory = model_inventory_sha256(model_root)
    if inventory != feature_manifest["model_inventory_sha256"]:
        raise DevelopmentTrainingError("RoboRefer checkpoint inventory changed")
    # NumPy's legacy RNG accepts only uint32 seeds.  Keep the preregistered
    # master seed in provenance and use its deterministic uint32 projection at
    # this runtime boundary; do not alter shared development utilities.
    runtime_seed = int(config["seeds"]["master"]) % (2**32)
    seed_runtime(runtime_seed, strict=False)
    cache_root = cache_root.resolve()
    feature_root, index_root = cache_root / "features", cache_root / "indexes"
    feature_root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)
    by_key: dict[str, dict[str, Any]] = {}
    sample_to_feature: dict[str, str] = {}
    for entry in entries:
        key = entry["feature_key"]
        sample_to_feature[entry["sample_id"]] = key
        if key in by_key and (by_key[key]["rgb_sha256"], by_key[key]["depth_sha256"]) != (
            entry["rgb_sha256"], entry["depth_sha256"]
        ):
            raise DevelopmentTrainingError(f"Calibration feature-key collision: {key}")
        by_key.setdefault(key, entry)
    if len(sample_to_feature) != 1000:
        raise DevelopmentTrainingError("Calibration sample-to-feature coverage is not 1,000")

    model = None
    new_count = reused_count = 0
    rows: dict[str, Any] = {}
    latencies: list[float] = []
    started = time.perf_counter()
    try:
        for position, key in enumerate(sorted(by_key), 1):
            entry = by_key[key]
            tensor_path = feature_root / f"{key}.safetensors"
            metadata_path = feature_root / f"{key}.json"
            depth_path = safe_resolve(WORKSPACE, entry["depth_path"])
            summary = {
                "zero_valued_depth_input_fraction": zero_depth_fraction(depth_path),
                "definition": "fraction_of_640x480_depth_model_input_pixels_with_all_channels_zero",
            }
            if tensor_path.is_file() and metadata_path.is_file():
                metadata = read_json(metadata_path)
                if (
                    metadata.get("feature_key") != key
                    or metadata.get("sha256") != sha256_file(tensor_path)
                    or metadata.get("observable_input_summary") != summary
                    or not tensor_payload_valid(tensor_path)
                ):
                    raise DevelopmentTrainingError(f"Existing Calibration cache pair invalid: {key}")
                reused_count += 1
            else:
                if tensor_path.exists() or metadata_path.exists():
                    raise DevelopmentTrainingError(f"Partial immutable Calibration cache pair: {key}")
                if model is None:
                    model = load_model(model_root)
                rgb_path = safe_resolve(WORKSPACE, entry["rgb_path"])
                if sha256_file(rgb_path) != entry["rgb_sha256"] or sha256_file(depth_path) != entry["depth_sha256"]:
                    raise DevelopmentTrainingError(f"Calibration feature input hash changed: {entry['sample_id']}")
                seed_runtime(runtime_seed, strict=False)
                extracted = extract_features(model, rgb_path, depth_path)
                r_grid, r_thumb = pool_raw_feature(extracted["r0"])
                d_grid, d_thumb = pool_raw_feature(extracted["d0"])
                payload = {
                    "R0_GRID": r_grid.half().contiguous(), "D0_GRID": d_grid.half().contiguous(),
                    "R0_THUMB": r_thumb.half().contiguous(), "D0_THUMB": d_thumb.half().contiguous(),
                }
                temporary = tensor_path.parent / f".{tensor_path.name}.tmp-{os.getpid()}"
                save_safetensors(payload, str(temporary))
                os.replace(temporary, tensor_path)
                metadata = {
                    "schema_version": 1, "protocol_id": PROTOCOL_ID, "feature_key": key,
                    "rgb_sha256": entry["rgb_sha256"], "depth_sha256": entry["depth_sha256"],
                    "model_inventory_sha256": inventory, "raw_stage": "tower_output_before_projector",
                    "pooling": "locked_3x4_tiles_avgpool4_layernorm_v1",
                    "tensor_shapes": {name: list(value.shape) for name, value in payload.items()},
                    "dtype": "torch.float16", "sha256": sha256_file(tensor_path),
                    "bytes": tensor_path.stat().st_size, "observable_input_summary": summary,
                    "runtime": {"preprocess_latency_ms": extracted["preprocess_latency_ms"],
                                "feature_latency_ms": extracted["feature_latency_ms"]},
                    "safety": {"evaluator_artifacts_read": False, "oracle_labels_cached": False,
                               "model_training_performed": False, "test_opened": False},
                }
                if forbidden_cache_findings(metadata):
                    raise DevelopmentTrainingError(f"Generated Calibration cache metadata is not oracle-free: {key}")
                write_json(metadata_path, metadata)
                new_count += 1
                latencies.append(float(extracted["feature_latency_ms"]))
            rows[key] = {
                "path": str(tensor_path.relative_to(cache_root)), "sha256": sha256_file(tensor_path),
                "bytes": tensor_path.stat().st_size, "rgb_sha256": entry["rgb_sha256"],
                "depth_sha256": entry["depth_sha256"], "tensor_shapes": EXPECTED_SHAPES,
                "dtype": "torch.float16", "observable_input_summary": summary,
            }
            if position % 25 == 0 or position == len(by_key):
                print(f"CALIBRATION_FEATURE_CACHE {position}/{len(by_key)} new={new_count} reused={reused_count}", flush=True)
    finally:
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    index = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "scope": "full", "status": "COMPLETE",
        "created_at_utc": utc_now(), "model_inventory_sha256": inventory,
        "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST),
        "pooling_contract_sha256": canonical_sha256(config["feature_input"]),
        "features": rows, "sample_to_feature": sample_to_feature,
        "counts": {"samples": 1000, "unique_feature_pairs": len(rows),
                   "newly_extracted": new_count, "reused_verified": reused_count},
        "runtime": {"elapsed_seconds": time.perf_counter() - started,
                    "median_new_feature_latency_ms": float(np.median(latencies)) if latencies else None,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "torch": torch.__version__},
        "safety": {"inference_manifest_only": True, "evaluator_artifacts_read": False,
                   "model_training_performed": False, "oracle_labels_cached": False,
                   "test_opened": False, "baseline_modified": False},
    }
    validate_schema(index)
    if forbidden_cache_findings(index):
        raise DevelopmentTrainingError("Generated Calibration feature index is not oracle-free")
    index_path = index_root / "index_full.json"
    if index_path.exists():
        old = read_json(index_path)
        stable = lambda value: {k: v for k, v in value.items() if k not in {"created_at_utc", "runtime", "counts"}}
        if stable(old) != stable(index):
            raise DevelopmentTrainingError(f"Refusing to replace non-equivalent Calibration index: {index_path}")
    write_json(index_path, index)
    return {"status": "COMPLETE", "index_path": str(index_path.relative_to(WORKSPACE)), **index["counts"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT.relative_to(WORKSPACE)))
    args = parser.parse_args()
    print(json.dumps(build(safe_resolve(WORKSPACE, args.cache_root)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
