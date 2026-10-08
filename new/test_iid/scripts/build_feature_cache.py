#!/usr/bin/env python3
"""Build the resumable, oracle-free Test-IID RoboRefer feature cache."""

from __future__ import annotations

import json
import os
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


ROOT = Path(__file__).resolve().parents[3]
TEST_ROOT = ROOT / "new/test_iid"
MANIFEST_PATH = TEST_ROOT / "protocol/test_iid_feature_manifest.json"
GATE_PATH = TEST_ROOT / "protocol/frozen_inference_gate.json"
CACHE_ROOT = TEST_ROOT / "feature_cache"
PROTOCOL_ID = "roborefer_dataset_v2_1_test_iid_capture_200"
EXPECTED_UNIQUE_PAIRS = 619
EXPECTED_SHAPES = {
    "R0_GRID": [24, 32, 1152], "D0_GRID": [24, 32, 1152],
    "R0_THUMB": [1152], "D0_THUMB": [1152],
}

sys.path.insert(0, str(ROOT / "old/protocol"))
from pcra_u_development_common import pool_raw_feature, seed_runtime  # noqa: E402
from wp2_common import read_json, sha256_file, write_json  # noqa: E402
from wp3_feature_hook_smoke import (  # noqa: E402
    extract_features, forbidden_cache_findings, load_model, model_inventory_sha256,
)


class TestIIDFeatureError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_workspace_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise TestIIDFeatureError(f"path escapes workspace: {relative}")
    return path


def tensor_valid(path: Path) -> bool:
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
        raise TestIIDFeatureError(f"invalid observable depth: {path}")
    zero = image == 0 if image.ndim == 2 else np.all(image == 0, axis=2)
    return float(np.mean(zero))


def build() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    gate = read_json(GATE_PATH)
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("sample_count") != 1000
        or manifest.get("unique_feature_pair_count") != EXPECTED_UNIQUE_PAIRS
        or gate.get("decision") != "GO_FROZEN_BEST_V2_TEST_IID_INFERENCE"
        or gate.get("training_authorized") is not False
    ):
        raise TestIIDFeatureError("feature manifest/inference gate differs")
    if sha256_file(MANIFEST_PATH) != gate["feature_manifest_sha256"]:
        raise TestIIDFeatureError("feature manifest hash differs from gate")
    if forbidden_cache_findings(manifest):
        raise TestIIDFeatureError("feature manifest contains evaluator/oracle-like keys")
    model_root = safe_workspace_path(manifest["model_root"])
    inventory = model_inventory_sha256(model_root)
    if inventory != manifest["model_inventory_sha256"]:
        raise TestIIDFeatureError("RoboRefer model inventory changed")

    feature_root = CACHE_ROOT / "features"
    index_root = CACHE_ROOT / "indexes"
    feature_root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)
    by_key: dict[str, dict[str, Any]] = {}
    sample_to_feature: dict[str, str] = {}
    for entry in manifest["entries"]:
        key = entry["feature_key"]
        sample_to_feature[entry["sample_id"]] = key
        if key in by_key and (by_key[key]["rgb_sha256"], by_key[key]["depth_sha256"]) != (
            entry["rgb_sha256"], entry["depth_sha256"]
        ):
            raise TestIIDFeatureError(f"feature-key collision: {key}")
        by_key.setdefault(key, entry)
    if len(sample_to_feature) != 1000 or len(by_key) != EXPECTED_UNIQUE_PAIRS:
        raise TestIIDFeatureError("feature/sample coverage differs")

    runtime_seed = 24082026
    seed_runtime(runtime_seed, strict=False)
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
            rgb_path = safe_workspace_path(entry["rgb_path"])
            depth_path = safe_workspace_path(entry["depth_path"])
            if sha256_file(rgb_path) != entry["rgb_sha256"] or sha256_file(depth_path) != entry["depth_sha256"]:
                raise TestIIDFeatureError(f"observable input hash changed: {entry['sample_id']}")
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
                    or not tensor_valid(tensor_path)
                ):
                    raise TestIIDFeatureError(f"existing cache pair invalid: {key}")
                reused_count += 1
            else:
                if tensor_path.exists() or metadata_path.exists():
                    raise TestIIDFeatureError(f"partial cache pair: {key}")
                if model is None:
                    model = load_model(model_root)
                seed_runtime(runtime_seed, strict=False)
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
                metadata = {
                    "schema_version": 1, "protocol_id": PROTOCOL_ID,
                    "feature_key": key, "rgb_sha256": entry["rgb_sha256"],
                    "depth_sha256": entry["depth_sha256"],
                    "model_inventory_sha256": inventory,
                    "raw_stage": "tower_output_before_projector",
                    "pooling": "locked_3x4_tiles_avgpool4_layernorm_v1",
                    "tensor_shapes": {name: list(value.shape) for name, value in payload.items()},
                    "dtype": "torch.float16", "sha256": sha256_file(tensor_path),
                    "bytes": tensor_path.stat().st_size,
                    "observable_input_summary": summary,
                    "runtime": {"preprocess_latency_ms": extracted["preprocess_latency_ms"],
                                "feature_latency_ms": extracted["feature_latency_ms"]},
                    "safety": {"evaluator_artifacts_read": False, "oracle_labels_cached": False,
                               "model_training_performed": False, "frozen_test_inference": True},
                }
                if forbidden_cache_findings(metadata):
                    raise TestIIDFeatureError(f"generated metadata contains oracle key: {key}")
                write_json(metadata_path, metadata)
                new_count += 1
                latencies.append(float(extracted["feature_latency_ms"]))
            rows[key] = {
                "path": str(tensor_path.relative_to(CACHE_ROOT)),
                "sha256": sha256_file(tensor_path), "bytes": tensor_path.stat().st_size,
                "rgb_sha256": entry["rgb_sha256"], "depth_sha256": entry["depth_sha256"],
                "tensor_shapes": EXPECTED_SHAPES, "dtype": "torch.float16",
                "observable_input_summary": summary,
            }
            if position % 25 == 0 or position == len(by_key):
                print(
                    f"TEST_IID_FEATURE_CACHE {position}/{len(by_key)} "
                    f"new={new_count} reused={reused_count}", flush=True
                )
    finally:
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    index = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "scope": "full",
        "status": "COMPLETE", "created_at_utc": utc_now(),
        "model_inventory_sha256": inventory,
        "feature_manifest_sha256": sha256_file(MANIFEST_PATH),
        "features": rows, "sample_to_feature": sample_to_feature,
        "counts": {"samples": 1000, "unique_feature_pairs": len(rows),
                   "newly_extracted": new_count, "reused_verified": reused_count},
        "runtime": {"elapsed_seconds": time.perf_counter() - started,
                    "median_new_feature_latency_ms": float(np.median(latencies)) if latencies else None,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "torch": torch.__version__},
        "safety": {"inference_manifest_only": True, "evaluator_artifacts_read": False,
                   "model_training_performed": False, "oracle_labels_cached": False,
                   "frozen_test_iid_inference": True},
    }
    if forbidden_cache_findings(index):
        raise TestIIDFeatureError("generated feature index contains oracle key")
    index_path = index_root / "index_full.json"
    write_json(index_path, index)
    report = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "checkpoint": "TEST_IID_FEATURE_CACHE_COMPLETE", "created_at_utc": utc_now(),
        "feature_index": str(index_path.relative_to(ROOT)),
        "feature_index_sha256": sha256_file(index_path), **index["counts"],
        "training_performed": False,
    }
    write_json(TEST_ROOT / "dataset/report_assets/checkpoints/03_feature_cache.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2, sort_keys=True))
