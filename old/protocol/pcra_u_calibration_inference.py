#!/usr/bin/env python3
"""Run immutable P-CRA-U raw inference on oracle-free Calibration features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

if __package__:
    from .pcra_u_calibration_feature_cache import validate_schema
    from .pcra_u_development_common import (
        ANSWERABILITY_CLASSES,
        SOURCE_CLASSES,
        PCRAUDevelopmentV1,
        DevelopmentTrainingError,
        chunks,
        parse_relation,
        read_json,
        safe_resolve,
        seed_runtime,
        sha256_file,
        tokenize,
        write_json,
    )
    from .training_checkpoint_manager import load_model_for_evaluation
    from .wp3_feature_hook_smoke import forbidden_cache_findings
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pcra_u_calibration_feature_cache import validate_schema  # type: ignore
    from pcra_u_development_common import (  # type: ignore
        ANSWERABILITY_CLASSES,
        SOURCE_CLASSES,
        PCRAUDevelopmentV1,
        DevelopmentTrainingError,
        chunks,
        parse_relation,
        read_json,
        safe_resolve,
        seed_runtime,
        sha256_file,
        tokenize,
        write_json,
    )
    from training_checkpoint_manager import load_model_for_evaluation  # type: ignore
    from wp3_feature_hook_smoke import forbidden_cache_findings  # type: ignore


WORKSPACE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "pcra_u_calibration_v1"
CONFIG_PATH = WORKSPACE / "protocol/pcra_u_calibration_config.json"
FEATURE_MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_calibration_feature_manifest.json"
CACHE_ROOT = WORKSPACE / "results/pcra_u_feature_cache/pcra_u_calibration_v1"
CACHE_INDEX_PATH = CACHE_ROOT / "indexes/index_full.json"
RUN_ROOT = WORKSPACE / "results/pcra_u_runs/pcra_u_calibration_20260824"
PREDICTIONS_PATH = RUN_ROOT / "predictions/calibration_raw_predictions.jsonl"
ATTESTATION_PATH = RUN_ROOT / "checks/calibration_raw_inference_manifest.json"
ARCHITECTURE_LOCK_PATH = WORKSPACE / "protocol/pcra_u_architecture_freeze_lock.json"

DEVELOPMENT_CONFIG = WORKSPACE / "protocol/pcra_u_development_train_config.json"
DEVELOPMENT_MANIFEST = WORKSPACE / "protocol/pcra_u_development_train_manifest.json"
DEVELOPMENT_FEATURE_MANIFEST = WORKSPACE / "protocol/pcra_u_development_feature_manifest.json"
DEVELOPMENT_FEATURE_INDEX = WORKSPACE / "results/pcra_u_feature_cache/pcra_u_development_v1/indexes/index_full.json"
DEVELOPMENT_CONTRACT = WORKSPACE / "protocol/PCRA_U_DEVELOPMENT_TRAIN_CONTRACT.md"
DEVELOPMENT_EXECUTION_LOCK = WORKSPACE / "protocol/pcra_u_development_execution_lock.json"
DEVELOPMENT_TRAIN_CODE = WORKSPACE / "protocol/pcra_u_development_train.py"
DEVELOPMENT_COMMON_CODE = WORKSPACE / "protocol/pcra_u_development_common.py"
CHECKPOINT_MANAGER_CODE = WORKSPACE / "protocol/training_checkpoint_manager.py"


class CalibrationInferenceError(DevelopmentTrainingError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def development_checkpoint_identity() -> dict[str, str]:
    """Recreate the original development identity, not a Calibration identity."""
    return {
        "config_sha256": sha256_file(DEVELOPMENT_CONFIG),
        "train_manifest_sha256": sha256_file(DEVELOPMENT_MANIFEST),
        "feature_manifest_sha256": sha256_file(DEVELOPMENT_FEATURE_MANIFEST),
        "feature_cache_index_sha256": sha256_file(DEVELOPMENT_FEATURE_INDEX),
        "contract_sha256": sha256_file(DEVELOPMENT_CONTRACT),
        "execution_lock_sha256": sha256_file(DEVELOPMENT_EXECUTION_LOCK),
        "training_code_sha256": sha256_file(DEVELOPMENT_TRAIN_CODE),
        "common_code_sha256": sha256_file(DEVELOPMENT_COMMON_CODE),
        "checkpoint_manager_sha256": sha256_file(CHECKPOINT_MANAGER_CODE),
    }


def connected_mode_count(probability: np.ndarray) -> int:
    maximum = float(probability.max())
    if maximum <= 0.0:
        return 0
    tensor = torch.from_numpy(probability).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)[0, 0].numpy()
    candidates = (probability >= 0.5 * maximum) & np.isclose(probability, pooled, rtol=0.0, atol=1e-7)
    component_count, _ = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=8)
    return int(component_count - 1)


def summarize_prediction(
    heatmap_logits: torch.Tensor,
    answer_logits: torch.Tensor,
    source_logits: torch.Tensor,
    r_thumb: torch.Tensor,
    d_thumb: torch.Tensor,
    zero_depth_fraction: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    heat_logits = heatmap_logits.detach().float().cpu().numpy()
    probability = torch.sigmoid(heatmap_logits).detach().float().cpu().numpy()
    unit_mass = probability / max(float(probability.sum()), 1e-12)
    flat = int(np.argmax(probability))
    grid_y, grid_x = divmod(flat, 32)
    map_x = min(639, int((grid_x + 0.5) / 32 * 640))
    map_y = min(479, int((grid_y + 0.5) / 24 * 480))
    y0, y1 = max(0, grid_y - 1), min(24, grid_y + 2)
    x0, x1 = max(0, grid_x - 1), min(32, grid_x + 2)
    positive = unit_mass[unit_mass > 0]
    normalized_entropy = float(-(positive * np.log(positive)).sum() / math.log(24 * 32))
    top = np.partition(probability.reshape(-1), -2)[-2:]
    peak_margin = float(top.max() - top.min())
    local_mass = float(unit_mass[y0:y1, x0:x1].sum())
    mode_count = connected_mode_count(probability)
    answer_values = answer_logits.detach().float().cpu()
    source_values = source_logits.detach().float().cpu()
    answer_probability = torch.softmax(answer_values, dim=0)
    source_probability = torch.sigmoid(source_values)
    found_log_odds = float((answer_values[0] - torch.logsumexp(answer_values[1:], dim=0)).item())
    cosine = float(F.cosine_similarity(r_thumb.float(), d_thumb.float(), dim=0).item())
    mean_gap = float(torch.mean(torch.abs(r_thumb.float() - d_thumb.float())).item())
    spatial_terms = config["raw_evidence"]["spatial_score"]
    spatial_score = (
        float(spatial_terms["answer_found_log_odds"]) * found_log_odds
        + float(spatial_terms["one_minus_normalized_entropy"]) * (1.0 - normalized_entropy)
        + float(spatial_terms["local_map_mass_3x3"]) * local_mass
        + float(spatial_terms["top1_top2_peak_margin"]) * peak_margin
        + float(spatial_terms["log1p_mode_count"]) * math.log1p(mode_count)
    )
    additions = config["raw_evidence"]["multimodal_score_additions"]
    mean_source_probability = float(source_probability.mean().item())
    multimodal_score = (
        spatial_score
        + float(additions["mean_source_probability"]) * mean_source_probability
        + float(additions["rgb_depth_thumbnail_cosine"]) * cosine
        + float(additions["rgb_depth_thumbnail_mean_absolute_gap"]) * mean_gap
        + float(additions["invalid_depth_fraction"]) * zero_depth_fraction
    )
    heat_bytes = np.ascontiguousarray(heat_logits.astype("<f4")).tobytes()
    return {
        "answerability_logits": [float(value) for value in answer_values.tolist()],
        "answerability_probabilities": {
            name: float(answer_probability[index].item()) for index, name in enumerate(ANSWERABILITY_CLASSES)
        },
        "source_logits": [float(value) for value in source_values.tolist()],
        "source_probabilities": {
            name: float(source_probability[index].item()) for index, name in enumerate(SOURCE_CLASSES)
        },
        "heatmap_grid_shape": [24, 32],
        "heatmap_logits": [float(value) for value in heat_logits.reshape(-1).tolist()],
        "heatmap_logits_float32_sha256": hashlib.sha256(heat_bytes).hexdigest(),
        "map_grid_x": grid_x,
        "map_grid_y": grid_y,
        "map_x": map_x,
        "map_y": map_y,
        "answer_found_log_odds": found_log_odds,
        "normalized_heatmap_entropy": normalized_entropy,
        "local_map_mass_3x3": local_mass,
        "top1_top2_peak_margin": peak_margin,
        "mode_count": mode_count,
        "mean_source_probability": mean_source_probability,
        "rgb_depth_thumbnail_cosine": cosine,
        "rgb_depth_thumbnail_mean_absolute_gap": mean_gap,
        # The model input does not encode a separate validity plane.  This is
        # transparently the observable zero-pixel proxy committed by the cache.
        "invalid_depth_fraction": zero_depth_fraction,
        "invalid_depth_fraction_definition": "zero_valued_depth_model_input_pixel_fraction_observable_proxy",
        "spatial_raw_score": float(spatial_score),
        "multimodal_raw_score": float(multimodal_score),
    }


def load_feature_batch(
    entries: Sequence[Mapping[str, Any]], index: Mapping[str, Any], device: torch.device,
    development_config: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    tensors_by_name: dict[str, list[torch.Tensor]] = {name: [] for name in ("R0_GRID", "D0_GRID", "R0_THUMB", "D0_THUMB")}
    tokens, relation_ids, metadata = [], [], []
    relation_names = development_config["language"]["relations"]
    for entry in entries:
        key = index["sample_to_feature"][entry["sample_id"]]
        row = index["features"][key]
        path = safe_resolve(CACHE_ROOT, row["path"])
        if sha256_file(path) != row["sha256"]:
            raise CalibrationInferenceError(f"Calibration cached feature hash mismatch: {key}")
        loaded = load_safetensors(str(path), device="cpu")
        if set(loaded) != set(tensors_by_name):
            raise CalibrationInferenceError(f"Unexpected cached tensor names: {key}")
        for name in tensors_by_name:
            tensors_by_name[name].append(loaded[name])
        prompt = entry["prompt"]
        relation = parse_relation(prompt)
        tokens.append(tokenize(prompt, int(development_config["language"]["max_tokens"]), int(development_config["language"]["vocab_size"])))
        relation_ids.append(relation_names.index(relation))
        metadata.append({"feature_key": key, "relation": relation,
                         "zero_depth_fraction": float(row["observable_input_summary"]["zero_valued_depth_input_fraction"])})
    batch = {
        "r0": torch.stack(tensors_by_name["R0_GRID"]).to(device=device, dtype=torch.float32),
        "d0": torch.stack(tensors_by_name["D0_GRID"]).to(device=device, dtype=torch.float32),
        "r_thumb": torch.stack(tensors_by_name["R0_THUMB"]).to(device=device, dtype=torch.float32),
        "d_thumb": torch.stack(tensors_by_name["D0_THUMB"]).to(device=device, dtype=torch.float32),
        "tokens": torch.tensor(tokens, dtype=torch.long, device=device),
        "relation_ids": torch.tensor(relation_ids, dtype=torch.long, device=device),
    }
    return batch, metadata


def run() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    development_config = read_json(DEVELOPMENT_CONFIG)
    feature_manifest = read_json(FEATURE_MANIFEST_PATH)
    index = read_json(CACHE_INDEX_PATH)
    validate_schema(index)
    if (
        feature_manifest.get("protocol_id") != PROTOCOL_ID
        or feature_manifest.get("sample_count") != 1000
        or feature_manifest.get("split") != "calibration"
        or index.get("counts", {}).get("samples") != 1000
        or sha256_file(FEATURE_MANIFEST_PATH) != index.get("feature_manifest_sha256")
    ):
        raise CalibrationInferenceError("Calibration oracle-free feature inputs are incomplete")
    if forbidden_cache_findings(feature_manifest) or forbidden_cache_findings(index):
        raise CalibrationInferenceError("Oracle/evaluator-like key found at raw inference boundary")
    architecture_lock = read_json(ARCHITECTURE_LOCK_PATH)
    checkpoint = safe_resolve(WORKSPACE, config["frozen_model"]["checkpoint"])
    if (
        architecture_lock.get("architecture_frozen") is not True
        or architecture_lock.get("checkpoint_frozen") is not True
        or architecture_lock["selected_checkpoint"]["path"] != config["frozen_model"]["checkpoint"]
        or sha256_file(checkpoint / "model.safetensors") != config["frozen_model"]["model_sha256"]
    ):
        raise CalibrationInferenceError("Frozen architecture/checkpoint binding changed")
    expected_identity = development_checkpoint_identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Preserve the locked master seed while satisfying NumPy's uint32 runtime
    # requirement locally.  Shared development/baseline code stays untouched.
    runtime_seed = int(config["seeds"]["master"]) % (2**32)
    seed_runtime(runtime_seed, strict=device.type == "cuda")
    model = PCRAUDevelopmentV1(development_config).to(device)
    loaded = load_model_for_evaluation(checkpoint, model=model, expected_identity=expected_identity)
    if not loaded["verification"]["passed"] or loaded["incompatible_model_keys"] != {"missing": [], "unexpected": []}:
        raise CalibrationInferenceError("Safe frozen-checkpoint load failed")
    model.eval().requires_grad_(False)
    entries = feature_manifest["entries"]
    predictions: list[dict[str, Any]] = []
    batch_size = int(development_config["optimization"]["batch_size"])
    started = time.perf_counter()
    with torch.inference_mode():
        for indices in chunks(list(range(len(entries))), batch_size):
            selected = [entries[index_value] for index_value in indices]
            batch, observable = load_feature_batch(selected, index, device, development_config)
            output = model(batch)
            for local_index, entry in enumerate(selected):
                row = {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL_ID,
                    "split": "calibration",
                    "sample_id": entry["sample_id"],
                    "family_id": entry["family_id"],
                    "variant": entry["variant"],
                    "prompt_sha256": entry["prompt_sha256"],
                    "feature_key": observable[local_index]["feature_key"],
                    "prompt_parsed_relation": observable[local_index]["relation"],
                    **summarize_prediction(
                        output["heatmap_logits"][local_index], output["answerability_logits"][local_index],
                        output["source_logits"][local_index], batch["r_thumb"][local_index],
                        batch["d_thumb"][local_index], observable[local_index]["zero_depth_fraction"], config,
                    ),
                }
                numeric = [row["spatial_raw_score"], row["multimodal_raw_score"], row["map_x"], row["map_y"]]
                if not all(math.isfinite(float(value)) for value in numeric):
                    raise CalibrationInferenceError(f"Non-finite raw prediction: {entry['sample_id']}")
                predictions.append(row)
            if len(predictions) % 100 == 0 or len(predictions) == len(entries):
                print(f"CALIBRATION_RAW_INFERENCE {len(predictions)}/{len(entries)}", flush=True)
    if len(predictions) != 1000 or len({row["sample_id"] for row in predictions}) != 1000:
        raise CalibrationInferenceError("Raw Calibration prediction coverage mismatch")
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PREDICTIONS_PATH.parent / f".{PREDICTIONS_PATH.name}.tmp-{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, PREDICTIONS_PATH)
    attestation = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE_ORACLE_FREE_RAW_INFERENCE",
        "created_at_utc": utc_now(),
        "split": "calibration",
        "prediction_count": 1000,
        "predictions_path": str(PREDICTIONS_PATH.relative_to(WORKSPACE)),
        "predictions_sha256": sha256_file(PREDICTIONS_PATH),
        "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
        "feature_cache_index_sha256": sha256_file(CACHE_INDEX_PATH),
        "checkpoint_path": str(checkpoint.relative_to(WORKSPACE)),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "checkpoint_original_development_identity": expected_identity,
        "architecture_freeze_lock_sha256": sha256_file(ARCHITECTURE_LOCK_PATH),
        "inference_code_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(CONFIG_PATH),
        "runtime": {"elapsed_seconds": time.perf_counter() - started, "device": str(device),
                    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                    "torch": torch.__version__, "optimizer_steps": 0},
        "oracle_or_annotation_read": False,
        "model_training_performed": False,
        "checkpoint_selection_performed": False,
        "test_opened": False,
        "test_iid_opened": False,
        "test_ood_opened": False,
        "robot_publish": False,
        "baseline_modified": False,
    }
    write_json(ATTESTATION_PATH, attestation)
    return attestation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
