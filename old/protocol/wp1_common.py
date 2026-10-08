#!/usr/bin/env python3
"""Shared, oracle-free primitives for the WP1 depth-sensitivity protocol."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import math
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np


PROTOCOL_ID = "roborefer_depth_sensitivity_v1"
PILOT_PROTOCOL_ID = "roborefer_pilot_v0"
RANDOM_SEED = 8132026
EXPECTED_MODEL_INVENTORY_SHA256 = (
    "5508fb49888c1f62b875d928266f7fff3e1224c795f6fc8b971631143a45a7fa"
)
CONDITIONS = (
    "rgb_only",
    "correct_depth",
    "flat_depth",
    "shuffled_depth",
    "inverted_depth",
    "localized_holes_edge",
)
CORRUPTED_DEPTH_CONDITIONS = CONDITIONS[2:]
STRICT_POINT_ANSWER = re.compile(
    r"^\s*\[\s*\(\s*-?(?:\d+(?:\.\d*)?|\.\d+)\s*,\s*"
    r"-?(?:\d+(?:\.\d*)?|\.\d+)\s*\)\s*\]\s*$"
)
FORBIDDEN_INFERENCE_KEY_TOKENS = {
    "target",
    "targetid",
    "targetmodel",
    "targetpose",
    "targetpixel",
    "targetbbox",
    "targetmask",
    "targetsemanticlabel",
    "targetsemanticlabels",
    "tasktype",
    "scenefamilyid",
    "semanticlabel",
    "semanticlabels",
    "plausiblemodel",
    "plausiblemodels",
    "plausiblesemanticlabel",
    "plausiblesemanticlabels",
    "absentmodel",
    "absentsemanticlabel",
    "oracle",
    "groundtruth",
    "annotation",
    "annotations",
}


class WP1Error(RuntimeError):
    """Raised when a locked WP1 invariant is violated."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WP1Error(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WP1Error(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WP1Error(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise WP1Error(f"JSONL record is not an object: {path}:{line_number}")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for record in records:
            stream.write(canonical_json_bytes(record))


def pilot_tree_digest(workspace_root: Path, pilot_relative: str) -> str:
    """Match the WP0 shell digest, including workspace-relative path strings."""

    pilot = workspace_root / pilot_relative
    relative_paths = [
        str(path.relative_to(workspace_root))
        for path in pilot.rglob("*")
        if path.is_file()
    ]
    # WP0 used GNU ``sort -z`` under the workspace locale.  Reuse that exact
    # collation instead of Python's code-point ordering so the historical
    # df8ec3... digest remains comparable byte for byte.
    sorted_bytes = subprocess.run(
        ["sort", "-z"],
        input=b"\0".join(path.encode("utf-8") for path in relative_paths) + b"\0",
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    digest = hashlib.sha256()
    for raw_relative in sorted_bytes.rstrip(b"\0").split(b"\0"):
        relative = raw_relative.decode("utf-8")
        digest.update(
            f"{sha256_file(workspace_root / relative)}  {relative}\n".encode("utf-8")
        )
    return digest.hexdigest()


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_forbidden_inference_keys(value: Any, prefix: str = "$") -> list[str]:
    findings = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            child_path = f"{prefix}.{raw_key}"
            if normalize_key(str(raw_key)) in FORBIDDEN_INFERENCE_KEY_TOKENS:
                findings.append(child_path)
            findings.extend(find_forbidden_inference_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_forbidden_inference_keys(child, f"{prefix}[{index}]"))
    return findings


def parse_point(answer: str, width: int, height: int) -> dict:
    try:
        value = ast.literal_eval(str(answer).strip())
    except (SyntaxError, ValueError):
        return {"parse_status": "NO_POINT", "normalized_points_xy": [], "pixel_points_xy": []}
    if not isinstance(value, list) or len(value) != 1:
        return {"parse_status": "MULTIPLE_OR_INVALID_POINTS", "normalized_points_xy": [], "pixel_points_xy": []}
    point = value[0]
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return {"parse_status": "NO_POINT", "normalized_points_xy": [], "pixel_points_xy": []}
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in point):
        return {"parse_status": "NO_POINT", "normalized_points_xy": [], "pixel_points_xy": []}
    x_value, y_value = float(point[0]), float(point[1])
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in (x_value, y_value)):
        return {"parse_status": "OUT_OF_RANGE", "normalized_points_xy": [[x_value, y_value]], "pixel_points_xy": []}
    pixel = [
        min(width - 1, int(round(x_value * (width - 1)))),
        min(height - 1, int(round(y_value * (height - 1)))),
    ]
    status = (
        "EXACT_ONE_NORMALIZED_POINT"
        if STRICT_POINT_ANSWER.fullmatch(str(answer))
        else "FORMAT_VIOLATION"
    )
    return {
        "parse_status": status,
        "normalized_points_xy": [[x_value, y_value]],
        "pixel_points_xy": [pixel],
    }


def encode_image(image: np.ndarray, extension: str, params=None) -> bytes:
    success, encoded = cv2.imencode(extension, image, params or [])
    if not success:
        raise WP1Error(f"cannot encode image as {extension}")
    return encoded.tobytes()


def normalize_metric_depth(depth_m: np.ndarray) -> np.ndarray:
    """Locked inverse-depth-percentile transform used by the pilot."""

    values = np.asarray(depth_m, dtype=np.float32)
    if values.ndim != 2:
        raise WP1Error(f"depth input must be HxW, got {values.shape}")
    valid = np.isfinite(values) & (values >= 0.05) & (values <= 2.0)
    if int(np.count_nonzero(valid)) < 16:
        raise WP1Error("registered depth contains fewer than 16 valid pixels")
    valid_depth = values[valid]
    near = float(np.percentile(valid_depth, 2.0))
    far = float(np.percentile(valid_depth, 98.0))
    if far - near < 1e-4:
        near, far = float(valid_depth.min()), float(valid_depth.max())
    if far - near < 1e-6:
        raise WP1Error("registered depth has no usable range")
    clipped = np.clip(values, near, far)
    inverse = 1.0 / np.maximum(clipped, 1e-6)
    normalized = (inverse - 1.0 / far) / max(1.0 / near - 1.0 / far, 1e-6)
    gray = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    gray[~valid] = 0
    return np.repeat(gray[..., np.newaxis], 3, axis=-1)


def flat_depth_view(correct: np.ndarray, value: int = 127) -> np.ndarray:
    return np.full_like(np.asarray(correct, dtype=np.uint8), int(value), dtype=np.uint8)


def inverted_depth_view(correct: np.ndarray) -> np.ndarray:
    return 255 - np.asarray(correct, dtype=np.uint8)


def localized_holes_edge_view(correct: np.ndarray) -> tuple[np.ndarray, dict]:
    """Deterministically remove depth edges and three fixed normalized patches."""

    source = np.asarray(correct, dtype=np.uint8)
    gray = source[..., 0]
    edge_mask = cv2.Canny(gray, 24, 72) > 0
    edge_mask = cv2.dilate(edge_mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    height, width = gray.shape
    mask = edge_mask.copy()
    patch_width = max(8, int(round(0.10 * width)))
    patch_height = max(8, int(round(0.10 * height)))
    centers = ((0.25, 0.35), (0.50, 0.50), (0.75, 0.65))
    rectangles = []
    for normalized_x, normalized_y in centers:
        center_x = int(round(normalized_x * (width - 1)))
        center_y = int(round(normalized_y * (height - 1)))
        x0, x1 = max(0, center_x - patch_width // 2), min(width, center_x + patch_width // 2)
        y0, y1 = max(0, center_y - patch_height // 2), min(height, center_y + patch_height // 2)
        mask[y0:y1, x0:x1] = True
        rectangles.append([x0, y0, x1 - 1, y1 - 1])
    output = source.copy()
    output[mask] = 0
    return output, {
        "edge_detector": "Canny(24,72)+dilate_7x7",
        "fixed_normalized_patch_centers": [list(item) for item in centers],
        "rectangles_xyxy": rectangles,
        "corrupted_pixel_count": int(np.count_nonzero(mask)),
        "corrupted_fraction": float(np.count_nonzero(mask) / mask.size),
    }


def shuffled_source_scene(scene_ids: Sequence[str], scene_id: str) -> str:
    index = list(scene_ids).index(scene_id)
    return list(scene_ids)[(index + 1) % len(scene_ids)]


def query_model(
    server_url: str,
    prompt: str,
    rgb_jpeg: bytes,
    depth_png: Optional[bytes],
    timeout_sec: float,
    expected_model_inventory_sha256: str,
) -> tuple[str, float, dict]:
    body = {
        "image_url": [base64.b64encode(rgb_jpeg).decode("ascii")],
        "depth_url": [base64.b64encode(depth_png).decode("ascii")] if depth_png is not None else [],
        "enable_depth": 1 if depth_png is not None else 0,
        "text": prompt,
    }
    request_bytes = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        server_url.rstrip("/") + "/query",
        data=request_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    import time

    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_sec)) as response:
            response_bytes = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise WP1Error(f"RoboRefer HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise WP1Error(f"RoboRefer unavailable: {exc}") from exc
    latency_ms = 1000.0 * (time.monotonic() - started)
    payload = json.loads(response_bytes.decode("utf-8"))
    if status != 200 or payload.get("result") != 1:
        raise WP1Error(f"RoboRefer response failed: HTTP={status}, payload={payload}")
    if payload.get("generation_mode") != "greedy" or payload.get("random_seed") != RANDOM_SEED:
        raise WP1Error("server is not locked to greedy decoding and seed 8132026")
    expected_generation = {"do_sample": False, "temperature": None, "top_k": None, "top_p": None}
    if payload.get("generation_config") != expected_generation:
        raise WP1Error(f"unexpected generation config: {payload.get('generation_config')}")
    fingerprint = payload.get("model_fingerprint", {})
    if fingerprint.get("inventory_sha256") != expected_model_inventory_sha256:
        raise WP1Error(f"checkpoint fingerprint mismatch: {fingerprint}")
    answer = str(payload.get("answer", "")).strip()
    return answer, latency_ms, {
        "http_status": status,
        "response_sha256": sha256_bytes(response_bytes),
        "response_body_base64": base64.b64encode(response_bytes).decode("ascii"),
        "response_json": payload,
        "model_fingerprint": fingerprint,
        "generation_mode": payload["generation_mode"],
        "generation_config": payload["generation_config"],
        "random_seed": payload["random_seed"],
    }


def decision_from_metrics(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Apply the preregistered, exhaustive three-way P-CRA-F decision rule."""

    reasons = []
    correct = metrics["condition_summary"]["correct_depth"]
    rgb = metrics["condition_summary"]["rgb_only"]
    if correct["strict_parse_rate"] < float(thresholds["required_correct_depth_parse_rate"]):
        reasons.append("CORRECT_DEPTH_PARSE_GATE_FAILED")
    if correct["point_hit_rate_single_targets"] < float(thresholds["minimum_correct_depth_point_hit_rate"]):
        reasons.append("CORRECT_DEPTH_ACCURACY_GATE_FAILED")
    if correct["point_hit_count"] < rgb["point_hit_count"] - int(thresholds["maximum_correct_depth_hit_deficit_vs_rgb"]):
        reasons.append("CORRECT_DEPTH_WORSE_THAN_RGB")
    if reasons:
        return "FIX_DEPTH_PIPELINE_FIRST", reasons

    sensitive_rate = float(metrics["corruption_aggregate"]["displacement_sensitive_rate"])
    switch_rate = float(metrics["corruption_aggregate"]["instance_switch_rate"])
    low_sensitivity = (
        sensitive_rate <= float(thresholds["go_pcra_f_max_sensitive_pair_rate"])
        and switch_rate <= float(thresholds["go_pcra_f_max_instance_switch_rate"])
    )
    if low_sensitivity:
        return "GO_PCRA_F", ["EXISTING_MODEL_NEARLY_INVARIANT_TO_DEPTH_COUNTERFACTUALS"]
    return "NO_GO_PCRA_F", [
        "EXISTING_MODEL_RESPONDS_TO_DEPTH_COUNTERFACTUALS",
        "PRIORITIZE_PCRA_U_CALIBRATION_AND_INTERVENTION",
    ]


def validate_prediction_lock(result_root: Path, protocol_manifest_path: Path) -> tuple[dict, list[dict]]:
    """Verify every inference byte before any evaluator/oracle file is opened."""

    result_root = result_root.resolve()
    lock = read_json(result_root / "prediction_lock.json")
    if lock.get("status") != "LOCKED" or lock.get("protocol_id") != PROTOCOL_ID:
        raise WP1Error("prediction lock is missing or invalid")
    if sha256_file(protocol_manifest_path) != lock.get("protocol_manifest_sha256"):
        raise WP1Error("protocol manifest differs from pre-inference lock")
    inference_manifest = result_root / "inference/inference_manifest.json"
    predictions_path = result_root / "inference/predictions.jsonl"
    artifact_manifest_path = result_root / "inference/artifact_manifest.json"
    for path, key in (
        (inference_manifest, "inference_manifest_sha256"),
        (predictions_path, "predictions_sha256"),
        (artifact_manifest_path, "artifact_manifest_sha256"),
    ):
        if sha256_file(path) != lock.get(key):
            raise WP1Error(f"locked hash mismatch: {path}")
    artifact_manifest = read_json(artifact_manifest_path)
    for item in artifact_manifest.get("files", []):
        path = result_root / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise WP1Error(f"inference artifact mismatch: {item['path']}")
    records = read_jsonl(predictions_path)
    expected = len(CONDITIONS) * 10
    if len(records) != expected or lock.get("record_count") != expected:
        raise WP1Error(f"expected {expected} locked predictions, got {len(records)}")
    keys = {(item.get("scene_id"), item.get("condition")) for item in records}
    if len(keys) != expected:
        raise WP1Error("scene/condition prediction keys are not unique")
    if {item.get("condition") for item in records} != set(CONDITIONS):
        raise WP1Error("prediction condition set mismatch")
    for record in records:
        expected_hash = record.get("prediction_sha256")
        body = dict(record)
        body.pop("prediction_sha256", None)
        if expected_hash != canonical_json_sha256(body):
            raise WP1Error(f"prediction record hash mismatch: {record.get('record_file')}")
    return lock, records
