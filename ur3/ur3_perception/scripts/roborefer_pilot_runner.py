#!/usr/bin/env python3
"""Run the preregistered RoboRefer B0/B1/B2 pilot without robot motion.

B0 and B1 receive byte-identical RGB JPEGs and byte-identical prompts.  B0
disables the depth stream, while B1 supplies registered camera depth.  B2 does
not call the model: it applies the frozen generic depth-component gate to the
already persisted B1 point.  The runner never imports evaluator annotations,
semantic labels, Gazebo object poses, or any manipulation interface.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
import numpy as np
import yaml

from roborefer_grounder import (
    normalize_depth_for_roborefer,
    segment_seeded_depth_component,
)
from spatial_point_utils import parse_points, points_to_pixels


SCHEMA_VERSION = 1
PROTOCOL_ID = "roborefer_pilot_v0"
EXPECTED_SCENE_COUNT = 10
MODES = ("B0", "B1", "B2")
PILOT_RANDOM_SEED = 8132026
REQUEST_PREPROCESSING = {
    "rgb": {
        "source": "locked_capture_rgb_png",
        "encoding": "jpeg",
        "quality": 90,
    },
    "depth": {
        "source": "locked_registered_metric_depth_npy",
        "transform": "inverse_depth_percentile_v1",
        "valid_min_m": 0.05,
        "valid_max_m": 2.0,
        "near_percentile": 2.0,
        "far_percentile": 98.0,
        "encoding": "png",
        "channels": 3,
    },
}
FORBIDDEN_INPUT_KEYS = {
    "scene_family_id",
    "task_type",
    "target_model",
    "target_id",
    "target_pose",
    "target_pixel",
    "target_bbox",
    "target_mask",
    "semantic_label",
    "semantic_labels",
    "plausible_models",
    "plausible_semantic_labels",
    "absent_model",
    "absent_semantic_label",
    "oracle",
    "ground_truth",
}
STRICT_POINT_ANSWER = re.compile(
    r"^\s*\[\s*\(\s*-?(?:\d+(?:\.\d*)?|\.\d+)\s*,\s*"
    r"-?(?:\d+(?:\.\d*)?|\.\d+)\s*\)\s*\]\s*$"
)


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


def write_json(path: Path, payload: Any) -> None:
    path.write_bytes(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def find_forbidden_keys(value: Any, prefix: str = "$") -> List[str]:
    """Return paths of evaluator/oracle keys found in inference input data."""
    findings: List[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            child_path = f"{prefix}.{raw_key}"
            if key in FORBIDDEN_INPUT_KEYS:
                findings.append(child_path)
            findings.extend(find_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_forbidden_keys(child, f"{prefix}[{index}]"))
    return findings


def parse_model_answer(
    answer: str, width: int, height: int
) -> Tuple[str, List[Tuple[float, float]], List[Tuple[int, int]]]:
    """Parse a response while distinguishing model-format failure modes."""
    points = parse_points(str(answer))
    if not points:
        return "NO_POINT", [], []
    if len(points) != 1:
        return "MULTIPLE_POINTS", points, points_to_pixels(points, width, height)
    if any(value < 0.0 or value > 1.0 for value in points[0]):
        return "OUT_OF_RANGE", points, []
    pixels = points_to_pixels(points, width, height)
    if len(pixels) != 1:
        return "OUT_OF_RANGE", points, []
    if not STRICT_POINT_ANSWER.fullmatch(str(answer)):
        # Coordinates remain auditable, but this violates the preregistered
        # exact-output contract and is not actionable under this pilot.
        return "FORMAT_VIOLATION", points, pixels
    return "EXACT_ONE_NORMALIZED_POINT", points, pixels


def encode_image(image: np.ndarray, extension: str, params=None) -> bytes:
    success, encoded = cv2.imencode(extension, image, params or [])
    if not success:
        raise RuntimeError(f"cannot encode image as {extension}")
    return encoded.tobytes()


def query_model(
    server_url: str,
    prompt: str,
    rgb_jpeg: bytes,
    depth_png: Optional[bytes],
    timeout_sec: float,
    expected_model_inventory_sha256: Optional[str] = None,
) -> Tuple[str, float, dict]:
    """Perform exactly one model request; retry policy is deliberately absent."""
    body = {
        "image_url": [base64.b64encode(rgb_jpeg).decode("ascii")],
        "depth_url": (
            [base64.b64encode(depth_png).decode("ascii")]
            if depth_png is not None
            else []
        ),
        "enable_depth": 1 if depth_png is not None else 0,
        "text": prompt,
    }
    request_body = json.dumps(body, separators=(",", ":")).encode("utf-8")
    http_request = urlrequest.Request(
        server_url.rstrip("/") + "/query",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlrequest.urlopen(http_request, timeout=float(timeout_sec)) as response:
            response_body = response.read()
            http_status = int(response.status)
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"RoboRefer HTTP {exc.code}: {detail}") from exc
    except (urlerror.URLError, TimeoutError) as exc:
        raise RuntimeError(f"RoboRefer unavailable: {exc}") from exc
    latency_ms = 1000.0 * (time.monotonic() - started)
    result = json.loads(response_body.decode("utf-8"))
    if http_status != 200 or int(result.get("result", 0)) != 1:
        raise RuntimeError(f"RoboRefer unsuccessful response: {result}")
    if result.get("generation_mode") != "greedy":
        raise RuntimeError(
            "RoboRefer server is not locked to greedy decoding: "
            f"{result.get('generation_mode')!r}"
        )
    if result.get("random_seed") != PILOT_RANDOM_SEED:
        raise RuntimeError(
            f"RoboRefer server seed differs from pilot lock: {result.get('random_seed')!r}"
        )
    generation_config = result.get("generation_config", {})
    if generation_config != {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }:
        raise RuntimeError(
            f"unexpected server generation config: {generation_config!r}"
        )
    server_model = result.get("model_fingerprint", {})
    if (
        expected_model_inventory_sha256 is not None
        and server_model.get("inventory_sha256")
        != expected_model_inventory_sha256
    ):
        raise RuntimeError(
            "server checkpoint fingerprint differs from locked model: "
            f"{server_model!r}"
        )
    return str(result.get("answer", "")).strip(), latency_ms, {
        "http_status": http_status,
        "response_sha256": sha256_bytes(response_body),
        "response_body_base64": base64.b64encode(response_body).decode("ascii"),
        "response_json": result,
        "generation_mode": "greedy",
        "generation_config": generation_config,
        "model_fingerprint": server_model,
    }


def model_fingerprint(model_root: Path) -> dict:
    """Hash the local checkpoint files used for this frozen pilot."""
    if not model_root.is_dir():
        raise ValueError(f"model directory not found: {model_root}")
    entries = []
    for path in sorted(item for item in model_root.rglob("*") if item.is_file()):
        entries.append({
            "path": str(path.relative_to(model_root)),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    if not entries:
        raise ValueError(f"model directory is empty: {model_root}")
    inventory_hash = sha256_bytes(canonical_json_bytes(entries))
    return {
        "model_name": model_root.name,
        "model_root": str(model_root),
        "file_count": len(entries),
        "inventory_sha256": inventory_hash,
        "files": entries,
    }


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _record_hash(record: dict) -> str:
    return sha256_bytes(canonical_json_bytes(record))


def _base_record(
    scene: dict,
    mode: str,
    prompt: str,
    width: int,
    height: int,
    rgb_hash: str,
    depth_hash: Optional[str],
) -> dict:
    request_hashes = {"rgb": rgb_hash, "depth": depth_hash}
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "scene_id": scene["scene_id"],
        "mode": mode,
        "policy": (
            "no_uncertainty_gate_shadow"
            if mode in ("B0", "B1")
            else "hard_gate_shadow"
        ),
        "instruction": scene["instruction"],
        "prompt": prompt,
        "image_width": int(width),
        "image_height": int(height),
        "request_rgb_sha256": rgb_hash,
        "request_depth_sha256": depth_hash,
        "request_sha256": request_hashes,
        "source_input_sha256": dict(scene["input_sha256"]),
        "target_handoff_published": False,
        "robot_manipulation_performed": False,
        "shadow_evaluation_only": True,
    }


def _save_overlay(
    path: Path,
    rgb: np.ndarray,
    points_xy: Sequence[Tuple[int, int]],
    segmentation: Optional[dict] = None,
) -> None:
    overlay = rgb.copy()
    if segmentation is not None:
        mask = segmentation["mask"] > 0
        tint = np.zeros_like(overlay)
        tint[..., 1] = 230
        overlay[mask] = cv2.addWeighted(
            overlay[mask], 0.55, tint[mask], 0.45, 0.0
        )
        contours, _ = cv2.findContours(
            segmentation["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    for index, (x_pixel, y_pixel) in enumerate(points_xy, start=1):
        cv2.drawMarker(
            overlay,
            (int(x_pixel), int(y_pixel)),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=3,
        )
        cv2.putText(
            overlay,
            f"P{index}",
            (int(x_pixel) + 8, max(18, int(y_pixel) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"cannot write overlay: {path}")


def _load_and_validate_inputs(dataset_root: Path) -> List[dict]:
    capture_manifest_path = dataset_root / "capture_manifest.json"
    input_manifest_path = dataset_root / "input_manifest.jsonl"
    if not capture_manifest_path.is_file() or not input_manifest_path.is_file():
        raise ValueError("dataset is missing capture_manifest.json/input_manifest.jsonl")
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    if capture_manifest.get("status") != "COMPLETE":
        raise ValueError("capture manifest is not COMPLETE")
    if capture_manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("capture protocol mismatch")
    if capture_manifest.get("robot_manipulation_performed") is not False:
        raise ValueError("pilot input is not marked no-manipulation")
    expected_manifest_hash = capture_manifest.get("input_manifest_sha256")
    if expected_manifest_hash != sha256_file(input_manifest_path):
        raise ValueError("input manifest SHA256 does not match capture lock")
    scenes = read_jsonl(input_manifest_path)
    if len(scenes) != EXPECTED_SCENE_COUNT:
        raise ValueError(f"pilot requires exactly {EXPECTED_SCENE_COUNT} scenes")
    if len({record.get("scene_id") for record in scenes}) != len(scenes):
        raise ValueError("input manifest scene_id values are not unique")
    leaked = find_forbidden_keys(scenes)
    if leaked:
        raise ValueError(f"forbidden evaluator keys in inference manifest: {leaked}")
    for scene in scenes:
        if scene.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"protocol mismatch in {scene.get('scene_id')}")
        if scene.get("capture", {}).get("robot_manipulation_performed") is not False:
            raise ValueError(f"scene is not no-manipulation: {scene.get('scene_id')}")
        for key, relative_path in scene.get("input_files", {}).items():
            path = (dataset_root / relative_path).resolve()
            if dataset_root.resolve() not in path.parents:
                raise ValueError(f"input path escapes dataset root: {relative_path}")
            if not path.is_file():
                raise ValueError(f"missing input file: {relative_path}")
            expected_hash = scene.get("input_sha256", {}).get(key)
            if expected_hash != sha256_file(path):
                raise ValueError(f"input SHA256 mismatch: {relative_path}")
    return scenes


def run_pilot(
    dataset_root: Path,
    output_root: Path,
    gate_config_path: Path,
    model_root: Path,
    server_url: str,
    timeout_sec: float = 90.0,
    require_greedy_metadata: bool = True,
) -> Path:
    """Run all 30 records and atomically create the prediction lock."""
    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    gate_config_path = gate_config_path.expanduser().resolve()
    model_root = model_root.expanduser().resolve()
    # This validator opens only inference-facing capture inputs.  It checks
    # sensor quality/provenance and explicitly never traverses evaluator/
    # artifacts, so a bad or frozen camera stream cannot reach the VLM.
    from roborefer_pilot_validate import validate_capture

    validate_capture(dataset_root)
    scenes = _load_and_validate_inputs(dataset_root)
    if output_root.exists():
        raise FileExistsError(f"immutable prediction output already exists: {output_root}")
    gate_config = yaml.safe_load(gate_config_path.read_text(encoding="utf-8"))
    if gate_config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("gate protocol mismatch")
    if gate_config.get("frozen_before_pilot") is not True:
        raise ValueError("gate must be marked frozen_before_pilot")
    if (
        float(gate_config.get("min_depth_m", -1.0))
        != REQUEST_PREPROCESSING["depth"]["valid_min_m"]
        or float(gate_config.get("max_depth_m", -1.0))
        != REQUEST_PREPROCESSING["depth"]["valid_max_m"]
    ):
        raise ValueError("gate depth range differs from pilot request preprocessing")
    gate_hash = sha256_file(gate_config_path)
    capture_manifest_path = dataset_root / "capture_manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    if capture_manifest.get("gate_config_bytes_hashed_before_capture") is not True:
        raise ValueError("capture did not attest that gate bytes were frozen before capture")
    if capture_manifest.get("gate_config_file_sha256") != gate_hash:
        raise ValueError("gate config differs from the copy locked before capture")
    output_root.mkdir(parents=True, exist_ok=False)

    locked_model_fingerprint = model_fingerprint(model_root)
    if (
        locked_model_fingerprint["inventory_sha256"]
        != capture_manifest.get("model_inventory_sha256_preregistered")
    ):
        raise ValueError("model inventory differs from the pretrial source lock")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "RUNNING",
        "created_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_root": str(dataset_root),
        "capture_manifest_sha256": sha256_file(dataset_root / "capture_manifest.json"),
        "input_manifest_sha256": sha256_file(dataset_root / "input_manifest.jsonl"),
        "pretrial_source_lock_sha256": capture_manifest[
            "pretrial_source_lock_sha256"
        ],
        "gate_config_path": str(gate_config_path),
        "gate_config_sha256": gate_hash,
        "server_url": server_url,
        "query_order": "scene_order_then_B0_RGB_only_then_B1_RGB_D",
        "query_attempts_per_mode_per_scene": 1,
        "generation_mode": "greedy",
        "sampling_enabled": False,
        "random_seed": PILOT_RANDOM_SEED,
        "b2_requeries_model": False,
        "same_rgb_and_prompt_for_B0_B1": True,
        "request_preprocessing": REQUEST_PREPROCESSING,
        "target_handoff_published": False,
        "robot_manipulation_performed": False,
        "oracle_or_annotation_read_by_runner": False,
        "model": locked_model_fingerprint,
    }
    write_json(output_root / "prediction_manifest.json", manifest)
    predictions_path = output_root / "predictions.jsonl"
    records: List[dict] = []

    gate_args = {
        "min_depth_m": float(gate_config["min_depth_m"]),
        "max_depth_m": float(gate_config["max_depth_m"]),
        "seed_radius_px": int(gate_config["depth_seed_radius_px"]),
        "roi_radius_px": int(gate_config["depth_roi_radius_px"]),
        "near_tolerance_m": float(gate_config["depth_near_tolerance_m"]),
        "far_tolerance_m": float(gate_config["depth_far_tolerance_m"]),
        "min_area_px": int(gate_config["min_mask_area_px"]),
        "max_area_fraction": float(gate_config["max_mask_area_fraction"]),
        "bbox_padding_px": int(gate_config["bbox_padding_px"]),
    }

    for scene_index, scene in enumerate(scenes, start=1):
        scene_id = str(scene["scene_id"])
        scene_dir = output_root / scene_id
        scene_dir.mkdir(parents=True, exist_ok=False)
        rgb_path = dataset_root / scene["input_files"]["rgb"]
        depth_path = dataset_root / scene["input_files"]["depth_m"]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth_m = np.load(depth_path, allow_pickle=False)
        if rgb is None:
            raise RuntimeError(f"cannot decode RGB: {rgb_path}")
        if depth_m.ndim != 2 or rgb.shape[:2] != depth_m.shape:
            raise RuntimeError(
                f"registered RGB/depth mismatch for {scene_id}: "
                f"{rgb.shape[:2]} vs {depth_m.shape}"
            )
        height, width = rgb.shape[:2]
        rgb_jpeg = encode_image(
            rgb,
            ".jpg",
            [cv2.IMWRITE_JPEG_QUALITY, REQUEST_PREPROCESSING["rgb"]["quality"]],
        )
        depth_view = normalize_depth_for_roborefer(
            depth_m, gate_args["min_depth_m"], gate_args["max_depth_m"]
        )
        depth_png = encode_image(depth_view, ".png")
        rgb_request_path = scene_dir / "request_rgb_q90.jpg"
        depth_request_path = scene_dir / "request_depth_registered_view.png"
        rgb_request_path.write_bytes(rgb_jpeg)
        depth_request_path.write_bytes(depth_png)
        rgb_hash = sha256_bytes(rgb_jpeg)
        depth_hash = sha256_bytes(depth_png)
        prompt = str(scene["instruction"]).strip() + " " + str(
            scene["coordinate_suffix"]
        ).strip()

        mode_records: Dict[str, dict] = {}
        for mode, request_depth in (("B0", None), ("B1", depth_png)):
            record = _base_record(
                scene,
                mode,
                prompt,
                width,
                height,
                rgb_hash,
                depth_hash if request_depth is not None else None,
            )
            record.update({
                "model_query_performed": True,
                "request_enable_depth": request_depth is not None,
                "request_rgb_file": _relative(rgb_request_path, output_root),
                "request_depth_file": (
                    _relative(depth_request_path, output_root)
                    if request_depth is not None
                    else None
                ),
                "query_attempt": 1,
            })
            try:
                answer, latency_ms, response_meta = query_model(
                    server_url,
                    prompt,
                    rgb_jpeg,
                    request_depth,
                    timeout_sec,
                    locked_model_fingerprint["inventory_sha256"],
                )
                response_json = response_meta.get("response_json")
                if (
                    not isinstance(response_json, dict)
                    or str(response_json.get("answer", "")).strip() != answer
                ):
                    raise RuntimeError("query response JSON/answer provenance is invalid")
                if require_greedy_metadata and (
                    response_meta.get("generation_mode") != "greedy"
                    or response_meta.get("generation_config")
                    != {
                        "do_sample": False,
                        "temperature": None,
                        "top_p": None,
                        "top_k": None,
                    }
                ):
                    raise RuntimeError(
                        "query response does not prove locked greedy decoding"
                    )
                parse_status, normalized_points, pixel_points = parse_model_answer(
                    answer, width, height
                )
                record.update({
                    "query_status": "SUCCESS",
                    "raw_answer": answer,
                    "parse_status": parse_status,
                    "normalized_points_xy": [list(point) for point in normalized_points],
                    "pixel_points_xy": [list(point) for point in pixel_points],
                    "inference_latency_ms": round(float(latency_ms), 3),
                    "response": response_meta,
                    "would_execute": parse_status == "EXACT_ONE_NORMALIZED_POINT",
                })
            except Exception as exc:
                record.update({
                    "query_status": "ERROR",
                    "query_error": f"{type(exc).__name__}: {exc}",
                    "raw_answer": "",
                    "parse_status": "QUERY_ERROR",
                    "normalized_points_xy": [],
                    "pixel_points_xy": [],
                    "inference_latency_ms": None,
                    "would_execute": False,
                })
            record_path = scene_dir / f"{mode.lower()}_prediction.json"
            record["record_file"] = _relative(record_path, output_root)
            record["prediction_sha256"] = _record_hash(record)
            write_json(record_path, record)
            _save_overlay(
                scene_dir / f"{mode.lower()}_point_overlay.png",
                rgb,
                [tuple(point) for point in record["pixel_points_xy"]],
            )
            mode_records[mode] = record
            records.append(record)

        b1 = mode_records["B1"]
        if any(
            mode_records[mode].get("query_status") != "SUCCESS"
            for mode in ("B0", "B1")
        ):
            raise RuntimeError(
                f"model query failed for {scene_id}; refusing to create a "
                "COMPLETE manifest or prediction lock"
            )
        b2 = _base_record(
            scene, "B2", prompt, width, height, rgb_hash, depth_hash
        )
        b2.update({
            "model_query_performed": False,
            "query_status": "NOT_QUERIED_DERIVED_FROM_B1",
            "request_enable_depth": None,
            "registered_depth_used_by_gate": True,
            "request_rgb_file": _relative(rgb_request_path, output_root),
            "request_depth_file": _relative(depth_request_path, output_root),
            "source_mode": "B1",
            "source_b1_prediction_sha256": b1["prediction_sha256"],
            "raw_answer": b1["raw_answer"],
            "parse_status": b1["parse_status"],
            "normalized_points_xy": b1["normalized_points_xy"],
            "pixel_points_xy": b1["pixel_points_xy"],
            "inference_latency_ms": 0.0,
        })
        segmentation = None
        gate_started = time.monotonic()
        if b1["parse_status"] != "EXACT_ONE_NORMALIZED_POINT":
            gate_result = {
                "accepted": False,
                "reason": f"B1_{b1['parse_status']}",
                "profile_id": gate_config["gate_profile_id"],
                "mask_file": None,
            }
        else:
            try:
                segmentation = segment_seeded_depth_component(
                    depth_m,
                    [tuple(point) for point in b1["pixel_points_xy"]],
                    **gate_args,
                )
                mask_path = scene_dir / "b2_depth_component_mask.png"
                if not cv2.imwrite(str(mask_path), segmentation["mask"]):
                    raise RuntimeError(f"cannot write mask: {mask_path}")
                exact_support = len(segmentation["supported_points_xy"])
                seed_supported = exact_support == 1
                gate_result = {
                    "accepted": bool(seed_supported),
                    "reason": (
                        "DEPTH_COMPONENT_ACCEPTED"
                        if seed_supported
                        else "DEPTH_COMPONENT_REJECTED: seed pixel is outside mask"
                    ),
                    "profile_id": gate_config["gate_profile_id"],
                    "mask_file": _relative(mask_path, output_root),
                    "mask_sha256": sha256_file(mask_path),
                    "bbox_xyxy": segmentation["bbox_xyxy"],
                    "grasp_pixel_xy": list(segmentation["grasp_pixel_xy"]),
                    "seed_depth_m": float(segmentation["seed_depth_m"]),
                    "mask_area_px": int(segmentation["mask_area_px"]),
                    "mask_depth_min_m": float(segmentation["mask_depth_min_m"]),
                    "mask_depth_max_m": float(segmentation["mask_depth_max_m"]),
                    "supported_points_xy": [
                        list(point) for point in segmentation["supported_points_xy"]
                    ],
                    "geometric_support_ratio": exact_support / 1.0,
                }
            except ValueError as exc:
                gate_result = {
                    "accepted": False,
                    "reason": f"DEPTH_COMPONENT_REJECTED: {type(exc).__name__}: {exc}",
                    "profile_id": gate_config["gate_profile_id"],
                    "mask_file": None,
                }
        gate_result["latency_ms"] = round(
            1000.0 * (time.monotonic() - gate_started), 3
        )
        b2["gate"] = gate_result
        b2["would_execute"] = bool(gate_result["accepted"])
        b2_path = scene_dir / "b2_prediction.json"
        b2["record_file"] = _relative(b2_path, output_root)
        b2["prediction_sha256"] = _record_hash(b2)
        write_json(b2_path, b2)
        _save_overlay(
            scene_dir / "b2_gate_overlay.png",
            rgb,
            [tuple(point) for point in b2["pixel_points_xy"]],
            segmentation,
        )
        records.append(b2)
        print(
            f"PILOT_INFERENCE {scene_index:02d}/{EXPECTED_SCENE_COUNT} "
            f"{scene_id}: B0={mode_records['B0']['parse_status']} "
            f"B1={b1['parse_status']} B2={'ACCEPT' if b2['would_execute'] else 'REJECT'}",
            flush=True,
        )

    with predictions_path.open("wb") as stream:
        for record in records:
            stream.write(canonical_json_bytes(record))
    if len(records) != EXPECTED_SCENE_COUNT * len(MODES):
        raise RuntimeError(f"internal prediction-count error: {len(records)}")
    manifest.update({
        "status": "COMPLETE",
        "completed_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scene_count": EXPECTED_SCENE_COUNT,
        "record_count": len(records),
        "predictions_file": "predictions.jsonl",
        "predictions_sha256": sha256_file(predictions_path),
    })
    manifest_path = output_root / "prediction_manifest.json"
    write_json(manifest_path, manifest)
    lock = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED",
        "locked_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "predictions_sha256": sha256_file(predictions_path),
        "record_count": len(records),
        "oracle_opened_before_lock": False,
        "immutable_warning": "Do not modify predictions after this lock.",
    }
    lock_path = output_root / "prediction_lock.json"
    write_json(lock_path, lock)
    print(f"PILOT_PREDICTIONS_LOCKED: {lock_path}", flush=True)
    return lock_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:25547")
    parser.add_argument("--timeout-sec", type=float, default=90.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        run_pilot(
            dataset_root=arguments.dataset_root,
            output_root=arguments.output_root,
            gate_config_path=arguments.gate_config,
            model_root=arguments.model_root,
            server_url=arguments.server_url,
            timeout_sec=arguments.timeout_sec,
        )
    except Exception as exc:
        print(f"PILOT_RUNNER_FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
