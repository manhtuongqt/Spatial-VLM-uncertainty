#!/usr/bin/env python3
"""Validate the immutable inputs and predictions of the RoboRefer pilot.

This module deliberately has no knowledge of evaluator annotations or Gazebo
semantic-label images.  It validates only the inference-facing capture and the
locked B0/B1/B2 prediction artifacts, so it is safe to run before opening the
oracle side of the dataset.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from roborefer_grounder import segment_seeded_depth_component


SCHEMA_VERSION = 1
PROTOCOL_ID = "roborefer_pilot_v0"
EXPECTED_SCENE_COUNT = 10
EXPECTED_MODES = ("B0", "B1", "B2")

# Compare normalized spellings as well as exact spellings.  This catches easy
# aliases such as semanticLabels and target-model without looking at natural
# language instructions (which necessarily contain object descriptions).
FORBIDDEN_INPUT_KEY_TOKENS = {
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
    "captureoracle",
    "groundtruth",
    "annotation",
    "annotations",
}
FORBIDDEN_PATH_MARKERS = (
    "/evaluator/",
    "evaluator/capture_oracle",
    "capture_oracle",
    "semantic_labels",
    "semantic-labels",
    "annotations.yaml",
    "annotation.yaml",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MIN_RGB_STD = 5.0
MIN_RGB_DYNAMIC_RANGE = 20.0
MIN_RGB_NONZERO_FRACTION = 0.05
MIN_VALID_DEPTH_FRACTION = 0.10
MIN_DEPTH_DYNAMIC_RANGE_M = 0.05
MIN_VALID_DEPTH_M = 0.05
MAX_VALID_DEPTH_M = 2.0
MAX_SENSOR_SPREAD_SEC = 0.020001
REQUIRED_SOURCE_ARTIFACTS = {
    "RoboRefer/API/api.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_capture.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_runner.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_validate.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_evaluator.py",
    "ur3/ur3_perception/scripts/roborefer_grounder.py",
    "ur3/ur3_perception/scripts/spatial_point_utils.py",
    "ur3/ur3_perception/scripts/move_camera_to_view.py",
    "ur3/ur3_perception/launch/roborefer_pilot_capture.launch.py",
    "ur3/ur3_perception/config/roborefer_pilot_v0_scenes.yaml",
    "ur3/ur3_perception/config/roborefer_pilot_v0_gate.yaml",
    "ur3/ur3_perception/config/roborefer_pilot_v0_annotations.yaml",
    "ur3/ur3_perception/schemas/roborefer_pilot_prediction.schema.json",
    "ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro",
    "ur3/ur3_moveit_control/launch/ur3_susgrip_sim.launch.py",
    "ur3/ur_simulation_gz/worlds/ur3_pick_place.sdf",
    "ur3/ur_simulation_gz/launch/ur_sim_control.launch.py",
}
POINT_PATTERN = re.compile(
    r"\(\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*\)"
)
STRICT_POINT_ANSWER = re.compile(
    r"^\s*\[\s*\(\s*-?(?:\d+(?:\.\d*)?|\.\d+)\s*,\s*"
    r"-?(?:\d+(?:\.\d*)?|\.\d+)\s*\)\s*\]\s*$"
)


def _replay_parse(answer: str, width: int, height: int) -> Tuple[str, list, list]:
    """Independently replay the runner's deterministic point parser."""

    points = []
    for match in POINT_PATTERN.finditer(str(answer)):
        x_value, y_value = float(match.group(1)), float(match.group(2))
        if np.isfinite(x_value) and np.isfinite(y_value):
            points.append((x_value, y_value))
    if not points:
        return "NO_POINT", [], []

    def to_pixels(values):
        pixels = []
        for x_value, y_value in values:
            if 0.0 <= x_value <= 1.0 and 0.0 <= y_value <= 1.0:
                pixels.append((
                    min(width - 1, int(round(x_value * (width - 1)))),
                    min(height - 1, int(round(y_value * (height - 1)))),
                ))
        return pixels

    if len(points) != 1:
        pixels = to_pixels(points)
        return "MULTIPLE_POINTS", [list(point) for point in points], [
            list(point) for point in pixels
        ]
    if any(value < 0.0 or value > 1.0 for value in points[0]):
        return "OUT_OF_RANGE", [list(points[0])], []
    pixels = to_pixels(points)
    if len(pixels) != 1:
        return "OUT_OF_RANGE", [list(points[0])], []
    status = (
        "EXACT_ONE_NORMALIZED_POINT"
        if STRICT_POINT_ANSWER.fullmatch(str(answer))
        else "FORMAT_VIOLATION"
    )
    return status, [list(points[0])], [list(pixels[0])]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _encode_image(image: np.ndarray, extension: str, params=None) -> bytes:
    success, encoded = cv2.imencode(extension, image, params or [])
    if not success:
        raise ValidationError(f"cannot deterministically encode request as {extension}")
    return encoded.tobytes()


def _normalize_depth_request(depth_m: np.ndarray) -> np.ndarray:
    """Mirror the locked runner's inverse-depth-percentile preprocessing."""

    values = np.asarray(depth_m, dtype=np.float32)
    _expect(values.ndim == 2, f"depth input must be HxW, got {values.shape}")
    valid = (
        np.isfinite(values)
        & (values >= MIN_VALID_DEPTH_M)
        & (values <= MAX_VALID_DEPTH_M)
    )
    _expect(int(np.count_nonzero(valid)) >= 16,
            "registered depth image contains too few valid pixels")
    valid_depth = values[valid]
    near = float(np.percentile(valid_depth, 2.0))
    far = float(np.percentile(valid_depth, 98.0))
    if far - near < 1e-4:
        near = float(valid_depth.min())
        far = float(valid_depth.max())
    _expect(far - near >= 1e-6, "registered depth has no usable range")
    clipped = np.clip(values, near, far)
    inverse = 1.0 / np.maximum(clipped, 1e-6)
    normalized = (
        (inverse - 1.0 / far) / max(1.0 / near - 1.0 / far, 1e-6)
    )
    gray = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    gray[~valid] = 0
    return np.repeat(gray[..., np.newaxis], 3, axis=-1)


def _gate_args_from_config(gate: Mapping[str, Any]) -> dict:
    return {
        "min_depth_m": float(gate["min_depth_m"]),
        "max_depth_m": float(gate["max_depth_m"]),
        "seed_radius_px": int(gate["depth_seed_radius_px"]),
        "roi_radius_px": int(gate["depth_roi_radius_px"]),
        "near_tolerance_m": float(gate["depth_near_tolerance_m"]),
        "far_tolerance_m": float(gate["depth_far_tolerance_m"]),
        "min_area_px": int(gate["min_mask_area_px"]),
        "max_area_fraction": float(gate["max_mask_area_fraction"]),
        "bbox_padding_px": int(gate["bbox_padding_px"]),
    }


class ValidationError(RuntimeError):
    """Raised when a pilot invariant or an artifact hash is invalid."""


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


def sha256_canonical_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValidationError(f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        raise ValidationError(f"missing JSONL file: {path}")
    records: List[dict] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValidationError(
                        f"{path}:{line_number} is not a JSON object"
                    )
                records.append(value)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSONL {path}: {exc}") from exc
    return records


def _normalized_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def find_forbidden_input_paths(value: Any, prefix: str = "$") -> List[str]:
    """Return recursive paths that could leak evaluator-only information.

    Keys are checked broadly, while string values are checked only for explicit
    evaluator/annotation file markers.  Consequently ordinary prompts such as
    "point to the target fruit" are not rejected.
    """

    findings: List[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            child_path = f"{prefix}.{raw_key}"
            key = _normalized_key(raw_key)
            if (
                key in FORBIDDEN_INPUT_KEY_TOKENS
                or key.startswith("targetmodel")
                or key.startswith("targetsemanticlabel")
                or "groundtruth" in key
                or "semanticlabel" in key
                or "oracle" in key
            ):
                findings.append(child_path)
            findings.extend(find_forbidden_input_paths(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            findings.extend(find_forbidden_input_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, str):
        lowered = value.replace("\\", "/").lower()
        if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
            findings.append(prefix)
    return findings


# Backward-friendly public alias matching the runner's helper name.
find_forbidden_keys = find_forbidden_input_paths


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _resolve_confined(root: Path, relative_value: Any, description: str) -> Path:
    _expect(isinstance(relative_value, str) and bool(relative_value),
            f"{description} must be a non-empty relative path")
    relative_path = Path(relative_value)
    _expect(not relative_path.is_absolute(), f"{description} must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative_path).resolve()
    _expect(
        resolved != resolved_root and resolved_root in resolved.parents,
        f"{description} escapes its artifact root: {relative_value}",
    )
    return resolved


def _expected_prompt(scene: Mapping[str, Any]) -> str:
    return str(scene.get("instruction", "")).strip() + " " + str(
        scene.get("coordinate_suffix", "")
    ).strip()


def _request_hash(record: Mapping[str, Any], kind: str) -> Any:
    """Accept the runner's flat fields and its explicit mapping aliases."""

    flat = record.get(f"request_{kind}_sha256")
    mapped = record.get("request_sha256")
    if isinstance(mapped, Mapping):
        mapped_value = mapped.get(kind)
        if flat is not None and mapped_value != flat:
            raise ValidationError(
                f"{record.get('scene_id')}/{record.get('mode')} has disagreeing "
                f"flat/mapped {kind} request hashes"
            )
        return mapped_value
    return flat


def validate_capture(dataset_root: Path) -> dict:
    """Validate the ten inference inputs without opening evaluator artifacts."""

    root = Path(dataset_root).expanduser().resolve()
    _expect(root.is_dir(), f"dataset root is not a directory: {root}")
    capture_path = root / "capture_manifest.json"
    input_manifest_path = root / "input_manifest.jsonl"
    capture = _read_json(capture_path)

    _expect(capture.get("schema_version") == SCHEMA_VERSION,
            "capture schema_version mismatch")
    _expect(capture.get("protocol_id") == PROTOCOL_ID,
            "capture protocol_id mismatch")
    _expect(capture.get("status") == "COMPLETE", "capture manifest is not COMPLETE")
    _expect(capture.get("scene_count") == EXPECTED_SCENE_COUNT,
            "capture manifest scene_count must be exactly 10")
    _expect(capture.get("annotation_semantics_consumed_by_capture") is False,
            "capture does not attest that annotation semantics stayed closed")
    _expect(capture.get("annotation_bytes_hashed_before_inference") is True,
            "capture did not attest hash-only annotation preregistration")
    _expect(
        isinstance(capture.get("annotation_file_sha256"), str)
        and len(capture.get("annotation_file_sha256")) == 64,
        "capture did not preregister the evaluator annotation hash",
    )
    _expect(capture.get("semantic_labels_are_evaluator_only") is True,
            "capture does not mark semantic labels evaluator-only")
    _expect(capture.get("target_handoff_published") is False,
            "capture published a target handoff")
    _expect(capture.get("robot_manipulation_performed") is False,
            "capture performed robot manipulation")
    _expect(capture.get("source_artifacts_verified_before_capture") is True,
            "capture did not verify preregistered source artifacts")
    pretrial_path_value = capture.get("pretrial_source_lock_file")
    _expect(isinstance(pretrial_path_value, str) and bool(pretrial_path_value),
            "capture does not reference a pretrial source lock")
    pretrial_path = Path(pretrial_path_value).expanduser().resolve()
    _expect(pretrial_path.is_file(), f"pretrial source lock is missing: {pretrial_path}")
    _expect(capture.get("pretrial_source_lock_sha256") == sha256_file(pretrial_path),
            "pretrial source lock changed after capture")
    pretrial = _read_json(pretrial_path)
    _expect(
        pretrial.get("protocol_id") == PROTOCOL_ID
        and pretrial.get("status") == "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "pretrial source lock status/protocol is invalid",
    )
    workspace_root = Path(str(pretrial.get("workspace_root", ""))).resolve()
    source_hashes = pretrial.get("source_artifact_sha256")
    _expect(workspace_root.is_dir() and isinstance(source_hashes, Mapping),
            "pretrial source lock lacks workspace/artifact hashes")
    _expect(
        REQUIRED_SOURCE_ARTIFACTS.issubset(set(source_hashes)),
        "pretrial source lock misses required production artifacts",
    )
    for relative_name, expected_hash in source_hashes.items():
        artifact = (workspace_root / str(relative_name)).resolve()
        _expect(
            workspace_root in artifact.parents and artifact.is_file(),
            f"pretrial source artifact is missing/unsafe: {relative_name}",
        )
        _expect(
            SHA256_PATTERN.fullmatch(str(expected_hash)) is not None
            and sha256_file(artifact) == expected_hash,
            f"pretrial source artifact changed: {relative_name}",
        )
    _expect(
        capture.get("model_inventory_sha256_preregistered")
        == pretrial.get("model_inventory_sha256")
        and SHA256_PATTERN.fullmatch(
            str(capture.get("model_inventory_sha256_preregistered", ""))
        ) is not None,
        "capture model fingerprint differs from pretrial source lock",
    )
    _expect(capture.get("gate_config_bytes_hashed_before_capture") is True,
            "capture does not attest gate bytes were hashed before capture")
    _expect(
        isinstance(capture.get("gate_config_file_sha256"), str)
        and SHA256_PATTERN.fullmatch(capture["gate_config_file_sha256"]) is not None,
        "capture gate_config_file_sha256 must be a 64-character lowercase SHA256",
    )
    _expect(capture.get("input_manifest_sha256") == sha256_file(input_manifest_path),
            "input_manifest.jsonl SHA256 differs from capture lock")
    manifest_sensor_qc = capture.get("input_only_sensor_qc")
    _expect(
        isinstance(manifest_sensor_qc, Mapping)
        and manifest_sensor_qc.get("passed") is True
        and manifest_sensor_qc.get("all_scene_rgb_sha256_unique") is True
        and manifest_sensor_qc.get("rgb_scene_count") == EXPECTED_SCENE_COUNT,
        "capture manifest input-only sensor QC did not pass for 10 unique RGB frames",
    )
    evaluator_hashes = capture.get("evaluator_artifact_sha256")
    _expect(
        isinstance(evaluator_hashes, Mapping)
        and len(evaluator_hashes) == EXPECTED_SCENE_COUNT * 2,
        "capture must lock semantic-label and capture-oracle hashes for 10 scenes",
    )

    scenes = _read_jsonl(input_manifest_path)
    _expect(len(scenes) == EXPECTED_SCENE_COUNT,
            "input manifest must contain exactly 10 records")
    scene_ids = [str(scene.get("scene_id", "")) for scene in scenes]
    _expect(all(scene_ids), "input manifest contains an empty scene_id")
    _expect(len(set(scene_ids)) == EXPECTED_SCENE_COUNT,
            "input manifest scene_id values are not unique")
    rgb_hashes = [scene.get("input_sha256", {}).get("rgb") for scene in scenes]
    _expect(
        all(isinstance(value, str) for value in rgb_hashes)
        and len(set(rgb_hashes)) == EXPECTED_SCENE_COUNT,
        "input manifest does not contain 10 byte-distinct RGB captures",
    )

    leaked = find_forbidden_input_paths(scenes)
    _expect(not leaked, f"forbidden evaluator data in input manifest: {leaked}")

    required_files = {"rgb", "depth_m", "camera_info", "tf_snapshot"}
    verified_file_count = 0
    rgb_hashes = []
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        _expect(scene.get("schema_version") == SCHEMA_VERSION,
                f"{scene_id}: schema_version mismatch")
        _expect(scene.get("protocol_id") == PROTOCOL_ID,
                f"{scene_id}: protocol_id mismatch")
        _expect(scene.get("capture", {}).get("robot_manipulation_performed") is False,
                f"{scene_id}: capture is not marked no-manipulation")
        capture_qc = scene.get("capture", {})
        _expect(capture_qc.get("registered_metric_depth") is True,
                f"{scene_id}: depth is not marked registered metric depth")
        _expect(
            capture_qc.get("input_only_sensor_qc", {}).get("passed") is True,
            f"{scene_id}: input-only sensor QC did not pass",
        )
        spread = capture_qc.get("rgb_depth_label_spread_sec")
        _expect(
            isinstance(spread, (int, float))
            and 0.0 <= float(spread) <= MAX_SENSOR_SPREAD_SEC,
            f"{scene_id}: RGB/depth/label spread exceeds 0.020001 seconds",
        )
        files = scene.get("input_files")
        hashes = scene.get("input_sha256")
        _expect(isinstance(files, Mapping), f"{scene_id}: input_files missing")
        _expect(isinstance(hashes, Mapping), f"{scene_id}: input_sha256 missing")
        _expect(set(files) == required_files,
                f"{scene_id}: input_files must be exactly {sorted(required_files)}")
        _expect(set(hashes) == required_files,
                f"{scene_id}: input_sha256 must be exactly {sorted(required_files)}")
        for key in sorted(required_files):
            path = _resolve_confined(root, files[key], f"{scene_id}.{key}")
            _expect(path.is_file(), f"{scene_id}: missing input file {files[key]}")
            _expect(hashes[key] == sha256_file(path),
                    f"{scene_id}: SHA256 mismatch for {files[key]}")
            leaked_path = find_forbidden_input_paths(str(files[key]), f"$.{scene_id}.{key}")
            _expect(not leaked_path,
                    f"{scene_id}: evaluator path leaked into input_files: {leaked_path}")
            verified_file_count += 1

        rgb_path = _resolve_confined(root, files["rgb"], f"{scene_id}.rgb")
        depth_path = _resolve_confined(root, files["depth_m"], f"{scene_id}.depth_m")
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        _expect(rgb is not None and rgb.ndim == 3 and rgb.shape[2] == 3,
                f"{scene_id}: RGB cannot be decoded as BGR8")
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        rgb_std = float(np.std(gray))
        rgb_dynamic_range = float(
            np.percentile(gray, 99.0) - np.percentile(gray, 1.0)
        )
        rgb_nonzero_fraction = float(np.count_nonzero(gray)) / float(gray.size)
        _expect(
            rgb_std >= MIN_RGB_STD
            and rgb_dynamic_range >= MIN_RGB_DYNAMIC_RANGE
            and rgb_nonzero_fraction >= MIN_RGB_NONZERO_FRACTION,
            f"{scene_id}: RGB input-only QC failed "
            f"(std={rgb_std:.3f}, range={rgb_dynamic_range:.3f}, "
            f"nonzero={rgb_nonzero_fraction:.6f})",
        )
        try:
            depth_m = np.load(depth_path, allow_pickle=False)
        except Exception as exc:
            raise ValidationError(f"{scene_id}: cannot load metric depth: {exc}") from exc
        _expect(depth_m.ndim == 2 and depth_m.shape == rgb.shape[:2],
                f"{scene_id}: registered RGB/depth dimensions mismatch")
        valid_depth = (
            np.isfinite(depth_m)
            & (depth_m >= MIN_VALID_DEPTH_M)
            & (depth_m <= MAX_VALID_DEPTH_M)
        )
        valid_fraction = float(np.count_nonzero(valid_depth)) / float(depth_m.size)
        values = depth_m[valid_depth]
        depth_range = (
            float(np.percentile(values, 98.0) - np.percentile(values, 2.0))
            if values.size
            else 0.0
        )
        _expect(
            valid_fraction >= MIN_VALID_DEPTH_FRACTION
            and depth_range >= MIN_DEPTH_DYNAMIC_RANGE_M,
            f"{scene_id}: depth input-only QC failed "
            f"(valid={valid_fraction:.6f}, range={depth_range:.6f}m)",
        )
        rgb_hashes.append(hashes["rgb"])

        # Capture writes a human-readable copy beside the actual inputs.  It is
        # an inference payload too, so validate both its contents and equality
        # with the locked JSONL record.  Never traverse ../evaluator here.
        scene_input_path = root / scene_id / "input" / "scene_input.json"
        _expect(scene_input_path.is_file(), f"{scene_id}: scene_input.json missing")
        scene_input = _read_json(scene_input_path)
        _expect(scene_input == scene,
                f"{scene_id}: scene_input.json differs from input_manifest.jsonl")
        scene_leaks = find_forbidden_input_paths(scene_input)
        _expect(not scene_leaks,
                f"{scene_id}: forbidden evaluator data in scene_input.json: {scene_leaks}")

    _expect(len(set(rgb_hashes)) == EXPECTED_SCENE_COUNT,
            "capture contains byte-identical RGB frames across pilot scenes")

    return {
        "valid": True,
        "stage": "capture",
        "protocol_id": PROTOCOL_ID,
        "dataset_root": str(root),
        "scene_count": len(scenes),
        "verified_input_file_count": verified_file_count,
        "capture_manifest_sha256": sha256_file(capture_path),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "gate_config_file_sha256": capture["gate_config_file_sha256"],
        "pretrial_source_lock_sha256": capture["pretrial_source_lock_sha256"],
        "model_inventory_sha256_preregistered": capture[
            "model_inventory_sha256_preregistered"
        ],
        "scene_ids": scene_ids,
        "records": scenes,
        "oracle_files_opened": False,
    }


def _verify_record_hash(record: Mapping[str, Any]) -> None:
    expected = record.get("prediction_sha256")
    payload = dict(record)
    payload.pop("prediction_sha256", None)
    _expect(isinstance(expected, str) and len(expected) == 64,
            f"{record.get('scene_id')}/{record.get('mode')}: prediction hash missing")
    _expect(expected == sha256_canonical_json(payload),
            f"{record.get('scene_id')}/{record.get('mode')}: prediction hash mismatch")


def verify_prediction_lock(prediction_root: Path) -> dict:
    """Verify COMPLETE manifest, prediction bytes, and the immutable lock."""

    root = Path(prediction_root).expanduser().resolve()
    _expect(root.is_dir(), f"prediction root is not a directory: {root}")
    manifest_path = root / "prediction_manifest.json"
    lock_path = root / "prediction_lock.json"
    manifest = _read_json(manifest_path)
    lock = _read_json(lock_path)
    _expect(manifest.get("schema_version") == SCHEMA_VERSION,
            "prediction manifest schema_version mismatch")
    _expect(manifest.get("protocol_id") == PROTOCOL_ID,
            "prediction manifest protocol mismatch")
    _expect(manifest.get("status") == "COMPLETE",
            "prediction manifest is not COMPLETE")
    _expect(lock.get("schema_version") == SCHEMA_VERSION,
            "prediction lock schema_version mismatch")
    _expect(lock.get("protocol_id") == PROTOCOL_ID,
            "prediction lock protocol mismatch")
    _expect(lock.get("status") == "LOCKED", "prediction lock is not LOCKED")
    _expect(lock.get("oracle_opened_before_lock") is False,
            "prediction lock reports oracle exposure before lock")
    _expect(lock.get("prediction_manifest_sha256") == sha256_file(manifest_path),
            "prediction manifest changed after prediction lock")
    predictions_rel = manifest.get("predictions_file", "predictions.jsonl")
    predictions_path = _resolve_confined(root, predictions_rel, "predictions_file")
    _expect(predictions_path.is_file(), f"predictions file missing: {predictions_path}")
    predictions_hash = sha256_file(predictions_path)
    _expect(manifest.get("predictions_sha256") == predictions_hash,
            "prediction JSONL differs from COMPLETE manifest")
    _expect(lock.get("predictions_sha256") == predictions_hash,
            "prediction JSONL changed after prediction lock")
    records = _read_jsonl(predictions_path)
    _expect(lock.get("record_count") == len(records),
            "prediction lock record_count mismatch")
    _expect(manifest.get("record_count") == len(records),
            "prediction manifest record_count mismatch")
    return {
        "root": root,
        "manifest": manifest,
        "lock": lock,
        "manifest_path": manifest_path,
        "lock_path": lock_path,
        "predictions_path": predictions_path,
        "records": records,
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "predictions_sha256": predictions_hash,
    }


def validate_predictions(
    dataset_root: Path,
    prediction_root: Path,
    gate_config_path: Optional[Path] = None,
) -> dict:
    """Validate all B0/B1/B2 records and their no-manipulation invariants."""

    capture_report = validate_capture(dataset_root)
    dataset = Path(dataset_root).expanduser().resolve()
    locked = verify_prediction_lock(prediction_root)
    root: Path = locked["root"]
    manifest = locked["manifest"]
    records: List[dict] = locked["records"]
    if gate_config_path is None:
        gate_config_path = Path(str(manifest.get("gate_config_path", "")))
    gate_path = Path(gate_config_path).expanduser().resolve()
    _expect(gate_path.is_file(), f"gate config file is missing: {gate_path}")
    _expect(sha256_file(gate_path) == manifest.get("gate_config_sha256"),
            "gate config bytes differ from prediction manifest")
    try:
        import yaml
        gate_config = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"cannot read frozen gate config: {exc}") from exc
    _expect(isinstance(gate_config, Mapping), "frozen gate config is not a mapping")
    gate_args = _gate_args_from_config(gate_config)

    _expect(manifest.get("scene_count") == EXPECTED_SCENE_COUNT,
            "prediction scene_count must be exactly 10")
    _expect(manifest.get("record_count") == EXPECTED_SCENE_COUNT * len(EXPECTED_MODES),
            "prediction record_count must be exactly 30")
    _expect(len(records) == EXPECTED_SCENE_COUNT * len(EXPECTED_MODES),
            "predictions.jsonl must contain exactly 30 records")
    _expect(manifest.get("capture_manifest_sha256") ==
            sha256_file(dataset / "capture_manifest.json"),
            "prediction manifest references a different capture manifest")
    _expect(manifest.get("input_manifest_sha256") ==
            sha256_file(dataset / "input_manifest.jsonl"),
            "prediction manifest references a different input manifest")
    _expect(
        manifest.get("gate_config_sha256") ==
        capture_report.get("gate_config_file_sha256"),
        "prediction gate hash differs from gate frozen before capture",
    )
    _expect(manifest.get("same_rgb_and_prompt_for_B0_B1") is True,
            "prediction manifest does not attest identical B0/B1 RGB and prompt")
    expected_preprocessing = {
        "rgb": {
            "source": "locked_capture_rgb_png",
            "encoding": "jpeg",
            "quality": 90,
        },
        "depth": {
            "source": "locked_registered_metric_depth_npy",
            "transform": "inverse_depth_percentile_v1",
            "valid_min_m": MIN_VALID_DEPTH_M,
            "valid_max_m": MAX_VALID_DEPTH_M,
            "near_percentile": 2.0,
            "far_percentile": 98.0,
            "encoding": "png",
            "channels": 3,
        },
    }
    _expect(manifest.get("request_preprocessing") == expected_preprocessing,
            "prediction request preprocessing contract is missing or changed")
    _expect(manifest.get("b2_requeries_model") is False,
            "prediction manifest permits a B2 model re-query")
    _expect(
        manifest.get("generation_mode") == "greedy"
        and manifest.get("sampling_enabled") is False,
        "prediction manifest does not prove deterministic greedy decoding",
    )
    _expect(manifest.get("oracle_or_annotation_read_by_runner") is False,
            "runner reports reading evaluator/oracle information")
    _expect(manifest.get("target_handoff_published") is False,
            "prediction run published target handoff")
    _expect(manifest.get("robot_manipulation_performed") is False,
            "prediction run performed robot manipulation")
    leaked_predictions = find_forbidden_input_paths(records)
    _expect(
        not leaked_predictions,
        f"forbidden evaluator data in inference/prediction payloads: "
        f"{leaked_predictions}",
    )

    scenes = {scene["scene_id"]: scene for scene in capture_report["records"]}
    derived_request_hashes: Dict[str, Dict[str, str]] = {}
    for scene_id, scene in scenes.items():
        capture_rgb_path = _resolve_confined(
            dataset, scene["input_files"]["rgb"], f"{scene_id}.capture_rgb"
        )
        capture_depth_path = _resolve_confined(
            dataset, scene["input_files"]["depth_m"], f"{scene_id}.capture_depth"
        )
        capture_rgb = cv2.imread(str(capture_rgb_path), cv2.IMREAD_COLOR)
        _expect(capture_rgb is not None, f"{scene_id}: cannot decode capture RGB")
        capture_depth = np.load(capture_depth_path, allow_pickle=False)
        derived_request_hashes[scene_id] = {
            "rgb": _sha256_bytes(_encode_image(
                capture_rgb, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90]
            )),
            "depth": _sha256_bytes(_encode_image(
                _normalize_depth_request(capture_depth), ".png"
            )),
        }
    grouped: Dict[str, Dict[str, dict]] = {scene_id: {} for scene_id in scenes}
    for record in records:
        scene_id = str(record.get("scene_id", ""))
        mode = str(record.get("mode", ""))
        _expect(scene_id in scenes, f"prediction references unknown scene: {scene_id}")
        _expect(mode in EXPECTED_MODES, f"{scene_id}: unknown mode {mode}")
        _expect(mode not in grouped[scene_id], f"{scene_id}: duplicate mode {mode}")
        grouped[scene_id][mode] = record
        _expect(record.get("schema_version") == SCHEMA_VERSION,
                f"{scene_id}/{mode}: schema_version mismatch")
        _expect(record.get("protocol_id") == PROTOCOL_ID,
                f"{scene_id}/{mode}: protocol mismatch")
        _expect(record.get("target_handoff_published") is False,
                f"{scene_id}/{mode}: target handoff was published")
        _expect(record.get("robot_manipulation_performed") is False,
                f"{scene_id}/{mode}: robot manipulation was performed")
        _expect(record.get("shadow_evaluation_only") is True,
                f"{scene_id}/{mode}: record is not shadow-only")
        _expect(isinstance(record.get("would_execute"), bool),
                f"{scene_id}/{mode}: would_execute must be Boolean")
        _expect(record.get("prompt") == _expected_prompt(scenes[scene_id]),
                f"{scene_id}/{mode}: prompt differs from locked input")
        _expect(record.get("instruction") == scenes[scene_id].get("instruction"),
                f"{scene_id}/{mode}: instruction differs from locked input")
        source_hash = record.get("source_input_sha256")
        _expect(source_hash == scenes[scene_id].get("input_sha256"),
                f"{scene_id}/{mode}: source input hash mapping missing or mismatched")
        _verify_record_hash(record)

        record_path = _resolve_confined(root, record.get("record_file"),
                                        f"{scene_id}/{mode}.record_file")
        _expect(record_path.is_file(), f"{scene_id}/{mode}: record file missing")
        _expect(_read_json(record_path) == record,
                f"{scene_id}/{mode}: per-scene record differs from JSONL")

        rgb_path = _resolve_confined(root, record.get("request_rgb_file"),
                                     f"{scene_id}/{mode}.request_rgb_file")
        _expect(rgb_path.is_file(), f"{scene_id}/{mode}: request RGB missing")
        _expect(_request_hash(record, "rgb") == sha256_file(rgb_path),
                f"{scene_id}/{mode}: request RGB hash mismatch")
        _expect(
            _request_hash(record, "rgb") == derived_request_hashes[scene_id]["rgb"],
            f"{scene_id}/{mode}: request RGB is not the locked q90 derivation "
            "of capture RGB",
        )
        depth_file = record.get("request_depth_file")
        if depth_file is None:
            _expect(_request_hash(record, "depth") is None,
                    f"{scene_id}/{mode}: null depth file has non-null hash")
        else:
            depth_path = _resolve_confined(root, depth_file,
                                           f"{scene_id}/{mode}.request_depth_file")
            _expect(depth_path.is_file(), f"{scene_id}/{mode}: request depth missing")
            _expect(_request_hash(record, "depth") == sha256_file(depth_path),
                    f"{scene_id}/{mode}: request depth hash mismatch")
            _expect(
                _request_hash(record, "depth")
                == derived_request_hashes[scene_id]["depth"],
                f"{scene_id}/{mode}: request depth is not the locked inverse-depth "
                "derivation of registered capture depth",
            )

    for scene_id, mode_records in grouped.items():
        _expect(set(mode_records) == set(EXPECTED_MODES),
                f"{scene_id}: expected exactly B0/B1/B2")
        b0, b1, b2 = (mode_records[mode] for mode in EXPECTED_MODES)
        _expect(b0.get("policy") == "no_uncertainty_gate_shadow",
                f"{scene_id}/B0: incorrect policy")
        _expect(b1.get("policy") == "no_uncertainty_gate_shadow",
                f"{scene_id}/B1: incorrect policy")
        _expect(b2.get("policy") == "hard_gate_shadow",
                f"{scene_id}/B2: incorrect policy")
        _expect(b0.get("model_query_performed") is True,
                f"{scene_id}/B0: model query not recorded")
        _expect(b1.get("model_query_performed") is True,
                f"{scene_id}/B1: model query not recorded")
        for queried in (b0, b1):
            expected_model_fingerprint = manifest.get("model", {}).get(
                "inventory_sha256"
            )
            _expect(
                queried.get("query_status") == "SUCCESS"
                and queried.get("response", {}).get("generation_mode") == "greedy"
                and queried.get("response", {}).get("generation_config")
                == {
                    "do_sample": False,
                    "temperature": None,
                    "top_p": None,
                    "top_k": None,
                },
                f"{scene_id}/{queried.get('mode')}: query failed or decoding was not greedy",
            )
            _expect(
                isinstance(expected_model_fingerprint, str)
                and queried.get("response", {}).get("model_fingerprint", {}).get(
                    "inventory_sha256"
                ) == expected_model_fingerprint,
                f"{scene_id}/{queried.get('mode')}: response model fingerprint "
                "differs from prediction manifest",
            )
            response = queried.get("response", {})
            response_body_b64 = response.get("response_body_base64")
            _expect(
                isinstance(response_body_b64, str) and bool(response_body_b64),
                f"{scene_id}/{queried.get('mode')}: raw HTTP response bytes missing",
            )
            try:
                response_body = base64.b64decode(response_body_b64, validate=True)
                response_json = json.loads(response_body.decode("utf-8"))
            except Exception as exc:
                raise ValidationError(
                    f"{scene_id}/{queried.get('mode')}: invalid stored HTTP response: {exc}"
                ) from exc
            _expect(
                response.get("response_sha256") == _sha256_bytes(response_body),
                f"{scene_id}/{queried.get('mode')}: HTTP response hash mismatch",
            )
            _expect(
                response.get("response_json") == response_json,
                f"{scene_id}/{queried.get('mode')}: parsed/stored HTTP response differs",
            )
            _expect(
                str(response_json.get("answer", "")).strip()
                == str(queried.get("raw_answer", "")),
                f"{scene_id}/{queried.get('mode')}: raw answer differs from HTTP response",
            )
            _expect(
                response_json.get("generation_mode") == response.get("generation_mode")
                and response_json.get("generation_config")
                == response.get("generation_config")
                and response_json.get("model_fingerprint")
                == response.get("model_fingerprint"),
                f"{scene_id}/{queried.get('mode')}: HTTP metadata differs from record",
            )
            replay_status, replay_normalized, replay_pixels = _replay_parse(
                queried.get("raw_answer", ""),
                int(queried.get("image_width", 0)),
                int(queried.get("image_height", 0)),
            )
            _expect(
                queried.get("parse_status") == replay_status
                and queried.get("normalized_points_xy") == replay_normalized
                and queried.get("pixel_points_xy") == replay_pixels,
                f"{scene_id}/{queried.get('mode')}: point fields differ from "
                "independently replayed raw-answer parser",
            )
            _expect(
                queried.get("would_execute")
                == (replay_status == "EXACT_ONE_NORMALIZED_POINT"),
                f"{scene_id}/{queried.get('mode')}: would_execute differs from "
                "strict parse contract",
            )
        _expect(b0.get("request_enable_depth") is False,
                f"{scene_id}/B0: RGB-only mode enabled depth")
        _expect(b1.get("request_enable_depth") is True,
                f"{scene_id}/B1: RGB-D mode did not enable depth")
        _expect(
            isinstance(b1.get("request_depth_file"), str)
            and bool(b1.get("request_depth_file"))
            and isinstance(_request_hash(b1, "depth"), str)
            and len(_request_hash(b1, "depth")) == 64,
            f"{scene_id}/B1: RGB-D request is missing depth bytes/hash",
        )
        _expect(b0.get("request_depth_file") is None and
                _request_hash(b0, "depth") is None,
                f"{scene_id}/B0: RGB-only request contains depth")
        _expect(b0.get("prompt") == b1.get("prompt"),
                f"{scene_id}: B0/B1 prompts are not identical")
        _expect(_request_hash(b0, "rgb") == _request_hash(b1, "rgb"),
                f"{scene_id}: B0/B1 RGB bytes are not identical")
        _expect(b0.get("request_rgb_file") == b1.get("request_rgb_file"),
                f"{scene_id}: B0/B1 do not reference the same RGB file")

        _expect(b2.get("model_query_performed") is False,
                f"{scene_id}/B2: illegal model query")
        _expect(
            b2.get("registered_depth_used_by_gate") is True
            and isinstance(b2.get("request_depth_file"), str)
            and bool(b2.get("request_depth_file"))
            and isinstance(_request_hash(b2, "depth"), str)
            and len(_request_hash(b2, "depth")) == 64,
            f"{scene_id}/B2: gate is missing registered depth provenance",
        )
        _expect(b2.get("query_status") == "NOT_QUERIED_DERIVED_FROM_B1",
                f"{scene_id}/B2: query status does not prove no re-query")
        _expect(b2.get("source_mode") == "B1", f"{scene_id}/B2: source is not B1")
        _expect(b2.get("source_b1_prediction_sha256") == b1.get("prediction_sha256"),
                f"{scene_id}/B2: source B1 hash mismatch")
        _expect("response" not in b2 and "query_attempt" not in b2,
                f"{scene_id}/B2: HTTP-query metadata is present")
        for field in (
            "prompt",
            "instruction",
            "raw_answer",
            "parse_status",
            "normalized_points_xy",
            "pixel_points_xy",
            "request_rgb_file",
            "request_depth_file",
        ):
            _expect(b2.get(field) == b1.get(field),
                    f"{scene_id}/B2: {field} differs from locked B1")
        _expect(_request_hash(b2, "rgb") == _request_hash(b1, "rgb") and
                _request_hash(b2, "depth") == _request_hash(b1, "depth"),
                f"{scene_id}/B2: request hashes differ from locked B1")
        gate = b2.get("gate")
        _expect(isinstance(gate, Mapping) and isinstance(gate.get("accepted"), bool),
                f"{scene_id}/B2: gate decision missing")
        _expect(b2.get("would_execute") == gate.get("accepted"),
                f"{scene_id}/B2: would_execute differs from gate decision")
        mask_file = gate.get("mask_file")
        if mask_file is None:
            _expect(gate.get("mask_sha256") is None,
                    f"{scene_id}/B2: null mask file has a mask hash")
        else:
            mask_path = _resolve_confined(root, mask_file,
                                          f"{scene_id}/B2.gate.mask_file")
            _expect(mask_path.is_file(), f"{scene_id}/B2: gate mask missing")
            _expect(gate.get("mask_sha256") == sha256_file(mask_path),
                    f"{scene_id}/B2: gate mask hash mismatch")

        # Replay B2 from only the locked B1 point, registered metric depth and
        # frozen gate bytes.  This prevents a resealed artifact from inventing
        # an accept/reject decision or mask after seeing evaluator labels.
        capture_depth_path = _resolve_confined(
            dataset,
            scenes[scene_id]["input_files"]["depth_m"],
            f"{scene_id}.capture_depth_for_gate_replay",
        )
        replay_depth = np.load(capture_depth_path, allow_pickle=False)
        if b1.get("parse_status") != "EXACT_ONE_NORMALIZED_POINT":
            _expect(
                gate.get("accepted") is False
                and gate.get("reason") == f"B1_{b1.get('parse_status')}"
                and gate.get("mask_file") is None,
                f"{scene_id}/B2: non-actionable B1 gate outcome differs from replay",
            )
        else:
            try:
                replay = segment_seeded_depth_component(
                    replay_depth,
                    [tuple(point) for point in b1["pixel_points_xy"]],
                    **gate_args,
                )
            except ValueError as exc:
                expected_prefix = f"DEPTH_COMPONENT_REJECTED: {type(exc).__name__}: {exc}"
                _expect(
                    gate.get("accepted") is False
                    and gate.get("reason") == expected_prefix
                    and gate.get("mask_file") is None,
                    f"{scene_id}/B2: rejected gate result differs from replay",
                )
            else:
                replay_supported = [
                    list(point) for point in replay["supported_points_xy"]
                ]
                accepted = len(replay_supported) == 1
                _expect(gate.get("accepted") is accepted,
                        f"{scene_id}/B2: gate acceptance differs from replay")
                _expect(
                    gate.get("reason") == (
                        "DEPTH_COMPONENT_ACCEPTED"
                        if accepted
                        else "DEPTH_COMPONENT_REJECTED: seed pixel is outside mask"
                    ),
                    f"{scene_id}/B2: gate reason differs from replay",
                )
                _expect(gate.get("bbox_xyxy") == replay["bbox_xyxy"],
                        f"{scene_id}/B2: gate bbox differs from replay")
                _expect(gate.get("grasp_pixel_xy") == list(replay["grasp_pixel_xy"]),
                        f"{scene_id}/B2: gate grasp point differs from replay")
                _expect(gate.get("supported_points_xy") == replay_supported,
                        f"{scene_id}/B2: gate support differs from replay")
                _expect(gate.get("mask_area_px") == int(replay["mask_area_px"]),
                        f"{scene_id}/B2: gate mask area differs from replay")
                for key in ("seed_depth_m", "mask_depth_min_m", "mask_depth_max_m"):
                    _expect(
                        abs(float(gate.get(key)) - float(replay[key])) <= 1e-6,
                        f"{scene_id}/B2: {key} differs from replay",
                    )
                _expect(
                    abs(float(gate.get("geometric_support_ratio"))
                        - float(len(replay_supported))) <= 1e-12,
                    f"{scene_id}/B2: support ratio differs from replay",
                )
                _expect(isinstance(mask_file, str),
                        f"{scene_id}/B2: replay accepted component has no mask file")
                stored_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                _expect(
                    stored_mask is not None
                    and stored_mask.shape == replay["mask"].shape
                    and np.array_equal(stored_mask, replay["mask"]),
                    f"{scene_id}/B2: stored mask pixels differ from replay",
                )

    return {
        "valid": True,
        "stage": "predictions",
        "protocol_id": PROTOCOL_ID,
        "dataset_root": str(dataset),
        "prediction_root": str(root),
        "scene_count": len(grouped),
        "record_count": len(records),
        "mode_counts": {
            mode: sum(record.get("mode") == mode for record in records)
            for mode in EXPECTED_MODES
        },
        "prediction_manifest_sha256": locked["prediction_manifest_sha256"],
        "predictions_sha256": locked["predictions_sha256"],
        "prediction_lock_sha256": sha256_file(locked["lock_path"]),
        "oracle_files_opened": False,
        "records": records,
        "capture": capture_report,
    }


def validate_all(
    dataset_root: Path,
    prediction_root: Path,
    gate_config_path: Optional[Path] = None,
) -> dict:
    """Public convenience wrapper for a complete pre-oracle validation."""

    return validate_predictions(dataset_root, prediction_root, gate_config_path)


def _json_safe_report(report: dict) -> dict:
    """Remove bulky in-memory records from the command-line report."""

    result = dict(report)
    result.pop("records", None)
    capture = result.get("capture")
    if isinstance(capture, dict):
        capture = dict(capture)
        capture.pop("records", None)
        result["capture"] = capture
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path)
    parser.add_argument("--gate-config", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = (
            validate_capture(args.dataset_root)
            if args.prediction_root is None
            else validate_all(args.dataset_root, args.prediction_root, args.gate_config)
        )
    except Exception as exc:
        print(f"PILOT_VALIDATION_FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_json_safe_report(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
