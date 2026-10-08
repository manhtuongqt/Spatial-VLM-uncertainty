#!/usr/bin/env python3
"""Oracle-free runner for 10 scenes x 6 WP1 depth conditions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from wp1_common import (
    CONDITIONS,
    PROTOCOL_ID,
    WP1Error,
    canonical_json_sha256,
    encode_image,
    find_forbidden_inference_keys,
    flat_depth_view,
    inverted_depth_view,
    localized_holes_edge_view,
    normalize_metric_depth,
    parse_point,
    pilot_tree_digest,
    query_model,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    shuffled_source_scene,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def save_overlay(path: Path, rgb: np.ndarray, point: Optional[list[int]], condition: str) -> None:
    image = rgb.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 38), (20, 20, 20), -1)
    cv2.putText(image, condition, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 210, 255), 2)
    if point is not None:
        center = (int(point[0]), int(point[1]))
        cv2.circle(image, center, 13, (255, 255, 255), 3)
        cv2.drawMarker(image, center, (50, 70, 255), cv2.MARKER_CROSS, 22, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise WP1Error(f"cannot write overlay: {path}")


def load_locked_inputs(dataset_root: Path) -> list[dict]:
    capture_path = dataset_root / "capture_manifest.json"
    input_manifest_path = dataset_root / "input_manifest.jsonl"
    capture = read_json(capture_path)
    if capture.get("status") != "COMPLETE" or capture.get("protocol_id") != "roborefer_pilot_v0":
        raise WP1Error("pilot capture manifest is not COMPLETE/roborefer_pilot_v0")
    if capture.get("input_manifest_sha256") != sha256_file(input_manifest_path):
        raise WP1Error("pilot input manifest hash differs from capture lock")
    scenes = read_jsonl(input_manifest_path)
    if len(scenes) != 10 or len({item.get("scene_id") for item in scenes}) != 10:
        raise WP1Error("WP1 requires exactly 10 unique locked pilot scenes")
    forbidden = find_forbidden_inference_keys(scenes)
    if forbidden:
        raise WP1Error(f"inference input leaks evaluator keys: {forbidden}")
    for scene in scenes:
        if scene.get("protocol_id") != "roborefer_pilot_v0":
            raise WP1Error(f"input protocol mismatch: {scene.get('scene_id')}")
        for key, file_name in scene.get("input_files", {}).items():
            path = (dataset_root / file_name).resolve()
            if dataset_root.resolve() not in path.parents or not path.is_file():
                raise WP1Error(f"invalid input path: {file_name}")
            if sha256_file(path) != scene.get("input_sha256", {}).get(key):
                raise WP1Error(f"input hash mismatch: {file_name}")
    return scenes


def build_condition_views(
    scene_ids: Sequence[str],
    correct_views: dict[str, np.ndarray],
    scene_id: str,
) -> dict[str, tuple[Optional[np.ndarray], dict]]:
    correct = correct_views[scene_id]
    shuffled_scene = shuffled_source_scene(scene_ids, scene_id)
    holes, holes_meta = localized_holes_edge_view(correct)
    return {
        "rgb_only": (None, {"transform": "NO_DEPTH_INPUT"}),
        "correct_depth": (correct, {"transform": "inverse_depth_percentile_v1", "source_scene_id": scene_id}),
        "flat_depth": (flat_depth_view(correct), {"transform": "constant_uint8", "value": 127}),
        "shuffled_depth": (
            correct_views[shuffled_scene],
            {"transform": "cyclic_next_scene_derangement", "source_scene_id": shuffled_scene},
        ),
        "inverted_depth": (inverted_depth_view(correct), {"transform": "uint8_255_minus_correct"}),
        "localized_holes_edge": (holes, {"transform": "edge_and_fixed_patch_zeroing", **holes_meta}),
    }


def verify_source_lock(protocol: dict, protocol_path: Path) -> None:
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("status") != "LOCKED_BEFORE_INFERENCE":
        raise WP1Error("WP1 protocol is not locked before inference")
    if protocol.get("expected_record_count") != 60 or protocol.get("condition_order") != list(CONDITIONS):
        raise WP1Error("WP1 protocol does not lock 10 x 6 records")
    for relative_path, expected_hash in protocol.get("runner_source_sha256", {}).items():
        path = ROOT / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise WP1Error(f"runner source differs from protocol lock: {relative_path}")
    expected_pilot = protocol.get("pilot_lock", {}).get("tree_sha256")
    current_pilot = pilot_tree_digest(ROOT, protocol["pilot_lock"]["relative_path"])
    if current_pilot != expected_pilot:
        raise WP1Error("locked pilot tree changed before WP1 inference")


def run(protocol_path: Path, server_url: str, timeout_sec: float) -> Path:
    protocol_path = protocol_path.resolve()
    protocol = read_json(protocol_path)
    verify_source_lock(protocol, protocol_path)
    result_root = (ROOT / protocol["result_root"]).resolve()
    dataset_root = (ROOT / protocol["dataset_root"]).resolve()
    if result_root.exists():
        raise FileExistsError(f"WP1 output is immutable and already exists: {result_root}")
    scenes = load_locked_inputs(dataset_root)
    scene_ids = [str(item["scene_id"]) for item in scenes]
    result_root.mkdir(parents=True, exist_ok=False)
    inference_root = result_root / "inference"
    inference_root.mkdir()

    loaded = {}
    correct_views = {}
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        rgb_path = dataset_root / scene["input_files"]["rgb"]
        depth_path = dataset_root / scene["input_files"]["depth_m"]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth_m = np.load(depth_path, allow_pickle=False)
        if rgb is None or depth_m.ndim != 2 or rgb.shape[:2] != depth_m.shape:
            raise WP1Error(f"registered RGB/depth mismatch: {scene_id}")
        loaded[scene_id] = (rgb, depth_m)
        correct_views[scene_id] = normalize_metric_depth(depth_m)

    records = []
    query_index = 0
    started = time.monotonic()
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        rgb, _ = loaded[scene_id]
        height, width = rgb.shape[:2]
        prompt = str(scene["instruction"]).strip() + " " + str(scene["coordinate_suffix"]).strip()
        rgb_jpeg = encode_image(rgb, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90])
        scene_root = inference_root / "scenes" / scene_id
        scene_root.mkdir(parents=True)
        rgb_request_path = scene_root / "request_rgb_q90.jpg"
        rgb_request_path.write_bytes(rgb_jpeg)
        condition_views = build_condition_views(scene_ids, correct_views, scene_id)
        for condition in CONDITIONS:
            query_index += 1
            depth_view, transform = condition_views[condition]
            depth_png = None
            depth_path = None
            if depth_view is not None:
                depth_png = encode_image(depth_view, ".png")
                depth_path = scene_root / f"request_depth_{condition}.png"
                depth_path.write_bytes(depth_png)
            answer, latency_ms, response = query_model(
                server_url,
                prompt,
                rgb_jpeg,
                depth_png,
                timeout_sec,
                protocol["model"]["inventory_sha256"],
            )
            parsed = parse_point(answer, width, height)
            record_path = scene_root / f"prediction_{condition}.json"
            overlay_path = scene_root / f"overlay_{condition}.png"
            point = parsed["pixel_points_xy"][0] if parsed["pixel_points_xy"] else None
            record = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "scene_id": scene_id,
                "condition": condition,
                "query_index": query_index,
                "query_attempt": 1,
                "query_status": "SUCCESS",
                "instruction": scene["instruction"],
                "prompt": prompt,
                "image_width": width,
                "image_height": height,
                "request_enable_depth": depth_png is not None,
                "request_rgb_file": relative(rgb_request_path, result_root),
                "request_rgb_sha256": sha256_bytes(rgb_jpeg),
                "request_depth_file": relative(depth_path, result_root) if depth_path else None,
                "request_depth_sha256": sha256_bytes(depth_png) if depth_png else None,
                "request_contract_sha256": canonical_json_sha256({
                    "prompt": prompt,
                    "rgb_sha256": sha256_bytes(rgb_jpeg),
                    "depth_sha256": sha256_bytes(depth_png) if depth_png else None,
                    "enable_depth": depth_png is not None,
                }),
                "condition_provenance": transform,
                "source_input_sha256": dict(scene["input_sha256"]),
                "raw_answer": answer,
                **parsed,
                "inference_latency_ms": round(float(latency_ms), 3),
                "response": response,
                "record_file": relative(record_path, result_root),
                "overlay_file": relative(overlay_path, result_root),
                "oracle_or_annotation_read_by_runner": False,
                "target_handoff_published": False,
                "robot_manipulation_performed": False,
            }
            record["prediction_sha256"] = canonical_json_sha256(record)
            write_json(record_path, record)
            save_overlay(overlay_path, rgb, point, condition)
            records.append(record)
            print(
                f"WP1_INFERENCE {query_index:02d}/60 {scene_id}/{condition}: "
                f"{parsed['parse_status']} {answer}",
                flush=True,
            )

    predictions_path = inference_root / "predictions.jsonl"
    write_jsonl(predictions_path, records)
    exact_count = sum(item["parse_status"] == "EXACT_ONE_NORMALIZED_POINT" for item in records)
    inference_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_manifest": relative(protocol_path, ROOT),
        "protocol_manifest_sha256": sha256_file(protocol_path),
        "dataset_root": relative(dataset_root, ROOT),
        "result_root": relative(result_root, ROOT),
        "scene_count": len(scenes),
        "condition_count": len(CONDITIONS),
        "condition_order": list(CONDITIONS),
        "record_count": len(records),
        "successful_http_query_count": len(records),
        "exact_point_parse_count": exact_count,
        "query_attempts_per_scene_condition": 1,
        "query_order": "scene_order_then_locked_condition_order",
        "generation_mode": "greedy",
        "random_seed": 8132026,
        "model_inventory_sha256": protocol["model"]["inventory_sha256"],
        "predictions_file": relative(predictions_path, result_root),
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_s": round(time.monotonic() - started, 3),
        "oracle_or_annotation_read_by_runner": False,
        "robot_manipulation_performed": False,
    }
    inference_manifest_path = inference_root / "inference_manifest.json"
    write_json(inference_manifest_path, inference_manifest)

    artifact_files = []
    for path in sorted(item for item in inference_root.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        artifact_files.append({
            "path": relative(path, result_root),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    artifact_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scope": "All inference outputs except this closing manifest and prediction_lock.json; both are covered by the lock hash chain.",
        "file_count": len(artifact_files),
        "files": artifact_files,
    }
    artifact_manifest_path = inference_root / "artifact_manifest.json"
    write_json(artifact_manifest_path, artifact_manifest)
    lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "exact_point_parse_count": exact_count,
        "protocol_manifest_sha256": sha256_file(protocol_path),
        "inference_manifest_sha256": sha256_file(inference_manifest_path),
        "predictions_sha256": sha256_file(predictions_path),
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        "oracle_opened_before_lock": False,
        "immutable_warning": "Do not modify inference artifacts after this lock.",
    }
    lock_path = result_root / "prediction_lock.json"
    write_json(lock_path, lock)
    print(f"WP1_PREDICTIONS_LOCKED: {lock_path}", flush=True)
    return lock_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--protocol", type=Path, default=ROOT / "protocol/wp1_manifest.json")
    value.add_argument("--server-url", default="http://127.0.0.1:25548")
    value.add_argument("--timeout-sec", type=float, default=240.0)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        run(arguments.protocol, arguments.server_url, arguments.timeout_sec)
    except Exception as exc:
        print(f"WP1_RUN_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

