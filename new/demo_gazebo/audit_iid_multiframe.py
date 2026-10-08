#!/usr/bin/env python3
"""Freeze V2 predictions, then audit them against sealed Gazebo-only labels.

The semantic labels are *never* model inputs and must never become MoveIt
control inputs.  This is a simulation pre-handoff audit, not grasp success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def infer(run_dir: Path, prompt: str) -> dict:
    # No evaluator_only path or semantic-label topic appears in this stage.
    from infer_and_render import extract_frozen_features, frozen_paths, predict_v2
    from pcrau.policy import decide
    from pcrau.utils import read_json as load_config

    index = read_json(run_dir / "capture/capture_index.json")
    if index["camera"] != "wrist_camera" or index["view_pose"] != "test_iid":
        raise ValueError("Expected Test-IID-pose wrist camera")
    if index["frame_count"] < 2 or len(index["frames"]) != index["frame_count"]:
        raise ValueError("Need multiple distinct synchronized frames")
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")
    freeze, checkpoint, calibrator_path, config_path = frozen_paths()
    config = load_config(config_path)
    calibrator = load_config(calibrator_path)
    locked = []
    stamps = []
    for row in index["frames"]:
        frame_id = row["frame_id"]
        media = run_dir / "capture/inference" / frame_id
        rgb = media / "rgb_model_input.jpg"
        depth = media / "depth_relative_model_input.png"
        if file_hash(rgb) != row["inference_sha256"]["rgb_model_input"] or file_hash(depth) != row["inference_sha256"]["depth_relative_model_input"]:
            raise ValueError(f"Model input hash mismatch: {frame_id}")
        stamp = float(row["sensor_stamps_s"]["rgb"])
        if row["maximum_sensor_skew_s"] > 0.08 or (stamps and stamp <= stamps[-1] + 0.10):
            raise ValueError(f"Invalid multiframe timestamp/synchronization: {frame_id}")
        stamps.append(stamp)
        features, feature_runtime = extract_frozen_features(rgb, depth)
        prediction, model_runtime = predict_v2(features, prompt, config, checkpoint,
                                               config_path, "test_iid_pose_demo_audit")
        decision = decide(prediction, calibrator)
        prediction_path = media / "prediction.json"
        decision_path = media / "decision.json"
        write_json(prediction_path, prediction)
        write_json(decision_path, decision)
        locked.append({"frame_id": frame_id, "prediction_sha256": file_hash(prediction_path),
                       "decision_sha256": file_hash(decision_path),
                       "feature_runtime": feature_runtime, "model_runtime": model_runtime})
        print(f"INFERRED {frame_id} action={decision['action']} point={prediction['spatial']['map_pixel_xy']}", flush=True)
    lock = {"schema_version": 1, "status": "PREDICTIONS_LOCKED_BEFORE_EVALUATOR_ACCESS",
            "prompt": prompt, "checkpoint_sha256": freeze["selected_checkpoint"]["model_sha256"],
            "calibrator_sha256": freeze["frozen_calibrator"]["sha256"],
            "frame_count": len(locked), "frames": locked,
            "robot_motion_commanded": False, "evaluator_labels_accessed_by_v2": False}
    write_json(run_dir / "prediction_lock.json", lock)
    return lock


def label_for_model(world_path: Path, model_name: str) -> int:
    tree = ET.parse(world_path)
    model = tree.getroot().find(f"./world/model[@name='{model_name}']")
    if model is None:
        raise ValueError(f"Target model not in generated Gazebo world: {model_name}")
    labels = [node.text for node in model.findall("./plugin/label")]
    if len(labels) != 1:
        raise ValueError(f"Expected unique Gazebo evaluator label: {model_name}")
    return int(labels[0])


def check_without_oracle(run_dir: Path) -> dict:
    """RGB-D target check after V2 lock, before any evaluator-label read."""
    from independent_apple_rgbd import check_apple

    lock = read_json(run_dir / "prediction_lock.json")
    if lock["status"] != "PREDICTIONS_LOCKED_BEFORE_EVALUATOR_ACCESS":
        raise ValueError("V2 predictions are not locked")
    checks = []
    points = []
    output = run_dir / "independent_rgbd_check"
    output.mkdir(exist_ok=False)
    for item in lock["frames"]:
        frame_id = item["frame_id"]
        media = run_dir / "capture/inference" / frame_id
        prediction_path = media / "prediction.json"
        decision_path = media / "decision.json"
        if file_hash(prediction_path) != item["prediction_sha256"] or file_hash(decision_path) != item["decision_sha256"]:
            raise ValueError(f"Locked model output changed: {frame_id}")
        prediction = read_json(prediction_path)
        decision = read_json(decision_path)
        rgb = cv2.imread(str(media / "rgb_original.png"), cv2.IMREAD_COLOR)
        depth_m = np.load(media / "depth_metric.npy", allow_pickle=False)
        point = tuple(map(int, prediction["spatial"]["map_pixel_xy"]))
        result, mask = check_apple(rgb, depth_m, point)
        result.update({"frame_id": frame_id, "v2_action": decision["action"]})
        checks.append(result)
        points.append(point)
        cv2.imwrite(str(output / f"{frame_id}_component.png"), mask)
        print(f"RGBD_CHECK {frame_id} pass={result['pass']} action={decision['action']}", flush=True)
    point_array = np.asarray(points, dtype=np.float64)
    jitter = float(np.max(np.linalg.norm(point_array - np.median(point_array, axis=0), axis=1)))
    all_checker_pass = all(row["pass"] for row in checks)
    all_execute = all(row["v2_action"] == "EXECUTE" for row in checks)
    output_value = {"schema_version": 1, "status": "ORACLE_FREE_CHECK_LOCKED_BEFORE_EVALUATOR_ACCESS",
                    "frame_count": len(checks),
                    "checker_pass_count": sum(row["pass"] for row in checks),
                    "maximum_candidate_jitter_px": jitter,
                    "oracle_free_pre_handoff_pass": all_checker_pass and all_execute and jitter <= 30.0,
                    "moveit_connected": False, "robot_motion_commanded": False,
                    "gazebo_semantic_labels_accessed": False,
                    "frames": checks}
    path = run_dir / "independent_rgbd_check.json"
    write_json(path, output_value)
    write_json(run_dir / "independent_rgbd_check_lock.json", {
        "status": "ORACLE_FREE_CHECK_LOCKED_BEFORE_EVALUATOR_ACCESS",
        "sha256": file_hash(path), "frame_count": len(checks),
    })
    return output_value


def overlay(rgb: np.ndarray, target: np.ndarray, point: tuple[int, int],
            title: str, passed: bool) -> np.ndarray:
    image = rgb.copy()
    contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (40, 210, 60), 2)
    cv2.drawMarker(image, point, (230, 210, 20), cv2.MARKER_CROSS, 24, 3)
    cv2.rectangle(image, (0, 0), (639, 35), (28, 31, 35), -1)
    cv2.putText(image, title, (9, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.57,
                (60, 220, 75) if passed else (60, 65, 220), 2, cv2.LINE_AA)
    cv2.putText(image, "EVALUATOR-ONLY LABEL / NOT A ROBOT COMMAND", (8, 466),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def evaluate(run_dir: Path, target_model: str) -> dict:
    lock = read_json(run_dir / "prediction_lock.json")
    check_lock = read_json(run_dir / "independent_rgbd_check_lock.json")
    check_path = run_dir / "independent_rgbd_check.json"
    if check_lock["status"] != "ORACLE_FREE_CHECK_LOCKED_BEFORE_EVALUATOR_ACCESS" or file_hash(check_path) != check_lock["sha256"]:
        raise ValueError("Oracle-free RGB-D check must be locked before evaluator access")
    independent = read_json(check_path)
    checks_by_id = {row["frame_id"]: row for row in independent["frames"]}
    index = read_json(run_dir / "capture/capture_index.json")
    if lock["status"] != "PREDICTIONS_LOCKED_BEFORE_EVALUATOR_ACCESS" or lock["frame_count"] != index["frame_count"]:
        raise ValueError("Predictions must be locked before opening evaluator labels")
    label_id = label_for_model(run_dir / "world_with_static_ur3.sdf", target_model)
    capture_by_id = {row["frame_id"]: row for row in index["frames"]}
    results = []
    overlays = []
    points = []
    for item in lock["frames"]:
        frame_id = item["frame_id"]
        capture_row = capture_by_id[frame_id]
        media = run_dir / "capture/inference" / frame_id
        evaluator = run_dir / "capture/evaluator_only" / frame_id
        prediction_path = media / "prediction.json"
        decision_path = media / "decision.json"
        if file_hash(prediction_path) != item["prediction_sha256"] or file_hash(decision_path) != item["decision_sha256"]:
            raise ValueError(f"Locked prediction changed: {frame_id}")
        prediction = read_json(prediction_path)
        decision = read_json(decision_path)
        label_meta = read_json(evaluator / "label_meta.json")
        label_path = evaluator / "semantic_labels.png"
        if label_meta["frame_id"] != frame_id or file_hash(label_path) != label_meta["label_sha256"]:
            raise ValueError(f"Evaluator label mismatch: {frame_id}")
        if abs(float(label_meta["stamp_s"]) - float(capture_row["sensor_stamps_s"]["rgb"])) > 0.08:
            raise ValueError(f"Evaluator label frame not synchronized: {frame_id}")
        labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(media / "rgb_original.png"), cv2.IMREAD_COLOR)
        depth_m = np.load(media / "depth_metric.npy", allow_pickle=False)
        if labels is None or labels.shape != (480, 640) or rgb is None or rgb.shape[:2] != (480, 640) or depth_m.shape != (480, 640):
            raise ValueError(f"Evaluator sensor shape mismatch: {frame_id}")
        target = labels == label_id
        interior = cv2.erode(target.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        x, y = map(int, prediction["spatial"]["map_pixel_xy"])
        if not (0 <= x < 640 and 0 <= y < 480):
            raise ValueError(f"Model point outside image: {frame_id}")
        visible_px = int(target.sum())
        point_in_target = bool(target[y, x])
        point_in_interior = bool(interior[y, x])
        distance = float(cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 5)[y, x])
        point_depth = float(depth_m[y, x])
        target_depth = float(np.nanmedian(depth_m[target])) if visible_px else math.nan
        metric_consistent = (math.isfinite(point_depth) and 0.10 <= point_depth <= 2.0 and
                             math.isfinite(target_depth) and abs(point_depth - target_depth) <= 0.05)
        identity_pass = visible_px >= 100 and point_in_interior and metric_consistent
        action = str(decision["action"])
        result = {"frame_id": frame_id, "target_model": target_model,
                  "target_label_id": label_id, "target_visible_px": visible_px,
                  "candidate_xy": [x, y], "point_in_target": point_in_target,
                  "point_in_target_interior": point_in_interior,
                  "point_to_target_distance_px": distance,
                  "point_depth_m": point_depth, "target_median_depth_m": target_depth,
                  "metric_depth_consistent": metric_consistent,
                  "independent_sim_identity_check_pass": identity_pass,
                  "oracle_free_rgbd_check_pass": bool(checks_by_id[frame_id]["pass"]),
                  "v2_action": action, "v2_risk": decision["calibrated_grounding_risk"],
                  "frame_pre_handoff_pass": identity_pass and checks_by_id[frame_id]["pass"] and action == "EXECUTE"}
        results.append(result)
        points.append((x, y))
        display = overlay(rgb, target, (x, y),
                          f"{frame_id}: {action} | on target={point_in_target} | risk={result['v2_risk']:.3f}",
                          identity_pass)
        overlay_path = evaluator / "audit_overlay.png"
        cv2.imwrite(str(overlay_path), display)
        overlays.append(display)
        print(f"AUDIT {frame_id} identity={identity_pass} action={action} distance={distance:.1f}px", flush=True)
    point_array = np.asarray(points, dtype=np.float64)
    jitter_px = float(np.max(np.linalg.norm(point_array - np.median(point_array, axis=0), axis=1)))
    all_identity = all(row["independent_sim_identity_check_pass"] for row in results)
    all_execute = all(row["v2_action"] == "EXECUTE" for row in results)
    temporal_stable = jitter_px <= 30.0
    sim_pre_handoff_pass = all_identity and all_execute and temporal_stable and independent["oracle_free_pre_handoff_pass"]
    summary = {"schema_version": 1, "frame_count": len(results),
               "camera_pose": "test_iid_locked_joint_pose",
               "target_model": target_model, "target_label_id": label_id,
               "point_in_target_count": sum(row["point_in_target"] for row in results),
               "independent_identity_pass_count": sum(row["independent_sim_identity_check_pass"] for row in results),
               "oracle_free_rgbd_check_pass_count": independent["checker_pass_count"],
               "oracle_free_pre_handoff_pass": independent["oracle_free_pre_handoff_pass"],
               "action_counts": dict(Counter(row["v2_action"] for row in results)),
               "maximum_candidate_jitter_px": jitter_px,
               "temporal_stable_within_30px": temporal_stable,
               "sim_pre_handoff_pass": sim_pre_handoff_pass,
               "moveit_connected": False, "robot_motion_commanded": False,
               "oracle_used_by_v2": False,
               "oracle_used_only_for_post_prediction_sim_audit": True,
               "not_a_real_world_object_identity_gate": True,
               "frames": results}
    write_json(run_dir / "audit_summary.json", summary)
    blank = np.full((480, 640, 3), 245, dtype=np.uint8)
    columns = 2
    if len(overlays) % columns:
        overlays.append(blank)
    rows = math.ceil(len(overlays) / columns)
    montage = np.vstack([np.hstack(overlays[i:i + columns]) for i in range(0, rows * columns, columns)])
    cv2.imwrite(str(run_dir / "audit_montage.png"), montage)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("infer", "check", "evaluate"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--target-model", default="ycb_apple")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.stage == "infer":
        result = infer(run_dir, args.prompt)
    elif args.stage == "check":
        result = check_without_oracle(run_dir)
    else:
        result = evaluate(run_dir, args.target_model)
    print(json.dumps({key: value for key, value in result.items() if key != "frames"}, indent=2))
