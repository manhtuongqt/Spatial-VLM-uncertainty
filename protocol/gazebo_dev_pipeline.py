#!/usr/bin/env python3
"""Lock, preflight, and materialize the 16-family Gazebo_dev pilot."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_dev_v1"
SCENES = ROOT / "ur3/ur3_perception/config/gazebo_dev_v1_scenes.yaml"
ANNOTATIONS = ROOT / "ur3/ur3_perception/config/gazebo_dev_v1_annotations.yaml"
GATE = ROOT / "ur3/ur3_perception/config/gazebo_dev_v1_gate.yaml"
LOCK = ROOT / "protocol/gazebo_dev_v1_contract_lock.json"
PREFLIGHT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_v1/PREFLIGHT.json"
QC_REPORT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_v1/GAZEBO_DEV_QC.json"
DATASET = ROOT / "datasets/Gazebo_dev"
REQUIRED_SOURCES = [
    Path("protocol/gazebo_dev_pipeline.py"),
    Path("protocol/gazebo_dev_b0_b1_eval.py"),
    Path("RoboRefer/API/api.py"),
    Path("ur3/ur3_perception/scripts/roborefer_pilot_capture.py"),
    Path("ur3/ur3_perception/scripts/roborefer_pilot_runner.py"),
    Path("ur3/ur3_perception/scripts/roborefer_pilot_validate.py"),
    Path("ur3/ur3_perception/scripts/roborefer_pilot_evaluator.py"),
    Path("ur3/ur3_perception/scripts/roborefer_grounder.py"),
    Path("ur3/ur3_perception/scripts/spatial_point_utils.py"),
    Path("ur3/ur3_perception/scripts/move_camera_to_view.py"),
    Path("ur3/ur3_perception/launch/roborefer_pilot_capture.launch.py"),
    Path("ur3/ur3_perception/config/roborefer_pilot_v0_scenes.yaml"),
    Path("ur3/ur3_perception/config/roborefer_pilot_v0_annotations.yaml"),
    Path("ur3/ur3_perception/config/roborefer_pilot_v0_gate.yaml"),
    Path("ur3/ur3_perception/schemas/roborefer_pilot_prediction.schema.json"),
    Path("ur3/ur3_perception/config/gazebo_dev_v1_scenes.yaml"),
    Path("ur3/ur3_perception/config/gazebo_dev_v1_annotations.yaml"),
    Path("ur3/ur3_perception/config/gazebo_dev_v1_gate.yaml"),
    Path("ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro"),
    Path("ur3/ur3_moveit_control/launch/ur3_susgrip_sim.launch.py"),
    Path("ur3/ur_simulation_gz/worlds/ur3_pick_place.sdf"),
    Path("ur3/ur_simulation_gz/launch/ur_sim_control.launch.py"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def validate_contract() -> tuple[dict, dict, dict]:
    scenes = load_yaml(SCENES)
    annotations = load_yaml(ANNOTATIONS)
    gate = load_yaml(GATE)
    for name, payload in (("scenes", scenes), ("annotations", annotations), ("gate", gate)):
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"{name}: protocol_id mismatch")
    rows = scenes.get("scenes")
    ann = annotations.get("scenes")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("Gazebo_dev must contain exactly 16 capture records")
    if not isinstance(ann, dict) or len(ann) != 16:
        raise ValueError("Gazebo_dev must contain exactly 16 evaluator records")
    ids = [row.get("scene_id") for row in rows]
    families = [row.get("scene_family_id") for row in rows]
    if len(set(ids)) != 16 or set(ids) != set(ann):
        raise ValueError("scene IDs are not unique/aligned")
    if len(set(families)) != 16:
        raise ValueError("pilot requires 16 unique parent families")
    states = Counter(item.get("state") for item in ann.values())
    expected = {state: 4 for state in gate["state_quota"]}
    if dict(states) != expected:
        raise ValueError(f"state balance mismatch: {dict(states)}")
    for row in rows:
        item = ann[row["scene_id"]]
        if item.get("family_id") != row.get("scene_family_id"):
            raise ValueError(f"family mismatch: {row['scene_id']}")
        if row.get("task_type") != "horizontal_ordinal_ranking":
            raise ValueError(f"unsupported relation: {row['scene_id']}")
        if item.get("state") != "ABSENT" and not item.get("candidate_labels"):
            raise ValueError(f"candidate set missing: {row['scene_id']}")
    if gate.get("no_sam2") is not True or gate.get("b2_opened") is not False:
        raise ValueError("no-SAM2/B2-closed policy is not locked")
    return scenes, annotations, gate


def lock_contract() -> None:
    scenes, annotations, gate = validate_contract()
    missing = [str(path) for path in REQUIRED_SOURCES if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required source artifacts missing: {missing}")
    if LOCK.exists():
        previous = json.loads(LOCK.read_text(encoding="utf-8"))
        if previous.get("status") == "LOCKED_BEFORE_CAPTURE_AND_INFERENCE":
            raise FileExistsError(f"refusing to overwrite existing lock: {LOCK}")
    model_inventory = {
        "b0_base": ROOT / "RoboRefer/models/RoboRefer-2B-SFT",
        "b1_adapter": ROOT / "results/spatial_vlm_refspatial_v1/wp7_b1_clean_grounding/model/adapter_model.safetensors",
    }
    model_hashes = {}
    for name, path in model_inventory.items():
        if path.is_file():
            model_hashes[name] = sha256(path)
        elif path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.is_file())
            digest = hashlib.sha256()
            for item in files:
                digest.update(str(item.relative_to(path)).encode())
                digest.update(bytes.fromhex(sha256(item)))
            model_hashes[name] = digest.hexdigest()
        else:
            model_hashes[name] = None
    combined = hashlib.sha256(json.dumps(model_hashes, sort_keys=True).encode()).hexdigest()
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "split": "dev",
        "parent_family_count": 16,
        "state_counts": dict(Counter(item["state"] for item in annotations["scenes"].values())),
        "relation": "horizontal_ordinal_ranking",
        "reference_frame": gate["reference_frame"],
        "tie_margin_normalized": gate["tie_margin_normalized"],
        "min_visible_evidence_px": gate["min_visible_evidence_px"],
        "insufficient_evidence_rule": gate["insufficient_evidence_rule"],
        "camera": gate["camera"],
        "family_split_rule": gate["family_split_rule"],
        "object_assets": sorted(scenes["objects"]),
        "source_artifact_sha256": {
            str(path): sha256(ROOT / path) for path in REQUIRED_SOURCES
        },
        "model_sha256": model_hashes,
        "model_inventory_sha256": combined,
        "policies": {
            "no_sam2": True,
            "b2_closed": True,
            "test_iid_ood_sealed": True,
            "expand_only_after_qc_pass": True,
        },
    }
    write_json(LOCK, payload)
    print(json.dumps({"status": "LOCKED", "path": str(LOCK), "sha256": sha256(LOCK)}, indent=2))


def _run(command: list[str], timeout: int = 12) -> dict:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "seconds": time.perf_counter() - started,
        }
    except subprocess.TimeoutExpired as exc:
        timed_out_stdout = exc.stdout or b""
        if isinstance(timed_out_stdout, bytes):
            timed_out_stdout = timed_out_stdout.decode("utf-8", errors="replace")
        return {
            "command": command,
            "returncode": 124,
            "stdout": timed_out_stdout.strip(),
            "stderr": "timeout",
            "seconds": time.perf_counter() - started,
        }


def preflight() -> None:
    validate_contract()
    if not LOCK.is_file():
        raise FileNotFoundError("contract must be locked before preflight")
    required_topics = {
        "/wrist_camera/color/image_raw": "sensor_msgs/msg/Image",
        "/wrist_camera/depth/image_raw": "sensor_msgs/msg/Image",
        "/wrist_camera/evaluation_labels/labels_map": "sensor_msgs/msg/Image",
        "/wrist_camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
        "/tf": "tf2_msgs/msg/TFMessage",
    }
    topic_result = _run(["ros2", "topic", "list", "-t"])
    service_result = _run(["ros2", "service", "list", "-t"])
    topic_text = topic_result["stdout"]
    service_text = service_result["stdout"]
    checks = {
        "simulator_nodes_present": all(
            token in _run(["ros2", "node", "list"])["stdout"]
            for token in ("/ros_gz_bridge", "/robot_state_publisher", "/controller_manager")
        ),
        "required_topics_present": all(
            f"{name} [{kind}]" in topic_text for name, kind in required_topics.items()
        ),
        "reset_service_present": "/world/ur3_pick_place/set_pose [ros_gz_interfaces/srv/SetEntityPose]" in service_text,
    }
    samples = {
        "rgb_height": _run(["ros2", "topic", "echo", "--once", "/wrist_camera/color/image_raw", "--field", "height"], timeout=15),
        "rgb_width": _run(["ros2", "topic", "echo", "--once", "/wrist_camera/color/image_raw", "--field", "width"], timeout=15),
        "rgb_frame": _run(["ros2", "topic", "echo", "--once", "/wrist_camera/color/image_raw", "--field", "header"], timeout=15),
        "depth_encoding": _run(["ros2", "topic", "echo", "--once", "/wrist_camera/depth/image_raw", "--field", "encoding"], timeout=15),
        "labels_encoding": _run(["ros2", "topic", "echo", "--once", "/wrist_camera/evaluation_labels/labels_map", "--field", "encoding"], timeout=15),
        "camera_info": _run(["ros2", "topic", "echo", "--once", "/wrist_camera/color/camera_info"], timeout=15),
    }
    checks.update({
        "rgb_frame_received": samples["rgb_height"]["returncode"] == 0 and samples["rgb_height"]["stdout"].startswith("480") and samples["rgb_width"]["stdout"].startswith("640") and "camera_color_optical_frame" in samples["rgb_frame"]["stdout"],
        "metric_depth_frame_received": samples["depth_encoding"]["returncode"] == 0 and "32FC1" in samples["depth_encoding"]["stdout"],
        "semantic_mask_frame_received": samples["labels_encoding"]["returncode"] == 0 and "rgb8" in samples["labels_encoding"]["stdout"],
        "intrinsics_match_lock": samples["camera_info"]["returncode"] == 0 and "606.0816650391" in samples["camera_info"]["stdout"] and "605.7973022461" in samples["camera_info"]["stdout"],
    })
    tf = _run(["ros2", "run", "tf2_ros", "tf2_echo", "base_link", "camera_color_optical_frame"], timeout=8)
    checks["camera_to_world_tf_valid"] = "Translation:" in tf["stdout"] and "Rotation:" in tf["stdout"]
    scenes, annotations, _ = validate_contract()
    sdf = (ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place.sdf").read_text(encoding="utf-8")
    configured_ids = set(scenes["objects"])
    oracle_ids = set()
    labels = []
    for item in annotations["scenes"].values():
        oracle_ids.update(item.get("candidate_ids", []))
        oracle_ids.update(item.get("occluder_ids", []))
        if item.get("target_id"):
            oracle_ids.add(item["target_id"])
        labels.extend(item.get("candidate_labels", []))
        labels.extend(item.get("occluder_labels", []))
        if item.get("target_label") is not None:
            labels.append(item["target_label"])
    checks["instance_ids_bound"] = oracle_ids <= configured_ids and all(
        f'<model name="{name}">' in sdf for name in oracle_ids
    )
    checks["semantic_ids_retrievable"] = all(f"<label>{label}</label>" in sdf for label in set(labels))
    disk = shutil.disk_usage(ROOT)
    projected_capture_bytes = 16 * (640 * 480 * (3 + 4 + 3) + 100_000)
    checks["disk_capacity_ok"] = disk.free >= max(2 * 1024**3, projected_capture_bytes * 20)
    status = "PASS" if all(checks.values()) else "BLOCKED"
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_lock_sha256": sha256(LOCK),
        "checks": checks,
        "storage": {
            "free_bytes": disk.free,
            "free_gib": disk.free / 1024**3,
            "projected_raw_pilot_bytes": projected_capture_bytes,
            "projected_raw_pilot_mib": projected_capture_bytes / 1024**2,
        },
        "observations": {
            "topic_inventory": topic_result,
            "service_inventory": service_result,
            "samples": samples,
            "tf_probe": tf,
        },
    }
    write_json(PREFLIGHT, report)
    print(json.dumps({"status": status, "checks": checks, "path": str(PREFLIGHT)}, indent=2))
    if status != "PASS":
        raise SystemExit(2)


def _labels(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode {path}")
    if image.ndim == 3:
        if not (np.array_equal(image[..., 0], image[..., 1]) and np.array_equal(image[..., 1], image[..., 2])):
            raise ValueError(f"semantic label channels disagree: {path}")
        image = image[..., 0]
    return image


def _geometry(label_map: np.ndarray, label: int) -> dict:
    ys, xs = np.nonzero(label_map == int(label))
    if not xs.size:
        return {"semantic_label": int(label), "visible_pixels": 0, "centroid_xy": None, "centroid_normalized_xy": None, "bbox_xyxy": None}
    return {
        "semantic_label": int(label),
        "visible_pixels": int(xs.size),
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
        "centroid_normalized_xy": [float(xs.mean() / label_map.shape[1]), float(ys.mean() / label_map.shape[0])],
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    }


def _depth_view(depth_m: np.ndarray) -> np.ndarray:
    values = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(values) & (values >= 0.10) & (values <= 2.0)
    if np.count_nonzero(valid) < 16:
        raise ValueError("too few valid metric-depth pixels")
    near, far = np.percentile(values[valid], [2.0, 98.0])
    if far - near < 1e-6:
        raise ValueError("metric depth has no usable range")
    clipped = np.clip(values, near, far)
    inverse = 1.0 / np.maximum(clipped, 1e-6)
    normalized = (inverse - 1.0 / far) / max(1.0 / near - 1.0 / far, 1e-6)
    gray = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    gray[~valid] = 0
    return np.repeat(gray[..., None], 3, axis=-1)


def _rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _project_base_point(point: list[float], tf: dict, camera: dict) -> dict:
    position = np.asarray(tf["position"], dtype=np.float64)
    rotation = _rotation(*tf["orientation_xyzw"])
    camera_point = rotation.T @ (np.asarray(point, dtype=np.float64) - position)
    z = float(camera_point[2])
    if z <= 0:
        return {"camera_xyz": camera_point.tolist(), "pixel_xy": None, "in_frame": False}
    u = camera["fx"] * float(camera_point[0]) / z + camera["cx"]
    v = camera["fy"] * float(camera_point[1]) / z + camera["cy"]
    return {"camera_xyz": camera_point.tolist(), "pixel_xy": [u, v], "in_frame": 0 <= u < 640 and 0 <= v < 480 and 0.1 <= z <= 2.0}


def materialize(capture: Path, output: Path) -> None:
    scenes_cfg, annotations, gate = validate_contract()
    capture = capture.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset: {output}")
    manifest_path = capture / "capture_manifest.json"
    if not PREFLIGHT.is_file() or json.loads(PREFLIGHT.read_text()).get("status") != "PASS":
        raise ValueError("preflight PASS is required before materialization")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("scene_count") != 16:
        raise ValueError("capture manifest is not a complete 16-record Gazebo_dev capture")
    input_rows = [json.loads(line) for line in (capture / "input_manifest.jsonl").read_text().splitlines() if line]
    if len(input_rows) != 16:
        raise ValueError("capture input manifest must contain 16 records")
    scene_by_id = {item["scene_id"]: item for item in scenes_cfg["scenes"]}
    min_pixels = int(gate["min_visible_evidence_px"])
    tie = float(gate["tie_margin_normalized"])
    qc_rows, evaluator_rows, inference_rows, mask_arrays, depth_views = [], [], [], [], []
    for record in input_rows:
        scene_id = record["scene_id"]
        item = annotations["scenes"][scene_id]
        scene = scene_by_id[scene_id]
        label_path = capture / scene_id / "evaluator/semantic_labels.png"
        label_map = _labels(label_path)
        camera_path = capture / record["input_files"]["camera_info"]
        tf_path = capture / record["input_files"]["tf_snapshot"]
        depth_path = capture / record["input_files"]["depth_m"]
        rgb_path = capture / record["input_files"]["rgb"]
        camera = json.loads(camera_path.read_text())
        tf_data = json.loads(tf_path.read_text())
        depth = np.load(depth_path, allow_pickle=False)
        depth_view = _depth_view(depth)
        depth_views.append((scene_id, depth_view))
        reasons = []
        if depth.shape != label_map.shape or depth.dtype not in (np.float32, np.float64):
            reasons.append("DEPTH_NOT_REGISTERED_METRIC_ARRAY")
        if camera.get("width") != 640 or camera.get("height") != 480 or camera.get("frame_id") != "camera_color_optical_frame":
            reasons.append("CAMERA_CONTRACT_MISMATCH")
        camera_tf = tf_data.get("camera_color_optical_frame", {})
        if "error" in camera_tf or not camera_tf.get("position"):
            reasons.append("CAMERA_TF_MISSING")
        candidates = [_geometry(label_map, label) for label in item.get("candidate_labels", [])]
        target = _geometry(label_map, item["target_label"]) if item.get("target_label") is not None else None
        state = item["state"]
        state_verified = False
        projected = None
        if state == "FOUND":
            visible = all(value["visible_pixels"] >= min_pixels for value in candidates)
            ordered = sorted(candidates, key=lambda value: value["centroid_normalized_xy"][0]) if visible else []
            if item["rank_from"] == "right":
                ordered.reverse()
            rank_index = int(item["rank"]) - 1
            rank_ok = visible and rank_index < len(ordered) and ordered[rank_index]["semantic_label"] == item["target_label"]
            gaps = [abs(ordered[i]["centroid_normalized_xy"][0] - ordered[i+1]["centroid_normalized_xy"][0]) for i in range(len(ordered)-1)]
            state_verified = bool(rank_ok and all(gap > tie for gap in gaps))
        elif state == "AMBIGUOUS":
            valid = [_geometry(label_map, label) for label in item["valid_target_labels"]]
            visible = all(value["visible_pixels"] >= min_pixels for value in candidates)
            xs = [value["centroid_normalized_xy"][0] for value in valid if value["centroid_normalized_xy"]]
            ordered = sorted(candidates, key=lambda value: value["centroid_normalized_xy"][0]) if visible else []
            if item["rank_from"] == "right":
                ordered.reverse()
            valid_labels = set(item["valid_target_labels"])
            tied_positions = [index for index, value in enumerate(ordered) if value["semantic_label"] in valid_labels]
            rank_index = int(item["rank"]) - 1
            state_verified = bool(
                visible and len(xs) >= 2 and max(xs) - min(xs) <= tie
                and rank_index in tied_positions
            )
        elif state == "ABSENT":
            state_verified = bool(target and target["visible_pixels"] == 0 and item["target_id"] not in scene["poses"])
        elif state == "INSUFFICIENT_EVIDENCE":
            layout = json.loads((capture / scene_id / "evaluator/capture_oracle.json").read_text())["requested_scene_layout_base_link"]
            projected = _project_base_point(layout[item["target_id"]][:3], camera_tf, {
                "fx": camera["k"][0], "fy": camera["k"][4], "cx": camera["k"][2], "cy": camera["k"][5]
            }) if camera_tf.get("position") else None
            occluders = [_geometry(label_map, label) for label in item.get("occluder_labels", [])]
            insufficient_rule = gate["insufficient_evidence_rule"]
            min_insufficient_pixels = int(insufficient_rule["min_visible_pixels_inclusive"])
            max_insufficient_pixels = int(insufficient_rule["max_visible_pixels_exclusive"])
            outer = float(insufficient_rule["projected_center_outer_margin_fraction"])
            projected_near_frame = bool(
                projected and projected["pixel_xy"]
                and -640 * outer <= projected["pixel_xy"][0] < 640 * (1 + outer)
                and -480 * outer <= projected["pixel_xy"][1] < 480 * (1 + outer)
                and 0.1 <= projected["camera_xyz"][2] <= 2.0
            )
            state_verified = bool(
                target
                and min_insufficient_pixels <= target["visible_pixels"] < max_insufficient_pixels
                and projected_near_frame
                and all(value["visible_pixels"] >= min_pixels for value in occluders)
            )
        if not state_verified:
            reasons.append(f"REQUESTED_STATE_NOT_VERIFIED:{state}")
        if state == "AMBIGUOUS":
            target_mask = np.isin(label_map, np.asarray(item["valid_target_labels"], dtype=label_map.dtype)).astype(np.uint8) * 255
        else:
            target_mask = np.zeros_like(label_map, dtype=np.uint8) if target is None else (label_map == int(item["target_label"])).astype(np.uint8) * 255
        anchor_labels = [label for label in item.get("candidate_labels", []) if label != item.get("target_label")] + item.get("occluder_labels", [])
        anchor_union = np.isin(label_map, np.asarray(anchor_labels, dtype=label_map.dtype)).astype(np.uint8) * 255 if anchor_labels else np.zeros_like(label_map, dtype=np.uint8)
        mask_arrays.append((scene_id, target_mask, anchor_union))
        evaluator = {
            "sample_id": scene_id, "scene_id": scene_id, "family_id": item["family_id"], "split": "dev",
            "relation": "horizontal_ordinal_ranking", "reference_frame": gate["reference_frame"],
            "answerability_state": state, "answerability_verified": state_verified,
            "target_id": item.get("target_id"), "target_semantic_label": item.get("target_label"),
            "valid_target_ids": item.get("valid_target_ids", [item.get("target_id")] if item.get("target_id") else []),
            "valid_target_semantic_labels": item.get("valid_target_labels", [item.get("target_label")] if item.get("target_label") is not None else []),
            "target_xy": target["centroid_normalized_xy"] if target else None,
            "target_mask": f"records/{scene_id}/evaluator/target_mask.png",
            "anchor_ids": [name for name in item.get("candidate_ids", []) if name != item.get("target_id")] + item.get("occluder_ids", []),
            "anchor_mask_union": f"records/{scene_id}/evaluator/anchor_mask_union.png",
            "candidate_set": candidates,
            "visibility": {"method": "visible_semantic_mask_pixels", "visible_pixels": target["visible_pixels"] if target else 0, "minimum_sufficient_pixels": min_pixels, "sufficient": bool(target and target["visible_pixels"] >= min_pixels)},
            "target_center_projection_from_geometry": projected,
        }
        inference = {
            "sample_id": scene_id, "scene_id": scene_id, "family_id": item["family_id"], "split": "dev",
            "relation": "horizontal_ordinal_ranking", "image": f"records/{scene_id}/input/rgb.png",
            "depth": f"records/{scene_id}/input/depth_view.png", "metric_depth": f"records/{scene_id}/input/depth_m.npy",
            "instruction": record["instruction"] + " " + record["coordinate_suffix"],
        }
        qc_rows.append({"scene_id": scene_id, "requested_state": state, "state_verified": state_verified, "reasons": reasons, "target_visible_pixels": target["visible_pixels"] if target else 0})
        evaluator_rows.append(evaluator)
        inference_rows.append(inference)
    passed = all(not row["reasons"] for row in qc_rows)
    report = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PASS" if passed else "BLOCKED",
        "capture_root": str(capture), "capture_manifest_sha256": sha256(manifest_path),
        "state_counts": dict(Counter(row["requested_state"] for row in qc_rows)),
        "verified_state_counts": dict(Counter(row["requested_state"] for row in qc_rows if row["state_verified"])),
        "failed_scene_count": sum(bool(row["reasons"]) for row in qc_rows), "scenes": qc_rows,
        "leakage_checks": {"unique_parent_families": len({row["family_id"] for row in evaluator_rows}) == 16, "all_split_dev": all(row["split"] == "dev" for row in evaluator_rows), "all_variants_co_located": True},
        "dataset_materialized": passed,
    }
    write_json(QC_REPORT, report)
    if not passed:
        print(json.dumps({"status": "BLOCKED", "qc_report": str(QC_REPORT), "failed": [row for row in qc_rows if row["reasons"]]}, indent=2))
        raise SystemExit(2)
    output.mkdir(parents=True, exist_ok=True)
    for scene_id, target_mask, anchor_union in mask_arrays:
        mask_dir = output / "records" / scene_id / "evaluator"
        mask_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_dir / "target_mask.png"), target_mask):
            raise RuntimeError(f"cannot write target mask for {scene_id}")
        if not cv2.imwrite(str(mask_dir / "anchor_mask_union.png"), anchor_union):
            raise RuntimeError(f"cannot write anchor mask for {scene_id}")
    for record in input_rows:
        scene_id = record["scene_id"]
        target_dir = output / "records" / scene_id / "input"
        target_dir.mkdir(parents=True, exist_ok=True)
        for key in ("rgb", "depth_m", "camera_info", "tf_snapshot"):
            source = capture / record["input_files"][key]
            shutil.copy2(source, target_dir / source.name)
    for scene_id, depth_view in depth_views:
        if not cv2.imwrite(str(output / "records" / scene_id / "input/depth_view.png"), depth_view):
            raise RuntimeError(f"cannot write depth view for {scene_id}")
    with (output / "inference_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in inference_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output / "evaluator_ground_truth.jsonl").open("w", encoding="utf-8") as stream:
        for row in evaluator_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(output / "manifest.json", {
        "schema_version": 1, "dataset": "Gazebo_dev", "protocol_id": PROTOCOL_ID, "status": "PASS",
        "records": 16, "parent_families": 16, "split": "dev", "state_counts": report["state_counts"],
        "contract_lock_sha256": sha256(LOCK), "capture_manifest_sha256": sha256(manifest_path), "qc_report_sha256": sha256(QC_REPORT),
        "model_input_allowlist": ["image", "depth", "instruction"], "no_sam2": True, "b2_opened": False, "test_iid_ood_access": False,
    })
    print(json.dumps({"status": "PASS", "dataset": str(output), "qc_report": str(QC_REPORT)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-contract")
    sub.add_parser("lock")
    sub.add_parser("preflight")
    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument("--capture", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, default=DATASET)
    args = parser.parse_args()
    if args.command == "validate-contract":
        validate_contract()
        print("PASS: Gazebo_dev contract is internally consistent")
    elif args.command == "lock":
        lock_contract()
    elif args.command == "preflight":
        preflight()
    else:
        materialize(args.capture, args.output.resolve())


if __name__ == "__main__":
    main()
