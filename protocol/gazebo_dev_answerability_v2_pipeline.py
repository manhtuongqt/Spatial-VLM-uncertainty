#!/usr/bin/env python3
"""Validate, lock, preflight, and materialize Gazebo answerability-v2."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from gazebo_dev_pipeline import (
    _depth_view,
    _geometry,
    _labels,
    _project_base_point,
    _run,
    load_yaml,
    sha256,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_dev_answerability_v2"
N = 64
SCENES = ROOT / "ur3/ur3_perception/config/gazebo_dev_answerability_v2_scenes.yaml"
ANNOTATIONS = ROOT / "ur3/ur3_perception/config/gazebo_dev_answerability_v2_annotations.yaml"
GATE = ROOT / "ur3/ur3_perception/config/gazebo_dev_answerability_v2_gate.yaml"
LOCK = ROOT / "protocol/gazebo_dev_answerability_v2_contract_lock.json"
RESULT_ROOT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2"
PREFLIGHT = RESULT_ROOT / "PREFLIGHT.json"
QC_REPORT = RESULT_ROOT / "GAZEBO_DEV_ANSWERABILITY_V2_QC.json"
DATASET = ROOT / "datasets/Gazebo_dev_answerability_v2"
V1_GT = ROOT / "datasets/Gazebo_dev/evaluator_ground_truth.jsonl"
V1_FROZEN = {
    "protocol/gazebo_dev_v1_contract_lock.json": "1e45ecbf3bb8578ed8b70fea143b8a82f94f12de1318583d182a5a9c6cd1d457",
    "datasets/Gazebo_dev/manifest.json": "39569255e33922df4ed57e036622c88a61e08f8719f899486b14f0e6f5efa825",
    "results/spatial_vlm_refspatial_v1/gazebo_dev_v1/GAZEBO_DEV_QC.json": "a59259892b6a450e71c680d4790c7f6416681320b4bb675066f5242db229f7c8",
}
REQUIRED_SOURCES = [
    Path("protocol/generate_gazebo_dev_answerability_v2_contract.py"),
    Path("protocol/gazebo_dev_answerability_v2_pipeline.py"),
    Path("protocol/gazebo_dev_answerability_v2_eval.py"),
    Path("protocol/gazebo_dev_pipeline.py"),
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
    Path("ur3/ur3_perception/config/gazebo_dev_answerability_v2_scenes.yaml"),
    Path("ur3/ur3_perception/config/gazebo_dev_answerability_v2_annotations.yaml"),
    Path("ur3/ur3_perception/config/gazebo_dev_answerability_v2_gate.yaml"),
    Path("ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro"),
    Path("ur3/ur3_moveit_control/launch/ur3_susgrip_sim.launch.py"),
    Path("ur3/ur_simulation_gz/worlds/ur3_pick_place.sdf"),
    Path("ur3/ur_simulation_gz/launch/ur_sim_control.launch.py"),
]
FRUITS = {"ycb_apple", "ycb_orange", "mango"}
STATES = {"FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"}
RELATIONS = {"leftmost", "rightmost", "second_from_left", "second_from_right"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _verify_v1_frozen() -> None:
    failures = []
    for relative, expected in V1_FROZEN.items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    if failures:
        raise ValueError(f"Gazebo_dev_v1 frozen artifacts changed: {failures}")


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def _verify_lock_sources(lock: dict) -> None:
    if lock.get("protocol_id") != PROTOCOL_ID or lock.get("status") != "LOCKED_BEFORE_CAPTURE_AND_INFERENCE":
        raise ValueError("v2 contract lock is invalid")
    changed = []
    for relative, expected in lock.get("source_artifact_sha256", {}).items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            changed.append({"path": relative, "expected": expected, "actual": actual})
    if changed:
        raise ValueError(f"locked source artifacts changed: {changed}")


def validate_contract() -> tuple[dict, dict, dict]:
    _verify_v1_frozen()
    scenes = load_yaml(SCENES)
    annotations = load_yaml(ANNOTATIONS)
    gate = load_yaml(GATE)
    for name, payload in (("scenes", scenes), ("annotations", annotations), ("gate", gate)):
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"{name}: protocol_id mismatch")
    rows = scenes.get("scenes")
    ann = annotations.get("scenes")
    if scenes.get("expected_scene_count") != N or not isinstance(rows, list) or len(rows) != N:
        raise ValueError("v2 must preregister exactly 64 capture records")
    if not isinstance(ann, dict) or len(ann) != N:
        raise ValueError("v2 must preregister exactly 64 evaluator records")
    ids = [row.get("scene_id") for row in rows]
    families = [row.get("scene_family_id") for row in rows]
    if len(set(ids)) != N or set(ids) != set(ann):
        raise ValueError("scene IDs are not unique/aligned")
    if len(set(families)) != N:
        raise ValueError("v2 requires 64 independent parent families")
    prior_families = {row["family_id"] for row in _jsonl(V1_GT)}
    if set(families) & prior_families:
        raise ValueError("v2 parent families overlap frozen Gazebo_dev_v1")
    state_counts = Counter(item.get("state") for item in ann.values())
    relation_counts = Counter(item.get("relation_variant") for item in ann.values())
    if set(state_counts) != STATES or dict(state_counts) != gate.get("state_quota"):
        raise ValueError(f"state quota mismatch: {dict(state_counts)}")
    if set(relation_counts) != RELATIONS or dict(relation_counts) != gate.get("relation_variant_quota"):
        raise ValueError(f"relation quota mismatch: {dict(relation_counts)}")
    focus_appearances = Counter()
    for row in rows:
        scene_id = row["scene_id"]
        item = ann[scene_id]
        if row.get("scene_family_id") != item.get("family_id") or item.get("split") != "dev":
            raise ValueError(f"family/split mismatch: {scene_id}")
        if row.get("task_type") != "horizontal_ordinal_ranking_answerability":
            raise ValueError(f"task type mismatch: {scene_id}")
        if item.get("relation_variant") not in RELATIONS or item.get("rank_from") not in {"left", "right"}:
            raise ValueError(f"invalid ordinal relation: {scene_id}")
        if set(row.get("poses", {})) - set(scenes.get("objects", {})):
            raise ValueError(f"unknown object in scene poses: {scene_id}")
        focus_appearances.update(FRUITS & set(row.get("poses", {})))
        state = item["state"]
        if state == "FOUND":
            if item.get("target_id") not in item.get("candidate_ids", []) or set(item.get("candidate_ids", [])) != FRUITS:
                raise ValueError(f"invalid FOUND oracle: {scene_id}")
        elif state == "AMBIGUOUS":
            if item.get("target_id") is not None or len(item.get("valid_target_ids", [])) < 2 or not set(item["valid_target_ids"]) <= set(item.get("candidate_ids", [])):
                raise ValueError(f"invalid AMBIGUOUS oracle: {scene_id}")
        elif state == "ABSENT":
            if not item.get("target_id") or item["target_id"] in row.get("poses", {}) or not item.get("context_ids"):
                raise ValueError(f"invalid ABSENT oracle: {scene_id}")
        elif state == "INSUFFICIENT_EVIDENCE":
            if not item.get("target_id") or item["target_id"] not in row.get("poses", {}) or "edge_truncation" not in item.get("failure_tags", []):
                raise ValueError(f"invalid INSUFFICIENT_EVIDENCE oracle: {scene_id}")
    for name, minimum in gate.get("focus_object_quota_min", {}).items():
        if focus_appearances[name] < minimum:
            raise ValueError(f"focus object quota not met for {name}: {focus_appearances[name]} < {minimum}")
    expected_output = {
        "point_regex": r"^POINT \[\(([0-9]*\.?[0-9]+), ([0-9]*\.?[0-9]+)\)\]$",
        "abstain_literal": "ABSTAIN",
        "allowed_actions": ["POINT", "ABSTAIN"],
        "point_state": "FOUND",
        "abstain_states": ["AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"],
    }
    if gate.get("output_contract") != expected_output:
        raise ValueError("POINT/ABSTAIN output grammar is not the frozen v2 grammar")
    if annotations.get("reference_frame") != "image_viewer_left_to_right" or gate.get("reference_frame") != "image_viewer_left_to_right":
        raise ValueError("reference frame mismatch")
    if gate.get("tie_margin_normalized") != 0.03 or gate.get("min_visible_evidence_px") != 120:
        raise ValueError("tie/visibility threshold mismatch with v1")
    if gate.get("no_sam2") is not True or gate.get("b2_opened") is not False or gate.get("test_iid_ood_access") is not False:
        raise ValueError("no-SAM2/B2-closed/Test-sealed policy is not locked")
    if gate.get("family_split_rule", {}).get("all_variants_same_parent_same_split") is not True:
        raise ValueError("family co-location policy missing")
    return scenes, annotations, gate


def lock_contract() -> None:
    scenes, annotations, gate = validate_contract()
    if LOCK.exists():
        raise FileExistsError(f"refusing to overwrite existing lock: {LOCK}")
    missing = [str(path) for path in REQUIRED_SOURCES if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required source artifacts missing: {missing}")
    base = ROOT / "RoboRefer/models/RoboRefer-2B-SFT"
    adapter = ROOT / "results/spatial_vlm_refspatial_v1/wp7_b1_clean_grounding/model/adapter_model.safetensors"
    model_hashes = {
        "b0_base": _directory_sha256(base) if base.is_dir() else None,
        "b1_adapter": sha256(adapter) if adapter.is_file() else None,
    }
    if any(value is None for value in model_hashes.values()):
        raise FileNotFoundError("B0 base or clean-B1 adapter is missing")
    model_inventory_hash = hashlib.sha256(json.dumps(model_hashes, sort_keys=True).encode("utf-8")).hexdigest()
    prior_families = sorted(row["family_id"] for row in _jsonl(V1_GT))
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "split": "dev",
        "parent_family_count": N,
        "state_counts": dict(Counter(item["state"] for item in annotations["scenes"].values())),
        "relation_variant_counts": dict(Counter(item["relation_variant"] for item in annotations["scenes"].values())),
        "relation": "horizontal_ordinal_ranking_answerability",
        "reference_frame": gate["reference_frame"],
        "output_contract": gate["output_contract"],
        "tie_margin_normalized": gate["tie_margin_normalized"],
        "min_visible_evidence_px": gate["min_visible_evidence_px"],
        "insufficient_evidence_rule": gate["insufficient_evidence_rule"],
        "camera": gate["camera"],
        "family_split_rule": gate["family_split_rule"],
        "focus_objects": sorted(FRUITS),
        "object_assets": sorted(scenes["objects"]),
        "promotion_rule": gate["promotion_rule"],
        "prior_v1_family_ids_sha256": hashlib.sha256(json.dumps(prior_families).encode("utf-8")).hexdigest(),
        "v1_frozen_artifact_sha256": V1_FROZEN,
        "source_artifact_sha256": {str(path): sha256(ROOT / path) for path in REQUIRED_SOURCES},
        "model_sha256": model_hashes,
        "model_inventory_sha256": model_inventory_hash,
        "policies": {
            "no_sam2": True,
            "b2_closed": True,
            "test_iid_ood_sealed": True,
            "no_model_before_lock": True,
            "expand_only_after_geometry_qc_pass": True,
        },
    }
    write_json(LOCK, payload)
    print(json.dumps({"status": "LOCKED", "path": str(LOCK), "sha256": sha256(LOCK)}, indent=2))


def preflight() -> None:
    scenes, annotations, gate = validate_contract()
    if not LOCK.is_file():
        raise FileNotFoundError("contract must be locked before preflight")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    _verify_lock_sources(lock)
    required_topics = {
        "/wrist_camera/color/image_raw": "sensor_msgs/msg/Image",
        "/wrist_camera/depth/image_raw": "sensor_msgs/msg/Image",
        "/wrist_camera/evaluation_labels/labels_map": "sensor_msgs/msg/Image",
        "/wrist_camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
        "/tf": "tf2_msgs/msg/TFMessage",
    }
    topic_result = _run(["ros2", "topic", "list", "-t"])
    service_result = _run(["ros2", "service", "list", "-t"])
    node_result = _run(["ros2", "node", "list"])
    topic_text, service_text = topic_result["stdout"], service_result["stdout"]
    checks = {
        "simulator_nodes_present": all(token in node_result["stdout"] for token in ("/ros_gz_bridge", "/robot_state_publisher", "/controller_manager")),
        "required_topics_present": all(f"{name} [{kind}]" in topic_text for name, kind in required_topics.items()),
        "reset_service_present": "/world/ur3_pick_place/set_pose [ros_gz_interfaces/srv/SetEntityPose]" in service_text,
    }
    samples = {
        "rgb_height": _run(["ros2", "topic", "echo", "--once", "/wrist_camera/color/image_raw", "--field", "height"], 15),
        "rgb_width": _run(["ros2", "topic", "echo", "--once", "/wrist_camera/color/image_raw", "--field", "width"], 15),
        "rgb_frame": _run(["ros2", "topic", "echo", "--once", "/wrist_camera/color/image_raw", "--field", "header"], 15),
        "depth_encoding": _run(["ros2", "topic", "echo", "--once", "/wrist_camera/depth/image_raw", "--field", "encoding"], 15),
        "labels_encoding": _run(["ros2", "topic", "echo", "--once", "/wrist_camera/evaluation_labels/labels_map", "--field", "encoding"], 15),
        "camera_info": _run(["ros2", "topic", "echo", "--once", "/wrist_camera/color/camera_info"], 15),
    }
    camera_lock = gate["camera"]["intrinsics"]
    checks.update({
        "rgb_frame_received": samples["rgb_height"]["returncode"] == 0 and samples["rgb_height"]["stdout"].startswith("480") and samples["rgb_width"]["stdout"].startswith("640") and "camera_color_optical_frame" in samples["rgb_frame"]["stdout"],
        "metric_depth_frame_received": samples["depth_encoding"]["returncode"] == 0 and "32FC1" in samples["depth_encoding"]["stdout"],
        "semantic_mask_frame_received": samples["labels_encoding"]["returncode"] == 0 and "rgb8" in samples["labels_encoding"]["stdout"],
        "intrinsics_match_lock": samples["camera_info"]["returncode"] == 0 and str(camera_lock["fx"]) in samples["camera_info"]["stdout"] and str(camera_lock["fy"]) in samples["camera_info"]["stdout"],
    })
    tf_probe = _run(["ros2", "run", "tf2_ros", "tf2_echo", "base_link", "camera_color_optical_frame"], 8)
    checks["camera_to_world_tf_valid"] = "Translation:" in tf_probe["stdout"] and "Rotation:" in tf_probe["stdout"]
    sdf = (ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place.sdf").read_text(encoding="utf-8")
    oracle_ids, labels = set(), []
    for item in annotations["scenes"].values():
        for field in ("candidate_ids", "context_ids", "valid_target_ids", "occluder_ids"):
            oracle_ids.update(item.get(field, []))
        for field in ("candidate_labels", "context_labels", "valid_target_labels", "occluder_labels"):
            labels.extend(item.get(field, []))
        if item.get("target_id"):
            oracle_ids.add(item["target_id"])
        if item.get("target_label") is not None:
            labels.append(item["target_label"])
    checks["instance_ids_bound"] = oracle_ids <= set(scenes["objects"]) and all(f'<model name="{name}">' in sdf for name in oracle_ids)
    checks["semantic_ids_retrievable"] = all(f"<label>{label}</label>" in sdf for label in set(labels))
    disk = shutil.disk_usage(ROOT)
    projected_capture_bytes = N * (640 * 480 * (3 + 4 + 3) + 100_000)
    checks["disk_capacity_ok"] = disk.free >= max(2 * 1024**3, projected_capture_bytes * 20)
    status = "PASS" if all(checks.values()) else "BLOCKED"
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_lock_sha256": sha256(LOCK),
        "checks": checks,
        "storage": {"free_bytes": disk.free, "free_gib": disk.free / 1024**3, "projected_raw_bytes": projected_capture_bytes, "projected_raw_mib": projected_capture_bytes / 1024**2},
        "observations": {"topic_inventory": topic_result, "service_inventory": service_result, "node_inventory": node_result, "samples": samples, "tf_probe": tf_probe},
    }
    write_json(PREFLIGHT, report)
    print(json.dumps({"status": status, "checks": checks, "path": str(PREFLIGHT)}, indent=2))
    if status != "PASS":
        raise SystemExit(2)


def _sensor_reasons(label_map: np.ndarray, depth: np.ndarray, camera: dict, camera_tf: dict) -> list[str]:
    reasons = []
    if depth.shape != label_map.shape or depth.dtype not in (np.float32, np.float64):
        reasons.append("DEPTH_NOT_REGISTERED_METRIC_ARRAY")
    if camera.get("width") != 640 or camera.get("height") != 480 or camera.get("frame_id") != "camera_color_optical_frame":
        reasons.append("CAMERA_CONTRACT_MISMATCH")
    if "error" in camera_tf or not camera_tf.get("position"):
        reasons.append("CAMERA_TF_MISSING")
    return reasons


def materialize(capture: Path, output: Path) -> None:
    scenes_cfg, annotations, gate = validate_contract()
    capture, output = capture.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset: {output}")
    if not LOCK.is_file() or not PREFLIGHT.is_file():
        raise ValueError("lock and preflight PASS are required")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    _verify_lock_sources(lock)
    preflight_report = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight_report.get("status") != "PASS" or preflight_report.get("contract_lock_sha256") != sha256(LOCK):
        raise ValueError("preflight does not match the current contract lock")
    manifest_path = capture / "capture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_capture_hashes = {
        "pretrial_source_lock_sha256": sha256(LOCK),
        "scene_config_sha256": sha256(SCENES),
        "annotation_file_sha256": sha256(ANNOTATIONS),
        "gate_config_file_sha256": sha256(GATE),
    }
    if manifest.get("status") != "COMPLETE" or manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("scene_count") != N:
        raise ValueError("capture manifest is not a complete v2/64-record capture")
    for field, expected in expected_capture_hashes.items():
        if manifest.get(field) != expected:
            raise ValueError(f"capture manifest hash mismatch: {field}")
    if manifest.get("source_artifacts_verified_before_capture") is not True or manifest.get("model_inventory_sha256_preregistered") != lock["model_inventory_sha256"]:
        raise ValueError("capture did not verify preregistered sources/models")
    input_manifest_path = capture / "input_manifest.jsonl"
    if manifest.get("input_manifest_sha256") != sha256(input_manifest_path):
        raise ValueError("capture input manifest hash mismatch")
    input_rows = _jsonl(input_manifest_path)
    if len(input_rows) != N or len({row["scene_id"] for row in input_rows}) != N:
        raise ValueError("capture input manifest must contain 64 unique records")
    scene_by_id = {item["scene_id"]: item for item in scenes_cfg["scenes"]}
    min_pixels = int(gate["min_visible_evidence_px"])
    tie = float(gate["tie_margin_normalized"])
    qc_rows, evaluator_rows, inference_rows, masks, depth_views = [], [], [], [], []
    for record in input_rows:
        scene_id = record["scene_id"]
        if scene_id not in scene_by_id or scene_id not in annotations["scenes"]:
            raise ValueError(f"unregistered capture row: {scene_id}")
        item, scene = annotations["scenes"][scene_id], scene_by_id[scene_id]
        label_map = _labels(capture / scene_id / "evaluator/semantic_labels.png")
        camera = json.loads((capture / record["input_files"]["camera_info"]).read_text(encoding="utf-8"))
        tf_data = json.loads((capture / record["input_files"]["tf_snapshot"]).read_text(encoding="utf-8"))
        depth = np.load(capture / record["input_files"]["depth_m"], allow_pickle=False)
        depth_view = _depth_view(depth)
        depth_views.append((scene_id, depth_view))
        camera_tf = tf_data.get("camera_color_optical_frame", {})
        reasons = _sensor_reasons(label_map, depth, camera, camera_tf)
        candidates = [_geometry(label_map, label) for label in item.get("candidate_labels", [])]
        context = [_geometry(label_map, label) for label in item.get("context_labels", [])]
        target = _geometry(label_map, item["target_label"]) if item.get("target_label") is not None else None
        state, state_verified, projected = item["state"], False, None
        if state == "FOUND":
            visible = all(value["visible_pixels"] >= min_pixels for value in candidates)
            ordered = sorted(candidates, key=lambda value: value["centroid_normalized_xy"][0]) if visible else []
            if item["rank_from"] == "right":
                ordered.reverse()
            rank_index = int(item["rank"]) - 1
            gaps = [abs(ordered[index]["centroid_normalized_xy"][0] - ordered[index + 1]["centroid_normalized_xy"][0]) for index in range(len(ordered) - 1)]
            state_verified = bool(visible and rank_index < len(ordered) and ordered[rank_index]["semantic_label"] == item["target_label"] and all(gap > tie for gap in gaps))
        elif state == "AMBIGUOUS":
            valid = [_geometry(label_map, label) for label in item["valid_target_labels"]]
            visible = all(value["visible_pixels"] >= min_pixels for value in candidates)
            xs = [value["centroid_normalized_xy"][0] for value in valid if value["centroid_normalized_xy"]]
            ordered = sorted(candidates, key=lambda value: value["centroid_normalized_xy"][0]) if visible else []
            if item["rank_from"] == "right":
                ordered.reverse()
            tied_positions = [index for index, value in enumerate(ordered) if value["semantic_label"] in set(item["valid_target_labels"])]
            state_verified = bool(visible and len(xs) >= 2 and max(xs) - min(xs) <= tie and int(item["rank"]) - 1 in tied_positions)
        elif state == "ABSENT":
            state_verified = bool(target and target["visible_pixels"] == 0 and item["target_id"] not in scene["poses"] and all(value["visible_pixels"] >= min_pixels for value in context))
        elif state == "INSUFFICIENT_EVIDENCE":
            layout = json.loads((capture / scene_id / "evaluator/capture_oracle.json").read_text(encoding="utf-8"))["requested_scene_layout_base_link"]
            if camera_tf.get("position"):
                projected = _project_base_point(layout[item["target_id"]][:3], camera_tf, {"fx": camera["k"][0], "fy": camera["k"][4], "cx": camera["k"][2], "cy": camera["k"][5]})
            rule = gate["insufficient_evidence_rule"]
            outer = float(rule["projected_center_outer_margin_fraction"])
            near_frame = bool(projected and projected["pixel_xy"] and -640 * outer <= projected["pixel_xy"][0] < 640 * (1 + outer) and -480 * outer <= projected["pixel_xy"][1] < 480 * (1 + outer) and 0.1 <= projected["camera_xyz"][2] <= 2.0)
            other_candidates = [value for value in candidates if value["semantic_label"] != item["target_label"]]
            state_verified = bool(target and int(rule["min_visible_pixels_inclusive"]) <= target["visible_pixels"] < int(rule["max_visible_pixels_exclusive"]) and near_frame and all(value["visible_pixels"] >= min_pixels for value in other_candidates))
        if not state_verified:
            reasons.append(f"REQUESTED_STATE_NOT_VERIFIED:{state}")
        if state == "AMBIGUOUS":
            target_mask = np.isin(label_map, np.asarray(item["valid_target_labels"], dtype=label_map.dtype)).astype(np.uint8) * 255
        else:
            target_mask = np.zeros_like(label_map, dtype=np.uint8) if target is None else (label_map == int(item["target_label"])).astype(np.uint8) * 255
        anchor_labels = sorted((set(item.get("candidate_labels", [])) | set(item.get("context_labels", [])) | set(item.get("occluder_labels", []))) - ({item.get("target_label")} if item.get("target_label") is not None else set()))
        anchor_union = np.isin(label_map, np.asarray(anchor_labels, dtype=label_map.dtype)).astype(np.uint8) * 255 if anchor_labels else np.zeros_like(label_map, dtype=np.uint8)
        masks.append((scene_id, target_mask, anchor_union))
        valid_ids = item.get("valid_target_ids", [item.get("target_id")] if item.get("target_id") else [])
        valid_labels = item.get("valid_target_labels", [item.get("target_label")] if item.get("target_label") is not None else [])
        evaluator_rows.append({
            "sample_id": scene_id,
            "scene_id": scene_id,
            "family_id": item["family_id"],
            "split": "dev",
            "relation": "horizontal_ordinal_ranking_answerability",
            "relation_variant": item["relation_variant"],
            "reference_frame": gate["reference_frame"],
            "answerability_state": state,
            "answerability_verified": state_verified,
            "target_id": item.get("target_id"),
            "target_category": item["target_category"],
            "target_semantic_label": item.get("target_label"),
            "valid_target_ids": valid_ids,
            "valid_target_semantic_labels": valid_labels,
            "target_xy": target["centroid_normalized_xy"] if state == "FOUND" and target else None,
            "target_mask": f"records/{scene_id}/evaluator/target_mask.png",
            "anchor_ids": sorted((set(item.get("candidate_ids", [])) | set(item.get("context_ids", [])) | set(item.get("occluder_ids", []))) - ({item.get("target_id")} if item.get("target_id") else set())),
            "anchor_mask_union": f"records/{scene_id}/evaluator/anchor_mask_union.png",
            "candidate_set": candidates,
            "context_set": context,
            "visibility": {"method": "visible_semantic_mask_pixels", "visible_pixels": target["visible_pixels"] if target else 0, "minimum_sufficient_pixels": min_pixels, "sufficient": bool(target and target["visible_pixels"] >= min_pixels)},
            "target_center_projection_from_geometry": projected,
            "failure_tags": item["failure_tags"],
        })
        inference_rows.append({
            "sample_id": scene_id,
            "scene_id": scene_id,
            "family_id": item["family_id"],
            "split": "dev",
            "relation": "horizontal_ordinal_ranking_answerability",
            "image": f"records/{scene_id}/input/rgb.png",
            "depth": f"records/{scene_id}/input/depth_view.png",
            "metric_depth": f"records/{scene_id}/input/depth_m.npy",
            "instruction": record["instruction"] + " " + record["coordinate_suffix"],
        })
        qc_rows.append({"scene_id": scene_id, "family_id": item["family_id"], "requested_state": state, "relation_variant": item["relation_variant"], "state_verified": state_verified, "reasons": reasons, "target_visible_pixels": target["visible_pixels"] if target else 0})
    families = [row["family_id"] for row in evaluator_rows]
    prior_families = {row["family_id"] for row in _jsonl(V1_GT)}
    leakage = {
        "unique_parent_families": len(set(families)) == N,
        "all_split_dev": all(row["split"] == "dev" for row in evaluator_rows),
        "no_v1_family_overlap": not bool(set(families) & prior_families),
        "all_variants_co_located": all(len({row["split"] for row in evaluator_rows if row["family_id"] == family}) == 1 for family in set(families)),
        "inference_manifest_oracle_free": all(not ({"answerability_state", "target_id", "target_xy", "target_mask", "candidate_set", "semantic_labels", "failure_tags"} & set(row)) for row in inference_rows),
    }
    passed = all(not row["reasons"] for row in qc_rows) and all(leakage.values())
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if passed else "BLOCKED",
        "contract_lock_sha256": sha256(LOCK),
        "capture_root": str(capture),
        "capture_manifest_sha256": sha256(manifest_path),
        "state_counts": dict(Counter(row["requested_state"] for row in qc_rows)),
        "relation_variant_counts": dict(Counter(row["relation_variant"] for row in qc_rows)),
        "verified_state_counts": dict(Counter(row["requested_state"] for row in qc_rows if row["state_verified"])),
        "failed_scene_count": sum(bool(row["reasons"]) for row in qc_rows),
        "leakage_checks": leakage,
        "scenes": qc_rows,
        "dataset_materialized": passed,
    }
    write_json(QC_REPORT, report)
    if not passed:
        print(json.dumps({"status": "BLOCKED", "qc_report": str(QC_REPORT), "failed": [row for row in qc_rows if row["reasons"]], "leakage_checks": leakage}, indent=2))
        raise SystemExit(2)
    output.mkdir(parents=True, exist_ok=True)
    for scene_id, target_mask, anchor_union in masks:
        mask_dir = output / "records" / scene_id / "evaluator"
        mask_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_dir / "target_mask.png"), target_mask) or not cv2.imwrite(str(mask_dir / "anchor_mask_union.png"), anchor_union):
            raise RuntimeError(f"cannot write evaluator masks for {scene_id}")
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
    for filename, rows in (("inference_manifest.jsonl", inference_rows), ("evaluator_ground_truth.jsonl", evaluator_rows)):
        with (output / filename).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(output / "manifest.json", {
        "schema_version": 1,
        "dataset": "Gazebo_dev_answerability_v2",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "records": N,
        "parent_families": N,
        "split": "dev",
        "state_counts": report["state_counts"],
        "relation_variant_counts": report["relation_variant_counts"],
        "contract_lock_sha256": sha256(LOCK),
        "capture_manifest_sha256": sha256(manifest_path),
        "qc_report_sha256": sha256(QC_REPORT),
        "model_input_allowlist": ["image", "depth", "instruction"],
        "output_contract": gate["output_contract"],
        "no_sam2": True,
        "b2_opened": False,
        "test_iid_ood_access": False,
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
        print("PASS: Gazebo_dev_answerability_v2 contract is internally consistent")
    elif args.command == "lock":
        lock_contract()
    elif args.command == "preflight":
        preflight()
    else:
        materialize(args.capture, args.output)


if __name__ == "__main__":
    main()
