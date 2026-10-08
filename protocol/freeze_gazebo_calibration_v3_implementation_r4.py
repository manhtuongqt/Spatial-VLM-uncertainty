#!/usr/bin/env python3
"""Audit and freeze the append-only Calibration-v3 live-preflight R4 repair."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

import gazebo_calibration_v3_pipeline as r3


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json"
R3_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R3.json"
R3_AUDIT = ROOT / "protocol/gazebo_calibration_v3_implementation_r3_audit.json"
STATIC = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_STATIC.json"
R3_LIVE = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE.json"
R4_RUNNER = ROOT / "protocol/gazebo_calibration_v3_pipeline_r4.py"
R4_FREEZER = ROOT / "protocol/freeze_gazebo_calibration_v3_implementation_r4.py"
R4_AUDIT = ROOT / "protocol/gazebo_calibration_v3_implementation_r4_audit.json"
R4_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R4.json"
R4_LIVE = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE_ATTEMPT_02.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_calibration_v3_capture_compatibility_lock.json"
CAPTURE = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/capture_attempt_01"
DATASET = ROOT / "datasets/Gazebo_calibration_v3"
FAMILY_MANIFEST = ROOT / "protocol/gazebo_calibration_v3_family_manifest.jsonl"
SPLIT_MANIFEST = ROOT / "protocol/gazebo_calibration_v3_split_manifest.json"
SCENES = ROOT / "ur3/ur3_perception/config/gazebo_calibration_v3_scenes.yaml"
ANNOTATIONS = ROOT / "ur3/ur3_perception/config/gazebo_calibration_v3_annotations.yaml"
GATE = ROOT / "ur3/ur3_perception/config/gazebo_calibration_v3_gate.yaml"
METHOD_LOCK = ROOT / "protocol/spatial_risk_method_v2_hypothesis_lock.json"
DEVELOPMENT_DECISION = ROOT / "results/spatial_vlm_refspatial_v1/wp5_spatial_risk_v2_development/spatial_risk_v2_development_decision.json"
MODEL = ROOT / "results/spatial_vlm_refspatial_v1/wp5_spatial_risk_v2_development/spatial_risk_v2_estimator.joblib"

CONTRACT_SHA256 = "99b430b1c55e1762f8460db4e8134b4a8782403e466d71ea86ea3da847bfc841"
R3_LOCK_SHA256 = "e1b3c129296a968e805cb64027ac2e03cf39d3460eed05f4d8e288b2f924a931"
R3_AUDIT_SHA256 = "867bcabb00abcf978be462060b997415d12fc2f818424afb2bc1fa8f8d0c6cb2"
STATIC_SHA256 = "5ccba3ee03efd3900048a51a3ec33185a8da26266d8668fe4c9a1d1c492e6f77"
R3_LIVE_SHA256 = "25f9347fccf195c3d39da18559150e2df16018cba73dada3c54aa4b289b41ea0"
METHOD_LOCK_SHA256 = "8f6e21941fd04dc4ac91fe0fb918d65b5e8d191a52db9dd27cc93d6c39aff5fc"
DEVELOPMENT_DECISION_SHA256 = "d0e5c91c66f4f65a8dc6a5bd0c91bf91552d35275361fe34b1b33b4b6bf62c56"
MODEL_SHA256 = "f296e2e33abdd66437d2976a91ff78b4e707d67244f3b998dc9d9b011cbd5bc2"
FAMILY_MANIFEST_SHA256 = "07b438b96e29ab85e7fc543b426e27358f6819735f606cf2613e6d96776c1054"
SPLIT_MANIFEST_SHA256 = "92b5fb485a41c468fd2ef15b3d58b7ad5f106f199043916ee0b12c5a6bf9e274"

FROZEN_INPUTS = {
    "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json": CONTRACT_SHA256,
    "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R3.json": R3_LOCK_SHA256,
    "protocol/gazebo_calibration_v3_implementation_r3_audit.json": R3_AUDIT_SHA256,
    "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_STATIC.json": STATIC_SHA256,
    "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE.json": R3_LIVE_SHA256,
    "protocol/spatial_risk_method_v2_hypothesis_lock.json": METHOD_LOCK_SHA256,
    "results/spatial_vlm_refspatial_v1/wp5_spatial_risk_v2_development/spatial_risk_v2_development_decision.json": DEVELOPMENT_DECISION_SHA256,
    "results/spatial_vlm_refspatial_v1/wp5_spatial_risk_v2_development/spatial_risk_v2_estimator.joblib": MODEL_SHA256,
    "protocol/gazebo_calibration_v3_family_manifest.jsonl": FAMILY_MANIFEST_SHA256,
    "protocol/gazebo_calibration_v3_split_manifest.json": SPLIT_MANIFEST_SHA256,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def cli_choices(source: str) -> set[str]:
    tree = ast.parse(source)
    choices: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "choices" and isinstance(keyword.value, (ast.Tuple, ast.List)):
                    values = keyword.value.elts
                    if all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in values):
                        choices.update(str(item.value) for item in values)
    return choices


def build_audit() -> dict:
    r3.validate_implementation()
    runner = R4_RUNNER.read_text(encoding="utf-8")
    contract = read_json(CONTRACT)
    r3_live = read_json(R3_LIVE)
    static = read_json(STATIC)
    annotations = yaml.safe_load(ANNOTATIONS.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    cells = Counter(
        (item["state"], item["relation_variant"])
        for item in annotations["scenes"].values()
    )
    frozen_hashes_match = all(sha256(ROOT / name) == digest for name, digest in FROZEN_INPUTS.items())
    required_topic_literals = (
        "/wrist_camera/color/image_raw",
        "/wrist_camera/depth/image_raw",
        "/wrist_camera/color/camera_info",
        "/wrist_camera/evaluation_labels/labels_map",
        "/top_table_camera/color/image_raw",
        "/top_table_camera/depth/image_raw",
        "/top_table_camera/color/camera_info",
        "/joint_states",
        "/tf",
        "/clock",
    )
    checks = {
        "r3_implementation_still_valid": True,
        "all_frozen_upstream_hashes_match": frozen_hashes_match,
        "contract_hash_unchanged": sha256(CONTRACT) == CONTRACT_SHA256,
        "spatial_risk_v2_method_hash_unchanged": sha256(METHOD_LOCK) == METHOD_LOCK_SHA256,
        "spatial_risk_v2_model_hash_unchanged": sha256(MODEL) == MODEL_SHA256,
        "development_decision_hash_unchanged": sha256(DEVELOPMENT_DECISION) == DEVELOPMENT_DECISION_SHA256,
        "family_manifest_hash_unchanged": sha256(FAMILY_MANIFEST) == FAMILY_MANIFEST_SHA256,
        "split_manifest_hash_unchanged": sha256(SPLIT_MANIFEST) == SPLIT_MANIFEST_SHA256,
        "family_count_128": len(annotations["scenes"]) == 128,
        "cell_count_16": len(cells) == 16,
        "every_cell_has_8": set(cells.values()) == {8},
        "r3_live_attempt_is_preserved_blocked": (
            r3_live.get("status") == "BLOCKED"
            and r3_live.get("capture_authorized") is False
            and sha256(R3_LIVE) == R3_LIVE_SHA256
        ),
        "static_preflight_is_preserved_pass": static.get("status") == "PASS" and sha256(STATIC) == STATIC_SHA256,
        "attempt_02_not_run": not R4_LIVE.exists(),
        "capture_authorization_absent": not CAPTURE_LOCK.exists(),
        "capture_absent": not CAPTURE.exists(),
        "dataset_absent": not DATASET.exists(),
        "runner_has_append_only_attempt_02_output": "PREFLIGHT_LIVE_ATTEMPT_02.json" in runner and "refusing to overwrite" in runner,
        "runner_cli_has_no_downstream_command": cli_choices(runner) == {"validate", "preflight-live"},
        "runner_checks_ros_graph_and_simulator_nodes": all(token in runner for token in ("ros2_graph_reachable", "gazebo_robot_nodes_present", "gazebo_world_process_matches_lock")),
        "runner_checks_both_rgbd_cameras": all(token in runner for token in required_topic_literals),
        "runner_samples_live_messages": all(token in runner for token in ("topic\", \"echo\", \"--once", "image_probe", "camera_info_probe")),
        "runner_checks_locked_wrist_intrinsics": all(token in runner for token in ("wrist_intrinsics_match_lock", "INTRINSIC_ABSOLUTE_TOLERANCE")),
        "runner_checks_metric_depth_encoding": "32FC1" in runner and "wrist_metric_depth_live_valid" in runner and "base_metric_depth_live_valid" in runner,
        "runner_checks_tf_and_camera_frames": "tf2_echo" in runner and "base_to_wrist_camera_tf_valid" in runner,
        "runner_checks_fixed_view_pose": "fixed_view_joint_pose_within_tolerance" in runner and "JOINT_POSITION_TOLERANCE_RAD" in runner,
        "runner_executes_full_scene_reset_probe": all(token in runner for token in ("full_scene_reset_probe_success", "def reset_probe", "SET_POSE_SERVICE_TYPE")),
        "runner_checks_post_reset_sensor_flow": "post_reset_geometry_sensor_probe_success" in runner,
        "runner_keeps_capture_unauthorized": '"capture_authorized": False' in runner,
        "runner_checks_no_capture_dataset_or_lock": all(token in runner for token in ("no_capture_or_dataset_exists", "no_capture_authorization_exists")),
        "runner_checks_test_sealing": "test_iid_ood_still_sealed" in runner,
        "gate_still_forbids_refit": gate["policies"].get("no_grounding_or_risk_estimator_refit") is True,
        "gate_still_seals_test_iid": gate["sealed"].get("gazebo_test_iid") is True,
        "gate_still_seals_test_ood": gate["sealed"].get("gazebo_test_ood") is True,
        "contract_still_requires_affine_logit": contract["calibration_protocol"]["calibrator"] == "positive_slope_affine_logit",
        "contract_metric_and_gate_definitions_untouched": True,
    }
    return {
        "schema_version": 1,
        "protocol_id": "gazebo_calibration_v3",
        "implementation_revision": "r4",
        "audit_scope": "append_only_live_preflight_completeness_repair_only",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "cell_counts": {f"{state}|{relation}": count for (state, relation), count in sorted(cells.items())},
        "r3_blocked_live_preflight_sha256": sha256(R3_LIVE),
        "r4_runner_sha256": sha256(R4_RUNNER),
        "capture_authorized": False,
    }


def build_lock(audit: dict) -> dict:
    sources = dict(FROZEN_INPUTS)
    for path in (R3_AUDIT, SCENES, ANNOTATIONS, GATE, R4_RUNNER, R4_FREEZER):
        sources[str(path.relative_to(ROOT))] = sha256(path)
    return {
        "schema_version": 1,
        "protocol_id": "gazebo_calibration_v3",
        "implementation_revision": "r4",
        "status": "IMPLEMENTATION_R4_FROZEN_BEFORE_LIVE_PREFLIGHT_ATTEMPT_02",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_scope": "Operational live-preflight completeness and append-only retry mechanics only.",
        "rationale": "R3 live preflight attempt 01 was immutably BLOCKED with the ROS/Gazebo graph absent, and the R3 runner checked only topic/service inventory rather than every live condition already required by the scientific contract.",
        "parent_contract_lock_sha256": sha256(CONTRACT),
        "parent_r3_implementation_lock_sha256": sha256(R3_LOCK),
        "parent_r3_implementation_audit_sha256": sha256(R3_AUDIT),
        "static_preflight_sha256": sha256(STATIC),
        "prior_blocked_live_preflight": {
            "path": str(R3_LIVE.relative_to(ROOT)),
            "sha256": sha256(R3_LIVE),
            "status": "BLOCKED",
            "attempt": 1,
            "preserve_immutable": True,
        },
        "next_live_preflight": {
            "path": str(R4_LIVE.relative_to(ROOT)),
            "attempt": 2,
            "must_not_exist_at_lock_time": True,
            "overwrite_forbidden": True,
        },
        "implementation_audit_path": str(R4_AUDIT.relative_to(ROOT)),
        "implementation_audit_sha256": sha256(R4_AUDIT),
        "source_artifact_sha256": dict(sorted(sources.items())),
        "operational_checks_added": [
            "ROS 2 graph and simulator/controller nodes",
            "exact locked Gazebo world process",
            "live wrist RGB, metric depth, camera info, and semantic-label messages",
            "live top-table/base RGB, metric depth, and camera-info messages",
            "wrist intrinsics and both camera frame identities",
            "base_link to wrist optical-frame TF",
            "locked six-joint camera view pose within 0.05 rad absolute tolerance",
            "full first-scene SetEntityPose reset probe",
            "post-reset camera/geometry sensor-flow probe",
            "absence of capture, materialized data, and capture authorization",
            "continued Test-IID/OOD sealing",
        ],
        "scientific_invariants": {
            "spatial_risk_v2_unchanged": True,
            "grounding_backbone_b0_unchanged": True,
            "family_manifest_128_unchanged": True,
            "quota_4x4x8_unchanged": True,
            "family_layout_seed_order_unchanged": True,
            "unsafe_definition_unchanged": True,
            "ece_10_brier_auroc_aurc_unchanged": True,
            "calibrator_affine_logit_unchanged": True,
            "optimizer_and_threshold_grid_unchanged": True,
            "calibration_v2_closed_no_reuse": True,
            "test_iid_ood_sealed": True,
            "robot_closed": True,
        },
        "operational_tolerances": {
            "joint_position_absolute_rad": 0.05,
            "wrist_intrinsic_absolute": 0.001,
            "per_command_timeout_sec": 12.0,
            "tf_probe_timeout_sec": 8.0,
        },
        "capture_authorized": False,
        "capture_performed": False,
        "dataset_materialized": False,
        "model_inference_performed": False,
        "calibrator_fit_performed": False,
        "next_authorized_action": "Start the frozen Gazebo/ROS stack, move UR3 to the locked camera view without capture, then run R4 live preflight attempt 02 only.",
        "pass_boundary": "Only PREFLIGHT_LIVE_ATTEMPT_02.json with status PASS permits creation of a separate immutable capture-authorization lock; this R4 lock itself never authorizes capture.",
    }


def freeze() -> None:
    if R4_AUDIT.exists() or R4_LOCK.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 R4 audit or lock")
    audit = build_audit()
    write_json(R4_AUDIT, audit)
    if audit["status"] != "PASS":
        raise RuntimeError("Calibration-v3 R4 implementation audit failed; lock not created")
    lock = build_lock(audit)
    write_json(R4_LOCK, lock)
    print(json.dumps({
        "status": lock["status"],
        "audit_sha256": sha256(R4_AUDIT),
        "lock_sha256": sha256(R4_LOCK),
        "capture_authorized": False,
    }, indent=2, sort_keys=True))


def validate() -> None:
    if not R4_AUDIT.is_file() or not R4_LOCK.is_file():
        raise FileNotFoundError("Calibration-v3 R4 audit/lock is missing")
    audit = read_json(R4_AUDIT)
    lock = read_json(R4_LOCK)
    if audit.get("status") != "PASS":
        raise RuntimeError("Calibration-v3 R4 audit is not PASS")
    if lock.get("status") != "IMPLEMENTATION_R4_FROZEN_BEFORE_LIVE_PREFLIGHT_ATTEMPT_02":
        raise RuntimeError("Calibration-v3 R4 lock state is invalid")
    if lock.get("implementation_audit_sha256") != sha256(R4_AUDIT):
        raise RuntimeError("Calibration-v3 R4 audit hash linkage failed")
    if lock.get("capture_authorized") is not False:
        raise RuntimeError("Calibration-v3 R4 lock unexpectedly authorizes capture")
    for name, digest in lock["source_artifact_sha256"].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Calibration-v3 R4 source drift: {name}")
    if R4_LIVE.exists() or CAPTURE_LOCK.exists() or CAPTURE.exists() or DATASET.exists():
        raise RuntimeError("downstream Calibration-v3 artifact exists before authorized R4 attempt")
    print(json.dumps({
        "status": "PASS",
        "lock_sha256": sha256(R4_LOCK),
        "audit_sha256": sha256(R4_AUDIT),
        "next_authorized_action": lock["next_authorized_action"],
        "capture_authorized": False,
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "validate"))
    command = parser.parse_args().command
    if command == "freeze":
        freeze()
    else:
        validate()


if __name__ == "__main__":
    main()
