#!/usr/bin/env python3
"""Append-only live preflight and single capture executor for Calibration-v6 R5."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import gazebo_calibration_v3_pipeline_r4 as helpers
import gazebo_calibration_v3_pipeline_r6 as parsers
import gazebo_calibration_v6_pipeline_r2 as r2
import gazebo_calibration_v6_pipeline_r3 as r3
import gazebo_calibration_v6_pilot_r5 as r5
import generate_gazebo_calibration_v6_design as v6


ROOT = v6.ROOT
PID = v6.CALIBRATION
OUTPUT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_r5"
CAPTURE = OUTPUT / "capture_attempt_01"
PREFLIGHT = OUTPUT / "PREFLIGHT_LIVE_ATTEMPT_01.json"
PREFLIGHT_FAILURE = OUTPUT / "PREFLIGHT_LIVE_ATTEMPT_01_FAILURE_LOCK.json"
CAPTURE_FAILURE = OUTPUT / "CAPTURE_ATTEMPT_01_FAILURE_LOCK.json"
AUTH = ROOT / "protocol/gazebo_calibration_v6_r5_capture_compatibility_lock.json"
EXECUTION_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V6_R5_CALIBRATION_EXECUTION_LOCK.json"
QC_CODE = ROOT / "protocol/gazebo_calibration_v6_r5_calibration_qc.py"
CAPTURE_CODE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture_r4.py"
SCENES, ANNOTATIONS, GATE, FAMILY_MANIFEST, SPLIT_MANIFEST = v6.paths(PID)
PILOT_DECISION = r5.R5_DECISION
DATASET = ROOT / "datasets/Gazebo_calibration_v6"
REQUIRED_NODES = ("/ros_gz_bridge", "/robot_state_publisher", "/controller_manager")
INFERENCE_OUTPUTS = (
    DATASET,
    ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/b0_predictions",
    ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/B0_PREDICTION_LOCK.json",
    ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/spatial_risk_v2_raw_predictions.jsonl",
    ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/SPATIAL_RISK_V2_RAW_PREDICTION_LOCK.json",
    ROOT / "protocol/GAZEBO_CALIBRATION_V6_FIT_INPUT_LOCK.json",
    ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/CALIBRATOR_FIT_ATTEMPT_STARTED.json",
    ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/affine_logit_calibrator.json",
    ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/FINAL_CALIBRATION_V6_DECISION.json",
)


def sha256(path: Path) -> str:
    return r5.sha256(path)


def read_json(path: Path) -> dict[str, Any]:
    return r5.read_json(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return r5.read_jsonl(path)


def write_new(path: Path, value: dict[str, Any]) -> None:
    r5.write_new(path, value)


def verify_hashes(values: dict[str, str]) -> None:
    for name, digest in values.items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"frozen artifact drift: {name}")


def calibration_design_checks() -> dict[str, bool]:
    rows = read_jsonl(FAMILY_MANIFEST)
    scenes = yaml.safe_load(SCENES.read_text(encoding="utf-8"))["scenes"]
    cells = Counter((row["state"], row["relation_variant"]) for row in rows)
    expected_order = [row["scene_id"] for row in sorted(rows, key=lambda row: row["capture_order"])]
    return {
        "family_count_128": len(rows) == 128,
        "unique_family_scene_seed_layout_signature": all(
            len({row[field] for row in rows}) == 128
            for field in ("family_id", "scene_id", "deterministic_seed", "layout_id", "layout_signature_sha256")
        ),
        "four_state_by_four_relation_by_eight": len(cells) == 16 and set(cells.values()) == {8},
        "capture_order_0_to_127": sorted(row["capture_order"] for row in rows) == list(range(128)),
        "scene_order_matches_manifest": [scene["scene_id"] for scene in scenes] == expected_order,
    }


def verify_authorization() -> dict[str, Any]:
    r5.validate_implementation()
    value = read_json(AUTH)
    if (
        value.get("status") != "LOCKED_BEFORE_CAPTURE_AND_INFERENCE"
        or value.get("authorization_status") != "ONE_CALIBRATION_128_CAPTURE_AUTHORIZED_AFTER_R5_PILOT_32_OF_32_QC_PASS"
        or value.get("protocol_id") != PID
        or value.get("expected_parent_families") != 128
        or value.get("capture_attempts_authorized") != 1
        or value.get("authorized_output") != str(CAPTURE.relative_to(ROOT))
        or not all(value.get("policies", {}).values())
    ):
        raise RuntimeError("Calibration-v6 R5 authorization invalid")
    verify_hashes(value["source_artifact_sha256"])
    decision = read_json(PILOT_DECISION)
    if decision.get("decision") != "PILOT_R5_32_OF_32_QC_PASS" or decision.get("accepted_family_count") != 32:
        raise RuntimeError("R5 pilot 32/32 accepted decision required")
    checks = calibration_design_checks()
    if not all(checks.values()):
        raise RuntimeError(f"Calibration-v6 128 design drift: {checks}")
    return value


def downstream_absent() -> bool:
    return not CAPTURE.exists() and all(not path.exists() for path in INFERENCE_OUTPUTS)


def freeze() -> None:
    if EXECUTION_LOCK.exists():
        raise FileExistsError("Calibration-v6 R5 execution already frozen")
    if PREFLIGHT.exists() or PREFLIGHT_FAILURE.exists() or CAPTURE.exists() or CAPTURE_FAILURE.exists():
        raise RuntimeError("Calibration-v6 R5 downstream artifact exists before execution freeze")
    authorization = verify_authorization()
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    design_checks = calibration_design_checks()
    checks = {
        "r5_pilot_32_of_32_qc_pass": read_json(PILOT_DECISION).get("accepted_family_count") == 32,
        "calibration_authorization_valid": True,
        "calibration_128_design_valid": all(design_checks.values()),
        "calibration_capture_absent": not CAPTURE.exists(),
        "dataset_inference_and_fit_absent": all(not path.exists() for path in INFERENCE_OUTPUTS),
        "test_iid_ood_and_robot_sealed": all(
            gate["sealed"][name] is True for name in ("gazebo_test_iid", "gazebo_test_ood", "robot")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Calibration-v6 R5 freeze checks failed: {checks}")
    sources = [
        Path(__file__).resolve(), QC_CODE, AUTH, PILOT_DECISION,
        r5.DATA_LOCK, r5.IMPLEMENTATION_LOCK, r3.R3_LOCK, r3.R3_PRECHECK,
        r3.WORLD, CAPTURE_CODE, SCENES, ANNOTATIONS, GATE,
        FAMILY_MANIFEST, SPLIT_MANIFEST,
    ]
    payload = {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "CALIBRATION_V6_R5_EXECUTION_FROZEN_BEFORE_LIVE_PREFLIGHT_AND_SINGLE_CAPTURE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Calibration-v6 128 live preflight, one capture attempt, all-or-nothing QC",
        "authorization_sha256": sha256(AUTH),
        "pilot_r5_decision_sha256": sha256(PILOT_DECISION),
        "design_checks": design_checks,
        "checks": checks,
        "source_artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "policies": {
            "live_preflight_attempts_authorized": 1,
            "capture_attempts_authorized": authorization["capture_attempts_authorized"],
            "capture_requires_live_preflight_pass": True,
            "all_or_nothing_128_qc": True,
            "partial_row_reuse_forbidden": True,
            "row_retry_repair_filter_or_cross_attempt_merge_forbidden": True,
            "no_materialization_inference_or_fit_before_128_qc_pass": True,
            "test_iid_ood_sealed": True,
            "robot_policy_sealed": True,
        },
        "calibration_families_captured": 0,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(EXECUTION_LOCK, payload)
    print(json.dumps({"status": payload["status"], "sha256": sha256(EXECUTION_LOCK)}, indent=2))


def validate_execution() -> dict[str, Any]:
    verify_authorization()
    value = read_json(EXECUTION_LOCK)
    if value.get("status") != "CALIBRATION_V6_R5_EXECUTION_FROZEN_BEFORE_LIVE_PREFLIGHT_AND_SINGLE_CAPTURE":
        raise RuntimeError("Calibration-v6 R5 execution lock status invalid")
    verify_hashes(value["source_artifact_sha256"])
    if not all(value["checks"].values()):
        raise RuntimeError("Calibration-v6 R5 execution checks drift")
    return value


def ros_environment() -> dict[str, str]:
    env = r2.ros_environment()
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["IGN_PARTITION"] = "ur3_roborefer_calibration_v6_r5_calibration_local"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def converged_node_probe() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    for index in range(15):
        final = r2.run_probe(["ros2", "node", "list"])
        present = {node: node in final["stdout"] for node in REQUIRED_NODES}
        history.append({
            "sample_index": index,
            "returncode": final["returncode"],
            "required_nodes_present": present,
            "stdout": final["stdout"],
            "stderr": final["stderr"],
        })
        if final["returncode"] == 0 and all(present.values()):
            break
        if index < 14:
            time.sleep(1.0)
    return final, history


def preflight_live() -> None:
    if PREFLIGHT.exists() or PREFLIGHT_FAILURE.exists():
        raise FileExistsError("Calibration-v6 R5 live preflight already attempted")
    validate_execution()
    helpers.run_probe = r2.run_probe
    parsers.install_parser_repair()
    nodes, node_history = converged_node_probe()
    probes = {
        "topics": r2.run_probe(["ros2", "topic", "list", "-t"]),
        "nodes": nodes,
        "services": r2.run_probe(["ros2", "service", "list", "-t"]),
        "actions": r2.run_probe(["ros2", "action", "list", "-t"]),
        "processes": r2.run_probe(["ps", "-eo", "comm=,args="]),
    }
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    scenes = yaml.safe_load(SCENES.read_text(encoding="utf-8"))
    required_topics = {
        name: f"{name} [{kind}]" in probes["topics"]["stdout"]
        for name, kind in helpers.REQUIRED_TOPICS.items()
    }
    observations: dict[str, Any] = {
        "ros_graph": probes,
        "node_discovery_history": node_history,
        "required_topics": required_topics,
    }
    samples: dict[str, Any] = {}
    sample_checks: dict[str, bool] = {}
    graph_ok = (
        all(item["returncode"] == 0 for name, item in probes.items() if name != "processes")
        and all(required_topics.values())
    )
    base_tf = r3.lookup_tf()
    wrist_tf = r2.run_probe(
        ["ros2", "run", "tf2_ros", "tf2_echo", "base_link", "camera_color_optical_frame"], timeout=8
    )
    joint, pose_ok, comparison = parsers.joint_state_probe(gate["camera"]["view_joint_pose"])
    observations.update({"base_tf": base_tf, "wrist_tf": wrist_tf, "joint_state": joint, "joint_comparison": comparison})
    if graph_ok:
        width, height = (int(value) for value in gate["camera"]["resolution"])
        for prefix, topic, frame in (
            ("wrist", "/wrist_camera", gate["camera"]["frame"]),
            ("base", "/top_table_camera", r3.CHILD),
        ):
            for label, suffix, encodings in (
                ("rgb", "/color/image_raw", {"rgb8", "bgr8"}),
                ("depth", "/depth/image_raw", {"32FC1"}),
            ):
                sample = helpers.image_probe(topic + suffix)
                samples[f"{prefix}_{label}"] = sample
                sample_checks[f"{prefix}_{label}_fresh_valid"] = helpers.image_matches(
                    sample, width=width, height=height, encodings=encodings, frame_id=frame
                )
            info = parsers.camera_info_probe(topic + "/color/camera_info")
            samples[f"{prefix}_intrinsics"] = info
            sample_checks[f"{prefix}_intrinsics_valid"] = helpers.camera_info_valid(
                info,
                width=width,
                height=height,
                frame_id=frame,
                expected_intrinsics=gate["camera"]["intrinsics"] if prefix == "wrist" else None,
            )
        labels = helpers.image_probe("/wrist_camera/evaluation_labels/labels_map")
        samples["labels"] = labels
        sample_checks["labels_fresh_valid"] = helpers.image_matches(
            labels, width=width, height=height, encodings={"rgb8"}, frame_id=gate["camera"]["frame"]
        )
        reset = helpers.reset_probe(scenes)
        observations["full_scene_reset"] = reset
        sample_checks["full_scene_reset_success"] = bool(reset.get("success"))
        post_depth = helpers.image_probe("/wrist_camera/depth/image_raw")
        samples["post_reset_depth"] = post_depth
        sample_checks["post_reset_metric_depth_valid"] = helpers.image_matches(
            post_depth, width=width, height=height, encodings={"32FC1"}, frame_id=gate["camera"]["frame"]
        )
    else:
        sample_checks["live_samples_and_reset"] = False
    observations["sensor_samples"] = samples
    process_lines = probes["processes"]["stdout"].splitlines()
    world_lines = [
        line for line in process_lines
        if line.split() and line.split()[0] in ("ruby", "ign", "gz", "gzserver") and r3.WORLD.name in line
    ]
    observations["locked_world_processes"] = world_lines
    checks = {
        "execution_lock_source_closure_validated": True,
        "r5_pilot_32_of_32_qc_pass": read_json(PILOT_DECISION).get("accepted_family_count") == 32,
        "calibration_authorization_valid": True,
        "calibration_128_design_valid": all(calibration_design_checks().values()),
        "ros_graph_and_required_topics": graph_ok,
        "sim_nodes_present": all(node in probes["nodes"]["stdout"] for node in REQUIRED_NODES),
        "bounded_node_discovery_followed": len(node_history) <= 15,
        "set_pose_service_present": helpers.SET_POSE_SERVICE in probes["services"]["stdout"],
        "trajectory_action_present": "/joint_trajectory_controller/follow_joint_trajectory" in probes["actions"]["stdout"],
        "locked_world_running": bool(world_lines),
        "r3_static_tf_publisher_running": any(
            "static_transform_publisher" in line and r3.CHILD in line for line in process_lines
        ),
        "r3_base_camera_tf_exact": r3.exact_tf_valid(base_tf, r3.camera_contract()),
        "wrist_camera_tf_resolved": "Translation:" in wrist_tf["stdout"] and "Rotation:" in wrist_tf["stdout"],
        "locked_camera_pose": pose_ok,
        "disk_free_over_2_gib": shutil.disk_usage(ROOT).free > 2 * 1024**3,
        "calibration_capture_absent": not CAPTURE.exists(),
        "dataset_inference_and_fit_absent": all(not path.exists() for path in INFERENCE_OUTPUTS),
        "scientific_hypothesis_not_tested": read_json(AUTH).get("scientific_hypothesis") == "NOT_TESTED",
        "test_iid_ood_and_robot_sealed": all(
            gate["sealed"][name] is True for name in ("gazebo_test_iid", "gazebo_test_ood", "robot")
        ),
        **sample_checks,
    }
    report = {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "attempt": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "observations": observations,
        "execution_lock_sha256": sha256(EXECUTION_LOCK),
        "calibration_authorization_sha256": sha256(AUTH),
        "pilot_r5_decision_sha256": sha256(PILOT_DECISION),
        "manifest_config_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (SCENES, ANNOTATIONS, GATE, FAMILY_MANIFEST, SPLIT_MANIFEST)
        },
        "r3_static_tf_lock_sha256": sha256(r3.R3_LOCK),
        "capture_authorized_by_this_artifact": False,
        "calibration_families_captured": 0,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(PREFLIGHT, report)
    if report["status"] != "PASS":
        write_new(PREFLIGHT_FAILURE, {
            "schema_version": 1,
            "protocol_id": PID,
            "status": "CALIBRATION_V6_R5_LIVE_PREFLIGHT_BLOCKED_NO_CAPTURE_NO_RETRY_UNDER_EXECUTION_REVISION",
            "classification": "INFRASTRUCTURE_PREFLIGHT_FAILURE_NOT_SCIENTIFIC_RESULT",
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
            "preflight_sha256": sha256(PREFLIGHT),
            "calibration_families_captured": 0,
            "model_inference_run": False,
            "calibrator_fit_call_count": 0,
            "scientific_hypothesis": "NOT_TESTED",
            "test_iid_ood_access": False,
            "robot_access": False,
        })
    print(json.dumps({"status": report["status"], "checks": checks, "artifact": str(PREFLIGHT)}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def capture() -> None:
    validate_execution()
    if read_json(PREFLIGHT).get("status") != "PASS":
        raise RuntimeError("Calibration-v6 R5 live preflight PASS required")
    if CAPTURE.exists() or CAPTURE_FAILURE.exists():
        raise FileExistsError("Calibration-v6 R5 single capture attempt already used")
    command = ["/usr/bin/python3", str(CAPTURE_CODE), "--ros-args", "-p", "use_sim_time:=true"]
    for key, value in (
        ("scene_config_file", SCENES),
        ("annotation_file", ANNOTATIONS),
        ("gate_config_file", GATE),
        ("pretrial_lock_file", AUTH),
        ("output_root", CAPTURE),
        ("settle_sec", "1.5"),
        ("sync_slop_sec", "0.02"),
        ("capture_timeout_sec", "45.0"),
    ):
        command += ["-p", f"{key}:={value}"]
    completed = subprocess.run(command, cwd=ROOT, env=ros_environment(), check=False)
    manifest = CAPTURE / "capture_manifest.json"
    index = CAPTURE / "input_manifest.jsonl"
    records = read_jsonl(index) if index.exists() else []
    passed = (
        completed.returncode == 0
        and manifest.is_file()
        and read_json(manifest).get("status") == "COMPLETE"
        and len(records) == 128
    )
    if not passed:
        write_new(CAPTURE_FAILURE, {
            "schema_version": 1,
            "protocol_id": PID,
            "status": "CALIBRATION_V6_R5_CAPTURE_ATTEMPT_01_FROZEN_INCOMPLETE_NO_RETRY",
            "classification": "INFRASTRUCTURE_CAPTURE_FAILURE_NOT_SCIENTIFIC_RESULT",
            "expected_family_count": 128,
            "captured_record_count": len(records),
            "process_returncode": completed.returncode,
            "source_artifact_sha256": tree_hashes(CAPTURE) if CAPTURE.exists() else {},
            "partial_rows_quarantined": True,
            "model_inference_run": False,
            "calibrator_fit_call_count": 0,
            "scientific_hypothesis": "NOT_TESTED",
            "test_iid_ood_access": False,
            "robot_access": False,
        })
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "validate", "preflight-live", "capture"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "validate":
        validate_execution()
        print("PASS: Calibration-v6 R5 execution lock")
    elif args.command == "preflight-live":
        preflight_live()
    else:
        capture()


if __name__ == "__main__":
    main()
