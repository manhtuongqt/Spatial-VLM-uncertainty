#!/usr/bin/env python3
"""Append-only ROS graph discovery amendment for Calibration-v6 pilot R5.

Attempt 05 sampled the ROS graph before ``/controller_manager`` had converged,
although its controller action, all sensors, TF, robot pose and scene reset had
already passed.  This amendment preserves attempt 05 and changes only the node
discovery observation to a bounded wait before writing attempt 06.
"""
from __future__ import annotations

import argparse
import json
import shutil
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


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_05 = r5.PREFLIGHT
FAILURE_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V6_PILOT_R5_PREFLIGHT_ATTEMPT_05_FAILURE_LOCK.json"
AMENDMENT_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V6_PILOT_R5_PREFLIGHT_AMENDMENT_A1_LOCK.json"
ATTEMPT_06 = r5.OUTPUT / "PREFLIGHT_LIVE_ATTEMPT_06.json"
REQUIRED_NODES = ("/ros_gz_bridge", "/robot_state_publisher", "/controller_manager")


def verify_attempt_05_failure() -> dict[str, Any]:
    r5.validate_implementation()
    report = r5.read_json(ATTEMPT_05)
    checks = report.get("checks", {})
    failed = sorted(name for name, passed in checks.items() if not passed)
    nodes = report.get("observations", {}).get("ros_graph", {}).get("nodes", {}).get("stdout", "")
    if report.get("status") != "BLOCKED" or failed != ["sim_nodes_present"]:
        raise RuntimeError("attempt 05 is not the isolated ROS-node discovery failure")
    if "/ros_gz_bridge" not in nodes or "/robot_state_publisher" not in nodes or "/controller_manager" in nodes:
        raise RuntimeError("attempt 05 node snapshot does not match the preregistered A1 repair")
    if r5.AUTH.exists() or r5.CAPTURE.exists():
        raise RuntimeError("R5 capture authorization or attempt already exists")
    return report


def freeze() -> None:
    if FAILURE_LOCK.exists() or AMENDMENT_LOCK.exists():
        raise FileExistsError("R5 preflight amendment A1 is already frozen")
    if ATTEMPT_06.exists() or r5.AUTH.exists() or r5.CAPTURE.exists():
        raise RuntimeError("R5 A1 downstream artifact exists before freeze")
    report = verify_attempt_05_failure()
    common = {
        "schema_version": 1,
        "protocol_id": r5.PID,
        "pilot_accepted_families": 0,
        "calibration_families_captured": 0,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    failure = {
        **common,
        "status": "LIVE_PREFLIGHT_ATTEMPT_05_FROZEN_BLOCKED_NO_CAPTURE",
        "classification": "TRANSIENT_ROS_GRAPH_DISCOVERY_NOT_SENSOR_GEOMETRY_OR_SCIENTIFIC_FAILURE",
        "blocked_checks": ["sim_nodes_present"],
        "attempt_05_sha256": r5.sha256(ATTEMPT_05),
        "observed_result": (
            "The one-shot node snapshot omitted /controller_manager while its trajectory action, "
            "all required topics, fresh RGB/depth/labels, exact R3 TF, locked robot pose and full "
            "scene reset passed. No capture authorization or capture was created."
        ),
        "partial_capture_reuse": False,
    }
    r5.write_new(FAILURE_LOCK, failure)
    sources = [
        Path(__file__).resolve(), r5.DATA_LOCK, r5.IMPLEMENTATION_LOCK,
        ATTEMPT_05, FAILURE_LOCK, r5.AUDIT, r5.REGISTRY,
        r5.FAMILY_MANIFEST, r5.SPLIT_MANIFEST, r5.SCENES,
        r5.ANNOTATIONS, r5.GATE, r5.CAPTURE_CODE, r5.QC_CODE,
    ]
    amendment = {
        **common,
        "status": "PILOT_R5_PREFLIGHT_AMENDMENT_A1_FROZEN_BEFORE_ATTEMPT_06",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "bounded ROS node-discovery convergence wait only",
        "parent_attempt": "PREFLIGHT_LIVE_ATTEMPT_05.json",
        "new_attempt": "PREFLIGHT_LIVE_ATTEMPT_06.json",
        "checks": {
            "attempt_05_preserved_blocked": report.get("status") == "BLOCKED",
            "only_failed_check_was_sim_nodes_present": sorted(
                name for name, passed in report["checks"].items() if not passed
            ) == ["sim_nodes_present"],
            "attempt_05_sensor_tf_pose_reset_checks_passed": all(
                report["checks"][name]
                for name in (
                    "rgb_fresh_valid", "depth_fresh_valid", "labels_fresh_valid",
                    "r3_base_camera_tf_exact", "wrist_camera_tf_resolved",
                    "locked_camera_pose", "full_scene_reset_success",
                )
            ),
            "no_capture_authorization_or_capture": not r5.AUTH.exists() and not r5.CAPTURE.exists(),
            "data_design_and_qc_thresholds_unchanged": True,
            "one_pilot_capture_attempt_policy_unchanged": True,
            "test_robot_sealed": True,
        },
        "bounded_discovery_policy": {
            "required_nodes": list(REQUIRED_NODES),
            "maximum_samples": 15,
            "interval_seconds": 1.0,
            "pass_rule": "all required nodes must coexist in one successful ros2 node list sample",
        },
        "unchanged": [
            "B0, spatial-risk v2 and the 12 ordered features",
            "pilot R5 family, scene, seed, layout, signature and deterministic capture order",
            "Calibration-v6 128 population, calibrator, optimizer and scientific gate",
            "R4 sensor capture implementation and R3 static TF",
            "duplicate/near-duplicate and all other QC thresholds",
            "single all-or-nothing R5 pilot capture attempt",
            "Test-IID/OOD and robot-policy sealing",
        ],
        "capture_authorized": False,
        "source_artifact_sha256": {str(path.relative_to(ROOT)): r5.sha256(path) for path in sources},
    }
    if not all(amendment["checks"].values()):
        raise RuntimeError(f"R5 A1 freeze checks failed: {amendment['checks']}")
    r5.write_new(AMENDMENT_LOCK, amendment)
    print(json.dumps({
        "status": amendment["status"],
        "failure_lock_sha256": r5.sha256(FAILURE_LOCK),
        "amendment_lock_sha256": r5.sha256(AMENDMENT_LOCK),
    }, indent=2))


def validate_amendment() -> dict[str, Any]:
    verify_attempt_05_failure()
    value = r5.read_json(AMENDMENT_LOCK)
    if value.get("status") != "PILOT_R5_PREFLIGHT_AMENDMENT_A1_FROZEN_BEFORE_ATTEMPT_06":
        raise RuntimeError("R5 preflight amendment A1 status invalid")
    r5.verify_hash_map(value["source_artifact_sha256"])
    if not all(value["checks"].values()):
        raise RuntimeError("R5 preflight amendment A1 checks drift")
    return value


def converged_node_probe() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    for sample_index in range(15):
        final = r2.run_probe(["ros2", "node", "list"])
        present = {name: name in final["stdout"] for name in REQUIRED_NODES}
        history.append({
            "sample_index": sample_index,
            "elapsed_bound_seconds": float(sample_index),
            "returncode": final["returncode"],
            "required_nodes_present": present,
            "stdout": final["stdout"],
            "stderr": final["stderr"],
        })
        if final["returncode"] == 0 and all(present.values()):
            return final, history
        if sample_index < 14:
            time.sleep(1.0)
    return final, history


def preflight_live() -> None:
    if ATTEMPT_06.exists():
        raise FileExistsError("R5 live preflight attempt 06 already attempted")
    amendment = validate_amendment()
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
    required_topics = {
        name: f"{name} [{kind}]" in probes["topics"]["stdout"]
        for name, kind in helpers.REQUIRED_TOPICS.items()
    }
    gate = yaml.safe_load(r5.GATE.read_text(encoding="utf-8"))
    scenes = yaml.safe_load(r5.SCENES.read_text(encoding="utf-8"))
    base_tf = r3.lookup_tf()
    wrist_tf = r2.run_probe(
        ["ros2", "run", "tf2_ros", "tf2_echo", "base_link", "camera_color_optical_frame"],
        timeout=8,
    )
    joint, pose_ok, comparison = parsers.joint_state_probe(gate["camera"]["view_joint_pose"])
    observations: dict[str, Any] = {
        "ros_graph": probes,
        "node_discovery_history": node_history,
        "required_topics": required_topics,
        "base_tf": base_tf,
        "wrist_tf": wrist_tf,
        "joint_state": joint,
        "joint_comparison": comparison,
    }
    graph_ok = (
        all(item["returncode"] == 0 for name, item in probes.items() if name != "processes")
        and all(required_topics.values())
    )
    sample_checks: dict[str, bool] = {}
    samples: dict[str, Any] = {}
    if graph_ok:
        width, height = (int(value) for value in gate["camera"]["resolution"])
        for label, topic, encodings in (
            ("rgb", "/wrist_camera/color/image_raw", {"rgb8", "bgr8"}),
            ("depth", "/wrist_camera/depth/image_raw", {"32FC1"}),
            ("labels", "/wrist_camera/evaluation_labels/labels_map", {"rgb8"}),
        ):
            sample = helpers.image_probe(topic)
            samples[label] = sample
            sample_checks[label + "_fresh_valid"] = helpers.image_matches(
                sample, width=width, height=height, encodings=encodings,
                frame_id=gate["camera"]["frame"],
            )
        reset = helpers.reset_probe(scenes)
        observations["full_scene_reset"] = reset
        sample_checks["full_scene_reset_success"] = bool(reset.get("success"))
    else:
        sample_checks["live_samples_and_reset"] = False
    observations["sensor_samples"] = samples
    process_text = probes["processes"]["stdout"]
    checks = {
        "r5_implementation_unchanged": True,
        "preflight_amendment_a1_validated": True,
        "attempt_05_preserved_blocked": r5.read_json(ATTEMPT_05).get("status") == "BLOCKED",
        "r4_successful_32_scene_sensor_capture_bound": r5.read_json(
            r5.R4_ROOT / "capture_attempt_01/capture_manifest.json"
        ).get("status") == "COMPLETE",
        "r4_qc_rejection_was_duplicate_not_sensor_or_geometry": r5.read_json(
            r5.R4_ROOT / "GEOMETRY_QC.json"
        ).get("passed_scene_count") == 32,
        "ros_graph_and_required_topics": graph_ok,
        "sim_nodes_present": all(token in probes["nodes"]["stdout"] for token in REQUIRED_NODES),
        "bounded_node_discovery_policy_followed": len(node_history) <= amendment["bounded_discovery_policy"]["maximum_samples"],
        "set_pose_service_present": helpers.SET_POSE_SERVICE in probes["services"]["stdout"],
        "trajectory_action_present": "/joint_trajectory_controller/follow_joint_trajectory" in probes["actions"]["stdout"],
        "locked_world_running": any(r3.WORLD.name in line and "ign gazebo" in line for line in process_text.splitlines()),
        "r3_static_tf_publisher_running": any(
            "static_transform_publisher" in line and r3.CHILD in line for line in process_text.splitlines()
        ),
        "r3_base_camera_tf_exact": r3.exact_tf_valid(base_tf, r3.camera_contract()),
        "wrist_camera_tf_resolved": "Translation:" in wrist_tf["stdout"] and "Rotation:" in wrist_tf["stdout"],
        "locked_camera_pose": pose_ok,
        "disk_free_over_2_gib": shutil.disk_usage(ROOT).free > 2 * 1024**3,
        "r5_capture_and_authorization_absent": not r5.CAPTURE.exists() and not r5.AUTH.exists(),
        "calibration_capture_dataset_inference_absent": not r5.CALIBRATION_OUTPUT.exists() and not r2.DATASET_ROOT.exists(),
        "test_iid_ood_and_robot_sealed": all(
            gate["sealed"][name] is True for name in ("gazebo_test_iid", "gazebo_test_ood", "robot")
        ),
        **sample_checks,
    }
    report = {
        "schema_version": 1,
        "protocol_id": r5.PID,
        "attempt": 6,
        "implementation_revision": "r5_preflight_a1_bounded_ros_node_discovery_only",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "observations": observations,
        "implementation_lock_sha256": r5.sha256(r5.IMPLEMENTATION_LOCK),
        "preflight_amendment_lock_sha256": r5.sha256(AMENDMENT_LOCK),
        "capture_authorized": False,
        "pilot_accepted_families": 0,
        "calibration_families_captured": 0,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    r5.write_new(ATTEMPT_06, report)
    print(json.dumps({"status": report["status"], "checks": checks, "artifact": str(ATTEMPT_06)}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


def authorize_pilot() -> None:
    validate_amendment()
    if r5.AUTH.exists() or r5.CAPTURE.exists():
        raise RuntimeError("R5 pilot authorization or attempt already exists")
    if r5.read_json(ATTEMPT_06).get("status") != "PASS":
        raise RuntimeError("R5 live preflight attempt 06 PASS required")
    rows = r5.read_jsonl(r5.FAMILY_MANIFEST)
    cells = Counter((row["state"], row["relation_variant"]) for row in rows)
    if len(rows) != 32 or len(cells) != 16 or set(cells.values()) != {2}:
        raise RuntimeError("R5 pilot population drift")
    sources = r5.authorization_sources(r5.PID)
    for path in (Path(__file__).resolve(), FAILURE_LOCK, AMENDMENT_LOCK, ATTEMPT_06):
        sources[str(path.relative_to(ROOT))] = r5.sha256(path)
    payload = {
        "schema_version": 1,
        "protocol_id": r5.PID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "authorization_status": "ONE_R5_PILOT_CAPTURE_AUTHORIZED_AFTER_APPEND_ONLY_PREFLIGHT_ATTEMPT_06_PASS",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "parent_contract_lock_sha256": r5.sha256(r5.DATA_LOCK),
        "implementation_lock_sha256": r5.sha256(r5.IMPLEMENTATION_LOCK),
        "preflight_amendment_lock_sha256": r5.sha256(AMENDMENT_LOCK),
        "source_artifact_sha256": sources,
        "expected_parent_families": 32,
        "population": "4 state x 4 relation x 2",
        "capture_attempts_authorized": 1,
        "authorized_output": str(r5.CAPTURE.relative_to(ROOT)),
        "model_inventory_sha256": "NO_MODEL_INFERENCE_DURING_R5_PILOT",
        "policies": {
            "single_complete_capture_attempt_only": True,
            "partial_attempt_reuse_forbidden": True,
            "row_filter_repair_or_replacement_forbidden": True,
            "cross_attempt_merge_forbidden": True,
            "r3_r4_row_or_image_reuse_forbidden": True,
            "duplicate_threshold_relaxation_forbidden": True,
            "no_model_inference": True,
            "no_materialization": True,
            "no_calibrator_fit": True,
            "test_iid_ood_sealed": True,
            "robot_policy_sealed": True,
        },
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    r5.write_new(r5.AUTH, payload)
    print(json.dumps({"status": payload["authorization_status"], "sha256": r5.sha256(r5.AUTH)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "validate", "preflight-live", "authorize-pilot"))
    command = parser.parse_args().command
    if command == "freeze":
        freeze()
    elif command == "validate":
        validate_amendment()
        print("PASS: Calibration-v6 pilot R5 preflight amendment A1")
    elif command == "preflight-live":
        preflight_live()
    else:
        authorize_pilot()


if __name__ == "__main__":
    main()
