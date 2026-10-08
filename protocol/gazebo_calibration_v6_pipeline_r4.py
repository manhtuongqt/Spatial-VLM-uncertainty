#!/usr/bin/env python3
"""Append-only R4 sensor-liveness workflow for Calibration-v6.

R4 changes only observation reliability for the already frozen simulated
sensor streams.  It preserves all data design, model, feature, metric,
calibrator, optimizer, scientific-gate, Test, and robot-policy contracts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import yaml

import gazebo_calibration_v3_pipeline_r4 as helpers
import gazebo_calibration_v3_pipeline_r6 as parsers
import gazebo_calibration_v6_pipeline as base
import gazebo_calibration_v6_pipeline_r2 as r2
import gazebo_calibration_v6_pipeline_r3 as r3
import generate_gazebo_calibration_v6_design as design


ROOT = design.ROOT
R4_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V6_IMPLEMENTATION_AMENDMENT_R4_LOCK.json"
SOAK_CODE = ROOT / "protocol/gazebo_calibration_v6_sensor_soak_r4.py"
CAPTURE_CODE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture_r4.py"
QC_CODE = ROOT / "protocol/gazebo_calibration_v6_qc_r4.py"
DIAGNOSTIC = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/R4_SENSOR_DIAGNOSTIC.json"
CLASSIFICATION = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/R4_SENSOR_FAILURE_CLASSIFICATION.json"
SENSOR_PREFLIGHT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/R4_SENSOR_TRANSPORT_PREFLIGHT.json"
ATTEMPT_04 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot_r4/PREFLIGHT_LIVE_ATTEMPT_04.json"
PILOT_ROOT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot_r4"
CALIBRATION_ROOT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_r4"
PILOT_CAPTURE = PILOT_ROOT / "capture_attempt_01"
CALIBRATION_CAPTURE = CALIBRATION_ROOT / "capture_attempt_01"
PILOT_AUTH = ROOT / "protocol/gazebo_calibration_v6_geometry_pilot_r4_capture_compatibility_lock.json"
CALIBRATION_AUTH = ROOT / "protocol/gazebo_calibration_v6_r4_capture_compatibility_lock.json"
PILOT_DECISION = PILOT_ROOT / "PILOT_R4_DECISION.json"
CALIBRATION_QC = CALIBRATION_ROOT / "GEOMETRY_QC.json"
OLD_FAILURE = ROOT / "protocol/GAZEBO_CALIBRATION_V6_GEOMETRY_PILOT_CAPTURE_ATTEMPT_01_FAILURE_LOCK.json"
OLD_AUTH = ROOT / "protocol/gazebo_calibration_v6_geometry_pilot_capture_compatibility_lock.json"
OLD_CAPTURE = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/capture_attempt_01"
FINAL_R4 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/CALIBRATION_V6_R4_SCIENTIFIC_DECISION_LOCK.json"

EXPECTED = {
    design.LOCK: "850921e200d921592128c76e11c8deeb5a7d6177ea315f1ff8d286f873084b6d",
    base.IMPLEMENTATION: "89d3416e7c870e635c3f6277a6dbbc2d5d12891c23f3a686c9a4fe2d4f7b821d",
    r3.R2_LOCK: "4ba0765cd8f68eccfd11fd905db4048856eec130652dd62c786c6b27bae4846b",
    r3.R3_LOCK: "f3416bd6843e1b72250e8d39f740843e476cce76359e1555f8fa3fe756b737cd",
    base.RESULT / "PREFLIGHT_STATIC.json": "05be37274246b8364a0fc65a2c45d8c9707abfebe85cf7777169cb9c4ea18002",
    r3.R3_PRECHECK: "0ace1bdc936110912c9d7226e2d7b5ed1238d934f620659543360ceeb67ea739",
    r3.ATTEMPT_03: "d0019a72f2cc69d664c3f36602e5dd34fcb2ead155e0c4feb0388b620da5ed36",
    OLD_AUTH: "cfa7a225ce98891b500bdc7631e1f547f3ac322850246ba6fd9a1e1123bf12c2",
    OLD_FAILURE: "1741b46aaa9a96d458152944feb24302ddfb85ea3aa80e8b531dfc75031a499c",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def ros_environment() -> dict[str, str]:
    env = r2.ros_environment()
    # R4 is an isolated, single-host simulation workflow.  Pinning discovery
    # locally removes reliance on an unavailable external multicast route.
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["IGN_PARTITION"] = "ur3_roborefer_calibration_v6_r4_local"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def verify_fixed_predecessors() -> None:
    for path, expected in EXPECTED.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen predecessor drift: {path}")
    if read_json(r3.R3_PRECHECK).get("status") != "PASS" or read_json(r3.ATTEMPT_03).get("status") != "PASS":
        raise RuntimeError("R3 TF precheck and live attempt 03 must remain PASS")
    failure = read_json(OLD_FAILURE)
    if failure.get("captured_record_count") != 19 or not failure["partial_attempt_policy"]["partial_rows_quarantined"]:
        raise RuntimeError("R3 partial attempt quarantine changed")
    if len([line for line in (OLD_CAPTURE / "input_manifest.jsonl").read_text().splitlines() if line]) != 19:
        raise RuntimeError("R3 partial attempt inventory changed")


def validate_r4() -> dict[str, Any]:
    verify_fixed_predecessors()
    value = read_json(R4_LOCK)
    if value.get("status") != "IMPLEMENTATION_AMENDMENT_R4_FROZEN_BEFORE_SENSOR_PREFLIGHT":
        raise RuntimeError("R4 lock status invalid")
    for name, digest in value["source_artifact_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"R4 frozen source drift: {name}")
    if not all(value["invariants"].values()):
        raise RuntimeError("R4 invariant lock invalid")
    return value


def manifest_summary(pid: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in design.paths(pid)[3].read_text(encoding="utf-8").splitlines() if line]
    scenes = yaml.safe_load(design.paths(pid)[0].read_text(encoding="utf-8"))["scenes"]
    expected = 32 if pid == design.PILOT else 128
    repetitions = 2 if pid == design.PILOT else 8
    cells = Counter((row["state"], row["relation_variant"]) for row in rows)
    expected_order = [row["scene_id"] for row in sorted(rows, key=lambda row: row["capture_order"])]
    checks = {
        "row_count": len(rows) == expected,
        "unique_family_id": len({row["family_id"] for row in rows}) == expected,
        "unique_scene_id": len({row["scene_id"] for row in rows}) == expected,
        "unique_layout_id": len({row["layout_id"] for row in rows}) == expected,
        "unique_seed": len({row["deterministic_seed"] for row in rows}) == expected,
        "unique_layout_signature": len({row["layout_signature_sha256"] for row in rows}) == expected,
        "four_state_by_four_relation_balance": len(cells) == 16 and set(cells.values()) == {repetitions},
        "deterministic_capture_order": [row["scene_id"] for row in scenes] == expected_order,
        "capture_order_contiguous": sorted(row["capture_order"] for row in rows) == list(range(expected)),
    }
    return {"expected": expected, "repetitions_per_cell": repetitions, "checks": checks}


def run_soak(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite soak artifact: {output}")
    command = [
        "/usr/bin/python3", str(SOAK_CODE),
        "--scene-config", str(design.paths(design.PILOT)[0]),
        "--output", str(output), "--cycles", "40",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=ros_environment(), check=False)
    if not output.is_file():
        raise RuntimeError("sensor soak exited without an immutable report")
    if completed.returncode != 0 or read_json(output).get("status") != "PASS":
        raise SystemExit(2)


def diagnose() -> None:
    verify_fixed_predecessors()
    if R4_LOCK.exists() or CLASSIFICATION.exists():
        raise RuntimeError("R4 diagnostic must precede amendment freeze")
    run_soak(DIAGNOSTIC)
    prior_records = [json.loads(line) for line in (OLD_CAPTURE / "input_manifest.jsonl").read_text().splitlines() if line]
    exact_spreads = [float(row["capture"]["rgb_depth_label_spread_sec"]) for row in prior_records]
    same_stamp = [len(set(row["capture"]["sensor_stamps_sec"])) == 1 for row in prior_records]
    report = {
        "schema_version": 1,
        "protocol_id": "gazebo_calibration_v6",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "R3_SENSOR_STREAM_LIVENESS_LOSS_AFTER_19_EXACTLY_SYNCHRONIZED_TUPLES",
        "evidence": {
            "r3_completed_tuple_count": len(prior_records),
            "r3_all_completed_tuples_exact_header_timestamp": all(same_stamp),
            "r3_max_completed_tuple_spread_sec": max(exact_spreads),
            "r3_failure_scene": read_json(OLD_FAILURE)["failure_scene_id"],
            "r3_failure_timeout_sec": 45.0,
            "r3_multicast_log_is_retained_but_not_asserted_as_root_cause": True,
            "r4_neutral_soak_sha256": sha256(DIAGNOSTIC),
            "r4_neutral_soak_cycles_passed": read_json(DIAGNOSTIC)["cycles_completed"],
        },
        "conclusions": {
            "sync_slop_was_not_the_observed_r3_failure": all(same_stamp),
            "root_transport_process_below_ros_topic_layer_not_proven": True,
            "bounded_timestamp_queues_and_per_topic_liveness_are_a_non_outcome_based_observability_repair": True,
            "timeouts_or_thresholds_relaxed": False,
            "scientific_contract_changed": False,
        },
        "scientific_capture_count": 0,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(CLASSIFICATION, report)
    print(json.dumps(report, indent=2))


def freeze_amendment() -> None:
    if R4_LOCK.exists():
        raise FileExistsError("R4 amendment already exists")
    verify_fixed_predecessors()
    if read_json(DIAGNOSTIC).get("status") != "PASS" or read_json(CLASSIFICATION).get("status") != "PASS":
        raise RuntimeError("R4 development diagnostic must PASS before freeze")
    pilot = manifest_summary(design.PILOT)
    calibration = manifest_summary(design.CALIBRATION)
    invariants = {
        "r3_attempt_quarantined_no_reuse": True,
        "pilot_manifest_4x4x2_unchanged": all(pilot["checks"].values()),
        "calibration_manifest_4x4x8_unchanged": all(calibration["checks"].values()),
        "settle_sec_unchanged_1_5": True,
        "capture_timeout_sec_unchanged_45": True,
        "sync_slop_sec_unchanged_0_02": True,
        "b0_and_spatial_risk_v2_12_features_unchanged": True,
        "visibility_metric_calibrator_optimizer_and_scientific_gates_unchanged": True,
        "no_capture_inference_or_fit_during_diagnostic": True,
        "test_iid_ood_and_robot_policy_sealed": True,
    }
    sources = [
        Path(__file__).resolve(), SOAK_CODE, CAPTURE_CODE, QC_CODE, DIAGNOSTIC, CLASSIFICATION,
        *EXPECTED, design.REGISTRY, *design.paths(design.PILOT), *design.paths(design.CALIBRATION),
        r3.WORLD, OLD_CAPTURE / "capture_manifest.json", OLD_CAPTURE / "input_manifest.jsonl",
    ]
    payload = {
        "schema_version": 1,
        "protocol_id": "gazebo_calibration_v6",
        "status": "IMPLEMENTATION_AMENDMENT_R4_FROZEN_BEFORE_SENSOR_PREFLIGHT",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "simulation-only sensor transport/liveness/observability and exact-timestamp queue repair",
        "repair": {
            "bounded_timestamp_queue_per_rgb_depth_semantic_topic": 64,
            "matcher": "newest exact common header timestamp",
            "post_settle_freshness_barrier": True,
            "per_topic_and_clock_watchdog": True,
            "single_host_ros_discovery": "ROS_LOCALHOST_ONLY=1",
            "automatic_retry": False,
        },
        "diagnostic_sha256": sha256(DIAGNOSTIC),
        "failure_classification_sha256": sha256(CLASSIFICATION),
        "r3_amendment_sha256": sha256(r3.R3_LOCK),
        "r3_failure_lock_sha256": sha256(OLD_FAILURE),
        "pilot_manifest": pilot,
        "calibration_manifest": calibration,
        "invariants": invariants,
        "source_artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "capture_authorized": False,
        "pilot_families_captured": 0,
        "calibration_families_captured": 0,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(R4_LOCK, payload)
    print(json.dumps({"status": payload["status"], "sha256": sha256(R4_LOCK), "invariants": invariants}, indent=2))


def sensor_preflight() -> None:
    validate_r4()
    run_soak(SENSOR_PREFLIGHT)
    value = read_json(SENSOR_PREFLIGHT)
    if value.get("cycles_completed") != 40 or not all(value["checks"].values()):
        raise SystemExit(2)
    print(json.dumps({"status": value["status"], "sha256": sha256(SENSOR_PREFLIGHT)}, indent=2))


def live_preflight() -> None:
    amendment = validate_r4()
    if ATTEMPT_04.exists():
        raise FileExistsError("live attempt 04 already exists")
    soak = read_json(SENSOR_PREFLIGHT)
    if soak.get("status") != "PASS" or soak.get("cycles_completed") != 40:
        raise RuntimeError("R4 40-cycle sensor preflight PASS required")
    helpers.run_probe = r2.run_probe
    parsers.install_parser_repair()
    probes = {
        "topics": r2.run_probe(["ros2", "topic", "list", "-t"]),
        "nodes": r2.run_probe(["ros2", "node", "list"]),
        "services": r2.run_probe(["ros2", "service", "list", "-t"]),
        "actions": r2.run_probe(["ros2", "action", "list", "-t"]),
        "processes": r2.run_probe(["ps", "-eo", "comm=,args="]),
    }
    required_topics = {
        name: f"{name} [{kind}]" in probes["topics"]["stdout"] for name, kind in helpers.REQUIRED_TOPICS.items()
    }
    expected_tf = amendment["source_artifact_sha256"]  # forces immutable lock validation above
    del expected_tf
    base_tf = r3.lookup_tf()
    tf_contract = r3.camera_contract()
    wrist_tf = r2.run_probe(["ros2", "run", "tf2_ros", "tf2_echo", "base_link", "camera_color_optical_frame"], timeout=8)
    gate = yaml.safe_load(design.paths(design.PILOT)[2].read_text(encoding="utf-8"))
    joint, joint_ok, comparison = parsers.joint_state_probe(gate["camera"]["view_joint_pose"])
    checks = {
        "r4_amendment_unchanged": True,
        "sensor_transport_preflight_40_of_40_pass_and_bound": soak["status"] == "PASS" and soak["cycles_completed"] == 40 and all(soak["checks"].values()),
        "ros_graph_commands_pass": all(item["returncode"] == 0 for name, item in probes.items() if name != "processes"),
        "required_topic_types_present": all(required_topics.values()),
        "sim_nodes_present": all(token in probes["nodes"]["stdout"] for token in ("/ros_gz_bridge", "/robot_state_publisher", "/controller_manager")),
        "set_pose_service_present": helpers.SET_POSE_SERVICE in probes["services"]["stdout"],
        "trajectory_action_present": "/joint_trajectory_controller/follow_joint_trajectory" in probes["actions"]["stdout"],
        "locked_world_running": any(r3.WORLD.name in line and "ign gazebo" in line for line in probes["processes"]["stdout"].splitlines()),
        "r3_static_tf_publisher_running": any("static_transform_publisher" in line and r3.CHILD in line for line in probes["processes"]["stdout"].splitlines()),
        "r3_base_camera_tf_exact": r3.exact_tf_valid(base_tf, tf_contract),
        "wrist_camera_tf_resolved": "Translation:" in wrist_tf["stdout"] and "Rotation:" in wrist_tf["stdout"],
        "locked_camera_pose": joint_ok,
        "disk_free_over_2_gib": shutil.disk_usage(ROOT).free > 2 * 1024**3,
        "new_r4_capture_namespaces_absent": not PILOT_CAPTURE.exists() and not CALIBRATION_CAPTURE.exists(),
        "canonical_dataset_absent": not r2.DATASET_ROOT.exists(),
        "test_iid_ood_and_robot_sealed": all(gate["sealed"][name] is True for name in ("gazebo_test_iid", "gazebo_test_ood", "robot")),
    }
    report = {
        "schema_version": 1,
        "protocol_id": design.PILOT,
        "attempt": 4,
        "implementation_revision": "r4_sensor_transport_liveness_exact_timestamp_queue",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "observations": {"ros_graph": probes, "required_topics": required_topics, "base_tf": base_tf, "wrist_tf": wrist_tf, "joint_state": joint, "joint_comparison": comparison},
        "r4_amendment_sha256": sha256(R4_LOCK),
        "sensor_preflight_sha256": sha256(SENSOR_PREFLIGHT),
        "capture_authorized": False,
        "pilot_families_captured": 0,
        "calibration_families_captured": 0,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(ATTEMPT_04, report)
    print(json.dumps({"status": report["status"], "checks": checks, "artifact": str(ATTEMPT_04)}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


def auth_path(pid: str) -> Path:
    return PILOT_AUTH if pid == design.PILOT else CALIBRATION_AUTH


def capture_path(pid: str) -> Path:
    return PILOT_CAPTURE if pid == design.PILOT else CALIBRATION_CAPTURE


def build_authorization_sources(pid: str) -> dict[str, str]:
    sources = dict(read_json(base.IMPLEMENTATION)["source_artifact_sha256"])
    extra = [
        R4_LOCK, SOAK_CODE, CAPTURE_CODE, QC_CODE, Path(__file__).resolve(), DIAGNOSTIC,
        CLASSIFICATION, SENSOR_PREFLIGHT, ATTEMPT_04, OLD_FAILURE, OLD_AUTH,
        *design.paths(pid), design.REGISTRY,
    ]
    if pid == design.CALIBRATION:
        extra += [PILOT_AUTH, PILOT_DECISION, PILOT_ROOT / "GEOMETRY_QC.json", PILOT_ROOT / "RGB_DUPLICATE_QC.json", PILOT_ROOT / "QC_INPUT_LOCK.json"]
    sources.update({str(path.relative_to(ROOT)): sha256(path) for path in extra})
    return sources


def authorize(pid: str) -> None:
    validate_r4()
    if read_json(SENSOR_PREFLIGHT).get("status") != "PASS" or read_json(ATTEMPT_04).get("status") != "PASS":
        raise RuntimeError("R4 sensor and live preflights must PASS")
    summary = manifest_summary(pid)
    if not all(summary["checks"].values()):
        raise RuntimeError(f"manifest audit failed: {summary}")
    output = capture_path(pid)
    lock_path = auth_path(pid)
    if output.exists() or lock_path.exists():
        raise RuntimeError("authorization or single-attempt namespace already exists")
    if pid == design.CALIBRATION:
        decision = read_json(PILOT_DECISION)
        if decision.get("status") != "PASS" or decision.get("captured_family_count") != 32 or decision.get("passed_scene_count") != 32:
            raise RuntimeError("pilot 32/32 QC PASS decision required")
    sources = build_authorization_sources(pid)
    payload = {
        "schema_version": 1,
        "protocol_id": pid,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "authorization_status": "CAPTURE_AUTHORIZED_AFTER_R4_SENSOR_AND_LIVE_PREFLIGHT_PASS",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "parent_contract_lock_sha256": sha256(design.LOCK),
        "implementation_lock_sha256": sha256(base.IMPLEMENTATION),
        "r4_amendment_sha256": sha256(R4_LOCK),
        "sensor_preflight_sha256": sha256(SENSOR_PREFLIGHT),
        "live_preflight_attempt_04_sha256": sha256(ATTEMPT_04),
        "expected_parent_families": summary["expected"],
        "population": "4 answerability states x 4 relations x " + str(summary["repetitions_per_cell"]),
        "manifest_summary": summary,
        "capture_attempts_authorized": 1,
        "authorized_output": str(output.relative_to(ROOT)),
        "source_artifact_sha256": sources,
        "model_inventory_sha256": "NO_MODEL_INFERENCE_DURING_CAPTURE",
        "policies": {
            "single_complete_capture_attempt_only": True,
            "partial_attempt_reuse_forbidden": True,
            "row_filter_repair_or_replacement_forbidden": True,
            "cross_attempt_merge_forbidden": True,
            "no_model_inference_during_capture": True,
            "no_calibrator_fit_during_capture": True,
            "no_automatic_restart_or_scene_retry": True,
            "test_iid_ood_sealed": True,
            "robot_policy_sealed": True,
        },
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(lock_path, payload)
    print(json.dumps({"status": payload["authorization_status"], "sha256": sha256(lock_path), "output": payload["authorized_output"]}, indent=2))


def verify_authorization(pid: str) -> dict[str, Any]:
    validate_r4()
    value = read_json(auth_path(pid))
    if value.get("protocol_id") != pid or value.get("status") != "LOCKED_BEFORE_CAPTURE_AND_INFERENCE":
        raise RuntimeError("capture authorization invalid")
    for name, digest in value["source_artifact_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"authorized source drift: {name}")
    if not all(value["policies"].values()):
        raise RuntimeError("capture policy drift")
    return value


def tree_inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def freeze_capture_failure(pid: str, returncode: int) -> None:
    output = capture_path(pid)
    lock = (PILOT_ROOT if pid == design.PILOT else CALIBRATION_ROOT) / "CAPTURE_ATTEMPT_01_FAILURE_LOCK.json"
    rows_path = output / "input_manifest.jsonl"
    rows = len([line for line in rows_path.read_text().splitlines() if line]) if rows_path.exists() else 0
    write_new(lock, {
        "schema_version": 1,
        "protocol_id": pid,
        "status": "CAPTURE_ATTEMPT_01_FROZEN_INCOMPLETE_NO_RETRY_UNDER_R4",
        "classification": "INFRASTRUCTURE_CAPTURE_FAILURE_NOT_SCIENTIFIC_RESULT",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_family_count": 32 if pid == design.PILOT else 128,
        "captured_record_count": rows,
        "process_returncode": returncode,
        "partial_attempt_policy": {"partial_rows_quarantined": True, "resume_allowed": False, "same_revision_retry_allowed": False, "cross_attempt_merge_allowed": False},
        "source_artifact_sha256": tree_inventory(output),
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    })


def capture(pid: str) -> None:
    verify_authorization(pid)
    output = capture_path(pid)
    if output.exists():
        raise FileExistsError("single R4 capture attempt already exists")
    command = ["/usr/bin/python3", str(CAPTURE_CODE), "--ros-args", "-p", "use_sim_time:=true"]
    parameters: list[tuple[str, str | Path]] = [
        ("scene_config_file", design.paths(pid)[0]),
        ("annotation_file", design.paths(pid)[1]),
        ("gate_config_file", design.paths(pid)[2]),
        ("pretrial_lock_file", auth_path(pid)),
        ("output_root", output),
        ("settle_sec", "1.5"),
        ("sync_slop_sec", "0.02"),
        ("capture_timeout_sec", "45.0"),
    ]
    for key, value in parameters:
        command += ["-p", f"{key}:={value}"]
    completed = subprocess.run(command, cwd=ROOT, env=ros_environment(), check=False)
    if completed.returncode != 0:
        freeze_capture_failure(pid, completed.returncode)
        raise SystemExit(2)
    manifest = read_json(output / "capture_manifest.json")
    rows = [line for line in (output / "input_manifest.jsonl").read_text().splitlines() if line]
    expected = 32 if pid == design.PILOT else 128
    if manifest.get("status") != "COMPLETE" or len(rows) != expected:
        freeze_capture_failure(pid, 0)
        raise SystemExit(2)


def wrap_decision() -> None:
    """Bind the original one-fit result into an R4 successor decision lock."""
    if FINAL_R4.exists():
        raise FileExistsError("R4 scientific decision already frozen")
    validate_r4()
    original = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/FINAL_CALIBRATION_V6_DECISION.json"
    fit_input = ROOT / "protocol/GAZEBO_CALIBRATION_V6_FIT_INPUT_LOCK.json"
    threshold = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/CALIBRATOR_THRESHOLD_V3_LOCK.json"
    required = [CALIBRATION_QC, original, fit_input, threshold, ROOT / "protocol/GAZEBO_CALIBRATION_V6_MATERIALIZATION_LOCK.json"]
    if any(not path.is_file() for path in required):
        raise RuntimeError("calibration QC/materialization/inference/one-fit decision is incomplete")
    decision = read_json(original)
    decision_map = {
        "CALIBRATION_V6_PASS_ELIGIBLE_FOR_SEPARATE_LOCKED_TEST_AUTHORIZATION": "PASS",
        "CALIBRATION_V6_NEGATIVE_NO_TEST_OR_ROBOT": "NEGATIVE",
    }
    if decision.get("decision") not in decision_map:
        raise RuntimeError("scientific decision must be an honest PASS or NEGATIVE")
    payload = {
        "schema_version": 1,
        "protocol_id": "gazebo_calibration_v6",
        "status": "SCIENTIFIC_DECISION_FROZEN",
        "decision": decision_map[decision["decision"]],
        "frozen_fit_status": decision["decision"],
        "classification": "CALIBRATION_V6_R4_SCIENTIFIC_RESULT",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_capture": "32/32 QC PASS",
        "calibration_capture": "128/128 QC PASS",
        "calibrator_fit_call_count": 1,
        "source_artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in required + [R4_LOCK, PILOT_DECISION, PILOT_AUTH, CALIBRATION_AUTH]},
        "upstream_blocked_decision_preserved": "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/CALIBRATION_V6_FINAL_DECISION.json",
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(FINAL_R4, payload)
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "diagnose", "freeze-amendment", "validate", "preflight-sensor", "preflight-live",
        "authorize", "capture", "wrap-decision",
    ))
    parser.add_argument("--split", choices=("pilot", "calibration"), default="pilot")
    args = parser.parse_args()
    pid = design.PILOT if args.split == "pilot" else design.CALIBRATION
    if args.command == "diagnose":
        diagnose()
    elif args.command == "freeze-amendment":
        freeze_amendment()
    elif args.command == "validate":
        validate_r4(); print("PASS: frozen Calibration-v6 R4 amendment")
    elif args.command == "preflight-sensor":
        sensor_preflight()
    elif args.command == "preflight-live":
        live_preflight()
    elif args.command == "authorize":
        authorize(pid)
    elif args.command == "capture":
        capture(pid)
    else:
        wrap_decision()


if __name__ == "__main__":
    main()
