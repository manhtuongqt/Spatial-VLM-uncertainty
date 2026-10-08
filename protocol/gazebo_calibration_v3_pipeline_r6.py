#!/usr/bin/env python3
"""Append-only Calibration-v3 R6 parser repair and gated capture runner."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import gazebo_calibration_v3_pipeline_r4 as r4
import gazebo_calibration_v3_pipeline_r5 as r5


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_calibration_v3"
CONTRACT = r5.CONTRACT
R5_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R5.json"
R5_AUDIT = ROOT / "protocol/gazebo_calibration_v3_implementation_r5_audit.json"
R6_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R6.json"
R6_AUDIT = ROOT / "protocol/gazebo_calibration_v3_implementation_r6_audit.json"
STATIC_PREFLIGHT = r5.STATIC_PREFLIGHT
ATTEMPT_01 = r5.ATTEMPT_01
ATTEMPT_02 = r5.ATTEMPT_02
ATTEMPT_03 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE_ATTEMPT_03.json"
ATTEMPT_04 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE_ATTEMPT_04.json"
CAPTURE_LOCK = r5.CAPTURE_LOCK
CAPTURE_ROOT = r5.CAPTURE_ROOT
DATASET_ROOT = r5.DATASET_ROOT
SCENES = r5.SCENES
ANNOTATIONS = r5.ANNOTATIONS
GATE = r5.GATE
WORLD = r5.WORLD
CAPTURE_REQUIRED_SOURCES = r5.CAPTURE_REQUIRED_SOURCES

R5_LOCK_SHA256 = "f522def330478deb2edde9365f4f4cc5b4f561b2238f845211900c3dd31dedb4"
R5_AUDIT_SHA256 = "913056cbdfd7dc0a89d786df8f3b29a9ad0890d414d7f2dcd126cc7c7a1f9952"
ATTEMPT_03_SHA256 = "25d8785802e4fe4efdea74471d4c370da513b93b92f2a38f106975905b023a7e"


sha256 = r5.sha256
read_json = r5.read_json
write_json = r5.write_json
ros_environment = r5.ros_environment
run_probe = r5.run_probe


def parse_ros_numeric_array(output: str, expected_length: int) -> list[float] | None:
    """Parse either YAML comma lists or ROS NumPy-style whitespace arrays."""
    left = output.find("[")
    right = output.rfind("]")
    if left < 0 or right <= left:
        return None
    body = output[left + 1:right].strip()
    tokens = [token for token in re.split(r"[\s,]+", body) if token]
    if len(tokens) != expected_length:
        return None
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        return None
    return values if all(math.isfinite(value) for value in values) else None


def parse_joint_state_document(output: str) -> dict[str, Any] | None:
    """Ignore ROS warning preambles and parse the actual JointState YAML body."""
    match = re.search(r"(?m)^header:\s*$", output)
    if match is None:
        return None
    try:
        for document in yaml.safe_load_all(output[match.start():]):
            if isinstance(document, dict) and "name" in document and "position" in document:
                return document
    except yaml.YAMLError:
        return None
    return None


def camera_info_probe(topic: str) -> dict:
    fields = {
        field: r4.echo_field(topic, field)
        for field in ("height", "width", "k", "header.frame_id")
    }
    values = {
        field: r4.parsed_field(result)
        for field, result in fields.items()
        if field != "k"
    }
    k_probe = fields["k"]
    values["k"] = (
        parse_ros_numeric_array(str(k_probe.get("stdout", "")), 9)
        if k_probe.get("returncode") == 0
        else None
    )
    return {"commands": fields, "values": values}


def joint_state_probe(expected_positions: list[float]) -> tuple[dict, bool, dict]:
    probe = run_probe(["ros2", "topic", "echo", "--once", "/joint_states"])
    document = (
        parse_joint_state_document(str(probe.get("stdout", "")))
        if probe.get("returncode") == 0
        else None
    )
    comparison: dict[str, dict[str, float]] = {}
    passed = False
    if isinstance(document, dict):
        names = document.get("name")
        positions = document.get("position")
        if isinstance(names, list) and isinstance(positions, list) and len(names) == len(positions):
            observed = dict(zip((str(name) for name in names), positions))
            if all(name in observed and r4.finite_number(observed[name]) for name in r4.EXPECTED_JOINT_NAMES):
                comparison = {
                    name: {
                        "observed": float(observed[name]),
                        "expected": float(target),
                        "absolute_error": abs(float(observed[name]) - float(target)),
                    }
                    for name, target in zip(r4.EXPECTED_JOINT_NAMES, expected_positions)
                }
                passed = all(
                    item["absolute_error"] <= r4.JOINT_POSITION_TOLERANCE_RAD
                    for item in comparison.values()
                )
    return probe, passed, comparison


def install_parser_repair() -> None:
    r4.camera_info_probe = camera_info_probe
    r4.joint_state_probe = joint_state_probe


def parser_regression_results() -> dict:
    attempt = read_json(ATTEMPT_03)
    observations = attempt["observations"]
    wrist = observations["samples"]["wrist_camera_info"]
    base = observations["samples"]["base_camera_info"]
    joint = observations["joint_state"]
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    wrist_k = parse_ros_numeric_array(wrist["commands"]["k"]["stdout"], 9)
    base_k = parse_ros_numeric_array(base["commands"]["k"]["stdout"], 9)
    document = parse_joint_state_document(joint["stdout"])
    comparison: dict[str, dict[str, float]] = {}
    joint_pass = False
    if document is not None:
        names = document.get("name", [])
        positions = document.get("position", [])
        observed = dict(zip(names, positions)) if len(names) == len(positions) else {}
        expected = [float(value) for value in gate["camera"]["view_joint_pose"]]
        if all(name in observed and r4.finite_number(observed[name]) for name in r4.EXPECTED_JOINT_NAMES):
            comparison = {
                name: {
                    "observed": float(observed[name]),
                    "expected": target,
                    "absolute_error": abs(float(observed[name]) - target),
                }
                for name, target in zip(r4.EXPECTED_JOINT_NAMES, expected)
            }
            joint_pass = all(
                row["absolute_error"] <= r4.JOINT_POSITION_TOLERANCE_RAD
                for row in comparison.values()
            )
    checks = {
        "attempt_03_is_immutable_blocked": attempt.get("status") == "BLOCKED" and sha256(ATTEMPT_03) == ATTEMPT_03_SHA256,
        "wrist_k_parses_exactly_9_finite_floats": wrist_k is not None and len(wrist_k) == 9,
        "base_k_parses_exactly_9_finite_floats": base_k is not None and len(base_k) == 9,
        "joint_warning_preamble_is_ignored": document is not None,
        "all_six_locked_joints_parse": len(comparison) == 6,
        "recorded_joint_pose_passes_existing_tolerance": joint_pass,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "wrist_k": wrist_k,
        "base_k": base_k,
        "joint_comparison": comparison,
    }


def validate_r6_implementation() -> tuple[dict, dict, dict]:
    r5.validate_r5_implementation()
    fixed = {
        R5_LOCK: R5_LOCK_SHA256,
        R5_AUDIT: R5_AUDIT_SHA256,
        ATTEMPT_03: ATTEMPT_03_SHA256,
    }
    for path, digest in fixed.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Calibration-v3 R6 upstream drift: {path}")
    if not R6_LOCK.is_file() or not R6_AUDIT.is_file():
        raise RuntimeError("Calibration-v3 R6 implementation is not frozen")
    lock = read_json(R6_LOCK)
    audit = read_json(R6_AUDIT)
    if lock.get("status") != "IMPLEMENTATION_R6_FROZEN_BEFORE_LIVE_PREFLIGHT_ATTEMPT_04":
        raise RuntimeError("invalid Calibration-v3 R6 implementation state")
    if lock.get("capture_authorized") is not False:
        raise RuntimeError("R6 implementation lock must not authorize capture")
    if audit.get("status") != "PASS" or lock.get("implementation_audit_sha256") != sha256(R6_AUDIT):
        raise RuntimeError("Calibration-v3 R6 audit linkage failed")
    for name, digest in lock["source_artifact_sha256"].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Calibration-v3 R6 source drift: {name}")
    if parser_regression_results()["status"] != "PASS":
        raise RuntimeError("Calibration-v3 R6 parser regression failed")
    scenes = yaml.safe_load(SCENES.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    return scenes, gate, read_json(CONTRACT)


# Compatibility aliases used by the inherited R5 downstream adapter.
validate_r5_implementation = validate_r6_implementation


def live_preflight_attempt_04() -> None:
    if ATTEMPT_04.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 live preflight attempt 04")
    validate_r6_implementation()
    install_parser_repair()
    original_target = r5.ATTEMPT_03
    original_writer = r5.write_json

    def r6_report_writer(path: Path, value: dict) -> None:
        if path != ATTEMPT_04:
            raise RuntimeError(f"unexpected R6 live-preflight output: {path}")
        value["implementation_revision"] = "r6"
        value["attempt"] = 4
        value["r6_implementation_lock_sha256"] = sha256(R6_LOCK)
        value["prior_live_preflight_sha256"]["attempt_03"] = sha256(ATTEMPT_03)
        value["checks"]["attempt_03_preserved_blocked"] = (
            read_json(ATTEMPT_03).get("status") == "BLOCKED"
            and sha256(ATTEMPT_03) == ATTEMPT_03_SHA256
        )
        value["checks"]["r6_implementation_audit_pass"] = read_json(R6_AUDIT).get("status") == "PASS"
        value["checks"]["r6_parser_regression_pass"] = parser_regression_results()["status"] == "PASS"
        value["status"] = "PASS" if all(value["checks"].values()) else "BLOCKED"
        value["capture_authorized"] = False
        value["next_authorized_action_if_pass"] = "Create the separate immutable R6 Calibration-v3 capture-authorization lock."
        original_writer(path, value)

    r5.ATTEMPT_03 = ATTEMPT_04
    r5.write_json = r6_report_writer
    try:
        r5.live_preflight_attempt_03()
    finally:
        r5.ATTEMPT_03 = original_target
        r5.write_json = original_writer
    if read_json(ATTEMPT_04).get("status") != "PASS":
        raise SystemExit(2)


def authorize_capture() -> None:
    scenes, _, contract = validate_r6_implementation()
    if CAPTURE_LOCK.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 capture authorization")
    live = read_json(ATTEMPT_04)
    if live.get("status") != "PASS" or live.get("capture_authorized") is not False:
        raise RuntimeError("live preflight attempt 04 must PASS before capture authorization")
    if CAPTURE_ROOT.exists() or DATASET_ROOT.exists():
        raise RuntimeError("Calibration-v3 downstream data exists before capture authorization")
    r6_lock = read_json(R6_LOCK)
    source_names = set(r6_lock["source_artifact_sha256"]) | CAPTURE_REQUIRED_SOURCES | {
        "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R6.json",
        "protocol/gazebo_calibration_v3_implementation_r6_audit.json",
        "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_STATIC.json",
        "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE.json",
        "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE_ATTEMPT_02.json",
        "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE_ATTEMPT_03.json",
        "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE_ATTEMPT_04.json",
        "ur3/ur3_perception/config/gazebo_calibration_v3_scenes.yaml",
        "ur3/ur3_perception/config/gazebo_calibration_v3_annotations.yaml",
        "ur3/ur3_perception/config/gazebo_calibration_v3_gate.yaml",
        "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
    }
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "authorization_status": "CAPTURE_AUTHORIZED_AFTER_STATIC_AND_LIVE_PREFLIGHT_PASS",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "parent_contract_lock_sha256": sha256(CONTRACT),
        "implementation_lock_sha256": sha256(R6_LOCK),
        "implementation_audit_sha256": sha256(R6_AUDIT),
        "static_preflight_sha256": sha256(STATIC_PREFLIGHT),
        "live_preflight_sha256": sha256(ATTEMPT_04),
        "capture_attempts_authorized": 1,
        "expected_parent_families": len(scenes["scenes"]),
        "source_artifact_sha256": {
            name: sha256(ROOT / name) for name in sorted(source_names)
        },
        "model_inventory_sha256": "NO_MODEL_INFERENCE_DURING_CALIBRATION_V3_CAPTURE",
        "policies": {
            "no_model_inference_during_capture": True,
            "no_robot_manipulation": True,
            "semantic_labels_evaluator_only": True,
            "calibration_v2_closed": True,
            "test_iid_ood_sealed": True,
            "single_capture_attempt_only": True,
            "partial_row_reuse_forbidden": True,
        },
        "pass_authority_limit": contract["terminal_decision_rule"]["pass_authority_limit"],
    }
    write_json(CAPTURE_LOCK, payload)
    print(json.dumps({
        "status": payload["authorization_status"],
        "sha256": sha256(CAPTURE_LOCK),
    }, indent=2, sort_keys=True))


def verify_capture_authorization() -> None:
    validate_r6_implementation()
    lock = read_json(CAPTURE_LOCK)
    if lock.get("status") != "LOCKED_BEFORE_CAPTURE_AND_INFERENCE":
        raise RuntimeError("capture lock schema is not accepted by the frozen capture node")
    if lock.get("authorization_status") != "CAPTURE_AUTHORIZED_AFTER_STATIC_AND_LIVE_PREFLIGHT_PASS":
        raise RuntimeError("Calibration-v3 capture is not authorized")
    if lock.get("parent_contract_lock_sha256") != sha256(CONTRACT):
        raise RuntimeError("capture/contract linkage failed")
    if lock.get("implementation_lock_sha256") != sha256(R6_LOCK):
        raise RuntimeError("capture/R6 implementation linkage failed")
    if lock.get("live_preflight_sha256") != sha256(ATTEMPT_04) or read_json(ATTEMPT_04).get("status") != "PASS":
        raise RuntimeError("capture/live-preflight linkage failed")
    missing = CAPTURE_REQUIRED_SOURCES - set(lock.get("source_artifact_sha256", {}))
    if missing:
        raise RuntimeError(f"capture authorization misses frozen capture-node sources: {sorted(missing)}")
    for name, digest in lock["source_artifact_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"capture source drift: {name}")


def capture() -> None:
    verify_capture_authorization()
    if CAPTURE_ROOT.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 capture attempt 01")
    command = [
        "ros2", "launch", "ur3_perception", "roborefer_uq_capture.launch.py",
        f"world_file:={WORLD}",
        f"scene_config_file:={SCENES}",
        f"annotation_file:={ANNOTATIONS}",
        f"gate_config_file:={GATE}",
        f"pretrial_lock_file:={CAPTURE_LOCK}",
        f"output_root:={CAPTURE_ROOT}",
        "gazebo_gui:=false",
    ]
    subprocess.run(command, cwd=ROOT, env=ros_environment(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("validate", "parser-regression", "preflight-live", "authorize-capture", "capture"),
    )
    command = parser.parse_args().command
    if command == "validate":
        validate_r6_implementation()
        print("PASS: frozen Calibration-v3 implementation revision R6")
    elif command == "parser-regression":
        value = parser_regression_results()
        print(json.dumps(value, indent=2, sort_keys=True))
        if value["status"] != "PASS":
            raise SystemExit(2)
    elif command == "preflight-live":
        live_preflight_attempt_04()
    elif command == "authorize-capture":
        authorize_capture()
    else:
        capture()


if __name__ == "__main__":
    main()
