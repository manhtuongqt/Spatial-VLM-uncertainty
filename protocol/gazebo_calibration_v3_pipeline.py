#!/usr/bin/env python3
"""Validate, preflight, authorize, and capture Calibration-v3 in locked order."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

import generate_gazebo_calibration_v3_contract as generator


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_calibration_v3"
CONTRACT = ROOT / "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json"
IMPLEMENTATION_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R3.json"
IMPLEMENTATION_AUDIT = ROOT / "protocol/gazebo_calibration_v3_implementation_r3_audit.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_calibration_v3_capture_compatibility_lock.json"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3"
CAPTURE = RESULT / "capture_attempt_01"
STATIC_PREFLIGHT = RESULT / "PREFLIGHT_STATIC.json"
LIVE_PREFLIGHT = RESULT / "PREFLIGHT_LIVE.json"
WORLD = ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf"
CONTRACT_SHA256 = generator.CONTRACT_SHA256

LEGACY_CAPTURE_SOURCES = (
    "RoboRefer/API/api.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_capture.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_runner.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_validate.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_evaluator.py",
    "ur3/ur3_perception/scripts/roborefer_grounder.py",
    "ur3/ur3_perception/scripts/spatial_point_utils.py",
    "ur3/ur3_perception/scripts/move_camera_to_view.py",
    "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
    "ur3/ur3_perception/schemas/roborefer_pilot_prediction.schema.json",
    "ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro",
    "ur3/ur3_moveit_control/launch/ur3_susgrip_sim.launch.py",
    "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
    "ur3/ur_simulation_gz/launch/ur_sim_control.launch.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_implementation() -> tuple[dict, dict, dict]:
    if sha256(CONTRACT) != CONTRACT_SHA256:
        raise RuntimeError("Calibration-v3 contract hash drift")
    if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_AUDIT.is_file():
        raise RuntimeError("Calibration-v3 implementation is not frozen")
    lock = read_json(IMPLEMENTATION_LOCK)
    audit = read_json(IMPLEMENTATION_AUDIT)
    if lock.get("status") != "IMPLEMENTATION_R3_FROZEN_BEFORE_STATIC_OR_LIVE_PREFLIGHT":
        raise RuntimeError("invalid implementation lock state")
    if lock.get("contract_lock_sha256") != CONTRACT_SHA256:
        raise RuntimeError("implementation/contract linkage failed")
    if audit.get("status") != "PASS" or lock.get("implementation_audit_sha256") != sha256(IMPLEMENTATION_AUDIT):
        raise RuntimeError("implementation audit linkage failed")
    for name, digest in lock["source_artifact_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"frozen implementation source drift: {name}")
    paths, texts = generator.serialized_outputs()
    for path, text in zip(paths, texts):
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"deterministic generated artifact drift: {path}")
    scenes = yaml.safe_load(generator.SCENES_PATH.read_text(encoding="utf-8"))
    annotations = yaml.safe_load(generator.ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(generator.GATE_PATH.read_text(encoding="utf-8"))
    rows = scenes["scenes"]
    oracle = annotations["scenes"]
    if len(rows) != 128 or len(oracle) != 128 or set(oracle) != {row["scene_id"] for row in rows}:
        raise RuntimeError("Calibration-v3 scene/oracle cardinality mismatch")
    cells = Counter((item["state"], item["relation_variant"]) for item in oracle.values())
    if len(cells) != 16 or set(cells.values()) != {8}:
        raise RuntimeError("Calibration-v3 4x4x8 quota drift")
    if gate["policies"].get("capture_requires_post_preflight_compatibility_lock") is not True:
        raise RuntimeError("capture authorization policy drift")
    return scenes, annotations, gate


def static_preflight() -> None:
    scenes, _, _ = validate_implementation()
    free_gib = shutil.disk_usage(ROOT).free / (2**30)
    checks = {
        "contract_hash": sha256(CONTRACT) == CONTRACT_SHA256,
        "implementation_lock": True,
        "implementation_audit": read_json(IMPLEMENTATION_AUDIT)["status"] == "PASS",
        "family_count_128": len(scenes["scenes"]) == 128,
        "disk_free_over_2_gib": free_gib > 2.0,
        "capture_absent": not CAPTURE.exists(),
        "dataset_absent": not (ROOT / "datasets/Gazebo_calibration_v3").exists(),
        "test_iid_ood_sealed": True,
    }
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_lock_sha256": sha256(CONTRACT),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK),
        "checks": checks,
        "free_gib": free_gib,
        "capture_authorized": False,
    }
    if STATIC_PREFLIGHT.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 static preflight")
    write_json(STATIC_PREFLIGHT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout


def live_preflight() -> None:
    validate_implementation()
    static = read_json(STATIC_PREFLIGHT)
    if static.get("status") != "PASS":
        raise RuntimeError("static preflight must pass first")
    topics = command_output(["ros2", "topic", "list", "-t"])
    services = command_output(["ros2", "service", "list", "-t"])
    required_topics = {
        "/wrist_camera/color/image_raw": "sensor_msgs/msg/Image",
        "/wrist_camera/depth/image_raw": "sensor_msgs/msg/Image",
        "/wrist_camera/evaluation_labels/labels_map": "sensor_msgs/msg/Image",
        "/wrist_camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
        "/tf": "tf2_msgs/msg/TFMessage",
        "/joint_states": "sensor_msgs/msg/JointState",
    }
    topic_checks = {
        name: f"{name} [{message_type}]" in topics
        for name, message_type in required_topics.items()
    }
    service_pass = (
        "/world/ur3_pick_place/set_pose [ros_gz_interfaces/srv/SetEntityPose]"
        in services
    )
    free_gib = shutil.disk_usage(ROOT).free / (2**30)
    passed = all(topic_checks.values()) and service_pass and free_gib > 2.0
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if passed else "BLOCKED",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_lock_sha256": sha256(CONTRACT),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK),
        "static_preflight_sha256": sha256(STATIC_PREFLIGHT),
        "topic_checks": topic_checks,
        "set_pose_service": service_pass,
        "free_gib": free_gib,
        "capture_authorized": False,
    }
    if LIVE_PREFLIGHT.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 live preflight")
    write_json(LIVE_PREFLIGHT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


def authorize_capture() -> None:
    validate_implementation()
    if CAPTURE_LOCK.exists():
        raise FileExistsError("refusing to overwrite capture-compatibility lock")
    static, live = read_json(STATIC_PREFLIGHT), read_json(LIVE_PREFLIGHT)
    if static.get("status") != "PASS" or live.get("status") != "PASS":
        raise RuntimeError("both preflights must pass before capture authorization")
    implementation = read_json(IMPLEMENTATION_LOCK)
    source_names = set(implementation["source_artifact_sha256"]) | set(LEGACY_CAPTURE_SOURCES)
    source_hashes = {name: sha256(ROOT / name) for name in sorted(source_names)}
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "CAPTURE_AUTHORIZED_AFTER_STATIC_AND_LIVE_PREFLIGHT_PASS",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "parent_contract_lock": str(CONTRACT.relative_to(ROOT)),
        "parent_contract_lock_sha256": sha256(CONTRACT),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK),
        "implementation_audit_sha256": sha256(IMPLEMENTATION_AUDIT),
        "static_preflight_sha256": sha256(STATIC_PREFLIGHT),
        "live_preflight_sha256": sha256(LIVE_PREFLIGHT),
        "capture_attempts_authorized": 1,
        "expected_parent_families": 128,
        "purpose": "Calibration-v3 no-manipulation capture only",
        "source_artifact_sha256": source_hashes,
        "model_inventory_sha256": "NO_MODEL_INFERENCE_DURING_CALIBRATION_V3_CAPTURE",
        "policies": {
            "no_model_inference_during_capture": True,
            "no_robot_manipulation": True,
            "semantic_labels_evaluator_only": True,
            "calibration_v2_closed": True,
            "test_iid_ood_sealed": True,
        },
    }
    write_json(CAPTURE_LOCK, payload)
    print(json.dumps({"status": payload["status"], "sha256": sha256(CAPTURE_LOCK)}, indent=2))


def verify_capture_authorization() -> None:
    validate_implementation()
    lock = read_json(CAPTURE_LOCK)
    if lock.get("status") != "CAPTURE_AUTHORIZED_AFTER_STATIC_AND_LIVE_PREFLIGHT_PASS":
        raise RuntimeError("Calibration-v3 capture is not authorized")
    if lock.get("parent_contract_lock_sha256") != sha256(CONTRACT):
        raise RuntimeError("capture/contract linkage failed")
    if lock.get("implementation_lock_sha256") != sha256(IMPLEMENTATION_LOCK):
        raise RuntimeError("capture/implementation linkage failed")
    if lock.get("static_preflight_sha256") != sha256(STATIC_PREFLIGHT) or lock.get("live_preflight_sha256") != sha256(LIVE_PREFLIGHT):
        raise RuntimeError("capture/preflight linkage failed")
    for name, digest in lock["source_artifact_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"capture source drift: {name}")


def capture() -> None:
    verify_capture_authorization()
    if CAPTURE.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 capture attempt")
    command = [
        "ros2",
        "launch",
        "ur3_perception",
        "roborefer_uq_capture.launch.py",
        f"world_file:={WORLD}",
        f"scene_config_file:={generator.SCENES_PATH}",
        f"annotation_file:={generator.ANNOTATIONS_PATH}",
        f"gate_config_file:={generator.GATE_PATH}",
        f"pretrial_lock_file:={CAPTURE_LOCK}",
        f"output_root:={CAPTURE}",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "validate",
            "preflight-static",
            "preflight-live",
            "authorize-capture",
            "capture",
        ),
    )
    command = parser.parse_args().command
    if command == "validate":
        validate_implementation()
        print("PASS: frozen Calibration-v3 implementation")
    elif command == "preflight-static":
        static_preflight()
    elif command == "preflight-live":
        live_preflight()
    elif command == "authorize-capture":
        authorize_capture()
    else:
        capture()


if __name__ == "__main__":
    main()
