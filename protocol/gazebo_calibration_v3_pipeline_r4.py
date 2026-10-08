#!/usr/bin/env python3
"""Append-only Calibration-v3 live-preflight revision R4.

R4 repairs only the operational completeness of the live preflight.  It does
not authorize capture and deliberately exposes no capture, materialization,
inference, or calibration command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import gazebo_calibration_v3_pipeline as r3


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_calibration_v3"
CONTRACT = ROOT / "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json"
R3_IMPLEMENTATION_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R3.json"
R4_IMPLEMENTATION_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R4.json"
R4_IMPLEMENTATION_AUDIT = ROOT / "protocol/gazebo_calibration_v3_implementation_r4_audit.json"
STATIC_PREFLIGHT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_STATIC.json"
R3_BLOCKED_LIVE_PREFLIGHT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE.json"
R4_LIVE_PREFLIGHT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE_ATTEMPT_02.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_calibration_v3_capture_compatibility_lock.json"
CAPTURE_ROOT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/capture_attempt_01"
DATASET_ROOT = ROOT / "datasets/Gazebo_calibration_v3"
SCENES = ROOT / "ur3/ur3_perception/config/gazebo_calibration_v3_scenes.yaml"
GATE = ROOT / "ur3/ur3_perception/config/gazebo_calibration_v3_gate.yaml"
WORLD = ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf"

CONTRACT_SHA256 = "99b430b1c55e1762f8460db4e8134b4a8782403e466d71ea86ea3da847bfc841"
R3_IMPLEMENTATION_LOCK_SHA256 = "e1b3c129296a968e805cb64027ac2e03cf39d3460eed05f4d8e288b2f924a931"
STATIC_PREFLIGHT_SHA256 = "5ccba3ee03efd3900048a51a3ec33185a8da26266d8668fe4c9a1d1c492e6f77"
R3_BLOCKED_LIVE_PREFLIGHT_SHA256 = "25f9347fccf195c3d39da18559150e2df16018cba73dada3c54aa4b289b41ea0"

EXPECTED_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
JOINT_POSITION_TOLERANCE_RAD = 0.05
INTRINSIC_ABSOLUTE_TOLERANCE = 1.0e-3
COMMAND_TIMEOUT_SEC = 12.0

REQUIRED_TOPICS = {
    "/clock": "rosgraph_msgs/msg/Clock",
    "/joint_states": "sensor_msgs/msg/JointState",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/wrist_camera/color/image_raw": "sensor_msgs/msg/Image",
    "/wrist_camera/depth/image_raw": "sensor_msgs/msg/Image",
    "/wrist_camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
    "/wrist_camera/evaluation_labels/labels_map": "sensor_msgs/msg/Image",
    "/top_table_camera/color/image_raw": "sensor_msgs/msg/Image",
    "/top_table_camera/depth/image_raw": "sensor_msgs/msg/Image",
    "/top_table_camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
}
SET_POSE_SERVICE = "/world/ur3_pick_place/set_pose"
SET_POSE_SERVICE_TYPE = "ros_gz_interfaces/srv/SetEntityPose"


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


def run_probe(command: list[str], timeout: float = COMMAND_TIMEOUT_SEC) -> dict:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "seconds": time.monotonic() - started,
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "seconds": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "command": command,
            "returncode": 124,
            "stdout": stdout.strip(),
            "stderr": stderr.strip() or "timeout",
            "seconds": time.monotonic() - started,
        }


def first_yaml_document(output: str) -> Any:
    try:
        for document in yaml.safe_load_all(output):
            if document is not None:
                return document
    except yaml.YAMLError:
        return None
    return None


def echo_field(topic: str, field: str, timeout: float = COMMAND_TIMEOUT_SEC) -> dict:
    return run_probe(
        ["ros2", "topic", "echo", "--once", topic, "--field", field],
        timeout=timeout,
    )


def parsed_field(probe: dict) -> Any:
    if probe.get("returncode") != 0:
        return None
    return first_yaml_document(str(probe.get("stdout", "")))


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def image_probe(topic: str) -> dict:
    fields = {
        field: echo_field(topic, field)
        for field in ("height", "width", "encoding", "header.frame_id")
    }
    return {
        "commands": fields,
        "values": {field: parsed_field(result) for field, result in fields.items()},
    }


def camera_info_probe(topic: str) -> dict:
    fields = {
        field: echo_field(topic, field)
        for field in ("height", "width", "k", "header.frame_id")
    }
    return {
        "commands": fields,
        "values": {field: parsed_field(result) for field, result in fields.items()},
    }


def image_matches(
    probe: dict,
    *,
    width: int,
    height: int,
    encodings: set[str],
    frame_id: str,
) -> bool:
    values = probe.get("values", {})
    return (
        values.get("width") == width
        and values.get("height") == height
        and str(values.get("encoding", "")) in encodings
        and str(values.get("header.frame_id", "")) == frame_id
    )


def camera_info_valid(
    probe: dict,
    *,
    width: int,
    height: int,
    frame_id: str,
    expected_intrinsics: dict[str, float] | None,
) -> bool:
    values = probe.get("values", {})
    matrix = values.get("k")
    if not (
        values.get("width") == width
        and values.get("height") == height
        and str(values.get("header.frame_id", "")) == frame_id
        and isinstance(matrix, list)
        and len(matrix) == 9
        and all(finite_number(item) for item in matrix)
        and float(matrix[0]) > 0.0
        and float(matrix[4]) > 0.0
        and float(matrix[8]) == 1.0
    ):
        return False
    if expected_intrinsics is None:
        return True
    expected = (
        float(expected_intrinsics["fx"]),
        float(expected_intrinsics["fy"]),
        float(expected_intrinsics["cx"]),
        float(expected_intrinsics["cy"]),
    )
    observed = (float(matrix[0]), float(matrix[4]), float(matrix[2]), float(matrix[5]))
    return all(abs(actual - target) <= INTRINSIC_ABSOLUTE_TOLERANCE for actual, target in zip(observed, expected))


def joint_state_probe(expected_positions: list[float]) -> tuple[dict, bool, dict]:
    probe = run_probe(["ros2", "topic", "echo", "--once", "/joint_states"])
    document = first_yaml_document(probe.get("stdout", "")) if probe.get("returncode") == 0 else None
    comparison: dict[str, dict[str, float]] = {}
    passed = False
    if isinstance(document, dict):
        names = document.get("name")
        positions = document.get("position")
        if isinstance(names, list) and isinstance(positions, list) and len(names) == len(positions):
            observed = dict(zip((str(name) for name in names), positions))
            if all(name in observed and finite_number(observed[name]) for name in EXPECTED_JOINT_NAMES):
                comparison = {
                    name: {
                        "observed": float(observed[name]),
                        "expected": float(target),
                        "absolute_error": abs(float(observed[name]) - float(target)),
                    }
                    for name, target in zip(EXPECTED_JOINT_NAMES, expected_positions)
                }
                passed = all(item["absolute_error"] <= JOINT_POSITION_TOLERANCE_RAD for item in comparison.values())
    return probe, passed, comparison


def scene_layout(scene_config: dict, scene: dict) -> dict[str, list[float]]:
    layout: dict[str, list[float]] = {}
    overrides = scene.get("poses", {})
    for model_name, object_config in scene_config["objects"].items():
        x_value, y_value, yaw = overrides.get(model_name, object_config["storage_pose"])
        yaw = float(yaw)
        layout[model_name] = [
            float(x_value),
            float(y_value),
            float(object_config["z"]),
            0.0,
            0.0,
            math.sin(0.5 * yaw),
            math.cos(0.5 * yaw),
        ]
    return layout


def set_pose_request(model_name: str, pose: list[float]) -> str:
    x_value, y_value, z_value, qx, qy, qz, qw = pose
    return (
        "{entity: {name: '" + model_name + "', type: 2}, pose: {position: {"
        f"x: {x_value:.12g}, y: {y_value:.12g}, z: {z_value:.12g}"
        "}, orientation: {"
        f"x: {qx:.12g}, y: {qy:.12g}, z: {qz:.12g}, w: {qw:.12g}"
        "}}}"
    )


def service_call_succeeded(result: dict) -> bool:
    normalized = str(result.get("stdout", "")).lower().replace(" ", "")
    return result.get("returncode") == 0 and (
        "success:true" in normalized or "success=true" in normalized
    )


def reset_probe(scene_config: dict) -> dict:
    scene = scene_config["scenes"][0]
    calls = []
    for model_name, pose in scene_layout(scene_config, scene).items():
        result = run_probe(
            [
                "ros2",
                "service",
                "call",
                SET_POSE_SERVICE,
                SET_POSE_SERVICE_TYPE,
                set_pose_request(model_name, pose),
            ],
            timeout=COMMAND_TIMEOUT_SEC,
        )
        calls.append(
            {
                "model_name": model_name,
                "pose": pose,
                "returncode": result["returncode"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "seconds": result["seconds"],
            }
        )
        if not service_call_succeeded(result):
            break
    return {
        "scene_id": scene["scene_id"],
        "expected_entity_count": len(scene_config["objects"]),
        "completed_entity_count": len(calls),
        "calls": calls,
        "success": len(calls) == len(scene_config["objects"])
        and all(service_call_succeeded(call) for call in calls),
    }


def validate_r4_implementation() -> tuple[dict, dict, dict]:
    r3.validate_implementation()
    required_hashes = {
        CONTRACT: CONTRACT_SHA256,
        R3_IMPLEMENTATION_LOCK: R3_IMPLEMENTATION_LOCK_SHA256,
        STATIC_PREFLIGHT: STATIC_PREFLIGHT_SHA256,
        R3_BLOCKED_LIVE_PREFLIGHT: R3_BLOCKED_LIVE_PREFLIGHT_SHA256,
    }
    for path, expected in required_hashes.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"R4 frozen upstream drift: {path}")
    if not R4_IMPLEMENTATION_LOCK.is_file() or not R4_IMPLEMENTATION_AUDIT.is_file():
        raise RuntimeError("Calibration-v3 R4 implementation is not frozen")
    lock = read_json(R4_IMPLEMENTATION_LOCK)
    audit = read_json(R4_IMPLEMENTATION_AUDIT)
    if lock.get("status") != "IMPLEMENTATION_R4_FROZEN_BEFORE_LIVE_PREFLIGHT_ATTEMPT_02":
        raise RuntimeError("invalid R4 implementation lock state")
    if lock.get("capture_authorized") is not False:
        raise RuntimeError("R4 implementation lock must not authorize capture")
    if audit.get("status") != "PASS" or lock.get("implementation_audit_sha256") != sha256(R4_IMPLEMENTATION_AUDIT):
        raise RuntimeError("R4 implementation audit linkage failed")
    for name, digest in lock["source_artifact_sha256"].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"R4 frozen source drift: {name}")
    scene_config = yaml.safe_load(SCENES.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    contract = read_json(CONTRACT)
    return scene_config, gate, contract


def live_preflight_attempt_02() -> None:
    if R4_LIVE_PREFLIGHT.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 live preflight attempt 02")
    scene_config, gate, contract = validate_r4_implementation()
    static = read_json(STATIC_PREFLIGHT)
    blocked = read_json(R3_BLOCKED_LIVE_PREFLIGHT)
    nodes = run_probe(["ros2", "node", "list"])
    topics = run_probe(["ros2", "topic", "list", "-t"])
    services = run_probe(["ros2", "service", "list", "-t"])
    actions = run_probe(["ros2", "action", "list", "-t"])
    processes = run_probe(["ps", "-eo", "pid=,args="])
    topic_text = topics["stdout"]
    service_text = services["stdout"]
    node_text = nodes["stdout"]
    action_text = actions["stdout"]
    relevant_processes = [
        line.strip()
        for line in processes["stdout"].splitlines()
        if any(token in line for token in ("gz sim", "ign gazebo", "ur3_pick_place_uq_occlusion_v2.sdf"))
    ]
    topic_presence = {
        name: f"{name} [{message_type}]" in topic_text
        for name, message_type in REQUIRED_TOPICS.items()
    }
    service_present = f"{SET_POSE_SERVICE} [{SET_POSE_SERVICE_TYPE}]" in service_text
    graph_ready = (
        nodes["returncode"] == 0
        and topics["returncode"] == 0
        and services["returncode"] == 0
        and bool(node_text.strip())
    )
    world_process_valid = any(WORLD.name in line for line in relevant_processes)
    simulator_nodes_present = all(
        token in node_text for token in ("/ros_gz_bridge", "/robot_state_publisher", "/controller_manager")
    )
    trajectory_action_present = (
        "/joint_trajectory_controller/follow_joint_trajectory [control_msgs/action/FollowJointTrajectory]"
        in action_text
    )

    observations: dict[str, Any] = {
        "node_inventory": nodes,
        "topic_inventory": topics,
        "service_inventory": services,
        "action_inventory": actions,
        "relevant_processes": relevant_processes,
    }
    samples: dict[str, Any] = {}
    sample_prerequisites = all(topic_presence.values()) and service_present and graph_ready
    if sample_prerequisites:
        samples = {
            "wrist_rgb": image_probe("/wrist_camera/color/image_raw"),
            "wrist_depth": image_probe("/wrist_camera/depth/image_raw"),
            "wrist_labels": image_probe("/wrist_camera/evaluation_labels/labels_map"),
            "wrist_camera_info": camera_info_probe("/wrist_camera/color/camera_info"),
            "base_rgb": image_probe("/top_table_camera/color/image_raw"),
            "base_depth": image_probe("/top_table_camera/depth/image_raw"),
            "base_camera_info": camera_info_probe("/top_table_camera/color/camera_info"),
        }
        joint_probe, fixed_view_pass, joint_comparison = joint_state_probe(
            [float(value) for value in gate["camera"]["view_joint_pose"]]
        )
        tf_probe = run_probe(
            ["ros2", "run", "tf2_ros", "tf2_echo", gate["camera"]["base_frame"], gate["camera"]["frame"]],
            timeout=8.0,
        )
        reset = reset_probe(scene_config)
        post_reset = {
            "wrist_rgb_frame": echo_field("/wrist_camera/color/image_raw", "header.frame_id"),
            "wrist_depth_frame": echo_field("/wrist_camera/depth/image_raw", "header.frame_id"),
            "wrist_label_frame": echo_field("/wrist_camera/evaluation_labels/labels_map", "header.frame_id"),
            "base_rgb_frame": echo_field("/top_table_camera/color/image_raw", "header.frame_id"),
            "base_depth_frame": echo_field("/top_table_camera/depth/image_raw", "header.frame_id"),
        }
        observations.update(
            {
                "samples": samples,
                "joint_state": joint_probe,
                "joint_comparison": joint_comparison,
                "tf_probe": tf_probe,
                "reset_probe": reset,
                "post_reset_sensor_probe": post_reset,
            }
        )
    else:
        fixed_view_pass = False
        joint_comparison = {}
        tf_probe = {"returncode": 125, "stdout": "", "stderr": "skipped: live prerequisites failed"}
        reset = {"success": False, "calls": [], "reason": "skipped: live prerequisites failed"}
        post_reset = {}
        observations["sample_skip_reason"] = "required graph/topic/service prerequisites are not all available"

    resolution = tuple(int(value) for value in gate["camera"]["resolution"])
    width, height = resolution
    wrist_frame = str(gate["camera"]["frame"])
    base_frame = "top_table_camera_optical_frame"
    if samples:
        wrist_rgb_valid = image_matches(
            samples["wrist_rgb"], width=width, height=height,
            encodings={"rgb8", "bgr8"}, frame_id=wrist_frame,
        )
        wrist_depth_valid = image_matches(
            samples["wrist_depth"], width=width, height=height,
            encodings={"32FC1"}, frame_id=wrist_frame,
        )
        wrist_labels_valid = image_matches(
            samples["wrist_labels"], width=width, height=height,
            encodings={"rgb8"}, frame_id=wrist_frame,
        )
        wrist_intrinsics_valid = camera_info_valid(
            samples["wrist_camera_info"], width=width, height=height,
            frame_id=wrist_frame, expected_intrinsics=gate["camera"]["intrinsics"],
        )
        base_rgb_valid = image_matches(
            samples["base_rgb"], width=width, height=height,
            encodings={"rgb8", "bgr8"}, frame_id=base_frame,
        )
        base_depth_valid = image_matches(
            samples["base_depth"], width=width, height=height,
            encodings={"32FC1"}, frame_id=base_frame,
        )
        base_intrinsics_valid = camera_info_valid(
            samples["base_camera_info"], width=width, height=height,
            frame_id=base_frame, expected_intrinsics=None,
        )
        tf_valid = "Translation:" in tf_probe["stdout"] and "Rotation:" in tf_probe["stdout"]
        post_reset_valid = all(
            result.get("returncode") == 0 and parsed_field(result) == expected_frame
            for key, result in post_reset.items()
            for expected_frame in ([base_frame] if key.startswith("base_") else [wrist_frame])
        )
    else:
        wrist_rgb_valid = wrist_depth_valid = wrist_labels_valid = wrist_intrinsics_valid = False
        base_rgb_valid = base_depth_valid = base_intrinsics_valid = False
        tf_valid = post_reset_valid = False

    access = contract["access_and_sealing"]
    checks = {
        "contract_hash_unchanged": sha256(CONTRACT) == CONTRACT_SHA256,
        "r3_implementation_hash_unchanged": sha256(R3_IMPLEMENTATION_LOCK) == R3_IMPLEMENTATION_LOCK_SHA256,
        "static_preflight_pass_and_unchanged": static.get("status") == "PASS" and sha256(STATIC_PREFLIGHT) == STATIC_PREFLIGHT_SHA256,
        "r3_blocked_attempt_preserved": blocked.get("status") == "BLOCKED" and sha256(R3_BLOCKED_LIVE_PREFLIGHT) == R3_BLOCKED_LIVE_PREFLIGHT_SHA256,
        "r4_implementation_audit_pass": read_json(R4_IMPLEMENTATION_AUDIT).get("status") == "PASS",
        "ros2_graph_reachable": graph_ready,
        "gazebo_world_process_matches_lock": world_process_valid,
        "gazebo_robot_nodes_present": simulator_nodes_present,
        "ur3_trajectory_action_present": trajectory_action_present,
        "all_required_topic_types_present": all(topic_presence.values()),
        "set_pose_service_present": service_present,
        "wrist_rgb_live_valid": wrist_rgb_valid,
        "wrist_metric_depth_live_valid": wrist_depth_valid,
        "wrist_semantic_labels_live_valid": wrist_labels_valid,
        "wrist_intrinsics_match_lock": wrist_intrinsics_valid,
        "base_rgb_live_valid": base_rgb_valid,
        "base_metric_depth_live_valid": base_depth_valid,
        "base_intrinsics_valid": base_intrinsics_valid,
        "base_to_wrist_camera_tf_valid": tf_valid,
        "fixed_view_joint_pose_within_tolerance": fixed_view_pass,
        "full_scene_reset_probe_success": bool(reset.get("success")),
        "post_reset_geometry_sensor_probe_success": post_reset_valid,
        "no_capture_or_dataset_exists": not CAPTURE_ROOT.exists() and not DATASET_ROOT.exists(),
        "no_capture_authorization_exists": not CAPTURE_LOCK.exists(),
        "test_iid_ood_still_sealed": (
            access.get("test_iid") == "SEALED_AND_UNAUTHORIZED"
            and access.get("test_ood") == "SEALED_AND_UNAUTHORIZED"
            and gate["sealed"].get("gazebo_test_iid") is True
            and gate["sealed"].get("gazebo_test_ood") is True
        ),
        "disk_free_over_2_gib": shutil.disk_usage(ROOT).free > 2 * 1024**3,
    }
    status = "PASS" if all(checks.values()) else "BLOCKED"
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "implementation_revision": "r4",
        "attempt": 2,
        "status": status,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_lock_sha256": sha256(CONTRACT),
        "r3_implementation_lock_sha256": sha256(R3_IMPLEMENTATION_LOCK),
        "r4_implementation_lock_sha256": sha256(R4_IMPLEMENTATION_LOCK),
        "static_preflight_sha256": sha256(STATIC_PREFLIGHT),
        "prior_blocked_live_preflight_sha256": sha256(R3_BLOCKED_LIVE_PREFLIGHT),
        "topic_presence": topic_presence,
        "checks": checks,
        "tolerances": {
            "joint_position_absolute_rad": JOINT_POSITION_TOLERANCE_RAD,
            "wrist_intrinsic_absolute": INTRINSIC_ABSOLUTE_TOLERANCE,
        },
        "storage": {
            "free_bytes": shutil.disk_usage(ROOT).free,
            "free_gib": shutil.disk_usage(ROOT).free / 2**30,
        },
        "observations": observations,
        "capture_authorized": False,
        "next_authorized_action_if_pass": "Create a separate immutable Calibration-v3 capture-authorization lock.",
        "next_authorized_action_if_blocked": "Stop without capture; preserve this artifact and preregister any retry repair append-only.",
    }
    write_json(R4_LIVE_PREFLIGHT, report)
    print(json.dumps({"status": status, "artifact": str(R4_LIVE_PREFLIGHT), "checks": checks}, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "preflight-live"))
    command = parser.parse_args().command
    if command == "validate":
        validate_r4_implementation()
        print("PASS: frozen Calibration-v3 implementation revision R4")
    else:
        live_preflight_attempt_02()


if __name__ == "__main__":
    main()
