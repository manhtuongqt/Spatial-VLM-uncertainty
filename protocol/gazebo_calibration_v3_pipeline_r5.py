#!/usr/bin/env python3
"""Append-only Calibration-v3 R5 preflight, authorization, and capture gate.

R5 repairs ROS Python-path propagation and binds a capture lock schema that is
accepted by the already-frozen capture node.  Scientific method, population,
metrics, and gates remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import gazebo_calibration_v3_pipeline_r4 as r4


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_calibration_v3"
CONTRACT = ROOT / "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json"
R4_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R4.json"
R5_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R5.json"
R5_AUDIT = ROOT / "protocol/gazebo_calibration_v3_implementation_r5_audit.json"
STATIC_PREFLIGHT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_STATIC.json"
ATTEMPT_01 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE.json"
ATTEMPT_02 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE_ATTEMPT_02.json"
ATTEMPT_03 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE_ATTEMPT_03.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_calibration_v3_capture_compatibility_lock.json"
CAPTURE_ROOT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/capture_attempt_01"
DATASET_ROOT = ROOT / "datasets/Gazebo_calibration_v3"
SCENES = ROOT / "ur3/ur3_perception/config/gazebo_calibration_v3_scenes.yaml"
ANNOTATIONS = ROOT / "ur3/ur3_perception/config/gazebo_calibration_v3_annotations.yaml"
GATE = ROOT / "ur3/ur3_perception/config/gazebo_calibration_v3_gate.yaml"
WORLD = ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf"

CONTRACT_SHA256 = "99b430b1c55e1762f8460db4e8134b4a8782403e466d71ea86ea3da847bfc841"
R4_LOCK_SHA256 = "36b97adae26a5e4927bfe3c4358836056a4564ff7a89371c7a76e7e343393644"
STATIC_SHA256 = "5ccba3ee03efd3900048a51a3ec33185a8da26266d8668fe4c9a1d1c492e6f77"
ATTEMPT_01_SHA256 = "25f9347fccf195c3d39da18559150e2df16018cba73dada3c54aa4b289b41ea0"
ATTEMPT_02_SHA256 = "53d1a2c70fa4fc2f2ccf25176d24381bd1b0b80ca1743c92597feb7f84d90686"
ROS_SITE_PACKAGES = "/opt/ros/humble/lib/python3.10/site-packages"
COMMAND_TIMEOUT_SEC = 12.0

CAPTURE_REQUIRED_SOURCES = {
    "RoboRefer/API/api.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_capture.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_runner.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_validate.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_evaluator.py",
    "ur3/ur3_perception/scripts/roborefer_grounder.py",
    "ur3/ur3_perception/scripts/spatial_point_utils.py",
    "ur3/ur3_perception/scripts/move_camera_to_view.py",
    "ur3/ur3_perception/launch/roborefer_pilot_capture.launch.py",
    "ur3/ur3_perception/config/roborefer_pilot_v0_scenes.yaml",
    "ur3/ur3_perception/config/roborefer_pilot_v0_gate.yaml",
    "ur3/ur3_perception/config/roborefer_pilot_v0_annotations.yaml",
    "ur3/ur3_perception/schemas/roborefer_pilot_prediction.schema.json",
    "ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro",
    "ur3/ur3_moveit_control/launch/ur3_susgrip_sim.launch.py",
    "ur3/ur_simulation_gz/worlds/ur3_pick_place.sdf",
    "ur3/ur_simulation_gz/launch/ur_sim_control.launch.py",
}


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


def ros_environment() -> dict[str, str]:
    env = os.environ.copy()
    existing = [item for item in env.get("PYTHONPATH", "").split(":") if item]
    required = [str(ROOT / "protocol"), ROS_SITE_PACKAGES]
    for item in existing:
        if item not in required:
            required.append(item)
    env["PYTHONPATH"] = ":".join(required)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("ROS_LOCALHOST_ONLY", "1")
    env.setdefault("IGN_PARTITION", "ur3_roborefer_pilot_v0_local")
    return env


def run_probe(command: list[str], timeout: float = COMMAND_TIMEOUT_SEC) -> dict:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=ros_environment(),
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


def configure_r4_helpers() -> None:
    r4.run_probe = run_probe


def validate_r5_implementation() -> tuple[dict, dict, dict]:
    r4.validate_r4_implementation()
    fixed = {
        CONTRACT: CONTRACT_SHA256,
        R4_LOCK: R4_LOCK_SHA256,
        STATIC_PREFLIGHT: STATIC_SHA256,
        ATTEMPT_01: ATTEMPT_01_SHA256,
        ATTEMPT_02: ATTEMPT_02_SHA256,
    }
    for path, digest in fixed.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Calibration-v3 R5 upstream drift: {path}")
    if not R5_LOCK.is_file() or not R5_AUDIT.is_file():
        raise RuntimeError("Calibration-v3 R5 implementation is not frozen")
    lock = read_json(R5_LOCK)
    audit = read_json(R5_AUDIT)
    if lock.get("status") != "IMPLEMENTATION_R5_FROZEN_BEFORE_LIVE_PREFLIGHT_ATTEMPT_03":
        raise RuntimeError("invalid Calibration-v3 R5 implementation state")
    if lock.get("capture_authorized") is not False:
        raise RuntimeError("R5 implementation lock must not authorize capture")
    if audit.get("status") != "PASS" or lock.get("implementation_audit_sha256") != sha256(R5_AUDIT):
        raise RuntimeError("Calibration-v3 R5 audit linkage failed")
    for name, digest in lock["source_artifact_sha256"].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Calibration-v3 R5 source drift: {name}")
    scenes = yaml.safe_load(SCENES.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    contract = read_json(CONTRACT)
    return scenes, gate, contract


def live_preflight_attempt_03() -> None:
    if ATTEMPT_03.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 live preflight attempt 03")
    scene_config, gate, contract = validate_r5_implementation()
    configure_r4_helpers()
    nodes = run_probe(["ros2", "node", "list"])
    topics = run_probe(["ros2", "topic", "list", "-t"])
    services = run_probe(["ros2", "service", "list", "-t"])
    actions = run_probe(["ros2", "action", "list", "-t"])
    processes = run_probe(["ps", "-eo", "pid=,args="])
    node_text = nodes["stdout"]
    topic_text = topics["stdout"]
    service_text = services["stdout"]
    action_text = actions["stdout"]
    relevant_processes = [
        line.strip()
        for line in processes["stdout"].splitlines()
        if any(token in line for token in ("gz sim", "ign gazebo", WORLD.name))
    ]
    topic_presence = {
        name: f"{name} [{message_type}]" in topic_text
        for name, message_type in r4.REQUIRED_TOPICS.items()
    }
    service_present = f"{r4.SET_POSE_SERVICE} [{r4.SET_POSE_SERVICE_TYPE}]" in service_text
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
        "effective_ros_pythonpath": ros_environment()["PYTHONPATH"],
        "node_inventory": nodes,
        "topic_inventory": topics,
        "service_inventory": services,
        "action_inventory": actions,
        "relevant_processes": relevant_processes,
    }
    samples: dict[str, Any] = {}
    prerequisites = graph_ready and all(topic_presence.values()) and service_present
    if prerequisites:
        samples = {
            "wrist_rgb": r4.image_probe("/wrist_camera/color/image_raw"),
            "wrist_depth": r4.image_probe("/wrist_camera/depth/image_raw"),
            "wrist_labels": r4.image_probe("/wrist_camera/evaluation_labels/labels_map"),
            "wrist_camera_info": r4.camera_info_probe("/wrist_camera/color/camera_info"),
            "base_rgb": r4.image_probe("/top_table_camera/color/image_raw"),
            "base_depth": r4.image_probe("/top_table_camera/depth/image_raw"),
            "base_camera_info": r4.camera_info_probe("/top_table_camera/color/camera_info"),
        }
        joint_probe, fixed_view_pass, joint_comparison = r4.joint_state_probe(
            [float(value) for value in gate["camera"]["view_joint_pose"]]
        )
        tf_probe = run_probe(
            ["ros2", "run", "tf2_ros", "tf2_echo", gate["camera"]["base_frame"], gate["camera"]["frame"]],
            timeout=8.0,
        )
        reset = r4.reset_probe(scene_config)
        post_reset = {
            "wrist_rgb_frame": r4.echo_field("/wrist_camera/color/image_raw", "header.frame_id"),
            "wrist_depth_frame": r4.echo_field("/wrist_camera/depth/image_raw", "header.frame_id"),
            "wrist_label_frame": r4.echo_field("/wrist_camera/evaluation_labels/labels_map", "header.frame_id"),
            "base_rgb_frame": r4.echo_field("/top_table_camera/color/image_raw", "header.frame_id"),
            "base_depth_frame": r4.echo_field("/top_table_camera/depth/image_raw", "header.frame_id"),
        }
        observations.update({
            "samples": samples,
            "joint_state": joint_probe,
            "joint_comparison": joint_comparison,
            "tf_probe": tf_probe,
            "reset_probe": reset,
            "post_reset_sensor_probe": post_reset,
        })
    else:
        fixed_view_pass = False
        tf_probe = {"returncode": 125, "stdout": "", "stderr": "skipped: live prerequisites failed"}
        reset = {"success": False, "calls": [], "reason": "skipped: live prerequisites failed"}
        post_reset = {}
        observations["sample_skip_reason"] = "required graph/topic/service prerequisites are not all available"

    width, height = (int(value) for value in gate["camera"]["resolution"])
    wrist_frame = str(gate["camera"]["frame"])
    base_frame = "top_table_camera_optical_frame"
    if samples:
        wrist_rgb_valid = r4.image_matches(samples["wrist_rgb"], width=width, height=height, encodings={"rgb8", "bgr8"}, frame_id=wrist_frame)
        wrist_depth_valid = r4.image_matches(samples["wrist_depth"], width=width, height=height, encodings={"32FC1"}, frame_id=wrist_frame)
        wrist_labels_valid = r4.image_matches(samples["wrist_labels"], width=width, height=height, encodings={"rgb8"}, frame_id=wrist_frame)
        wrist_intrinsics_valid = r4.camera_info_valid(
            samples["wrist_camera_info"], width=width, height=height,
            frame_id=wrist_frame, expected_intrinsics=gate["camera"]["intrinsics"],
        )
        base_rgb_valid = r4.image_matches(samples["base_rgb"], width=width, height=height, encodings={"rgb8", "bgr8"}, frame_id=base_frame)
        base_depth_valid = r4.image_matches(samples["base_depth"], width=width, height=height, encodings={"32FC1"}, frame_id=base_frame)
        base_intrinsics_valid = r4.camera_info_valid(
            samples["base_camera_info"], width=width, height=height,
            frame_id=base_frame, expected_intrinsics=None,
        )
        tf_valid = "Translation:" in tf_probe["stdout"] and "Rotation:" in tf_probe["stdout"]
        post_reset_valid = all(
            result.get("returncode") == 0
            and r4.parsed_field(result) == (base_frame if key.startswith("base_") else wrist_frame)
            for key, result in post_reset.items()
        )
    else:
        wrist_rgb_valid = wrist_depth_valid = wrist_labels_valid = wrist_intrinsics_valid = False
        base_rgb_valid = base_depth_valid = base_intrinsics_valid = False
        tf_valid = post_reset_valid = False

    access = contract["access_and_sealing"]
    checks = {
        "contract_hash_unchanged": sha256(CONTRACT) == CONTRACT_SHA256,
        "r4_implementation_hash_unchanged": sha256(R4_LOCK) == R4_LOCK_SHA256,
        "static_preflight_pass_and_unchanged": read_json(STATIC_PREFLIGHT).get("status") == "PASS" and sha256(STATIC_PREFLIGHT) == STATIC_SHA256,
        "attempt_01_preserved_blocked": read_json(ATTEMPT_01).get("status") == "BLOCKED" and sha256(ATTEMPT_01) == ATTEMPT_01_SHA256,
        "attempt_02_preserved_blocked": read_json(ATTEMPT_02).get("status") == "BLOCKED" and sha256(ATTEMPT_02) == ATTEMPT_02_SHA256,
        "r5_implementation_audit_pass": read_json(R5_AUDIT).get("status") == "PASS",
        "ros2_graph_reachable": graph_ready,
        "ros_pythonpath_preserves_ros2cli": ROS_SITE_PACKAGES in ros_environment()["PYTHONPATH"],
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
        "implementation_revision": "r5",
        "attempt": 3,
        "status": status,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_lock_sha256": sha256(CONTRACT),
        "r4_implementation_lock_sha256": sha256(R4_LOCK),
        "r5_implementation_lock_sha256": sha256(R5_LOCK),
        "static_preflight_sha256": sha256(STATIC_PREFLIGHT),
        "prior_live_preflight_sha256": {
            "attempt_01": sha256(ATTEMPT_01),
            "attempt_02": sha256(ATTEMPT_02),
        },
        "topic_presence": topic_presence,
        "checks": checks,
        "tolerances": {
            "joint_position_absolute_rad": r4.JOINT_POSITION_TOLERANCE_RAD,
            "wrist_intrinsic_absolute": r4.INTRINSIC_ABSOLUTE_TOLERANCE,
        },
        "storage": {
            "free_bytes": shutil.disk_usage(ROOT).free,
            "free_gib": shutil.disk_usage(ROOT).free / 2**30,
        },
        "observations": observations,
        "capture_authorized": False,
        "next_authorized_action_if_pass": "Create the separate immutable Calibration-v3 capture-authorization lock.",
        "next_authorized_action_if_blocked": "Stop without capture and preserve this artifact.",
    }
    write_json(ATTEMPT_03, report)
    print(json.dumps({"status": status, "artifact": str(ATTEMPT_03), "checks": checks}, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


def authorize_capture() -> None:
    scenes, _, contract = validate_r5_implementation()
    if CAPTURE_LOCK.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 capture authorization")
    live = read_json(ATTEMPT_03)
    if live.get("status") != "PASS" or live.get("capture_authorized") is not False:
        raise RuntimeError("live preflight attempt 03 must PASS before capture authorization")
    if CAPTURE_ROOT.exists() or DATASET_ROOT.exists():
        raise RuntimeError("Calibration-v3 downstream data exists before capture authorization")
    r5_lock = read_json(R5_LOCK)
    source_names = set(r5_lock["source_artifact_sha256"]) | CAPTURE_REQUIRED_SOURCES | {
        "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R5.json",
        "protocol/gazebo_calibration_v3_implementation_r5_audit.json",
        "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_STATIC.json",
        "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE.json",
        "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE_ATTEMPT_02.json",
        "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE_ATTEMPT_03.json",
        "ur3/ur3_perception/config/gazebo_calibration_v3_scenes.yaml",
        "ur3/ur3_perception/config/gazebo_calibration_v3_annotations.yaml",
        "ur3/ur3_perception/config/gazebo_calibration_v3_gate.yaml",
        "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
    }
    source_hashes = {name: sha256(ROOT / name) for name in sorted(source_names)}
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "authorization_status": "CAPTURE_AUTHORIZED_AFTER_STATIC_AND_LIVE_PREFLIGHT_PASS",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "parent_contract_lock_sha256": sha256(CONTRACT),
        "implementation_lock_sha256": sha256(R5_LOCK),
        "implementation_audit_sha256": sha256(R5_AUDIT),
        "static_preflight_sha256": sha256(STATIC_PREFLIGHT),
        "live_preflight_sha256": sha256(ATTEMPT_03),
        "capture_attempts_authorized": 1,
        "expected_parent_families": len(scenes["scenes"]),
        "source_artifact_sha256": source_hashes,
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
    print(json.dumps({"status": payload["authorization_status"], "sha256": sha256(CAPTURE_LOCK)}, indent=2, sort_keys=True))


def verify_capture_authorization() -> None:
    validate_r5_implementation()
    lock = read_json(CAPTURE_LOCK)
    if lock.get("status") != "LOCKED_BEFORE_CAPTURE_AND_INFERENCE":
        raise RuntimeError("capture lock schema is not accepted by the frozen capture node")
    if lock.get("authorization_status") != "CAPTURE_AUTHORIZED_AFTER_STATIC_AND_LIVE_PREFLIGHT_PASS":
        raise RuntimeError("Calibration-v3 capture is not authorized")
    if lock.get("parent_contract_lock_sha256") != sha256(CONTRACT):
        raise RuntimeError("capture/contract linkage failed")
    if lock.get("implementation_lock_sha256") != sha256(R5_LOCK):
        raise RuntimeError("capture/R5 implementation linkage failed")
    if lock.get("live_preflight_sha256") != sha256(ATTEMPT_03) or read_json(ATTEMPT_03).get("status") != "PASS":
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
    parser.add_argument("command", choices=("validate", "preflight-live", "authorize-capture", "capture"))
    command = parser.parse_args().command
    if command == "validate":
        validate_r5_implementation()
        print("PASS: frozen Calibration-v3 implementation revision R5")
    elif command == "preflight-live":
        live_preflight_attempt_03()
    elif command == "authorize-capture":
        authorize_capture()
    else:
        capture()


if __name__ == "__main__":
    main()
