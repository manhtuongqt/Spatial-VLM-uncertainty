#!/usr/bin/env python3
"""Append-only fixed-camera TF repair for Calibration-v6 live attempt 03.

Only the simulation fixed-camera transform is added.  No capture, inference,
calibrator fitting, Test access, or robot policy is implemented here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import gazebo_calibration_v3_pipeline_r4 as helpers
import gazebo_calibration_v3_pipeline_r6 as parsers
import gazebo_calibration_v6_pipeline_r2 as r2


ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf"
LAUNCH = ROOT / "ur3/ur_simulation_gz/launch/ur_sim_control.launch.py"
R2 = ROOT / "protocol/gazebo_calibration_v6_pipeline_r2.py"
R2_LOCK = r2.AMENDMENT
R2_ENV = r2.ENV_PREFLIGHT
ATTEMPT_02 = r2.ATTEMPT_02
ATTEMPT_02_FAILURE = ROOT / "protocol/GAZEBO_CALIBRATION_V6_LIVE_ATTEMPT_02_FAILURE_LOCK.json"
R3_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V6_IMPLEMENTATION_AMENDMENT_R3_LOCK.json"
R3_PRECHECK = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/R3_FIXED_CAMERA_TF_PREFLIGHT.json"
ATTEMPT_03 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/PREFLIGHT_LIVE_ATTEMPT_03.json"
PARENT = "base_link"
CHILD = "top_table_camera_optical_frame"
POSITION_TOL_M = 1e-6
ANGLE_TOL_RAD = 1e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def multiply(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def normalize(q: tuple[float, ...]) -> tuple[float, ...]:
    length = math.sqrt(sum(value * value for value in q))
    return tuple(value / length for value in q)


def from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
    qx = (math.sin(roll / 2), 0.0, 0.0, math.cos(roll / 2))
    qy = (0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2))
    qz = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    return normalize(multiply(multiply(qz, qy), qx))


def rotate(q: tuple[float, ...], vector: tuple[float, ...]) -> tuple[float, ...]:
    product = multiply(multiply(q, (*vector, 0.0)), (-q[0], -q[1], -q[2], q[3]))
    return product[:3]


def camera_contract() -> dict[str, Any]:
    world = ET.parse(WORLD).getroot()
    model = world.find("./world/model[@name='top_table_camera']")
    if model is None or model.findtext("static") != "true":
        raise RuntimeError("locked SDF fixed-camera model is absent or non-static")
    sensor = model.find("./link[@name='top_table_camera_link']/sensor[@name='top_table_rgbd']")
    if sensor is None or sensor.get("type") != "rgbd_camera":
        raise RuntimeError("locked SDF RGB-D sensor is absent")
    pose = tuple(float(v) for v in (model.findtext("pose") or "").split())
    if len(pose) != 6:
        raise RuntimeError("SDF fixed-camera pose is not six-dimensional")
    if sensor.findtext("camera/optical_frame_id") != CHILD or sensor.findtext("gz_frame_id") != CHILD:
        raise RuntimeError("SDF camera frame IDs differ from the requested optical frame")
    if sensor.find("pose") is not None or model.find("link/pose") is not None:
        raise RuntimeError("unhandled non-identity camera link/sensor offset")
    # Gazebo camera body: +X forward, +Y left, +Z up. ROS optical:
    # +Z forward, +X image-right, +Y image-down. This fixed body->optical
    # rotation is Rz(-pi/2) Rx(-pi/2), composed after the SDF model pose.
    body_q = from_rpy(*pose[3:])
    optical_q = from_rpy(-math.pi / 2, 0.0, -math.pi / 2)
    q = normalize(multiply(body_q, optical_q))
    basis = {
        "optical_x_in_base": rotate(q, (1.0, 0.0, 0.0)),
        "optical_y_in_base": rotate(q, (0.0, 1.0, 0.0)),
        "optical_z_in_base": rotate(q, (0.0, 0.0, 1.0)),
    }
    # The UR model is spawned without pose flags by the locked launch file;
    # the robot base_link is thus coincident with world at initialization.
    launch_text = LAUNCH.read_text(encoding="utf-8")
    spawn_block = launch_text.split("gz_spawn_entity = Node(", 1)[1].split("gz_launch_description_with_gui", 1)[0]
    if any(flag in spawn_block for flag in ('"-x"', '"-y"', '"-z"', '"-R"', '"-P"', '"-Y"')):
        raise RuntimeError("robot spawn now has a world pose offset")
    return {
        "sdf_pose_xyz_rpy": pose,
        "translation_xyz_m": pose[:3],
        "rotation_xyzw": q,
        "body_to_optical_rpy": (-math.pi / 2, 0.0, -math.pi / 2),
        "basis": basis,
        "parent_frame": PARENT,
        "child_frame": CHILD,
        "world_to_base_assumption": "UR spawn has no pose flags; base_link coincides with world origin",
    }


def static_command(contract: dict[str, Any]) -> list[str]:
    xyz = contract["translation_xyz_m"]
    q = contract["rotation_xyzw"]
    return [
        "ros2", "run", "tf2_ros", "static_transform_publisher",
        "--x", repr(xyz[0]), "--y", repr(xyz[1]), "--z", repr(xyz[2]),
        "--qx", repr(q[0]), "--qy", repr(q[1]), "--qz", repr(q[2]), "--qw", repr(q[3]),
        "--frame-id", PARENT, "--child-frame-id", CHILD,
    ]


def verify_unchanged() -> None:
    r2.validate_amendment()
    locked = {
        r2.DATA_DESIGN: "850921e200d921592128c76e11c8deeb5a7d6177ea315f1ff8d286f873084b6d",
        r2.BASE_IMPLEMENTATION: "89d3416e7c870e635c3f6277a6dbbc2d5d12891c23f3a686c9a4fe2d4f7b821d",
        R2_LOCK: "4ba0765cd8f68eccfd11fd905db4048856eec130652dd62c786c6b27bae4846b",
        R2_ENV: "3eb820ca14faf3f1ede3e252976cedf79824610cceee9a4e7eaa749018e8c9c3",
        ATTEMPT_02: "b426317c8e10190d5704dd78e832b729110412bb4fea7ba5d7e69a6d9d527381",
        WORLD: "6ee5cd6ede77d3fc8247557746ca25f0c03d0e08e0156d0360059d8e691c1193",
    }
    for path, digest in locked.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"frozen predecessor drift: {path}")
    if read_json(ATTEMPT_02).get("status") != "BLOCKED":
        raise RuntimeError("attempt 02 must remain BLOCKED")
    if read_json(ATTEMPT_02_FAILURE).get("blocked_check") != "base_tf_valid":
        raise RuntimeError("attempt 02 failure lock changed")
    if any(path.exists() for path in (r2.CAPTURE_ROOT, r2.CALIBRATION_CAPTURE_ROOT, r2.DATASET_ROOT, r2.CAPTURE_LOCK)):
        raise RuntimeError("unexpected capture, dataset, or authorization")


def freeze() -> None:
    if R3_LOCK.exists():
        raise FileExistsError("R3 amendment is already frozen")
    if R3_PRECHECK.exists() or ATTEMPT_03.exists():
        raise RuntimeError("R3 downstream artifact exists before freeze")
    verify_unchanged()
    contract = camera_contract()
    basis = contract["basis"]
    expected_basis = {
        "optical_x_in_base": (0.0, -1.0, 0.0),
        "optical_y_in_base": (-1.0, 0.0, 0.0),
        "optical_z_in_base": (0.0, 0.0, -1.0),
    }
    checks = {
        "frozen_predecessors_unchanged": True,
        "attempts_01_02_immutable": True,
        "sdf_model_static_and_frame_ids_match": True,
        "optical_basis_matches_sdf_downward_view": all(
            math.dist(basis[name], expected) < 1e-9 for name, expected in expected_basis.items()
        ),
        "pilot_calibration_manifests_unchanged": all(
            sha256(path) == read_json(R2_LOCK)["source_artifact_sha256"][str(path.relative_to(ROOT))]
            for path in (r2.PILOT_MANIFEST, r2.CALIBRATION_MANIFEST)
        ),
        "no_capture_dataset_or_authorization": True,
        "test_robot_sealed": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"R3 static proof failed: {checks}")
    sources = [
        Path(__file__).resolve(), R2, R2_LOCK, R2_ENV, ATTEMPT_02, ATTEMPT_02_FAILURE,
        r2.ATTEMPT_01, r2.BASE_IMPLEMENTATION, r2.STATIC_PREFLIGHT,
        r2.DATA_DESIGN, r2.CURRENT_DECISION, WORLD, LAUNCH,
        r2.PILOT_MANIFEST, r2.CALIBRATION_MANIFEST, r2.SCENES, r2.GATE,
    ]
    payload = {
        "schema_version": 1,
        "protocol_id": "gazebo_calibration_v6",
        "status": "IMPLEMENTATION_AMENDMENT_R3_FROZEN_BEFORE_TF_PREFLIGHT_AND_ATTEMPT_03",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "fixed-camera static TF publication for simulation-only live preflight",
        "fixed_camera_tf": contract,
        "publisher_command": static_command(contract),
        "checks": checks,
        "source_artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "unchanged": [
            "Calibration-v6 data design and pilot 32/calibration 128 manifests, IDs, seeds, layouts",
            "B0, frozen spatial-risk v2 and 12 ordered features",
            "visibility thresholds, unsafe definition, metrics, calibrator, optimizer and scientific gates",
            "attempts 01/02, their failure locks, and existing workflow decision",
            "Test-IID/OOD and robot policy sealing",
        ],
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
    write_new(R3_LOCK, payload)
    print(json.dumps({"status": payload["status"], "sha256": sha256(R3_LOCK), "checks": checks}, indent=2))


def validate() -> dict[str, Any]:
    verify_unchanged()
    value = read_json(R3_LOCK)
    if value.get("status") != "IMPLEMENTATION_AMENDMENT_R3_FROZEN_BEFORE_TF_PREFLIGHT_AND_ATTEMPT_03":
        raise RuntimeError("R3 lock status invalid")
    for name, digest in value["source_artifact_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"R3 frozen source drift: {name}")
    if value["fixed_camera_tf"] != json.loads(json.dumps(camera_contract())):
        raise RuntimeError("derived SDF camera transform changed")
    if not all(value["checks"].values()):
        raise RuntimeError("R3 lock has failed checks")
    return value


def lookup_tf(timeout_sec: float = 8.0) -> dict[str, Any]:
    import rclpy
    from rclpy.duration import Duration
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    rclpy.init(args=None)
    node = rclpy.create_node("calibration_v6_r3_tf_probe")
    buffer = Buffer()
    listener = TransformListener(buffer, node, spin_thread=False)
    started = time.monotonic()
    observation: dict[str, Any] = {"rclpy_path": rclpy.__file__}
    try:
        while time.monotonic() - started < timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.2)
            if buffer.can_transform(PARENT, CHILD, Time(), timeout=Duration(seconds=0.0)):
                result = buffer.lookup_transform(PARENT, CHILD, Time())
                t, q = result.transform.translation, result.transform.rotation
                observation.update({
                    "parent_frame": result.header.frame_id,
                    "child_frame": result.child_frame_id,
                    "translation_xyz_m": (t.x, t.y, t.z),
                    "rotation_xyzw": (q.x, q.y, q.z, q.w),
                    "resolved": True,
                })
                break
        else:
            observation["resolved"] = False
    finally:
        del listener
        node.destroy_node()
        rclpy.shutdown()
    return observation


def exact_tf_valid(observation: dict[str, Any], expected: dict[str, Any]) -> bool:
    if not observation.get("resolved") or observation.get("parent_frame") != PARENT or observation.get("child_frame") != CHILD:
        return False
    translation_error = math.dist(observation["translation_xyz_m"], expected["translation_xyz_m"])
    qa = normalize(tuple(observation["rotation_xyzw"]))
    qb = normalize(tuple(expected["rotation_xyzw"]))
    dot = min(1.0, abs(sum(a * b for a, b in zip(qa, qb))))
    observation["translation_error_m"] = translation_error
    observation["rotation_error_rad"] = 2.0 * math.acos(dot)
    return translation_error <= POSITION_TOL_M and observation["rotation_error_rad"] <= ANGLE_TOL_RAD


def tf_precheck() -> None:
    if R3_PRECHECK.exists():
        raise FileExistsError("R3 TF precheck already exists")
    value = validate()
    import rclpy

    r2.configure_helpers()
    tf = lookup_tf()
    info = parsers.camera_info_probe("/top_table_camera/color/camera_info")
    rgb = helpers.image_probe("/top_table_camera/color/image_raw")
    depth = helpers.image_probe("/top_table_camera/depth/image_raw")
    processes = r2.run_probe(["ps", "-eo", "comm=,args="])
    checks = {
        "r3_lock_unchanged": True,
        "rclpy_from_ros_humble": str(rclpy.__file__).startswith(r2.ROS_LOCAL_DIST_PACKAGES),
        "base_tf_matches_sdf_and_optical_convention": exact_tf_valid(tf, value["fixed_camera_tf"]),
        "camera_info_optical_frame_matches": info.get("values", {}).get("header.frame_id") == CHILD,
        "rgb_optical_frame_matches": rgb.get("values", {}).get("header.frame_id") == CHILD,
        "metric_depth_optical_frame_matches": depth.get("values", {}).get("header.frame_id") == CHILD and depth.get("values", {}).get("encoding") == "32FC1",
        "locked_world_running": any(WORLD.name in line and "ign gazebo" in line for line in processes["stdout"].splitlines()),
        "static_publisher_running": any("static_transform_publisher" in line and CHILD in line for line in processes["stdout"].splitlines()),
        "attempt_03_absent": not ATTEMPT_03.exists(),
        "no_capture_dataset_or_authorization": not any(path.exists() for path in (r2.CAPTURE_ROOT, r2.CALIBRATION_CAPTURE_ROOT, r2.DATASET_ROOT, r2.CAPTURE_LOCK)),
    }
    report = {
        "schema_version": 1,
        "protocol_id": "gazebo_calibration_v6",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "independent ROS environment and fixed-camera TF check; no capture",
        "checks": checks,
        "observed_tf": tf,
        "camera_info": info,
        "base_rgb": rgb,
        "base_depth": depth,
        "r3_amendment_sha256": sha256(R3_LOCK),
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
    write_new(R3_PRECHECK, report)
    print(json.dumps({"status": report["status"], "checks": checks, "tf": tf}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


def live_preflight() -> None:
    if ATTEMPT_03.exists():
        raise FileExistsError("refusing to overwrite live attempt 03")
    value = validate()
    precheck = read_json(R3_PRECHECK)
    if precheck["status"] != "PASS" or precheck["r3_amendment_sha256"] != sha256(R3_LOCK):
        raise RuntimeError("independent R3 TF precheck did not PASS under this lock")
    r2.configure_helpers()
    gate = yaml.safe_load(r2.GATE.read_text(encoding="utf-8"))
    scenes = yaml.safe_load(r2.SCENES.read_text(encoding="utf-8"))
    probes = {name: r2.run_probe(["ros2", name, "list"] + (["-t"] if name != "node" else [])) for name in ("topic", "node", "service", "action")}
    processes = r2.run_probe(["ps", "-eo", "comm=,args="])
    topics, nodes, services, actions = (probes[name] for name in ("topic", "node", "service", "action"))
    required_topics = {name: f"{name} [{kind}]" in topics["stdout"] for name, kind in helpers.REQUIRED_TOPICS.items()}
    checks: dict[str, bool] = {
        "r3_amendment_locked_and_unchanged": True,
        "independent_tf_environment_precheck_pass_and_bound": precheck["status"] == "PASS" and precheck["r3_amendment_sha256"] == sha256(R3_LOCK),
        "attempts_01_02_preserved_blocked": read_json(r2.ATTEMPT_01)["status"] == "BLOCKED" and read_json(ATTEMPT_02)["status"] == "BLOCKED",
        "existing_decision_preserved": sha256(r2.CURRENT_DECISION) == r2.EXPECTED_FROZEN_SHA256[r2.CURRENT_DECISION],
        "ros_graph_probes_returncode_zero": all(item["returncode"] == 0 for item in probes.values()),
        "all_required_topic_types_present": all(required_topics.values()),
        "simulator_nodes_present": all(token in nodes["stdout"] for token in ("/ros_gz_bridge", "/robot_state_publisher", "/controller_manager")),
        "set_pose_service_present": helpers.SET_POSE_SERVICE in services["stdout"],
        "trajectory_action_present": "/joint_trajectory_controller/follow_joint_trajectory" in actions["stdout"],
        "locked_world_running": any(WORLD.name in line and "ign gazebo" in line for line in processes["stdout"].splitlines()),
        "static_publisher_running": any("static_transform_publisher" in line and CHILD in line for line in processes["stdout"].splitlines()),
        "no_capture_or_dataset_exists": not any(path.exists() for path in (r2.CAPTURE_ROOT, r2.CALIBRATION_CAPTURE_ROOT, r2.DATASET_ROOT)),
        "no_capture_authorization_exists": not r2.CAPTURE_LOCK.exists(),
    }
    observations: dict[str, Any] = {"ros_graph": probes, "required_topic_presence": required_topics}
    prerequisites = all(checks[name] for name in ("ros_graph_probes_returncode_zero", "all_required_topic_types_present", "set_pose_service_present"))
    if prerequisites:
        width, height = (int(v) for v in gate["camera"]["resolution"])
        for prefix, topic, frame in (("wrist", "/wrist_camera", gate["camera"]["frame"]), ("base", "/top_table_camera", CHILD)):
            rgb = helpers.image_probe(topic + "/color/image_raw")
            depth = helpers.image_probe(topic + "/depth/image_raw")
            info = parsers.camera_info_probe(topic + "/color/camera_info")
            tf = lookup_tf() if prefix == "base" else r2.run_probe(["ros2", "run", "tf2_ros", "tf2_echo", PARENT, frame], timeout=8.0)
            observations[prefix] = {"rgb": rgb, "depth": depth, "camera_info": info, "tf": tf}
            checks[prefix + "_rgb_live_valid"] = helpers.image_matches(rgb, width=width, height=height, encodings={"rgb8", "bgr8"}, frame_id=frame)
            checks[prefix + "_metric_depth_live_valid"] = helpers.image_matches(depth, width=width, height=height, encodings={"32FC1"}, frame_id=frame)
            checks[prefix + "_intrinsics_valid"] = helpers.camera_info_valid(info, width=width, height=height, frame_id=frame, expected_intrinsics=gate["camera"]["intrinsics"] if prefix == "wrist" else None)
            checks[prefix + "_tf_valid"] = exact_tf_valid(tf, value["fixed_camera_tf"]) if prefix == "base" else "Translation:" in tf["stdout"] and "Rotation:" in tf["stdout"]
        labels = helpers.image_probe("/wrist_camera/evaluation_labels/labels_map")
        observations["wrist_semantic_labels"] = labels
        checks["wrist_semantic_labels_live_valid"] = helpers.image_matches(labels, width=width, height=height, encodings={"rgb8"}, frame_id=gate["camera"]["frame"])
        joint, fixed_pose, comparison = parsers.joint_state_probe([float(v) for v in gate["camera"]["view_joint_pose"]])
        observations["joint_state"] = joint
        observations["joint_comparison"] = comparison
        checks["fixed_view_joint_pose_within_tolerance"] = fixed_pose
        reset = helpers.reset_probe(scenes)
        observations["reset_probe"] = reset
        checks["full_scene_reset_probe_success"] = bool(reset.get("success"))
        post = helpers.image_probe("/wrist_camera/depth/image_raw")
        observations["post_reset_depth"] = post
        checks["post_reset_geometry_sensor_probe_success"] = helpers.image_matches(post, width=width, height=height, encodings={"32FC1"}, frame_id=gate["camera"]["frame"])
    else:
        checks["live_sensor_intrinsics_tf_pose_reset_complete"] = False
        observations["sample_skip_reason"] = "ROS graph/topic/service prerequisites failed"
    checks["test_iid_ood_and_robot_still_sealed"] = all(gate["sealed"][name] is True for name in ("gazebo_test_iid", "gazebo_test_ood", "robot"))
    checks["disk_free_over_2_gib"] = shutil.disk_usage(ROOT).free > 2 * 1024**3
    report = {
        "schema_version": 1,
        "protocol_id": r2.PILOT_ID,
        "attempt": 3,
        "implementation_revision": "r3_fixed_camera_tf_only",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "observations": observations,
        "r3_amendment_sha256": sha256(R3_LOCK),
        "independent_tf_precheck_sha256": sha256(R3_PRECHECK),
        "prior_live_attempt_02_sha256": sha256(ATTEMPT_02),
        "capture_authorized": False,
        "pilot_families_captured": 0,
        "calibration_families_captured": 0,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
        "next_authorized_action_if_pass": "Stop; a separate pilot capture-authorization lock is required.",
        "next_authorized_action_if_blocked": "Freeze this negative infrastructure attempt and stop; no retry under R3.",
    }
    write_new(ATTEMPT_03, report)
    print(json.dumps({"status": report["status"], "checks": checks, "artifact": str(ATTEMPT_03)}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze-amendment", "validate", "static-command", "precheck-tf", "preflight-live"))
    command = parser.parse_args().command
    if command == "freeze-amendment":
        freeze()
    elif command == "validate":
        validate()
        print("PASS: frozen Calibration-v6 R3 amendment")
    elif command == "static-command":
        value = validate()
        print(json.dumps(value["publisher_command"]))
    elif command == "precheck-tf":
        tf_precheck()
    else:
        live_preflight()


if __name__ == "__main__":
    main()
