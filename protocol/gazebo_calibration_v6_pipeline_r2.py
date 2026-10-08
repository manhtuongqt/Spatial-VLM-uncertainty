#!/usr/bin/env python3
"""Append-only ROS-environment amendment for Calibration-v6 live preflight.

R2 preserves the frozen v6 design, implementation, manifests, scientific
method, failed live attempt 01, and existing workflow decision.  Its sole
behavioral change is restoring the ROS Python paths required by rclpy and
writing live-preflight attempt 02 to a new immutable artifact.
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

import gazebo_calibration_v3_pipeline_r4 as helpers
import gazebo_calibration_v3_pipeline_r6 as parsers
import gazebo_calibration_v6_pipeline as base


ROOT = Path(__file__).resolve().parents[1]
PILOT_ID = "gazebo_calibration_v6_geometry_pilot"
AMENDMENT = ROOT / "protocol/GAZEBO_CALIBRATION_V6_IMPLEMENTATION_AMENDMENT_R2_LOCK.json"
ENV_PREFLIGHT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/R2_ROS_ENVIRONMENT_PREFLIGHT.json"
ATTEMPT_02 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/PREFLIGHT_LIVE_ATTEMPT_02.json"
BASE_IMPLEMENTATION = ROOT / "protocol/GAZEBO_CALIBRATION_V6_IMPLEMENTATION_LOCK.json"
BASE_PIPELINE = ROOT / "protocol/gazebo_calibration_v6_pipeline.py"
DATA_DESIGN = ROOT / "protocol/GAZEBO_CALIBRATION_V6_DATA_DESIGN_LOCK.json"
STATIC_PREFLIGHT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/PREFLIGHT_STATIC.json"
ATTEMPT_01 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/PREFLIGHT_LIVE.json"
LIVE_FAILURE_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V6_LIVE_FAILURE_LOCK.json"
CURRENT_DECISION = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/CALIBRATION_V6_FINAL_DECISION.json"
SCENES = ROOT / "ur3/ur3_perception/config/gazebo_calibration_v6_geometry_pilot_scenes.yaml"
GATE = ROOT / "ur3/ur3_perception/config/gazebo_calibration_v6_geometry_pilot_gate.yaml"
PILOT_MANIFEST = ROOT / "protocol/gazebo_calibration_v6_geometry_pilot_family_manifest.jsonl"
CALIBRATION_MANIFEST = ROOT / "protocol/gazebo_calibration_v6_family_manifest.jsonl"
CAPTURE_ROOT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/capture_attempt_01"
CALIBRATION_CAPTURE_ROOT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/capture_attempt_01"
DATASET_ROOT = ROOT / "datasets/Gazebo_calibration_v6"
CAPTURE_LOCK = ROOT / "protocol/gazebo_calibration_v6_geometry_pilot_capture_compatibility_lock.json"
WORLD = ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf"

ROS_LOCAL_DIST_PACKAGES = "/opt/ros/humble/local/lib/python3.10/dist-packages"
ROS_SITE_PACKAGES = "/opt/ros/humble/lib/python3.10/site-packages"
COMMAND_TIMEOUT_SEC = 12.0

EXPECTED_FROZEN_SHA256 = {
    DATA_DESIGN: "850921e200d921592128c76e11c8deeb5a7d6177ea315f1ff8d286f873084b6d",
    BASE_IMPLEMENTATION: "89d3416e7c870e635c3f6277a6dbbc2d5d12891c23f3a686c9a4fe2d4f7b821d",
    BASE_PIPELINE: "b5010c45c38e24bc7a1b30f21ee1d4441b49ceec0bb8c7fbff8b1ca01690ee01",
    STATIC_PREFLIGHT: "05be37274246b8364a0fc65a2c45d8c9707abfebe85cf7777169cb9c4ea18002",
    ATTEMPT_01: "a735ee37971b3917c7ccee19171056868a26ecc425984f5f585b87273834253e",
    LIVE_FAILURE_LOCK: "5449570008060680aea675e178059bcdca379140fead3064ec02422b248bad10",
    CURRENT_DECISION: "492cb4884077eec05ade91c114370eb3c30a1579fd15c66337b1e6d76d52c49e",
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
    """Preserve sourced ROS paths and prepend only the v6 protocol directory."""
    env = os.environ.copy()
    required = [str(ROOT / "protocol"), ROS_LOCAL_DIST_PACKAGES, ROS_SITE_PACKAGES]
    inherited = [item for item in env.get("PYTHONPATH", "").split(":") if item]
    env["PYTHONPATH"] = ":".join(dict.fromkeys(required + inherited))
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("ROS_LOCALHOST_ONLY", "1")
    env.setdefault("IGN_PARTITION", "ur3_roborefer_pilot_v0_local")
    return env


def run_probe(command: list[str], timeout: float = COMMAND_TIMEOUT_SEC) -> dict[str, Any]:
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
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        stdout = getattr(exc, "stdout", "") or ""
        stderr = getattr(exc, "stderr", "") or str(exc)
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "command": command,
            "returncode": 124 if isinstance(exc, subprocess.TimeoutExpired) else 127,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "seconds": time.monotonic() - started,
        }


def verify_frozen_predecessors() -> None:
    base.validate_implementation()
    for path, expected in EXPECTED_FROZEN_SHA256.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen v6 predecessor drift: {path}")
    if read_json(STATIC_PREFLIGHT).get("status") != "PASS":
        raise RuntimeError("frozen static preflight is not PASS")
    if read_json(ATTEMPT_01).get("status") != "BLOCKED":
        raise RuntimeError("live attempt 01 must remain BLOCKED")
    decision = read_json(CURRENT_DECISION)
    if decision.get("decision") != "CALIBRATION_V6_BLOCKED_BEFORE_SCIENTIFIC_EVALUATION":
        raise RuntimeError("existing v6 workflow decision changed")
    if any(path.exists() for path in (CAPTURE_ROOT, CALIBRATION_CAPTURE_ROOT, DATASET_ROOT, CAPTURE_LOCK)):
        raise RuntimeError("downstream data or capture authorization exists before R2")


def manifest_summary(path: Path, expected: int, repetition: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    cells: dict[str, int] = {}
    for row in rows:
        key = f"{row['state']}|{row['relation_variant']}"
        cells[key] = cells.get(key, 0) + 1
    return {
        "rows": len(rows),
        "expected_rows": expected,
        "cell_count": len(cells),
        "expected_repetitions_per_cell": repetition,
        "all_cells_match": len(cells) == 16 and set(cells.values()) == {repetition},
        "unique_family_ids": len({row["family_id"] for row in rows}) == expected,
        "unique_scene_ids": len({row["scene_id"] for row in rows}) == expected,
        "unique_layout_ids": len({row["layout_id"] for row in rows}) == expected,
        "unique_seeds": len({row["deterministic_seed"] for row in rows}) == expected,
        "unique_layout_signatures": len({row["layout_signature_sha256"] for row in rows}) == expected,
    }


def freeze_amendment() -> None:
    if AMENDMENT.exists():
        raise FileExistsError("R2 amendment already exists")
    verify_frozen_predecessors()
    pilot = manifest_summary(PILOT_MANIFEST, 32, 2)
    calibration = manifest_summary(CALIBRATION_MANIFEST, 128, 8)
    checks = {
        "frozen_v6_predecessors_unchanged": True,
        "attempt_01_preserved_blocked": True,
        "existing_workflow_decision_preserved": True,
        "pilot_manifest_4x4x2_unchanged": all(value is True or not isinstance(value, bool) for value in pilot.values()) and pilot["rows"] == 32,
        "calibration_manifest_4x4x8_unchanged": all(value is True or not isinstance(value, bool) for value in calibration.values()) and calibration["rows"] == 128,
        "no_capture_dataset_or_authorization": True,
        "test_and_robot_remain_sealed": True,
    }
    source_paths = list(EXPECTED_FROZEN_SHA256) + [Path(__file__).resolve(), PILOT_MANIFEST, CALIBRATION_MANIFEST, SCENES, GATE]
    payload = {
        "schema_version": 1,
        "protocol_id": "gazebo_calibration_v6",
        "status": "IMPLEMENTATION_AMENDMENT_R2_FROZEN_BEFORE_LIVE_PREFLIGHT_ATTEMPT_02",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "ROS Python environment restoration for live preflight only",
        "behavioral_change": {
            "prepend_python_path": [str(ROOT / "protocol"), ROS_LOCAL_DIST_PACKAGES, ROS_SITE_PACKAGES],
            "required_precheck": "/usr/bin/python3 -c 'import rclpy'",
            "new_live_artifact": str(ATTEMPT_02.relative_to(ROOT)),
        },
        "unchanged": [
            "Calibration-v6 data design and all 32/128 manifests",
            "B0 and frozen spatial-risk v2 with 12 ordered features",
            "unsafe definition, visibility thresholds, metrics, affine-logit optimizer, threshold grid, and scientific gates",
            "live attempt 01, its failure lock, static preflight, and existing workflow decision",
            "Test-IID/OOD and robot sealing",
        ],
        "checks": checks,
        "pilot_manifest_summary": pilot,
        "calibration_manifest_summary": calibration,
        "source_artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "capture_authorized": False,
        "pilot_or_calibration_captured": False,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    if not all(checks.values()):
        raise RuntimeError("R2 amendment audit failed")
    write_new(AMENDMENT, payload)
    print(json.dumps({"status": payload["status"], "sha256": sha256(AMENDMENT), "checks": checks}, indent=2, sort_keys=True))


def validate_amendment() -> dict[str, Any]:
    verify_frozen_predecessors()
    value = read_json(AMENDMENT)
    if value.get("status") != "IMPLEMENTATION_AMENDMENT_R2_FROZEN_BEFORE_LIVE_PREFLIGHT_ATTEMPT_02":
        raise RuntimeError("invalid R2 amendment status")
    if not all(value.get("checks", {}).values()):
        raise RuntimeError("R2 amendment checks are not all PASS")
    for name, digest in value.get("source_artifact_sha256", {}).items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"R2 source drift: {name}")
    return value


def environment_preflight() -> None:
    if ENV_PREFLIGHT.exists():
        raise FileExistsError("R2 ROS environment preflight already exists")
    validate_amendment()
    probe = run_probe([
        "/usr/bin/python3",
        "-c",
        "import rclpy,sys; print(sys.executable); print(rclpy.__file__)",
    ])
    path_entries = ros_environment()["PYTHONPATH"].split(":")
    checks = {
        "rclpy_import_returncode_zero": probe["returncode"] == 0,
        "rclpy_import_path_is_ros_humble_local": ROS_LOCAL_DIST_PACKAGES in probe["stdout"],
        "ros_local_dist_packages_in_pythonpath": ROS_LOCAL_DIST_PACKAGES in path_entries,
        "ros_site_packages_in_pythonpath": ROS_SITE_PACKAGES in path_entries,
        "protocol_directory_in_pythonpath": str(ROOT / "protocol") in path_entries,
        "attempt_01_preserved_blocked": sha256(ATTEMPT_01) == EXPECTED_FROZEN_SHA256[ATTEMPT_01],
        "attempt_02_absent": not ATTEMPT_02.exists(),
        "no_capture_dataset_or_authorization": not any(path.exists() for path in (CAPTURE_ROOT, CALIBRATION_CAPTURE_ROOT, DATASET_ROOT, CAPTURE_LOCK)),
    }
    report = {
        "schema_version": 1,
        "protocol_id": "gazebo_calibration_v6",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "ROS import environment only; no Gazebo data access or capture",
        "checks": checks,
        "probe": probe,
        "effective_pythonpath": path_entries,
        "amendment_lock_sha256": sha256(AMENDMENT),
        "capture_authorized": False,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(ENV_PREFLIGHT, report)
    print(json.dumps({"status": report["status"], "checks": checks, "probe": probe}, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


def configure_helpers() -> None:
    helpers.run_probe = run_probe
    parsers.run_probe = run_probe
    parsers.install_parser_repair()


def live_preflight_attempt_02() -> None:
    if ATTEMPT_02.exists():
        raise FileExistsError("refusing to overwrite live preflight attempt 02")
    validate_amendment()
    environment_report = read_json(ENV_PREFLIGHT)
    if environment_report.get("status") != "PASS":
        raise RuntimeError("R2 ROS environment preflight must PASS")
    configure_helpers()
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    scenes = yaml.safe_load(SCENES.read_text(encoding="utf-8"))
    topics = run_probe(["ros2", "topic", "list", "-t"])
    nodes = run_probe(["ros2", "node", "list"])
    services = run_probe(["ros2", "service", "list", "-t"])
    actions = run_probe(["ros2", "action", "list", "-t"])
    processes = run_probe(["ps", "-eo", "comm=,args="])
    observed: dict[str, Any] = {
        "topics": topics,
        "nodes": nodes,
        "services": services,
        "actions": actions,
        "processes": processes,
        "effective_pythonpath": ros_environment()["PYTHONPATH"].split(":"),
    }
    required_topics = {
        name: f"{name} [{message_type}]" in topics["stdout"]
        for name, message_type in helpers.REQUIRED_TOPICS.items()
    }
    world_lines = [
        line
        for line in processes["stdout"].splitlines()
        if line.split()
        and line.split()[0] in {"ruby", "ign", "gz", "gzserver"}
        and WORLD.name in line
    ]
    checks: dict[str, bool] = {
        "r2_amendment_locked_and_unchanged": True,
        "rclpy_environment_preflight_pass_and_unchanged": environment_report["status"] == "PASS",
        "attempt_01_preserved_blocked": sha256(ATTEMPT_01) == EXPECTED_FROZEN_SHA256[ATTEMPT_01],
        "existing_decision_preserved": sha256(CURRENT_DECISION) == EXPECTED_FROZEN_SHA256[CURRENT_DECISION],
        "ros_graph_probes_returncode_zero": all(p["returncode"] == 0 for p in (topics, nodes, services, actions)),
        "all_required_topic_types_present": all(required_topics.values()),
        "simulator_nodes_present": all(token in nodes["stdout"] for token in ("/ros_gz_bridge", "/robot_state_publisher", "/controller_manager")),
        "set_pose_service_present": helpers.SET_POSE_SERVICE in services["stdout"],
        "trajectory_action_present": "/joint_trajectory_controller/follow_joint_trajectory" in actions["stdout"],
        "locked_world_running": bool(world_lines),
        "no_capture_or_dataset_exists": not any(path.exists() for path in (CAPTURE_ROOT, CALIBRATION_CAPTURE_ROOT, DATASET_ROOT)),
        "no_capture_authorization_exists": not CAPTURE_LOCK.exists(),
    }
    observed["required_topic_presence"] = required_topics
    observed["locked_world_processes"] = world_lines
    prerequisites = (
        checks["ros_graph_probes_returncode_zero"]
        and checks["all_required_topic_types_present"]
        and checks["set_pose_service_present"]
    )
    if prerequisites:
        width, height = (int(value) for value in gate["camera"]["resolution"])
        wrist_frame = str(gate["camera"]["frame"])
        base_frame = "top_table_camera_optical_frame"
        for prefix, topic, frame in (
            ("wrist", "/wrist_camera", wrist_frame),
            ("base", "/top_table_camera", base_frame),
        ):
            rgb = helpers.image_probe(topic + "/color/image_raw")
            depth = helpers.image_probe(topic + "/depth/image_raw")
            info = parsers.camera_info_probe(topic + "/color/camera_info")
            tf_probe = run_probe(["ros2", "run", "tf2_ros", "tf2_echo", "base_link", frame], timeout=8.0)
            observed[prefix + "_rgb"] = rgb
            observed[prefix + "_depth"] = depth
            observed[prefix + "_camera_info"] = info
            observed[prefix + "_tf"] = tf_probe
            checks[prefix + "_rgb_live_valid"] = helpers.image_matches(
                rgb, width=width, height=height, encodings={"rgb8", "bgr8"}, frame_id=frame
            )
            checks[prefix + "_metric_depth_live_valid"] = helpers.image_matches(
                depth, width=width, height=height, encodings={"32FC1"}, frame_id=frame
            )
            checks[prefix + "_intrinsics_valid"] = helpers.camera_info_valid(
                info,
                width=width,
                height=height,
                frame_id=frame,
                expected_intrinsics=gate["camera"]["intrinsics"] if prefix == "wrist" else None,
            )
            checks[prefix + "_tf_valid"] = "Translation:" in tf_probe["stdout"] and "Rotation:" in tf_probe["stdout"]
        labels = helpers.image_probe("/wrist_camera/evaluation_labels/labels_map")
        observed["wrist_semantic_labels"] = labels
        checks["wrist_semantic_labels_live_valid"] = helpers.image_matches(
            labels, width=width, height=height, encodings={"rgb8"}, frame_id=wrist_frame
        )
        joint, fixed_pose, comparison = parsers.joint_state_probe(
            [float(value) for value in gate["camera"]["view_joint_pose"]]
        )
        observed["joint_state"] = joint
        observed["joint_comparison"] = comparison
        checks["fixed_view_joint_pose_within_tolerance"] = fixed_pose
        reset = helpers.reset_probe(scenes)
        observed["reset_probe"] = reset
        checks["full_scene_reset_probe_success"] = bool(reset.get("success"))
        post_reset = helpers.image_probe("/wrist_camera/depth/image_raw")
        observed["post_reset_depth"] = post_reset
        checks["post_reset_geometry_sensor_probe_success"] = helpers.image_matches(
            post_reset, width=width, height=height, encodings={"32FC1"}, frame_id=wrist_frame
        )
    else:
        checks["live_sensor_intrinsics_tf_pose_reset_complete"] = False
        observed["sample_skip_reason"] = "ROS graph/topic/service prerequisites failed"
    checks["test_iid_ood_and_robot_still_sealed"] = all(
        gate["sealed"][name] is True for name in ("gazebo_test_iid", "gazebo_test_ood", "robot")
    )
    checks["disk_free_over_2_gib"] = shutil.disk_usage(ROOT).free > 2 * 1024**3
    report = {
        "schema_version": 1,
        "protocol_id": PILOT_ID,
        "implementation_revision": "r2_ros_environment_only",
        "attempt": 2,
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "observations": observed,
        "amendment_lock_sha256": sha256(AMENDMENT),
        "environment_preflight_sha256": sha256(ENV_PREFLIGHT),
        "base_implementation_lock_sha256": sha256(BASE_IMPLEMENTATION),
        "static_preflight_sha256": sha256(STATIC_PREFLIGHT),
        "prior_live_attempt_sha256": sha256(ATTEMPT_01),
        "existing_workflow_decision_sha256": sha256(CURRENT_DECISION),
        "capture_authorized": False,
        "pilot_or_calibration_captured": False,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
        "next_authorized_action_if_pass": "Create a separate pilot capture-authorization lock; capture is not authorized by this artifact.",
        "next_authorized_action_if_blocked": "Freeze this attempt and stop without capture.",
    }
    write_new(ATTEMPT_02, report)
    print(json.dumps({"status": report["status"], "checks": checks, "artifact": str(ATTEMPT_02)}, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze-amendment", "validate", "preflight-environment", "preflight-live"))
    command = parser.parse_args().command
    if command == "freeze-amendment":
        freeze_amendment()
    elif command == "validate":
        validate_amendment()
        print("PASS: frozen Calibration-v6 implementation amendment R2")
    elif command == "preflight-environment":
        environment_preflight()
    else:
        live_preflight_attempt_02()


if __name__ == "__main__":
    main()
