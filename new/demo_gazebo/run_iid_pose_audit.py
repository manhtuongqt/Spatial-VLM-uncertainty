#!/usr/bin/env python3
"""Test-IID-pose multi-frame V2/independent-label audit; never starts MoveIt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from build_static_robot_world import TEST_IID_PLAN, build, selected_view_pose, verify_test_iid_optical_pose
from run_demo import ROOT, HERE, stop_owned_process


DEFAULT_PROMPT = (
    "In this robot-mounted camera observation, locate the apple that is left of "
    "the tomato soup can. Base the localization on the current frame. "
    "Output the target pixel as (x, y) in image coordinates."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--interval-s", type=float, default=0.8)
    parser.add_argument("--capture-timeout-s", type=float, default=90.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--target-model", default="ycb_apple")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output-root", type=Path, default=HERE / "runs")
    args = parser.parse_args()
    if args.frames < 2 or args.frames > 12:
        raise ValueError("Choose 2–12 frames for this audit")
    if args.gui and not os.environ.get("DISPLAY"):
        raise RuntimeError("--gui requires DISPLAY")
    if args.target_model != "ycb_apple":
        raise ValueError("This locked default audit currently supports ycb_apple only")
    if "apple" not in args.prompt.lower():
        raise ValueError("Prompt must identify the audited apple target")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"iid_pose_audit_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{os.getpid()}"
    run_dir.mkdir(exist_ok=False)
    print(f"RUN_DIR={run_dir}", flush=True)
    world = run_dir / "world_with_static_ur3.sdf"
    build(world, demo_layout=True, view_pose="test_iid")
    optical_pose_check = verify_test_iid_optical_pose(world)
    (run_dir / "view_pose.json").write_text(json.dumps({
        "pose_name": "camera_v2_1_relation",
        "joint_positions": selected_view_pose("test_iid"),
        "source_plan": str(TEST_IID_PLAN),
        "source_plan_sha256": hashlib.sha256(TEST_IID_PLAN.read_bytes()).hexdigest(),
        "optical_pose_check": optical_pose_check,
        "source_world_and_urdf_modified": False,
    }, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["IGN_PARTITION"] = f"pcrau_iid_pose_audit_{os.getpid()}"
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    gazebo = bridge = gui = None
    with (run_dir / "gazebo.log").open("x") as gazebo_log, \
         (run_dir / "bridge.log").open("x") as bridge_log, \
         (run_dir / "capture.log").open("x") as capture_log, \
         (run_dir / "inference.log").open("x") as inference_log, \
         (run_dir / "audit.log").open("x") as audit_log:
        try:
            gazebo = subprocess.Popen(
                ["ign", "gazebo", "-s", "-r", "--headless-rendering", "-v", "2", str(world)],
                cwd=ROOT, env=env, stdout=gazebo_log, stderr=subprocess.STDOUT,
                start_new_session=True)
            if args.gui:
                gui = subprocess.Popen(["ign", "gazebo", "-g"], cwd=ROOT, env=env,
                                       stdout=gazebo_log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            bridge_command = (
                f"source {ROOT / 'install/setup.bash'} && exec ros2 run ros_gz_bridge parameter_bridge "
                "'/wrist_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image' "
                "'/wrist_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image' "
                "'/wrist_camera/evaluation_labels/labels_map@sensor_msgs/msg/Image[ignition.msgs.Image'"
            )
            bridge = subprocess.Popen(["bash", "-c", bridge_command], cwd=ROOT, env=env,
                                      stdout=bridge_log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            time.sleep(2)
            if gazebo.poll() is not None or bridge.poll() is not None:
                raise RuntimeError("Gazebo/bridge stopped early; inspect logs")
            subprocess.run([
                "/usr/bin/python3", str(HERE / "capture_iid_multiframe.py"),
                "--output", str(run_dir / "capture"),
                "--count", str(args.frames),
                "--interval-s", str(args.interval_s),
                "--timeout-s", str(args.capture_timeout_s),
            ], cwd=ROOT, env=env, stdout=capture_log, stderr=subprocess.STDOUT,
                check=True, timeout=args.capture_timeout_s + 15)
            print(f"CAPTURED {args.frames} synchronized RGB-D + evaluator-label frames", flush=True)
            subprocess.run([
                sys.executable, str(HERE / "audit_iid_multiframe.py"),
                "--stage", "infer", "--run-dir", str(run_dir), "--prompt", args.prompt,
            ], cwd=ROOT, env=env, stdout=inference_log, stderr=subprocess.STDOUT,
                check=True, timeout=max(120, args.frames * 45))
            print("PREDICTIONS_LOCKED before evaluator-label access", flush=True)
            subprocess.run([
                "/usr/bin/python3", str(HERE / "audit_iid_multiframe.py"),
                "--stage", "check", "--run-dir", str(run_dir),
            ], cwd=ROOT, env=env, stdout=audit_log, stderr=subprocess.STDOUT,
                check=True, timeout=90)
            print("ORACLE_FREE_RGBD_CHECK_LOCKED before evaluator-label access", flush=True)
            subprocess.run([
                "/usr/bin/python3", str(HERE / "audit_iid_multiframe.py"),
                "--stage", "evaluate", "--run-dir", str(run_dir),
                "--target-model", args.target_model,
            ], cwd=ROOT, env=env, stdout=audit_log, stderr=subprocess.STDOUT,
                check=True, timeout=90)
            summary = json.loads((run_dir / "audit_summary.json").read_text(encoding="utf-8"))
            print(json.dumps({"status": "MULTIFRAME_SIM_AUDIT_COMPLETE", "run_dir": str(run_dir),
                              "frames": summary["frame_count"],
                              "point_in_target": summary["point_in_target_count"],
                              "identity_pass": summary["independent_identity_pass_count"],
                              "oracle_free_rgbd_check_pass": summary["oracle_free_rgbd_check_pass_count"],
                              "actions": summary["action_counts"],
                              "sim_pre_handoff_pass": summary["sim_pre_handoff_pass"],
                              "moveit_connected": False,
                              "robot_motion_commanded": False}, indent=2), flush=True)
        except Exception:
            print(f"AUDIT_FAILED; retained logs in {run_dir}", file=sys.stderr, flush=True)
            raise
        finally:
            stop_owned_process(gui)
            stop_owned_process(bridge)
            stop_owned_process(gazebo)


if __name__ == "__main__":
    main()
