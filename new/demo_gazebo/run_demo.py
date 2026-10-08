#!/usr/bin/env python3
"""Launch a perception-only Gazebo scene, capture RGB-D, and run frozen best V2."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORLD = ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf"
DEFAULT_OUTPUT_ROOT = HERE / "runs"
DEFAULT_PROMPT = (
    "For the presently visible object arrangement, locate the banana. "
    "Use the current RGB-D observation. Return the target pixel as (x, y) in image coordinates."
)
APPLE_DASHBOARD_PROMPT = (
    "In this robot-mounted camera observation, locate the apple that is left of "
    "the tomato soup can. Base the localization on the current frame. "
    "Output the target pixel as (x, y) in image coordinates."
)


def stop_owned_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="A referring expression for the fresh Gazebo RGB-D observation")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--capture-timeout-s", type=float, default=90.0)
    parser.add_argument("--gui", action="store_true", help="Also open the Gazebo GUI when DISPLAY is available")
    parser.add_argument("--camera", choices=("top", "oblique", "opposite", "wrist", "left_oblique"), default="top",
                        help="Fixed Gazebo RGB-D viewpoint")
    parser.add_argument("--capture-only", action="store_true",
                        help="Save the RGB-D view without running V2 inference")
    parser.add_argument("--with-static-robot", action="store_true",
                        help="Show a stationary UR3 preview at the original robot base")
    parser.add_argument("--demo-layout", action="store_true",
                        help="Preview-only apple-centered wrist pose, boxes near bin, left-oblique view")
    parser.add_argument("--dashboard", action="store_true",
                        help="Capture both views, run frozen V2 on wrist RGB-D, render four-panel UI")
    parser.add_argument("--show-dashboard", action="store_true",
                        help="Keep Gazebo and the dashboard image window open until closed")
    parser.add_argument("--live-dashboard", action="store_true",
                        help="Live four-panel RGB-D/V2 monitor; still no robot actuation")
    parser.add_argument("--record-video", action="store_true",
                        help="With --live-dashboard, save the four-panel display as MP4")
    parser.add_argument("--live-seconds", type=float, default=None,
                        help="With --live-dashboard, close automatically after this many seconds")
    args = parser.parse_args()
    if args.show_dashboard:
        args.dashboard = True
        args.gui = True
    if args.dashboard:
        if args.capture_only:
            raise ValueError("--dashboard and --capture-only cannot be combined")
        args.camera = "wrist"
        args.with_static_robot = True
        args.demo_layout = True
        if args.prompt == DEFAULT_PROMPT:
            args.prompt = APPLE_DASHBOARD_PROMPT
    if args.live_dashboard:
        if args.dashboard or args.capture_only:
            raise ValueError("--live-dashboard cannot be combined with --dashboard or --capture-only")
        args.gui = True
        args.camera = "wrist"
        args.with_static_robot = True
        args.demo_layout = True
        if args.prompt == DEFAULT_PROMPT:
            args.prompt = APPLE_DASHBOARD_PROMPT
    if not WORLD.is_file():
        raise FileNotFoundError(WORLD)
    if args.gui and not os.environ.get("DISPLAY"):
        raise RuntimeError("--gui requires DISPLAY; omit it for the headless Gazebo scene")
    if args.capture_timeout_s <= 0:
        raise ValueError("capture timeout must be positive")
    if args.camera == "wrist" and not args.with_static_robot:
        raise ValueError("The wrist camera requires --with-static-robot")
    if args.demo_layout and not args.with_static_robot:
        raise ValueError("--demo-layout requires --with-static-robot")
    if args.camera == "left_oblique" and not args.demo_layout:
        raise ValueError("The left-oblique camera requires --demo-layout")
    if not args.capture_only and not args.dashboard and not args.live_dashboard and (args.camera != "top" or args.with_static_robot):
        raise ValueError("V2 inference is currently supported only for the top view without the static robot; use --capture-only")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"gazebo_v2_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{os.getpid()}"
    run_dir.mkdir(exist_ok=False)
    print(f"RUN_DIR={run_dir}", flush=True)
    world = WORLD
    if args.with_static_robot:
        world = run_dir / "world_with_static_ur3.sdf"
        build_command = [sys.executable, str(HERE / "build_static_robot_world.py"),
                         "--output-world", str(world)]
        if args.demo_layout:
            build_command.append("--demo-layout")
        subprocess.run(build_command, cwd=ROOT, check=True)

    env = os.environ.copy()
    env["IGN_PARTITION"] = f"pcrau_observation_demo_{os.getpid()}"
    env["ROS_LOCALHOST_ONLY"] = "1"
    # Prevent ~/.local torch from shadowing the checkpoint environment's
    # matching torch/torchvision pair during frozen RoboRefer feature loading.
    env["PYTHONNOUSERSITE"] = "1"
    camera_topic = {
        "top": "/top_table_camera",
        "oblique": "/oblique_table_camera",
        "opposite": "/opposite_table_camera",
        "wrist": "/wrist_camera",
        "left_oblique": "/left_oblique_table_camera",
    }[args.camera]
    gazebo: subprocess.Popen | None = None
    bridge: subprocess.Popen | None = None
    gui: subprocess.Popen | None = None
    with (run_dir / "gazebo.log").open("x", encoding="utf-8") as gazebo_log, \
         (run_dir / "bridge.log").open("x", encoding="utf-8") as bridge_log, \
         (run_dir / "capture.log").open("x", encoding="utf-8") as capture_log, \
         (run_dir / "inference.log").open("x", encoding="utf-8") as inference_log:
        try:
            gazebo = subprocess.Popen(
                ["ign", "gazebo", "-s", "-r", "--headless-rendering", "-v", "2", str(world)],
                cwd=ROOT, env=env, stdout=gazebo_log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if args.gui:
                gui = subprocess.Popen(
                    ["ign", "gazebo", "-g"], cwd=ROOT, env=env,
                    stdout=gazebo_log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            bridge_command = (
                f"source {ROOT / 'install/setup.bash'} && exec ros2 run ros_gz_bridge parameter_bridge "
                f"'{camera_topic}/image@sensor_msgs/msg/Image[ignition.msgs.Image' "
                f"'{camera_topic}/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image'"
            )
            if args.dashboard:
                bridge_command += " '/left_oblique_table_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image'"
            if args.live_dashboard:
                bridge_command += " '/left_oblique_table_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image'"
            bridge = subprocess.Popen(
                ["bash", "-c", bridge_command], cwd=ROOT, env=env,
                stdout=bridge_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            time.sleep(2)
            if gazebo.poll() is not None or bridge.poll() is not None:
                raise RuntimeError("Gazebo or ROS-Gazebo bridge exited early; inspect gazebo.log / bridge.log")
            if args.live_dashboard:
                command = ["/usr/bin/python3", str(HERE / "live_dashboard.py"),
                           "--output", str(run_dir / "live"), "--prompt", args.prompt,
                           "--overview-topic", "/left_oblique_table_camera/image",
                           "--wrist-topic", "/wrist_camera/image",
                           "--depth-topic", "/wrist_camera/depth_image"]
                if args.record_video:
                    command.append("--record-video")
                if args.live_seconds is not None:
                    command.extend(["--max-seconds", str(args.live_seconds)])
                subprocess.run(command, cwd=ROOT, env=env, check=True,
                               stdout=inference_log, stderr=subprocess.STDOUT)
                print(json.dumps({"status": "LIVE_OBSERVATION_COMPLETE",
                                  "run_dir": str(run_dir),
                                  "last_dashboard": str(run_dir / "live/last_dashboard.png"),
                                  "robot_motion_commanded": False}, indent=2), flush=True)
                return
            capture_command = [
                "/usr/bin/python3", str(HERE / "capture_ros_rgbd.py"),
                "--output-dir", str(run_dir / "capture"),
                "--timeout-s", str(args.capture_timeout_s),
                "--rgb-topic", f"{camera_topic}/image",
                "--depth-topic", f"{camera_topic}/depth_image",
            ]
            if args.with_static_robot:
                capture_command.append("--static-robot-present")
            if args.dashboard:
                capture_command.extend(["--overview-topic", "/left_oblique_table_camera/image"])
            subprocess.run(capture_command, cwd=ROOT, env=env, check=True,
                           stdout=capture_log, stderr=subprocess.STDOUT,
                           timeout=args.capture_timeout_s + 12)
            print("CAPTURE_COMPLETE: synchronized Gazebo RGB-D pair saved", flush=True)
            if args.capture_only:
                print(json.dumps({
                    "status": "RGBD_CAPTURE_COMPLETE",
                    "camera": args.camera,
                    "run_dir": str(run_dir),
                    "rgb": str(run_dir / "capture/rgb_original.png"),
                    "robot_spawned": args.with_static_robot,
                    "robot_motion_commanded": False,
                }, ensure_ascii=False, indent=2), flush=True)
                return
            inference_command = [
                sys.executable, str(HERE / "infer_and_render.py"),
                "--run-dir", str(run_dir), "--prompt", args.prompt,
            ]
            subprocess.run(inference_command, cwd=ROOT, env=env, check=True,
                           stdout=inference_log, stderr=subprocess.STDOUT)
            if args.dashboard:
                subprocess.run([sys.executable, str(HERE / "render_dashboard.py"),
                                "--run-dir", str(run_dir)], cwd=ROOT, env=env, check=True,
                               stdout=inference_log, stderr=subprocess.STDOUT)
            decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
            print(json.dumps({
                "status": "OBSERVATION_ONLY_DEMO_COMPLETE",
                "run_dir": str(run_dir),
                "panel": str(run_dir / "demo_panel.png"),
                "dashboard": str(run_dir / "dashboard.png") if args.dashboard else None,
                "action": decision["action"],
                "model_reported_risk": decision["calibrated_grounding_risk"],
                "robot_spawned": args.with_static_robot,
                "robot_motion_commanded": False,
                "risk_validated_for_this_preview": False,
            }, ensure_ascii=False, indent=2), flush=True)
            if args.show_dashboard:
                subprocess.run(["eog", str(run_dir / "dashboard.png")], cwd=ROOT, env=env,
                               check=False)
        except Exception:
            print(f"DEMO_FAILED; retained logs and any outputs in {run_dir}", file=sys.stderr, flush=True)
            raise
        finally:
            stop_owned_process(gui)
            stop_owned_process(bridge)
            stop_owned_process(gazebo)


if __name__ == "__main__":
    main()
