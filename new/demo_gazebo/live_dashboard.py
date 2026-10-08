#!/usr/bin/env python3
"""Live four-panel Gazebo monitor with asynchronous frozen-V2 observations.

This monitor never publishes a manipulation target or a robot command.  A
separate pick/place runner may be observed through its episode-event topic;
its motion must not be attributed to V2 merely because both appear here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np

from capture_ros_rgbd import decode_depth_m, decode_rgb, relative_depth, stamp_seconds


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FONT = cv2.FONT_HERSHEY_SIMPLEX
NAVY = (37, 32, 25)
WHITE = (248, 248, 248)
CYAN = (225, 208, 24)
RED = (55, 57, 192)
GREEN = (88, 148, 51)
AMBER = (34, 147, 220)


def label(image: np.ndarray, text: str, x: int, y: int, scale: float = 0.58,
          color: tuple[int, int, int] = WHITE, thickness: int = 1) -> None:
    cv2.putText(image, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def image_tile(frame: np.ndarray | None, title: str, fallback: str,
               point: tuple[int, int] | None = None,
               heatmap: np.ndarray | None = None) -> np.ndarray:
    tile = np.full((540, 640, 3), (20, 23, 27), dtype=np.uint8)
    label(tile, title, 14, 31, 0.62, WHITE, 2)
    if frame is None:
        label(tile, fallback, 32, 255, 0.58, AMBER)
        return tile
    view = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    if heatmap is not None:
        grid = np.asarray(heatmap, dtype=np.float32)
        if grid.shape == (24, 32) and np.isfinite(grid).all() and grid.max() > 0:
            density = cv2.resize(grid, (640, 480), interpolation=cv2.INTER_LINEAR)
            density = np.clip(density / grid.max() * 255, 0, 255).astype(np.uint8)
            coloured = cv2.applyColorMap(density, cv2.COLORMAP_MAGMA)
            mask = density > 5
            view[mask] = cv2.addWeighted(view, 0.38, coloured, 0.62, 0)[mask]
    if point is not None:
        x, y = point
        if 0 <= x < 640 and 0 <= y < 480:
            cv2.drawMarker(view, (x, y), CYAN, cv2.MARKER_CROSS, 24, 3)
    tile[48:528] = view
    return tile


def score_bar(canvas: np.ndarray, title: str, score: float, y: int,
              color: tuple[int, int, int]) -> None:
    label(canvas, f"{title:<16} {score:.3f}", 26, y, 0.49)
    cv2.rectangle(canvas, (280, y - 13), (605, y), (65, 69, 74), -1)
    cv2.rectangle(canvas, (280, y - 13), (280 + int(325 * np.clip(score, 0, 1)), y), color, -1)


def policy_tile(result: dict | None, age_s: float | None, robot_event: str,
                inference_status: str) -> np.ndarray:
    tile = np.full((540, 640, 3), (20, 23, 27), dtype=np.uint8)
    label(tile, "4  V2 POLICY / NO CONTROL", 14, 31, 0.62, WHITE, 2)
    if result is None:
        label(tile, f"Inference: {inference_status[:72]}", 26, 100, 0.5, AMBER)
        label(tile, "Decision: PENDING (motion not authorized)", 26, 142, 0.54, RED)
    else:
        decision = result["decision"]
        prediction = result["prediction"]
        action = decision["action"]
        fresh = age_s is not None and age_s <= 8.0
        action_color = GREEN if action == "EXECUTE" and fresh else RED
        label(tile, f"V2: {action}" + ("" if fresh else "  / OLD FRAME"), 26, 89, 0.83, action_color, 2)
        label(tile, f"risk {float(decision['calibrated_grounding_risk']):.3f}  /  tau "
                    f"{float(decision['risk_threshold']):.3f}  /  age {age_s:.1f}s",
              26, 125, 0.54)
        answer = prediction["answerability_probabilities"]
        source = prediction["source_probabilities"]
        label(tile, "Answerability", 26, 169, 0.55, CYAN, 2)
        for index, key in enumerate(("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")):
            score_bar(tile, key.replace("INSUFFICIENT_EVIDENCE", "INSUFFICIENT"),
                      float(answer[key]), 196 + 27 * index, CYAN)
        label(tile, "Source uncertainty (raw scores)", 26, 330, 0.55, CYAN, 2)
        for index, key in enumerate(("semantic", "relation", "spatial", "depth", "occlusion")):
            score_bar(tile, key, float(source[key]), 356 + 27 * index, AMBER)
    label(tile, "Robot (separate controller): " + robot_event[:40], 26, 510, 0.44, AMBER)
    return tile


def compose(overview: np.ndarray | None, wrist: np.ndarray | None, result: dict | None,
            age_s: float | None, robot_event: str, inference_status: str,
            prompt: str) -> np.ndarray:
    prediction = result["prediction"] if result is not None else None
    point = tuple(map(int, prediction["spatial"]["map_pixel_xy"])) if prediction else None
    heat = prediction["spatial"]["probability_grid"] if prediction else None
    frozen_rgb = result.get("frame_bgr", wrist) if result is not None else wrist
    top = np.hstack((image_tile(overview, "1  GAZEBO OVERVIEW / LIVE", "Waiting for top camera"),
                     image_tile(wrist, "2  WRIST RGB / LIVE", "Waiting for wrist RGB")))
    bottom = np.hstack((image_tile(frozen_rgb, "3  V2 HEATMAP / CAPTURED FRAME",
                                   "Waiting for wrist RGB", point, heat),
                        policy_tile(result, age_s, robot_event, inference_status)))
    page = np.vstack((top, bottom))
    cv2.rectangle(page, (0, 1058), (1279, 1079), NAVY, -1)
    label(page, "Prompt: " + prompt[:160], 12, 1075, 0.42)
    return page


class V2Worker:
    def __init__(self, output: Path, prompt: str) -> None:
        self.output = output
        self.prompt = prompt
        self.lock = threading.Lock()
        self.pending: tuple[np.ndarray, np.ndarray, float, float] | None = None
        self.result: dict | None = None
        self.result_capture_time: float | None = None
        self.status = "loading frozen models"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=False)
        self.thread.start()

    def offer(self, rgb: np.ndarray, depth_m: np.ndarray, stamp: float) -> None:
        with self.lock:
            self.pending = (rgb.copy(), depth_m.copy(), stamp, time.monotonic())

    def run(self) -> None:
        try:
            import torch
            from infer_and_render import (frozen_paths, predict_v2)
            from infer_and_render import ROOT
            from pcra_u_development_common import pool_raw_feature, seed_runtime
            from wp3_feature_hook_smoke import extract_features, load_model, model_inventory_sha256
            from pcrau.policy import decide
            from pcrau.utils import read_json

            freeze, checkpoint, calibrator_path, config_path = frozen_paths()
            manifest = read_json(ROOT / "old/protocol/pcra_u_calibration_feature_manifest.json")
            model_root = ROOT / "RoboRefer/models/RoboRefer-2B-SFT"
            if model_inventory_sha256(model_root) != manifest["model_inventory_sha256"]:
                raise ValueError("Frozen RoboRefer inventory mismatch")
            seed_runtime(24082026, strict=False)
            backbone = load_model(model_root)
            config = read_json(config_path)
            calibrator = read_json(calibrator_path)
            self.status = "ready"
            cycle = 0
            while not self.stop.is_set():
                with self.lock:
                    snapshot, self.pending = self.pending, None
                if snapshot is None:
                    self.stop.wait(0.1)
                    continue
                rgb, depth_m, stamp, capture_time = snapshot
                cycle += 1
                self.status = f"inferencing frame {cycle}"
                cycle_dir = self.output / "inference" / f"frame_{cycle:05d}"
                cycle_dir.mkdir(parents=True, exist_ok=False)
                rgb_path = cycle_dir / "rgb.jpg"
                depth_path = cycle_dir / "relative_depth.png"
                relative_bgr, summary = relative_depth(depth_m)
                cv2.imwrite(str(rgb_path), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(depth_path), relative_bgr)
                extracted = extract_features(backbone, rgb_path, depth_path)
                r_grid, r_thumb = pool_raw_feature(extracted["r0"])
                d_grid, d_thumb = pool_raw_feature(extracted["d0"])
                features = {"r0": r_grid, "d0": d_grid, "r_thumb": r_thumb, "d_thumb": d_thumb}
                prediction, runtime = predict_v2(features, self.prompt, config, checkpoint,
                                                 config_path, "live_dynamic_wrist")
                decision = decide(prediction, calibrator)
                record = {"frame": cycle, "stamp_s": stamp,
                          "created_at_utc": datetime.now(timezone.utc).isoformat(),
                          "prediction": prediction, "decision": decision,
                          "depth_summary": summary, "runtime": runtime,
                          "motion_authorized": False,
                          "risk_validated_for_this_layout": False,
                          "checkpoint_sha256": freeze["selected_checkpoint"]["model_sha256"]}
                (cycle_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
                with self.lock:
                    record["frame_bgr"] = rgb
                    self.result = record
                    self.result_capture_time = capture_time
                self.status = f"frame {cycle}: {decision['action']}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as exc:
            self.status = f"ERROR {type(exc).__name__}: {exc}"
            (self.output / "inference_error.txt").write_text(self.status + "\n")


class ExternalV2Worker:
    """Keep ROS/GUI Python separate from the checkpoint's headless OpenCV."""

    def __init__(self, output: Path, prompt: str) -> None:
        self.output = output
        self.prompt = prompt
        self.process: subprocess.Popen | None = None
        self.request_dir: Path | None = None
        self.result: dict | None = None
        self.result_capture_time: float | None = None
        self.request_capture_time: float | None = None
        self.status = "waiting for synchronized RGB-D"
        self.count = 0

    def offer(self, rgb: np.ndarray, depth_m: np.ndarray, stamp: float) -> bool:
        self.poll()
        if self.process is not None:
            return False
        self.count += 1
        request_dir = self.output / "requests" / f"request_{self.count:05d}"
        request_dir.mkdir(parents=True, exist_ok=False)
        if not cv2.imwrite(str(request_dir / "rgb.png"), rgb):
            raise OSError("Could not save RGB inference request")
        np.save(request_dir / "depth_m.npy", depth_m, allow_pickle=False)
        (request_dir / "stamp.json").write_text(json.dumps({"stamp_s": stamp}) + "\n")
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        command = [str(ROOT / ".conda-roborefer/bin/python3.10"), str(HERE / "live_dashboard.py"),
                   "--infer-request", str(request_dir), "--output", str(request_dir),
                   "--prompt", self.prompt]
        log = (request_dir / "worker.log").open("w")
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, env=env,
                                            stdout=log, stderr=subprocess.STDOUT)
        finally:
            log.close()
        self.request_dir = request_dir
        self.request_capture_time = time.monotonic()
        self.status = f"inferencing request {self.count}"
        return True

    def poll(self) -> None:
        if self.process is None or self.process.poll() is None:
            return
        assert self.request_dir is not None
        result_path = self.request_dir / "inference/frame_00001/result.json"
        if self.process.returncode == 0 and result_path.is_file():
            result = json.loads(result_path.read_text())
            result["frame_bgr"] = cv2.imread(str(self.request_dir / "rgb.png"))
            self.result = result
            self.result_capture_time = self.request_capture_time
            self.status = f"request {self.count}: {result['decision']['action']}"
        else:
            self.status = f"ERROR worker exit {self.process.returncode}; see {self.request_dir / 'worker.log'}"
        self.process = None

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def run_live(args: argparse.Namespace) -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    class Monitor(Node):
        def __init__(self) -> None:
            super().__init__("pcrau_live_dashboard")
            self.overview: np.ndarray | None = None
            self.wrist: np.ndarray | None = None
            self.depth: np.ndarray | None = None
            self.rgb_stamp = 0.0
            self.depth_stamp = 0.0
            self.robot_event = "No pick/place event received"
            self.create_subscription(Image, args.overview_topic, self.on_overview, qos_profile_sensor_data)
            self.create_subscription(Image, args.wrist_topic, self.on_wrist, qos_profile_sensor_data)
            self.create_subscription(Image, args.depth_topic, self.on_depth, qos_profile_sensor_data)
            self.create_subscription(String, "/ur3_dataset/episode_event", self.on_event, 10)

        def on_overview(self, message: Image) -> None:
            self.overview = decode_rgb(message)

        def on_wrist(self, message: Image) -> None:
            self.wrist = decode_rgb(message)
            self.rgb_stamp = stamp_seconds(message)

        def on_depth(self, message: Image) -> None:
            self.depth = decode_depth_m(message)
            self.depth_stamp = stamp_seconds(message)

        def on_event(self, message: String) -> None:
            try:
                event = json.loads(message.data)
                self.robot_event = str(event.get("event", event))
                with (args.output / "robot_events.jsonl").open("a") as handle:
                    handle.write(message.data + "\n")
            except (ValueError, OSError):
                self.robot_event = message.data

    rclpy.init()
    node = Monitor()
    worker = ExternalV2Worker(args.output, args.prompt)
    writer = None
    page = None
    last_offer = 0.0
    started = time.monotonic()
    cv2.namedWindow("UR3 Spatial-VLM live demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("UR3 Spatial-VLM live demo", 1280, 1080)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.monotonic()
            worker.poll()
            if (node.wrist is not None and node.depth is not None and
                abs(node.rgb_stamp - node.depth_stamp) <= 0.08 and
                now - last_offer >= args.inference_interval_s):
                if worker.offer(node.wrist, node.depth, node.rgb_stamp):
                    last_offer = now
            result = worker.result
            age_s = now - worker.result_capture_time if worker.result_capture_time else None
            page = compose(node.overview, node.wrist, result, age_s,
                           node.robot_event, worker.status, args.prompt)
            cv2.imshow("UR3 Spatial-VLM live demo", page)
            if args.record_video:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output / "live_dashboard.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 1080))
                writer.write(page)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            if args.max_seconds is not None and now - started >= args.max_seconds:
                break
    finally:
        worker.stop()
        if writer is not None:
            writer.release()
        if page is not None:
            cv2.imwrite(str(args.output / "last_dashboard.png"), page)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=(
        "In this robot-mounted camera observation, locate the apple that is left of "
        "the tomato soup can. Base the localization on the current frame. "
        "Output the target pixel as (x, y) in image coordinates."))
    parser.add_argument("--output", type=Path, default=HERE / "runs" /
                        f"live_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    parser.add_argument("--overview-topic", default="/top_table_camera/color/image_raw")
    parser.add_argument("--wrist-topic", default="/wrist_camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/wrist_camera/depth/image_raw")
    parser.add_argument("--inference-interval-s", type=float, default=15.0)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Optional finite live-view duration for testing")
    parser.add_argument("--replay-run", type=Path, help="Render one saved real Gazebo capture for UI QA")
    parser.add_argument("--infer-request", type=Path,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.infer_request:
        request = args.infer_request.resolve()
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        rgb = cv2.imread(str(request / "rgb.png"))
        depth_m = np.load(request / "depth_m.npy", allow_pickle=False)
        if rgb is None or rgb.shape[:2] != (480, 640) or depth_m.shape != (480, 640):
            raise ValueError("Inference request must contain aligned 640x480 RGB-D")
        stamp = float(json.loads((request / "stamp.json").read_text())["stamp_s"])
        worker = V2Worker(output, args.prompt)
        worker.offer(rgb, depth_m, stamp)
        try:
            while worker.result is None and not worker.status.startswith("ERROR"):
                time.sleep(0.1)
            if worker.result is None:
                raise RuntimeError(worker.status)
            print(worker.status, flush=True)
        finally:
            worker.stop.set()
            worker.thread.join()
        return
    if args.replay_run:
        run = args.replay_run.resolve()
        overview = cv2.imread(str(run / "capture/overview_original.png"))
        wrist = cv2.imread(str(run / "capture/rgb_original.png"))
        if overview is None or wrist is None:
            raise FileNotFoundError("Replay requires real saved overview and wrist frames")
        result = {"prediction": json.loads((run / "prediction.json").read_text()),
                  "decision": json.loads((run / "decision.json").read_text())}
        output = compose(overview, wrist, result, 0.0, "Replay: no robot motion",
                         "saved observation", result["prediction"]["prompt"])
        path = run / "live_dashboard_replay.png"
        cv2.imwrite(str(path), output)
        print(path)
        return
    if args.inference_interval_s < 1:
        raise ValueError("inference interval must be >= 1 second")
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("max-seconds must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    run_live(args)


if __name__ == "__main__":
    main()
