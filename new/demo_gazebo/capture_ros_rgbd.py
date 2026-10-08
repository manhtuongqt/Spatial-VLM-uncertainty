#!/usr/bin/python3
"""Capture one synchronized RGB-D observation from Gazebo; no robot control."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def stamp_seconds(message: Image) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9


def decode_rgb(message: Image) -> np.ndarray:
    encoding = message.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(encoding)
    if channels is None:
        raise ValueError(f"Unsupported RGB encoding: {message.encoding}")
    raw = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    pixels = raw[:, : message.width * channels].reshape(message.height, message.width, channels)
    if encoding == "rgb8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    if encoding == "mono8":
        return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    return pixels.copy()


def decode_depth_m(message: Image) -> np.ndarray:
    encoding = message.encoding.lower()
    if encoding == "32fc1":
        dtype, scale = np.dtype("f4"), 1.0
    elif encoding == "16uc1":
        dtype, scale = np.dtype("u2"), 0.001
    else:
        raise ValueError(f"Unsupported depth encoding: {message.encoding}")
    dtype = dtype.newbyteorder(">" if message.is_bigendian else "<")
    raw = np.frombuffer(message.data, dtype=dtype).reshape(message.height, message.step // dtype.itemsize)
    return (raw[:, : message.width].astype(np.float32) * scale).copy()


def relative_depth(depth_m: np.ndarray) -> tuple[np.ndarray, dict]:
    """Same percentile inverse-depth transform as Test-IID materialization."""
    values = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(values) & (values >= 0.10) & (values <= 2.0)
    if int(np.count_nonzero(valid)) < 16:
        raise ValueError("Fewer than 16 valid depth pixels in [0.10, 2.0] m")
    visible = values[valid]
    near, far = float(np.percentile(visible, 2)), float(np.percentile(visible, 98))
    if far - near < 1e-4:
        near, far = float(visible.min()), float(visible.max())
    if far - near < 1e-6:
        raise ValueError("Metric depth has no usable range")
    clipped = np.clip(values, near, far)
    inverse = 1.0 / np.maximum(clipped, 1e-6)
    normalized = (inverse - 1.0 / far) / max(1.0 / near - 1.0 / far, 1e-6)
    gray = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    gray[~valid] = 0
    return np.repeat(gray[..., None], 3, axis=2), {
        "valid_fraction": float(valid.mean()), "near_percentile_2_m": near,
        "far_percentile_98_m": far,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RGBDCapture(Node):
    def __init__(self, rgb_topic: str, depth_topic: str, tolerance_s: float,
                 overview_topic: str | None = None):
        super().__init__("pcrau_gazebo_observation_capture")
        self.latest_rgb: Image | None = None
        self.latest_depth: Image | None = None
        self.latest_overview: Image | None = None
        self.pair: tuple[Image, Image, Image | None] | None = None
        self.tolerance_s = tolerance_s
        self.create_subscription(Image, rgb_topic, self.on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, depth_topic, self.on_depth, qos_profile_sensor_data)
        if overview_topic:
            self.create_subscription(Image, overview_topic, self.on_overview,
                                     qos_profile_sensor_data)
        self.expect_overview = overview_topic is not None

    def on_rgb(self, message: Image) -> None:
        self.latest_rgb = message
        self.try_pair()

    def on_depth(self, message: Image) -> None:
        self.latest_depth = message
        self.try_pair()

    def on_overview(self, message: Image) -> None:
        self.latest_overview = message
        self.try_pair()

    def try_pair(self) -> None:
        if self.pair is not None or self.latest_rgb is None or self.latest_depth is None:
            return
        if abs(stamp_seconds(self.latest_rgb) - stamp_seconds(self.latest_depth)) > self.tolerance_s:
            return
        if self.expect_overview:
            if self.latest_overview is None:
                return
            if abs(stamp_seconds(self.latest_rgb) - stamp_seconds(self.latest_overview)) > 0.10:
                return
        self.pair = self.latest_rgb, self.latest_depth, self.latest_overview


def capture(output_dir: Path, rgb_topic: str, depth_topic: str, timeout_s: float,
            tolerance_s: float, static_robot_present: bool = False,
            overview_topic: str | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = RGBDCapture(rgb_topic, depth_topic, tolerance_s, overview_topic)
    try:
        deadline = time.monotonic() + timeout_s
        while node.pair is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if node.pair is None:
            raise TimeoutError(f"No synchronized RGB-D pair within {timeout_s:g} s")
        rgb_msg, depth_msg, overview_msg = node.pair
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if (rgb_msg.width, rgb_msg.height) != (640, 480) or (depth_msg.width, depth_msg.height) != (640, 480):
        raise ValueError("Frozen V2 expects aligned 640x480 RGB and depth")
    rgb_bgr = decode_rgb(rgb_msg)
    depth_m = decode_depth_m(depth_msg)
    relative_bgr, depth_summary = relative_depth(depth_m)
    paths = {
        "rgb_original": output_dir / "rgb_original.png",
        "rgb_model_input": output_dir / "rgb_model_input.jpg",
        "depth_metric": output_dir / "depth_metric.npy",
        "depth_model_input": output_dir / "depth_relative_model_input.png",
    }
    if not cv2.imwrite(str(paths["rgb_original"]), rgb_bgr):
        raise OSError("Could not write RGB PNG")
    if not cv2.imwrite(str(paths["rgb_model_input"]), rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError("Could not write model RGB JPEG")
    np.save(paths["depth_metric"], depth_m, allow_pickle=False)
    if not cv2.imwrite(str(paths["depth_model_input"]), relative_bgr):
        raise OSError("Could not write model depth PNG")
    if overview_msg is not None:
        if (overview_msg.width, overview_msg.height) != (640, 480):
            raise ValueError("Overview must be 640x480")
        paths["overview_original"] = output_dir / "overview_original.png"
        if not cv2.imwrite(str(paths["overview_original"]), decode_rgb(overview_msg)):
            raise OSError("Could not write overview PNG")
    result = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_backend": f"Gazebo Fortress fixed {rgb_topic.rsplit('/', 1)[0]} via ROS-Gazebo bridge",
        "rgb_topic": rgb_topic,
        "depth_topic": depth_topic,
        "rgb_encoding": rgb_msg.encoding,
        "depth_encoding": depth_msg.encoding,
        "rgb_stamp_s": stamp_seconds(rgb_msg),
        "depth_stamp_s": stamp_seconds(depth_msg),
        "sync_delta_s": abs(stamp_seconds(rgb_msg) - stamp_seconds(depth_msg)),
        "overview_topic": overview_topic,
        "overview_stamp_s": stamp_seconds(overview_msg) if overview_msg is not None else None,
        "overview_sync_delta_s": (abs(stamp_seconds(rgb_msg) - stamp_seconds(overview_msg))
                                  if overview_msg is not None else None),
        "shape_hw": [480, 640],
        "depth": depth_summary,
        "artifacts_sha256": {name: sha256(path) for name, path in paths.items()},
        "safety": {"robot_spawned": static_robot_present, "robot_motion_commanded": False,
                   "semantic_labels_subscribed": False, "physical_task_success_measured": False},
    }
    with (output_dir / "capture.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rgb-topic", default="/top_table_camera/image")
    parser.add_argument("--depth-topic", default="/top_table_camera/depth_image")
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--sync-tolerance-s", type=float, default=0.06)
    parser.add_argument("--static-robot-present", action="store_true")
    parser.add_argument("--overview-topic", default=None)
    args = parser.parse_args()
    result = capture(args.output_dir.resolve(), args.rgb_topic, args.depth_topic,
                     args.timeout_s, args.sync_tolerance_s, args.static_robot_present,
                     args.overview_topic)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
