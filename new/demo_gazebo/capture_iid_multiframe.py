#!/usr/bin/python3
"""Capture distinct, synchronized wrist RGB-D frames plus sealed evaluator labels."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from capture_ros_rgbd import decode_depth_m, decode_rgb, relative_depth, sha256, stamp_seconds


class MultiFrameCapture(Node):
    def __init__(self) -> None:
        super().__init__("pcrau_iid_pose_multiframe_capture")
        self.latest: dict[str, Image] = {}
        self.bridge = CvBridge()
        topics = {
            "rgb": "/wrist_camera/image",
            "depth": "/wrist_camera/depth_image",
            "labels": "/wrist_camera/evaluation_labels/labels_map",
        }
        for key, topic in topics.items():
            self.create_subscription(Image, topic, lambda msg, name=key: self.latest.__setitem__(name, msg),
                                     qos_profile_sensor_data)

    def synchronized(self, last_rgb_stamp: float) -> tuple[dict[str, Image], dict[str, float]] | None:
        if set(self.latest) != {"rgb", "depth", "labels"}:
            return None
        messages = dict(self.latest)
        stamps = {key: stamp_seconds(msg) for key, msg in messages.items()}
        if stamps["rgb"] <= last_rgb_stamp + 0.10:
            return None
        if max(stamps.values()) - min(stamps.values()) > 0.08:
            return None
        if any((msg.width, msg.height) != (640, 480) for msg in messages.values()):
            raise ValueError("Expected 640x480 registered RGB, depth, and evaluator labels")
        return messages, stamps

    def decode_labels(self, message: Image) -> np.ndarray:
        value = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough"))
        if value.ndim == 3:
            if value.shape[2] != 3 or not np.array_equal(value[..., 0], value[..., 1]) or not np.array_equal(value[..., 1], value[..., 2]):
                raise ValueError("Semantic label channels disagree")
            value = value[..., 0]
        if value.shape != (480, 640) or value.max() > 255:
            raise ValueError("Semantic labels violate uint8 640x480 contract")
        return value.astype(np.uint8).copy()


def capture(output: Path, count: int, timeout_s: float, interval_s: float) -> dict:
    if count < 2 or interval_s < 0.2 or timeout_s <= 0:
        raise ValueError("Require >=2 frames, >=0.2 s interval, positive timeout")
    output.mkdir(parents=True, exist_ok=False)
    (output / "inference").mkdir()
    (output / "evaluator_only").mkdir()
    rclpy.init()
    node = MultiFrameCapture()
    rows = []
    last_stamp = float("-inf")
    next_wall = time.monotonic()
    deadline = time.monotonic() + timeout_s
    try:
        while len(rows) < count and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() < next_wall:
                continue
            bundle = node.synchronized(last_stamp)
            if bundle is None:
                continue
            messages, stamps = bundle
            frame_id = f"frame_{len(rows) + 1:03d}"
            inference_dir = output / "inference" / frame_id
            evaluator_dir = output / "evaluator_only" / frame_id
            inference_dir.mkdir()
            evaluator_dir.mkdir()
            rgb = decode_rgb(messages["rgb"])
            depth_m = decode_depth_m(messages["depth"])
            labels = node.decode_labels(messages["labels"])
            relative_bgr, depth_summary = relative_depth(depth_m)
            files = {
                "rgb_original": inference_dir / "rgb_original.png",
                "rgb_model_input": inference_dir / "rgb_model_input.jpg",
                "depth_relative_model_input": inference_dir / "depth_relative_model_input.png",
            }
            if not cv2.imwrite(str(files["rgb_original"]), rgb):
                raise OSError("RGB write failed")
            if not cv2.imwrite(str(files["rgb_model_input"]), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError("Model RGB write failed")
            if not cv2.imwrite(str(files["depth_relative_model_input"]), relative_bgr):
                raise OSError("Relative-depth write failed")
            np.save(inference_dir / "depth_metric.npy", depth_m, allow_pickle=False)
            label_path = evaluator_dir / "semantic_labels.png"
            if not cv2.imwrite(str(label_path), labels):
                raise OSError("Evaluator-label write failed")
            row = {
                "frame_id": frame_id,
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "sensor_stamps_s": stamps,
                "maximum_sensor_skew_s": max(stamps.values()) - min(stamps.values()),
                "depth": depth_summary,
                "inference_sha256": {name: sha256(path) for name, path in files.items()},
                "depth_metric_sha256": sha256(inference_dir / "depth_metric.npy"),
            }
            rows.append(row)
            (evaluator_dir / "label_meta.json").write_text(json.dumps({
                "frame_id": frame_id, "stamp_s": stamps["labels"],
                "label_sha256": sha256(label_path),
                "access": "evaluator_only_after_prediction_lock",
            }, indent=2) + "\n", encoding="utf-8")
            last_stamp = stamps["rgb"]
            next_wall = time.monotonic() + interval_s
            print(f"CAPTURED {frame_id} rgb={stamps['rgb']:.3f} skew={row['maximum_sensor_skew_s']:.3f}", flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if len(rows) != count:
        raise TimeoutError(f"Captured {len(rows)}/{count} synchronized frames within {timeout_s}s")
    index = {"schema_version": 1, "frame_count": count, "frames": rows,
             "camera": "wrist_camera", "view_pose": "test_iid",
             "evaluator_labels_visible_to_v2": False,
             "robot_motion_commanded": False}
    (output / "capture_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--interval-s", type=float, default=0.8)
    args = parser.parse_args()
    print(json.dumps(capture(args.output.resolve(), args.count, args.timeout_s, args.interval_s)))
