#!/usr/bin/python3
"""Capture WP2 raw RGB-D and evaluator semantic labels from Gazebo.

The node consumes only the preregistered capture plan. It does not construct a
RoboRefer request and it never publishes a manipulation target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformException, TransformListener

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp2_common import (  # noqa: E402
    PROTOCOL_ID,
    WP2Error,
    artifact_entry,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)


CAPTURE_ID = re.compile(r"^wp2_family_[0-9]{4}__(?:clean|occlusion)_capture$")
LOCKED_VIEW_JOINT_POSE = [1.3815, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601]
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def stamp_sec(message) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class WP2Capture(Node):
    def __init__(self, capture_plan: dict, output_root: Path, settle_sec: float) -> None:
        super().__init__("wp2_gazebo_capture")
        self.plan = capture_plan
        self.output_root = output_root
        self.capture_root = output_root / "raw" / "captures"
        self.capture_root.mkdir(parents=True, exist_ok=True)
        self.settle_sec = float(settle_sec)
        self.bridge = CvBridge()
        self.latest_rgb: Optional[tuple[float, np.ndarray]] = None
        self.latest_depth: Optional[tuple[float, np.ndarray]] = None
        self.latest_labels: Optional[tuple[float, np.ndarray]] = None
        self.camera_info: Optional[CameraInfo] = None
        self.joint_state: Optional[JointState] = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=60.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.reset_client = self.create_client(
            SetEntityPose, "/world/ur3_pick_place/set_pose"
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            Image, "/wrist_camera/color/image_raw", self.rgb_callback, sensor_qos
        )
        self.create_subscription(
            Image, "/wrist_camera/depth/image_raw", self.depth_callback, sensor_qos
        )
        self.create_subscription(
            Image,
            "/wrist_camera/evaluation_labels/labels_map",
            self.labels_callback,
            sensor_qos,
        )
        self.create_subscription(
            CameraInfo,
            "/wrist_camera/color/camera_info",
            self.camera_info_callback,
            sensor_qos,
        )
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)

    def rgb_callback(self, message: Image) -> None:
        try:
            value = np.asarray(
                self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            ).copy()
            self.latest_rgb = (stamp_sec(message), value)
        except Exception as exc:
            self.get_logger().warning(f"RGB decode failed: {exc}")

    def depth_callback(self, message: Image) -> None:
        try:
            value = np.asarray(
                self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            )
            if value.ndim == 3:
                value = value[..., 0]
            value = (
                value.astype(np.float32) * 0.001
                if message.encoding.upper() in {"16UC1", "MONO16"}
                else value.astype(np.float32)
            )
            self.latest_depth = (stamp_sec(message), value.copy())
        except Exception as exc:
            self.get_logger().warning(f"depth decode failed: {exc}")

    def labels_callback(self, message: Image) -> None:
        try:
            value = np.asarray(
                self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            )
            if value.ndim == 3:
                if not (
                    value.shape[2] == 3
                    and np.array_equal(value[..., 0], value[..., 1])
                    and np.array_equal(value[..., 1], value[..., 2])
                ):
                    raise ValueError("semantic-label RGB channels disagree")
                value = value[..., 0]
            self.latest_labels = (stamp_sec(message), value.astype(np.uint8).copy())
        except Exception as exc:
            self.get_logger().warning(f"label decode failed: {exc}")

    def camera_info_callback(self, message: CameraInfo) -> None:
        self.camera_info = message

    def joint_callback(self, message: JointState) -> None:
        self.joint_state = message

    def spin_until(self, predicate, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return bool(predicate())

    def view_ready(self) -> bool:
        if self.joint_state is None:
            return False
        positions = dict(zip(self.joint_state.name, self.joint_state.position))
        return all(
            name in positions and abs(float(positions[name]) - expected) <= 0.025
            for name, expected in zip(JOINT_NAMES, LOCKED_VIEW_JOINT_POSE)
        )

    def synchronized_tuple(self):
        if self.latest_rgb is None or self.latest_depth is None or self.latest_labels is None:
            return None
        stamps = [self.latest_rgb[0], self.latest_depth[0], self.latest_labels[0]]
        if max(stamps) - min(stamps) > 0.035:
            return None
        rgb, depth, labels = self.latest_rgb[1], self.latest_depth[1], self.latest_labels[1]
        if rgb.shape[:2] != depth.shape or depth.shape != labels.shape:
            raise WP2Error(
                f"registered sensor shape mismatch: {rgb.shape}, {depth.shape}, {labels.shape}"
            )
        return rgb.copy(), depth.copy(), labels.copy(), stamps

    def set_pose(self, model_name: str, xyz_yaw: list[float]) -> None:
        registry = self.plan["object_registry"]
        if model_name not in registry:
            raise WP2Error(f"unknown capture object: {model_name}")
        x_value, y_value, yaw = [float(item) for item in xyz_yaw]
        request = SetEntityPose.Request()
        request.entity.name = model_name
        request.entity.type = Entity.MODEL
        request.pose.position.x = x_value
        request.pose.position.y = y_value
        request.pose.position.z = float(registry[model_name]["z"])
        request.pose.orientation.z = math.sin(0.5 * yaw)
        request.pose.orientation.w = math.cos(0.5 * yaw)
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        result = future.result() if future.done() else None
        if result is None or not result.success:
            raise WP2Error(f"SetEntityPose failed: {model_name}")

    @staticmethod
    def transform_dict(transform) -> dict:
        value = transform.transform
        return {
            "parent_frame": transform.header.frame_id,
            "child_frame": transform.child_frame_id,
            "position": [value.translation.x, value.translation.y, value.translation.z],
            "orientation_xyzw": [
                value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w
            ],
        }

    def tf_snapshot(self) -> dict:
        snapshot = {}
        for child in ("camera_color_optical_frame", "gripper_tcp"):
            try:
                transform = self.tf_buffer.lookup_transform(
                    "base_link", child, Time(), timeout=Duration(seconds=0.5)
                )
                snapshot[child] = self.transform_dict(transform)
            except TransformException as exc:
                snapshot[child] = {"error": str(exc)}
        return snapshot

    def save_capture(self, capture: dict, sensor_tuple) -> dict:
        capture_id = str(capture["capture_id"])
        if CAPTURE_ID.fullmatch(capture_id) is None:
            raise WP2Error(f"unsafe capture_id: {capture_id}")
        final_dir = self.capture_root / capture_id
        if final_dir.exists():
            existing = read_json(final_dir / "capture_meta.json")
            for relative, expected in existing["artifact_sha256"].items():
                if sha256_file(final_dir / relative) != expected:
                    raise WP2Error(f"existing capture is corrupt: {capture_id}/{relative}")
            return existing

        rgb, depth_m, labels, stamps = sensor_tuple
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        valid_depth = np.isfinite(depth_m) & (depth_m >= 0.10) & (depth_m <= 2.0)
        if float(np.std(gray)) < 5.0:
            raise WP2Error(f"RGB QC failed: {capture_id}")
        if float(np.count_nonzero(valid_depth) / valid_depth.size) < 0.10:
            raise WP2Error(f"depth QC failed: {capture_id}")
        if np.unique(labels).size < 2:
            raise WP2Error(f"semantic label QC failed: {capture_id}")

        temp_parent = self.output_root / "raw" / ".capture_tmp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{capture_id}__", dir=temp_parent))
        try:
            rgb_path = temp_dir / "rgb_original.png"
            depth_path = temp_dir / "depth_metric.npy"
            labels_path = temp_dir / "semantic_instance_labels.png"
            camera_path = temp_dir / "camera_info.json"
            tf_path = temp_dir / "tf_snapshot.json"
            if not cv2.imwrite(str(rgb_path), rgb):
                raise WP2Error(f"cannot write {rgb_path}")
            np.save(depth_path, depth_m.astype(np.float32), allow_pickle=False)
            if not cv2.imwrite(str(labels_path), labels):
                raise WP2Error(f"cannot write {labels_path}")
            info = self.camera_info
            if info is None:
                raise WP2Error("camera info disappeared")
            write_json(camera_path, {
                "frame_id": info.header.frame_id,
                "width": int(info.width), "height": int(info.height),
                "distortion_model": info.distortion_model,
                "d": list(info.d), "k": list(info.k), "r": list(info.r), "p": list(info.p),
            })
            write_json(tf_path, self.tf_snapshot())
            artifact_hashes = {
                path.name: sha256_file(path)
                for path in (rgb_path, depth_path, labels_path, camera_path, tf_path)
            }
            meta = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "capture_id": capture_id,
                "family_id": capture["family_id"],
                "condition": capture["condition"],
                "seed": capture["seed"],
                "capture_backend": "Gazebo Fortress RGB-D + semantic segmentation",
                "requested_layout_base_link": capture["layout"],
                "capture_timestamps": {
                    "rgb_sec": stamps[0], "depth_sec": stamps[1], "labels_sec": stamps[2],
                    "max_spread_sec": max(stamps) - min(stamps),
                },
                "sensor_qc": {
                    "passed": True,
                    "rgb_gray_std": float(np.std(gray)),
                    "valid_depth_fraction": float(np.count_nonzero(valid_depth) / valid_depth.size),
                    "valid_depth_min_m": float(np.min(depth_m[valid_depth])),
                    "valid_depth_max_m": float(np.max(depth_m[valid_depth])),
                    "visible_semantic_labels": [int(value) for value in np.unique(labels)],
                },
                "artifact_sha256": artifact_hashes,
            }
            write_json(temp_dir / "capture_meta.json", meta)
            os.rename(temp_dir, final_dir)
            return meta
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def run(self, captures: list[dict]) -> list[dict]:
        if not self.reset_client.wait_for_service(timeout_sec=60.0):
            raise WP2Error("Gazebo SetEntityPose service unavailable")
        if not self.spin_until(lambda: self.camera_info is not None, 45.0):
            raise WP2Error("camera info unavailable")
        if not self.spin_until(self.view_ready, 45.0):
            raise WP2Error("locked wrist-camera VIEW_POSE not reached")
        records = []
        for capture_index, capture in enumerate(captures, start=1):
            capture_id = capture["capture_id"]
            final_dir = self.capture_root / capture_id
            if final_dir.is_dir():
                records.append(self.save_capture(capture, None))
                self.get_logger().info(
                    f"WP2_CAPTURE_REUSE {capture_index}/{len(captures)} {capture_id}"
                )
                continue
            self.get_logger().info(
                f"WP2_CAPTURE_RESET {capture_index}/{len(captures)} {capture_id}"
            )
            for model_name, pose in capture["layout"].items():
                self.set_pose(model_name, pose)
            self.latest_rgb = self.latest_depth = self.latest_labels = None
            deadline = time.monotonic() + self.settle_sec
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
            frozen = None

            def acquire() -> bool:
                nonlocal frozen
                frozen = self.synchronized_tuple()
                return frozen is not None

            if not self.spin_until(acquire, 20.0):
                raise WP2Error(f"synchronized sensor tuple unavailable: {capture_id}")
            records.append(self.save_capture(capture, frozen))
            self.get_logger().info(
                f"WP2_CAPTURE_SAVED {capture_index}/{len(captures)} {capture_id}"
            )
        return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-plan", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selection", choices=("smoke", "all"), default="all")
    parser.add_argument("--smoke-selection")
    parser.add_argument("--settle-sec", type=float, default=0.75)
    args = parser.parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    plan_path = Path(args.capture_plan).expanduser().resolve()
    plan = read_json(plan_path)
    if plan.get("protocol_id") != PROTOCOL_ID or plan.get("capture_count") != 100:
        raise WP2Error("capture plan does not match locked WP2 protocol")
    captures = list(plan["captures"])
    if args.selection == "smoke":
        if not args.smoke_selection:
            raise WP2Error("--smoke-selection is required")
        smoke = read_json(Path(args.smoke_selection).expanduser().resolve())
        selected = set(smoke["capture_ids"])
        captures = [item for item in captures if item["capture_id"] in selected]
        if len(captures) != 6:
            raise WP2Error(f"smoke must select 6 captures, got {len(captures)}")

    rclpy.init()
    node = WP2Capture(plan, output_root, args.settle_sec)
    try:
        node.run(captures)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    all_capture_dirs = sorted(
        path for path in (output_root / "raw" / "captures").iterdir() if path.is_dir()
    )
    inventory = []
    for directory in all_capture_dirs:
        meta = read_json(directory / "capture_meta.json")
        inventory.append({
            "capture_id": meta["capture_id"],
            "family_id": meta["family_id"],
            "condition": meta["condition"],
            "capture_meta_sha256": sha256_file(directory / "capture_meta.json"),
            "artifact_sha256": meta["artifact_sha256"],
        })
    write_json(output_root / "raw" / "raw_capture_manifest.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "capture_plan_sha256": sha256_file(plan_path),
        "capture_count": len(inventory),
        "captures": inventory,
    })
    checkpoint_name = (
        "01_smoke_raw_capture.json" if args.selection == "smoke"
        else "03_full_raw_capture.json"
    )
    write_json(output_root / "report_assets" / "checkpoints" / checkpoint_name, {
        "schema_version": 1,
        "checkpoint": "SMOKE_RAW_CAPTURED" if args.selection == "smoke" else "FULL_RAW_CAPTURED",
        "created_at_utc": utc_now(),
        "requested_selection": args.selection,
        "raw_capture_count": len(inventory),
        "raw_capture_manifest_sha256": sha256_file(output_root / "raw" / "raw_capture_manifest.json"),
        "capture_source_sha256": sha256_file(Path(__file__).resolve()),
    })
    print(f"WP2_CAPTURE_COMPLETE selection={args.selection} captures={len(inventory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
