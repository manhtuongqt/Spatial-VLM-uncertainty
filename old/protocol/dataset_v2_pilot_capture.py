#!/usr/bin/python3
"""Capture the execution-locked Dataset V2 pilot from Gazebo."""

from __future__ import annotations

import argparse
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
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformException, TransformListener
from ur3_moveit_control.action import UR3Control

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp2_common import WP2Error, read_json, sha256_file, utc_now, write_json  # noqa: E402


PROTOCOL_ID = "roborefer_dataset_v2_pilot_only"
CAPTURE_ID = re.compile(r"^v2pilot_family_[0-9]{6}__(?:clean|occlusion)_capture$")
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def stamp_sec(message) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9


def verify_execution_lock(workspace: Path, plan_path: Path, lock_path: Path, output_root: Path) -> dict:
    lock = read_json(lock_path)
    if lock.get("protocol_id") != PROTOCOL_ID or not lock.get("capture_authorized"):
        raise WP2Error("pilot execution lock does not authorize Gazebo capture")
    if lock.get("decision") != "GO_PILOT_GAZEBO_CAPTURE_60":
        raise WP2Error(f"unexpected pilot execution decision: {lock.get('decision')}")
    expected_output = (workspace / str(lock["capture_output_root"])).resolve()
    if output_root != expected_output:
        raise WP2Error(f"output root differs from execution lock: {output_root} != {expected_output}")
    for relative, expected in lock["locked_artifact_sha256"].items():
        path = (workspace / relative).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise WP2Error(f"execution-lock artifact changed: {relative}")
    if sha256_file(plan_path) != lock["capture_plan_sha256"]:
        raise WP2Error("capture plan hash differs from execution lock")
    return lock


class PilotCapture(Node):
    def __init__(self, plan: dict, output_root: Path, settle_sec: float) -> None:
        super().__init__("dataset_v2_pilot_capture")
        self.plan = plan
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
        self.pose_client = self.create_client(SetEntityPose, "/world/ur3_pick_place/set_pose")
        self.motion_client = ActionClient(self, UR3Control, "/ur3_control")
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Image, "/wrist_camera/color/image_raw", self.rgb_callback, sensor_qos)
        self.create_subscription(Image, "/wrist_camera/depth/image_raw", self.depth_callback, sensor_qos)
        self.create_subscription(Image, "/wrist_camera/evaluation_labels/labels_map", self.labels_callback, sensor_qos)
        self.create_subscription(CameraInfo, "/wrist_camera/color/camera_info", self.camera_info_callback, sensor_qos)
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)

    def rgb_callback(self, message: Image) -> None:
        try:
            value = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")).copy()
            self.latest_rgb = (stamp_sec(message), value)
        except Exception as exc:
            self.get_logger().warning(f"RGB decode failed: {exc}")

    def depth_callback(self, message: Image) -> None:
        try:
            value = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough"))
            if value.ndim == 3:
                value = value[..., 0]
            value = value.astype(np.float32) * (0.001 if message.encoding.upper() in {"16UC1", "MONO16"} else 1.0)
            self.latest_depth = (stamp_sec(message), value.copy())
        except Exception as exc:
            self.get_logger().warning(f"depth decode failed: {exc}")

    def labels_callback(self, message: Image) -> None:
        try:
            value = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough"))
            if value.ndim == 3:
                if not (value.shape[2] == 3 and np.array_equal(value[..., 0], value[..., 1]) and np.array_equal(value[..., 1], value[..., 2])):
                    raise ValueError("semantic-label RGB channels disagree")
                value = value[..., 0]
            if int(np.max(value)) > 255:
                raise ValueError("semantic labels exceed uint8 contract")
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
            rclpy.spin_once(self, timeout_sec=.05)
        return bool(predicate())

    def joint_positions(self) -> dict[str, float]:
        if self.joint_state is None:
            return {}
        return {name: float(value) for name, value in zip(self.joint_state.name, self.joint_state.position)}

    def view_ready(self, requested: list[float]) -> bool:
        positions = self.joint_positions()
        return all(name in positions and abs(positions[name] - float(expected)) <= .025 for name, expected in zip(JOINT_NAMES, requested))

    def move_camera(self, requested: list[float]) -> None:
        if len(requested) != 6 or not all(math.isfinite(float(value)) for value in requested):
            raise WP2Error("view_joint_pose must contain six finite values")
        if self.view_ready(requested):
            return
        if not self.motion_client.wait_for_server(timeout_sec=60.0):
            raise WP2Error("UR3 motion action server unavailable")
        goal = UR3Control.Goal()
        goal.command_type = UR3Control.Goal.MOVE_JOINT
        goal.joint_goal.position = [float(value) for value in requested]
        future = self.motion_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        handle = future.result() if future.done() else None
        if handle is None or not handle.accepted:
            raise WP2Error("camera joint motion rejected")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=90.0)
        result = result_future.result() if result_future.done() else None
        if result is None or not result.result.success:
            raise WP2Error("camera joint motion failed")
        if not self.spin_until(lambda: self.view_ready(requested), 20.0):
            raise WP2Error("camera joint pose did not reach tolerance")

    def synchronized_tuple(self):
        if self.latest_rgb is None or self.latest_depth is None or self.latest_labels is None:
            return None
        stamps = [self.latest_rgb[0], self.latest_depth[0], self.latest_labels[0]]
        if max(stamps) - min(stamps) > float(self.plan["sensor_contract"]["max_timestamp_spread_sec"]):
            return None
        rgb, depth, labels = self.latest_rgb[1], self.latest_depth[1], self.latest_labels[1]
        if rgb.shape[:2] != depth.shape or depth.shape != labels.shape:
            raise WP2Error(f"registered sensor shape mismatch: {rgb.shape}, {depth.shape}, {labels.shape}")
        return rgb.copy(), depth.copy(), labels.copy(), stamps

    def set_pose(self, model_name: str, xyz_yaw: list[float]) -> None:
        registry = self.plan["object_registry"]
        if model_name not in registry:
            raise WP2Error(f"unknown capture object: {model_name}")
        request = SetEntityPose.Request()
        request.entity.name = model_name
        request.entity.type = Entity.MODEL
        request.pose.position.x = float(xyz_yaw[0])
        request.pose.position.y = float(xyz_yaw[1])
        request.pose.position.z = float(registry[model_name]["z"])
        request.pose.orientation.z = math.sin(.5 * float(xyz_yaw[2]))
        request.pose.orientation.w = math.cos(.5 * float(xyz_yaw[2]))
        future = self.pose_client.call_async(request)
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
            "orientation_xyzw": [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
        }

    def tf_snapshot(self) -> dict:
        snapshot = {}
        for child in ("camera_color_optical_frame", "gripper_tcp"):
            try:
                transform = self.tf_buffer.lookup_transform("base_link", child, Time(), timeout=Duration(seconds=.5))
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
        if sensor_tuple is None:
            raise WP2Error(f"missing sensor tuple for new capture: {capture_id}")
        rgb, depth_m, labels, stamps = sensor_tuple
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        valid_depth = np.isfinite(depth_m) & (depth_m >= .10) & (depth_m <= 2.0)
        valid_fraction = float(np.count_nonzero(valid_depth) / valid_depth.size)
        if float(np.std(gray)) < 5.0 or valid_fraction < .10 or np.unique(labels).size < 2:
            raise WP2Error(f"raw sensor QC failed: {capture_id}")
        visible_labels = {int(value) for value in np.unique(labels)}
        required_labels = {int(value) for value in capture.get("required_visible_label_ids", [])}
        missing_labels = sorted(required_labels - visible_labels)
        if missing_labels:
            raise WP2Error(
                f"required task evidence not visible: {capture_id} missing_labels={missing_labels}"
            )

        temp_parent = self.output_root / "raw" / ".capture_tmp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{capture_id}__", dir=temp_parent))
        try:
            paths = {
                "rgb_original.png": temp_dir / "rgb_original.png",
                "depth_metric.npy": temp_dir / "depth_metric.npy",
                "semantic_instance_labels.png": temp_dir / "semantic_instance_labels.png",
                "camera_info.json": temp_dir / "camera_info.json",
                "tf_snapshot.json": temp_dir / "tf_snapshot.json",
            }
            if not cv2.imwrite(str(paths["rgb_original.png"]), rgb):
                raise WP2Error("cannot write RGB")
            np.save(paths["depth_metric.npy"], depth_m.astype(np.float32), allow_pickle=False)
            if not cv2.imwrite(str(paths["semantic_instance_labels.png"]), labels):
                raise WP2Error("cannot write semantic labels")
            info = self.camera_info
            if info is None:
                raise WP2Error("camera info unavailable at save")
            write_json(paths["camera_info.json"], {
                "frame_id": info.header.frame_id, "width": int(info.width), "height": int(info.height),
                "distortion_model": info.distortion_model, "d": list(info.d), "k": list(info.k),
                "r": list(info.r), "p": list(info.p),
            })
            write_json(paths["tf_snapshot.json"], self.tf_snapshot())
            hashes = {name: sha256_file(path) for name, path in paths.items()}
            meta = {
                "schema_version": 1, "protocol_id": PROTOCOL_ID,
                "capture_id": capture_id, "family_id": capture["family_id"],
                "condition": capture["condition"], "seed": int(capture["seed"]),
                "ood_axis": capture["ood_axis"], "camera_bin_id": capture["camera_bin_id"],
                "capture_backend": "Gazebo Fortress RGB-D + semantic segmentation",
                "requested_layout_base_link": capture["layout"],
                "requested_view_joint_pose": capture["view_joint_pose"],
                "realized_view_joint_pose": [self.joint_positions().get(name) for name in JOINT_NAMES],
                "capture_timestamps": {
                    "rgb_sec": stamps[0], "depth_sec": stamps[1], "labels_sec": stamps[2],
                    "max_spread_sec": max(stamps) - min(stamps),
                },
                "sensor_qc": {
                    "passed": True, "rgb_gray_std": float(np.std(gray)),
                    "valid_depth_fraction": valid_fraction,
                    "valid_depth_min_m": float(np.min(depth_m[valid_depth])),
                    "valid_depth_max_m": float(np.max(depth_m[valid_depth])),
                    "visible_semantic_labels": sorted(visible_labels),
                    "required_visible_semantic_labels": sorted(required_labels),
                    "registered_shape_hw": [int(depth_m.shape[0]), int(depth_m.shape[1])],
                },
                "artifact_sha256": hashes,
            }
            write_json(temp_dir / "capture_meta.json", meta)
            os.rename(temp_dir, final_dir)
            return meta
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def run(self, captures: list[dict]) -> list[dict]:
        if not self.pose_client.wait_for_service(timeout_sec=60.0):
            raise WP2Error("Gazebo SetEntityPose service unavailable")
        if not self.spin_until(lambda: self.camera_info is not None and self.joint_state is not None, 60.0):
            raise WP2Error("camera info or joint state unavailable")
        records = []
        for index, capture in enumerate(captures, start=1):
            capture_id = capture["capture_id"]
            if (self.capture_root / capture_id).is_dir():
                records.append(self.save_capture(capture, None))
                self.get_logger().info(f"V2_PILOT_CAPTURE_REUSE {index}/{len(captures)} {capture_id}")
                continue
            self.get_logger().info(f"V2_PILOT_CAPTURE_RESET {index}/{len(captures)} {capture_id}")
            self.move_camera([float(value) for value in capture["view_joint_pose"]])
            for model_name, pose in capture["layout"].items():
                self.set_pose(model_name, pose)
            self.latest_rgb = self.latest_depth = self.latest_labels = None
            deadline = time.monotonic() + self.settle_sec
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=.05)
            frozen = None

            def acquire() -> bool:
                nonlocal frozen
                frozen = self.synchronized_tuple()
                return frozen is not None

            if not self.spin_until(acquire, 20.0):
                raise WP2Error(f"synchronized sensor tuple unavailable: {capture_id}")
            records.append(self.save_capture(capture, frozen))
            self.get_logger().info(f"V2_PILOT_CAPTURE_SAVED {index}/{len(captures)} {capture_id}")
        return records


def write_raw_manifest(output_root: Path, plan_path: Path, lock_path: Path) -> dict:
    inventory = []
    capture_root = output_root / "raw" / "captures"
    for directory in sorted(path for path in capture_root.iterdir() if path.is_dir()):
        meta = read_json(directory / "capture_meta.json")
        inventory.append({
            "capture_id": meta["capture_id"], "family_id": meta["family_id"],
            "condition": meta["condition"], "capture_meta_sha256": sha256_file(directory / "capture_meta.json"),
            "artifact_sha256": meta["artifact_sha256"],
        })
    manifest = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "capture_plan_sha256": sha256_file(plan_path), "execution_lock_sha256": sha256_file(lock_path),
        "capture_count": len(inventory), "complete": len(inventory) == 60, "captures": inventory,
    }
    write_json(output_root / "raw" / "raw_capture_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-plan", required=True)
    parser.add_argument("--execution-lock", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--settle-sec", type=float, default=.8)
    parser.add_argument("--max-captures", type=int, default=0, help="diagnostic prefix; 0 captures all 60")
    parser.add_argument(
        "--family-ids", default="",
        help="diagnostic comma-separated family IDs; empty captures all families",
    )
    args = parser.parse_args()
    plan_path = Path(args.capture_plan).expanduser().resolve()
    lock_path = Path(args.execution_lock).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    workspace = Path(__file__).resolve().parents[1]
    plan = read_json(plan_path)
    if plan.get("protocol_id") != PROTOCOL_ID or plan.get("capture_count") != 60:
        raise WP2Error("capture plan does not match Dataset V2 pilot contract")
    verify_execution_lock(workspace, plan_path, lock_path, output_root)
    captures = list(plan["captures"])
    selected_family_ids = {value.strip() for value in args.family_ids.split(",") if value.strip()}
    if selected_family_ids and args.max_captures:
        raise WP2Error("--family-ids and --max-captures are mutually exclusive")
    if selected_family_ids:
        known_family_ids = {item["family_id"] for item in captures}
        unknown = selected_family_ids - known_family_ids
        if unknown:
            raise WP2Error(f"unknown --family-ids: {sorted(unknown)}")
        captures = [item for item in captures if item["family_id"] in selected_family_ids]
    if args.max_captures:
        if args.max_captures < 1 or args.max_captures > 60:
            raise WP2Error("--max-captures must be 1..60")
        captures = captures[:args.max_captures]
    output_root.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = PilotCapture(plan, output_root, args.settle_sec)
    try:
        node.run(captures)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    manifest = write_raw_manifest(output_root, plan_path, lock_path)
    write_json(output_root / "report_assets" / "checkpoints" / "01_raw_capture.json", {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "checkpoint": "FULL_RAW_CAPTURED" if manifest["complete"] else "PARTIAL_RAW_CAPTURED",
        "created_at_utc": utc_now(), "raw_capture_count": manifest["capture_count"],
        "raw_capture_manifest_sha256": sha256_file(output_root / "raw" / "raw_capture_manifest.json"),
        "capture_source_sha256": sha256_file(Path(__file__).resolve()),
    })
    print(f"DATASET_V2_PILOT_CAPTURE_COMPLETE captures={manifest['capture_count']} complete={manifest['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
