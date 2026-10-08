#!/usr/bin/env python3
"""Record auditable RoboRefer RGB-D demo artifacts into one result folder."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


class RoboReferDemoRecorder(Node):
    """Persist inputs, outputs and lifecycle evidence without affecting inference."""

    def __init__(self) -> None:
        super().__init__("roborefer_demo_recorder")
        self.declare_parameter("result_dir", "")
        self.declare_parameter("main_prompt", "")
        self.declare_parameter("random_seed", 0)
        self.declare_parameter("scenario_name", "blind_compact_grocery_target")
        result_value = str(self.get_parameter("result_dir").value).strip()
        if not result_value:
            raise ValueError("result_dir must be non-empty")
        self._result_dir = Path(result_value).expanduser().resolve()
        self._result_dir.mkdir(parents=True, exist_ok=True)
        self._bridge = CvBridge()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth_m: Optional[np.ndarray] = None
        self._camera_info: Optional[CameraInfo] = None
        self._artifacts = set()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        retained_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Image, "/wrist_camera/color/image_raw", self._rgb_callback, sensor_qos
        )
        self.create_subscription(
            Image, "/wrist_camera/depth/image_raw", self._depth_callback, sensor_qos
        )
        self.create_subscription(
            CameraInfo,
            "/wrist_camera/color/camera_info",
            self._camera_info_callback,
            sensor_qos,
        )
        self.create_subscription(
            Image,
            "/ur3_perception/roborefer/image",
            lambda message: self._save_image_message(
                "roborefer_point_bbox_overlay.png", message
            ),
            retained_qos,
        )
        self.create_subscription(
            Image,
            "/ur3_perception/roborefer/mask",
            lambda message: self._save_image_message(
                "roborefer_depth_mask.png", message
            ),
            retained_qos,
        )
        self.create_subscription(
            Image,
            "/ur3_perception/roborefer/dimension_comparison",
            lambda message: self._save_image_message(
                "five_step_reasoning_panel.png", message
            ),
            retained_qos,
        )
        self.create_subscription(
            String,
            "/ur3_perception/roborefer/status",
            lambda message: self._json_status_callback(
                "roborefer_grounding_status.json", message, capture_rgbd=True
            ),
            10,
        )
        self.create_subscription(
            String,
            "/ur3_perception/roborefer/dimension_status",
            lambda message: self._json_status_callback(
                "five_step_reasoning_status.json", message
            ),
            retained_qos,
        )
        self.create_subscription(
            String,
            "/ur3_perception/target_selection",
            lambda message: self._json_status_callback(
                "verified_target_handoff.json", message
            ),
            retained_qos,
        )
        self.create_subscription(
            String, "/ur3_dataset/episode_event", self._episode_callback, retained_qos
        )
        self.create_subscription(Log, "/rosout", self._rosout_callback, 50)

        manifest = {
            "scenario": str(self.get_parameter("scenario_name").value),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": "RoboRefer-2B-SFT",
            "main_prompt": str(self.get_parameter("main_prompt").value),
            "random_seed_recorded_after_launch_creation": int(
                self.get_parameter("random_seed").value
            ),
            "vlm_inputs": ["wrist_rgb", "registered_depth_view", "main_prompt"],
            "target_coordinates_supplied_to_vlm": False,
            "target_pixel_supplied_to_vlm": False,
            "target_bbox_supplied_to_vlm": False,
            "gazebo_oracle_supplied_to_vlm": False,
            "registry_identity_release": "only_after_five_of_five_gate_passes",
            "post_hoc_episode_event_may_record_reset_pose_for_audit": True,
        }
        self._write_json("protocol_manifest.json", manifest)
        self.get_logger().info(f"DEMO_RECORDER_READY: {self._result_dir}")

    def _write_json(self, filename: str, payload) -> None:
        path = self._result_dir / filename
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._artifacts.add(filename)
        self._write_inventory()

    def _write_inventory(self) -> None:
        inventory = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": sorted(self._artifacts),
        }
        (self._result_dir / "artifact_inventory.json").write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _decode_json(message: String):
        try:
            return json.loads(message.data)
        except json.JSONDecodeError:
            return {"raw": message.data, "json_decode_error": True}

    def _rgb_callback(self, message: Image) -> None:
        try:
            self._latest_rgb = np.asarray(
                self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            ).copy()
        except Exception as exc:
            self.get_logger().warning(f"Recorder cannot decode RGB: {exc}")

    def _depth_callback(self, message: Image) -> None:
        try:
            depth = np.asarray(
                self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            )
            if depth.ndim == 3:
                depth = depth[..., 0]
            self._latest_depth_m = (
                depth.astype(np.float32) * 0.001
                if message.encoding.upper() in ("16UC1", "MONO16")
                else depth.astype(np.float32)
            )
        except Exception as exc:
            self.get_logger().warning(f"Recorder cannot decode depth: {exc}")

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self._camera_info = message

    def _save_image_message(self, filename: str, message: Image) -> None:
        if filename in self._artifacts:
            return
        try:
            frame = np.asarray(
                self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            )
            if frame.ndim == 3 and message.encoding.lower() == "rgb8":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(self._result_dir / filename), frame):
                raise RuntimeError("cv2.imwrite returned false")
            self._artifacts.add(filename)
            self._write_inventory()
        except Exception as exc:
            self.get_logger().warning(f"Recorder cannot save {filename}: {exc}")

    def _capture_locked_rgbd(self) -> None:
        if self._latest_rgb is not None and "locked_wrist_rgb.png" not in self._artifacts:
            if cv2.imwrite(
                str(self._result_dir / "locked_wrist_rgb.png"), self._latest_rgb
            ):
                self._artifacts.add("locked_wrist_rgb.png")
        if self._latest_depth_m is not None and "locked_depth_m.npy" not in self._artifacts:
            np.save(self._result_dir / "locked_depth_m.npy", self._latest_depth_m)
            self._artifacts.add("locked_depth_m.npy")
            valid = np.isfinite(self._latest_depth_m) & (self._latest_depth_m > 0.0)
            if np.any(valid):
                near, far = np.percentile(self._latest_depth_m[valid], [2.0, 98.0])
                scale = max(float(far - near), 1e-6)
                view = np.zeros_like(self._latest_depth_m, dtype=np.uint8)
                view[valid] = np.clip(
                    255.0 * (far - self._latest_depth_m[valid]) / scale,
                    0.0,
                    255.0,
                ).astype(np.uint8)
                cv2.imwrite(str(self._result_dir / "locked_depth_view.png"), view)
                self._artifacts.add("locked_depth_view.png")
        if self._camera_info is not None and "camera_intrinsics.json" not in self._artifacts:
            info = self._camera_info
            self._write_json("camera_intrinsics.json", {
                "frame_id": info.header.frame_id,
                "width": int(info.width),
                "height": int(info.height),
                "k": list(info.k),
                "d": list(info.d),
                "distortion_model": info.distortion_model,
            })
        self._write_inventory()

    def _json_status_callback(
        self, filename: str, message: String, capture_rgbd: bool = False
    ) -> None:
        payload = self._decode_json(message)
        self._write_json(filename, payload)
        if capture_rgbd and payload.get("status") == "SELECTED":
            self._capture_locked_rgbd()

    def _episode_callback(self, message: String) -> None:
        payload = self._decode_json(message)
        path = self._result_dir / "episode_events.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._artifacts.add("episode_events.jsonl")
        if payload.get("event") in ("episode_end", "run_complete"):
            self._write_json("run_summary.json", payload)
        else:
            self._write_inventory()

    def _rosout_callback(self, message: Log) -> None:
        name = str(message.name).lower()
        if not any(marker in name for marker in (
            "roborefer", "fixed_pick_place", "rgbd_object_pose"
        )):
            return
        record = {
            "stamp": {
                "sec": int(message.stamp.sec),
                "nanosec": int(message.stamp.nanosec),
            },
            "level": int(message.level),
            "name": message.name,
            "message": message.msg,
        }
        with (self._result_dir / "rosout.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._artifacts.add("rosout.jsonl")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoboReferDemoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
