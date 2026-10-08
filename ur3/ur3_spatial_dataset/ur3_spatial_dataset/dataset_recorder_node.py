"""Record reproducible UR3 RGB-D episodes and their spatial annotations."""

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
import rosbag2_py
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import serialize_message
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64, String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException, TransformListener
from ur3_perception_interfaces.msg import ObjectObservation


def _stamp_ns(message: Any, fallback_ns: int) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if value > 0:
            return value
    return int(fallback_ns)


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DatasetRecorder(Node):
    """Own one immutable run directory and one rosbag per episode."""

    def __init__(self) -> None:
        super().__init__("ur3_dataset_recorder")
        self.declare_parameter("dataset_root", "/tmp/ur3_spatial_dataset")
        self.declare_parameter("run_id", "")
        self.declare_parameter("random_seed", 42)
        self.declare_parameter("episode_count", 1)
        self.declare_parameter("registry_file", "")
        self.declare_parameter("world_file", "")
        self.declare_parameter("control_config_file", "")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_depth_optical_frame")
        self.declare_parameter("tcp_frame", "gripper_tcp")
        self.declare_parameter("sync_slop_sec", 0.12)
        self.declare_parameter("episode_event_topic", "/ur3_dataset/episode_event")
        self.declare_parameter("spatial_scene_topic", "/ur3_spatial/scene")

        root = Path(str(self.get_parameter("dataset_root").value)).expanduser()
        run_id = str(self.get_parameter("run_id").value).strip()
        if not run_id:
            run_id = time.strftime("run_%Y%m%d_%H%M%S")
        self._run_id = run_id
        self._run_root = root / run_id
        self._run_root.mkdir(parents=True, exist_ok=False)
        self._index_path = self._run_root / "dataset_index.jsonl"
        self._seed = int(self.get_parameter("random_seed").value)
        self._episode_count = int(self.get_parameter("episode_count").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._camera_frame = str(self.get_parameter("camera_frame").value)
        self._tcp_frame = str(self.get_parameter("tcp_frame").value)
        self._sync_slop_ns = int(
            float(self.get_parameter("sync_slop_sec").value) * 1_000_000_000
        )
        self._bridge = CvBridge()
        self._tf_buffer = Buffer(cache_time=Duration(seconds=60.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._writer = None
        self._episode_dir: Optional[Path] = None
        self._episode_record: Optional[dict] = None
        self._capture_requested = False
        self._capture_complete = False
        self._latest_rgb: Optional[Image] = None
        self._latest_depth: Optional[Image] = None
        self._latest_info: Optional[CameraInfo] = None
        self._latest_joint_state: Optional[JointState] = None
        self._latest_oracle: Dict[str, dict] = {}
        self._latest_scene: Optional[dict] = None

        sensor_qos = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        reliable_qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        transient_qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        static_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        event_topic = str(self.get_parameter("episode_event_topic").value)
        scene_topic = str(self.get_parameter("spatial_scene_topic").value)
        self._topic_types = {
            "/wrist_camera/color/image_raw": "sensor_msgs/msg/Image",
            "/wrist_camera/depth/image_raw": "sensor_msgs/msg/Image",
            "/wrist_camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
            "/tf": "tf2_msgs/msg/TFMessage",
            "/tf_static": "tf2_msgs/msg/TFMessage",
            "/joint_states": "sensor_msgs/msg/JointState",
            "/ur3_perception/oracle_tf": "tf2_msgs/msg/TFMessage",
            "/ur3_perception/object_observation":
                "ur3_perception_interfaces/msg/ObjectObservation",
            "/ur3_perception/object_pose_stable": "geometry_msgs/msg/PoseStamped",
            "/ur3_perception/status": "std_msgs/msg/String",
            "/ur3_perception/position_error_m": "std_msgs/msg/Float64",
            scene_topic: "std_msgs/msg/String",
            event_topic: "std_msgs/msg/String",
        }
        self._event_topic = event_topic
        self._scene_topic = scene_topic

        self.create_subscription(
            Image, "/wrist_camera/color/image_raw", self._rgb_callback, sensor_qos
        )
        self.create_subscription(
            Image, "/wrist_camera/depth/image_raw", self._depth_callback, sensor_qos
        )
        self.create_subscription(
            CameraInfo, "/wrist_camera/color/camera_info", self._info_callback, sensor_qos
        )
        self.create_subscription(TFMessage, "/tf", lambda msg: self._bag_only("/tf", msg), sensor_qos)
        self.create_subscription(
            TFMessage, "/tf_static", lambda msg: self._bag_only("/tf_static", msg), static_qos
        )
        self.create_subscription(
            JointState, "/joint_states", self._joint_callback, sensor_qos
        )
        self.create_subscription(
            TFMessage, "/ur3_perception/oracle_tf", self._oracle_callback, sensor_qos
        )
        self.create_subscription(
            ObjectObservation,
            "/ur3_perception/object_observation",
            lambda msg: self._bag_only("/ur3_perception/object_observation", msg),
            reliable_qos,
        )
        self.create_subscription(
            PoseStamped,
            "/ur3_perception/object_pose_stable",
            lambda msg: self._bag_only("/ur3_perception/object_pose_stable", msg),
            reliable_qos,
        )
        self.create_subscription(
            String,
            "/ur3_perception/status",
            lambda msg: self._bag_only("/ur3_perception/status", msg),
            reliable_qos,
        )
        self.create_subscription(
            Float64,
            "/ur3_perception/position_error_m",
            lambda msg: self._bag_only("/ur3_perception/position_error_m", msg),
            reliable_qos,
        )
        self.create_subscription(String, scene_topic, self._scene_callback, transient_qos)
        self.create_subscription(String, event_topic, self._event_callback, transient_qos)

        self._write_manifest()
        self.get_logger().info(f"DATASET_RUN_READY: {self._run_root}")

    def _write_manifest(self) -> None:
        source_files = {
            "object_registry": Path(str(self.get_parameter("registry_file").value)),
            "gazebo_world": Path(str(self.get_parameter("world_file").value)),
            "control_config": Path(str(self.get_parameter("control_config_file").value)),
        }
        manifest = {
            "schema_version": 1,
            "run_id": self._run_id,
            "created_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "robot": "ur3",
            "random_seed": self._seed,
            "episode_count": self._episode_count,
            "coordinate_convention": {
                "poses": self._base_frame,
                "left_right_above_below": "frozen_VIEW_POSE_image",
                "front_behind": "frozen_camera_optical_depth",
                "inside_near": self._base_frame,
            },
            "oracle_policy": "dataset_and_evaluation_only_not_control",
            "source_files": {
                label: {"path": str(path), "sha256": _sha256(path)}
                for label, path in source_files.items()
            },
            "bag_topics": self._topic_types,
            "reproduce_command": (
                "ros2 launch ur3_spatial_dataset spatial_dataset_demo.launch.py "
                f"random_seed:={self._seed} episode_count:={self._episode_count} "
                f"run_id:={self._run_id}_repro "
                f"dataset_root:={self._run_root.parent}"
            ),
        }
        _json_write(self._run_root / "manifest.json", manifest)

    def _start_episode(self, event: dict) -> None:
        if self._writer is not None:
            self._finish_episode({
                "success": False,
                "failure_stage": "INTERRUPTED_BY_NEXT_EPISODE",
                "failure_reason": "a new episode started before episode_end",
            })
        episode = int(event.get("episode", 0))
        self._episode_dir = self._run_root / f"episode_{episode:04d}"
        self._episode_dir.mkdir(parents=False, exist_ok=False)
        (self._episode_dir / "rgb").mkdir()
        (self._episode_dir / "depth").mkdir()
        (self._episode_dir / "tf").mkdir()

        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(
                uri=str(self._episode_dir / "rosbag"), storage_id="sqlite3"
            ),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )
        for topic, type_name in self._topic_types.items():
            writer.create_topic(rosbag2_py.TopicMetadata(
                name=topic, type=type_name, serialization_format="cdr"
            ))
        self._writer = writer
        self._capture_requested = False
        self._capture_complete = False
        self._episode_record = {
            "schema_version": 1,
            "run_id": self._run_id,
            "scene_id": f"scene_{episode:04d}",
            "episode": episode,
            "random_seed": int(event.get("random_seed", self._seed)),
            "object_pose_source": event.get("object_pose_source"),
            "randomized": bool(event.get("randomized", False)),
            "requested_object_pose": event.get("requested_object_pose"),
            "error_injection": {
                "axis": event.get("error_injection_axis"),
                "value": event.get("error_injection_value"),
            },
            "started": event.get("wall_time", time.time()),
            "files": {},
        }
        _json_write(self._episode_dir / "episode_start.json", self._episode_record)
        self.get_logger().info(f"RECORDING_EPISODE: {episode} -> {self._episode_dir}")

    def _write_bag(self, topic: str, message: Any) -> None:
        if self._writer is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        self._writer.write(topic, serialize_message(message), _stamp_ns(message, now_ns))

    def _bag_only(self, topic: str, message: Any) -> None:
        self._write_bag(topic, message)

    def _rgb_callback(self, message: Image) -> None:
        self._latest_rgb = message
        self._write_bag("/wrist_camera/color/image_raw", message)
        self._maybe_capture()

    def _depth_callback(self, message: Image) -> None:
        self._latest_depth = message
        self._write_bag("/wrist_camera/depth/image_raw", message)
        self._maybe_capture()

    def _info_callback(self, message: CameraInfo) -> None:
        self._latest_info = message
        self._write_bag("/wrist_camera/color/camera_info", message)
        self._maybe_capture()

    def _joint_callback(self, message: JointState) -> None:
        self._latest_joint_state = message
        self._write_bag("/joint_states", message)

    def _oracle_callback(self, message: TFMessage) -> None:
        for transform in message.transforms:
            value = transform.transform
            self._latest_oracle[transform.child_frame_id] = {
                "parent_frame": transform.header.frame_id,
                "position": [value.translation.x, value.translation.y, value.translation.z],
                "orientation_xyzw": [
                    value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w
                ],
            }
        self._write_bag("/ur3_perception/oracle_tf", message)

    def _scene_callback(self, message: String) -> None:
        try:
            self._latest_scene = json.loads(message.data)
        except json.JSONDecodeError:
            self._latest_scene = None
        self._write_bag(self._scene_topic, message)
        self._maybe_capture()

    def _event_callback(self, message: String) -> None:
        try:
            event = json.loads(message.data)
        except json.JSONDecodeError:
            return
        event_name = event.get("event")
        if event_name == "episode_start":
            self._start_episode(event)
            self._write_bag(self._event_topic, message)
            return
        self._write_bag(self._event_topic, message)
        if self._writer is None:
            return
        if (
            event_name == "step_end"
            and event.get("state") in ("FREEZE_PERCEPTION_POSE", "MOVE_VIEW")
            and bool(event.get("success"))
        ):
            self._capture_requested = True
            self._maybe_capture()
        elif event_name == "episode_end":
            self._capture_requested = True
            self._maybe_capture(force=True)
            self._finish_episode(event.get("result", {}))

    def _latest_sensor_tuple(self) -> Optional[Tuple[Image, Image, CameraInfo, bool, float]]:
        if self._latest_rgb is None or self._latest_depth is None or self._latest_info is None:
            return None
        now_ns = self.get_clock().now().nanoseconds
        stamps = [
            _stamp_ns(self._latest_rgb, now_ns),
            _stamp_ns(self._latest_depth, now_ns),
            _stamp_ns(self._latest_info, now_ns),
        ]
        spread_ns = max(stamps) - min(stamps)
        return (
            self._latest_rgb,
            self._latest_depth,
            self._latest_info,
            spread_ns <= self._sync_slop_ns,
            spread_ns / 1_000_000_000.0,
        )

    @staticmethod
    def _transform_dict(transform) -> dict:
        value = transform.transform
        return {
            "parent_frame": transform.header.frame_id,
            "child_frame": transform.child_frame_id,
            "stamp": {
                "sec": int(transform.header.stamp.sec),
                "nanosec": int(transform.header.stamp.nanosec),
            },
            "position": [value.translation.x, value.translation.y, value.translation.z],
            "orientation_xyzw": [
                value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w
            ],
        }

    def _tf_snapshot(self) -> dict:
        snapshot = {}
        for child in (self._camera_frame, self._tcp_frame):
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._base_frame, child, Time(), timeout=Duration(seconds=0.2)
                )
                snapshot[child] = self._transform_dict(transform)
            except TransformException as exc:
                snapshot[child] = {"error": str(exc)}
        return snapshot

    def _maybe_capture(self, force: bool = False) -> None:
        if not self._capture_requested or self._capture_complete or self._episode_dir is None:
            return
        sensor_tuple = self._latest_sensor_tuple()
        if sensor_tuple is None:
            return
        rgb_message, depth_message, info, synchronized, spread_sec = sensor_tuple
        if not synchronized and not force:
            return
        if (
            not force
            and self._latest_scene is not None
            and not bool(self._latest_scene.get("view_transform_frozen"))
        ):
            return
        try:
            rgb = self._bridge.imgmsg_to_cv2(rgb_message, desired_encoding="bgr8")
            depth = self._bridge.imgmsg_to_cv2(depth_message, desired_encoding="passthrough")
        except Exception as exc:  # cv_bridge raises different exception types by distro
            self.get_logger().error(f"CAPTURE_CONVERSION_FAILED: {exc}")
            return
        rgb_path = self._episode_dir / "rgb" / "view.png"
        depth_path = self._episode_dir / "depth" / "view.npy"
        if not cv2.imwrite(str(rgb_path), rgb):
            self.get_logger().error(f"CAPTURE_WRITE_FAILED: {rgb_path}")
            return
        np.save(depth_path, np.asarray(depth, dtype=np.float32), allow_pickle=False)

        camera_info = {
            "header_frame_id": info.header.frame_id,
            "width": int(info.width),
            "height": int(info.height),
            "distortion_model": info.distortion_model,
            "d": list(info.d),
            "k": list(info.k),
            "r": list(info.r),
            "p": list(info.p),
            "sensor_stamp_spread_sec": spread_sec,
            "synchronized": synchronized,
        }
        _json_write(self._episode_dir / "camera_info.json", camera_info)
        _json_write(self._episode_dir / "tf" / "tf_snapshot.json", self._tf_snapshot())
        if self._latest_joint_state is not None:
            joint = self._latest_joint_state
            _json_write(self._episode_dir / "joint_state.json", {
                "name": list(joint.name),
                "position": list(joint.position),
                "velocity": list(joint.velocity),
                "effort": list(joint.effort),
            })
        if self._latest_scene is not None:
            _json_write(self._episode_dir / "spatial_scene.json", self._latest_scene)
        _json_write(self._episode_dir / "ground_truth.json", {
            "oracle_usage": "dataset_and_evaluation_only_not_control",
            "transforms": self._latest_oracle,
            "canonical_objects": (
                self._latest_scene.get("objects", []) if self._latest_scene else []
            ),
        })
        self._capture_complete = True
        if self._episode_record is not None:
            self._episode_record["files"].update({
                "rgb": "rgb/view.png",
                "depth": "depth/view.npy",
                "camera_info": "camera_info.json",
                "tf_snapshot": "tf/tf_snapshot.json",
                "joint_state": "joint_state.json",
                "ground_truth": "ground_truth.json",
                "spatial_scene": "spatial_scene.json",
                "rosbag": "rosbag",
            })
            self._episode_record["sensor_sync"] = {
                "valid": synchronized,
                "stamp_spread_sec": spread_sec,
                "allowed_sec": self._sync_slop_ns / 1_000_000_000.0,
            }
        self.get_logger().info(
            f"DATASET_SNAPSHOT_SAVED: synchronized={synchronized}, spread={spread_sec:.4f}s"
        )

    def _finish_episode(self, result: dict) -> None:
        if self._writer is None or self._episode_dir is None or self._episode_record is None:
            return
        self._episode_record["result"] = result
        self._episode_record["success"] = bool(result.get("success", False))
        self._episode_record["failure_stage"] = result.get("failure_stage", "")
        self._episode_record["failure_reason"] = result.get("failure_reason", "")
        self._episode_record["finished_wall_time"] = time.time()
        self._episode_record["snapshot_complete"] = self._capture_complete
        if self._latest_scene is not None:
            _json_write(self._episode_dir / "spatial_scene_final.json", self._latest_scene)
            self._episode_record["files"]["spatial_scene_final"] = "spatial_scene_final.json"
        _json_write(self._episode_dir / "ground_truth_final.json", {
            "oracle_usage": "dataset_and_evaluation_only_not_control",
            "transforms": self._latest_oracle,
            "canonical_objects": (
                self._latest_scene.get("objects", []) if self._latest_scene else []
            ),
        })
        self._episode_record["files"]["ground_truth_final"] = "ground_truth_final.json"
        _json_write(self._episode_dir / "result.json", result)
        _json_write(self._episode_dir / "episode.json", self._episode_record)
        with self._index_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "episode": self._episode_record["episode"],
                "scene_id": self._episode_record["scene_id"],
                "path": self._episode_dir.name,
                "success": self._episode_record["success"],
                "failure_stage": self._episode_record["failure_stage"],
                "snapshot_complete": self._capture_complete,
            }, ensure_ascii=False) + "\n")
        episode = self._episode_record["episode"]
        success = self._episode_record["success"]
        self._writer = None
        gc.collect()
        self.get_logger().info(
            f"EPISODE_DATASET_COMPLETE: episode={episode}, success={success}"
        )
        self._episode_dir = None
        self._episode_record = None
        self._capture_requested = False
        self._capture_complete = False

    def close(self) -> None:
        if self._writer is not None:
            self._capture_requested = True
            self._maybe_capture(force=True)
            self._finish_episode({
                "success": False,
                "failure_stage": "RECORDER_SHUTDOWN",
                "failure_reason": "node stopped before episode_end",
            })


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatasetRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
