"""Publish a canonical object registry and spatial relation graph for UR3 scenes."""

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException, TransformListener
from ur3_perception_interfaces.msg import ObjectObservation

from .spatial_relations import compute_relations, project_point, transform_point


class SpatialSceneNode(Node):
    """Create the dataset-only ground-truth graph; never publishes control poses."""

    def __init__(self) -> None:
        super().__init__("spatial_scene")
        self.declare_parameter("registry_file", "")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_depth_optical_frame")
        self.declare_parameter("oracle_topic", "/ur3_perception/oracle_tf")
        self.declare_parameter("observation_topic", "/ur3_perception/object_observation")
        self.declare_parameter("camera_info_topic", "/wrist_camera/color/camera_info")
        self.declare_parameter("episode_event_topic", "/ur3_dataset/episode_event")
        self.declare_parameter("scene_topic", "/ur3_spatial/scene")
        self.declare_parameter("publish_rate_hz", 2.0)

        registry_path = Path(str(self.get_parameter("registry_file").value)).expanduser()
        if not registry_path.is_file():
            raise ValueError(f"registry_file does not exist: {registry_path}")
        self._registry_path = registry_path
        self._registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._camera_frame = str(self.get_parameter("camera_frame").value)
        self._objects_config = self._registry["objects"]
        self._thresholds = self._registry.get("thresholds", {})
        self._ground_truth = {}
        self._observation = None
        self._intrinsics = None
        self._view_transform = None
        self._freeze_view_requested = False
        self._scene_id = "scene_0000"
        self._episode = 0
        self._last_log = 0.0

        self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        sensor_qos = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        event_qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        scene_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            TFMessage, str(self.get_parameter("oracle_topic").value),
            self._oracle_callback, sensor_qos,
        )
        self.create_subscription(
            ObjectObservation, str(self.get_parameter("observation_topic").value),
            self._observation_callback, 10,
        )
        self.create_subscription(
            CameraInfo, str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback, sensor_qos,
        )
        self.create_subscription(
            String, str(self.get_parameter("episode_event_topic").value),
            self._event_callback, event_qos,
        )
        self._scene_pub = self.create_publisher(
            String, str(self.get_parameter("scene_topic").value), scene_qos
        )
        rate = max(0.2, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / rate, self._publish_scene)
        self.get_logger().info(
            f"Canonical spatial registry loaded: {len(self._objects_config)} objects, "
            f"relations fixed to {self._registry['view_pose_id']}"
        )

    def _oracle_callback(self, msg: TFMessage) -> None:
        known_frames = {item["model_frame"] for item in self._objects_config}
        for transform in msg.transforms:
            if transform.child_frame_id not in known_frames:
                continue
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            self._ground_truth[transform.child_frame_id] = [
                float(translation.x), float(translation.y), float(translation.z),
                float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w),
            ]

    def _observation_callback(self, msg: ObjectObservation) -> None:
        self._observation = copy.deepcopy(msg)

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self._intrinsics = {
            "fx": float(msg.k[0]), "fy": float(msg.k[4]),
            "cx": float(msg.k[2]), "cy": float(msg.k[5]),
            "width": int(msg.width), "height": int(msg.height),
        }

    def _event_callback(self, msg: String) -> None:
        try:
            event = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        event_name = event.get("event")
        if event_name == "episode_start":
            self._episode = int(event.get("episode", 0))
            self._scene_id = f"scene_{self._episode:04d}"
            self._view_transform = None
            self._freeze_view_requested = False
        elif (
            event_name == "step_end"
            and event.get("state") == "MOVE_VIEW"
            and bool(event.get("success"))
        ):
            self._freeze_view_requested = True
            self._try_freeze_view_transform()

    def _try_freeze_view_transform(self) -> None:
        if not self._freeze_view_requested or self._view_transform is not None:
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                self._camera_frame, self._base_frame, Time(),
                timeout=Duration(seconds=0.1),
            ).transform
        except TransformException:
            return
        self._view_transform = {
            "translation": [
                float(transform.translation.x),
                float(transform.translation.y),
                float(transform.translation.z),
            ],
            "orientation_xyzw": [
                float(transform.rotation.x), float(transform.rotation.y),
                float(transform.rotation.z), float(transform.rotation.w),
            ],
        }
        self._freeze_view_requested = False
        self.get_logger().info(
            f"VIEW_TRANSFORM_FROZEN: {self._registry['view_pose_id']} "
            f"({self._base_frame} -> {self._camera_frame})"
        )

    @staticmethod
    def _pose_dict(values) -> dict:
        return {
            "position": [round(float(value), 7) for value in values[:3]],
            "orientation_xyzw": [round(float(value), 7) for value in values[3:7]],
        }

    def _canonical_objects(self) -> list:
        objects = []
        for configured in self._objects_config:
            pose_values = self._ground_truth.get(configured["model_frame"])
            if pose_values is None:
                pose_values = configured.get("fallback_pose")
            if pose_values is None:
                continue
            item = {
                "id": str(configured["id"]),
                "category": str(configured["category"]),
                "attributes": dict(configured.get("attributes", {})),
                "pose": self._pose_dict(pose_values),
                "dimensions": [float(value) for value in configured["dimensions"]],
                "pose_source": str(configured["pose_source"]),
                "model_frame": str(configured["model_frame"]),
            }
            if "container_inner_dimensions" in configured:
                item["container_inner_dimensions"] = [
                    float(value) for value in configured["container_inner_dimensions"]
                ]
            if configured["id"] == "cube_red_01" and self._observation is not None:
                pose = self._observation.pose.pose
                item["perception"] = {
                    "pose": {
                        "position": [pose.position.x, pose.position.y, pose.position.z],
                        "orientation_xyzw": [
                            pose.orientation.x, pose.orientation.y,
                            pose.orientation.z, pose.orientation.w,
                        ],
                    },
                    "dimensions": [
                        self._observation.dimensions.x,
                        self._observation.dimensions.y,
                        self._observation.dimensions.z,
                    ],
                    "status": self._observation.status,
                    "confidence": float(self._observation.confidence),
                    "valid_frames": int(self._observation.valid_frames),
                    "position_stddev_m": float(self._observation.position_stddev_m),
                }
            objects.append(item)
        return objects

    def _projections(self, objects: list) -> dict:
        if self._view_transform is None or self._intrinsics is None:
            return {}
        transform = self._view_transform
        projections = {}
        for item in objects:
            camera_point = transform_point(
                item["pose"]["position"],
                transform["translation"],
                transform["orientation_xyzw"],
            )
            projected = project_point(camera_point, self._intrinsics)
            if projected is None:
                continue
            u, v, depth = projected
            # Relations in the main dataset are only defined for visible
            # instances in the frozen view, as required by the research plan.
            if 0.0 <= u < self._intrinsics["width"] and 0.0 <= v < self._intrinsics["height"]:
                projections[item["id"]] = (u, v, depth)
                item["view_projection"] = {
                    "u": round(u, 3), "v": round(v, 3), "depth_m": round(depth, 6)
                }
        return projections

    def _publish_scene(self) -> None:
        self._try_freeze_view_transform()
        objects = self._canonical_objects()
        projections = self._projections(objects)
        relations = compute_relations(
            objects,
            projections,
            pixel_margin=float(self._thresholds.get("pixel_margin", 8.0)),
            depth_margin_m=float(self._thresholds.get("depth_margin_m", 0.015)),
            near_threshold_m=float(self._thresholds.get("near_threshold_m", 0.12)),
        )
        stamp = self.get_clock().now().to_msg()
        scene = {
            "schema_version": 1,
            "scene_id": self._scene_id,
            "episode": self._episode,
            "timestamp": {"sec": stamp.sec, "nanosec": stamp.nanosec},
            "frame_id": self._base_frame,
            "camera_frame": self._camera_frame,
            "view_pose_id": self._registry["view_pose_id"],
            "view_transform_frozen": self._view_transform is not None,
            "oracle_usage": "dataset_and_evaluation_only_not_control",
            "objects": objects,
            "relations": relations,
        }
        message = String()
        message.data = json.dumps(scene, ensure_ascii=False, separators=(",", ":"))
        self._scene_pub.publish(message)
        now = time.monotonic()
        if self._view_transform is not None and now - self._last_log >= 3.0:
            counts = {}
            for relation in relations:
                counts[relation["predicate"]] = counts.get(relation["predicate"], 0) + 1
            self.get_logger().info(
                f"SPATIAL_SCENE {self._scene_id}: objects={len(objects)}, "
                f"visible={len(projections)}, relations={counts}"
            )
            self._last_log = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpatialSceneNode()
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
