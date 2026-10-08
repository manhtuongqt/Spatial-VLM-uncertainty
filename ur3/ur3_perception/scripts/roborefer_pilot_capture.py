#!/usr/bin/env python3
"""Capture immutable no-manipulation RGB-D pilot scenes for RoboRefer.

This process owns scene reset and sensor capture only.  It never loads the VLM,
publishes a target, reads target annotations, or commands a grasp.  Semantic
labels and reset poses are written under evaluator/ and are deliberately absent
from the input manifest consumed by the offline benchmark runner.
"""

import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple

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
import yaml


FORBIDDEN_INPUT_KEYS = {
    "scene_family_id",
    "task_type",
    "target_model",
    "target_id",
    "target_pose",
    "target_pixel",
    "target_bbox",
    "target_mask",
    "semantic_label",
    "oracle",
    "ground_truth",
}

# Input-only sensor checks are fixed protocol constants.  They deliberately do
# not inspect semantic labels or target annotations, so they can fail closed
# before the VLM is queried without opening the evaluation oracle.
MIN_RGB_STD = 5.0
MIN_RGB_DYNAMIC_RANGE = 20.0
MIN_RGB_NONZERO_FRACTION = 0.05
MIN_VALID_DEPTH_FRACTION = 0.10
MIN_DEPTH_DYNAMIC_RANGE_M = 0.05
MIN_VALID_DEPTH_M = 0.05
MAX_VALID_DEPTH_M = 2.0
REQUIRED_SOURCE_ARTIFACTS = {
    "RoboRefer/API/api.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_capture.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_runner.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_validate.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_evaluator.py",
    "ur3/ur3_perception/scripts/roborefer_grounder.py",
    "ur3/ur3_perception/scripts/spatial_point_utils.py",
    "ur3/ur3_perception/scripts/move_camera_to_view.py",
    "ur3/ur3_perception/launch/roborefer_pilot_capture.launch.py",
    "ur3/ur3_perception/config/roborefer_pilot_v0_scenes.yaml",
    "ur3/ur3_perception/config/roborefer_pilot_v0_gate.yaml",
    "ur3/ur3_perception/config/roborefer_pilot_v0_annotations.yaml",
    "ur3/ur3_perception/schemas/roborefer_pilot_prediction.schema.json",
    "ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro",
    "ur3/ur3_moveit_control/launch/ur3_susgrip_sim.launch.py",
    "ur3/ur_simulation_gz/worlds/ur3_pick_place.sdf",
    "ur3/ur_simulation_gz/launch/ur_sim_control.launch.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stamp_seconds(message) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def validate_scene_config(config: dict) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("scene config schema_version must be 1")
    scenes = config.get("scenes")
    objects = config.get("objects")
    expected_scene_count = int(config.get("expected_scene_count", 10))
    if expected_scene_count <= 0:
        raise ValueError("expected_scene_count must be positive")
    if not isinstance(scenes, list) or len(scenes) != expected_scene_count:
        raise ValueError(
            f"scene config must preregister exactly {expected_scene_count} scenes"
        )
    if not isinstance(objects, dict) or not objects:
        raise ValueError("scene config objects must be a non-empty mapping")
    scene_ids = [str(item.get("scene_id", "")) for item in scenes]
    if not all(scene_ids) or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("scene_id values must be unique and non-empty")
    for model_name, item in objects.items():
        if not isinstance(item, dict) or "z" not in item or "storage_pose" not in item:
            raise ValueError(f"invalid object entry: {model_name}")
        if len(item["storage_pose"]) != 3:
            raise ValueError(f"storage_pose must be [x,y,yaw]: {model_name}")
    for scene in scenes:
        for required in ("scene_family_id", "task_type", "instruction", "poses"):
            if required not in scene:
                raise ValueError(f"{scene['scene_id']} missing {required}")
        unknown = set(scene["poses"]) - set(objects)
        if unknown:
            raise ValueError(f"{scene['scene_id']} references unknown objects: {unknown}")
        if any(len(values) != 3 for values in scene["poses"].values()):
            raise ValueError(f"{scene['scene_id']} pose overrides must be [x,y,yaw]")


def scene_layout(config: dict, scene: dict) -> Dict[str, list]:
    """Return full [x,y,z,qx,qy,qz,qw] reset layout for one scene."""
    layout = {}
    overrides = scene.get("poses", {})
    for model_name, object_config in config["objects"].items():
        x_value, y_value, yaw = overrides.get(
            model_name, object_config["storage_pose"]
        )
        z_value = float(object_config["z"])
        yaw = float(yaw)
        layout[model_name] = [
            float(x_value),
            float(y_value),
            z_value,
            0.0,
            0.0,
            math.sin(0.5 * yaw),
            math.cos(0.5 * yaw),
        ]
    return layout


class PilotCapture(Node):
    """Reset preregistered scenes and save synchronized evaluator pairs."""

    def __init__(self) -> None:
        super().__init__("roborefer_pilot_capture")
        self.declare_parameter("scene_config_file", "")
        self.declare_parameter("output_root", "")
        self.declare_parameter("reset_service", "/world/ur3_pick_place/set_pose")
        self.declare_parameter("rgb_topic", "/wrist_camera/color/image_raw")
        self.declare_parameter("depth_topic", "/wrist_camera/depth/image_raw")
        self.declare_parameter(
            "semantic_label_topic", "/wrist_camera/evaluation_labels/labels_map"
        )
        self.declare_parameter("camera_info_topic", "/wrist_camera/color/camera_info")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("tcp_frame", "gripper_tcp")
        self.declare_parameter("settle_sec", 1.5)
        self.declare_parameter("sync_slop_sec", 0.15)
        self.declare_parameter("service_timeout_sec", 60.0)
        self.declare_parameter("capture_timeout_sec", 30.0)
        self.declare_parameter("view_joint_tolerance_rad", 0.025)
        self.declare_parameter("annotation_file", "")
        self.declare_parameter("gate_config_file", "")
        self.declare_parameter("pretrial_lock_file", "")

        config_path = Path(
            str(self.get_parameter("scene_config_file").value)
        ).expanduser().resolve()
        output_root = Path(
            str(self.get_parameter("output_root").value)
        ).expanduser().resolve()
        if not config_path.is_file():
            raise ValueError(f"scene_config_file not found: {config_path}")
        if not str(self.get_parameter("output_root").value).strip():
            raise ValueError("output_root must be non-empty")
        self._config_path = config_path
        self._config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        validate_scene_config(self._config)
        annotation_path = Path(
            str(self.get_parameter("annotation_file").value)
        ).expanduser().resolve()
        if not annotation_path.is_file():
            raise ValueError(f"annotation_file not found: {annotation_path}")
        self._annotation_path = annotation_path
        gate_config_path = Path(
            str(self.get_parameter("gate_config_file").value)
        ).expanduser().resolve()
        if not gate_config_path.is_file():
            raise ValueError(f"gate_config_file not found: {gate_config_path}")
        # The gate is not used during capture.  Its bytes are hashed here so
        # the later inference runner can prove that thresholds were frozen
        # before any pilot image or model prediction was inspected.
        self._gate_config_path = gate_config_path
        pretrial_lock_path = Path(
            str(self.get_parameter("pretrial_lock_file").value)
        ).expanduser().resolve()
        if not pretrial_lock_path.is_file():
            raise ValueError(f"pretrial_lock_file not found: {pretrial_lock_path}")
        pretrial_lock = json.loads(pretrial_lock_path.read_text(encoding="utf-8"))
        if (
            pretrial_lock.get("protocol_id") != self._config["protocol_id"]
            or pretrial_lock.get("status") != "LOCKED_BEFORE_CAPTURE_AND_INFERENCE"
        ):
            raise ValueError("pretrial source lock is not valid for this protocol")
        workspace_root = Path(str(pretrial_lock.get("workspace_root", ""))).resolve()
        artifact_hashes = pretrial_lock.get("source_artifact_sha256")
        if not workspace_root.is_dir() or not isinstance(artifact_hashes, dict):
            raise ValueError("pretrial source lock lacks workspace/artifact hashes")
        missing_required = REQUIRED_SOURCE_ARTIFACTS - set(artifact_hashes)
        if missing_required:
            raise ValueError(
                f"pretrial source lock misses required artifacts: {sorted(missing_required)}"
            )
        for relative_name, expected_hash in artifact_hashes.items():
            artifact = (workspace_root / str(relative_name)).resolve()
            if workspace_root not in artifact.parents or not artifact.is_file():
                raise ValueError(f"pretrial artifact missing/unsafe: {relative_name}")
            if sha256_file(artifact) != expected_hash:
                raise ValueError(f"pretrial artifact changed: {relative_name}")
        if artifact_hashes.get(str(config_path.relative_to(workspace_root))) != sha256_file(config_path):
            raise ValueError("scene config is not the preregistered source artifact")
        if artifact_hashes.get(str(annotation_path.relative_to(workspace_root))) != sha256_file(annotation_path):
            raise ValueError("annotation config is not the preregistered source artifact")
        if artifact_hashes.get(str(gate_config_path.relative_to(workspace_root))) != sha256_file(gate_config_path):
            raise ValueError("gate config is not the preregistered source artifact")
        self._pretrial_lock_path = pretrial_lock_path
        self._pretrial_lock = pretrial_lock
        output_root.mkdir(parents=True, exist_ok=False)
        self._output_root = output_root
        self._bridge = CvBridge()
        self._latest_rgb: Optional[Tuple[float, np.ndarray]] = None
        self._latest_depth: Optional[Tuple[float, np.ndarray]] = None
        self._latest_labels: Optional[Tuple[float, np.ndarray]] = None
        self._camera_info: Optional[CameraInfo] = None
        self._joint_state: Optional[JointState] = None
        self._tf_buffer = Buffer(cache_time=Duration(seconds=60.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._reset_client = self.create_client(
            SetEntityPose, str(self.get_parameter("reset_service").value)
        )

        # ros_gz_bridge camera streams use sensor-data (best-effort) QoS.
        # A default reliable subscription is incompatible with those writers
        # on some ROS 2 Humble/Fast-DDS configurations and can yield a black or
        # permanently empty capture despite Gazebo publishing valid frames.
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("rgb_topic").value),
            self._rgb_callback,
            sensor_qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._depth_callback,
            sensor_qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("semantic_label_topic").value),
            self._labels_callback,
            sensor_qos,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback,
            sensor_qos,
        )
        self.create_subscription(JointState, "/joint_states", self._joint_callback, 10)

    def _rgb_callback(self, message: Image) -> None:
        try:
            frame = np.asarray(
                self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            ).copy()
            self._latest_rgb = (stamp_seconds(message), frame)
        except Exception as exc:
            self.get_logger().warning(f"RGB decode failed: {exc}")

    def _depth_callback(self, message: Image) -> None:
        try:
            depth = np.asarray(
                self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            )
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth_m = (
                depth.astype(np.float32) * 0.001
                if message.encoding.upper() in ("16UC1", "MONO16")
                else depth.astype(np.float32)
            )
            self._latest_depth = (stamp_seconds(message), depth_m.copy())
        except Exception as exc:
            self.get_logger().warning(f"Depth decode failed: {exc}")

    def _labels_callback(self, message: Image) -> None:
        try:
            labels = np.asarray(
                self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            )
            if labels.ndim == 3:
                if labels.shape[2] != 3 or not (
                    np.array_equal(labels[..., 0], labels[..., 1])
                    and np.array_equal(labels[..., 1], labels[..., 2])
                ):
                    raise ValueError("semantic labels_map RGB channels disagree")
                labels = labels[..., 0]
            self._latest_labels = (
                stamp_seconds(message), labels.astype(np.uint8).copy()
            )
        except Exception as exc:
            self.get_logger().warning(f"Semantic-label decode failed: {exc}")

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self._camera_info = message

    def _joint_callback(self, message: JointState) -> None:
        self._joint_state = message

    def _view_pose_ready(self) -> bool:
        if self._joint_state is None:
            return False
        required_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        positions = dict(zip(self._joint_state.name, self._joint_state.position))
        if any(name not in positions for name in required_names):
            return False
        expected = [float(value) for value in self._config["view_joint_pose"]]
        tolerance = float(self.get_parameter("view_joint_tolerance_rad").value)
        return all(
            abs(float(positions[name]) - target) <= tolerance
            for name, target in zip(required_names, expected)
        )

    def _spin_until(self, predicate, timeout_sec: float) -> bool:
        deadline = time.monotonic() + float(timeout_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return bool(predicate())

    def _reset_entity(self, model_name: str, pose_values: list) -> None:
        request = SetEntityPose.Request()
        request.entity.name = model_name
        request.entity.type = Entity.MODEL
        (
            request.pose.position.x,
            request.pose.position.y,
            request.pose.position.z,
        ) = pose_values[:3]
        (
            request.pose.orientation.x,
            request.pose.orientation.y,
            request.pose.orientation.z,
            request.pose.orientation.w,
        ) = pose_values[3:7]
        future = self._reset_client.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=float(self.get_parameter("service_timeout_sec").value),
        )
        result = future.result() if future.done() else None
        if result is None or not result.success:
            raise RuntimeError(f"SetEntityPose failed for {model_name}")

    def _sensor_tuple(self):
        if (
            self._latest_rgb is None
            or self._latest_depth is None
            or self._latest_labels is None
            or self._camera_info is None
        ):
            return None
        stamps = [
            self._latest_rgb[0],
            self._latest_depth[0],
            self._latest_labels[0],
        ]
        spread = max(stamps) - min(stamps)
        allowed = float(self.get_parameter("sync_slop_sec").value)
        if spread > allowed:
            return None
        rgb = self._latest_rgb[1]
        depth = self._latest_depth[1]
        labels = self._latest_labels[1]
        if rgb.shape[:2] != depth.shape[:2] or rgb.shape[:2] != labels.shape[:2]:
            raise RuntimeError(
                f"sensor size mismatch RGB={rgb.shape[:2]}, depth={depth.shape[:2]}, "
                f"labels={labels.shape[:2]}"
            )
        return rgb.copy(), depth.copy(), labels.copy(), spread, stamps

    @staticmethod
    def _transform_dict(transform) -> dict:
        value = transform.transform
        return {
            "parent_frame": transform.header.frame_id,
            "child_frame": transform.child_frame_id,
            "position": [
                value.translation.x,
                value.translation.y,
                value.translation.z,
            ],
            "orientation_xyzw": [
                value.rotation.x,
                value.rotation.y,
                value.rotation.z,
                value.rotation.w,
            ],
        }

    def _tf_snapshot(self) -> dict:
        base = str(self.get_parameter("base_frame").value)
        snapshot = {}
        for child in (
            str(self.get_parameter("camera_frame").value),
            str(self.get_parameter("tcp_frame").value),
        ):
            try:
                transform = self._tf_buffer.lookup_transform(
                    base, child, Time(), timeout=Duration(seconds=0.5)
                )
                snapshot[child] = self._transform_dict(transform)
            except TransformException as exc:
                snapshot[child] = {"error": str(exc)}
        return snapshot

    def _save_scene(self, scene: dict, layout: dict, sensor_tuple) -> dict:
        rgb, depth_m, labels, spread, stamps = sensor_tuple
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise RuntimeError(f"invalid RGB shape/type: {rgb.shape}/{rgb.dtype}")
        if depth_m.ndim != 2 or depth_m.shape != rgb.shape[:2]:
            raise RuntimeError(
                f"registered depth shape mismatch: {depth_m.shape} vs {rgb.shape[:2]}"
            )
        if labels.ndim != 2 or labels.shape != rgb.shape[:2]:
            raise RuntimeError(
                f"semantic-label shape mismatch: {labels.shape} vs {rgb.shape[:2]}"
            )
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        rgb_std = float(np.std(gray))
        rgb_dynamic_range = float(
            np.percentile(gray, 99.0) - np.percentile(gray, 1.0)
        )
        rgb_nonzero_fraction = float(np.count_nonzero(gray)) / float(gray.size)
        if (
            rgb_std < MIN_RGB_STD
            or rgb_dynamic_range < MIN_RGB_DYNAMIC_RANGE
            or rgb_nonzero_fraction < MIN_RGB_NONZERO_FRACTION
        ):
            raise RuntimeError(
                "RGB_INPUT_QC_FAILED: "
                f"std={rgb_std:.3f}, dynamic_range={rgb_dynamic_range:.3f}, "
                f"nonzero_fraction={rgb_nonzero_fraction:.6f}"
            )
        valid_depth = (
            np.isfinite(depth_m)
            & (depth_m >= MIN_VALID_DEPTH_M)
            & (depth_m <= MAX_VALID_DEPTH_M)
        )
        valid_depth_fraction = float(np.count_nonzero(valid_depth)) / float(depth_m.size)
        valid_values = depth_m[valid_depth]
        depth_dynamic_range_m = (
            float(np.percentile(valid_values, 98.0) - np.percentile(valid_values, 2.0))
            if valid_values.size
            else 0.0
        )
        if (
            valid_depth_fraction < MIN_VALID_DEPTH_FRACTION
            or depth_dynamic_range_m < MIN_DEPTH_DYNAMIC_RANGE_M
        ):
            raise RuntimeError(
                "DEPTH_INPUT_QC_FAILED: "
                f"valid_fraction={valid_depth_fraction:.6f}, "
                f"dynamic_range_m={depth_dynamic_range_m:.6f}"
            )
        configured_slop = float(self.get_parameter("sync_slop_sec").value)
        if float(spread) > configured_slop + 1e-6:
            raise RuntimeError(
                f"SENSOR_SYNC_QC_FAILED: spread={spread:.9f} > {configured_slop:.9f}"
            )
        scene_dir = self._output_root / scene["scene_id"]
        input_dir = scene_dir / "input"
        evaluator_dir = scene_dir / "evaluator"
        input_dir.mkdir(parents=True, exist_ok=False)
        evaluator_dir.mkdir(parents=True, exist_ok=False)

        rgb_path = input_dir / "rgb.png"
        depth_path = input_dir / "depth_m.npy"
        labels_path = evaluator_dir / "semantic_labels.png"
        if not cv2.imwrite(str(rgb_path), rgb):
            raise RuntimeError(f"cannot write {rgb_path}")
        np.save(depth_path, depth_m, allow_pickle=False)
        if not cv2.imwrite(str(labels_path), labels):
            raise RuntimeError(f"cannot write {labels_path}")

        info = self._camera_info
        camera_info_path = input_dir / "camera_info.json"
        tf_path = input_dir / "tf_snapshot.json"
        write_json(camera_info_path, {
            "frame_id": info.header.frame_id,
            "width": int(info.width),
            "height": int(info.height),
            "distortion_model": info.distortion_model,
            "d": list(info.d),
            "k": list(info.k),
            "r": list(info.r),
            "p": list(info.p),
        })
        write_json(tf_path, self._tf_snapshot())
        write_json(evaluator_dir / "capture_oracle.json", {
            "oracle_usage": "evaluation_only_open_after_prediction_lock",
            "semantic_labels_file": "semantic_labels.png",
            "requested_scene_layout_base_link": layout,
            "visible_label_pixel_counts": {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(labels, return_counts=True))
            },
        })

        relative = lambda path: str(path.relative_to(self._output_root))
        record = {
            "schema_version": 1,
            "protocol_id": self._config["protocol_id"],
            "scene_id": scene["scene_id"],
            "instruction": str(scene["instruction"]),
            "coordinate_suffix": str(self._config["coordinate_suffix"]),
            "input_files": {
                "rgb": relative(rgb_path),
                "depth_m": relative(depth_path),
                "camera_info": relative(camera_info_path),
                "tf_snapshot": relative(tf_path),
            },
            "input_sha256": {
                "rgb": sha256_file(rgb_path),
                "depth_m": sha256_file(depth_path),
                "camera_info": sha256_file(camera_info_path),
                "tf_snapshot": sha256_file(tf_path),
            },
            "capture": {
                "rgb_depth_label_spread_sec": float(spread),
                "sensor_stamps_sec": [float(value) for value in stamps],
                "registered_metric_depth": True,
                "robot_manipulation_performed": False,
                "input_only_sensor_qc": {
                    "passed": True,
                    "rgb_gray_std": rgb_std,
                    "rgb_gray_p99_minus_p01": rgb_dynamic_range,
                    "rgb_nonzero_fraction": rgb_nonzero_fraction,
                    "valid_depth_fraction": valid_depth_fraction,
                    "depth_p98_minus_p02_m": depth_dynamic_range_m,
                },
            },
        }
        def forbidden_paths(value, prefix="$", findings=None):
            findings = [] if findings is None else findings
            if isinstance(value, dict):
                for raw_key, child in value.items():
                    child_path = f"{prefix}.{raw_key}"
                    if str(raw_key).strip().lower() in FORBIDDEN_INPUT_KEYS:
                        findings.append(child_path)
                    forbidden_paths(child, child_path, findings)
            elif isinstance(value, list):
                for item_index, child in enumerate(value):
                    forbidden_paths(child, f"{prefix}[{item_index}]", findings)
            return findings

        forbidden = forbidden_paths(record)
        if forbidden:
            raise RuntimeError(f"unsafe keys leaked into input record: {forbidden}")
        write_json(input_dir / "scene_input.json", record)
        return record

    def run(self) -> bool:
        if not self._reset_client.wait_for_service(
            timeout_sec=float(self.get_parameter("service_timeout_sec").value)
        ):
            self.get_logger().error("Gazebo SetEntityPose service unavailable")
            return False
        if not self._spin_until(
            lambda: self._camera_info is not None,
            float(self.get_parameter("capture_timeout_sec").value),
        ):
            self.get_logger().error("camera info unavailable")
            return False
        if not self._spin_until(
            self._view_pose_ready,
            float(self.get_parameter("capture_timeout_sec").value),
        ):
            self.get_logger().error("locked wrist-camera view joint pose was not reached")
            return False

        manifest = {
            "schema_version": 1,
            "protocol_id": self._config["protocol_id"],
            "created_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "capture_mode": "no_manipulation_shadow",
            "scene_count": len(self._config["scenes"]),
            "scene_config_sha256": sha256_file(self._config_path),
            # Hash-only preregistration: capture never parses this evaluator
            # file, so target labels cannot influence inputs or scene reset.
            "annotation_semantics_consumed_by_capture": False,
            "annotation_bytes_hashed_before_inference": True,
            "annotation_file_sha256": sha256_file(self._annotation_path),
            "gate_config_bytes_hashed_before_capture": True,
            "gate_config_file_sha256": sha256_file(self._gate_config_path),
            "pretrial_source_lock_file": str(self._pretrial_lock_path),
            "pretrial_source_lock_sha256": sha256_file(self._pretrial_lock_path),
            "source_artifacts_verified_before_capture": True,
            "model_inventory_sha256_preregistered": self._pretrial_lock.get(
                "model_inventory_sha256"
            ),
            "semantic_labels_are_evaluator_only": True,
            "target_handoff_published": False,
            "robot_manipulation_performed": False,
            "view_joint_pose_verified": True,
            "view_joint_pose": self._config["view_joint_pose"],
        }
        write_json(self._output_root / "capture_manifest.json", manifest)
        index_path = self._output_root / "input_manifest.jsonl"
        captured_rgb_hashes = []
        total_scenes = len(self._config["scenes"])

        for index, scene in enumerate(self._config["scenes"], start=1):
            layout = scene_layout(self._config, scene)
            self.get_logger().info(
                f"PILOT_CAPTURE_RESET {index}/{total_scenes}: {scene['scene_id']}"
            )
            for model_name, pose_values in layout.items():
                self._reset_entity(model_name, pose_values)

            # Clear all previous scene frames.  Only messages received after
            # this point can become evidence for the next immutable sample.
            self._latest_rgb = None
            self._latest_depth = None
            self._latest_labels = None
            settle_deadline = time.monotonic() + float(
                self.get_parameter("settle_sec").value
            )
            while rclpy.ok() and time.monotonic() < settle_deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
            captured_tuple = None

            def freeze_synchronized_tuple():
                nonlocal captured_tuple
                captured_tuple = self._sensor_tuple()
                return captured_tuple is not None

            if not self._spin_until(
                freeze_synchronized_tuple,
                float(self.get_parameter("capture_timeout_sec").value),
            ):
                self.get_logger().error(
                    f"synchronized RGB-D-label tuple unavailable: {scene['scene_id']}"
                )
                return False
            # Persist the exact tuple that satisfied synchronization.  Do not
            # fetch "latest" messages again between the check and disk write.
            record = self._save_scene(scene, layout, captured_tuple)
            captured_rgb_hashes.append(record["input_sha256"]["rgb"])
            with index_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.get_logger().info(
                f"PILOT_SCENE_CAPTURED {index}/{total_scenes}: {scene['scene_id']}"
            )

        if len(set(captured_rgb_hashes)) != len(captured_rgb_hashes):
            raise RuntimeError(
                "RGB_FROZEN_STREAM_QC_FAILED: at least two pilot scenes have "
                "byte-identical captured RGB images"
            )
        manifest["input_only_sensor_qc"] = {
            "passed": True,
            "all_scene_rgb_sha256_unique": True,
            "rgb_scene_count": len(captured_rgb_hashes),
            "thresholds": {
                "min_rgb_std": MIN_RGB_STD,
                "min_rgb_dynamic_range": MIN_RGB_DYNAMIC_RANGE,
                "min_rgb_nonzero_fraction": MIN_RGB_NONZERO_FRACTION,
                "min_valid_depth_fraction": MIN_VALID_DEPTH_FRACTION,
                "min_depth_dynamic_range_m": MIN_DEPTH_DYNAMIC_RANGE_M,
                "max_rgb_depth_label_spread_sec": float(
                    self.get_parameter("sync_slop_sec").value
                ),
            },
        }
        manifest["completed_wall_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest["input_manifest_sha256"] = sha256_file(index_path)
        evaluator_hashes = {}
        for scene in self._config["scenes"]:
            scene_id = scene["scene_id"]
            for relative_name in (
                "evaluator/semantic_labels.png",
                "evaluator/capture_oracle.json",
            ):
                artifact = self._output_root / scene_id / relative_name
                evaluator_hashes[f"{scene_id}/{relative_name}"] = sha256_file(
                    artifact
                )
        manifest["evaluator_artifact_sha256"] = evaluator_hashes
        manifest["status"] = "COMPLETE"
        write_json(self._output_root / "capture_manifest.json", manifest)
        self.get_logger().info(
            f"PILOT_CAPTURE_COMPLETE: {self._output_root} "
            f"({total_scenes} scenes, no manipulation)"
        )
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PilotCapture()
    try:
        success = node.run()
    except Exception as exc:
        node.get_logger().error(f"PILOT_CAPTURE_FATAL: {exc}")
        success = False
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
