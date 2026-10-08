#!/usr/bin/env python3
"""RGB-D cube observation with temporal stability gating.

Gazebo ground truth is deliberately isolated in the evaluation publisher.  It
never contributes to the detected pose, dimensions, stability decision, or
confidence used by manipulation.
"""

from collections import deque
import json
import math
import time
from typing import Optional

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from image_geometry import PinholeCameraModel
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64, String
from tf2_geometry_msgs import do_transform_point
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException, TransformListener
from ur3_perception_interfaces.msg import ObjectObservation
from ur3_perception_interfaces.srv import ProjectPixel
from visualization_msgs.msg import Marker


def valid_depth_median(
    depth: np.ndarray,
    u: int,
    v: int,
    radius: int,
    min_depth: float,
    max_depth: float,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Return a local median after rejecting NaN, zero and invalid depths."""
    height, width = depth.shape[:2]
    x0, x1 = max(0, u - radius), min(width, u + radius + 1)
    y0, y1 = max(0, v - radius), min(height, v + radius + 1)
    values = depth[y0:y1, x0:x1].astype(np.float64, copy=False)
    valid = np.isfinite(values) & (values >= min_depth) & (values <= max_depth)
    if mask is not None:
        valid &= mask[y0:y1, x0:x1] > 0
    samples = values[valid]
    return float(np.median(samples)) if samples.size else math.nan


def project_pixel(u: int, v: int, depth_m: float, model: PinholeCameraModel):
    """Back-project an image pixel to the registered optical frame."""
    ray = model.projectPixelTo3dRay((float(u), float(v)))
    scale = depth_m / float(ray[2])
    return float(ray[0] * scale), float(ray[1] * scale), float(depth_m)


def aggregate_stable_samples(samples, max_stddev_m: float):
    """Return (accepted, mean xyz, mean dimensions, scalar position stddev)."""
    positions = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    dimensions = np.asarray([sample[1] for sample in samples], dtype=np.float64)
    mean_position = positions.mean(axis=0)
    mean_dimensions = dimensions.mean(axis=0)
    position_stddev = float(np.linalg.norm(positions.std(axis=0)))
    return (
        position_stddev <= max_stddev_m,
        mean_position,
        mean_dimensions,
        position_stddev,
    )


def stability_reason_code(
    sample_count: int,
    window_size: int,
    accepted: bool,
) -> Optional[str]:
    """Return the structured rejection reason once the temporal window is full."""
    if int(sample_count) < int(window_size) or accepted:
        return None
    return "UNSTABLE_3D"


class RgbdObjectPose(Node):
    def __init__(self) -> None:
        super().__init__("rgbd_object_pose")
        self.declare_parameter("color_topic", "/wrist_camera/color/image_raw")
        self.declare_parameter("depth_topic", "/wrist_camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/wrist_camera/color/camera_info")
        self.declare_parameter("camera_frame", "camera_depth_optical_frame")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("oracle_topic", "/ur3_perception/oracle_tf")
        self.declare_parameter("oracle_object_name", "red_cube")
        self.declare_parameter("default_object_id", "cube_red_01")
        self.declare_parameter("default_target_color", "red")
        self.declare_parameter("use_grounder_grasp_pixel", False)
        self.declare_parameter("grounder_grasp_proxy_dimensions", [0.04, 0.04, 0.024])
        # Optional generic selector for a future detector/grounder. The
        # built-in RGB-D baseline uses the configured default colour and does
        # not depend on any language or vision-language model.
        self.declare_parameter("target_selection_topic", "/ur3_perception/target_selection")
        self.declare_parameter("require_target_selection", False)
        self.declare_parameter("oracle_offset_z", 0.0)
        self.declare_parameter("sync_slop_sec", 0.08)
        self.declare_parameter("depth_window_radius", 3)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 2.0)
        self.declare_parameter("min_mask_area_px", 80.0)
        self.declare_parameter("object_height_m", 0.06)
        self.declare_parameter("min_dimension_m", 0.04)
        self.declare_parameter("max_dimension_m", 0.08)
        self.declare_parameter("stability_window_size", 7)
        self.declare_parameter("max_position_stddev_m", 0.003)
        self.declare_parameter("max_pose_age_sec", 1.0)
        self.declare_parameter("workspace_min", [0.15, 0.12, 0.0])
        self.declare_parameter("workspace_max", [0.50, 0.34, 0.12])
        self.declare_parameter("estimate_yaw", False)

        self._camera_frame = str(self.get_parameter("camera_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._oracle_name = str(self.get_parameter("oracle_object_name").value)
        self._target_object_id = str(self.get_parameter("default_object_id").value)
        self._target_color = str(self.get_parameter("default_target_color").value).lower()
        self._target_bbox = None
        self._use_grounder_grasp_pixel = bool(
            self.get_parameter("use_grounder_grasp_pixel").value
        )
        self._grasp_pixel = None
        self._require_target_selection = bool(
            self.get_parameter("require_target_selection").value
        )
        self._selection_received = False
        self._last_selection_wait_status = 0.0
        self._oracle_offset = float(self.get_parameter("oracle_offset_z").value)
        self._radius = int(self.get_parameter("depth_window_radius").value)
        self._min_depth = float(self.get_parameter("min_depth_m").value)
        self._max_depth = float(self.get_parameter("max_depth_m").value)
        self._min_area = float(self.get_parameter("min_mask_area_px").value)
        self._object_height = float(self.get_parameter("object_height_m").value)
        self._min_dimension = float(self.get_parameter("min_dimension_m").value)
        self._max_dimension = float(self.get_parameter("max_dimension_m").value)
        self._window_size = int(self.get_parameter("stability_window_size").value)
        self._max_stddev = float(self.get_parameter("max_position_stddev_m").value)
        self._max_pose_age = float(self.get_parameter("max_pose_age_sec").value)
        self._workspace_min = np.asarray(
            self.get_parameter("workspace_min").value, dtype=np.float64
        )
        self._workspace_max = np.asarray(
            self.get_parameter("workspace_max").value, dtype=np.float64
        )
        self._estimate_yaw = bool(self.get_parameter("estimate_yaw").value)
        if not 5 <= self._window_size <= 10:
            raise ValueError("stability_window_size must be between 5 and 10")

        self._bridge = CvBridge()
        self._model = PinholeCameraModel()
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._latest_depth = None
        self._latest_info = None
        self._latest_stamp = None
        self._oracle_point = None
        self._last_valid_wall_time = None
        self._last_log = 0.0
        self._samples = deque(maxlen=self._window_size)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        color_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter("color_topic").value),
            qos_profile=sensor_qos,
        )
        depth_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter("depth_topic").value),
            qos_profile=sensor_qos,
        )
        info_sub = message_filters.Subscriber(
            self, CameraInfo, str(self.get_parameter("camera_info_topic").value),
            qos_profile=sensor_qos,
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=10,
            slop=float(self.get_parameter("sync_slop_sec").value),
        )
        self._sync.registerCallback(self._rgbd_callback)

        self.create_subscription(
            TFMessage, str(self.get_parameter("oracle_topic").value),
            self._oracle_callback, sensor_qos,
        )
        selection_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, str(self.get_parameter("target_selection_topic").value),
            self._target_selection_callback, selection_qos,
        )
        self._pose_pub = self.create_publisher(
            PoseStamped, "/ur3_perception/object_pose", 10
        )
        self._stable_pose_pub = self.create_publisher(
            PoseStamped, "/ur3_perception/object_pose_stable", 10
        )
        self._observation_pub = self.create_publisher(
            ObjectObservation, "/ur3_perception/object_observation", 10
        )
        self._camera_point_pub = self.create_publisher(
            PointStamped, "/ur3_perception/object_point_camera", 10
        )
        self._oracle_pub = self.create_publisher(
            PoseStamped, "/ur3_perception/oracle_pose", 10
        )
        self._error_pub = self.create_publisher(
            Float64, "/ur3_perception/position_error_m", 10
        )
        self._debug_pub = self.create_publisher(
            Image, "/ur3_perception/debug_image", sensor_qos
        )
        self._status_pub = self.create_publisher(String, "/ur3_perception/status", 10)
        self._marker_pub = self.create_publisher(
            Marker, "/ur3_perception/object_marker", 10
        )
        self.create_service(
            ProjectPixel, "/ur3_perception/project_pixel", self._project_service
        )
        self.create_timer(0.5, self._stale_timer)
        self.get_logger().info(
            f"RGB-D stability gate ready: {self._window_size} frames, "
            f"max stddev={self._max_stddev * 1000.0:.1f} mm"
        )

    def _oracle_callback(self, msg: TFMessage) -> None:
        """Evaluation-only callback; values are never read by control logic."""
        for transform in msg.transforms:
            if transform.child_frame_id != self._oracle_name:
                continue
            point = PointStamped()
            point.header.stamp = self.get_clock().now().to_msg()
            point.header.frame_id = self._base_frame
            point.point.x = transform.transform.translation.x
            point.point.y = transform.transform.translation.y
            point.point.z = transform.transform.translation.z + self._oracle_offset
            self._oracle_point = point
            pose = PoseStamped()
            pose.header = point.header
            pose.pose.position = point.point
            pose.pose.orientation = transform.transform.rotation
            self._oracle_pub.publish(pose)
            return

    def _target_selection_callback(self, msg: String) -> None:
        """Accept a registry-backed target; geometry remains RGB-D derived."""
        try:
            selection = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Ignoring malformed target selection JSON")
            return
        allowed = {
            "cube_red_01": ("red", "red_cube", 0.06),
            "cube_green_01": ("green", "green_cube", 0.05),
            "cube_blue_01": ("blue", "blue_cube", 0.05),
            "cube_yellow_01": ("yellow", "yellow_cube", 0.05),
            "cube_orange_01": ("orange", "orange_cube", 0.05),
            "cube_purple_01": ("purple", "purple_cube", 0.05),
            "cube_pink_01": ("pink", "pink_cube", 0.05),
            "ycb_cracker_box_01": ("red", "ycb_cracker_box", 0.213421),
            "ycb_sugar_box_01": ("yellow", "ycb_sugar_box", 0.176214),
            "ycb_tomato_soup_can_01": ("red", "ycb_tomato_soup_can", 0.101895),
            "ycb_mustard_bottle_01": ("yellow", "ycb_mustard_bottle", 0.191390),
            "ycb_banana_01": ("yellow", "ycb_banana", 0.036752),
            "ycb_apple_01": ("red", "ycb_apple", 0.071907),
            "ycb_orange_01": ("orange", "ycb_orange", 0.071368),
            "ycb_power_drill_01": ("blue", "ycb_power_drill", 0.057377),
            "mango_01": ("orange", "mango", 0.060),
        }
        object_id = str(selection.get("object_id", ""))
        color = str(selection.get("color", "")).lower()
        if selection.get("status") != "SELECTED" or object_id not in allowed:
            self.get_logger().warning(f"Ignoring ungrounded target selection: {object_id}")
            return
        expected_color, oracle_frame, object_height = allowed[object_id]
        if color != expected_color:
            self.get_logger().warning(
                f"Ignoring inconsistent target selection: {object_id} cannot have color={color}"
            )
            return
        bbox = selection.get("bbox_xyxy")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            self.get_logger().warning(
                f"Ignoring target selection without a valid bbox: {object_id}"
            )
            return
        changed = object_id != self._target_object_id
        self._target_object_id = object_id
        self._target_color = color
        self._oracle_name = oracle_frame
        self._object_height = object_height
        self._target_bbox = tuple(int(round(value)) for value in bbox)
        self._grasp_pixel = None
        raw_grasp_pixel = selection.get("grasp_pixel_xy")
        if self._use_grounder_grasp_pixel:
            if (
                not isinstance(raw_grasp_pixel, list)
                or len(raw_grasp_pixel) != 2
                or not all(isinstance(value, (int, float)) for value in raw_grasp_pixel)
            ):
                self.get_logger().warning(
                    "Ignoring selection without a semantic grasp pixel"
                )
                return
            grasp_pixel = tuple(int(round(value)) for value in raw_grasp_pixel)
            x0, y0, x1, y1 = self._target_bbox
            if not (x0 <= grasp_pixel[0] <= x1 and y0 <= grasp_pixel[1] <= y1):
                self.get_logger().warning(
                    "Ignoring semantic grasp pixel outside its target bbox"
                )
                return
            self._grasp_pixel = grasp_pixel
        self._selection_received = True
        # Always discard samples taken before semantic selection/frozen view.
        self._samples.clear()
        self._last_valid_wall_time = None
        self.get_logger().info(
            f"PERCEPTION_TARGET_SELECTED: {object_id}, color={color}, "
            f"bbox={self._target_bbox}, "
            f"grasp_pixel={self._grasp_pixel}, "
            f"stability gate reset{' (target changed)' if changed else ''}"
        )

    @staticmethod
    def _color_mask(bgr: np.ndarray, color: str) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ranges = {
            "red": [((0, 90, 50), (6, 255, 255)), ((176, 90, 50), (180, 255, 255))],
            "orange": [((7, 80, 45), (20, 255, 255))],
            "yellow": [((21, 70, 45), (34, 255, 255))],
            "green": [((35, 70, 35), (89, 255, 255))],
            "blue": [((90, 70, 35), (134, 255, 255))],
            "purple": [((135, 70, 35), (155, 255, 255))],
            "pink": [((156, 70, 45), (175, 255, 255))],
            "black": [((0, 0, 0), (180, 255, 80))],
        }
        if color not in ranges:
            return np.zeros(bgr.shape[:2], dtype=np.uint8)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges[color]:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(hsv, np.asarray(lower), np.asarray(upper)),
            )
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def _lookup_transform(self, stamp):
        try:
            return self._tf_buffer.lookup_transform(
                self._base_frame, self._camera_frame, Time.from_msg(stamp),
                timeout=Duration(seconds=0.25),
            )
        except TransformException:
            return self._tf_buffer.lookup_transform(
                self._base_frame, self._camera_frame, Time(),
                timeout=Duration(seconds=0.25),
            )

    def _camera_point(self, u: int, v: int, depth_m: float, stamp) -> PointStamped:
        x, y, z = project_pixel(u, v, depth_m, self._model)
        point = PointStamped()
        point.header.stamp = stamp
        point.header.frame_id = self._camera_frame
        point.point.x, point.point.y, point.point.z = x, y, z
        return point

    def _publish_failure(self, status: str, detail: str) -> None:
        self._status_pub.publish(String(data=json.dumps({
            "status": status,
            "detail": detail,
            "object_id": self._target_object_id,
            "color": self._target_color,
            "valid_frames": len(self._samples),
            "oracle_error_m": None,
        })))

    def _stale_timer(self) -> None:
        if self._last_valid_wall_time is None:
            return
        age = time.monotonic() - self._last_valid_wall_time
        if age <= self._max_pose_age:
            return
        if self._samples:
            self._samples.clear()
            self._publish_failure("STALE", f"no valid RGB-D candidate for {age:.2f}s")

    def _estimate_dimensions(self, contour, depth_m: float, info: CameraInfo):
        rectangle = cv2.minAreaRect(contour)
        width_px, height_px = rectangle[1]
        width_m = width_px * depth_m / float(info.k[0])
        height_m = height_px * depth_m / float(info.k[4])
        planar = np.clip(
            [width_m, height_m], self._min_dimension, self._max_dimension
        )
        dimensions = np.array([planar[0], planar[1], self._object_height])
        angle_deg = float(rectangle[2])
        yaw = math.radians(angle_deg) if self._estimate_yaw else 0.0
        return dimensions, yaw

    def _rgbd_callback(self, color_msg: Image, depth_msg: Image, info: CameraInfo):
        try:
            bgr = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        except Exception as exc:
            self._publish_failure("INVALID_IMAGE", str(exc))
            return

        self._model.fromCameraInfo(info)
        self._latest_depth = np.asarray(depth)
        self._latest_info = info
        self._latest_stamp = depth_msg.header.stamp
        if self._require_target_selection and not self._selection_received:
            now = time.monotonic()
            if now - self._last_selection_wait_status >= 1.0:
                self._publish_failure(
                    "WAITING_FOR_TARGET_SELECTION",
                    "no grounded semantic target has been accepted",
                )
                self._last_selection_wait_status = now
            debug = bgr.copy()
            cv2.putText(
                debug,
                "Waiting for semantic target selection",
                (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
            )
            self._publish_debug(debug, color_msg)
            return
        use_grasp_pixel = (
            self._use_grounder_grasp_pixel and self._grasp_pixel is not None
        )
        mask = self._color_mask(bgr, self._target_color)
        # Color alone is ambiguous for yellow/orange because the tray/table can
        # share that hue. Geometry is therefore restricted to an optional
        # externally supplied candidate ROI; depth and TF still determine
        # every metric coordinate.
        if self._target_bbox is not None:
            height, width = mask.shape[:2]
            x0, y0, x1, y1 = self._target_bbox
            x0, x1 = max(0, x0), min(width, x1)
            y0, y1 = max(0, y0), min(height, y1)
            roi = np.zeros_like(mask)
            if x1 > x0 and y1 > y0:
                roi[y0:y1, x0:x1] = 255
            mask = cv2.bitwise_and(mask, roi)
        debug = bgr.copy()
        if self._target_bbox is not None:
            cv2.rectangle(
                debug,
                (max(0, self._target_bbox[0]), max(0, self._target_bbox[1])),
                (min(debug.shape[1] - 1, self._target_bbox[2]),
                 min(debug.shape[0] - 1, self._target_bbox[3])),
                (255, 255, 255),
                2,
            )
        contour = None
        if use_grasp_pixel:
            u, v = self._grasp_pixel
            height, width = depth.shape[:2]
            if not (0 <= u < width and 0 <= v < height):
                self._publish_failure("NO_DETECTION", "semantic grasp pixel is outside the image")
                self._publish_debug(debug, color_msg)
                return
            # A direct depth sample makes the controller follow the grounder's
            # semantic grasp location instead of a fragile colour centroid.
            area = self._min_area * 5.0
            eroded = None
        else:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                self._publish_failure(
                    "NO_DETECTION", f"{self._target_color} object mask is empty"
                )
                self._publish_debug(debug, color_msg)
                return
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area < self._min_area:
                self._publish_failure("NO_DETECTION", f"mask area {area:.1f}px is too small")
                self._publish_debug(debug, color_msg)
                return
            moments = cv2.moments(contour)
            if moments["m00"] <= 0.0:
                self._publish_failure("NO_DETECTION", "invalid contour moments")
                return
            u = int(round(moments["m10"] / moments["m00"]))
            v = int(round(moments["m01"] / moments["m00"]))
            eroded = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
        depth_m = valid_depth_median(
            depth, u, v, self._radius, self._min_depth, self._max_depth, eroded
        )
        if not math.isfinite(depth_m):
            self._samples.clear()
            self._publish_failure("INVALID_DEPTH", "no valid registered depth")
            cv2.drawContours(debug, [contour], -1, (0, 165, 255), 2)
            self._publish_debug(debug, color_msg)
            return

        camera_point = self._camera_point(u, v, depth_m, depth_msg.header.stamp)
        try:
            base_surface = do_transform_point(
                camera_point, self._lookup_transform(depth_msg.header.stamp)
            )
        except TransformException as exc:
            self._publish_failure("TF_UNAVAILABLE", str(exc))
            return
        base_surface.header.frame_id = self._base_frame
        base_surface.header.stamp = camera_point.header.stamp
        self._camera_point_pub.publish(camera_point)

        raw_pose = PoseStamped()
        raw_pose.header = base_surface.header
        raw_pose.pose.position = base_surface.point
        raw_pose.pose.orientation.w = 1.0
        self._pose_pub.publish(raw_pose)

        if use_grasp_pixel:
            dimensions = np.asarray(
                self.get_parameter("grounder_grasp_proxy_dimensions").value,
                dtype=np.float64,
            )
            yaw = 0.0
            # The sampled point is intentionally the semantic grasp point, so
            # it enters manipulation pose generation unchanged.
            center = np.array([
                base_surface.point.x,
                base_surface.point.y,
                base_surface.point.z,
            ])
        else:
            dimensions, yaw = self._estimate_dimensions(contour, depth_m, info)
            center = np.array([
                base_surface.point.x,
                base_surface.point.y,
                base_surface.point.z - 0.5 * dimensions[2],
            ])
        if np.any(center < self._workspace_min) or np.any(center > self._workspace_max):
            self._samples.clear()
            self._publish_failure(
                "OUTSIDE_WORKSPACE",
                f"center={center.round(4).tolist()}",
            )
            self._publish_debug(debug, color_msg)
            return

        self._last_valid_wall_time = time.monotonic()
        self._samples.append((center, dimensions, yaw, depth_msg.header.stamp))
        accepted = False
        stddev = math.nan
        stable_pose = None
        mean_dimensions = dimensions
        if len(self._samples) >= self._window_size:
            accepted, mean_position, mean_dimensions, stddev = aggregate_stable_samples(
                self._samples, self._max_stddev
            )
            if accepted:
                stable_pose = PoseStamped()
                stable_pose.header.frame_id = self._base_frame
                stable_pose.header.stamp = depth_msg.header.stamp
                stable_pose.pose.position.x = float(mean_position[0])
                stable_pose.pose.position.y = float(mean_position[1])
                stable_pose.pose.position.z = float(mean_position[2])
                mean_yaw = float(np.mean([sample[2] for sample in self._samples]))
                stable_pose.pose.orientation.z = math.sin(0.5 * mean_yaw)
                stable_pose.pose.orientation.w = math.cos(0.5 * mean_yaw)
                self._stable_pose_pub.publish(stable_pose)

        error_m = math.nan
        if accepted and stable_pose is not None and self._oracle_point is not None:
            dx = stable_pose.pose.position.x - self._oracle_point.point.x
            dy = stable_pose.pose.position.y - self._oracle_point.point.y
            dz = stable_pose.pose.position.z - self._oracle_point.point.z
            error_m = math.sqrt(dx * dx + dy * dy + dz * dz)
            self._error_pub.publish(Float64(data=error_m))

        observation = ObjectObservation()
        observation.object_id = self._target_object_id
        observation.color = self._target_color
        observation.pose = stable_pose if stable_pose is not None else raw_pose
        observation.dimensions.x = float(mean_dimensions[0])
        observation.dimensions.y = float(mean_dimensions[1])
        observation.dimensions.z = float(mean_dimensions[2])
        observation.yaw = float(np.mean([sample[2] for sample in self._samples]))
        observation.yaw_valid = self._estimate_yaw
        observation.valid_frames = len(self._samples)
        observation.position_stddev_m = float(stddev) if math.isfinite(stddev) else -1.0
        reason_code = stability_reason_code(
            len(self._samples), self._window_size, accepted
        )
        if len(self._samples) < self._window_size:
            observation.status = "ACCUMULATING"
            observation.confidence = len(self._samples) / self._window_size
        elif not accepted:
            observation.status = "UNSTABLE"
            observation.confidence = 0.0
        else:
            observation.status = "STABLE"
            area_confidence = min(1.0, area / max(self._min_area * 5.0, 1.0))
            stability_confidence = max(0.0, 1.0 - stddev / self._max_stddev)
            observation.confidence = 0.5 * area_confidence + 0.5 * stability_confidence
        self._observation_pub.publish(observation)

        status = {
            "status": observation.status,
            "object_id": self._target_object_id,
            "color": self._target_color,
            "pose_source": "registered_rgbd_depth_tf2",
            "synchronized": True,
            "pixel": [u, v],
            "depth_m": round(depth_m, 6),
            "camera_frame": self._camera_frame,
            "base_point_m": [round(float(value), 6) for value in center],
            "dimensions_m": [round(float(value), 6) for value in mean_dimensions],
            "yaw": observation.yaw if observation.yaw_valid else None,
            "confidence": round(observation.confidence, 6),
            "valid_frames": observation.valid_frames,
            "position_stddev_m": observation.position_stddev_m,
            "reason_code": reason_code,
            "oracle_error_m": round(error_m, 6) if math.isfinite(error_m) else None,
        }
        self._status_pub.publish(String(data=json.dumps(status)))
        if stable_pose is not None:
            self._publish_marker(stable_pose, mean_dimensions)

        now = time.monotonic()
        if now - self._last_log > 2.0:
            error_text = f", eval error={error_m * 1000.0:.1f}mm" if math.isfinite(error_m) else ""
            self.get_logger().info(
                f"{observation.status}: frames={len(self._samples)}, "
                f"center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})"
                f"{error_text}"
            )
            self._last_log = now
        if contour is not None:
            cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)
        if use_grasp_pixel:
            cv2.drawMarker(debug, (u, v), (0, 255, 255), cv2.MARKER_DIAMOND, 18, 2)
        else:
            cv2.circle(debug, (u, v), 5, (255, 255, 255), -1)
        label = (
            f"Semantic point:{observation.status} n={len(self._samples)}"
            if use_grasp_pixel
            else f"{self._target_color}:{observation.status} n={len(self._samples)}"
        )
        cv2.putText(debug, label, (max(5, u - 80), max(20, v - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        self._publish_debug(debug, color_msg)

    def _publish_debug(self, image: np.ndarray, source: Image) -> None:
        msg = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
        msg.header = source.header
        self._debug_pub.publish(msg)

    def _publish_marker(self, pose: PoseStamped, dimensions) -> None:
        marker = Marker()
        marker.header = pose.header
        marker.ns = "ur3_perception"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = float(dimensions[0])
        marker.scale.y = float(dimensions[1])
        marker.scale.z = float(dimensions[2])
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.1, 1.0, 0.2, 0.55
        self._marker_pub.publish(marker)

    def _project_service(self, request, response):
        if self._latest_depth is None or self._latest_info is None:
            response.message = "No synchronized RGB-D frame received yet"
            return response
        height, width = self._latest_depth.shape[:2]
        if request.u >= width or request.v >= height:
            response.message = f"Pixel outside {width}x{height} image"
            return response
        radius = int(request.window_radius) if request.window_radius else self._radius
        depth_m = valid_depth_median(
            self._latest_depth, int(request.u), int(request.v), radius,
            self._min_depth, self._max_depth,
        )
        if not math.isfinite(depth_m):
            response.message = "No valid depth in requested window"
            return response
        self._model.fromCameraInfo(self._latest_info)
        camera_point = self._camera_point(
            int(request.u), int(request.v), depth_m, self._latest_stamp
        )
        try:
            base_point = do_transform_point(
                camera_point, self._lookup_transform(self._latest_stamp)
            )
        except TransformException as exc:
            response.message = f"TF lookup failed: {exc}"
            return response
        base_point.header.frame_id = self._base_frame
        base_point.header.stamp = camera_point.header.stamp
        response.success = True
        response.message = "ok"
        response.depth_m = depth_m
        response.point_camera = camera_point
        response.point_base = base_point
        return response


def main(args=None):
    rclpy.init(args=args)
    node = RgbdObjectPose()
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
