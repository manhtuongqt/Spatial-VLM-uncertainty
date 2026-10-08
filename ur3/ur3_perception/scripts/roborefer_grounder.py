#!/usr/bin/env python3
"""Create a target mask and grasp point from RoboRefer points plus metric depth."""

import base64
import json
import math
import time
from typing import List, Optional, Sequence, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String

from depth_component_gate_v2 import (
    DepthComponentError,
    segment_seeded_depth_component_v2,
)
from spatial_point_utils import (
    canonical_target,
    manipulation_event_reaches_state,
    parse_points,
    points_to_pixels,
    representative_grasp_point,
)


COORDINATE_SUFFIX = (
    " Your answer should be formatted as a list of tuples, i.e. "
    "[(x1, y1), (x2, y2), ...], where each tuple contains the x and y "
    "coordinates of a point satisfying the conditions above. The coordinates "
    "should be between 0 and 1, indicating normalized pixel locations in the "
    "RGB image. Return coordinates only."
)


def stamp_seconds(message: Image) -> float:
    """Convert a ROS image timestamp to seconds."""
    return float(message.header.stamp.sec) + 1e-9 * float(
        message.header.stamp.nanosec
    )


def normalize_depth_for_roborefer(
    depth_m: np.ndarray,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
) -> np.ndarray:
    """Convert registered metric depth to RoboRefer's 8-bit relative-depth view.

    RoboRefer's reference API feeds it the grayscale output of monocular Depth
    Anything V2.  That representation is relative inverse depth rather than raw
    RealSense/Gazebo metres.  This conversion preserves the robot sensor's true
    geometry while matching that expected near-bright, far-dark appearance.
    """
    values = np.asarray(depth_m, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"depth must be HxW, got shape={values.shape}")
    valid = (
        np.isfinite(values)
        & (values >= float(min_depth_m))
        & (values <= float(max_depth_m))
    )
    if int(np.count_nonzero(valid)) < 16:
        raise ValueError("registered depth image contains too few valid pixels")

    valid_depth = values[valid]
    near = float(np.percentile(valid_depth, 2.0))
    far = float(np.percentile(valid_depth, 98.0))
    if far - near < 1e-4:
        near = float(valid_depth.min())
        far = float(valid_depth.max())
    if far - near < 1e-6:
        raise ValueError("registered depth image has no usable depth range")

    clipped = np.clip(values, near, far)
    inverse = 1.0 / np.maximum(clipped, 1e-6)
    inverse_near = 1.0 / near
    inverse_far = 1.0 / far
    normalized = (inverse - inverse_far) / max(inverse_near - inverse_far, 1e-6)
    gray = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    gray[~valid] = 0
    return np.repeat(gray[..., np.newaxis], 3, axis=-1)


def segment_seeded_depth_component(
    depth_m: np.ndarray,
    seed_points_xy: Sequence[Tuple[int, int]],
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
    seed_radius_px: int = 5,
    roi_radius_px: int = 110,
    near_tolerance_m: float = 0.015,
    far_tolerance_m: float = 0.055,
    min_area_px: int = 120,
    max_area_fraction: float = 0.12,
    bbox_padding_px: int = 5,
) -> dict:
    """Grow the depth surface containing RoboRefer's semantic seed points.

    The asymmetric interval is intentional: a point normally lands on the top
    of an object, so the visible sides are farther from the camera while almost
    no object surface should be substantially nearer.  Connected-component
    selection prevents a similarly deep but spatially separate object from
    being merged into the target.
    """
    values = np.asarray(depth_m, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"depth must be HxW, got shape={values.shape}")
    height, width = values.shape
    points = [
        (int(x), int(y))
        for x, y in seed_points_xy
        if 0 <= int(x) < width and 0 <= int(y) < height
    ]
    if not points:
        raise ValueError("no RoboRefer seed point lies inside the depth image")

    valid = (
        np.isfinite(values)
        & (values >= float(min_depth_m))
        & (values <= float(max_depth_m))
    )
    radius = max(1, int(seed_radius_px))
    local_depths = []
    for x_pixel, y_pixel in points:
        x0, x1 = max(0, x_pixel - radius), min(width, x_pixel + radius + 1)
        y0, y1 = max(0, y_pixel - radius), min(height, y_pixel + radius + 1)
        samples = values[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
        if samples.size:
            local_depths.append(float(np.median(samples)))
    if not local_depths:
        raise ValueError("no valid metric depth around any RoboRefer seed point")
    seed_depth_m = float(np.median(np.asarray(local_depths, dtype=np.float32)))

    center_x, center_y = representative_grasp_point(points)
    roi_radius = max(radius + 2, int(roi_radius_px))
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[
        max(0, center_y - roi_radius):min(height, center_y + roi_radius + 1),
        max(0, center_x - roi_radius):min(width, center_x + roi_radius + 1),
    ] = 255
    candidate = (
        valid
        & (values >= seed_depth_m - max(0.0, float(near_tolerance_m)))
        & (values <= seed_depth_m + max(0.0, float(far_tolerance_m)))
        & (roi > 0)
    ).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate, connectivity=8
    )
    if component_count <= 1:
        raise ValueError("depth threshold produced no connected target surface")

    max_area = max(1, int(round(width * height * float(max_area_fraction))))
    search_radius = max(2, radius * 2)
    best_label = None
    best_score = None
    for component in range(1, component_count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < int(min_area_px) or area > max_area:
            continue
        component_mask = (labels == component).astype(np.uint8)
        expanded = cv2.dilate(
            component_mask,
            np.ones((2 * search_radius + 1, 2 * search_radius + 1), np.uint8),
        )
        support = sum(bool(expanded[y, x]) for x, y in points)
        component_pixels_y, component_pixels_x = np.nonzero(component_mask)
        distance_sq = float(np.min(
            (component_pixels_x - center_x) ** 2
            + (component_pixels_y - center_y) ** 2
        ))
        # Seed support dominates, then proximity, then a stable larger surface.
        score = (int(support), -distance_sq, area)
        if best_score is None or score > best_score:
            best_score = score
            best_label = component
    if best_label is None or best_score[0] <= 0:
        raise ValueError("no valid depth component is connected to a RoboRefer seed")

    mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    y_pixels, x_pixels = np.nonzero(mask)
    if not x_pixels.size:
        raise ValueError("selected depth component became empty after cleanup")
    padding = max(0, int(bbox_padding_px))
    bbox = [
        max(0, int(x_pixels.min()) - padding),
        max(0, int(y_pixels.min()) - padding),
        min(width - 1, int(x_pixels.max()) + padding),
        min(height - 1, int(y_pixels.max()) + padding),
    ]

    supported_points = [(x, y) for x, y in points if mask[y, x] > 0]
    if supported_points:
        grasp_pixel = representative_grasp_point(supported_points)
    else:
        nearest_index = int(np.argmin(
            (x_pixels - center_x) ** 2 + (y_pixels - center_y) ** 2
        ))
        grasp_pixel = (
            int(x_pixels[nearest_index]), int(y_pixels[nearest_index])
        )
    selected_depth = values[mask > 0]
    selected_depth = selected_depth[np.isfinite(selected_depth)]
    return {
        "mask": mask,
        "bbox_xyxy": bbox,
        "grasp_pixel_xy": grasp_pixel,
        "supported_points_xy": supported_points,
        "seed_depth_m": seed_depth_m,
        "mask_area_px": int(np.count_nonzero(mask)),
        "mask_depth_min_m": float(np.min(selected_depth)),
        "mask_depth_max_m": float(np.max(selected_depth)),
    }


class RoboReferGrounder(Node):
    """One-shot RGB-D semantic point selection for the UR3 frozen view."""

    def __init__(self) -> None:
        super().__init__("roborefer_grounder")
        self.declare_parameter("image_topic", "/wrist_camera/color/image_raw")
        self.declare_parameter("depth_topic", "/wrist_camera/depth/image_raw")
        self.declare_parameter(
            "target_selection_topic", "/ur3_perception/target_selection"
        )
        self.declare_parameter("status_topic", "/ur3_perception/roborefer/status")
        self.declare_parameter("output_topic", "/ur3_perception/roborefer/image")
        self.declare_parameter("mask_topic", "/ur3_perception/roborefer/mask")
        self.declare_parameter(
            "instruction",
            "Point to the center of the red apple that the robot should pick up.",
        )
        self.declare_parameter("server_url", "http://127.0.0.1:25547")
        self.declare_parameter("worker_timeout_sec", 45.0)
        self.declare_parameter("max_rate_hz", 0.2)
        self.declare_parameter("max_rgb_depth_skew_sec", 0.15)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 2.0)
        self.declare_parameter("depth_seed_radius_px", 5)
        self.declare_parameter("depth_roi_radius_px", 110)
        self.declare_parameter("depth_near_tolerance_m", 0.015)
        self.declare_parameter("depth_far_tolerance_m", 0.055)
        self.declare_parameter("min_mask_area_px", 120)
        self.declare_parameter("max_mask_area_fraction", 0.12)
        self.declare_parameter("bbox_padding_px", 5)
        self.declare_parameter("reject_border_truncated", False)
        self.declare_parameter("target_color", "red")
        self.declare_parameter("target_object", "")
        self.declare_parameter("target_label_hint", "")
        self.declare_parameter("publish_target_selection", True)
        self.declare_parameter("grasp_region", "object")
        self.declare_parameter("wait_for_manipulation_state", False)
        self.declare_parameter("episode_event_topic", "/ur3_dataset/episode_event")
        self.declare_parameter("trigger_state", "FREEZE_PERCEPTION_POSE")

        self._bridge = CvBridge()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_depth_m: Optional[np.ndarray] = None
        self._latest_image_message: Optional[Image] = None
        self._latest_rgb_stamp = 0.0
        self._latest_depth_stamp = 0.0
        self._next_inference_time = 0.0
        self._completed = False
        self._inference_active = False
        self._last_log_time = 0.0
        self._manipulation_ready = not bool(
            self.get_parameter("wait_for_manipulation_state").value
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        retained_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._selection_pub = self.create_publisher(
            String,
            str(self.get_parameter("target_selection_topic").value),
            retained_qos,
        )
        self._status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self._image_pub = self.create_publisher(
            Image, str(self.get_parameter("output_topic").value), retained_qos
        )
        self._mask_pub = self.create_publisher(
            Image, str(self.get_parameter("mask_topic").value), retained_qos
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self._image_callback,
            sensor_qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._depth_callback,
            sensor_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("episode_event_topic").value),
            self._episode_event_callback,
            retained_qos,
        )
        self.create_timer(0.1, self._maybe_infer)
        self.get_logger().info(
            "RoboRefer-only RGB-D grounder ready; point-seeded depth segmentation; "
            f"manipulation_gate={'ready' if self._manipulation_ready else 'waiting'}"
        )

    def _episode_event_callback(self, message: String) -> None:
        if self._manipulation_ready:
            return
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        trigger_state = str(self.get_parameter("trigger_state").value)
        if not manipulation_event_reaches_state(payload, trigger_state):
            return
        self._manipulation_ready = True
        self._next_inference_time = 0.0
        self.get_logger().info(
            f"ROBOREFER_INFERENCE_GATE_OPEN: runner entered {trigger_state}"
        )

    def _image_callback(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Cannot decode wrist RGB frame: {exc}")
            return
        self._latest_frame = np.asarray(frame).copy()
        self._latest_image_message = message
        self._latest_rgb_stamp = stamp_seconds(message)

    def _depth_callback(self, message: Image) -> None:
        try:
            depth = np.asarray(
                self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            )
            if depth.ndim == 3:
                depth = depth[..., 0]
            if depth.dtype == np.uint16:
                depth_m = depth.astype(np.float32) * 0.001
            else:
                depth_m = depth.astype(np.float32)
        except Exception as exc:
            self.get_logger().warning(f"Cannot decode registered depth frame: {exc}")
            return
        self._latest_depth_m = depth_m.copy()
        self._latest_depth_stamp = stamp_seconds(message)

    def _publish_status(self, state: str, detail: str = "", **values) -> None:
        payload = {
            "status": state,
            "source": "roborefer_rgbd_depth_component",
            "instruction": str(self.get_parameter("instruction").value),
            "detail": detail,
        }
        payload.update(values)
        self._status_pub.publish(String(data=json.dumps(payload)))

    @staticmethod
    def _encode_image(image: np.ndarray, extension: str, params=None) -> str:
        success, encoded = cv2.imencode(extension, image, params or [])
        if not success:
            raise RuntimeError(f"Cannot encode image as {extension}")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _infer_points(
        self, frame: np.ndarray, depth_m: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], str, np.ndarray]:
        depth_view = normalize_depth_for_roborefer(
            depth_m,
            float(self.get_parameter("min_depth_m").value),
            float(self.get_parameter("max_depth_m").value),
        )
        instruction = str(self.get_parameter("instruction").value).strip()
        body = json.dumps({
            "image_url": [self._encode_image(
                frame, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90]
            )],
            "depth_url": [self._encode_image(depth_view, ".png")],
            "enable_depth": 1,
            "text": instruction + COORDINATE_SUFFIX,
        }).encode("utf-8")
        endpoint = str(self.get_parameter("server_url").value).rstrip("/") + "/query"
        request = urlrequest.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(
                request, timeout=float(self.get_parameter("worker_timeout_sec").value)
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"RoboRefer server HTTP {exc.code}: {detail}") from exc
        except (urlerror.URLError, TimeoutError) as exc:
            raise RuntimeError(f"RoboRefer server unavailable: {exc}") from exc
        answer = str(result.get("answer", "")).strip()
        points = parse_points(answer)
        if not points:
            raise RuntimeError(f"RoboRefer returned no coordinate tuples: {answer[:240]}")
        return points, answer, depth_view

    def _publish_overlay(
        self,
        points_xy: Sequence[Tuple[int, int]],
        segmentation: Optional[dict],
        depth_view: np.ndarray,
    ) -> None:
        if self._latest_frame is None or self._latest_image_message is None:
            return
        overlay = self._latest_frame.copy()
        grasp_pixel_xy = None
        if segmentation is not None:
            mask = segmentation["mask"]
            tint = np.zeros_like(overlay)
            tint[..., 1] = 220
            selected_pixels = mask > 0
            overlay[selected_pixels] = cv2.addWeighted(
                overlay[selected_pixels], 0.55, tint[selected_pixels], 0.45, 0.0
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
            x0, y0, x1, y1 = segmentation["bbox_xyxy"]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"DEPTH MASK {segmentation['mask_area_px']} px",
                (x0, max(18, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                2,
            )
            grasp_pixel_xy = segmentation["grasp_pixel_xy"]
        for x_pixel, y_pixel in points_xy:
            cv2.drawMarker(
                overlay,
                (x_pixel, y_pixel),
                (255, 170, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=3,
            )
        if grasp_pixel_xy is not None:
            cv2.drawMarker(
                overlay,
                grasp_pixel_xy,
                (0, 255, 255),
                markerType=cv2.MARKER_DIAMOND,
                markerSize=22,
                thickness=2,
            )

        # A small inset makes it explicit that the selected point was inferred
        # from the simulator's registered RGB-D pair, not RGB alone.
        inset_width = max(120, overlay.shape[1] // 4)
        inset_height = max(90, overlay.shape[0] // 4)
        depth_inset = cv2.resize(depth_view, (inset_width, inset_height))
        x_start = overlay.shape[1] - inset_width - 8
        y_start = overlay.shape[0] - inset_height - 8
        overlay[y_start:y_start + inset_height, x_start:x_start + inset_width] = depth_inset
        cv2.rectangle(
            overlay,
            (x_start, y_start),
            (x_start + inset_width, y_start + inset_height),
            (255, 255, 255),
            1,
        )
        cv2.putText(
            overlay,
            "REGISTERED DEPTH",
            (x_start + 5, y_start + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
        )
        label = "SELECTED" if segmentation is not None else "NO DEPTH COMPONENT"
        cv2.putText(
            overlay,
            f"RoboRefer point + registered depth: {label}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
        )
        message = self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        message.header = self._latest_image_message.header
        self._image_pub.publish(message)

        if segmentation is not None:
            mask_message = self._bridge.cv2_to_imgmsg(
                segmentation["mask"], encoding="mono8"
            )
            mask_message.header = self._latest_image_message.header
            self._mask_pub.publish(mask_message)

    def _configured_target(self) -> Tuple[str, str, str]:
        colour = str(self.get_parameter("target_color").value).strip()
        object_name = str(self.get_parameter("target_object").value).strip()
        label_hint = str(self.get_parameter("target_label_hint").value).strip()
        for candidate in (label_hint, f"{colour} {object_name}", object_name):
            target = canonical_target(candidate)
            if target is not None:
                return target
        raise ValueError(
            "target_object/target_label_hint does not map to a configured robot target"
        )

    def _maybe_infer(self) -> None:
        if (
            self._completed
            or self._inference_active
            or not self._manipulation_ready
            or self._latest_frame is None
            or self._latest_depth_m is None
        ):
            return
        now = time.monotonic()
        if now < self._next_inference_time:
            return
        interval = 1.0 / max(0.02, float(self.get_parameter("max_rate_hz").value))
        self._next_inference_time = now + interval
        self._inference_active = True

        inference_started = time.monotonic()
        try:
            skew = abs(self._latest_rgb_stamp - self._latest_depth_stamp)
            allowed_skew = float(
                self.get_parameter("max_rgb_depth_skew_sec").value
            )
            if skew > allowed_skew:
                raise RuntimeError(
                    f"latest RGB/depth frames are not synchronized: skew={skew:.3f}s"
                )
            frame = self._latest_frame.copy()
            depth_m = self._latest_depth_m.copy()
            if frame.shape[:2] != depth_m.shape[:2]:
                raise RuntimeError(
                    f"registered RGB/depth size mismatch: {frame.shape[:2]} vs "
                    f"{depth_m.shape[:2]}"
                )
            normalized_points, raw_answer, depth_view = self._infer_points(
                frame, depth_m
            )
            inference_latency_ms = 1000.0 * (time.monotonic() - inference_started)
            height, width = frame.shape[:2]
            pixel_points = points_to_pixels(normalized_points, width, height)
            if not pixel_points:
                raise RuntimeError("RoboRefer returned no valid normalized RGB point")
            segmentation = segment_seeded_depth_component_v2(
                depth_m,
                pixel_points,
                min_depth_m=float(self.get_parameter("min_depth_m").value),
                max_depth_m=float(self.get_parameter("max_depth_m").value),
                seed_radius_px=int(self.get_parameter("depth_seed_radius_px").value),
                roi_radius_px=int(self.get_parameter("depth_roi_radius_px").value),
                near_tolerance_m=float(
                    self.get_parameter("depth_near_tolerance_m").value
                ),
                far_tolerance_m=float(
                    self.get_parameter("depth_far_tolerance_m").value
                ),
                min_area_px=int(self.get_parameter("min_mask_area_px").value),
                max_area_fraction=float(
                    self.get_parameter("max_mask_area_fraction").value
                ),
                bbox_padding_px=int(self.get_parameter("bbox_padding_px").value),
                reject_border_truncated=bool(
                    self.get_parameter("reject_border_truncated").value
                ),
            )
            self._publish_overlay(pixel_points, segmentation, depth_view)

            support = len(segmentation["supported_points_xy"])
            consensus = support / max(1, len(pixel_points))
            # A valid seeded connected component is the geometric confidence
            # gate.  Consensus records how consistently all VLM points land on it.
            confidence = max(0.5, consensus)
            grasp_pixel = segmentation["grasp_pixel_xy"]
            payload = {
                "status": "SELECTED",
                "source": "roborefer_rgbd_depth_component",
                "bbox_xyxy": segmentation["bbox_xyxy"],
                "confidence": round(confidence, 4),
                "roborefer_consensus": round(consensus, 4),
                "roborefer_points_normalized": [
                    [float(x), float(y)] for x, y in normalized_points
                ],
                "roborefer_points_xy": [list(point) for point in pixel_points],
                "roborefer_supported_points_xy": [
                    list(point) for point in segmentation["supported_points_xy"]
                ],
                "grasp_pixel_xy": list(grasp_pixel),
                "grasp_region": str(self.get_parameter("grasp_region").value),
                "depth_source": "registered_wrist_camera",
                "segmentation_method": "roborefer_seeded_asymmetric_depth_component",
                "seed_depth_m": round(segmentation["seed_depth_m"], 6),
                "mask_area_px": segmentation["mask_area_px"],
                "mask_depth_range_m": [
                    round(segmentation["mask_depth_min_m"], 6),
                    round(segmentation["mask_depth_max_m"], 6),
                ],
                "reason_code": segmentation["reason_code"],
                "geometry_diagnostics": segmentation["diagnostics"],
                "rgb_depth_skew_sec": round(skew, 6),
                "inference_latency_ms": round(inference_latency_ms, 3),
                "instruction": str(self.get_parameter("instruction").value),
                "roborefer_raw": raw_answer,
            }
            publish_selection = bool(
                self.get_parameter("publish_target_selection").value
            )
            payload["selection_published"] = publish_selection
            if publish_selection:
                object_id, colour, oracle_name = self._configured_target()
                payload.update({
                    "object_id": object_id,
                    "color": colour,
                    "oracle_name": oracle_name,
                })
                self._selection_pub.publish(String(data=json.dumps(payload)))
                detail = "RGB-D target accepted"
                log_prefix = f"ROBOREFER_TARGET_SELECTED: {object_id}"
            else:
                # Blind evaluation deliberately withholds the configured
                # registry identity and Gazebo oracle from the model output.
                # A referee may inspect ground truth only after this point is
                # frozen, but it cannot influence inference or manipulation.
                detail = "blind RGB-D point locked; target identity withheld"
                log_prefix = "ROBOREFER_BLIND_POINT_LOCKED"
            self._publish_status("SELECTED", detail, **payload)
            self._completed = True
            self.get_logger().info(
                f"{log_prefix}, point={grasp_pixel}, support={support}/"
                f"{len(pixel_points)}, mask={segmentation['mask_area_px']}px, "
                f"latency={inference_latency_ms:.1f}ms"
            )
        except DepthComponentError as exc:
            inference_latency_ms = 1000.0 * (time.monotonic() - inference_started)
            if now - self._last_log_time >= 5.0:
                self.get_logger().warning(f"RoboRefer geometry rejected: {exc}")
                self._last_log_time = now
            self._publish_status(
                "GEOMETRY_REJECTED",
                exc.detail,
                reason_code=exc.code,
                geometry_diagnostics=exc.diagnostics,
                inference_latency_ms=round(inference_latency_ms, 3),
            )
        except Exception as exc:
            inference_latency_ms = 1000.0 * (time.monotonic() - inference_started)
            if now - self._last_log_time >= 5.0:
                self.get_logger().warning(f"RoboRefer grounding unavailable: {exc}")
                self._last_log_time = now
            self._publish_status(
                "INFERENCE_ERROR",
                str(exc),
                inference_latency_ms=round(inference_latency_ms, 3),
            )
        finally:
            self._inference_active = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoboReferGrounder()
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
