#!/usr/bin/env python3
"""RoboRefer seeds depth regions for metric 3D spatial demonstrations."""

import base64
import json
import time
from typing import Dict, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from spatial_point_utils import parse_points, points_to_pixels
from roborefer_grounder import (
    COORDINATE_SUFFIX,
    normalize_depth_for_roborefer,
    stamp_seconds,
)


OBJECT_COLOURS_BGR = {
    "A": (255, 190, 40),
    "B": (220, 70, 255),
    "C": (70, 230, 90),
}


def quaternion_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Return the 3x3 rotation matrix for a normalized XYZW quaternion."""
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("transform quaternion has zero length")
    x, y, z, w = quaternion / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def backproject_mask_to_camera(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: Tuple[float, float, float, float],
) -> np.ndarray:
    """Back-project all valid mask pixels into the registered optical frame."""
    depth = np.asarray(depth_m, dtype=np.float64)
    selected = np.asarray(mask) > 0
    if depth.ndim != 2 or selected.shape != depth.shape:
        raise ValueError("depth and mask must be equally sized HxW arrays")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    selected &= np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(selected)
    if columns.size < 32:
        raise ValueError("depth mask contains too few valid 3D samples")
    z_camera = depth[rows, columns]
    x_camera = (columns.astype(np.float64) - cx) * z_camera / fx
    y_camera = (rows.astype(np.float64) - cy) * z_camera / fy
    return np.column_stack((x_camera, y_camera, z_camera))


def transform_points(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Transform Nx3 row-vector points to the target coordinate frame."""
    values = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(rotation, dtype=np.float64)
    offset = np.asarray(translation, dtype=np.float64).reshape(1, 3)
    if values.ndim != 2 or values.shape[1] != 3 or matrix.shape != (3, 3):
        raise ValueError("invalid point cloud or rigid transform shape")
    return values @ matrix.T + offset


def depth_to_base_image(
    depth_m: np.ndarray,
    intrinsics: Tuple[float, float, float, float],
    rotation: np.ndarray,
    translation: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Vectorize a registered depth image as an HxWx3 base-frame point map."""
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must be an HxW array")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    rows, columns = np.indices(depth.shape, dtype=np.float64)
    camera = np.stack((
        (columns - cx) * depth / fx,
        (rows - cy) * depth / fy,
        depth,
    ), axis=-1)
    valid = (
        np.isfinite(depth)
        & (depth >= float(min_depth_m))
        & (depth <= float(max_depth_m))
    )
    base = camera @ np.asarray(rotation, dtype=np.float64).T
    base += np.asarray(translation, dtype=np.float64).reshape(1, 1, 3)
    base[~valid] = np.nan
    return base


def estimate_support_plane_z(
    points_base_image: np.ndarray,
    search_min_z_m: float = -0.03,
    search_max_z_m: float = 0.04,
    histogram_bin_m: float = 0.002,
) -> float:
    """Estimate the dominant horizontal support plane from RGB-D, not an oracle."""
    points = np.asarray(points_base_image, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("base point map must have shape HxWx3")
    z_values = points[..., 2]
    samples = z_values[
        np.isfinite(z_values)
        & (z_values >= float(search_min_z_m))
        & (z_values <= float(search_max_z_m))
    ]
    if samples.size < 500:
        raise ValueError("too few RGB-D samples to estimate the support plane")
    bin_width = max(0.0005, float(histogram_bin_m))
    edges = np.arange(
        float(search_min_z_m), float(search_max_z_m) + bin_width, bin_width
    )
    histogram, edges = np.histogram(samples, bins=edges)
    peak = int(np.argmax(histogram))
    lower = edges[max(0, peak - 1)]
    upper = edges[min(len(edges) - 1, peak + 2)]
    inlier = samples[(samples >= lower) & (samples <= upper)]
    if inlier.size < 100:
        raise ValueError("dominant support-plane depth cluster is too small")
    return float(np.median(inlier))


def segment_seeded_above_plane_component(
    points_base_image: np.ndarray,
    seed_xy: Tuple[int, int],
    support_plane_z_m: float,
    roi_radius_px: int = 120,
    plane_clearance_m: float = 0.004,
    max_object_height_m: float = 0.30,
    min_area_px: int = 120,
    max_area_fraction: float = 0.12,
    bbox_padding_px: int = 5,
) -> Dict[str, object]:
    """Select the connected above-table region containing a RoboRefer seed.

    Unlike a fixed seed-depth band, this keeps the complete vertical silhouette
    of tall objects. The table plane is estimated from depth and removed before
    connected-component selection.
    """
    points = np.asarray(points_base_image, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("base point map must have shape HxWx3")
    height, width = points.shape[:2]
    seed_x, seed_y = (int(seed_xy[0]), int(seed_xy[1]))
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ValueError("RoboRefer seed lies outside the base point map")
    seed_point = points[seed_y, seed_x]
    if not np.all(np.isfinite(seed_point)):
        raise ValueError("RoboRefer seed has no valid registered 3D point")

    radius = max(8, int(roi_radius_px))
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[
        max(0, seed_y - radius):min(height, seed_y + radius + 1),
        max(0, seed_x - radius):min(width, seed_x + radius + 1),
    ] = 255
    z_values = points[..., 2]
    above_plane = (
        np.isfinite(z_values)
        & (z_values >= float(support_plane_z_m) + float(plane_clearance_m))
        & (z_values <= float(support_plane_z_m) + float(max_object_height_m))
        & (roi > 0)
    )
    candidate = above_plane.astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate, connectivity=8
    )
    if component_count <= 1:
        raise ValueError("support-plane removal produced no object component")

    max_area = max(1, int(round(width * height * float(max_area_fraction))))
    search_radius = 12
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
        support = int(bool(expanded[seed_y, seed_x]))
        component_rows, component_columns = np.nonzero(component_mask)
        distance_sq = float(np.min(
            (component_columns - seed_x) ** 2 + (component_rows - seed_y) ** 2
        ))
        score = (support, -distance_sq, area)
        if best_score is None or score > best_score:
            best_score = score
            best_label = component
    if best_label is None or best_score[0] <= 0:
        raise ValueError("no above-plane object component supports the RoboRefer seed")

    mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Morphology may bridge a thin gap down onto the table. Preserve filled
    # holes, but never re-admit a pixel rejected by the physical plane gate.
    mask[~above_plane] = 0
    rows, columns = np.nonzero(mask)
    if columns.size < int(min_area_px):
        raise ValueError("selected above-plane component is too small")
    padding = max(0, int(bbox_padding_px))
    bbox = [
        max(0, int(columns.min()) - padding),
        max(0, int(rows.min()) - padding),
        min(width - 1, int(columns.max()) + padding),
        min(height - 1, int(rows.max()) + padding),
    ]
    selected_z = z_values[mask > 0]
    return {
        "mask": mask,
        "bbox_xyxy": bbox,
        "grasp_pixel_xy": [seed_x, seed_y],
        "mask_area_px": int(np.count_nonzero(mask)),
        "seed_point_base_m": [float(value) for value in seed_point],
        "mask_base_z_min_m": float(np.min(selected_z)),
        "mask_base_z_max_m": float(np.max(selected_z)),
        "segmentation_method": "roborefer_seeded_above_support_plane_component",
    }


def estimate_base_dimensions(
    points_base: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    support_plane_z_m: Optional[float] = None,
) -> Dict[str, object]:
    """Estimate robust object dimensions in metres from a visible point cloud.

    Height follows base-frame Z. Horizontal width/depth follow PCA axes in the
    base XY plane, so the result is independent of image perspective and of the
    object's yaw on the table. Percentiles suppress isolated depth-edge pixels.
    """
    points = np.asarray(points_base, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 32:
        raise ValueError("too few finite base-frame points for 3D measurement")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("invalid robust percentile range")

    horizontal = points[:, :2]
    horizontal_center = np.median(horizontal, axis=0)
    covariance = np.cov(horizontal - horizontal_center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    projected = (horizontal - horizontal_center) @ axes
    horizontal_low = np.percentile(projected, lower_percentile, axis=0)
    horizontal_high = np.percentile(projected, upper_percentile, axis=0)
    horizontal_extents = np.maximum(0.0, horizontal_high - horizontal_low)
    horizontal_extents = np.sort(horizontal_extents)[::-1]
    z_low, z_high = np.percentile(
        points[:, 2], [lower_percentile, upper_percentile]
    )
    center = np.median(points, axis=0)
    width_m = float(horizontal_extents[0])
    depth_m = float(horizontal_extents[1])
    visible_surface_z_span_m = float(max(0.0, z_high - z_low))
    if support_plane_z_m is None:
        height_m = visible_surface_z_span_m
        height_method = "robust_visible_base_z_extent"
    else:
        # A depth camera cannot observe the lower hemisphere of a round object
        # from above. In this tabletop demo both objects rest on the support
        # plane, so top-minus-plane recovers full physical height without using
        # an object pose, model registry or Gazebo ground truth.
        height_m = float(max(0.0, z_high - float(support_plane_z_m)))
        height_method = "robust_top_z_minus_depth_estimated_support_plane"
    return {
        "width_m": width_m,
        "depth_m": depth_m,
        "height_m": height_m,
        "visible_surface_z_span_m": visible_surface_z_span_m,
        "visible_bbox_volume_m3": width_m * depth_m * height_m,
        "center_base_m": [float(value) for value in center],
        "point_count": int(points.shape[0]),
        "method": "base_xy_pca_width_and_support_plane_height",
        "height_method": height_method,
        "support_plane_z_m": (
            None if support_plane_z_m is None else float(support_plane_z_m)
        ),
        "percentile_range": [float(lower_percentile), float(upper_percentile)],
    }


def comparison_winner(
    first: Dict[str, object],
    second: Dict[str, object],
    key: str,
    tie_tolerance_m: float,
) -> str:
    """Return A, B or TIE for one metric dimension."""
    difference = float(first[key]) - float(second[key])
    if abs(difference) <= float(tie_tolerance_m):
        return "TIE"
    return "A" if difference > 0.0 else "B"


def classify_point_by_masks(
    pixel_xy: Tuple[int, int],
    masks: Dict[str, np.ndarray],
    max_distance_px: float = 30.0,
) -> Tuple[str, float]:
    """Assign a predicted point to A/B without using configured object poses."""
    x_pixel, y_pixel = int(pixel_xy[0]), int(pixel_xy[1])
    direct = []
    distances = {}
    for key, raw_mask in masks.items():
        mask = np.asarray(raw_mask) > 0
        if mask.ndim != 2:
            raise ValueError("classification masks must be HxW")
        height, width = mask.shape
        if 0 <= x_pixel < width and 0 <= y_pixel < height and mask[y_pixel, x_pixel]:
            direct.append(key)
            distances[key] = 0.0
            continue
        rows, columns = np.nonzero(mask)
        if columns.size:
            distances[key] = float(np.sqrt(np.min(
                (columns - x_pixel) ** 2 + (rows - y_pixel) ** 2
            )))
    if len(direct) == 1:
        return direct[0], 0.0
    if len(direct) > 1:
        return "AMBIGUOUS", 0.0
    if not distances:
        return "UNKNOWN", float("inf")
    closest = min(distances, key=distances.get)
    distance = distances[closest]
    if distance > float(max_distance_px):
        return "UNKNOWN", distance
    return closest, distance


def mask_centroid_pixel(mask: np.ndarray) -> Tuple[int, int]:
    """Return the integer centroid of a non-empty inferred object mask."""
    rows, columns = np.nonzero(np.asarray(mask) > 0)
    if columns.size == 0:
        raise ValueError("cannot find the centroid of an empty mask")
    return int(round(float(np.mean(columns)))), int(round(float(np.mean(rows))))


def _signed_relation(
    delta: float,
    positive_label: str,
    negative_label: str,
    aligned_label: str,
    tolerance: float,
) -> Dict[str, object]:
    """Classify one signed axis difference with an explicit uncertainty gate."""
    value = float(delta)
    threshold = max(0.0, float(tolerance))
    if value > threshold:
        label = positive_label
    elif value < -threshold:
        label = negative_label
    else:
        label = aligned_label
    return {
        "relation": label,
        "signed_delta": value,
        "magnitude": abs(value),
        "tolerance": threshold,
    }


def relative_position_between_objects(
    first_center_base_m,
    second_center_base_m,
    first_centroid_pixel_xy,
    second_centroid_pixel_xy,
    first_camera_depth_m: float,
    second_camera_depth_m: float,
    metric_tolerance_m: float = 0.015,
    image_tolerance_px: float = 12.0,
) -> Dict[str, object]:
    """Describe B relative to A in both camera view and robot base frame.

    ``base_link`` follows ROS convention: +X forward, +Y left, +Z up. Camera
    image coordinates follow +U right and +V down. Relations inside the
    tolerance band are reported as aligned instead of inventing an ordering.
    """
    first_base = np.asarray(first_center_base_m, dtype=np.float64).reshape(3)
    second_base = np.asarray(second_center_base_m, dtype=np.float64).reshape(3)
    base_delta = second_base - first_base
    first_pixel = np.asarray(first_centroid_pixel_xy, dtype=np.float64).reshape(2)
    second_pixel = np.asarray(second_centroid_pixel_xy, dtype=np.float64).reshape(2)
    image_delta = second_pixel - first_pixel
    depth_delta = float(second_camera_depth_m) - float(first_camera_depth_m)
    return {
        "subject": "B",
        "reference": "A",
        "semantics": "B relative to A using object-region centers",
        "base_link_axes": "+X forward, +Y left, +Z up",
        "camera_axes": "+U right, +V down, +depth farther",
        "base_delta_m": {
            "x": float(base_delta[0]),
            "y": float(base_delta[1]),
            "z": float(base_delta[2]),
        },
        "base_relations": {
            "x": _signed_relation(
                base_delta[0], "FORWARD", "BEHIND", "SAME_X",
                metric_tolerance_m,
            ),
            "y": _signed_relation(
                base_delta[1], "LEFT", "RIGHT", "SAME_Y",
                metric_tolerance_m,
            ),
            "z": _signed_relation(
                base_delta[2], "HIGHER", "LOWER", "SAME_Z",
                metric_tolerance_m,
            ),
        },
        "camera_delta": {
            "u_px": float(image_delta[0]),
            "v_px": float(image_delta[1]),
            "depth_m": depth_delta,
        },
        "camera_relations": {
            "u": _signed_relation(
                image_delta[0], "RIGHT", "LEFT", "SAME_U",
                image_tolerance_px,
            ),
            "v": _signed_relation(
                image_delta[1], "BELOW", "ABOVE", "SAME_V",
                image_tolerance_px,
            ),
            "depth": _signed_relation(
                depth_delta, "FARTHER", "CLOSER", "SAME_DEPTH",
                metric_tolerance_m,
            ),
        },
        "metric_tolerance_m": float(metric_tolerance_m),
        "image_tolerance_px": float(image_tolerance_px),
    }


def evaluate_five_step_reference(
    objects: Dict[str, Dict[str, object]],
    trigger_payload: Dict[str, object],
    relationship: Dict[str, object],
    assignment_limit_px: float = 30.0,
    rule_profile: str = "relative_round_target",
    forbidden_target_terms: Tuple[str, ...] = ("apple", "táo", "ycb_apple"),
    anchor_min_aspect_ratio: float = 1.5,
    compact_target_max_aspect_ratio: float = 1.75,
    size_margin_m: float = 0.005,
    between_corridor_m: float = 0.10,
) -> Dict[str, object]:
    """Score five observable clauses after the model's final point is locked.

    RoboRefer-2B-SFT emits a final point rather than an auditable chain of
    thought.  These checks therefore measure whether that point satisfies the
    five-clause spatial reference; they never claim to expose hidden reasoning.
    Direct A/B queries are post-inference referees and cannot change the locked
    model point.
    """
    required_keys = (
        ("A", "B", "C")
        if rule_profile == "two_reference_between_target"
        else ("A", "B")
    )
    if not all(key in objects for key in required_keys):
        raise ValueError(
            "five-step evaluation requires post-inference "
            + "/".join(required_keys)
            + " regions"
        )
    raw_points = trigger_payload.get("roborefer_points_xy")
    if (
        not isinstance(raw_points, list)
        or not raw_points
        or not isinstance(raw_points[0], list)
        or len(raw_points[0]) != 2
    ):
        raise ValueError("trigger payload has no locked raw RoboRefer point")
    model_pixel = tuple(int(round(value)) for value in raw_points[0])
    masks = {key: np.asarray(objects[key]["mask"]) for key in required_keys}
    model_selection, assignment_distance = classify_point_by_masks(
        model_pixel, masks, assignment_limit_px
    )

    dimensions_a = objects["A"]["dimensions"]
    dimensions_b = objects["B"]["dimensions"]
    anchor_ratio = float(dimensions_a["height_m"]) / max(
        float(dimensions_a["width_m"]), 1e-9
    )
    target_ratio = float(dimensions_b["height_m"]) / max(
        float(dimensions_b["width_m"]), 1e-9
    )
    camera = (
        relationship["camera_relations"]
        if rule_profile == "relative_round_target"
        else {}
    )
    tri_reference_geometry = None
    if rule_profile == "two_reference_between_target":
        dimensions_c = objects["C"]["dimensions"]
        horizontal_roundness_a = float(dimensions_a["width_m"]) / max(
            float(dimensions_a["depth_m"]), 1e-9
        )
        horizontal_elongation_b = float(dimensions_b["width_m"]) / max(
            float(dimensions_b["depth_m"]), 1e-9
        )
        center_a_xy = np.asarray(
            dimensions_a["center_base_m"][:2], dtype=np.float64
        )
        center_b_xy = np.asarray(
            dimensions_b["center_base_m"][:2], dtype=np.float64
        )
        center_c_xy = np.asarray(
            dimensions_c["center_base_m"][:2], dtype=np.float64
        )
        reference_segment = center_b_xy - center_a_xy
        segment_length_sq = float(np.dot(reference_segment, reference_segment))
        if segment_length_sq < 1e-9:
            between_fraction = float("nan")
            corridor_distance = float("inf")
        else:
            between_fraction = float(
                np.dot(center_c_xy - center_a_xy, reference_segment)
                / segment_length_sq
            )
            closest = center_a_xy + between_fraction * reference_segment
            corridor_distance = float(np.linalg.norm(center_c_xy - closest))
        is_between = (
            0.05 <= between_fraction <= 0.95
            and corridor_distance <= float(between_corridor_m)
        )
        height_a = float(dimensions_a["height_m"])
        height_b = float(dimensions_b["height_m"])
        height_c = float(dimensions_c["height_m"])
        tri_reference_geometry = {
            "reference_a_to_b_distance_m": float(np.sqrt(segment_length_sq)),
            "candidate_projection_fraction": between_fraction,
            "candidate_corridor_distance_m": corridor_distance,
            "corridor_limit_m": float(between_corridor_m),
        }
        steps = [
            {
                "step": 1,
                "clause": "reference A is horizontally compact/round",
                "passed": horizontal_roundness_a <= float(
                    compact_target_max_aspect_ratio
                ),
                "evidence": (
                    f"A width/depth={horizontal_roundness_a:.2f} "
                    f"(required <={float(compact_target_max_aspect_ratio):.2f})"
                ),
            },
            {
                "step": 2,
                "clause": "reference B is horizontally elongated",
                "passed": horizontal_elongation_b >= float(
                    anchor_min_aspect_ratio
                ),
                "evidence": (
                    f"B width/depth={horizontal_elongation_b:.2f} "
                    f"(required >={float(anchor_min_aspect_ratio):.2f})"
                ),
            },
            {
                "step": 3,
                "clause": "candidate lies between both references in base XY",
                "passed": is_between,
                "evidence": (
                    f"fraction={between_fraction:.2f}; "
                    f"corridor={1000.0 * corridor_distance:.0f}mm "
                    f"(limit {1000.0 * float(between_corridor_m):.0f}mm)"
                ),
            },
            {
                "step": 4,
                "clause": "candidate is physically taller than both references",
                "passed": (
                    height_c > max(height_a, height_b) + float(size_margin_m)
                ),
                "evidence": (
                    f"H_C={1000.0 * height_c:.0f}mm vs "
                    f"H_A={1000.0 * height_a:.0f}mm, "
                    f"H_B={1000.0 * height_b:.0f}mm"
                ),
            },
            {
                "step": 5,
                "clause": "locked model point lies inside the surviving candidate",
                "passed": model_selection == "C" and assignment_distance == 0.0,
                "evidence": (
                    f"raw point={model_pixel}; region={model_selection}; "
                    f"distance={assignment_distance:.1f}px"
                ),
            },
        ]
    elif rule_profile == "compact_target":
        height_a = float(dimensions_a["height_m"])
        height_b = float(dimensions_b["height_m"])
        width_a = float(dimensions_a["width_m"])
        width_b = float(dimensions_b["width_m"])
        steps = [
            {
                "step": 1,
                "clause": "reference is a tall upright package",
                "passed": anchor_ratio >= float(anchor_min_aspect_ratio),
                "evidence": (
                    f"A height/width={anchor_ratio:.2f} "
                    f"(required >={float(anchor_min_aspect_ratio):.2f})"
                ),
            },
            {
                "step": 2,
                "clause": "candidate is spatially compact",
                "passed": target_ratio <= float(compact_target_max_aspect_ratio),
                "evidence": (
                    f"B height/width={target_ratio:.2f} "
                    f"(required <={float(compact_target_max_aspect_ratio):.2f})"
                ),
            },
            {
                "step": 3,
                "clause": "candidate is physically shorter than reference",
                "passed": height_b + float(size_margin_m) < height_a,
                "evidence": (
                    f"H_B={1000.0 * height_b:.0f}mm vs "
                    f"H_A={1000.0 * height_a:.0f}mm"
                ),
            },
            {
                "step": 4,
                "clause": "candidate is physically narrower than reference",
                "passed": width_b + float(size_margin_m) < width_a,
                "evidence": (
                    f"W_B={1000.0 * width_b:.0f}mm vs "
                    f"W_A={1000.0 * width_a:.0f}mm"
                ),
            },
            {
                "step": 5,
                "clause": "locked model point lies inside the surviving candidate",
                "passed": model_selection == "B" and assignment_distance == 0.0,
                "evidence": (
                    f"raw point={model_pixel}; region={model_selection}; "
                    f"distance={assignment_distance:.1f}px"
                ),
            },
        ]
    elif rule_profile == "relative_round_target":
        steps = [
            {
                "step": 1,
                "clause": "anchor is a tall upright object",
                "passed": anchor_ratio >= 1.5,
                "evidence": f"A height/width={anchor_ratio:.2f} (required >=1.50)",
            },
            {
                "step": 2,
                "clause": "round candidate is to the right of anchor",
                "passed": (
                    0.65 <= target_ratio <= 1.45
                    and camera["u"]["relation"] == "RIGHT"
                ),
                "evidence": (
                    f"B height/width={target_ratio:.2f}; "
                    f"B {camera['u']['relation']} {camera['u']['magnitude']:.0f}px"
                ),
            },
            {
                "step": 3,
                "clause": "candidate center is lower in the image",
                "passed": camera["v"]["relation"] == "BELOW",
                "evidence": (
                    f"B {camera['v']['relation']} {camera['v']['magnitude']:.0f}px"
                ),
            },
            {
                "step": 4,
                "clause": "candidate is farther from the camera",
                "passed": camera["depth"]["relation"] == "FARTHER",
                "evidence": (
                    f"B {camera['depth']['relation']} "
                    f"{1000.0 * camera['depth']['magnitude']:.0f}mm"
                ),
            },
            {
                "step": 5,
                "clause": "locked model point lies inside the surviving candidate",
                "passed": model_selection == "B" and assignment_distance == 0.0,
                "evidence": (
                    f"raw point={model_pixel}; region={model_selection}; "
                    f"distance={assignment_distance:.1f}px"
                ),
            },
        ]
    else:
        raise ValueError(f"unknown five-step rule profile: {rule_profile}")
    for step in steps:
        step["verdict"] = "PASS" if step["passed"] else "FAIL"
    passed_count = sum(bool(step["passed"]) for step in steps)
    instruction = str(trigger_payload.get("instruction", ""))
    lower_instruction = instruction.lower()
    forbidden_terms_found = [
        str(term) for term in forbidden_target_terms
        if str(term).strip() and str(term).strip().lower() in lower_instruction
    ]
    protocol_valid = not forbidden_terms_found
    return {
        "evaluation_kind": "observable_five_clause_behavioral_test",
        "not_chain_of_thought": True,
        "model_checkpoint_stage": "SFT",
        "main_inference_prompt": instruction,
        "main_roborefer_raw": str(trigger_payload.get("roborefer_raw", "")),
        "main_inference_latency_ms": trigger_payload.get(
            "inference_latency_ms"
        ),
        "main_grounder_selection_published_before_gate": bool(
            trigger_payload.get("selection_published", False)
        ),
        "main_refined_grasp_pixel_xy": trigger_payload.get("grasp_pixel_xy"),
        "rule_profile": rule_profile,
        "target_name_in_main_prompt": bool(forbidden_terms_found),
        "forbidden_target_terms_found": forbidden_terms_found,
        "protocol_valid": protocol_valid,
        "target_coordinates_supplied_before_inference": False,
        "post_inference_referee_used": True,
        "post_inference_referee_can_change_locked_point": False,
        "locked_model_point_xy": list(model_pixel),
        "locked_model_point_region": model_selection,
        "locked_model_point_assignment_distance_px": assignment_distance,
        "two_reference_geometry": tri_reference_geometry,
        "steps": steps,
        "passed_steps": passed_count,
        "total_steps": len(steps),
        "verdict": (
            "PASS" if passed_count == len(steps) and protocol_valid else "FAIL"
        ),
    }


def build_verified_target_selection(
    trigger_payload: Dict[str, object],
    evaluation: Dict[str, object],
    object_id: str,
    colour: str,
    oracle_name: str,
) -> Dict[str, object]:
    """Create the manipulation handoff only after all five checks pass."""
    if evaluation.get("verdict") != "PASS":
        raise ValueError("five-step evaluation did not pass; target remains blocked")
    bbox = trigger_payload.get("bbox_xyxy")
    grasp_pixel = trigger_payload.get("grasp_pixel_xy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("verified handoff requires a valid inferred bbox")
    if not isinstance(grasp_pixel, list) or len(grasp_pixel) != 2:
        raise ValueError("verified handoff requires a valid inferred grasp pixel")
    selection = dict(trigger_payload)
    selection.update({
        "status": "SELECTED",
        "source": "roborefer_five_step_verified",
        "selection_published": True,
        "object_id": str(object_id),
        "color": str(colour).lower(),
        "oracle_name": str(oracle_name),
        "five_step_verdict": "PASS",
        "five_step_passed_steps": int(evaluation["passed_steps"]),
        "five_step_total_steps": int(evaluation["total_steps"]),
        "registry_identity_added_after_inference": True,
    })
    return selection


class RoboReferDimensionComparator(Node):
    """Measure RoboRefer-selected objects using only registered RGB-D + TF."""

    def __init__(self) -> None:
        super().__init__("roborefer_dimension_comparator")
        self.declare_parameter("image_topic", "/wrist_camera/color/image_raw")
        self.declare_parameter("depth_topic", "/wrist_camera/depth/image_raw")
        self.declare_parameter(
            "camera_info_topic", "/wrist_camera/color/camera_info"
        )
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter(
            "trigger_topic", "/ur3_perception/roborefer/status"
        )
        self.declare_parameter(
            "output_topic", "/ur3_perception/roborefer/dimension_comparison"
        )
        self.declare_parameter(
            "status_topic", "/ur3_perception/roborefer/dimension_status"
        )
        self.declare_parameter("server_url", "http://127.0.0.1:25547")
        self.declare_parameter("worker_timeout_sec", 45.0)
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
        self.declare_parameter("support_plane_search_min_z_m", -0.03)
        self.declare_parameter("support_plane_search_max_z_m", 0.04)
        self.declare_parameter("support_plane_histogram_bin_m", 0.002)
        self.declare_parameter("support_plane_clearance_m", 0.004)
        self.declare_parameter("max_object_height_m", 0.30)
        self.declare_parameter("dimension_lower_percentile", 0.5)
        self.declare_parameter("dimension_upper_percentile", 99.5)
        self.declare_parameter("dimension_tie_tolerance_m", 0.005)
        self.declare_parameter("comparison_mode", "dimensions")
        self.declare_parameter("near_far_assignment_max_distance_px", 30.0)
        self.declare_parameter("relative_position_tolerance_m", 0.015)
        self.declare_parameter("relative_position_image_tolerance_px", 12.0)
        self.declare_parameter("five_step_publish_target_selection", False)
        self.declare_parameter("five_step_target_selection_topic", "/ur3_perception/target_selection")
        self.declare_parameter("five_step_target_object_id", "ycb_apple_01")
        self.declare_parameter("five_step_target_color", "red")
        self.declare_parameter("five_step_target_oracle_name", "ycb_apple")
        self.declare_parameter("five_step_rule_profile", "relative_round_target")
        self.declare_parameter(
            "five_step_forbidden_target_terms_csv", "apple,táo,ycb_apple"
        )
        self.declare_parameter("five_step_anchor_min_aspect_ratio", 1.5)
        self.declare_parameter("five_step_compact_target_max_aspect_ratio", 1.75)
        self.declare_parameter("five_step_size_margin_m", 0.005)
        self.declare_parameter("five_step_between_corridor_m", 0.10)
        self.declare_parameter("object_a_label", "MUSTARD BOTTLE")
        self.declare_parameter(
            "object_a_description",
            "the tall yellow mustard bottle-shaped object standing on the table",
        )
        self.declare_parameter("object_b_label", "RED APPLE")
        self.declare_parameter(
            "object_b_description", "the round red apple resting on the table"
        )
        self.declare_parameter("object_c_label", "")
        self.declare_parameter("object_c_description", "")

        self._bridge = CvBridge()
        self._frame: Optional[np.ndarray] = None
        self._depth_m: Optional[np.ndarray] = None
        self._frame_message: Optional[Image] = None
        self._camera_info: Optional[CameraInfo] = None
        self._rgb_stamp = 0.0
        self._depth_stamp = 0.0
        self._triggered = False
        self._trigger_payload: Dict[str, object] = {}
        self._active = False
        self._completed = False
        self._next_attempt = 0.0
        self._tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=3,
        )
        retained_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
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
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback,
            sensor_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("trigger_topic").value),
            self._trigger_callback,
            10,
        )
        self._image_pub = self.create_publisher(
            Image, str(self.get_parameter("output_topic").value), retained_qos
        )
        self._status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), retained_qos
        )
        self._verified_selection_pub = self.create_publisher(
            String,
            str(self.get_parameter("five_step_target_selection_topic").value),
            retained_qos,
        )
        self.create_timer(0.2, self._maybe_compare)
        mode = str(self.get_parameter("comparison_mode").value).strip().lower()
        if mode not in (
            "dimensions", "near_far_ablation", "relative_position",
            "five_step_reasoning",
        ):
            raise ValueError(
                "comparison_mode must be 'dimensions', 'near_far_ablation' "
                "'relative_position', or 'five_step_reasoning'"
            )
        self.get_logger().info(
            f"RoboRefer comparison ready: mode={mode}; "
            "RoboRefer point + registered depth + metric 3D"
        )

    def _image_callback(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Cannot decode comparison RGB frame: {exc}")
            return
        self._frame = np.asarray(frame).copy()
        self._frame_message = message
        self._rgb_stamp = stamp_seconds(message)

    def _depth_callback(self, message: Image) -> None:
        try:
            depth = np.asarray(
                self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            )
            if depth.ndim == 3:
                depth = depth[..., 0]
            self._depth_m = (
                depth.astype(np.float32) * 0.001
                if depth.dtype == np.uint16
                else depth.astype(np.float32)
            )
            self._depth_stamp = stamp_seconds(message)
        except Exception as exc:
            self.get_logger().warning(f"Cannot decode comparison depth frame: {exc}")

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self._camera_info = message

    def _trigger_callback(self, message: String) -> None:
        if self._triggered or self._completed:
            return
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if (
            payload.get("status") == "SELECTED"
            and payload.get("source") == "roborefer_rgbd_depth_component"
        ):
            self._trigger_payload = dict(payload)
            self._triggered = True
            self.get_logger().info(
                "METRIC_COMPARISON_TRIGGERED: frozen RGB-D frame available"
            )

    @staticmethod
    def _encode_image(image: np.ndarray, extension: str, params=None) -> str:
        success, encoded = cv2.imencode(extension, image, params or [])
        if not success:
            raise RuntimeError(f"Cannot encode comparison image as {extension}")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _query_point(
        self,
        frame: np.ndarray,
        depth_view: np.ndarray,
        prompt: str,
        enable_depth: bool,
    ) -> Tuple[Tuple[float, float], str, float]:
        body = json.dumps({
            "image_url": [self._encode_image(
                frame, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90]
            )],
            "depth_url": (
                [self._encode_image(depth_view, ".png")] if enable_depth else []
            ),
            "enable_depth": int(enable_depth),
            "text": prompt + COORDINATE_SUFFIX,
        }).encode("utf-8")
        endpoint = str(self.get_parameter("server_url").value).rstrip("/") + "/query"
        request = urlrequest.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urlrequest.urlopen(
                request,
                timeout=float(self.get_parameter("worker_timeout_sec").value),
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"RoboRefer server HTTP {exc.code}: {detail}") from exc
        except (urlerror.URLError, TimeoutError) as exc:
            raise RuntimeError(f"RoboRefer server unavailable: {exc}") from exc
        latency_ms = 1000.0 * (time.monotonic() - started)
        answer = str(result.get("answer", "")).strip()
        points = parse_points(answer)
        if not points:
            raise RuntimeError(f"RoboRefer returned no point: {answer[:240]}")
        return points[0], answer, latency_ms

    def _query_object_point(
        self,
        frame: np.ndarray,
        depth_view: np.ndarray,
        description: str,
        other_description: str,
    ) -> Tuple[Tuple[float, float], str, float, str]:
        prompt = (
            f"Locate only {description} in the complete RGB-D tabletop scene. "
            "Point to one visible point well inside its silhouette near its "
            f"center. Do not point to {other_description}, the tray, cubes, "
            "robot, gripper, or text labels."
        )
        point, answer, latency_ms = self._query_point(
            frame, depth_view, prompt, True
        )
        return point, answer, latency_ms, prompt

    def _segment(
        self,
        points_base_image: np.ndarray,
        pixel: Tuple[int, int],
        support_plane_z_m: float,
    ) -> Dict:
        return segment_seeded_above_plane_component(
            points_base_image,
            pixel,
            support_plane_z_m,
            roi_radius_px=int(self.get_parameter("depth_roi_radius_px").value),
            plane_clearance_m=float(
                self.get_parameter("support_plane_clearance_m").value
            ),
            max_object_height_m=float(
                self.get_parameter("max_object_height_m").value
            ),
            min_area_px=int(self.get_parameter("min_mask_area_px").value),
            max_area_fraction=float(
                self.get_parameter("max_mask_area_fraction").value
            ),
            bbox_padding_px=int(self.get_parameter("bbox_padding_px").value),
        )

    def _lookup_rigid_transform(self, message: Image, source_frame: str):
        base_frame = str(self.get_parameter("base_frame").value)
        try:
            transform = self._tf_buffer.lookup_transform(
                base_frame,
                source_frame,
                Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=0.5),
            )
        except TransformException:
            transform = self._tf_buffer.lookup_transform(
                base_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.5),
            )
        rotation_message = transform.transform.rotation
        translation_message = transform.transform.translation
        rotation = quaternion_rotation_matrix(
            rotation_message.x,
            rotation_message.y,
            rotation_message.z,
            rotation_message.w,
        )
        translation = np.asarray([
            translation_message.x,
            translation_message.y,
            translation_message.z,
        ], dtype=np.float64)
        return rotation, translation, base_frame

    @staticmethod
    def _draw_double_arrow(
        image: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        colour: Tuple[int, int, int],
    ) -> None:
        cv2.arrowedLine(image, start, end, colour, 2, tipLength=0.08)
        cv2.arrowedLine(image, end, start, colour, 2, tipLength=0.08)

    @classmethod
    def _draw_results(
        cls,
        frame: np.ndarray,
        objects: Dict[str, Dict],
        comparison: Dict[str, str],
    ) -> np.ndarray:
        """Render a self-explanatory one-frame pipeline and metric result."""
        overlay = frame.copy()
        object_keys = [key for key in ("A", "B", "C") if key in objects]
        for key in object_keys:
            result = objects[key]
            mask = result["mask"] > 0
            colour = OBJECT_COLOURS_BGR[key]
            tint = np.empty_like(overlay)
            tint[:] = colour
            overlay[mask] = cv2.addWeighted(
                overlay[mask], 0.50, tint[mask], 0.50, 0.0
            )
            contours, _ = cv2.findContours(
                result["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, colour, 3)
            x0, y0, x1, y1 = result["bbox_xyxy"]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, 2)
            x_pixel, y_pixel = result["pixel_xy"]
            cv2.drawMarker(
                overlay,
                (x_pixel, y_pixel),
                colour,
                markerType=cv2.MARKER_CROSS,
                markerSize=22,
                thickness=3,
            )
            horizontal_y = min(388, y1 + 10)
            cls._draw_double_arrow(
                overlay, (x0, horizontal_y), (x1, horizontal_y), colour
            )
            vertical_x = max(8, x0 - 9)
            cls._draw_double_arrow(
                overlay, (vertical_x, y0), (vertical_x, y1), colour
            )
            dimensions = result["dimensions"]
            card_x = max(7, min(x0 + 5, overlay.shape[1] - 185))
            card_y = max(75, min(y0 + 7, overlay.shape[0] - 132))
            card = overlay.copy()
            cv2.rectangle(
                card, (card_x, card_y), (card_x + 178, card_y + 48), (5, 5, 5), -1
            )
            overlay = cv2.addWeighted(card, 0.78, overlay, 0.22, 0.0)
            cv2.putText(
                overlay,
                f"{key} {result['label'][:16]}",
                (card_x + 6, card_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                colour,
                2,
            )
            cv2.putText(
                overlay,
                f"W {1000.0 * dimensions['width_m']:.0f} | "
                f"H {1000.0 * dimensions['height_m']:.0f} mm",
                (card_x + 6, card_y + 39),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                colour,
                2,
            )

        header = overlay.copy()
        cv2.rectangle(header, (0, 0), (overlay.shape[1], 68), (8, 8, 8), -1)
        overlay = cv2.addWeighted(header, 0.82, overlay, 0.18, 0.0)
        cv2.putText(
            overlay,
            "ROBOREFER A/B -> DEPTH MASKS -> METRIC 3D (base_link)",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            "W=XY extent | H=top-detected table | RoboRefer RGB-D only",
            (12, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (185, 225, 255),
            1,
        )

        footer_top = max(0, overlay.shape[0] - 76)
        footer = overlay.copy()
        cv2.rectangle(
            footer,
            (0, footer_top),
            (overlay.shape[1], overlay.shape[0]),
            (8, 8, 8),
            -1,
        )
        overlay = cv2.addWeighted(footer, 0.84, overlay, 0.16, 0.0)
        names = {key: objects[key]["label"] for key in ("A", "B")}
        height_winner = comparison["height_winner"]
        width_winner = comparison["width_winner"]
        height_text = (
            "HEIGHT: TIE"
            if height_winner == "TIE"
            else f"HEIGHT: {height_winner} {names[height_winner]} is taller"
        )
        width_text = (
            "WIDTH: TIE"
            if width_winner == "TIE"
            else f"WIDTH: {width_winner} {names[width_winner]} is wider"
        )
        cv2.putText(
            overlay,
            height_text,
            (14, footer_top + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            OBJECT_COLOURS_BGR.get(height_winner, (255, 255, 255)),
            2,
        )
        cv2.putText(
            overlay,
            width_text,
            (14, footer_top + 59),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            OBJECT_COLOURS_BGR.get(width_winner, (255, 255, 255)),
            2,
        )
        return overlay

    @classmethod
    def _draw_near_far_results(
        cls,
        frame: np.ndarray,
        objects: Dict[str, Dict],
        ablation: Dict[str, object],
    ) -> np.ndarray:
        """Show same-RGB/same-prompt RGB-only versus RGB-D near/far outputs."""
        overlay = frame.copy()
        for key in ("A", "B"):
            result = objects[key]
            mask = result["mask"] > 0
            colour = OBJECT_COLOURS_BGR[key]
            tint = np.empty_like(overlay)
            tint[:] = colour
            overlay[mask] = cv2.addWeighted(
                overlay[mask], 0.62, tint[mask], 0.38, 0.0
            )
            contours, _ = cv2.findContours(
                result["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, colour, 3)
            x0, y0, x1, y1 = result["bbox_xyxy"]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, 2)
            card_x = max(7, min(x0 + 5, overlay.shape[1] - 185))
            card_y = max(75, min(y0 + 7, overlay.shape[0] - 154))
            card = overlay.copy()
            cv2.rectangle(
                card, (card_x, card_y), (card_x + 178, card_y + 47), (5, 5, 5), -1
            )
            overlay = cv2.addWeighted(card, 0.80, overlay, 0.20, 0.0)
            cv2.putText(
                overlay,
                f"{key} {result['label'][:16]}",
                (card_x + 6, card_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                colour,
                2,
            )
            cv2.putText(
                overlay,
                f"DEPTH {1000.0 * result['median_camera_depth_m']:.0f} mm",
                (card_x + 6, card_y + 39),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                colour,
                2,
            )

        rgb_point = tuple(ablation["rgb_only"]["pixel_xy"])
        rgbd_point = tuple(ablation["rgbd"]["pixel_xy"])
        rgb_colour = (0, 165, 255)
        rgbd_colour = (80, 255, 80)
        cv2.circle(overlay, rgb_point, 18, rgb_colour, 3)
        cv2.putText(
            overlay,
            "RGB",
            (rgb_point[0] + 12, max(88, rgb_point[1] - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            rgb_colour,
            2,
        )
        cv2.drawMarker(
            overlay,
            rgbd_point,
            rgbd_colour,
            markerType=cv2.MARKER_DIAMOND,
            markerSize=30,
            thickness=3,
        )
        cv2.putText(
            overlay,
            "RGB-D",
            (rgbd_point[0] + 12, min(358, rgbd_point[1] + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            rgbd_colour,
            2,
        )

        header = overlay.copy()
        cv2.rectangle(header, (0, 0), (overlay.shape[1], 68), (8, 8, 8), -1)
        overlay = cv2.addWeighted(header, 0.84, overlay, 0.16, 0.0)
        cv2.putText(
            overlay,
            "NEAR/FAR ABLATION: SAME RGB + SAME PROMPT",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            "orange circle=RGB only | green diamond=RGB-D | depth is referee",
            (12, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (190, 230, 255),
            1,
        )

        footer_top = max(0, overlay.shape[0] - 108)
        footer = overlay.copy()
        cv2.rectangle(
            footer,
            (0, footer_top),
            (overlay.shape[1], overlay.shape[0]),
            (8, 8, 8),
            -1,
        )
        overlay = cv2.addWeighted(footer, 0.88, overlay, 0.12, 0.0)
        nearer = str(ablation["metric_nearer"])
        farther = "B" if nearer == "A" else "A"
        margin_mm = 1000.0 * float(ablation["metric_depth_margin_m"])
        depth_line = (
            f"DEPTH REFEREE: {nearer} nearer than {farther} "
            f"by {margin_mm:.0f} mm"
        )
        rgb = ablation["rgb_only"]
        rgbd = ablation["rgbd"]
        outcome_line = (
            f"RGB ONLY -> {rgb['selection']} [{rgb['verdict']}]     "
            f"RGB-D -> {rgbd['selection']} [{rgbd['verdict']}]"
        )
        verdict = str(ablation["verdict"])
        verdict_lines = {
            "RGBD_IMPROVES": "RESULT: RGB-D corrects the RGB-only error",
            "BOTH_CORRECT": "RESULT: both correct; RGB-D adds metric evidence",
            "RGB_ONLY_BETTER": "RESULT: RGB-only wins this trial; inspect depth fusion",
            "BOTH_FAIL": "RESULT: both fail this trial",
        }
        cv2.putText(
            overlay,
            depth_line,
            (14, footer_top + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            OBJECT_COLOURS_BGR.get(nearer, (255, 255, 255)),
            2,
        )
        cv2.putText(
            overlay,
            outcome_line,
            (14, footer_top + 61),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (235, 235, 235),
            2,
        )
        cv2.putText(
            overlay,
            verdict_lines.get(verdict, f"RESULT: {verdict}"),
            (14, footer_top + 93),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            rgbd_colour if rgbd["verdict"] == "PASS" else rgb_colour,
            2,
        )
        return overlay

    @classmethod
    def _draw_relative_position_results(
        cls,
        frame: np.ndarray,
        objects: Dict[str, Dict],
        relationship: Dict[str, object],
    ) -> np.ndarray:
        """Render B relative to A in camera and base_link coordinates."""
        overlay = frame.copy()
        centroids = {}
        for key in ("A", "B"):
            result = objects[key]
            mask = np.asarray(result["mask"]) > 0
            colour = OBJECT_COLOURS_BGR[key]
            tint = np.empty_like(overlay)
            tint[:] = colour
            overlay[mask] = cv2.addWeighted(
                overlay[mask], 0.60, tint[mask], 0.40, 0.0
            )
            contours, _ = cv2.findContours(
                result["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, colour, 3)
            x0, y0, x1, y1 = result["bbox_xyxy"]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, 2)
            center_pixel = tuple(result["mask_centroid_pixel_xy"])
            centroids[key] = center_pixel
            cv2.circle(overlay, center_pixel, 8, colour, -1)
            cv2.circle(overlay, center_pixel, 13, (255, 255, 255), 2)

            center_base = result["dimensions"]["center_base_m"]
            card_x = max(7, min(x0 + 5, overlay.shape[1] - 205))
            card_y = max(74, min(y0 + 7, overlay.shape[0] - 170))
            card = overlay.copy()
            cv2.rectangle(
                card, (card_x, card_y), (card_x + 198, card_y + 49), (5, 5, 5), -1
            )
            overlay = cv2.addWeighted(card, 0.82, overlay, 0.18, 0.0)
            cv2.putText(
                overlay,
                f"{key} {result['label'][:16]}",
                (card_x + 6, card_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                colour,
                2,
            )
            xyz_mm = [1000.0 * float(value) for value in center_base]
            cv2.putText(
                overlay,
                f"XYZ {xyz_mm[0]:+.0f}, {xyz_mm[1]:+.0f}, {xyz_mm[2]:+.0f} mm",
                (card_x + 6, card_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                colour,
                1,
            )

        cv2.arrowedLine(
            overlay,
            centroids["A"],
            centroids["B"],
            (255, 255, 255),
            3,
            tipLength=0.08,
        )
        midpoint = (
            (centroids["A"][0] + centroids["B"][0]) // 2,
            (centroids["A"][1] + centroids["B"][1]) // 2,
        )
        cv2.putText(
            overlay,
            "A -> B",
            (midpoint[0] - 30, max(82, midpoint[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            2,
        )

        header = overlay.copy()
        cv2.rectangle(header, (0, 0), (overlay.shape[1], 68), (8, 8, 8), -1)
        overlay = cv2.addWeighted(header, 0.86, overlay, 0.14, 0.0)
        cv2.putText(
            overlay,
            "RELATIVE POSITION: B RELATIVE TO A",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.59,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            "RoboRefer A/B -> depth regions -> 3D centers in base_link",
            (12, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (190, 230, 255),
            1,
        )

        base = relationship["base_relations"]
        camera = relationship["camera_relations"]
        camera_line = (
            f"CAMERA: B {camera['u']['relation']} {camera['u']['magnitude']:.0f}px | "
            f"{camera['v']['relation']} {camera['v']['magnitude']:.0f}px | "
            f"{camera['depth']['relation']} "
            f"{1000.0 * camera['depth']['magnitude']:.0f}mm"
        )
        robot_line = (
            f"ROBOT: B {base['y']['relation']} {1000.0 * base['y']['magnitude']:.0f}mm | "
            f"{base['x']['relation']} {1000.0 * base['x']['magnitude']:.0f}mm | "
            f"{base['z']['relation']} {1000.0 * base['z']['magnitude']:.0f}mm"
        )
        footer_top = max(0, overlay.shape[0] - 112)
        footer = overlay.copy()
        cv2.rectangle(
            footer,
            (0, footer_top),
            (overlay.shape[1], overlay.shape[0]),
            (8, 8, 8),
            -1,
        )
        overlay = cv2.addWeighted(footer, 0.90, overlay, 0.10, 0.0)
        cv2.putText(
            overlay,
            camera_line,
            (12, footer_top + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (100, 230, 255),
            2,
        )
        cv2.putText(
            overlay,
            robot_line,
            (12, footer_top + 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (100, 255, 150),
            2,
        )
        cv2.putText(
            overlay,
            "base_link axes: +X forward | +Y left | +Z up | center-to-center",
            (12, footer_top + 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1,
        )
        return overlay

    @classmethod
    def _draw_five_step_results(
        cls,
        frame: np.ndarray,
        objects: Dict[str, Dict],
        evaluation: Dict[str, object],
    ) -> np.ndarray:
        """Show the locked point and all five independently scored clauses."""
        scene = frame.copy()
        object_keys = [key for key in ("A", "B", "C") if key in objects]
        for key in object_keys:
            result = objects[key]
            mask = np.asarray(result["mask"]) > 0
            colour = OBJECT_COLOURS_BGR[key]
            tint = np.empty_like(scene)
            tint[:] = colour
            scene[mask] = cv2.addWeighted(
                scene[mask], 0.58, tint[mask], 0.42, 0.0
            )
            contours, _ = cv2.findContours(
                result["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(scene, contours, -1, colour, 3)
            x0, y0, x1, y1 = result["bbox_xyxy"]
            cv2.rectangle(scene, (x0, y0), (x1, y1), colour, 2)
            cv2.putText(
                scene,
                f"{key}: {result['label']}",
                (x0, max(22, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                colour,
                2,
            )
        center_a = tuple(objects["A"]["mask_centroid_pixel_xy"])
        center_b = tuple(objects["B"]["mask_centroid_pixel_xy"])
        if "C" in objects:
            center_c = tuple(objects["C"]["mask_centroid_pixel_xy"])
            cv2.arrowedLine(
                scene, center_a, center_c, (255, 255, 255), 3, tipLength=0.08
            )
            cv2.arrowedLine(
                scene, center_c, center_b, (255, 255, 255), 3, tipLength=0.08
            )
        else:
            cv2.arrowedLine(
                scene, center_a, center_b, (255, 255, 255), 3, tipLength=0.08
            )
        locked = tuple(evaluation["locked_model_point_xy"])
        verdict_colour = (
            (60, 235, 100) if evaluation["verdict"] == "PASS" else (60, 80, 255)
        )
        cv2.drawMarker(
            scene, locked, verdict_colour, cv2.MARKER_DIAMOND, 28, 4
        )
        cv2.putText(
            scene,
            "LOCKED MODEL POINT",
            (max(8, locked[0] - 85), max(22, locked[1] - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            verdict_colour,
            2,
        )

        height, width = frame.shape[:2]
        canvas = np.full_like(frame, 10)
        cv2.rectangle(canvas, (0, 0), (width, 58), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            "ROBOREFER: 5-CLAUSE TWO-REFERENCE GATE -> UR3",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            "Main prompt: no target identity, no coordinates | checks after point lock",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (185, 225, 255),
            1,
        )
        scene_bottom = min(height, 246)
        canvas[58:scene_bottom] = cv2.resize(
            scene, (width, scene_bottom - 58), interpolation=cv2.INTER_AREA
        )
        row_y = scene_bottom + 23
        for step in evaluation["steps"]:
            passed = bool(step["passed"])
            colour = (60, 235, 100) if passed else (60, 80, 255)
            text = (
                f"{step['step']}  {step['verdict']:<4} | "
                f"{step['evidence']}"
            )
            cv2.putText(
                canvas,
                text[:86],
                (12, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
            )
            row_y += 31
        cv2.rectangle(canvas, (0, height - 43), (width, height), (22, 22, 22), -1)
        verdict_text = (
            "5/5 PASS -> TARGET RELEASED TO UR3 PICK-PLACE"
            if evaluation["verdict"] == "PASS"
            else f"{evaluation['passed_steps']}/5 FAIL -> PICK TARGET BLOCKED"
        )
        cv2.putText(
            canvas,
            verdict_text,
            (12, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            verdict_colour,
            2,
        )
        return canvas

    def _maybe_compare(self) -> None:
        if (
            not self._triggered
            or self._active
            or self._completed
            or self._frame is None
            or self._depth_m is None
            or self._frame_message is None
            or self._camera_info is None
            or time.monotonic() < self._next_attempt
        ):
            return
        self._active = True
        try:
            skew = abs(self._rgb_stamp - self._depth_stamp)
            allowed = float(self.get_parameter("max_rgb_depth_skew_sec").value)
            if skew > allowed:
                raise RuntimeError(f"RGB/depth skew {skew:.3f}s exceeds {allowed:.3f}s")
            frame = self._frame.copy()
            depth_m = self._depth_m.copy()
            source_message = self._frame_message
            camera_info = self._camera_info
            if frame.shape[:2] != depth_m.shape[:2]:
                raise RuntimeError("comparison RGB and registered depth sizes differ")
            if int(camera_info.width) != frame.shape[1] or int(camera_info.height) != frame.shape[0]:
                raise RuntimeError("CameraInfo resolution does not match RGB-D frame")
            source_frame = camera_info.header.frame_id or source_message.header.frame_id
            if not source_frame:
                raise RuntimeError("registered camera optical frame is empty")
            rotation, translation, base_frame = self._lookup_rigid_transform(
                source_message, source_frame
            )
            intrinsics = (
                float(camera_info.k[0]),
                float(camera_info.k[4]),
                float(camera_info.k[2]),
                float(camera_info.k[5]),
            )
            depth_view = normalize_depth_for_roborefer(
                depth_m,
                float(self.get_parameter("min_depth_m").value),
                float(self.get_parameter("max_depth_m").value),
            )
            points_base_image = depth_to_base_image(
                depth_m,
                intrinsics,
                rotation,
                translation,
                float(self.get_parameter("min_depth_m").value),
                float(self.get_parameter("max_depth_m").value),
            )
            support_plane_z_m = estimate_support_plane_z(
                points_base_image,
                float(self.get_parameter("support_plane_search_min_z_m").value),
                float(self.get_parameter("support_plane_search_max_z_m").value),
                float(self.get_parameter("support_plane_histogram_bin_m").value),
            )
            self.get_logger().info(
                f"SUPPORT_PLANE_FROM_DEPTH: z={support_plane_z_m:.4f}m"
            )
            height, width = frame.shape[:2]
            descriptions = {
                "A": str(self.get_parameter("object_a_description").value),
                "B": str(self.get_parameter("object_b_description").value),
            }
            labels = {
                "A": str(self.get_parameter("object_a_label").value),
                "B": str(self.get_parameter("object_b_label").value),
            }
            rule_profile = str(self.get_parameter(
                "five_step_rule_profile"
            ).value).strip()
            object_keys = ["A", "B"]
            if rule_profile == "two_reference_between_target":
                descriptions["C"] = str(self.get_parameter(
                    "object_c_description"
                ).value).strip()
                labels["C"] = str(self.get_parameter(
                    "object_c_label"
                ).value).strip()
                if not descriptions["C"] or not labels["C"]:
                    raise RuntimeError(
                        "two-reference profile requires object C label/description"
                    )
                object_keys.append("C")
            objects: Dict[str, Dict] = {}
            for key in object_keys:
                other_description = ", ".join(
                    descriptions[other_key]
                    for other_key in object_keys
                    if other_key != key
                )
                normalized, raw_answer, latency_ms, prompt = self._query_object_point(
                    frame,
                    depth_view,
                    descriptions[key],
                    other_description,
                )
                pixel = points_to_pixels([normalized], width, height)[0]
                segmentation = self._segment(
                    points_base_image, pixel, support_plane_z_m
                )
                base_points = points_base_image[segmentation["mask"] > 0]
                dimensions = estimate_base_dimensions(
                    base_points,
                    float(self.get_parameter("dimension_lower_percentile").value),
                    float(self.get_parameter("dimension_upper_percentile").value),
                    support_plane_z_m,
                )
                selected_depth = depth_m[segmentation["mask"] > 0]
                selected_depth = selected_depth[
                    np.isfinite(selected_depth) & (selected_depth > 0.0)
                ]
                if selected_depth.size < 32:
                    raise RuntimeError(
                        f"object {key} region has too few metric depth samples"
                    )
                objects[key] = {
                    "label": labels[key],
                    "description": descriptions[key],
                    "prompt": prompt,
                    "normalized_point_xy": list(normalized),
                    "pixel_xy": list(pixel),
                    "raw_answer": raw_answer,
                    "inference_latency_ms": round(latency_ms, 3),
                    "mask": segmentation["mask"],
                    "mask_centroid_pixel_xy": list(mask_centroid_pixel(
                        segmentation["mask"]
                    )),
                    "mask_area_px": segmentation["mask_area_px"],
                    "bbox_xyxy": segmentation["bbox_xyxy"],
                    "seed_point_base_m": segmentation["seed_point_base_m"],
                    "mask_base_z_range_m": [
                        segmentation["mask_base_z_min_m"],
                        segmentation["mask_base_z_max_m"],
                    ],
                    "segmentation_method": segmentation["segmentation_method"],
                    "median_camera_depth_m": float(np.median(selected_depth)),
                    "camera_depth_percentile_range_m": [
                        float(np.percentile(selected_depth, 5.0)),
                        float(np.percentile(selected_depth, 95.0)),
                    ],
                    "dimensions": dimensions,
                }
                self.get_logger().info(
                    f"OBJECT_{key}: {labels[key]}, point={pixel}, "
                    f"mask={segmentation['mask_area_px']}px, "
                    f"W={1000.0 * dimensions['width_m']:.1f}mm, "
                    f"H={1000.0 * dimensions['height_m']:.1f}mm, "
                    f"latency={latency_ms:.1f}ms"
                )

            overlap_fraction = 0.0
            overlap_pair = ""
            for first_index, first_key in enumerate(object_keys):
                for second_key in object_keys[first_index + 1:]:
                    first_mask = objects[first_key]["mask"] > 0
                    second_mask = objects[second_key]["mask"] > 0
                    overlap = int(np.count_nonzero(first_mask & second_mask))
                    smaller_area = min(
                        int(np.count_nonzero(first_mask)),
                        int(np.count_nonzero(second_mask)),
                    )
                    fraction = overlap / max(1, smaller_area)
                    if fraction > overlap_fraction:
                        overlap_fraction = fraction
                        overlap_pair = f"{first_key}/{second_key}"
            mode = str(self.get_parameter("comparison_mode").value).strip().lower()
            if overlap_fraction > 0.20 and mode != "five_step_reasoning":
                raise RuntimeError(
                    "RoboRefer points produced overlapping depth regions "
                    f"for {overlap_pair} ({100.0 * overlap_fraction:.1f}%); "
                    "refusing a false comparison"
                )

            if mode == "five_step_reasoning":
                def relationship_between(first_key, second_key):
                    return relative_position_between_objects(
                        objects[first_key]["dimensions"]["center_base_m"],
                        objects[second_key]["dimensions"]["center_base_m"],
                        objects[first_key]["mask_centroid_pixel_xy"],
                        objects[second_key]["mask_centroid_pixel_xy"],
                        objects[first_key]["median_camera_depth_m"],
                        objects[second_key]["median_camera_depth_m"],
                        float(self.get_parameter(
                            "relative_position_tolerance_m"
                        ).value),
                        float(self.get_parameter(
                            "relative_position_image_tolerance_px"
                        ).value),
                    )

                if rule_profile == "two_reference_between_target":
                    relationship = {
                        "A_to_B": relationship_between("A", "B"),
                        "A_to_C": relationship_between("A", "C"),
                        "C_to_B": relationship_between("C", "B"),
                        "semantics": (
                            "C candidate evaluated between reference A and "
                            "reference B using inferred RGB-D region centers"
                        ),
                    }
                else:
                    relationship = relationship_between("A", "B")
                comparison = evaluate_five_step_reference(
                    objects,
                    self._trigger_payload,
                    relationship,
                    float(self.get_parameter(
                        "near_far_assignment_max_distance_px"
                    ).value),
                    rule_profile,
                    tuple(
                        term.strip()
                        for term in str(self.get_parameter(
                            "five_step_forbidden_target_terms_csv"
                        ).value).split(",")
                        if term.strip()
                    ),
                    float(self.get_parameter(
                        "five_step_anchor_min_aspect_ratio"
                    ).value),
                    float(self.get_parameter(
                        "five_step_compact_target_max_aspect_ratio"
                    ).value),
                    float(self.get_parameter(
                        "five_step_size_margin_m"
                    ).value),
                    float(self.get_parameter(
                        "five_step_between_corridor_m"
                    ).value),
                )
                comparison["relationship_referee"] = relationship
                comparison["region_overlap_gate"] = {
                    "passed": overlap_fraction <= 0.20,
                    "pair": overlap_pair,
                    "fraction": overlap_fraction,
                    "maximum_allowed_fraction": 0.20,
                }
                if overlap_fraction > 0.20:
                    pair_keys = set(overlap_pair.split("/"))
                    affected_steps = set()
                    if "A" in pair_keys:
                        affected_steps.add(1)
                    if "B" in pair_keys:
                        affected_steps.add(2)
                    if "C" in pair_keys:
                        affected_steps.add(5)
                    for step in comparison["steps"]:
                        if int(step["step"]) in affected_steps:
                            step["passed"] = False
                            step["verdict"] = "FAIL"
                            step["evidence"] += (
                                f"; region overlap {overlap_pair}="
                                f"{100.0 * overlap_fraction:.1f}%"
                            )
                    comparison["passed_steps"] = sum(
                        bool(step["passed"]) for step in comparison["steps"]
                    )
                    comparison["verdict"] = "FAIL"
                publish_verified = bool(self.get_parameter(
                    "five_step_publish_target_selection"
                ).value)
                comparison["target_handoff_requested"] = publish_verified
                comparison["target_handoff_published"] = False
                if comparison["verdict"] == "PASS" and publish_verified:
                    verified_selection = build_verified_target_selection(
                        self._trigger_payload,
                        comparison,
                        str(self.get_parameter(
                            "five_step_target_object_id"
                        ).value),
                        str(self.get_parameter(
                            "five_step_target_color"
                        ).value),
                        str(self.get_parameter(
                            "five_step_target_oracle_name"
                        ).value),
                    )
                    self._verified_selection_pub.publish(
                        String(data=json.dumps(verified_selection))
                    )
                    comparison["target_handoff_published"] = True
                    comparison["target_handoff"] = {
                        "object_id": verified_selection["object_id"],
                        "color": verified_selection["color"],
                        "oracle_name": verified_selection["oracle_name"],
                        "source": verified_selection["source"],
                        "identity_added_after_inference": True,
                    }
                    self.get_logger().info(
                        "FIVE_STEP_TARGET_RELEASED: all checks passed; "
                        f"object_id={verified_selection['object_id']}"
                    )
                visualization = self._draw_five_step_results(
                    frame, objects, comparison
                )
                payload_source = "roborefer_sft_five_step_behavioral_test"
                pipeline = [
                    "single_composite_five_clause_rgbd_prompt",
                    "lock_raw_roborefer_point_before_referee",
                    (
                        "post_inference_direct_A_B_C_region_referee"
                        if rule_profile == "two_reference_between_target"
                        else "post_inference_direct_A_B_region_referee"
                    ),
                    "registered_depth_geometric_clause_checks",
                    "release_pick_place_target_only_on_5_of_5_pass",
                ]
                log_message = (
                    "FIVE_STEP_REASONING_COMPLETE: "
                    f"passed={comparison['passed_steps']}/5, "
                    f"point_region={comparison['locked_model_point_region']}, "
                    f"verdict={comparison['verdict']}, "
                    f"handoff={comparison['target_handoff_published']}"
                )
            elif mode == "relative_position":
                comparison = relative_position_between_objects(
                    objects["A"]["dimensions"]["center_base_m"],
                    objects["B"]["dimensions"]["center_base_m"],
                    objects["A"]["mask_centroid_pixel_xy"],
                    objects["B"]["mask_centroid_pixel_xy"],
                    objects["A"]["median_camera_depth_m"],
                    objects["B"]["median_camera_depth_m"],
                    float(self.get_parameter(
                        "relative_position_tolerance_m"
                    ).value),
                    float(self.get_parameter(
                        "relative_position_image_tolerance_px"
                    ).value),
                )
                visualization = self._draw_relative_position_results(
                    frame, objects, comparison
                )
                payload_source = "roborefer_seeded_rgbd_relative_position"
                pipeline = [
                    "roborefer_select_two_objects_without_coordinates",
                    "registered_depth_object_regions",
                    "camera_intrinsic_backprojection",
                    "tf_to_base_link",
                    "center_to_center_relative_position_with_tolerance",
                ]
                base_relations = comparison["base_relations"]
                camera_relations = comparison["camera_relations"]
                log_message = (
                    "RELATIVE_POSITION_COMPLETE: B relative A; "
                    f"camera={camera_relations['u']['relation']},"
                    f"{camera_relations['v']['relation']},"
                    f"{camera_relations['depth']['relation']}; "
                    f"base={base_relations['x']['relation']},"
                    f"{base_relations['y']['relation']},"
                    f"{base_relations['z']['relation']}"
                )
            elif mode == "near_far_ablation":
                # This text is byte-for-byte identical for the RGB-only and
                # RGB-D calls; only enable_depth/depth_url differ.
                comparison_prompt = (
                    f"Between {descriptions['A']} and {descriptions['B']}, "
                    "which object is closer to the camera? Point to one visible "
                    "point well inside the closer object's silhouette. Judge "
                    "distance along the camera view, not physical size or image "
                    "position. Return exactly one point."
                )
                rgb_normalized, rgb_raw, rgb_latency = self._query_point(
                    frame, depth_view, comparison_prompt, False
                )
                rgbd_normalized, rgbd_raw, rgbd_latency = self._query_point(
                    frame, depth_view, comparison_prompt, True
                )
                rgb_pixel = points_to_pixels([rgb_normalized], width, height)[0]
                rgbd_pixel = points_to_pixels([rgbd_normalized], width, height)[0]
                assignment_limit = float(
                    self.get_parameter("near_far_assignment_max_distance_px").value
                )
                masks = {key: objects[key]["mask"] for key in ("A", "B")}
                rgb_selection, rgb_distance = classify_point_by_masks(
                    rgb_pixel, masks, assignment_limit
                )
                rgbd_selection, rgbd_distance = classify_point_by_masks(
                    rgbd_pixel, masks, assignment_limit
                )
                depth_a = float(objects["A"]["median_camera_depth_m"])
                depth_b = float(objects["B"]["median_camera_depth_m"])
                metric_nearer = "A" if depth_a < depth_b else "B"
                metric_margin = abs(depth_a - depth_b)
                rgb_pass = rgb_selection == metric_nearer
                rgbd_pass = rgbd_selection == metric_nearer
                if rgbd_pass and not rgb_pass:
                    overall_verdict = "RGBD_IMPROVES"
                elif rgbd_pass and rgb_pass:
                    overall_verdict = "BOTH_CORRECT"
                elif rgb_pass:
                    overall_verdict = "RGB_ONLY_BETTER"
                else:
                    overall_verdict = "BOTH_FAIL"
                comparison = {
                    "prompt": comparison_prompt,
                    "prompt_identical_between_modalities": True,
                    "rgb_image_identical_between_modalities": True,
                    "metric_nearer": metric_nearer,
                    "metric_depth_margin_m": metric_margin,
                    "rgb_only": {
                        "enable_depth": False,
                        "raw_answer": rgb_raw,
                        "normalized_point_xy": list(rgb_normalized),
                        "pixel_xy": list(rgb_pixel),
                        "selection": rgb_selection,
                        "assignment_distance_px": rgb_distance,
                        "inference_latency_ms": round(rgb_latency, 3),
                        "verdict": "PASS" if rgb_pass else "FAIL",
                    },
                    "rgbd": {
                        "enable_depth": True,
                        "depth_source": "registered_wrist_camera",
                        "raw_answer": rgbd_raw,
                        "normalized_point_xy": list(rgbd_normalized),
                        "pixel_xy": list(rgbd_pixel),
                        "selection": rgbd_selection,
                        "assignment_distance_px": rgbd_distance,
                        "inference_latency_ms": round(rgbd_latency, 3),
                        "verdict": "PASS" if rgbd_pass else "FAIL",
                    },
                    "verdict": overall_verdict,
                }
                visualization = self._draw_near_far_results(
                    frame, objects, comparison
                )
                payload_source = "roborefer_rgb_vs_rgbd_near_far_ablation"
                pipeline = [
                    "same_frozen_rgb_frame",
                    "same_near_far_prompt",
                    "roborefer_rgb_only",
                    "roborefer_with_registered_depth",
                    "metric_depth_referee_after_inference",
                ]
                log_message = (
                    "NEAR_FAR_ABLATION_COMPLETE: "
                    f"metric={metric_nearer}, rgb={rgb_selection}, "
                    f"rgbd={rgbd_selection}, verdict={overall_verdict}"
                )
            else:
                tie_tolerance = float(
                    self.get_parameter("dimension_tie_tolerance_m").value
                )
                comparison = {
                    "height_winner": comparison_winner(
                        objects["A"]["dimensions"],
                        objects["B"]["dimensions"],
                        "height_m",
                        tie_tolerance,
                    ),
                    "width_winner": comparison_winner(
                        objects["A"]["dimensions"],
                        objects["B"]["dimensions"],
                        "width_m",
                        tie_tolerance,
                    ),
                    "tie_tolerance_m": tie_tolerance,
                }
                visualization = self._draw_results(frame, objects, comparison)
                payload_source = "roborefer_seeded_metric_3d_comparison"
                pipeline = [
                    "roborefer_select_two_objects",
                    "registered_depth_connected_components",
                    "camera_intrinsic_backprojection",
                    "tf_to_base_frame",
                    "robust_3d_dimension_comparison",
                ]
                log_message = (
                    "METRIC_COMPARISON_COMPLETE: "
                    f"height={comparison['height_winner']}, "
                    f"width={comparison['width_winner']}"
                )

            output = self._bridge.cv2_to_imgmsg(visualization, encoding="bgr8")
            output.header = source_message.header
            self._image_pub.publish(output)

            serializable_objects = {}
            for key, result in objects.items():
                serializable_objects[key] = {
                    field: value
                    for field, value in result.items()
                    if field != "mask"
                }
            payload = {
                "status": "COMPLETE",
                "source": payload_source,
                "comparison_mode": mode,
                "pipeline": pipeline,
                "objects": serializable_objects,
                "comparison": comparison,
                "camera_frame": source_frame,
                "measurement_frame": base_frame,
                "support_plane": {
                    "source": "dominant_plane_estimated_from_registered_depth",
                    "z_base_m": round(support_plane_z_m, 6),
                    "clearance_m": float(
                        self.get_parameter("support_plane_clearance_m").value
                    ),
                },
                "rgb_depth_skew_sec": round(skew, 6),
                "mask_overlap_fraction": round(overlap_fraction, 6),
                "target_coordinates_supplied_before_inference": False,
                "oracle_used_for_inference": False,
                "sam2_used": False,
            }
            self._status_pub.publish(String(data=json.dumps(payload)))
            self._completed = True
            self.get_logger().info(log_message)
        except Exception as exc:
            self._next_attempt = time.monotonic() + 5.0
            self.get_logger().warning(f"RoboRefer RGB-D comparison unavailable: {exc}")
            self._status_pub.publish(String(data=json.dumps({
                "status": "ERROR",
                "source": "roborefer_seeded_rgbd_spatial_demo",
                "detail": str(exc),
            })))
        finally:
            self._active = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoboReferDimensionComparator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
