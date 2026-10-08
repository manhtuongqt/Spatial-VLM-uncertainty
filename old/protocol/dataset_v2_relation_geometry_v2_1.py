#!/usr/bin/env python3
"""Observable and predictive relation geometry for Dataset V2.1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


SCHEMA_VERSION = 1
GEOMETRY_VERSION = "dataset_v2_relation_geometry_v2_1.1.0"
DEPTH_MIN_M = 0.10
DEPTH_MAX_M = 2.00
DEPTH_MARGIN_M = 0.020
PREDICTOR_SCREENING_MARGIN_M = 0.035
HORIZONTAL_MARGIN_PX = 12.0
PREDICTOR_HORIZONTAL_MARGIN_PX = 24.0
MASK_EROSION_ITERATIONS = 2
MIN_INTERIOR_PIXELS = 64


class RelationGeometryError(RuntimeError):
    """Raised when observable relation evidence is invalid."""


@dataclass(frozen=True)
class InstanceEvidence:
    label: int
    mask_pixels: int
    interior_valid_depth_pixels: int
    centroid_x_px: float
    centroid_y_px: float
    median_depth_m: float
    depth_mad_m: float
    depth_p10_m: float
    depth_p90_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mask_pixels": self.mask_pixels,
            "interior_valid_depth_pixels": self.interior_valid_depth_pixels,
            "centroid_x_px": self.centroid_x_px,
            "centroid_y_px": self.centroid_y_px,
            "median_depth_m": self.median_depth_m,
            "depth_mad_m": self.depth_mad_m,
            "depth_p10_m": self.depth_p10_m,
            "depth_p90_m": self.depth_p90_m,
        }


def instance_evidence(labels: np.ndarray, depth: np.ndarray, label: int) -> InstanceEvidence:
    if labels.shape != depth.shape:
        raise RelationGeometryError("semantic labels and metric depth shapes differ")
    mask = labels == int(label)
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise RelationGeometryError(f"semantic label is not visible: {label}")
    interior = cv2.erode(
        mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
        iterations=MASK_EROSION_ITERATIONS,
    ).astype(bool)
    valid = interior & np.isfinite(depth) & (depth >= DEPTH_MIN_M) & (depth <= DEPTH_MAX_M)
    values = depth[valid].astype(np.float64)
    if values.size < MIN_INTERIOR_PIXELS:
        raise RelationGeometryError(
            f"insufficient valid interior metric depth for label {label}: "
            f"{values.size} < {MIN_INTERIOR_PIXELS}"
        )
    median = float(np.median(values))
    return InstanceEvidence(
        label=int(label),
        mask_pixels=int(xs.size),
        interior_valid_depth_pixels=int(values.size),
        centroid_x_px=float(np.mean(xs)),
        centroid_y_px=float(np.mean(ys)),
        median_depth_m=median,
        depth_mad_m=float(np.median(np.abs(values - median))),
        depth_p10_m=float(np.percentile(values, 10)),
        depth_p90_m=float(np.percentile(values, 90)),
    )


def relation_predicate(
    relation: str,
    target_x_px: float,
    target_depth_m: float,
    anchor_x_px: list[float],
    anchor_depth_m: list[float],
    *,
    depth_margin_m: float = DEPTH_MARGIN_M,
    horizontal_margin_px: float = HORIZONTAL_MARGIN_PX,
) -> tuple[bool, float]:
    if relation == "direct":
        return True, math.inf
    if not anchor_depth_m or not anchor_x_px:
        return False, -math.inf
    if relation == "right_of":
        signed = target_x_px - anchor_x_px[0] - horizontal_margin_px
    elif relation == "left_of":
        signed = anchor_x_px[0] - target_x_px - horizontal_margin_px
    elif relation in {"nearer_than", "front_of"}:
        signed = anchor_depth_m[0] - target_depth_m - depth_margin_m
    elif relation in {"farther_than", "behind"}:
        signed = target_depth_m - anchor_depth_m[0] - depth_margin_m
    elif relation == "between_in_depth" and len(anchor_depth_m) >= 2:
        lower = target_depth_m - min(anchor_depth_m) - depth_margin_m
        upper = max(anchor_depth_m) - target_depth_m - depth_margin_m
        signed = min(lower, upper)
    elif relation == "nearer_than_both" and len(anchor_depth_m) >= 2:
        signed = min(value - target_depth_m - depth_margin_m for value in anchor_depth_m)
    elif relation == "farther_than_both" and len(anchor_depth_m) >= 2:
        signed = min(target_depth_m - value - depth_margin_m for value in anchor_depth_m)
    else:
        raise RelationGeometryError(f"unsupported relation or anchor cardinality: {relation}")
    return signed > 0.0, float(signed)


def evaluate_observed_relation(
    relation: str,
    labels: np.ndarray,
    depth: np.ndarray,
    target_label: int,
    anchor_labels: list[int],
) -> dict[str, Any]:
    target = instance_evidence(labels, depth, target_label)
    anchors = [instance_evidence(labels, depth, value) for value in anchor_labels]
    passed, signed_margin = relation_predicate(
        relation,
        target.centroid_x_px,
        target.median_depth_m,
        [value.centroid_x_px for value in anchors],
        [value.median_depth_m for value in anchors],
    )
    return {
        "geometry_version": GEOMETRY_VERSION,
        "relation": relation,
        "passed": passed,
        "signed_margin": signed_margin,
        "margin_unit": "px" if relation in {"left_of", "right_of"} else "m",
        "target": target.as_dict(),
        "anchors": [value.as_dict() for value in anchors],
    }


def quaternion_rotation_matrix(xyzw: list[float]) -> np.ndarray:
    if len(xyzw) != 4 or not all(math.isfinite(float(value)) for value in xyzw):
        raise RelationGeometryError("quaternion must contain four finite xyzw values")
    x, y, z, w = [float(value) for value in xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise RelationGeometryError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def project_base_point_to_camera(
    point_base_xyz: list[float], camera_tf_base: dict[str, Any]
) -> np.ndarray:
    translation = np.asarray(camera_tf_base["position"], dtype=np.float64)
    rotation_camera_in_base = quaternion_rotation_matrix(camera_tf_base["orientation_xyzw"])
    point = np.asarray(point_base_xyz, dtype=np.float64)
    if point.shape != (3,):
        raise RelationGeometryError("base point must have xyz")
    return rotation_camera_in_base.T @ (point - translation)


def predict_median_depth(
    pose_xyyaw: list[float], registry_row: dict[str, Any], camera_tf_base: dict[str, Any],
    asset_calibration: dict[str, Any],
) -> float:
    point = [float(pose_xyyaw[0]), float(pose_xyyaw[1]), float(registry_row["z"])]
    optical = project_base_point_to_camera(point, camera_tf_base)
    offset = float(asset_calibration["median_surface_offset_m"])
    return float(optical[2] - offset)


def predicted_relation(
    relation: str,
    target_pose: list[float],
    target_registry: dict[str, Any],
    target_calibration: dict[str, Any],
    anchor_poses: list[list[float]],
    anchor_registry: list[dict[str, Any]],
    anchor_calibration: list[dict[str, Any]],
    camera_tf_base: dict[str, Any],
    camera_intrinsics: dict[str, float] | None = None,
    *,
    depth_margin_m: float = PREDICTOR_SCREENING_MARGIN_M,
) -> dict[str, Any]:
    target_point = project_base_point_to_camera(
        [float(target_pose[0]), float(target_pose[1]), float(target_registry["z"])], camera_tf_base
    )
    anchor_points = [
        project_base_point_to_camera(
            [float(pose[0]), float(pose[1]), float(row["z"])], camera_tf_base
        )
        for pose, row in zip(anchor_poses, anchor_registry)
    ]
    target_depth = float(target_point[2] - float(target_calibration["median_surface_offset_m"]))
    anchor_depth = [
        float(point[2] - float(calibration["median_surface_offset_m"]))
        for point, calibration in zip(anchor_points, anchor_calibration)
    ]
    def horizontal_value(point: np.ndarray) -> float:
        if camera_intrinsics is None:
            return float(point[0])
        return float(camera_intrinsics["fx"] * point[0] / point[2] + camera_intrinsics["cx"])

    passed, signed = relation_predicate(
        relation,
        horizontal_value(target_point),
        target_depth,
        [horizontal_value(value) for value in anchor_points],
        anchor_depth,
        depth_margin_m=depth_margin_m,
        horizontal_margin_px=(
            PREDICTOR_HORIZONTAL_MARGIN_PX if camera_intrinsics is not None else 0.03
        ),
    )
    return {
        "passed": passed,
        "screening_depth_margin_m": float(depth_margin_m),
        "signed_screening_margin": signed,
        "target_predicted_median_depth_m": target_depth,
        "anchor_predicted_median_depth_m": anchor_depth,
        "target_camera_center_xyz": target_point.tolist(),
        "anchor_camera_center_xyz": [value.tolist() for value in anchor_points],
        "target_predicted_centroid_x_px": horizontal_value(target_point),
        "anchor_predicted_centroid_x_px": [horizontal_value(value) for value in anchor_points],
    }


def asset_shape_group(registry_row: dict[str, Any]) -> str:
    semantic = str(registry_row["semantic_class"])
    height = 2.0 * float(registry_row["z"])
    diameter = 2.0 * float(registry_row["footprint_radius_m"])
    if any(token in semantic for token in ("cube", "box")):
        return "box"
    if semantic in {"apple", "orange", "lemon", "mango"}:
        return "round"
    if height >= 0.14 or height / max(diameter, 1e-6) >= 1.7:
        return "tall"
    return "low"
