#!/usr/bin/env python3
"""Structured point-seeded depth-component gate for RoboRefer.

The locked ``roborefer_pilot_v0`` protocol keeps using the legacy gate in
``roborefer_grounder.py``.  This module is the forward-compatible Gate v2: it
records every connected component considered and raises a stable reason code
instead of collapsing distinct failures into one ``ValueError`` message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import cv2
import numpy as np


AREA_TOO_LARGE = "AREA_TOO_LARGE"
AREA_TOO_SMALL = "AREA_TOO_SMALL"
NO_SEED_SUPPORT = "NO_SEED_SUPPORT"
NO_VALID_DEPTH = "NO_VALID_DEPTH"
PLANE_MERGE = "PLANE_MERGE"
BORDER_TRUNCATED = "BORDER_TRUNCATED"
UNSTABLE_3D = "UNSTABLE_3D"

DEPTH_COMPONENT_REASON_CODES = (
    AREA_TOO_LARGE,
    AREA_TOO_SMALL,
    NO_SEED_SUPPORT,
    NO_VALID_DEPTH,
    PLANE_MERGE,
    BORDER_TRUNCATED,
)


@dataclass
class DepthComponentError(ValueError):
    """A fail-closed geometry rejection with machine-readable diagnostics."""

    code: str
    detail: str
    diagnostics: Dict[str, Any]

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accepted": False,
            "reason_code": self.code,
            "detail": self.detail,
            "diagnostics": self.diagnostics,
        }


def _representative_point(points: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    values = np.asarray(points, dtype=np.float64)
    return int(round(float(values[:, 0].mean()))), int(
        round(float(values[:, 1].mean()))
    )


def _raise(
    code: str,
    detail: str,
    diagnostics: Dict[str, Any],
) -> None:
    raise DepthComponentError(code, detail, diagnostics)


def segment_seeded_depth_component_v2(
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
    reject_border_truncated: bool = False,
) -> dict:
    """Select the seeded surface or reject it with a structured reason.

    ``PLANE_MERGE`` is a diagnostic heuristic, not semantic ground truth.  It
    is emitted when an over-size seeded component reaches the finite ROI
    boundary, which is the observable signature of a surface continuing into
    a table or another similarly deep region.  A bounded over-size component
    remains ``AREA_TOO_LARGE``.
    """

    values = np.asarray(depth_m, dtype=np.float32)
    base_diagnostics: Dict[str, Any] = {
        "input_shape": list(values.shape),
        "seed_points_xy_input": [list(point) for point in seed_points_xy],
        "thresholds": {
            "min_depth_m": float(min_depth_m),
            "max_depth_m": float(max_depth_m),
            "min_area_px": int(min_area_px),
            "max_area_fraction": float(max_area_fraction),
            "reject_border_truncated": bool(reject_border_truncated),
        },
    }
    if values.ndim != 2:
        _raise(
            NO_SEED_SUPPORT,
            f"depth must be HxW, got shape={values.shape}",
            base_diagnostics,
        )

    height, width = values.shape
    points = [
        (int(x), int(y))
        for x, y in seed_points_xy
        if 0 <= int(x) < width and 0 <= int(y) < height
    ]
    base_diagnostics["seed_points_xy_valid"] = [list(point) for point in points]
    if not points:
        _raise(
            NO_SEED_SUPPORT,
            "no RoboRefer seed point lies inside the depth image",
            base_diagnostics,
        )

    valid = (
        np.isfinite(values)
        & (values >= float(min_depth_m))
        & (values <= float(max_depth_m))
    )
    base_diagnostics["valid_depth_pixels"] = int(np.count_nonzero(valid))
    radius = max(1, int(seed_radius_px))
    local_depths = []
    for x_pixel, y_pixel in points:
        x0, x1 = max(0, x_pixel - radius), min(width, x_pixel + radius + 1)
        y0, y1 = max(0, y_pixel - radius), min(height, y_pixel + radius + 1)
        samples = values[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
        if samples.size:
            local_depths.append(float(np.median(samples)))
    if not local_depths:
        _raise(
            NO_VALID_DEPTH,
            "no valid metric depth around any RoboRefer seed point",
            base_diagnostics,
        )
    seed_depth_m = float(np.median(np.asarray(local_depths, dtype=np.float32)))
    base_diagnostics["seed_depth_m"] = seed_depth_m

    center_x, center_y = _representative_point(points)
    roi_radius = max(radius + 2, int(roi_radius_px))
    roi_x0 = max(0, center_x - roi_radius)
    roi_x1 = min(width - 1, center_x + roi_radius)
    roi_y0 = max(0, center_y - roi_radius)
    roi_y1 = min(height - 1, center_y + roi_radius)
    base_diagnostics["roi_xyxy"] = [roi_x0, roi_y0, roi_x1, roi_y1]

    roi = np.zeros((height, width), dtype=np.uint8)
    roi[roi_y0 : roi_y1 + 1, roi_x0 : roi_x1 + 1] = 255
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
    max_area = max(1, int(round(width * height * float(max_area_fraction))))
    base_diagnostics["max_area_px"] = max_area
    base_diagnostics["foreground_component_count"] = max(0, component_count - 1)
    search_radius = max(2, radius * 2)
    components = []
    for component in range(1, component_count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        x0 = int(stats[component, cv2.CC_STAT_LEFT])
        y0 = int(stats[component, cv2.CC_STAT_TOP])
        component_width = int(stats[component, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        x1 = x0 + component_width - 1
        y1 = y0 + component_height - 1
        component_mask = (labels == component).astype(np.uint8)
        expanded = cv2.dilate(
            component_mask,
            np.ones((2 * search_radius + 1, 2 * search_radius + 1), np.uint8),
        )
        support = sum(bool(expanded[y, x]) for x, y in points)
        component_pixels_y, component_pixels_x = np.nonzero(component_mask)
        distance_sq = float(
            np.min(
                (component_pixels_x - center_x) ** 2
                + (component_pixels_y - center_y) ** 2
            )
        )
        touches_image_border = bool(
            x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1
        )
        touches_roi_boundary = bool(
            x0 <= roi_x0 or y0 <= roi_y0 or x1 >= roi_x1 or y1 >= roi_y1
        )
        components.append(
            {
                "label": component,
                "area_px": area,
                "bbox_xyxy": [x0, y0, x1, y1],
                "seed_support_count": int(support),
                "distance_to_seed_px": float(np.sqrt(distance_sq)),
                "touches_image_border": touches_image_border,
                "touches_roi_boundary": touches_roi_boundary,
            }
        )
    base_diagnostics["components"] = components

    supported = [item for item in components if item["seed_support_count"] > 0]
    if not supported:
        _raise(
            NO_SEED_SUPPORT,
            "no connected depth surface supports the RoboRefer seed",
            base_diagnostics,
        )

    selected = max(
        supported,
        key=lambda item: (
            item["seed_support_count"],
            -item["distance_to_seed_px"],
            item["area_px"],
        ),
    )
    base_diagnostics["selected_component"] = selected
    area = int(selected["area_px"])
    if area < int(min_area_px):
        _raise(
            AREA_TOO_SMALL,
            f"seeded component area {area}px is below minimum {int(min_area_px)}px",
            base_diagnostics,
        )
    if bool(selected["touches_image_border"]) and reject_border_truncated:
        _raise(
            BORDER_TRUNCATED,
            "seeded component touches the image boundary",
            base_diagnostics,
        )
    if area > max_area:
        if bool(selected["touches_roi_boundary"]):
            _raise(
                PLANE_MERGE,
                f"seeded component area {area}px exceeds {max_area}px and continues to the ROI boundary",
                base_diagnostics,
            )
        _raise(
            AREA_TOO_LARGE,
            f"bounded seeded component area {area}px exceeds maximum {max_area}px",
            base_diagnostics,
        )

    best_label = int(selected["label"])
    mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    y_pixels, x_pixels = np.nonzero(mask)
    if not x_pixels.size:
        _raise(
            NO_SEED_SUPPORT,
            "selected depth component became empty after cleanup",
            base_diagnostics,
        )

    padding = max(0, int(bbox_padding_px))
    bbox = [
        max(0, int(x_pixels.min()) - padding),
        max(0, int(y_pixels.min()) - padding),
        min(width - 1, int(x_pixels.max()) + padding),
        min(height - 1, int(y_pixels.max()) + padding),
    ]
    supported_points = [(x, y) for x, y in points if mask[y, x] > 0]
    if supported_points:
        grasp_pixel = _representative_point(supported_points)
    else:
        nearest_index = int(
            np.argmin((x_pixels - center_x) ** 2 + (y_pixels - center_y) ** 2)
        )
        grasp_pixel = (int(x_pixels[nearest_index]), int(y_pixels[nearest_index]))
    selected_depth = values[mask > 0]
    selected_depth = selected_depth[np.isfinite(selected_depth)]
    return {
        "accepted": True,
        "reason_code": "DEPTH_COMPONENT_ACCEPTED",
        "mask": mask,
        "bbox_xyxy": bbox,
        "grasp_pixel_xy": grasp_pixel,
        "supported_points_xy": supported_points,
        "seed_depth_m": seed_depth_m,
        "mask_area_px": int(np.count_nonzero(mask)),
        "mask_depth_min_m": float(np.min(selected_depth)),
        "mask_depth_max_m": float(np.max(selected_depth)),
        "diagnostics": base_diagnostics,
    }
