#!/usr/bin/env python3
"""Conservative apple-specific RGB-D verifier with no Gazebo oracle input.

This is a demo-scene safety check, not a general object recognizer. A matching
red round component does not prove semantic identity; fail closed on ambiguity.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def check_apple(rgb_bgr: np.ndarray, depth_m: np.ndarray,
                point_xy: tuple[int, int]) -> tuple[dict[str, Any], np.ndarray]:
    if rgb_bgr.shape != (480, 640, 3) or depth_m.shape != (480, 640):
        raise ValueError("Expected registered 640x480 RGB-D")
    x, y = map(int, point_xy)
    if not (0 <= x < 640 and 0 <= y < 480):
        raise ValueError("Candidate outside camera image")
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 75, 35), (10, 255, 255))
    red |= cv2.inRange(hsv, (170, 75, 35), (180, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(red)
    valid = []
    masks: dict[int, np.ndarray] = {}
    for component_id in range(1, count):
        area_px = int(stats[component_id, cv2.CC_STAT_AREA])
        if not 4000 <= area_px <= 20000:
            continue
        left = int(stats[component_id, cv2.CC_STAT_LEFT])
        top = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        ratio = width / max(height, 1)
        if not 0.80 <= ratio <= 1.25:
            continue
        component = (components == component_id).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * contour_area / (perimeter * perimeter) if perimeter else 0.0
        if circularity < 0.82:
            continue
        valid.append({"component_id": component_id, "area_px": area_px,
                      "bbox_xyxy": [left, top, left + width, top + height],
                      "aspect_ratio": ratio, "circularity": circularity})
        filled = np.zeros_like(component)
        cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
        masks[component_id] = filled

    candidate_ids = [row["component_id"] for row in valid]
    selected_id = next((row["component_id"] for row in valid if masks[row["component_id"]][y, x]), 0)
    selected = next((row for row in valid if row["component_id"] == selected_id), None)
    selected_mask = masks.get(selected_id, np.zeros((480, 640), dtype=np.uint8))
    interior = cv2.erode(selected_mask, np.ones((5, 5), np.uint8))
    point_inside = bool(interior[y, x])
    point_depth = float(depth_m[y, x])
    surface_depths = depth_m[selected_mask > 0]
    surface_depths = surface_depths[np.isfinite(surface_depths) & (surface_depths >= 0.10) & (surface_depths <= 2.0)]
    median_depth = float(np.median(surface_depths)) if len(surface_depths) else math.nan
    depth_consistent = (math.isfinite(point_depth) and 0.10 <= point_depth <= 2.0 and
                        math.isfinite(median_depth) and abs(point_depth - median_depth) <= 0.03)
    unique = len(valid) == 1
    passed = unique and selected is not None and point_inside and depth_consistent
    result = {"checker": "apple_red_round_registered_rgbd_v1",
              "uses_gazebo_semantic_labels": False,
              "uses_model_point_as_candidate_seed": True,
              "demo_pose_specific_thresholds_not_independently_validated": True,
              "candidate_xy": [x, y], "eligible_round_red_components": valid,
              "eligible_component_count": len(candidate_ids),
              "candidate_component_id": selected_id,
              "candidate_on_component_interior": point_inside,
              "point_depth_m": point_depth,
              "component_median_depth_m": median_depth,
              "depth_consistent_within_0_03m": depth_consistent,
              "pass": bool(passed)}
    return result, selected_mask * 255
