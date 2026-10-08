#!/usr/bin/env python3
"""Deterministic helpers shared by RoboRefer spatial-grounding nodes."""

import math
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np


CANONICAL_CUBES = {
    "red": ("cube_red_01", "red_cube"),
    "green": ("cube_green_01", "green_cube"),
    "blue": ("cube_blue_01", "blue_cube"),
    "yellow": ("cube_yellow_01", "yellow_cube"),
    "orange": ("cube_orange_01", "orange_cube"),
    "purple": ("cube_purple_01", "purple_cube"),
    "pink": ("cube_pink_01", "pink_cube"),
}

CANONICAL_KITCHEN_TARGETS = {
    "cracker": ("ycb_cracker_box_01", "red", "ycb_cracker_box"),
    "sugar": ("ycb_sugar_box_01", "yellow", "ycb_sugar_box"),
    "tomato soup": ("ycb_tomato_soup_can_01", "red", "ycb_tomato_soup_can"),
    "mustard": ("ycb_mustard_bottle_01", "yellow", "ycb_mustard_bottle"),
    "banana": ("ycb_banana_01", "yellow", "ycb_banana"),
    "apple": ("ycb_apple_01", "red", "ycb_apple"),
    "orange": ("ycb_orange_01", "orange", "ycb_orange"),
    "power drill": ("ycb_power_drill_01", "blue", "ycb_power_drill"),
    "drill": ("ycb_power_drill_01", "blue", "ycb_power_drill"),
    "mango": ("mango_01", "orange", "mango"),
    "cup": ("cup_01", "blue", "cup"),
    "kettle": ("kettle_01", "silver", "kettle"),
}

POINT_PATTERN = re.compile(
    r"\(\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*\)"
)


def manipulation_event_reaches_state(payload: dict, trigger_state: str) -> bool:
    """Return true only for the runner's start event for ``trigger_state``."""
    return (
        isinstance(payload, dict)
        and payload.get("event") == "step_start"
        and str(payload.get("state", "")) == str(trigger_state)
    )


def canonical_target(label: str) -> Optional[Tuple[str, str, str]]:
    """Return registry id, colour and Gazebo oracle name for a known object."""
    lowered = label.lower()
    if "cube" in lowered:
        for colour, (object_id, oracle_name) in CANONICAL_CUBES.items():
            if colour in lowered:
                return object_id, colour, oracle_name
        return None
    for keyword, target in CANONICAL_KITCHEN_TARGETS.items():
        if keyword in lowered:
            return target
    return None


def parse_points(text: str) -> List[Tuple[float, float]]:
    """Extract finite normalized coordinate tuples from a model answer."""
    points = []
    for match in POINT_PATTERN.finditer(text):
        point = (float(match.group(1)), float(match.group(2)))
        if all(math.isfinite(value) for value in point):
            points.append(point)
    return points


def points_to_pixels(
    points: Sequence[Tuple[float, float]], width: int, height: int
) -> List[Tuple[int, int]]:
    """Convert normalized points to clamped camera pixels."""
    pixels = []
    for x_value, y_value in points:
        if not (0.0 <= x_value <= 1.0 and 0.0 <= y_value <= 1.0):
            continue
        x_pixel = int(round(x_value * max(0, width - 1)))
        y_pixel = int(round(y_value * max(0, height - 1)))
        pixels.append((x_pixel, y_pixel))
    return pixels


def representative_grasp_point(
    points_xy: Sequence[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """Return a robust image-space grasp point from model-supported pixels."""
    if not points_xy:
        return None
    values = np.asarray(points_xy, dtype=np.float64)
    return tuple(int(round(value)) for value in np.median(values, axis=0))
