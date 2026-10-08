"""Unit tests for the RoboRefer RGB-D ROS adapter."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "roborefer_grounder.py"
SPEC = importlib.util.spec_from_file_location("roborefer_grounder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

from spatial_point_utils import (  # noqa: E402
    canonical_target,
    manipulation_event_reaches_state,
    parse_points,
    points_to_pixels,
    representative_grasp_point,
)


def test_spatial_point_helpers_are_model_neutral():
    assert parse_points("answer: [(0.25, 0.50), (.75, 0.125)]") == [
        (0.25, 0.5),
        (0.75, 0.125),
    ]
    assert points_to_pixels([(0.0, 0.0), (1.0, 1.0)], 640, 480) == [
        (0, 0),
        (639, 479),
    ]
    assert representative_grasp_point([(10, 20), (14, 24)]) == (12, 22)
    assert canonical_target("red apple") == (
        "ycb_apple_01",
        "red",
        "ycb_apple",
    )
    assert manipulation_event_reaches_state(
        {"event": "step_start", "state": "FREEZE_PERCEPTION_POSE"},
        "FREEZE_PERCEPTION_POSE",
    )


def test_metric_depth_is_converted_to_near_bright_rgb_view():
    depth_m = np.linspace(0.2, 1.2, 20 * 30, dtype=np.float32).reshape(20, 30)
    view = MODULE.normalize_depth_for_roborefer(depth_m, 0.05, 2.0)

    assert view.shape == (20, 30, 3)
    assert view.dtype == np.uint8
    assert int(view[0, 0, 0]) > int(view[-1, -1, 0])
    assert np.array_equal(view[..., 0], view[..., 1])
    assert np.array_equal(view[..., 1], view[..., 2])


def test_invalid_registered_depth_is_black_and_not_used_for_scaling():
    depth_m = np.linspace(0.2, 1.0, 20 * 30, dtype=np.float32).reshape(20, 30)
    depth_m[0, 0] = np.nan
    depth_m[0, 1] = np.inf
    depth_m[0, 2] = 0.0
    view = MODULE.normalize_depth_for_roborefer(depth_m, 0.05, 2.0)

    assert np.all(view[0, :3] == 0)


def test_flat_or_missing_depth_fails_closed():
    with pytest.raises(ValueError, match="no usable depth range"):
        MODULE.normalize_depth_for_roborefer(
            np.full((20, 20), 0.5, dtype=np.float32)
        )
    with pytest.raises(ValueError, match="too few valid pixels"):
        MODULE.normalize_depth_for_roborefer(
            np.zeros((20, 20), dtype=np.float32)
        )


def _synthetic_tabletop_depth():
    height, width = 120, 180
    depth = np.full((height, width), 0.75, dtype=np.float32)
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((xx - 70) ** 2 + (yy - 55) ** 2)
    apple = radius <= 22
    depth[apple] = 0.615 + 0.0015 * radius[apple]
    # A distractor at nearly the same depth must not join the seeded object.
    distractor = (xx - 142) ** 2 + (yy - 82) ** 2 <= 14 ** 2
    depth[distractor] = 0.625
    return depth, apple, distractor


def test_roborefer_seed_creates_bbox_and_mask_without_detector():
    depth, apple, distractor = _synthetic_tabletop_depth()
    result = MODULE.segment_seeded_depth_component(
        depth,
        [(69, 54), (73, 57)],
        roi_radius_px=100,
        near_tolerance_m=0.015,
        far_tolerance_m=0.055,
        min_area_px=100,
        max_area_fraction=0.20,
        bbox_padding_px=4,
    )

    assert result["mask"].dtype == np.uint8
    assert result["mask"][55, 70] == 255
    assert result["mask"][82, 142] == 0
    assert np.count_nonzero(result["mask"] & distractor.astype(np.uint8)) == 0
    assert result["mask_area_px"] > 1000
    x0, y0, x1, y1 = result["bbox_xyxy"]
    assert 42 <= x0 <= 52
    assert 27 <= y0 <= 37
    assert 88 <= x1 <= 98
    assert 73 <= y1 <= 83
    grasp_x, grasp_y = result["grasp_pixel_xy"]
    assert result["mask"][grasp_y, grasp_x] == 255


def test_depth_component_fails_closed_without_seed_depth():
    depth = np.full((60, 80), 0.7, dtype=np.float32)
    depth[25:35, 35:45] = np.nan
    with pytest.raises(ValueError, match="no valid metric depth"):
        MODULE.segment_seeded_depth_component(depth, [(40, 30)], seed_radius_px=2)


def test_depth_component_rejects_unbounded_surface():
    depth = np.full((80, 100), 0.62, dtype=np.float32)
    with pytest.raises(ValueError, match="no valid depth component"):
        MODULE.segment_seeded_depth_component(
            depth,
            [(50, 40)],
            roi_radius_px=100,
            max_area_fraction=0.20,
        )
