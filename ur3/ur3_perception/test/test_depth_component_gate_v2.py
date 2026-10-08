"""CPU-only contract tests for structured Gate v2 reason codes."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "depth_component_gate_v2", SCRIPTS / "depth_component_gate_v2.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _expect_reason(depth, seed, code, **kwargs):
    with pytest.raises(MODULE.DepthComponentError) as caught:
        MODULE.segment_seeded_depth_component_v2(depth, [seed], **kwargs)
    assert caught.value.code == code
    assert caught.value.as_dict()["reason_code"] == code
    assert isinstance(caught.value.diagnostics.get("components", []), list)
    return caught.value


def test_area_too_small_is_distinct_from_missing_seed_support():
    depth = np.full((100, 100), np.nan, dtype=np.float32)
    depth[46:54, 46:54] = 0.62
    error = _expect_reason(
        depth,
        (50, 50),
        MODULE.AREA_TOO_SMALL,
        min_area_px=120,
        max_area_fraction=0.5,
    )
    assert error.diagnostics["selected_component"]["area_px"] < 120


def test_bounded_large_component_reports_area_too_large():
    depth = np.full((120, 120), np.nan, dtype=np.float32)
    depth[20:80, 20:80] = 0.62
    error = _expect_reason(
        depth,
        (50, 50),
        MODULE.AREA_TOO_LARGE,
        roi_radius_px=55,
        max_area_fraction=0.10,
    )
    assert not error.diagnostics["selected_component"]["touches_roi_boundary"]


def test_surface_continuing_to_roi_boundary_reports_plane_merge():
    depth = np.full((120, 120), 0.62, dtype=np.float32)
    error = _expect_reason(
        depth,
        (60, 60),
        MODULE.PLANE_MERGE,
        roi_radius_px=35,
        max_area_fraction=0.10,
    )
    assert error.diagnostics["selected_component"]["touches_roi_boundary"]


def test_border_contact_can_fail_closed_with_border_truncated():
    depth = np.full((100, 120), np.nan, dtype=np.float32)
    depth[30:75, 0:35] = 0.62
    error = _expect_reason(
        depth,
        (18, 52),
        MODULE.BORDER_TRUNCATED,
        roi_radius_px=80,
        max_area_fraction=0.5,
        reject_border_truncated=True,
    )
    assert error.diagnostics["selected_component"]["touches_image_border"]


def test_out_of_image_seed_reports_no_seed_support():
    depth = np.full((80, 100), 0.62, dtype=np.float32)
    _expect_reason(depth, (-1, 40), MODULE.NO_SEED_SUPPORT)


def test_in_bounds_seed_without_metric_depth_reports_no_valid_depth():
    depth = np.full((80, 100), np.nan, dtype=np.float32)
    error = _expect_reason(depth, (50, 40), MODULE.NO_VALID_DEPTH)
    assert error.diagnostics["seed_points_xy_valid"] == [[50, 40]]
    assert error.diagnostics["valid_depth_pixels"] == 0


def test_valid_seeded_component_returns_diagnostics_and_accepts():
    depth = np.full((100, 120), np.nan, dtype=np.float32)
    depth[35:65, 45:75] = 0.62
    result = MODULE.segment_seeded_depth_component_v2(
        depth,
        [(60, 50)],
        min_area_px=120,
        max_area_fraction=0.5,
    )
    assert result["accepted"] is True
    assert result["reason_code"] == "DEPTH_COMPONENT_ACCEPTED"
    assert result["mask"][50, 60] == 255
    assert result["diagnostics"]["selected_component"]["seed_support_count"] == 1
