import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "fixed_pick_place.py"
SPEC = importlib.util.spec_from_file_location("fixed_pick_place", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dynamic_pose_generation_tracks_observed_object():
    poses = MODULE.generate_manipulation_pose_values(
        object_xyz=[0.31, -0.17, 0.03],
        object_yaw=0.0,
        grasp_offset_xyz=[0.0, 0.0, 0.0],
        approach_distance=0.17,
        lift_distance=0.17,
        bin_object_pose=[0.32, 0.22, 0.04, 0.0],
        place_offset_xyz=[0.0, 0.0, 0.01],
        place_approach_distance=0.16,
    )
    assert set(poses) == {
        "object_pose", "grasp_pose", "pregrasp_pose", "lift_pose",
        "preplace_pose", "place_pose", "retreat_pose",
    }
    assert np.allclose(poses["grasp_pose"][:3], [0.31, -0.17, 0.03])
    assert np.allclose(poses["pregrasp_pose"][:3], [0.31, -0.17, 0.20])
    assert np.allclose(poses["place_pose"][:3], [0.32, 0.22, 0.05])
    assert poses["retreat_pose"] == poses["preplace_pose"]


def test_yaw_changes_grasp_orientation_without_changing_place():
    poses = MODULE.generate_manipulation_pose_values(
        [0.32, -0.18, 0.03], 0.2, [0.0, 0.0, 0.0], 0.17, 0.17,
        [0.32, 0.22, 0.03, 0.0], [0.0, 0.0, 0.01], 0.16,
    )
    assert not np.allclose(poses["grasp_pose"][3:], poses["place_pose"][3:])


def test_industrial_profile_keeps_one_top_down_orientation_for_every_waypoint():
    poses = MODULE.generate_manipulation_pose_values(
        [0.32, -0.18, 0.03], 0.7, [0.0, 0.0, 0.0], 0.17, 0.17,
        [0.32, 0.22, 0.04, 0.4], [0.0, 0.0, 0.01], 0.16,
        tool_yaw=0.0,
        place_tool_yaw=0.0,
    )
    expected = [1.0, 0.0, 0.0, 0.0]
    for name in (
        "grasp_pose", "pregrasp_pose", "lift_pose", "preplace_pose",
        "place_pose", "retreat_pose",
    ):
        assert np.allclose(poses[name][3:], expected), name


def test_bin_slot_allocator_prevents_aabb_overlap_and_stays_inside_tray():
    occupied = []
    slots = [
        [-0.055, -0.075], [0.0, -0.075], [0.055, -0.075],
        [-0.055, 0.0], [0.0, 0.0], [0.055, 0.0],
        [-0.055, 0.075], [0.0, 0.075], [0.055, 0.075],
    ]
    for _ in range(7):
        allocation = MODULE.select_non_overlapping_bin_slot(
            [0.0, 0.38], [0.18, 0.23], slots, [0.05, 0.05], occupied
        )
        assert allocation is not None
        assert all(
            not MODULE.xy_aabbs_overlap(allocation["aabb_xyxy"], item["aabb_xyxy"])
            for item in occupied
        )
        occupied.append(allocation)
    assert len({item["slot_id"] for item in occupied}) == 7


def test_bin_slot_allocator_reports_full_instead_of_stacking():
    slots = [[0.0, 0.0]]
    first = MODULE.select_non_overlapping_bin_slot(
        [0.0, 0.38], [0.18, 0.23], slots, [0.06, 0.06], []
    )
    second = MODULE.select_non_overlapping_bin_slot(
        [0.0, 0.38], [0.18, 0.23], slots, [0.06, 0.06], [first]
    )
    assert first is not None
    assert second is None


def test_random_source_keepout_rejects_tray_and_accepts_safe_workspace():
    tray_center = [0.0, 0.38]
    tray_size = [0.20, 0.25]
    assert MODULE.source_pose_overlaps_keepout(
        [0.0, 0.38], [0.06, 0.06], tray_center, tray_size, 0.005
    )
    assert not MODULE.source_pose_overlaps_keepout(
        [-0.13, 0.18], [0.06, 0.06], tray_center, tray_size, 0.005
    )
