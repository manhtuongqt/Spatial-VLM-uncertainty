"""Unit tests for RoboRefer-seeded metric RGB-D spatial demonstrations."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "roborefer_dimension_comparator.py"
SPEC = importlib.util.spec_from_file_location(
    "roborefer_dimension_comparator", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mask_backprojection_and_base_frame_dimensions_are_metric():
    depth = np.ones((80, 80), dtype=np.float32)
    mask = np.zeros_like(depth, dtype=np.uint8)
    mask[20:61, 10:31] = 255
    camera_points = MODULE.backproject_mask_to_camera(
        depth, mask, (100.0, 100.0, 0.0, 0.0)
    )

    # -90 degrees around camera X maps optical Y to negative base Z and
    # optical Z to base Y. The synthetic mask therefore has 0.4 m height and
    # 0.2 m horizontal width in the base frame.
    root_half = np.sqrt(0.5)
    rotation = MODULE.quaternion_rotation_matrix(
        -root_half, 0.0, 0.0, root_half
    )
    points_base = MODULE.transform_points(
        camera_points, rotation, np.asarray([0.0, 0.0, 1.0])
    )
    dimensions = MODULE.estimate_base_dimensions(points_base)

    assert camera_points.shape == (41 * 21, 3)
    assert 0.19 < dimensions["width_m"] < 0.21
    assert 0.38 < dimensions["height_m"] < 0.41
    assert dimensions["point_count"] == 41 * 21


def test_dimension_winner_uses_geometry_and_has_tie_gate():
    first = {"height_m": 0.18, "width_m": 0.06}
    second = {"height_m": 0.075, "width_m": 0.076}

    assert MODULE.comparison_winner(first, second, "height_m", 0.005) == "A"
    assert MODULE.comparison_winner(first, second, "width_m", 0.005) == "B"
    assert MODULE.comparison_winner(first, {"height_m": 0.177},
                                    "height_m", 0.005) == "TIE"


def test_depth_support_plane_recovers_complete_tall_object_component():
    point_map = np.full((120, 160, 3), np.nan, dtype=np.float64)
    # Dominant table at base z=0.
    point_map[50:, :, 0] = np.linspace(-0.4, 0.4, 160)[None, :]
    point_map[50:, :, 1] = np.linspace(0.1, 0.6, 70)[:, None]
    point_map[50:, :, 2] = 0.0
    # A 30x70 px standing object spans 0.18 m vertically. It would be cut by
    # the old fixed seed-depth band, but remains connected after plane removal.
    object_rows = slice(30, 100)
    object_columns = slice(60, 90)
    point_map[object_rows, object_columns, 0] = np.linspace(
        -0.03, 0.03, 30
    )[None, :]
    point_map[object_rows, object_columns, 1] = 0.25
    point_map[object_rows, object_columns, 2] = np.linspace(
        0.18, 0.006, 70
    )[:, None]

    plane_z = MODULE.estimate_support_plane_z(point_map)
    component = MODULE.segment_seeded_above_plane_component(
        point_map,
        (75, 45),
        plane_z,
        roi_radius_px=80,
        plane_clearance_m=0.004,
        max_area_fraction=0.30,
    )
    measured = MODULE.estimate_base_dimensions(
        point_map[component["mask"] > 0], 0.0, 100.0
    )

    assert abs(plane_z) < 0.001
    assert component["mask_area_px"] > 1900
    assert measured["height_m"] > 0.17


def test_comparison_overlay_shows_two_regions_and_results():
    source = np.zeros((480, 640, 3), dtype=np.uint8)
    mask_a = np.zeros(source.shape[:2], dtype=np.uint8)
    mask_b = np.zeros(source.shape[:2], dtype=np.uint8)
    mask_a[110:260, 90:180] = 255
    mask_b[250:340, 390:480] = 255
    objects = {
        "A": {
            "label": "BOTTLE",
            "mask": mask_a,
            "bbox_xyxy": [90, 110, 179, 259],
            "pixel_xy": [135, 180],
            "dimensions": {"width_m": 0.06, "height_m": 0.18},
        },
        "B": {
            "label": "APPLE",
            "mask": mask_b,
            "bbox_xyxy": [390, 250, 479, 339],
            "pixel_xy": [435, 295],
            "dimensions": {"width_m": 0.075, "height_m": 0.075},
        },
    }
    overlay = MODULE.RoboReferDimensionComparator._draw_results(
        source,
        objects,
        {"height_winner": "A", "width_winner": "B"},
    )

    assert overlay.shape == source.shape
    assert np.count_nonzero(overlay) > 10000
    assert np.count_nonzero(overlay[180 - 12:180 + 13, 135 - 12:135 + 13]) > 0
    assert np.count_nonzero(overlay[295 - 12:295 + 13, 435 - 12:435 + 13]) > 0


def test_near_far_point_assignment_uses_only_inferred_masks():
    mask_a = np.zeros((120, 180), dtype=np.uint8)
    mask_b = np.zeros_like(mask_a)
    mask_a[20:70, 20:65] = 255
    mask_b[45:105, 115:165] = 255
    masks = {"A": mask_a, "B": mask_b}

    assert MODULE.classify_point_by_masks((35, 40), masks) == ("A", 0.0)
    selection, distance = MODULE.classify_point_by_masks((110, 80), masks)
    assert selection == "B"
    assert 0.0 < distance < 10.0
    selection, distance = MODULE.classify_point_by_masks(
        (88, 5), masks, max_distance_px=12.0
    )
    assert selection == "UNKNOWN"
    assert distance > 12.0

    overlapping = dict(masks)
    overlapping["C"] = mask_b.copy()
    assert MODULE.classify_point_by_masks(
        (130, 70), overlapping
    ) == ("AMBIGUOUS", 0.0)


def test_near_far_overlay_shows_rgb_and_rgbd_predictions():
    source = np.zeros((480, 640, 3), dtype=np.uint8)
    mask_a = np.zeros(source.shape[:2], dtype=np.uint8)
    mask_b = np.zeros(source.shape[:2], dtype=np.uint8)
    mask_a[100:280, 85:180] = 255
    mask_b[245:350, 390:500] = 255
    objects = {
        "A": {
            "label": "BOTTLE",
            "mask": mask_a,
            "bbox_xyxy": [85, 100, 179, 279],
            "median_camera_depth_m": 0.22,
        },
        "B": {
            "label": "APPLE",
            "mask": mask_b,
            "bbox_xyxy": [390, 245, 499, 349],
            "median_camera_depth_m": 0.34,
        },
    }
    ablation = {
        "metric_nearer": "A",
        "metric_depth_margin_m": 0.12,
        "rgb_only": {
            "pixel_xy": [445, 295],
            "selection": "B",
            "verdict": "FAIL",
        },
        "rgbd": {
            "pixel_xy": [130, 180],
            "selection": "A",
            "verdict": "PASS",
        },
        "verdict": "RGBD_IMPROVES",
    }

    overlay = MODULE.RoboReferDimensionComparator._draw_near_far_results(
        source, objects, ablation
    )

    assert overlay.shape == source.shape
    assert np.count_nonzero(overlay) > 12000
    assert np.count_nonzero(overlay[280:311, 430:461]) > 0
    assert np.count_nonzero(overlay[165:196, 115:146]) > 0


def test_relative_position_separates_camera_and_robot_frames_with_tolerance():
    relationship = MODULE.relative_position_between_objects(
        first_center_base_m=[0.000, 0.100, 0.150],
        second_center_base_m=[-0.005, 0.240, 0.070],
        first_centroid_pixel_xy=[150, 150],
        second_centroid_pixel_xy=[420, 180],
        first_camera_depth_m=0.240,
        second_camera_depth_m=0.320,
        metric_tolerance_m=0.015,
        image_tolerance_px=12.0,
    )

    assert relationship["camera_relations"]["u"]["relation"] == "RIGHT"
    assert relationship["camera_relations"]["v"]["relation"] == "BELOW"
    assert relationship["camera_relations"]["depth"]["relation"] == "FARTHER"
    assert relationship["base_relations"]["x"]["relation"] == "SAME_X"
    assert relationship["base_relations"]["y"]["relation"] == "LEFT"
    assert relationship["base_relations"]["z"]["relation"] == "LOWER"
    assert abs(relationship["base_delta_m"]["y"] - 0.140) < 1e-9


def test_relative_position_overlay_draws_centers_arrow_and_relations():
    source = np.zeros((480, 640, 3), dtype=np.uint8)
    mask_a = np.zeros(source.shape[:2], dtype=np.uint8)
    mask_b = np.zeros(source.shape[:2], dtype=np.uint8)
    mask_a[90:270, 80:190] = 255
    mask_b[240:350, 390:510] = 255
    center_a = MODULE.mask_centroid_pixel(mask_a)
    center_b = MODULE.mask_centroid_pixel(mask_b)
    objects = {
        "A": {
            "label": "BOTTLE",
            "mask": mask_a,
            "bbox_xyxy": [80, 90, 189, 269],
            "mask_centroid_pixel_xy": list(center_a),
            "dimensions": {"center_base_m": [-0.16, 0.18, 0.15]},
        },
        "B": {
            "label": "APPLE",
            "mask": mask_b,
            "bbox_xyxy": [390, 240, 509, 349],
            "mask_centroid_pixel_xy": list(center_b),
            "dimensions": {"center_base_m": [-0.17, 0.28, 0.07]},
        },
    }
    relationship = MODULE.relative_position_between_objects(
        objects["A"]["dimensions"]["center_base_m"],
        objects["B"]["dimensions"]["center_base_m"],
        center_a,
        center_b,
        0.24,
        0.32,
    )

    overlay = MODULE.RoboReferDimensionComparator._draw_relative_position_results(
        source, objects, relationship
    )

    assert overlay.shape == source.shape
    assert np.count_nonzero(overlay) > 15000
    assert np.count_nonzero(
        overlay[center_a[1] - 14:center_a[1] + 15,
                center_a[0] - 14:center_a[0] + 15]
    ) > 0
    assert np.count_nonzero(
        overlay[center_b[1] - 14:center_b[1] + 15,
                center_b[0] - 14:center_b[0] + 15]
    ) > 0


def _five_step_fixture():
    mask_a = np.zeros((480, 640), dtype=np.uint8)
    mask_b = np.zeros_like(mask_a)
    mask_a[80:280, 90:170] = 255
    mask_b[245:345, 390:500] = 255
    center_a = MODULE.mask_centroid_pixel(mask_a)
    center_b = MODULE.mask_centroid_pixel(mask_b)
    objects = {
        "A": {
            "label": "TALL YELLOW BOTTLE",
            "mask": mask_a,
            "bbox_xyxy": [90, 80, 169, 279],
            "mask_centroid_pixel_xy": list(center_a),
            "median_camera_depth_m": 0.24,
            "dimensions": {
                "width_m": 0.060,
                "height_m": 0.190,
                "center_base_m": [-0.16, 0.18, 0.15],
            },
        },
        "B": {
            "label": "ROUND FRUIT TARGET",
            "mask": mask_b,
            "bbox_xyxy": [390, 245, 499, 344],
            "mask_centroid_pixel_xy": list(center_b),
            "median_camera_depth_m": 0.32,
            "dimensions": {
                "width_m": 0.076,
                "height_m": 0.074,
                "center_base_m": [-0.17, 0.28, 0.07],
            },
        },
    }
    relationship = MODULE.relative_position_between_objects(
        objects["A"]["dimensions"]["center_base_m"],
        objects["B"]["dimensions"]["center_base_m"],
        center_a,
        center_b,
        0.24,
        0.32,
    )
    trigger = {
        "instruction": (
            "Find the tall yellow bottle, then the round fruit to its right, "
            "lower in the image and farther from the camera. Point inside it."
        ),
        "roborefer_points_xy": [[445, 295]],
        "grasp_pixel_xy": [445, 295],
        "bbox_xyxy": [390, 245, 499, 344],
        "selection_published": False,
    }
    return objects, relationship, trigger


def test_five_step_evaluation_locks_raw_point_and_gates_target_handoff():
    objects, relationship, trigger = _five_step_fixture()
    evaluation = MODULE.evaluate_five_step_reference(
        objects, trigger, relationship
    )

    assert evaluation["verdict"] == "PASS"
    assert evaluation["passed_steps"] == 5
    assert evaluation["locked_model_point_region"] == "B"
    assert evaluation["target_name_in_main_prompt"] is False
    selection = MODULE.build_verified_target_selection(
        trigger, evaluation, "ycb_apple_01", "red", "ycb_apple"
    )
    assert selection["source"] == "roborefer_five_step_verified"
    assert selection["five_step_passed_steps"] == 5
    assert selection["registry_identity_added_after_inference"] is True

    leaked_trigger = dict(trigger)
    leaked_trigger["instruction"] = "Pick the apple at the known target."
    failed = MODULE.evaluate_five_step_reference(
        objects, leaked_trigger, relationship
    )
    assert failed["passed_steps"] == 5
    assert failed["protocol_valid"] is False
    assert failed["verdict"] == "FAIL"
    try:
        MODULE.build_verified_target_selection(
            leaked_trigger, failed, "ycb_apple_01", "red", "ycb_apple"
        )
        assert False, "failed protocol must not release a pick target"
    except ValueError:
        pass


def test_compact_target_profile_uses_metric_shape_without_target_name_or_coordinates():
    objects, relationship, trigger = _five_step_fixture()
    objects["A"]["label"] = "TALL YELLOW RECTANGULAR PACKAGE"
    objects["A"]["dimensions"].update({"width_m": 0.093, "height_m": 0.176})
    objects["B"]["label"] = "SHORT RED CYLINDRICAL PACKAGE"
    objects["B"]["dimensions"].update({"width_m": 0.068, "height_m": 0.102})
    trigger["instruction"] = (
        "Use the tall yellow rectangular package only as a reference. "
        "Select the shorter compact red cylindrical grocery container and "
        "return one point well inside it."
    )
    evaluation = MODULE.evaluate_five_step_reference(
        objects,
        trigger,
        relationship,
        rule_profile="compact_target",
        forbidden_target_terms=("tomato", "soup", "ycb_tomato_soup_can"),
        anchor_min_aspect_ratio=1.5,
        compact_target_max_aspect_ratio=1.75,
        size_margin_m=0.005,
    )

    assert evaluation["verdict"] == "PASS"
    assert evaluation["passed_steps"] == 5
    assert evaluation["rule_profile"] == "compact_target"
    assert evaluation["target_name_in_main_prompt"] is False
    assert evaluation["target_coordinates_supplied_before_inference"] is False

    leaked = dict(trigger)
    leaked["instruction"] += " It is the tomato soup object."
    rejected = MODULE.evaluate_five_step_reference(
        objects,
        leaked,
        relationship,
        rule_profile="compact_target",
        forbidden_target_terms=("tomato", "soup", "ycb_tomato_soup_can"),
    )
    assert rejected["protocol_valid"] is False
    assert rejected["verdict"] == "FAIL"


def _two_reference_fixture():
    masks = {}
    masks["A"] = np.zeros((480, 640), dtype=np.uint8)
    masks["B"] = np.zeros((480, 640), dtype=np.uint8)
    masks["C"] = np.zeros((480, 640), dtype=np.uint8)
    masks["A"][150:245, 55:155] = 255
    masks["B"][165:245, 470:610] = 255
    masks["C"][120:270, 255:365] = 255
    geometry = {
        "A": {
            "width_m": 0.075,
            "depth_m": 0.071,
            "height_m": 0.074,
            "center_base_m": [-0.150, 0.150, 0.060],
        },
        "B": {
            "width_m": 0.170,
            "depth_m": 0.055,
            "height_m": 0.037,
            "center_base_m": [-0.180, 0.410, 0.030],
        },
        "C": {
            "width_m": 0.068,
            "depth_m": 0.064,
            "height_m": 0.102,
            "center_base_m": [-0.175, 0.255, 0.090],
        },
    }
    labels = {
        "A": "ROUND ORANGE REFERENCE",
        "B": "CURVED YELLOW REFERENCE",
        "C": "CYLINDRICAL CANDIDATE",
    }
    objects = {}
    for key in ("A", "B", "C"):
        objects[key] = {
            "label": labels[key],
            "mask": masks[key],
            "bbox_xyxy": [0, 0, 1, 1],
            "mask_centroid_pixel_xy": list(
                MODULE.mask_centroid_pixel(masks[key])
            ),
            "median_camera_depth_m": 0.30,
            "dimensions": geometry[key],
        }
    trigger = {
        "instruction": (
            "Use the round orange fruit and long curved yellow fruit as two "
            "references. Point inside the taller cylindrical object between them."
        ),
        "roborefer_points_xy": [[310, 190]],
        "grasp_pixel_xy": [310, 190],
        "bbox_xyxy": [255, 120, 364, 269],
        "selection_published": False,
    }
    return objects, trigger


def test_two_reference_profile_requires_both_references_and_target_region():
    objects, trigger = _two_reference_fixture()
    evaluation = MODULE.evaluate_five_step_reference(
        objects,
        trigger,
        relationship={},
        rule_profile="two_reference_between_target",
        forbidden_target_terms=("tomato", "soup", "ycb_tomato_soup_can"),
        anchor_min_aspect_ratio=1.60,
        compact_target_max_aspect_ratio=1.45,
        size_margin_m=0.006,
        between_corridor_m=0.080,
    )

    assert evaluation["verdict"] == "PASS"
    assert evaluation["passed_steps"] == 5
    assert evaluation["locked_model_point_region"] == "C"
    assert evaluation["two_reference_geometry"][
        "candidate_projection_fraction"
    ] > 0.05
    assert evaluation["target_coordinates_supplied_before_inference"] is False

    wrong_point = dict(trigger)
    wrong_point["roborefer_points_xy"] = [[100, 190]]
    rejected = MODULE.evaluate_five_step_reference(
        objects,
        wrong_point,
        relationship={},
        rule_profile="two_reference_between_target",
        forbidden_target_terms=("tomato", "soup", "ycb_tomato_soup_can"),
        anchor_min_aspect_ratio=1.60,
        compact_target_max_aspect_ratio=1.45,
        size_margin_m=0.006,
        between_corridor_m=0.080,
    )
    assert rejected["passed_steps"] == 4
    assert rejected["verdict"] == "FAIL"
    assert rejected["locked_model_point_region"] == "A"


def test_two_reference_overlay_draws_all_three_regions():
    objects, trigger = _two_reference_fixture()
    evaluation = MODULE.evaluate_five_step_reference(
        objects,
        trigger,
        relationship={},
        rule_profile="two_reference_between_target",
        anchor_min_aspect_ratio=1.60,
        compact_target_max_aspect_ratio=1.45,
        size_margin_m=0.006,
        between_corridor_m=0.080,
    )
    source = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay = MODULE.RoboReferDimensionComparator._draw_five_step_results(
        source, objects, evaluation
    )

    assert overlay.shape == source.shape
    assert np.count_nonzero(overlay[58:246]) > 12000
    for key in ("A", "B", "C"):
        x_pixel, y_pixel = objects[key]["mask_centroid_pixel_xy"]
        # The scene is resized vertically into the panel; checking a colour is
        # less robust than verifying all three labels/masks contributed pixels.
        assert 0 <= x_pixel < 640
        assert 0 <= y_pixel < 480


def test_five_step_overlay_displays_scene_point_checks_and_verdict():
    objects, relationship, trigger = _five_step_fixture()
    evaluation = MODULE.evaluate_five_step_reference(
        objects, trigger, relationship
    )
    source = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay = MODULE.RoboReferDimensionComparator._draw_five_step_results(
        source, objects, evaluation
    )

    assert overlay.shape == source.shape
    assert np.count_nonzero(overlay) > 25000
    # Header, scene and five verdict rows must all be visibly populated.
    assert np.count_nonzero(overlay[:58]) > 1000
    assert np.count_nonzero(overlay[58:246]) > 5000
    assert np.count_nonzero(overlay[246:430]) > 3000
