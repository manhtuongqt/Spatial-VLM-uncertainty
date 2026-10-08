import numpy as np

from ur3_spatial_dataset.spatial_relations import (
    compute_relations,
    is_inside,
    project_point,
    surface_distance,
    transform_point,
)


def _object(object_id, position, dimensions=(0.05, 0.05, 0.05), **extra):
    value = {
        "id": object_id,
        "pose": {"position": list(position)},
        "dimensions": list(dimensions),
    }
    value.update(extra)
    return value


def test_transform_and_projection_follow_optical_convention():
    point = transform_point(
        [0.1, -0.2, 0.5],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    assert np.allclose(point, [0.1, -0.2, 0.5])
    assert project_point(point, {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0}) == (
        70.0, 0.0, 0.5
    )


def test_image_and_depth_relations_are_directed_and_symmetric():
    left = _object("left", [0.0, 0.0, 0.0])
    right = _object("right", [1.0, 0.0, 0.0])
    relations = compute_relations(
        [left, right],
        {"left": (100.0, 120.0, 0.40), "right": (140.0, 160.0, 0.55)},
        near_threshold_m=0.01,
    )
    triples = {(r["subject_id"], r["predicate"], r["object_id"]) for r in relations}
    assert ("left", "left_of", "right") in triples
    assert ("right", "right_of", "left") in triples
    assert ("left", "above", "right") in triples
    assert ("right", "below", "left") in triples
    assert ("left", "front_of", "right") in triples
    assert ("right", "behind", "left") in triples


def test_metric_near_and_inside_use_full_aabb():
    cube = _object("cube", [0.0, 0.0, 0.03], (0.06, 0.06, 0.06))
    bin_object = _object(
        "bin", [0.0, 0.0, 0.035], (0.20, 0.25, 0.07),
        container_inner_dimensions=[0.18, 0.23, 0.07],
    )
    assert is_inside(cube, bin_object)
    assert surface_distance(cube, bin_object) == 0.0
    triples = {
        (r["subject_id"], r["predicate"], r["object_id"])
        for r in compute_relations([cube, bin_object], near_threshold_m=0.12)
    }
    assert ("cube", "inside", "bin") in triples
    assert ("cube", "near", "bin") in triples


def test_pixel_deadband_prevents_unstable_relation_flip():
    first = _object("first", [0.0, 0.0, 0.0])
    second = _object("second", [1.0, 0.0, 0.0])
    relations = compute_relations(
        [first, second],
        {"first": (100.0, 100.0, 0.5), "second": (105.0, 104.0, 0.506)},
        pixel_margin=8.0,
        depth_margin_m=0.015,
        near_threshold_m=0.01,
    )
    assert relations == []
