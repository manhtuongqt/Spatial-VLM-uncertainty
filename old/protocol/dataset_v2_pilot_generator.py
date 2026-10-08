#!/usr/bin/env python3
"""Materialize the locked 30-family Dataset V2 pilot manifest and capture plan.

This generator is intentionally independent of WP2. It consumes the V2 design
lock and emits only pilot-only assignments; it performs no Gazebo capture.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_dataset_expansion_v2 import (  # noqa: E402
    VARIANTS,
    assignment_commitment,
    build_development_assignments,
    build_pilot_assignments,
)
from wp2_common import canonical_json_sha256, read_json, sha256_file, write_json  # noqa: E402


PROTOCOL_ID = "roborefer_dataset_v2_pilot_only"
GENERATOR_VERSION = "dataset_v2_pilot_generator_v1.2.0"
EXPECTED_PILOT_COMMITMENT = "45f0ec073d1b71d3e4a07336262c98e5ef875ac5befd2a585b51bdb16b3cb22c"
IID_VIEW = [1.3815, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601]
# Azimuth-shifted wrist view.  Keep wrist_1 at the IID pitch: the earlier
# simultaneous shoulder/wrist change made optical depth almost perfectly
# vertical, so tabletop x/y no longer provided enough metric-depth separation
# to realize front/behind relations across objects of different heights.
OOD_VIEW = [1.1815, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601]
COORDINATE_SUFFIX = "Output the target pixel as (x, y) in image coordinates."


class PilotGenerationError(RuntimeError):
    """Raised when a locked pilot artifact cannot be generated exactly."""


def obj(
    object_id: str,
    semantic_class: str,
    label: int,
    z: float,
    radius: float,
    storage: tuple[float, float, float],
    partition: str = "seen",
) -> dict[str, Any]:
    return {
        "id": object_id,
        "semantic_class": semantic_class,
        "label": label,
        "z": z,
        "footprint_radius_m": radius,
        "storage_pose_xyyaw": list(storage),
        "asset_partition": partition,
    }


def object_registry() -> dict[str, dict[str, Any]]:
    """Gazebo model name -> evaluator identity and conservative footprint."""

    return {
        "red_cube": obj("cube_red_01", "red cube", 1, .0300, .043, (-.72, .84, 0)),
        "blue_cube": obj("cube_blue_01", "blue cube", 2, .0250, .037, (-.54, .84, 0)),
        "green_cube": obj("cube_green_01", "green cube", 3, .0250, .037, (-.36, .84, 0)),
        "yellow_cube": obj("cube_yellow_01", "yellow cube", 4, .0250, .037, (-.18, .84, 0)),
        "orange_cube": obj("cube_orange_01", "orange cube", 5, .0250, .037, (0, .84, 0)),
        "purple_cube": obj("cube_purple_01", "purple cube", 6, .0250, .037, (.18, .84, 0)),
        "pink_cube": obj("cube_pink_01", "pink cube", 7, .0250, .037, (.36, .84, 0)),
        "ycb_cracker_box": obj("ycb_cracker_box_01", "cracker box", 21, .106710, .091, (.54, .84, 0)),
        "ycb_sugar_box": obj("ycb_sugar_box_01", "sugar box", 22, .088107, .057, (.72, .84, 0)),
        "ycb_tomato_soup_can": obj("ycb_tomato_soup_can_01", "tomato soup can", 23, .050948, .038, (-.72, 1.08, 0)),
        "ycb_mustard_bottle": obj("ycb_mustard_bottle_01", "mustard bottle", 24, .095695, .035, (-.54, 1.08, 0)),
        "ycb_banana": obj("ycb_banana_01", "banana", 25, .018376, .106, (-.36, 1.08, .25)),
        "ycb_apple": obj("ycb_apple_01", "apple", 26, .037710, .042, (-.18, 1.08, 0)),
        "ycb_orange": obj("ycb_orange_01", "orange", 27, .037010, .042, (0, 1.08, 0)),
        "ycb_power_drill": obj("ycb_power_drill_01", "power drill", 28, .028689, .132, (.18, 1.08, -.25)),
        "mango": obj("mango_01", "mango", 29, .0300, .055, (.36, 1.08, .35)),
        "ycb_banana_02": obj("ycb_banana_02", "banana", 101, .018876, .106, (.54, 1.08, -.45)),
        "ycb_apple_02": obj("ycb_apple_02", "apple", 102, .036453, .042, (.72, 1.08, .2)),
        "ycb_orange_02": obj("ycb_orange_02", "orange", 103, .036184, .042, (-.72, 1.32, -.2)),
        "ycb_lemon": obj("ycb_lemon_01", "lemon", 104, .026999, .041, (-.54, 1.32, .35)),
        "ycb_pear": obj("ycb_pear_01", "pear", 105, .033322, .048, (-.36, 1.32, -.3), "test_ood_heldout"),
        "ycb_banana_03": obj("ycb_banana_03", "banana", 106, .018876, .106, (-.18, 1.32, .4)),
        "ycb_apple_03": obj("ycb_apple_03", "apple", 107, .036453, .042, (0, 1.32, -.35)),
        "ycb_orange_03": obj("ycb_orange_03", "orange", 108, .036184, .042, (.18, 1.32, .4)),
        "ycb_plum": obj("ycb_plum_01", "plum", 109, .027025, .039, (.36, 1.32, .55), "test_ood_heldout"),
        "ycb_tuna_fish_can": obj("ycb_tuna_fish_can_01", "tuna fish can", 110, .017264, .046, (.54, 1.32, .1), "test_ood_heldout"),
        "ycb_mug": obj("ycb_mug_01", "mug", 111, .041177, .064, (.72, 1.32, -.8)),
    }


def ranked(values: list[str], seed: int, namespace: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{seed}|{namespace}|{value}".encode()).hexdigest(),
            value,
        ),
    )


def submode_map(assignments: list[dict[str, Any]], master_seed: int) -> dict[str, str]:
    plans = {
        "AMBIGUOUS": [
            "same_class_duplicate", "same_class_duplicate",
            "attribute_tie", "attribute_tie",
            "relation_tie", "relation_tie",
            "multi_anchor_conflict",
        ],
        "ABSENT": [
            "target_absent", "target_absent", "target_absent",
            "anchor_absent", "anchor_absent",
            "unsatisfied_relation", "unsatisfied_relation",
        ],
        "INSUFFICIENT_EVIDENCE": [
            "depth_invalid_or_corrupt", "depth_invalid_or_corrupt",
            "occlusion", "occlusion",
            "too_small_or_out_of_view", "too_small_or_out_of_view",
            "cross_modal_conflict", "cross_modal_conflict",
        ],
    }
    result: dict[str, str] = {}
    for state, labels in plans.items():
        ids = [a["family_id"] for a in assignments if a["primary_answerability_stratum"] == state]
        ids = ranked(ids, master_seed, f"pilot_submode_{state}")
        if len(ids) != len(labels):
            raise PilotGenerationError(f"unexpected {state} count: {len(ids)}")
        result.update(dict(zip(ids, labels, strict=True)))
    for assignment in assignments:
        if assignment["primary_answerability_stratum"] == "FOUND":
            result[assignment["family_id"]] = "unique_visible_referent"
    return result


def model_for_id(registry: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {value["id"]: name for name, value in registry.items()}


SEEN_TARGETS = [
    "ycb_apple_01", "ycb_orange_01", "ycb_lemon_01", "ycb_mug_01",
    "cube_orange_01", "cube_blue_01", "ycb_tomato_soup_can_01",
    "ycb_mustard_bottle_01", "mango_01",
]
HELDOUT_TARGETS = ["ycb_pear_01", "ycb_plum_01", "ycb_tuna_fish_can_01"]
SEEN_ANCHORS = [
    "cube_yellow_01", "cube_green_01", "cube_purple_01",
    "ycb_orange_01", "ycb_apple_01", "ycb_lemon_01",
]
SEMANTIC_CF = ["cube_pink_01", "ycb_mustard_bottle_01", "ycb_tomato_soup_can_01"]


def pick_unique(pool: list[str], used: set[str], start: int) -> str:
    for offset in range(len(pool)):
        value = pool[(start + offset) % len(pool)]
        if value not in used:
            used.add(value)
            return value
    raise PilotGenerationError("object pool exhausted")


def inverse_relation(relation: str) -> str:
    return {
        "right_of": "left_of", "left_of": "right_of",
        "nearer_than": "farther_than", "farther_than": "nearer_than",
        "front_of": "behind", "behind": "front_of",
        "nearer_than_both": "farther_than_both",
        "between_in_depth": "between_in_depth",
        "direct": "direct",
    }[relation]


def relation_for(category: str, index: int) -> tuple[str, str]:
    if category == "direct_grounding":
        return "direct", "object_semantics"
    if category == "relation_2d":
        return ("right_of" if index % 2 == 0 else "left_of"), "image"
    if category == "nearer_farther":
        return ("nearer_than" if index % 2 == 0 else "farther_than"), "camera_color_optical_frame"
    if category == "front_behind_camera":
        return ("front_of" if index % 2 == 0 else "behind"), "camera_color_optical_frame"
    if category == "multi_anchor_depth_order":
        return ("between_in_depth" if index % 2 == 0 else "nearer_than_both"), "camera_color_optical_frame"
    if category == "occlusion_depth_evidence":
        return "front_of", "camera_color_optical_frame"
    raise PilotGenerationError(f"unknown category: {category}")


def instruction_for(
    semantic: str,
    relation: str,
    anchors: list[str],
    registry_by_id: dict[str, dict[str, Any]],
    language_bank: str,
) -> str:
    anchor_names = [registry_by_id[value]["semantic_class"] for value in anchors]
    if relation == "direct":
        standard = f"Locate the {semantic}."
        heldout = f"Which image location corresponds to the {semantic}?"
    elif relation in {"right_of", "left_of", "nearer_than", "farther_than", "front_of", "behind"}:
        phrase = relation.replace("_", " ")
        standard = f"Locate the {semantic} that is {phrase} the {anchor_names[0]}."
        heldout = f"Point out the {semantic}; it lies {phrase} relative to the {anchor_names[0]}."
    elif relation == "between_in_depth":
        standard = f"Locate the {semantic} between the {anchor_names[0]} and {anchor_names[1]} in camera depth."
        heldout = f"Which {semantic} occupies the depth interval bounded by the {anchor_names[0]} and {anchor_names[1]}?"
    elif relation == "nearer_than_both":
        standard = f"Locate the {semantic} nearer to the camera than both the {anchor_names[0]} and {anchor_names[1]}."
        heldout = f"Point out the {semantic} preceding both the {anchor_names[0]} and {anchor_names[1]} along the viewing ray."
    else:
        raise PilotGenerationError(f"no instruction template for {relation}")
    return heldout if language_bank == "language_v2_ood_heldout_templates" else standard


def state_fields(state: str, submode: str) -> dict[str, Any]:
    if state == "FOUND":
        return {"answerable": True, "sources": [], "severity": 0, "expected_intervention": "EXECUTE"}
    if state == "AMBIGUOUS":
        return {"answerable": False, "sources": ["semantic", "relation"], "severity": 2, "expected_intervention": "ASK_USER"}
    if state == "ABSENT":
        return {"answerable": False, "sources": ["semantic", "relation"], "severity": 3, "expected_intervention": "ABSTAIN"}
    source = "depth" if submode in {"depth_invalid_or_corrupt", "cross_modal_conflict"} else "occlusion"
    return {"answerable": False, "sources": [source], "severity": 2, "expected_intervention": "REOBSERVE"}


def pose_is_collision_free(
    layout: dict[str, list[float]], model_name: str, pose: tuple[float, float, float],
    registry: dict[str, dict[str, Any]], margin_m: float = .004,
) -> bool:
    radius = float(registry[model_name]["footprint_radius_m"])
    for other, other_pose in layout.items():
        if other == model_name or float(other_pose[1]) > .65:
            continue
        required = radius + float(registry[other]["footprint_radius_m"]) + margin_m
        if math.hypot(float(pose[0]) - float(other_pose[0]), float(pose[1]) - float(other_pose[1])) < required:
            return False
    return True


def place_auxiliary(
    layout: dict[str, list[float]], model_name: str,
    registry: dict[str, dict[str, Any]], candidates: list[tuple[float, float, float]],
) -> None:
    for pose in candidates:
        if pose_is_collision_free(layout, model_name, pose, registry):
            layout[model_name] = [round(value, 6) for value in pose]
            return
    raise PilotGenerationError(f"no collision-free auxiliary slot for {model_name}")


def place_safe_occluder(
    layout: dict[str, list[float]], target_model: str, occluder_model: str,
    registry: dict[str, dict[str, Any]],
) -> None:
    tx, ty, _ = layout[target_model]
    required = (
        float(registry[target_model]["footprint_radius_m"])
        + float(registry[occluder_model]["footprint_radius_m"])
        + .007
    )
    # Candidates remain closer to the camera (smaller base-y) but avoid physical
    # overlap. Oblique candidates preserve partial projected occlusion near the
    # front workspace boundary.
    vectors = [
        (0, -required),
        (.55 * required, -.84 * required),
        (-.55 * required, -.84 * required),
        (.82 * required, -.58 * required),
        (-.82 * required, -.58 * required),
        (required, -.25 * required),
        (-required, -.25 * required),
    ]
    vectors.extend(
        (x_factor * required, y_factor * required)
        for y_factor in (-1.6, -1.3, -1.0, -.7, -.4, -.1)
        for x_factor in (0, .6, -.6, 1.0, -1.0, 1.4, -1.4, 1.8, -1.8)
        if math.hypot(x_factor, y_factor) >= 1.0
    )
    candidates = [
        (float(tx) + dx, float(ty) + dy, 0)
        for dx, dy in vectors
        if -.58 <= float(tx) + dx <= .58 and .06 <= float(ty) + dy <= .65
    ]
    place_auxiliary(layout, occluder_model, registry, candidates)


def primary_scene(
    assignment: dict[str, Any], submode: str, family_index: int,
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_id = {value["id"]: value for value in registry.values()}
    used: set[str] = set()
    axis = assignment["ood_axis"]
    state = assignment["primary_answerability_stratum"]
    category = assignment["family_category"]
    target_pool = HELDOUT_TARGETS if axis == "asset" else SEEN_TARGETS
    first = pick_unique(target_pool, used, family_index)
    candidates = [first]
    if state == "AMBIGUOUS":
        if axis == "asset":
            candidates.append(pick_unique(HELDOUT_TARGETS, used, family_index + 1))
        elif submode == "same_class_duplicate":
            duplicate_pairs = [
                ("ycb_apple_01", "ycb_apple_02"),
                ("ycb_orange_01", "ycb_orange_02"),
            ]
            first, second = duplicate_pairs[family_index % len(duplicate_pairs)]
            used.update((first, second))
            candidates = [first, second]
        else:
            candidates.append(pick_unique(SEEN_TARGETS, used, family_index + 3))

    relation, frame = relation_for(category, family_index)
    anchor_count = 0 if relation == "direct" else (2 if relation in {"between_in_depth", "nearer_than_both"} else 1)
    anchors = [pick_unique(SEEN_ANCHORS, used, family_index + 2 + offset) for offset in range(anchor_count)]
    if axis == "asset" and state == "ABSENT" and submode == "anchor_absent" and anchors:
        anchors[0] = pick_unique(HELDOUT_TARGETS, used, family_index + 1)

    semantic_cf = pick_unique(SEMANTIC_CF, used, family_index)
    occluder = pick_unique(
        [
            "ycb_tomato_soup_can_01", "cube_blue_01", "cube_purple_01",
            "cube_green_01", "cube_orange_01", "cube_yellow_01",
            "ycb_mustard_bottle_01", "ycb_lemon_01", "mango_01",
        ],
        used,
        family_index,
    )

    valid = list(candidates)
    active_candidates = list(candidates)
    active_anchors = list(anchors)
    if state == "ABSENT":
        valid = []
        if submode == "target_absent":
            active_candidates = []
        elif submode == "anchor_absent":
            active_anchors = []
        elif submode == "unsatisfied_relation":
            if relation == "direct":
                active_candidates = []
                submode = "target_absent"
        else:
            raise PilotGenerationError(f"unknown ABSENT submode: {submode}")

    if state == "INSUFFICIENT_EVIDENCE" and len(valid) != 1:
        valid = [candidates[0]]
        active_candidates = [candidates[0]]

    layout = {name: list(value["storage_pose_xyyaw"]) for name, value in registry.items()}
    id_to_model = model_for_id(registry)

    def place(object_id: str, pose: tuple[float, float, float]) -> None:
        layout[id_to_model[object_id]] = [round(value, 6) for value in pose]

    # Calibrated visible geometry: y is monotonic with camera depth and
    # projected image x for the locked wrist view.  Keep task objects on the
    # x=-.20 lane and away from the y edges; the first real pilot attempt
    # showed that diagonal edge slots could be physically valid yet hidden in
    # the wrist-camera projection.  The OOD viewpoint uses a narrower y span.
    near_y = .16
    far_y = .40 if axis == "viewpoint" else .47
    mid_near_y = .27 if axis == "viewpoint" else .29
    mid_far_y = .30 if axis == "viewpoint" else .36
    target_poses: list[tuple[float, float, float]] = []
    anchor_poses: list[tuple[float, float, float]] = []
    if relation == "direct":
        target_poses = [(-.20, .18, 0), (-.20, far_y, .2)]
    elif relation in {"right_of", "left_of"}:
        target_y, second_y, anchor_y = (
            (mid_far_y, far_y, near_y)
            if relation == "right_of"
            else (near_y, mid_near_y, far_y)
        )
        target_poses = [(-.20, target_y, 0), (-.20, second_y, .2)]
        anchor_poses = [(-.20, anchor_y, 0)]
    elif relation in {"nearer_than", "front_of"}:
        target_poses = [(-.20, near_y, 0), (-.20, mid_near_y, .2)]
        anchor_poses = [(-.20, far_y, 0), (-.20, .52, 0)]
    elif relation in {"farther_than", "behind"}:
        target_poses = [(-.20, mid_far_y, 0), (-.20, far_y, .2)]
        anchor_poses = [(-.20, near_y, 0), (-.20, .10, 0)]
    elif relation == "between_in_depth":
        target_poses = [(-.20, .25, 0), (-.20, mid_far_y, .2)]
        anchor_poses = [(-.20, near_y, 0), (-.20, far_y, .1)]
    elif relation == "nearer_than_both":
        target_poses = [(-.20, near_y, 0), (-.20, mid_near_y, .2)]
        anchor_poses = [(-.20, mid_far_y, 0), (-.20, far_y, .1)]
    else:
        raise PilotGenerationError(f"unsupported relation: {relation}")

    if state == "ABSENT" and submode == "unsatisfied_relation" and active_candidates and active_anchors:
        # Realize the wrong relation inside the empirically visible band.  Both
        # the candidate and anchor remain observable, so ABSENT is caused by
        # the unsatisfied predicate rather than accidental field-of-view loss.
        if relation in {"right_of", "nearer_than", "front_of"}:
            target_poses[0] = (-.20, far_y, 0)
            anchor_poses[0] = (-.20, near_y, 0)
        elif relation in {"left_of", "farther_than", "behind"}:
            target_poses[0] = (-.20, near_y, 0)
            anchor_poses[0] = (-.20, far_y, 0)
        elif relation == "between_in_depth":
            target_poses[0] = (-.20, far_y, 0)
            anchor_poses = [(-.20, near_y, 0), (-.20, mid_far_y, .1)]
        elif relation == "nearer_than_both":
            target_poses[0] = (-.20, far_y, 0)
            anchor_poses = [(-.20, near_y, 0), (-.20, mid_near_y, .1)]

    if state == "INSUFFICIENT_EVIDENCE" and submode == "too_small_or_out_of_view":
        target_poses[0] = (.55, target_poses[0][1], 0)

    for index, object_id in enumerate(active_candidates):
        place(object_id, target_poses[min(index, len(target_poses) - 1)])
    for index, object_id in enumerate(active_anchors):
        place(object_id, anchor_poses[min(index, len(anchor_poses) - 1)])

    # A visible and robot-reachable semantic counterfactual object is always
    # present.  Do not use x=-.04: that lane intersects the target-bin fixture
    # and falls outside the locked action-reachability proxy.
    place_auxiliary(
        layout, id_to_model[semantic_cf], registry,
        [
            (-.08, .18, .15), (-.08, .12, .15), (-.38, .36, .15),
            (-.38, .22, .15), (-.20, .32, .15), (-.20, .43, .15),
        ],
    )

    # Layout OOD changes density/topology only; the other five axes do not.
    density_ids: list[str] = []
    if axis == "layout":
        for offset, pool in enumerate((["cube_orange_01", "cube_blue_01"], ["ycb_lemon_01", "mango_01"])):
            candidate = pick_unique(list(pool), used, family_index + offset)
            density_ids.append(candidate)
        density_slots = [
            (-.43, .14, .1), (-.43, .34, .1), (-.43, .55, .1),
            (-.02, .14, -.2), (-.02, .34, -.2), (-.02, .55, -.2),
        ]
        for object_id in density_ids:
            place_auxiliary(layout, id_to_model[object_id], registry, density_slots)

    # Some primary insufficient-evidence cases are realized in the clean capture.
    if state == "INSUFFICIENT_EVIDENCE" and submode == "occlusion" and active_candidates:
        place_safe_occluder(
            layout, id_to_model[active_candidates[0]], id_to_model[occluder], registry
        )

    clean_layout = copy.deepcopy(layout)
    occlusion_layout = copy.deepcopy(layout)
    occlusion_target = (
        active_candidates[0]
        if active_candidates and state != "ABSENT"
        else semantic_cf
    )
    if occlusion_target:
        place_safe_occluder(
            occlusion_layout, id_to_model[occlusion_target], id_to_model[occluder], registry
        )

    semantic_name = (
        "fruit" if len(candidates) > 1 and all(by_id[item]["semantic_class"] in {"apple", "orange", "lemon", "pear", "plum", "mango"} for item in candidates)
        else by_id[candidates[0]]["semantic_class"]
    )
    language_bank = assignment["language_template_bank_id"]
    instruction = instruction_for(semantic_name, relation, anchors, by_id, language_bank)
    state_values = state_fields(state, submode)

    return {
        "state_submode": submode,
        "relation": relation,
        "reference_frame": frame,
        "candidate_target_ids": candidates,
        "valid_target_ids": valid,
        "anchor_ids": anchors,
        "active_scene_ids": sorted(set(active_candidates + active_anchors + [semantic_cf] + density_ids + ([occluder] if state == "INSUFFICIENT_EVIDENCE" and submode == "occlusion" else []))),
        "instruction": instruction,
        "semantic_cf_id": semantic_cf,
        "occluder_id": occluder,
        "clean_layout": clean_layout,
        "occlusion_layout": occlusion_layout,
        **state_values,
    }


def variant_specs(
    assignment: dict[str, Any], scene: dict[str, Any],
    registry: dict[str, dict[str, Any]], family_index: int,
) -> list[dict[str, Any]]:
    by_id = {value["id"]: value for value in registry.values()}
    family_id = assignment["family_id"]
    state = assignment["primary_answerability_stratum"]
    base = {
        "instruction": scene["instruction"],
        "candidate_target_ids": list(scene["candidate_target_ids"]),
        "valid_target_ids": list(scene["valid_target_ids"]),
        "anchor_ids": list(scene["anchor_ids"]),
        "relations": [scene["relation"]],
        "reference_frame": scene["reference_frame"],
        "answerability_state": state,
        "state_submode": scene["state_submode"],
        "answerable": scene["answerable"],
        "uncertainty_sources": list(scene["sources"]),
        "severity": scene["severity"],
        "expected_intervention": scene["expected_intervention"],
        "perturbation": {"kind": "none", "severity": 0},
    }

    clean = {"variant": "clean", "capture_id": f"{family_id}__clean_capture", **copy.deepcopy(base)}

    semantic_id = scene["semantic_cf_id"]
    semantic = {
        "variant": "semantic_counterfactual",
        "capture_id": f"{family_id}__clean_capture",
        "instruction": instruction_for(by_id[semantic_id]["semantic_class"], "direct", [], by_id, assignment["language_template_bank_id"]),
        "candidate_target_ids": [semantic_id], "valid_target_ids": [semantic_id],
        "anchor_ids": [], "relations": ["direct"], "reference_frame": "object_semantics",
        "answerability_state": "FOUND", "state_submode": "unique_visible_referent",
        "answerable": True, "uncertainty_sources": [], "severity": 0,
        "expected_intervention": "EXECUTE",
        "perturbation": {"kind": "semantic_swap", "severity": 1},
    }

    if (
        scene["anchor_ids"]
        and scene["candidate_target_ids"]
        and scene["relation"] not in {"between_in_depth", "nearer_than_both"}
    ):
        cf_target = scene["anchor_ids"][0]
        cf_anchor = scene["candidate_target_ids"][0]
        cf_relation = inverse_relation(scene["relation"])
        both_active = cf_target in scene["active_scene_ids"] and cf_anchor in scene["active_scene_ids"]
        if not both_active or state == "ABSENT":
            cf_state = "ABSENT"
            cf_submode = scene["state_submode"] if state == "ABSENT" else "anchor_absent"
            cf_valid = []
        elif state == "INSUFFICIENT_EVIDENCE":
            cf_state = "INSUFFICIENT_EVIDENCE"
            cf_submode = scene["state_submode"]
            cf_valid = [cf_target]
        else:
            cf_state = "FOUND"
            cf_submode = "unique_visible_referent"
            cf_valid = [cf_target]
        cf_values = state_fields(cf_state, cf_submode)
        relation_instruction = instruction_for(by_id[cf_target]["semantic_class"], cf_relation, [cf_anchor], by_id, assignment["language_template_bank_id"])
        relation_anchors = [cf_anchor]
    else:
        cf_target = semantic_id
        cf_relation = "direct"
        cf_state = "FOUND"
        cf_submode = "unique_visible_referent"
        cf_valid = [cf_target]
        cf_values = state_fields("FOUND", "unique_visible_referent")
        relation_instruction = instruction_for(by_id[cf_target]["semantic_class"], "direct", [], by_id, assignment["language_template_bank_id"])
        relation_anchors = []
    relation_cf = {
        "variant": "relation_counterfactual", "capture_id": f"{family_id}__clean_capture",
        "instruction": relation_instruction, "candidate_target_ids": [cf_target],
        "valid_target_ids": cf_valid, "anchor_ids": relation_anchors,
        "relations": [cf_relation], "reference_frame": scene["reference_frame"] if cf_relation != "direct" else "object_semantics",
        "answerability_state": cf_state,
        "state_submode": cf_submode,
        "answerable": cf_values["answerable"], "uncertainty_sources": cf_values["sources"],
        "severity": cf_values["severity"], "expected_intervention": cf_values["expected_intervention"],
        "perturbation": {"kind": "relation_inverse", "severity": 1},
    }

    depth = copy.deepcopy(base)
    depth.update({"variant": "depth_corruption", "capture_id": f"{family_id}__clean_capture"})
    if state == "FOUND":
        depth.update({
            "answerability_state": "INSUFFICIENT_EVIDENCE", "state_submode": "depth_invalid_or_corrupt",
            "answerable": False, "uncertainty_sources": ["depth"], "severity": 2,
            "expected_intervention": "REOBSERVE",
        })
    depth_kind = "heldout_stripe_dropout" if assignment["ood_axis"] == "depth_noise" else ["bias_noise", "localized_holes_edges", "inversion_shift"][family_index % 3]
    depth["perturbation"] = {"kind": depth_kind, "severity": 2}

    occlusion = copy.deepcopy(base)
    occlusion.update({"variant": "occlusion_view_counterfactual", "capture_id": f"{family_id}__occlusion_capture"})
    if state == "FOUND":
        occlusion.update({
            "answerability_state": "INSUFFICIENT_EVIDENCE", "state_submode": "occlusion",
            "answerable": False, "uncertainty_sources": ["occlusion"], "severity": 2,
            "expected_intervention": "REOBSERVE",
        })
    occlusion["perturbation"] = {"kind": "physical_occluder", "severity": 2, "occluder_id": scene["occluder_id"]}
    values = [clean, semantic, relation_cf, depth, occlusion]
    if [item["variant"] for item in values] != list(VARIANTS):
        raise PilotGenerationError("variant order changed")
    return values


def build_artifacts(workspace: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = workspace / "protocol"
    seed_path = protocol / "dataset_expansion_v2_seed_lock.json"
    partition_path = protocol / "dataset_expansion_v2_asset_partition.json"
    seed_lock = read_json(seed_path)
    partition = read_json(partition_path)
    assignments = build_pilot_assignments(seed_lock, partition)
    commitment = assignment_commitment(assignments)
    if commitment != EXPECTED_PILOT_COMMITMENT or commitment != seed_lock["pilot_only"]["assignment_commitment_sha256"]:
        raise PilotGenerationError(f"pilot assignment commitment mismatch: {commitment}")
    registry = object_registry()
    by_id = {value["id"]: value for value in registry.values()}
    if len(by_id) != len(registry) or len({value["label"] for value in registry.values()}) != len(registry):
        raise PilotGenerationError("object IDs or semantic-instance labels are not unique")
    submodes = submode_map(assignments, int(seed_lock["master_seed"]))
    families = []
    captures = []
    for index, assignment in enumerate(assignments):
        try:
            scene = primary_scene(assignment, submodes[assignment["family_id"]], index, registry)
        except PilotGenerationError as exc:
            raise PilotGenerationError(f"{assignment['family_id']}: {exc}") from exc
        specs = variant_specs(assignment, scene, registry, index)
        family = {
            **assignment,
            "pilot_exclusion_policy": "NEVER_REUSE_ID_SEED_CAPTURE_LAYOUT_OR_SAMPLE_IN_OFFICIAL_DATASET",
            "state_submode": scene["state_submode"],
            "primary_scene": {
                key: scene[key] for key in (
                    "candidate_target_ids", "valid_target_ids", "anchor_ids", "active_scene_ids",
                    "relation", "reference_frame", "instruction", "occluder_id",
                )
            },
            "variant_specs": specs,
        }
        families.append(family)
        view = OOD_VIEW if assignment["ood_axis"] == "viewpoint" else IID_VIEW
        for condition, layout, seed_key in (
            ("clean", scene["clean_layout"], "sensor"),
            ("occlusion", scene["occlusion_layout"], "occlusion"),
        ):
            capture_id = f"{assignment['family_id']}__{condition}_capture"
            relevant_specs = [spec for spec in specs if spec["capture_id"] == capture_id]
            required_visible_ids: set[str] = set()
            for spec in relevant_specs:
                spec_state = spec["answerability_state"]
                if spec_state in {"FOUND", "AMBIGUOUS"}:
                    required_visible_ids.update(spec["valid_target_ids"])
                    required_visible_ids.update(spec["anchor_ids"])
                elif spec_state == "INSUFFICIENT_EVIDENCE":
                    relation_cf_out_of_view = (
                        spec["variant"] == "relation_counterfactual"
                        and spec["state_submode"] == "too_small_or_out_of_view"
                    )
                    if not relation_cf_out_of_view:
                        required_visible_ids.update(spec["anchor_ids"])
                    if spec["state_submode"] != "too_small_or_out_of_view" or relation_cf_out_of_view:
                        required_visible_ids.update(spec["valid_target_ids"])
                elif spec_state == "ABSENT" and spec["state_submode"] == "unsatisfied_relation":
                    required_visible_ids.update(spec["candidate_target_ids"])
                    required_visible_ids.update(spec["anchor_ids"])
            captures.append({
                "capture_id": capture_id,
                "family_id": assignment["family_id"], "split": "pilot_only",
                "condition": condition, "seed": int(assignment["seed_bundle"][seed_key]),
                "ood_axis": assignment["ood_axis"],
                "camera_bin_id": assignment["camera_bin_id"],
                "view_joint_pose": list(view),
                "layout_generator_id": assignment["layout_generator_id"],
                "layout": layout,
                "required_visible_object_ids": sorted(required_visible_ids),
                "required_visible_label_ids": sorted(int(by_id[value]["label"]) for value in required_visible_ids),
            })

    development = build_development_assignments(seed_lock, partition)
    reserved = {
        "family_ids_sha256": canonical_json_sha256(sorted(item["family_id"] for item in development)),
        "all_seed_values_sha256": canonical_json_sha256(sorted(value for item in development for value in item["seed_bundle"].values())),
        "assignment_commitment_sha256": assignment_commitment(development),
    }
    manifest = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION, "status": "MATERIALIZED_STATIC_NOT_CAPTURED",
        "split": "pilot_only", "eligible_for_official_dataset": False,
        "family_count": len(families), "sample_count": sum(len(item["variant_specs"]) for item in families),
        "variant_count_per_family": len(VARIANTS),
        "design_assignment_commitment_sha256": commitment,
        "design_seed_lock_sha256": sha256_file(seed_path),
        "asset_partition_sha256": sha256_file(partition_path),
        "reserved_development_commitments": reserved,
        "counts": {
            "primary_answerability": dict(sorted(Counter(item["primary_answerability_stratum"] for item in families).items())),
            "family_category": dict(sorted(Counter(item["family_category"] for item in families).items())),
            "ood_axis": dict(sorted(Counter(item["ood_axis"] for item in families).items())),
            "state_submode": dict(sorted(Counter(item["state_submode"] for item in families).items())),
        },
        "families": families,
    }
    world = workspace / "ur3/ur_simulation_gz/worlds/ur3_pick_place_inventory_v2.sdf"
    capture_plan = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "status": "STATIC_PLAN_CAPTURE_FORBIDDEN_UNTIL_PREFLIGHT_PASS",
        "world_file": str(world.relative_to(workspace)), "world_sha256": sha256_file(world),
        "coordinate_suffix": COORDINATE_SUFFIX,
        "capture_count": len(captures), "captures_per_family": 2,
        "sensor_contract": {
            "rgb_topic": "/wrist_camera/color/image_raw",
            "depth_topic": "/wrist_camera/depth/image_raw",
            "semantic_label_topic": "/wrist_camera/evaluation_labels/labels_map",
            "max_timestamp_spread_sec": .035, "metric_depth_unit": "metre",
        },
        "camera_poses": {"camera_v2_iid": IID_VIEW, "camera_v2_ood_elevation_azimuth": OOD_VIEW},
        "workspace_roi_base_link": {"x_m": [-.60, .60], "y_m": [.06, .65]},
        "object_registry": registry,
        "manifest_sha256_pending": True,
        "captures": captures,
    }
    registry_artifact = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "world_file": str(world.relative_to(workspace)), "world_sha256": sha256_file(world),
        "instance_count": len(registry), "objects": registry,
    }
    return manifest, capture_plan, registry_artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    protocol = workspace / "protocol"
    manifest, capture_plan, registry = build_artifacts(workspace)
    manifest_path = protocol / "dataset_v2_pilot_manifest.json"
    capture_path = protocol / "dataset_v2_pilot_capture_plan.json"
    registry_path = protocol / "dataset_v2_pilot_object_registry.json"
    write_json(manifest_path, manifest)
    capture_plan["manifest_sha256"] = sha256_file(manifest_path)
    capture_plan.pop("manifest_sha256_pending", None)
    write_json(capture_path, capture_plan)
    write_json(registry_path, registry)
    print(
        "DATASET_V2_PILOT_STATIC_MATERIALIZED "
        f"families={manifest['family_count']} samples={manifest['sample_count']} "
        f"captures={capture_plan['capture_count']} commitment={manifest['design_assignment_commitment_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
