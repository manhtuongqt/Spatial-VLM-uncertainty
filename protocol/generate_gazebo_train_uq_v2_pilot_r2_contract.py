#!/usr/bin/env python3
"""Generate an immutable 32-family geometry pilot for Train-UQ v2 revision r2.

Revision r2 fixes the invalid v2 pilot design in which several relation queries
were declared as independent families while reusing the same physical layout.
Every row generated here has a distinct full-scene pose signature.  Geometry is
still accepted only from fresh rendered semantic masks after capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from generate_gazebo_train_uq_v1_contract import (
    CAMERA,
    FRUITS,
    LABELS,
    NAMES,
    OBJECTS,
    RELATIONS,
    rank_spec,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ur3/ur3_perception/config"
PROTOCOL_ID = "gazebo_train_uq_v2_pilot_r2"
SCENES_PATH = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS_PATH = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE_PATH = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
OUTPUT_PATHS = (SCENES_PATH, ANNOTATIONS_PATH, GATE_PATH)
STATES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")


def seed_for(scene_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{PROTOCOL_ID}:{scene_id}".encode()).digest()[:4],
        "big",
    )


def instruction(relation: str) -> str:
    wording = {
        "leftmost": "leftmost object",
        "rightmost": "rightmost object",
        "second_from_left": "second object from left to right",
        "second_from_right": "second object from right to left",
    }[relation]
    return (
        "Among the red apple, orange, and yellow mango, identify the "
        f"{wording} in the image."
    )


def rotated(values: list[str], shift: int) -> list[str]:
    offset = shift % len(values)
    return values[offset:] + values[:offset]


def target_index(relation: str) -> int:
    return {
        "leftmost": 0,
        "rightmost": 2,
        "second_from_left": 1,
        "second_from_right": 1,
    }[relation]


def found_case(rep: int, relation: str) -> tuple[str, dict, dict]:
    relation_index = RELATIONS.index(relation)
    order = rotated(FRUITS, 2 * rep + relation_index)
    left = 0.105 + 0.010 * relation_index + 0.007 * rep
    middle = 0.282 + 0.006 * ((relation_index + rep) % 3)
    right = 0.485 - 0.008 * relation_index - 0.006 * rep
    ys = (left, middle, right)
    poses = {
        name: [
            -0.225 - 0.010 * ((relation_index + item + rep) % 4),
            y,
            0.075 * (1 + relation_index + 2 * rep + item),
        ]
        for item, (name, y) in enumerate(zip(order, ys))
    }
    target = order[target_index(relation)]
    rank_from, rank = rank_spec(relation)
    return instruction(relation), poses, {
        "state": "FOUND",
        "target_id": target,
        "target_label": LABELS[target],
        "candidate_ids": order,
        "candidate_labels": [LABELS[name] for name in order],
        "rank_from": rank_from,
        "rank": rank,
        "relation_variant": relation,
        "target_category": target,
        "failure_tags": ["fruit_identity", "horizontal_ordinal"],
    }


def ambiguous_case(rep: int, relation: str) -> tuple[str, dict, dict]:
    relation_index = RELATIONS.index(relation)
    left_side = relation in ("leftmost", "second_from_left")
    # Existing rendered evidence showed that base-x separation 0.04 was too
    # close to the frozen 0.03 image-space margin.  r2 uses 0.025 and varies
    # the whole layout by relation/rep.  Fresh semantic masks remain decisive.
    if left_side:
        pair_y = 0.250 + 0.006 * relation_index + 0.004 * rep
        third_y = 0.478 - 0.011 * relation_index - 0.006 * rep
        poses = {
            "ycb_apple": [-0.225 - 0.004 * rep, pair_y, 0.08 + 0.11 * relation_index + 0.05 * rep],
            "ycb_orange": [-0.250, pair_y, 0.19 + 0.09 * relation_index + 0.04 * rep],
            "mango": [-0.238 + 0.004 * rep, third_y, 0.31 + 0.07 * relation_index + 0.05 * rep],
        }
        tie_side = "left"
    else:
        third_y = 0.112 + 0.010 * relation_index + 0.006 * rep
        pair_y = 0.342 + 0.006 * relation_index + 0.004 * rep
        poses = {
            "mango": [-0.238 + 0.004 * rep, third_y, 0.23 + 0.07 * relation_index + 0.05 * rep],
            "ycb_apple": [-0.225 - 0.004 * rep, pair_y, 0.09 + 0.10 * relation_index + 0.05 * rep],
            "ycb_orange": [-0.250, pair_y, 0.20 + 0.08 * relation_index + 0.04 * rep],
        }
        tie_side = "right"
    rank_from, rank = rank_spec(relation)
    return instruction(relation), poses, {
        "state": "AMBIGUOUS",
        "target_id": None,
        "target_label": None,
        "valid_target_ids": ["ycb_apple", "ycb_orange"],
        "valid_target_labels": [LABELS["ycb_apple"], LABELS["ycb_orange"]],
        "candidate_ids": FRUITS,
        "candidate_labels": [LABELS[name] for name in FRUITS],
        "rank_from": rank_from,
        "rank": rank,
        "relation_variant": relation,
        "target_category": "apple_orange_tie",
        "failure_tags": ["fruit_identity", "horizontal_tie", f"tie_{tie_side}"],
    }


def absent_case(rep: int, relation: str) -> tuple[str, dict, dict]:
    relation_index = RELATIONS.index(relation)
    target = FRUITS[(2 * rep + relation_index) % len(FRUITS)]
    visible = [name for name in FRUITS if name != target]
    poses = {
        visible[0]: [
            -0.220 - 0.010 * ((relation_index + rep) % 3),
            0.150 + 0.014 * relation_index + 0.009 * rep,
            0.10 + 0.08 * relation_index + 0.04 * rep,
        ],
        visible[1]: [
            -0.270 + 0.008 * ((relation_index + rep) % 3),
            0.455 - 0.012 * relation_index - 0.008 * rep,
            0.27 + 0.07 * relation_index + 0.05 * rep,
        ],
    }
    wording = relation.replace("_", " ")
    rank_from, rank = rank_spec(relation)
    return (
        f"Identify the {wording} {NAMES[target]} among the visible tabletop objects in the image.",
        poses,
        {
            "state": "ABSENT",
            "target_id": target,
            "target_label": LABELS[target],
            "candidate_ids": [],
            "candidate_labels": [],
            "context_ids": visible,
            "context_labels": [LABELS[name] for name in visible],
            "rank_from": rank_from,
            "rank": rank,
            "relation_variant": relation,
            "target_category": target,
            "failure_tags": ["absent_target", "fruit_identity", "horizontal_ordinal"],
        },
    )


# All target edge poses below previously produced 1--119 visible pixels in the
# frozen camera in a prior captured audit.  r2 changes the other-object layout;
# it does not copy a prior full-scene layout.  Fresh rendered QC is mandatory.
INSUFFICIENT_TARGETS = {
    (0, "leftmost"): ("ycb_apple", [-0.200, 0.054, 0.110]),
    (1, "leftmost"): ("ycb_orange", [-0.200, 0.056, 0.165]),
    (0, "rightmost"): ("mango", [-0.200, 0.546, 0.055]),
    (1, "rightmost"): ("ycb_apple", [-0.200, 0.552, 0.110]),
    (0, "second_from_left"): ("ycb_orange", [-0.200, 0.056, 0.110]),
    (1, "second_from_left"): ("ycb_orange", [-0.200, 0.056, 0.220]),
    (0, "second_from_right"): ("ycb_apple", [-0.200, 0.552, 0.055]),
    (1, "second_from_right"): ("ycb_orange", [-0.200, 0.550, 0.110]),
}


def insufficient_case(rep: int, relation: str) -> tuple[str, dict, dict]:
    relation_index = RELATIONS.index(relation)
    target, target_pose = INSUFFICIENT_TARGETS[(rep, relation)]
    others = [name for name in FRUITS if name != target]
    left_edge = relation in ("leftmost", "second_from_left")
    ys = (
        (0.260 + 0.012 * relation_index + 0.006 * rep, 0.445 - 0.008 * relation_index - 0.006 * rep)
        if left_edge
        else (0.145 + 0.010 * relation_index + 0.006 * rep, 0.325 + 0.008 * relation_index + 0.005 * rep)
    )
    poses = {target: list(target_pose)}
    for item, (name, y) in enumerate(zip(others, ys)):
        poses[name] = [
            -0.255 - 0.012 * ((relation_index + item + rep) % 3),
            y,
            0.21 + 0.09 * relation_index + 0.06 * rep + 0.05 * item,
        ]
    rank_from, rank = rank_spec(relation)
    return instruction(relation), poses, {
        "state": "INSUFFICIENT_EVIDENCE",
        "target_id": target,
        "target_label": LABELS[target],
        "candidate_ids": FRUITS,
        "candidate_labels": [LABELS[name] for name in FRUITS],
        "occluder_ids": [],
        "occluder_labels": [],
        "rank_from": rank_from,
        "rank": rank,
        "relation_variant": relation,
        "target_category": target,
        "failure_tags": [
            "insufficient_visibility",
            "edge_truncation",
            "geometry_pilot",
        ],
    }


BUILDERS = {
    "FOUND": found_case,
    "AMBIGUOUS": ambiguous_case,
    "ABSENT": absent_case,
    "INSUFFICIENT_EVIDENCE": insufficient_case,
}


def full_layout_payload(poses: dict) -> dict:
    return {
        name: poses.get(name, item["storage_pose"])
        for name, item in sorted(OBJECTS.items())
    }


def layout_signature(poses: dict) -> str:
    payload = json.dumps(
        full_layout_payload(poses), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build() -> tuple[dict, dict, dict]:
    rows: list[dict] = []
    annotations: dict[str, dict] = {}
    index = 0
    for rep in range(2):
        for state in STATES:
            for relation in RELATIONS:
                scene_id = f"gazebo_uq_v2_pilot_r2_{index:03d}"
                family_id = (
                    f"spatial_vlm_uq_v2_pilot_r2/pilot/parent_{index:03d}"
                )
                prompt, poses, annotation = BUILDERS[state](rep, relation)
                signature = layout_signature(poses)
                rows.append(
                    {
                        "scene_id": scene_id,
                        "scene_family_id": family_id,
                        "layout_id": f"layout_{index:03d}",
                        "layout_signature_sha256": signature,
                        "task_type": "horizontal_ordinal_ranking_answerability",
                        "instruction": prompt,
                        "poses": poses,
                    }
                )
                annotations[scene_id] = {
                    **annotation,
                    "family_id": family_id,
                    "layout_id": f"layout_{index:03d}",
                    "layout_signature_sha256": signature,
                    "split": "geometry_pilot",
                    "seed": seed_for(scene_id),
                }
                index += 1

    coordinate_suffix = (
        "If exactly one requested target is identifiable and sufficiently visible, "
        "return exactly POINT [(x, y)], using one normalized image point with x "
        "and y between 0 and 1. If the request is ambiguous, the target is absent, "
        "or visual evidence is insufficient, return exactly ABSTAIN. Return no other text."
    )
    output_contract = {
        "point_regex": r"^POINT \[\(([0-9]*\.?[0-9]+), ([0-9]*\.?[0-9]+)\)\]$",
        "abstain_literal": "ABSTAIN",
        "allowed_actions": ["POINT", "ABSTAIN"],
        "point_state": "FOUND",
        "abstain_states": ["AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"],
    }
    scenes = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "expected_scene_count": 32,
        "capture_mode": "no_manipulation_shadow",
        "random_seed": 14092026,
        "view_joint_pose": CAMERA["view_joint_pose"],
        "camera_frame": CAMERA["frame"],
        "base_frame": CAMERA["base_frame"],
        "coordinate_suffix": coordinate_suffix,
        "objects": OBJECTS,
        "scenes": rows,
    }
    oracle = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "oracle_usage": "geometry_qc_only",
        "semantic_label_source": "gazebo_evaluation_camera_not_vlm_input",
        "reference_frame": "image_viewer_left_to_right",
        "horizontal_coordinate": "normalized_centroid_x",
        "tie_margin_normalized": 0.03,
        "min_visible_evidence_px": 120,
        "states": list(STATES),
        "scenes": annotations,
    }
    gate = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "frozen_before_capture": True,
        "parent_family_count": 32,
        "split_parent_family_count": {"geometry_pilot": 32},
        "state_quota": {state: 8 for state in STATES},
        "relation_variant_quota": {relation: 8 for relation in RELATIONS},
        "state_relation_cell_quota": 2,
        "layout_uniqueness": {
            "full_pose_signatures_unique": True,
            "no_pose_signature_overlap_with_prior_uq_attempts": True,
            "byte_identical_rgb_forbidden": True,
            "perceptual_near_duplicate_rule": {
                "metric": "whole-frame grayscale mean absolute difference and changed-pixel fraction",
                "gray_absdiff_threshold_for_changed_pixel": 3,
                "near_duplicate_if_gray_mad_below": 0.05,
                "and_changed_pixel_fraction_below": 0.002,
            },
        },
        "reference_frame": "image_viewer_left_to_right",
        "horizontal_coordinate": "normalized_centroid_x",
        "tie_margin_normalized": 0.03,
        "min_visible_evidence_px": 120,
        "insufficient_evidence_rule": {
            "min_visible_pixels_inclusive": 1,
            "max_visible_pixels_exclusive": 120,
            "projected_center_outer_margin_fraction": 0.20,
        },
        "output_contract": output_contract,
        "camera": CAMERA,
        "family_split_rule": {
            "all_variants_same_parent_same_split": True,
            "identical_rendered_scene_queries_share_parent_family": True,
            "co_locate": [
                "views",
                "paraphrases",
                "visibility_variants",
                "counterfactuals",
                "queries_over_identical_layout",
            ],
        },
        "model_input_allowlist": ["rgb", "depth_m", "instruction"],
        "evaluator_only": [
            "semantic_labels",
            "answerability_state",
            "target_id",
            "candidate_set",
            "scene_layout",
        ],
        "policies": {
            "no_sam2": True,
            "no_training": True,
            "no_model_inference": True,
            "no_materialization": True,
            "no_b2": True,
            "no_test_access": True,
            "no_dev_v2_fit_or_selection": True,
            "no_robot_manipulation": True,
            "full_train_uq_only_after_pilot_pass": True,
        },
    }
    return scenes, oracle, gate


def serialized(payload: dict) -> str:
    return yaml.safe_dump(
        payload, sort_keys=False, allow_unicode=True, width=140
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write == args.check:
        parser.error("choose exactly one of --write or --check")
    outputs = dict(zip(OUTPUT_PATHS, build()))
    if args.write:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite existing r2 contract inputs: {existing}"
            )
        for path, payload in outputs.items():
            path.write_text(serialized(payload), encoding="utf-8")
            print(path)
    else:
        mismatches = [
            str(path)
            for path, payload in outputs.items()
            if not path.is_file()
            or path.read_text(encoding="utf-8") != serialized(payload)
        ]
        if mismatches:
            raise ValueError(f"r2 generated inputs differ: {mismatches}")
        print("PASS: deterministic Gazebo Train-UQ v2 pilot r2 contract inputs")


if __name__ == "__main__":
    main()
