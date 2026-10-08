#!/usr/bin/env python3
"""Generate the immutable 32-family Gazebo Train-UQ v2 pilot revision r5."""
from __future__ import annotations

import argparse
import hashlib
import json

import yaml

import generate_gazebo_train_uq_v2_pilot_r4_contract as r4
from generate_gazebo_train_uq_v1_contract import CAMERA, FRUITS, LABELS, OBJECTS, RELATIONS, rank_spec

ROOT = r4.ROOT
CONFIG = r4.CONFIG
PROTOCOL_ID = "gazebo_train_uq_v2_pilot_r5"
SCENES_PATH = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS_PATH = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE_PATH = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
OUTPUT_PATHS = (SCENES_PATH, ANNOTATIONS_PATH, GATE_PATH)
STATES = r4.STATES
PANEL = "uq_neutral_occluder"
PANEL_LABEL = 39
OBJECTS_R5 = {**OBJECTS, PANEL: {"z": 0.060, "storage_pose": [-0.60, 1.20, 0.0]}}


def seed_for(scene_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{PROTOCOL_ID}:{scene_id}".encode()).digest()[:4], "big")


# Selected only from evaluator-only v8/v9 characterization.
INSUFFICIENT_TARGETS = {
    (0, "leftmost"): ("ycb_apple", 0.0, 0.03330),
    (0, "rightmost"): ("ycb_orange", 0.0, 0.03615),
    (0, "second_from_left"): ("mango", 0.0, 0.04325),
    (0, "second_from_right"): ("ycb_apple", 0.35, 0.03270),
    (1, "leftmost"): ("ycb_orange", 0.35, 0.03625),
    (1, "rightmost"): ("mango", 0.35, 0.04330),
    (1, "second_from_left"): ("ycb_apple", 0.0, 0.03330),
    (1, "second_from_right"): ("ycb_orange", 0.0, 0.03615),
}


def insufficient_case(rep: int, relation: str):
    target, yaw, offset = INSUFFICIENT_TARGETS[(rep, relation)]
    others = [name for name in FRUITS if name != target]
    ys = {
        "leftmost": (0.390, 0.475),
        "rightmost": (0.120, 0.195),
        "second_from_left": (0.135, 0.465),
        "second_from_right": (0.135, 0.465),
    }[relation]
    poses = {
        target: [-0.220, 0.300, yaw],
        PANEL: [-0.2168, round(0.2947 - offset, 6), 1.5707963267948966],
    }
    for item, (name, y) in enumerate(zip(others, ys)):
        poses[name] = [-0.050 + 0.014 * item + 0.005 * rep, y, 0.19 + 0.12 * item + 0.04 * rep]
    rank_from, rank = rank_spec(relation)
    return r4.instruction(relation), poses, {
        "state": "INSUFFICIENT_EVIDENCE", "target_id": target, "target_label": LABELS[target],
        "candidate_ids": FRUITS, "candidate_labels": [LABELS[name] for name in FRUITS],
        "occluder_ids": [PANEL], "occluder_labels": [PANEL_LABEL],
        "rank_from": rank_from, "rank": rank, "relation_variant": relation,
        "target_category": target,
        "failure_tags": ["insufficient_visibility", "controlled_partial_occlusion",
                         "collision_free_visual_only_occluder", "characterization_v8_v9_locked_geometry",
                         "geometry_pilot"],
    }


BUILDERS = {"FOUND": r4.found_case, "AMBIGUOUS": r4.ambiguous_case,
            "ABSENT": r4.absent_case, "INSUFFICIENT_EVIDENCE": insufficient_case}


def revised_layout(index: int, state: str, poses: dict) -> dict:
    if state == "INSUFFICIENT_EVIDENCE":
        return {name: list(pose) for name, pose in poses.items()}
    dx = 0.0120 + 0.0007 * (index % 4)
    dy = 0.0100 if index % 2 == 0 else -0.0100
    return {name: [pose[0] + dx, pose[1] + dy, pose[2] + 0.013] for name, pose in poses.items()}


def full_layout_payload(poses: dict) -> dict:
    return {name: poses.get(name, item["storage_pose"]) for name, item in sorted(OBJECTS_R5.items())}


def layout_signature(poses: dict) -> str:
    return hashlib.sha256(json.dumps(full_layout_payload(poses), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build():
    rows, annotations = [], {}
    for rep in range(2):
        for state in STATES:
            for relation in RELATIONS:
                index = len(rows)
                sid = f"gazebo_uq_v2_pilot_r5_{index:03d}"
                family = f"spatial_vlm_uq_v2_pilot_r5/pilot/parent_{index:03d}"
                prompt, poses, annotation = BUILDERS[state](rep, relation)
                poses = revised_layout(index, state, poses)
                signature = layout_signature(poses)
                rows.append({"scene_id": sid, "scene_family_id": family, "layout_id": f"layout_r5_{index:03d}",
                             "layout_signature_sha256": signature,
                             "task_type": "horizontal_ordinal_ranking_answerability",
                             "instruction": prompt, "poses": poses})
                annotations[sid] = {**annotation, "family_id": family, "layout_id": f"layout_r5_{index:03d}",
                                    "layout_signature_sha256": signature, "split": "geometry_pilot",
                                    "seed": seed_for(sid)}
    coordinate_suffix = (
        "If exactly one requested target is identifiable and sufficiently visible, return exactly POINT [(x, y)], "
        "using one normalized image point with x and y between 0 and 1. If the request is ambiguous, the target is "
        "absent, or visual evidence is insufficient, return exactly ABSTAIN. Return no other text."
    )
    output_contract = {"point_regex": r"^POINT \[\(([0-9]*\.?[0-9]+), ([0-9]*\.?[0-9]+)\)\]$",
                       "abstain_literal": "ABSTAIN", "allowed_actions": ["POINT", "ABSTAIN"],
                       "point_state": "FOUND", "abstain_states": ["AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]}
    scenes = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "expected_scene_count": 32,
              "capture_mode": "no_manipulation_shadow", "random_seed": 14092036,
              "view_joint_pose": CAMERA["view_joint_pose"], "camera_frame": CAMERA["frame"],
              "base_frame": CAMERA["base_frame"], "coordinate_suffix": coordinate_suffix,
              "objects": OBJECTS_R5, "scenes": rows}
    oracle = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "oracle_usage": "geometry_qc_only",
              "semantic_label_source": "gazebo_evaluation_camera_not_vlm_input",
              "reference_frame": "image_viewer_left_to_right", "horizontal_coordinate": "normalized_centroid_x",
              "tie_margin_normalized": 0.03, "min_visible_evidence_px": 120,
              "states": list(STATES), "scenes": annotations}
    gate = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "frozen_before_capture": True,
            "parent_family_count": 32, "split_parent_family_count": {"geometry_pilot": 32},
            "state_quota": {state: 8 for state in STATES},
            "relation_variant_quota": {relation: 8 for relation in RELATIONS},
            "state_relation_cell_quota": 2,
            "layout_uniqueness": {"full_pose_signatures_unique": True,
                                  "no_pose_signature_overlap_with_prior_uq_attempts": True,
                                  "byte_identical_rgb_forbidden": True,
                                  "perceptual_near_duplicate_rule": {
                                      "metric": "whole-frame grayscale mean absolute difference and changed-pixel fraction",
                                      "gray_absdiff_threshold_for_changed_pixel": 3,
                                      "near_duplicate_if_gray_mad_below": 0.05,
                                      "and_changed_pixel_fraction_below": 0.002}},
            "reference_frame": "image_viewer_left_to_right", "horizontal_coordinate": "normalized_centroid_x",
            "tie_margin_normalized": 0.03, "min_visible_evidence_px": 120,
            "insufficient_evidence_rule": {"min_visible_pixels_inclusive": 1,
                                           "max_visible_pixels_exclusive": 120,
                                           "projected_center_outer_margin_fraction": 0.20},
            "output_contract": output_contract, "camera": CAMERA,
            "family_split_rule": {"all_variants_same_parent_same_split": True,
                                  "identical_rendered_scene_queries_share_parent_family": True,
                                  "co_locate": ["views", "paraphrases", "visibility_variants", "counterfactuals",
                                                "queries_over_identical_layout"]},
            "model_input_allowlist": ["rgb", "depth_m", "instruction"],
            "evaluator_only": ["semantic_labels", "answerability_state", "target_id", "candidate_set", "scene_layout"],
            "policies": {"no_sam2": True, "no_training": True, "no_model_inference": True,
                         "no_materialization": True, "no_b2": True, "no_test_access": True,
                         "no_dev_v2_fit_or_selection": True, "no_robot_manipulation": True,
                         "full_train_uq_only_after_pilot_pass": True}}
    return scenes, oracle, gate


def serialized(payload: dict) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--write", action="store_true"); parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write == args.check: parser.error("choose exactly one of --write or --check")
    outputs = dict(zip(OUTPUT_PATHS, build()))
    if args.write:
        existing = [str(path) for path in outputs if path.exists()]
        if existing: raise FileExistsError(f"refusing overwrite: {existing}")
        for path, payload in outputs.items(): path.write_text(serialized(payload), encoding="utf-8"); print(path)
    else:
        mismatches = [str(path) for path, payload in outputs.items() if not path.is_file() or path.read_text() != serialized(payload)]
        if mismatches: raise ValueError(f"generated inputs differ: {mismatches}")
        print("PASS: deterministic Gazebo Train-UQ v2 pilot r5 contract inputs")


if __name__ == "__main__":
    main()
