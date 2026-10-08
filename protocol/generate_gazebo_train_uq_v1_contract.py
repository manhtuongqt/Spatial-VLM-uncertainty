#!/usr/bin/env python3
"""Generate deterministic Train-UQ/Val-UQ capture inputs before Gazebo capture.

The generated YAML files are protocol inputs, not captured data.  They define
320 independent parent families: 256 ``train_uq`` and 64 ``val_uq``.  The
evaluation oracle remains in the annotations file and is never model input.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ur3/ur3_perception/config"
SCENES_PATH = CONFIG / "gazebo_train_uq_v1_scenes.yaml"
ANNOTATIONS_PATH = CONFIG / "gazebo_train_uq_v1_annotations.yaml"
GATE_PATH = CONFIG / "gazebo_train_uq_v1_gate.yaml"
PROTOCOL_ID = "gazebo_train_uq_v1"

# Preserve the qualified assets and camera convention from answerability-v2.
OBJECTS = {
    "red_cube": {"z": 0.030000, "storage_pose": [-0.65, 1.40, 0.0]},
    "blue_cube": {"z": 0.025000, "storage_pose": [-0.45, 1.40, 0.0]},
    "green_cube": {"z": 0.025000, "storage_pose": [-0.25, 1.40, 0.0]},
    "yellow_cube": {"z": 0.025000, "storage_pose": [-0.05, 1.40, 0.0]},
    "orange_cube": {"z": 0.025000, "storage_pose": [0.15, 1.40, 0.0]},
    "purple_cube": {"z": 0.025000, "storage_pose": [0.35, 1.40, 0.0]},
    "pink_cube": {"z": 0.025000, "storage_pose": [0.55, 1.40, 0.0]},
    "ycb_cracker_box": {"z": 0.106710, "storage_pose": [-0.02, 0.85, 0.0]},
    "ycb_sugar_box": {"z": 0.088107, "storage_pose": [0.20, 0.85, 0.0]},
    "ycb_tomato_soup_can": {"z": 0.050948, "storage_pose": [-0.55, 1.10, 0.0]},
    "ycb_mustard_bottle": {"z": 0.095695, "storage_pose": [-0.38, 1.10, 0.0]},
    "ycb_banana": {"z": 0.018376, "storage_pose": [-0.30, 0.85, 0.25]},
    "ycb_apple": {"z": 0.037710, "storage_pose": [-0.21, 1.10, 0.0]},
    "ycb_orange": {"z": 0.037010, "storage_pose": [-0.04, 1.10, 0.0]},
    "ycb_power_drill": {"z": 0.028689, "storage_pose": [-0.62, 0.85, -0.25]},
    "mango": {"z": 0.030000, "storage_pose": [0.38, 0.85, 0.35]},
}
FRUITS = ["ycb_apple", "ycb_orange", "mango"]
LABELS = {"ycb_apple": 26, "ycb_orange": 27, "mango": 29}
NAMES = {"ycb_apple": "red apple", "ycb_orange": "orange", "mango": "yellow mango"}
STATES = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
RELATIONS = ["leftmost", "rightmost", "second_from_left", "second_from_right"]
SPLITS = {"train_uq": 16, "val_uq": 4}  # repetitions per state x relation cell
CAMERA = {
    "frame": "camera_color_optical_frame", "base_frame": "base_link", "resolution": [640, 480],
    "intrinsics": {"fx": 606.0816650391, "fy": 605.7973022461, "cx": 325.5436706543, "cy": 249.9961547852},
    "view_joint_pose": [1.3815, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601],
    "depth_unit": "metre", "valid_depth_range_m": [0.10, 2.0],
}


def seed_for(scene_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{PROTOCOL_ID}:{scene_id}".encode()).digest()[:4], "big")


def rank_spec(relation: str) -> tuple[str, int]:
    return {
        "leftmost": ("left", 1), "rightmost": ("right", 1),
        "second_from_left": ("left", 2), "second_from_right": ("right", 2),
    }[relation]


def instruction(relation: str) -> str:
    wording = {
        "leftmost": "leftmost object", "rightmost": "rightmost object",
        "second_from_left": "second object from left to right",
        "second_from_right": "second object from right to left",
    }[relation]
    return f"Among the red apple, orange, and yellow mango, identify the {wording} in the image."


def rotated_fruits(index: int) -> list[str]:
    shift = index % len(FRUITS)
    return FRUITS[shift:] + FRUITS[:shift]


def ordinal_target(ordered: list[str], relation: str) -> tuple[str, int]:
    position = {"leftmost": 0, "rightmost": 2, "second_from_left": 1, "second_from_right": 1}[relation]
    return ordered[position], position


def found_case(rep: int, relation: str, yaw: float) -> tuple[str, dict, dict]:
    ordered = rotated_fruits(rep + RELATIONS.index(relation))
    # In the fixed v2 camera convention, increasing table-y maps left-to-right
    # in the viewer frame; geometry QC verifies the rendered order after capture.
    ys = [0.115 + 0.006 * (rep % 8), 0.295 + 0.004 * (rep % 8), 0.475 + 0.003 * (rep % 8)]
    poses = {name: [-0.25 - 0.004 * (rep % 3), y, yaw + 0.17 * item] for item, (name, y) in enumerate(zip(ordered, ys))}
    target, position = ordinal_target(ordered, relation)
    rank_from, rank = rank_spec(relation)
    tags = ["fruit_identity", "horizontal_ordinal"]
    tags.append("edge_truncation" if position in (0, 2) else "central_target")
    return instruction(relation), poses, {
        "state": "FOUND", "target_id": target, "target_label": LABELS[target],
        "candidate_ids": ordered, "candidate_labels": [LABELS[name] for name in ordered],
        "rank_from": rank_from, "rank": rank, "relation_variant": relation,
        "target_category": target, "failure_tags": tags,
    }


def ambiguous_case(rep: int, relation: str, yaw: float) -> tuple[str, dict, dict]:
    # Apple and orange deliberately share the same projected ordering coordinate.
    if relation in ("leftmost", "second_from_left"):
        poses = {"ycb_apple": [-0.17, 0.246 + 0.002 * (rep % 8), yaw], "ycb_orange": [-0.25, 0.246 + 0.002 * (rep % 8), yaw + 0.2], "mango": [-0.25, 0.480, yaw - 0.15]}
        tie_side = "left"
    else:
        poses = {"mango": [-0.25, 0.120, yaw - 0.15], "ycb_apple": [-0.17, 0.354 + 0.002 * (rep % 8), yaw], "ycb_orange": [-0.25, 0.354 + 0.002 * (rep % 8), yaw + 0.2]}
        tie_side = "right"
    rank_from, rank = rank_spec(relation)
    return instruction(relation), poses, {
        "state": "AMBIGUOUS", "target_id": None, "target_label": None,
        "valid_target_ids": ["ycb_apple", "ycb_orange"], "valid_target_labels": [26, 27],
        "candidate_ids": FRUITS, "candidate_labels": [26, 27, 29],
        "rank_from": rank_from, "rank": rank, "relation_variant": relation,
        "target_category": "apple_orange_tie", "failure_tags": ["fruit_identity", "horizontal_tie", f"tie_{tie_side}"],
    }


def absent_case(rep: int, relation: str, yaw: float) -> tuple[str, dict, dict]:
    target = FRUITS[(rep + RELATIONS.index(relation)) % len(FRUITS)]
    visible = [name for name in FRUITS if name != target]
    poses = {visible[0]: [-0.26, 0.180 + 0.006 * (rep % 8), yaw], visible[1]: [-0.26, 0.415 + 0.004 * (rep % 8), yaw + 0.25]}
    wording = {"leftmost": "leftmost", "rightmost": "rightmost", "second_from_left": "second from left to right", "second_from_right": "second from right to left"}[relation]
    rank_from, rank = rank_spec(relation)
    return f"Identify the {wording} {NAMES[target]} among the visible tabletop objects in the image.", poses, {
        "state": "ABSENT", "target_id": target, "target_label": LABELS[target],
        "candidate_ids": [], "candidate_labels": [], "context_ids": visible, "context_labels": [LABELS[name] for name in visible],
        "rank_from": rank_from, "rank": rank, "relation_variant": relation,
        "target_category": target, "failure_tags": ["absent_target", "fruit_identity", "horizontal_ordinal"],
    }


def insufficient_case(rep: int, relation: str, yaw: float) -> tuple[str, dict, dict]:
    target = FRUITS[(rep + RELATIONS.index(relation)) % len(FRUITS)]
    is_left = relation in ("leftmost", "second_from_left")
    target_y = 0.052 + 0.002 * (rep % 4) if is_left else 0.552 - 0.002 * (rep % 4)
    other_y = [0.285, 0.465] if is_left else [0.135, 0.315]
    others = [name for name in FRUITS if name != target]
    poses = {target: [-0.20, target_y, yaw]}
    poses.update({name: [-0.28, y, yaw + 0.2 + 0.15 * index] for index, (name, y) in enumerate(zip(others, other_y))})
    rank_from, rank = rank_spec(relation)
    return instruction(relation), poses, {
        "state": "INSUFFICIENT_EVIDENCE", "target_id": target, "target_label": LABELS[target],
        "candidate_ids": FRUITS, "candidate_labels": [26, 27, 29], "occluder_ids": [], "occluder_labels": [],
        "rank_from": rank_from, "rank": rank, "relation_variant": relation,
        "target_category": target, "failure_tags": ["insufficient_visibility", "edge_truncation", "fruit_identity", "horizontal_ordinal"],
    }


BUILDERS = {"FOUND": found_case, "AMBIGUOUS": ambiguous_case, "ABSENT": absent_case, "INSUFFICIENT_EVIDENCE": insufficient_case}


def build() -> tuple[dict, dict, dict]:
    rows: list[dict] = []
    annotations: dict[str, dict] = {}
    global_index = 0
    for split, repetitions in SPLITS.items():
        for rep in range(repetitions):
            for state in STATES:
                for relation in RELATIONS:
                    scene_id = f"gazebo_uq_v1_{split}_{global_index:03d}"
                    family_id = f"spatial_vlm_uq_v1/{split}/parent_{global_index:03d}"
                    prompt, poses, annotation = BUILDERS[state](rep, relation, 0.055 * ((rep + RELATIONS.index(relation)) % 7))
                    rows.append({"scene_id": scene_id, "scene_family_id": family_id, "task_type": "horizontal_ordinal_ranking_answerability", "instruction": prompt, "poses": poses})
                    annotations[scene_id] = {**annotation, "family_id": family_id, "seed": seed_for(scene_id), "split": split}
                    global_index += 1
    split_counts = {split: repetitions * len(STATES) * len(RELATIONS) for split, repetitions in SPLITS.items()}
    state_quota = {split: {state: repetitions * len(RELATIONS) for state in STATES} for split, repetitions in SPLITS.items()}
    relation_quota = {split: {relation: repetitions * len(STATES) for relation in RELATIONS} for split, repetitions in SPLITS.items()}
    cell_quota = {split: repetitions for split, repetitions in SPLITS.items()}
    scenes = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "expected_scene_count": global_index,
        "capture_mode": "no_manipulation_shadow", "random_seed": 12092026,
        "view_joint_pose": CAMERA["view_joint_pose"], "camera_frame": CAMERA["frame"], "base_frame": CAMERA["base_frame"],
        "coordinate_suffix": "If exactly one requested target is identifiable and sufficiently visible, return exactly POINT [(x, y)], using one normalized image point with x and y between 0 and 1. If the request is ambiguous, the target is absent, or visual evidence is insufficient, return exactly ABSTAIN. Return no other text.",
        "objects": OBJECTS, "scenes": rows,
    }
    oracle = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "oracle_usage": "evaluation_only_open_after_prediction_lock",
        "semantic_label_source": "gazebo_evaluation_camera_not_vlm_input", "reference_frame": "image_viewer_left_to_right",
        "horizontal_coordinate": "normalized_centroid_x", "tie_margin_normalized": 0.03, "min_visible_evidence_px": 120,
        "states": STATES, "scenes": annotations,
    }
    gate = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "frozen_before_capture": True,
        "parent_family_count": global_index, "split_parent_family_count": split_counts,
        "state_quota": state_quota, "relation_variant_quota": relation_quota, "state_relation_cell_quota": cell_quota,
        "focus_object_quota_min": {"train_uq": {name: 200 for name in FRUITS}, "val_uq": {name: 50 for name in FRUITS}},
        "hard_case_quota_min": {"train_uq": {"edge_truncation": 96}, "val_uq": {"edge_truncation": 24}},
        "reference_frame": "image_viewer_left_to_right", "horizontal_coordinate": "normalized_centroid_x",
        "tie_margin_normalized": 0.03, "min_visible_evidence_px": 120,
        "insufficient_evidence_rule": {"min_visible_pixels_inclusive": 1, "max_visible_pixels_exclusive": 120, "projected_center_outer_margin_fraction": 0.20},
        "output_contract": {"point_regex": r"^POINT \[\(([0-9]*\.?[0-9]+), ([0-9]*\.?[0-9]+)\)\]$", "abstain_literal": "ABSTAIN", "allowed_actions": ["POINT", "ABSTAIN"], "point_state": "FOUND", "abstain_states": ["AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]},
        "camera": CAMERA, "family_split_rule": {"all_variants_same_parent_same_split": True, "co_locate": ["views", "paraphrases", "visibility_variants", "counterfactuals"]},
        "model_input_allowlist": ["rgb", "depth_m", "instruction"],
        "evaluator_only": ["semantic_labels", "target_id", "target_mask", "context_ids", "candidate_set", "answerability_state", "scene_layout", "camera_to_world"],
        "model_selection": {"selection_split": "val_uq", "forbidden_selection_splits": ["train_uq", "gazebo_dev_answerability_v2", "gazebo_calibration", "gazebo_test_iid", "gazebo_test_ood"], "checkpoint_rule": "lexicographic: maximize Val-UQ safe-task accuracy; minimize Val-UQ false-accept rate; maximize Val-UQ FOUND Hit@0.08; then lowest validation loss", "threshold_rule": "select only on Val-UQ after model checkpoint freeze"},
        "post_val_dev_rule": {"frozen_dev_dataset": "Gazebo_dev_answerability_v2", "allowed": "one blinded paired report after B1-UQ checkpoint and parser are frozen", "forbidden": ["checkpoint_selection", "threshold_selection", "prompt_editing", "training", "repeat_until_success"]},
        "future_splits": {"gazebo_calibration": "SEALED_NOT_MATERIALIZED", "gazebo_test_iid": "SEALED_NOT_MATERIALIZED", "gazebo_test_ood": "SEALED_NOT_MATERIALIZED"},
        "no_sam2": True, "b2_opened": False, "test_iid_ood_access": False,
    }
    return scenes, oracle, gate


def serialized(payload: dict) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write == args.check:
        parser.error("choose exactly one of --write or --check")
    outputs = dict(zip((SCENES_PATH, ANNOTATIONS_PATH, GATE_PATH), build()))
    if args.write:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing contract inputs: {existing}")
        for path, payload in outputs.items():
            path.write_text(serialized(payload), encoding="utf-8")
            print(path)
    else:
        mismatches = [str(path) for path, payload in outputs.items() if not path.is_file() or path.read_text(encoding="utf-8") != serialized(payload)]
        if mismatches:
            raise ValueError(f"generated contract inputs differ: {mismatches}")
        print("PASS: Train-UQ/Val-UQ contract inputs are deterministic")


if __name__ == "__main__":
    main()
