#!/usr/bin/env python3
"""Generate the preregistered 64-family Gazebo answerability-v2 YAML inputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ur3/ur3_perception/config"
SCENES_PATH = CONFIG / "gazebo_dev_answerability_v2_scenes.yaml"
ANNOTATIONS_PATH = CONFIG / "gazebo_dev_answerability_v2_annotations.yaml"
GATE_PATH = CONFIG / "gazebo_dev_answerability_v2_gate.yaml"
PROTOCOL_ID = "gazebo_dev_answerability_v2"

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
RELATIONS = ["leftmost", "rightmost", "second_from_left", "second_from_right"]
STATES = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]


def seed_for(scene_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{PROTOCOL_ID}:{scene_id}".encode()).digest()[:4], "big")


def instruction(relation: str, subject: str = "the red apple, orange, and yellow mango") -> str:
    relation_text = {
        "leftmost": "leftmost object",
        "rightmost": "rightmost object",
        "second_from_left": "second object from left to right",
        "second_from_right": "second object from right to left",
    }[relation]
    return f"Among {subject}, identify the {relation_text} in the image."


def rank_spec(relation: str) -> tuple[str, int]:
    return {
        "leftmost": ("left", 1),
        "rightmost": ("right", 1),
        "second_from_left": ("left", 2),
        "second_from_right": ("right", 2),
    }[relation]


def found_case(i: int, yaw: float) -> tuple[str, dict, dict]:
    relation = RELATIONS[i % 4]
    permutation = [
        FRUITS,
        ["ycb_orange", "mango", "ycb_apple"],
        ["mango", "ycb_apple", "ycb_orange"],
        ["ycb_apple", "mango", "ycb_orange"],
    ][i % 4]
    ys = [0.13 + 0.004 * (i // 4), 0.30 + 0.003 * (i // 4), 0.47 + 0.002 * (i // 4)]
    poses = {name: [-0.25 - 0.004 * (i % 2), y, yaw + j * 0.17] for j, (name, y) in enumerate(zip(permutation, ys))}
    target_index = 0 if relation == "leftmost" else 2 if relation == "rightmost" else 1
    target = permutation[target_index]
    rank_from, rank = rank_spec(relation)
    return instruction(relation), poses, {
        "state": "FOUND", "target_id": target, "target_label": LABELS[target],
        "candidate_ids": FRUITS, "candidate_labels": [LABELS[name] for name in FRUITS],
        "rank_from": rank_from, "rank": rank, "relation_variant": relation,
        "target_category": target, "failure_tags": ["fruit_identity", "horizontal_ordinal", "edge_truncation" if target_index in (0, 2) else "central_target"],
    }


def ambiguous_case(i: int, yaw: float) -> tuple[str, dict, dict]:
    relation = RELATIONS[i % 4]
    if relation in ("leftmost", "second_from_left"):
        poses = {
            "ycb_apple": [-0.17, 0.250 + 0.001 * (i // 4), yaw],
            "ycb_orange": [-0.25, 0.240 + 0.001 * (i // 4), yaw + 0.2],
            "mango": [-0.25, 0.480, yaw - 0.15],
        }
        tie_side = "left"
    else:
        poses = {
            "mango": [-0.25, 0.125, yaw - 0.15],
            "ycb_apple": [-0.17, 0.360 + 0.001 * (i // 4), yaw],
            "ycb_orange": [-0.25, 0.350 + 0.001 * (i // 4), yaw + 0.2],
        }
        tie_side = "right"
    rank_from, rank = rank_spec(relation)
    return instruction(relation), poses, {
        "state": "AMBIGUOUS", "target_id": None, "target_label": None,
        "valid_target_ids": ["ycb_apple", "ycb_orange"], "valid_target_labels": [26, 27],
        "candidate_ids": FRUITS, "candidate_labels": [26, 27, 29],
        "rank_from": rank_from, "rank": rank, "relation_variant": relation,
        "target_category": "apple_or_orange_tie", "failure_tags": ["fruit_identity", "horizontal_tie", f"tie_{tie_side}"],
    }


def absent_case(i: int, yaw: float) -> tuple[str, dict, dict]:
    relation = RELATIONS[i % 4]
    target = FRUITS[i % 3]
    visible = [name for name in FRUITS if name != target]
    poses = {
        visible[0]: [-0.26, 0.19 + 0.006 * (i // 4), yaw],
        visible[1]: [-0.26, 0.41 + 0.004 * (i // 4), yaw + 0.25],
    }
    ordinal = {
        "leftmost": "leftmost", "rightmost": "rightmost",
        "second_from_left": "second from left to right", "second_from_right": "second from right to left",
    }[relation]
    prompt = f"Identify the {ordinal} {NAMES[target]} among the visible tabletop objects in the image."
    rank_from, rank = rank_spec(relation)
    return prompt, poses, {
        "state": "ABSENT", "target_id": target, "target_label": LABELS[target],
        "candidate_ids": [], "candidate_labels": [], "context_ids": visible,
        "context_labels": [LABELS[name] for name in visible],
        "rank_from": rank_from, "rank": rank, "relation_variant": relation,
        "target_category": target, "failure_tags": ["absent_target", "fruit_identity", "horizontal_ordinal"],
    }


def insufficient_case(i: int, yaw: float) -> tuple[str, dict, dict]:
    relation = RELATIONS[i % 4]
    pattern = i % 4
    if pattern == 0:
        target = "ycb_apple"; target_pose = [-0.20, 0.055, yaw]; others = [("ycb_orange", 0.30), ("mango", 0.46)]
    elif pattern == 1:
        target = "ycb_orange"; target_pose = [-0.20, 0.551, yaw]; others = [("ycb_apple", 0.15), ("mango", 0.34)]
    elif pattern == 2:
        target = "mango"; target_pose = [-0.18, 0.550, yaw]; others = [("ycb_apple", 0.14), ("ycb_orange", 0.32)]
    else:
        target = "ycb_apple"; target_pose = [-0.20, 0.055, yaw]; others = [("ycb_orange", 0.29), ("mango", 0.46)]
    poses = {target: target_pose}
    poses.update({name: [-0.28, y, yaw + 0.2 + j * 0.15] for j, (name, y) in enumerate(others)})
    rank_from, rank = rank_spec(relation)
    return instruction(relation), poses, {
        "state": "INSUFFICIENT_EVIDENCE", "target_id": target, "target_label": LABELS[target],
        "candidate_ids": FRUITS, "candidate_labels": [26, 27, 29], "occluder_ids": [], "occluder_labels": [],
        "rank_from": rank_from, "rank": rank, "relation_variant": relation,
        "target_category": target, "failure_tags": ["insufficient_visibility", "edge_truncation", "fruit_identity", "horizontal_ordinal"],
    }


def build() -> tuple[dict, dict, dict]:
    scenes = []
    annotations = {}
    builders = {"FOUND": found_case, "AMBIGUOUS": ambiguous_case, "ABSENT": absent_case, "INSUFFICIENT_EVIDENCE": insufficient_case}
    counters = {state: 0 for state in STATES}
    for index in range(64):
        state = STATES[index % 4]
        within = counters[state]; counters[state] += 1
        scene_id = f"gazebo_dev_ans_v2_{index:03d}"
        family_id = f"spatial_vlm_v2/dev/answerability/parent_{index:03d}"
        prompt, poses, annotation = builders[state](within, 0.07 * (within % 5))
        scenes.append({"scene_id": scene_id, "scene_family_id": family_id, "task_type": "horizontal_ordinal_ranking_answerability", "instruction": prompt, "poses": poses})
        annotation.update({"family_id": family_id, "seed": seed_for(scene_id), "split": "dev"})
        annotations[scene_id] = annotation
    scene_payload = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "expected_scene_count": 64,
        "capture_mode": "no_manipulation_shadow", "random_seed": 12092026,
        "view_joint_pose": [1.3815, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601],
        "camera_frame": "camera_color_optical_frame", "base_frame": "base_link",
        "coordinate_suffix": (
            "If exactly one requested target is identifiable and sufficiently visible, return exactly "
            "POINT [(x, y)], using one normalized image point with x and y between 0 and 1. "
            "If the request is ambiguous, the target is absent, or visual evidence is insufficient, "
            "return exactly ABSTAIN. Return no other text."
        ),
        "objects": OBJECTS, "scenes": scenes,
    }
    annotation_payload = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "oracle_usage": "evaluation_only_open_after_prediction_lock",
        "semantic_label_source": "gazebo_evaluation_camera_not_vlm_input",
        "reference_frame": "image_viewer_left_to_right", "horizontal_coordinate": "normalized_centroid_x",
        "tie_margin_normalized": 0.03, "min_visible_evidence_px": 120,
        "states": STATES, "scenes": annotations,
    }
    gate_payload = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "frozen_before_capture": True,
        "split": "dev", "parent_family_count": 64,
        "state_quota": {state: 16 for state in STATES},
        "relation_variant_quota": {relation: 16 for relation in RELATIONS},
        "focus_object_quota_min": {name: 16 for name in FRUITS},
        "reference_frame": "image_viewer_left_to_right", "horizontal_coordinate": "normalized_centroid_x",
        "tie_margin_normalized": 0.03, "min_visible_evidence_px": 120,
        "insufficient_evidence_rule": {"min_visible_pixels_inclusive": 1, "max_visible_pixels_exclusive": 120, "projected_center_outer_margin_fraction": 0.20},
        "output_contract": {
            "point_regex": r"^POINT \[\(([0-9]*\.?[0-9]+), ([0-9]*\.?[0-9]+)\)\]$",
            "abstain_literal": "ABSTAIN", "allowed_actions": ["POINT", "ABSTAIN"],
            "point_state": "FOUND", "abstain_states": ["AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"],
        },
        "promotion_rule": {"requires_b1_found_hit_strictly_higher": True, "requires_b1_safe_task_accuracy_strictly_higher": True, "requires_no_exact_format_regression": True},
        "max_sensor_spread_sec": 0.02,
        "camera": {
            "frame": "camera_color_optical_frame", "base_frame": "base_link", "resolution": [640, 480],
            "intrinsics": {"fx": 606.0816650391, "fy": 605.7973022461, "cx": 325.5436706543, "cy": 249.9961547852},
            "view_joint_pose": [1.3815, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601],
            "depth_unit": "metre", "valid_depth_range_m": [0.10, 2.0],
        },
        "family_split_rule": {"all_variants_same_parent_same_split": True, "co_locate": ["views", "paraphrases", "visibility_variants", "counterfactuals"]},
        "model_input_allowlist": ["rgb", "depth_m", "instruction"],
        "evaluator_only": ["semantic_labels", "target_id", "target_mask", "context_ids", "candidate_set", "answerability_state", "scene_layout", "camera_to_world"],
        "no_sam2": True, "b2_opened": False, "test_iid_ood_access": False,
    }
    return scene_payload, annotation_payload, gate_payload


def serialized(payload: dict) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write == args.check:
        parser.error("choose exactly one of --write or --check")
    payloads = dict(zip((SCENES_PATH, ANNOTATIONS_PATH, GATE_PATH), build()))
    if args.write:
        existing = [str(path) for path in payloads if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing contract inputs: {existing}")
        for path, payload in payloads.items():
            path.write_text(serialized(payload), encoding="utf-8")
            print(path)
    else:
        mismatches = [str(path) for path, payload in payloads.items() if not path.is_file() or path.read_text(encoding="utf-8") != serialized(payload)]
        if mismatches:
            raise ValueError(f"generated contract inputs differ: {mismatches}")
        print("PASS: generated v2 contract inputs are deterministic")


if __name__ == "__main__":
    main()
