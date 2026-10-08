#!/usr/bin/env python3
"""Generate a new, independent 32-family geometry-validation pilot for UQ v2."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import yaml

from generate_gazebo_train_uq_v1_contract import CAMERA, FRUITS, LABELS, NAMES, OBJECTS, RELATIONS, rank_spec

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ur3/ur3_perception/config"
PROTOCOL = "gazebo_train_uq_v2_pilot"
OUT = (CONFIG / "gazebo_train_uq_v2_pilot_scenes.yaml", CONFIG / "gazebo_train_uq_v2_pilot_annotations.yaml", CONFIG / "gazebo_train_uq_v2_pilot_gate.yaml")
STATES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")


def instruction(relation: str) -> str:
    words = {"leftmost": "leftmost object", "rightmost": "rightmost object", "second_from_left": "second object from left to right", "second_from_right": "second object from right to left"}
    return f"Among the red apple, orange, and yellow mango, identify the {words[relation]} in the image."


def found(rep: int, relation: str) -> tuple[str, dict, dict]:
    order = FRUITS[rep % 3:] + FRUITS[:rep % 3]
    poses = {name: [-.25, y, .055*(rep+idx)] for idx, (name, y) in enumerate(zip(order, (.12, .30, .48)))}
    rank_from, rank = rank_spec(relation); index = {"leftmost": 0, "rightmost": 2, "second_from_left": 1, "second_from_right": 1}[relation]
    target = order[index]
    return instruction(relation), poses, {"state": "FOUND", "target_id": target, "target_label": LABELS[target], "candidate_ids": order, "candidate_labels": [LABELS[n] for n in order], "rank_from": rank_from, "rank": rank, "relation_variant": relation, "target_category": target, "failure_tags": ["fruit_identity", "horizontal_ordinal"]}


def ambiguous(rep: int, relation: str) -> tuple[str, dict, dict]:
    # v1 used a 0.08 base-x gap and rendered 0.036--0.046 image-x gap.  This
    # pilot uses 0.04 base-x while preserving a shared ordering coordinate; QC
    # accepts it only if the *rendered* gap is <= 0.03 and both masks survive.
    pair_y = .255 + .002*rep
    if relation in ("leftmost", "second_from_left"):
        poses = {"ycb_apple": [-.21, pair_y, .055], "ycb_orange": [-.25, pair_y, .165], "mango": [-.25, .48, .275]}
    else:
        poses = {"mango": [-.25, .12, .275], "ycb_apple": [-.21, pair_y+.10, .055], "ycb_orange": [-.25, pair_y+.10, .165]}
    rank_from, rank = rank_spec(relation)
    return instruction(relation), poses, {"state": "AMBIGUOUS", "target_id": None, "target_label": None, "valid_target_ids": ["ycb_apple", "ycb_orange"], "valid_target_labels": [26, 27], "candidate_ids": FRUITS, "candidate_labels": [26, 27, 29], "rank_from": rank_from, "rank": rank, "relation_variant": relation, "target_category": "apple_orange_tie", "failure_tags": ["fruit_identity", "horizontal_tie", "rendered_tie_required"]}


def absent(rep: int, relation: str) -> tuple[str, dict, dict]:
    target = FRUITS[(rep + RELATIONS.index(relation)) % 3]; visible = [x for x in FRUITS if x != target]
    poses = {visible[0]: [-.26, .18, .055], visible[1]: [-.26, .42, .165]}; rank_from, rank = rank_spec(relation)
    return f"Identify the {relation.replace('_', ' ')} {NAMES[target]} among the visible tabletop objects in the image.", poses, {"state": "ABSENT", "target_id": target, "target_label": LABELS[target], "candidate_ids": [], "candidate_labels": [], "context_ids": visible, "context_labels": [LABELS[x] for x in visible], "rank_from": rank_from, "rank": rank, "relation_variant": relation, "target_category": target, "failure_tags": ["absent_target", "fruit_identity"]}


def insufficient(rep: int, relation: str) -> tuple[str, dict, dict]:
    # These four target configurations are empirical *candidate* poses from
    # v1 raw capture that had 10--71 visible target pixels.  They are not
    # labels: v2 acceptance still relies solely on fresh rendered geometry.
    target_spec = {
        "leftmost": ("ycb_apple", [-.20, .054, .11]),
        "rightmost": ("mango", [-.20, .546, .055]),
        "second_from_left": ("ycb_orange", [-.20, .056, .11]),
        "second_from_right": ("ycb_apple", [-.20, .552, .11]),
    }[relation]
    target, pose = target_spec; others = [x for x in FRUITS if x != target]
    poses = {target: pose, others[0]: [-.28, .285, .165+.02*rep], others[1]: [-.28, .465, .275+.02*rep]}; rank_from, rank = rank_spec(relation)
    return instruction(relation), poses, {"state": "INSUFFICIENT_EVIDENCE", "target_id": target, "target_label": LABELS[target], "candidate_ids": FRUITS, "candidate_labels": [26, 27, 29], "occluder_ids": [], "occluder_labels": [], "rank_from": rank_from, "rank": rank, "relation_variant": relation, "target_category": target, "failure_tags": ["insufficient_visibility", "edge_truncation", "geometry_pilot"]}


BUILDERS = {"FOUND": found, "AMBIGUOUS": ambiguous, "ABSENT": absent, "INSUFFICIENT_EVIDENCE": insufficient}


def build() -> tuple[dict, dict, dict]:
    scenes, annotations, index = [], {}, 0
    for rep in range(2):
        for state in STATES:
            for relation in RELATIONS:
                sid = f"gazebo_uq_v2_pilot_{index:03d}"; family = f"spatial_vlm_uq_v2/pilot/parent_{index:03d}"
                prompt, poses, ann = BUILDERS[state](rep, relation)
                scenes.append({"scene_id": sid, "scene_family_id": family, "task_type": "horizontal_ordinal_ranking_answerability", "instruction": prompt, "poses": poses})
                annotations[sid] = {**ann, "family_id": family, "split": "geometry_pilot", "seed": int.from_bytes(hashlib.sha256(f"{PROTOCOL}:{sid}".encode()).digest()[:4], "big")}; index += 1
    suffix = "If exactly one requested target is identifiable and sufficiently visible, return exactly POINT [(x, y)], using one normalized image point with x and y between 0 and 1. If the request is ambiguous, the target is absent, or visual evidence is insufficient, return exactly ABSTAIN. Return no other text."
    output = {"point_regex": r"^POINT \[\(([0-9]*\.?[0-9]+), ([0-9]*\.?[0-9]+)\)\]$", "abstain_literal": "ABSTAIN", "allowed_actions": ["POINT", "ABSTAIN"], "point_state": "FOUND", "abstain_states": ["AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]}
    return ({"schema_version": 1, "protocol_id": PROTOCOL, "expected_scene_count": 32, "capture_mode": "no_manipulation_shadow", "random_seed": 13092026, "view_joint_pose": CAMERA["view_joint_pose"], "camera_frame": CAMERA["frame"], "base_frame": CAMERA["base_frame"], "coordinate_suffix": suffix, "objects": OBJECTS, "scenes": scenes}, {"schema_version": 1, "protocol_id": PROTOCOL, "oracle_usage": "geometry_qc_only", "semantic_label_source": "gazebo_evaluation_camera_not_vlm_input", "reference_frame": "image_viewer_left_to_right", "horizontal_coordinate": "normalized_centroid_x", "tie_margin_normalized": .03, "min_visible_evidence_px": 120, "states": list(STATES), "scenes": annotations}, {"schema_version": 1, "protocol_id": PROTOCOL, "frozen_before_capture": True, "parent_family_count": 32, "split_parent_family_count": {"geometry_pilot": 32}, "state_quota": {state: 8 for state in STATES}, "relation_variant_quota": {relation: 8 for relation in RELATIONS}, "state_relation_cell_quota": 2, "reference_frame": "image_viewer_left_to_right", "horizontal_coordinate": "normalized_centroid_x", "tie_margin_normalized": .03, "min_visible_evidence_px": 120, "insufficient_evidence_rule": {"min_visible_pixels_inclusive": 1, "max_visible_pixels_exclusive": 120, "projected_center_outer_margin_fraction": .20}, "output_contract": output, "camera": CAMERA, "model_input_allowlist": ["rgb", "depth_m", "instruction"], "evaluator_only": ["semantic_labels", "answerability_state", "target_id", "candidate_set"], "policies": {"no_sam2": True, "no_training": True, "no_b2": True, "no_test_access": True, "no_dev_v2_fit_or_selection": True}})


def text(value: dict) -> str: return yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=140)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--write", action="store_true"); parser.add_argument("--check", action="store_true"); args = parser.parse_args()
    if args.write == args.check: parser.error("choose --write or --check")
    payloads = build()
    if args.write:
        if any(p.exists() for p in OUT): raise FileExistsError("refusing to overwrite v2 pilot contract input")
        for p, value in zip(OUT, payloads): p.write_text(text(value), encoding="utf-8"); print(p)
    else:
        bad = [str(p) for p, value in zip(OUT, payloads) if not p.is_file() or p.read_text(encoding="utf-8") != text(value)]
        if bad: raise ValueError(bad)
        print("PASS: deterministic v2 geometry-pilot contract")


if __name__ == "__main__": main()
