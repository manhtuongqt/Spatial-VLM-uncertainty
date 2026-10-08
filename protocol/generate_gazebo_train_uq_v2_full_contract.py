#!/usr/bin/env python3
"""Generate immutable family-disjoint Train-UQ/Val-UQ v2 from PASS r6 templates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

import generate_gazebo_train_uq_v2_pilot_r5_contract as geometry

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ur3/ur3_perception/config"
PROTOCOL_ID = "gazebo_train_uq_v2_full"
SCENES_PATH = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS_PATH = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE_PATH = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
FAMILY_MANIFEST_PATH = ROOT / "protocol/gazebo_train_uq_v2_full_family_manifest.jsonl"
SPLIT_MANIFEST_PATH = ROOT / "protocol/gazebo_train_uq_v2_full_split_manifest.json"
OUTPUT_PATHS = (SCENES_PATH, ANNOTATIONS_PATH, GATE_PATH)
STATES = geometry.STATES
RELATIONS = geometry.RELATIONS
SPLITS = {"train_uq": 16, "val_uq": 4}


def seed_for(scene_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{PROTOCOL_ID}:{scene_id}".encode()).digest()[:4], "big")


def jitter_poses(poses: dict, state: str, repetition: int, split_index: int, relation: str) -> dict:
    result = copy.deepcopy(poses)
    code = repetition + 17 * split_index + 5 * RELATIONS.index(relation)
    if state == "INSUFFICIENT_EVIDENCE":
        # Target/screen geometry remains in its characterized band. Only
        # context evidence is diversified; pilot rows themselves are not reused.
        target_names = set(result) - {geometry.PANEL}
        # The target is always the object at the characterized central pose.
        target = min(target_names, key=lambda name: abs(result[name][0] + .220) + abs(result[name][1] - .300))
        for index, (name, pose) in enumerate(sorted(result.items())):
            if name in (target, geometry.PANEL):
                continue
            pose[0] += 0.010 + 0.0015 * ((code + index) % 5)
            pose[1] += 0.0015 * (((code + 2 * index) % 5) - 2)
            pose[2] += 0.021 * ((code + index) % 7)
        # Move within the evaluator-characterized transition, away from 1 px.
        offset_delta = (0.00010, 0.00020, 0.00030, 0.00040)[repetition % 4]
        panel_pose = result[geometry.PANEL]
        panel_pose[1] -= offset_delta
    else:
        dx = 0.0015 * (((code % 7) - 3))
        dy = 0.0018 * ((((code // 2) % 7) - 3))
        for index, pose in enumerate(result.values()):
            pose[0] += dx + 0.0004 * index
            pose[1] += dy
            pose[2] += 0.019 * ((code + index) % 9)
    return result


def build():
    pilot_scenes, pilot_oracle, pilot_gate = geometry.build()
    template_rows = pilot_scenes["scenes"]
    template_oracle = pilot_oracle["scenes"]
    templates = {}
    for index, row in enumerate(template_rows):
        item = template_oracle[row["scene_id"]]
        templates[(index // 16, item["state"], item["relation_variant"])] = (row, item)
    rows, oracle, family_rows = [], {}, []
    for split_index, (split, repetitions) in enumerate(SPLITS.items()):
        for repetition in range(repetitions):
            for state in STATES:
                for relation in RELATIONS:
                    template_rep = repetition % 2
                    source, source_item = templates[(template_rep, state, relation)]
                    index = len(rows)
                    sid = f"gazebo_uq_v2_full_{split}_{index:03d}"
                    family = f"spatial_vlm_uq_v2_full/{split}/parent_{index:03d}"
                    layout = f"uq_v2_full_layout_{index:03d}"
                    poses = jitter_poses(source["poses"], state, repetition, split_index, relation)
                    signature = geometry.layout_signature(poses)
                    rows.append({"scene_id": sid, "scene_family_id": family, "layout_id": layout,
                                 "layout_signature_sha256": signature,
                                 "task_type": "horizontal_ordinal_ranking_answerability",
                                 "instruction": source["instruction"], "poses": poses})
                    oracle[sid] = {**copy.deepcopy(source_item), "family_id": family, "layout_id": layout,
                                   "layout_signature_sha256": signature, "split": split, "seed": seed_for(sid),
                                   "template_provenance": "r6_PASS_geometry_class_only_not_pilot_data",
                                   "cell_repetition": repetition}
                    family_rows.append({"scene_id": sid, "family_id": family, "layout_id": layout,
                                        "layout_signature_sha256": signature, "split": split,
                                        "state": state, "relation_variant": relation,
                                        "deterministic_seed": seed_for(sid)})
    scenes = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "expected_scene_count": 320,
              "capture_mode": "no_manipulation_shadow", "random_seed": 14092038,
              "view_joint_pose": pilot_scenes["view_joint_pose"], "camera_frame": pilot_scenes["camera_frame"],
              "base_frame": pilot_scenes["base_frame"], "coordinate_suffix": pilot_scenes["coordinate_suffix"],
              "objects": pilot_scenes["objects"], "scenes": rows}
    annotations = {**copy.deepcopy(pilot_oracle), "protocol_id": PROTOCOL_ID,
                   "oracle_usage": "evaluation_only_open_after_prediction_lock", "scenes": oracle}
    gate = {**copy.deepcopy(pilot_gate), "protocol_id": PROTOCOL_ID, "parent_family_count": 320,
            "split_parent_family_count": {"train_uq": 256, "val_uq": 64},
            "state_quota": {"train_uq": {state: 64 for state in STATES},
                            "val_uq": {state: 16 for state in STATES}},
            "relation_variant_quota": {"train_uq": {relation: 64 for relation in RELATIONS},
                                       "val_uq": {relation: 16 for relation in RELATIONS}},
            "state_relation_cell_quota": {"train_uq": 16, "val_uq": 4},
            "full_dataset_only_after_pilot": {"protocol_id": "gazebo_train_uq_v2_pilot_r6",
                                               "required_geometry_qc": "32/32 PASS"},
            "inference_lock": {"backbone": "B0", "greedy_decoding": True, "max_new_tokens": 40,
                               "stochastic_draws": 3, "temperature": 0.7, "top_p": 0.9, "top_k": 50,
                               "seed_rule": "sha256(protocol_id:sample_id:model_id:draw)[0:4]",
                               "self_consistency_is_calibrated_probability": False},
            "risk_feature_schema": ["b0_action_one_hot", "b0_exact_contract", "b0_self_consistency",
                                    "stochastic_point_dispersion", "predicted_x", "predicted_y",
                                    "distance_to_image_boundary", "local_depth_valid_fraction",
                                    "local_depth_median", "local_depth_mad", "instruction_relation_one_hot"],
            "proposed_method": {"maximum_methods": 1, "type": "L2_regularized_logistic_regression",
                                "normalization_fit_split": "train_uq_only",
                                "C_grid_selected_on_val_uq": [0.01, 0.1, 1.0, 10.0],
                                "decision_threshold_grid": [round(i / 100, 2) for i in range(101)]},
            "selection_rule": {"selection_split": "val_uq_only",
                               "found_hit_delta_min": -0.125,
                               "nonfound_abstain_recall": "strictly_higher_than_b0",
                               "nonfound_false_accept": "strictly_lower_than_b0",
                               "exact_contract": "not_lower_than_b0",
                               "invalid_is_abstain": False, "aurc": "not_worse_than_b0",
                               "tie_break": ["lower_false_accept", "higher_found_hit", "simpler_method"]},
            "statistical_protocol": {"unit": "parent_family", "bootstrap_draws": 10000,
                                     "bootstrap_seed": 13092026, "confidence_interval": 0.95,
                                     "paired": True, "mcnemar": "exact_two_sided_when_applicable"},
            "sealed": {"gazebo_dev_answerability_v2": "one_confirmation_only_after_val_freeze",
                       "gazebo_calibration": True, "gazebo_test_iid": True, "gazebo_test_ood": True},
            "policies": {**pilot_gate["policies"], "no_training": False,
                         "no_model_inference": False, "no_materialization": False,
                         "full_train_uq_only_after_pilot_pass": True}}
    split_manifest = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PREREGISTERED",
                      "train_uq": {"families": 256, "cell_quota": 16},
                      "val_uq": {"families": 64, "cell_quota": 4},
                      "family_disjoint": True, "pilot_families_excluded": True}
    return scenes, annotations, gate, family_rows, split_manifest


def serialized(payload):
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--write", action="store_true"); parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write == args.check: parser.error("choose exactly one of --write or --check")
    scenes, annotations, gate, families, splits = build()
    if args.write:
        paths = (*OUTPUT_PATHS, FAMILY_MANIFEST_PATH, SPLIT_MANIFEST_PATH)
        if any(path.exists() for path in paths): raise FileExistsError("refusing to overwrite full v2 contract")
        for path, payload in zip(OUTPUT_PATHS, (scenes, annotations, gate)): path.write_text(serialized(payload), encoding="utf-8")
        FAMILY_MANIFEST_PATH.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in families), encoding="utf-8")
        SPLIT_MANIFEST_PATH.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "GENERATED", "families": len(families)}, indent=2))
    else:
        mismatches = [str(path) for path,payload in zip(OUTPUT_PATHS,(scenes,annotations,gate)) if not path.is_file() or path.read_text()!=serialized(payload)]
        expected_family = "".join(json.dumps(row, sort_keys=True) + "\n" for row in families)
        expected_split = json.dumps(splits, indent=2, sort_keys=True) + "\n"
        if not FAMILY_MANIFEST_PATH.is_file() or FAMILY_MANIFEST_PATH.read_text()!=expected_family: mismatches.append(str(FAMILY_MANIFEST_PATH))
        if not SPLIT_MANIFEST_PATH.is_file() or SPLIT_MANIFEST_PATH.read_text()!=expected_split: mismatches.append(str(SPLIT_MANIFEST_PATH))
        if mismatches: raise ValueError(f"contract drift: {mismatches}")
        print("PASS: deterministic 320-family full v2 contract")


if __name__ == "__main__": main()
