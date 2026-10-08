#!/usr/bin/env python3
"""Generate a fresh, family-disjoint 128-family final-calibration contract."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

import generate_gazebo_train_uq_v2_full_r3_contract as source


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ur3/ur3_perception/config"
PROTOCOL_ID = "gazebo_calibration_v1"
SCENES_PATH = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS_PATH = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE_PATH = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
FAMILY_MANIFEST_PATH = ROOT / "protocol/gazebo_calibration_v1_family_manifest.jsonl"
SPLIT_MANIFEST_PATH = ROOT / "protocol/gazebo_calibration_v1_split_manifest.json"
OUTPUT_PATHS = (SCENES_PATH, ANNOTATIONS_PATH, GATE_PATH)
STATES = source.STATES
RELATIONS = source.RELATIONS
REPETITIONS = 8
PANEL = source.PANEL
TARGET_IE = source.MANGO


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def seed_for(scene_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{PROTOCOL_ID}:{scene_id}".encode()).digest()[:4], "big"
    )


def fresh_poses(poses: dict, state: str, repetition: int, relation_index: int) -> dict:
    """Create a fresh signature while preserving the full-r3 PASS geometry class."""
    result = copy.deepcopy(poses)
    code = 11 * repetition + 3 * relation_index
    if state == "INSUFFICIENT_EVIDENCE":
        # Keep the empirically stable 31--92 px mango/panel pair exact.  Move
        # only contextual objects, which makes every layout new without using
        # visibility labels to tune a post-capture scene.
        for index, name in enumerate(sorted(set(result) - {TARGET_IE, PANEL})):
            result[name][0] += 0.0045 + 0.0007 * ((code + index) % 5)
            result[name][1] += 0.0012 * (((code + 2 * index) % 5) - 2)
            result[name][2] += 0.031 + 0.009 * ((code + index) % 4)
    else:
        # A common translation preserves ordinal/tie geometry.  Per-object yaw
        # provides additional rendered diversity without changing relation GT.
        dx = 0.0015 * ((repetition % 5) - 2)
        dy = 0.0020 * (((repetition + relation_index) % 5) - 2)
        for index, pose in enumerate(result.values()):
            pose[0] += dx
            pose[1] += dy
            pose[2] += 0.027 + 0.007 * ((code + index) % 5)
    return result


def build():
    prior_scenes, prior_annotations, prior_gate, _, _ = copy.deepcopy(source.build())
    source_rows = prior_scenes["scenes"][: REPETITIONS * len(STATES) * len(RELATIONS)]
    source_oracle = prior_annotations["scenes"]
    rows, oracle, families = [], {}, []
    for index, old_row in enumerate(source_rows):
        old_item = source_oracle[old_row["scene_id"]]
        state = old_item["state"]
        relation = old_item["relation_variant"]
        repetition = index // (len(STATES) * len(RELATIONS))
        relation_index = RELATIONS.index(relation)
        sid = f"gazebo_calibration_v1_{index:03d}"
        family = f"spatial_vlm_gazebo_calibration_v1/calibration/parent_{index:03d}"
        layout = f"gazebo_calibration_v1_layout_{index:03d}"
        poses = fresh_poses(old_row["poses"], state, repetition, relation_index)
        signature = source.v1.geometry.layout_signature(poses)
        row = copy.deepcopy(old_row)
        row.update(
            scene_id=sid,
            scene_family_id=family,
            layout_id=layout,
            layout_signature_sha256=signature,
            poses=poses,
        )
        item = copy.deepcopy(old_item)
        item.update(
            family_id=family,
            layout_id=layout,
            layout_signature_sha256=signature,
            split="calibration",
            seed=seed_for(sid),
            cell_repetition=repetition,
            geometry_provenance="fresh_transform_of_locked_full_r3_PASS_geometry_class",
        )
        rows.append(row)
        oracle[sid] = item
        families.append(
            {
                "scene_id": sid,
                "family_id": family,
                "layout_id": layout,
                "layout_signature_sha256": signature,
                "split": "calibration",
                "state": state,
                "relation_variant": relation,
                "cell_repetition": repetition,
                "deterministic_seed": seed_for(sid),
            }
        )

    scenes = copy.deepcopy(prior_scenes)
    scenes.update(
        protocol_id=PROTOCOL_ID,
        expected_scene_count=128,
        random_seed=14092041,
        scenes=rows,
    )
    annotations = copy.deepcopy(prior_annotations)
    annotations.update(
        protocol_id=PROTOCOL_ID,
        oracle_usage="calibrator_and_operating_threshold_fit_only_after_prediction_lock",
        scenes=oracle,
    )
    gate = copy.deepcopy(prior_gate)
    gate.update(
        protocol_id=PROTOCOL_ID,
        parent_family_count=128,
        split_parent_family_count={"calibration": 128},
        state_quota={"calibration": {state: 32 for state in STATES}},
        relation_variant_quota={"calibration": {relation: 32 for relation in RELATIONS}},
        state_relation_cell_quota={"calibration": REPETITIONS},
        calibration_protocol={
            "input_score": "frozen_wp5_estimated_unsafe_probability",
            "calibrator_count": 1,
            "calibrator": "binary_temperature_scaling_on_clipped_logit",
            "formula": "sigmoid(logit(clip(p_raw,1e-6,1-1e-6))/T)",
            "fit_objective": "binary_negative_log_likelihood",
            "parameter": "T_positive_scalar_only",
            "optimizer": "deterministic_golden_section_on_log_T",
            "log_T_bounds": [-5.0, 5.0],
            "iterations": 200,
            "probability_gate": "ECE-10 and Brier must both strictly decrease versus raw frozen WP5 risk",
            "ranking_invariance": "AURC and AUROC-error must be unchanged within 1e-12",
        },
        operating_point_rule={
            "threshold_grid": [round(i / 100, 2) for i in range(101)],
            "action": "ABSTAIN iff calibrated_risk >= threshold; otherwise preserve frozen B0 action",
            "eligibility_vs_frozen_b0": {
                "found_hit_at_008_delta_min": -0.125,
                "nonfound_abstain_recall": "strictly_higher",
                "nonfound_false_accept": "strictly_lower",
                "exact_contract": "not_lower",
                "invalid_is_abstain": False,
            },
            "selection_lexicographic": [
                "lowest_nonfound_false_accept",
                "highest_found_hit_at_008",
                "highest_corrected_safe_task_accuracy",
                "highest_coverage",
                "lowest_threshold",
            ],
        },
        sealed={
            "gazebo_train_uq_and_val_uq": "read_only_for_hash_and_frozen_estimator_provenance",
            "gazebo_dev_answerability_v2": "closed_no_reuse_after_single_confirmation",
            "gazebo_test_iid": True,
            "gazebo_test_ood": True,
        },
        policies={
            **prior_gate["policies"],
            "no_training": True,
            "no_model_inference_during_capture": True,
            "no_materialization_before_128_of_128_geometry_pass": True,
            "no_grounding_or_risk_estimator_refit": True,
            "no_robot_manipulation": True,
            "no_test_access": True,
        },
    )
    split_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PREREGISTERED",
        "calibration": {"families": 128, "state_relation_cell_quota": REPETITIONS},
        "family_disjoint_from": [
            "Gazebo_train_uq",
            "Gazebo_val_uq",
            "Gazebo_dev_answerability_v2",
            "all Gazebo UQ pilots and failed full attempts",
        ],
        "test_iid_ood_sealed": True,
    }
    return scenes, annotations, gate, families, split_manifest


def serialized(payload: dict) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write == args.check:
        parser.error("choose exactly one of --write or --check")
    values = build()
    texts = [serialized(x) for x in values[:3]] + [
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in values[3]),
        json.dumps(values[4], indent=2, sort_keys=True) + "\n",
    ]
    paths = [*OUTPUT_PATHS, FAMILY_MANIFEST_PATH, SPLIT_MANIFEST_PATH]
    if args.write:
        if any(path.exists() for path in paths):
            raise FileExistsError("refusing to overwrite calibration-v1 contract inputs")
        for path, text in zip(paths, texts):
            path.write_text(text, encoding="utf-8")
        print(json.dumps({"status": "GENERATED", "families": len(values[3])}, indent=2))
    else:
        drift = [str(path) for path, text in zip(paths, texts) if not path.is_file() or path.read_text() != text]
        if drift:
            raise ValueError(f"calibration contract drift: {drift}")
        print("PASS: deterministic fresh 128-family calibration-v1 contract")


if __name__ == "__main__":
    main()
