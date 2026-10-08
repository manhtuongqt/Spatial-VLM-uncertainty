#!/usr/bin/env python3
"""Audit and freeze the Calibration-v3 implementation before any preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

import generate_gazebo_calibration_v3_contract as generator


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json"
AUDIT = ROOT / "protocol/gazebo_calibration_v3_implementation_r3_audit.json"
LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R3.json"
CONTRACT_SHA256 = generator.CONTRACT_SHA256
CODE_AND_RUNTIME_SOURCES = (
    "protocol/generate_gazebo_calibration_v3_contract.py",
    "protocol/gazebo_calibration_v3_pipeline.py",
    "protocol/materialize_gazebo_calibration_v3.py",
    "protocol/gazebo_calibration_v3_infer.py",
    "protocol/gazebo_calibration_v3_fit.py",
    "protocol/freeze_gazebo_calibration_v3_implementation.py",
    "protocol/generate_gazebo_calibration_v2_contract.py",
    "protocol/generate_gazebo_calibration_v1_contract.py",
    "protocol/generate_gazebo_train_uq_v2_full_r3_contract.py",
    "protocol/generate_gazebo_train_uq_v2_full_contract.py",
    "protocol/generate_gazebo_train_uq_v2_pilot_r5_contract.py",
    "protocol/generate_gazebo_train_uq_v2_pilot_r4_contract.py",
    "protocol/generate_gazebo_train_uq_v1_contract.py",
    "protocol/materialize_gazebo_calibration_v1.py",
    "protocol/materialize_gazebo_train_uq_v1.py",
    "protocol/materialize_gazebo_train_uq_v2_full.py",
    "protocol/gazebo_train_uq_v1_infer.py",
    "protocol/spatial_risk_v2_development.py",
    "RoboRefer/API/api.py",
    "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_capture.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_runner.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_validate.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_evaluator.py",
    "ur3/ur3_perception/scripts/roborefer_grounder.py",
    "ur3/ur3_perception/scripts/spatial_point_utils.py",
    "ur3/ur3_perception/scripts/move_camera_to_view.py",
    "ur3/ur3_perception/schemas/roborefer_pilot_prediction.schema.json",
    "ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro",
    "ur3/ur3_moveit_control/launch/ur3_susgrip_sim.launch.py",
    "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
    "ur3/ur_simulation_gz/launch/ur_sim_control.launch.py",
)
GENERATED_SOURCES = (
    "protocol/gazebo_calibration_v3_family_manifest.jsonl",
    "protocol/gazebo_calibration_v3_split_manifest.json",
    "ur3/ur3_perception/config/gazebo_calibration_v3_scenes.yaml",
    "ur3/ur3_perception/config/gazebo_calibration_v3_annotations.yaml",
    "ur3/ur3_perception/config/gazebo_calibration_v3_gate.yaml",
)
FROZEN_UPSTREAM = (
    "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK.json",
    "protocol/gazebo_calibration_v3_implementation_audit.json",
    "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_R1_INVALID.json",
    "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R2.json",
    "protocol/gazebo_calibration_v3_implementation_r2_audit.json",
    "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_R2_INVALID.json",
    "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json",
    "protocol/spatial_risk_method_v2_hypothesis_lock.json",
    "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/grounding_model_lock.json",
    "results/spatial_vlm_refspatial_v1/wp5_spatial_risk_v2_development/spatial_risk_v2_estimator.joblib",
    "results/spatial_vlm_refspatial_v1/wp5_spatial_risk_v2_development/spatial_risk_v2_estimator.json",
    "results/spatial_vlm_refspatial_v1/wp5_spatial_risk_v2_development/spatial_risk_v2_development_decision.json",
)
PRIOR_DATASET_ORACLES = tuple(
    path
    for path in (
        ROOT / "datasets/Gazebo_dev/evaluator_ground_truth.jsonl",
        ROOT / "datasets/Gazebo_dev_answerability_v2/evaluator_ground_truth.jsonl",
        ROOT / "datasets/Gazebo_train_uq_v2_full_r3/evaluator_ground_truth.jsonl",
        ROOT / "datasets/Gazebo_calibration_v2/evaluator_ground_truth.jsonl",
    )
    if path.is_file()
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_strings(values: set[str] | list[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prior_registry() -> dict[str, set]:
    registry = {
        "sample_ids": set(),
        "scene_ids": set(),
        "family_ids": set(),
        "layout_ids": set(),
        "seeds": set(),
        "physical_layout_signatures": set(),
    }
    for path in sorted((ROOT / "protocol").glob("gazebo_*family_manifest.jsonl")):
        if path == generator.FAMILY_MANIFEST_PATH or "test" in path.name.lower():
            continue
        for row in read_jsonl(path):
            for source_key, target_key in (
                ("scene_id", "scene_ids"),
                ("family_id", "family_ids"),
                ("layout_id", "layout_ids"),
            ):
                if row.get(source_key) is not None:
                    registry[target_key].add(str(row[source_key]))
            if row.get("deterministic_seed") is not None:
                registry["seeds"].add(int(row["deterministic_seed"]))
    for path in PRIOR_DATASET_ORACLES:
        for row in read_jsonl(path):
            registry["sample_ids"].add(str(row["sample_id"]))
            registry["scene_ids"].add(str(row["scene_id"]))
            registry["family_ids"].add(str(row["family_id"]))
    for path in sorted((ROOT / "ur3/ur3_perception/config").glob("gazebo*_scenes.yaml")):
        if path == generator.SCENES_PATH or "test" in path.name.lower():
            continue
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        objects = config.get("objects", {})
        for row in config.get("scenes", []):
            if row.get("scene_id") is not None:
                registry["scene_ids"].add(str(row["scene_id"]))
            if row.get("scene_family_id") is not None:
                registry["family_ids"].add(str(row["scene_family_id"]))
            if row.get("layout_id") is not None:
                registry["layout_ids"].add(str(row["layout_id"]))
            if row.get("poses"):
                registry["physical_layout_signatures"].add(
                    generator.layout_signature(row, objects)
                )
    return registry


def audit() -> dict:
    if sha256(CONTRACT) != CONTRACT_SHA256:
        raise RuntimeError("Calibration-v3 contract hash drift")
    paths, expected_texts = generator.serialized_outputs()
    deterministic = {
        str(path.relative_to(ROOT)): path.is_file()
        and path.read_text(encoding="utf-8") == text
        for path, text in zip(paths, expected_texts)
    }
    scenes = yaml.safe_load(generator.SCENES_PATH.read_text(encoding="utf-8"))
    annotations = yaml.safe_load(generator.ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(generator.GATE_PATH.read_text(encoding="utf-8"))
    families = read_jsonl(generator.FAMILY_MANIFEST_PATH)
    split = json.loads(generator.SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = scenes["scenes"]
    oracle = annotations["scenes"]

    identities = {
        "sample_ids": {row["scene_id"] for row in rows},
        "scene_ids": {row["scene_id"] for row in rows},
        "family_ids": {row["scene_family_id"] for row in rows},
        "layout_ids": {row["layout_id"] for row in rows},
        "seeds": {int(row["deterministic_seed"]) for row in families},
        "physical_layout_signatures": {
            row["layout_signature_sha256"] for row in rows
        },
    }
    cell_counts = Counter(
        (item["state"], item["relation_variant"]) for item in oracle.values()
    )
    family_by_scene = {row["scene_id"]: row for row in families}
    signature_checks = {
        row["scene_id"]: (
            row["layout_signature_sha256"]
            == generator.layout_signature(row, scenes["objects"])
            == oracle[row["scene_id"]]["layout_signature_sha256"]
        )
        for row in rows
    }
    seed_checks = {
        row["scene_id"]: row["deterministic_seed"]
        == generator.seed_for(
            row["family_id"], row["state"], row["relation_variant"], row["cell_repetition"]
        )
        for row in families
    }
    ordered_families = sorted(families, key=lambda row: row["capture_order_sha256"])
    capture_order_checks = {
        "manifest_orders_are_0_to_127": [row["capture_order"] for row in ordered_families] == list(range(128)),
        "scene_config_follows_capture_order": [row["scene_id"] for row in rows]
        == [row["scene_id"] for row in ordered_families],
    }
    prior = prior_registry()
    overlap = {
        name: sorted(values & prior[name]) for name, values in identities.items()
    }
    uniqueness = {
        name: len(values) == 128 for name, values in identities.items()
    }
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    exclusion_hash_checks = {
        name: (ROOT / name).is_file() and sha256(ROOT / name) == digest
        for name, digest in contract["family_and_leakage_contract"]["exclusion_source_sha256"].items()
    }
    checks = {
        "contract_hash_matches": sha256(CONTRACT) == CONTRACT_SHA256,
        "deterministic_generated_files": all(deterministic.values()),
        "scene_count_128": len(rows) == 128,
        "oracle_count_128": len(oracle) == 128,
        "family_manifest_count_128": len(families) == 128,
        "scene_oracle_alignment": set(oracle) == {row["scene_id"] for row in rows},
        "family_scene_alignment": set(family_by_scene) == set(oracle),
        "cell_count_16": len(cell_counts) == 16,
        "every_cell_has_8": set(cell_counts.values()) == {8},
        "identities_and_signatures_unique": all(uniqueness.values()),
        "layout_signatures_recompute": all(signature_checks.values()),
        "seeds_recompute": all(seed_checks.values()),
        "capture_order_recomputes": all(
            row["capture_order_sha256"] == generator.capture_order(row["family_id"])
            for row in families
        ),
        "capture_order_valid": all(capture_order_checks.values()),
        "zero_prior_identity_seed_layout_overlap": all(not values for values in overlap.values()),
        "exclusion_source_hashes_match_contract": all(exclusion_hash_checks.values()),
        "split_manifest_seals_test": split.get("test_iid_ood_sealed") is True,
        "split_manifest_does_not_authorize_capture": split.get("capture_authorized") is False,
        "gate_seals_test_iid": gate["sealed"].get("gazebo_test_iid") is True,
        "gate_seals_test_ood": gate["sealed"].get("gazebo_test_ood") is True,
        "gate_forbids_refit": gate["policies"].get("no_grounding_or_risk_estimator_refit") is True,
        "gate_requires_post_preflight_capture_lock": gate["policies"].get("capture_requires_post_preflight_compatibility_lock") is True,
        "calibrator_is_affine_logit": gate["calibration_protocol"].get("calibrator") == "positive_slope_affine_logit",
        "calibrator_count_one": gate["calibration_protocol"].get("calibrator_count") == 1,
        "no_inherited_logistic_method_block": "proposed_method" not in gate,
        "no_inherited_train_pilot_dependency": "full_dataset_only_after_pilot" not in gate,
        "threshold_count_101": len(gate["operating_point_rule"]["threshold_grid"]) == 101,
        "bootstrap_contract_10000": gate.get("statistical_protocol", {}).get("bootstrap_draws") == 10000,
    }
    return {
        "schema_version": 1,
        "protocol_id": generator.PROTOCOL_ID,
        "status": "PASS" if all(checks.values()) else "REJECT",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_lock_sha256": sha256(CONTRACT),
        "checks": checks,
        "deterministic_file_checks": deterministic,
        "counts": {
            "families": len(families),
            "states": dict(Counter(item["state"] for item in oracle.values())),
            "relations": dict(Counter(item["relation_variant"] for item in oracle.values())),
            "state_relation_cells": {
                f"{state}|{relation}": count
                for (state, relation), count in sorted(cell_counts.items())
            },
        },
        "uniqueness": uniqueness,
        "overlap": overlap,
        "identity_set_sha256": {
            name: digest_strings(values) for name, values in identities.items()
        },
        "prior_registry_counts": {name: len(values) for name, values in prior.items()},
        "prior_registry_set_sha256": {
            name: digest_strings(values) for name, values in prior.items()
        },
        "exclusion_source_hash_checks": exclusion_hash_checks,
        "formal_static_preflight_run": False,
        "formal_live_preflight_run": False,
        "capture_performed": False,
        "model_inference_performed": False,
        "calibrator_fit_performed": False,
        "test_iid_ood_access": False,
    }


def freeze() -> None:
    if AUDIT.exists() or LOCK.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 audit or implementation lock")
    report = audit()
    write_json(AUDIT, report)
    if report["status"] != "PASS":
        raise RuntimeError(f"Calibration-v3 implementation audit rejected: {AUDIT}")
    source_names = CODE_AND_RUNTIME_SOURCES + GENERATED_SOURCES + FROZEN_UPSTREAM
    missing = [name for name in source_names if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"implementation source missing: {missing}")
    value = {
        "schema_version": 1,
        "protocol_id": generator.PROTOCOL_ID,
        "status": "IMPLEMENTATION_R3_FROZEN_BEFORE_STATIC_OR_LIVE_PREFLIGHT",
        "implementation_revision": "r3",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_lock_sha256": sha256(CONTRACT),
        "implementation_audit_path": str(AUDIT.relative_to(ROOT)),
        "implementation_audit_sha256": sha256(AUDIT),
        "parent_family_count": 128,
        "state_relation_cell_quota": 8,
        "family_manifest_sha256": sha256(generator.FAMILY_MANIFEST_PATH),
        "split_manifest_sha256": sha256(generator.SPLIT_MANIFEST_PATH),
        "physical_layout_signature_set_sha256": report["identity_set_sha256"]["physical_layout_signatures"],
        "source_artifact_sha256": {
            name: sha256(ROOT / name) for name in sorted(set(source_names))
        },
        "audit_summary": {
            "four_by_four_by_eight": True,
            "unique_ids_seeds_and_layouts": True,
            "zero_prior_family_scene_layout_seed_overlap": True,
            "deterministic_regeneration": True,
        },
        "formal_static_preflight_run": False,
        "formal_live_preflight_run": False,
        "capture_authorized": False,
        "capture_performed": False,
        "model_inference_performed": False,
        "calibrator_fit_performed": False,
        "test_iid_ood_access": False,
        "next_authorized_action": "Run Calibration-v3 static preflight only.",
    }
    write_json(LOCK, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "audit_sha256": sha256(AUDIT),
                "implementation_lock_sha256": sha256(LOCK),
            },
            indent=2,
        )
    )


def validate() -> None:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    if value.get("status") != "IMPLEMENTATION_R3_FROZEN_BEFORE_STATIC_OR_LIVE_PREFLIGHT":
        raise RuntimeError("invalid implementation lock")
    if value.get("implementation_audit_sha256") != sha256(AUDIT):
        raise RuntimeError("implementation audit hash drift")
    for name, digest in value["source_artifact_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"implementation source drift: {name}")
    if audit()["status"] != "PASS":
        raise RuntimeError("current implementation audit no longer passes")
    print("PASS: Calibration-v3 implementation lock and sources")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "validate"))
    command = parser.parse_args().command
    if command == "freeze":
        freeze()
    else:
        validate()


if __name__ == "__main__":
    main()
