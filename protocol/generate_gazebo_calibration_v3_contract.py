#!/usr/bin/env python3
"""Generate the deterministic 128-family Calibration-v3 implementation inputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

import generate_gazebo_calibration_v2_contract as source


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ur3/ur3_perception/config"
PROTOCOL_ID = "gazebo_calibration_v3"
CONTRACT = ROOT / "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json"
SCENES_PATH = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS_PATH = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE_PATH = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
FAMILY_MANIFEST_PATH = ROOT / "protocol/gazebo_calibration_v3_family_manifest.jsonl"
SPLIT_MANIFEST_PATH = ROOT / "protocol/gazebo_calibration_v3_split_manifest.json"
OUTPUT_PATHS = (SCENES_PATH, ANNOTATIONS_PATH, GATE_PATH)
STATES = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
RELATIONS = ["leftmost", "rightmost", "second_from_left", "second_from_right"]
REPETITIONS = 8
PANEL = source.v1.PANEL
TARGET_IE = source.v1.TARGET_IE
WORLD_ID = "ur3_pick_place_uq_occlusion_v2.sdf"
CONTRACT_SHA256 = "99b430b1c55e1762f8460db4e8134b4a8782403e466d71ea86ea3da847bfc841"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def seed_for(family_id: str, state: str, relation: str, repetition: int) -> int:
    payload = (
        f"gazebo_calibration_v3_seed_v1|{family_id}|{state}|{relation}|{repetition}"
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def capture_order(family_id: str) -> str:
    return hashlib.sha256(
        f"gazebo_calibration_v3_capture_order_v1|{family_id}".encode("utf-8")
    ).hexdigest()


def physical_layout_payload(scene: dict, objects: dict) -> dict:
    canonical_objects = []
    for name in sorted(scene["poses"]):
        x, y, yaw = scene["poses"][name]
        z = objects.get(name, {}).get("z", 0.0)
        canonical_objects.append(
            {
                "model": name,
                "pose_xyzrpy": [
                    round(float(x), 6),
                    round(float(y), 6),
                    round(float(z), 6),
                    0.0,
                    0.0,
                    round(float(yaw), 6),
                ],
            }
        )
    return {
        "world": WORLD_ID,
        "camera": {
            "frame": "camera_color_optical_frame",
            "base_frame": "base_link",
            "resolution": [640, 480],
            "intrinsics": {
                "fx": 606.0816650391,
                "fy": 605.7973022461,
                "cx": 325.5436706543,
                "cy": 249.9961547852,
            },
            "view_joint_pose": [
                1.3815,
                -1.8273,
                1.3428,
                -1.2863,
                -1.5708,
                -1.9601,
            ],
        },
        "objects": canonical_objects,
    }


def layout_signature(scene: dict, objects: dict) -> str:
    payload = json.dumps(
        physical_layout_payload(scene, objects),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256_bytes(payload)


def fresh_poses(
    poses: dict, state: str, state_index: int, relation_index: int, repetition: int
) -> dict:
    """Make a new physical layout without changing the locked geometry class."""
    result = copy.deepcopy(poses)
    code = state_index * 97 + relation_index * 23 + repetition * 7
    if state == "INSUFFICIENT_EVIDENCE":
        # Preserve the empirically accepted target/occluder pair. Only context
        # objects move, so no post-capture visibility tuning is introduced.
        for offset, name in enumerate(sorted(set(result) - {TARGET_IE, PANEL})):
            result[name][0] += 0.0060 + 0.0008 * ((code + offset) % 5)
            result[name][1] += 0.0015 * (((code + 2 * offset) % 5) - 2)
            result[name][2] += 0.0700 + 0.0110 * ((code + offset) % 7)
    else:
        # A common translation preserves ordinal/tie relationships. Yaw changes
        # add rendering diversity while leaving object centers unchanged.
        dx = 0.0040 + 0.0007 * ((code + repetition) % 5)
        dy = -0.0050 + 0.0014 * ((code + relation_index) % 7)
        for offset, name in enumerate(sorted(result)):
            result[name][0] += dx
            result[name][1] += dy
            result[name][2] += 0.0550 + 0.0090 * ((code + offset) % 9)
    return result


def build() -> tuple[dict, dict, dict, list[dict], dict]:
    prior_scenes, prior_annotations, prior_gate, _, _ = copy.deepcopy(source.build())
    source_by_cell: dict[tuple[str, str, int], tuple[dict, dict]] = {}
    for scene in prior_scenes["scenes"]:
        oracle = prior_annotations["scenes"][scene["scene_id"]]
        key = (
            oracle["state"],
            oracle["relation_variant"],
            int(oracle["cell_repetition"]),
        )
        source_by_cell[key] = (scene, oracle)

    rows: list[dict] = []
    oracle_rows: dict[str, dict] = {}
    families: list[dict] = []
    objects = copy.deepcopy(prior_scenes["objects"])
    for state_index, state in enumerate(STATES):
        for relation_index, relation in enumerate(RELATIONS):
            for repetition in range(REPETITIONS):
                index = ((state_index * 4 + relation_index) * 8) + repetition
                old_scene, old_oracle = source_by_cell[(state, relation, repetition)]
                scene_id = f"gazebo_calibration_v3_{index:03d}"
                family_id = (
                    f"spatial_vlm_gazebo_calibration_v3/calibration/parent_{index:03d}"
                )
                layout_id = f"gazebo_calibration_v3_layout_{index:03d}"
                scene = copy.deepcopy(old_scene)
                scene["poses"] = fresh_poses(
                    old_scene["poses"], state, state_index, relation_index, repetition
                )
                scene.update(
                    scene_id=scene_id,
                    scene_family_id=family_id,
                    layout_id=layout_id,
                )
                signature = layout_signature(scene, objects)
                scene["layout_signature_sha256"] = signature
                item = copy.deepcopy(old_oracle)
                item.update(
                    family_id=family_id,
                    layout_id=layout_id,
                    layout_signature_sha256=signature,
                    split="calibration",
                    seed=seed_for(family_id, state, relation, repetition),
                    cell_repetition=repetition,
                    geometry_provenance=(
                        "fresh_v3_transform_of_calibration_v2_pass_geometry_class_"
                        "before_capture_no_outcome_tuning"
                    ),
                )
                rows.append(scene)
                oracle_rows[scene_id] = item
                families.append(
                    {
                        "family_index": index,
                        "scene_id": scene_id,
                        "family_id": family_id,
                        "layout_id": layout_id,
                        "layout_signature_sha256": signature,
                        "split": "calibration",
                        "state": state,
                        "relation_variant": relation,
                        "cell_repetition": repetition,
                        "deterministic_seed": seed_for(
                            family_id, state, relation, repetition
                        ),
                        "capture_order_sha256": capture_order(family_id),
                    }
                )

    order_by_family = {
        row["family_id"]: order
        for order, row in enumerate(
            sorted(families, key=lambda value: value["capture_order_sha256"])
        )
    }
    for family in families:
        family["capture_order"] = order_by_family[family["family_id"]]
    rows.sort(key=lambda row: order_by_family[row["scene_family_id"]])

    scenes = copy.deepcopy(prior_scenes)
    scenes.update(
        protocol_id=PROTOCOL_ID,
        expected_scene_count=128,
        random_seed=15092026,
        contract_lock_sha256=CONTRACT_SHA256,
        capture_order="sha256_sort_gazebo_calibration_v3_capture_order_v1",
        scenes=rows,
    )
    annotations = copy.deepcopy(prior_annotations)
    annotations.update(
        protocol_id=PROTOCOL_ID,
        contract_lock_sha256=CONTRACT_SHA256,
        oracle_usage=(
            "affine_logit_calibrator_and_operating_threshold_only_after_"
            "frozen_v2_raw_prediction_lock"
        ),
        scenes=oracle_rows,
    )
    gate = copy.deepcopy(prior_gate)
    # These blocks describe the predecessor's development-time Logistic-L2
    # search and pilot dependency. They are inapplicable and contradictory once
    # the already-frozen HGB v2 estimator enters independent calibration.
    gate.pop("proposed_method", None)
    gate.pop("full_dataset_only_after_pilot", None)
    gate.update(
        protocol_id=PROTOCOL_ID,
        contract_lock_sha256=CONTRACT_SHA256,
        parent_family_count=128,
        split_parent_family_count={"calibration": 128},
        state_quota={"calibration": {state: 32 for state in STATES}},
        relation_variant_quota={
            "calibration": {relation: 32 for relation in RELATIONS}
        },
        state_relation_cell_quota={"calibration": REPETITIONS},
        risk_feature_schema=[
            "action_POINT",
            "action_INVALID",
            "self_consistency",
            "point_dispersion",
            "boundary_distance",
            "local_depth_valid_fraction",
            "local_depth_median",
            "local_depth_mad",
            "relation_leftmost",
            "relation_rightmost",
            "relation_second_from_left",
            "relation_second_from_right",
        ],
        frozen_risk_estimator={
            "type": "sklearn.ensemble.HistGradientBoostingClassifier",
            "joblib_sha256": "f296e2e33abdd66437d2976a91ff78b4e707d67244f3b998dc9d9b011cbd5bc2",
            "refit_allowed": False,
        },
        calibration_protocol={
            "input_score": "frozen_spatial_risk_v2_raw_probability",
            "calibrator_count": 1,
            "calibrator": "positive_slope_affine_logit",
            "formula": (
                "p=clip(p_raw,1e-6,1-1e-6); z=logit(p); "
                "p_cal=sigmoid(exp(log_a)*z+b)"
            ),
            "fit_objective": "mean_binary_negative_log_likelihood",
            "optimizer": "scipy.optimize.minimize_L-BFGS-B",
            "initial_parameters": {"log_a": 0.0, "b": 0.0},
            "bounds": {"log_a": [-5.0, 5.0], "b": [-10.0, 10.0]},
            "optimizer_options": {
                "maxiter": 1000,
                "ftol": 1e-12,
                "gtol": 1e-8,
            },
            "probability_gate": "ECE-10 and Brier both strictly decrease",
            "ranking_invariance": (
                "AURC and AUROC-error absolute difference at most 1e-12"
            ),
        },
        operating_point_rule={
            "threshold_grid": [round(index / 100, 2) for index in range(101)],
            "action": (
                "ABSTAIN iff calibrated_risk >= threshold; otherwise preserve "
                "frozen B0 action"
            ),
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
            "train_uq": "read_only_hash_and_frozen_provenance",
            "val_uq": "closed_for_selection",
            "gazebo_dev_answerability_v2": "closed_no_reuse",
            "gazebo_calibration_v2": "closed_historical_negative_no_reuse",
            "gazebo_test_iid": True,
            "gazebo_test_ood": True,
            "robot": True,
        },
        policies={
            **prior_gate["policies"],
            "no_training": True,
            "no_model_inference_during_capture": True,
            "no_materialization_before_128_of_128_geometry_pass": True,
            "no_grounding_or_risk_estimator_refit": True,
            "no_robot_manipulation": True,
            "no_test_access": True,
            "capture_requires_post_preflight_compatibility_lock": True,
        },
    )
    split_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PREREGISTERED_IMPLEMENTATION_INPUT",
        "contract_lock_sha256": CONTRACT_SHA256,
        "calibration": {
            "families": 128,
            "states": 4,
            "relations": 4,
            "state_relation_cell_quota": REPETITIONS,
        },
        "family_disjoint_from": [
            "Gazebo_train_uq and Gazebo_val_uq including failed attempts/pilots",
            "Gazebo_dev_answerability_v2",
            "Gazebo_calibration_v1 and Gazebo_calibration_v2",
            "all reserved or materialized Test-IID/Test-OOD namespaces",
        ],
        "capture_order_rule": "sha256_sort_gazebo_calibration_v3_capture_order_v1",
        "test_iid_ood_sealed": True,
        "capture_authorized": False,
    }
    return scenes, annotations, gate, families, split_manifest


def serialized_yaml(payload: dict) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140)


def serialized_outputs() -> tuple[list[Path], list[str]]:
    values = build()
    texts = [serialized_yaml(value) for value in values[:3]]
    texts.extend(
        [
            "".join(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                for row in values[3]
            ),
            json.dumps(values[4], indent=2, sort_keys=True, allow_nan=False) + "\n",
        ]
    )
    return [*OUTPUT_PATHS, FAMILY_MANIFEST_PATH, SPLIT_MANIFEST_PATH], texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write == args.check:
        parser.error("choose exactly one of --write or --check")
    if hashlib.sha256(CONTRACT.read_bytes()).hexdigest() != CONTRACT_SHA256:
        raise RuntimeError("Calibration-v3 contract lock hash drift")
    paths, texts = serialized_outputs()
    if args.write:
        if any(path.exists() for path in paths):
            raise FileExistsError("refusing to overwrite Calibration-v3 implementation inputs")
        for path, text in zip(paths, texts):
            path.write_text(text, encoding="utf-8")
        print(json.dumps({"status": "GENERATED", "families": 128}, indent=2))
    else:
        drift = [
            str(path)
            for path, text in zip(paths, texts)
            if not path.is_file() or path.read_text(encoding="utf-8") != text
        ]
        if drift:
            raise RuntimeError(f"Calibration-v3 deterministic drift: {drift}")
        print("PASS: deterministic Calibration-v3 implementation inputs")


if __name__ == "__main__":
    main()
