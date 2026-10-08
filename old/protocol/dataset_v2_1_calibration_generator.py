#!/usr/bin/env python3
"""Build the independent, capture-only Dataset V2.1 calibration design.

The observable-geometry implementation is reused from the qualified V2.1
development generator.  Development families are composition templates only:
every calibration family receives a fresh ID, seed bundle, instruction,
capture ID and deterministic layout.  This module never starts ROS/Gazebo and
never fits or evaluates a model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_v2_1_development_generator as geometry_generator  # noqa: E402
from dataset_v2_1_development_generator import (  # noqa: E402
    collect_seed_values,
    collect_values,
)
from dataset_v2_pilot_generator import COORDINATE_SUFFIX, object_registry  # noqa: E402
from dataset_v2_relation_geometry_v2_1 import (  # noqa: E402
    GEOMETRY_VERSION,
    PREDICTOR_SCREENING_MARGIN_M,
)
from dataset_v2_relation_repair_generator import (  # noqa: E402
    BETWEEN_DEPTH_SCREENING_MARGIN_M,
)
from validate_dataset_expansion_v2 import VARIANTS  # noqa: E402
from wp2_common import canonical_json_sha256, read_json, sha256_file, write_json  # noqa: E402


PROTOCOL_ID = "roborefer_dataset_v2_1_calibration_capture_200"
GENERATOR_VERSION = "dataset_v2_1_calibration_generator_v1.0.0"
MASTER_SEED = 240820263001
FAMILY_PREFIX = "v211cal_family_"
FAMILY_COUNT = 200
CAPTURE_COUNT = 400
SAMPLE_COUNT = 1000
OUTPUT_ROOT = "datasets/roborefer_dataset_v2_1_calibration_200_20260824"
WORLD_FILE = geometry_generator.WORLD_FILE
CAMERA_LOCK = geometry_generator.CAMERA_LOCK
DEPTH_CALIBRATION = geometry_generator.CALIBRATION
DEVELOPMENT_TEMPLATE = "protocol/dataset_v2_1_development_manifest.json"
PARTITION = geometry_generator.PARTITION
ARCHITECTURE_FREEZE_LOCK = "protocol/pcra_u_architecture_freeze_lock.json"

EXPECTED_STATE = {
    "FOUND": 80,
    "INSUFFICIENT_EVIDENCE": 50,
    "AMBIGUOUS": 35,
    "ABSENT": 35,
}
EXPECTED_CATEGORY = {
    "direct_grounding": 30,
    "front_behind_camera": 30,
    "multi_anchor_depth_order": 30,
    "nearer_farther": 35,
    "occlusion_depth_evidence": 35,
    "relation_2d": 40,
}

# One half of every state/category cell, with the six odd residuals assigned
# deterministically so both the state and category marginals are exact.
CELL_QUOTAS = {
    ("FOUND", "direct_grounding"): 13,
    ("FOUND", "front_behind_camera"): 15,
    ("FOUND", "multi_anchor_depth_order"): 9,
    ("FOUND", "nearer_farther"): 13,
    ("FOUND", "occlusion_depth_evidence"): 14,
    ("FOUND", "relation_2d"): 16,
    ("INSUFFICIENT_EVIDENCE", "direct_grounding"): 6,
    ("INSUFFICIENT_EVIDENCE", "front_behind_camera"): 5,
    ("INSUFFICIENT_EVIDENCE", "multi_anchor_depth_order"): 11,
    ("INSUFFICIENT_EVIDENCE", "nearer_farther"): 11,
    ("INSUFFICIENT_EVIDENCE", "occlusion_depth_evidence"): 8,
    ("INSUFFICIENT_EVIDENCE", "relation_2d"): 9,
    ("AMBIGUOUS", "direct_grounding"): 7,
    ("AMBIGUOUS", "front_behind_camera"): 6,
    ("AMBIGUOUS", "multi_anchor_depth_order"): 6,
    ("AMBIGUOUS", "nearer_farther"): 4,
    ("AMBIGUOUS", "occlusion_depth_evidence"): 8,
    ("AMBIGUOUS", "relation_2d"): 4,
    ("ABSENT", "direct_grounding"): 4,
    ("ABSENT", "front_behind_camera"): 4,
    ("ABSENT", "multi_anchor_depth_order"): 4,
    ("ABSENT", "nearer_farther"): 7,
    ("ABSENT", "occlusion_depth_evidence"): 5,
    ("ABSENT", "relation_2d"): 11,
}

# These are read-only historical sources.  The new calibration artifact paths
# are intentionally absent so deterministic replay remains possible.
DENYLIST_JSON = tuple(dict.fromkeys((*geometry_generator.DENYLIST_JSON,
    "protocol/dataset_v2_1_development_manifest.json",
    "protocol/dataset_v2_1_development_capture_plan.json",
    "datasets/roborefer_dataset_v2_1_1_development_400_20260824/family_manifest.json",
    "datasets/roborefer_dataset_v2_1_1_development_400_20260824/capture_plan.json",
    "datasets/roborefer_dataset_v2_1_1_development_400_20260824/dataset_index.json",
    "datasets/roborefer_dataset_v2_1_1_development_400_20260824/raw/raw_capture_manifest.json",
    "protocol/pcra_u_development_train_manifest.json",
    "protocol/pcra_u_development_eval_manifest.json",
)))


class CalibrationGenerationError(RuntimeError):
    """Raised when an exact independent calibration design cannot be built."""


def _bind_geometry_namespace() -> None:
    """Set only module-local generation policy used by reused pure functions."""

    geometry_generator.PROTOCOL_ID = PROTOCOL_ID
    geometry_generator.MASTER_SEED = MASTER_SEED
    geometry_generator.FAMILY_PREFIX = FAMILY_PREFIX


def load_calibration_denylists(workspace: Path) -> dict[str, set[Any]]:
    result: dict[str, set[Any]] = {
        "family_ids": set(),
        "capture_ids": set(),
        "seeds": set(),
        "instructions": set(),
        "template_ids": set(),
        "layout_fingerprints": set(),
    }
    for relative in DENYLIST_JSON:
        path = workspace / relative
        if not path.is_file():
            continue
        payload = read_json(path)
        result["family_ids"].update(collect_values(payload, {"family_id"}))
        result["capture_ids"].update(collect_values(payload, {"capture_id"}))
        result["seeds"].update(collect_seed_values(payload))
        result["instructions"].update(collect_values(payload, {"instruction"}))
        result["template_ids"].update(
            collect_values(payload, {"language_template_family_id"})
        )
        result["layout_fingerprints"].update(collect_values(
            payload,
            {"layout_instance_fingerprint_sha256", "layout_fingerprint_sha256"},
        ))
    return result


def select_templates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for (state, category), amount in sorted(CELL_QUOTAS.items()):
        candidates = [
            row for row in payload["families"]
            if row["primary_answerability_stratum"] == state
            and row["family_category"] == category
        ]
        candidates.sort(key=lambda row: hashlib.sha256(
            f"{PROTOCOL_ID}|{MASTER_SEED}|template|{row['family_id']}".encode()
        ).hexdigest())
        if len(candidates) < amount:
            raise CalibrationGenerationError(
                f"template cell exhausted: state={state} category={category}"
            )
        selected.extend(candidates[:amount])
    selected.sort(key=lambda row: hashlib.sha256(
        f"{PROTOCOL_ID}|{MASTER_SEED}|order|{row['family_id']}".encode()
    ).hexdigest())
    if len(selected) != FAMILY_COUNT or len({row["family_id"] for row in selected}) != FAMILY_COUNT:
        raise CalibrationGenerationError("template selection is not exactly 200 unique families")
    return selected


def allocate_batches(families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Canary: five families per answerability state and all six categories.
    selected: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for category in sorted(EXPECTED_CATEGORY):
        available = [
            row for row in families
            if row not in selected and row["family_category"] == category
            and state_counts[row["primary_answerability_stratum"]] < 5
        ]
        available.sort(key=lambda row: (
            state_counts[row["primary_answerability_stratum"]],
            hashlib.sha256(f"{MASTER_SEED}|canary-category|{row['family_id']}".encode()).hexdigest(),
        ))
        if not available:
            raise CalibrationGenerationError(f"cannot cover canary category: {category}")
        selected.append(available[0])
        state_counts[available[0]["primary_answerability_stratum"]] += 1
        category_counts[category] += 1
    for state in EXPECTED_STATE:
        available = [
            row for row in families
            if row not in selected and row["primary_answerability_stratum"] == state
        ]
        available.sort(key=lambda row: (
            category_counts[row["family_category"]],
            hashlib.sha256(f"{MASTER_SEED}|canary-state|{row['family_id']}".encode()).hexdigest(),
        ))
        need = 5 - state_counts[state]
        if len(available) < need:
            raise CalibrationGenerationError(f"cannot fill canary state: {state}")
        for row in available[:need]:
            selected.append(row)
            state_counts[state] += 1
            category_counts[row["family_category"]] += 1
    if len(selected) != 20 or set(state_counts.values()) != {5} or len(category_counts) != 6:
        raise CalibrationGenerationError(
            f"invalid calibration canary: states={dict(state_counts)} categories={dict(category_counts)}"
        )
    selected_ids = {row["family_id"] for row in selected}
    remainder = sorted(
        [row for row in families if row["family_id"] not in selected_ids],
        key=lambda row: hashlib.sha256(
            f"{MASTER_SEED}|post-canary|{row['family_id']}".encode()
        ).hexdigest(),
    )
    chunks = [selected, remainder[:90], remainder[90:]]
    if [len(chunk) for chunk in chunks] != [20, 90, 90]:
        raise CalibrationGenerationError("batch sizes differ from 20 + 90 + 90")
    batches = []
    for order, chunk in enumerate(chunks):
        batch_id = "canary_000" if order == 0 else f"batch_{order:03d}"
        batches.append({
            "batch_id": batch_id,
            "order": order,
            "family_count": len(chunk),
            "planned_capture_count": 2 * len(chunk),
            "family_ids": [row["family_id"] for row in chunk],
            "split_counts": {"calibration": len(chunk)},
            "primary_answerability_counts": dict(sorted(Counter(
                row["primary_answerability_stratum"] for row in chunk
            ).items())),
            "family_category_counts": dict(sorted(Counter(
                row["family_category"] for row in chunk
            ).items())),
            "prerequisite_batch": None if order == 0 else (
                "canary_000" if order == 1 else f"batch_{order - 1:03d}"
            ),
            "next_batch_requires_raw_qc_pass": True,
        })
    return batches


def build_artifacts(workspace: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _bind_geometry_namespace()
    template = read_json(workspace / DEVELOPMENT_TEMPLATE)
    partition = read_json(workspace / PARTITION)
    camera_lock = read_json(workspace / CAMERA_LOCK)
    depth_calibration = read_json(workspace / DEPTH_CALIBRATION)
    freeze_lock = read_json(workspace / ARCHITECTURE_FREEZE_LOCK)
    if (
        template.get("family_count") != 400
        or not freeze_lock.get("architecture_frozen")
        or not freeze_lock.get("checkpoint_frozen")
        or freeze_lock.get("allowed_next_action") != "CAPTURE_CALIBRATION"
        or freeze_lock.get("test_opened") is not False
    ):
        raise CalibrationGenerationError("development template or architecture freeze prerequisite differs")

    repair_view = camera_lock["repair_view"]
    camera_tf = repair_view["fk_transform_base_to_camera"]
    intrinsics = camera_lock["intrinsics"]
    asset_calibration = depth_calibration["asset_calibration"]
    registry = object_registry()
    by_id = {row["id"]: row for row in registry.values()}
    denylists = load_calibration_denylists(workspace)
    used_seeds: set[int] = set()
    used_instructions: set[str] = set()
    used_fingerprints: set[str] = set(denylists["layout_fingerprints"])
    families: list[dict[str, Any]] = []
    captures_by_family: dict[str, list[dict[str, Any]]] = {}

    for index, old in enumerate(select_templates(template), start=1):
        family_id = f"{FAMILY_PREFIX}{index:06d}"
        if family_id in denylists["family_ids"]:
            raise CalibrationGenerationError(f"family ID collides with historical data: {family_id}")
        seed_bundle = {
            key: geometry_generator.derived_seed(key, index, denylists["seeds"], used_seeds)
            for key in ("layout", "sensor", "occlusion", "language", "counterfactual")
        }
        specs = copy.deepcopy(old["variant_specs"])
        for spec in specs:
            condition = "occlusion" if spec["variant"] == "occlusion_view_counterfactual" else "clean"
            spec["capture_id"] = f"{family_id}__{condition}_capture"
            instruction, template_id = geometry_generator.new_instruction(
                spec,
                by_id,
                f"{family_id}__{spec['variant']}",
                denylists["instructions"],
                used_instructions,
            )
            spec["instruction"] = instruction
            spec["instruction_sha256"] = hashlib.sha256(instruction.encode()).hexdigest()
            spec["language_template_family_id"] = template_id.replace(
                "v21_development_template_", "v21_calibration_template_"
            )
            if spec["language_template_family_id"] in denylists["template_ids"]:
                raise CalibrationGenerationError(
                    f"language template ID collides: {spec['language_template_family_id']}"
                )
        family = {
            "family_id": family_id,
            "split": "calibration",
            "primary_answerability_stratum": old["primary_answerability_stratum"],
            "state_submode": old["state_submode"],
            "family_category": old["family_category"],
            "depth_dependent": bool(old["depth_dependent"]),
            "asset_pool": "seen",
            "ood_axis": "none",
            "layout_generator_id": "layout_v2_1_observable_relation_calibration_iid",
            "camera_bin_id": repair_view["camera_bin_id"],
            "language_template_bank_id": "language_v2_1_calibration_iid",
            "depth_noise_generator_id": "depth_noise_v2_1_calibration_iid",
            "seed_bundle": seed_bundle,
            "eligible_for_calibration": True,
            "eligible_for_training_or_dev": False,
            "eligible_for_test": False,
            "capture_policy": "OFFICIAL_V2_1_CALIBRATION_ONLY",
            "all_prior_capture_reuse_forbidden": True,
            "composition_template_family_sha256": canonical_json_sha256({
                "state": old["primary_answerability_stratum"],
                "submode": old["state_submode"],
                "category": old["family_category"],
                "depth_dependent": bool(old["depth_dependent"]),
                "primary_scene": old["primary_scene"],
                "variant_semantics": [{
                    key: spec[key] for key in (
                        "variant", "answerability_state", "state_submode",
                        "answerable", "candidate_target_ids", "valid_target_ids",
                        "anchor_ids", "relations", "reference_frame",
                        "uncertainty_sources", "severity", "expected_intervention",
                        "perturbation",
                    )
                } for spec in old["variant_specs"]],
            }),
            "primary_scene": copy.deepcopy(old["primary_scene"]),
            "variants": list(VARIANTS),
            "variant_specs": specs,
        }
        family["primary_scene"]["instruction"] = specs[0]["instruction"]
        clean, occlusion, layout_seed, geometry_rows = geometry_generator.select_family_layouts(
            index,
            family,
            specs,
            registry,
            asset_calibration,
            camera_tf,
            intrinsics,
            denylists["seeds"],
            used_seeds,
            used_fingerprints,
        )
        family["selected_layout_seed"] = layout_seed
        family["observable_geometry_screening"] = {
            "geometry_version": GEOMETRY_VERSION,
            "camera_bin_id": repair_view["camera_bin_id"],
            "selection_policy": "FIRST_DETERMINISTIC_FRESH_CANDIDATE_PASSING_V2_1_RELATION_AND_VISIBILITY_PREDICTOR",
            "predictor_rows": geometry_rows,
        }
        families.append(family)
        id_to_model = {row["id"]: name for name, row in registry.items()}
        rows = []
        for condition, layout, seed_key in (
            ("clean", clean, "sensor"),
            ("occlusion", occlusion, "occlusion"),
        ):
            capture_id = f"{family_id}__{condition}_capture"
            if capture_id in denylists["capture_ids"]:
                raise CalibrationGenerationError(f"capture ID collides: {capture_id}")
            active = {name: pose for name, pose in layout.items() if float(pose[1]) <= .65}
            required = geometry_generator.required_visible_ids(family, specs, condition)
            rows.append({
                "capture_id": capture_id,
                "family_id": family_id,
                "split": "calibration",
                "condition": condition,
                "seed": int(seed_bundle[seed_key]),
                "layout_seed": int(layout_seed),
                "ood_axis": "none",
                "camera_bin_id": repair_view["camera_bin_id"],
                "view_joint_pose": list(repair_view["requested_view_joint_pose"]),
                "layout_generator_id": family["layout_generator_id"],
                "layout_instance_fingerprint_sha256": canonical_json_sha256(
                    {key: active[key] for key in sorted(active)}
                ),
                "layout": layout,
                "required_visible_object_ids": required,
                "required_visible_label_ids": sorted(int(by_id[value]["label"]) for value in required),
            })
        captures_by_family[family_id] = rows

    state_counts = dict(Counter(row["primary_answerability_stratum"] for row in families))
    category_counts = dict(Counter(row["family_category"] for row in families))
    if state_counts != EXPECTED_STATE or category_counts != EXPECTED_CATEGORY:
        raise CalibrationGenerationError(
            f"locked marginals differ: states={state_counts} categories={category_counts}"
        )
    batches = allocate_batches(families)
    captures = []
    for batch in batches:
        for family_id in batch["family_ids"]:
            for capture in captures_by_family[family_id]:
                capture["batch_id"] = batch["batch_id"]
                captures.append(capture)

    source_paths = (
        DEVELOPMENT_TEMPLATE,
        PARTITION,
        CAMERA_LOCK,
        DEPTH_CALIBRATION,
        ARCHITECTURE_FREEZE_LOCK,
    )
    source_hashes = {relative: sha256_file(workspace / relative) for relative in source_paths}
    denylist_hashes = {
        relative: sha256_file(workspace / relative)
        for relative in DENYLIST_JSON if (workspace / relative).is_file()
    }
    assignments = [{
        "family_id": row["family_id"],
        "state": row["primary_answerability_stratum"],
        "category": row["family_category"],
        "submode": row["state_submode"],
        "seed_bundle": row["seed_bundle"],
        "selected_layout_seed": row["selected_layout_seed"],
    } for row in families]
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "status": "STATIC_CALIBRATION_PLAN_NOT_CAPTURED",
        "scientific_status": "PLANNED_INDEPENDENT_CALIBRATION_NOT_CAPTURED",
        "master_seed": MASTER_SEED,
        "eligible_for_calibration": True,
        "eligible_for_training_or_dev": False,
        "eligible_for_test": False,
        "allowed_splits": ["calibration"],
        "sealed_splits_not_created": ["test_iid", "test_ood"],
        "family_count": FAMILY_COUNT,
        "sample_count": SAMPLE_COUNT,
        "variant_count_per_family": 5,
        "counts": {
            "split": {"calibration": FAMILY_COUNT},
            "primary_answerability": state_counts,
            "family_category": category_counts,
            "ood_axis": {"none": FAMILY_COUNT},
            "state_submode": dict(sorted(Counter(row["state_submode"] for row in families).items())),
        },
        "assignment_commitment_sha256": canonical_json_sha256(assignments),
        "development_composition_template_only": True,
        "all_prior_id_seed_instruction_layout_capture_reuse_forbidden": True,
        "source_artifact_sha256": source_hashes,
        "denylist_artifact_sha256": denylist_hashes,
        "asset_partition_sha256": sha256_file(workspace / PARTITION),
        "architecture_freeze_lock_sha256": sha256_file(workspace / ARCHITECTURE_FREEZE_LOCK),
        "frozen_checkpoint": copy.deepcopy(freeze_lock["selected_checkpoint"]),
        "batch_assignments": batches,
        "capture_performed": False,
        "model_inference_performed": False,
        "calibrator_fit_performed": False,
        "test_opened": False,
        "families": families,
    }
    plan = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "status": "STATIC_PLAN_CAPTURE_FORBIDDEN_UNTIL_PREFLIGHT_PASS",
        "scientific_status": "PLANNED_INDEPENDENT_CALIBRATION_NOT_CAPTURED",
        "capture_output_root": OUTPUT_ROOT,
        "world_file": WORLD_FILE,
        "world_sha256": sha256_file(workspace / WORLD_FILE),
        "family_count": FAMILY_COUNT,
        "sample_count_after_materialization": SAMPLE_COUNT,
        "capture_count": len(captures),
        "captures_per_family": 2,
        "batch_execution_order": [row["batch_id"] for row in batches],
        "batches": batches,
        "resume_contract": {
            "identity_key": "capture_id",
            "write_mode": "ATOMIC_CREATE_ONLY",
            "existing_complete_capture": "VERIFY_ALL_DECLARED_HASHES_THEN_SKIP",
            "existing_partial_or_corrupt_capture": "FAIL_NEVER_OVERWRITE",
            "checkpoint_after_each_family": True,
            "checkpoint_at_batch_boundary": True,
            "next_batch_requires_previous_raw_qc_pass": True,
        },
        "sensor_contract": {
            "rgb_topic": "/wrist_camera/color/image_raw",
            "depth_topic": "/wrist_camera/depth/image_raw",
            "semantic_label_topic": "/wrist_camera/evaluation_labels/labels_map",
            "max_timestamp_spread_sec": 0.035,
            "metric_depth_unit": "metre",
            "required_capture_artifacts": [
                "rgb_original.png", "depth_metric.npy", "semantic_instance_labels.png",
                "camera_info.json", "tf_snapshot.json", "capture_meta.json",
            ],
        },
        "camera_poses": {
            repair_view["camera_bin_id"]: list(repair_view["requested_view_joint_pose"])
        },
        "camera_predictor_transform": camera_tf,
        "camera_intrinsics": intrinsics,
        "observable_relation_contract": {
            "geometry_version": GEOMETRY_VERSION,
            "label_source": "synchronized clean-capture semantic-mask centroid and robust median metric depth",
            "runtime_depth_margin_m": 0.020,
            "predictor_screening_margin_m": PREDICTOR_SCREENING_MARGIN_M,
            "between_depth_predictor_screening_margin_m": BETWEEN_DEPTH_SCREENING_MARGIN_M,
            "horizontal_runtime_margin_px": 12.0,
            "simulator_object_center_role": "SECONDARY_STATIC_SCREENING_ONLY",
        },
        "ordered_shutdown_contract": {
            "qualified_gate_decision": "PASS_V2_1_SHUTDOWN_GATE",
            "launcher": "protocol/dataset_v2_1_calibration_capture.launch.py",
            "coordinator": "protocol/dataset_v2_relation_repair_shutdown.py",
            "sequence": ["capture_node", "action_server_and_servo", "move_group", "gazebo"],
            "baseline_launch_files_modified": False,
        },
        "workspace_roi_base_link": {"x_m": [-0.60, 0.60], "y_m": [0.06, 0.65]},
        "inference_payload_template": {
            "rgb_model_input": "TO_BE_CAPTURED",
            "depth_relative_model_input": "TO_BE_DERIVED",
            "enable_depth": True,
            "prompt": "{instruction} " + COORDINATE_SUFFIX,
            "coordinate_suffix": COORDINATE_SUFFIX,
        },
        "object_registry": registry,
        "training_authorized": False,
        "model_inference_authorized": False,
        "calibrator_fit_authorized": False,
        "test_authorized": False,
        "manifest_sha256_pending": True,
        "captures": captures,
    }
    if len(captures) != CAPTURE_COUNT:
        raise CalibrationGenerationError(f"capture count differs: {len(captures)}")
    return manifest, plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    protocol = workspace / "protocol"
    manifest, plan = build_artifacts(workspace)
    manifest_path = protocol / "dataset_v2_1_calibration_manifest.json"
    plan_path = protocol / "dataset_v2_1_calibration_capture_plan.json"
    write_json(manifest_path, manifest)
    plan["manifest_sha256"] = sha256_file(manifest_path)
    plan.pop("manifest_sha256_pending", None)
    write_json(plan_path, plan)
    print(
        "DATASET_V2_1_CALIBRATION_STATIC_MATERIALIZED "
        f"families={FAMILY_COUNT} samples={SAMPLE_COUNT} captures={CAPTURE_COUNT} "
        "batches=[20, 90, 90]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
