#!/usr/bin/env python3
"""Validate the Dataset Expansion V2 design lock without capture or training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import jsonschema

PROTOCOL_DIR = Path(__file__).resolve().parent
WORKSPACE = PROTOCOL_DIR.parent
sys.path.insert(0, str(PROTOCOL_DIR))

from wp2_common import sha256_file, tree_digest, write_json  # noqa: E402


PROTOCOL_ID = "roborefer_dataset_expansion_v2"
VARIANTS = (
    "clean",
    "semantic_counterfactual",
    "relation_counterfactual",
    "depth_corruption",
    "occlusion_view_counterfactual",
)
STATES = ("FOUND", "INSUFFICIENT_EVIDENCE", "AMBIGUOUS", "ABSENT")
CATEGORIES = (
    "direct_grounding",
    "relation_2d",
    "nearer_farther",
    "front_behind_camera",
    "multi_anchor_depth_order",
    "occlusion_depth_evidence",
)
DEPTH_CATEGORIES = {
    "nearer_farther",
    "front_behind_camera",
    "multi_anchor_depth_order",
    "occlusion_depth_evidence",
}
PURPOSES = ("layout", "language", "sensor", "counterfactual", "occlusion")
DESIGN_ARTIFACTS = (
    "protocol/DATASET_EXPANSION_V2_CONTRACT.md",
    "protocol/dataset_expansion_v2_spec.json",
    "protocol/dataset_expansion_v2_asset_partition.json",
    "protocol/dataset_expansion_v2_seed_lock.json",
    "protocol/dataset_split_v2.schema.json",
    "protocol/validate_dataset_expansion_v2.py",
    "protocol/test_dataset_expansion_v2.py",
)


class DesignValidationError(RuntimeError):
    """Raised for malformed design artifacts."""


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DesignValidationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DesignValidationError(f"JSON root must be an object: {path}")
    return value


def canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ranked_ids(
    family_ids: list[str], master_seed: int, namespace: str, dimension: str
) -> list[str]:
    version = "sha256_hash_sort_v1"

    def key(family_id: str) -> tuple[str, str]:
        payload = f"{version}|{master_seed}|{namespace}|{dimension}|{family_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), family_id

    return sorted(family_ids, key=key)


def allocate_labels(
    family_ids: list[str],
    counts: Mapping[str, int],
    master_seed: int,
    namespace: str,
    dimension: str,
) -> dict[str, str]:
    if sum(int(value) for value in counts.values()) != len(family_ids):
        raise DesignValidationError(
            f"{namespace}/{dimension}: counts do not sum to {len(family_ids)}"
        )
    labels: list[str] = []
    for label, count in counts.items():
        if int(count) < 0:
            raise DesignValidationError(f"negative count: {namespace}/{dimension}/{label}")
        labels.extend([str(label)] * int(count))
    ordered = ranked_ids(family_ids, master_seed, namespace, dimension)
    return dict(zip(ordered, labels, strict=True))


def derive_seed(master_seed: int, namespace: str, family_id: str, purpose: str) -> int:
    version = "sha256_seed_bundle_v1"
    payload = f"{version}|{master_seed}|{namespace}|{family_id}|{purpose}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % 2147483646 + 1


def family_ids(config: Mapping[str, Any]) -> list[str]:
    first = int(config["first_index"])
    count = int(config["family_count"])
    prefix = str(config["family_id_prefix"])
    return [f"{prefix}{index:06d}" for index in range(first, first + count)]


def axis_fields(axis: str, partition: Mapping[str, Any]) -> dict[str, str]:
    axis_config = partition["ood_axes"][axis]
    return {
        "asset_pool": str(axis_config["asset_pool"]),
        "layout_generator_id": str(axis_config["layout_generator_id"]),
        "camera_bin_id": str(axis_config["camera_bin_id"]),
        "language_template_bank_id": str(axis_config["language_template_bank_id"]),
        "depth_noise_generator_id": str(axis_config["depth_noise_generator_id"]),
    }


def assignment_record(
    family_id: str,
    split: str,
    state: str,
    category: str,
    axis: str,
    namespace: str,
    master_seed: int,
    partition: Mapping[str, Any],
    eligible: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "family_id": family_id,
        "split": split,
        "primary_answerability_stratum": state,
        "family_category": category,
        "depth_dependent": category in DEPTH_CATEGORIES,
        "ood_axis": axis,
    }
    record.update(axis_fields(axis, partition))
    record["seed_bundle"] = {
        purpose: derive_seed(master_seed, namespace, family_id, purpose)
        for purpose in PURPOSES
    }
    record["variants"] = list(VARIANTS)
    record["eligible_for_official_dataset"] = eligible
    return record


def build_pilot_assignments(
    seed_lock: Mapping[str, Any], partition: Mapping[str, Any]
) -> list[dict[str, Any]]:
    config = seed_lock["pilot_only"]
    master_seed = int(seed_lock["master_seed"])
    namespace = str(config["namespace"])
    ids = family_ids(config)
    split = allocate_labels(ids, config["split_counts"], master_seed, namespace, "split")
    states = allocate_labels(
        ids, config["primary_answerability_counts"], master_seed, namespace, "state"
    )
    categories = allocate_labels(
        ids, config["family_category_counts"], master_seed, namespace, "category"
    )
    axes = allocate_labels(ids, config["ood_axis_counts"], master_seed, namespace, "ood_axis")
    return [
        assignment_record(
            family_id,
            split[family_id],
            states[family_id],
            categories[family_id],
            axes[family_id],
            namespace,
            master_seed,
            partition,
            bool(config["eligible_for_official_dataset"]),
        )
        for family_id in sorted(ids)
    ]


def build_development_assignments(
    seed_lock: Mapping[str, Any], partition: Mapping[str, Any]
) -> list[dict[str, Any]]:
    config = seed_lock["development"]
    master_seed = int(seed_lock["master_seed"])
    namespace = str(config["namespace"])
    ids = family_ids(config)
    split_map = allocate_labels(ids, config["split_counts"], master_seed, namespace, "split")
    assignments: list[dict[str, Any]] = []
    for split_name in config["split_counts"]:
        split_ids = sorted(family_id for family_id in ids if split_map[family_id] == split_name)
        states = allocate_labels(
            split_ids,
            config["primary_answerability_counts_by_split"][split_name],
            master_seed,
            namespace,
            f"{split_name}_state",
        )
        categories = allocate_labels(
            split_ids,
            config["family_category_counts_by_split"][split_name],
            master_seed,
            namespace,
            f"{split_name}_category",
        )
        axes = allocate_labels(
            split_ids,
            config["ood_axis_counts_by_split"][split_name],
            master_seed,
            namespace,
            f"{split_name}_ood_axis",
        )
        assignments.extend(
            assignment_record(
                family_id,
                split_name,
                states[family_id],
                categories[family_id],
                axes[family_id],
                namespace,
                master_seed,
                partition,
                bool(config["eligible_for_official_dataset"]),
            )
            for family_id in split_ids
        )
    return sorted(assignments, key=lambda item: item["family_id"])


def assignment_commitment(assignments: list[dict[str, Any]]) -> str:
    return canonical_sha256(assignments)


def count_field(assignments: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item[field]) for item in assignments).items()))


def build_manifest(
    manifest_id: str,
    assignments: list[dict[str, Any]],
    asset_partition_sha256: str,
    seed_lock_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "manifest_id": manifest_id,
        "lifecycle_status": "DESIGN_LOCKED_NOT_CAPTURED",
        "created_before_capture": True,
        "family_definition": "scene_query_family",
        "variants": list(VARIANTS),
        "asset_partition_sha256": asset_partition_sha256,
        "seed_lock_sha256": seed_lock_sha256,
        "summary": {
            "family_count": len(assignments),
            "sample_count_planned": len(assignments) * len(VARIANTS),
            "counts_by_split": count_field(assignments, "split"),
            "counts_by_primary_answerability": count_field(
                assignments, "primary_answerability_stratum"
            ),
            "counts_by_family_category": count_field(assignments, "family_category"),
            "depth_dependent_family_count": sum(
                bool(item["depth_dependent"]) for item in assignments
            ),
        },
        "families": assignments,
    }


def expected_ontology() -> dict[str, dict[str, Any]]:
    return {
        "FOUND": {
            "denotation_cardinality_min": 1,
            "denotation_cardinality_max": 1,
            "world_referent_exists": True,
            "observation_sufficient": True,
            "answerable": True,
            "default_intervention": "EXECUTE",
        },
        "AMBIGUOUS": {
            "denotation_cardinality_min": 2,
            "denotation_cardinality_max": None,
            "world_referent_exists": True,
            "observation_sufficient": True,
            "answerable": False,
            "default_intervention": "ASK_USER",
        },
        "ABSENT": {
            "denotation_cardinality_min": 0,
            "denotation_cardinality_max": 0,
            "world_referent_exists": False,
            "observation_sufficient": True,
            "answerable": False,
            "default_intervention": "ABSTAIN",
        },
        "INSUFFICIENT_EVIDENCE": {
            "denotation_cardinality_min": 1,
            "denotation_cardinality_max": 1,
            "world_referent_exists": True,
            "observation_sufficient": False,
            "answerable": False,
            "default_intervention": "REOBSERVE",
        },
    }


def run_validation(workspace: Path, require_design_lock: bool = True) -> dict[str, Any]:
    workspace = workspace.resolve()
    protocol = workspace / "protocol"
    spec_path = protocol / "dataset_expansion_v2_spec.json"
    partition_path = protocol / "dataset_expansion_v2_asset_partition.json"
    seed_lock_path = protocol / "dataset_expansion_v2_seed_lock.json"
    schema_path = protocol / "dataset_split_v2.schema.json"
    design_lock_path = protocol / "dataset_expansion_v2_design_lock.json"

    spec = read_object(spec_path)
    partition = read_object(partition_path)
    seed_lock = read_object(seed_lock_path)
    schema = read_object(schema_path)
    design_lock = read_object(design_lock_path) if design_lock_path.exists() else None

    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    check(
        "protocol_ids_match",
        {spec.get("protocol_id"), partition.get("protocol_id"), seed_lock.get("protocol_id")}
        == {PROTOCOL_ID},
        [spec.get("protocol_id"), partition.get("protocol_id"), seed_lock.get("protocol_id")],
    )
    check(
        "design_status_and_next_action",
        spec.get("decision") == "GO_DATASET_EXPANSION_DESIGN_LOCK"
        and spec.get("status") == "LOCKED_DESIGN_NO_CAPTURE_NO_TRAINING"
        and spec.get("current_allowed_next_action") == "GO_DRY_RUN_30_PILOT_ONLY",
        {
            "decision": spec.get("decision"),
            "status": spec.get("status"),
            "next": spec.get("current_allowed_next_action"),
        },
    )

    ontology_findings = []
    for state, expected in expected_ontology().items():
        observed = spec.get("answerability_ontology", {}).get(state, {})
        for field, value in expected.items():
            if observed.get(field) != value:
                ontology_findings.append(
                    {"state": state, "field": field, "expected": value, "observed": observed.get(field)}
                )
    check("answerability_ontology_exact", not ontology_findings, ontology_findings)

    variants = tuple(spec.get("family_unit", {}).get("variants", []))
    check(
        "family_unit_and_variants_locked",
        spec.get("family_unit", {}).get("name") == "scene_query_family"
        and spec.get("family_unit", {}).get("split_key") == "family_id"
        and spec.get("family_unit", {}).get(
            "all_views_paraphrases_clean_corruptions_counterfactuals_same_split"
        )
        is True
        and variants == VARIANTS,
        {"unit": spec.get("family_unit", {}).get("name"), "variants": list(variants)},
    )

    pilot = build_pilot_assignments(seed_lock, partition)
    development = build_development_assignments(seed_lock, partition)
    pilot_commitment = assignment_commitment(pilot)
    development_commitment = assignment_commitment(development)
    check(
        "pilot_assignment_commitment",
        pilot_commitment == seed_lock["pilot_only"].get("assignment_commitment_sha256"),
        {
            "expected": seed_lock["pilot_only"].get("assignment_commitment_sha256"),
            "actual": pilot_commitment,
        },
    )
    check(
        "development_assignment_commitment",
        development_commitment
        == seed_lock["development"].get("assignment_commitment_sha256"),
        {
            "expected": seed_lock["development"].get("assignment_commitment_sha256"),
            "actual": development_commitment,
        },
    )

    pilot_ids = {item["family_id"] for item in pilot}
    development_ids = {item["family_id"] for item in development}
    pilot_seeds = {
        seed for item in pilot for seed in item["seed_bundle"].values()
    }
    development_seeds = {
        seed for item in development for seed in item["seed_bundle"].values()
    }
    check(
        "pilot_official_ids_and_seeds_disjoint",
        not (pilot_ids & development_ids) and not (pilot_seeds & development_seeds),
        {
            "family_id_overlap": sorted(pilot_ids & development_ids),
            "seed_overlap_count": len(pilot_seeds & development_seeds),
        },
    )
    check(
        "pilot_not_official",
        len(pilot) == 30
        and all(item["split"] == "pilot_only" for item in pilot)
        and not any(item["eligible_for_official_dataset"] for item in pilot),
        {
            "families": len(pilot),
            "splits": count_field(pilot, "split"),
            "eligible_count": sum(item["eligible_for_official_dataset"] for item in pilot),
        },
    )
    check(
        "development_exact_scale_and_split",
        len(development) == 400
        and count_field(development, "split") == {"dev": 80, "train": 320}
        and all(item["eligible_for_official_dataset"] for item in development)
        and seed_lock["development"].get("capture_authorized") is False,
        {
            "families": len(development),
            "splits": count_field(development, "split"),
            "capture_authorized": seed_lock["development"].get("capture_authorized"),
        },
    )

    spec_split_findings = []
    for split_name, split_spec in spec["development"]["splits"].items():
        selected = [item for item in development if item["split"] == split_name]
        observed_states = count_field(selected, "primary_answerability_stratum")
        observed_categories = count_field(selected, "family_category")
        if observed_states != dict(sorted(split_spec["primary_answerability_family_counts"].items())):
            spec_split_findings.append({"split": split_name, "dimension": "state", "observed": observed_states})
        if observed_categories != dict(sorted(split_spec["family_category_counts"].items())):
            spec_split_findings.append(
                {"split": split_name, "dimension": "category", "observed": observed_categories}
            )
        if set(observed_states) != set(STATES) or set(observed_categories) != set(CATEGORIES):
            spec_split_findings.append({"split": split_name, "dimension": "coverage"})
    check("development_state_category_targets_exact", not spec_split_findings, spec_split_findings)

    submode_totals = {
        state: sum(int(value) for value in values.values())
        for state, values in spec["state_submodes"].items()
    }
    development_states = Counter(item["primary_answerability_stratum"] for item in development)
    submode_findings = {
        state: {"submode_total": total, "development_state_total": development_states[state]}
        for state, total in submode_totals.items()
        if total != development_states[state]
    }
    check("negative_submode_totals_exact", not submode_findings, submode_findings)

    depth_count = sum(item["depth_dependent"] for item in development)
    check(
        "development_depth_target_exact",
        depth_count == spec["development"].get("depth_dependent_family_count") == 260,
        {"actual": depth_count, "locked": spec["development"].get("depth_dependent_family_count")},
    )

    seen = set(partition["seen_pool"]["asset_ids"])
    heldout = set(partition["test_ood_asset_holdout"]["asset_ids"])
    excluded = set(partition["excluded_assets"]["asset_ids"])
    expected_heldout = {"ycb_pear_01", "ycb_plum_01", "ycb_tuna_fish_can_01"}
    check(
        "asset_sets_disjoint_and_holdout_exact",
        not (seen & heldout)
        and not (seen & excluded)
        and not (heldout & excluded)
        and heldout == expected_heldout,
        {
            "seen_heldout_overlap": sorted(seen & heldout),
            "seen_excluded_overlap": sorted(seen & excluded),
            "heldout_excluded_overlap": sorted(heldout & excluded),
            "heldout": sorted(heldout),
        },
    )
    check(
        "scene_cleanup_exclusions_locked",
        {"ycb_bleach_cleanser_01", "cup", "kettle"}.issubset(excluded),
        sorted(excluded),
    )

    inventory_path = workspace / partition["source_inventory"]["path"]
    color_gate_path = workspace / partition["asset_color_gate"]["path"]
    inventory = read_object(inventory_path)
    color_gate = read_object(color_gate_path)
    inventory_assets = {item["ycb_id"]: item for item in inventory["assets"]}
    heldout_hash_findings = []
    for asset_id, item in partition["test_ood_asset_holdout"]["source_assets"].items():
        observed = inventory_assets.get(item["ycb_id"], {}).get("glb_sha256")
        if observed != item["glb_sha256"]:
            heldout_hash_findings.append(
                {"asset_id": asset_id, "expected": item["glb_sha256"], "observed": observed}
            )
    check(
        "asset_inventory_and_color_gate_verified",
        sha256_file(inventory_path) == partition["source_inventory"]["sha256"]
        and sha256_file(color_gate_path) == partition["asset_color_gate"]["sha256"]
        and color_gate.get("status") == partition["asset_color_gate"]["required_status"]
        and not heldout_hash_findings,
        {
            "inventory_sha256": sha256_file(inventory_path),
            "color_gate_sha256": sha256_file(color_gate_path),
            "color_gate_status": color_gate.get("status"),
            "heldout_hash_findings": heldout_hash_findings,
        },
    )

    clone_groups = partition["clone_policy"]["clone_groups"]
    clone_findings = []
    all_clone_instances = []
    for canonical, instances in clone_groups.items():
        if canonical not in instances or len(instances) != len(set(instances)):
            clone_findings.append(canonical)
        all_clone_instances.extend(instances)
    check(
        "clone_policy_nonindependent",
        partition["clone_policy"].get("clone_instances_are_not_independent_asset_categories")
        is True
        and len(all_clone_instances) == len(set(all_clone_instances))
        and not clone_findings,
        {"groups": clone_groups, "findings": clone_findings},
    )

    axis_baseline = partition["ood_axes"]["none"]
    axis_findings = []
    dimension_keys = (
        "asset_pool",
        "layout_generator_id",
        "camera_bin_id",
        "language_template_bank_id",
        "depth_noise_generator_id",
    )
    expected_changed_key = {
        "asset": "asset_pool",
        "layout": "layout_generator_id",
        "viewpoint": "camera_bin_id",
        "language": "language_template_bank_id",
        "depth_noise": "depth_noise_generator_id",
    }
    for axis, expected_key in expected_changed_key.items():
        config = partition["ood_axes"][axis]
        changed = [key for key in dimension_keys if config[key] != axis_baseline[key]]
        if changed != [expected_key]:
            axis_findings.append({"axis": axis, "changed": changed, "expected": [expected_key]})
    composition = partition["test_ood_composition_policy"]
    check(
        "single_axis_ood_isolation",
        not axis_findings
        and float(composition["single_primary_axis_fraction_min"]) >= 0.8
        and float(composition["crossed_axis_fraction_max"]) <= 0.2,
        {"axis_findings": axis_findings, "composition": composition},
    )

    minimum_negative = spec["future_candidate_full"]["minimum_ambiguous_plus_absent_families"]
    lifecycle = spec["split_lifecycle"]
    check(
        "calibration_and_test_minimum_negative_locked",
        all(int(minimum_negative[name]) >= 60 for name in ("calibration", "test_iid", "test_ood")),
        minimum_negative,
    )
    check(
        "sealed_split_lifecycle",
        lifecycle["calibration"]["first_authorized_phase"]
        == "AFTER_CANDIDATE_MODEL_AND_CHECKPOINT_FREEZE"
        and lifecycle["test_iid"]["first_authorized_phase"]
        == "AFTER_CALIBRATOR_AND_THRESHOLDS_FREEZE"
        and lifecycle["test_ood"]["first_authorized_phase"]
        == "AFTER_CALIBRATOR_AND_THRESHOLDS_FREEZE"
        and lifecycle["test_iid"]["open_once"] is True
        and lifecycle["test_ood"]["open_once"] is True,
        lifecycle,
    )

    wp2_policy = spec["wp2_policy"]
    check(
        "wp2_prototype_excluded_from_final_eval",
        wp2_policy.get("status") == "PROTOTYPE_ENGINEERING_ONLY"
        and {"calibration_fit", "test_iid", "test_ood", "generalization_claim"}.issubset(
            set(wp2_policy.get("forbidden_use", []))
        ),
        wp2_policy,
    )
    reporting = spec["reporting_boundary"]
    check(
        "reporting_boundary_prevents_fake_results",
        set(reporting.get("may_update", []))
        == {"TABLE_01_DATASET_INDEPENDENCE"}
        and reporting.get("figures_allowed_before_capture") == []
        and set(reporting.get("figures_forbidden_before_capture", []))
        == {"F01_DATASET_BALANCE"}
        and reporting.get("figure_release_condition")
        == "AFTER_REAL_CAPTURE_AND_DATASET_QC_PASS"
        and reporting.get("figure_generation_policy")
        == {
            "must_be_generated_from_hashed_csv_or_json": True,
            "source_data_must_be_archived_beside_figure": True,
            "family_denominator_must_be_explicit": True,
            "planned_counts_must_not_be_presented_as_observed_evidence": True,
            "manual_decorative_gui_style_figure_forbidden": True,
        }
        and set(reporting.get("must_remain_not_run", []))
        == {
            "TABLE_02_GROUNDING_MAIN",
            "TABLE_03_ANSWERABILITY_ERROR",
            "TABLE_04_CALIBRATION",
            "TABLE_05_SOURCE_ATTRIBUTION",
            "TABLE_06_ABLATION",
        }
        and reporting.get("planned_counts_must_be_labeled") == "PLANNED_NOT_CAPTURED",
        reporting,
    )

    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        schema_meta_valid = True
        schema_error = None
    except jsonschema.SchemaError as exc:
        schema_meta_valid = False
        schema_error = str(exc)
    asset_sha = sha256_file(partition_path)
    seed_sha = sha256_file(seed_lock_path)
    pilot_manifest = build_manifest("dataset_expansion_v2_pilot_design", pilot, asset_sha, seed_sha)
    development_manifest = build_manifest(
        "dataset_expansion_v2_development_design", development, asset_sha, seed_sha
    )
    manifest_schema_findings = []
    if schema_meta_valid:
        validator = jsonschema.Draft202012Validator(schema)
        for name, manifest in (("pilot", pilot_manifest), ("development", development_manifest)):
            for error in sorted(validator.iter_errors(manifest), key=lambda item: list(item.path)):
                manifest_schema_findings.append(
                    {"manifest": name, "path": list(error.path), "message": error.message}
                )
    check(
        "split_schema_and_design_manifests_valid",
        schema_meta_valid and not manifest_schema_findings,
        {"schema_error": schema_error, "manifest_findings": manifest_schema_findings},
    )

    protected_file_mismatches = []
    for relative, expected in seed_lock["protected_files"].items():
        path = workspace / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            protected_file_mismatches.append(
                {"path": relative, "expected": expected, "actual": actual}
            )
    protected_tree_mismatches = []
    for relative, expected in seed_lock["protected_tree_digests"].items():
        path = workspace / relative
        actual = tree_digest(workspace, relative) if path.is_dir() else None
        if actual != expected:
            protected_tree_mismatches.append(
                {"path": relative, "expected": expected, "actual": actual}
            )
    check("protected_wp0_wp3_files_unchanged", not protected_file_mismatches, protected_file_mismatches)
    check("protected_wp2_and_prior_trees_unchanged", not protected_tree_mismatches, protected_tree_mismatches)

    design_artifact_hashes = {
        relative: sha256_file(workspace / relative) for relative in DESIGN_ARTIFACTS
    }
    if design_lock is None:
        check("design_lock_present", not require_design_lock, "missing")
    else:
        lock_hash_findings = []
        for relative, actual in design_artifact_hashes.items():
            expected = design_lock.get("artifacts", {}).get(relative)
            if actual != expected:
                lock_hash_findings.append(
                    {"path": relative, "expected": expected, "actual": actual}
                )
        check("design_lock_artifact_hashes_match", not lock_hash_findings, lock_hash_findings)
        check(
            "design_lock_commitments_match",
            design_lock.get("assignment_commitments", {}).get("pilot_only") == pilot_commitment
            and design_lock.get("assignment_commitments", {}).get("development")
            == development_commitment,
            {
                "lock": design_lock.get("assignment_commitments"),
                "computed": {"pilot_only": pilot_commitment, "development": development_commitment},
            },
        )
        check(
            "design_lock_authorization_boundary",
            design_lock.get("status") == "LOCKED_NO_CAPTURE_NO_TRAINING"
            and design_lock.get("allowed_next_action") == "GO_DRY_RUN_30_PILOT_ONLY"
            and design_lock.get("training_performed") is False
            and design_lock.get("official_capture_performed") is False
            and design_lock.get("pilot_capture_performed") is False,
            {
                "status": design_lock.get("status"),
                "allowed_next_action": design_lock.get("allowed_next_action"),
                "training_performed": design_lock.get("training_performed"),
                "official_capture_performed": design_lock.get("official_capture_performed"),
                "pilot_capture_performed": design_lock.get("pilot_capture_performed"),
            },
        )

    passed = all(item["passed"] for item in checks.values())
    return {
        "schema_version": 2,
        "gate": "GO_DATASET_EXPANSION_DESIGN_LOCK",
        "status": "PASS" if passed else "FAIL",
        "decision": "GO_DRY_RUN_30_PILOT_ONLY" if passed else "FIX_DESIGN_LOCK_FIRST",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "pilot_capture_performed": False,
        "official_capture_performed": False,
        "official_family_count_captured": 0,
        "planned_counts": {
            "pilot_only_families": len(pilot),
            "development_families": len(development),
            "development_samples": len(development) * len(VARIANTS),
            "development_splits": count_field(development, "split"),
            "development_primary_answerability": count_field(
                development, "primary_answerability_stratum"
            ),
            "development_categories": count_field(development, "family_category"),
            "development_depth_dependent_families": sum(
                item["depth_dependent"] for item in development
            ),
        },
        "assignment_commitments": {
            "pilot_only": pilot_commitment,
            "development": development_commitment,
        },
        "design_artifact_hashes": design_artifact_hashes,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(WORKSPACE))
    parser.add_argument("--report")
    parser.add_argument("--allow-missing-design-lock", action="store_true")
    parser.add_argument("--print-commitments", action="store_true")
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()

    if args.print_commitments:
        seed_lock = read_object(workspace / "protocol/dataset_expansion_v2_seed_lock.json")
        partition = read_object(
            workspace / "protocol/dataset_expansion_v2_asset_partition.json"
        )
        values = {
            "pilot_only": assignment_commitment(build_pilot_assignments(seed_lock, partition)),
            "development": assignment_commitment(
                build_development_assignments(seed_lock, partition)
            ),
        }
        print(json.dumps(values, indent=2, sort_keys=True))
        return 0

    report = run_validation(workspace, require_design_lock=not args.allow_missing_design_lock)
    if args.report:
        report_path = Path(args.report).expanduser()
        if not report_path.is_absolute():
            report_path = workspace / report_path
        write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
