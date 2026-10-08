#!/usr/bin/env python3
"""Fail-closed static preflight for the add-only P-CRA-U V1.1 revision.

This program performs no model forward pass and creates no optimizer.  Its only
permitted write is the V1.1 preflight lock, and that write occurs atomically
after every integrity, design, data, implementation, and sealing check passes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "pcra_u_development_v1_1"
CONTRACT_PATH = WORKSPACE / "protocol/PCRA_U_DEVELOPMENT_V1_1_REVISION_CONTRACT.md"
CONFIG_PATH = WORKSPACE / "protocol/pcra_u_development_v1_1_config.json"
MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_development_v1_1_manifest.json"
LOCK_PATH = WORKSPACE / "protocol/pcra_u_development_v1_1_preflight_lock.json"
IMPLEMENTATION_PATHS = [
    WORKSPACE / "protocol/pcra_u_development_v1_1_common.py",
    WORKSPACE / "protocol/pcra_u_development_v1_1_train.py",
]


class V11PreflightError(RuntimeError):
    """Raised when a preflight artifact is malformed or a lock would drift."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V11PreflightError(f"Cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise V11PreflightError(f"Expected a JSON object: {path}")
    return value


def safe_path(relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise V11PreflightError(f"Path must be non-empty and relative: {relative!r}")
    path = (WORKSPACE / relative).resolve()
    root = WORKSPACE.resolve()
    if path != root and root not in path.parents:
        raise V11PreflightError(f"Path escapes workspace: {relative}")
    return path


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def call_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def run_checks() -> tuple[dict[str, bool], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = read_json(CONFIG_PATH)
    manifest = read_json(MANIFEST_PATH)
    parent_manifest = read_json(
        safe_path(manifest["dataset_reuse"]["parent_manifest_path"])
    )
    feature_manifest = read_json(
        safe_path(manifest["dataset_reuse"]["feature_manifest_path"])
    )
    feature_index = read_json(
        safe_path(manifest["dataset_reuse"]["feature_cache_index_path"])
    )

    checks: dict[str, bool] = {}
    observed: dict[str, Any] = {}

    def check(name: str, condition: bool) -> None:
        checks[name] = bool(condition)

    check(
        "design_files_exist",
        all(path.is_file() and path.stat().st_size > 0 for path in [CONTRACT_PATH, CONFIG_PATH, MANIFEST_PATH]),
    )
    check(
        "protocol_identity_locked",
        config.get("schema_version") == 1
        and config.get("protocol_id") == PROTOCOL_ID
        and manifest.get("schema_version") == 1
        and manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("status") == "LOCKED_BEFORE_PREFLIGHT_NO_OPTIMIZER_STEP_AUTHORIZED",
    )

    parent_hashes: dict[str, str | None] = {}
    parent_hashes_valid = True
    for relative, expected in manifest.get("parent_artifacts", {}).items():
        path = safe_path(relative)
        actual = sha256_file(path) if path.is_file() else None
        parent_hashes[relative] = actual
        parent_hashes_valid &= actual == expected
    observed["parent_hashes"] = parent_hashes
    check("all_exact_parent_hashes_match", parent_hashes_valid and bool(parent_hashes))

    parent_config = read_json(WORKSPACE / "protocol/pcra_u_development_train_config.json")
    feature_keys = [
        "model_root",
        "stage",
        "raw_shape",
        "raw_dtype",
        "local_tile_count",
        "thumbnail_index",
        "tile_grid_rows_columns",
        "patch_grid_per_tile",
        "pool_kernel",
        "pooled_grid_shape",
        "pooled_thumbnail_shape",
        "cache_dtype",
        "training_dtype",
        "deduplication_key",
        "raw_cache_retained",
    ]
    feature_compatible = all(
        config["feature_input"].get(key) == parent_config["feature_input"].get(key)
        for key in feature_keys
    )
    model_compatible = all(
        config["model"].get(key) == parent_config["model"].get(key)
        for key in [
            "hidden_dim",
            "answerability_classes",
            "source_classes",
            "deferred_source_classes",
            "roborefer_frozen",
        ]
    )
    check(
        "v1_input_language_and_model_shape_compatible",
        feature_compatible and config.get("language") == parent_config.get("language") and model_compatible,
    )
    check(
        "dropout_revision_exact",
        config["model"].get("dropout") == 0.1
        and config["model"].get("dropout_placement")
        == "after_hidden_gelu_in_target_answerability_and_source_heads",
    )
    loss = config["loss"]
    check(
        "normalized_clean_found_objective_exact",
        loss.get("heatmap_weight") == 1.0
        and loss.get("answerability_weight") == 0.5
        and loss.get("source_weight") == 0.5
        and loss.get("clean_found_heatmap_multiplier") == 4.0
        and loss.get("clean_found_weight_normalization")
        == "sum_weighted_per_sample_loss_divided_by_sum_found_sample_weights",
    )
    optimization = config["optimization"]
    score_weights = optimization.get("selection_score", {})
    check(
        "optimizer_and_equal_epoch_plan_exact",
        optimization.get("optimizer") == "AdamW"
        and optimization.get("learning_rate") == 0.0001
        and optimization.get("weight_decay") == 0.001
        and optimization.get("batch_size") == 8
        and optimization.get("max_epochs") == 20
        and optimization.get("min_epochs") == 6
        and optimization.get("run_all_epochs") is True
        and optimization.get("warmup_steps") == 200
        and optimization.get("gradient_clip_norm") == 5.0,
    )
    check(
        "scientific_checkpoint_score_exact",
        optimization.get("selection_split") == "dev"
        and optimization.get("selection_metric") == "dev_scientific_score_v1_1"
        and optimization.get("selection_mode") == "max"
        and score_weights
        == {
            "clean_found_point_in_target": 0.45,
            "all_found_point_in_target": 0.2,
            "answerability_macro_f1": 0.2,
            "one_minus_false_found_rate": 0.15,
        }
        and abs(sum(score_weights.values()) - 1.0) < 1e-12,
    )

    seed_runs = manifest.get("seed_runs", [])
    campaign_root = "results/pcra_u_runs/pcra_u_development_v1_1_campaign_20260824"
    expected_run_roots = [f"{campaign_root}_seed_{seed}" for seed in [24082026, 24082027, 24082028]]
    check(
        "three_fresh_locked_seeds_exact",
        [row.get("seed") for row in seed_runs] == [24082026, 24082027, 24082028]
        and len({row.get("run_id") for row in seed_runs}) == 3
        and all(row.get("initialization") == "fresh_sidecar_random_initialization" for row in seed_runs)
        and all(row.get("resume_checkpoint") is None for row in seed_runs)
        and all(row.get("epochs") == 20 for row in seed_runs)
        and [row.get("run_root") for row in seed_runs] == expected_run_roots,
    )
    execution_paths = manifest.get("execution_paths", {})
    check(
        "add_only_execution_paths_exact",
        execution_paths
        == {
            "preflight_lock": "protocol/pcra_u_development_v1_1_preflight_lock.json",
            "canary_run_root": "results/pcra_u_runs/pcra_u_development_v1_1_canary_20260824",
            "campaign_run_root": campaign_root,
        }
        and not safe_path(execution_paths["canary_run_root"]).exists()
        and not safe_path(execution_paths["campaign_run_root"]).exists()
        and all(not safe_path(relative).exists() for relative in expected_run_roots),
    )
    selection = manifest.get("checkpoint_selection", {})
    eligibility = selection.get("eligibility", {})
    check(
        "eligibility_thresholds_exact",
        eligibility
        == {
            "min_epoch": 6,
            "min_all_found_point_in_target": 0.8371428571,
            "min_answerability_macro_f1": 0.7220238435,
            "max_false_found_rate": 0.1147368421,
        },
    )
    check(
        "selection_manifest_exact",
        selection.get("metric") == "dev_scientific_score_v1_1"
        and selection.get("mode") == "max"
        and selection.get("pointer_name") == "best_dev_scientific_score_v1_1.json"
        and selection.get("score", {}).get("formula")
        == "0.45*Gc + 0.20*Ga + 0.20*A + 0.15*(1-F)"
        and selection.get("total_loss_is_selection_metric") is False
        and selection.get("relaxation_after_observation_forbidden") is True,
    )

    acceptance = manifest.get("acceptance_policy", {})
    bootstrap = acceptance.get("paired_family_bootstrap", {})
    check(
        "acceptance_and_bootstrap_policy_exact",
        acceptance.get("require_three_eligible_seeds") is True
        and acceptance.get("min_clean_found_point_in_target_each_seed") == 0.5
        and acceptance.get("strict_clean_improvement_threshold") == 0.53125
        and acceptance.get("min_seed_count_strictly_above_clean_threshold") == 2
        and acceptance.get("point_estimate_false_found_strict_improvement") is True
        and bootstrap.get("conditional_on_seed_gates") is True
        and bootstrap.get("replicates") == 10000
        and bootstrap.get("seed") == 24082029
        and bootstrap.get("interval") == "two_sided_95_percent_percentile_2.5_97.5"
        and bootstrap.get("delta_direction") == "v1_1_minus_v1_for_Gc_Ga_A_and_F"
        and bootstrap.get("clean_superiority_min_lower_bound") == 0.0
        and bootstrap.get("all_found_noninferiority_min_lower_bound") == -0.02
        and bootstrap.get("answerability_noninferiority_min_lower_bound") == -0.02
        and bootstrap.get("false_found_noninferiority_max_upper_bound") == 0.02,
    )

    entries = parent_manifest.get("entries", [])
    train = [entry for entry in entries if entry.get("split") == "train"]
    dev = [entry for entry in entries if entry.get("split") == "dev"]
    train_families = {entry["family_id"] for entry in train}
    dev_families = {entry["family_id"] for entry in dev}
    family_variants: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        family_variants[entry["family_id"]][entry["variant"]] += 1
    expected_variants = set(parent_manifest.get("variant_order", []))
    found_counts = {
        "train_found": sum(entry["supervision"]["answerability_state"] == "FOUND" for entry in train),
        "train_clean_found": sum(
            entry["supervision"]["answerability_state"] == "FOUND" and entry["variant"] == "clean"
            for entry in train
        ),
        "dev_found": sum(entry["supervision"]["answerability_state"] == "FOUND" for entry in dev),
        "dev_clean_found": sum(
            entry["supervision"]["answerability_state"] == "FOUND" and entry["variant"] == "clean"
            for entry in dev
        ),
        "dev_ambiguous_or_absent": sum(
            entry["supervision"]["answerability_state"] in {"AMBIGUOUS", "ABSENT"} for entry in dev
        ),
    }
    found_counts["train_other_found"] = found_counts["train_found"] - found_counts["train_clean_found"]
    found_counts["dev_other_found"] = found_counts["dev_found"] - found_counts["dev_clean_found"]
    observed["dataset_counts"] = {
        "train_families": len(train_families),
        "dev_families": len(dev_families),
        "train_samples": len(train),
        "dev_samples": len(dev),
        **found_counts,
    }
    reuse = manifest["dataset_reuse"]
    expected_counts = {
        "train_families": 320,
        "train_samples": 1600,
        "dev_families": 80,
        "dev_samples": 400,
        "train_found": 673,
        "train_clean_found": 128,
        "train_other_found": 545,
        "dev_found": 168,
        "dev_clean_found": 32,
        "dev_other_found": 136,
        "dev_ambiguous_or_absent": 95,
    }
    observed_counts = observed["dataset_counts"]
    check(
        "exact_family_preserving_dataset_counts",
        all(observed_counts[key] == value == reuse.get(key) for key, value in expected_counts.items())
        and not (train_families & dev_families)
        and len(entries) == len({entry["sample_id"] for entry in entries}) == 2000
        and len(family_variants) == 400
        and all(
            set(counter) == expected_variants
            and len(counter) == 5
            and all(count == 1 for count in counter.values())
            for counter in family_variants.values()
        ),
    )

    feature_entries = feature_manifest.get("entries", [])
    sample_ids = [entry["sample_id"] for entry in entries]
    feature_sample_ids = [entry["sample_id"] for entry in feature_entries]
    index_sample_ids = set(feature_index.get("sample_to_feature", {}))
    check(
        "immutable_feature_cache_identity_and_alignment",
        feature_index.get("status") == "COMPLETE"
        and feature_index.get("scope") == "full"
        and feature_index.get("feature_manifest_sha256")
        == manifest["parent_artifacts"]["protocol/pcra_u_development_feature_manifest.json"]
        and feature_index.get("counts", {}).get("samples") == 2000
        and feature_index.get("counts", {}).get("unique_feature_pairs") == 1242
        and len(feature_index.get("features", {})) == 1242
        and sample_ids == feature_sample_ids
        and index_sample_ids == set(sample_ids),
    )

    v1_report = read_json(
        WORKSPACE / "results/pcra_u_runs/pcra_u_development_train_20260824/PCRA_U_DEVELOPMENT_TRAIN_REPORT.json"
    )
    v1_predictions_payload = read_json(
        WORKSPACE
        / "results/pcra_u_runs/pcra_u_development_train_20260824/predictions/dev_predictions_exploratory.json"
    )
    predictions = v1_predictions_payload.get("predictions", [])
    clean_found = [
        row for row in predictions if row.get("variant") == "clean" and row.get("answerability_truth") == "FOUND"
    ]
    all_found = [row for row in predictions if row.get("answerability_truth") == "FOUND"]
    risky = [row for row in predictions if row.get("answerability_truth") in {"AMBIGUOUS", "ABSENT"}]
    reference_observed = {
        "clean_found_correct": sum(row.get("point_in_target") is True for row in clean_found),
        "clean_found_denominator": len(clean_found),
        "all_found_correct": sum(row.get("point_in_target") is True for row in all_found),
        "all_found_denominator": len(all_found),
        "false_found_count": sum(row.get("answerability_prediction") == "FOUND" for row in risky),
        "false_found_denominator": len(risky),
    }
    observed["v1_reference_counts"] = reference_observed
    reference = manifest["v1_reference"]
    metrics = v1_report["selected_checkpoint_dev_metrics"]
    check(
        "immutable_v1_reference_metrics_reproduced",
        reference_observed["clean_found_correct"] == reference.get("clean_found_correct") == 17
        and reference_observed["clean_found_denominator"] == reference.get("clean_found_denominator") == 32
        and reference_observed["all_found_correct"] == reference.get("all_found_correct") == 144
        and reference_observed["all_found_denominator"] == reference.get("all_found_denominator") == 168
        and reference_observed["false_found_count"] == reference.get("false_found_count") == 9
        and reference_observed["false_found_denominator"] == reference.get("false_found_denominator") == 95
        and reference.get("clean_found_point_in_target") == 17 / 32
        and metrics["grounding_found"]["point_in_target"] == reference.get("all_found_point_in_target")
        and metrics["answerability"]["macro_f1"] == reference.get("answerability_macro_f1")
        and metrics["false_found_on_ambiguous_or_absent"]["rate"] == reference.get("false_found_rate")
        and v1_report.get("selected_checkpoint_model_sha256") == reference.get("model_sha256"),
    )

    implementation_sources: dict[str, str] = {}
    implementation_trees: dict[str, ast.AST] = {}
    implementation_valid = True
    for path in IMPLEMENTATION_PATHS:
        if not path.is_file() or path.stat().st_size == 0:
            implementation_valid = False
            continue
        try:
            source = path.read_text(encoding="utf-8")
            implementation_sources[path.name] = source
            implementation_trees[path.name] = ast.parse(source, filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            implementation_valid = False
    check("implementation_files_exist_and_parse", implementation_valid and len(implementation_trees) == 2)
    common_source = implementation_sources.get("pcra_u_development_v1_1_common.py", "")
    train_source = implementation_sources.get("pcra_u_development_v1_1_train.py", "")
    all_calls = set().union(*(call_names(tree) for tree in implementation_trees.values())) if implementation_trees else set()
    check(
        "implementation_contains_locked_revision",
        ".Dropout(" in common_source
        and "clean_found_heatmap_multiplier" in common_source
        and "eligible_for_best" in train_source
        and "dev_scientific_score_v1_1" in train_source
        and "pcra_u_development_v1_1_config.json" in train_source
        and "pcra_u_development_v1_1_manifest.json" in train_source,
    )
    forbidden_roots = manifest["calibration_test_boundary"]["forbidden_input_roots"]
    combined_implementation = "\n".join(implementation_sources.values())
    check(
        "implementation_has_no_resume_old_pointer_or_calibration_input",
        "resume_training" not in all_calls
        and "best_dev_total_loss" not in combined_implementation
        and all(root not in combined_implementation for root in forbidden_roots),
    )

    calibrator_lock = read_json(WORKSPACE / "protocol/pcra_u_calibrator_threshold_freeze_lock.json")
    calibration_report = read_json(
        WORKSPACE / "results/pcra_u_runs/pcra_u_calibration_20260824/PCRA_U_CALIBRATION_REPORT.json"
    )
    lock_boundary = calibrator_lock.get("reporting_boundary", {})
    report_boundary = calibration_report.get("reporting_boundary", {})
    boundary = manifest["calibration_test_boundary"]
    check(
        "historical_calibration_quarantined_and_tests_still_unopened",
        boundary.get("historical_v1_calibration_exists") is True
        and boundary.get("historical_v1_calibration_reuse_for_v1_1") is False
        and boundary.get("v1_1_calibration_fit") is False
        and boundary.get("test_iid_opened") is False
        and boundary.get("test_ood_opened") is False
        and calibrator_lock.get("status") == "FROZEN"
        and calibrator_lock.get("tests_opened_during_calibration") is False
        and lock_boundary.get("test_iid_opened") is False
        and lock_boundary.get("test_ood_opened") is False
        and report_boundary.get("test_iid_opened") is False
        and report_boundary.get("test_ood_opened") is False,
    )
    check(
        "preflight_has_no_training_authority_or_optimizer_step",
        config["safety"].get("gradient_split") == "train"
        and config["safety"].get("checkpoint_selection_split") == "dev"
        and config["safety"].get("forbidden_splits") == ["calibration", "test_iid", "test_ood"]
        and config["safety"].get("v1_1_calibration_fit") is False
        and config["safety"].get("test_opened") is False,
    )
    return checks, observed, config, manifest


def build_lock(
    checks: dict[str, bool],
    observed: dict[str, Any],
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    artifact_paths = [
        CONTRACT_PATH,
        CONFIG_PATH,
        MANIFEST_PATH,
        Path(__file__).resolve(),
        *IMPLEMENTATION_PATHS,
        WORKSPACE / "protocol/training_checkpoint_manager.py",
    ]
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_AFTER_PREFLIGHT_BEFORE_V1_1_OPTIMIZER_STEP",
        "created_at_utc": utc_now(),
        "decision": "GO_RUN_THREE_FRESH_V1_1_SEEDS",
        "allowed_next_action": "RUN_EXACTLY_20_EPOCHS_FOR_EACH_LOCKED_SEED",
        "checks": checks,
        "artifacts": {
            str(path.relative_to(WORKSPACE)): sha256_file(path) for path in artifact_paths
        },
        "parent_artifacts": manifest["parent_artifacts"],
        "validated_dataset_counts": observed["dataset_counts"],
        "validated_v1_reference_counts": observed["v1_reference_counts"],
        "seed_runs": manifest["seed_runs"],
        "checkpoint_selection": manifest["checkpoint_selection"],
        "acceptance_policy": manifest["acceptance_policy"],
        "gradient_split": "train",
        "selection_split": "dev",
        "fresh_initialization_only": True,
        "training_performed": False,
        "optimizer_step_performed": False,
        "v1_1_calibration_fit": False,
        "historical_v1_calibration_used": False,
        "test_iid_opened": False,
        "test_ood_opened": False,
        "tables_02_to_06": "NOT_RUN",
        "config_protocol_id": config["protocol_id"],
    }


def lock_matches(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    keys = [
        "schema_version",
        "protocol_id",
        "status",
        "decision",
        "allowed_next_action",
        "checks",
        "artifacts",
        "parent_artifacts",
        "validated_dataset_counts",
        "validated_v1_reference_counts",
        "seed_runs",
        "checkpoint_selection",
        "acceptance_policy",
        "gradient_split",
        "selection_split",
        "fresh_initialization_only",
        "training_performed",
        "optimizer_step_performed",
        "v1_1_calibration_fit",
        "historical_v1_calibration_used",
        "test_iid_opened",
        "test_ood_opened",
        "tables_02_to_06",
        "config_protocol_id",
    ]
    return all(existing.get(key) == candidate.get(key) for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate everything but do not create the preflight lock",
    )
    args = parser.parse_args()
    checks, observed, config, manifest = run_checks()
    passed = all(checks.values())
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "passed": passed,
        "decision": "GO_WRITE_V1_1_PREFLIGHT_LOCK" if passed else "FIX_V1_1_PREFLIGHT_FIRST",
        "checks": checks,
        "observed": observed,
        "lock_path": str(LOCK_PATH.relative_to(WORKSPACE)),
        "lock_written": False,
        "training_performed": False,
        "optimizer_step_performed": False,
        "v1_1_calibration_fit": False,
        "test_iid_opened": False,
        "test_ood_opened": False,
    }
    if passed and not args.check_only:
        candidate = build_lock(checks, observed, config, manifest)
        if LOCK_PATH.exists():
            existing = read_json(LOCK_PATH)
            if not lock_matches(existing, candidate):
                raise V11PreflightError(
                    f"Refusing to overwrite a non-equivalent V1.1 preflight lock: {LOCK_PATH}"
                )
            report["decision"] = "EXISTING_V1_1_PREFLIGHT_LOCK_VERIFIED"
        else:
            atomic_write_json(LOCK_PATH, candidate)
            report["lock_written"] = True
            report["decision"] = "V1_1_PREFLIGHT_LOCK_CREATED"
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
