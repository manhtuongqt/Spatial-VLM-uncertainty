#!/usr/bin/env python3
"""Locked, add-only three-seed P-CRA-U V1.1 development campaign.

This runtime never resumes a V1 checkpoint and never mutates V1 sources,
datasets, cached features, or result roots.  It performs a two-epoch engineering
canary followed by three independent, exactly 20-epoch full-development runs.
Only train entries receive gradients and only dev entries select checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import shutil
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

if __package__:
    from .pcra_u_development_common import (
        ANSWERABILITY_CLASSES,
        DevelopmentTrainingError,
        FrozenFeatureDataset,
        canonical_sha256,
        chunks,
        deterministic_entry_order,
        global_class_weights,
        gradient_groups,
        model_inputs,
        read_json,
        safe_resolve,
        seed_runtime,
        sha256_file,
        write_json,
    )
    from .pcra_u_development_train import (
        append_jsonl,
        scheduler_for,
        verify_execution_lock,
        verify_feature_index,
        write_csv,
    )
    from .pcra_u_development_train_preflight import protected_now
    from .pcra_u_development_v1_1_common import (
        PCRAUDevelopmentV11,
        SELECTION_METRIC,
        SELECTION_MODE,
        compute_loss_v11,
        evaluate_dataset_v11,
        trainable_parameter_count,
    )
    from .training_checkpoint_manager import (
        audit_checkpoint_root,
        load_model_for_evaluation,
        save_checkpoint,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pcra_u_development_common import (  # type: ignore
        ANSWERABILITY_CLASSES,
        DevelopmentTrainingError,
        FrozenFeatureDataset,
        canonical_sha256,
        chunks,
        deterministic_entry_order,
        global_class_weights,
        gradient_groups,
        model_inputs,
        read_json,
        safe_resolve,
        seed_runtime,
        sha256_file,
        write_json,
    )
    from pcra_u_development_train import (  # type: ignore
        append_jsonl,
        scheduler_for,
        verify_execution_lock,
        verify_feature_index,
        write_csv,
    )
    from pcra_u_development_train_preflight import protected_now  # type: ignore
    from pcra_u_development_v1_1_common import (  # type: ignore
        PCRAUDevelopmentV11,
        SELECTION_METRIC,
        SELECTION_MODE,
        compute_loss_v11,
        evaluate_dataset_v11,
        trainable_parameter_count,
    )
    from training_checkpoint_manager import (  # type: ignore
        audit_checkpoint_root,
        load_model_for_evaluation,
        save_checkpoint,
    )


WORKSPACE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "pcra_u_development_v1_1"
CONFIG_PATH = WORKSPACE / "protocol/pcra_u_development_v1_1_config.json"
MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_development_v1_1_manifest.json"
PREFLIGHT_LOCK_PATH = WORKSPACE / "protocol/pcra_u_development_v1_1_preflight_lock.json"
COMMON_PATH = WORKSPACE / "protocol/pcra_u_development_v1_1_common.py"
TRAIN_PATH = Path(__file__).resolve()

PARENT_CONFIG_PATH = WORKSPACE / "protocol/pcra_u_development_train_config.json"
PARENT_MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_development_train_manifest.json"
PARENT_FEATURE_MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_development_feature_manifest.json"
PARENT_EXECUTION_LOCK_PATH = WORKSPACE / "protocol/pcra_u_development_execution_lock.json"
PARENT_COMMON_PATH = WORKSPACE / "protocol/pcra_u_development_common.py"
PARENT_TRAIN_PATH = WORKSPACE / "protocol/pcra_u_development_train.py"
CHECKPOINT_MANAGER_PATH = WORKSPACE / "protocol/training_checkpoint_manager.py"
FEATURE_CACHE_ROOT = WORKSPACE / "results/pcra_u_feature_cache/pcra_u_development_v1"
PARENT_RUN_ROOT = WORKSPACE / "results/pcra_u_runs/pcra_u_development_train_20260824"
PARENT_REPORT_PATH = PARENT_RUN_ROOT / "PCRA_U_DEVELOPMENT_TRAIN_REPORT.json"

LOCKED_SEEDS = (24082026, 24082027, 24082028)
LOCKED_CANARY_EPOCHS = 2
LOCKED_FULL_EPOCHS = 20
LOCKED_BATCH_SIZE = 8
LOCKED_LEARNING_RATE = 1e-4
LOCKED_WEIGHT_DECAY = 1e-3
LOCKED_WARMUP_STEPS = 200
LOCKED_GRADIENT_CLIP = 5.0
LOCKED_DROPOUT = 0.1
LOCKED_CLEAN_FOUND_MULTIPLIER = 4.0
EXPECTED_FULL_COUNTS = {
    "train_samples": 1600,
    "dev_samples": 400,
    "train_families": 320,
    "dev_families": 80,
    "dev_found": 168,
    "dev_clean_found": 32,
    "dev_false_found_denominator": 95,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE.resolve()))


def _require_exact(name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        try:
            equal = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
        except (TypeError, ValueError):
            equal = False
    else:
        equal = actual == expected
    if not equal:
        raise DevelopmentTrainingError(f"Locked V1.1 value mismatch for {name}: {actual!r} != {expected!r}")


def _first_mapping_value(mapping: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    for key in candidates:
        if key in mapping:
            return mapping[key]
    return None


def _resolve_locked_path(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise DevelopmentTrainingError(f"V1.1 manifest must lock {label} as a relative string path")
    return safe_resolve(WORKSPACE, value)


def _manifest_campaign_spec(manifest: Mapping[str, Any]) -> dict[str, Any]:
    campaign = manifest.get("campaign", {})
    if not isinstance(campaign, Mapping):
        campaign = {}
    run_roots = manifest.get("run_roots", {})
    if not isinstance(run_roots, Mapping):
        run_roots = {}
    execution_paths = manifest.get("execution_paths", {})
    if not isinstance(execution_paths, Mapping):
        execution_paths = {}

    canary_value = (
        _first_mapping_value(campaign, ["canary_run_root", "canary_root"])
        or _first_mapping_value(execution_paths, ["canary_run_root", "canary_root"])
        or _first_mapping_value(run_roots, ["canary", "canary_run_root"])
        or manifest.get("canary_run_root")
    )
    campaign_value = (
        _first_mapping_value(campaign, ["campaign_root", "aggregate_root", "run_root"])
        or _first_mapping_value(execution_paths, ["campaign_run_root", "campaign_root", "aggregate_root"])
        or _first_mapping_value(run_roots, ["campaign", "aggregate", "campaign_root"])
        or manifest.get("campaign_root")
    )
    # The locked manifest fixes run IDs and the date-stamped campaign identity.
    # Older manifests may omit redundant root strings; in that case these
    # deterministic add-only roots are derived without accepting a CLI override.
    canary_value = canary_value or "results/pcra_u_runs/pcra_u_development_v1_1_canary_20260824"
    campaign_value = campaign_value or "results/pcra_u_runs/pcra_u_development_v1_1_campaign_20260824"
    canary_root = _resolve_locked_path(canary_value, "canary_run_root")
    campaign_root = _resolve_locked_path(campaign_value, "campaign_root")

    raw_records = (
        campaign.get("seed_runs")
        or manifest.get("seed_runs")
        or manifest.get("seeds")
    )
    if not isinstance(raw_records, list) or len(raw_records) != len(LOCKED_SEEDS):
        raise DevelopmentTrainingError("V1.1 manifest must contain exactly three locked seed-run records")
    seed_root_map = manifest.get("seed_run_roots", {})
    if not isinstance(seed_root_map, Mapping):
        seed_root_map = run_roots.get("seeds", {}) if isinstance(run_roots.get("seeds"), Mapping) else {}
    records: list[dict[str, Any]] = []
    for raw in raw_records:
        if isinstance(raw, Mapping):
            seed = int(raw["seed"])
            root_value = raw.get("run_root") or raw.get("root")
            if root_value is None and isinstance(raw.get("run_id"), str):
                root_value = f"results/pcra_u_runs/pcra_u_development_{raw['run_id']}_20260824"
        else:
            seed = int(raw)
            root_value = seed_root_map.get(str(seed)) or seed_root_map.get(seed)
        if root_value is None:
            root_value = f"results/pcra_u_runs/pcra_u_development_v1_1_seed_{seed}_20260824"
        records.append({"seed": seed, "run_root": _resolve_locked_path(root_value, f"run_root for seed {seed}")})
    if tuple(record["seed"] for record in records) != LOCKED_SEEDS:
        raise DevelopmentTrainingError(
            f"Locked seed order must be {list(LOCKED_SEEDS)}, got {[record['seed'] for record in records]}"
        )
    roots = [canary_root, campaign_root, *(record["run_root"] for record in records)]
    if len({path.resolve() for path in roots}) != len(roots):
        raise DevelopmentTrainingError("Canary, campaign, and all seed run roots must be distinct")
    if any(path == PARENT_RUN_ROOT or PARENT_RUN_ROOT in path.parents for path in roots):
        raise DevelopmentTrainingError("A V1.1 output root overlaps the immutable V1 result root")
    return {
        "canary_root": canary_root,
        "campaign_root": campaign_root,
        "seed_runs": records,
    }


def _eligibility_spec(manifest: Mapping[str, Any]) -> dict[str, float | int]:
    raw = manifest.get("eligibility")
    if not isinstance(raw, Mapping):
        selection = manifest.get("checkpoint_selection", {})
        raw = selection.get("eligibility") if isinstance(selection, Mapping) else None
    if not isinstance(raw, Mapping):
        raise DevelopmentTrainingError("V1.1 manifest lacks checkpoint eligibility thresholds")
    result: dict[str, float | int] = {
        "min_epoch": int(raw["min_epoch"]),
        "min_all_found_point_in_target": float(raw["min_all_found_point_in_target"]),
        "min_answerability_macro_f1": float(raw["min_answerability_macro_f1"]),
        "max_false_found_rate": float(raw["max_false_found_rate"]),
    }
    _require_exact("eligibility.min_epoch", result["min_epoch"], 6)
    for key in [
        "min_all_found_point_in_target",
        "min_answerability_macro_f1",
        "max_false_found_rate",
    ]:
        value = float(result[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise DevelopmentTrainingError(f"Invalid eligibility threshold {key}: {value}")
    return result


def validate_locked_config(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("protocol_id") != PROTOCOL_ID or manifest.get("protocol_id") != PROTOCOL_ID:
        raise DevelopmentTrainingError("V1.1 config/manifest protocol_id mismatch")
    _require_exact("model.dropout", config["model"]["dropout"], LOCKED_DROPOUT)
    _require_exact(
        "loss.clean_found_heatmap_multiplier",
        config["loss"]["clean_found_heatmap_multiplier"],
        LOCKED_CLEAN_FOUND_MULTIPLIER,
    )
    _require_exact("loss.heatmap_weight", config["loss"]["heatmap_weight"], 1.0)
    _require_exact("loss.answerability_weight", config["loss"]["answerability_weight"], 0.5)
    _require_exact("loss.source_weight", config["loss"]["source_weight"], 0.5)
    optimization = config["optimization"]
    _require_exact("optimization.learning_rate", optimization["learning_rate"], LOCKED_LEARNING_RATE)
    _require_exact("optimization.weight_decay", optimization["weight_decay"], LOCKED_WEIGHT_DECAY)
    _require_exact("optimization.batch_size", optimization["batch_size"], LOCKED_BATCH_SIZE)
    _require_exact("optimization.max_epochs", optimization["max_epochs"], LOCKED_FULL_EPOCHS)
    _require_exact("optimization.min_epochs", optimization["min_epochs"], 6)
    _require_exact("optimization.warmup_steps", optimization["warmup_steps"], LOCKED_WARMUP_STEPS)
    _require_exact("optimization.gradient_clip_norm", optimization["gradient_clip_norm"], LOCKED_GRADIENT_CLIP)
    if optimization.get("selection_metric") not in {None, SELECTION_METRIC}:
        raise DevelopmentTrainingError("Config selection metric does not match V1.1 scientific score")
    if optimization.get("selection_mode") not in {None, SELECTION_MODE}:
        raise DevelopmentTrainingError("Config selection mode must be max")
    locked_score = optimization.get("selection_score", {})
    expected_score = {
        "clean_found_point_in_target": 0.45,
        "all_found_point_in_target": 0.20,
        "answerability_macro_f1": 0.20,
        "one_minus_false_found_rate": 0.15,
    }
    if not isinstance(locked_score, Mapping):
        raise DevelopmentTrainingError("Config lacks the locked scientific selection score")
    for key, value in expected_score.items():
        _require_exact(f"optimization.selection_score.{key}", locked_score.get(key), value)
    canary = config.get("canary", {})
    if isinstance(canary, Mapping) and "epochs" in canary:
        _require_exact("canary.epochs", canary["epochs"], LOCKED_CANARY_EPOCHS)
    safety = config.get("safety", {})
    forbidden = set(safety.get("forbidden_splits", [])) if isinstance(safety, Mapping) else set()
    if not {"calibration", "test_iid", "test_ood"}.issubset(forbidden):
        raise DevelopmentTrainingError("V1.1 safety lock does not seal all forbidden splits")
    for key in ["calibration_fit", "test_opened", "baseline_modification"]:
        if bool(safety.get(key, False)):
            raise DevelopmentTrainingError(f"V1.1 safety flag must remain false: {key}")
    for key in ["v1_1_calibration_fit", "robot_publish", "parent_artifact_modification"]:
        if safety.get(key) is not False:
            raise DevelopmentTrainingError(f"V1.1 safety flag must be explicitly false: {key}")
    if config["model"].get("roborefer_frozen") is not True:
        raise DevelopmentTrainingError("RoboRefer must remain frozen")
    if config["model"].get("fresh_sidecar_initialization") is not True:
        raise DevelopmentTrainingError("V1.1 must lock fresh sidecar initialization")
    if config["model"].get("checkpoint_initialization_forbidden") is not True:
        raise DevelopmentTrainingError("V1.1 checkpoint initialization must be forbidden")
    raw_seed_runs = manifest.get("seed_runs")
    if not isinstance(raw_seed_runs, list):
        raise DevelopmentTrainingError("V1.1 manifest seed_runs are absent")
    for record in raw_seed_runs:
        if not isinstance(record, Mapping):
            raise DevelopmentTrainingError("Every V1.1 seed run must be a locked record")
        _require_exact(f"seed {record.get('seed')} epochs", record.get("epochs"), LOCKED_FULL_EPOCHS)
        if record.get("resume_checkpoint", "missing") is not None:
            raise DevelopmentTrainingError(f"Seed {record.get('seed')} is not fresh initialization")
        if record.get("initialization") != "fresh_sidecar_random_initialization":
            raise DevelopmentTrainingError(f"Seed {record.get('seed')} initialization policy drifted")
    checkpoint_selection = manifest.get("checkpoint_selection", {})
    if not isinstance(checkpoint_selection, Mapping):
        raise DevelopmentTrainingError("V1.1 checkpoint-selection lock is absent")
    _require_exact("checkpoint_selection.split", checkpoint_selection.get("split"), "dev")
    _require_exact("checkpoint_selection.metric", checkpoint_selection.get("metric"), SELECTION_METRIC)
    _require_exact("checkpoint_selection.mode", checkpoint_selection.get("mode"), SELECTION_MODE)
    _require_exact(
        "checkpoint_selection.pointer_name",
        checkpoint_selection.get("pointer_name"),
        f"best_{SELECTION_METRIC}.json",
    )
    if checkpoint_selection.get("total_loss_is_selection_metric") is not False:
        raise DevelopmentTrainingError("Development total loss cannot select a V1.1 checkpoint")
    boundary = manifest.get("calibration_test_boundary", {})
    if not isinstance(boundary, Mapping):
        raise DevelopmentTrainingError("V1.1 calibration/test boundary is absent")
    for key in ["v1_1_calibration_fit", "test_iid_opened", "test_ood_opened"]:
        if boundary.get(key) is not False:
            raise DevelopmentTrainingError(f"V1.1 boundary must explicitly keep {key}=false")
    if boundary.get("historical_v1_calibration_reuse_for_v1_1") is not False:
        raise DevelopmentTrainingError("Historical V1 calibration reuse is forbidden")
    return {
        "campaign": _manifest_campaign_spec(manifest),
        "eligibility": _eligibility_spec(manifest),
    }


def verify_preflight_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    if not PREFLIGHT_LOCK_PATH.is_file():
        raise DevelopmentTrainingError(f"V1.1 preflight lock is required: {PREFLIGHT_LOCK_PATH}")
    lock = read_json(PREFLIGHT_LOCK_PATH)
    status = str(lock.get("status", ""))
    if lock.get("protocol_id") != PROTOCOL_ID or not status.startswith("LOCKED"):
        raise DevelopmentTrainingError("V1.1 preflight lock has wrong protocol/status")
    artifacts = lock.get("artifacts", lock.get("locked_artifacts"))
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise DevelopmentTrainingError("V1.1 preflight lock has no artifact hashes")
    required = {
        _relative(CONFIG_PATH),
        _relative(MANIFEST_PATH),
        _relative(COMMON_PATH),
        _relative(TRAIN_PATH),
    }
    missing_required = sorted(required - set(artifacts))
    if missing_required:
        raise DevelopmentTrainingError(f"V1.1 preflight lock omits required artifacts: {missing_required}")
    mismatches = []
    for relative, expected in artifacts.items():
        path = safe_resolve(WORKSPACE, str(relative))
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(str(relative))
    if mismatches:
        raise DevelopmentTrainingError(f"V1.1 preflight-lock artifact mismatch: {mismatches}")
    locked_config_hash = artifacts.get(_relative(CONFIG_PATH))
    if locked_config_hash != sha256_file(CONFIG_PATH):
        raise DevelopmentTrainingError("Active V1.1 config is not the preflight-locked config")
    return lock


def _protected_snapshot() -> dict[str, Any]:
    parent_config = read_json(PARENT_CONFIG_PATH)
    parent_manifest = read_json(PARENT_MANIFEST_PATH)
    current = protected_now(parent_manifest, parent_config)
    expected = parent_manifest["protected_inputs"]
    for key, value in current.items():
        if expected.get(key) != value:
            raise DevelopmentTrainingError(f"Protected parent input mismatch: {key}")
    return current


def _parent_file_paths() -> list[Path]:
    paths = [
        PARENT_CONFIG_PATH,
        PARENT_MANIFEST_PATH,
        PARENT_FEATURE_MANIFEST_PATH,
        PARENT_EXECUTION_LOCK_PATH,
        PARENT_COMMON_PATH,
        PARENT_TRAIN_PATH,
        CHECKPOINT_MANAGER_PATH,
        PARENT_REPORT_PATH,
    ]
    report = read_json(PARENT_REPORT_PATH)
    selected_value = report.get("selected_checkpoint")
    if not isinstance(selected_value, str):
        raise DevelopmentTrainingError("Parent report lacks selected_checkpoint")
    selected = safe_resolve(WORKSPACE, selected_value)
    for name in ["metadata.json", "manifest.json", "model.safetensors", "training_state.pt"]:
        paths.append(selected / name)
    return paths


def parent_integrity_snapshot(*, feature_scope: str) -> dict[str, Any]:
    verify_execution_lock()
    index_path, index = verify_feature_index(feature_scope)
    files = _parent_file_paths()
    missing = [_relative(path) for path in files if not path.is_file()]
    if missing:
        raise DevelopmentTrainingError(f"Missing immutable parent artifacts: {missing}")
    hashes = {_relative(path): sha256_file(path) for path in files}
    locked_parent = read_json(MANIFEST_PATH).get("parent_artifacts", {})
    if not isinstance(locked_parent, Mapping):
        raise DevelopmentTrainingError("V1.1 manifest lacks locked parent hashes")
    mismatches = [
        relative
        for relative, digest in hashes.items()
        if relative in locked_parent and locked_parent[relative] != digest
    ]
    if mismatches:
        raise DevelopmentTrainingError(f"Immutable V1 parent artifact mismatch: {mismatches}")
    index_relative = _relative(index_path)
    index_digest = sha256_file(index_path)
    if index_relative in locked_parent and locked_parent[index_relative] != index_digest:
        raise DevelopmentTrainingError(f"Immutable feature-cache index mismatch: {index_relative}")
    return {
        "protected": _protected_snapshot(),
        "parent_file_sha256": hashes,
        "feature_cache_index": {
            "path": _relative(index_path),
            "sha256": index_digest,
            "scope": index["scope"],
            "samples": index["counts"]["samples"],
            "features": index["counts"].get("features", len(index["features"])),
        },
    }


def _verify_split_sealing(entries: Sequence[Mapping[str, Any]]) -> None:
    splits = {entry.get("split") for entry in entries}
    if not splits or not splits.issubset({"train", "dev"}):
        raise DevelopmentTrainingError(f"Runtime manifest contains a forbidden split: {sorted(splits)}")
    for entry in entries:
        for section in [entry.get("feature_input", {}), entry.get("supervision", {})]:
            if isinstance(section, Mapping):
                for key, value in section.items():
                    if key.endswith("_path") and isinstance(value, str):
                        safe_resolve(WORKSPACE, value)


def checkpoint_identity_v11(index_path: Path, *, seed: int, scope: str, run_root: Path) -> dict[str, str]:
    descriptor = {
        "protocol_id": PROTOCOL_ID,
        "seed": seed,
        "scope": scope,
        "run_root": _relative(run_root),
        "fresh_initialization": True,
        "resume_checkpoint": None,
    }
    return {
        "v1_1_config_sha256": sha256_file(CONFIG_PATH),
        "v1_1_manifest_sha256": sha256_file(MANIFEST_PATH),
        "v1_1_preflight_lock_sha256": sha256_file(PREFLIGHT_LOCK_PATH),
        "v1_1_common_code_sha256": sha256_file(COMMON_PATH),
        "v1_1_training_code_sha256": sha256_file(TRAIN_PATH),
        "parent_config_sha256": sha256_file(PARENT_CONFIG_PATH),
        "parent_train_manifest_sha256": sha256_file(PARENT_MANIFEST_PATH),
        "parent_feature_manifest_sha256": sha256_file(PARENT_FEATURE_MANIFEST_PATH),
        "parent_execution_lock_sha256": sha256_file(PARENT_EXECUTION_LOCK_PATH),
        "feature_cache_index_sha256": sha256_file(index_path),
        "parent_common_code_sha256": sha256_file(PARENT_COMMON_PATH),
        "checkpoint_manager_sha256": sha256_file(CHECKPOINT_MANAGER_PATH),
        "seed_run_descriptor_sha256": canonical_sha256(descriptor),
    }


def _copy_locked_inputs(run_root: Path, index_path: Path, *, canary_report: Path | None = None) -> None:
    destination = run_root / "locked_inputs"
    destination.mkdir(parents=True, exist_ok=False)
    paths = [
        CONFIG_PATH,
        MANIFEST_PATH,
        PREFLIGHT_LOCK_PATH,
        COMMON_PATH,
        TRAIN_PATH,
        PARENT_CONFIG_PATH,
        PARENT_MANIFEST_PATH,
        PARENT_FEATURE_MANIFEST_PATH,
        PARENT_EXECUTION_LOCK_PATH,
        index_path,
        PARENT_REPORT_PATH,
    ]
    if canary_report is not None:
        paths.append(canary_report)
    seen_names: set[str] = set()
    for path in paths:
        if path.name in seen_names:
            raise DevelopmentTrainingError(f"Locked-input basename collision: {path.name}")
        seen_names.add(path.name)
        shutil.copy2(path, destination / path.name)


def _create_run_root(run_root: Path, directories: Sequence[str], index_path: Path, *, canary_report: Path | None = None) -> None:
    if run_root.exists():
        raise DevelopmentTrainingError(f"Refusing to overwrite V1.1 run root: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    for directory in directories:
        (run_root / directory).mkdir(parents=True, exist_ok=False)
    _copy_locked_inputs(run_root, index_path, canary_report=canary_report)


def _ensure_all_absent(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise DevelopmentTrainingError(f"V1.1 add-only roots already exist; refusing any write: {existing}")


def _training_sequence(entries: Sequence[Mapping[str, Any]], epochs: int, batch_size: int, seed: int) -> list[list[int]]:
    sequence: list[list[int]] = []
    for epoch in range(epochs):
        order = deterministic_entry_order(entries, seed, epoch)
        sequence.extend(chunks(order, batch_size))
    return sequence


def _eligibility(metrics: Mapping[str, Any], epoch: int, thresholds: Mapping[str, float | int]) -> dict[str, Any]:
    ga = float(metrics["grounding_found"]["point_in_target"])
    answerability = float(metrics["answerability"]["macro_f1"])
    false_found = float(metrics["false_found_on_ambiguous_or_absent"]["rate"])
    checks = {
        "epoch_at_least_minimum": epoch >= int(thresholds["min_epoch"]),
        "all_found_grounding_at_least_minimum": ga >= float(thresholds["min_all_found_point_in_target"]),
        "answerability_macro_f1_at_least_minimum": answerability
        >= float(thresholds["min_answerability_macro_f1"]),
        "false_found_rate_at_most_maximum": false_found <= float(thresholds["max_false_found_rate"]),
    }
    return {"eligible": all(checks.values()), "checks": checks, "thresholds": dict(thresholds)}


def _metric_row(epoch: int, global_step: int, train_loss: float, dev: Mapping[str, Any], eligibility: Mapping[str, Any]) -> dict[str, Any]:
    components = dev["selection_components"]
    return {
        "epoch": epoch,
        "global_step": global_step,
        "train_epoch_mean_total_loss": train_loss,
        "dev_total_loss": dev["loss"]["total"],
        "G_c_clean_found_raw_point_in_target": components["G_c_clean_found_raw_point_in_target"],
        "clean_joint_success": components["clean_joint_success"],
        "G_a_all_found_raw_point_in_target": components["G_a_all_found_raw_point_in_target"],
        "A_answerability_macro_f1": components["A_answerability_macro_f1"],
        "F_false_found_rate": components["F_false_found_rate"],
        "S_source_micro_f1": components["S_source_micro_f1"],
        "C_dev_scientific_score_v1_1": components["C_dev_scientific_score_v1_1"],
        "eligible": eligibility["eligible"],
    }


def _finalize_best_pointer(checkpoint_root: Path, history: Sequence[Mapping[str, Any]]) -> tuple[Path | None, dict[str, Any] | None]:
    eligible = [row for row in history if row["selection"]["eligible"]]
    if not eligible:
        return None, None
    chosen = min(
        eligible,
        key=lambda row: (
            -float(row["selection"]["value"]),
            float(row["dev"]["false_found_on_ambiguous_or_absent"]["rate"]),
            -float(row["dev"]["clean_found"]["raw_point_in_target"]),
            int(row["epoch"]),
        ),
    )
    directory = checkpoint_root / f"step_{int(chosen['global_step']):09d}"
    metadata = read_json(directory / "metadata.json")
    pointer = {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "checkpoint_id": directory.name,
        "checkpoint_path": directory.name,
        "manifest_sha256": sha256_file(directory / "manifest.json"),
        "selection": metadata["selection"],
        "updated_at_utc": utc_now(),
        "tie_break": "max_C_then_min_F_then_max_Gc_then_earliest_epoch",
    }
    pointer_path = checkpoint_root / f"best_{SELECTION_METRIC}.json"
    write_json(pointer_path, pointer)
    return directory, chosen


def _checkpoint_table(run_root: Path, history: Sequence[Mapping[str, Any]], chosen: Path | None) -> None:
    rows = []
    for row in history:
        checkpoint = f"step_{int(row['global_step']):09d}"
        metric = _metric_row(
            int(row["epoch"]),
            int(row["global_step"]),
            float(row["train_epoch_mean_total_loss"]),
            row["dev"],
            row["selection"],
        )
        metric.update({"checkpoint": checkpoint, "selected": chosen is not None and checkpoint == chosen.name})
        rows.append(metric)
    fields = list(rows[0]) if rows else []
    if rows:
        write_csv(run_root / "tables/checkpoint_trajectory.csv", fields, rows)


def train_one_run(
    *,
    run_root: Path,
    run_name: str,
    scope: str,
    seed: int,
    epochs: int,
    train_dataset: FrozenFeatureDataset,
    dev_dataset: FrozenFeatureDataset,
    train_entries: Sequence[Mapping[str, Any]],
    dev_entries: Sequence[Mapping[str, Any]],
    all_train_entries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    identity: Mapping[str, str],
    thresholds: Mapping[str, float | int],
    permit_best: bool,
) -> dict[str, Any]:
    seed_runtime(seed, strict=True)
    if not torch.cuda.is_available():
        raise DevelopmentTrainingError("V1.1 locked training requires CUDA")
    device = torch.device("cuda")
    model = PCRAUDevelopmentV11(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LOCKED_LEARNING_RATE,
        weight_decay=LOCKED_WEIGHT_DECAY,
    )
    batch_size = LOCKED_BATCH_SIZE
    steps_per_epoch = math.ceil(len(train_entries) / batch_size)
    total_steps = epochs * steps_per_epoch
    scheduler = scheduler_for(optimizer, total_steps, LOCKED_WARMUP_STEPS)
    answer_weights, source_weights = global_class_weights(all_train_entries, device)
    checkpoint_root = run_root / "checkpoints" / run_name
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    log_path = run_root / "logs" / f"{run_name}_metrics.jsonl"
    sequence = _training_sequence(train_entries, epochs, batch_size, seed)
    global_step = 0
    history: list[dict[str, Any]] = []
    max_gradient: defaultdict[str, float] = defaultdict(float)
    all_step_values_finite = True
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    for epoch_index in range(epochs):
        epoch = epoch_index + 1
        epoch_rows: list[dict[str, Any]] = []
        epoch_batches = sequence[epoch_index * steps_per_epoch : epoch * steps_per_epoch]
        for batch_indices in epoch_batches:
            global_step += 1
            batch = train_dataset.make_batch(batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(model_inputs(batch))
            losses = compute_loss_v11(output, batch, answer_weights, source_weights, config)
            primary_loss_names = ["total", "heatmap", "heatmap_bce", "heatmap_dice", "answerability", "source"]
            finite = all(bool(torch.isfinite(losses[name]).item()) for name in primary_loss_names)
            if not finite:
                raise DevelopmentTrainingError(f"Non-finite V1.1 loss at {run_name} step {global_step}")
            losses["total"].backward()
            groups = gradient_groups(model)  # V1.1 intentionally preserves the named module boundary.
            for name, value in groups.items():
                if not math.isfinite(value):
                    raise DevelopmentTrainingError(f"Non-finite {name} gradient at step {global_step}")
                max_gradient[name] = max(max_gradient[name], value)
            total_gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), LOCKED_GRADIENT_CLIP).item())
            if not math.isfinite(total_gradient) or total_gradient <= 0.0:
                raise DevelopmentTrainingError(f"Invalid V1.1 gradient norm at step {global_step}: {total_gradient}")
            optimizer.step()
            scheduler.step()
            row = {
                "run": run_name,
                "scope": scope,
                "seed": seed,
                "epoch": epoch,
                "step": global_step,
                "total_loss": float(losses["total"].item()),
                "heatmap_loss": float(losses["heatmap"].item()),
                "heatmap_bce": float(losses["heatmap_bce"].item()),
                "heatmap_dice": float(losses["heatmap_dice"].item()),
                "answerability_loss": float(losses["answerability"].item()),
                "source_loss": float(losses["source"].item()),
                "gradient_norm_preclip": total_gradient,
                "target_head_gradient": groups["target_head"],
                "answer_head_gradient": groups["answer_head"],
                "source_head_gradient": groups["source_head"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "finite": True,
            }
            all_step_values_finite &= finite and math.isfinite(total_gradient)
            epoch_rows.append(row)
            append_jsonl(log_path, row)

        dev_metrics = evaluate_dataset_v11(
            model,
            dev_dataset,
            dev_entries,
            device,
            config,
            answer_weights,
            source_weights,
            keep_predictions=False,
        )
        mean_train_loss = statistics.fmean(row["total_loss"] for row in epoch_rows)
        score = float(dev_metrics["selection_components"]["C_dev_scientific_score_v1_1"])
        eligibility = _eligibility(dev_metrics, epoch, thresholds)
        eligible_for_best = bool(permit_best and eligibility["eligible"])
        save_checkpoint(
            checkpoint_root,
            run_id=f"{run_root.name}_{run_name}_seed_{seed}",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            selection_split="dev",
            selection_metric=SELECTION_METRIC,
            selection_mode=SELECTION_MODE,
            selection_value=score,
            eligible_for_best=eligible_for_best,
            metrics={
                "train_epoch_mean_total_loss": mean_train_loss,
                "dev_total_loss": float(dev_metrics["loss"]["total"]),
                "dev_clean_found_raw_point_in_target": float(dev_metrics["clean_found"]["raw_point_in_target"]),
                "dev_clean_joint_success": float(dev_metrics["clean_found"]["joint_found_and_point_in_target"]),
                "dev_all_found_raw_point_in_target": float(dev_metrics["grounding_found"]["point_in_target"]),
                "dev_answerability_macro_f1": float(dev_metrics["answerability"]["macro_f1"]),
                "dev_false_found_rate": float(dev_metrics["false_found_on_ambiguous_or_absent"]["rate"]),
                "dev_source_micro_f1": float(dev_metrics["source_micro_f1"]),
                "dev_scientific_score_v1_1": score,
            },
            identity=identity,
            sampler_state={
                "kind": "deterministic_hash_order",
                "seed": seed,
                "next_epoch": epoch + 1,
                "next_global_step": global_step + 1,
            },
            extra_state={
                "fresh_initialization": True,
                "resume_checkpoint": None,
                "steps_per_epoch": steps_per_epoch,
                "eligibility": eligibility,
            },
            runtime={
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "seed": seed,
                "scope": scope,
                "dropout": LOCKED_DROPOUT,
                "sidecar_dtype": "torch.float32",
                "feature_cache_dtype": "torch.float16",
            },
        )
        selection = {
            "metric": SELECTION_METRIC,
            "mode": SELECTION_MODE,
            "value": score,
            "eligible": eligible_for_best,
            "eligibility_checks": eligibility["checks"],
            "thresholds": eligibility["thresholds"],
        }
        epoch_summary = {
            "epoch": epoch,
            "global_step": global_step,
            "train_epoch_mean_total_loss": mean_train_loss,
            "dev": dev_metrics,
            "selection": selection,
        }
        history.append(epoch_summary)
        write_json(run_root / "metrics" / f"{run_name}_epoch_{epoch:03d}.json", epoch_summary)
        print(
            f"V11 {scope} seed={seed} epoch={epoch}/{epochs} step={global_step} "
            f"train={mean_train_loss:.5f} C={score:.6f} "
            f"Gc={dev_metrics['clean_found']['raw_point_in_target']:.4f} "
            f"Ga={dev_metrics['grounding_found']['point_in_target']:.4f} "
            f"A={dev_metrics['answerability']['macro_f1']:.4f} "
            f"F={dev_metrics['false_found_on_ambiguous_or_absent']['rate']:.4f} "
            f"eligible={eligible_for_best}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    chosen, chosen_history = _finalize_best_pointer(checkpoint_root, history) if permit_best else (None, None)
    audit = audit_checkpoint_root(checkpoint_root)
    _checkpoint_table(run_root, history, chosen)
    selected_metrics = None
    selected_predictions = None
    load_verification = None
    if chosen is not None and chosen_history is not None:
        load_verification = load_model_for_evaluation(chosen, model=model, expected_identity=identity)
        selected_metrics = evaluate_dataset_v11(
            model,
            dev_dataset,
            dev_entries,
            device,
            config,
            answer_weights,
            source_weights,
            keep_predictions=True,
        )
        selected_predictions = selected_metrics.pop("predictions")
        expected_score = float(chosen_history["selection"]["value"])
        measured_score = float(selected_metrics["selection_components"]["C_dev_scientific_score_v1_1"])
        if not math.isclose(expected_score, measured_score, rel_tol=0.0, abs_tol=1e-12):
            raise DevelopmentTrainingError(
                f"Selected checkpoint metric replay mismatch: {expected_score} != {measured_score}"
            )

    result = {
        "run_name": run_name,
        "scope": scope,
        "seed": seed,
        "fresh_initialization": True,
        "resume_checkpoint": None,
        "epochs_completed": len(history),
        "global_steps": global_step,
        "steps_per_epoch": steps_per_epoch,
        "elapsed_seconds": elapsed,
        "steps_per_second": global_step / elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "trainable_parameters": trainable_parameter_count(model),
        "max_gradient_norms": dict(max_gradient),
        "all_step_values_finite": all_step_values_finite,
        "history": history,
        "checkpoint_root": _relative(checkpoint_root),
        "checkpoint_audit": audit,
        "selected_checkpoint": _relative(chosen) if chosen is not None else None,
        "selected_history": chosen_history,
        "selected_metrics": selected_metrics,
        "selected_predictions": selected_predictions,
        "selected_load_verification": load_verification,
    }
    del optimizer, scheduler, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _load_locked_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = read_json(CONFIG_PATH)
    manifest = read_json(MANIFEST_PATH)
    parsed = validate_locked_config(config, manifest)
    lock = verify_preflight_lock(config)
    return config, manifest, parsed, lock


def run_canary() -> dict[str, Any]:
    config, manifest, parsed, lock = _load_locked_inputs()
    canary_root: Path = parsed["campaign"]["canary_root"]
    _ensure_all_absent([canary_root])
    before = parent_integrity_snapshot(feature_scope="canary")
    index_path, index = verify_feature_index("canary")
    parent_manifest = read_json(PARENT_MANIFEST_PATH)
    entries = parent_manifest["entries"]
    _verify_split_sealing(entries)
    canary_ids = set(parent_manifest["canary_family_ids"])
    train_entries = [entry for entry in entries if entry["split"] == "train" and entry["family_id"] in canary_ids]
    dev_entries = [entry for entry in entries if entry["split"] == "dev" and entry["family_id"] in canary_ids]
    all_train_entries = [entry for entry in entries if entry["split"] == "train"]
    if len(train_entries) != 80 or len(dev_entries) != 20:
        raise DevelopmentTrainingError("V1.1 canary scope must be exactly 80 train / 20 dev samples")
    _create_run_root(
        canary_root,
        ["logs", "metrics", "checks", "tables", "checkpoints"],
        index_path,
    )
    train_dataset = FrozenFeatureDataset(WORKSPACE, train_entries, FEATURE_CACHE_ROOT, index, config)
    dev_dataset = FrozenFeatureDataset(WORKSPACE, dev_entries, FEATURE_CACHE_ROOT, index, config)
    seed = LOCKED_SEEDS[0]
    identity = checkpoint_identity_v11(index_path, seed=seed, scope="canary", run_root=canary_root)
    trained = train_one_run(
        run_root=canary_root,
        run_name="canary",
        scope="canary",
        seed=seed,
        epochs=LOCKED_CANARY_EPOCHS,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        train_entries=train_entries,
        dev_entries=dev_entries,
        all_train_entries=all_train_entries,
        config=config,
        identity=identity,
        thresholds=parsed["eligibility"],
        permit_best=False,
    )
    after = parent_integrity_snapshot(feature_scope="canary")
    gates = {
        "preflight_lock_verified": bool(lock),
        "exact_two_epochs": trained["epochs_completed"] == LOCKED_CANARY_EPOCHS,
        "exact_20_steps": trained["global_steps"] == 20,
        "fresh_initialization_no_resume": trained["fresh_initialization"] and trained["resume_checkpoint"] is None,
        "all_loss_and_gradient_values_finite": trained["all_step_values_finite"],
        "all_active_gradient_groups_nonzero": all(value > 1e-10 for value in trained["max_gradient_norms"].values()),
        "checkpoint_audit_pass": trained["checkpoint_audit"]["passed"],
        "no_best_pointer_before_eligibility_epoch": not (
            canary_root / "checkpoints/canary" / f"best_{SELECTION_METRIC}.json"
        ).exists(),
        "parent_integrity_unchanged": before == after,
        "only_train_and_dev_entries_loaded": True,
        "calibration_and_tests_sealed": True,
    }
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "scientific_status": "ENGINEERING_CANARY_NOT_SCIENTIFIC_RESULT",
        "passed": passed,
        "decision": "GO_V1_1_THREE_SEED_CAMPAIGN" if passed else "FIX_V1_1_CANARY_FIRST",
        "seed": seed,
        "run_root": _relative(canary_root),
        "gates": gates,
        "observed": {
            "epochs_completed": trained["epochs_completed"],
            "optimizer_steps": trained["global_steps"],
            "elapsed_seconds": trained["elapsed_seconds"],
            "steps_per_second": trained["steps_per_second"],
            "peak_vram_bytes": trained["peak_vram_bytes"],
            "trainable_parameters": trained["trainable_parameters"],
            "final_dev_metrics": trained["history"][-1]["dev"],
        },
        "checkpoint_audit": trained["checkpoint_audit"],
        "protected_before": before,
        "protected_after": after,
        "training_performed": True,
        "gradient_split": "train_canary_only",
        "selection_split": "dev_canary_diagnostic_only",
        "calibration_fit": False,
        "test_opened": False,
    }
    write_json(canary_root / "PCRA_U_DEVELOPMENT_V1_1_CANARY_REPORT.json", report)
    write_json(canary_root / "checks/canary_gate.json", report)
    return report


def _seed_report(run_root: Path, trained: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    selected = trained["selected_checkpoint"]
    selected_history = trained["selected_history"]
    selected_metrics = trained["selected_metrics"]
    selection = {
        "metric": SELECTION_METRIC,
        "mode": SELECTION_MODE,
        "eligible": selected is not None,
        "value": selected_history["selection"]["value"] if selected_history else None,
        "epoch": selected_history["epoch"] if selected_history else None,
        "global_step": selected_history["global_step"] if selected_history else None,
        "eligibility_checks": selected_history["selection"]["eligibility_checks"] if selected_history else None,
        "tie_break": "max_C_then_min_F_then_max_Gc_then_earliest_epoch",
    }
    exact_denominators = bool(
        selected_metrics
        and selected_metrics["sample_count"] == EXPECTED_FULL_COUNTS["dev_samples"]
        and selected_metrics["grounding_found"]["found_total"] == EXPECTED_FULL_COUNTS["dev_found"]
        and selected_metrics["clean_found"]["clean_found_total"] == EXPECTED_FULL_COUNTS["dev_clean_found"]
        and selected_metrics["false_found_on_ambiguous_or_absent"]["denominator"]
        == EXPECTED_FULL_COUNTS["dev_false_found_denominator"]
    )
    gates = {
        "fresh_initialization_no_resume": trained["fresh_initialization"] and trained["resume_checkpoint"] is None,
        "exact_20_epochs": trained["epochs_completed"] == LOCKED_FULL_EPOCHS,
        "exact_4000_optimizer_steps": trained["global_steps"] == LOCKED_FULL_EPOCHS * 200,
        "all_loss_and_gradient_values_finite": trained["all_step_values_finite"],
        "checkpoint_each_epoch": len(trained["checkpoint_audit"]["valid_checkpoints"]) == LOCKED_FULL_EPOCHS,
        "checkpoint_audit_pass": trained["checkpoint_audit"]["passed"],
        "eligible_checkpoint_selected": selected is not None,
        "selected_checkpoint_safetensors_verified": bool(
            trained["selected_load_verification"]
            and trained["selected_load_verification"]["verification"]["passed"]
        ),
        "exact_locked_dev_denominators": exact_denominators,
        "parent_integrity_unchanged": before == after,
        "calibration_and_tests_sealed": True,
    }
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "scientific_status": "EXPLORATORY_DEV_ONLY_CALIBRATION_TESTS_SEALED",
        "passed": passed,
        "decision": "V1_1_SEED_COMPLETE" if passed else "V1_1_SEED_FAILED_GATES",
        "seed": trained["seed"],
        "run_root": _relative(run_root),
        "selected_checkpoint": selected,
        "selected_checkpoint_model_sha256": (
            sha256_file(safe_resolve(WORKSPACE, selected) / "model.safetensors") if selected else None
        ),
        "selected_checkpoint_dev_metrics": selected_metrics,
        "selection": selection,
        "gates": gates,
        "observed": {
            "epochs_completed": trained["epochs_completed"],
            "optimizer_steps": trained["global_steps"],
            "elapsed_seconds": trained["elapsed_seconds"],
            "steps_per_second": trained["steps_per_second"],
            "peak_vram_bytes": trained["peak_vram_bytes"],
            "trainable_parameters": trained["trainable_parameters"],
        },
        "checkpoint_audit": trained["checkpoint_audit"],
        "protected_before": before,
        "protected_after": after,
        "gradient_split": "train",
        "selection_split": "dev",
        "training_performed": True,
        "calibration_fit": False,
        "test_opened": False,
    }
    write_json(run_root / "PCRA_U_DEVELOPMENT_V1_1_SEED_REPORT.json", report)
    write_json(run_root / "checks/seed_gate.json", report)
    if selected is not None:
        prediction_payload = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "scientific_status": "EXPLORATORY_DEV_ONLY",
            "seed": trained["seed"],
            "selection": selection,
            "selected_checkpoint": selected,
            "selected_checkpoint_model_sha256": report["selected_checkpoint_model_sha256"],
            "metrics": selected_metrics,
            "predictions": trained["selected_predictions"],
        }
        write_json(run_root / "predictions/selected_dev_predictions_exploratory.json", prediction_payload)
    return report


def _aggregate_metric(seed_reports: Sequence[Mapping[str, Any]], path: Sequence[str]) -> dict[str, Any]:
    values: list[float] = []
    for report in seed_reports:
        value: Any = report
        for key in path:
            value = value[key]
        values.append(float(value))
    return {
        "values_by_seed": {str(report["seed"]): value for report, value in zip(seed_reports, values)},
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _campaign_winner(seed_reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [report for report in seed_reports if report["selection"]["eligible"]]
    if not eligible:
        raise DevelopmentTrainingError("No seed produced an eligible V1.1 checkpoint")
    return min(
        eligible,
        key=lambda report: (
            -float(report["selection"]["value"]),
            float(report["selected_checkpoint_dev_metrics"]["false_found_on_ambiguous_or_absent"]["rate"]),
            -float(report["selected_checkpoint_dev_metrics"]["clean_found"]["raw_point_in_target"]),
            int(report["seed"]),
        ),
    )


def run_campaign() -> dict[str, Any]:
    config, manifest, parsed, lock = _load_locked_inputs()
    campaign_spec = parsed["campaign"]
    canary_report_path = campaign_spec["canary_root"] / "PCRA_U_DEVELOPMENT_V1_1_CANARY_REPORT.json"
    canary = read_json(canary_report_path)
    canary_gates = canary.get("gates", {})
    if (
        canary.get("protocol_id") != PROTOCOL_ID
        or canary.get("passed") is not True
        or canary.get("decision") != "GO_V1_1_THREE_SEED_CAMPAIGN"
        or int(canary.get("seed", -1)) != LOCKED_SEEDS[0]
        or not isinstance(canary_gates, Mapping)
        or not canary_gates
        or not all(value is True for value in canary_gates.values())
    ):
        raise DevelopmentTrainingError("Three-seed campaign is blocked until the locked V1.1 canary passes")
    output_roots = [campaign_spec["campaign_root"], *(row["run_root"] for row in campaign_spec["seed_runs"])]
    _ensure_all_absent(output_roots)
    before_campaign = parent_integrity_snapshot(feature_scope="full")
    index_path, index = verify_feature_index("full")
    parent_manifest = read_json(PARENT_MANIFEST_PATH)
    entries = parent_manifest["entries"]
    _verify_split_sealing(entries)
    train_entries = [entry for entry in entries if entry["split"] == "train"]
    dev_entries = [entry for entry in entries if entry["split"] == "dev"]
    if (
        len(train_entries) != EXPECTED_FULL_COUNTS["train_samples"]
        or len(dev_entries) != EXPECTED_FULL_COUNTS["dev_samples"]
        or len({entry["family_id"] for entry in train_entries}) != EXPECTED_FULL_COUNTS["train_families"]
        or len({entry["family_id"] for entry in dev_entries}) != EXPECTED_FULL_COUNTS["dev_families"]
    ):
        raise DevelopmentTrainingError("Full V1.1 train/dev scope differs from the locked 320/80-family split")

    campaign_root: Path = campaign_spec["campaign_root"]
    _create_run_root(
        campaign_root,
        ["checks", "tables", "seed_reports"],
        index_path,
        canary_report=canary_report_path,
    )
    # These dataset objects retain the verified frozen tensors in CPU memory and
    # are intentionally reused across all seeds.  No cache extraction is rerun.
    train_dataset = FrozenFeatureDataset(WORKSPACE, train_entries, FEATURE_CACHE_ROOT, index, config)
    dev_dataset = FrozenFeatureDataset(WORKSPACE, dev_entries, FEATURE_CACHE_ROOT, index, config)
    seed_reports: list[dict[str, Any]] = []
    for record in campaign_spec["seed_runs"]:
        seed = int(record["seed"])
        run_root: Path = record["run_root"]
        before_seed = parent_integrity_snapshot(feature_scope="full")
        _create_run_root(
            run_root,
            ["logs", "metrics", "checks", "tables", "predictions", "checkpoints"],
            index_path,
            canary_report=canary_report_path,
        )
        identity = checkpoint_identity_v11(index_path, seed=seed, scope="full", run_root=run_root)
        trained = train_one_run(
            run_root=run_root,
            run_name="development",
            scope="full",
            seed=seed,
            epochs=LOCKED_FULL_EPOCHS,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            train_entries=train_entries,
            dev_entries=dev_entries,
            all_train_entries=train_entries,
            config=config,
            identity=identity,
            thresholds=parsed["eligibility"],
            permit_best=True,
        )
        after_seed = parent_integrity_snapshot(feature_scope="full")
        report = _seed_report(run_root, trained, before_seed, after_seed)
        seed_reports.append(report)
        shutil.copy2(
            run_root / "PCRA_U_DEVELOPMENT_V1_1_SEED_REPORT.json",
            campaign_root / "seed_reports" / f"seed_{seed}.json",
        )

    after_campaign = parent_integrity_snapshot(feature_scope="full")
    all_selected = all(report["selection"]["eligible"] for report in seed_reports)
    winner = _campaign_winner(seed_reports) if all_selected else None
    aggregate_paths = {
        "C_dev_scientific_score_v1_1": ["selection", "value"],
        "G_c_clean_found_raw_point_in_target": ["selected_checkpoint_dev_metrics", "clean_found", "raw_point_in_target"],
        "clean_joint_success": [
            "selected_checkpoint_dev_metrics",
            "clean_found",
            "joint_found_and_point_in_target",
        ],
        "G_a_all_found_raw_point_in_target": ["selected_checkpoint_dev_metrics", "grounding_found", "point_in_target"],
        "A_answerability_macro_f1": ["selected_checkpoint_dev_metrics", "answerability", "macro_f1"],
        "F_false_found_rate": [
            "selected_checkpoint_dev_metrics",
            "false_found_on_ambiguous_or_absent",
            "rate",
        ],
        "S_source_micro_f1": ["selected_checkpoint_dev_metrics", "source_micro_f1"],
        "dev_total_loss": ["selected_checkpoint_dev_metrics", "loss", "total"],
    }
    aggregates = (
        {name: _aggregate_metric(seed_reports, path) for name, path in aggregate_paths.items()}
        if all_selected
        else {}
    )
    gates = {
        "preflight_lock_verified": bool(lock),
        "canary_passed": canary.get("passed") is True,
        "exact_three_locked_seeds": tuple(report["seed"] for report in seed_reports) == LOCKED_SEEDS,
        "all_runs_exact_20_epochs_equal_search_budget": all(
            report["observed"]["epochs_completed"] == LOCKED_FULL_EPOCHS for report in seed_reports
        ),
        "all_runs_fresh_initialization_no_resume": all(
            report["gates"]["fresh_initialization_no_resume"] for report in seed_reports
        ),
        "all_seed_reports_pass": all(report["passed"] for report in seed_reports),
        "all_seed_winners_eligible": all_selected,
        "parent_integrity_unchanged": before_campaign == after_campaign,
        "only_train_and_dev_entries_loaded": True,
        "calibration_and_tests_sealed": True,
    }
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "scientific_status": "EXPLORATORY_DEV_ONLY_THREE_SEED_CALIBRATION_TESTS_SEALED",
        "passed": passed,
        "decision": "V1_1_CAMPAIGN_COMPLETE" if passed else "V1_1_CAMPAIGN_FAILED_GATES",
        "campaign_root": _relative(campaign_root),
        "selection_rule": {
            "metric": SELECTION_METRIC,
            "mode": SELECTION_MODE,
            "formula": "0.45*Gc + 0.20*Ga + 0.20*A + 0.15*(1-F)",
            "eligibility": parsed["eligibility"],
            "within_seed_tie_break": "max_C_then_min_F_then_max_Gc_then_earliest_epoch",
            "campaign_winner_tie_break": "max_C_then_min_F_then_max_Gc_then_lowest_seed",
        },
        "seed_reports": seed_reports,
        "aggregate_selected_checkpoint_metrics": aggregates,
        "campaign_winner": (
            {
                "seed": winner["seed"],
                "run_root": winner["run_root"],
                "checkpoint": winner["selected_checkpoint"],
                "model_sha256": winner["selected_checkpoint_model_sha256"],
                "score": winner["selection"]["value"],
                "note": "Deployment candidate only; scientific reporting uses all-seed aggregates.",
            }
            if winner is not None
            else None
        ),
        "gates": gates,
        "protected_before": before_campaign,
        "protected_after": after_campaign,
        "training_performed": True,
        "gradient_split": "train",
        "selection_split": "dev",
        "calibration_fit": False,
        "test_opened": False,
    }
    write_json(campaign_root / "PCRA_U_DEVELOPMENT_V1_1_CAMPAIGN_REPORT.json", report)
    write_json(campaign_root / "checks/campaign_gate.json", report)
    summary_rows = []
    for seed_report in seed_reports:
        metrics = seed_report["selected_checkpoint_dev_metrics"]
        summary_rows.append(
            {
                "seed": seed_report["seed"],
                "checkpoint": seed_report["selected_checkpoint"],
                "epoch": seed_report["selection"]["epoch"],
                "C": seed_report["selection"]["value"],
                "Gc": metrics["clean_found"]["raw_point_in_target"],
                "clean_joint": metrics["clean_found"]["joint_found_and_point_in_target"],
                "Ga": metrics["grounding_found"]["point_in_target"],
                "A": metrics["answerability"]["macro_f1"],
                "F": metrics["false_found_on_ambiguous_or_absent"]["rate"],
                "S": metrics["source_micro_f1"],
                "eligible": seed_report["selection"]["eligible"],
                "seed_report_passed": seed_report["passed"],
            }
        )
    if summary_rows:
        write_csv(campaign_root / "tables/three_seed_selected_checkpoints.csv", list(summary_rows[0]), summary_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("canary", help="Run the locked two-epoch V1.1 engineering canary")
    subparsers.add_parser("campaign", help="Run all three fresh 20-epoch V1.1 seeds after canary PASS")
    args = parser.parse_args()
    report = run_canary() if args.command == "canary" else run_campaign()
    print(
        json.dumps(
            {"passed": report["passed"], "decision": report["decision"], "gates": report["gates"]},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
