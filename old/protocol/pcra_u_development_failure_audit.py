#!/usr/bin/env python3
"""Dev-only P-CRA-U failure audit and architecture-decision gate.

The script has three deliberately separated stages:

``prepare``
    Locks the 80-family / 400-sample dev evaluation manifest before inference.
``baseline``
    Queries the frozen RoboRefer server with inference payloads only (B0/B1).
``evaluate``
    Opens dev evaluator data, evaluates B0/B1/B2/P1 and the two post-hoc
    P-CRA-U interventions, performs family-cluster bootstrap, and writes the
    architecture freeze/revision decision.

Calibration, Test-IID and Test-OOD are not representable as command inputs.
No optimizer or robot-control path is imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch

if __package__:
    from .pcra_u_development_common import (
        ANSWERABILITY_CLASSES,
        SOURCE_CLASSES,
        FrozenFeatureDataset,
        PCRAUDevelopmentV1,
        chunks,
        confusion_metrics,
        global_class_weights,
        model_inputs,
        read_json,
        safe_resolve,
        seed_runtime,
        sha256_file,
        write_json,
    )
    from .pcra_u_development_train import checkpoint_identity, verify_feature_index
    from .pcra_u_development_train_preflight import protected_now
    from .training_checkpoint_manager import audit_checkpoint_root, load_model_for_evaluation
    from .wp1_common import parse_point, query_model
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pcra_u_development_common import (  # type: ignore
        ANSWERABILITY_CLASSES,
        SOURCE_CLASSES,
        FrozenFeatureDataset,
        PCRAUDevelopmentV1,
        chunks,
        confusion_metrics,
        global_class_weights,
        model_inputs,
        read_json,
        safe_resolve,
        seed_runtime,
        sha256_file,
        write_json,
    )
    from pcra_u_development_train import checkpoint_identity, verify_feature_index  # type: ignore
    from pcra_u_development_train_preflight import protected_now  # type: ignore
    from training_checkpoint_manager import audit_checkpoint_root, load_model_for_evaluation  # type: ignore
    from wp1_common import parse_point, query_model  # type: ignore


WORKSPACE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "pcra_u_development_failure_audit_v1"
CONTRACT_PATH = WORKSPACE / "protocol/PCRA_U_DEVELOPMENT_FAILURE_AUDIT_CONTRACT.md"
EVAL_MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_development_eval_manifest.json"
DECISION_LOCK_PATH = WORKSPACE / "protocol/pcra_u_architecture_freeze_lock.json"
PROTOCOL_REPORT_PATH = WORKSPACE / "protocol/PCRA_U_DEVELOPMENT_FAILURE_AUDIT_REPORT.md"
TRAIN_CONFIG_PATH = WORKSPACE / "protocol/pcra_u_development_train_config.json"
TRAIN_MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_development_train_manifest.json"
TRAIN_REPORT_PATH = WORKSPACE / "results/pcra_u_runs/pcra_u_development_train_20260824/PCRA_U_DEVELOPMENT_TRAIN_REPORT.json"
TRAIN_RUN_ROOT = WORKSPACE / "results/pcra_u_runs/pcra_u_development_train_20260824"
CHECKPOINT_ROOT = TRAIN_RUN_ROOT / "checkpoints/development"
CHECKPOINT_PATH = CHECKPOINT_ROOT / "step_000002000"
FEATURE_CACHE_ROOT = WORKSPACE / "results/pcra_u_feature_cache/pcra_u_development_v1"
DEFAULT_RESULT_ROOT = WORKSPACE / "results/pcra_u_runs/pcra_u_development_failure_audit_20260824"
GEOMETRY_GATE_PATH = WORKSPACE / "ur3/ur3_perception/scripts/depth_component_gate_v2.py"
GEOMETRY_GATE_CONFIG_PATH = WORKSPACE / "ur3/ur3_perception/config/roborefer_pilot_v0_gate.yaml"
DATASET_MANIFEST_PATH = WORKSPACE / "protocol/dataset_v2_1_development_manifest.json"

BOOTSTRAP_SEED = 24082027
BOOTSTRAP_REPLICATES = 5000
BASELINE_SEED = 8132026
EXPECTED_MODEL_INVENTORY = "5508fb49888c1f62b875d928266f7fff3e1224c795f6fc8b971631143a45a7fa"
DATASET_PIXEL_SUFFIX = "Output the target pixel as (x, y) in image coordinates."
ROBOREFER_NORMALIZED_SUFFIX = (
    "Your answer must be exactly one list containing one normalized image point, "
    "formatted as [(x, y)], where x and y are between 0 and 1. Return no words."
)
VARIANT_ORDER = [
    "clean",
    "semantic_counterfactual",
    "relation_counterfactual",
    "depth_corruption",
    "occlusion_view_counterfactual",
]
METHOD_ORDER = ["B0", "B1", "B2", "U1", "P1", "A_NO_DEPTH"]
METHOD_LABELS = {
    "B0": "RoboRefer RGB-only forced point",
    "B1": "RoboRefer RGB-D forced point",
    "B2": "B1 + locked depth-component gate v2 (clean primary)",
    "U1": "P-CRA-U checkpoint with explicit relation slot set to direct",
    "P1": "P-CRA-U development v1 checkpoint step_000002000",
    "A_NO_DEPTH": "P-CRA-U checkpoint with D0 and depth thumbnail zeroed",
}
GATE_PARAMETERS = {
    "min_depth_m": 0.05,
    "max_depth_m": 2.0,
    "seed_radius_px": 5,
    "roi_radius_px": 110,
    "near_tolerance_m": 0.015,
    "far_tolerance_m": 0.055,
    "min_area_px": 120,
    "max_area_fraction": 0.12,
    "bbox_padding_px": 5,
    "reject_border_truncated": False,
}


class AuditError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_bytes(row))
    os.replace(temporary, path)


def append_jsonl_fsync(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AuditError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE.resolve()))


def verify_locked_manifest() -> dict[str, Any]:
    manifest = read_json(EVAL_MANIFEST_PATH)
    allowed_statuses = {"LOCKED_BEFORE_EVALUATION", "LOCKED_WITH_STATISTICAL_CORRECTION_01"}
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("status") not in allowed_statuses:
        raise AuditError("Evaluation manifest is not an accepted locked revision")
    if manifest.get("status") == "LOCKED_WITH_STATISTICAL_CORRECTION_01":
        correction = manifest.get("statistical_correction_01", {})
        if not (
            correction.get("inference_entries_unchanged") is True
            and correction.get("baseline_predictions_reused_without_new_inference") is True
            and correction.get("training_performed") is False
            and correction.get("calibration_or_test_opened") is False
        ):
            raise AuditError("Statistical correction provenance is incomplete")
    locks = manifest["artifact_locks"]
    for name, path in {
        "contract_sha256": CONTRACT_PATH,
        "audit_code_sha256": Path(__file__).resolve(),
        "train_config_sha256": TRAIN_CONFIG_PATH,
        "train_manifest_sha256": TRAIN_MANIFEST_PATH,
        "train_report_sha256": TRAIN_REPORT_PATH,
        "checkpoint_model_sha256": CHECKPOINT_PATH / "model.safetensors",
        "checkpoint_manifest_sha256": CHECKPOINT_PATH / "manifest.json",
        "checkpoint_metadata_sha256": CHECKPOINT_PATH / "metadata.json",
        "best_pointer_sha256": CHECKPOINT_ROOT / "best_dev_total_loss.json",
        "geometry_gate_code_sha256": GEOMETRY_GATE_PATH,
        "geometry_gate_config_sha256": GEOMETRY_GATE_CONFIG_PATH,
    }.items():
        if not path.is_file() or sha256_file(path) != locks[name]:
            raise AuditError(f"Locked artifact differs: {name} -> {path}")
    return manifest


def prepare_manifest() -> dict[str, Any]:
    if EVAL_MANIFEST_PATH.exists():
        raise AuditError(f"Refusing to overwrite locked eval manifest: {EVAL_MANIFEST_PATH}")
    if not CONTRACT_PATH.is_file():
        raise AuditError(f"Missing contract: {CONTRACT_PATH}")
    train_manifest = read_json(TRAIN_MANIFEST_PATH)
    config = read_json(TRAIN_CONFIG_PATH)
    train_report = read_json(TRAIN_REPORT_PATH)
    dataset_manifest = read_json(DATASET_MANIFEST_PATH)
    if train_report.get("selected_checkpoint") != relative(CHECKPOINT_PATH):
        raise AuditError("Training report does not select step_000002000")
    dev_entries = [entry for entry in train_manifest["entries"] if entry["split"] == "dev"]
    dev_entries.sort(key=lambda row: (row["family_id"], VARIANT_ORDER.index(row["variant"])))
    if len(dev_entries) != 400 or len({row["family_id"] for row in dev_entries}) != 80:
        raise AuditError("Expected exactly 80 dev families / 400 dev samples")
    family_lookup = {row["family_id"]: row for row in dataset_manifest["families"]}
    inference_entries = []
    for entry in dev_entries:
        family = family_lookup[entry["family_id"]]
        variant_spec = next(row for row in family["variant_specs"] if row["variant"] == entry["variant"])
        source_prompt = str(entry["feature_input"]["prompt"]).strip()
        if not source_prompt.endswith(DATASET_PIXEL_SUFFIX):
            raise AuditError(f"Unexpected dataset coordinate suffix: {entry['sample_id']}")
        instruction = source_prompt[: -len(DATASET_PIXEL_SUFFIX)].strip()
        baseline_prompt = f"{instruction} {ROBOREFER_NORMALIZED_SUFFIX}"
        inference_entries.append(
            {
                "sample_id": entry["sample_id"],
                "family_id": entry["family_id"],
                "split": "dev",
                "variant": entry["variant"],
                "source_prompt": source_prompt,
                "source_prompt_sha256": entry["feature_input"]["prompt_sha256"],
                "baseline_prompt": baseline_prompt,
                "baseline_prompt_sha256": hashlib.sha256(baseline_prompt.encode("utf-8")).hexdigest(),
                "baseline_prompt_adapter": "replace_exact_pixel_suffix_with_roborefer_normalized_point_suffix_v1",
                "rgb_path": entry["feature_input"]["rgb_path"],
                "rgb_sha256": entry["feature_input"]["rgb_sha256"],
                "depth_path": entry["feature_input"]["depth_path"],
                "depth_sha256": entry["feature_input"]["depth_sha256"],
                "capture_id": variant_spec["capture_id"],
            }
        )
    answer_counts = Counter(row["supervision"]["answerability_state"] for row in dev_entries)
    relation_counts = Counter(row["audit_only"]["relation"] for row in dev_entries)
    protected = protected_now(train_manifest, config)
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_EVALUATION",
        "scientific_status": "EXPLORATORY_DEV_ONLY_CALIBRATION_TESTS_SEALED",
        "created_at_utc": utc_now(),
        "scope": {
            "split": "dev",
            "family_count": 80,
            "sample_count": 400,
            "variants_per_family": 5,
            "family_is_independent_unit": True,
            "gradient_or_optimizer_steps": 0,
        },
        "checkpoint": {
            "path": relative(CHECKPOINT_PATH),
            "epoch": 10,
            "global_step": 2000,
            "selection_split": "dev",
            "selection_metric": "dev_total_loss",
            "model_sha256": sha256_file(CHECKPOINT_PATH / "model.safetensors"),
        },
        "methods": {
            "B0": {"input": "RGB", "output": "forced_point", "answerability_policy": "FOUND_if_parse_else_INSUFFICIENT_EVIDENCE"},
            "B1": {"input": "RGB-D", "output": "forced_point", "answerability_policy": "FOUND_if_parse_else_INSUFFICIENT_EVIDENCE"},
            "B2": {"input": "B1_point_plus_metric_depth", "gate": "depth_component_gate_v2", "primary_scope": "clean_only"},
            "U1": {"checkpoint": "same_as_P1", "intervention": "explicit_relation_id_set_to_direct", "lexical_relation_tokens_retained": True},
            "P1": {"checkpoint": "step_000002000", "intervention": "none"},
            "A_NO_DEPTH": {"checkpoint": "same_as_P1", "intervention": "zero_D0_grid_and_D0_thumbnail"},
        },
        "bootstrap": {
            "unit": "family",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "seed_application": "reinitialize exact seed for every method and paired contrast; no offsets",
            "confidence_level": 0.95,
            "resample_rule": "sample 80 family IDs with replacement and retain all five variants",
        },
        "decision_thresholds": {
            "max_p1_false_found_point_estimate": 0.10,
            "min_p1_grounding_family_bootstrap_lower_95": 0.70,
            "max_allowed_ablation_delta_ci_lower": -0.02,
            "ablation_non_degradation_rule": "for each intervention, P1-minus-intervention lower CI > -0.02 for family_found_point_in_target OR answerability_macro_f1",
        },
        "observed_dev_commitment_counts": {
            "answerability_samples": dict(sorted(answer_counts.items())),
            "relation_samples": dict(sorted(relation_counts.items())),
        },
        "inference_entries": inference_entries,
        "inference_boundary": {
            "allowed_fields": ["baseline_prompt", "rgb_path", "rgb_sha256", "depth_path", "depth_sha256"],
            "oracle_or_evaluator_fields_present_per_entry": False,
        },
        "geometry_gate_parameters": GATE_PARAMETERS,
        "sealing": {
            "calibration_fit": False,
            "test_opened": False,
            "forbidden_splits": ["calibration", "test_iid", "test_ood"],
            "tables_02_to_06": "NOT_RUN",
        },
        "protected_inputs": protected,
        "artifact_locks": {
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "audit_code_sha256": sha256_file(Path(__file__).resolve()),
            "train_config_sha256": sha256_file(TRAIN_CONFIG_PATH),
            "train_manifest_sha256": sha256_file(TRAIN_MANIFEST_PATH),
            "train_report_sha256": sha256_file(TRAIN_REPORT_PATH),
            "checkpoint_model_sha256": sha256_file(CHECKPOINT_PATH / "model.safetensors"),
            "checkpoint_manifest_sha256": sha256_file(CHECKPOINT_PATH / "manifest.json"),
            "checkpoint_metadata_sha256": sha256_file(CHECKPOINT_PATH / "metadata.json"),
            "best_pointer_sha256": sha256_file(CHECKPOINT_ROOT / "best_dev_total_loss.json"),
            "geometry_gate_code_sha256": sha256_file(GEOMETRY_GATE_PATH),
            "geometry_gate_config_sha256": sha256_file(GEOMETRY_GATE_CONFIG_PATH),
            "dataset_index_sha256": protected["dataset_index_sha256"],
            "dataset_tree_sha256": protected["dataset_tree_sha256"],
            "model_inventory_sha256": protected["model_inventory_sha256"],
        },
    }
    write_json(EVAL_MANIFEST_PATH, manifest)
    return manifest


def validate_inference_entry(entry: Mapping[str, Any]) -> tuple[bytes, bytes, str]:
    if set(entry) != {
        "sample_id", "family_id", "split", "variant", "source_prompt", "source_prompt_sha256",
        "baseline_prompt", "baseline_prompt_sha256", "baseline_prompt_adapter",
        "rgb_path", "rgb_sha256", "depth_path", "depth_sha256", "capture_id",
    }:
        raise AuditError(f"Unexpected inference-entry keys: {entry.get('sample_id')}")
    if entry["split"] != "dev":
        raise AuditError("Baseline runner received a non-dev entry")
    source_prompt = str(entry["source_prompt"])
    prompt = str(entry["baseline_prompt"])
    if hashlib.sha256(source_prompt.encode("utf-8")).hexdigest() != entry["source_prompt_sha256"]:
        raise AuditError(f"Source prompt hash mismatch: {entry['sample_id']}")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != entry["baseline_prompt_sha256"]:
        raise AuditError(f"Adapted baseline prompt hash mismatch: {entry['sample_id']}")
    if not source_prompt.endswith(DATASET_PIXEL_SUFFIX) or not prompt.endswith(ROBOREFER_NORMALIZED_SUFFIX):
        raise AuditError(f"Baseline prompt adapter contract mismatch: {entry['sample_id']}")
    rgb_path = safe_resolve(WORKSPACE, str(entry["rgb_path"]))
    depth_path = safe_resolve(WORKSPACE, str(entry["depth_path"]))
    if sha256_file(rgb_path) != entry["rgb_sha256"] or sha256_file(depth_path) != entry["depth_sha256"]:
        raise AuditError(f"Input hash mismatch: {entry['sample_id']}")
    return rgb_path.read_bytes(), depth_path.read_bytes(), prompt


def run_baselines(result_root: Path, server_url: str, timeout_seconds: float) -> dict[str, Any]:
    manifest = verify_locked_manifest()
    result_root.mkdir(parents=True, exist_ok=True)
    predictions_path = result_root / "predictions/roborefer_baselines_dev.jsonl"
    existing = read_jsonl(predictions_path)
    keys: dict[tuple[str, str], dict[str, Any]] = {}
    for row in existing:
        key = (str(row.get("sample_id")), str(row.get("method")))
        if key in keys or key[1] not in {"B0", "B1"}:
            raise AuditError(f"Duplicate/invalid resume row: {key}")
        keys[key] = row
    expected_keys = {(entry["sample_id"], method) for entry in manifest["inference_entries"] for method in ("B0", "B1")}
    if not set(keys).issubset(expected_keys):
        raise AuditError("Existing baseline prediction file contains out-of-manifest rows")
    started = time.monotonic()
    completed_now = 0
    for entry in manifest["inference_entries"]:
        rgb, depth, prompt = validate_inference_entry(entry)
        for method in ("B0", "B1"):
            key = (entry["sample_id"], method)
            if key in keys:
                continue
            depth_payload = None if method == "B0" else depth
            answer, latency_ms, response = query_model(
                server_url,
                prompt,
                rgb,
                depth_payload,
                timeout_seconds,
                EXPECTED_MODEL_INVENTORY,
            )
            parsed = parse_point(answer, 640, 480)
            row = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "scientific_status": "EXPLORATORY_DEV_ONLY",
                "sample_id": entry["sample_id"],
                "family_id": entry["family_id"],
                "variant": entry["variant"],
                "method": method,
                "source_prompt_sha256": entry["source_prompt_sha256"],
                "baseline_prompt_sha256": entry["baseline_prompt_sha256"],
                "rgb_sha256": entry["rgb_sha256"],
                "depth_sha256": entry["depth_sha256"] if method == "B1" else None,
                "enable_depth": method == "B1",
                "raw_answer": answer,
                **parsed,
                "latency_ms": float(latency_ms),
                "generation_mode": response["generation_mode"],
                "generation_config": response["generation_config"],
                "random_seed": response["random_seed"],
                "model_inventory_sha256": response["model_fingerprint"]["inventory_sha256"],
                "response_sha256": response["response_sha256"],
                "oracle_or_annotation_read_by_runner": False,
                "robot_manipulation_performed": False,
            }
            append_jsonl_fsync(predictions_path, row)
            keys[key] = row
            completed_now += 1
            if completed_now % 20 == 0:
                print(f"BASELINE {len(keys):03d}/800 complete", flush=True)
    if set(keys) != expected_keys:
        raise AuditError(f"Incomplete baseline inference: {len(keys)}/800")
    ordered = [keys[(entry["sample_id"], method)] for entry in manifest["inference_entries"] for method in ("B0", "B1")]
    write_jsonl_atomic(predictions_path, ordered)
    parse_counts = Counter((row["method"], row["parse_status"]) for row in ordered)
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "scientific_status": "EXPLORATORY_DEV_ONLY",
        "created_at_utc": utc_now(),
        "record_count": len(ordered),
        "sample_count": 400,
        "family_count": 80,
        "method_counts": dict(Counter(row["method"] for row in ordered)),
        "parse_counts": {f"{method}:{status}": count for (method, status), count in sorted(parse_counts.items())},
        "mean_latency_ms": {
            method: statistics.fmean(row["latency_ms"] for row in ordered if row["method"] == method)
            for method in ("B0", "B1")
        },
        "predictions_path": relative(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "completed_in_this_invocation": completed_now,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "greedy_seed": BASELINE_SEED,
        "model_inventory_sha256": EXPECTED_MODEL_INVENTORY,
        "oracle_or_annotation_read_by_runner": False,
        "calibration_fit": False,
        "test_opened": False,
        "robot_manipulation_performed": False,
    }
    write_json(result_root / "checks/baseline_inference_manifest.json", report)
    return report


def import_geometry_gate():
    spec = importlib.util.spec_from_file_location("pcra_audit_depth_component_gate_v2", GEOMETRY_GATE_PATH)
    if spec is None or spec.loader is None:
        raise AuditError("Cannot import locked geometry gate")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def answer_metrics(truth: Sequence[int], prediction: Sequence[int]) -> dict[str, Any]:
    return confusion_metrics(truth, prediction, len(ANSWERABILITY_CLASSES))


def score_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    eligible = [row for row in rows if row.get("eligible", True)]
    truth = [ANSWERABILITY_CLASSES.index(str(row["answerability_truth"])) for row in eligible]
    prediction = [ANSWERABILITY_CLASSES.index(str(row["answerability_prediction"])) for row in eligible]
    answer = answer_metrics(truth, prediction)
    risky = [row for row in eligible if row["answerability_truth"] in {"AMBIGUOUS", "ABSENT"}]
    found = [row for row in eligible if row["answerability_truth"] == "FOUND"]
    accepted = [row for row in eligible if row["answerability_prediction"] == "FOUND"]
    accepted_found = [row for row in found if row["answerability_prediction"] == "FOUND"]
    parsed_found = [row for row in found if row.get("point_in_target") is not None]
    family_answer: defaultdict[str, list[float]] = defaultdict(list)
    family_ground: defaultdict[str, list[float]] = defaultdict(list)
    for row in eligible:
        family_answer[str(row["family_id"])].append(float(row["answerability_truth"] == row["answerability_prediction"]))
        if row["answerability_truth"] == "FOUND":
            family_ground[str(row["family_id"])].append(float(bool(row.get("point_in_target", False))))
    source_tp = source_fp = source_fn = source_rows_seen = 0
    for row in eligible:
        if "source_truth" not in row:
            continue
        source_rows_seen += 1
        source_truth = set(row["source_truth"])
        source_pred = set(row["source_prediction"])
        source_tp += len(source_truth & source_pred)
        source_fp += len(source_pred - source_truth)
        source_fn += len(source_truth - source_pred)
    source_den = 2 * source_tp + source_fp + source_fn
    return {
        "eligible_families": len({row["family_id"] for row in eligible}),
        "eligible_samples": len(eligible),
        "answerability_accuracy": float(answer["accuracy"]),
        "answerability_macro_f1": float(answer["macro_f1"]),
        "false_found_rate": sum(row["answerability_prediction"] == "FOUND" for row in risky) / len(risky) if risky else 0.0,
        "false_found_count": sum(row["answerability_prediction"] == "FOUND" for row in risky),
        "false_found_denominator": len(risky),
        "found_point_in_target": sum(bool(row.get("point_in_target", False)) for row in found) / len(found) if found else 0.0,
        "found_point_in_interior": sum(bool(row.get("point_in_interior", False)) for row in found) / len(found) if found else 0.0,
        "found_parse_rate": len(parsed_found) / len(found) if found else 0.0,
        "found_count": len(found),
        "coverage_predicted_found": len(accepted) / len(eligible) if eligible else 0.0,
        "accepted_found_point_in_target": sum(bool(row.get("point_in_target", False)) for row in accepted_found) / len(accepted_found) if accepted_found else 0.0,
        "accepted_found_count": len(accepted_found),
        "family_answerability_accuracy": statistics.fmean(statistics.fmean(values) for values in family_answer.values()) if family_answer else 0.0,
        "family_found_point_in_target": statistics.fmean(statistics.fmean(values) for values in family_ground.values()) if family_ground else 0.0,
        "families_with_found": len(family_ground),
        "source_micro_f1": (2 * source_tp / source_den if source_den else 1.0) if source_rows_seen else None,
    }


def percentile_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


BOOTSTRAP_METRICS = [
    "answerability_accuracy",
    "answerability_macro_f1",
    "false_found_rate",
    "found_point_in_target",
    "found_point_in_interior",
    "coverage_predicted_found",
    "family_answerability_accuracy",
    "family_found_point_in_target",
]


def family_bootstrap(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, dict[str, float]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("eligible", True):
            grouped[str(row["family_id"])].append(row)
    families = sorted(grouped)
    if not families:
        raise AuditError("Cannot bootstrap zero families")
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(families, size=len(families), replace=True)
        expanded: list[Mapping[str, Any]] = []
        for draw_index, family in enumerate(sampled):
            for row in grouped[str(family)]:
                copy = dict(row)
                copy["family_id"] = f"bootstrap_{draw_index:04d}_{family}"
                expanded.append(copy)
        metrics = score_rows(expanded)
        for metric in BOOTSTRAP_METRICS:
            draws[metric].append(float(metrics[metric]))
    point = score_rows(rows)
    return {
        metric: {
            "estimate": float(point[metric]),
            "ci95_low": percentile_ci(values)[0],
            "ci95_high": percentile_ci(values)[1],
        }
        for metric, values in draws.items()
    }


def paired_bootstrap(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], seed: int
) -> dict[str, dict[str, float]]:
    left_group: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    right_group: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in left:
        if row.get("eligible", True):
            left_group[str(row["family_id"])].append(row)
    for row in right:
        if row.get("eligible", True):
            right_group[str(row["family_id"])].append(row)
    families = sorted(set(left_group) & set(right_group))
    if not families:
        raise AuditError("No paired families")
    # The grounding contrast is deliberately family-equal: first average the
    # eligible FOUND variants inside each sampled family, then average those
    # family values.  Using the sample-level metric here would violate the
    # locked independent-unit contract when families have unequal eligible
    # FOUND counts.
    metrics_of_interest = ["answerability_macro_f1", "false_found_rate", "family_found_point_in_target"]
    point_left, point_right = score_rows(left), score_rows(right)
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in metrics_of_interest}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(families, size=len(families), replace=True)
        left_rows: list[Mapping[str, Any]] = []
        right_rows: list[Mapping[str, Any]] = []
        for draw_index, family in enumerate(sampled):
            alias = f"bootstrap_{draw_index:04d}_{family}"
            for source, destination in ((left_group[str(family)], left_rows), (right_group[str(family)], right_rows)):
                for row in source:
                    copy = dict(row)
                    copy["family_id"] = alias
                    destination.append(copy)
        left_metrics, right_metrics = score_rows(left_rows), score_rows(right_rows)
        for metric in metrics_of_interest:
            draws[metric].append(float(left_metrics[metric]) - float(right_metrics[metric]))
    return {
        metric: {
            "estimate": float(point_left[metric]) - float(point_right[metric]),
            "ci95_low": percentile_ci(values)[0],
            "ci95_high": percentile_ci(values)[1],
        }
        for metric, values in draws.items()
    }


def point_metrics(entry: Mapping[str, Any], point: Sequence[int] | None) -> dict[str, Any]:
    if point is None:
        return {"point_in_target": None, "point_in_interior": None, "mass_in_target": None, "map_x": None, "map_y": None}
    x, y = int(point[0]), int(point[1])
    target = cv2.imread(str(safe_resolve(WORKSPACE, entry["supervision"]["target_mask_path"])), cv2.IMREAD_GRAYSCALE)
    interior = cv2.imread(str(safe_resolve(WORKSPACE, entry["supervision"]["target_interior_mask_path"])), cv2.IMREAD_GRAYSCALE)
    if target is None or interior is None:
        raise AuditError(f"Cannot load evaluator mask: {entry['sample_id']}")
    return {
        "point_in_target": bool(target[y, x] > 0) if entry["supervision"]["answerability_state"] == "FOUND" else None,
        "point_in_interior": bool(interior[y, x] > 0) if entry["supervision"]["answerability_state"] == "FOUND" else None,
        "mass_in_target": None,
        "map_x": x,
        "map_y": y,
    }


@torch.no_grad()
def evaluate_sidecar_methods(
    model: PCRAUDevelopmentV1,
    dataset: FrozenFeatureDataset,
    entries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    model.eval()
    output_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ("P1", "U1", "A_NO_DEPTH")}
    p1_heatmaps: dict[str, np.ndarray] = {}
    batch_size = int(config["optimization"]["batch_size"])
    for indices in chunks(list(range(len(entries))), batch_size):
        batch = dataset.make_batch(indices, device)
        for method in ("P1", "U1", "A_NO_DEPTH"):
            inputs = model_inputs(batch)
            if method == "U1":
                inputs = dict(inputs)
                inputs["relation_ids"] = torch.zeros_like(inputs["relation_ids"])
            elif method == "A_NO_DEPTH":
                inputs = dict(inputs)
                inputs["d0"] = torch.zeros_like(inputs["d0"])
                inputs["d_thumb"] = torch.zeros_like(inputs["d_thumb"])
            started = time.perf_counter()
            output = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = 1000.0 * (time.perf_counter() - started)
            answer_prob = torch.softmax(output["answerability_logits"], dim=1)
            source_prob = torch.sigmoid(output["source_logits"])
            heatmap_prob = torch.sigmoid(output["heatmap_logits"])
            for local_index, entry in enumerate(batch["entries"]):
                answer_index = int(answer_prob[local_index].argmax().item())
                flat = int(heatmap_prob[local_index].argmax().item())
                grid_y, grid_x = divmod(flat, 32)
                x = min(639, int((grid_x + 0.5) / 32 * 640))
                y = min(479, int((grid_y + 0.5) / 24 * 480))
                truth = entry["supervision"]["answerability_state"]
                target_grid = batch["heatmap_targets"][local_index]
                mass = None
                in_target = in_interior = None
                if truth == "FOUND":
                    in_target = bool(batch["full_masks"][local_index][y, x])
                    in_interior = bool(batch["full_interiors"][local_index][y, x])
                    mass = float(
                        (heatmap_prob[local_index] * target_grid).sum().item()
                        / heatmap_prob[local_index].sum().clamp_min(1e-12).item()
                    )
                row = {
                    "sample_id": entry["sample_id"],
                    "family_id": entry["family_id"],
                    "variant": entry["variant"],
                    "relation": entry["audit_only"]["relation"],
                    "method": method,
                    "eligible": True,
                    "answerability_truth": truth,
                    "answerability_prediction": ANSWERABILITY_CLASSES[answer_index],
                    "answerability_probabilities": {
                        name: float(answer_prob[local_index, index].item()) for index, name in enumerate(ANSWERABILITY_CLASSES)
                    },
                    "map_x": x,
                    "map_y": y,
                    "point_in_target": in_target,
                    "point_in_interior": in_interior,
                    "mass_in_target": mass,
                    "source_truth": entry["supervision"]["source_labels"],
                    "source_prediction": [
                        name for index, name in enumerate(SOURCE_CLASSES) if float(source_prob[local_index, index].item()) >= 0.5
                    ],
                    "source_probabilities": {
                        name: float(source_prob[local_index, index].item()) for index, name in enumerate(SOURCE_CLASSES)
                    },
                    "latency_ms_sidecar_only": elapsed_ms / len(indices),
                }
                output_rows[method].append(row)
                if method == "P1":
                    p1_heatmaps[entry["sample_id"]] = heatmap_prob[local_index].detach().cpu().numpy()
    return output_rows, p1_heatmaps


def baseline_method_rows(
    predictions: Sequence[Mapping[str, Any]], entries: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    by_sample = {entry["sample_id"]: entry for entry in entries}
    rows: dict[str, list[dict[str, Any]]] = {"B0": [], "B1": []}
    for prediction in predictions:
        method = str(prediction["method"])
        entry = by_sample[str(prediction["sample_id"])]
        points = prediction.get("pixel_points_xy", [])
        point = points[0] if len(points) == 1 else None
        parse_success = point is not None
        rows[method].append(
            {
                "sample_id": entry["sample_id"],
                "family_id": entry["family_id"],
                "variant": entry["variant"],
                "relation": entry["audit_only"]["relation"],
                "method": method,
                "eligible": True,
                "answerability_truth": entry["supervision"]["answerability_state"],
                "answerability_prediction": "FOUND" if parse_success else "INSUFFICIENT_EVIDENCE",
                "parse_status": prediction["parse_status"],
                **point_metrics(entry, point),
                "latency_ms": prediction["latency_ms"],
            }
        )
    for method in rows:
        rows[method].sort(key=lambda row: (row["family_id"], VARIANT_ORDER.index(row["variant"])))
    return rows


def b2_rows(
    b1_rows: Sequence[Mapping[str, Any]], entries: Sequence[Mapping[str, Any]], gate_module: Any
) -> tuple[list[dict[str, Any]], Counter[str]]:
    by_sample = {entry["sample_id"]: entry for entry in entries}
    result = []
    reasons: Counter[str] = Counter()
    for b1 in b1_rows:
        if b1["variant"] != "clean":
            continue
        entry = by_sample[str(b1["sample_id"])]
        record = read_json(safe_resolve(WORKSPACE, entry["record_path"]))
        depth_path = safe_resolve(
            safe_resolve(WORKSPACE, read_json(TRAIN_MANIFEST_PATH)["dataset_root"]),
            record["sensor_evidence"]["depth_metric"]["path"],
        )
        depth = np.load(depth_path, allow_pickle=False)
        point = None if b1["map_x"] is None else [int(b1["map_x"]), int(b1["map_y"])]
        accepted = False
        reason = "NO_PARSED_POINT"
        if point is not None:
            try:
                gate = gate_module.segment_seeded_depth_component_v2(depth, [tuple(point)], **GATE_PARAMETERS)
                accepted = bool(gate["accepted"])
                reason = str(gate["reason_code"])
            except gate_module.DepthComponentError as exc:
                reason = str(exc.code)
        reasons[reason] += 1
        result.append(
            {
                **dict(b1),
                "method": "B2",
                "eligible": True,
                "answerability_prediction": "FOUND" if accepted else "INSUFFICIENT_EVIDENCE",
                "geometry_gate_accepted": accepted,
                "geometry_gate_reason": reason,
                "primary_scope": "clean_only",
            }
        )
    return result, reasons


def mask_evidence(mask: np.ndarray, depth: np.ndarray | None = None) -> dict[str, Any]:
    active = mask > 0
    ys, xs = np.where(active)
    result: dict[str, Any] = {
        "mask_pixels": int(xs.size),
        "mask_fraction": float(xs.size / mask.size),
        "centroid_x_px": float(np.mean(xs)) if xs.size else None,
        "centroid_y_px": float(np.mean(ys)) if ys.size else None,
    }
    if depth is not None and xs.size:
        valid = active & np.isfinite(depth) & (depth >= 0.10) & (depth <= 2.0)
        values = depth[valid].astype(np.float64)
        result.update(
            {
                "valid_depth_pixels": int(values.size),
                "valid_depth_fraction_in_mask": float(values.size / xs.size),
                "median_depth_m": float(np.median(values)) if values.size else None,
                "depth_mad_m": float(np.median(np.abs(values - np.median(values)))) if values.size else None,
            }
        )
    return result


def asset_class(object_id: str) -> str:
    value = re.sub(r"_\d+$", "", str(object_id))
    return value.removeprefix("ycb_").removeprefix("cube_")


def mask_size_bin(fraction: float) -> str:
    if fraction <= 0:
        return "empty"
    if fraction < 0.0025:
        return "tiny_<0.25pct"
    if fraction < 0.0075:
        return "small_0.25-0.75pct"
    if fraction < 0.015:
        return "medium_0.75-1.5pct"
    return "large_>=1.5pct"


def depth_bin(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value < 0.55:
        return "near_<0.55m"
    if value < 0.70:
        return "mid_0.55-0.70m"
    return "far_>=0.70m"


def enrich_audit_row(
    prediction: Mapping[str, Any], entry: Mapping[str, Any], dataset_root: Path
) -> dict[str, Any]:
    record = read_json(safe_resolve(WORKSPACE, entry["record_path"]))
    target_path = safe_resolve(WORKSPACE, entry["supervision"]["target_mask_path"])
    target_mask = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    interior_mask = cv2.imread(
        str(safe_resolve(WORKSPACE, entry["supervision"]["target_interior_mask_path"])),
        cv2.IMREAD_GRAYSCALE,
    )
    if target_mask is None or interior_mask is None:
        raise AuditError(f"Missing target mask: {entry['sample_id']}")
    depth_path = safe_resolve(dataset_root, record["sensor_evidence"]["depth_metric"]["path"])
    depth = np.load(depth_path, allow_pickle=False)
    target = mask_evidence(target_mask, depth)
    interior = mask_evidence(interior_mask, depth)
    spatial = record["evaluator_only"]["spatial_label"]
    target_ids = spatial.get("valid_target_ids") or spatial.get("candidate_target_ids") or []
    anchor_ids = spatial.get("anchor_ids") or []
    anchor_values = []
    for anchor in record["evaluator_only"]["masks"]["anchor"]:
        mask = cv2.imread(str(safe_resolve(dataset_root, anchor["path"])), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            anchor_values.append(mask_evidence(mask, depth))
    signed_margin = None
    if target["centroid_x_px"] is not None and target.get("median_depth_m") is not None and anchor_values:
        anchor_x = [value["centroid_x_px"] for value in anchor_values if value["centroid_x_px"] is not None]
        anchor_depth = [value.get("median_depth_m") for value in anchor_values if value.get("median_depth_m") is not None]
        relation = str(prediction["relation"])
        try:
            geometry = import_geometry_relation()
            _, margin = geometry.relation_predicate(
                relation,
                target["centroid_x_px"],
                target["median_depth_m"],
                anchor_x,
                anchor_depth,
            )
            signed_margin = None if math.isinf(margin) else float(margin)
        except Exception:
            signed_margin = None
    point_error = None
    if prediction.get("map_x") is not None and target["centroid_x_px"] is not None:
        point_error = math.hypot(
            float(prediction["map_x"]) - float(target["centroid_x_px"]),
            float(prediction["map_y"]) - float(target["centroid_y_px"]),
        ) / math.hypot(640, 480)
    tf_path = safe_resolve(dataset_root, record["sensor_evidence"]["tf_snapshot"]["path"])
    camera = read_json(tf_path)["camera_color_optical_frame"]
    return {
        **dict(prediction),
        "family_category": record["family_category"],
        "state_submode": record["evaluator_only"]["uncertainty_label"]["state_submode"],
        "target_asset_ids": "|".join(target_ids),
        "target_asset_classes": "|".join(asset_class(value) for value in target_ids),
        "anchor_asset_ids": "|".join(anchor_ids),
        "anchor_asset_classes": "|".join(asset_class(value) for value in anchor_ids),
        "target_mask_pixels": target["mask_pixels"],
        "target_mask_fraction": target["mask_fraction"],
        "target_interior_pixels": interior["mask_pixels"],
        "target_interior_fraction": interior["mask_fraction"],
        "mask_size_bin": mask_size_bin(target["mask_fraction"]),
        "target_median_depth_m": target.get("median_depth_m"),
        "target_depth_mad_m": target.get("depth_mad_m"),
        "target_valid_depth_fraction": target.get("valid_depth_fraction_in_mask"),
        "anchor_median_depth_m": "|".join(
            "" if value.get("median_depth_m") is None else f"{value['median_depth_m']:.9f}"
            for value in anchor_values
        ),
        "anchor_depth_mad_m": "|".join(
            "" if value.get("depth_mad_m") is None else f"{value['depth_mad_m']:.9f}"
            for value in anchor_values
        ),
        "depth_bin": depth_bin(target.get("median_depth_m")),
        "observed_relation_signed_margin": signed_margin,
        "relation_margin_unit": "px" if prediction["relation"] in {"left_of", "right_of"} else "m",
        "occlusion_variant": prediction["variant"] == "occlusion_view_counterfactual",
        "depth_corruption_variant": prediction["variant"] == "depth_corruption",
        "normalized_centroid_error": point_error,
        "camera_x_m": camera["position"][0],
        "camera_y_m": camera["position"][1],
        "camera_z_m": camera["position"][2],
    }


_GEOMETRY_RELATION_MODULE = None


def import_geometry_relation():
    global _GEOMETRY_RELATION_MODULE
    if _GEOMETRY_RELATION_MODULE is None:
        path = WORKSPACE / "protocol/dataset_v2_relation_geometry_v2_1.py"
        spec = importlib.util.spec_from_file_location("pcra_audit_relation_geometry", path)
        if spec is None or spec.loader is None:
            raise AuditError("Cannot import relation geometry")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _GEOMETRY_RELATION_MODULE = module
    return _GEOMETRY_RELATION_MODULE


def failure_strata(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dimensions = {
        "relation": lambda row: row["relation"],
        "variant": lambda row: row["variant"],
        "target_asset": lambda row: row.get("target_asset_classes") or "unavailable",
        "mask_size": lambda row: row["mask_size_bin"],
        "depth": lambda row: row["depth_bin"],
        "occlusion": lambda row: "occlusion" if row["occlusion_variant"] else "non_occlusion",
    }
    output = []
    for dimension, key_fn in dimensions.items():
        groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(key_fn(row))].append(row)
        for stratum, values in sorted(groups.items()):
            risky = [row for row in values if row["answerability_truth"] in {"AMBIGUOUS", "ABSENT"}]
            found = [row for row in values if row["answerability_truth"] == "FOUND"]
            output.append(
                {
                    "scientific_status": "EXPLORATORY_DEV_ONLY",
                    "dimension": dimension,
                    "stratum": stratum,
                    "families": len({row["family_id"] for row in values}),
                    "samples": len(values),
                    "false_found_count": sum(row["answerability_prediction"] == "FOUND" for row in risky),
                    "false_found_denominator": len(risky),
                    "false_found_rate": sum(row["answerability_prediction"] == "FOUND" for row in risky) / len(risky) if risky else None,
                    "outside_target_count": sum(not bool(row.get("point_in_target", False)) for row in found),
                    "found_denominator": len(found),
                    "outside_target_rate": sum(not bool(row.get("point_in_target", False)) for row in found) / len(found) if found else None,
                    "evidence_flag": "INSUFFICIENT_FAMILIES" if len({row["family_id"] for row in values}) < 10 else "DESCRIPTIVE_DEV",
                }
            )
    return output


def source_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in SOURCE_CLASSES:
        tp = sum(source in row["source_truth"] and source in row["source_prediction"] for row in rows)
        fp = sum(source not in row["source_truth"] and source in row["source_prediction"] for row in rows)
        fn = sum(source in row["source_truth"] and source not in row["source_prediction"] for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output.append(
            {
                "scientific_status": "EXPLORATORY_DEV_ONLY",
                "head_type": "four_label_multilabel_not_four_way_classification",
                "source": source,
                "positive_samples": sum(source in row["source_truth"] for row in rows),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return output


def confusion_table(rows: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row.get("eligible", True)]
    truth = [ANSWERABILITY_CLASSES.index(row["answerability_truth"]) for row in eligible]
    prediction = [ANSWERABILITY_CLASSES.index(row["answerability_prediction"]) for row in eligible]
    matrix = answer_metrics(truth, prediction)["confusion_matrix"]
    output = []
    for truth_index, truth_name in enumerate(ANSWERABILITY_CLASSES):
        for prediction_index, prediction_name in enumerate(ANSWERABILITY_CLASSES):
            output.append(
                {
                    "scientific_status": "EXPLORATORY_DEV_ONLY",
                    "method": method,
                    "truth": truth_name,
                    "prediction": prediction_name,
                    "count": matrix[truth_index][prediction_index],
                }
            )
    return output


def render_failure_images(
    result_root: Path,
    failures: Sequence[Mapping[str, Any]],
    entries: Mapping[str, Mapping[str, Any]],
    heatmaps: Mapping[str, np.ndarray],
) -> int:
    ranked = sorted(failures, key=lambda row: hashlib.sha256(f"{BOOTSTRAP_SEED}:{row['sample_id']}:{row['failure_type']}".encode()).hexdigest())
    selected = ranked[:12]
    output_root = result_root / "images/failure_cases_real"
    output_root.mkdir(parents=True, exist_ok=True)
    for row in selected:
        entry = entries[row["sample_id"]]
        rgb = cv2.imread(str(safe_resolve(WORKSPACE, entry["feature_input"]["rgb_path"])), cv2.IMREAD_COLOR)
        target = cv2.imread(str(safe_resolve(WORKSPACE, entry["supervision"]["target_mask_path"])), cv2.IMREAD_GRAYSCALE)
        heat = cv2.resize(heatmaps[row["sample_id"]], (640, 480), interpolation=cv2.INTER_CUBIC)
        heat = (255 * (heat - heat.min()) / max(1e-12, heat.max() - heat.min())).astype(np.uint8)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(rgb, 0.65, colored, 0.35, 0)
        contours, _ = cv2.findContours((target > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        if row.get("map_x") is not None:
            cv2.drawMarker(overlay, (int(row["map_x"]), int(row["map_y"])), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        caption = f"{row['failure_type']} | {row['sample_id']} | {row['relation']}"
        cv2.rectangle(overlay, (0, 0), (640, 32), (10, 10, 10), -1)
        cv2.putText(overlay, caption[:88], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(output_root / f"{row['sample_id']}__{row['failure_type']}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return len(selected)


def forest_svg(path: Path, paired_rows: Sequence[Mapping[str, Any]]) -> None:
    selected = [row for row in paired_rows if row["metric"] in {"family_found_point_in_target", "answerability_macro_f1"}]
    width = 1000
    height = 80 + 58 * len(selected)
    x0, x1 = 430, 930
    min_x, max_x = -0.35, 0.35
    def px(value: float) -> float:
        return x0 + (float(value) - min_x) / (max_x - min_x) * (x1 - x0)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="30" font-family="sans-serif" font-size="20">Dev paired family-bootstrap deltas (left minus right)</text>',
        f'<line x1="{px(0):.1f}" y1="52" x2="{px(0):.1f}" y2="{height-25}" stroke="#555" stroke-dasharray="4 4"/>',
    ]
    for index, row in enumerate(selected):
        y = 75 + index * 58
        label = f"{row['contrast']} · {row['metric']}"
        body.append(f'<text x="30" y="{y+5}" font-family="sans-serif" font-size="13">{label}</text>')
        body.append(f'<line x1="{px(row["ci95_low"]):.1f}" y1="{y}" x2="{px(row["ci95_high"]):.1f}" y2="{y}" stroke="#205493" stroke-width="3"/>')
        body.append(f'<circle cx="{px(row["estimate"]):.1f}" cy="{y}" r="5" fill="#b52b27"/>')
        body.append(f'<text x="{x1+8}" y="{y+5}" font-family="monospace" font-size="12">{row["estimate"]:+.3f} [{row["ci95_low"]:+.3f}, {row["ci95_high"]:+.3f}]</text>')
    body.append('</svg>')
    write_text_atomic(path, "\n".join(body) + "\n")


def report_markdown(report: Mapping[str, Any]) -> str:
    p1 = report["method_metrics"]["P1"]
    decision = report["decision"]
    lines = [
        "# P-CRA-U development failure audit and architecture decision",
        "",
        f"- Decision: `{decision}`",
        "- Status: `EXPLORATORY_DEV_ONLY`; 80 family là đơn vị độc lập, 400 variants không được coi là 400 quan sát độc lập.",
        "- Checkpoint: `step_000002000` (epoch 10), không có optimizer step hay retraining trong audit.",
        "- Calibration/Test-IID/Test-OOD vẫn đóng; Bảng 2–6 vẫn `NOT_RUN`.",
        "",
        "## Kết quả trọng tâm",
        "",
        f"- P1 false-FOUND trên AMBIGUOUS/ABSENT: `{p1['false_found_count']}/{p1['false_found_denominator']} = {p1['false_found_rate']:.4f}`.",
        f"- P1 point-outside-target trên FOUND: `{report['failure_counts']['point_outside_target']}/{p1['found_count']}`; point-in-target `{p1['found_point_in_target']:.4f}`.",
        f"- Answerability macro-F1: `{p1['answerability_macro_f1']:.4f}`; source micro-F1 bốn nhãn: `{p1['source_micro_f1']:.4f}`.",
        f"- Bootstrap: `{report['bootstrap']['replicates']}` lần, resample theo family, seed `{report['bootstrap']['seed']}`.",
        "",
        "## Diễn giải baseline và ablation",
        "",
        "- B0/B1 là RoboRefer thật với greedy decoding; vì luôn buộc trả point nên không có head answerability bốn trạng thái.",
        "- B2 dùng gate độ sâu hiện có và chỉ lấy clean dev làm phạm vi so sánh chính; không dùng oracle mask trong inference.",
        "- U1 chỉ vô hiệu explicit relation slot; từ chỉ quan hệ vẫn còn trong prompt tokens.",
        "- `A_NO_DEPTH` zero D0/thumbnail tại inference. Hai phép này là post-hoc intervention trên cùng checkpoint, không phải retrained causal ablation.",
        "",
        "## Gate quyết định",
        "",
    ]
    lines.extend(f"- `{name}`: `{'PASS' if value else 'FAIL'}`" for name, value in report["decision_gates"].items())
    lines += ["", "## Phạm vi bước sau", ""]
    if decision == "FREEZE_PCRA_U_ARCHITECTURE_AND_CHECKPOINT":
        lines.append("Kiến trúc/checkpoint được khóa; bước được phép tiếp theo là capture Calibration. Test vẫn đóng.")
    else:
        lines.append("Chỉ được sửa các failure mechanism ghi trong decision lock và train P-CRA-U V1.1 đúng một lần; Calibration/Test vẫn đóng.")
    lines += [
        "",
        "Các bảng so sánh trong run đều mang tên `table_dev_*_exploratory.csv`; chúng không phải Bảng 2–6 kết quả cuối.",
    ]
    return "\n".join(lines) + "\n"


def artifact_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": 1, "file_count": len(rows), "files": rows, "tree_commitment_sha256": canonical_sha256(rows)}


def run_evaluation(result_root: Path) -> dict[str, Any]:
    manifest = verify_locked_manifest()
    existing_report_path = result_root / "PCRA_U_DEVELOPMENT_FAILURE_AUDIT_REPORT.json"
    if existing_report_path.exists():
        existing_report = read_json(existing_report_path)
        correction = manifest.get("statistical_correction_01")
        if not correction or existing_report.get("statistical_correction_01"):
            raise AuditError(f"Refusing to overwrite completed audit: {result_root}")
        snapshot_root = result_root / "checks/pre_statistical_correction_01"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        snapshot_sources = [
            existing_report_path,
            result_root / "PCRA_U_DEVELOPMENT_FAILURE_AUDIT_REPORT.md",
            result_root / "figures/F09_dev_ablation_paired_family_bootstrap.svg",
            result_root / "checks/development_failure_audit_gate.json",
            result_root / "checks/artifact_manifest.json",
            DECISION_LOCK_PATH,
            PROTOCOL_REPORT_PATH,
            *sorted((result_root / "tables").glob("*.csv")),
        ]
        for source in snapshot_sources:
            if source.is_file():
                destination = snapshot_root / source.name
                if not destination.exists():
                    destination.write_bytes(source.read_bytes())
    config = read_json(TRAIN_CONFIG_PATH)
    train_manifest = read_json(TRAIN_MANIFEST_PATH)
    before = protected_now(train_manifest, config)
    if before != manifest["protected_inputs"]:
        raise AuditError("Protected dataset/model/baseline changed before evaluation")
    baseline_manifest_path = result_root / "checks/baseline_inference_manifest.json"
    baseline_manifest = read_json(baseline_manifest_path)
    baseline_predictions_path = safe_resolve(WORKSPACE, baseline_manifest["predictions_path"])
    if (
        baseline_manifest.get("status") != "COMPLETE"
        or baseline_manifest.get("record_count") != 800
        or sha256_file(baseline_predictions_path) != baseline_manifest.get("predictions_sha256")
        or baseline_manifest.get("oracle_or_annotation_read_by_runner") is not False
    ):
        raise AuditError("RoboRefer baseline inference is incomplete or invalid")
    baseline_predictions = read_jsonl(baseline_predictions_path)
    dev_entries = [entry for entry in train_manifest["entries"] if entry["split"] == "dev"]
    dev_entries.sort(key=lambda row: (row["family_id"], VARIANT_ORDER.index(row["variant"])))
    train_entries = [entry for entry in train_manifest["entries"] if entry["split"] == "train"]
    index_path, index = verify_feature_index("full")
    identity = checkpoint_identity(index_path)
    checkpoint_audit = audit_checkpoint_root(CHECKPOINT_ROOT)
    if not checkpoint_audit["passed"]:
        raise AuditError("Checkpoint root audit failed")
    device = torch.device("cuda")
    seed_runtime(int(config["seed"]), strict=True)
    model = PCRAUDevelopmentV1(config).to(device)
    load_result = load_model_for_evaluation(CHECKPOINT_PATH, model=model, expected_identity=identity)
    if not load_result["verification"]["passed"]:
        raise AuditError("Selected checkpoint failed safe evaluation load")
    dataset = FrozenFeatureDataset(WORKSPACE, dev_entries, FEATURE_CACHE_ROOT, index, config)
    sidecar, p1_heatmaps = evaluate_sidecar_methods(model, dataset, dev_entries, config, device)
    baseline = baseline_method_rows(baseline_predictions, dev_entries)
    gate_module = import_geometry_gate()
    b2, b2_reasons = b2_rows(baseline["B1"], dev_entries, gate_module)
    methods: dict[str, list[dict[str, Any]]] = {**baseline, **sidecar, "B2": b2}
    method_metrics = {method: score_rows(rows) for method, rows in methods.items()}
    # Reinitialize the exact preregistered seed for every method/contrast.
    # This avoids undisclosed derived seeds and makes the sampling schedule
    # directly reproducible from the contract and manifest.
    bootstrap = {method: family_bootstrap(rows, BOOTSTRAP_SEED) for method, rows in methods.items()}
    contrasts = {
        "P1_minus_U1": paired_bootstrap(methods["P1"], methods["U1"], BOOTSTRAP_SEED),
        "P1_minus_A_NO_DEPTH": paired_bootstrap(methods["P1"], methods["A_NO_DEPTH"], BOOTSTRAP_SEED),
        "B1_minus_B0": paired_bootstrap(methods["B1"], methods["B0"], BOOTSTRAP_SEED),
        "P1_minus_B1": paired_bootstrap(methods["P1"], methods["B1"], BOOTSTRAP_SEED),
        "P1_minus_B2_clean": paired_bootstrap(
            [row for row in methods["P1"] if row["variant"] == "clean"], methods["B2"], BOOTSTRAP_SEED
        ),
    }
    dataset_root = safe_resolve(WORKSPACE, train_manifest["dataset_root"])
    entry_by_sample = {entry["sample_id"]: entry for entry in dev_entries}
    p1_enriched = [enrich_audit_row(row, entry_by_sample[row["sample_id"]], dataset_root) for row in methods["P1"]]
    clean_area = {
        row["family_id"]: row["target_mask_pixels"]
        for row in p1_enriched
        if row["variant"] == "clean"
    }
    for row in p1_enriched:
        denominator = clean_area.get(row["family_id"], 0)
        row["occlusion_visibility_ratio_vs_clean"] = (
            row["target_mask_pixels"] / denominator
            if row["variant"] == "occlusion_view_counterfactual" and denominator > 0
            else None
        )
    false_found = [
        {**row, "failure_type": "false_FOUND"}
        for row in p1_enriched
        if row["answerability_truth"] in {"AMBIGUOUS", "ABSENT"} and row["answerability_prediction"] == "FOUND"
    ]
    outside = [
        {**row, "failure_type": "point_outside_target"}
        for row in p1_enriched
        if row["answerability_truth"] == "FOUND" and row["point_in_target"] is False
    ]
    failures = false_found + outside
    if len(false_found) != 9 or len(outside) != 24:
        raise AuditError(f"Checkpoint failure counts changed: false_FOUND={len(false_found)}, outside={len(outside)}")
    output_dirs = ["tables", "figures", "images", "predictions", "checks", "locked_inputs"]
    for directory in output_dirs:
        (result_root / directory).mkdir(parents=True, exist_ok=True)
    write_json(
        result_root / "predictions/dev_failure_audit_predictions.json",
        {"scientific_status": "EXPLORATORY_DEV_ONLY", "methods": methods},
    )
    method_rows = []
    ci_rows = []
    for method in METHOD_ORDER:
        metrics = method_metrics[method]
        method_rows.append({"scientific_status": "EXPLORATORY_DEV_ONLY", "method": method, "label": METHOD_LABELS[method], **metrics})
        for metric, interval in bootstrap[method].items():
            ci_rows.append({"scientific_status": "EXPLORATORY_DEV_ONLY", "method": method, "metric": metric, **interval, "family_bootstrap_replicates": BOOTSTRAP_REPLICATES})
    paired_rows = []
    for contrast, values in contrasts.items():
        for metric, interval in values.items():
            paired_rows.append({"scientific_status": "EXPLORATORY_DEV_ONLY", "contrast": contrast, "metric": metric, **interval, "paired_family_count": 80 if "B2" not in contrast else 80})
    write_csv(result_root / "tables/table_dev_method_comparison_exploratory.csv", method_rows)
    write_csv(result_root / "tables/table_dev_family_bootstrap_ci_exploratory.csv", ci_rows)
    write_csv(result_root / "tables/table_dev_paired_deltas_exploratory.csv", paired_rows)
    write_csv(
        result_root / "tables/table_dev_ablation_exploratory.csv",
        [row for row in method_rows if row["method"] in {"P1", "U1", "A_NO_DEPTH"}],
    )
    failure_fields = [
        "scientific_status", "failure_type", "sample_id", "family_id", "variant", "relation",
        "answerability_truth", "answerability_prediction", "target_asset_ids", "target_asset_classes",
        "anchor_asset_ids", "anchor_asset_classes", "family_category", "state_submode", "map_x", "map_y",
        "point_in_target", "target_mask_pixels", "target_mask_fraction", "mask_size_bin",
        "target_interior_pixels", "target_interior_fraction",
        "target_median_depth_m", "target_depth_mad_m", "target_valid_depth_fraction", "depth_bin",
        "anchor_median_depth_m", "anchor_depth_mad_m",
        "observed_relation_signed_margin", "relation_margin_unit", "occlusion_variant", "depth_corruption_variant",
        "occlusion_visibility_ratio_vs_clean",
        "normalized_centroid_error", "camera_x_m", "camera_y_m", "camera_z_m",
    ]
    write_csv(
        result_root / "tables/table_dev_failure_cases_exploratory.csv",
        [{"scientific_status": "EXPLORATORY_DEV_ONLY", **row} for row in failures],
        failure_fields,
    )
    strata = failure_strata(p1_enriched)
    write_csv(result_root / "tables/table_dev_failure_strata_exploratory.csv", strata)
    sources = source_table(methods["P1"])
    write_csv(result_root / "tables/table_dev_source_head_4class_exploratory.csv", sources)
    confusion_rows = []
    for method in METHOD_ORDER:
        confusion_rows.extend(confusion_table(methods[method], method))
    write_csv(result_root / "tables/table_dev_answerability_confusion_exploratory.csv", confusion_rows)
    relation_rows = [row for row in strata if row["dimension"] == "relation"]
    write_csv(result_root / "tables/table_dev_grounding_by_relation_exploratory.csv", relation_rows)
    write_json(
        result_root / "tables/TABLES_02_TO_06_STATUS.json",
        {"status": "NOT_RUN", "reason": "Only exploratory dev was opened; Calibration/Test-IID/Test-OOD remain sealed"},
    )
    forest_svg(result_root / "figures/F09_dev_ablation_paired_family_bootstrap.svg", paired_rows)
    rendered = render_failure_images(result_root, failures, entry_by_sample, p1_heatmaps)
    p1_ci = bootstrap["P1"]
    relation_delta = contrasts["P1_minus_U1"]
    depth_delta = contrasts["P1_minus_A_NO_DEPTH"]
    threshold = float(manifest["decision_thresholds"]["max_allowed_ablation_delta_ci_lower"])
    relation_non_degraded = (
        relation_delta["family_found_point_in_target"]["ci95_low"] > threshold
        or relation_delta["answerability_macro_f1"]["ci95_low"] > threshold
    )
    depth_non_degraded = (
        depth_delta["family_found_point_in_target"]["ci95_low"] > threshold
        or depth_delta["answerability_macro_f1"]["ci95_low"] > threshold
    )
    after = protected_now(train_manifest, config)
    sealing = {
        "only_train_dev_splits_in_training_manifest": set(row["split"] for row in train_manifest["entries"]) == {"train", "dev"},
        "calibration_fit": False,
        "test_opened": False,
        "tables_02_to_06": "NOT_RUN",
    }
    decision_gates = {
        "selected_checkpoint_integrity": checkpoint_audit["passed"] and load_result["verification"]["passed"],
        "protected_dataset_model_baseline_unchanged": before == after == manifest["protected_inputs"],
        "calibration_and_tests_sealed": sealing["only_train_dev_splits_in_training_manifest"] and not sealing["calibration_fit"] and not sealing["test_opened"],
        "exact_expected_failure_counts_reproduced": len(false_found) == 9 and len(outside) == 24,
        "p1_false_found_point_estimate_at_most_0_10": method_metrics["P1"]["false_found_rate"] <= 0.10,
        "p1_family_grounding_ci_lower_at_least_0_70": p1_ci["family_found_point_in_target"]["ci95_low"] >= 0.70,
        "explicit_relation_intervention_non_degradation": relation_non_degraded,
        "depth_intervention_non_degradation": depth_non_degraded,
    }
    decision = "FREEZE_PCRA_U_ARCHITECTURE_AND_CHECKPOINT" if all(decision_gates.values()) else "ONE_CONTROLLED_DEVELOPMENT_REVISION"
    reasons = [name for name, passed in decision_gates.items() if not passed]
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "scientific_status": "EXPLORATORY_DEV_ONLY_CALIBRATION_TESTS_SEALED",
        "passed": True,
        "decision": decision,
        "decision_gates": decision_gates,
        "decision_reasons": reasons,
        "failure_counts": {"false_found": len(false_found), "point_outside_target": len(outside)},
        "method_metrics": method_metrics,
        "family_bootstrap_ci": bootstrap,
        "paired_family_bootstrap_deltas": contrasts,
        "bootstrap": manifest["bootstrap"],
        "statistical_correction_01": manifest.get("statistical_correction_01"),
        "b2_geometry_gate_reason_counts_clean": dict(sorted(b2_reasons.items())),
        "source_head": {
            "type": "four_label_multilabel",
            "classes": SOURCE_CLASSES,
            "per_class": sources,
            "spatial_source_deferred_no_positive_label": True,
            "semantic_relation_truth_vectors_identical": all(
                (("semantic" in row["source_truth"]) == ("relation" in row["source_truth"]))
                for row in methods["P1"]
            ),
        },
        "special_relation_audit": {
            row["stratum"]: row for row in relation_rows if row["stratum"] in {"right_of", "between_in_depth"}
        },
        "observed": {"real_failure_images": rendered, "dev_families": 80, "dev_samples": 400},
        "checkpoint": manifest["checkpoint"],
        "checkpoint_audit": checkpoint_audit,
        "protected_before": before,
        "protected_after": after,
        "sealing": sealing,
        "training_performed": False,
        "optimizer_steps": 0,
        "calibration_fit": False,
        "test_opened": False,
        "tables_02_to_06": "NOT_RUN",
        "limitations": [
            "U1 retains lexical relation words and only removes the explicit relation ID slot.",
            "A_NO_DEPTH is an inference-time zeroing intervention, not a separately retrained model.",
            "B2 primary metrics are clean-only because its metric-depth evidence is not equivalent to the corrupted relative-depth variant.",
            "Dev subgroup denominators below ten families are descriptive and marked INSUFFICIENT_FAMILIES.",
            "Semantic and relation source labels are perfectly co-occurring in this dev set, so separate attribution is not identifiable.",
        ],
    }
    report_json_path = result_root / "PCRA_U_DEVELOPMENT_FAILURE_AUDIT_REPORT.json"
    write_json(report_json_path, report)
    markdown = report_markdown(report)
    write_text_atomic(result_root / "PCRA_U_DEVELOPMENT_FAILURE_AUDIT_REPORT.md", markdown)
    write_text_atomic(PROTOCOL_REPORT_PATH, markdown)
    revision_scope = []
    if not relation_non_degraded:
        revision_scope.append("RELATION_CONDITIONING_ONLY")
    if not depth_non_degraded:
        revision_scope.append("DEPTH_FUSION_ONLY")
    if not decision_gates["p1_false_found_point_estimate_at_most_0_10"]:
        revision_scope.append("ANSWERABILITY_FALSE_FOUND_ONLY")
    if not decision_gates["p1_family_grounding_ci_lower_at_least_0_70"]:
        revision_scope.append("TARGET_HEATMAP_ONLY")
    decision_lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN" if decision.startswith("FREEZE") else "NOT_FROZEN_ONE_REVISION_AUTHORIZED",
        "decision": decision,
        "architecture_frozen": decision.startswith("FREEZE"),
        "checkpoint_frozen": decision.startswith("FREEZE"),
        "selected_checkpoint": manifest["checkpoint"],
        "checkpoint_manifest_sha256": sha256_file(CHECKPOINT_PATH / "manifest.json"),
        "checkpoint_metadata_sha256": sha256_file(CHECKPOINT_PATH / "metadata.json"),
        "eval_manifest_sha256": sha256_file(EVAL_MANIFEST_PATH),
        "audit_contract_sha256": sha256_file(CONTRACT_PATH),
        "audit_code_sha256": sha256_file(Path(__file__).resolve()),
        "audit_report_sha256": sha256_file(report_json_path),
        "dev_family_count": 80,
        "dev_sample_count": 400,
        "bootstrap": manifest["bootstrap"],
        "decision_gates": decision_gates,
        "authorized_revision_scope": revision_scope,
        "allowed_next_action": "CAPTURE_CALIBRATION" if decision.startswith("FREEZE") else "TRAIN_PCRA_U_V1_1_ONCE",
        "calibration_fit": False,
        "test_opened": False,
        "tables_02_to_06": "NOT_RUN",
        "protected_inputs": after,
        "created_at_utc": utc_now(),
    }
    write_json(DECISION_LOCK_PATH, decision_lock)
    write_json(result_root / "checks/development_failure_audit_gate.json", report)
    write_json(
        result_root / "checks/selected_checkpoint_integrity.json",
        {"passed": checkpoint_audit["passed"] and load_result["verification"]["passed"], "checkpoint_audit": checkpoint_audit, "load_verification": load_result["verification"]},
    )
    write_json(result_root / "checks/sealed_split_attestation.json", sealing)
    for source in [CONTRACT_PATH, EVAL_MANIFEST_PATH, CHECKPOINT_PATH / "metadata.json", CHECKPOINT_PATH / "manifest.json", CHECKPOINT_ROOT / "best_dev_total_loss.json"]:
        destination = result_root / "locked_inputs" / source.name
        source_bytes = source.read_bytes()
        if destination.exists() and destination.read_bytes() != source_bytes:
            historical = result_root / "locked_inputs/pre_statistical_correction_01" / source.name
            historical.parent.mkdir(parents=True, exist_ok=True)
            if not historical.exists():
                historical.write_bytes(destination.read_bytes())
        destination.write_bytes(source_bytes)
    run_card = (
        "# Run card — P-CRA-U development failure audit\n\n"
        f"- Decision: `{decision}`\n"
        "- Scope: 80 dev families / 400 dependent variants; family-cluster bootstrap.\n"
        "- Training: none. Calibration/Test: sealed. Robot action: none.\n"
        f"- Checkpoint model SHA-256: `{manifest['checkpoint']['model_sha256']}`.\n"
    )
    write_text_atomic(result_root / "RUN_CARD.md", run_card)
    write_json(result_root / "checks/artifact_manifest.json", artifact_manifest(result_root))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    baseline = sub.add_parser("baseline")
    baseline.add_argument("--server-url", default="http://127.0.0.1:25547")
    baseline.add_argument("--timeout-seconds", type=float, default=180.0)
    baseline.add_argument("--result-root", default=relative(DEFAULT_RESULT_ROOT))
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--result-root", default=relative(DEFAULT_RESULT_ROOT))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_manifest()
        summary = {"status": result["status"], "families": result["scope"]["family_count"], "samples": result["scope"]["sample_count"]}
    elif args.command == "baseline":
        result = run_baselines(safe_resolve(WORKSPACE, args.result_root), args.server_url, args.timeout_seconds)
        summary = {"status": result["status"], "records": result["record_count"], "parse_counts": result["parse_counts"]}
    else:
        result = run_evaluation(safe_resolve(WORKSPACE, args.result_root))
        summary = {"passed": result["passed"], "decision": result["decision"], "failure_counts": result["failure_counts"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
