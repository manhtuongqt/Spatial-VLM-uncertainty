#!/usr/bin/env python3
"""Fit and freeze P-CRA-U v1 risk calibration on the locked Calibration split.

This is deliberately the *second* phase of calibration.  Frozen-model raw
inference must already have completed without evaluator access.  Only this
script joins those immutable raw predictions to evaluator masks, constructs
``y_error``, fits the two predeclared one-dimensional Platt models, and freezes
the FOUND/ABSTAIN operating point.

The script cannot train or load P-CRA-U, cannot choose a checkpoint, and has no
Test-IID/Test-OOD input.  All uncertainty intervals cluster-resample complete
scene-query families, retaining all five dependent variants.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


WORKSPACE = Path(__file__).resolve().parents[1]
CONTRACT_PATH = WORKSPACE / "protocol/PCRA_U_CALIBRATION_CONTRACT.md"
CONFIG_PATH = WORKSPACE / "protocol/pcra_u_calibration_config.json"
EVAL_MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_calibration_eval_manifest.json"
RAW_PREDICTIONS_PATH = (
    WORKSPACE
    / "results/pcra_u_runs/pcra_u_calibration_20260824/predictions/calibration_raw_predictions.jsonl"
)
RAW_ATTESTATION_PATH = (
    WORKSPACE
    / "results/pcra_u_runs/pcra_u_calibration_20260824/checks/calibration_raw_inference_manifest.json"
)
RESULT_ROOT = WORKSPACE / "results/pcra_u_runs/pcra_u_calibration_20260824"
FREEZE_LOCK_PATH = WORKSPACE / "protocol/pcra_u_calibrator_threshold_freeze_lock.json"
ARCHITECTURE_LOCK_PATH = WORKSPACE / "protocol/pcra_u_architecture_freeze_lock.json"
CHECKPOINT_PATH = (
    WORKSPACE
    / "results/pcra_u_runs/pcra_u_development_train_20260824/checkpoints/development/step_000002000"
)
DEVELOPMENT_DATASET_INDEX = (
    WORKSPACE / "datasets/roborefer_dataset_v2_1_1_development_400_20260824/dataset_index.json"
)

PROTOCOL_ID = "pcra_u_calibration_v1"
SCIENTIFIC_STATUS = "CALIBRATION_FIT_ONLY"
ANSWERABILITY_CLASSES = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
SOURCE_CLASSES = ["semantic", "relation", "depth", "occlusion"]
METRIC_NAMES = [
    "brier",
    "nll",
    "ece_10bin",
    "aurc",
    "risk_at_80pct_coverage",
    "coverage_at_5pct_risk",
]
PLOT_METHODS = ["multimodal_raw_proxy", "spatial_platt_crossfit", "multimodal_platt_crossfit"]
FORBIDDEN_RAW_KEYS = {
    "answerability_state",
    "answerability_truth",
    "candidate_target_ids",
    "evaluator_only",
    "label",
    "labels",
    "object_id",
    "object_ids",
    "oracle",
    "point_in_target",
    "query_graph",
    "target_mask",
    "target_mask_path",
    "truth",
    "y_error",
}


class CalibrationFitError(RuntimeError):
    """Raised for a protocol, provenance, or numerical gate failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            json_clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    write_bytes_atomic(path, canonical_bytes(value))


def write_text(path: Path, text: str) -> None:
    write_bytes_atomic(path, text.encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CalibrationFitError(f"Missing required JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalibrationFitError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CalibrationFitError(f"Missing raw prediction JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationFitError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise CalibrationFitError(f"Prediction row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE.resolve()))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = {}
            for field in fields:
                value = source.get(field)
                if isinstance(value, float) and not math.isfinite(value):
                    value = ""
                row[field] = value
            writer.writerow(row)
    os.replace(temporary, path)


def resolve_workspace_path(value: str, dataset_root: Path | None = None) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [WORKSPACE / raw]
    if dataset_root is not None and not raw.is_absolute():
        candidates.append(dataset_root / raw)
    existing = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    resolved = existing.resolve()
    try:
        resolved.relative_to(WORKSPACE.resolve())
    except ValueError as exc:
        raise CalibrationFitError(f"Path escapes workspace: {value}") from exc
    return resolved


def nested_flag(payload: Mapping[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    for parent in ("safety", "attestation", "inference_attestation", "checks"):
        child = payload.get(parent)
        if isinstance(child, Mapping) and key in child:
            return child[key]
    return None


def manifest_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("entries", "records", "samples"):
        value = manifest.get(key)
        if isinstance(value, list):
            if not all(isinstance(item, dict) for item in value):
                raise CalibrationFitError(f"Evaluator manifest `{key}` has non-object rows")
            return [dict(item) for item in value]
    raise CalibrationFitError("Evaluator manifest must contain `entries` (preferred), `records`, or `samples`")


def vector_field(row: Mapping[str, Any], key: str, names: Sequence[str]) -> np.ndarray:
    value = row.get(key)
    if isinstance(value, Mapping):
        try:
            values = [float(value[name]) for name in names]
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationFitError(f"{row.get('sample_id')}: malformed `{key}` mapping") from exc
    elif isinstance(value, list) and len(value) == len(names):
        try:
            values = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise CalibrationFitError(f"{row.get('sample_id')}: malformed `{key}` list") from exc
    else:
        raise CalibrationFitError(
            f"{row.get('sample_id')}: `{key}` must be length-{len(names)} list or named mapping"
        )
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise CalibrationFitError(f"{row.get('sample_id')}: `{key}` contains non-finite values")
    return array


def assert_oracle_free_row(row: Mapping[str, Any]) -> None:
    stack: list[Mapping[str, Any]] = [row]
    while stack:
        current = stack.pop()
        for key, value in current.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_RAW_KEYS:
                raise CalibrationFitError(
                    f"Raw prediction {row.get('sample_id')} contains forbidden evaluator/oracle key `{key}`"
                )
            if isinstance(value, Mapping):
                stack.append(value)


def recompute_raw_scores(row: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[float, float]:
    answer_logits = vector_field(row, "answerability_logits", ANSWERABILITY_CLASSES)
    source_logits = vector_field(row, "source_logits", SOURCE_CLASSES)
    found_log_odds = float(answer_logits[0] - logsumexp(answer_logits[1:]))

    aliases = {
        "heatmap_entropy_normalized": ("heatmap_entropy_normalized", "normalized_heatmap_entropy"),
        "local_map_mass_3x3": ("local_map_mass_3x3", "map_mass_3x3"),
        "peak_margin": ("peak_margin", "top1_top2_peak_margin"),
        "mode_count": ("mode_count", "heatmap_mode_count"),
        "rgb_depth_thumbnail_cosine": ("rgb_depth_thumbnail_cosine", "rgb_depth_cosine"),
        "rgb_depth_thumbnail_mean_absolute_gap": (
            "rgb_depth_thumbnail_mean_absolute_gap",
            "rgb_depth_thumbnail_mae",
        ),
        "valid_depth_fraction": ("valid_depth_fraction", "depth_valid_fraction"),
        "invalid_depth_fraction": ("invalid_depth_fraction", "zero_depth_fraction"),
    }

    def scalar(name: str) -> float:
        for alias in aliases[name]:
            if alias in row:
                try:
                    result = float(row[alias])
                except (TypeError, ValueError) as exc:
                    raise CalibrationFitError(f"{row.get('sample_id')}: `{alias}` is not numeric") from exc
                if not math.isfinite(result):
                    raise CalibrationFitError(f"{row.get('sample_id')}: `{alias}` is non-finite")
                return result
        raise CalibrationFitError(f"{row.get('sample_id')}: missing raw evidence `{name}`")

    entropy = scalar("heatmap_entropy_normalized")
    local_mass = scalar("local_map_mass_3x3")
    peak_margin = scalar("peak_margin")
    mode_count = scalar("mode_count")
    cosine = scalar("rgb_depth_thumbnail_cosine")
    mean_gap = scalar("rgb_depth_thumbnail_mean_absolute_gap")
    # Raw inference currently persists the observable zero-pixel proxy as
    # ``invalid_depth_fraction``.  Accept an explicit valid fraction only as a
    # schema-compatible alternative, never combine the two silently.
    if any(alias in row for alias in aliases["invalid_depth_fraction"]):
        invalid_depth = scalar("invalid_depth_fraction")
        valid_depth = 1.0 - invalid_depth
    else:
        valid_depth = scalar("valid_depth_fraction")
        invalid_depth = 1.0 - valid_depth
    if not (0.0 <= entropy <= 1.0 + 1e-6 and 0.0 <= local_mass <= 1.0 + 1e-6):
        raise CalibrationFitError(f"{row.get('sample_id')}: probability summary outside [0,1]")
    if mode_count < 0 or not (
        0.0 <= valid_depth <= 1.0 + 1e-6 and 0.0 <= invalid_depth <= 1.0 + 1e-6
    ):
        raise CalibrationFitError(f"{row.get('sample_id')}: invalid mode/depth summary")
    if "top1_probability" in row and "top2_probability" in row:
        implied_margin = float(row["top1_probability"]) - float(row["top2_probability"])
        if abs(implied_margin - peak_margin) > 2e-5:
            raise CalibrationFitError(f"{row.get('sample_id')}: peak-margin fields disagree")

    weights = config["raw_evidence"]["spatial_score"]
    spatial = (
        float(weights["answer_found_log_odds"]) * found_log_odds
        + float(weights["one_minus_normalized_entropy"]) * (1.0 - entropy)
        + float(weights["local_map_mass_3x3"]) * local_mass
        + float(weights["top1_top2_peak_margin"]) * peak_margin
        + float(weights["log1p_mode_count"]) * math.log1p(mode_count)
    )
    additions = config["raw_evidence"]["multimodal_score_additions"]
    multimodal = spatial + (
        float(additions["mean_source_probability"]) * float(expit(source_logits).mean())
        + float(additions["rgb_depth_thumbnail_cosine"]) * cosine
        + float(additions["rgb_depth_thumbnail_mean_absolute_gap"]) * mean_gap
        + float(additions["invalid_depth_fraction"]) * invalid_depth
    )
    return float(spatial), float(multimodal)


def validate_raw_inputs(
    config: Mapping[str, Any], predictions_path: Path, attestation_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attestation = read_json(attestation_path)
    rows = read_jsonl(predictions_path)
    expected_count = int(config["dataset"]["sample_count"])
    reported_count = attestation.get("prediction_count", attestation.get("sample_count"))
    reported_hash = attestation.get("predictions_sha256", attestation.get("raw_predictions_sha256"))
    actual_hash = sha256_file(predictions_path)
    if reported_count != expected_count or len(rows) != expected_count:
        raise CalibrationFitError(
            f"Raw prediction count mismatch: manifest={reported_count}, file={len(rows)}, expected={expected_count}"
        )
    if reported_hash != actual_hash:
        raise CalibrationFitError("Raw prediction SHA-256 does not match inference attestation")
    if nested_flag(attestation, "oracle_or_annotation_read") is not False:
        raise CalibrationFitError("Raw inference attestation does not prove oracle_or_annotation_read=false")
    if nested_flag(attestation, "test_opened") not in (None, False):
        raise CalibrationFitError("Raw inference attestation indicates a locked test was opened")
    if attestation.get("split", "calibration") != "calibration":
        raise CalibrationFitError("Raw inference split is not calibration")
    model_hash = config["frozen_model"]["model_sha256"]
    attested_model_hash = attestation.get("checkpoint_model_sha256", attestation.get("model_sha256"))
    if attested_model_hash != model_hash:
        raise CalibrationFitError("Raw inference checkpoint hash is not the frozen model hash")

    seen: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        assert_oracle_free_row(row)
        sample_id = str(row.get("sample_id", ""))
        family_id = str(row.get("family_id", ""))
        if not sample_id or not family_id or sample_id in seen:
            raise CalibrationFitError(f"Missing/duplicate raw prediction identity: {sample_id!r}")
        seen.add(sample_id)
        for field in ("map_x", "map_y", "spatial_raw_score", "multimodal_raw_score"):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise CalibrationFitError(f"{sample_id}: missing or invalid `{field}`") from exc
            if not math.isfinite(value):
                raise CalibrationFitError(f"{sample_id}: non-finite `{field}`")
        computed_spatial, computed_multimodal = recompute_raw_scores(row, config)
        if abs(computed_spatial - float(row["spatial_raw_score"])) > 2e-5:
            raise CalibrationFitError(f"{sample_id}: spatial_raw_score violates locked evidence formula")
        if abs(computed_multimodal - float(row["multimodal_raw_score"])) > 2e-5:
            raise CalibrationFitError(f"{sample_id}: multimodal_raw_score violates locked evidence formula")
        normalized = dict(row)
        normalized["spatial_raw_score"] = computed_spatial
        normalized["multimodal_raw_score"] = computed_multimodal
        normalized_rows.append(normalized)
    return normalized_rows, attestation


def join_evaluator(
    config: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]], eval_manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest = read_json(eval_manifest_path)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise CalibrationFitError("Evaluator manifest protocol_id mismatch")
    if manifest.get("status") != "CALIBRATION_EVALUATOR_ONLY_LABELS_SEPARATED":
        raise CalibrationFitError("Evaluator manifest is not the locked labels-separated revision")
    if manifest.get("split", "calibration") != "calibration":
        raise CalibrationFitError("Evaluator manifest is not Calibration-only")
    if nested_flag(manifest, "test_opened") not in (None, False):
        raise CalibrationFitError("Evaluator manifest indicates Test was opened")
    entries = manifest_entries(manifest)
    expected_samples = int(config["dataset"]["sample_count"])
    expected_families = int(config["dataset"]["family_count"])
    if len(entries) != expected_samples:
        raise CalibrationFitError(f"Evaluator entry count {len(entries)} != {expected_samples}")
    by_sample: dict[str, dict[str, Any]] = {}
    for entry in entries:
        sample_id = str(entry.get("sample_id", ""))
        if not sample_id or sample_id in by_sample:
            raise CalibrationFitError(f"Missing/duplicate evaluator sample ID: {sample_id!r}")
        if entry.get("split", "calibration") != "calibration":
            raise CalibrationFitError(f"Non-calibration evaluator row: {sample_id}")
        by_sample[sample_id] = entry
    raw_ids = {str(row["sample_id"]) for row in raw_rows}
    if raw_ids != set(by_sample):
        missing = sorted(set(by_sample) - raw_ids)[:5]
        extra = sorted(raw_ids - set(by_sample))[:5]
        raise CalibrationFitError(f"Raw/evaluator sample set mismatch; missing={missing}, extra={extra}")

    dataset_value = manifest.get("dataset_root", config["dataset"]["root"])
    dataset_root = resolve_workspace_path(str(dataset_value))
    joined: list[dict[str, Any]] = []
    family_variants: defaultdict[str, set[str]] = defaultdict(set)
    family_clean_state: dict[str, str] = {}
    coordinate_failures = 0
    verified_masks = 0
    for raw in raw_rows:
        sample_id = str(raw["sample_id"])
        entry = by_sample[sample_id]
        family_id = str(entry.get("family_id", raw["family_id"]))
        variant = str(entry.get("variant", raw.get("variant", "")))
        if family_id != str(raw["family_id"]):
            raise CalibrationFitError(f"Family identity mismatch: {sample_id}")
        supervision = entry.get("supervision")
        if not isinstance(supervision, Mapping):
            raise CalibrationFitError(f"Missing evaluator supervision: {sample_id}")
        state = str(supervision.get("answerability_state", supervision.get("state", "")))
        if state not in ANSWERABILITY_CLASSES:
            raise CalibrationFitError(f"Invalid answerability state for {sample_id}: {state!r}")
        if variant == "clean":
            family_clean_state[family_id] = state
        family_variants[family_id].add(variant)

        x, y = int(float(raw["map_x"])), int(float(raw["map_y"]))
        selected_correct = False
        mask_shape: tuple[int, int] | None = None
        if state == "FOUND":
            mask_value = supervision.get("target_mask_path")
            mask_hash = supervision.get("target_mask_sha256")
            if not mask_value or not mask_hash:
                raise CalibrationFitError(f"FOUND row lacks locked target mask: {sample_id}")
            mask_path = resolve_workspace_path(str(mask_value), dataset_root)
            if not mask_path.is_file() or sha256_file(mask_path) != str(mask_hash):
                raise CalibrationFitError(f"Target-mask integrity failure: {sample_id}")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or int((mask > 0).sum()) <= 0:
                raise CalibrationFitError(f"Target mask is unreadable/empty: {sample_id}")
            height, width = mask.shape[:2]
            mask_shape = (height, width)
            in_bounds = 0 <= x < width and 0 <= y < height
            if not in_bounds:
                coordinate_failures += 1
            selected_correct = bool(in_bounds and mask[y, x] > 0)
            verified_masks += 1
        else:
            # Non-FOUND is unsafe by the predeclared event, independent of any
            # model point.  The evaluator mask is intentionally not consulted.
            in_bounds = 0 <= x < 640 and 0 <= y < 480
            if not in_bounds:
                coordinate_failures += 1
        joined.append(
            {
                "sample_id": sample_id,
                "family_id": family_id,
                "variant": variant,
                "relation": str(
                    raw.get(
                        "prompt_parsed_relation",
                        raw.get("relation", entry.get("audit_only", {}).get("relation", "unknown")),
                    )
                ),
                "answerability_state": state,
                "map_x": x,
                "map_y": y,
                "map_in_bounds": bool(in_bounds),
                "selected_grounding_correct": bool(selected_correct),
                "y_error": int(not selected_correct),
                "spatial_raw_score": float(raw["spatial_raw_score"]),
                "multimodal_raw_score": float(raw["multimodal_raw_score"]),
                "mask_height": mask_shape[0] if mask_shape else None,
                "mask_width": mask_shape[1] if mask_shape else None,
            }
        )

    if len(family_variants) != expected_families:
        raise CalibrationFitError(f"Evaluator family count {len(family_variants)} != {expected_families}")
    expected_variants = int(config["dataset"]["variants_per_family"])
    bad_variant_families = [family for family, values in family_variants.items() if len(values) != expected_variants]
    if bad_variant_families:
        raise CalibrationFitError(f"Families without exactly {expected_variants} variants: {bad_variant_families[:5]}")

    # Primary-stratum commitment is recovered from an explicit audit field if
    # present, otherwise from the clean variant, never from model output.
    primary: dict[str, str] = {}
    for entry in entries:
        family_id = str(entry["family_id"])
        audit = entry.get("audit_only", {})
        if isinstance(audit, Mapping):
            value = audit.get("primary_answerability_stratum")
            if value is not None:
                primary[family_id] = str(value)
    for family_id, state in family_clean_state.items():
        primary.setdefault(family_id, state)
    if set(primary) != set(family_variants):
        raise CalibrationFitError("Cannot determine primary answerability stratum for every Calibration family")
    observed_strata = Counter(primary.values())
    expected_strata = Counter({str(k): int(v) for k, v in config["dataset"]["primary_family_strata"].items()})
    if observed_strata != expected_strata:
        raise CalibrationFitError(
            f"Primary family strata mismatch: observed={dict(observed_strata)}, expected={dict(expected_strata)}"
        )
    join_audit = {
        "sample_count": len(joined),
        "family_count": len(family_variants),
        "variants_per_family": expected_variants,
        "primary_family_strata": dict(sorted(observed_strata.items())),
        "variant_state_counts": dict(sorted(Counter(row["answerability_state"] for row in joined).items())),
        "error_count": int(sum(row["y_error"] for row in joined)),
        "safe_execution_count": int(sum(row["selected_grounding_correct"] for row in joined)),
        "found_target_masks_verified": verified_masks,
        "coordinate_out_of_bounds_count": coordinate_failures,
        "strict_join_by_sample_id": True,
    }
    return joined, manifest, join_audit


def fold_for_family(seed: int, family_id: str, fold_count: int) -> int:
    digest = hashlib.sha256(f"{seed}|{family_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % fold_count


def fit_platt(x: np.ndarray, y: np.ndarray, l2_slope: float, max_iterations: int) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise CalibrationFitError("Invalid Platt fit arrays")
    if set(np.unique(y).tolist()) != {0.0, 1.0}:
        raise CalibrationFitError("Platt fitting requires both safe and error outcomes")
    prevalence = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    initial = np.asarray([0.0, math.log(prevalence / (1.0 - prevalence))], dtype=np.float64)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        slope, intercept = float(theta[0]), float(theta[1])
        linear = slope * x + intercept
        probability = expit(linear)
        value = float(np.mean(np.logaddexp(0.0, linear) - y * linear) + 0.5 * l2_slope * slope**2)
        residual = probability - y
        gradient = np.asarray(
            [float(np.mean(residual * x) + l2_slope * slope), float(np.mean(residual))],
            dtype=np.float64,
        )
        return value, gradient

    result = minimize(
        fun=lambda theta: objective(theta)[0],
        x0=initial,
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iterations), "ftol": 1e-14, "gtol": 1e-9, "maxls": 50},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise CalibrationFitError(f"Deterministic Platt fit failed: status={result.status}, {result.message}")
    slope, intercept = (float(result.x[0]), float(result.x[1]))
    return {
        "slope": slope,
        "intercept": intercept,
        "l2_slope": float(l2_slope),
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "optimizer": "scipy_L-BFGS-B_analytic_gradient",
        "converged": True,
    }


def apply_platt(x: np.ndarray, fit: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(expit(float(fit["slope"]) * x + float(fit["intercept"])), dtype=np.float64)


def crossfit_platt(
    x: np.ndarray,
    y: np.ndarray,
    family_ids: Sequence[str],
    seed: int,
    fold_count: int,
    l2_slope: float,
    max_iterations: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    folds = np.asarray([fold_for_family(seed, family, fold_count) for family in family_ids], dtype=np.int64)
    predicted = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    all_families = sorted(set(family_ids))
    for fold in range(fold_count):
        train = folds != fold
        held_out = folds == fold
        if not train.any() or not held_out.any():
            raise CalibrationFitError(f"Grouped cross-fit fold {fold} is empty")
        fit = fit_platt(x[train], y[train], l2_slope, max_iterations)
        predicted[held_out] = apply_platt(x[held_out], fit)
        held_families = {family_ids[index] for index in np.flatnonzero(held_out)}
        fold_rows.append(
            {
                "fold": fold,
                "train_families": len(set(all_families) - held_families),
                "held_out_families": len(held_families),
                "train_samples": int(train.sum()),
                "held_out_samples": int(held_out.sum()),
                "held_out_errors": int(y[held_out].sum()),
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "optimizer_iterations": fit["iterations"],
            }
        )
    if not np.isfinite(predicted).all():
        raise CalibrationFitError("Cross-fit did not generate exactly one prediction per Calibration sample")
    return predicted, fold_rows


def risk_coverage_curve(probability: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(probability, kind="mergesort")
    p_sorted, y_sorted = probability[order], y[order]
    cumulative_error = np.cumsum(y_sorted)
    # Thresholds operate on unique risks; all tied observations enter together.
    endpoints = np.flatnonzero(np.r_[p_sorted[1:] != p_sorted[:-1], True])
    accepted = endpoints + 1
    coverage = accepted.astype(np.float64) / len(y)
    risk = cumulative_error[endpoints] / accepted
    return {
        "threshold": p_sorted[endpoints],
        "coverage": coverage,
        "risk": risk,
        "accepted": accepted.astype(np.int64),
    }


def calibration_metrics(probability: np.ndarray, y: np.ndarray, bins: int = 10) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y, dtype=np.float64)
    if len(probability) != len(y) or len(y) == 0:
        raise CalibrationFitError("Cannot score empty/mismatched calibration arrays")
    brier = float(np.mean((probability - y) ** 2))
    nll = float(-np.mean(y * np.log(probability) + (1.0 - y) * np.log1p(-probability)))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if selected.any():
            ece += float(selected.mean()) * abs(float(probability[selected].mean()) - float(y[selected].mean()))
    curve = risk_coverage_curve(probability, y)
    # Tie-aware trapezoidal AURC.  The zero-coverage origin has risk zero and is
    # included explicitly; this definition is locked in the output report.
    aurc = float(np.trapezoid(np.r_[0.0, curve["risk"]], np.r_[0.0, curve["coverage"]]))
    at_80 = np.flatnonzero(curve["coverage"] >= 0.80)
    risk_80 = float(curve["risk"][at_80[0]]) if len(at_80) else float(curve["risk"][-1])
    feasible = curve["risk"] <= 0.05 + 1e-15
    coverage_5 = float(curve["coverage"][feasible].max()) if feasible.any() else 0.0
    return {
        "brier": brier,
        "nll": nll,
        "ece_10bin": float(ece),
        "aurc": aurc,
        "risk_at_80pct_coverage": risk_80,
        "coverage_at_5pct_risk": coverage_5,
    }


def reliability_rows(
    method: str, probability: np.ndarray, y: np.ndarray, family_ids: Sequence[str], bins: int
) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        indices = np.flatnonzero(selected)
        rows.append(
            {
                "scientific_status": SCIENTIFIC_STATUS,
                "method": method,
                "bin_index": index,
                "risk_lower": float(edges[index]),
                "risk_upper": float(edges[index + 1]),
                "sample_count": int(len(indices)),
                "family_count": len({family_ids[item] for item in indices}),
                "mean_predicted_risk": float(probability[selected].mean()) if selected.any() else float("nan"),
                "empirical_error_rate": float(y[selected].mean()) if selected.any() else float("nan"),
            }
        )
    return rows


def grouped_bootstrap_intervals(
    predictions: Mapping[str, np.ndarray],
    y: np.ndarray,
    family_ids: Sequence[str],
    replicates: int,
    seed: int,
    bins: int,
) -> dict[str, dict[str, dict[str, float]]]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(family_ids):
        grouped[str(family)].append(index)
    families = sorted(grouped)
    if not families:
        raise CalibrationFitError("Cannot bootstrap zero families")
    groups = [np.asarray(grouped[family], dtype=np.int64) for family in families]
    rng = np.random.default_rng(seed)
    draws = {
        method: {metric: np.empty(replicates, dtype=np.float64) for metric in METRIC_NAMES}
        for method in predictions
    }
    for replicate in range(replicates):
        selected_families = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected_families])
        target = y[indices]
        for method, values in predictions.items():
            metrics = calibration_metrics(values[indices], target, bins)
            for metric in METRIC_NAMES:
                draws[method][metric][replicate] = metrics[metric]
    result: dict[str, dict[str, dict[str, float]]] = {}
    for method, values in predictions.items():
        point = calibration_metrics(values, y, bins)
        result[method] = {}
        for metric in METRIC_NAMES:
            result[method][metric] = {
                "estimate": float(point[metric]),
                "ci95_low": float(np.percentile(draws[method][metric], 2.5)),
                "ci95_high": float(np.percentile(draws[method][metric], 97.5)),
            }
    return result


def select_operating_threshold(
    probability: np.ndarray,
    y: np.ndarray,
    family_ids: Sequence[str],
    target_risk: float,
    minimum_families: int,
) -> dict[str, Any]:
    curve = risk_coverage_curve(probability, y)
    candidates: list[dict[str, Any]] = []
    family_array = np.asarray(family_ids, dtype=object)
    for threshold, coverage, risk, accepted_count in zip(
        curve["threshold"], curve["coverage"], curve["risk"], curve["accepted"]
    ):
        accepted = probability <= threshold
        accepted_families = len(set(family_array[accepted].tolist()))
        if risk <= target_risk + 1e-15 and accepted_families >= minimum_families:
            candidates.append(
                {
                    "threshold": float(threshold),
                    "coverage": float(coverage),
                    "empirical_accepted_error_risk": float(risk),
                    "accepted_samples": int(accepted_count),
                    "accepted_families": accepted_families,
                }
            )
    if not candidates:
        return {
            "mode": "ABSTAIN_ALL",
            "threshold": None,
            "coverage": 0.0,
            "empirical_accepted_error_risk": None,
            "accepted_samples": 0,
            "accepted_families": 0,
            "qualifying_threshold_count": 0,
            "reason": f"No threshold met risk<={target_risk:.6f} and accepted_families>={minimum_families}",
        }
    candidates.sort(key=lambda row: (-row["coverage"], row["threshold"]))
    chosen = dict(candidates[0])
    chosen.update(
        {
            "mode": "RISK_THRESHOLD",
            "qualifying_threshold_count": len(candidates),
            "reason": "Maximum sample coverage; ties choose the lower predicted-risk threshold",
        }
    )
    return chosen


def bootstrap_operating_point(
    probability: np.ndarray,
    y: np.ndarray,
    family_ids: Sequence[str],
    operating: Mapping[str, Any],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if operating["mode"] == "ABSTAIN_ALL":
        return {
            "coverage_ci95": [0.0, 0.0],
            "accepted_error_risk_ci95": [None, None],
            "accepted_family_draws_ci95": [0.0, 0.0],
            "bootstrap_upper_95pct_accepted_error_risk": None,
        }
    threshold = float(operating["threshold"])
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(family_ids):
        grouped[str(family)].append(index)
    groups = [np.asarray(grouped[family], dtype=np.int64) for family in sorted(grouped)]
    rng = np.random.default_rng(seed)
    coverage = np.empty(replicates, dtype=np.float64)
    risk = np.empty(replicates, dtype=np.float64)
    accepted_family_draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[item] for item in sampled])
        accepted = probability[indices] <= threshold
        coverage[replicate] = float(accepted.mean())
        risk[replicate] = float(y[indices][accepted].mean()) if accepted.any() else float("nan")
        accepted_family_draws[replicate] = float(
            sum(bool((probability[groups[item]] <= threshold).any()) for item in sampled)
        )
    finite_risk = risk[np.isfinite(risk)]
    return {
        "coverage_ci95": [float(np.percentile(coverage, 2.5)), float(np.percentile(coverage, 97.5))],
        "accepted_error_risk_ci95": (
            [float(np.percentile(finite_risk, 2.5)), float(np.percentile(finite_risk, 97.5))]
            if len(finite_risk)
            else [None, None]
        ),
        "accepted_family_draws_ci95": [
            float(np.percentile(accepted_family_draws, 2.5)),
            float(np.percentile(accepted_family_draws, 97.5)),
        ],
        "bootstrap_upper_95pct_accepted_error_risk": (
            float(np.percentile(finite_risk, 95.0)) if len(finite_risk) else None
        ),
    }


def protected_snapshot(config: Mapping[str, Any]) -> dict[str, str]:
    architecture = read_json(ARCHITECTURE_LOCK_PATH)
    expected = {
        CHECKPOINT_PATH / "model.safetensors": config["frozen_model"]["model_sha256"],
        CHECKPOINT_PATH / "metadata.json": architecture["checkpoint_metadata_sha256"],
        CHECKPOINT_PATH / "manifest.json": architecture["checkpoint_manifest_sha256"],
        DEVELOPMENT_DATASET_INDEX: architecture["protected_inputs"]["dataset_index_sha256"],
    }
    for raw_path, digest in architecture["protected_inputs"]["wp3_locked_baseline_artifacts"].items():
        expected[WORKSPACE / raw_path] = digest
    observed: dict[str, str] = {}
    for path, digest in expected.items():
        if not path.is_file():
            raise CalibrationFitError(f"Protected upstream artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != digest:
            raise CalibrationFitError(f"Protected upstream hash changed: {relative(path)}")
        observed[relative(path)] = actual
    observed[relative(ARCHITECTURE_LOCK_PATH)] = sha256_file(ARCHITECTURE_LOCK_PATH)
    return dict(sorted(observed.items()))


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": "2026-08-24", "Creator": "pcra_u_calibration_fit.py"},
    )
    plt.close(fig)


def plot_reliability(
    path: Path,
    reliability: Sequence[Mapping[str, Any]],
    family_count: int,
    sample_count: int,
) -> None:
    labels = {
        "multimodal_raw_proxy": "Before: sigmoid(-multimodal score)",
        "spatial_platt_crossfit": "Spatial Platt: grouped cross-fit",
        "multimodal_platt_crossfit": "Multimodal Platt: grouped cross-fit",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), sharex=True, sharey=True)
    for axis, method in zip(axes, PLOT_METHODS):
        rows = [row for row in reliability if row["method"] == method and row["sample_count"] > 0]
        x = np.asarray([row["mean_predicted_risk"] for row in rows])
        y = np.asarray([row["empirical_error_rate"] for row in rows])
        count = np.asarray([row["sample_count"] for row in rows])
        axis.plot([0, 1], [0, 1], "--", color="#6b7280", linewidth=1.2, label="perfect")
        axis.plot(x, y, color="#185fa5", linewidth=1.8, alpha=0.8)
        axis.scatter(x, y, s=25 + 1.4 * np.sqrt(count), color="#185fa5", edgecolor="white", linewidth=0.7)
        for x_value, y_value, number in zip(x, y, count):
            axis.annotate(f"n={number}", (x_value, y_value), xytext=(4, 5), textcoords="offset points", fontsize=7)
        axis.set_title(labels[method], fontsize=10.5)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.18)
        axis.set_xlabel("Mean predicted execution-error risk")
    axes[0].set_ylabel("Empirical execution-error rate")
    fig.suptitle(
        "F06 — Calibration reliability (CALIBRATION_FIT_ONLY)", fontsize=14, fontweight="bold", y=1.03
    )
    fig.text(
        0.5,
        -0.015,
        f"10 equal-width bins; {family_count} independent families / {sample_count} dependent variants. "
        "Cross-fit curves are diagnostic; no Test result.",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout()
    save_figure(fig, path)


def plot_risk_coverage(
    path: Path,
    predictions: Mapping[str, np.ndarray],
    y: np.ndarray,
    operating: Mapping[str, Any],
) -> None:
    labels = {
        "multimodal_raw_proxy": "Raw multimodal proxy",
        "spatial_platt_crossfit": "Spatial Platt cross-fit",
        "multimodal_platt_crossfit": "Multimodal Platt cross-fit",
        "multimodal_platt_fullfit": "Primary full-fit (threshold fit)",
    }
    colors = {
        "multimodal_raw_proxy": "#6b7280",
        "spatial_platt_crossfit": "#d97706",
        "multimodal_platt_crossfit": "#185fa5",
        "multimodal_platt_fullfit": "#16803c",
    }
    fig, axis = plt.subplots(figsize=(8.4, 6.0))
    for method in labels:
        curve = risk_coverage_curve(predictions[method], y)
        axis.plot(curve["coverage"], curve["risk"], label=labels[method], color=colors[method], linewidth=2)
    axis.axhline(0.05, color="#c92a2a", linestyle="--", linewidth=1.5, label="target risk = 5%")
    axis.axvline(0.80, color="#7c3aed", linestyle=":", linewidth=1.5, label="reference coverage = 80%")
    if operating["mode"] == "RISK_THRESHOLD":
        axis.scatter(
            [operating["coverage"]],
            [operating["empirical_accepted_error_risk"]],
            s=90,
            marker="*",
            color="#111827",
            zorder=6,
            label=f"frozen τ ({operating['accepted_families']} families)",
        )
    else:
        axis.text(
            0.04,
            0.93,
            "Frozen decision: ABSTAIN_ALL\n(no qualifying threshold)",
            transform=axis.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff4e6", "edgecolor": "#c92a2a"},
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, max(0.12, min(1.0, float(y.mean()) * 1.15)))
    axis.set_xlabel("Accepted sample coverage")
    axis.set_ylabel("Observed error risk among accepted samples")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8.5, loc="upper left")
    axis.set_title("F07 — Selective risk–coverage (CALIBRATION_FIT_ONLY)", fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Curves use Calibration variants; confidence intervals are family-clustered in the accompanying table. No Test claim.",
        ha="center",
        fontsize=8.5,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, path)


def plot_distributions(
    path: Path,
    probability: np.ndarray,
    y: np.ndarray,
    states: Sequence[str],
    operating: Mapping[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    bins = np.linspace(0, 1, 21)
    axes[0].hist(probability[y == 0], bins=bins, alpha=0.72, color="#16803c", label=f"safe MAP (n={int((y == 0).sum())})")
    axes[0].hist(probability[y == 1], bins=bins, alpha=0.63, color="#c92a2a", label=f"unsafe/error (n={int((y == 1).sum())})")
    if operating["threshold"] is not None:
        axes[0].axvline(float(operating["threshold"]), color="#111827", linestyle="--", linewidth=2, label="frozen τ")
    axes[0].set_xlabel("Primary full-fit multimodal predicted error risk")
    axes[0].set_ylabel("Variant count")
    axes[0].set_title("A. Risk separation for the locked execution event")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.18)

    order = ANSWERABILITY_CLASSES
    values = [probability[np.asarray(states) == state] for state in order]
    box = axes[1].boxplot(
        values,
        tick_labels=[state.replace("INSUFFICIENT_EVIDENCE", "INSUFFICIENT") for state in order],
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], ["#16803c", "#7c3aed", "#c92a2a", "#d97706"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)
    if operating["threshold"] is not None:
        axes[1].axhline(float(operating["threshold"]), color="#111827", linestyle="--", linewidth=2)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Primary full-fit multimodal predicted error risk")
    axes[1].set_title("B. Risk by evaluator answerability state")
    axes[1].tick_params(axis="x", labelrotation=18, labelsize=8)
    axes[1].grid(axis="y", alpha=0.18)
    fig.suptitle("F08 — Calibration risk distribution and frozen threshold", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "CALIBRATION_FIT_ONLY. Full-fit distributions are descriptive and optimistic; locked Test remains unopened.",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(fig, path)


def artifact_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "schema_version": 1,
        "file_count": len(rows),
        "files": rows,
        "tree_commitment_sha256": canonical_sha256(rows),
    }


def report_markdown(report: Mapping[str, Any]) -> str:
    primary = report["metrics"]["multimodal_platt_crossfit"]
    spatial = report["metrics"]["spatial_platt_crossfit"]
    operating = report["operating_point"]
    lines = [
        "# P-CRA-U calibration fit report",
        "",
        "- Status khoa học: `CALIBRATION_FIT_ONLY`; đây không phải kết quả tổng quát hóa trên Test.",
        f"- Gate: `{report['gate_status']}`; quyết định vận hành: `{operating['mode']}`.",
        f"- Dữ liệu: `{report['data']['family_count']}` family độc lập / `{report['data']['sample_count']}` variants phụ thuộc.",
        "- P-CRA-U và RoboRefer hoàn toàn đóng băng; chỉ hai mô hình Platt một biến được fit.",
        "- Test-IID/Test-OOD không được mở hay đọc.",
        "",
        "## Chẩn đoán cross-fit chính",
        "",
        "Năm fold được chia xác định theo family. Full-fit được lưu để dùng về sau; cross-fit dưới đây dùng để giảm lạc quan khi mô tả Calibration.",
        "",
        "| Calibrator | Brier ↓ | NLL ↓ | ECE ↓ | AURC ↓ | Risk@80% ↓ | Coverage@5% ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Spatial Platt cross-fit | {spatial['brier']['estimate']:.4f} | {spatial['nll']['estimate']:.4f} | {spatial['ece_10bin']['estimate']:.4f} | {spatial['aurc']['estimate']:.4f} | {spatial['risk_at_80pct_coverage']['estimate']:.4f} | {spatial['coverage_at_5pct_risk']['estimate']:.4f} |",
        f"| Multimodal Platt cross-fit | {primary['brier']['estimate']:.4f} | {primary['nll']['estimate']:.4f} | {primary['ece_10bin']['estimate']:.4f} | {primary['aurc']['estimate']:.4f} | {primary['risk_at_80pct_coverage']['estimate']:.4f} | {primary['coverage_at_5pct_risk']['estimate']:.4f} |",
        "",
        "Mọi khoảng tin cậy trong CSV được bootstrap 5.000 lần theo family, giữ nguyên năm variants của family được lấy mẫu.",
        "",
        "## Operating point đã khóa",
        "",
    ]
    if operating["mode"] == "RISK_THRESHOLD":
        lines += [
            f"- `tau = {operating['threshold']:.10g}` trên `multimodal_platt` full-fit.",
            f"- Coverage Calibration: `{operating['accepted_samples']}/{report['data']['sample_count']} = {operating['coverage']:.4f}`.",
            f"- Accepted families: `{operating['accepted_families']}/{report['data']['family_count']}`.",
            f"- Empirical accepted error risk: `{operating['empirical_accepted_error_risk']:.4f}` (mục tiêu ≤ 0.05).",
            f"- Bootstrap upper 95th percentile của accepted risk: `{operating['bootstrap']['bootstrap_upper_95pct_accepted_error_risk']:.4f}`. Giá trị này chỉ báo cáo, không thay thế rule đã khóa.",
        ]
    else:
        lines += [
            "- Không threshold nào đồng thời đạt empirical risk ≤ 5% và ít nhất 60 accepted family.",
            "- Kết quả trung thực được khóa là `ABSTAIN_ALL`, coverage bằng 0; không nới rule sau khi xem số liệu.",
        ]
    multimodal_better = (
        primary["brier"]["estimate"] < spatial["brier"]["estimate"]
        and primary["aurc"]["estimate"] < spatial["aurc"]["estimate"]
    )
    lines += [
        "",
        "## Diễn giải trung thực",
        "",
        (
            "Trên cross-fit Calibration, multimodal có point estimate Brier và AURC cùng thấp hơn spatial. Đây vẫn chưa phải bằng chứng Test."
            if multimodal_better
            else "Trên cross-fit Calibration, multimodal không đồng thời cải thiện cả Brier và AURC so với spatial; không tuyên bố multimodal tốt hơn."
        ),
        "",
        "AURC dùng tích phân hình thang trên risk–coverage curve theo các threshold risk duy nhất. Risk/coverage là theo variants; CI cluster theo family và threshold yêu cầu ít nhất 60 family độc lập.",
        "",
        "## Ranh giới kết quả",
        "",
        "- Chỉ Bảng 4 dạng `CALIBRATION_FIT_ONLY` được tạo.",
        "- Bảng 2, 3, 5 và 6 vẫn `NOT_RUN`.",
        "- Sau freeze PASS, hành động duy nhất được phép là capture và mở locked Test-IID/Test-OOD một lần.",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    eval_path = Path(args.eval_manifest).resolve()
    predictions_path = Path(args.raw_predictions).resolve()
    attestation_path = Path(args.raw_attestation).resolve()
    result_root = Path(args.output_root).resolve()
    freeze_path = Path(args.freeze_lock).resolve()
    if freeze_path.exists():
        existing = read_json(freeze_path)
        if existing.get("status") == "FROZEN":
            raise CalibrationFitError(f"Refusing to overwrite frozen calibrator lock: {freeze_path}")

    config = read_json(config_path)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise CalibrationFitError("Calibration config protocol_id mismatch")
    if not CONTRACT_PATH.is_file():
        raise CalibrationFitError("Calibration contract is missing")
    if config["safety"] != {
        "model_training": False,
        "checkpoint_selection": False,
        "architecture_selection": False,
        "calibration_fit": True,
        "test_iid_opened": False,
        "test_ood_opened": False,
        "robot_publish": False,
        "baseline_modification": False,
    }:
        raise CalibrationFitError("Calibration safety block differs from the locked policy")

    protected_before = protected_snapshot(config)
    raw_rows, raw_attestation = validate_raw_inputs(config, predictions_path, attestation_path)
    joined, eval_manifest, join_audit = join_evaluator(config, raw_rows, eval_path)
    y = np.asarray([row["y_error"] for row in joined], dtype=np.float64)
    family_ids = [str(row["family_id"]) for row in joined]
    spatial = np.asarray([row["spatial_raw_score"] for row in joined], dtype=np.float64)
    multimodal = np.asarray([row["multimodal_raw_score"] for row in joined], dtype=np.float64)

    calibrator_config = config["calibrators"]
    spatial_fit = fit_platt(
        spatial,
        y,
        float(calibrator_config["spatial_platt"]["l2_slope"]),
        int(calibrator_config["spatial_platt"]["max_iterations"]),
    )
    multimodal_fit = fit_platt(
        multimodal,
        y,
        float(calibrator_config["multimodal_platt"]["l2_slope"]),
        int(calibrator_config["multimodal_platt"]["max_iterations"]),
    )
    diagnostics = config["diagnostics"]
    folds = int(diagnostics["grouped_crossfit_folds"])
    crossfit_seed = int(config["seeds"]["grouped_crossfit"])
    spatial_crossfit, spatial_fold_rows = crossfit_platt(
        spatial,
        y,
        family_ids,
        crossfit_seed,
        folds,
        float(calibrator_config["spatial_platt"]["l2_slope"]),
        int(calibrator_config["spatial_platt"]["max_iterations"]),
    )
    multimodal_crossfit, multimodal_fold_rows = crossfit_platt(
        multimodal,
        y,
        family_ids,
        crossfit_seed,
        folds,
        float(calibrator_config["multimodal_platt"]["l2_slope"]),
        int(calibrator_config["multimodal_platt"]["max_iterations"]),
    )
    predictions = {
        "spatial_raw_proxy": np.asarray(expit(-spatial), dtype=np.float64),
        "spatial_platt_fullfit": apply_platt(spatial, spatial_fit),
        "spatial_platt_crossfit": spatial_crossfit,
        "multimodal_raw_proxy": np.asarray(expit(-multimodal), dtype=np.float64),
        "multimodal_platt_fullfit": apply_platt(multimodal, multimodal_fit),
        "multimodal_platt_crossfit": multimodal_crossfit,
    }
    bootstrap_replicates = int(diagnostics["bootstrap_replicates"])
    bins = int(diagnostics["ece_bins"])
    metric_intervals = grouped_bootstrap_intervals(
        predictions,
        y,
        family_ids,
        bootstrap_replicates,
        int(config["seeds"]["family_bootstrap"]),
        bins,
    )

    threshold_config = config["threshold_policy"]
    operating = select_operating_threshold(
        predictions["multimodal_platt_fullfit"],
        y,
        family_ids,
        float(threshold_config["target_accepted_error_risk"]),
        int(threshold_config["minimum_accepted_families"]),
    )
    operating["bootstrap"] = bootstrap_operating_point(
        predictions["multimodal_platt_fullfit"],
        y,
        family_ids,
        operating,
        bootstrap_replicates,
        int(config["seeds"]["family_bootstrap"]),
    )

    reliability: list[dict[str, Any]] = []
    for method, values in predictions.items():
        reliability.extend(reliability_rows(method, values, y, family_ids, bins))
    table_rows: list[dict[str, Any]] = []
    method_meta = {
        "spatial_raw_proxy": ("No", "raw_proxy", "sigmoid(-spatial_raw_score); descriptive only"),
        "spatial_platt_fullfit": ("Platt", "full_fit", "optimistic in-sample fit"),
        "spatial_platt_crossfit": ("Platt", "family_grouped_5fold_crossfit", "diagnostic"),
        "multimodal_raw_proxy": ("No", "raw_proxy", "sigmoid(-multimodal_raw_score); descriptive only"),
        "multimodal_platt_fullfit": ("Platt", "full_fit", "frozen deployable calibrator; in-sample diagnostic"),
        "multimodal_platt_crossfit": ("Platt", "family_grouped_5fold_crossfit", "primary diagnostic"),
    }
    for method, metric_values in metric_intervals.items():
        calibration, scope, note = method_meta[method]
        row: dict[str, Any] = {
            "scientific_status": SCIENTIFIC_STATUS,
            "risk_model": method,
            "calibration": calibration,
            "prediction_scope": scope,
            "families": int(config["dataset"]["family_count"]),
            "samples": int(config["dataset"]["sample_count"]),
            "error_count": int(y.sum()),
            "note": note,
        }
        for metric in METRIC_NAMES:
            row[metric] = metric_values[metric]["estimate"]
            row[f"{metric}_ci95_low"] = metric_values[metric]["ci95_low"]
            row[f"{metric}_ci95_high"] = metric_values[metric]["ci95_high"]
        table_rows.append(row)

    table_root = result_root / "tables"
    figure_root = result_root / "figures"
    calibrator_root = result_root / "calibrators"
    checks_root = result_root / "checks"
    table_fields = [
        "scientific_status",
        "risk_model",
        "calibration",
        "prediction_scope",
        "families",
        "samples",
        "error_count",
    ]
    for metric in METRIC_NAMES:
        table_fields.extend([metric, f"{metric}_ci95_low", f"{metric}_ci95_high"])
    table_fields.append("note")
    write_csv(table_root / "table_04_calibration_fit_only.csv", table_rows, table_fields)
    write_csv(
        table_root / "table_calibration_reliability_bins.csv",
        reliability,
        [
            "scientific_status",
            "method",
            "bin_index",
            "risk_lower",
            "risk_upper",
            "sample_count",
            "family_count",
            "mean_predicted_risk",
            "empirical_error_rate",
        ],
    )
    fold_table: list[dict[str, Any]] = []
    for calibrator, rows in (("spatial_platt", spatial_fold_rows), ("multimodal_platt", multimodal_fold_rows)):
        for row in rows:
            fold_table.append({"scientific_status": SCIENTIFIC_STATUS, "calibrator": calibrator, **row})
    write_csv(
        table_root / "table_calibration_grouped_crossfit_folds.csv",
        fold_table,
        [
            "scientific_status",
            "calibrator",
            "fold",
            "train_families",
            "held_out_families",
            "train_samples",
            "held_out_samples",
            "held_out_errors",
            "slope",
            "intercept",
            "optimizer_iterations",
        ],
    )
    state_rows = []
    states = [row["answerability_state"] for row in joined]
    for state in ANSWERABILITY_CLASSES:
        selected = np.asarray(states) == state
        state_rows.append(
            {
                "scientific_status": SCIENTIFIC_STATUS,
                "answerability_state": state,
                "samples": int(selected.sum()),
                "families": len({family_ids[index] for index in np.flatnonzero(selected)}),
                "errors": int(y[selected].sum()),
                "mean_primary_fullfit_risk": float(predictions["multimodal_platt_fullfit"][selected].mean()),
                "median_primary_fullfit_risk": float(np.median(predictions["multimodal_platt_fullfit"][selected])),
            }
        )
    write_csv(
        table_root / "table_calibration_risk_by_answerability_state.csv",
        state_rows,
        [
            "scientific_status",
            "answerability_state",
            "samples",
            "families",
            "errors",
            "mean_primary_fullfit_risk",
            "median_primary_fullfit_risk",
        ],
    )
    operating_row = {
        "scientific_status": SCIENTIFIC_STATUS,
        "calibrator": "multimodal_platt_fullfit",
        "mode": operating["mode"],
        "threshold": operating["threshold"],
        "target_error_risk": threshold_config["target_accepted_error_risk"],
        "minimum_accepted_families": threshold_config["minimum_accepted_families"],
        "coverage": operating["coverage"],
        "accepted_samples": operating["accepted_samples"],
        "accepted_families": operating["accepted_families"],
        "empirical_accepted_error_risk": operating["empirical_accepted_error_risk"],
        "coverage_ci95_low": operating["bootstrap"]["coverage_ci95"][0],
        "coverage_ci95_high": operating["bootstrap"]["coverage_ci95"][1],
        "risk_ci95_low": operating["bootstrap"]["accepted_error_risk_ci95"][0],
        "risk_ci95_high": operating["bootstrap"]["accepted_error_risk_ci95"][1],
        "risk_bootstrap_upper_95pct": operating["bootstrap"]["bootstrap_upper_95pct_accepted_error_risk"],
        "reason": operating["reason"],
    }
    write_csv(
        table_root / "table_calibration_operating_point.csv",
        [operating_row],
        list(operating_row),
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "pcra_u_calibration_v1_20260824",
        }
    )
    plot_reliability(
        figure_root / "F06_calibration_reliability",
        reliability,
        int(config["dataset"]["family_count"]),
        int(config["dataset"]["sample_count"]),
    )
    plot_risk_coverage(figure_root / "F07_calibration_risk_coverage", predictions, y, operating)
    plot_distributions(
        figure_root / "F08_calibration_risk_distribution_threshold",
        predictions["multimodal_platt_fullfit"],
        y,
        states,
        operating,
    )

    calibrator_payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "target": "y_error = NOT(truth FOUND AND MAP point inside target mask)",
        "input_score_direction": "higher raw score is stronger evidence; Platt slope is fit without sign constraint",
        "spatial_platt": spatial_fit,
        "multimodal_platt": {**multimodal_fit, "primary": True},
        "operating_point": operating,
        "frozen_model_sha256": config["frozen_model"]["model_sha256"],
        "checkpoint_path": config["frozen_model"]["checkpoint"],
        "raw_predictions_sha256": sha256_file(predictions_path),
        "eval_manifest_sha256": sha256_file(eval_path),
        "config_sha256": sha256_file(config_path),
        "fit_code_sha256": sha256_file(Path(__file__).resolve()),
        "test_iid_opened": False,
        "test_ood_opened": False,
    }
    calibrator_path = calibrator_root / "pcra_u_calibrator_v1.json"
    write_json(calibrator_path, calibrator_payload)

    protected_after = protected_snapshot(config)
    gates = {
        "raw_inference_count_and_hash_attested": len(raw_rows) == int(config["dataset"]["sample_count"]),
        "raw_inference_oracle_free": nested_flag(raw_attestation, "oracle_or_annotation_read") is False,
        "strict_raw_evaluator_sample_join": join_audit["strict_join_by_sample_id"],
        "calibration_family_and_sample_counts_exact": (
            join_audit["family_count"] == int(config["dataset"]["family_count"])
            and join_audit["sample_count"] == int(config["dataset"]["sample_count"])
        ),
        "four_primary_strata_match_lock": join_audit["primary_family_strata"]
        == dict(sorted(config["dataset"]["primary_family_strata"].items())),
        "map_coordinates_in_bounds": join_audit["coordinate_out_of_bounds_count"] == 0,
        "both_outcome_classes_present": set(np.unique(y).tolist()) == {0.0, 1.0},
        "platt_fits_converged": bool(spatial_fit["converged"] and multimodal_fit["converged"]),
        "grouped_crossfit_complete": bool(np.isfinite(spatial_crossfit).all() and np.isfinite(multimodal_crossfit).all()),
        "bootstrap_5000_family_clustered": bootstrap_replicates == 5000,
        "primary_calibrator_is_multimodal_platt": bool(calibrator_config["multimodal_platt"].get("primary")),
        "threshold_rule_applied_without_relaxation": operating["mode"] in {"RISK_THRESHOLD", "ABSTAIN_ALL"},
        "upstream_model_dataset_baseline_unchanged": protected_before == protected_after,
        "test_iid_and_test_ood_unopened": True,
        "model_training_and_checkpoint_selection_not_performed": True,
    }
    gate_status = "PASS" if all(gates.values()) else "FAIL"
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "created_at_utc": utc_now(),
        "gate_status": gate_status,
        "decision": "FREEZE_CALIBRATOR_AND_THRESHOLD" if gate_status == "PASS" else "FIX_CALIBRATION_PIPELINE_FIRST",
        "data": {
            **join_audit,
            "dataset_root": relative(resolve_workspace_path(str(eval_manifest.get("dataset_root", config["dataset"]["root"])))),
        },
        "event_definition": "y_error = NOT(truth FOUND AND frozen P-CRA-U MAP point inside evaluator target mask)",
        "calibrators": {"spatial_platt": spatial_fit, "multimodal_platt": multimodal_fit},
        "metrics": metric_intervals,
        "metric_definitions": {
            "ece": "10 equal-width predicted-risk bins, sample weighted",
            "aurc": "tie-aware trapezoidal integral over unique-threshold risk-coverage curve including origin",
            "risk_at_80pct_coverage": "first unique-risk threshold reaching at least 80% sample coverage",
            "coverage_at_5pct_risk": "maximum sample coverage among unique-risk thresholds with empirical risk <= 5%",
            "confidence_intervals": "percentile 95% CI from 5000 family-cluster bootstrap resamples retaining all variants",
        },
        "operating_point": operating,
        "gates": gates,
        "provenance": {
            "contract": {"path": relative(CONTRACT_PATH), "sha256": sha256_file(CONTRACT_PATH)},
            "config": {"path": relative(config_path), "sha256": sha256_file(config_path)},
            "eval_manifest": {"path": relative(eval_path), "sha256": sha256_file(eval_path)},
            "raw_predictions": {"path": relative(predictions_path), "sha256": sha256_file(predictions_path)},
            "raw_inference_attestation": {"path": relative(attestation_path), "sha256": sha256_file(attestation_path)},
            "fit_code": {"path": relative(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "calibrator": {"path": relative(calibrator_path), "sha256": sha256_file(calibrator_path)},
            "protected_before": protected_before,
            "protected_after": protected_after,
        },
        "reporting_boundary": {
            "table_04": "CALIBRATION_FIT_ONLY",
            "tables_02_03_05_06": "NOT_RUN",
            "final_generalization_claim": False,
            "test_iid_opened": False,
            "test_ood_opened": False,
        },
    }
    report_json_path = result_root / "PCRA_U_CALIBRATION_REPORT.json"
    report_md_path = result_root / "PCRA_U_CALIBRATION_REPORT.md"
    write_json(report_json_path, report)
    write_text(report_md_path, report_markdown(report))
    fit_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": gate_status,
        "phase": "EVALUATOR_JOIN_AND_CALIBRATOR_FIT_ONLY",
        "inputs": report["provenance"],
        "outputs": {
            "calibrator": {"path": relative(calibrator_path), "sha256": sha256_file(calibrator_path)},
            "report_json": {"path": relative(report_json_path), "sha256": sha256_file(report_json_path)},
            "report_md": {"path": relative(report_md_path), "sha256": sha256_file(report_md_path)},
            "tables": [relative(path) for path in sorted(table_root.glob("*.csv"))],
            "figures": [relative(path) for path in sorted(figure_root.glob("F0[678]_calibration_*.*"))],
        },
        "safety": {
            "model_loaded": False,
            "optimizer_created": False,
            "model_training": False,
            "checkpoint_selection": False,
            "test_inputs_read": False,
            "evaluator_opened_only_after_raw_inference_attestation": True,
        },
        "gates": gates,
    }
    write_json(checks_root / "calibration_fit_manifest.json", fit_manifest)
    write_json(checks_root / "calibration_fit_gate.json", {"status": gate_status, "gates": gates})
    manifest = artifact_manifest(result_root)
    write_json(checks_root / "artifact_manifest.json", manifest)

    if gate_status == "PASS":
        freeze_lock = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "FROZEN",
            "decision": "FREEZE_CALIBRATOR_AND_THRESHOLD",
            "scientific_status": SCIENTIFIC_STATUS,
            "created_at_utc": report["created_at_utc"],
            "allowed_next_action": "CAPTURE_AND_OPEN_LOCKED_TEST_IID_OOD",
            "forbidden_actions": [
                "MODEL_RETRAIN",
                "ARCHITECTURE_CHANGE",
                "CHECKPOINT_RESELECTION",
                "CALIBRATOR_REFIT_AFTER_TEST",
                "THRESHOLD_CHANGE_AFTER_TEST",
            ],
            "frozen_checkpoint": {
                "path": config["frozen_model"]["checkpoint"],
                "model_sha256": config["frozen_model"]["model_sha256"],
                "epoch": config["frozen_model"]["epoch"],
                "step": config["frozen_model"]["step"],
            },
            "frozen_calibrator": {
                "path": relative(calibrator_path),
                "sha256": sha256_file(calibrator_path),
                "primary": "multimodal_platt",
                "spatial_coefficients": {
                    "slope": spatial_fit["slope"],
                    "intercept": spatial_fit["intercept"],
                },
                "multimodal_coefficients": {
                    "slope": multimodal_fit["slope"],
                    "intercept": multimodal_fit["intercept"],
                },
            },
            "frozen_operating_point": operating,
            "target_policy": {
                "accepted_error_risk": threshold_config["target_accepted_error_risk"],
                "minimum_accepted_families": threshold_config["minimum_accepted_families"],
                "fallback": threshold_config["fallback"],
            },
            "bound_inputs": {
                "architecture_freeze_lock_sha256": sha256_file(ARCHITECTURE_LOCK_PATH),
                "checkpoint_model_sha256": config["frozen_model"]["model_sha256"],
                "calibration_contract_sha256": sha256_file(CONTRACT_PATH),
                "calibration_config_sha256": sha256_file(config_path),
                "calibration_eval_manifest_sha256": sha256_file(eval_path),
                "raw_predictions_sha256": sha256_file(predictions_path),
                "raw_inference_manifest_sha256": sha256_file(attestation_path),
                "fit_code_sha256": sha256_file(Path(__file__).resolve()),
                "calibration_report_sha256": sha256_file(report_json_path),
                "result_artifact_tree_sha256": manifest["tree_commitment_sha256"],
            },
            "protected_upstream": protected_after,
            "reporting_boundary": report["reporting_boundary"],
            "tests_opened_during_calibration": False,
        }
        write_json(freeze_path, freeze_lock)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--eval-manifest", default=str(EVAL_MANIFEST_PATH))
    parser.add_argument("--raw-predictions", default=str(RAW_PREDICTIONS_PATH))
    parser.add_argument("--raw-attestation", default=str(RAW_ATTESTATION_PATH))
    parser.add_argument("--output-root", default=str(RESULT_ROOT))
    parser.add_argument("--freeze-lock", default=str(FREEZE_LOCK_PATH))
    return parser.parse_args()


def main() -> int:
    try:
        report = run(parse_args())
    except CalibrationFitError as exc:
        print(f"CALIBRATION FIT BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["gate_status"], "decision": report["decision"]}, sort_keys=True))
    return 0 if report["gate_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
