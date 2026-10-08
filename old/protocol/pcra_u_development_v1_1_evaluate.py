#!/usr/bin/env python3
"""Conditional exploratory DEV comparison of P-CRA-U V1.1 against locked V1.

This is a post-campaign, read-only-input evaluator.  It never loads Calibration,
Test-IID, or Test-OOD.  It consumes the three development-seed winner reports
and predictions, verifies the locked V1 epoch-10 comparator, performs the
precommitted paired family bootstrap, and writes only to a new results root.

Example (run only after all three campaign seeds have completed)::

    python protocol/pcra_u_development_v1_1_evaluate.py \
      --campaign-report results/pcra_u_runs/<campaign>/PCRA_U_DEVELOPMENT_V1_1_CAMPAIGN_REPORT.json \
      --output-root results/pcra_u_runs/<new_evaluation_root>

The resulting confidence intervals are conditional exploratory DEV intervals:
DEV was also used to select checkpoints, so these are not confirmatory test
intervals and cannot authorize opening a test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
RESULTS_ROOT = WORKSPACE / "results"

V1_REPORT = WORKSPACE / "results/pcra_u_runs/pcra_u_development_train_20260824/PCRA_U_DEVELOPMENT_TRAIN_REPORT.json"
V1_PREDICTIONS = (
    WORKSPACE
    / "results/pcra_u_runs/pcra_u_development_train_20260824/predictions/dev_predictions_exploratory.json"
)
V1_CHECKPOINT = (
    WORKSPACE
    / "results/pcra_u_runs/pcra_u_development_train_20260824/checkpoints/development/step_000002000"
)
V1_MODEL_SHA256 = "14dba40d2de437f082aece9b342931a48c58202ae4242d577d4a4ffd7eecaab3"
V1_PREDICTIONS_SHA256 = "3f6c68dc50611205cd4213911b8b59822cefd85487780f610c6ae409720d22d2"

EXPECTED_SEEDS = (24082026, 24082027, 24082028)
BOOTSTRAP_SEED = 24082029
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_FAMILIES = 80
EXPECTED_SAMPLES = 400
EXPECTED_VARIANTS = {
    "clean",
    "semantic_counterfactual",
    "relation_counterfactual",
    "depth_corruption",
    "occlusion_view_counterfactual",
}
ANSWERABILITY_CLASSES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")
SOURCE_CLASSES = ("semantic", "relation", "depth", "occlusion")

# Locked V1.1 scientific checkpoint score.  Source is deliberately secondary.
SCORE_WEIGHTS = {"Gc": 0.45, "Ga": 0.20, "A": 0.20, "one_minus_F": 0.15}
NONINFERIORITY_MARGIN = 0.02
V1_CLEAN_FOUND_REFERENCE = 0.53125
MIN_PER_SEED_CLEAN_FOUND = 0.50
MIN_ELIGIBLE_GA = 0.8371428571
MIN_ELIGIBLE_A = 0.7220238435
MAX_ELIGIBLE_F = 0.1147368421
FLOAT_TOLERANCE = 1e-12
SCIENTIFIC_STATUS = "EXPLORATORY_DEV_ONLY_CONDITIONAL_ON_CHECKPOINT_SELECTION"


class V11EvaluationError(RuntimeError):
    """Fail-closed post-campaign evaluation error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V11EvaluationError(f"Cannot read valid JSON: {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def workspace_path(raw: str | Path, *, must_exist: bool = True) -> Path:
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (WORKSPACE / path).resolve()
    try:
        resolved.relative_to(WORKSPACE.resolve())
    except ValueError as exc:
        raise V11EvaluationError(f"Path must remain inside workspace: {raw}") from exc
    if must_exist and not resolved.exists():
        raise V11EvaluationError(f"Required path does not exist: {resolved}")
    return resolved


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE.resolve()))


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise V11EvaluationError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise V11EvaluationError(f"{label} is not finite: {result}")
    return result


def close(left: float, right: float, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def prediction_rows(document: Any, label: str) -> list[dict[str, Any]]:
    rows = document.get("predictions") if isinstance(document, Mapping) else document
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise V11EvaluationError(f"{label} must contain a prediction list")
    return [dict(row) for row in rows]


def answerability_metrics(rows: Sequence[Mapping[str, Any]]) -> tuple[float, list[list[int]]]:
    class_index = {name: index for index, name in enumerate(ANSWERABILITY_CLASSES)}
    matrix = [[0 for _ in ANSWERABILITY_CLASSES] for _ in ANSWERABILITY_CLASSES]
    for row in rows:
        truth = str(row.get("answerability_truth"))
        prediction = str(row.get("answerability_prediction"))
        if truth not in class_index or prediction not in class_index:
            raise V11EvaluationError(f"Invalid answerability label in {row.get('sample_id')}: {truth}/{prediction}")
        matrix[class_index[truth]][class_index[prediction]] += 1
    f1_values = []
    for index in range(len(ANSWERABILITY_CLASSES)):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(matrix)) if row != index)
        false_negative = sum(matrix[index][column] for column in range(len(matrix)) if column != index)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    return statistics.fmean(f1_values), matrix


def validate_prediction_set(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    if len(rows) != EXPECTED_SAMPLES:
        raise V11EvaluationError(f"{label}: expected 400 predictions, observed {len(rows)}")
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if len(set(sample_ids)) != len(sample_ids) or any(value in {"", "None"} for value in sample_ids):
        raise V11EvaluationError(f"{label}: sample IDs are missing or duplicated")
    by_family: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row.get("family_id"))
        variant = str(row.get("variant"))
        if family in {"", "None"} or variant not in EXPECTED_VARIANTS:
            raise V11EvaluationError(f"{label}: invalid family/variant for {row.get('sample_id')}")
        by_family[family].append(row)
        truth = str(row.get("answerability_truth"))
        point = row.get("point_in_target")
        if truth == "FOUND" and not isinstance(point, bool):
            raise V11EvaluationError(f"{label}: FOUND row lacks boolean raw point-in-target: {row.get('sample_id')}")
        if truth != "FOUND" and point is not None:
            raise V11EvaluationError(f"{label}: non-FOUND row has grounding oracle value: {row.get('sample_id')}")
        for field in ("source_truth", "source_prediction"):
            labels = row.get(field)
            if not isinstance(labels, list) or len(labels) != len(set(labels)):
                raise V11EvaluationError(f"{label}: invalid {field} in {row.get('sample_id')}")
            if not set(labels).issubset(SOURCE_CLASSES):
                raise V11EvaluationError(f"{label}: unknown {field} label in {row.get('sample_id')}")
    if len(by_family) != EXPECTED_FAMILIES:
        raise V11EvaluationError(f"{label}: expected 80 families, observed {len(by_family)}")
    for family, family_rows in by_family.items():
        variants = {str(row["variant"]) for row in family_rows}
        if len(family_rows) != 5 or variants != EXPECTED_VARIANTS:
            raise V11EvaluationError(f"{label}: family {family} does not retain exactly all five variants")

    macro_f1, confusion = answerability_metrics(rows)
    clean_found = [row for row in rows if row["variant"] == "clean" and row["answerability_truth"] == "FOUND"]
    all_found = [row for row in rows if row["answerability_truth"] == "FOUND"]
    risky = [row for row in rows if row["answerability_truth"] in {"AMBIGUOUS", "ABSENT"}]
    gc_correct = sum(row["point_in_target"] is True for row in clean_found)
    ga_correct = sum(row["point_in_target"] is True for row in all_found)
    false_found = sum(row["answerability_prediction"] == "FOUND" for row in risky)
    source_tp = source_fp = source_fn = 0
    for row in rows:
        truth = set(row["source_truth"])
        prediction = set(row["source_prediction"])
        source_tp += len(truth & prediction)
        source_fp += len(prediction - truth)
        source_fn += len(truth - prediction)
    source_denominator = 2 * source_tp + source_fp + source_fn
    if not clean_found or not all_found or not risky:
        raise V11EvaluationError(f"{label}: locked metric denominator is zero")
    metrics = {
        "Gc": gc_correct / len(clean_found),
        "Ga": ga_correct / len(all_found),
        "A": macro_f1,
        "F": false_found / len(risky),
        "S": 2 * source_tp / source_denominator if source_denominator else 1.0,
        "counts": {
            "Gc_correct": gc_correct,
            "Gc_denominator": len(clean_found),
            "Ga_correct": ga_correct,
            "Ga_denominator": len(all_found),
            "false_found_count": false_found,
            "false_found_denominator": len(risky),
            "source_tp": source_tp,
            "source_fp": source_fp,
            "source_fn": source_fn,
        },
        "answerability_confusion_matrix": confusion,
    }
    metrics["C"] = scientific_score(metrics)
    return metrics


def scientific_score(metrics: Mapping[str, Any]) -> float:
    return (
        SCORE_WEIGHTS["Gc"] * finite_float(metrics["Gc"], "Gc")
        + SCORE_WEIGHTS["Ga"] * finite_float(metrics["Ga"], "Ga")
        + SCORE_WEIGHTS["A"] * finite_float(metrics["A"], "A")
        + SCORE_WEIGHTS["one_minus_F"] * (1.0 - finite_float(metrics["F"], "F"))
    )


def prediction_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[Any, ...]]:
    return {
        str(row["sample_id"]): (
            str(row["family_id"]),
            str(row["variant"]),
            str(row["answerability_truth"]),
            tuple(sorted(str(value) for value in row["source_truth"])),
        )
        for row in rows
    }


def validate_same_locked_dev(reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]], label: str) -> None:
    if prediction_identity(reference) != prediction_identity(candidate):
        raise V11EvaluationError(f"{label}: prediction identities/truth differ from locked V1 DEV")


def nested_metric(report: Mapping[str, Any], path: Sequence[str]) -> float | None:
    value: Any = report
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def verify_reported_metrics(report: Mapping[str, Any], measured: Mapping[str, Any], label: str) -> None:
    paths = {
        "Gc": ("selected_checkpoint_dev_metrics", "clean_found", "raw_point_in_target"),
        "Ga": ("selected_checkpoint_dev_metrics", "grounding_found", "point_in_target"),
        "A": ("selected_checkpoint_dev_metrics", "answerability", "macro_f1"),
        "F": ("selected_checkpoint_dev_metrics", "false_found_on_ambiguous_or_absent", "rate"),
        "S": ("selected_checkpoint_dev_metrics", "source_micro_f1"),
    }
    for metric, path in paths.items():
        reported = nested_metric(report, path)
        if reported is not None and not close(reported, float(measured[metric])):
            raise V11EvaluationError(
                f"{label}: reported {metric}={reported} disagrees with predictions={measured[metric]}"
            )


def extract_reported_eligibility(report: Mapping[str, Any], label: str) -> tuple[bool, Mapping[str, Any]]:
    selection = report.get("selection")
    if not isinstance(selection, Mapping) or not isinstance(selection.get("eligible"), bool):
        raise V11EvaluationError(f"{label}: missing required boolean selection.eligible")
    return bool(selection["eligible"]), selection


def all_boolean_gates_pass(report: Mapping[str, Any]) -> bool:
    gates = report.get("gates")
    if not isinstance(gates, Mapping):
        return False
    values = [value for value in gates.values() if isinstance(value, bool)]
    return bool(values) and all(values)


def selected_checkpoint_path(report: Mapping[str, Any], run_root: Path, label: str) -> Path:
    raw = report.get("selected_checkpoint")
    if not isinstance(raw, str) or not raw:
        raise V11EvaluationError(f"{label}: selected_checkpoint is missing")
    candidate = workspace_path(raw)
    try:
        candidate.relative_to(run_root)
    except ValueError as exc:
        raise V11EvaluationError(f"{label}: selected checkpoint is outside its seed run root") from exc
    if not candidate.is_dir() or not (candidate / "model.safetensors").is_file():
        raise V11EvaluationError(f"{label}: selected checkpoint has no model.safetensors")
    return candidate


def campaign_seed_entries(campaign: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = campaign.get("seed_reports")
    if not isinstance(entries, list) or not all(isinstance(entry, Mapping) for entry in entries):
        raise V11EvaluationError("Campaign report must contain seed_reports[]")
    observed = sorted(int(entry.get("seed")) for entry in entries)
    if observed != list(EXPECTED_SEEDS):
        raise V11EvaluationError(f"Campaign must contain exactly seeds {EXPECTED_SEEDS}; observed {observed}")
    return sorted(entries, key=lambda entry: int(entry["seed"]))


def load_winner(
    campaign_entry: Mapping[str, Any],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(campaign_entry["seed"])
    run_root_raw = campaign_entry.get("run_root")
    if not isinstance(run_root_raw, str):
        raise V11EvaluationError(f"seed {seed}: campaign run_root missing")
    run_root = workspace_path(run_root_raw)
    report_raw = campaign_entry.get("report_path")
    report_path = (
        workspace_path(report_raw)
        if isinstance(report_raw, str)
        else run_root / "PCRA_U_DEVELOPMENT_V1_1_SEED_REPORT.json"
    )
    report = read_json(report_path)
    if not isinstance(report, Mapping) or int(report.get("seed", -1)) != seed:
        raise V11EvaluationError(f"seed {seed}: invalid/mismatched seed report")
    if canonical_sha256(campaign_entry) != canonical_sha256(report):
        raise V11EvaluationError(f"seed {seed}: embedded campaign seed report differs from run-root report")
    reported_run_root = report.get("run_root")
    if isinstance(reported_run_root, str) and workspace_path(reported_run_root) != run_root:
        raise V11EvaluationError(f"seed {seed}: campaign and seed-report run roots disagree")
    prediction_path = run_root / "predictions/selected_dev_predictions_exploratory.json"
    prediction_document = read_json(prediction_path)
    if not isinstance(prediction_document, Mapping) or int(prediction_document.get("seed", -1)) != seed:
        raise V11EvaluationError(f"seed {seed}: invalid/mismatched selected prediction document")
    rows = prediction_rows(prediction_document, f"seed {seed}")
    measured = validate_prediction_set(rows, f"seed {seed}")
    validate_same_locked_dev(baseline_rows, rows, f"seed {seed}")
    verify_reported_metrics(report, measured, f"seed {seed}")
    checkpoint = selected_checkpoint_path(report, run_root, f"seed {seed}")
    model_sha = sha256_file(checkpoint / "model.safetensors")
    reported_model_sha = report.get("selected_checkpoint_model_sha256")
    prediction_model_sha = prediction_document.get("selected_checkpoint_model_sha256")
    if model_sha != reported_model_sha or model_sha != prediction_model_sha:
        raise V11EvaluationError(f"seed {seed}: checkpoint model hash does not bind report and predictions")
    if isinstance(campaign_entry.get("selected_checkpoint_model_sha256"), str):
        if campaign_entry["selected_checkpoint_model_sha256"] != model_sha:
            raise V11EvaluationError(f"seed {seed}: campaign checkpoint hash disagrees")
    reported_eligible, selection = extract_reported_eligibility(report, f"seed {seed}")
    selection_value = None
    for key in ("value", "score", "C_dev_scientific_score_v1_1"):
        if key in selection:
            selection_value = finite_float(selection[key], f"seed {seed} selection.{key}")
            break
    if selection_value is None or not close(selection_value, float(measured["C"])):
        raise V11EvaluationError(
            f"seed {seed}: selected scientific score does not match recomputed prediction score {measured['C']}"
        )
    prediction_checkpoint = prediction_document.get("selected_checkpoint")
    if isinstance(prediction_checkpoint, str) and workspace_path(prediction_checkpoint) != checkpoint:
        raise V11EvaluationError(f"seed {seed}: prediction and seed-report checkpoint paths disagree")
    campaign_checkpoint = campaign_entry.get("selected_checkpoint")
    if isinstance(campaign_checkpoint, str) and workspace_path(campaign_checkpoint) != checkpoint:
        raise V11EvaluationError(f"seed {seed}: campaign and seed-report checkpoint paths disagree")
    if "score" in campaign_entry and not close(
        finite_float(campaign_entry["score"], f"seed {seed} campaign score"), float(measured["C"])
    ):
        raise V11EvaluationError(f"seed {seed}: campaign score disagrees with selected predictions")
    point_eligibility = {
        "Ga_at_least_locked_minimum": measured["Ga"] >= MIN_ELIGIBLE_GA,
        "A_at_least_locked_minimum": measured["A"] >= MIN_ELIGIBLE_A,
        "F_at_most_locked_maximum": measured["F"] <= MAX_ELIGIBLE_F,
    }
    checkpoint_audit = report.get("checkpoint_audit")
    audit_pass = isinstance(checkpoint_audit, Mapping) and checkpoint_audit.get("passed") is True
    training_gates_pass = all_boolean_gates_pass(report)
    final_eligible = reported_eligible and all(point_eligibility.values()) and audit_pass and training_gates_pass

    trajectory: list[dict[str, Any]] = []
    trajectory_inputs: list[dict[str, Any]] = []
    metric_paths = sorted((run_root / "metrics").glob("development_epoch_*.json"))
    if not metric_paths:
        raise V11EvaluationError(f"seed {seed}: no measured epoch metric files for training curve")
    for metric_path in metric_paths:
        epoch_document = read_json(metric_path)
        if not isinstance(epoch_document, Mapping):
            raise V11EvaluationError(f"seed {seed}: invalid epoch document {metric_path}")
        epoch = int(epoch_document.get("epoch", 0))
        train_loss = nested_metric(epoch_document, ("train_epoch_mean_total_loss",))
        dev_loss = nested_metric(epoch_document, ("dev", "loss", "total"))
        score = nested_metric(epoch_document, ("dev", "selection_components", "C_dev_scientific_score_v1_1"))
        if score is None:
            score = nested_metric(epoch_document, ("selection", "value"))
        if epoch <= 0 or train_loss is None or dev_loss is None or score is None:
            raise V11EvaluationError(f"seed {seed}: incomplete measured epoch trajectory in {metric_path}")
        trajectory.append({"seed": seed, "epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss, "C": score})
        trajectory_inputs.append({"path": relative(metric_path), "sha256": sha256_file(metric_path)})
    trajectory.sort(key=lambda row: int(row["epoch"]))
    if len({int(row["epoch"]) for row in trajectory}) != len(trajectory):
        raise V11EvaluationError(f"seed {seed}: duplicate epoch trajectory")

    winner = {
        "seed": seed,
        "run_root": relative(run_root),
        "seed_report_path": relative(report_path),
        "seed_report_sha256": sha256_file(report_path),
        "prediction_path": relative(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "selected_checkpoint": relative(checkpoint),
        "selected_checkpoint_model_sha256": model_sha,
        "metrics": measured,
        "reported_selection_eligible": reported_eligible,
        "point_eligibility": point_eligibility,
        "checkpoint_audit_pass": audit_pass,
        "training_gates_pass": training_gates_pass,
        "eligible": final_eligible,
        "selection_record": dict(selection),
    }
    return winner, rows, trajectory, trajectory_inputs


def verify_locked_v1() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(V1_PREDICTIONS) != V1_PREDICTIONS_SHA256:
        raise V11EvaluationError("Locked V1 epoch-10 prediction artifact hash changed")
    if sha256_file(V1_CHECKPOINT / "model.safetensors") != V1_MODEL_SHA256:
        raise V11EvaluationError("Locked V1 epoch-10 model hash changed")
    report = read_json(V1_REPORT)
    if not isinstance(report, Mapping):
        raise V11EvaluationError("Locked V1 report is invalid")
    if report.get("selected_checkpoint_model_sha256") != V1_MODEL_SHA256:
        raise V11EvaluationError("Locked V1 report checkpoint hash mismatch")
    if workspace_path(str(report.get("selected_checkpoint"))) != V1_CHECKPOINT.resolve():
        raise V11EvaluationError("Locked V1 report no longer selects epoch 10")
    rows = prediction_rows(read_json(V1_PREDICTIONS), "V1")
    metrics = validate_prediction_set(rows, "V1")
    expected = {
        "Gc": 17 / 32,
        "Ga": 144 / 168,
        "A": 0.7420238435082854,
        "F": 9 / 95,
        "S": 0.5876288659793815,
    }
    for metric, value in expected.items():
        if not close(float(metrics[metric]), value):
            raise V11EvaluationError(f"Locked V1 {metric} changed: {metrics[metric]} != {value}")
    return dict(report), rows, metrics


# Family sufficient-statistic layout:
# Gc hit/n, Ga hit/n, false-FOUND hit/n, source tp/fp/fn, then 4x4 answer confusion.
FAMILY_STAT_WIDTH = 25


def family_sufficient_statistics(
    rows: Sequence[Mapping[str, Any]], families: Sequence[str]
) -> np.ndarray:
    family_index = {family: index for index, family in enumerate(families)}
    class_index = {name: index for index, name in enumerate(ANSWERABILITY_CLASSES)}
    result = np.zeros((len(families), FAMILY_STAT_WIDTH), dtype=np.float64)
    for row in rows:
        index = family_index[str(row["family_id"])]
        truth = str(row["answerability_truth"])
        prediction = str(row["answerability_prediction"])
        if truth == "FOUND":
            result[index, 2] += float(row["point_in_target"] is True)
            result[index, 3] += 1.0
            if row["variant"] == "clean":
                result[index, 0] += float(row["point_in_target"] is True)
                result[index, 1] += 1.0
        if truth in {"AMBIGUOUS", "ABSENT"}:
            result[index, 4] += float(prediction == "FOUND")
            result[index, 5] += 1.0
        source_truth = set(row["source_truth"])
        source_prediction = set(row["source_prediction"])
        result[index, 6] += len(source_truth & source_prediction)
        result[index, 7] += len(source_prediction - source_truth)
        result[index, 8] += len(source_truth - source_prediction)
        result[index, 9 + 4 * class_index[truth] + class_index[prediction]] += 1.0
    return result


def bootstrap_metric_arrays(family_stats: np.ndarray, draw_counts: np.ndarray) -> dict[str, np.ndarray]:
    aggregate = draw_counts @ family_stats
    for denominator_index, label in ((1, "Gc"), (3, "Ga"), (5, "F")):
        if np.any(aggregate[:, denominator_index] <= 0):
            raise V11EvaluationError(f"Bootstrap replicate has zero {label} denominator")
    gc = aggregate[:, 0] / aggregate[:, 1]
    ga = aggregate[:, 2] / aggregate[:, 3]
    false_found = aggregate[:, 4] / aggregate[:, 5]
    source_denominator = 2 * aggregate[:, 6] + aggregate[:, 7] + aggregate[:, 8]
    source = np.ones(len(aggregate), dtype=np.float64)
    np.divide(2 * aggregate[:, 6], source_denominator, out=source, where=source_denominator > 0)
    confusion = aggregate[:, 9:].reshape((-1, 4, 4))
    true_positive = np.diagonal(confusion, axis1=1, axis2=2)
    false_positive = confusion.sum(axis=1) - true_positive
    false_negative = confusion.sum(axis=2) - true_positive
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.zeros_like(true_positive)
    np.divide(2 * true_positive, f1_denominator, out=f1, where=f1_denominator > 0)
    macro_f1 = f1.mean(axis=1)
    score = (
        SCORE_WEIGHTS["Gc"] * gc
        + SCORE_WEIGHTS["Ga"] * ga
        + SCORE_WEIGHTS["A"] * macro_f1
        + SCORE_WEIGHTS["one_minus_F"] * (1.0 - false_found)
    )
    return {"Gc": gc, "Ga": ga, "A": macro_f1, "F": false_found, "S": source, "C": score}


def paired_family_bootstrap(
    baseline_rows: Sequence[Mapping[str, Any]], winner_rows: Sequence[Sequence[Mapping[str, Any]]]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    families = sorted({str(row["family_id"]) for row in baseline_rows})
    if len(families) != EXPECTED_FAMILIES:
        raise V11EvaluationError("Paired bootstrap requires exactly 80 locked DEV families")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(len(families), size=(BOOTSTRAP_REPLICATES, len(families)), replace=True)
    draw_counts = np.zeros((BOOTSTRAP_REPLICATES, len(families)), dtype=np.int16)
    for replicate in range(BOOTSTRAP_REPLICATES):
        draw_counts[replicate] = np.bincount(draws[replicate], minlength=len(families))
    baseline_arrays = bootstrap_metric_arrays(family_sufficient_statistics(baseline_rows, families), draw_counts)
    seed_arrays = [
        bootstrap_metric_arrays(family_sufficient_statistics(rows, families), draw_counts) for rows in winner_rows
    ]
    mean_v11 = {
        metric: np.mean(np.stack([arrays[metric] for arrays in seed_arrays], axis=0), axis=0)
        for metric in ("Gc", "Ga", "A", "F", "S", "C")
    }
    # Conventional deltas are V1.1 minus V1 for every metric.  Unlike the
    # other metrics, lower false-FOUND F is better, so its gate uses the upper
    # confidence bound and requires a negative point delta.
    delta_arrays = {
        "Gc": mean_v11["Gc"] - baseline_arrays["Gc"],
        "Ga": mean_v11["Ga"] - baseline_arrays["Ga"],
        "A": mean_v11["A"] - baseline_arrays["A"],
        "F": mean_v11["F"] - baseline_arrays["F"],
        "S": mean_v11["S"] - baseline_arrays["S"],
        "C": mean_v11["C"] - baseline_arrays["C"],
    }
    summary = {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "rng": "numpy.random.default_rng(PCG64).choice",
        "unit": "family",
        "resample_rule": "sample 80 family IDs with replacement and retain all five variants for every sampled occurrence",
        "v1_1_seed_rule": "recompute each nonlinear metric per seed within replicate, then arithmetic-mean three seed metrics",
        "paired_draw_matrix_sha256": hashlib.sha256(draws.tobytes(order="C")).hexdigest(),
        "confidence_interval": "two-sided percentile 95% [2.5%,97.5%]",
        "scientific_status": SCIENTIFIC_STATUS,
        "metrics": {},
    }
    for metric, values in delta_arrays.items():
        summary["metrics"][metric] = {
            "delta_convention": "V1.1_mean_minus_V1",
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
        }
    return summary, delta_arrays


def mean_sd_summary(winners: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in ("Gc", "Ga", "A", "F", "S", "C"):
        values = [float(winner["metrics"][metric]) for winner in winners]
        point_delta = statistics.fmean(values) - float(baseline[metric])
        output[metric] = {
            "v1": float(baseline[metric]),
            "v1_1_by_seed": {str(winner["seed"]): float(winner["metrics"][metric]) for winner in winners},
            "v1_1_mean": statistics.fmean(values),
            "v1_1_sample_sd": statistics.stdev(values),
            "v1_1_min": min(values),
            "v1_1_max": max(values),
            "point_delta": point_delta,
            "delta_convention": "V1.1_mean_minus_V1",
        }
    return output


def acceptance_gates(
    winners: Sequence[Mapping[str, Any]], summaries: Mapping[str, Any], bootstrap: Mapping[str, Any]
) -> dict[str, bool]:
    bootstrap_metrics = bootstrap["metrics"]
    return {
        "all_three_winners_eligible": len(winners) == 3 and all(bool(winner["eligible"]) for winner in winners),
        "each_seed_Gc_at_least_0_50": all(float(winner["metrics"]["Gc"]) >= MIN_PER_SEED_CLEAN_FOUND for winner in winners),
        "at_least_two_of_three_Gc_above_v1_0_53125": sum(
            float(winner["metrics"]["Gc"]) > V1_CLEAN_FOUND_REFERENCE for winner in winners
        )
        >= 2,
        "mean_C_above_v1_C": float(summaries["C"]["v1_1_mean"]) > float(summaries["C"]["v1"]),
        "ci95_low_delta_Gc_above_0": float(bootstrap_metrics["Gc"]["ci95_low"]) > 0.0,
        "ci95_low_delta_Ga_above_minus_0_02": float(bootstrap_metrics["Ga"]["ci95_low"])
        >= -NONINFERIORITY_MARGIN,
        "ci95_low_delta_A_above_minus_0_02": float(bootstrap_metrics["A"]["ci95_low"])
        >= -NONINFERIORITY_MARGIN,
        "point_delta_F_below_0": float(summaries["F"]["point_delta"]) < 0.0,
        "ci95_high_delta_F_at_most_plus_0_02": float(bootstrap_metrics["F"]["ci95_high"])
        <= NONINFERIORITY_MARGIN,
    }


def winner_csv_rows(winners: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for winner in winners:
        counts = winner["metrics"]["counts"]
        rows.append(
            {
                "scientific_status": SCIENTIFIC_STATUS,
                "seed": winner["seed"],
                "run_root": winner["run_root"],
                "selected_checkpoint": winner["selected_checkpoint"],
                "model_sha256": winner["selected_checkpoint_model_sha256"],
                "prediction_sha256": winner["prediction_sha256"],
                "Gc": winner["metrics"]["Gc"],
                "Gc_correct": counts["Gc_correct"],
                "Gc_denominator": counts["Gc_denominator"],
                "Ga": winner["metrics"]["Ga"],
                "Ga_correct": counts["Ga_correct"],
                "Ga_denominator": counts["Ga_denominator"],
                "A": winner["metrics"]["A"],
                "F": winner["metrics"]["F"],
                "false_found_count": counts["false_found_count"],
                "false_found_denominator": counts["false_found_denominator"],
                "S_secondary": winner["metrics"]["S"],
                "C": winner["metrics"]["C"],
                "reported_selection_eligible": winner["reported_selection_eligible"],
                "recomputed_eligible": winner["eligible"],
            }
        )
    return rows


def metric_summary_rows(summaries: Mapping[str, Any], seeds: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    for metric in ("Gc", "Ga", "A", "F", "S", "C"):
        summary = summaries[metric]
        row = {
            "scientific_status": SCIENTIFIC_STATUS,
            "metric": metric,
            "delta_convention": summary["delta_convention"],
            "v1": summary["v1"],
            "v1_1_mean": summary["v1_1_mean"],
            "v1_1_sample_sd": summary["v1_1_sample_sd"],
            "v1_1_min": summary["v1_1_min"],
            "v1_1_max": summary["v1_1_max"],
            "point_delta": summary["point_delta"],
        }
        for seed in seeds:
            row[f"seed_{seed}"] = summary["v1_1_by_seed"][str(seed)]
        rows.append(row)
    return rows


def bootstrap_summary_rows(bootstrap: Mapping[str, Any], summaries: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric in ("Gc", "Ga", "A", "F", "S", "C"):
        interval = bootstrap["metrics"][metric]
        rows.append(
            {
                "scientific_status": SCIENTIFIC_STATUS,
                "metric": metric,
                "delta_convention": interval["delta_convention"],
                "estimate": summaries[metric]["point_delta"],
                "ci95_low": interval["ci95_low"],
                "ci95_high": interval["ci95_high"],
                "primary_gate_metric": metric in {"Gc", "Ga", "A", "F"},
                "source_secondary": metric == "S",
            }
        )
    return rows


def svg_document(width: int, height: int, body: Sequence[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def text_svg(x: float, y: float, value: str, size: int = 13, anchor: str = "start", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-family="sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#222">{html.escape(value)}</text>'
    )


def training_curves_svg(trajectories: Sequence[Mapping[str, Any]]) -> str:
    width, height = 1120, 820
    left, right = 90, 1080
    colors = {24082026: "#1f77b4", 24082027: "#d95f02", 24082028: "#2b8c4b"}
    panels = (("train_loss", "Train epoch mean total loss"), ("dev_loss", "DEV total loss"), ("C", "Locked scientific composite C"))
    body = [
        text_svg(left, 32, "P-CRA-U V1.1 measured training trajectories", 20, weight="bold"),
        text_svg(left, 55, "Real epoch artifacts; DEV exploratory only", 13),
    ]
    for legend_index, (seed, color) in enumerate(colors.items()):
        x = left + 520 + legend_index * 155
        body.append(f'<line x1="{x:.2f}" y1="28" x2="{x+24:.2f}" y2="28" stroke="{color}" stroke-width="3"/>')
        body.append(text_svg(x + 30, 33, str(seed), 12))
    grouped: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trajectories:
        grouped[int(row["seed"])].append(row)
    for panel_index, (field, title) in enumerate(panels):
        top = 90 + panel_index * 235
        bottom = top + 175
        values = [float(row[field]) for row in trajectories]
        epochs = [int(row["epoch"]) for row in trajectories]
        y_min, y_max = min(values), max(values)
        padding = max((y_max - y_min) * 0.08, 1e-6)
        y_min -= padding
        y_max += padding
        x_min, x_max = min(epochs), max(epochs)
        body.extend(
            [
                text_svg(left, top - 10, title, 14, weight="bold"),
                f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#777"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#777"/>',
                text_svg(left - 8, top + 5, f"{y_max:.4f}", 11, anchor="end"),
                text_svg(left - 8, bottom, f"{y_min:.4f}", 11, anchor="end"),
                text_svg(left, bottom + 20, str(x_min), 11, anchor="middle"),
                text_svg(right, bottom + 20, str(x_max), 11, anchor="middle"),
            ]
        )
        for seed in EXPECTED_SEEDS:
            rows = sorted(grouped[seed], key=lambda row: int(row["epoch"]))
            points = []
            for row in rows:
                x = left + (int(row["epoch"]) - x_min) / max(1, x_max - x_min) * (right - left)
                y = bottom - (float(row[field]) - y_min) / max(1e-12, y_max - y_min) * (bottom - top)
                points.append(f"{x:.2f},{y:.2f}")
            body.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[seed]}" stroke-width="2.2"/>'
            )
    body.append(text_svg((left + right) / 2, height - 18, "Epoch (same locked budget per seed)", 13, anchor="middle"))
    return svg_document(width, height, body)


def comparison_svg(winners: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any], summaries: Mapping[str, Any]) -> str:
    width, height = 1200, 650
    left, right, top, bottom = 85, 1160, 100, 555
    metrics = ("Gc", "Ga", "A", "one_minus_F", "S", "C")
    labels = {"Gc": "Gc clean", "Ga": "Ga all", "A": "A macro-F1", "one_minus_F": "1-F", "S": "S source", "C": "C score"}
    series = [("V1", "#666666", baseline)]
    palette = ("#1f77b4", "#d95f02", "#2b8c4b")
    for color, winner in zip(palette, winners):
        series.append((str(winner["seed"]), color, winner["metrics"]))
    mean_values = {metric: summaries[metric]["v1_1_mean"] for metric in ("Gc", "Ga", "A", "F", "S", "C")}
    series.append(("V1.1 mean", "#6a3d9a", mean_values))
    body = [
        text_svg(left, 32, "Locked DEV metric comparison", 20, weight="bold"),
        text_svg(left, 56, "All bars are computed from real selected-checkpoint predictions", 13),
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#777"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#777"/>',
    ]
    for tick in range(6):
        value = tick / 5
        y = bottom - value * (bottom - top)
        body.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e3e3e3"/>')
        body.append(text_svg(left - 8, y + 4, f"{value:.1f}", 11, anchor="end"))
    group_width = (right - left) / len(metrics)
    bar_width = group_width / (len(series) + 1)
    for metric_index, metric in enumerate(metrics):
        group_left = left + metric_index * group_width
        for series_index, (_, color, values) in enumerate(series):
            value = 1.0 - float(values["F"]) if metric == "one_minus_F" else float(values[metric])
            x = group_left + (series_index + 0.5) * bar_width
            y = bottom - value * (bottom - top)
            body.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width*0.78:.2f}" height="{bottom-y:.2f}" fill="{color}"/>'
            )
        body.append(text_svg(group_left + group_width / 2, bottom + 22, labels[metric], 11, anchor="middle"))
    legend_x = left + 440
    for index, (name, color, _) in enumerate(series):
        x = legend_x + index * 135
        body.append(f'<rect x="{x}" y="72" width="14" height="10" fill="{color}"/>')
        body.append(text_svg(x + 19, 82, name, 11))
    body.append(text_svg(left, height - 24, "Source S is secondary and cannot rescue or fail the core decision.", 12))
    return svg_document(width, height, body)


def bootstrap_forest_svg(bootstrap: Mapping[str, Any], summaries: Mapping[str, Any]) -> str:
    width, height = 1050, 560
    left, right, top, bottom = 210, 995, 100, 470
    metrics = ("Gc", "Ga", "A", "F", "S", "C")
    intervals = bootstrap["metrics"]
    lows = [float(intervals[metric]["ci95_low"]) for metric in metrics]
    highs = [float(intervals[metric]["ci95_high"]) for metric in metrics]
    x_min = min(min(lows), -NONINFERIORITY_MARGIN, 0.0) - 0.015
    x_max = max(max(highs), NONINFERIORITY_MARGIN, 0.0) + 0.015

    def x_coordinate(value: float) -> float:
        return left + (value - x_min) / max(1e-12, x_max - x_min) * (right - left)

    body = [
        text_svg(45, 32, "Paired family-bootstrap deltas", 20, weight="bold"),
        text_svg(45, 56, "10,000 paired family draws; conditional exploratory DEV 95% percentile CI", 13),
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#777"/>',
        f'<line x1="{x_coordinate(0):.2f}" y1="{top-15}" x2="{x_coordinate(0):.2f}" y2="{bottom}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{x_coordinate(-NONINFERIORITY_MARGIN):.2f}" y1="{top-15}" x2="{x_coordinate(-NONINFERIORITY_MARGIN):.2f}" y2="{bottom}" stroke="#b44" stroke-dasharray="5,4"/>',
        f'<line x1="{x_coordinate(NONINFERIORITY_MARGIN):.2f}" y1="{top-15}" x2="{x_coordinate(NONINFERIORITY_MARGIN):.2f}" y2="{bottom}" stroke="#b44" stroke-dasharray="5,4"/>',
    ]
    for index, metric in enumerate(metrics):
        y = top + index * 58
        point = float(summaries[metric]["point_delta"])
        low = float(intervals[metric]["ci95_low"])
        high = float(intervals[metric]["ci95_high"])
        color = "#777777" if metric == "S" else "#1f5a93"
        body.append(text_svg(left - 18, y + 5, metric + (" secondary" if metric == "S" else ""), 13, anchor="end"))
        body.append(
            f'<line x1="{x_coordinate(low):.2f}" y1="{y}" x2="{x_coordinate(high):.2f}" y2="{y}" stroke="{color}" stroke-width="3"/>'
        )
        body.append(f'<circle cx="{x_coordinate(point):.2f}" cy="{y}" r="5" fill="{color}"/>')
        body.append(text_svg(right - 2, y - 9, f"{point:+.4f} [{low:+.4f}, {high:+.4f}]", 11, anchor="end"))
    body.append(text_svg(x_coordinate(0), bottom + 24, "0", 11, anchor="middle"))
    body.append(text_svg(x_coordinate(-NONINFERIORITY_MARGIN), bottom + 24, "-0.02 G/A", 11, anchor="middle"))
    body.append(text_svg(x_coordinate(NONINFERIORITY_MARGIN), bottom + 24, "+0.02 F", 11, anchor="middle"))
    body.append(text_svg((left + right) / 2, height - 24, "Conventional delta V1.1 − V1 (left is better only for false-FOUND F)", 13, anchor="middle"))
    return svg_document(width, height, body)


def report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P-CRA-U development V1.1 — conditional exploratory DEV evaluation",
        "",
        f"- Decision: `{report['decision']}`.",
        f"- Scientific status: `{report['scientific_status']}`.",
        "- DEV was used for checkpoint selection; all intervals below are conditional exploratory DEV intervals, not confirmatory test intervals.",
        "- Calibration, Test-IID and Test-OOD were neither accepted as inputs nor opened by this evaluator.",
        f"- Locked V1 comparator: `{report['v1']['checkpoint']}` / `{report['v1']['model_sha256']}`.",
        f"- Bootstrap: `{BOOTSTRAP_REPLICATES}` paired family replicates, seed `{BOOTSTRAP_SEED}`; each sampled family retains all five variants.",
        "",
        "## Per-seed selected winners",
        "",
        "| Seed | Gc clean FOUND | Ga raw FOUND | A macro-F1 | F false-FOUND | S source (secondary) | C | Eligible |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for winner in report["v1_1"]["winners"]:
        metrics = winner["metrics"]
        lines.append(
            f"| {winner['seed']} | {metrics['Gc']:.6f} | {metrics['Ga']:.6f} | {metrics['A']:.6f} | "
            f"{metrics['F']:.6f} | {metrics['S']:.6f} | {metrics['C']:.6f} | "
            f"{'PASS' if winner['eligible'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Three-seed mean ± sample SD and V1 comparison",
            "",
            "| Metric | V1 | V1.1 mean ± SD | Oriented delta |",
            "|:---|---:|---:|---:|",
        ]
    )
    for metric in ("Gc", "Ga", "A", "F", "S", "C"):
        summary = report["metric_summary"][metric]
        lines.append(
            f"| {metric} | {summary['v1']:.6f} | {summary['v1_1_mean']:.6f} ± "
            f"{summary['v1_1_sample_sd']:.6f} | {summary['point_delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "Every delta is conventional `V1.1 − V1`. Positive is better for Gc/Ga/A/S/C; negative is better for false-FOUND F.",
            "",
            "## Paired family bootstrap",
            "",
            "| Metric | Point delta | 95% CI | Role |",
            "|:---|---:|:---|:---|",
        ]
    )
    for metric in ("Gc", "Ga", "A", "F", "S", "C"):
        interval = report["paired_family_bootstrap"]["metrics"][metric]
        point = report["metric_summary"][metric]["point_delta"]
        role = "secondary" if metric == "S" else ("descriptive" if metric == "C" else "core gate")
        lines.append(
            f"| {metric} | {point:+.6f} | [{interval['ci95_low']:+.6f}, {interval['ci95_high']:+.6f}] | {role} |"
        )
    lines.extend(["", "## Acceptance gates", ""])
    lines.extend(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`" for name, passed in report["acceptance_gates"].items())
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A promotion decision means only that the selected V1.1 candidate may proceed to a fresh, independently governed Calibration stage. It does not make V1.1 a final model and does not authorize Test-IID/Test-OOD access. If any core gate fails, V1 epoch 10 remains the retained development checkpoint.",
            "",
            "Source micro-F1 is a secondary reported diagnostic; it cannot rescue or fail the core decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def deployment_candidate(decision: str, winners: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if decision != "PROMOTE_V1_1_TO_FRESH_CALIBRATION":
        return {
            "model": "P-CRA-U-development-v1",
            "checkpoint": relative(V1_CHECKPOINT),
            "model_sha256": V1_MODEL_SHA256,
            "reason": "one_or_more_precommitted_v1_1_acceptance_gates_failed",
        }
    selected = max(
        winners,
        key=lambda winner: (
            float(winner["metrics"]["C"]),
            -float(winner["metrics"]["F"]),
            float(winner["metrics"]["Gc"]),
            -int(winner["seed"]),
        ),
    )
    return {
        "model": "P-CRA-U-development-v1.1",
        "seed": selected["seed"],
        "checkpoint": selected["selected_checkpoint"],
        "model_sha256": selected["selected_checkpoint_model_sha256"],
        "selection_rule": "max C; tie F low, Gc high, seed low",
        "status": "DEV_SELECTED_REQUIRES_FRESH_CALIBRATION",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-report",
        required=True,
        help="V1.1 campaign report containing exactly the three locked seed reports",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="A new, nonexistent directory below results/ for evaluation artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_path = workspace_path(args.campaign_report)
    output_root = workspace_path(args.output_root, must_exist=False)
    try:
        output_root.relative_to(RESULTS_ROOT.resolve())
    except ValueError as exc:
        raise V11EvaluationError("--output-root must be a new directory below results/") from exc
    if output_root.exists():
        raise V11EvaluationError(f"Refusing to overwrite existing output root: {output_root}")
    campaign = read_json(campaign_path)
    if not isinstance(campaign, Mapping):
        raise V11EvaluationError("Campaign report must be a JSON object")
    if campaign.get("passed") is not True or campaign.get("decision") != "V1_1_CAMPAIGN_COMPLETE":
        raise V11EvaluationError("Post-campaign analysis requires a passed, complete three-seed V1.1 campaign")

    v1_report, v1_rows, v1_metrics = verify_locked_v1()
    seed_entries = campaign_seed_entries(campaign)
    winners: list[dict[str, Any]] = []
    rows_by_seed: list[list[dict[str, Any]]] = []
    trajectories: list[dict[str, Any]] = []
    trajectory_inputs: list[dict[str, Any]] = []
    for entry in seed_entries:
        winner, rows, trajectory, inputs = load_winner(entry, v1_rows)
        winners.append(winner)
        rows_by_seed.append(rows)
        trajectories.extend(trajectory)
        trajectory_inputs.extend(inputs)

    summaries = mean_sd_summary(winners, v1_metrics)
    bootstrap, delta_arrays = paired_family_bootstrap(v1_rows, rows_by_seed)
    for metric in ("Gc", "Ga", "A", "F", "S", "C"):
        bootstrap["metrics"][metric]["estimate"] = summaries[metric]["point_delta"]
    gates = acceptance_gates(winners, summaries, bootstrap)
    decision = "PROMOTE_V1_1_TO_FRESH_CALIBRATION" if all(gates.values()) else "RETAIN_V1"
    candidate = deployment_candidate(decision, winners)
    if decision == "PROMOTE_V1_1_TO_FRESH_CALIBRATION":
        campaign_winner = campaign.get("campaign_winner")
        if not isinstance(campaign_winner, Mapping):
            raise V11EvaluationError("Promotable campaign lacks its locked campaign_winner record")
        expected_campaign_candidate = (
            int(campaign_winner.get("seed", -1)),
            str(campaign_winner.get("checkpoint")),
            str(campaign_winner.get("model_sha256")),
        )
        evaluated_candidate = (
            int(candidate["seed"]),
            str(candidate["checkpoint"]),
            str(candidate["model_sha256"]),
        )
        if expected_campaign_candidate != evaluated_candidate:
            raise V11EvaluationError("Post-campaign candidate disagrees with the locked campaign winner tie-break")

    input_manifest = {
        "schema_version": 1,
        "scientific_status": SCIENTIFIC_STATUS,
        "evaluator_path": relative(Path(__file__)),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "campaign_report": {"path": relative(campaign_path), "sha256": sha256_file(campaign_path)},
        "v1": {
            "report": {"path": relative(V1_REPORT), "sha256": sha256_file(V1_REPORT)},
            "predictions": {"path": relative(V1_PREDICTIONS), "sha256": sha256_file(V1_PREDICTIONS)},
            "checkpoint_model": {
                "path": relative(V1_CHECKPOINT / "model.safetensors"),
                "sha256": sha256_file(V1_CHECKPOINT / "model.safetensors"),
            },
        },
        "v1_1": [
            {
                "seed": winner["seed"],
                "seed_report": {"path": winner["seed_report_path"], "sha256": winner["seed_report_sha256"]},
                "predictions": {"path": winner["prediction_path"], "sha256": winner["prediction_sha256"]},
                "checkpoint_model": {
                    "path": f"{winner['selected_checkpoint']}/model.safetensors",
                    "sha256": winner["selected_checkpoint_model_sha256"],
                },
            }
            for winner in winners
        ],
        "epoch_trajectory_inputs": trajectory_inputs,
        "canonical_manifest_sha256": None,
    }
    input_manifest["canonical_manifest_sha256"] = canonical_sha256(
        {key: value for key, value in input_manifest.items() if key != "canonical_manifest_sha256"}
    )
    baseline_hashes_after = {
        "predictions": sha256_file(V1_PREDICTIONS),
        "checkpoint_model": sha256_file(V1_CHECKPOINT / "model.safetensors"),
    }
    baseline_unchanged = baseline_hashes_after == {
        "predictions": V1_PREDICTIONS_SHA256,
        "checkpoint_model": V1_MODEL_SHA256,
    }
    if not baseline_unchanged:
        raise V11EvaluationError("Locked V1 changed during read-only post-campaign analysis")

    public_winners = []
    for winner in winners:
        public_winners.append({key: value for key, value in winner.items() if key != "selection_record"})
    report = {
        "schema_version": 1,
        "protocol_id": "pcra_u_development_v1_1_post_campaign_evaluation",
        "generated_at_utc": utc_now(),
        "scientific_status": SCIENTIFIC_STATUS,
        "decision": decision,
        "passed": decision == "PROMOTE_V1_1_TO_FRESH_CALIBRATION",
        "score": {
            "formula": "C = 0.45*Gc + 0.20*Ga + 0.20*A + 0.15*(1-F)",
            "weights": SCORE_WEIGHTS,
            "source_role": "secondary_report_only_not_an_acceptance_gate",
        },
        "v1": {
            "checkpoint": relative(V1_CHECKPOINT),
            "model_sha256": V1_MODEL_SHA256,
            "prediction_sha256": V1_PREDICTIONS_SHA256,
            "selected_epoch": 10,
            "metrics": v1_metrics,
            "report_decision": v1_report.get("decision"),
        },
        "v1_1": {"seeds": list(EXPECTED_SEEDS), "winners": public_winners},
        "metric_summary": summaries,
        "paired_family_bootstrap": bootstrap,
        "acceptance_gates": gates,
        "decision_lock_candidate": candidate,
        "input_manifest_sha256": input_manifest["canonical_manifest_sha256"],
        "locked_v1_inputs_unchanged": baseline_unchanged,
        "calibration_fit": False,
        "test_iid_opened": False,
        "test_ood_opened": False,
        "tables_02_to_06": "NOT_RUN",
        "limitations": [
            "DEV was used for checkpoint selection; bootstrap intervals are conditional exploratory DEV intervals.",
            "The family bootstrap treats the three trained seeds as fixed repeats and averages their recomputed metrics.",
            "Only 32 clean DEV records have GT FOUND; Gc has a correspondingly discrete denominator.",
            "Source micro-F1 is secondary and is not an acceptance gate.",
        ],
    }
    decision_lock = {
        "schema_version": 1,
        "status": "DECISION_LOCK_CANDIDATE_NOT_FINAL_MODEL_LOCK",
        "generated_at_utc": report["generated_at_utc"],
        "scientific_status": SCIENTIFIC_STATUS,
        "decision": decision,
        "candidate": candidate,
        "acceptance_gates": gates,
        "input_manifest_sha256": input_manifest["canonical_manifest_sha256"],
        "requires_fresh_calibration": decision == "PROMOTE_V1_1_TO_FRESH_CALIBRATION",
        "authorizes_test_access": False,
        "calibration_fit": False,
        "test_opened": False,
    }

    output_root.mkdir(parents=True, exist_ok=False)
    for directory in ("checks", "figures", "manifests", "metrics", "tables"):
        (output_root / directory).mkdir(parents=True, exist_ok=False)
    write_json(output_root / "manifests/input_manifest.json", input_manifest)
    write_json(output_root / "metrics/per_seed_winners.json", {"scientific_status": SCIENTIFIC_STATUS, "winners": public_winners})
    write_json(output_root / "metrics/three_seed_metric_summary.json", summaries)
    write_json(output_root / "metrics/paired_family_bootstrap.json", bootstrap)
    write_json(output_root / "checks/acceptance_gates.json", {"decision": decision, "gates": gates})
    write_json(output_root / "checks/decision_lock_candidate.json", decision_lock)
    write_json(output_root / "PCRA_U_DEVELOPMENT_V1_1_EVALUATION.json", report)
    (output_root / "PCRA_U_DEVELOPMENT_V1_1_EVALUATION.md").write_text(report_markdown(report), encoding="utf-8")

    winner_fields = list(winner_csv_rows(winners)[0])
    write_csv(output_root / "tables/per_seed_winners.csv", winner_fields, winner_csv_rows(winners))
    summary_rows = metric_summary_rows(summaries, EXPECTED_SEEDS)
    write_csv(output_root / "tables/v1_comparison_mean_sd.csv", list(summary_rows[0]), summary_rows)
    bootstrap_rows = bootstrap_summary_rows(bootstrap, summaries)
    write_csv(output_root / "tables/paired_family_bootstrap_summary.csv", list(bootstrap_rows[0]), bootstrap_rows)
    write_csv(
        output_root / "tables/acceptance_gates.csv",
        ["scientific_status", "gate", "passed"],
        [
            {"scientific_status": SCIENTIFIC_STATUS, "gate": name, "passed": passed}
            for name, passed in gates.items()
        ],
    )
    write_csv(
        output_root / "tables/training_trajectory_measured.csv",
        ["seed", "epoch", "train_loss", "dev_loss", "C"],
        trajectories,
    )
    write_csv(
        output_root / "metrics/paired_family_bootstrap_draws.csv",
        ["replicate", "delta_Gc", "delta_Ga", "delta_A", "delta_F", "delta_S", "delta_C"],
        (
            {
                "replicate": index,
                "delta_Gc": delta_arrays["Gc"][index],
                "delta_Ga": delta_arrays["Ga"][index],
                "delta_A": delta_arrays["A"][index],
                "delta_F": delta_arrays["F"][index],
                "delta_S": delta_arrays["S"][index],
                "delta_C": delta_arrays["C"][index],
            }
            for index in range(BOOTSTRAP_REPLICATES)
        ),
    )
    (output_root / "figures/F01_v1_1_training_curves_measured.svg").write_text(
        training_curves_svg(trajectories), encoding="utf-8"
    )
    (output_root / "figures/F02_v1_vs_v1_1_metric_comparison.svg").write_text(
        comparison_svg(winners, v1_metrics, summaries), encoding="utf-8"
    )
    (output_root / "figures/F03_paired_family_bootstrap.svg").write_text(
        bootstrap_forest_svg(bootstrap, summaries), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "output_root": relative(output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
