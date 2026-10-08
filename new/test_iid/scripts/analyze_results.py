#!/usr/bin/env python3
"""Compute the complete frozen best-V2 Test-IID evaluation report."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TEST_ROOT = ROOT / "new/test_iid"
DATASET_ROOT = TEST_ROOT / "dataset"
EVAL_ROOT = TEST_ROOT / "evaluation/best_v2"
PREDICTIONS = EVAL_ROOT / "predictions.jsonl"
DECISIONS = EVAL_ROOT / "decisions.jsonl"
BASE_METRICS = EVAL_ROOT / "metrics.json"
EVAL_MANIFEST = TEST_ROOT / "protocol/test_iid_eval_manifest.json"
FEATURE_INDEX = TEST_ROOT / "feature_cache/indexes/index_full.json"
FREEZE_LOCK = TEST_ROOT / "contracts/best_v2_freeze_lock.json"
CALIBRATOR = ROOT / "new/outputs/pcrau_target_v2_full_seed_24082026/evaluation/calibration/calibrator.json"
ANSWER_CLASSES = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
SOURCE_CLASSES = ["semantic", "relation", "spatial", "depth", "occlusion"]

sys.path.insert(0, str(ROOT / "new/src"))
from pcrau.calibration import apply_calibrator  # noqa: E402
from pcrau.utils import atomic_json, read_json, sha256_file  # noqa: E402


class TestIIDAnalysisError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def binary_metrics(truth: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    truth = truth.astype(bool)
    prediction = probability >= threshold
    tp = int((truth & prediction).sum())
    tn = int(((~truth) & (~prediction)).sum())
    fp = int(((~truth) & prediction).sum())
    fn = int((truth & (~prediction)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return {
        "threshold": threshold, "support_positive": int(truth.sum()),
        "predicted_positive": int(prediction.sum()), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / len(truth),
        "roc_auc": roc_auc(truth, probability),
        "average_precision": average_precision(truth, probability),
        "brier": float(np.mean((probability - truth.astype(float)) ** 2)),
        "ece_10": ece(probability, truth.astype(float)),
    }


def roc_auc(truth: np.ndarray, score: np.ndarray) -> float | None:
    truth = truth.astype(bool)
    positives, negatives = int(truth.sum()), int((~truth).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(truth: np.ndarray, score: np.ndarray) -> float | None:
    truth = truth.astype(bool)
    positives = int(truth.sum())
    if not positives:
        return None
    order = np.argsort(-score, kind="mergesort")
    ordered = truth[order].astype(int)
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / positives)


def ece(probability: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (probability >= low) & (probability < high if index + 1 < bins else probability <= high)
        if selected.any():
            result += float(selected.mean()) * abs(float(probability[selected].mean() - truth[selected].mean()))
    return float(result)


def nll_binary(probability: np.ndarray, truth: np.ndarray) -> float:
    p = probability.clip(1e-7, 1 - 1e-7)
    return float(-np.mean(truth * np.log(p) + (1 - truth) * np.log(1 - p)))


def confusion_metrics(truth: np.ndarray, prediction: np.ndarray, classes: list[str]) -> dict[str, Any]:
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for target, predicted in zip(truth, prediction):
        matrix[int(target), int(predicted)] += 1
    rows = {}
    for index, name in enumerate(classes):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows[name] = {
            "support": int(matrix[index].sum()), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    return {
        "confusion_matrix": matrix.tolist(),
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "macro_f1": float(np.mean([row["f1"] for row in rows.values()])),
        "per_class": rows,
    }


def point_to_mask_distance(mask_path: Path, point_xy: list[int]) -> float | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or not np.any(mask > 0):
        return None
    outside = (mask == 0).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    x, y = point_xy
    return float(distance[int(y), int(x)])


def summarize_stratum(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_rows = [row for row in rows if row["target_exists"]]
    return {
        "samples": len(rows),
        "grounding_evaluated": len(target_rows),
        "grounding_accuracy": float(np.mean([row["inside"] for row in target_rows])) if target_rows else None,
        "answerability_accuracy": float(np.mean([row["answer_correct"] for row in rows])) if rows else None,
        "mean_calibrated_risk": float(np.mean([row["risk"] for row in rows])) if rows else None,
        "observed_error_rate": float(np.mean([row["error"] for row in rows])) if rows else None,
        "policy_action_accuracy": float(np.mean([row["action_correct"] for row in rows])) if rows else None,
    }


def stratum_table(rows: list[dict[str, Any]], fields: Iterable[str]) -> dict[str, Any]:
    output = {}
    for field in fields:
        values = sorted({str(row[field]) for row in rows})
        output[field] = {value: summarize_stratum([row for row in rows if str(row[field]) == value]) for value in values}
    return output


def analyze() -> dict[str, Any]:
    freeze = read_json(FREEZE_LOCK)
    calibrator = read_json(CALIBRATOR)
    manifest = read_json(EVAL_MANIFEST)
    feature_index = read_json(FEATURE_INDEX)
    base = read_json(BASE_METRICS)
    predictions = read_jsonl(PREDICTIONS)
    decisions = read_jsonl(DECISIONS)
    if len(predictions) != 1000 or len(decisions) != 1000 or len(manifest["entries"]) != 1000:
        raise TestIIDAnalysisError("expected exactly 1,000 predictions/decisions/evaluator rows")
    if feature_index.get("status") != "COMPLETE" or feature_index["counts"]["samples"] != 1000:
        raise TestIIDAnalysisError("feature cache is not complete")
    selected = freeze["selected_checkpoint"]
    if sha256_file(ROOT / selected["path"]) != selected["model_sha256"]:
        raise TestIIDAnalysisError("checkpoint hash changed after inference")
    if sha256_file(CALIBRATOR) != freeze["frozen_calibrator"]["sha256"]:
        raise TestIIDAnalysisError("calibrator hash changed after inference")

    entries = {row["sample_id"]: row for row in manifest["entries"]}
    decision_map = {row["sample_id"]: row for row in decisions}
    if set(entries) != {row["sample_id"] for row in predictions} or set(entries) != set(decision_map):
        raise TestIIDAnalysisError("prediction/evaluator/decision sample identity differs")

    rows = []
    distance_values = []
    target_mass = []
    required_mass = []
    region_areas = []
    conformal_hits = []
    conformal_mass = float(calibrator["conformal"]["probability_mass"])
    for prediction in predictions:
        sample_id = prediction["sample_id"]
        entry = entries[sample_id]
        decision = decision_map[sample_id]
        evaluation = prediction["evaluation"]
        truth_index = int(evaluation["answerability_index"])
        answer_probs = prediction["answerability_probabilities"]
        predicted_state = max(answer_probs, key=answer_probs.get)
        predicted_index = ANSWER_CLASSES.index(predicted_state)
        target_exists = bool(evaluation["target_exists"])
        inside = bool(evaluation["map_inside_target"])
        error = bool(evaluation["error_event"])
        risk = float(apply_calibrator(prediction, calibrator))
        expected_action = entry["audit_only"]["expected_intervention"]
        row = {
            "sample_id": sample_id, "family_id": entry["family_id"],
            "variant": entry["variant"], "relation": entry["audit_only"]["relation"],
            "family_category": entry["audit_only"]["family_category"],
            "target_object_group": entry["audit_only"]["target_object_group"],
            "answerability_state": ANSWER_CLASSES[truth_index],
            "state_submode": entry["audit_only"]["state_submode"],
            "target_exists": target_exists, "inside": inside,
            "answer_correct": predicted_index == truth_index,
            "risk": risk, "error": error,
            "action": decision["action"], "expected_action": expected_action,
            "action_correct": decision["action"] == expected_action,
        }
        rows.append(row)
        if target_exists:
            mask_path = DATASET_ROOT / entry["supervision"]["target_mask_path"]
            distance = point_to_mask_distance(mask_path, prediction["spatial"]["map_pixel_xy"])
            if distance is not None:
                distance_values.append(distance)
            target_mass.append(float(evaluation["target_probability_mass"]))
            mass = float(evaluation["target_required_hd_mass"])
            required_mass.append(mass)
            conformal_hits.append(mass <= conformal_mass)
            if decision.get("confidence_region"):
                region_areas.append(float(decision["confidence_region"]["grid_area_fraction"]))

    answer_truth = np.asarray([int(row["evaluation"]["answerability_index"]) for row in predictions])
    answer_probability = np.asarray([
        [row["answerability_probabilities"][name] for name in ANSWER_CLASSES] for row in predictions
    ], dtype=float)
    answer_prediction = answer_probability.argmax(axis=1)
    onehot = np.eye(len(ANSWER_CLASSES))[answer_truth]
    confidence = answer_probability.max(axis=1)
    correct = (answer_prediction == answer_truth).astype(float)
    answer_metrics = confusion_metrics(answer_truth, answer_prediction, ANSWER_CLASSES)
    answer_metrics["multiclass_nll"] = float(-np.mean(np.log(answer_probability[np.arange(1000), answer_truth].clip(1e-7))))
    answer_metrics["multiclass_brier"] = float(np.mean(np.sum((answer_probability - onehot) ** 2, axis=1)))
    answer_metrics["confidence_ece_10"] = ece(confidence, correct)

    source_truth = np.asarray([row["evaluation"]["source_multihot_5"] for row in predictions], dtype=bool)
    source_probability = np.asarray([
        [row["source_probabilities"][name] for name in SOURCE_CLASSES] for row in predictions
    ], dtype=float)
    source_rows = {
        name: binary_metrics(source_truth[:, index], source_probability[:, index], 0.5)
        for index, name in enumerate(SOURCE_CLASSES)
    }
    source_prediction = source_probability >= 0.5
    tp = int((source_truth & source_prediction).sum())
    fp = int(((~source_truth) & source_prediction).sum())
    fn = int((source_truth & (~source_prediction)).sum())
    source_metrics = {
        "threshold_policy": "frozen_v2_default_0.5_no_test_tuning",
        "micro_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "macro_f1": float(np.mean([row["f1"] for row in source_rows.values()])),
        "macro_roc_auc": float(np.mean([row["roc_auc"] for row in source_rows.values()])),
        "macro_average_precision": float(np.mean([row["average_precision"] for row in source_rows.values()])),
        "per_source": source_rows,
    }

    risk = np.asarray([row["risk"] for row in rows], dtype=float)
    errors = np.asarray([row["error"] for row in rows], dtype=float)
    risk_threshold = float(calibrator["risk_policy"]["threshold"])
    accepted = risk <= risk_threshold
    order = np.argsort(risk, kind="mergesort")
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    oracle_order = np.argsort(errors, kind="mergesort")
    oracle_curve = np.cumsum(errors[oracle_order]) / np.arange(1, len(errors) + 1)
    risk_metrics = {
        "definition": "error = non-FOUND ground truth OR MAP point outside target",
        "positive_error_rate": float(errors.mean()),
        "brier": float(np.mean((risk - errors) ** 2)),
        "nll": nll_binary(risk, errors), "ece_10": ece(risk, errors),
        "roc_auc": roc_auc(errors.astype(bool), risk),
        "average_precision": average_precision(errors.astype(bool), risk),
        "aurc": float(cumulative_risk.mean()),
        "oracle_aurc": float(oracle_curve.mean()),
        "excess_aurc": float(cumulative_risk.mean() - oracle_curve.mean()),
        "frozen_threshold": risk_threshold,
        "coverage_at_frozen_threshold": float(accepted.mean()),
        "accepted_samples": int(accepted.sum()),
        "accepted_families": len({row["family_id"] for row, keep in zip(rows, accepted) if keep}),
        "selective_risk_at_frozen_threshold": float(errors[accepted].mean()) if accepted.any() else None,
    }

    actions = Counter(row["action"] for row in rows)
    executes = [row for row in rows if row["action"] == "EXECUTE"]
    policy = {
        "action_counts": dict(sorted(actions.items())),
        "action_rates": {key: value / len(rows) for key, value in sorted(actions.items())},
        "expected_action_accuracy": float(np.mean([row["action_correct"] for row in rows])),
        "execute_count": len(executes), "execute_rate": len(executes) / len(rows),
        "execute_error_rate": float(np.mean([row["error"] for row in executes])) if executes else None,
        "execute_grounding_accuracy": float(np.mean([row["inside"] for row in executes])) if executes else None,
        "unsafe_execute_non_found_count": sum(row["answerability_state"] != "FOUND" for row in executes),
    }
    distances = np.asarray(distance_values, dtype=float)
    grounding = {
        "evaluated": len(distance_values),
        "point_in_target_accuracy": float(np.mean([row["inside"] for row in rows if row["target_exists"]])),
        "point_to_target_distance_px": {
            "mean": float(distances.mean()), "median": float(np.median(distances)),
            "p95": float(np.quantile(distances, 0.95)), "maximum": float(distances.max()),
            "mean_normalized_image_diagonal": float(distances.mean() / 800.0),
        },
        "mean_target_probability_mass": float(np.mean(target_mass)),
    }
    conformal = {
        "alpha": float(calibrator["conformal"]["alpha"]),
        "frozen_probability_mass": conformal_mass,
        "target_samples": len(conformal_hits),
        "empirical_coverage": float(np.mean(conformal_hits)),
        "target_coverage_nominal": 1.0 - float(calibrator["conformal"]["alpha"]),
        "mean_grid_area_fraction": float(np.mean(region_areas)),
        "median_grid_area_fraction": float(np.median(region_areas)),
    }
    strata = stratum_table(rows, [
        "variant", "relation", "family_category", "target_object_group",
        "answerability_state", "state_submode",
    ])
    report = {
        "schema_version": 1,
        "protocol_id": "roborefer_dataset_v2_1_test_iid_capture_200",
        "status": "OFFICIAL_FROZEN_BEST_V2_TEST_IID_EVALUATION_COMPLETE",
        "family_count": 200, "sample_count": 1000,
        "checkpoint": {
            "path": selected["path"], "sha256": selected["model_sha256"],
            "epoch": selected["epoch"], "global_step": selected["global_step"],
        },
        "calibrator": {"path": freeze["frozen_calibrator"]["path"],
                       "sha256": freeze["frozen_calibrator"]["sha256"]},
        "loss": base["loss"], "grounding": grounding,
        "answerability": answer_metrics, "source_uncertainty": source_metrics,
        "relation_edge": {"accuracy": base["relation_edge_accuracy"],
                          "evaluated_edges": base["relation_edges_evaluated"]},
        "risk_calibration_and_ranking": risk_metrics,
        "conformal_spatial_region": conformal,
        "selective_policy": policy, "strata": strata,
        "scientific_boundary": {
            "test_tuning_performed": False, "checkpoint_selection_on_test": False,
            "calibrator_refit_on_test": False, "source_threshold_refit_on_test": False,
            "source_threshold_note": "Frozen V2 calibrator predates per-source thresholds; 0.5 retained.",
        },
        "artifacts": {
            "predictions_sha256": sha256_file(PREDICTIONS),
            "decisions_sha256": sha256_file(DECISIONS),
            "feature_index_sha256": sha256_file(FEATURE_INDEX),
            "eval_manifest_sha256": sha256_file(EVAL_MANIFEST),
        },
    }
    atomic_json(EVAL_ROOT / "full_metrics.json", report)

    csv_path = EVAL_ROOT / "stratified_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dimension", "value", "samples", "grounding_evaluated", "grounding_accuracy",
                         "answerability_accuracy", "mean_calibrated_risk", "observed_error_rate",
                         "policy_action_accuracy"])
        for dimension, values in strata.items():
            for value, metrics in values.items():
                writer.writerow([dimension, value, *[metrics[key] for key in (
                    "samples", "grounding_evaluated", "grounding_accuracy", "answerability_accuracy",
                    "mean_calibrated_risk", "observed_error_rate", "policy_action_accuracy",
                )]])

    markdown = f"""# Best V2 — Test-IID frozen evaluation

- Samples/families: **1,000 / 200**
- Grounding point-in-target: **{grounding['point_in_target_accuracy']:.2%}** ({grounding['evaluated']} target-present samples)
- Mean / median point-to-target error: **{grounding['point_to_target_distance_px']['mean']:.2f} / {grounding['point_to_target_distance_px']['median']:.2f} px**
- Answerability accuracy / macro-F1: **{answer_metrics['accuracy']:.2%} / {answer_metrics['macro_f1']:.2%}**
- Source uncertainty micro/macro-F1: **{source_metrics['micro_f1']:.2%} / {source_metrics['macro_f1']:.2%}**
- Relation-edge accuracy: **{base['relation_edge_accuracy']:.2%}** ({base['relation_edges_evaluated']} edges)
- Risk AUROC / AUPRC: **{risk_metrics['roc_auc']:.4f} / {risk_metrics['average_precision']:.4f}**
- Risk calibration Brier / NLL / ECE: **{risk_metrics['brier']:.4f} / {risk_metrics['nll']:.4f} / {risk_metrics['ece_10']:.4f}**
- Frozen-threshold coverage / selective risk: **{risk_metrics['coverage_at_frozen_threshold']:.2%} / {risk_metrics['selective_risk_at_frozen_threshold']:.2%}**
- EXECUTE count / error rate: **{policy['execute_count']} / {policy['execute_error_rate']:.2%}**
- Conformal region empirical/nominal coverage: **{conformal['empirical_coverage']:.2%} / {conformal['target_coverage_nominal']:.2%}**

No model, calibrator, threshold, or prompt was fitted on Test-IID.
"""
    (EVAL_ROOT / "SUMMARY.md").write_text(markdown, encoding="utf-8")
    return report


if __name__ == "__main__":
    report = analyze()
    print(json.dumps({
        "status": report["status"],
        "grounding": report["grounding"]["point_in_target_accuracy"],
        "answerability_accuracy": report["answerability"]["accuracy"],
        "answerability_macro_f1": report["answerability"]["macro_f1"],
        "source_micro_f1": report["source_uncertainty"]["micro_f1"],
        "source_macro_f1": report["source_uncertainty"]["macro_f1"],
        "risk": report["risk_calibration_and_ranking"],
        "policy": report["selective_policy"],
        "conformal": report["conformal_spatial_region"],
    }, ensure_ascii=False, indent=2))
