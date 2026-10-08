from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


FEATURE_NAMES = [
    "heatmap_entropy",
    "peak_margin",
    "mode_count",
    "found_probability",
    "ambiguous_probability",
    "absent_probability",
    "insufficient_probability",
    "semantic_uncertainty",
    "relation_uncertainty",
    "spatial_uncertainty",
    "depth_uncertainty",
    "occlusion_uncertainty",
    "relation_consistency",
    "fusion_gate_mean",
    "rgb_depth_cosine",
    "rgb_depth_mae",
    "roborefer_disagreement",
    "roborefer_missing",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evidence_vector(row: Mapping[str, Any]) -> np.ndarray:
    answer = row["answerability_probabilities"]
    source = row["source_probabilities"]
    disagreement = row.get("roborefer_disagreement_normalized")
    values = [
        row["spatial"]["entropy_normalized"],
        row["spatial"]["peak_margin"],
        row["spatial"]["mode_count"],
        answer["FOUND"],
        answer["AMBIGUOUS"],
        answer["ABSENT"],
        answer["INSUFFICIENT_EVIDENCE"],
        source["semantic"],
        source["relation"],
        source["spatial"],
        source["depth"],
        source["occlusion"],
        row["relation_consistency"],
        row["fusion_gate_mean"],
        row["rgb_depth_cosine"],
        row["rgb_depth_mae"],
        0.0 if disagreement is None else disagreement,
        float(disagreement is None),
    ]
    return np.asarray(values, dtype=np.float64)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, iterations: int = 100) -> tuple[np.ndarray, float]:
    """Deterministic damped Newton fit with an unregularized intercept."""
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        probability = _sigmoid(design @ weights)
        gradient = design.T @ (probability - y) / len(y) + penalty @ weights
        curvature = probability * (1.0 - probability)
        hessian = (design.T * curvature) @ design / len(y) + penalty
        hessian += np.eye(design.shape[1]) * 1e-8
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return weights[1:], float(weights[0])


def predict_risk(x: np.ndarray, coefficients: np.ndarray, intercept: float) -> np.ndarray:
    return _sigmoid(x @ coefficients + intercept)


def _fold(family_id: str, folds: int) -> int:
    digest = hashlib.sha256(family_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def expected_calibration_error(probability: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for low in np.linspace(0.0, 1.0, bins, endpoint=False):
        high = low + 1.0 / bins
        selected = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if selected.any():
            result += selected.mean() * abs(float(probability[selected].mean() - truth[selected].mean()))
    return float(result)


def _wilson_upper(errors: int, count: int, z: float = 1.96) -> float:
    if count == 0:
        return 1.0
    rate = errors / count
    denominator = 1.0 + z * z / count
    center = rate + z * z / (2.0 * count)
    radius = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
    return min(1.0, (center + radius) / denominator)


def choose_risk_threshold(
    probability: np.ndarray,
    truth: np.ndarray,
    families: list[str],
    risk_target: float,
    minimum_families: int,
) -> dict[str, Any]:
    candidates = sorted(set(float(value) for value in probability))
    best: dict[str, Any] | None = None
    for threshold in candidates:
        accepted = probability <= threshold
        family_count = len({family for family, keep in zip(families, accepted) if keep})
        count = int(accepted.sum())
        errors = int(truth[accepted].sum())
        upper = _wilson_upper(errors, count)
        if family_count >= minimum_families and upper <= risk_target:
            best = {
                "threshold": threshold,
                "accepted_samples": count,
                "accepted_families": family_count,
                "empirical_risk": errors / max(1, count),
                "wilson_upper_95": upper,
                "coverage": count / len(truth),
            }
    if best is None:
        return {
            "threshold": None,
            "accepted_samples": 0,
            "accepted_families": 0,
            "empirical_risk": None,
            "wilson_upper_95": None,
            "coverage": 0.0,
            "status": "NO_THRESHOLD_MEETS_DECLARED_RISK",
        }
    best["status"] = "PASS"
    return best


def fit_source_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    source_names = ["semantic", "relation", "spatial", "depth", "occlusion"]
    for index, name in enumerate(source_names):
        truth = np.asarray(
            [bool(row["evaluation"]["source_multihot_5"][index]) for row in rows], dtype=bool
        )
        probability = np.asarray([row["source_probabilities"][name] for row in rows])
        candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], probability]))
        best = None
        for threshold in candidates:
            prediction = probability >= threshold
            tp = int((truth & prediction).sum())
            fp = int(((~truth) & prediction).sum())
            fn = int((truth & (~prediction)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            candidate = (f1, precision, -abs(float(threshold) - 0.5), float(threshold), recall)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        assert best is not None
        result[name] = {
            "threshold": best[3], "calibration_f1": best[0],
            "calibration_precision": best[1], "calibration_recall": best[4],
            "status": "CALIBRATION_FIT_ONLY",
        }
    return result


def fit_calibrator(rows: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if not rows or any("evaluation" not in row for row in rows):
        raise ValueError("Calibration predictions must contain evaluator-joined labels")
    x = np.stack([evidence_vector(row) for row in rows])
    y = np.asarray([row["evaluation"]["error_event"] for row in rows], dtype=np.float64)
    families = [row["family_id"] for row in rows]
    folds = min(5, len(set(families)))
    crossfit = np.full(len(rows), np.nan, dtype=np.float64)
    for fold in range(folds):
        held = np.asarray([_fold(family, folds) == fold for family in families])
        train = ~held
        if not held.any():
            continue
        fold_mean = x[train].mean(0)
        fold_scale = x[train].std(0)
        fold_scale[fold_scale < 1e-8] = 1.0
        if len(np.unique(y[train])) < 2:
            crossfit[held] = float(y[train].mean())
            continue
        coefficients, intercept = fit_logistic(
            (x[train] - fold_mean) / fold_scale, y[train], float(config["calibration"]["l2"])
        )
        crossfit[held] = predict_risk((x[held] - fold_mean) / fold_scale, coefficients, intercept)
    if np.isnan(crossfit).any():
        raise ValueError("Family cross-fitting left calibration rows unassigned")
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-8] = 1.0
    standardized = (x - mean) / scale
    coefficients, intercept = fit_logistic(standardized, y, float(config["calibration"]["l2"]))
    final_probability = predict_risk(standardized, coefficients, intercept)
    threshold = choose_risk_threshold(
        crossfit, y, families, float(config["calibration"]["risk_target"]),
        int(config["calibration"]["minimum_accepted_families"]),
    )
    alpha = float(config["calibration"]["conformal_alpha"])
    required_masses = [
        float(row["evaluation"]["target_required_hd_mass"])
        for row in rows if row["evaluation"]["target_exists"]
    ]
    conformity_scores = np.sort(np.asarray(required_masses, dtype=np.float64))
    conformal_index = min(len(conformity_scores) - 1, math.ceil((len(conformity_scores) + 1) * (1.0 - alpha)) - 1)
    conformal_mass = float(conformity_scores[conformal_index]) if len(conformity_scores) else None
    return {
        "schema_version": 1,
        "method": "standardized_logistic_family_5fold_crossfit",
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": intercept,
        "samples": len(rows),
        "families": len(set(families)),
        "positive_error_rate": float(y.mean()),
        "crossfit_metrics": {
            "brier": float(np.mean((crossfit - y) ** 2)),
            "nll": float(-np.mean(y * np.log(crossfit.clip(1e-7, 1 - 1e-7)) + (1 - y) * np.log((1 - crossfit).clip(1e-7, 1 - 1e-7)))),
            "ece": expected_calibration_error(crossfit, y),
        },
        "risk_policy": threshold,
        "source_thresholds": fit_source_thresholds(rows),
        "conformal": {
            "alpha": alpha,
            "coverage_event": "highest-density grid region intersects target mask",
            "calibration_target_samples": len(conformity_scores),
            "probability_mass": conformal_mass,
        },
        "fit_probability_summary": {
            "minimum": float(final_probability.min()),
            "median": float(np.median(final_probability)),
            "maximum": float(final_probability.max()),
        },
    }


def apply_calibrator(row: Mapping[str, Any], calibrator: Mapping[str, Any]) -> float:
    vector = evidence_vector(row)
    mean = np.asarray(calibrator["mean"], dtype=np.float64)
    scale = np.asarray(calibrator["scale"], dtype=np.float64)
    coefficients = np.asarray(calibrator["coefficients"], dtype=np.float64)
    probability = predict_risk(((vector - mean) / scale)[None], coefficients, float(calibrator["intercept"]))
    return float(probability[0])
