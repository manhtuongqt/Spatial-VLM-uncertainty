"""Train mining and separately split post-freeze calibration experiments."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .calibration import (
    FEATURE_NAMES, _wilson_upper, evidence_vector, expected_calibration_error,
    fit_logistic, predict_risk,
)


EXTRA_RISK_FEATURES = [
    "target_sigmoid_max", "target_sigmoid_mean", "spatial_peak", "spatial_top5_mass",
    "spatial_top10_mass", "interior_sigmoid_max", "interior_sigmoid_mean",
    "target_interior_peak_distance", "anchor_sigmoid_max_mean", "anchor_sigmoid_max_min",
    "edge_probability_min", "base_found", "base_ambiguous", "base_absent", "base_insufficient_evidence",
]


def vector(row, extended=False, ranker_feature=False):
    base = evidence_vector(row)
    if extended:
        observed = row["observable_answerability_evidence"]
        base = np.concatenate([base, np.asarray([observed[name] for name in EXTRA_RISK_FEATURES])])
    if ranker_feature:
        base = np.concatenate([base, np.asarray([row["ranker_error_logit"]])])
    return base


def fit_risk(rows, extended=False, l2=0.001, folds=5, ranker_feature=False):
    families = [row["family_id"] for row in rows]
    unique = sorted(set(families), key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    if len(unique) < 2:
        raise ValueError("Risk fitting requires at least two families")
    folds = min(folds, len(unique))
    assignment = {family: index % folds for index, family in enumerate(unique)}
    x = np.stack([vector(row, extended, ranker_feature) for row in rows])
    y = np.asarray([row["evaluation"]["error_event"] for row in rows], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite risk evidence")
    crossfit = np.full(len(rows), np.nan)
    for fold in range(folds):
        held = np.asarray([assignment[family] == fold for family in families])
        train = ~held
        mean, scale = x[train].mean(0), x[train].std(0)
        scale[scale < 1e-8] = 1.0
        if len(np.unique(y[train])) == 1:
            crossfit[held] = y[train].mean()
        else:
            coefficient, intercept = fit_logistic((x[train]-mean)/scale, y[train], l2)
            crossfit[held] = predict_risk((x[held]-mean)/scale, coefficient, intercept)
    if not np.isfinite(crossfit).all():
        raise ValueError("Invalid crossfit risks")
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1.0
    coefficient, intercept = fit_logistic((x-mean)/scale, y, l2)
    result = {
        "method": "logistic_family_crossfit_fit_partition_only", "extended": extended, "l2": l2,
        "feature_names": FEATURE_NAMES + (EXTRA_RISK_FEATURES if extended else []) + (["ranker_error_logit"] if ranker_feature else []),
        "mean": mean.tolist(), "scale": scale.tolist(), "coefficients": coefficient.tolist(),
        "intercept": intercept, "fit_samples": len(rows), "fit_families": len(unique),
        "crossfit_metrics": probability_metrics(crossfit, y),
        "family_folds": assignment,
    }
    if ranker_feature:
        result["ranker_feature"] = True
    return result, crossfit


def apply_risk(rows, calibrator):
    if calibrator.get('method') == 'logistic_support_scalar_split':
        x = np.asarray([[row['ranker_error_logit']] for row in rows], dtype=float)
    else:
        x = np.stack([vector(row, calibrator["extended"], calibrator.get("ranker_feature", False)) for row in rows])
    mean, scale, coefficients = [np.asarray(calibrator[key]) for key in ("mean", "scale", "coefficients")]
    return predict_risk((x-mean)/scale, coefficients, calibrator["intercept"])


def probability_metrics(risk, error):
    risk, error = np.asarray(risk), np.asarray(error)
    clipped = risk.clip(1e-7, 1-1e-7)
    order = np.argsort(risk, kind="stable")
    selective = np.cumsum(error[order]) / np.arange(1, len(error)+1)
    return {"brier": float(np.mean((risk-error)**2)),
            "nll": float(-np.mean(error*np.log(clipped)+(1-error)*np.log(1-clipped))),
            "ece_10": expected_calibration_error(risk, error), "aurc": float(selective.mean())}


def choose_policy_threshold(risk, error, families, eligible, target, minimum_families=20):
    risk, error, eligible = np.asarray(risk), np.asarray(error), np.asarray(eligible, dtype=bool)
    if not 0 < target < 1:
        raise ValueError("Risk target must be between zero and one")
    best = {"status": "NO_THRESHOLD_MEETS_DECLARED_RISK", "threshold": None, "coverage": 0.0,
            "accepted_samples": 0, "accepted_families": 0, "errors": 0,
            "empirical_risk": None, "wilson_upper_95": None}
    for threshold in np.unique(risk[eligible]):
        accepted = eligible & (risk <= threshold)
        count, errors = int(accepted.sum()), int(error[accepted].sum())
        family_count = len({family for family, keep in zip(families, accepted) if keep})
        upper = _wilson_upper(errors, count)
        if family_count >= minimum_families and upper <= target:
            best = {"status": "PASS", "threshold": float(threshold), "coverage": count/len(risk),
                    "accepted_samples": count, "accepted_families": family_count, "errors": errors,
                    "empirical_risk": errors/count, "wilson_upper_95": upper}
    return {"risk_target": target, "minimum_accepted_families": minimum_families, **best}


def predicted_answer(row):
    return max(row["answerability_probabilities"], key=row["answerability_probabilities"].get)


def mine_hard_examples(rows, proxy_risk, threshold):
    probability = np.asarray(proxy_risk)
    answers = np.asarray([predicted_answer(row) for row in rows])
    correct = np.asarray([not row["evaluation"]["error_event"] for row in rows])
    truth_found = np.asarray([row["evaluation"]["answerability_state"] == "FOUND" for row in rows])
    high = probability > threshold if threshold is not None else probability >= 0.1
    low = ~high
    positive = correct & ((answers != "FOUND") | high)
    negative = (~truth_found) & ((answers == "FOUND") | low)
    location_failure = truth_found & ~correct
    # Localization failures retain FOUND class supervision; their binary risk label is error.
    return positive, negative, location_failure


def weighted_answer_loss(logits, labels, class_weights, example_weights):
    terms = F.cross_entropy(logits.float(), labels, weight=class_weights, reduction="none")
    denominator = (class_weights[labels]*example_weights).sum().clamp_min(1e-8)
    return (terms*example_weights).sum()/denominator


def experimental_action(row, risk, policy, threshold):
    """Annotation-free action rule for offline experimental profiles."""
    if policy not in {"hard_found", "risk_only"}:
        raise ValueError("Unknown experimental policy")
    answer = predicted_answer(row)
    accepted = (policy == "risk_only" or answer == "FOUND") and threshold is not None and risk <= threshold
    if accepted:
        return "EXECUTE"
    if answer == "ABSENT":
        return "ABSTAIN"
    if answer == "AMBIGUOUS":
        return "ASK_USER"
    if answer == "INSUFFICIENT_EVIDENCE":
        return "REOBSERVE"
    source = row["source_probabilities"]
    return "ASK_USER" if source["semantic"] >= max(source["depth"],source["occlusion"]) else "REOBSERVE"


def policy_result(rows, risk, policy, threshold):
    records = []
    for row, value in zip(rows, risk):
        answer = predicted_answer(row)
        action = experimental_action(row, float(value), policy, threshold)
        records.append({"sample_id": row["sample_id"], "family_id": row["family_id"],
                        "predicted_answerability": answer, "action": action, "risk": float(value),
                        "threshold": threshold, "policy": policy,
                        "evaluation": row["evaluation"]})
    accepted = np.asarray([r["action"] == "EXECUTE" for r in records])
    errors = np.asarray([row["evaluation"]["error_event"] for row in rows], dtype=bool)
    found = np.asarray([row["evaluation"]["answerability_state"] == "FOUND" for row in rows])
    count, error_count = int(accepted.sum()), int(errors[accepted].sum())
    safe = found & ~errors
    bypass = accepted & np.asarray([predicted_answer(row) != "FOUND" for row in rows])
    false_found = accepted & ~found
    metrics = {"samples": len(rows), "accepted": count, "coverage": count/len(rows),
               "errors": error_count, "empirical_risk": error_count/count if count else None,
               "wilson_upper_95": _wilson_upper(error_count, count) if count else None,
               "correct_acceptances": int((accepted & ~errors).sum()),
               "correct_found_rejected": int((safe & ~accepted).sum()),
               "correct_found_total": int(safe.sum()),
               "accepted_non_found_truth": int(false_found.sum()),
               "accepted_absent_truth": sum(r["action"]=="EXECUTE" and r["evaluation"]["answerability_state"]=="ABSENT" for r in records),
               "answer_gate_bypassed": int(bypass.sum()), "bypass_correct": int((bypass & ~errors).sum()),
               "bypass_errors": int((bypass & errors).sum()),
               "actions": {a: sum(r["action"]==a for r in records) for a in ("EXECUTE","REOBSERVE","ASK_USER","ABSTAIN")}}
    return metrics, records
