#!/usr/bin/env python3
"""Fit one locked temperature scaler and freeze the final operating point."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import wp5_spatial_uncertainty_estimator as est


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/Gazebo_calibration_v1"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v1"
PREDICTIONS = RESULT / "b0_predictions/predictions.jsonl"
PREDICTION_RUN = RESULT / "b0_predictions/run.json"
CONTRACT = ROOT / "protocol/GAZEBO_CALIBRATION_CONTRACT_LOCK.json"
MATERIALIZATION_LOCK = ROOT / "protocol/gazebo_calibration_v1_materialization_lock.json"
WP5 = ROOT / "results/spatial_vlm_refspatial_v1/wp5_spatial_uncertainty"
ESTIMATOR = WP5 / "selected_estimator.json"
ESTIMATOR_LOCK = WP5 / "wp5_checkpoint_or_estimator_lock.json"
FIT_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_FIT_INPUT_LOCK.json"
CALIBRATOR = RESULT / "temperature_calibrator.json"
METRICS = RESULT / "GAZEBO_CALIBRATION_METRICS.json"
REPORT = RESULT / "GAZEBO_CALIBRATION_RESULT.md"
FINAL_LOCK = RESULT / "CALIBRATOR_THRESHOLD_LOCK.json"
EPS = 1e-6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sigmoid(value):
    value = np.asarray(value, dtype=float)
    return np.where(value >= 0, 1 / (1 + np.exp(-value)), np.exp(value) / (1 + np.exp(value)))


def raw_risk(data):
    model = json.loads(ESTIMATOR.read_text())
    if model["C"] != 0.1 or len(model["feature_names"]) != 16:
        raise ValueError("frozen estimator contract mismatch")
    X = np.stack([row["feature"] for row in data])
    mean, scale = np.asarray(model["scaler_mean"]), np.asarray(model["scaler_scale"])
    coefficient = np.asarray(model["coefficient"])
    return sigmoid(((X - mean) / scale) @ coefficient + float(model["intercept"]))


def nll(log_t: float, logits, labels) -> float:
    probability = np.clip(sigmoid(logits / math.exp(log_t)), EPS, 1 - EPS)
    labels = np.asarray(labels, dtype=float)
    return float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)))


def golden_temperature(logits, labels):
    left, right = -5.0, 5.0
    ratio = (math.sqrt(5) - 1) / 2
    c, d = right - ratio * (right - left), left + ratio * (right - left)
    fc, fd = nll(c, logits, labels), nll(d, logits, labels)
    for _ in range(200):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left); fc = nll(c, logits, labels)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left); fd = nll(d, logits, labels)
    log_t = (left + right) / 2
    return math.exp(log_t), nll(log_t, logits, labels)


def corrected_safe(data, risk, threshold):
    decisions = est.decisions(data, risk, threshold)
    return float(np.mean([
        (row["action"] == "POINT" and row["hit"]) if row["state"] == "FOUND"
        else row["action"] == "ABSTAIN" for row in decisions
    ]))


def metric(data, risk, threshold):
    value = est.metrics(data, risk, threshold)
    value["safe_task_accuracy"] = corrected_safe(data, risk, threshold)
    return value


def lock_fit_inputs():
    if FIT_LOCK.exists():
        raise FileExistsError("refusing to overwrite calibration fit-input lock")
    manifest = json.loads((DATA / "manifest.json").read_text())
    run = json.loads(PREDICTION_RUN.read_text())
    predictions = est.rows(PREDICTIONS)
    if manifest.get("status") != "PASS" or manifest.get("records") != 128:
        raise ValueError("Calibration dataset is not a 128/128 PASS materialization")
    if run.get("status") != "COMPLETED" or len(predictions) != 128 or any("error" in row for row in predictions):
        raise ValueError("frozen B0 Calibration inference is incomplete")
    source_paths = [Path(__file__).resolve(), CONTRACT, MATERIALIZATION_LOCK, DATA / "manifest.json",
                    DATA / "inference_manifest.jsonl", DATA / "evaluator_ground_truth.jsonl",
                    PREDICTIONS, PREDICTION_RUN, ESTIMATOR, ESTIMATOR_LOCK,
                    ROOT / "protocol/wp5_spatial_uncertainty_estimator.py"]
    payload = {"schema_version": 1, "protocol_id": "gazebo_calibration_v1",
               "status": "LOCKED_AFTER_PREDICTION_BEFORE_ORACLE_JOIN_AND_CALIBRATOR_FIT",
               "locked_at_utc": datetime.now(timezone.utc).isoformat(),
               "prediction_count": 128, "calibrator_count": 1, "test_iid_ood_sealed": True,
               "source_artifact_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths}}
    dump(FIT_LOCK, payload)
    print(json.dumps({"status": payload["status"], "sha256": sha(FIT_LOCK)}, indent=2))


def verify_fit_lock():
    lock = json.loads(FIT_LOCK.read_text())
    if lock.get("status") != "LOCKED_AFTER_PREDICTION_BEFORE_ORACLE_JOIN_AND_CALIBRATOR_FIT":
        raise ValueError("invalid fit-input lock")
    for name, digest in lock["source_artifact_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"fit input changed: {name}")


def reliability(labels, risks):
    rows = []
    labels, risks = np.asarray(labels), np.asarray(risks)
    for index in range(10):
        lo, hi = index / 10, (index + 1) / 10
        mask = (risks >= lo) & ((risks < hi) if index < 9 else (risks <= hi))
        if mask.any():
            rows.append({"bin": index, "lower": lo, "upper": hi, "count": int(mask.sum()),
                         "mean_probability": float(risks[mask].mean()), "unsafe_frequency": float(labels[mask].mean())})
    return rows


def fit():
    verify_fit_lock()
    if any(path.exists() for path in (CALIBRATOR, METRICS, REPORT, FINAL_LOCK)):
        raise FileExistsError("refusing to overwrite Calibration result")
    data = est.joined(DATA, PREDICTIONS)
    if len(data) != 128 or any(row["split"] != "calibration" for row in data):
        raise ValueError("unexpected Calibration rows")
    raw = np.asarray(raw_risk(data), dtype=float)
    labels = np.asarray([row["unsafe"] for row in data], dtype=int)
    logits = np.log(np.clip(raw, EPS, 1 - EPS) / np.clip(1 - raw, EPS, 1 - EPS))
    temperature, calibrated_nll = golden_temperature(logits, labels)
    calibrated = sigmoid(logits / temperature)
    raw_probability = metric(data, raw, 0.72)
    calibrated_probability = metric(data, calibrated, 0.72)
    b0_risk = est.baseline_risk(data)
    b0 = metric(data, b0_risk, 1.01)
    candidates = []
    for threshold in np.linspace(0, 1, 101):
        m = metric(data, calibrated, float(threshold))
        eligible = (m["found_hit_at_008"] - b0["found_hit_at_008"] >= -0.125 - 1e-12
                    and m["nonfound_abstain_recall"] > b0["nonfound_abstain_recall"]
                    and m["nonfound_false_accept"] < b0["nonfound_false_accept"]
                    and m["exact_contract"] >= b0["exact_contract"])
        if eligible:
            candidates.append((m["nonfound_false_accept"], -m["found_hit_at_008"],
                               -m["safe_task_accuracy"], -m["coverage"], float(threshold), m))
    selected = min(candidates) if candidates else None
    threshold, operating = (selected[4], selected[5]) if selected else (None, None)
    ranking_invariant = (abs(raw_probability["aurc"] - calibrated_probability["aurc"]) <= 1e-12
                         and abs(raw_probability["auroc_error"] - calibrated_probability["auroc_error"]) <= 1e-12)
    probability_pass = (calibrated_probability["ece_10bin"] < raw_probability["ece_10bin"]
                        and calibrated_probability["brier"] < raw_probability["brier"] and ranking_invariant)
    passed = bool(probability_pass and selected)
    status = "CALIBRATION_PASS_READY_FOR_LOCKED_TEST" if passed else "CALIBRATION_NEGATIVE_NO_ROBOT_DEPLOYMENT"
    calibrator = {"schema_version": 1, "type": "binary_temperature_scaling_on_clipped_logit",
                  "temperature": temperature, "logit_clip_epsilon": EPS, "log_T_bounds": [-5.0, 5.0],
                  "optimizer": "deterministic_golden_section", "iterations": 200,
                  "fit_objective": "binary_negative_log_likelihood", "fit_nll": calibrated_nll,
                  "operating_threshold": threshold, "status": status}
    dump(CALIBRATOR, calibrator)
    scored = []
    for row, r0, r1 in zip(data, raw, calibrated):
        action = "ABSTAIN" if threshold is not None and r1 >= threshold else row["b0"]["action"]
        scored.append({"sample_id": row["sample_id"], "family_id": row["family_id"],
                       "state": row["state"], "relation_variant": row["relation"], "unsafe": row["unsafe"],
                       "raw_risk": float(r0), "calibrated_risk": float(r1), "action": action,
                       "grounding_hit_at_008": row["hit"]})
    with (RESULT / "calibration_scored_predictions.jsonl").open("w") as stream:
        for row in scored:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    payload = {"schema_version": 1, "status": status, "records": 128,
               "unsafe_count": int(labels.sum()), "safe_count": int((1 - labels).sum()),
               "temperature": temperature, "selected_threshold": threshold,
               "gates": {"ece_strictly_lower": calibrated_probability["ece_10bin"] < raw_probability["ece_10bin"],
                         "brier_strictly_lower": calibrated_probability["brier"] < raw_probability["brier"],
                         "ranking_invariant": ranking_invariant, "eligible_operating_point_exists": bool(selected)},
               "b0": b0, "raw_wp5_at_val_threshold_0_72": raw_probability,
               "calibrated_at_reference_threshold_0_72": calibrated_probability,
               "calibrated_selected_operating_point": operating,
               "reliability": {"raw": reliability(labels, raw), "calibrated": reliability(labels, calibrated)},
               "scope": "Calibration fit evidence only; no generalization or robot-deployment claim",
               "test_iid_ood_access": False}
    dump(METRICS, payload)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    for name, values, marker in (("raw", raw, "o"), ("temperature-scaled", calibrated, "s")):
        bins = reliability(labels, values)
        ax.plot([x["mean_probability"] for x in bins], [x["unsafe_frequency"] for x in bins], marker=marker, label=name)
    ax.set(xlabel="Predicted unsafe probability", ylabel="Observed unsafe frequency", xlim=(0, 1), ylim=(0, 1),
           title="Calibration-v1 reliability")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULT / "CALIBRATION_RELIABILITY.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in (("raw", raw), ("temperature-scaled", calibrated)):
        curve = est.curve(labels, values)
        ax.plot([x["coverage"] for x in curve], [x["risk"] for x in curve], label=name)
    ax.set(xlabel="Coverage", ylabel="Selective risk", title="Calibration-v1 risk–coverage")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(RESULT / "CALIBRATION_RISK_COVERAGE.png", dpi=180); plt.close(fig)
    op = operating or {key: None for key in ("found_hit_at_008", "nonfound_false_accept", "nonfound_abstain_recall", "coverage")}
    REPORT.write_text(f'''# Gazebo Final Calibration v1

## Decision

**{status}**

Exactly one preregistered binary temperature scaler was fitted on 128 independent Calibration families. Frozen B0, the 16-feature schema, Logistic-L2 (`C=0.1`), and all Test splits remained unchanged and sealed.

| Metric | Raw WP5 risk | Temperature-scaled risk |
|---|---:|---:|
| ECE-10 | {raw_probability['ece_10bin']:.6f} | {calibrated_probability['ece_10bin']:.6f} |
| Brier | {raw_probability['brier']:.6f} | {calibrated_probability['brier']:.6f} |
| AURC | {raw_probability['aurc']:.6f} | {calibrated_probability['aurc']:.6f} |
| AUROC-error | {raw_probability['auroc_error']:.6f} | {calibrated_probability['auroc_error']:.6f} |

Temperature: `{temperature:.9f}`. Selected operating threshold: `{threshold}`.

At the selected operating point: false accept `{op['nonfound_false_accept']}`, abstain recall `{op['nonfound_abstain_recall']}`, FOUND Hit@.08 `{op['found_hit_at_008']}`, coverage `{op['coverage']}`.

This is calibration-fit evidence, not a Test-IID/OOD generalization result and not authorization for robot deployment.
''')
    print(json.dumps({"status": status, "temperature": temperature, "threshold": threshold,
                      "ece_raw": raw_probability["ece_10bin"], "ece_calibrated": calibrated_probability["ece_10bin"],
                      "brier_raw": raw_probability["brier"], "brier_calibrated": calibrated_probability["brier"]}, indent=2))


def freeze():
    verify_fit_lock()
    if FINAL_LOCK.exists():
        raise FileExistsError("refusing to overwrite final calibrator lock")
    metrics = json.loads(METRICS.read_text())
    calibrator = json.loads(CALIBRATOR.read_text())
    artifacts = [FIT_LOCK, CONTRACT, MATERIALIZATION_LOCK, ESTIMATOR, ESTIMATOR_LOCK,
                 CALIBRATOR, METRICS, REPORT, RESULT / "calibration_scored_predictions.jsonl",
                 RESULT / "CALIBRATION_RELIABILITY.png", RESULT / "CALIBRATION_RISK_COVERAGE.png"]
    payload = {"schema_version": 1, "protocol_id": "gazebo_calibration_v1", "status": metrics["status"],
               "locked_at_utc": datetime.now(timezone.utc).isoformat(),
               "temperature": calibrator["temperature"], "operating_threshold": calibrator["operating_threshold"],
               "decision_rule": "ABSTAIN iff calibrated unsafe probability >= operating_threshold; otherwise preserve frozen B0 action",
               "frozen_b0": True, "frozen_feature_schema_count": 16, "frozen_estimator_C": 0.1,
               "test_iid_ood_sealed_at_freeze": True, "robot_deployment_authorized": False,
               "artifact_sha256": {str(p.relative_to(ROOT)): sha(p) for p in artifacts}}
    dump(FINAL_LOCK, payload)
    print(json.dumps({"status": payload["status"], "lock_sha256": sha(FINAL_LOCK)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("lock-fit-inputs", "fit", "freeze"))
    command = parser.parse_args().command
    if command == "lock-fit-inputs":
        lock_fit_inputs()
    elif command == "fit":
        fit()
    else:
        freeze()


if __name__ == "__main__":
    main()
