#!/usr/bin/env python3
"""Fit exactly one locked affine-logit calibrator and freeze Calibration-v3."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import spatial_risk_v2_development as metrics_impl


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_calibration_v3"
DATASET = ROOT / "datasets/Gazebo_calibration_v3"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3"
CONTRACT = ROOT / "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json"
IMPLEMENTATION_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R3.json"
MATERIALIZATION_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_MATERIALIZATION_LOCK.json"
B0_PREDICTIONS = RESULT / "b0_predictions/predictions.jsonl"
B0_LOCK = RESULT / "B0_PREDICTION_LOCK.json"
RISK_PREDICTIONS = RESULT / "spatial_risk_v2_raw_predictions.jsonl"
RISK_LOCK = RESULT / "SPATIAL_RISK_V2_RAW_PREDICTION_LOCK.json"
FIT_INPUT_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_FIT_INPUT_LOCK.json"
CALIBRATOR = RESULT / "affine_logit_calibrator.json"
METRICS = RESULT / "GAZEBO_CALIBRATION_V3_METRICS.json"
SCORED = RESULT / "calibration_v3_scored_predictions.jsonl"
THRESHOLD_SCAN = RESULT / "calibration_v3_threshold_scan.jsonl"
BOOTSTRAP = RESULT / "calibration_v3_family_bootstrap_10000.json"
DECISION = RESULT / "FINAL_CALIBRATION_V3_DECISION.json"
FINAL_LOCK = RESULT / "CALIBRATOR_THRESHOLD_V3_LOCK.json"
EXPECTED_RUNTIME = {
    "python": "3.10.12",
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "scikit_learn": "1.7.2",
    "joblib": "1.5.3",
}
EPSILON = 1e-6
BOUNDS = [(-5.0, 5.0), (-10.0, 10.0)]
INITIAL = np.asarray([0.0, 0.0], dtype=np.float64)
OPTIONS = {"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 13_092_026


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def runtime() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "joblib": version("joblib"),
    }


def verify_hash_lock(path: Path, expected_status: str) -> dict:
    value = read_json(path)
    if value.get("status") != expected_status:
        raise RuntimeError(f"unexpected lock status: {path}")
    for name, digest in value.get("source_artifact_sha256", {}).items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"locked source drift: {name}")
    return value


def lock_fit_inputs() -> None:
    if FIT_INPUT_LOCK.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 fit-input lock")
    verify_hash_lock(MATERIALIZATION_LOCK, "MATERIALIZATION_FROZEN_128_OF_128_QC_PASS")
    verify_hash_lock(B0_LOCK, "B0_LOCKED_BEFORE_SPATIAL_RISK_AND_ORACLE_JOIN")
    verify_hash_lock(RISK_LOCK, "SPATIAL_RISK_V2_RAW_LOCKED_BEFORE_ORACLE_JOIN")
    manifest = read_json(DATASET / "manifest.json")
    raw = read_jsonl(RISK_PREDICTIONS)
    oracle_count = sum(1 for line in (DATASET / "evaluator_ground_truth.jsonl").read_text(encoding="utf-8").splitlines() if line)
    if (
        manifest.get("status") != "PASS"
        or manifest.get("records") != 128
        or len(raw) != 128
        or oracle_count != 128
    ):
        raise RuntimeError("Calibration-v3 fit inputs are incomplete")
    paths = [
        Path(__file__).resolve(), CONTRACT, IMPLEMENTATION_LOCK, MATERIALIZATION_LOCK,
        DATASET / "manifest.json", DATASET / "inference_manifest.jsonl",
        DATASET / "evaluator_ground_truth.jsonl", B0_PREDICTIONS, B0_LOCK,
        RISK_PREDICTIONS, RISK_LOCK,
        ROOT / "protocol/spatial_risk_v2_development.py",
    ]
    value = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_AFTER_RAW_PREDICTIONS_BEFORE_ORACLE_JOIN_AND_SINGLE_CALIBRATOR_FIT",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": 128,
        "calibrator_count": 1,
        "fit_attempts_authorized": 1,
        "source_artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in paths
        },
        "calibration_v2_reuse": False,
        "test_iid_ood_access": False,
    }
    write_json(FIT_INPUT_LOCK, value)
    print(json.dumps({"status": value["status"], "sha256": sha256(FIT_INPUT_LOCK)}, indent=2))


def verify_fit_inputs() -> None:
    verify_hash_lock(
        FIT_INPUT_LOCK,
        "LOCKED_AFTER_RAW_PREDICTIONS_BEFORE_ORACLE_JOIN_AND_SINGLE_CALIBRATOR_FIT",
    )


def join_calibration() -> list[dict]:
    inference_rows = read_jsonl(DATASET / "inference_manifest.jsonl")
    truth_rows = read_jsonl(DATASET / "evaluator_ground_truth.jsonl")
    prediction_rows = read_jsonl(B0_PREDICTIONS)
    inference = {row["sample_id"]: row for row in inference_rows}
    truth = {row["sample_id"]: row for row in truth_rows}
    predictions = {row["sample_id"]: row for row in prediction_rows}
    if len(inference) != 128 or set(inference) != set(truth) or set(inference) != set(predictions):
        raise RuntimeError("Calibration-v3 inference/oracle/B0 ID sets differ")
    output = []
    for inference_row in inference_rows:
        sample_id = inference_row["sample_id"]
        target = truth[sample_id]
        prediction = predictions[sample_id]
        if not (
            inference_row["family_id"] == target["family_id"] == prediction["family_id"]
            and inference_row["split"] == target["split"] == prediction["split"] == "calibration"
            and inference_row["relation_variant"] == target["relation_variant"]
        ):
            raise RuntimeError(f"Calibration-v3 provenance mismatch: {sample_id}")
        state = target["answerability_state"]
        point = prediction.get("prediction_xy")
        if state == "FOUND" and prediction.get("action") == "POINT" and point is not None:
            distance = math.dist(point, target["target_xy"])
        else:
            distance = math.inf
        hit = state == "FOUND" and distance <= 0.08
        output.append(
            {
                "sample_id": sample_id,
                "family_id": target["family_id"],
                "split": "calibration",
                "state": state,
                "relation": target["relation_variant"],
                "distance": distance if math.isfinite(distance) else None,
                "hit": bool(hit),
                "unsafe": bool(state != "FOUND" or not hit),
                "b0": prediction,
            }
        )
    return output


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def affine_logits(parameters: np.ndarray, raw_logits: np.ndarray) -> np.ndarray:
    log_a, intercept = parameters
    return math.exp(float(log_a)) * raw_logits + float(intercept)


def objective(parameters: np.ndarray, raw_logits: np.ndarray, labels: np.ndarray) -> float:
    calibrated_logits = affine_logits(parameters, raw_logits)
    return float(np.mean(np.logaddexp(0.0, calibrated_logits) - labels * calibrated_logits))


def gate_result(method: dict, baseline: dict) -> dict:
    checks = {
        "found_hit_delta_at_least_minus_0_125": method["found_hit_at_008"] - baseline["found_hit_at_008"] >= -0.125 - 1e-12,
        "nonfound_abstain_recall_strictly_higher": method["nonfound_abstain_recall"] > baseline["nonfound_abstain_recall"],
        "nonfound_false_accept_strictly_lower": method["nonfound_false_accept"] < baseline["nonfound_false_accept"],
        "exact_contract_not_lower": method["exact_contract"] >= baseline["exact_contract"],
        "invalid_not_counted_as_abstain": True,
    }
    return {"checks": checks, "all_pass": all(checks.values())}


def bootstrap_interval(values: list[float]) -> dict:
    return {
        "valid_draws": len(values),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ] if values else None,
    }


def family_bootstrap(
    data: list[dict],
    raw: np.ndarray,
    calibrated: np.ndarray,
    b0_risk: np.ndarray,
    selected_threshold: float | None,
) -> dict:
    families = sorted({row["family_id"] for row in data})
    family_to_index = {row["family_id"]: index for index, row in enumerate(data)}
    if len(families) != 128 or len(family_to_index) != len(data):
        raise RuntimeError("Calibration-v3 bootstrap requires one row per parent family")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    probability_names = ["ece_10bin", "brier", "aurc", "auroc_error"]
    probability_effects: dict[str, list[float]] = {name: [] for name in probability_names}
    operating_names = [
        "safe_task_accuracy",
        "found_hit_at_008",
        "nonfound_abstain_recall",
        "nonfound_false_accept",
        "coverage",
    ]
    operating_effects: dict[str, list[float]] = {name: [] for name in operating_names}
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(families, size=len(families), replace=True)
        indices = [family_to_index[family] for family in sampled]
        sample = [data[index] for index in indices]
        raw_metrics = metrics_impl.metrics(sample, raw[indices], 0.75)
        calibrated_metrics = metrics_impl.metrics(sample, calibrated[indices], 0.75)
        for name in probability_names:
            if raw_metrics[name] is not None and calibrated_metrics[name] is not None:
                probability_effects[name].append(
                    float(calibrated_metrics[name] - raw_metrics[name])
                )
        if selected_threshold is not None:
            baseline_metrics = metrics_impl.metrics(sample, b0_risk[indices], None)
            operating_metrics = metrics_impl.metrics(
                sample, calibrated[indices], selected_threshold
            )
            for name in operating_names:
                if baseline_metrics[name] is not None and operating_metrics[name] is not None:
                    operating_effects[name].append(
                        float(operating_metrics[name] - baseline_metrics[name])
                    )
    raw_point = metrics_impl.metrics(data, raw, 0.75)
    calibrated_point = metrics_impl.metrics(data, calibrated, 0.75)
    probability = {
        name: {
            "delta_calibrated_minus_raw": calibrated_point[name] - raw_point[name],
            **bootstrap_interval(values),
        }
        for name, values in probability_effects.items()
    }
    operating = None
    if selected_threshold is not None:
        baseline_point = metrics_impl.metrics(data, b0_risk, None)
        operating_point = metrics_impl.metrics(data, calibrated, selected_threshold)
        operating = {
            name: {
                "delta_calibrated_policy_minus_b0": operating_point[name] - baseline_point[name],
                **bootstrap_interval(operating_effects[name]),
            }
            for name in operating_names
        }
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "paired": True,
        "resampling_unit": "parent_family",
        "parent_family_count": len(families),
        "probability_effects_calibrated_minus_raw": probability,
        "operating_effects_calibrated_policy_minus_b0": operating,
        "operating_threshold": selected_threshold,
    }


def fit_once() -> None:
    verify_fit_inputs()
    outputs = (CALIBRATOR, METRICS, SCORED, THRESHOLD_SCAN, BOOTSTRAP, DECISION, FINAL_LOCK)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing a second Calibration-v3 fit or result overwrite")
    actual_runtime = runtime()
    if actual_runtime != EXPECTED_RUNTIME:
        raise RuntimeError(f"runtime drift: expected {EXPECTED_RUNTIME}, got {actual_runtime}")
    data = join_calibration()
    raw_rows = {row["sample_id"]: row for row in read_jsonl(RISK_PREDICTIONS)}
    if len(data) != 128 or set(raw_rows) != {row["sample_id"] for row in data}:
        raise RuntimeError("Calibration-v3 oracle/raw prediction join mismatch")
    raw = np.asarray(
        [raw_rows[row["sample_id"]]["spatial_risk_v2_raw_probability"] for row in data],
        dtype=np.float64,
    )
    labels = np.asarray([row["unsafe"] for row in data], dtype=np.float64)
    clipped = np.clip(raw, EPSILON, 1.0 - EPSILON)
    raw_logits = np.log(clipped / (1.0 - clipped))
    fit_call_count = 1
    result = minimize(
        objective,
        INITIAL.copy(),
        args=(raw_logits, labels),
        method="L-BFGS-B",
        bounds=BOUNDS,
        options=OPTIONS,
    )
    calibrated = sigmoid(affine_logits(result.x, raw_logits))
    slope = math.exp(float(result.x[0]))
    raw_probability = metrics_impl.metrics(data, raw, 0.75)
    calibrated_probability = metrics_impl.metrics(data, calibrated, 0.75)
    b0_risk = metrics_impl.baseline_risk(data)
    b0 = metrics_impl.metrics(data, b0_risk, None)
    ranking_checks = {
        "positive_slope": slope > 0.0,
        "aurc_absolute_difference_at_most_1e_12": abs(raw_probability["aurc"] - calibrated_probability["aurc"]) <= 1e-12,
        "auroc_error_absolute_difference_at_most_1e_12": abs(raw_probability["auroc_error"] - calibrated_probability["auroc_error"]) <= 1e-12,
    }
    probability_checks = {
        "ece_10_strictly_lower": calibrated_probability["ece_10bin"] < raw_probability["ece_10bin"],
        "brier_strictly_lower": calibrated_probability["brier"] < raw_probability["brier"],
    }
    scans = []
    for threshold in [index / 100.0 for index in range(101)]:
        value = metrics_impl.metrics(data, calibrated, threshold)
        gates = gate_result(value, b0)
        scans.append({"threshold": threshold, "metrics": value, "gates": gates})
    eligible = [row for row in scans if row["gates"]["all_pass"]]
    selected = min(
        eligible,
        key=lambda row: (
            row["metrics"]["nonfound_false_accept"],
            -row["metrics"]["found_hit_at_008"],
            -row["metrics"]["safe_task_accuracy"],
            -row["metrics"]["coverage"],
            row["threshold"],
        ),
    ) if eligible else None
    passed = bool(
        result.success
        and all(probability_checks.values())
        and all(ranking_checks.values())
        and selected is not None
    )
    status = (
        "CALIBRATION_V3_PASS_ELIGIBLE_FOR_SEPARATE_LOCKED_TEST_AUTHORIZATION"
        if passed
        else "CALIBRATION_V3_NEGATIVE_NO_TEST_OR_ROBOT"
    )
    selected_threshold = selected["threshold"] if selected else None
    calibrator = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "type": "positive_slope_affine_logit",
        "formula": "sigmoid(exp(log_a)*logit(clip(p_raw,1e-6,1-1e-6))+b)",
        "log_a": float(result.x[0]),
        "a_positive_slope": slope,
        "b": float(result.x[1]),
        "optimizer": "scipy.optimize.minimize_L-BFGS-B",
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "objective_nll": float(result.fun),
        "fit_call_count": fit_call_count,
        "operating_threshold": selected_threshold,
        "status": status,
    }
    write_json(CALIBRATOR, calibrator)
    write_jsonl(THRESHOLD_SCAN, scans)
    scored = []
    for row, raw_score, calibrated_score in zip(data, raw, calibrated):
        action = (
            "ABSTAIN"
            if selected_threshold is not None and calibrated_score >= selected_threshold
            else row["b0"].get("action", "INVALID")
        )
        scored.append(
            {
                "sample_id": row["sample_id"],
                "family_id": row["family_id"],
                "answerability_state": row["state"],
                "relation_variant": row["relation"],
                "unsafe": row["unsafe"],
                "raw_risk": float(raw_score),
                "calibrated_risk": float(calibrated_score),
                "action": action,
                "grounding_hit_at_008": row["hit"],
            }
        )
    write_jsonl(SCORED, scored)
    bootstrap_result = family_bootstrap(
        data, raw, calibrated, b0_risk, selected_threshold
    )
    write_json(BOOTSTRAP, bootstrap_result)
    metrics_payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "records": 128,
        "unsafe_count": int(labels.sum()),
        "safe_count": int(len(labels) - labels.sum()),
        "fit_call_count": fit_call_count,
        "optimizer_success": bool(result.success),
        "probability_gates": probability_checks,
        "ranking_invariance_gates": ranking_checks,
        "eligible_operating_point_exists": selected is not None,
        "eligible_threshold_count": len(eligible),
        "selected_threshold": selected_threshold,
        "b0": b0,
        "raw_v2_at_development_threshold_reference_0_75": raw_probability,
        "calibrated_at_development_threshold_reference_0_75": calibrated_probability,
        "calibrated_selected_operating_point": selected["metrics"] if selected else None,
        "scope": "Calibration-fit evidence only; no Test, generalization, deployment, or robot claim.",
        "test_iid_ood_access": False,
    }
    write_json(METRICS, metrics_payload)
    decision = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "decision": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_call_count": fit_call_count,
        "calibrator_sha256": sha256(CALIBRATOR),
        "metrics_sha256": sha256(METRICS),
        "scored_predictions_sha256": sha256(SCORED),
        "threshold_scan_sha256": sha256(THRESHOLD_SCAN),
        "family_bootstrap_10000_sha256": sha256(BOOTSTRAP),
        "probability_gates": probability_checks,
        "ranking_invariance_gates": ranking_checks,
        "selected_threshold": selected_threshold,
        "eligible_threshold_count": len(eligible),
        "test_iid_ood_access": False,
        "robot_authorized": False,
        "next_authorized_action": (
            "Create and review a separate Test-IID/OOD access lock; Test remains sealed."
            if passed
            else "Stop Calibration-v3; do not open Test or robot."
        ),
    }
    write_json(DECISION, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


def freeze() -> None:
    verify_fit_inputs()
    if FINAL_LOCK.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 final lock")
    decision = read_json(DECISION)
    if decision.get("decision") not in {
        "CALIBRATION_V3_PASS_ELIGIBLE_FOR_SEPARATE_LOCKED_TEST_AUTHORIZATION",
        "CALIBRATION_V3_NEGATIVE_NO_TEST_OR_ROBOT",
    }:
        raise RuntimeError("missing terminal Calibration-v3 decision")
    paths = [
        Path(__file__).resolve(), CONTRACT, IMPLEMENTATION_LOCK, MATERIALIZATION_LOCK,
        FIT_INPUT_LOCK, B0_LOCK, RISK_LOCK, CALIBRATOR, METRICS, SCORED,
        THRESHOLD_SCAN, BOOTSTRAP, DECISION,
    ]
    value = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": decision["decision"],
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "operating_threshold": decision["selected_threshold"],
        "calibrator_type": "positive_slope_affine_logit",
        "frozen_b0": True,
        "frozen_spatial_risk_v2": True,
        "test_iid_ood_sealed_at_freeze": True,
        "robot_authorized": False,
        "artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in paths
        },
    }
    write_json(FINAL_LOCK, value)
    print(json.dumps({"status": value["status"], "sha256": sha256(FINAL_LOCK)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("lock-fit-inputs", "fit-once", "freeze"))
    command = parser.parse_args().command
    if command == "lock-fit-inputs":
        lock_fit_inputs()
    elif command == "fit-once":
        fit_once()
    else:
        freeze()


if __name__ == "__main__":
    main()
