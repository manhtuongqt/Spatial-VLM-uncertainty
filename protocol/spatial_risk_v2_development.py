#!/usr/bin/env python3
"""Run the single preregistered spatial-risk v2 Train-UQ/Val-UQ evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocol/spatial_risk_method_v2_hypothesis_lock.json"
DATASET = ROOT / "datasets/Gazebo_train_uq_v2_full_r3"
PREDICTIONS = (
    ROOT
    / "results/spatial_vlm_refspatial_v1/wp5_spatial_uncertainty/b0_full_r3/predictions.jsonl"
)
B0_LOCK = (
    ROOT
    / "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/grounding_model_lock.json"
)
OUTPUT = (
    ROOT
    / "results/spatial_vlm_refspatial_v1/wp5_spatial_risk_v2_development"
)

EXPECTED_HASHES = {
    "protocol/spatial_risk_method_v2_hypothesis_lock.json": "8f6e21941fd04dc4ac91fe0fb918d65b5e8d191a52db9dd27cc93d6c39aff5fc",
    "datasets/Gazebo_train_uq_v2_full_r3/manifest.json": "87b2c73222b142a40b058ad63d94e34f5668dc60e78330153fcaf6cec4ab6271",
    "datasets/Gazebo_train_uq_v2_full_r3/inference_manifest.jsonl": "fd9862e30bac0a723c0cfe64c51a91bbe3f9e50dc98cacf973f538f512ad1caf",
    "datasets/Gazebo_train_uq_v2_full_r3/evaluator_ground_truth.jsonl": "6df14c195678318159059a649d623ba91462654f48b29879a057caee60947ea2",
    "results/spatial_vlm_refspatial_v1/wp5_spatial_uncertainty/b0_full_r3/predictions.jsonl": "31e1a19560bbb5228a4cbdf3e1af5adedbb0035196dbfa5e502ad8fd2578706c",
    "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/grounding_model_lock.json": "6e1da7e9759eacfaadf7be07467f4996e95575fd2a6225af25d87e4f74418b09",
}
FEATURE_NAMES = [
    "action_POINT",
    "action_INVALID",
    "self_consistency",
    "point_dispersion",
    "boundary_distance",
    "local_depth_valid_fraction",
    "local_depth_median",
    "local_depth_mad",
    "relation_leftmost",
    "relation_rightmost",
    "relation_second_from_left",
    "relation_second_from_right",
]
RELATIONS = ["leftmost", "rightmost", "second_from_left", "second_from_right"]
HYPERPARAMETERS = {
    "learning_rate": 0.03,
    "max_iter": 100,
    "max_leaf_nodes": 3,
    "max_depth": None,
    "min_samples_leaf": 16,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": 15092026,
}
THRESHOLDS = [i / 100.0 for i in range(101)]
HIT_RADIUS = 0.08
DEPTH_RADIUS = 8
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 13_092_026


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sanitized(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitized(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(sanitized(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(sanitized(record), sort_keys=True, allow_nan=False) + "\n"
            )


def hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def verify_preregistration() -> dict[str, Any]:
    for name, expected in EXPECTED_HASHES.items():
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"required locked input is absent: {name}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"locked input hash drift for {name}: expected {expected}, got {actual}"
            )
    lock = read_json(LOCK)
    if lock["status"] != "PREREGISTERED_BEFORE_V2_FIT_OR_NEW_DATA_ACCESS":
        raise RuntimeError("method lock is not in the preregistered state")
    if lock["locked_features_in_order"] != FEATURE_NAMES:
        raise RuntimeError("feature list differs from the preregistered order")
    if lock["method_v2"]["hyperparameters"] != HYPERPARAMETERS:
        raise RuntimeError("estimator parameters differ from the preregistration")
    if lock["development_data_and_selection"]["candidate_count"] != 1:
        raise RuntimeError("preregistration does not authorize exactly one candidate")
    boundary = lock["data_access_boundary"]
    if boundary["test_iid"] != "SEALED_AND_UNAUTHORIZED":
        raise RuntimeError("Test-IID boundary is not sealed")
    if boundary["test_ood"] != "SEALED_AND_UNAUTHORIZED":
        raise RuntimeError("Test-OOD boundary is not sealed")
    if boundary["gazebo_robot"] != "CLOSED":
        raise RuntimeError("Gazebo robot boundary is not closed")
    return lock


def point_dispersion(prediction: dict[str, Any]) -> float:
    points = np.asarray(
        [
            draw["prediction_xy"]
            for draw in prediction.get("stochastic_draws", [])
            if draw.get("action") == "POINT" and draw.get("prediction_xy") is not None
        ],
        dtype=np.float64,
    )
    if len(points) < 2:
        return 1.0
    return float(np.mean(np.linalg.norm(points - points.mean(axis=0), axis=1)))


def locked_depth_path(dataset: Path, item: dict[str, Any]) -> Path:
    path = (dataset / item["metric_depth"]).resolve()
    records_root = (dataset / "records").resolve()
    if path != records_root and records_root not in path.parents:
        raise RuntimeError(f"metric depth escapes the authorized dataset: {path}")
    return path


def depth_features(path: Path, prediction: dict[str, Any]) -> list[float]:
    if prediction.get("action") != "POINT" or prediction.get("prediction_xy") is None:
        return [0.0, 0.0, 0.0]
    depth = np.load(path, allow_pickle=False)
    x = int(round(prediction["prediction_xy"][0] * (depth.shape[1] - 1)))
    y = int(round(prediction["prediction_xy"][1] * (depth.shape[0] - 1)))
    patch = depth[
        max(0, y - DEPTH_RADIUS) : min(depth.shape[0], y + DEPTH_RADIUS + 1),
        max(0, x - DEPTH_RADIUS) : min(depth.shape[1], x + DEPTH_RADIUS + 1),
    ]
    valid = np.isfinite(patch) & (patch >= 0.05) & (patch <= 2.0)
    if not valid.any():
        return [0.0, 0.0, 0.0]
    values = patch[valid]
    median = float(np.median(values))
    return [
        float(valid.mean()),
        median,
        float(np.median(np.abs(values - median))),
    ]


def feature_vector(
    inference: dict[str, Any], prediction: dict[str, Any], dataset: Path
) -> np.ndarray:
    action = prediction.get("action", "INVALID")
    xy = prediction.get("prediction_xy") if action == "POINT" else None
    x, y = xy if xy is not None else (-1.0, -1.0)
    boundary = min(x, y, 1.0 - x, 1.0 - y) if xy is not None else -1.0
    relation = inference["relation_variant"]
    values = np.asarray(
        [
            float(action == "POINT"),
            float(action == "INVALID"),
            float(prediction.get("self_consistency_confidence", 0.0)),
            point_dispersion(prediction),
            boundary,
            *depth_features(locked_depth_path(dataset, inference), prediction),
            *[float(relation == candidate) for candidate in RELATIONS],
        ],
        dtype=np.float64,
    )
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid locked feature vector for {inference['sample_id']}")
    return values


def load_authorized_data() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(DATASET / "manifest.json")
    if manifest.get("status") != "PASS":
        raise RuntimeError("Train-UQ/Val-UQ dataset manifest did not pass")
    if manifest.get("test_iid_ood_access") is not False:
        raise RuntimeError("dataset manifest does not attest sealed Test-IID/OOD")
    if manifest.get("b2_opened") is not False:
        raise RuntimeError("dataset manifest reports B2 access")
    if manifest.get("split_counts") != {"train_uq": 256, "val_uq": 64}:
        raise RuntimeError("unexpected Train-UQ/Val-UQ split counts")

    inference_rows = read_jsonl(DATASET / "inference_manifest.jsonl")
    ground_truth_rows = read_jsonl(DATASET / "evaluator_ground_truth.jsonl")
    prediction_rows = read_jsonl(PREDICTIONS)
    inference = {row["sample_id"]: row for row in inference_rows}
    ground_truth = {row["sample_id"]: row for row in ground_truth_rows}
    predictions = {row["sample_id"]: row for row in prediction_rows}
    if len(inference) != 320 or set(inference) != set(ground_truth) or set(inference) != set(predictions):
        raise RuntimeError("inference, oracle, and B0 prediction ID sets differ")

    data: list[dict[str, Any]] = []
    depth_records: list[dict[str, Any]] = []
    for inference_row in inference_rows:
        sample_id = inference_row["sample_id"]
        truth = ground_truth[sample_id]
        prediction = predictions[sample_id]
        if not (
            inference_row["split"] == truth["split"] == prediction["split"]
            and inference_row["family_id"] == truth["family_id"] == prediction["family_id"]
            and inference_row["relation_variant"] == truth["relation_variant"]
        ):
            raise RuntimeError(f"joined provenance mismatch for {sample_id}")
        state = truth["answerability_state"]
        if state == "FOUND" and prediction.get("action") == "POINT" and prediction.get("prediction_xy") is not None:
            distance = math.dist(prediction["prediction_xy"], truth["target_xy"])
        else:
            distance = math.inf
        hit = state == "FOUND" and distance <= HIT_RADIUS
        unsafe = state != "FOUND" or not hit
        depth_path = locked_depth_path(DATASET, inference_row)
        depth_records.append(
            {
                "sample_id": sample_id,
                "path": relative(depth_path),
                "bytes": depth_path.stat().st_size,
                "sha256": sha256(depth_path),
            }
        )
        data.append(
            {
                "sample_id": sample_id,
                "family_id": truth["family_id"],
                "split": truth["split"],
                "state": state,
                "relation": truth["relation_variant"],
                "target_category": truth.get("target_category"),
                "distance": distance if math.isfinite(distance) else None,
                "hit": bool(hit),
                "unsafe": bool(unsafe),
                "feature": feature_vector(inference_row, prediction, DATASET),
                "b0": prediction,
            }
        )

    if {row["split"] for row in data} != {"train_uq", "val_uq"}:
        raise RuntimeError("unauthorized split encountered")
    if len({row["family_id"] for row in data}) != len(data):
        raise RuntimeError("expected one record per parent family")
    train = [row for row in data if row["split"] == "train_uq"]
    validation = [row for row in data if row["split"] == "val_uq"]
    if len(train) != 256 or len(validation) != 64:
        raise RuntimeError("materialized split sizes do not match the lock")
    if set(row["family_id"] for row in train) & set(row["family_id"] for row in validation):
        raise RuntimeError("Train-UQ and Val-UQ family leakage detected")
    for partition in (train, validation):
        labels = {row["unsafe"] for row in partition}
        if labels != {False, True}:
            raise RuntimeError("a development partition lacks one unsafe class")

    return data, {
        "manifest": manifest,
        "depth_records": depth_records,
        "train_count": len(train),
        "val_count": len(validation),
    }


def aurc(data: list[dict[str, Any]], risk: np.ndarray) -> float:
    order = sorted(range(len(data)), key=lambda index: (float(risk[index]), data[index]["sample_id"]))
    labels = np.asarray([data[index]["unsafe"] for index in order], dtype=np.float64)
    return float(np.mean(np.cumsum(labels) / np.arange(1, len(labels) + 1)))


def ece_10(labels: np.ndarray, risk: np.ndarray) -> float:
    total = 0.0
    for index in range(10):
        lower, upper = index / 10.0, (index + 1) / 10.0
        mask = (risk >= lower) & ((risk < upper) if index < 9 else (risk <= upper))
        if mask.any():
            total += float(mask.mean()) * abs(float(risk[mask].mean() - labels[mask].mean()))
    return float(total)


def mean_boolean(values: list[bool]) -> float | None:
    return float(np.mean(values)) if values else None


def decisions(
    data: list[dict[str, Any]], risk: np.ndarray, threshold: float | None
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, score in zip(data, risk):
        injected_abstain = threshold is not None and float(score) >= threshold
        action = "ABSTAIN" if injected_abstain else row["b0"].get("action", "INVALID")
        exact = True if injected_abstain else bool(row["b0"].get("exact_contract", False))
        correct = row["hit"] if row["state"] == "FOUND" else action == "ABSTAIN"
        output.append(
            {
                "action": action,
                "exact": exact,
                "correct": bool(correct),
                "injected_abstain": bool(injected_abstain),
            }
        )
    return output


def metrics(
    data: list[dict[str, Any]], risk: np.ndarray, threshold: float | None
) -> dict[str, Any]:
    labels = np.asarray([row["unsafe"] for row in data], dtype=np.int64)
    risk = np.asarray(risk, dtype=np.float64)
    if not np.isfinite(risk).all() or np.any(risk < 0.0) or np.any(risk > 1.0):
        raise RuntimeError("risk score outside the locked finite [0,1] range")
    chosen = decisions(data, risk, threshold)
    found = [index for index, row in enumerate(data) if row["state"] == "FOUND"]
    nonfound = [index for index, row in enumerate(data) if row["state"] != "FOUND"]
    return {
        "n": len(data),
        "unsafe_rate": float(labels.mean()),
        "risk_min": float(risk.min()),
        "risk_max": float(risk.max()),
        "risk_mean": float(risk.mean()),
        "aurc": aurc(data, risk),
        "auroc_error": float(roc_auc_score(labels, risk)) if len(set(labels)) == 2 else None,
        "ece_10bin": ece_10(labels, risk),
        "brier": float(brier_score_loss(labels, risk)),
        "coverage": mean_boolean([item["action"] == "POINT" for item in chosen]),
        "found_hit_at_008": mean_boolean(
            [chosen[index]["action"] == "POINT" and data[index]["hit"] for index in found]
        ),
        "nonfound_abstain_recall": mean_boolean(
            [chosen[index]["action"] == "ABSTAIN" for index in nonfound]
        ),
        "nonfound_false_accept": mean_boolean(
            [chosen[index]["action"] == "POINT" for index in nonfound]
        ),
        "invalid_rate": mean_boolean([item["action"] == "INVALID" for item in chosen]),
        "exact_contract": mean_boolean([item["exact"] for item in chosen]),
        "safe_task_accuracy": mean_boolean([item["correct"] for item in chosen]),
        "threshold": threshold,
    }


def baseline_risk(data: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [float(row["b0"].get("predictive_uncertainty", 1.0)) for row in data],
        dtype=np.float64,
    )


def evaluate_gates(method: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "found_hit_delta_at_least_minus_0_125": method["found_hit_at_008"] - baseline["found_hit_at_008"] >= -0.125 - 1e-12,
        "nonfound_abstain_recall_strictly_higher": method["nonfound_abstain_recall"] > baseline["nonfound_abstain_recall"],
        "nonfound_false_accept_strictly_lower": method["nonfound_false_accept"] < baseline["nonfound_false_accept"],
        "exact_contract_not_lower": method["exact_contract"] >= baseline["exact_contract"],
        "aurc_no_worse": method["aurc"] <= baseline["aurc"] + 1e-12,
        "invalid_not_counted_as_abstain": True,
    }
    return {"checks": checks, "all_pass": all(checks.values())}


def threshold_scan(
    validation: list[dict[str, Any]], risk: np.ndarray, baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for threshold in THRESHOLDS:
        score = metrics(validation, risk, threshold)
        gates = evaluate_gates(score, baseline)
        output.append({"threshold": threshold, "metrics": score, "gates": gates})
    return output


def select_threshold(scan: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in scan if row["gates"]["all_pass"]]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            row["metrics"]["nonfound_false_accept"],
            -row["metrics"]["found_hit_at_008"],
            -row["metrics"]["safe_task_accuracy"],
            -row["metrics"]["coverage"],
            row["threshold"],
        ),
    )


def diagnostic_threshold(scan: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        scan,
        key=lambda row: (
            -sum(bool(value) for value in row["gates"]["checks"].values()),
            row["metrics"]["nonfound_false_accept"],
            -row["metrics"]["found_hit_at_008"],
            -row["metrics"]["safe_task_accuracy"],
            -row["metrics"]["coverage"],
            row["threshold"],
        ),
    )


def subset_metrics(
    data: list[dict[str, Any]], risk: np.ndarray, threshold: float | None, key: str
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value in sorted({str(row[key]) for row in data}):
        indices = [index for index, row in enumerate(data) if str(row[key]) == value]
        subset = [data[index] for index in indices]
        output[value] = metrics(subset, risk[indices], threshold)
    return output


def bootstrap(
    validation: list[dict[str, Any]],
    b0_risk: np.ndarray,
    method_risk: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    family_to_indices: dict[str, list[int]] = {}
    for index, row in enumerate(validation):
        family_to_indices.setdefault(row["family_id"], []).append(index)
    families = sorted(family_to_indices)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = [
        "safe_task_accuracy",
        "found_hit_at_008",
        "nonfound_abstain_recall",
        "nonfound_false_accept",
        "aurc",
    ]
    effects: dict[str, list[float]] = {name: [] for name in names}
    point_baseline = metrics(validation, b0_risk, None)
    point_method = metrics(validation, method_risk, threshold)
    for _ in range(BOOTSTRAP_DRAWS):
        sampled_families = rng.choice(families, size=len(families), replace=True)
        indices = [index for family in sampled_families for index in family_to_indices[family]]
        sampled_data = [validation[index] for index in indices]
        baseline = metrics(sampled_data, b0_risk[indices], None)
        method = metrics(sampled_data, method_risk[indices], threshold)
        for name in names:
            if baseline[name] is not None and method[name] is not None:
                effects[name].append(float(method[name] - baseline[name]))
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "paired": True,
        "resampling_unit": "parent_family",
        "parent_family_count": len(families),
        "effects_method_minus_b0": {
            name: {
                "delta": point_method[name] - point_baseline[name],
                "ci95": [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ],
                "valid_draws": len(values),
            }
            for name, values in effects.items()
        },
    }


def mcnemar(
    validation: list[dict[str, Any]],
    b0_risk: np.ndarray,
    method_risk: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    baseline = decisions(validation, b0_risk, None)
    method = decisions(validation, method_risk, threshold)
    n01 = sum((not left["correct"]) and right["correct"] for left, right in zip(baseline, method))
    n10 = sum(left["correct"] and (not right["correct"]) for left, right in zip(baseline, method))
    discordant = n01 + n10
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(n01, n10)
        p_value = min(
            1.0,
            2.0
            * sum(math.comb(discordant, index) for index in range(smaller + 1))
            / (2**discordant),
        )
    return {
        "b0_wrong_method_right": n01,
        "b0_right_method_wrong": n10,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "joblib": version("joblib"),
    }


def preflight() -> None:
    verify_preregistration()
    data, provenance = load_authorized_data()
    train = [row for row in data if row["split"] == "train_uq"]
    validation = [row for row in data if row["split"] == "val_uq"]
    train_features = np.stack([row["feature"] for row in train])
    val_features = np.stack([row["feature"] for row in validation])
    print(
        json.dumps(
            {
                "status": "PREFLIGHT_PASS_NO_FIT_PERFORMED",
                "train_count": provenance["train_count"],
                "val_count": provenance["val_count"],
                "feature_count": train_features.shape[1],
                "train_feature_matrix_sha256": hash_array(train_features),
                "val_feature_matrix_sha256": hash_array(val_features),
                "environment": environment(),
                "test_iid_ood_access": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_once() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing a second fit: output already exists at {OUTPUT}")
    lock = verify_preregistration()
    data, provenance = load_authorized_data()
    train = [row for row in data if row["split"] == "train_uq"]
    validation = [row for row in data if row["split"] == "val_uq"]
    train_features = np.stack([row["feature"] for row in train])
    train_labels = np.asarray([row["unsafe"] for row in train], dtype=np.int64)
    val_features = np.stack([row["feature"] for row in validation])
    val_labels = np.asarray([row["unsafe"] for row in validation], dtype=np.int64)

    temporary = OUTPUT.parent / f".{OUTPUT.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise RuntimeError(f"temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    fit_call_count = 0
    try:
        depth_manifest_path = temporary / "spatial_risk_v2_depth_input_manifest.jsonl"
        write_jsonl(depth_manifest_path, provenance["depth_records"])

        estimator = HistGradientBoostingClassifier(**HYPERPARAMETERS)
        fit_call_count += 1
        estimator.fit(train_features, train_labels)
        if fit_call_count != 1:
            raise RuntimeError("fit call count violated the preregistration")

        train_risk = estimator.predict_proba(train_features)[:, 1]
        val_risk = estimator.predict_proba(val_features)[:, 1]
        train_b0_risk = baseline_risk(train)
        val_b0_risk = baseline_risk(validation)
        baseline_val = metrics(validation, val_b0_risk, None)
        scan = threshold_scan(validation, val_risk, baseline_val)
        selected = select_threshold(scan)
        diagnostic = diagnostic_threshold(scan)
        eligible = selected is not None
        reported = selected if selected is not None else diagnostic
        selected_threshold = selected["threshold"] if selected is not None else None
        reported_threshold = float(reported["threshold"])
        decision_state = (
            "SPATIAL_RISK_V2_METHOD_FROZEN_AWAITING_NEW_CALIBRATION_CONTRACT"
            if eligible
            else "SPATIAL_RISK_V2_NEGATIVE_DO_NOT_CAPTURE_CALIBRATION_V3"
        )

        model_path = temporary / "spatial_risk_v2_estimator.joblib"
        joblib.dump(
            {
                "estimator": estimator,
                "feature_names": FEATURE_NAMES,
                "selected_threshold": selected_threshold,
                "method_lock_sha256": EXPECTED_HASHES[relative(LOCK)],
            },
            model_path,
        )
        estimator_json_path = temporary / "spatial_risk_v2_estimator.json"
        write_json(
            estimator_json_path,
            {
                "schema_version": 1,
                "status": "FROZEN_DEVELOPMENT_ESTIMATOR" if eligible else "NEGATIVE_DEVELOPMENT_ESTIMATOR_PRESERVED",
                "method": "sklearn.ensemble.HistGradientBoostingClassifier",
                "feature_names": FEATURE_NAMES,
                "hyperparameters_explicitly_locked": HYPERPARAMETERS,
                "all_runtime_parameters": estimator.get_params(deep=False),
                "classes": estimator.classes_.tolist(),
                "n_iter": int(estimator.n_iter_),
                "fit_split": "train_uq_only",
                "fit_call_count": fit_call_count,
                "candidate_count": 1,
                "selected_threshold": selected_threshold,
                "train_feature_matrix_sha256": hash_array(train_features),
                "train_label_array_sha256": hash_array(train_labels),
                "val_feature_matrix_sha256": hash_array(val_features),
                "val_label_array_sha256": hash_array(val_labels),
                "method_lock_sha256": EXPECTED_HASHES[relative(LOCK)],
                "environment": environment(),
            },
        )

        threshold_path = temporary / "spatial_risk_v2_threshold_scan.jsonl"
        write_jsonl(threshold_path, scan)
        prediction_records: list[dict[str, Any]] = []
        all_risk = np.concatenate([train_risk, val_risk])
        for row, risk in zip(train + validation, all_risk):
            proposed_action = "ABSTAIN" if risk >= reported_threshold else row["b0"].get("action", "INVALID")
            prediction_records.append(
                {
                    "sample_id": row["sample_id"],
                    "family_id": row["family_id"],
                    "split": row["split"],
                    "answerability_state": row["state"],
                    "relation_variant": row["relation"],
                    "unsafe": row["unsafe"],
                    "b0_action": row["b0"].get("action", "INVALID"),
                    "b0_risk_proxy": float(row["b0"].get("predictive_uncertainty", 1.0)),
                    "spatial_risk_v2_raw_probability": float(risk),
                    "threshold_role": "selected" if eligible else "diagnostic_not_selected",
                    "threshold": reported_threshold,
                    "proposed_action": proposed_action,
                    "grounding_hit_at_008": row["hit"],
                }
            )
        predictions_path = temporary / "spatial_risk_v2_predictions.jsonl"
        write_jsonl(predictions_path, prediction_records)

        bootstrap_result = bootstrap(validation, val_b0_risk, val_risk, reported_threshold)
        bootstrap_result["threshold_role"] = "selected" if eligible else "diagnostic_not_selected"
        bootstrap_path = temporary / "spatial_risk_v2_val_family_bootstrap_10000.json"
        write_json(bootstrap_path, bootstrap_result)
        mcnemar_result = mcnemar(validation, val_b0_risk, val_risk, reported_threshold)

        metrics_all = {
            "b0_train": metrics(train, train_b0_risk, None),
            "method_train_at_reported_threshold": metrics(train, train_risk, reported_threshold),
            "b0_val": baseline_val,
            "method_val_at_reported_threshold": metrics(validation, val_risk, reported_threshold),
            "val_by_relation": {
                "b0": subset_metrics(validation, val_b0_risk, None, "relation"),
                "method": subset_metrics(validation, val_risk, reported_threshold, "relation"),
            },
            "val_by_answerability_state": {
                "b0": subset_metrics(validation, val_b0_risk, None, "state"),
                "method": subset_metrics(validation, val_risk, reported_threshold, "state"),
            },
        }
        metrics_path = temporary / "spatial_risk_v2_metrics.json"
        write_json(metrics_path, metrics_all)

        source_hashes = dict(EXPECTED_HASHES)
        source_hashes[relative(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
        generated_hashes = {
            relative(OUTPUT / model_path.name): sha256(model_path),
            relative(OUTPUT / estimator_json_path.name): sha256(estimator_json_path),
            relative(OUTPUT / predictions_path.name): sha256(predictions_path),
            relative(OUTPUT / threshold_path.name): sha256(threshold_path),
            relative(OUTPUT / bootstrap_path.name): sha256(bootstrap_path),
            relative(OUTPUT / metrics_path.name): sha256(metrics_path),
            relative(OUTPUT / depth_manifest_path.name): sha256(depth_manifest_path),
        }
        selected_gate_result = selected["gates"] if selected is not None else None
        decision = {
            "schema_version": 1,
            "decision": decision_state,
            "scientific_status": "DEVELOPMENT_GATE_PASS" if eligible else "DEVELOPMENT_NEGATIVE_RESULT",
            "completed_at_utc": utc_now(),
            "method_lock": relative(LOCK),
            "method_lock_sha256": EXPECTED_HASHES[relative(LOCK)],
            "single_candidate": True,
            "fit_call_count": fit_call_count,
            "fit_split": "Train-UQ only",
            "selection_split": "Val-UQ only",
            "train_count": len(train),
            "val_count": len(validation),
            "feature_names": FEATURE_NAMES,
            "feature_count": len(FEATURE_NAMES),
            "model": {
                "type": "sklearn.ensemble.HistGradientBoostingClassifier",
                "hyperparameters": HYPERPARAMETERS,
                "joblib_sha256": generated_hashes[relative(OUTPUT / model_path.name)],
                "inspectable_estimator_sha256": generated_hashes[relative(OUTPUT / estimator_json_path.name)],
            },
            "input_and_code_sha256": source_hashes,
            "derived_input_sha256": {
                "depth_input_manifest": generated_hashes[relative(OUTPUT / depth_manifest_path.name)],
                "train_feature_matrix": hash_array(train_features),
                "train_labels": hash_array(train_labels),
                "val_feature_matrix": hash_array(val_features),
                "val_labels": hash_array(val_labels),
            },
            "generated_artifact_sha256": generated_hashes,
            "threshold_selection": {
                "grid": {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "count": 101},
                "eligible_threshold_count": sum(row["gates"]["all_pass"] for row in scan),
                "selected_threshold": selected_threshold,
                "selected_gate_result": selected_gate_result,
                "diagnostic_threshold_if_no_selection": None if eligible else reported_threshold,
                "reported_threshold": reported_threshold,
                "reported_threshold_role": "selected" if eligible else "diagnostic_not_selected",
            },
            "metrics": metrics_all,
            "bootstrap": bootstrap_result,
            "mcnemar_safe_task": mcnemar_result,
            "gate_interpretation": (
                "The single preregistered v2 estimator and selected Val-UQ operating point pass every unchanged development gate. This is development evidence only, not calibration or test evidence."
                if eligible
                else "No Val-UQ threshold passes every unchanged development gate. Preserve this as a negative result; Calibration-v3 must not be captured."
            ),
            "next_authorized_action": (
                "Create and review a separate pre-capture, family-disjoint Calibration-v3 contract; do not capture or access Calibration-v3 under this decision alone."
                if eligible
                else "Stop this method revision. Do not capture Calibration-v3, open Test-IID/OOD, or evaluate the robot."
            ),
            "sealed_state": {
                "calibration_v2": "CLOSED_HISTORICAL_NEGATIVE_NOT_ACCESSED_BY_THIS_RUN",
                "calibration_v3": "NOT_MATERIALIZED_OR_ACCESSED",
                "test_iid": "SEALED_AND_UNAUTHORIZED",
                "test_ood": "SEALED_AND_UNAUTHORIZED",
                "gazebo_robot": "CLOSED",
                "real_ur3": "CLOSED_OPTIONAL_FUTURE_ONLY",
            },
            "environment": environment(),
            "claim_boundary": "A completed pipeline is not proof of the hypothesis; only the stated development gate decision is supported here.",
        }
        decision_path = temporary / "spatial_risk_v2_development_decision.json"
        write_json(decision_path, decision)

        artifact_paths = sorted(path for path in temporary.iterdir() if path.is_file())
        artifact_manifest = {
            "schema_version": 1,
            "status": "COMPLETE",
            "created_at_utc": utc_now(),
            "decision": decision_state,
            "artifacts": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in artifact_paths
            },
            "sealed_splits_accessed": [],
        }
        write_json(temporary / "spatial_risk_v2_artifact_manifest.json", artifact_manifest)
        temporary.rename(OUTPUT)
        print(
            json.dumps(
                {
                    "decision": decision_state,
                    "selected_threshold": selected_threshold,
                    "eligible_threshold_count": sum(row["gates"]["all_pass"] for row in scan),
                    "val_metrics": metrics_all["method_val_at_reported_threshold"],
                    "output": str(OUTPUT),
                    "fit_call_count": fit_call_count,
                },
                indent=2,
                sort_keys=True,
            )
        )
    except Exception:
        failure_marker = temporary / "DO_NOT_RERUN_FIT_WITHOUT_AUDIT.txt"
        failure_marker.write_text(
            "A fit may already have occurred in this temporary directory. Audit before any retry.\n",
            encoding="utf-8",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run-once"])
    arguments = parser.parse_args()
    if arguments.command == "preflight":
        preflight()
    else:
        run_once()


if __name__ == "__main__":
    main()
