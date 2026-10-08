#!/usr/bin/env python3
"""Run and lock blinded B0 plus frozen spatial-risk v2 inference on Calibration-v6."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

import gazebo_train_uq_v1_infer as b0_runner
import spatial_risk_v2_development as risk_v2


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_calibration_v6"
DATASET = ROOT / "datasets/Gazebo_calibration_v6"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6"
B0_OUTPUT = RESULT / "b0_predictions"
B0_PREDICTIONS = B0_OUTPUT / "predictions.jsonl"
B0_RUN = B0_OUTPUT / "run.json"
B0_LOCK = RESULT / "B0_PREDICTION_LOCK.json"
RISK_PREDICTIONS = RESULT / "spatial_risk_v2_raw_predictions.jsonl"
RISK_RUN = RESULT / "spatial_risk_v2_raw_run.json"
RISK_LOCK = RESULT / "SPATIAL_RISK_V2_RAW_PREDICTION_LOCK.json"
CONTRACT = ROOT / "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json"
IMPLEMENTATION_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V6_IMPLEMENTATION_LOCK.json"
MATERIALIZATION_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V6_MATERIALIZATION_LOCK.json"
MODEL = ROOT / "results/spatial_vlm_refspatial_v1/wp5_spatial_risk_v2_development/spatial_risk_v2_estimator.joblib"
MODEL_SHA256 = "f296e2e33abdd66437d2976a91ff78b4e707d67244f3b998dc9d9b011cbd5bc2"
FORBIDDEN = {
    "answerability_state",
    "target_id",
    "target_xy",
    "candidate_set",
    "failure_tags",
    "supervision",
    "unsafe",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def verify_materialization() -> tuple[dict, list[dict]]:
    from gazebo_calibration_v6_pipeline import validate_implementation
    validate_implementation()
    lock = read_json(MATERIALIZATION_LOCK)
    if lock.get("status") != "MATERIALIZATION_FROZEN_128_OF_128_QC_PASS":
        raise RuntimeError("Calibration-v6 materialization is not frozen PASS")
    for name, digest in lock["source_artifact_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"materialization artifact drift: {name}")
    manifest = read_json(DATASET / "manifest.json")
    rows = read_jsonl(DATASET / "inference_manifest.jsonl")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("records") != 128
        or len(rows) != 128
        or len({row["sample_id"] for row in rows}) != 128
    ):
        raise RuntimeError("invalid Calibration-v6 inference materialization")
    leaks = [(row["sample_id"], sorted(FORBIDDEN & set(row))) for row in rows if FORBIDDEN & set(row)]
    if leaks:
        raise RuntimeError(f"oracle leakage in inference manifest: {leaks[:3]}")
    if any(row.get("split") != "calibration" for row in rows):
        raise RuntimeError("non-calibration split in inference manifest")
    return manifest, rows


def run_b0(base_model: Path) -> None:
    verify_materialization()
    if B0_LOCK.exists():
        raise RuntimeError("B0 prediction lock already exists")
    b0_runner.PROTOCOL_ID = PROTOCOL_ID
    b0_runner.run(
        argparse.Namespace(
            dataset=DATASET,
            output=B0_OUTPUT,
            base=base_model,
            adapter=None,
            model_id="b0",
            draws=3,
            max_new_tokens=40,
        )
    )


def lock_b0() -> None:
    if B0_LOCK.exists():
        raise FileExistsError("refusing to overwrite B0 prediction lock")
    _, rows = verify_materialization()
    run = read_json(B0_RUN)
    predictions = read_jsonl(B0_PREDICTIONS)
    if (
        run.get("status") != "COMPLETED"
        or len(predictions) != 128
        or {row["sample_id"] for row in rows} != {row["sample_id"] for row in predictions}
        or any("error" in row for row in predictions)
    ):
        raise RuntimeError("B0 inference is incomplete or contains errors")
    sources = [
        Path(__file__).resolve(), CONTRACT, IMPLEMENTATION_LOCK, MATERIALIZATION_LOCK,
        DATASET / "manifest.json", DATASET / "inference_manifest.jsonl",
        B0_PREDICTIONS, B0_RUN, ROOT / "protocol/gazebo_train_uq_v1_infer.py",
        ROOT / "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/grounding_model_lock.json",
    ]
    value = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "B0_LOCKED_BEFORE_SPATIAL_RISK_AND_ORACLE_JOIN",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_count": 128,
        "oracle_opened": False,
        "source_artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "test_iid_ood_access": False,
    }
    write_json(B0_LOCK, value)
    print(json.dumps({"status": value["status"], "sha256": sha256(B0_LOCK)}, indent=2))


def verify_b0_lock() -> None:
    value = read_json(B0_LOCK)
    if value.get("status") != "B0_LOCKED_BEFORE_SPATIAL_RISK_AND_ORACLE_JOIN":
        raise RuntimeError("invalid B0 prediction lock")
    for name, digest in value["source_artifact_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"B0 inference artifact drift: {name}")


def run_risk() -> None:
    verify_b0_lock()
    if any(path.exists() for path in (RISK_PREDICTIONS, RISK_RUN, RISK_LOCK)):
        raise FileExistsError("refusing to overwrite or rerun spatial-risk v2 inference")
    _, inference_rows = verify_materialization()
    predictions = {row["sample_id"]: row for row in read_jsonl(B0_PREDICTIONS)}
    if sha256(MODEL) != MODEL_SHA256:
        raise RuntimeError("frozen spatial-risk v2 model hash drift")
    bundle = joblib.load(MODEL)
    if bundle["feature_names"] != risk_v2.FEATURE_NAMES:
        raise RuntimeError("frozen spatial-risk v2 feature order drift")
    features = np.stack(
        [risk_v2.feature_vector(row, predictions[row["sample_id"]], DATASET) for row in inference_rows]
    )
    probabilities = bundle["estimator"].predict_proba(features)[:, 1]
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise RuntimeError("spatial-risk v2 produced invalid probabilities")
    records = [
        {
            "sample_id": row["sample_id"],
            "scene_id": row["scene_id"],
            "family_id": row["family_id"],
            "split": row["split"],
            "relation_variant": row["relation_variant"],
            "b0_action": predictions[row["sample_id"]].get("action", "INVALID"),
            "feature_names": risk_v2.FEATURE_NAMES,
            "feature_values": features[index].tolist(),
            "spatial_risk_v2_raw_probability": float(probabilities[index]),
        }
        for index, row in enumerate(inference_rows)
    ]
    write_jsonl(RISK_PREDICTIONS, records)
    write_json(
        RISK_RUN,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETED",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "records": 128,
            "model_sha256": sha256(MODEL),
            "feature_count": len(risk_v2.FEATURE_NAMES),
            "oracle_opened": False,
            "test_iid_ood_access": False,
        },
    )


def lock_risk() -> None:
    if RISK_LOCK.exists():
        raise FileExistsError("refusing to overwrite spatial-risk prediction lock")
    verify_b0_lock()
    run = read_json(RISK_RUN)
    predictions = read_jsonl(RISK_PREDICTIONS)
    if run.get("status") != "COMPLETED" or len(predictions) != 128:
        raise RuntimeError("spatial-risk v2 inference is incomplete")
    sources = [
        Path(__file__).resolve(), CONTRACT, IMPLEMENTATION_LOCK, MATERIALIZATION_LOCK,
        B0_LOCK, RISK_RUN, RISK_PREDICTIONS, MODEL,
        ROOT / "protocol/spatial_risk_v2_development.py",
        DATASET / "manifest.json", DATASET / "inference_manifest.jsonl",
    ]
    value = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "SPATIAL_RISK_V2_RAW_LOCKED_BEFORE_ORACLE_JOIN",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_count": 128,
        "model_sha256": MODEL_SHA256,
        "oracle_opened": False,
        "source_artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "test_iid_ood_access": False,
    }
    write_json(RISK_LOCK, value)
    print(json.dumps({"status": value["status"], "sha256": sha256(RISK_LOCK)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-b0", "lock-b0", "run-risk", "lock-risk"))
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "RoboRefer/models/RoboRefer-2B-SFT",
    )
    args = parser.parse_args()
    if args.command == "run-b0":
        run_b0(args.base.resolve())
    elif args.command == "lock-b0":
        lock_b0()
    elif args.command == "run-risk":
        run_risk()
    else:
        lock_risk()


if __name__ == "__main__":
    main()
