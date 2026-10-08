#!/usr/bin/env python3
"""R5 provenance adapter for the frozen Calibration-v3 downstream pipeline.

This module changes no scientific method.  It binds the already-preregistered
materialization, frozen inference, affine-logit fit, statistical report, and
artifact-manifest operations to the append-only R5 implementation lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import gazebo_calibration_v3_fit as fit
import gazebo_calibration_v3_infer as inference
import gazebo_calibration_v3_pipeline_r5 as pipeline
import materialize_gazebo_calibration_v3 as materialization


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3"
DATASET = ROOT / "datasets/Gazebo_calibration_v3"
R5_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R5.json"
ARTIFACT_MANIFEST = RESULT / "calibration_v3_artifact_manifest.json"
MCNEMAR = RESULT / "calibration_v3_mcnemar_exact.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_json_replace_before_final_lock(path: Path, value: dict) -> None:
    if fit.FINAL_LOCK.exists():
        raise RuntimeError("refusing to modify a result after the final Calibration-v3 lock")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def configure() -> None:
    pipeline.validate_r5_implementation()
    for module in (materialization, inference, fit):
        module.IMPLEMENTATION_LOCK = R5_LOCK


def add_locked_mcnemar_report() -> None:
    """Add the contract-required exact paired test before final result freeze."""
    if MCNEMAR.exists():
        raise FileExistsError("refusing to recompute Calibration-v3 McNemar result")
    if fit.FINAL_LOCK.exists():
        raise RuntimeError("Calibration-v3 final lock already exists")
    metrics = read_json(fit.METRICS)
    decision = read_json(fit.DECISION)
    selected_threshold = metrics.get("selected_threshold")
    if selected_threshold is None:
        result = {
            "schema_version": 1,
            "protocol_id": fit.PROTOCOL_ID,
            "status": "NOT_APPLICABLE_NO_ELIGIBLE_OPERATING_THRESHOLD",
            "endpoint": "paired_safe_task_correctness",
            "exact_two_sided_p": None,
        }
    else:
        data = fit.join_calibration()
        raw_rows = {
            row["sample_id"]: row for row in fit.read_jsonl(fit.RISK_PREDICTIONS)
        }
        raw = np.asarray(
            [raw_rows[row["sample_id"]]["spatial_risk_v2_raw_probability"] for row in data],
            dtype=np.float64,
        )
        calibrator = read_json(fit.CALIBRATOR)
        raw_logits = np.log(
            np.clip(raw, fit.EPSILON, 1.0 - fit.EPSILON)
            / (1.0 - np.clip(raw, fit.EPSILON, 1.0 - fit.EPSILON))
        )
        calibrated = fit.sigmoid(
            fit.affine_logits(
                np.asarray([calibrator["log_a"], calibrator["b"]], dtype=np.float64),
                raw_logits,
            )
        )
        b0_risk = fit.metrics_impl.baseline_risk(data)
        result = {
            "schema_version": 1,
            "protocol_id": fit.PROTOCOL_ID,
            "status": "COMPLETED_EXACT_TWO_SIDED",
            "endpoint": "paired_safe_task_correctness",
            "threshold": float(selected_threshold),
            **fit.metrics_impl.mcnemar(
                data, b0_risk, calibrated, float(selected_threshold)
            ),
        }
    result.update(
        computed_at_utc=datetime.now(timezone.utc).isoformat(),
        implementation_lock_sha256=sha256(R5_LOCK),
        test_iid_ood_access=False,
    )
    write_json_new(MCNEMAR, result)
    metrics["mcnemar_safe_task"] = result
    metrics["mcnemar_exact_sha256"] = sha256(MCNEMAR)
    write_json_replace_before_final_lock(fit.METRICS, metrics)
    decision["metrics_sha256"] = sha256(fit.METRICS)
    decision["mcnemar_safe_task"] = result
    decision["mcnemar_exact_sha256"] = sha256(MCNEMAR)
    write_json_replace_before_final_lock(fit.DECISION, decision)


def fit_once_with_required_statistics() -> None:
    configure()
    fit.fit_once()
    add_locked_mcnemar_report()


def artifact_entry(path: Path) -> dict:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def create_artifact_manifest() -> None:
    configure()
    if ARTIFACT_MANIFEST.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 artifact manifest")
    decision = read_json(fit.DECISION)
    final_lock = read_json(fit.FINAL_LOCK)
    if final_lock.get("status") != decision.get("decision"):
        raise RuntimeError("terminal decision/final-lock mismatch")
    required_outputs = (
        fit.METRICS,
        fit.THRESHOLD_SCAN,
        fit.BOOTSTRAP,
        fit.DECISION,
        fit.FINAL_LOCK,
        fit.CALIBRATOR,
        fit.SCORED,
        MCNEMAR,
    )
    if not all(path.is_file() for path in required_outputs):
        raise RuntimeError("cannot manifest an incomplete Calibration-v3 result")
    protocol_and_code = (
        pipeline.CONTRACT,
        R5_LOCK,
        pipeline.R5_AUDIT,
        Path(__file__).resolve(),
        ROOT / "protocol/gazebo_calibration_v3_pipeline_r5.py",
        ROOT / "protocol/materialize_gazebo_calibration_v3.py",
        ROOT / "protocol/gazebo_calibration_v3_infer.py",
        ROOT / "protocol/gazebo_calibration_v3_fit.py",
        ROOT / "protocol/spatial_risk_v2_development.py",
    )
    model_and_locks = (
        inference.MODEL,
        pipeline.CAPTURE_LOCK,
        materialization.INPUT_LOCK,
        materialization.FINAL_LOCK,
        inference.B0_LOCK,
        inference.RISK_LOCK,
        fit.FIT_INPUT_LOCK,
    )
    dataset_files = tuple(sorted(path for path in DATASET.rglob("*") if path.is_file()))
    result_files = tuple(
        sorted(
            path
            for path in RESULT.rglob("*")
            if path.is_file() and path != ARTIFACT_MANIFEST
        )
    )
    groups = {
        "protocol_and_code": protocol_and_code,
        "model_and_stage_locks": model_and_locks,
        "materialized_inputs_and_oracle": dataset_files,
        "results_and_outputs": result_files,
    }
    payload = {
        "schema_version": 1,
        "protocol_id": fit.PROTOCOL_ID,
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision["decision"],
        "implementation_lock_sha256": sha256(R5_LOCK),
        "artifacts": {
            group: {
                str(path.relative_to(ROOT)): artifact_entry(path)
                for path in paths
            }
            for group, paths in groups.items()
        },
        "sealed_splits_accessed": [],
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_json_new(ARTIFACT_MANIFEST, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "sha256": sha256(ARTIFACT_MANIFEST),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "lock-materialization-inputs",
            "materialize",
            "freeze-materialization",
            "run-b0",
            "lock-b0",
            "run-risk",
            "lock-risk",
            "lock-fit-inputs",
            "fit-once",
            "freeze-calibration",
            "create-artifact-manifest",
        ),
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "RoboRefer/models/RoboRefer-2B-SFT",
    )
    args = parser.parse_args()
    configure()
    if args.command == "lock-materialization-inputs":
        materialization.lock_inputs()
    elif args.command == "materialize":
        materialization.materialize(DATASET)
    elif args.command == "freeze-materialization":
        materialization.freeze_materialization()
    elif args.command == "run-b0":
        inference.run_b0(args.base.resolve())
    elif args.command == "lock-b0":
        inference.lock_b0()
    elif args.command == "run-risk":
        inference.run_risk()
    elif args.command == "lock-risk":
        inference.lock_risk()
    elif args.command == "lock-fit-inputs":
        fit.lock_fit_inputs()
    elif args.command == "fit-once":
        fit_once_with_required_statistics()
    elif args.command == "freeze-calibration":
        fit.freeze()
    else:
        create_artifact_manifest()


if __name__ == "__main__":
    main()
