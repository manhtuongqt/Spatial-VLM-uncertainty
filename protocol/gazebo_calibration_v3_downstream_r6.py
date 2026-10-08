#!/usr/bin/env python3
"""Bind the preregistered Calibration-v3 downstream pipeline to R6."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import gazebo_calibration_v3_downstream_r5 as d5
import gazebo_calibration_v3_pipeline_r6 as pipeline


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3"
DATASET = ROOT / "datasets/Gazebo_calibration_v3"
R6_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R6.json"
ARTIFACT_MANIFEST = RESULT / "calibration_v3_artifact_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configure() -> None:
    pipeline.validate_r6_implementation()
    d5.R5_LOCK = R6_LOCK
    d5.pipeline = pipeline
    for module in (d5.materialization, d5.inference, d5.fit):
        module.IMPLEMENTATION_LOCK = R6_LOCK


def artifact_entry(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def create_artifact_manifest() -> None:
    configure()
    if ARTIFACT_MANIFEST.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 artifact manifest")
    decision = read_json(d5.fit.DECISION)
    final_lock = read_json(d5.fit.FINAL_LOCK)
    if final_lock.get("status") != decision.get("decision"):
        raise RuntimeError("terminal decision/final-lock mismatch")
    required_outputs = (
        d5.fit.METRICS, d5.fit.THRESHOLD_SCAN, d5.fit.BOOTSTRAP,
        d5.fit.DECISION, d5.fit.FINAL_LOCK, d5.fit.CALIBRATOR,
        d5.fit.SCORED, d5.MCNEMAR,
    )
    if not all(path.is_file() for path in required_outputs):
        raise RuntimeError("cannot manifest an incomplete Calibration-v3 result")
    protocol_and_code = (
        pipeline.CONTRACT, R6_LOCK, pipeline.R6_AUDIT,
        Path(__file__).resolve(),
        ROOT / "protocol/gazebo_calibration_v3_pipeline_r6.py",
        ROOT / "protocol/gazebo_calibration_v3_downstream_r5.py",
        ROOT / "protocol/materialize_gazebo_calibration_v3.py",
        ROOT / "protocol/gazebo_calibration_v3_infer.py",
        ROOT / "protocol/gazebo_calibration_v3_fit.py",
        ROOT / "protocol/spatial_risk_v2_development.py",
    )
    model_and_locks = (
        d5.inference.MODEL, pipeline.CAPTURE_LOCK,
        d5.materialization.INPUT_LOCK, d5.materialization.FINAL_LOCK,
        d5.inference.B0_LOCK, d5.inference.RISK_LOCK, d5.fit.FIT_INPUT_LOCK,
    )
    groups = {
        "protocol_and_code": protocol_and_code,
        "model_and_stage_locks": model_and_locks,
        "materialized_inputs_and_oracle": tuple(
            sorted(path for path in DATASET.rglob("*") if path.is_file())
        ),
        "results_and_outputs": tuple(
            sorted(
                path for path in RESULT.rglob("*")
                if path.is_file() and path != ARTIFACT_MANIFEST
            )
        ),
    }
    payload = {
        "schema_version": 1,
        "protocol_id": pipeline.PROTOCOL_ID,
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision["decision"],
        "implementation_lock_sha256": sha256(R6_LOCK),
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
    ARTIFACT_MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "decision": payload["decision"],
        "sha256": sha256(ARTIFACT_MANIFEST),
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "lock-materialization-inputs", "materialize", "freeze-materialization",
            "run-b0", "lock-b0", "run-risk", "lock-risk",
            "lock-fit-inputs", "fit-once", "freeze-calibration",
            "create-artifact-manifest",
        ),
    )
    parser.add_argument(
        "--base", type=Path,
        default=ROOT / "RoboRefer/models/RoboRefer-2B-SFT",
    )
    args = parser.parse_args()
    configure()
    if args.command == "lock-materialization-inputs":
        d5.materialization.lock_inputs()
    elif args.command == "materialize":
        d5.materialization.materialize(DATASET)
    elif args.command == "freeze-materialization":
        d5.materialization.freeze_materialization()
    elif args.command == "run-b0":
        d5.inference.run_b0(args.base.resolve())
    elif args.command == "lock-b0":
        d5.inference.lock_b0()
    elif args.command == "run-risk":
        d5.inference.run_risk()
    elif args.command == "lock-risk":
        d5.inference.lock_risk()
    elif args.command == "lock-fit-inputs":
        d5.fit.lock_fit_inputs()
    elif args.command == "fit-once":
        d5.fit_once_with_required_statistics()
    elif args.command == "freeze-calibration":
        d5.fit.freeze()
    else:
        create_artifact_manifest()


if __name__ == "__main__":
    main()
