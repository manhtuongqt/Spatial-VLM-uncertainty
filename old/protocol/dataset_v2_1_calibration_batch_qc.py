#!/usr/bin/python3
"""Observed RGB-D/semantic QC for one independent calibration batch."""

from __future__ import annotations

import argparse
from pathlib import Path

import dataset_v2_1_development_batch_qc as qualified_qc
from wp2_common import utc_now


PROTOCOL_ID = "roborefer_dataset_v2_1_calibration_capture_200"
DECISION = "GO_CALIBRATION_CAPTURE_200"
LAST_BATCH = "batch_002"

# The qualified checks are protocol-agnostic after these two identity values:
# synchronization, hashes, masks/depth relation geometry, visibility, replay
# signatures, oracle boundary and ordered-shutdown evidence remain unchanged.
qualified_qc.PROTOCOL_ID = PROTOCOL_ID
qualified_qc.DECISION = DECISION


def run_qc(
    workspace: Path,
    dataset_root: Path,
    plan_path: Path,
    lock_path: Path,
    manifest_path: Path,
    batch_id: str,
) -> dict:
    report = qualified_qc.run_qc(
        workspace, dataset_root, plan_path, lock_path, manifest_path, batch_id
    )
    report["decision"] = (
        "GO_CALIBRATION_MATERIALIZATION_FULL_QC"
        if report["passed"] and batch_id == LAST_BATCH
        else "GO_NEXT_CALIBRATION_BATCH"
        if report["passed"]
        else "FIX_CURRENT_CALIBRATION_BATCH_FIRST"
    )
    report.pop("calibration_or_test_created", None)
    report.update({
        "scientific_status": "INDEPENDENT_CALIBRATION_CAPTURE_QC",
        "calibration_dataset_capture_performed": True,
        "model_inference_performed": False,
        "calibrator_fit_performed": False,
        "training_performed": False,
        "test_opened": False,
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument(
        "--capture-plan", default="protocol/dataset_v2_1_calibration_capture_plan.json"
    )
    parser.add_argument(
        "--execution-lock", default="protocol/dataset_v2_1_calibration_execution_lock.json"
    )
    parser.add_argument(
        "--manifest", default="protocol/dataset_v2_1_calibration_manifest.json"
    )
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    try:
        report = run_qc(
            workspace,
            dataset_root,
            (workspace / args.capture_plan).resolve(),
            (workspace / args.execution_lock).resolve(),
            (workspace / args.manifest).resolve(),
            args.batch_id,
        )
    except Exception as exc:
        report = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "generated_at_utc": utc_now(),
            "batch_id": args.batch_id,
            "passed": False,
            "decision": "FIX_CURRENT_CALIBRATION_BATCH_FIRST",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "calibration_dataset_capture_performed": True,
            "model_inference_performed": False,
            "calibrator_fit_performed": False,
            "training_performed": False,
            "test_opened": False,
        }
    checkpoint = (
        dataset_root / "report_assets/checkpoints/batch_qc" / f"{args.batch_id}.json"
    )
    qualified_qc.atomic_json(checkpoint, report)
    print(
        "DATASET_V2_1_CALIBRATION_BATCH_QC "
        f"batch={args.batch_id} passed={report['passed']} decision={report['decision']} "
        f"errors={len(report['errors'])}"
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
