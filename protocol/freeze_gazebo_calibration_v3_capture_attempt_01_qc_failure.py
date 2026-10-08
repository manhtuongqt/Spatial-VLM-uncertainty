#!/usr/bin/env python3
"""Freeze Calibration-v3 capture attempt 01 after the preregistered QC gate failed."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_calibration_v3"
PROTOCOL = ROOT / "protocol"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3"
CAPTURE = RESULT / "capture_attempt_01"
DATASET = ROOT / "datasets/Gazebo_calibration_v3"
OUTPUT = RESULT / "CALIBRATION_V3_CAPTURE_ATTEMPT_01_QC_FAILURE_LOCK.json"

R6_LOCK = PROTOCOL / "GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R6.json"
CAPTURE_AUTH = PROTOCOL / "gazebo_calibration_v3_capture_compatibility_lock.json"
INPUT_LOCK = PROTOCOL / "GAZEBO_CALIBRATION_V3_MATERIALIZATION_INPUT_LOCK.json"
COMPAT_LOCK = PROTOCOL / "GAZEBO_CALIBRATION_V3_MATERIALIZATION_COMPAT_LOCK_R2.json"
COMPAT_AUDIT = PROTOCOL / "gazebo_calibration_v3_materialization_compat_r2_audit.json"
WITHIN_QC = RESULT / "GAZEBO_CALIBRATION_V3_RGB_DUPLICATE_QC.json"
CROSS_QC = RESULT / "GAZEBO_CALIBRATION_V3_CROSS_SPLIT_RGB_QC.json"
GEOMETRY_QC = RESULT / "GAZEBO_CALIBRATION_V3_GEOMETRY_QC.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable failure lock: {OUTPUT}")
    required = (
        R6_LOCK, CAPTURE_AUTH, INPUT_LOCK, COMPAT_LOCK, COMPAT_AUDIT,
        CAPTURE / "capture_manifest.json", CAPTURE / "input_manifest.jsonl",
        WITHIN_QC, CROSS_QC, GEOMETRY_QC,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze incomplete evidence: {missing}")

    capture_manifest = read_json(CAPTURE / "capture_manifest.json")
    inputs = jsonl(CAPTURE / "input_manifest.jsonl")
    within, cross, geometry = map(read_json, (WITHIN_QC, CROSS_QC, GEOMETRY_QC))
    if capture_manifest.get("status") != "COMPLETE" or len(inputs) != 128:
        raise RuntimeError("capture attempt is not COMPLETE with 128 rows")
    if len({row["scene_id"] for row in inputs}) != 128:
        raise RuntimeError("capture attempt does not contain 128 unique scene IDs")
    if within.get("status") != "PASS" or cross.get("status") != "PASS":
        raise RuntimeError("RGB QC state does not match the observed failure path")
    if geometry.get("status") != "BLOCKED" or geometry.get("failed_scene_count") != 6:
        raise RuntimeError("geometry QC state does not match the observed failure path")
    if geometry.get("dataset_materialized") is not False:
        raise RuntimeError("geometry QC unexpectedly reports a materialized dataset")
    if DATASET.exists() and any(DATASET.iterdir()):
        raise RuntimeError("dataset exists after a blocked all-or-nothing QC gate")

    forbidden = (
        RESULT / "b0_predictions.jsonl",
        RESULT / "spatial_risk_v2_predictions.jsonl",
        RESULT / "GAZEBO_CALIBRATION_V3_METRICS.json",
        RESULT / "calibration_v3_threshold_scan.jsonl",
        RESULT / "calibration_v3_family_bootstrap_10000.json",
        RESULT / "calibration_v3_mcnemar_exact.json",
        RESULT / "FINAL_CALIBRATION_V3_DECISION.json",
        RESULT / "CALIBRATOR_THRESHOLD_V3_LOCK.json",
        RESULT / "calibration_v3_artifact_manifest.json",
    )
    existing_forbidden = [str(path.relative_to(ROOT)) for path in forbidden if path.exists()]
    if existing_forbidden:
        raise RuntimeError(f"downstream artifacts exist despite blocked QC: {existing_forbidden}")

    failed = [row for row in geometry["scenes"] if row.get("reasons")]
    if len(failed) != 6 or any(
        row.get("requested_state") != "INSUFFICIENT_EVIDENCE" for row in failed
    ):
        raise RuntimeError("unexpected geometry failure composition")
    capture_files = tuple(sorted(path for path in CAPTURE.rglob("*") if path.is_file()))
    source_paths = (
        Path(__file__).resolve(), R6_LOCK, CAPTURE_AUTH, INPUT_LOCK,
        COMPAT_LOCK, COMPAT_AUDIT, WITHIN_QC, CROSS_QC, GEOMETRY_QC,
    )
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "CAPTURE_ATTEMPT_01_FROZEN_DATA_QC_FAILURE_NO_INFERENCE",
        "classification": "DATA_ATTEMPT_FAILURE_NOT_SCIENTIFIC_CALIBRATOR_DECISION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture": {
            "attempt": 1,
            "status": capture_manifest["status"],
            "records": len(inputs),
            "unique_scene_ids": len({row["scene_id"] for row in inputs}),
            "capture_file_count": len(capture_files),
        },
        "design_balance": {
            "state_counts": geometry["state_counts"],
            "relation_variant_counts": geometry["relation_variant_counts"],
            "split_counts": geometry["split_counts"],
            "quota_4_state_x_4_relation_x_8": True,
        },
        "qc": {
            "within_split_rgb": {
                "status": within["status"],
                "rgb_count": within["rgb_count"],
                "unique_rgb_sha256": within["unique_rgb_sha256"],
                "exact_duplicate_groups": len(within["exact_duplicate_groups"]),
                "perceptual_near_duplicate_pairs": len(within["perceptual_near_duplicate_pairs"]),
            },
            "cross_split_rgb": {
                "status": cross["status"],
                "current_images": cross["current_images"],
                "prior_images": cross["prior_images"],
                "exact_duplicate_count": cross["exact_duplicate_count"],
                "perceptual_near_duplicate_count": cross["perceptual_near_duplicate_count"],
            },
            "geometry": {
                "status": geometry["status"],
                "records": geometry["records"],
                "failed_scene_count": geometry["failed_scene_count"],
                "verified_state_counts": geometry["verified_state_counts"],
                "failure_reason_counts": dict(sorted(Counter(
                    reason for row in failed for reason in row["reasons"]
                ).items())),
                "failed_scenes": [
                    {
                        "scene_id": row["scene_id"],
                        "family_id": row["family_id"],
                        "requested_state": row["requested_state"],
                        "relation_variant": row["relation_variant"],
                        "target_visible_pixels": row.get("target_visible_pixels"),
                        "reasons": row["reasons"],
                    }
                    for row in failed
                ],
                "leakage_checks": geometry["leakage_checks"],
                "dataset_materialized": geometry["dataset_materialized"],
            },
        },
        "required_stop_actions": {
            "capture_attempt_frozen": True,
            "row_filtering_or_replacement": False,
            "recapture": False,
            "dataset_materialized": False,
            "b0_inference_run": False,
            "spatial_risk_v2_inference_run": False,
            "calibrator_fit_call_count": 0,
            "final_scientific_decision_created": False,
        },
        "source_artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
        "capture_file_inventory": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in capture_files
        },
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "failed_scene_count": geometry["failed_scene_count"],
        "capture_file_count": len(capture_files),
        "sha256": sha256(OUTPUT),
        "path": str(OUTPUT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
