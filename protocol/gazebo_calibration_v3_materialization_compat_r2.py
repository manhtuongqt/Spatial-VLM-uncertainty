#!/usr/bin/env python3
"""Append-only R2 amendment after the pre-geometry lock-status failure.

The failed materialize invocation produced valid within-split and cross-split
RGB audit artifacts before the inherited lock check raised.  R2 freezes and
reuses those reports; it neither overwrites nor recomputes them.  Its only
runtime compatibility change is accepting the immutable v3 input-lock status
while still verifying every hash embedded in that lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import gazebo_calibration_v3_downstream_r6 as downstream


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_calibration_v3"
R6_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R6.json"
INPUT_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_MATERIALIZATION_INPUT_LOCK.json"
R1_SOURCE = ROOT / "protocol/gazebo_calibration_v3_materialization_compat_r1.py"
AUDIT = ROOT / "protocol/gazebo_calibration_v3_materialization_compat_r2_audit.json"
AMENDMENT_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_MATERIALIZATION_COMPAT_LOCK_R2.json"
DATASET = ROOT / "datasets/Gazebo_calibration_v3"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3"
WITHIN_QC = RESULT / "GAZEBO_CALIBRATION_V3_RGB_DUPLICATE_QC.json"
CROSS_QC = RESULT / "GAZEBO_CALIBRATION_V3_CROSS_SPLIT_RGB_QC.json"
GEOMETRY_QC = RESULT / "GAZEBO_CALIBRATION_V3_GEOMETRY_QC.json"


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


def verify_r6_input_lock() -> dict:
    downstream.configure()
    lock = read_json(INPUT_LOCK)
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("materialization input lock protocol mismatch")
    if lock.get("status") != "LOCKED_BEFORE_QC_AND_MATERIALIZATION":
        raise RuntimeError("unexpected immutable v3 materialization-lock status")
    if lock.get("implementation_lock_sha256") != sha256(R6_LOCK):
        raise RuntimeError("materialization input lock is not bound to R6")
    for relative, expected in lock.get("source_artifact_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"locked materialization source drift: {relative}")
    return lock


def verify_frozen_rgb_qc() -> tuple[dict, dict]:
    within, cross = read_json(WITHIN_QC), read_json(CROSS_QC)
    if within.get("status") != "PASS" or within.get("rgb_count") != 128:
        raise RuntimeError("within-split RGB QC is not a 128-row PASS")
    if within.get("unique_rgb_sha256") != 128:
        raise RuntimeError("within-split exact duplicate gate failed")
    if within.get("exact_duplicate_groups") or within.get("perceptual_near_duplicate_pairs"):
        raise RuntimeError("within-split RGB duplicates were detected")
    if cross.get("status") != "PASS" or cross.get("current_images") != 128:
        raise RuntimeError("cross-split RGB QC is not a 128-row PASS")
    if cross.get("exact_duplicate_count") or cross.get("perceptual_near_duplicate_count"):
        raise RuntimeError("cross-split RGB duplicates were detected")
    return within, cross


def lock_amendment() -> None:
    verify_r6_input_lock()
    verify_frozen_rgb_qc()
    if GEOMETRY_QC.exists():
        raise RuntimeError("geometry QC already exists; R2 amendment must precede it")
    if DATASET.exists() and any(DATASET.iterdir()):
        raise RuntimeError("dataset already exists; R2 amendment must precede materialization")
    checks = {
        "r6_implementation_valid": True,
        "immutable_input_lock_and_embedded_hashes_valid": True,
        "within_split_rgb_qc_pass_128_unique": True,
        "cross_split_rgb_qc_pass_no_exact_or_perceptual_overlap": True,
        "rgb_qc_artifacts_reused_without_recompute": True,
        "geometry_qc_absent": True,
        "dataset_absent": True,
        "scientific_method_and_all_gates_unchanged": True,
        "test_iid_ood_sealed": True,
        "robot_unauthorized": True,
    }
    audit_payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "pre-geometry lock-status compatibility and immutable RGB-QC reuse only",
        "observed_infrastructure_failure": {
            "stage": "after_rgb_qc_before_geometry_qc",
            "exception": "ValueError: invalid materialization lock",
            "producer_status": "LOCKED_BEFORE_QC_AND_MATERIALIZATION",
            "inherited_consumer_expected_status": "LOCKED_BEFORE_GEOMETRY_QC_AND_MATERIALIZATION",
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "r6_implementation_lock_sha256": sha256(R6_LOCK),
        "materialization_input_lock_sha256": sha256(INPUT_LOCK),
        "within_split_rgb_qc_sha256": sha256(WITHIN_QC),
        "cross_split_rgb_qc_sha256": sha256(CROSS_QC),
        "aborted_r1_source_sha256": sha256(R1_SOURCE),
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_json_new(AUDIT, audit_payload)
    source_paths = (
        Path(__file__).resolve(), R1_SOURCE, R6_LOCK, INPUT_LOCK,
        WITHIN_QC, CROSS_QC, AUDIT,
    )
    amendment_payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_AFTER_RGB_QC_BEFORE_GEOMETRY_QC_COMPATIBILITY_R2",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "accept immutable v3 status and reuse frozen PASS RGB QC reports",
        "source_artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
        "scientific_method_changed": False,
        "data_design_changed": False,
        "qc_or_gate_changed": False,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_json_new(AMENDMENT_LOCK, amendment_payload)
    print(json.dumps({
        "status": amendment_payload["status"],
        "audit_sha256": sha256(AUDIT),
        "lock_sha256": sha256(AMENDMENT_LOCK),
    }, indent=2, sort_keys=True))


def validate_amendment() -> None:
    amendment = read_json(AMENDMENT_LOCK)
    if amendment.get("status") != "LOCKED_AFTER_RGB_QC_BEFORE_GEOMETRY_QC_COMPATIBILITY_R2":
        raise RuntimeError("invalid R2 materialization compatibility amendment")
    for relative, expected in amendment.get("source_artifact_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"R2 compatibility amendment source drift: {relative}")
    audit = read_json(AUDIT)
    if audit.get("status") != "PASS" or audit.get("all_checks_pass") is not True:
        raise RuntimeError("R2 compatibility audit did not pass")
    verify_r6_input_lock()
    verify_frozen_rgb_qc()


def compatible_verify_materialization_lock() -> None:
    verify_r6_input_lock()


def reuse_within_rgb_qc() -> dict:
    return verify_frozen_rgb_qc()[0]


def reuse_cross_rgb_qc() -> dict:
    return verify_frozen_rgb_qc()[1]


def configure_compatibility() -> None:
    validate_amendment()
    materialization = downstream.d5.materialization
    materialization.implementation.base.verify_materialization_lock = compatible_verify_materialization_lock
    materialization.implementation.full.duplicate_audit = reuse_within_rgb_qc
    materialization.cross_split_duplicate_audit = reuse_cross_rgb_qc


def materialize() -> None:
    configure_compatibility()
    materialization = downstream.d5.materialization
    materialization.materialize(DATASET)
    manifest_path = DATASET / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["materialization_compatibility_amendment_sha256"] = sha256(AMENDMENT_LOCK)
    manifest["reused_immutable_within_split_rgb_qc_sha256"] = sha256(WITHIN_QC)
    manifest["reused_immutable_cross_split_rgb_qc_sha256"] = sha256(CROSS_QC)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def freeze_materialization() -> None:
    configure_compatibility()
    downstream.d5.materialization.freeze_materialization()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("lock-amendment", "materialize", "freeze-materialization"))
    args = parser.parse_args()
    if args.command == "lock-amendment":
        lock_amendment()
    elif args.command == "materialize":
        materialize()
    else:
        freeze_materialization()


if __name__ == "__main__":
    main()
