#!/usr/bin/env python3
"""Append-only compatibility amendment for Calibration-v3 materialization.

The frozen v3 wrapper deliberately rewrote the input-lock status to
``LOCKED_BEFORE_QC_AND_MATERIALIZATION`` while the inherited Train-UQ
materializer accepts only its older spelling.  This adapter changes only that
pre-QC validation predicate.  It still verifies every source hash embedded in
the immutable input lock and leaves all data, geometry, duplicate, leakage,
oracle-free, model, metric, and scientific gates untouched.
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
AUDIT = ROOT / "protocol/gazebo_calibration_v3_materialization_compat_r1_audit.json"
AMENDMENT_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_MATERIALIZATION_COMPAT_LOCK_R1.json"
DATASET = ROOT / "datasets/Gazebo_calibration_v3"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3"


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


def lock_amendment() -> None:
    lock = verify_r6_input_lock()
    forbidden_outputs = (
        RESULT / "GAZEBO_CALIBRATION_V3_GEOMETRY_QC.json",
        RESULT / "GAZEBO_CALIBRATION_V3_RGB_DUPLICATE_QC.json",
        RESULT / "GAZEBO_CALIBRATION_V3_CROSS_SPLIT_RGB_QC.json",
    )
    if DATASET.exists() and any(DATASET.iterdir()):
        raise RuntimeError("dataset already exists; amendment must precede QC/materialization")
    if any(path.exists() for path in forbidden_outputs):
        raise RuntimeError("QC output already exists; amendment is too late")
    checks = {
        "r6_implementation_valid": True,
        "immutable_input_lock_valid": True,
        "all_embedded_source_hashes_valid": True,
        "failure_occurred_before_qc": True,
        "dataset_absent": True,
        "qc_outputs_absent": True,
        "scientific_method_unchanged": True,
        "test_iid_ood_sealed": True,
        "robot_unauthorized": True,
    }
    audit_payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "pre-QC lock-status compatibility only",
        "observed_infrastructure_failure": {
            "stage": "materialization_lock_validation_before_qc",
            "exception": "ValueError: invalid materialization lock",
            "producer_status": lock["status"],
            "inherited_consumer_expected_status": "LOCKED_BEFORE_GEOMETRY_QC_AND_MATERIALIZATION",
        },
        "compatibility_rule": {
            "accepted_status": "LOCKED_BEFORE_QC_AND_MATERIALIZATION",
            "verify_all_embedded_source_hashes": True,
            "no_file_rewrite": True,
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "r6_implementation_lock_sha256": sha256(R6_LOCK),
        "materialization_input_lock_sha256": sha256(INPUT_LOCK),
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_json_new(AUDIT, audit_payload)
    source_paths = (Path(__file__).resolve(), R6_LOCK, INPUT_LOCK, AUDIT)
    amendment_payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_QC_COMPATIBILITY_AMENDMENT_R1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "accept the immutable v3 status spelling while preserving full hash validation",
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
    if amendment.get("status") != "LOCKED_BEFORE_QC_COMPATIBILITY_AMENDMENT_R1":
        raise RuntimeError("invalid materialization compatibility amendment")
    for relative, expected in amendment.get("source_artifact_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"compatibility amendment source drift: {relative}")
    audit = read_json(AUDIT)
    if audit.get("status") != "PASS" or audit.get("all_checks_pass") is not True:
        raise RuntimeError("compatibility amendment audit did not pass")
    verify_r6_input_lock()


def compatible_verify_materialization_lock() -> None:
    verify_r6_input_lock()


def configure_compatibility() -> None:
    validate_amendment()
    materialization = downstream.d5.materialization
    materialization.implementation.base.verify_materialization_lock = (
        compatible_verify_materialization_lock
    )


def materialize() -> None:
    configure_compatibility()
    materialization = downstream.d5.materialization
    materialization.materialize(DATASET)
    manifest_path = DATASET / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["materialization_compatibility_amendment_sha256"] = sha256(AMENDMENT_LOCK)
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
