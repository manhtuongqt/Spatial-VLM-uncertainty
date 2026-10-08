#!/usr/bin/env python3
"""Materialization-only r2 wrapper for the immutable Calibration-v2 capture.

This revision changes only module path propagation. Scientific contracts,
capture bytes, geometry gates, duplicate rules, and all-or-nothing policy are
inherited unchanged from Calibration-v2.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import materialize_gazebo_calibration_v1 as impl


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "gazebo_calibration_v2"
CONFIG = ROOT / "ur3/ur3_perception/config"
SCENES = CONFIG / f"{PROTOCOL}_scenes.yaml"
ANNOTATIONS = CONFIG / f"{PROTOCOL}_annotations.yaml"
GATE = CONFIG / f"{PROTOCOL}_gate.yaml"
PRIMARY_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V2_CONTRACT_LOCK.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_calibration_v2_capture_compatibility_lock.json"
CAPTURE = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v2/capture_attempt_01"
OUT = ROOT / "datasets/Gazebo_calibration_v2"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v2"
LOCK = ROOT / "protocol/gazebo_calibration_v2_materialization_r2_lock.json"
AMENDMENT = ROOT / "protocol/gazebo_calibration_v2_materialization_r2_amendment.json"
QC = RESULT / "GAZEBO_CALIBRATION_V2_GEOMETRY_QC.json"
DUPLICATE_QC = RESULT / "GAZEBO_CALIBRATION_V2_RGB_DUPLICATE_QC.json"
CROSS_DUPLICATE_QC = RESULT / "GAZEBO_CALIBRATION_V2_CROSS_SPLIT_RGB_QC.json"
FAILED_DECISION = RESULT / "MATERIALIZATION_ATTEMPT_01_DECISION.md"


def configure() -> None:
    common = {
        "PROTOCOL": PROTOCOL,
        "SCENES": SCENES,
        "ANNOTATIONS": ANNOTATIONS,
        "GATE": GATE,
        "PRIMARY_LOCK": PRIMARY_LOCK,
        "CAPTURE_LOCK": CAPTURE_LOCK,
        "CAPTURE": CAPTURE,
        "OUT": OUT,
        "RESULT": RESULT,
        "LOCK": LOCK,
        "QC": QC,
    }
    for target in (impl, impl.base, impl.full):
        for name, value in common.items():
            setattr(target, name, value)
    impl.DEV_MANIFESTS = impl.PRIOR_MANIFESTS
    impl.base.DEV_MANIFESTS = impl.PRIOR_MANIFESTS
    impl.full.DUPLICATE_QC = DUPLICATE_QC
    impl.DUPLICATE_QC = DUPLICATE_QC
    impl.CROSS_DUPLICATE_QC = CROSS_DUPLICATE_QC
    # Functions in the inherited implementation resolve configure at runtime.
    impl.configure = configure


def lock_materialization() -> None:
    configure()
    if LOCK.exists():
        raise FileExistsError(f"refusing to overwrite lock: {LOCK}")
    impl.lock_materialization()
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    payload.update(
        materialization_revision="r2",
        status="LOCKED_BEFORE_GEOMETRY_QC_AND_MATERIALIZATION",
        r2_wrapper_source_sha256=impl.base.sha(Path(__file__).resolve()),
        amendment_sha256=impl.base.sha(AMENDMENT),
        failed_attempt_decision_sha256=impl.base.sha(FAILED_DECISION),
        immutable_capture_manifest_sha256=impl.base.sha(CAPTURE / "capture_manifest.json"),
        immutable_capture_input_manifest_sha256=impl.base.sha(CAPTURE / "input_manifest.jsonl"),
        locked_at_utc_r2=datetime.now(timezone.utc).isoformat(),
    )
    impl.base.dump(LOCK, payload)
    print(json.dumps({"status": "LOCKED_FINAL_R2", "sha256": impl.base.sha(LOCK)}, indent=2))


def verify_r2_lock() -> None:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    expected = {
        "r2_wrapper_source_sha256": impl.base.sha(Path(__file__).resolve()),
        "amendment_sha256": impl.base.sha(AMENDMENT),
        "failed_attempt_decision_sha256": impl.base.sha(FAILED_DECISION),
        "immutable_capture_manifest_sha256": impl.base.sha(CAPTURE / "capture_manifest.json"),
        "immutable_capture_input_manifest_sha256": impl.base.sha(CAPTURE / "input_manifest.jsonl"),
    }
    for key, digest in expected.items():
        if value.get(key) != digest:
            raise ValueError(f"materialization-r2 lock mismatch: {key}")


def materialize(output: Path) -> None:
    configure()
    verify_r2_lock()
    impl.materialize(output)
    manifest_path = output / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value.update(
        dataset="Gazebo_calibration_v2",
        protocol_id=PROTOCOL,
        materialization_revision="r2",
        materialization_lock_sha256=impl.base.sha(LOCK),
        amendment_sha256=impl.base.sha(AMENDMENT),
    )
    impl.base.dump(manifest_path, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("lock-materialization")
    run = sub.add_parser("materialize")
    run.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "lock-materialization":
        lock_materialization()
    else:
        materialize(args.output.resolve())


if __name__ == "__main__":
    main()
