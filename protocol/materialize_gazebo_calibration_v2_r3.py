#!/usr/bin/env python3
"""Materialization-only r3 for the immutable Calibration-v2 capture.

The r3 change is limited to an explicit, non-recursive 128-record capture
validator. Geometry and duplicate logic remain inherited and locked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import yaml

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
LOCK = ROOT / "protocol/gazebo_calibration_v2_materialization_r3_lock.json"
AMENDMENT = ROOT / "protocol/gazebo_calibration_v2_materialization_r3_amendment.json"
QC = RESULT / "GAZEBO_CALIBRATION_V2_GEOMETRY_QC.json"
DUPLICATE_QC = RESULT / "GAZEBO_CALIBRATION_V2_RGB_DUPLICATE_QC.json"
CROSS_DUPLICATE_QC = RESULT / "GAZEBO_CALIBRATION_V2_CROSS_SPLIT_RGB_QC.json"
FAILED_DECISION_01 = RESULT / "MATERIALIZATION_ATTEMPT_01_DECISION.md"
FAILED_DECISION_02 = RESULT / "MATERIALIZATION_ATTEMPT_02_DECISION.md"


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
    impl.DUPLICATE_QC = DUPLICATE_QC
    impl.full.DUPLICATE_QC = DUPLICATE_QC
    impl.CROSS_DUPLICATE_QC = CROSS_DUPLICATE_QC
    impl.configure = configure


def validate_capture():
    scenes, annotations, gate = (
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in (SCENES, ANNOTATIONS, GATE)
    )
    contract = json.loads(PRIMARY_LOCK.read_text(encoding="utf-8"))
    capture_lock = json.loads(CAPTURE_LOCK.read_text(encoding="utf-8"))
    manifest = json.loads((CAPTURE / "capture_manifest.json").read_text(encoding="utf-8"))
    inputs = impl.base.jsonl(CAPTURE / "input_manifest.jsonl")
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("protocol_id") != PROTOCOL
        or len(inputs) != 128
        or len({row["scene_id"] for row in inputs}) != 128
    ):
        raise ValueError("capture is incomplete or has a bad record count")
    expected = {
        "pretrial_source_lock_sha256": impl.base.sha(CAPTURE_LOCK),
        "scene_config_sha256": impl.base.sha(SCENES),
        "annotation_file_sha256": impl.base.sha(ANNOTATIONS),
        "gate_config_file_sha256": impl.base.sha(GATE),
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"capture provenance mismatch: {name}")
    if capture_lock.get("parent_contract_lock_sha256") != impl.base.sha(PRIMARY_LOCK):
        raise ValueError("capture lock is not linked to the frozen contract")
    if contract.get("protocol_id") != PROTOCOL:
        raise ValueError("wrong frozen calibration contract")
    if manifest.get("model_inventory_sha256_preregistered") != capture_lock.get("model_inventory_sha256"):
        raise ValueError("capture/model-inventory linkage failed")
    if manifest.get("input_manifest_sha256") != impl.base.sha(CAPTURE / "input_manifest.jsonl"):
        raise ValueError("capture input-manifest hash changed")
    if manifest.get("source_artifacts_verified_before_capture") is not True:
        raise ValueError("capture sources were not verified before capture")
    return scenes, annotations, gate, manifest, inputs


def lock_materialization() -> None:
    configure()
    if LOCK.exists():
        raise FileExistsError(f"refusing to overwrite lock: {LOCK}")
    impl.base.validate_capture = validate_capture
    impl.base.lock_materialization()
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    payload.update(
        materialization_revision="r3",
        r3_wrapper_source_sha256=impl.base.sha(Path(__file__).resolve()),
        amendment_sha256=impl.base.sha(AMENDMENT),
        failed_attempt_01_decision_sha256=impl.base.sha(FAILED_DECISION_01),
        failed_attempt_02_decision_sha256=impl.base.sha(FAILED_DECISION_02),
        immutable_capture_manifest_sha256=impl.base.sha(CAPTURE / "capture_manifest.json"),
        immutable_capture_input_manifest_sha256=impl.base.sha(CAPTURE / "input_manifest.jsonl"),
        locked_at_utc_r3=datetime.now(timezone.utc).isoformat(),
    )
    impl.base.dump(LOCK, payload)
    print(json.dumps({"status": "LOCKED_FINAL_R3", "sha256": impl.base.sha(LOCK)}, indent=2))


def verify_r3_lock() -> None:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    expected = {
        "r3_wrapper_source_sha256": impl.base.sha(Path(__file__).resolve()),
        "amendment_sha256": impl.base.sha(AMENDMENT),
        "failed_attempt_01_decision_sha256": impl.base.sha(FAILED_DECISION_01),
        "failed_attempt_02_decision_sha256": impl.base.sha(FAILED_DECISION_02),
        "immutable_capture_manifest_sha256": impl.base.sha(CAPTURE / "capture_manifest.json"),
        "immutable_capture_input_manifest_sha256": impl.base.sha(CAPTURE / "input_manifest.jsonl"),
    }
    for key, digest in expected.items():
        if value.get(key) != digest:
            raise ValueError(f"materialization-r3 lock mismatch: {key}")


def materialize(output: Path) -> None:
    configure()
    verify_r3_lock()
    impl.calibration_validate_capture = validate_capture
    impl.base.validate_capture = validate_capture
    impl.materialize(output)
    manifest_path = output / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value.update(
        dataset="Gazebo_calibration_v2",
        protocol_id=PROTOCOL,
        materialization_revision="r3",
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
