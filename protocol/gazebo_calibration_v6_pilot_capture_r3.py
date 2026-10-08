#!/usr/bin/env python3
"""R3-bound authorization and terminal decision for Calibration-v6 pilot.

This append-only gate authorizes exactly one 32-family geometry-pilot capture
after live-preflight attempt 03 PASS.  It never authorizes Calibration-v6
capture, materialization, model inference, calibrator fitting, Test, or robot
policy execution.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import gazebo_calibration_v3_pipeline_r5 as capture_contract
import gazebo_calibration_v6_pipeline as base
import gazebo_calibration_v6_pipeline_r3 as r3
import generate_gazebo_calibration_v6_design as design


ROOT = design.ROOT
PID = design.PILOT
LOCK = ROOT / "protocol/gazebo_calibration_v6_geometry_pilot_capture_compatibility_lock.json"
CAPTURE = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/capture_attempt_01"
DECISION = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/PILOT_CAPTURE_ATTEMPT_01_DECISION.json"
QC = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/GEOMETRY_QC.json"
RGB_QC = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/RGB_DUPLICATE_QC.json"
QC_INPUT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/QC_INPUT_LOCK.json"


def sha256(path: Path) -> str:
    return design.sha256(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: dict[str, Any]) -> None:
    design.write_new(path, value)


def manifest_rows() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in design.paths(PID)[3].read_text(encoding="utf-8").splitlines()
        if line
    ]


def validate_order_and_balance() -> dict[str, Any]:

    rows = manifest_rows()
    scenes = yaml.safe_load(design.paths(PID)[0].read_text(encoding="utf-8"))["scenes"]
    expected_order = [row["scene_id"] for row in sorted(rows, key=lambda row: row["capture_order"])]
    actual_order = [scene["scene_id"] for scene in scenes]
    cells = Counter((row["state"], row["relation_variant"]) for row in rows)
    return {
        "rows": len(rows),
        "unique_family_ids": len({row["family_id"] for row in rows}) == 32,
        "unique_scene_ids": len({row["scene_id"] for row in rows}) == 32,
        "unique_layout_ids": len({row["layout_id"] for row in rows}) == 32,
        "unique_seeds": len({row["deterministic_seed"] for row in rows}) == 32,
        "unique_layout_signatures": len({row["layout_signature_sha256"] for row in rows}) == 32,
        "state_relation_cells": len(cells),
        "two_per_state_relation_cell": len(cells) == 16 and set(cells.values()) == {2},
        "scene_config_follows_deterministic_capture_order": actual_order == expected_order,
        "capture_order_values_are_0_to_31": sorted(row["capture_order"] for row in rows) == list(range(32)),
    }


def frozen_predecessors() -> dict[Path, str]:
    return {
        design.LOCK: "850921e200d921592128c76e11c8deeb5a7d6177ea315f1ff8d286f873084b6d",
        base.IMPLEMENTATION: "89d3416e7c870e635c3f6277a6dbbc2d5d12891c23f3a686c9a4fe2d4f7b821d",
        r3.R2_LOCK: "4ba0765cd8f68eccfd11fd905db4048856eec130652dd62c786c6b27bae4846b",
        r3.R3_LOCK: "f3416bd6843e1b72250e8d39f740843e476cce76359e1555f8fa3fe756b737cd",
        base.RESULT / "PREFLIGHT_STATIC.json": "05be37274246b8364a0fc65a2c45d8c9707abfebe85cf7777169cb9c4ea18002",
        r3.R3_PRECHECK: "0ace1bdc936110912c9d7226e2d7b5ed1238d934f620659543360ceeb67ea739",
        r3.ATTEMPT_03: "d0019a72f2cc69d664c3f36602e5dd34fcb2ead155e0c4feb0388b620da5ed36",
        r3.ATTEMPT_02: "b426317c8e10190d5704dd78e832b729110412bb4fea7ba5d7e69a6d9d527381",
        r3.ATTEMPT_02_FAILURE: "9821413b61281bb5882955354354821ae57d8d4b86a3862ed0f7c8cd49308d04",
        r3.WORLD: "6ee5cd6ede77d3fc8247557746ca25f0c03d0e08e0156d0360059d8e691c1193",
    }


def verify_before_authorization() -> None:
    base.validate_implementation()
    for path, expected in frozen_predecessors().items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen R3 predecessor drift: {path}")
    if read_json(r3.ATTEMPT_03).get("status") != "PASS":
        raise RuntimeError("live-preflight attempt 03 must PASS")
    if read_json(r3.R3_PRECHECK).get("status") != "PASS":
        raise RuntimeError("independent R3 TF precheck must PASS")
    if read_json(r3.ATTEMPT_02).get("status") != "BLOCKED":
        raise RuntimeError("attempt 02 must remain immutable BLOCKED")
    if LOCK.exists() or CAPTURE.exists() or DECISION.exists():
        raise RuntimeError("pilot authorization or downstream attempt already exists")
    if (ROOT / "datasets/Gazebo_calibration_v6").exists():
        raise RuntimeError("Calibration-v6 dataset exists before pilot")
    if (ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6/capture_attempt_01").exists():
        raise RuntimeError("Calibration-v6 128-family capture exists before pilot")
    summary = validate_order_and_balance()
    if summary["rows"] != 32 or not all(value is True for value in summary.values() if isinstance(value, bool)):
        raise RuntimeError(f"pilot manifest/order audit failed: {summary}")


def authorize() -> None:
    verify_before_authorization()
    implementation = read_json(base.IMPLEMENTATION)
    extra = [
        Path(__file__).resolve(),
        base.IMPLEMENTATION,
        base.RESULT / "PREFLIGHT_STATIC.json",
        r3.R2,
        r3.R2_LOCK,
        r3.R2_ENV,
        r3.ATTEMPT_02,
        r3.ATTEMPT_02_FAILURE,
        r3.R3_LOCK,
        r3.R3_PRECHECK,
        r3.ATTEMPT_03,
        r3.WORLD,
        design.REGISTRY,
        *design.paths(PID),
        ROOT / "protocol/gazebo_calibration_v6_qc.py",
    ]
    sources = dict(implementation["source_artifact_sha256"])
    sources.update({str(path.relative_to(ROOT)): sha256(path) for path in extra})
    missing = capture_contract.CAPTURE_REQUIRED_SOURCES - set(sources)
    if missing:
        raise RuntimeError(f"capture source closure incomplete: {sorted(missing)}")
    payload = {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "authorization_status": "CAPTURE_AUTHORIZED_AFTER_STATIC_AND_LIVE_PREFLIGHT_PASS",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "parent_contract_lock_sha256": sha256(design.LOCK),
        "implementation_lock_sha256": sha256(base.IMPLEMENTATION),
        "r3_amendment_sha256": sha256(r3.R3_LOCK),
        "independent_tf_precheck_sha256": sha256(r3.R3_PRECHECK),
        "live_preflight_attempt_03_sha256": sha256(r3.ATTEMPT_03),
        "prior_attempts": {
            "attempt_01": "IMMUTABLE_BLOCKED_ROS_PYTHON_ENVIRONMENT",
            "attempt_02": "IMMUTABLE_BLOCKED_MISSING_BASE_CAMERA_TF",
            "attempt_03": "PASS_NO_CAPTURE_AUTHORIZATION",
        },
        "expected_parent_families": 32,
        "population": "4 answerability states x 4 relations x 2 repetitions",
        "capture_order": "scene config order equals manifest capture_order 0..31 sorted by locked SHA-256 rule",
        "manifest_summary": validate_order_and_balance(),
        "capture_attempts_authorized": 1,
        "authorized_output": str(CAPTURE.relative_to(ROOT)),
        "source_artifact_sha256": sources,
        "model_inventory_sha256": "NO_MODEL_INFERENCE_DURING_CALIBRATION_V6_GEOMETRY_PILOT_CAPTURE",
        "policies": {
            "single_complete_capture_attempt_only": True,
            "partial_attempt_reuse_forbidden": True,
            "row_filter_repair_or_replacement_forbidden": True,
            "cross_attempt_merge_forbidden": True,
            "no_materialization": True,
            "no_model_inference": True,
            "no_calibrator_fit": True,
            "no_robot_manipulation": True,
            "semantic_labels_evaluator_only": True,
            "calibration_128_capture_not_authorized": True,
            "test_iid_ood_sealed": True,
            "robot_policy_sealed": True,
        },
        "capture_authorized": True,
        "calibration_capture_authorized": False,
        "pilot_families_captured_before_lock": 0,
        "calibration_families_captured": 0,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(LOCK, payload)
    print(json.dumps({"status": payload["authorization_status"], "sha256": sha256(LOCK), "manifest_summary": payload["manifest_summary"]}, indent=2))


def verify_lock() -> dict[str, Any]:
    value = read_json(LOCK)
    if value.get("status") != "LOCKED_BEFORE_CAPTURE_AND_INFERENCE" or value.get("authorization_status") != "CAPTURE_AUTHORIZED_AFTER_STATIC_AND_LIVE_PREFLIGHT_PASS":
        raise RuntimeError("pilot capture authorization status invalid")
    if value.get("expected_parent_families") != 32 or value.get("capture_attempts_authorized") != 1:
        raise RuntimeError("pilot capture count/attempt policy changed")
    if value.get("r3_amendment_sha256") != sha256(r3.R3_LOCK) or value.get("live_preflight_attempt_03_sha256") != sha256(r3.ATTEMPT_03):
        raise RuntimeError("R3/attempt-03 linkage failed")
    for name, digest in value["source_artifact_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"authorized source drift: {name}")
    required_true = (
        "single_complete_capture_attempt_only", "partial_attempt_reuse_forbidden",
        "row_filter_repair_or_replacement_forbidden", "cross_attempt_merge_forbidden",
        "no_materialization", "no_model_inference", "no_calibrator_fit",
        "calibration_128_capture_not_authorized", "test_iid_ood_sealed", "robot_policy_sealed",
    )
    if not all(value["policies"].get(name) is True for name in required_true):
        raise RuntimeError("capture policy drift")
    return value


def finalize() -> None:
    if DECISION.exists():
        raise FileExistsError("pilot attempt 01 decision already exists")
    lock = verify_lock()
    if not all(path.is_file() for path in (CAPTURE / "capture_manifest.json", CAPTURE / "input_manifest.jsonl", QC, RGB_QC, QC_INPUT)):
        raise RuntimeError("capture/QC artifacts incomplete")
    capture_manifest = read_json(CAPTURE / "capture_manifest.json")
    qc = read_json(QC)
    rgb = read_json(RGB_QC)
    records = [line for line in (CAPTURE / "input_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    passed = (
        capture_manifest.get("status") == "COMPLETE"
        and len(records) == 32
        and qc.get("status") == "PASS"
        and qc.get("passed_scene_count") == 32
        and rgb.get("status") == "PASS"
        and rgb.get("unique_rgb_sha256") == 32
    )
    sources = [LOCK, CAPTURE / "capture_manifest.json", CAPTURE / "input_manifest.jsonl", QC_INPUT, QC, RGB_QC]
    payload = {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "PASS" if passed else "REJECT",
        "decision": "PILOT_32_OF_32_QC_PASS_STOP_BEFORE_CALIBRATION_CAPTURE_AUTHORIZATION" if passed else "PILOT_ATTEMPT_01_REJECTED_STOP_NO_RETRY_UNDER_R3",
        "classification": "GEOMETRY_PILOT_DATA_GATE_NOT_SCIENTIFIC_CALIBRATION_RESULT",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "captured_family_count": len(records),
        "passed_scene_count": qc.get("passed_scene_count"),
        "verified_state_counts": qc.get("verified_state_counts"),
        "rgb_unique_count": rgb.get("unique_rgb_sha256"),
        "exact_rgb_duplicates": len(rgb.get("exact_duplicates", [])),
        "perceptual_near_duplicates": len(rgb.get("perceptual_near_duplicates", [])),
        "source_artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "capture_authorization_sha256": sha256(LOCK),
        "next_authorized_action": "Create a separate append-only Calibration-v6 128-family capture authorization; do not capture in this step." if passed else "Freeze this attempt and stop; any retry needs a new append-only revision and new namespace.",
        "calibration_capture_authorized": False,
        "dataset_materialized": False,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
        "locked_policy": lock["policies"],
    }
    write_new(DECISION, payload)
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("authorize", "verify", "finalize"))
    command = parser.parse_args().command
    if command == "authorize":
        authorize()
    elif command == "verify":
        verify_lock()
        print("PASS: Calibration-v6 geometry-pilot capture authorization")
    else:
        finalize()


if __name__ == "__main__":
    main()
