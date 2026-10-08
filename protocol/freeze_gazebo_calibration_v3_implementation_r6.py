#!/usr/bin/env python3
"""Audit and freeze append-only Calibration-v3 implementation R6."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

import freeze_gazebo_calibration_v3_implementation_r5 as f5
import gazebo_calibration_v3_pipeline_r6 as r6


ROOT = Path(__file__).resolve().parents[1]
R6_RUNNER = ROOT / "protocol/gazebo_calibration_v3_pipeline_r6.py"
R6_DOWNSTREAM = ROOT / "protocol/gazebo_calibration_v3_downstream_r6.py"
R6_FREEZER = Path(__file__).resolve()
R6_AUDIT = ROOT / "protocol/gazebo_calibration_v3_implementation_r6_audit.json"
R6_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R6.json"
ATTEMPT_04 = r6.ATTEMPT_04


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def cli_choices(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    output: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            value = keyword.value
            if keyword.arg == "choices" and isinstance(value, (ast.Tuple, ast.List)):
                if all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts):
                    output.update(str(item.value) for item in value.elts)
    return output


def build_audit() -> dict:
    f5.validate()
    runner = R6_RUNNER.read_text(encoding="utf-8")
    downstream = R6_DOWNSTREAM.read_text(encoding="utf-8")
    contract = read_json(r6.CONTRACT)
    gate = yaml.safe_load(r6.GATE.read_text(encoding="utf-8"))
    annotations = yaml.safe_load(r6.ANNOTATIONS.read_text(encoding="utf-8"))
    cells = Counter(
        (item["state"], item["relation_variant"])
        for item in annotations["scenes"].values()
    )
    regression = r6.parser_regression_results()
    capture_required = f5.assigned_string_set(f5.CAPTURE_NODE, "REQUIRED_SOURCE_ARTIFACTS")
    expected_downstream = {
        "lock-materialization-inputs", "materialize", "freeze-materialization",
        "run-b0", "lock-b0", "run-risk", "lock-risk", "lock-fit-inputs",
        "fit-once", "freeze-calibration", "create-artifact-manifest",
    }
    checks = {
        "r5_lock_hash_unchanged": sha256(r6.R5_LOCK) == r6.R5_LOCK_SHA256,
        "r5_audit_hash_unchanged": sha256(r6.R5_AUDIT) == r6.R5_AUDIT_SHA256,
        "attempt_03_hash_unchanged": sha256(r6.ATTEMPT_03) == r6.ATTEMPT_03_SHA256,
        "attempts_01_02_03_preserved_blocked": all(
            read_json(path).get("status") == "BLOCKED"
            for path in (r6.ATTEMPT_01, r6.ATTEMPT_02, r6.ATTEMPT_03)
        ),
        "parser_regression_pass": regression["status"] == "PASS",
        "wrist_k_regression_pass": regression["checks"]["wrist_k_parses_exactly_9_finite_floats"],
        "base_k_regression_pass": regression["checks"]["base_k_parses_exactly_9_finite_floats"],
        "joint_warning_regression_pass": regression["checks"]["joint_warning_preamble_is_ignored"],
        "six_joint_pose_regression_pass": regression["checks"]["recorded_joint_pose_passes_existing_tolerance"],
        "r6_runner_parses": bool(ast.parse(runner)),
        "r6_downstream_parses": bool(ast.parse(downstream)),
        "r6_output_append_only_attempt_04": "PREFLIGHT_LIVE_ATTEMPT_04.json" in runner and "refusing to overwrite" in runner,
        "r6_camera_array_parser_strict_length_and_finite": all(token in runner for token in ("expected_length", "math.isfinite", "len(tokens)")),
        "r6_joint_parser_starts_at_header": 'r"(?m)^header:' in runner,
        "r6_retains_existing_tolerances": "JOINT_POSITION_TOLERANCE_RAD" in runner and "camera_info_valid" not in runner,
        "capture_node_required_sources_unchanged": r6.CAPTURE_REQUIRED_SOURCES == capture_required,
        "capture_lock_schema_exact": '"status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE"' in runner,
        "capture_requires_attempt_04_pass": "live preflight attempt 04 must PASS" in runner,
        "single_capture_attempt_only": '"capture_attempts_authorized": 1' in runner,
        "downstream_cli_complete": cli_choices(R6_DOWNSTREAM) == expected_downstream,
        "downstream_binds_r6_lock": "module.IMPLEMENTATION_LOCK = R6_LOCK" in downstream,
        "artifact_manifest_includes_r6_code": all(token in downstream for token in (
            "gazebo_calibration_v3_pipeline_r6.py", "Path(__file__).resolve()",
            "materialized_inputs_and_oracle", "results_and_outputs",
        )),
        "attempt_04_absent_before_lock": not ATTEMPT_04.exists(),
        "capture_authorization_absent": not r6.CAPTURE_LOCK.exists(),
        "capture_absent": not r6.CAPTURE_ROOT.exists(),
        "dataset_absent": not r6.DATASET_ROOT.exists(),
        "family_count_128": len(annotations["scenes"]) == 128,
        "cell_count_16": len(cells) == 16,
        "every_cell_has_8": set(cells.values()) == {8},
        "feature_count_12_unchanged": len(contract["frozen_upstream"]["feature_names_in_order"]) == 12,
        "affine_logit_unchanged": contract["calibration_protocol"]["calibrator"] == "positive_slope_affine_logit",
        "threshold_grid_unchanged": contract["operating_point_rule"]["threshold_grid"] == {
            "minimum": 0.0, "maximum": 1.0, "step": 0.01, "count": 101
        },
        "refit_forbidden": gate["policies"].get("no_grounding_or_risk_estimator_refit") is True,
        "test_iid_ood_sealed": gate["sealed"].get("gazebo_test_iid") is True and gate["sealed"].get("gazebo_test_ood") is True,
    }
    return {
        "schema_version": 1,
        "protocol_id": r6.PROTOCOL_ID,
        "implementation_revision": "r6",
        "audit_scope": "append_only_live_output_parser_repair_only",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "parser_regression": regression,
        "cell_counts": {
            f"{state}|{relation}": count
            for (state, relation), count in sorted(cells.items())
        },
        "capture_authorized": False,
        "scientific_method_changed": False,
    }


def source_paths() -> tuple[Path, ...]:
    inherited = tuple(ROOT / name for name in read_json(r6.R5_LOCK)["source_artifact_sha256"])
    return tuple(dict.fromkeys(inherited + (
        r6.R5_LOCK, r6.R5_AUDIT, r6.ATTEMPT_03,
        R6_RUNNER, R6_DOWNSTREAM, R6_FREEZER,
    )))


def build_lock(audit: dict) -> dict:
    sources = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in source_paths()
    }
    return {
        "schema_version": 1,
        "protocol_id": r6.PROTOCOL_ID,
        "implementation_revision": "r6",
        "status": "IMPLEMENTATION_R6_FROZEN_BEFORE_LIVE_PREFLIGHT_ATTEMPT_04",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_scope": "Parse ROS NumPy-style CameraInfo K arrays and warning-prefixed JointState YAML only; no scientific method or gate change.",
        "parent_contract_lock_sha256": sha256(r6.CONTRACT),
        "parent_r5_implementation_lock_sha256": sha256(r6.R5_LOCK),
        "prior_blocked_live_preflight_attempt_03_sha256": sha256(r6.ATTEMPT_03),
        "implementation_audit_path": str(R6_AUDIT.relative_to(ROOT)),
        "implementation_audit_sha256": sha256(R6_AUDIT),
        "source_artifact_sha256": dict(sorted(sources.items())),
        "scientific_invariants": {
            "grounding_backbone_b0_unchanged": True,
            "spatial_risk_v2_and_12_features_unchanged": True,
            "family_population_and_4x4x8_unchanged": True,
            "unsafe_metrics_bootstrap_and_gates_unchanged": True,
            "positive_slope_affine_logit_single_fit_unchanged": True,
            "threshold_grid_and_selection_unchanged": True,
            "calibration_v2_closed": True,
            "test_iid_ood_sealed": True,
            "robot_closed": True,
        },
        "capture_authorized": False,
        "capture_performed": False,
        "dataset_materialized": False,
        "model_inference_performed": False,
        "calibrator_fit_performed": False,
        "next_authorized_action": "Start simulation-only, move UR3 to the locked camera pose, and run live preflight attempt 04 exactly once.",
    }


def freeze() -> None:
    if R6_AUDIT.exists() or R6_LOCK.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 R6 audit or lock")
    audit = build_audit()
    write_json(R6_AUDIT, audit)
    if audit["status"] != "PASS":
        raise RuntimeError("Calibration-v3 R6 implementation audit failed; lock not created")
    lock = build_lock(audit)
    write_json(R6_LOCK, lock)
    print(json.dumps({
        "status": lock["status"],
        "audit_sha256": sha256(R6_AUDIT),
        "lock_sha256": sha256(R6_LOCK),
        "capture_authorized": False,
    }, indent=2, sort_keys=True))


def validate() -> None:
    audit = read_json(R6_AUDIT)
    lock = read_json(R6_LOCK)
    if audit.get("status") != "PASS":
        raise RuntimeError("Calibration-v3 R6 audit is not PASS")
    if lock.get("status") != "IMPLEMENTATION_R6_FROZEN_BEFORE_LIVE_PREFLIGHT_ATTEMPT_04":
        raise RuntimeError("Calibration-v3 R6 lock state is invalid")
    if lock.get("implementation_audit_sha256") != sha256(R6_AUDIT):
        raise RuntimeError("Calibration-v3 R6 audit hash linkage failed")
    for name, digest in lock["source_artifact_sha256"].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Calibration-v3 R6 source drift: {name}")
    print(json.dumps({
        "status": "PASS",
        "lock_sha256": sha256(R6_LOCK),
        "audit_sha256": sha256(R6_AUDIT),
        "capture_authorized_by_r6_lock": False,
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preview", "freeze", "validate"))
    command = parser.parse_args().command
    if command == "preview":
        print(json.dumps(build_audit(), indent=2, sort_keys=True))
    elif command == "freeze":
        freeze()
    else:
        validate()


if __name__ == "__main__":
    main()
