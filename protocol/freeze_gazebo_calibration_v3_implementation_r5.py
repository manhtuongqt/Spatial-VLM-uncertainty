#!/usr/bin/env python3
"""Audit and freeze Calibration-v3 implementation revision R5 before data."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

import gazebo_calibration_v3_pipeline_r4 as r4


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json"
R4_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R4.json"
R4_AUDIT = ROOT / "protocol/gazebo_calibration_v3_implementation_r4_audit.json"
STATIC = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_STATIC.json"
ATTEMPT_01 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE.json"
ATTEMPT_02 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE_ATTEMPT_02.json"
ATTEMPT_03 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/PREFLIGHT_LIVE_ATTEMPT_03.json"
R5_RUNNER = ROOT / "protocol/gazebo_calibration_v3_pipeline_r5.py"
R5_DOWNSTREAM = ROOT / "protocol/gazebo_calibration_v3_downstream_r5.py"
R5_FREEZER = Path(__file__).resolve()
R5_AUDIT = ROOT / "protocol/gazebo_calibration_v3_implementation_r5_audit.json"
R5_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R5.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_calibration_v3_capture_compatibility_lock.json"
CAPTURE = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/capture_attempt_01"
DATASET = ROOT / "datasets/Gazebo_calibration_v3"
FAMILY_MANIFEST = ROOT / "protocol/gazebo_calibration_v3_family_manifest.jsonl"
SPLIT_MANIFEST = ROOT / "protocol/gazebo_calibration_v3_split_manifest.json"
SCENES = ROOT / "ur3/ur3_perception/config/gazebo_calibration_v3_scenes.yaml"
ANNOTATIONS = ROOT / "ur3/ur3_perception/config/gazebo_calibration_v3_annotations.yaml"
GATE = ROOT / "ur3/ur3_perception/config/gazebo_calibration_v3_gate.yaml"
WORLD = ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf"
CAPTURE_NODE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
CAPTURE_LAUNCH = ROOT / "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py"
MATERIALIZER = ROOT / "protocol/materialize_gazebo_calibration_v3.py"
INFERENCE = ROOT / "protocol/gazebo_calibration_v3_infer.py"
FITTER = ROOT / "protocol/gazebo_calibration_v3_fit.py"
RISK_RUNNER = ROOT / "protocol/spatial_risk_v2_development.py"
METHOD_LOCK = ROOT / "protocol/spatial_risk_method_v2_hypothesis_lock.json"
DEVELOPMENT_DECISION = ROOT / "results/spatial_vlm_refspatial_v1/wp5_spatial_risk_v2_development/spatial_risk_v2_development_decision.json"
MODEL = ROOT / "results/spatial_vlm_refspatial_v1/wp5_spatial_risk_v2_development/spatial_risk_v2_estimator.joblib"

FIXED_HASHES = {
    CONTRACT: "99b430b1c55e1762f8460db4e8134b4a8782403e466d71ea86ea3da847bfc841",
    R4_LOCK: "36b97adae26a5e4927bfe3c4358836056a4564ff7a89371c7a76e7e343393644",
    STATIC: "5ccba3ee03efd3900048a51a3ec33185a8da26266d8668fe4c9a1d1c492e6f77",
    ATTEMPT_01: "25f9347fccf195c3d39da18559150e2df16018cba73dada3c54aa4b289b41ea0",
    ATTEMPT_02: "53d1a2c70fa4fc2f2ccf25176d24381bd1b0b80ca1743c92597feb7f84d90686",
    METHOD_LOCK: "8f6e21941fd04dc4ac91fe0fb918d65b5e8d191a52db9dd27cc93d6c39aff5fc",
    DEVELOPMENT_DECISION: "d0e5c91c66f4f65a8dc6a5bd0c91bf91552d35275361fe34b1b33b4b6bf62c56",
    MODEL: "f296e2e33abdd66437d2976a91ff78b4e707d67244f3b998dc9d9b011cbd5bc2",
    FAMILY_MANIFEST: "07b438b96e29ab85e7fc543b426e27358f6819735f606cf2613e6d96776c1054",
    SPLIT_MANIFEST: "92b5fb485a41c468fd2ef15b3d58b7ad5f106f199043916ee0b12c5a6bf9e274",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def assigned_string_set(path: Path, variable: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == variable for target in targets):
                value = node.value
                if isinstance(value, ast.Set) and all(
                    isinstance(item, ast.Constant) and isinstance(item.value, str)
                    for item in value.elts
                ):
                    return {str(item.value) for item in value.elts}
    raise RuntimeError(f"cannot statically extract {variable} from {path}")


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
    r4.validate_r4_implementation()
    runner = R5_RUNNER.read_text(encoding="utf-8")
    downstream = R5_DOWNSTREAM.read_text(encoding="utf-8")
    contract = read_json(CONTRACT)
    attempt_02 = read_json(ATTEMPT_02)
    annotations = yaml.safe_load(ANNOTATIONS.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    cells = Counter(
        (item["state"], item["relation_variant"])
        for item in annotations["scenes"].values()
    )
    capture_required = assigned_string_set(CAPTURE_NODE, "REQUIRED_SOURCE_ARTIFACTS")
    runner_required = assigned_string_set(R5_RUNNER, "CAPTURE_REQUIRED_SOURCES")
    expected_downstream_commands = {
        "lock-materialization-inputs", "materialize", "freeze-materialization",
        "run-b0", "lock-b0", "run-risk", "lock-risk", "lock-fit-inputs",
        "fit-once", "freeze-calibration", "create-artifact-manifest",
    }
    fixed_hashes_match = all(path.is_file() and sha256(path) == digest for path, digest in FIXED_HASHES.items())
    checks = {
        "r4_implementation_still_valid": True,
        "all_fixed_upstream_hashes_match": fixed_hashes_match,
        "attempt_02_preserved_blocked": attempt_02.get("status") == "BLOCKED",
        "attempt_03_absent_before_r5_lock": not ATTEMPT_03.exists(),
        "capture_authorization_absent": not CAPTURE_LOCK.exists(),
        "capture_absent": not CAPTURE.exists(),
        "dataset_absent": not DATASET.exists(),
        "r5_runner_parses": bool(ast.parse(runner)),
        "r5_downstream_parses": bool(ast.parse(downstream)),
        "r5_attempt_is_append_only_attempt_03": "PREFLIGHT_LIVE_ATTEMPT_03.json" in runner and "refusing to overwrite" in runner,
        "ros_pythonpath_preserves_existing_paths": all(token in runner for token in ("ROS_SITE_PACKAGES", "existing", "PYTHONNOUSERSITE")),
        "subprocess_probes_receive_preserved_environment": "env=ros_environment()" in runner,
        "full_r4_live_checks_retained": all(token in runner for token in (
            "wrist_rgb_live_valid", "wrist_metric_depth_live_valid",
            "wrist_intrinsics_match_lock", "base_rgb_live_valid",
            "base_metric_depth_live_valid", "base_intrinsics_valid",
            "base_to_wrist_camera_tf_valid", "fixed_view_joint_pose_within_tolerance",
            "full_scene_reset_probe_success", "post_reset_geometry_sensor_probe_success",
        )),
        "capture_node_schema_status_exact": '"status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE"' in runner,
        "capture_required_sources_match_node_exactly": runner_required == capture_required,
        "capture_requires_live_attempt_03_pass": all(token in runner for token in (
            "live.get(\"status\") != \"PASS\"", "live_preflight_sha256", "sha256(ATTEMPT_03)"
        )),
        "single_capture_attempt_and_no_overwrite": "capture_attempts_authorized\": 1" in runner and "refusing to overwrite Calibration-v3 capture attempt 01" in runner,
        "downstream_cli_complete": cli_choices(R5_DOWNSTREAM) == expected_downstream_commands,
        "downstream_binds_all_modules_to_r5": "for module in (materialization, inference, fit)" in downstream and "module.IMPLEMENTATION_LOCK = R5_LOCK" in downstream,
        "mcnemar_exact_report_preregistered": "metrics_impl.mcnemar" in downstream and "calibration_v3_mcnemar_exact.json" in downstream,
        "artifact_manifest_hashes_inputs_code_model_outputs": all(token in downstream for token in (
            "protocol_and_code", "model_and_stage_locks",
            "materialized_inputs_and_oracle", "results_and_outputs",
        )),
        "family_count_128": len(annotations["scenes"]) == 128,
        "cell_count_16": len(cells) == 16,
        "every_cell_has_8": set(cells.values()) == {8},
        "method_feature_count_12_unchanged": len(contract["frozen_upstream"]["feature_names_in_order"]) == 12,
        "affine_logit_and_optimizer_unchanged": (
            contract["calibration_protocol"]["calibrator"] == "positive_slope_affine_logit"
            and "L-BFGS-B" in contract["calibration_protocol"]["optimizer"]
        ),
        "threshold_grid_unchanged": contract["operating_point_rule"]["threshold_grid"] == {
            "minimum": 0.0, "maximum": 1.0, "step": 0.01, "count": 101
        },
        "gate_forbids_refit": gate["policies"].get("no_grounding_or_risk_estimator_refit") is True,
        "test_iid_ood_sealed": (
            gate["sealed"].get("gazebo_test_iid") is True
            and gate["sealed"].get("gazebo_test_ood") is True
        ),
    }
    return {
        "schema_version": 1,
        "protocol_id": "gazebo_calibration_v3",
        "implementation_revision": "r5",
        "audit_scope": "append_only_ros_environment_capture_compatibility_and_complete_preregistered_reporting",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "cell_counts": {
            f"{state}|{relation}": count
            for (state, relation), count in sorted(cells.items())
        },
        "capture_required_source_count": len(capture_required),
        "capture_authorized": False,
        "scientific_method_changed": False,
    }


def source_paths() -> tuple[Path, ...]:
    return (
        CONTRACT, R4_LOCK, R4_AUDIT, STATIC, ATTEMPT_01, ATTEMPT_02,
        FAMILY_MANIFEST, SPLIT_MANIFEST, SCENES, ANNOTATIONS, GATE, WORLD,
        CAPTURE_NODE, CAPTURE_LAUNCH, MATERIALIZER, INFERENCE, FITTER,
        RISK_RUNNER, METHOD_LOCK, DEVELOPMENT_DECISION, MODEL,
        R5_RUNNER, R5_DOWNSTREAM, R5_FREEZER,
        ROOT / "protocol/gazebo_calibration_v3_pipeline_r4.py",
        ROOT / "protocol/generate_gazebo_calibration_v3_contract.py",
        ROOT / "protocol/gazebo_train_uq_v1_infer.py",
        ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_runner.py",
        ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_validate.py",
        ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_evaluator.py",
        ROOT / "ur3/ur3_perception/scripts/roborefer_grounder.py",
        ROOT / "ur3/ur3_perception/scripts/spatial_point_utils.py",
        ROOT / "ur3/ur3_perception/scripts/move_camera_to_view.py",
        ROOT / "ur3/ur3_moveit_control/launch/ur3_susgrip_sim.launch.py",
    )


def build_lock(audit: dict) -> dict:
    sources = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in source_paths()
    }
    return {
        "schema_version": 1,
        "protocol_id": "gazebo_calibration_v3",
        "implementation_revision": "r5",
        "status": "IMPLEMENTATION_R5_FROZEN_BEFORE_LIVE_PREFLIGHT_ATTEMPT_03",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_scope": "Operational ROS Python environment preservation, exact capture-lock compatibility, R5 provenance propagation, and completion of already-preregistered McNemar/artifact-manifest reporting only.",
        "parent_contract_lock_sha256": sha256(CONTRACT),
        "parent_r4_implementation_lock_sha256": sha256(R4_LOCK),
        "static_preflight_sha256": sha256(STATIC),
        "prior_blocked_live_preflight": {
            "attempt_01_sha256": sha256(ATTEMPT_01),
            "attempt_02_sha256": sha256(ATTEMPT_02),
            "preserve_immutable": True,
        },
        "implementation_audit_path": str(R5_AUDIT.relative_to(ROOT)),
        "implementation_audit_sha256": sha256(R5_AUDIT),
        "source_artifact_sha256": dict(sorted(sources.items())),
        "scientific_invariants": {
            "grounding_backbone_b0_unchanged": True,
            "spatial_risk_v2_model_and_12_features_unchanged": True,
            "family_manifest_and_4x4x8_quota_unchanged": True,
            "unsafe_and_all_metric_definitions_unchanged": True,
            "positive_slope_affine_logit_and_single_fit_unchanged": True,
            "optimizer_initialization_bounds_and_options_unchanged": True,
            "threshold_grid_selection_and_success_gates_unchanged": True,
            "bootstrap_10000_seed_13092026_unchanged": True,
            "calibration_v2_closed_no_reuse": True,
            "test_iid_ood_sealed": True,
            "robot_closed": True,
        },
        "capture_authorized": False,
        "capture_performed": False,
        "dataset_materialized": False,
        "model_inference_performed": False,
        "calibrator_fit_performed": False,
        "next_authorized_action": "Start simulation-only, move UR3 to the locked camera pose, and run live preflight attempt 03. Capture remains forbidden until attempt 03 PASS and a separate capture lock exists.",
    }


def freeze() -> None:
    if R5_AUDIT.exists() or R5_LOCK.exists():
        raise FileExistsError("refusing to overwrite Calibration-v3 R5 audit or lock")
    audit = build_audit()
    write_json(R5_AUDIT, audit)
    if audit["status"] != "PASS":
        raise RuntimeError("Calibration-v3 R5 implementation audit failed; lock not created")
    lock = build_lock(audit)
    write_json(R5_LOCK, lock)
    print(json.dumps({
        "status": lock["status"],
        "audit_sha256": sha256(R5_AUDIT),
        "lock_sha256": sha256(R5_LOCK),
        "capture_authorized": False,
    }, indent=2, sort_keys=True))


def validate() -> None:
    audit = read_json(R5_AUDIT)
    lock = read_json(R5_LOCK)
    if audit.get("status") != "PASS":
        raise RuntimeError("Calibration-v3 R5 audit is not PASS")
    if lock.get("status") != "IMPLEMENTATION_R5_FROZEN_BEFORE_LIVE_PREFLIGHT_ATTEMPT_03":
        raise RuntimeError("Calibration-v3 R5 lock state is invalid")
    if lock.get("implementation_audit_sha256") != sha256(R5_AUDIT):
        raise RuntimeError("Calibration-v3 R5 audit hash linkage failed")
    for name, digest in lock["source_artifact_sha256"].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Calibration-v3 R5 source drift: {name}")
    print(json.dumps({
        "status": "PASS",
        "lock_sha256": sha256(R5_LOCK),
        "audit_sha256": sha256(R5_AUDIT),
        "capture_authorized_by_r5_lock": False,
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
