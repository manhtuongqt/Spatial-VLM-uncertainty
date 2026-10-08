#!/usr/bin/env python3
"""Static independence and safety gate for Dataset V2.1 Calibration."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2_1_calibration_generator import (  # noqa: E402
    ARCHITECTURE_FREEZE_LOCK,
    CAPTURE_COUNT,
    DENYLIST_JSON,
    EXPECTED_CATEGORY,
    EXPECTED_STATE,
    FAMILY_COUNT,
    FAMILY_PREFIX,
    GENERATOR_VERSION,
    MASTER_SEED,
    OUTPUT_ROOT,
    PROTOCOL_ID,
    SAMPLE_COUNT,
    build_artifacts,
    load_calibration_denylists,
)
from dataset_v2_1_development_generator import (  # noqa: E402
    evaluate_spec_geometry,
    geometry_constraints_pass,
)
from dataset_v2_relation_geometry_v2_1 import GEOMETRY_VERSION  # noqa: E402
from validate_dataset_expansion_v2 import VARIANTS  # noqa: E402
from wp2_common import (  # noqa: E402
    canonical_json_sha256,
    find_forbidden_inference_keys,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)


DECISION = "GO_CALIBRATION_CAPTURE_200"
EXPECTED_BATCHES = [20, 90, 90]
RUNTIME_ARTIFACTS = (
    "protocol/DATASET_V2_1_CALIBRATION_CAPTURE_CONTRACT.md",
    "protocol/PCRA_U_CALIBRATION_CONTRACT.md",
    "protocol/pcra_u_calibration_config.json",
    "protocol/pcra_u_calibration_seed_lock.json",
    "protocol/dataset_v2_1_calibration_generator.py",
    "protocol/dataset_v2_1_calibration_manifest.json",
    "protocol/dataset_v2_1_calibration_capture_plan.json",
    "protocol/dataset_v2_1_calibration_preflight.py",
    "protocol/dataset_v2_1_calibration_capture.py",
    "protocol/dataset_v2_1_calibration_capture.launch.py",
    "protocol/dataset_v2_1_calibration_batch_qc.py",
    "protocol/dataset_v2_1_calibration_materialize.py",
    "protocol/dataset_v2_1_calibration_full_qc.py",
    "protocol/dataset_v2_relation_repair_shutdown.py",
    "protocol/dataset_v2_relation_geometry_v2_1.py",
    "protocol/dataset_v2_relation_camera_lock_v2_1.json",
    "protocol/dataset_expansion_v2_asset_partition.json",
    "protocol/pcra_u_architecture_freeze_lock.json",
    "results/dataset_v2_relation_repair_20260821/repair_pilot_round2/RELATION_REPAIR_QC_REPORT.json",
    "results/dataset_v2_1_shutdown_gate_20260824/DATASET_V2_1_SHUTDOWN_GATE_REPORT.json",
)


def record(checks: dict[str, Any], failures: list[str], name: str, passed: bool, **details: Any) -> None:
    checks[name] = {"passed": bool(passed), **details}
    if not passed:
        failures.append(name)


def active_layout(capture: dict[str, Any]) -> dict[str, list[float]]:
    return {name: pose for name, pose in capture["layout"].items() if float(pose[1]) <= .65}


def run_preflight(workspace: Path) -> dict[str, Any]:
    protocol = workspace / "protocol"
    manifest_path = protocol / "dataset_v2_1_calibration_manifest.json"
    plan_path = protocol / "dataset_v2_1_calibration_capture_plan.json"
    manifest, plan = read_json(manifest_path), read_json(plan_path)
    checks: dict[str, Any] = {}
    failures: list[str] = []

    replay_manifest, replay_plan = build_artifacts(workspace)
    replay_plan["manifest_sha256"] = sha256_file(manifest_path)
    replay_plan.pop("manifest_sha256_pending", None)
    record(
        checks, failures, "deterministic_static_replay",
        replay_manifest == manifest and replay_plan == plan,
    )

    development = read_json(protocol / "dataset_v2_1_development_manifest.json")
    development_states = Counter(
        row["primary_answerability_stratum"] for row in development["families"]
    )
    development_categories = Counter(row["family_category"] for row in development["families"])
    half_ok = (
        EXPECTED_STATE == {key: value // 2 for key, value in development_states.items()}
        and EXPECTED_CATEGORY == {key: value // 2 for key, value in development_categories.items()}
    )
    population_ok = (
        manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("generator_version") == GENERATOR_VERSION
        and manifest.get("family_count") == FAMILY_COUNT
        and manifest.get("sample_count") == SAMPLE_COUNT
        and manifest.get("variant_count_per_family") == 5
        and manifest["counts"]["split"] == {"calibration": FAMILY_COUNT}
        and manifest["counts"]["primary_answerability"] == EXPECTED_STATE
        and manifest["counts"]["family_category"] == EXPECTED_CATEGORY
        and plan.get("family_count") == FAMILY_COUNT
        and plan.get("capture_count") == CAPTURE_COUNT
        and plan.get("captures_per_family") == 2
        and half_ok
    )
    record(
        checks, failures, "locked_calibration_population",
        population_ok, counts=manifest.get("counts"), half_of_development_marginals=half_ok,
    )

    denylists = load_calibration_denylists(workspace)
    families = manifest["families"]
    family_ids = [row["family_id"] for row in families]
    capture_ids = [row["capture_id"] for row in plan["captures"]]
    sample_ids = [
        f"{family['family_id']}__{spec['variant']}"
        for family in families for spec in family["variant_specs"]
    ]
    identity_ok = (
        len(family_ids) == len(set(family_ids)) == FAMILY_COUNT
        and len(capture_ids) == len(set(capture_ids)) == CAPTURE_COUNT
        and len(sample_ids) == len(set(sample_ids)) == SAMPLE_COUNT
        and all(re.fullmatch(r"v211cal_family_[0-9]{6}", value) for value in family_ids)
        and all(value.startswith(FAMILY_PREFIX) for value in family_ids)
        and not (set(family_ids) & denylists["family_ids"])
        and not (set(capture_ids) & denylists["capture_ids"])
        and all(
            [spec["variant"] for spec in family["variant_specs"]] == list(VARIANTS)
            for family in families
        )
    )
    record(
        checks, failures, "fresh_family_sample_capture_identity", identity_ok,
        family_prior_overlap=len(set(family_ids) & denylists["family_ids"]),
        capture_prior_overlap=len(set(capture_ids) & denylists["capture_ids"]),
    )

    seeds = [
        int(value) for family in families
        for value in (*family["seed_bundle"].values(), family["selected_layout_seed"])
    ]
    seed_ok = len(seeds) == len(set(seeds)) == 6 * FAMILY_COUNT and not (
        set(seeds) & denylists["seeds"]
    )
    record(
        checks, failures, "fresh_seed_namespace", seed_ok,
        count=len(seeds), unique=len(set(seeds)), prior_overlap=len(set(seeds) & denylists["seeds"]),
    )

    instructions = [spec["instruction"] for family in families for spec in family["variant_specs"]]
    instruction_ok = (
        len(instructions) == len(set(instructions)) == SAMPLE_COUNT
        and not (set(instructions) & denylists["instructions"])
    )
    record(
        checks, failures, "fresh_instruction_namespace", instruction_ok,
        count=len(instructions), unique=len(set(instructions)),
        prior_overlap=len(set(instructions) & denylists["instructions"]),
    )

    template_ids = [
        spec["language_template_family_id"]
        for family in families for spec in family["variant_specs"]
    ]
    template_ok = (
        len(template_ids) == len(set(template_ids)) == SAMPLE_COUNT
        and not (set(template_ids) & denylists["template_ids"])
    )
    record(
        checks, failures, "fresh_language_template_family_namespace", template_ok,
        count=len(template_ids), unique=len(set(template_ids)),
        prior_overlap=len(set(template_ids) & denylists["template_ids"]),
    )

    fingerprints = [row["layout_instance_fingerprint_sha256"] for row in plan["captures"]]
    layouts_ok = (
        len(fingerprints) == len(set(fingerprints)) == CAPTURE_COUNT
        and not (set(fingerprints) & denylists["layout_fingerprints"])
        and all(
            canonical_json_sha256({key: value for key, value in sorted(active_layout(row).items())})
            == row["layout_instance_fingerprint_sha256"]
            for row in plan["captures"]
        )
    )
    record(
        checks, failures, "fresh_layout_namespace", layouts_ok,
        count=len(fingerprints), unique=len(set(fingerprints)),
        prior_overlap=len(set(fingerprints) & denylists["layout_fingerprints"]),
    )

    batches = plan["batches"]
    canary = batches[0]
    batched = [value for batch in batches for value in batch["family_ids"]]
    batch_ok = (
        [row["batch_id"] for row in batches] == ["canary_000", "batch_001", "batch_002"]
        and [row["family_count"] for row in batches] == EXPECTED_BATCHES
        and len(batched) == len(set(batched)) == FAMILY_COUNT
        and set(batched) == set(family_ids)
        and canary["split_counts"] == {"calibration": 20}
        and canary["primary_answerability_counts"] == {key: 5 for key in EXPECTED_STATE}
        and len(canary["family_category_counts"]) == 6
        and sum(row["planned_capture_count"] for row in batches) == CAPTURE_COUNT
        and plan["resume_contract"]["identity_key"] == "capture_id"
        and plan["resume_contract"]["next_batch_requires_previous_raw_qc_pass"] is True
    )
    record(
        checks, failures, "batch_resume_and_canary_lock", batch_ok,
        batch_family_counts=[row["family_count"] for row in batches],
        canary_states=canary["primary_answerability_counts"],
        canary_categories=canary["family_category_counts"],
    )

    partition = read_json(protocol / "dataset_expansion_v2_asset_partition.json")
    allowed = set(partition["seen_pool"]["asset_ids"])
    for base, clones in partition["clone_policy"]["clone_groups"].items():
        if base in allowed:
            allowed.update(clones)
    heldout = set(partition["test_ood_asset_holdout"]["asset_ids"])
    excluded = set(partition["excluded_assets"]["asset_ids"])
    registry = plan["object_registry"]
    active_ids = {
        registry[name]["id"] for capture in plan["captures"] for name in active_layout(capture)
    }
    asset_ok = active_ids.issubset(allowed) and not (active_ids & (heldout | excluded))
    record(
        checks, failures, "seen_assets_only_ood_held_out", asset_ok,
        active_asset_ids=sorted(active_ids), heldout_active=sorted(active_ids & heldout),
        excluded_active=sorted(active_ids & excluded),
    )

    camera_lock = read_json(protocol / "dataset_v2_relation_camera_lock_v2_1.json")
    repair_view = camera_lock["repair_view"]
    calibration = read_json(
        workspace / "results/dataset_v2_relation_repair_20260821/audit/asset_depth_calibration.json"
    )["asset_calibration"]
    by_id = {row["id"]: row for row in registry.values()}
    id_to_model = {row["id"]: name for name, row in registry.items()}
    capture_by_id = {row["capture_id"]: row for row in plan["captures"]}
    geometry_rows = []
    for family in families:
        capture = capture_by_id[f"{family['family_id']}__clean_capture"]
        visible = set(capture["required_visible_object_ids"])
        poses = {object_id: capture["layout"][id_to_model[object_id]] for object_id in visible}
        for spec in family["variant_specs"]:
            if spec["capture_id"] == capture["capture_id"]:
                geometry_rows.extend(evaluate_spec_geometry(
                    spec, poses, by_id, id_to_model, calibration,
                    repair_view["fk_transform_base_to_camera"], camera_lock["intrinsics"],
                ))
    geometry_ok = (
        len(geometry_rows) > 0
        and geometry_constraints_pass(geometry_rows)
        and plan["observable_relation_contract"]["geometry_version"] == GEOMETRY_VERSION
    )
    record(
        checks, failures, "observable_relation_geometry_v2_1", geometry_ok,
        evaluated_predicates=len(geometry_rows), geometry_version=GEOMETRY_VERSION,
    )

    ontology_errors = []
    for family in families:
        spec = family["variant_specs"][0]
        state, valid = spec["answerability_state"], spec["valid_target_ids"]
        if state == "FOUND" and (len(valid) != 1 or not spec["answerable"]):
            ontology_errors.append(f"{family['family_id']}:FOUND")
        elif state == "AMBIGUOUS" and (len(valid) < 2 or spec["answerable"]):
            ontology_errors.append(f"{family['family_id']}:AMBIGUOUS")
        elif state == "ABSENT" and (valid or spec["answerable"]):
            ontology_errors.append(f"{family['family_id']}:ABSENT")
        elif state == "INSUFFICIENT_EVIDENCE" and (len(valid) != 1 or spec["answerable"]):
            ontology_errors.append(f"{family['family_id']}:INSUFFICIENT")
    record(checks, failures, "four_state_ontology", not ontology_errors, errors=ontology_errors[:20])

    freeze = read_json(workspace / ARCHITECTURE_FREEZE_LOCK)
    calibration_config = read_json(protocol / "pcra_u_calibration_config.json")
    seed_lock = read_json(protocol / "pcra_u_calibration_seed_lock.json")
    protocol_lock_ok = (
        calibration_config.get("protocol_id") == "pcra_u_calibration_v1"
        and int(calibration_config["seeds"]["master"]) == MASTER_SEED
        and int(seed_lock.get("dataset_master_seed", -1)) == MASTER_SEED
        and calibration_config["dataset"]["primary_family_strata"] == EXPECTED_STATE
        and int(calibration_config["dataset"]["family_count"]) == FAMILY_COUNT
        and int(calibration_config["dataset"]["sample_count"]) == SAMPLE_COUNT
        and int(calibration_config["dataset"]["capture_count"]) == CAPTURE_COUNT
        and seed_lock.get("test_iid_opened") is False
        and seed_lock.get("test_ood_opened") is False
    )
    record(
        checks, failures, "calibration_contract_config_seed_consistency", protocol_lock_ok,
        master_seed=MASTER_SEED,
    )
    checkpoint_root = workspace / freeze["selected_checkpoint"]["path"]
    checkpoint_ok = (
        freeze.get("architecture_frozen") is True
        and freeze.get("checkpoint_frozen") is True
        and freeze.get("allowed_next_action") == "CAPTURE_CALIBRATION"
        and freeze.get("calibration_fit") is False
        and freeze.get("test_opened") is False
        and sha256_file(checkpoint_root / "model.safetensors")
            == freeze["selected_checkpoint"]["model_sha256"]
        and sha256_file(checkpoint_root / "manifest.json") == freeze["checkpoint_manifest_sha256"]
        and sha256_file(checkpoint_root / "metadata.json") == freeze["checkpoint_metadata_sha256"]
    )
    record(checks, failures, "frozen_architecture_checkpoint_prerequisite", checkpoint_ok)

    protected_mismatches = []
    for relative, expected in freeze["protected_inputs"]["wp3_locked_baseline_artifacts"].items():
        path = workspace / relative
        if not path.is_file() or sha256_file(path) != expected:
            protected_mismatches.append(relative)
    record(
        checks, failures, "baseline_unchanged", not protected_mismatches,
        checked=len(freeze["protected_inputs"]["wp3_locked_baseline_artifacts"]),
        mismatches=protected_mismatches,
    )

    oracle = find_forbidden_inference_keys(plan["inference_payload_template"])
    boundary_ok = (
        not oracle
        and set(manifest["allowed_splits"]) == {"calibration"}
        and set(manifest["sealed_splits_not_created"]) == {"test_iid", "test_ood"}
        and all(row["split"] == "calibration" for row in families)
        and plan["training_authorized"] is False
        and plan["model_inference_authorized"] is False
        and plan["calibrator_fit_authorized"] is False
        and plan["test_authorized"] is False
        and manifest["test_opened"] is False
    )
    record(
        checks, failures, "oracle_free_calibration_only_test_sealed", boundary_ok,
        inference_payload_findings=oracle,
    )

    missing = [relative for relative in RUNTIME_ARTIFACTS if not (workspace / relative).is_file()]
    output_root = workspace / OUTPUT_ROOT
    output_clean = not output_root.exists() or not any(output_root.iterdir())
    shutdown = plan["ordered_shutdown_contract"]
    runtime_ok = (
        not missing and output_clean
        and shutdown["qualified_gate_decision"] == "PASS_V2_1_SHUTDOWN_GATE"
        and shutdown["sequence"] == ["capture_node", "action_server_and_servo", "move_group", "gazebo"]
        and shutdown["baseline_launch_files_modified"] is False
    )
    record(
        checks, failures, "runtime_output_and_ordered_shutdown", runtime_ok,
        missing=missing, output_absent_or_empty=output_clean,
    )

    source_mismatches = []
    for mapping in (manifest["source_artifact_sha256"], manifest["denylist_artifact_sha256"]):
        for relative, expected in mapping.items():
            path = workspace / relative
            if not path.is_file() or sha256_file(path) != expected:
                source_mismatches.append(relative)
    record(
        checks, failures, "locked_source_and_denylist_hashes", not source_mismatches,
        mismatches=source_mismatches,
    )

    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "gate": "LOCK_INDEPENDENT_CALIBRATION_CAPTURE_200",
        "scientific_status": "STATIC_PREFLIGHT_CALIBRATION_NOT_CAPTURED_TEST_SEALED",
        "passed": not failures,
        "decision": DECISION if not failures else "FIX_CALIBRATION_LOCK_FIRST",
        "authorized_next_action": "CAPTURE_CALIBRATION_CANARY_20" if not failures else "NONE",
        "failures": failures,
        "checks": checks,
        "capture_performed": False,
        "model_inference_performed": False,
        "calibrator_fit_performed": False,
        "training_performed": False,
        "test_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    protocol = workspace / "protocol"
    report = run_preflight(workspace)
    report_path = protocol / "dataset_v2_1_calibration_preflight_report.json"
    write_json(report_path, report)
    locked_hashes = {
        relative: sha256_file(workspace / relative)
        for relative in RUNTIME_ARTIFACTS if (workspace / relative).is_file()
    }
    manifest_path = protocol / "dataset_v2_1_calibration_manifest.json"
    plan_path = protocol / "dataset_v2_1_calibration_capture_plan.json"
    commitment = {
        "manifest_sha256": sha256_file(manifest_path),
        "capture_plan_sha256": sha256_file(plan_path),
        "preflight_report_sha256": sha256_file(report_path),
        "locked_artifact_sha256": locked_hashes,
        "architecture_freeze_lock_sha256": sha256_file(workspace / ARCHITECTURE_FREEZE_LOCK),
        "authorized_next_batch": "canary_000" if report["passed"] else None,
    }
    lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "gate": "LOCK_INDEPENDENT_CALIBRATION_CAPTURE_200",
        "locked_at_utc": utc_now(),
        "decision": report["decision"],
        "capture_authorized": bool(report["passed"]),
        "calibration_capture_authorized": bool(report["passed"]),
        "authorized_next_batch": commitment["authorized_next_batch"],
        "capture_output_root": OUTPUT_ROOT,
        "required_family_count": FAMILY_COUNT,
        "required_raw_capture_count": CAPTURE_COUNT,
        "required_materialized_sample_count": SAMPLE_COUNT,
        "batch_family_counts": EXPECTED_BATCHES,
        "next_batch_requires_previous_batch_qc": True,
        "all_prior_capture_reuse_forbidden": True,
        "ordered_shutdown_required": True,
        "training_authorized": False,
        "model_inference_authorized_before_full_qc": False,
        "calibrator_fit_authorized_before_full_qc": False,
        "test_authorized": False,
        "test_opened": False,
        **commitment,
        "lock_commitment_sha256": canonical_json_sha256(commitment),
    }
    write_json(protocol / "dataset_v2_1_calibration_execution_lock.json", lock)
    print(
        "DATASET_V2_1_CALIBRATION_PREFLIGHT "
        f"passed={report['passed']} decision={report['decision']} failures={len(report['failures'])}"
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
