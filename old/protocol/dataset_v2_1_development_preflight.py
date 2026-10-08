#!/usr/bin/env python3
"""Static gate and execution lock for Dataset V2.1 development capture."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2_1_development_generator import (  # noqa: E402
    DENYLIST_JSON,
    FAMILY_COUNT,
    FAMILY_PREFIX,
    GENERATOR_VERSION,
    OUTPUT_ROOT,
    PROTOCOL_ID,
    build_artifacts,
    collect_seed_values,
    collect_values,
    evaluate_spec_geometry,
    geometry_constraints_pass,
    load_denylists,
)
from dataset_v2_pilot_generator import object_registry  # noqa: E402
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


EXPECTED_SPLIT = {"train": 320, "dev": 80}
EXPECTED_STATE = {
    "train": {"FOUND": 128, "INSUFFICIENT_EVIDENCE": 80, "AMBIGUOUS": 56, "ABSENT": 56},
    "dev": {"FOUND": 32, "INSUFFICIENT_EVIDENCE": 20, "AMBIGUOUS": 14, "ABSENT": 14},
}
EXPECTED_CATEGORY = {
    "train": {
        "direct_grounding": 48, "relation_2d": 64, "nearer_farther": 56,
        "front_behind_camera": 48, "multi_anchor_depth_order": 48,
        "occlusion_depth_evidence": 56,
    },
    "dev": {
        "direct_grounding": 12, "relation_2d": 16, "nearer_farther": 14,
        "front_behind_camera": 12, "multi_anchor_depth_order": 12,
        "occlusion_depth_evidence": 14,
    },
}
EXPECTED_BATCHES = [20, 95, 95, 95, 95]
RUNTIME_ARTIFACTS = (
    "protocol/DATASET_V2_1_DEVELOPMENT_CAPTURE_CONTRACT.md",
    "protocol/dataset_v2_1_development_generator.py",
    "protocol/dataset_v2_1_development_manifest.json",
    "protocol/dataset_v2_1_development_capture_plan.json",
    "protocol/dataset_v2_1_development_preflight.py",
    "protocol/dataset_v2_1_development_capture.py",
    "protocol/dataset_v2_1_development_capture.launch.py",
    "protocol/dataset_v2_1_development_batch_qc.py",
    "protocol/dataset_v2_1_development_materialize.py",
    "protocol/dataset_v2_1_development_full_qc.py",
    "protocol/DATASET_V2_1_1_VISIBILITY_AMENDMENT.md",
    "protocol/dataset_v2_relation_repair_shutdown.py",
    "protocol/dataset_v2_development_capture.py",
    "protocol/DATASET_V2_RELATION_GEOMETRY_AMENDMENT_01.md",
    "protocol/DATASET_V2_RELATION_GEOMETRY_AMENDMENT_02.md",
    "protocol/dataset_v2_relation_geometry_v2_1.py",
    "protocol/dataset_v2_relation_camera_lock_v2_1.json",
    "results/dataset_v2_relation_repair_20260821/repair_pilot_round2/RELATION_REPAIR_QC_REPORT.json",
    "protocol/dataset_v2_1_shutdown_gate_execution_lock.json",
    "results/dataset_v2_1_shutdown_gate_20260824/DATASET_V2_1_SHUTDOWN_GATE_REPORT.json",
    "protocol/dataset_expansion_v2_asset_partition.json",
    "protocol/dataset_v2_development_manifest.json",
    "protocol/dataset_v2_development_capture_plan.json",
)


def fail(checks: dict[str, Any], failures: list[str], name: str, passed: bool, **details: Any) -> None:
    checks[name] = {"passed": bool(passed), **details}
    if not passed:
        failures.append(name)


def active_layout(capture: dict[str, Any]) -> dict[str, list[float]]:
    return {name: pose for name, pose in capture["layout"].items() if float(pose[1]) <= 0.65}


def run_preflight(workspace: Path) -> dict[str, Any]:
    protocol = workspace / "protocol"
    manifest_path = protocol / "dataset_v2_1_development_manifest.json"
    plan_path = protocol / "dataset_v2_1_development_capture_plan.json"
    manifest = read_json(manifest_path)
    plan = read_json(plan_path)
    failures: list[str] = []
    checks: dict[str, Any] = {}

    replay_manifest, replay_plan = build_artifacts(workspace)
    replay_plan["manifest_sha256"] = sha256_file(manifest_path)
    replay_plan.pop("manifest_sha256_pending", None)
    replay_ok = replay_manifest == manifest and replay_plan == plan
    fail(checks, failures, "deterministic_static_replay", replay_ok)

    counts_ok = (
        manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("generator_version") == GENERATOR_VERSION
        and manifest.get("family_count") == FAMILY_COUNT
        and manifest.get("sample_count") == 2000
        and manifest.get("variant_count_per_family") == 5
        and plan.get("family_count") == 400
        and plan.get("capture_count") == 800
        and plan.get("captures_per_family") == 2
        and manifest["counts"]["split"] == EXPECTED_SPLIT
        and manifest["counts"]["primary_answerability_by_split"] == EXPECTED_STATE
        and manifest["counts"]["family_category_by_split"] == EXPECTED_CATEGORY
        and manifest["counts"]["ood_axis_by_split"] == {"train": {"none": 320}, "dev": {"none": 80}}
    )
    fail(
        checks, failures, "locked_population", counts_ok,
        family_count=manifest.get("family_count"), sample_count=manifest.get("sample_count"),
        capture_count=plan.get("capture_count"), counts=manifest.get("counts"),
    )

    family_ids = [item["family_id"] for item in manifest["families"]]
    capture_ids = [item["capture_id"] for item in plan["captures"]]
    sample_ids = [
        f"{family['family_id']}__{spec['variant']}"
        for family in manifest["families"] for spec in family["variant_specs"]
    ]
    denylists = load_denylists(workspace)
    id_ok = (
        len(family_ids) == len(set(family_ids)) == 400
        and len(capture_ids) == len(set(capture_ids)) == 800
        and len(sample_ids) == len(set(sample_ids)) == 2000
        and all(re.fullmatch(r"v211dev_family_[0-9]{6}", value) for value in family_ids)
        and all(value.startswith(FAMILY_PREFIX) for value in family_ids)
        and not (set(family_ids) & denylists["family_ids"])
        and not (set(capture_ids) & denylists["capture_ids"])
        and all([spec["variant"] for spec in family["variant_specs"]] == list(VARIANTS)
                for family in manifest["families"])
    )
    fail(
        checks, failures, "fresh_family_sample_capture_identity", id_ok,
        family_overlap_count=len(set(family_ids) & denylists["family_ids"]),
        capture_overlap_count=len(set(capture_ids) & denylists["capture_ids"]),
    )

    seed_bundle_values = [
        int(value) for family in manifest["families"] for value in family["seed_bundle"].values()
    ]
    selected_layout_seeds = [int(family["selected_layout_seed"]) for family in manifest["families"]]
    new_seeds = seed_bundle_values + selected_layout_seeds
    seed_ok = (
        len(new_seeds) == len(set(new_seeds)) == 2400
        and not (set(new_seeds) & denylists["seeds"])
    )
    fail(
        checks, failures, "fresh_seed_namespace", seed_ok,
        seed_count=len(new_seeds), unique_seed_count=len(set(new_seeds)),
        prior_overlap_count=len(set(new_seeds) & denylists["seeds"]),
    )

    instructions = [
        spec["instruction"] for family in manifest["families"] for spec in family["variant_specs"]
    ]
    instruction_ok = (
        len(instructions) == len(set(instructions)) == 2000
        and not (set(instructions) & denylists["instructions"])
    )
    fail(
        checks, failures, "fresh_instruction_namespace", instruction_ok,
        instruction_count=len(instructions), unique_count=len(set(instructions)),
        prior_overlap_count=len(set(instructions) & denylists["instructions"]),
    )

    fingerprints = [item["layout_instance_fingerprint_sha256"] for item in plan["captures"]]
    split_by_fingerprint: defaultdict[str, set[str]] = defaultdict(set)
    for capture in plan["captures"]:
        split_by_fingerprint[capture["layout_instance_fingerprint_sha256"]].add(capture["split"])
    layout_hash_ok = (
        len(fingerprints) == len(set(fingerprints)) == 800
        and not (set(fingerprints) & denylists["layout_fingerprints"])
        and not any(len(value) > 1 for value in split_by_fingerprint.values())
        and all(
            canonical_json_sha256({key: value for key, value in sorted(active_layout(capture).items())})
            == capture["layout_instance_fingerprint_sha256"]
            for capture in plan["captures"]
        )
    )
    fail(
        checks, failures, "fresh_layouts_and_split_isolation", layout_hash_ok,
        layout_count=len(fingerprints), unique_count=len(set(fingerprints)),
        prior_overlap_count=len(set(fingerprints) & denylists["layout_fingerprints"]),
    )

    batches = plan["batches"]
    batch_ids = [item["batch_id"] for item in batches]
    batched_families = [value for item in batches for value in item["family_ids"]]
    canary = batches[0]
    batch_ok = (
        batch_ids == ["canary_000", "batch_001", "batch_002", "batch_003", "batch_004"]
        and [item["family_count"] for item in batches] == EXPECTED_BATCHES
        and len(batched_families) == len(set(batched_families)) == 400
        and set(batched_families) == set(family_ids)
        and canary["split_counts"] == {"train": 16, "dev": 4}
        and set(canary["primary_answerability_counts"]) == set(EXPECTED_STATE["train"])
        and len(canary["family_category_counts"]) == 6
        and sum(item["planned_capture_count"] for item in batches) == 800
        and plan["resume_contract"]["identity_key"] == "capture_id"
        and plan["resume_contract"]["next_batch_requires_previous_raw_qc_pass"] is True
    )
    fail(
        checks, failures, "batch_and_resume_lock", batch_ok,
        batch_family_counts=[item["family_count"] for item in batches],
        canary_split_counts=canary["split_counts"],
        canary_state_counts=canary["primary_answerability_counts"],
    )

    partition = read_json(protocol / "dataset_expansion_v2_asset_partition.json")
    allowed_ids = set(partition["seen_pool"]["asset_ids"])
    for base, clones in partition["clone_policy"]["clone_groups"].items():
        if base in allowed_ids:
            allowed_ids.update(clones)
    heldout = set(partition["test_ood_asset_holdout"]["asset_ids"])
    excluded = set(partition["excluded_assets"]["asset_ids"])
    registry = plan["object_registry"]
    by_model = {name: row for name, row in registry.items()}
    active_ids = {
        by_model[name]["id"]
        for capture in plan["captures"] for name in active_layout(capture)
    }
    asset_ok = active_ids.issubset(allowed_ids) and not (active_ids & heldout) and not (active_ids & excluded)
    fail(
        checks, failures, "seen_assets_only_and_ood_holdout", asset_ok,
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
    capture_by_id = {item["capture_id"]: item for item in plan["captures"]}
    geometry_rows = []
    for family in manifest["families"]:
        capture = capture_by_id[f"{family['family_id']}__clean_capture"]
        visible_ids = set(capture["required_visible_object_ids"])
        poses = {
            object_id: capture["layout"][id_to_model[object_id]] for object_id in visible_ids
        }
        for spec in family["variant_specs"]:
            if spec["capture_id"] == capture["capture_id"]:
                geometry_rows.extend(evaluate_spec_geometry(
                    spec, poses, by_id, id_to_model, calibration,
                    repair_view["fk_transform_base_to_camera"], camera_lock["intrinsics"],
                ))
    geometry_ok = (
        geometry_constraints_pass(geometry_rows)
        and len(geometry_rows) > 0
        and plan["observable_relation_contract"]["geometry_version"] == GEOMETRY_VERSION
        and all(item["camera_bin_id"] == repair_view["camera_bin_id"] for item in plan["captures"])
    )
    failed_geometry = [
        item for item in geometry_rows
        if (item["expected"] == "SATISFIED") != bool(item["passed"])
    ]
    fail(
        checks, failures, "observable_relation_geometry_v2_1", geometry_ok,
        geometry_version=GEOMETRY_VERSION, evaluated_predicates=len(geometry_rows),
        failed_predicates=len(failed_geometry), predicate_failures=failed_geometry[:20],
    )

    ontology_errors = []
    for family in manifest["families"]:
        spec = family["variant_specs"][0]
        state = spec["answerability_state"]
        valid = spec["valid_target_ids"]
        if state == "FOUND" and (len(valid) != 1 or not spec["answerable"]):
            ontology_errors.append(f"{family['family_id']}:FOUND")
        elif state == "AMBIGUOUS" and (len(valid) < 2 or spec["answerable"]):
            ontology_errors.append(f"{family['family_id']}:AMBIGUOUS")
        elif state == "ABSENT" and (valid or spec["answerable"]):
            ontology_errors.append(f"{family['family_id']}:ABSENT")
        elif state == "INSUFFICIENT_EVIDENCE" and (len(valid) != 1 or spec["answerable"]):
            ontology_errors.append(f"{family['family_id']}:INSUFFICIENT")
    fail(
        checks, failures, "four_state_ontology", not ontology_errors,
        errors=ontology_errors[:50],
    )

    oracle_findings = find_forbidden_inference_keys(plan["inference_payload_template"])
    boundary_ok = (
        not oracle_findings
        and set(manifest["allowed_splits"]) == {"train", "dev"}
        and set(manifest["sealed_splits_not_created"]) == {"calibration", "test_iid", "test_ood"}
        and all(item["split"] in {"train", "dev"} for item in manifest["families"])
        and plan["training_authorized"] is False
        and plan["calibration_test_authorized"] is False
        and manifest["training_performed"] is False
    )
    fail(
        checks, failures, "oracle_and_sealed_split_boundary", boundary_ok,
        inference_payload_findings=oracle_findings,
        sealed_splits=manifest["sealed_splits_not_created"],
    )

    repair_report_path = workspace / "results/dataset_v2_relation_repair_20260821/repair_pilot_round2/RELATION_REPAIR_QC_REPORT.json"
    shutdown_report_path = workspace / "results/dataset_v2_1_shutdown_gate_20260824/DATASET_V2_1_SHUTDOWN_GATE_REPORT.json"
    shutdown_lock_path = protocol / "dataset_v2_1_shutdown_gate_execution_lock.json"
    repair_report = read_json(repair_report_path)
    shutdown_report = read_json(shutdown_report_path)
    shutdown_lock = read_json(shutdown_lock_path)
    prereq_ok = (
        repair_report.get("passed") is True
        and repair_report.get("decision") == "PASS_RELATION_REPAIR_PILOT"
        and repair_report.get("training_performed") is False
        and shutdown_report.get("passed") is True
        and shutdown_report.get("decision") == "PASS_V2_1_SHUTDOWN_GATE"
        and shutdown_report.get("baseline_modified") is False
        and shutdown_report.get("training_performed") is False
        and shutdown_lock.get("training_authorized") is False
    )
    fail(
        checks, failures, "relation_and_shutdown_prerequisites", prereq_ok,
        relation_decision=repair_report.get("decision"),
        shutdown_decision=shutdown_report.get("decision"),
        relation_report_sha256=sha256_file(repair_report_path),
        shutdown_report_sha256=sha256_file(shutdown_report_path),
    )

    baseline_mismatches = []
    for relative, expected in shutdown_lock["baseline_expected_sha256"].items():
        path = workspace / relative
        if not path.is_file() or sha256_file(path) != expected:
            baseline_mismatches.append(relative)
    fail(
        checks, failures, "baseline_and_v2_history_unchanged", not baseline_mismatches,
        checked_count=len(shutdown_lock["baseline_expected_sha256"]),
        mismatches=baseline_mismatches,
    )

    missing = [relative for relative in RUNTIME_ARTIFACTS if not (workspace / relative).is_file()]
    output_root = workspace / OUTPUT_ROOT
    output_clean = not output_root.exists() or not any(output_root.iterdir())
    shutdown_contract = plan["ordered_shutdown_contract"]
    runtime_ok = (
        not missing and output_clean
        and shutdown_contract["qualified_gate_decision"] == "PASS_V2_1_SHUTDOWN_GATE"
        and shutdown_contract["sequence"] == [
            "capture_node", "action_server_and_servo", "move_group", "gazebo"
        ]
        and shutdown_contract["baseline_launch_files_modified"] is False
    )
    fail(
        checks, failures, "runtime_output_and_ordered_shutdown", runtime_ok,
        missing_runtime_artifacts=missing, capture_output_root=OUTPUT_ROOT,
        capture_output_root_absent_or_empty=output_clean,
        shutdown_sequence=shutdown_contract["sequence"],
    )

    source_mismatches = []
    for relative, expected in manifest["source_artifact_sha256"].items():
        if not (workspace / relative).is_file() or sha256_file(workspace / relative) != expected:
            source_mismatches.append(relative)
    for relative, expected in manifest["denylist_artifact_sha256"].items():
        if not (workspace / relative).is_file() or sha256_file(workspace / relative) != expected:
            source_mismatches.append(relative)
    fail(
        checks, failures, "locked_source_and_denylist_hashes", not source_mismatches,
        source_count=len(manifest["source_artifact_sha256"]),
        denylist_count=len(manifest["denylist_artifact_sha256"]),
        mismatches=source_mismatches,
    )

    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "gate": "LOCK_DEVELOPMENT_V2_1_CAPTURE_400",
        "scientific_status": "STATIC_PREFLIGHT_PLANNED_NOT_CAPTURED",
        "passed": not failures,
        "decision": "GO_DEVELOPMENT_V2_1_CAPTURE_400" if not failures else "FIX_V2_1_DEVELOPMENT_LOCK_FIRST",
        "authorized_next_action": "CAPTURE_CANARY_000_20_FAMILIES" if not failures else "NONE",
        "failures": failures,
        "checks": checks,
        "capture_performed": False,
        "training_performed": False,
        "calibration_or_test_created": False,
        "tables_02_to_06": "NOT_RUN",
        "figures_created": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    protocol = workspace / "protocol"
    report = run_preflight(workspace)
    report_path = protocol / "dataset_v2_1_development_preflight_report.json"
    write_json(report_path, report)
    locked_hashes = {
        relative: sha256_file(workspace / relative)
        for relative in RUNTIME_ARTIFACTS if (workspace / relative).is_file()
    }
    manifest_path = protocol / "dataset_v2_1_development_manifest.json"
    plan_path = protocol / "dataset_v2_1_development_capture_plan.json"
    commitment = {
        "manifest_sha256": sha256_file(manifest_path),
        "capture_plan_sha256": sha256_file(plan_path),
        "preflight_report_sha256": sha256_file(report_path),
        "locked_artifact_sha256": locked_hashes,
        "authorized_next_batch": "canary_000" if report["passed"] else None,
    }
    lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "gate": "LOCK_DEVELOPMENT_V2_1_CAPTURE_400",
        "locked_at_utc": utc_now(),
        "decision": report["decision"],
        "capture_authorized": bool(report["passed"]),
        "authorized_next_batch": commitment["authorized_next_batch"],
        "capture_output_root": OUTPUT_ROOT,
        "required_family_count": 400,
        "required_raw_capture_count": 800,
        "required_materialized_sample_count": 2000,
        "batch_family_counts": EXPECTED_BATCHES,
        "next_batch_requires_previous_batch_qc": True,
        "all_prior_capture_reuse_forbidden": True,
        "ordered_shutdown_required": True,
        "training_authorized": False,
        "training_forbidden_until_full_v2_1_qc": True,
        "calibration_test_authorized": False,
        **commitment,
        "lock_commitment_sha256": canonical_json_sha256(commitment),
    }
    write_json(protocol / "dataset_v2_1_development_execution_lock.json", lock)
    print(
        "DATASET_V2_1_DEVELOPMENT_PREFLIGHT "
        f"passed={report['passed']} decision={report['decision']} "
        f"failures={len(report['failures'])} next={report['authorized_next_action']}"
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
