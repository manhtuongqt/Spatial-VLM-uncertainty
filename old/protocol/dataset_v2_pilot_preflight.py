#!/usr/bin/env python3
"""Static preflight and execution-lock writer for Dataset V2 pilot-only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2_pilot_generator import (  # noqa: E402
    EXPECTED_PILOT_COMMITMENT,
    PROTOCOL_ID,
    VARIANTS,
    build_artifacts,
)
from validate_dataset_expansion_v2 import build_development_assignments  # noqa: E402
from wp2_common import (  # noqa: E402
    canonical_json_sha256,
    find_forbidden_inference_keys,
    read_json,
    sha256_file,
    tree_digest,
    utc_now,
    write_json,
)


EXPECTED_COUNTS = {
    "primary_answerability": {"FOUND": 8, "INSUFFICIENT_EVIDENCE": 8, "AMBIGUOUS": 7, "ABSENT": 7},
    "family_category": {
        "direct_grounding": 5, "relation_2d": 5, "nearer_farther": 5,
        "front_behind_camera": 5, "multi_anchor_depth_order": 5,
        "occlusion_depth_evidence": 5,
    },
    "ood_axis": {"none": 5, "asset": 5, "layout": 5, "viewpoint": 5, "language": 5, "depth_noise": 5},
}
REQUIRED_SUBMODES = {
    "AMBIGUOUS": {"same_class_duplicate", "attribute_tie", "relation_tie", "multi_anchor_conflict"},
    "ABSENT": {"target_absent", "anchor_absent", "unsatisfied_relation"},
    "INSUFFICIENT_EVIDENCE": {"depth_invalid_or_corrupt", "occlusion", "too_small_or_out_of_view", "cross_modal_conflict"},
}
RUNTIME_ARTIFACTS = (
    "protocol/DATASET_V2_PILOT_CONTRACT.md",
    "protocol/dataset_v2_pilot_generator.py",
    "protocol/dataset_v2_pilot_manifest.json",
    "protocol/dataset_v2_pilot_capture_plan.json",
    "protocol/dataset_v2_pilot_object_registry.json",
    "protocol/dataset_v2_pilot_preflight.py",
    "protocol/dataset_v2_pilot_capture.py",
    "protocol/dataset_v2_pilot_capture.launch.py",
    "protocol/dataset_v2_pilot_record.schema.json",
    "protocol/dataset_v2_pilot_materialize.py",
    "protocol/dataset_v2_pilot_validate.py",
    "protocol/DATASET_EXPANSION_V2_CONTRACT.md",
    "protocol/dataset_expansion_v2_spec.json",
    "protocol/dataset_expansion_v2_asset_partition.json",
    "protocol/dataset_expansion_v2_seed_lock.json",
    "protocol/dataset_split_v2.schema.json",
    "protocol/validate_dataset_expansion_v2.py",
    "ur3/ur_simulation_gz/worlds/ur3_pick_place_inventory_v2.sdf",
    "ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro",
)


def geometry_satisfied(
    relation: str, target_pose: list[float], anchor_poses: list[list[float]], margin: float = .025
) -> bool:
    target_y = float(target_pose[1])
    anchor_y = [float(value[1]) for value in anchor_poses]
    if relation == "direct":
        return True
    if not anchor_y:
        return False
    if relation in {"right_of", "farther_than", "behind"}:
        return target_y > anchor_y[0] + margin
    if relation in {"left_of", "nearer_than", "front_of"}:
        return target_y + margin < anchor_y[0]
    if relation == "between_in_depth":
        return len(anchor_y) >= 2 and min(anchor_y) + margin < target_y < max(anchor_y) - margin
    if relation == "nearer_than_both":
        return len(anchor_y) >= 2 and all(target_y + margin < value for value in anchor_y)
    if relation == "farther_than_both":
        return len(anchor_y) >= 2 and all(target_y > value + margin for value in anchor_y)
    return False


def run_preflight(workspace: Path) -> dict[str, Any]:
    protocol = workspace / "protocol"
    manifest_path = protocol / "dataset_v2_pilot_manifest.json"
    plan_path = protocol / "dataset_v2_pilot_capture_plan.json"
    registry_path = protocol / "dataset_v2_pilot_object_registry.json"
    manifest = read_json(manifest_path)
    plan = read_json(plan_path)
    registry_artifact = read_json(registry_path)
    seed_lock = read_json(protocol / "dataset_expansion_v2_seed_lock.json")
    partition = read_json(protocol / "dataset_expansion_v2_asset_partition.json")
    failures: list[str] = []
    checks: dict[str, Any] = {}

    # Deterministic static replay, including the manifest hash embedded in plan.
    replay_manifest, replay_plan, replay_registry = build_artifacts(workspace)
    replay_plan["manifest_sha256"] = sha256_file(manifest_path)
    replay_plan.pop("manifest_sha256_pending", None)
    replay_ok = manifest == replay_manifest and plan == replay_plan and registry_artifact == replay_registry
    checks["static_replay"] = {"passed": replay_ok}
    if not replay_ok:
        failures.append("static generator replay differs from materialized artifacts")

    exact_counts = (
        manifest.get("family_count") == 30 and manifest.get("sample_count") == 150
        and plan.get("capture_count") == 60 and plan.get("captures_per_family") == 2
        and manifest.get("design_assignment_commitment_sha256") == EXPECTED_PILOT_COMMITMENT
        and all(manifest["counts"].get(key) == dict(sorted(value.items())) for key, value in EXPECTED_COUNTS.items())
    )
    checks["locked_counts"] = {
        "passed": exact_counts, "family_count": manifest.get("family_count"),
        "sample_count": manifest.get("sample_count"), "capture_count": plan.get("capture_count"),
        "counts": manifest.get("counts"),
    }
    if not exact_counts:
        failures.append("locked pilot composition/counts do not match preregistration")

    family_ids = [item["family_id"] for item in manifest["families"]]
    sample_ids = [f"{item['family_id']}__{spec['variant']}" for item in manifest["families"] for spec in item["variant_specs"]]
    capture_ids = [item["capture_id"] for item in plan["captures"]]
    identifiers_ok = len(set(family_ids)) == 30 and len(set(sample_ids)) == 150 and len(set(capture_ids)) == 60
    variants_ok = all([spec["variant"] for spec in item["variant_specs"]] == list(VARIANTS) for item in manifest["families"])
    checks["identity_and_variants"] = {"passed": identifiers_ok and variants_ok}
    if not identifiers_ok or not variants_ok:
        failures.append("family/sample/capture identity or five-variant contract failed")

    registry = plan["object_registry"]
    by_id = {value["id"]: {**value, "model_name": name} for name, value in registry.items()}
    registry_ok = (
        len(registry) == len(set(registry)) == len(set(value["id"] for value in registry.values()))
        and len(registry) == len(set(int(value["label"]) for value in registry.values()))
        and registry_artifact.get("objects") == registry
    )
    world_text = (workspace / plan["world_file"]).read_text(encoding="utf-8")
    world_findings = []
    for model_name, value in registry.items():
        marker = f'<model name="{model_name}">'
        start = world_text.find(marker)
        end = world_text.find("</model>", start)
        if start < 0 or end < 0 or f"<label>{int(value['label'])}</label>" not in world_text[start:end]:
            world_findings.append(f"{model_name}: model/label {value['label']} missing")
    checks["object_registry_and_world_labels"] = {
        "passed": registry_ok and not world_findings, "instance_count": len(registry),
        "findings": world_findings,
    }
    if not registry_ok or world_findings:
        failures.append("object registry uniqueness or Gazebo label mapping failed")

    layout_failures = []
    roi = plan["workspace_roi_base_link"]
    x_min, x_max = [float(value) for value in roi["x_m"]]
    y_min, y_max = [float(value) for value in roi["y_m"]]
    # Static target-bin footprint from ur3_pick_place_inventory_v2.sdf,
    # expanded below by each object's conservative footprint and 3 mm margin.
    bin_x_min, bin_x_max = -.10, .10
    bin_y_min, bin_y_max = .255, .505
    for capture in plan["captures"]:
        layout = capture["layout"]
        if set(layout) != set(registry):
            layout_failures.append(f"{capture['capture_id']}: layout registry coverage")
            continue
        active = [(name, pose) for name, pose in layout.items() if float(pose[1]) <= y_max]
        for name, pose in active:
            if not (x_min <= float(pose[0]) <= x_max and y_min <= float(pose[1]) <= y_max):
                layout_failures.append(f"{capture['capture_id']}:{name}: outside active ROI")
            nearest_x = min(max(float(pose[0]), bin_x_min), bin_x_max)
            nearest_y = min(max(float(pose[1]), bin_y_min), bin_y_max)
            bin_distance = math.hypot(float(pose[0]) - nearest_x, float(pose[1]) - nearest_y)
            required_bin_clearance = float(registry[name]["footprint_radius_m"]) + .003
            if bin_distance < required_bin_clearance:
                layout_failures.append(
                    f"{capture['capture_id']}:{name}: target-bin collision "
                    f"{bin_distance:.4f} < {required_bin_clearance:.4f}"
                )
        for index, (name, pose) in enumerate(active):
            for other, other_pose in active[index + 1:]:
                distance = math.hypot(float(pose[0]) - float(other_pose[0]), float(pose[1]) - float(other_pose[1]))
                required = float(registry[name]["footprint_radius_m"]) + float(registry[other]["footprint_radius_m"]) + .003
                if distance < required:
                    layout_failures.append(
                        f"{capture['capture_id']}:{name}/{other}: collision {distance:.4f} < {required:.4f}"
                    )
    checks["layout_bounds_and_collision"] = {
        "passed": not layout_failures, "layouts_checked": len(plan["captures"]),
        "collision_margin_m": .003, "failures": layout_failures,
    }
    if layout_failures:
        failures.append("active layout bounds/collision preflight failed")

    # Empirical task visibility/action envelope derived from the failed first
    # real capture.  This is intentionally applied only where the ontology
    # requires visible evidence; too-small/out-of-view negatives remain valid.
    visibility_failures = []
    visibility_envelope = {"x_m": [-.42, -.08], "y_m": [.12, .50]}
    capture_by_id = {item["capture_id"]: item for item in plan["captures"]}
    for family in manifest["families"]:
        for spec in family["variant_specs"]:
            state = spec["answerability_state"]
            if state not in {"FOUND", "AMBIGUOUS"}:
                continue
            layout = capture_by_id[spec["capture_id"]]["layout"]
            required_ids = list(spec["valid_target_ids"]) + list(spec["anchor_ids"])
            for object_id in required_ids:
                pose = layout[by_id[object_id]["model_name"]]
                if not (
                    visibility_envelope["x_m"][0] <= float(pose[0]) <= visibility_envelope["x_m"][1]
                    and visibility_envelope["y_m"][0] <= float(pose[1]) <= visibility_envelope["y_m"][1]
                ):
                    visibility_failures.append(
                        f"{family['family_id']}__{spec['variant']}:{object_id}: "
                        f"outside calibrated visibility envelope at {pose[:2]}"
                    )
                if state == "FOUND" and spec["expected_intervention"] == "EXECUTE" and float(pose[0]) > -.05:
                    visibility_failures.append(
                        f"{family['family_id']}__{spec['variant']}:{object_id}: outside action envelope"
                    )
    checks["required_evidence_visibility_envelope"] = {
        "passed": not visibility_failures,
        "envelope_base_link": visibility_envelope,
        "failures": visibility_failures,
    }
    if visibility_failures:
        failures.append("required target/anchor evidence lies outside calibrated visibility/action envelope")

    ontology_failures = []
    relation_checks = 0
    for family in manifest["families"]:
        clean = family["variant_specs"][0]
        state = clean["answerability_state"]
        valid = clean["valid_target_ids"]
        candidates = clean["candidate_target_ids"]
        anchors = clean["anchor_ids"]
        active = set(family["primary_scene"]["active_scene_ids"])
        if state == "FOUND" and (len(valid) != 1 or not clean["answerable"]):
            ontology_failures.append(f"{family['family_id']}: FOUND cardinality")
        elif state == "AMBIGUOUS" and (len(valid) < 2 or clean["answerable"]):
            ontology_failures.append(f"{family['family_id']}: AMBIGUOUS cardinality")
        elif state == "ABSENT" and (len(valid) != 0 or clean["answerable"]):
            ontology_failures.append(f"{family['family_id']}: ABSENT cardinality")
        elif state == "INSUFFICIENT_EVIDENCE" and (len(valid) != 1 or clean["answerable"]):
            ontology_failures.append(f"{family['family_id']}: INSUFFICIENT cardinality")
        submode = clean["state_submode"]
        if submode == "target_absent" and any(value in active for value in candidates):
            ontology_failures.append(f"{family['family_id']}: target_absent candidate active")
        if submode == "anchor_absent" and anchors and all(value in active for value in anchors):
            ontology_failures.append(f"{family['family_id']}: anchor_absent anchor active")
        if state in {"FOUND", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"} and not all(value in active for value in valid):
            ontology_failures.append(f"{family['family_id']}: valid referent not in active task scene")

        capture = capture_by_id[clean["capture_id"]]
        layout = capture["layout"]
        relation = clean["relations"][0]
        anchor_poses = [layout[by_id[value]["model_name"]] for value in anchors if value in active]
        if submode == "unsatisfied_relation":
            relation_checks += 1
            satisfied = any(
                geometry_satisfied(relation, layout[by_id[value]["model_name"]], anchor_poses)
                for value in candidates if value in active
            )
            if satisfied or not all(value in active for value in candidates + anchors):
                ontology_failures.append(f"{family['family_id']}: unsatisfied relation not realized")
        elif state != "ABSENT" and relation != "direct":
            for value in valid:
                relation_checks += 1
                if not geometry_satisfied(relation, layout[by_id[value]["model_name"]], anchor_poses):
                    ontology_failures.append(f"{family['family_id']}:{value}: locked relation geometry false")
    observed_submodes = {
        state: {family["state_submode"] for family in manifest["families"] if family["primary_answerability_stratum"] == state}
        for state in REQUIRED_SUBMODES
    }
    for state, required in REQUIRED_SUBMODES.items():
        missing = required - observed_submodes[state]
        if missing:
            ontology_failures.append(f"{state}: missing submodes {sorted(missing)}")
    checks["ontology_target_anchor_relation"] = {
        "passed": not ontology_failures, "relation_checks": relation_checks,
        "observed_submodes": {key: sorted(value) for key, value in observed_submodes.items()},
        "failures": ontology_failures,
    }
    if ontology_failures:
        failures.append("ontology, target/anchor uniqueness, or static relation geometry failed")

    heldout = set(partition["test_ood_asset_holdout"]["asset_ids"])
    excluded = set(partition["excluded_assets"]["asset_ids"])
    runtime_reserved = {"cube_red_01"}
    partition_failures = []
    axis_fields = ("asset_pool", "layout_generator_id", "camera_bin_id", "language_template_bank_id", "depth_noise_generator_id")
    for family in manifest["families"]:
        expected_axis = partition["ood_axes"][family["ood_axis"]]
        for field in axis_fields:
            if family[field] != expected_axis[field]:
                partition_failures.append(f"{family['family_id']}:{field}: OOD-axis leakage")
        queried = set(family["primary_scene"]["candidate_target_ids"] + family["primary_scene"]["anchor_ids"])
        active = set(family["primary_scene"]["active_scene_ids"])
        if family["ood_axis"] == "asset" and not ((queried | active) & heldout):
            partition_failures.append(f"{family['family_id']}: asset OOD has no held-out entity")
        if family["ood_axis"] != "asset" and (queried | active) & heldout:
            partition_failures.append(f"{family['family_id']}: held-out asset leaked into non-asset axis")
        if (queried | active) & excluded:
            partition_failures.append(f"{family['family_id']}: excluded asset used")
        if (queried | active) & runtime_reserved:
            partition_failures.append(
                f"{family['family_id']}: gripper-attached runtime fixture used as task evidence"
            )
    if set(value["id"] for value in registry.values()) & excluded:
        partition_failures.append("excluded asset is present in object registry/world plan")
    checks["asset_partition_and_single_ood_axis"] = {
        "passed": not partition_failures, "heldout_assets": sorted(heldout),
        "excluded_assets": sorted(excluded), "runtime_reserved_assets": sorted(runtime_reserved),
        "failures": partition_failures,
    }
    if partition_failures:
        failures.append("asset partition or single-axis OOD isolation failed")

    leakage_findings = []
    for family in manifest["families"]:
        for spec in family["variant_specs"]:
            mock = {
                "rgb_model_input": "TO_BE_CAPTURED", "depth_relative_model_input": "TO_BE_DERIVED",
                "enable_depth": True,
                "prompt": f"{spec['instruction']} {plan['coordinate_suffix']}",
                "coordinate_suffix": plan["coordinate_suffix"],
            }
            findings = find_forbidden_inference_keys(mock)
            if findings:
                leakage_findings.append({"sample": f"{family['family_id']}__{spec['variant']}", "findings": findings})
    checks["oracle_leakage"] = {"passed": not leakage_findings, "findings": leakage_findings}
    if leakage_findings:
        failures.append("oracle/evaluator key leaked into inference payload template")

    development = build_development_assignments(seed_lock, partition)
    pilot_seeds = {value for family in manifest["families"] for value in family["seed_bundle"].values()}
    development_seeds = {value for family in development for value in family["seed_bundle"].values()}
    development_family_ids = {family["family_id"] for family in development}
    reserved_capture_ids = {
        f"{family_id}__{condition}_capture"
        for family_id in development_family_ids for condition in ("clean", "occlusion")
    }
    disjoint_ok = (
        not (set(family_ids) & development_family_ids)
        and not (pilot_seeds & development_seeds)
        and not (set(capture_ids) & reserved_capture_ids)
        and manifest["reserved_development_commitments"]["assignment_commitment_sha256"] == seed_lock["development"]["assignment_commitment_sha256"]
    )
    checks["pilot_official_disjointness"] = {
        "passed": disjoint_ok, "pilot_seed_count": len(pilot_seeds),
        "development_seed_count": len(development_seeds),
    }
    if not disjoint_ok:
        failures.append("pilot IDs/seeds/captures overlap reserved development data")

    protected_failures = []
    for relative, expected in seed_lock["protected_files"].items():
        path = workspace / relative
        if not path.is_file() or sha256_file(path) != expected:
            protected_failures.append(relative)
    tree_values = {}
    for relative, expected in seed_lock["protected_tree_digests"].items():
        actual = tree_digest(workspace, relative)
        tree_values[relative] = actual
        if actual != expected:
            protected_failures.append(relative)
    checks["protected_wp0_wp3_wp2_hashes"] = {
        "passed": not protected_failures, "failures": protected_failures,
        "tree_digests": tree_values,
    }
    if protected_failures:
        failures.append("protected WP0-WP3 or WP2 artifact/tree changed")

    missing_runtime = [relative for relative in RUNTIME_ARTIFACTS if not (workspace / relative).is_file()]
    checks["runtime_pipeline_complete"] = {"passed": not missing_runtime, "missing": missing_runtime}
    if missing_runtime:
        failures.append("pilot runtime pipeline is incomplete")

    return {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(), "passed": not failures,
        "decision": "GO_PILOT_GAZEBO_CAPTURE_60" if not failures else "FIX_DATASET_V2_GENERATOR_OR_CAPTURE_PIPELINE_FIRST",
        "failures": failures, "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    protocol = workspace / "protocol"
    report = run_preflight(workspace)
    report_path = protocol / "dataset_v2_pilot_preflight_report.json"
    lock_path = protocol / "dataset_v2_pilot_execution_lock.json"
    write_json(report_path, report)
    locked_hashes = {
        relative: sha256_file(workspace / relative)
        for relative in RUNTIME_ARTIFACTS if (workspace / relative).is_file()
    }
    lock = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "locked_at_utc": utc_now(), "decision": report["decision"],
        "capture_authorized": bool(report["passed"]),
        "capture_output_root": "datasets/roborefer_dataset_v2_pilot_only_20260821",
        "required_raw_capture_count": 60, "required_materialized_sample_count": 150,
        "pilot_reuse_forbidden": True, "training_forbidden": True,
        "capture_plan_sha256": sha256_file(protocol / "dataset_v2_pilot_capture_plan.json"),
        "manifest_sha256": sha256_file(protocol / "dataset_v2_pilot_manifest.json"),
        "preflight_report_sha256": sha256_file(report_path),
        "locked_artifact_sha256": locked_hashes,
        "lock_commitment_sha256": canonical_json_sha256({
            "capture_plan_sha256": sha256_file(protocol / "dataset_v2_pilot_capture_plan.json"),
            "manifest_sha256": sha256_file(protocol / "dataset_v2_pilot_manifest.json"),
            "preflight_report_sha256": sha256_file(report_path),
            "locked_artifact_sha256": locked_hashes,
        }),
    }
    write_json(lock_path, lock)
    print(f"DATASET_V2_PILOT_PREFLIGHT passed={report['passed']} decision={report['decision']} failures={len(report['failures'])}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
