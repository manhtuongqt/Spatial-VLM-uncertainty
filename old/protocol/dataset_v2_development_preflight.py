#!/usr/bin/env python3
"""Static scientific preflight and execution lock for 400-family development capture."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2_development_generator import (  # noqa: E402
    EXPECTED_DEVELOPMENT_COMMITMENT,
    PILOT_ROOT,
    PROTOCOL_ID,
    SUBMODE_COUNTS_BY_SPLIT,
    build_artifacts,
)
from validate_dataset_expansion_v2 import VARIANTS  # noqa: E402
from wp2_common import (  # noqa: E402
    canonical_json_sha256,
    find_forbidden_inference_keys,
    read_json,
    sha256_file,
    tree_digest,
    utc_now,
    write_json,
)


EXPECTED_SPLIT_COUNTS = {"train": 320, "dev": 80}
EXPECTED_STATE_COUNTS = {
    "train": {"FOUND": 128, "INSUFFICIENT_EVIDENCE": 80, "AMBIGUOUS": 56, "ABSENT": 56},
    "dev": {"FOUND": 32, "INSUFFICIENT_EVIDENCE": 20, "AMBIGUOUS": 14, "ABSENT": 14},
}
EXPECTED_CATEGORY_COUNTS = {
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
EXPECTED_GLOBAL_SUBMODES = {
    "same_class_duplicate": 28, "attribute_tie": 18, "relation_tie": 14,
    "multi_anchor_conflict": 10, "target_absent": 24, "anchor_absent": 24,
    "unsatisfied_relation": 22, "depth_invalid_or_corrupt": 30,
    "occlusion": 30, "too_small_or_out_of_view": 20,
    "cross_modal_conflict": 20, "unique_visible_referent": 160,
}
RUNTIME_ARTIFACTS = (
    "protocol/DATASET_V2_DEVELOPMENT_CAPTURE_CONTRACT.md",
    "protocol/dataset_v2_development_generator.py",
    "protocol/dataset_v2_development_manifest.json",
    "protocol/dataset_v2_development_capture_plan.json",
    "protocol/dataset_v2_development_preflight.py",
    "protocol/dataset_v2_development_capture.py",
    "protocol/dataset_v2_development_capture.launch.py",
    "protocol/dataset_v2_development_batch_qc.py",
    "protocol/DATASET_EXPANSION_V2_CONTRACT.md",
    "protocol/dataset_expansion_v2_spec.json",
    "protocol/dataset_expansion_v2_asset_partition.json",
    "protocol/dataset_expansion_v2_seed_lock.json",
    "protocol/dataset_split_v2.schema.json",
    "protocol/validate_dataset_expansion_v2.py",
    "protocol/DATASET_V2_PILOT_CONTRACT.md",
    "protocol/dataset_v2_pilot_generator.py",
    "protocol/dataset_v2_pilot_manifest.json",
    "protocol/dataset_v2_pilot_capture_plan.json",
    "protocol/dataset_v2_pilot_execution_lock.json",
    "protocol/dataset_v2_pilot_capture.py",
    "protocol/dataset_v2_pilot_validate.py",
    "protocol/dataset_v2_pilot_validator_amendment_01.json",
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


def active_layout(capture: dict[str, Any]) -> dict[str, list[float]]:
    return {name: pose for name, pose in capture["layout"].items() if float(pose[1]) <= .65}


def counts_by_split(families: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    return {
        split: dict(sorted(Counter(str(item[field]) for item in families if item["split"] == split).items()))
        for split in ("train", "dev")
    }


def run_preflight(workspace: Path) -> dict[str, Any]:
    protocol = workspace / "protocol"
    manifest_path = protocol / "dataset_v2_development_manifest.json"
    plan_path = protocol / "dataset_v2_development_capture_plan.json"
    manifest = read_json(manifest_path)
    plan = read_json(plan_path)
    seed_lock = read_json(protocol / "dataset_expansion_v2_seed_lock.json")
    partition = read_json(protocol / "dataset_expansion_v2_asset_partition.json")
    pilot_manifest = read_json(protocol / "dataset_v2_pilot_manifest.json")
    pilot_plan = read_json(protocol / "dataset_v2_pilot_capture_plan.json")
    pilot_qc_path = workspace / PILOT_ROOT / "PILOT_QC_REPORT.json"
    pilot_raw_path = workspace / PILOT_ROOT / "raw/raw_capture_manifest.json"
    pilot_qc = read_json(pilot_qc_path)
    pilot_raw = read_json(pilot_raw_path)
    failures: list[str] = []
    checks: dict[str, Any] = {}

    replay_manifest, replay_plan = build_artifacts(workspace)
    replay_plan["manifest_sha256"] = sha256_file(manifest_path)
    replay_plan.pop("manifest_sha256_pending", None)
    replay_ok = replay_manifest == manifest and replay_plan == plan
    checks["deterministic_static_replay"] = {"passed": replay_ok}
    if not replay_ok:
        failures.append("static generator replay differs from materialized manifest or plan")

    exact_counts = (
        manifest.get("family_count") == 400
        and manifest.get("sample_count") == 2000
        and manifest.get("variant_count_per_family") == 5
        and plan.get("family_count") == 400
        and plan.get("capture_count") == 800
        and plan.get("captures_per_family") == 2
        and manifest.get("design_assignment_commitment_sha256") == EXPECTED_DEVELOPMENT_COMMITMENT
        and manifest.get("design_assignment_commitment_sha256") == seed_lock["development"]["assignment_commitment_sha256"]
        and manifest["counts"]["split"] == EXPECTED_SPLIT_COUNTS
        and manifest["counts"]["primary_answerability_by_split"] == EXPECTED_STATE_COUNTS
        and manifest["counts"]["family_category_by_split"] == EXPECTED_CATEGORY_COUNTS
        and manifest["counts"]["ood_axis_by_split"] == {"train": {"none": 320}, "dev": {"none": 80}}
    )
    checks["locked_counts_and_commitment"] = {
        "passed": exact_counts,
        "family_count": manifest.get("family_count"),
        "sample_count": manifest.get("sample_count"),
        "capture_count": plan.get("capture_count"),
        "counts": manifest.get("counts"),
        "commitment": manifest.get("design_assignment_commitment_sha256"),
    }
    if not exact_counts:
        failures.append("development counts or assignment commitment differ from design lock")

    global_submodes = dict(sorted(Counter(item["state_submode"] for item in manifest["families"]).items()))
    declared_submode_split = {
        split: dict(sorted({**SUBMODE_COUNTS_BY_SPLIT[split]["AMBIGUOUS"],
                            **SUBMODE_COUNTS_BY_SPLIT[split]["ABSENT"],
                            **SUBMODE_COUNTS_BY_SPLIT[split]["INSUFFICIENT_EVIDENCE"],
                            "unique_visible_referent": EXPECTED_STATE_COUNTS[split]["FOUND"]}.items()))
        for split in ("train", "dev")
    }
    submode_ok = (
        global_submodes == dict(sorted(EXPECTED_GLOBAL_SUBMODES.items()))
        and manifest["counts"]["state_submode_by_split"] == declared_submode_split
    )
    checks["submode_counts"] = {
        "passed": submode_ok,
        "global": global_submodes,
        "by_split": manifest["counts"]["state_submode_by_split"],
    }
    if not submode_ok:
        failures.append("primary state-submode allocation differs from locked design")

    family_ids = [item["family_id"] for item in manifest["families"]]
    sample_ids = [
        f"{family['family_id']}__{spec['variant']}"
        for family in manifest["families"] for spec in family["variant_specs"]
    ]
    capture_ids = [item["capture_id"] for item in plan["captures"]]
    identity_ok = (
        len(set(family_ids)) == len(family_ids) == 400
        and len(set(sample_ids)) == len(sample_ids) == 2000
        and len(set(capture_ids)) == len(capture_ids) == 800
        and all([spec["variant"] for spec in family["variant_specs"]] == list(VARIANTS)
                for family in manifest["families"])
        and all(len({spec["capture_id"].split("__")[0] for spec in family["variant_specs"]}) == 1
                for family in manifest["families"])
    )
    checks["family_sample_capture_identity"] = {"passed": identity_ok}
    if not identity_ok:
        failures.append("family/sample/capture identity or five-variant cohesion failed")

    batches = plan["batches"]
    batch_sizes = [int(item["family_count"]) for item in batches]
    batched_ids = [family_id for batch in batches for family_id in batch["family_ids"]]
    canary = batches[0]
    batch_ok = (
        plan["batch_execution_order"] == ["canary_000", "batch_001", "batch_002", "batch_003", "batch_004"]
        and batch_sizes == [20, 95, 95, 95, 95]
        and len(set(batched_ids)) == len(batched_ids) == 400
        and set(batched_ids) == set(family_ids)
        and sum(int(item["planned_capture_count"]) for item in batches) == 800
        and set(canary["split_counts"]) == {"train", "dev"}
        and set(canary["primary_answerability_counts"]) == set(EXPECTED_STATE_COUNTS["train"])
        and set(canary["family_category_counts"]) == set(EXPECTED_CATEGORY_COUNTS["train"])
        and all(item["batch_id"] == next(
            batch["batch_id"] for batch in batches if item["family_id"] in batch["family_ids"]
        ) for item in plan["captures"])
        and plan["resume_contract"] == {
            "identity_key": "capture_id",
            "write_mode": "ATOMIC_CREATE_ONLY",
            "existing_complete_capture": "VERIFY_ALL_DECLARED_HASHES_THEN_SKIP",
            "existing_partial_or_corrupt_capture": "FAIL_NEVER_OVERWRITE",
            "checkpoint_after_each_family": True,
            "checkpoint_at_batch_boundary": True,
            "next_batch_requires_previous_raw_qc_pass": True,
        }
    )
    checks["batch_and_resume_contract"] = {
        "passed": batch_ok,
        "batch_sizes": batch_sizes,
        "canary": {
            "split_counts": canary["split_counts"],
            "state_counts": canary["primary_answerability_counts"],
            "category_counts": canary["family_category_counts"],
        },
        "resume_contract": plan["resume_contract"],
    }
    if not batch_ok:
        failures.append("batch composition, canary coverage or resume policy failed")

    registry = plan["object_registry"]
    by_id = {value["id"]: {**value, "model_name": name} for name, value in registry.items()}
    registry_ok = (
        len(registry) == len(set(registry))
        and len(registry) == len(set(by_id))
        and len(registry) == len(set(int(value["label"]) for value in registry.values()))
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
        "passed": registry_ok and not world_findings,
        "instance_count": len(registry),
        "findings": world_findings,
    }
    if not registry_ok or world_findings:
        failures.append("object registry uniqueness or world semantic-label mapping failed")

    layout_failures = []
    x_min, x_max = [float(value) for value in plan["workspace_roi_base_link"]["x_m"]]
    y_min, y_max = [float(value) for value in plan["workspace_roi_base_link"]["y_m"]]
    bin_x_min, bin_x_max, bin_y_min, bin_y_max = -.10, .10, .255, .505
    fingerprint_families: defaultdict[str, set[str]] = defaultdict(set)
    fingerprint_splits: defaultdict[str, set[str]] = defaultdict(set)
    for capture in plan["captures"]:
        layout = capture["layout"]
        if set(layout) != set(registry):
            layout_failures.append(f"{capture['capture_id']}: registry coverage")
            continue
        active = list(active_layout(capture).items())
        actual_fingerprint = canonical_json_sha256(dict(active))
        if actual_fingerprint != capture["layout_instance_fingerprint_sha256"]:
            layout_failures.append(f"{capture['capture_id']}: layout fingerprint differs")
        fingerprint_families[actual_fingerprint].add(capture["family_id"])
        fingerprint_splits[actual_fingerprint].add(capture["split"])
        for name, pose in active:
            if not (x_min <= float(pose[0]) <= x_max and y_min <= float(pose[1]) <= y_max):
                layout_failures.append(f"{capture['capture_id']}:{name}: outside active ROI")
            nearest_x = min(max(float(pose[0]), bin_x_min), bin_x_max)
            nearest_y = min(max(float(pose[1]), bin_y_min), bin_y_max)
            distance = math.hypot(float(pose[0]) - nearest_x, float(pose[1]) - nearest_y)
            required = float(registry[name]["footprint_radius_m"]) + .003
            if distance < required:
                layout_failures.append(f"{capture['capture_id']}:{name}: target-bin collision")
        for index, (name, pose) in enumerate(active):
            for other, other_pose in active[index + 1:]:
                distance = math.hypot(float(pose[0]) - float(other_pose[0]), float(pose[1]) - float(other_pose[1]))
                required = (
                    float(registry[name]["footprint_radius_m"])
                    + float(registry[other]["footprint_radius_m"]) + .003
                )
                if distance < required:
                    layout_failures.append(f"{capture['capture_id']}:{name}/{other}: collision")
    cross_family_layouts = [key for key, value in fingerprint_families.items() if len(value) > 1]
    cross_split_layouts = [key for key, value in fingerprint_splits.items() if len(value) > 1]
    if cross_family_layouts:
        layout_failures.append(f"exact active layout reused across families: {len(cross_family_layouts)}")
    if cross_split_layouts:
        layout_failures.append(f"exact active layout leaked across train/dev: {len(cross_split_layouts)}")
    checks["layout_bounds_collision_and_uniqueness"] = {
        "passed": not layout_failures,
        "layouts_checked": len(plan["captures"]),
        "unique_layout_fingerprints": len(fingerprint_families),
        "cross_family_duplicates": len(cross_family_layouts),
        "cross_split_duplicates": len(cross_split_layouts),
        "failures": layout_failures[:100],
    }
    if layout_failures:
        failures.append("layout bounds, collision or uniqueness preflight failed")

    visibility_failures = []
    visibility_envelope = {"x_m": [-.42, -.08], "y_m": [.12, .50]}
    for capture in plan["captures"]:
        for object_id in capture["required_visible_object_ids"]:
            pose = capture["layout"][by_id[object_id]["model_name"]]
            if not (
                visibility_envelope["x_m"][0] <= float(pose[0]) <= visibility_envelope["x_m"][1]
                and visibility_envelope["y_m"][0] <= float(pose[1]) <= visibility_envelope["y_m"][1]
            ):
                visibility_failures.append(f"{capture['capture_id']}:{object_id}:{pose[:2]}")
    checks["required_evidence_visibility_envelope"] = {
        "passed": not visibility_failures,
        "envelope_base_link": visibility_envelope,
        "failures": visibility_failures[:100],
    }
    if visibility_failures:
        failures.append("required target/anchor evidence outside calibrated visibility envelope")

    capture_by_id = {item["capture_id"]: item for item in plan["captures"]}
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
        elif state == "ABSENT" and (valid or clean["answerable"]):
            ontology_failures.append(f"{family['family_id']}: ABSENT cardinality")
        elif state == "INSUFFICIENT_EVIDENCE" and (len(valid) != 1 or clean["answerable"]):
            ontology_failures.append(f"{family['family_id']}: INSUFFICIENT cardinality")
        if len(set(candidates + anchors)) != len(candidates + anchors):
            ontology_failures.append(f"{family['family_id']}: target/anchor identity overlap")
        submode = clean["state_submode"]
        if submode == "target_absent" and any(value in active for value in candidates):
            ontology_failures.append(f"{family['family_id']}: target_absent candidate active")
        if submode == "anchor_absent" and (not anchors or any(value in active for value in anchors)):
            ontology_failures.append(f"{family['family_id']}: anchor_absent realization")
        if state in {"FOUND", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"} and not all(value in active for value in valid):
            ontology_failures.append(f"{family['family_id']}: valid referent outside active scene")
        if state == "AMBIGUOUS":
            classes = {by_id[value]["semantic_class"] for value in valid}
            if len(classes) > 1 and not classes.issubset({"apple", "orange", "lemon", "mango"}):
                ontology_failures.append(f"{family['family_id']}: ambiguous candidates lack common semantic query")
        if submode == "multi_anchor_conflict" and len(anchors) < 2:
            ontology_failures.append(f"{family['family_id']}: multi-anchor conflict lacks two anchors")
        if submode in {"relation_tie", "unsatisfied_relation"} and clean["relations"] == ["direct"]:
            ontology_failures.append(f"{family['family_id']}: relation submode uses direct query")

        capture = capture_by_id[clean["capture_id"]]
        layout = capture["layout"]
        relation = clean["relations"][0]
        anchor_poses = [layout[by_id[value]["model_name"]] for value in anchors if value in active]
        if submode == "unsatisfied_relation":
            relation_checks += len(candidates)
            satisfied = any(
                geometry_satisfied(relation, layout[by_id[value]["model_name"]], anchor_poses)
                for value in candidates if value in active
            )
            if satisfied or not all(value in active for value in candidates + anchors):
                ontology_failures.append(f"{family['family_id']}: unsatisfied relation not isolated")
        elif state != "ABSENT" and relation != "direct":
            for value in valid:
                relation_checks += 1
                if not geometry_satisfied(relation, layout[by_id[value]["model_name"]], anchor_poses):
                    ontology_failures.append(f"{family['family_id']}:{value}: relation geometry false")
    checks["ontology_semantics_and_relation_geometry"] = {
        "passed": not ontology_failures,
        "relation_checks": relation_checks,
        "failures": ontology_failures[:100],
    }
    if ontology_failures:
        failures.append("ontology, semantic ambiguity or relation geometry failed")

    seen = set(partition["seen_pool"]["asset_ids"])
    for clones in partition["clone_policy"]["clone_groups"].values():
        seen.update(clones)
    heldout = set(partition["test_ood_asset_holdout"]["asset_ids"])
    excluded = set(partition["excluded_assets"]["asset_ids"])
    runtime_reserved = {"cube_red_01"}
    partition_failures = []
    for family in manifest["families"]:
        active = set(family["primary_scene"]["active_scene_ids"])
        evidence = set()
        for spec in family["variant_specs"]:
            evidence.update(spec["candidate_target_ids"])
            evidence.update(spec["valid_target_ids"])
            evidence.update(spec["anchor_ids"])
        used = active | evidence
        if family["ood_axis"] != "none" or family["asset_pool"] != "seen":
            partition_failures.append(f"{family['family_id']}: non-IID development assignment")
        if used - seen:
            partition_failures.append(f"{family['family_id']}: non-seen IDs {sorted(used - seen)}")
        if used & heldout:
            partition_failures.append(f"{family['family_id']}: Test-OOD asset leakage")
        if used & excluded:
            partition_failures.append(f"{family['family_id']}: excluded asset leakage")
        if used & runtime_reserved:
            partition_failures.append(f"{family['family_id']}: runtime fixture used as task evidence")
    checks["seen_asset_partition"] = {
        "passed": not partition_failures,
        "seen_asset_count_including_clones": len(seen),
        "heldout_assets": sorted(heldout),
        "excluded_assets": sorted(excluded),
        "runtime_reserved_assets": sorted(runtime_reserved),
        "failures": partition_failures[:100],
    }
    if partition_failures:
        failures.append("seen-only development asset partition failed")

    development_instructions = [
        spec["instruction"] for family in manifest["families"] for spec in family["variant_specs"]
    ]
    pilot_instructions = {
        spec["instruction"] for family in pilot_manifest["families"] for spec in family["variant_specs"]
    }
    train_instructions = {
        spec["instruction"] for family in manifest["families"] if family["split"] == "train"
        for spec in family["variant_specs"]
    }
    dev_instructions = {
        spec["instruction"] for family in manifest["families"] if family["split"] == "dev"
        for spec in family["variant_specs"]
    }
    template_ids = [
        spec["language_template_family_id"]
        for family in manifest["families"] for spec in family["variant_specs"]
    ]
    instruction_ok = (
        len(development_instructions) == len(set(development_instructions)) == 2000
        and not (set(development_instructions) & pilot_instructions)
        and not (train_instructions & dev_instructions)
        and len(template_ids) == len(set(template_ids)) == 2000
        and not any(family_id in instruction for family_id in family_ids for instruction in development_instructions)
    )
    checks["instruction_and_template_disjointness"] = {
        "passed": instruction_ok,
        "development_instruction_count": len(development_instructions),
        "unique_instruction_count": len(set(development_instructions)),
        "pilot_overlap_count": len(set(development_instructions) & pilot_instructions),
        "train_dev_overlap_count": len(train_instructions & dev_instructions),
        "unique_template_family_count": len(set(template_ids)),
    }
    if not instruction_ok:
        failures.append("instruction or language-template family leakage failed")

    pilot_family_ids = {item["family_id"] for item in pilot_manifest["families"]}
    pilot_sample_ids = {
        f"{family['family_id']}__{spec['variant']}"
        for family in pilot_manifest["families"] for spec in family["variant_specs"]
    }
    pilot_capture_ids = {item["capture_id"] for item in pilot_plan["captures"]}
    development_seeds = [value for item in manifest["families"] for value in item["seed_bundle"].values()]
    pilot_seeds = {value for item in pilot_manifest["families"] for value in item["seed_bundle"].values()}
    pilot_fingerprints = {
        canonical_json_sha256(active_layout(item)) for item in pilot_plan["captures"]
    }
    development_fingerprints = set(fingerprint_families)
    train_layout_seeds = {
        item["seed_bundle"]["layout"] for item in manifest["families"] if item["split"] == "train"
    }
    dev_layout_seeds = {
        item["seed_bundle"]["layout"] for item in manifest["families"] if item["split"] == "dev"
    }
    disjoint_ok = (
        not (set(family_ids) & pilot_family_ids)
        and not (set(sample_ids) & pilot_sample_ids)
        and not (set(capture_ids) & pilot_capture_ids)
        and len(development_seeds) == len(set(development_seeds)) == 2000
        and not (set(development_seeds) & pilot_seeds)
        and not (development_fingerprints & pilot_fingerprints)
        and not (train_layout_seeds & dev_layout_seeds)
    )
    checks["pilot_and_split_disjointness"] = {
        "passed": disjoint_ok,
        "development_seed_count": len(development_seeds),
        "unique_development_seed_count": len(set(development_seeds)),
        "pilot_seed_overlap_count": len(set(development_seeds) & pilot_seeds),
        "pilot_layout_overlap_count": len(development_fingerprints & pilot_fingerprints),
        "train_dev_layout_seed_overlap_count": len(train_layout_seeds & dev_layout_seeds),
    }
    if not disjoint_ok:
        failures.append("pilot/development or train/dev identity, seed or layout leakage failed")

    payload_findings = find_forbidden_inference_keys(plan["inference_payload_template"])
    sealed_split_leakage = [
        item["family_id"] for item in manifest["families"]
        if item["split"] not in {"train", "dev"}
    ] + [
        item["capture_id"] for item in plan["captures"]
        if item["split"] not in {"train", "dev"}
    ]
    checks["oracle_and_sealed_split_boundary"] = {
        "passed": not payload_findings and not sealed_split_leakage,
        "inference_payload_findings": payload_findings,
        "sealed_split_leakage": sealed_split_leakage,
        "sealed_splits": manifest["sealed_splits_not_created"],
    }
    if payload_findings or sealed_split_leakage:
        failures.append("oracle-free payload or sealed calibration/test boundary failed")

    pilot_gate_ok = (
        pilot_qc.get("passed") is True
        and pilot_qc.get("decision") == "GO_DEVELOPMENT_CAPTURE_400"
        and pilot_qc.get("training_performed") is False
        and pilot_raw.get("complete") is True
        and pilot_raw.get("capture_count") == 60
    )
    pilot_tree = tree_digest(workspace, PILOT_ROOT)
    checks["pilot_gate_prerequisite"] = {
        "passed": pilot_gate_ok,
        "pilot_qc_sha256": sha256_file(pilot_qc_path),
        "pilot_raw_manifest_sha256": sha256_file(pilot_raw_path),
        "pilot_tree_digest": pilot_tree,
        "decision": pilot_qc.get("decision"),
    }
    if not pilot_gate_ok:
        failures.append("pilot prerequisite is not a complete QC PASS")

    protected_failures = []
    for relative, expected in seed_lock["protected_files"].items():
        path = workspace / relative
        if not path.is_file() or sha256_file(path) != expected:
            protected_failures.append(relative)
    protected_trees = {}
    for relative, expected in seed_lock["protected_tree_digests"].items():
        actual = tree_digest(workspace, relative)
        protected_trees[relative] = actual
        if actual != expected:
            protected_failures.append(relative)
    checks["protected_wp0_wp3_wp2_hashes"] = {
        "passed": not protected_failures,
        "failures": protected_failures,
        "tree_digests": protected_trees,
    }
    if protected_failures:
        failures.append("protected WP0-WP3 or WP2 artifact/tree changed")

    missing_runtime = [relative for relative in RUNTIME_ARTIFACTS if not (workspace / relative).is_file()]
    output_root = workspace / "datasets/roborefer_dataset_v2_development_400_20260821"
    output_clean = not output_root.exists() or not any(output_root.iterdir())
    checks["runtime_and_output_precondition"] = {
        "passed": not missing_runtime and output_clean,
        "missing_runtime_artifacts": missing_runtime,
        "capture_output_root": str(output_root.relative_to(workspace)),
        "capture_output_root_absent_or_empty": output_clean,
    }
    if missing_runtime or not output_clean:
        failures.append("development runtime incomplete or output root is not clean")

    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "scientific_status": "STATIC_PREFLIGHT_PLANNED_NOT_CAPTURED",
        "passed": not failures,
        "decision": "GO_DEVELOPMENT_CAPTURE_400" if not failures else "FIX_DEVELOPMENT_MANIFEST_OR_CAPTURE_PIPELINE_FIRST",
        "failures": failures,
        "checks": checks,
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
    report_path = protocol / "dataset_v2_development_preflight_report.json"
    lock_path = protocol / "dataset_v2_development_execution_lock.json"
    write_json(report_path, report)
    locked_hashes = {
        relative: sha256_file(workspace / relative)
        for relative in RUNTIME_ARTIFACTS if (workspace / relative).is_file()
    }
    pilot_tree = tree_digest(workspace, PILOT_ROOT)
    lock_payload = {
        "capture_plan_sha256": sha256_file(protocol / "dataset_v2_development_capture_plan.json"),
        "manifest_sha256": sha256_file(protocol / "dataset_v2_development_manifest.json"),
        "preflight_report_sha256": sha256_file(report_path),
        "locked_artifact_sha256": locked_hashes,
        "pilot_tree_digest": pilot_tree,
    }
    lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "locked_at_utc": utc_now(),
        "decision": report["decision"],
        "capture_authorized": bool(report["passed"]),
        "capture_output_root": "datasets/roborefer_dataset_v2_development_400_20260821",
        "required_family_count": 400,
        "required_raw_capture_count": 800,
        "required_materialized_sample_count": 2000,
        "batch_family_counts": [20, 95, 95, 95, 95],
        "capture_order_requires_previous_batch_qc": True,
        "pilot_reuse_forbidden": True,
        "training_forbidden_until_full_qc": True,
        "calibration_test_forbidden": True,
        **lock_payload,
        "lock_commitment_sha256": canonical_json_sha256(lock_payload),
    }
    write_json(lock_path, lock)
    print(
        "DATASET_V2_DEVELOPMENT_PREFLIGHT "
        f"passed={report['passed']} decision={report['decision']} failures={len(report['failures'])}"
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
