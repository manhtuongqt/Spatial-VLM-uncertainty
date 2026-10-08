#!/usr/bin/env python3
"""Static preflight and execution-lock writer for the V2.1 repair pilot."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2_relation_geometry_v2_1 import predicted_relation  # noqa: E402
from dataset_v2_relation_repair_generator import (  # noqa: E402
    BETWEEN_DEPTH_SCREENING_MARGIN_M,
    FAMILY_COUNT,
    OUTPUT_ROOT,
    PROTOCOL_ID,
    build_artifacts,
    collision_free,
    projection,
)
from wp2_common import (  # noqa: E402
    canonical_json_sha256,
    find_forbidden_inference_keys,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)


EXPECTED_RELATIONS = {
    "front_of": 4,
    "behind": 4,
    "nearer_than": 4,
    "farther_than": 4,
    "between_in_depth": 6,
    "nearer_than_both": 4,
    "left_of": 2,
    "right_of": 2,
}
RUNTIME_ARTIFACTS = (
    "protocol/DATASET_V2_RELATION_GEOMETRY_AMENDMENT_01.md",
    "protocol/DATASET_V2_RELATION_GEOMETRY_AMENDMENT_02.md",
    "protocol/dataset_v2_relation_geometry_v2_1.py",
    "protocol/dataset_v2_relation_geometry_audit.py",
    "protocol/dataset_v2_relation_geometry_audit_230.json",
    "protocol/dataset_v2_relation_camera_lock_v2_1.json",
    "protocol/dataset_v2_relation_repair_generator.py",
    "protocol/dataset_v2_relation_repair_manifest.json",
    "protocol/dataset_v2_relation_repair_seed_lock.json",
    "protocol/dataset_v2_relation_repair_capture_plan.json",
    "protocol/dataset_v2_relation_repair_candidate_audit.json",
    "protocol/dataset_v2_relation_repair_preflight.py",
    "protocol/dataset_v2_relation_repair_capture.py",
    "protocol/dataset_v2_relation_repair_capture.launch.py",
    "protocol/dataset_v2_relation_repair_qc.py",
    "results/dataset_v2_relation_repair_20260821/audit/asset_depth_calibration.json",
    "protocol/dataset_v2_development_execution_lock.json",
    "datasets/roborefer_dataset_v2_development_400_20260821/raw/raw_capture_manifest.json",
    "protocol/dataset_v2_pilot_execution_lock.json",
    "ur3/ur_simulation_gz/worlds/ur3_pick_place_inventory_v2.sdf",
    "ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro",
    "ur3/ur3_moveit_control/config/ur3_susgrip_gazebo_controllers.yaml",
    "ur3/ur_simulation_gz/launch/ur_sim_control.launch.py",
)


def run_preflight(workspace: Path) -> dict[str, Any]:
    protocol = workspace / "protocol"
    manifest_path = protocol / "dataset_v2_relation_repair_manifest.json"
    plan_path = protocol / "dataset_v2_relation_repair_capture_plan.json"
    seed_path = protocol / "dataset_v2_relation_repair_seed_lock.json"
    audit_path = protocol / "dataset_v2_relation_repair_candidate_audit.json"
    manifest = read_json(manifest_path)
    plan = read_json(plan_path)
    seed_lock = read_json(seed_path)
    candidate_audit = read_json(audit_path)
    failures: list[str] = []
    checks: dict[str, Any] = {}

    replay_manifest, replay_plan, replay_seed, replay_audit = build_artifacts(workspace)
    replay_ok = (
        manifest == replay_manifest and plan == replay_plan
        and seed_lock == replay_seed and candidate_audit == replay_audit
    )
    checks["static_generator_replay"] = {"passed": replay_ok}
    if not replay_ok:
        failures.append("static generator replay differs from materialized artifacts")

    relation_counts = dict(sorted(Counter(value["relation"] for value in manifest["families"]).items()))
    identities = [value["family_id"] for value in manifest["families"]]
    capture_ids = [value["capture_id"] for value in plan["captures"]]
    counts_ok = (
        manifest.get("family_count") == FAMILY_COUNT
        and manifest.get("capture_count") == FAMILY_COUNT
        and plan.get("family_count") == FAMILY_COUNT
        and plan.get("capture_count") == FAMILY_COUNT
        and len(set(identities)) == FAMILY_COUNT
        and len(set(capture_ids)) == FAMILY_COUNT
        and relation_counts == EXPECTED_RELATIONS
        and all(value.startswith("v21repair_family_") for value in identities)
        and all(value.endswith("__clean_capture") for value in capture_ids)
    )
    checks["locked_composition"] = {
        "passed": counts_ok,
        "family_count": len(identities),
        "capture_count": len(capture_ids),
        "relation_counts": relation_counts,
    }
    if not counts_ok:
        failures.append("repair family/capture/relation composition differs from contract")

    exclusion_ok = (
        manifest.get("all_families_permanently_excluded_from_official_dataset") is True
        and all(
            value.get("protocol_role") == "REPAIR_PILOT_ENGINEERING_ONLY_NOT_OFFICIAL_DATA"
            and value.get("official_dataset_eligible") is False
            and value.get("split") == "repair_pilot_only"
            for value in manifest["families"]
        )
    )
    checks["permanent_official_exclusion"] = {"passed": exclusion_ok}
    if not exclusion_ok:
        failures.append("repair pilot is not permanently excluded from official data")

    registry = plan["object_registry"]
    by_id = {value["id"]: {**value, "model_name": name} for name, value in registry.items()}
    world_text = (workspace / plan["world_file"]).read_text(encoding="utf-8")
    world_failures = []
    for name, value in registry.items():
        start = world_text.find(f'<model name="{name}">')
        end = world_text.find("</model>", start)
        if start < 0 or end < 0 or f"<label>{int(value['label'])}</label>" not in world_text[start:end]:
            world_failures.append(f"{name}: missing model or semantic label {value['label']}")
    registry_ok = (
        len(registry) == len({value["id"] for value in registry.values()})
        and len(registry) == len({int(value["label"]) for value in registry.values()})
        and not world_failures
    )
    checks["registry_world_semantic_labels"] = {
        "passed": registry_ok,
        "registry_count": len(registry),
        "failures": world_failures,
    }
    if not registry_ok:
        failures.append("object registry/world semantic-instance mapping failed")

    calibration = read_json(
        workspace / "results/dataset_v2_relation_repair_20260821/audit/asset_depth_calibration.json"
    )["asset_calibration"]
    camera_lock = read_json(workspace / "protocol/dataset_v2_relation_camera_lock_v2_1.json")
    repair_view = camera_lock["repair_view"]
    camera_tf = repair_view["fk_transform_base_to_camera"]
    intrinsics = camera_lock["intrinsics"]
    family_by_id = {value["family_id"]: value for value in manifest["families"]}
    geometry_failures = []
    fingerprints = set()
    for capture in plan["captures"]:
        family = family_by_id[capture["family_id"]]
        names = [family["target_model"], *family["anchor_models"]]
        poses = [capture["layout"][name] for name in names]
        rows = [registry[name] for name in names]
        if set(capture["layout"]) != set(registry):
            geometry_failures.append(f"{capture['capture_id']}: layout registry coverage")
            continue
        if not collision_free(poses, rows):
            geometry_failures.append(f"{capture['capture_id']}: collision or bin clearance")
        projections = [projection(pose, row, camera_tf, intrinsics) for pose, row in zip(poses, rows)]
        if not all(value["visible"] for value in projections):
            geometry_failures.append(f"{capture['capture_id']}: conservative projection envelope")
        if any(name not in calibration for name in names):
            geometry_failures.append(f"{capture['capture_id']}: missing asset calibration")
            continue
        evidence = predicted_relation(
            capture["relation"], poses[0], rows[0], calibration[names[0]],
            poses[1:], rows[1:], [calibration[name] for name in names[1:]],
            camera_tf, intrinsics,
            depth_margin_m=(
                BETWEEN_DEPTH_SCREENING_MARGIN_M
                if capture["relation"] == "between_in_depth"
                else 0.035
            ),
        )
        if not evidence["passed"]:
            geometry_failures.append(f"{capture['capture_id']}: predictor relation false")
        if canonical_json_sha256({name: capture["layout"][name] for name in sorted(names)}) in fingerprints:
            geometry_failures.append(f"{capture['capture_id']}: duplicate active layout")
        fingerprints.add(canonical_json_sha256({name: capture["layout"][name] for name in sorted(names)}))
        required = {int(by_id[value]["label"]) for value in [capture["target_id"], *capture["anchor_ids"]]}
        if required != set(capture["required_visible_label_ids"]):
            geometry_failures.append(f"{capture['capture_id']}: visible-label contract mismatch")
    checks["camera_frame_predictor_collision_visibility"] = {
        "passed": not geometry_failures,
        "layouts_checked": len(plan["captures"]),
        "unique_layout_fingerprints": len(fingerprints),
        "predictor_screening_margin_m": 0.035,
        "between_depth_predictor_screening_margin_m": BETWEEN_DEPTH_SCREENING_MARGIN_M,
        "failures": geometry_failures,
    }
    if geometry_failures:
        failures.append("camera-frame geometry/collision/visibility preflight failed")

    mustard_stress = [
        value for value in manifest["families"]
        if value["target_model"] == "ycb_mustard_bottle"
        and value["anchor_models"] == ["ycb_orange"]
        and value["relation"] in {"front_of", "behind"}
    ]
    soup_multi = [
        value for value in manifest["families"]
        if value["target_model"] == "ycb_tomato_soup_can" and len(value["anchor_models"]) == 2
    ]
    shape_groups = set(manifest["participating_shape_group_counts"])
    stress_ok = (
        {value["relation"] for value in mustard_stress} == {"front_of", "behind"}
        and len(soup_multi) >= 3
        and {"tall", "low", "round", "box"}.issubset(shape_groups)
    )
    checks["required_stress_cases"] = {
        "passed": stress_ok,
        "mustard_orange_physical_orders": sorted(value["relation"] for value in mustard_stress),
        "soup_multi_anchor_family_count": len(soup_multi),
        "shape_groups": sorted(shape_groups),
    }
    if not stress_ok:
        failures.append("required asset-pair, multi-anchor or shape stress coverage failed")

    candidate_failures = []
    for family in candidate_audit["families"]:
        rows = family["evaluated_candidates"]
        accepted = [value for value in rows if value["accepted"]]
        if (
            len(accepted) != 1 or rows[-1] != accepted[0]
            or accepted[0]["reason"] != "FIRST_PASSING_CANDIDATE"
            or family["selected_seed"] != accepted[0]["candidate_seed"]
        ):
            candidate_failures.append(f"{family['family_id']}: selection trace")
    candidate_ok = (
        not candidate_failures
        and candidate_audit.get("selection_performed_before_repair_capture") is True
        and candidate_audit.get("repair_capture_observations_used") == 0
    )
    checks["deterministic_candidate_selection"] = {
        "passed": candidate_ok,
        "evaluated_candidate_count": sum(value["evaluated_candidate_count"] for value in candidate_audit["families"]),
        "failures": candidate_failures,
    }
    if not candidate_ok:
        failures.append("deterministic first-passing candidate policy failed")

    old_plans = [
        read_json(workspace / "protocol/dataset_v2_pilot_capture_plan.json"),
        read_json(workspace / "protocol/dataset_v2_development_capture_plan.json"),
    ]
    old_family_ids = {value["family_id"] for artifact in old_plans for value in artifact["captures"]}
    old_capture_ids = {value["capture_id"] for artifact in old_plans for value in artifact["captures"]}
    new_seeds = set(seed_lock["selected_seeds"])
    old_seeds = set()
    for artifact in old_plans:
        old_seeds.update(int(value["seed"]) for value in artifact["captures"])
    disjoint_ok = (
        not (set(identities) & old_family_ids)
        and not (set(capture_ids) & old_capture_ids)
        and not (new_seeds & old_seeds)
        and seed_lock.get("official_reuse_forbidden") is True
    )
    checks["old_v2_pilot_disjointness"] = {
        "passed": disjoint_ok,
        "old_family_id_count": len(old_family_ids),
        "old_capture_id_count": len(old_capture_ids),
        "selected_seed_count": len(new_seeds),
    }
    if not disjoint_ok:
        failures.append("repair IDs/captures/seeds overlap pilot or failed V2 development")

    heldout_ids = {"ycb_pear_01", "ycb_plum_01", "ycb_tuna_fish_can_01"}
    queried_ids = {
        value for family in manifest["families"]
        for value in [family["target_id"], *family["anchor_ids"]]
    }
    partition_ok = not (queried_ids & heldout_ids)
    checks["asset_partition"] = {
        "passed": partition_ok,
        "heldout_assets": sorted(heldout_ids),
        "leaked_assets": sorted(queried_ids & heldout_ids),
    }
    if not partition_ok:
        failures.append("Test-OOD held-out asset leaked into repair task evidence")

    oracle_findings = find_forbidden_inference_keys(plan["inference_payload_template"])
    checks["oracle_leakage"] = {"passed": not oracle_findings, "findings": oracle_findings}
    if oracle_findings:
        failures.append("oracle/evaluator field leaked into inference payload template")

    launch_text = (protocol / "dataset_v2_relation_repair_capture.launch.py").read_text(encoding="utf-8")
    qualified_runner_ok = (
        "ur3_susgrip_sim.launch.py" in launch_text
        and "ur3_demo_gripper.launch.py" in launch_text
        and "shared.PilotCapture" in (protocol / "dataset_v2_relation_repair_capture.py").read_text(encoding="utf-8")
    )
    checks["qualified_capture_pipeline"] = {
        "passed": qualified_runner_ok,
        "camera_motion": "/ur3_control through qualified MoveIt stack",
    }
    if not qualified_runner_ok:
        failures.append("repair runner differs from qualified capture pipeline")

    missing_runtime = [value for value in RUNTIME_ARTIFACTS if not (workspace / value).is_file()]
    checks["runtime_artifacts_present"] = {"passed": not missing_runtime, "missing": missing_runtime}
    if missing_runtime:
        failures.append("runtime artifact missing")

    decision = "GO_RELATION_REPAIR_CAPTURE_30" if not failures else "FIX_RELATION_REPAIR_PREFLIGHT"
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": utc_now(),
        "decision": decision,
        "passed": not failures,
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "training_performed": False,
        "calibration_or_test_opened": False,
    }
    write_json(protocol / "dataset_v2_relation_repair_preflight_report.json", report)
    if report["passed"]:
        locked_hashes = {value: sha256_file(workspace / value) for value in RUNTIME_ARTIFACTS}
        lock = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "locked_at_utc": utc_now(),
            "decision": decision,
            "capture_authorized": True,
            "capture_output_root": OUTPUT_ROOT,
            "family_count": FAMILY_COUNT,
            "capture_count": FAMILY_COUNT,
            "clean_only": True,
            "capture_plan_sha256": sha256_file(plan_path),
            "manifest_sha256": sha256_file(manifest_path),
            "seed_lock_sha256": sha256_file(seed_path),
            "candidate_audit_sha256": sha256_file(audit_path),
            "preflight_report_sha256": sha256_file(protocol / "dataset_v2_relation_repair_preflight_report.json"),
            "locked_artifact_sha256": locked_hashes,
            "moveit_used": True,
            "training_authorized": False,
            "official_dataset_eligible": False,
        }
        lock["lock_commitment_sha256"] = canonical_json_sha256(lock)
        write_json(protocol / "dataset_v2_relation_repair_execution_lock.json", lock)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    report = run_preflight(Path(args.workspace).expanduser().resolve())
    print(
        "DATASET_V2_RELATION_REPAIR_PREFLIGHT "
        f"decision={report['decision']} failures={report['failure_count']}"
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
