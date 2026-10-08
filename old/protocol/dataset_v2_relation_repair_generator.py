#!/usr/bin/env python3
"""Generate the deterministic Dataset V2.1 relation-repair pilot.

This program performs static geometry only.  It never starts Gazebo and never
uses a model prediction or a repair-pilot observation to choose a layout.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2_pilot_generator import object_registry  # noqa: E402
from dataset_v2_relation_geometry_v2_1 import (  # noqa: E402
    GEOMETRY_VERSION,
    PREDICTOR_SCREENING_MARGIN_M,
    asset_shape_group,
    predicted_relation,
    project_base_point_to_camera,
)
from wp2_common import canonical_json_sha256, read_json, sha256_file, write_json  # noqa: E402


PROTOCOL_ID = "roborefer_dataset_v2_1_relation_repair"
GENERATOR_VERSION = "dataset_v2_relation_repair_generator_v1.1.0"
MASTER_SEED = 210820261
FAMILY_COUNT = 30
CANDIDATE_POOL_SIZE = 65536
BETWEEN_DEPTH_SCREENING_MARGIN_M = 0.070
OUTPUT_ROOT = "datasets/roborefer_dataset_v2_1_relation_repair_pilot_round2_only_20260821"
WORLD_FILE = "ur3/ur_simulation_gz/worlds/ur3_pick_place_inventory_v2.sdf"
CAMERA_LOCK = "protocol/dataset_v2_relation_camera_lock_v2_1.json"
CALIBRATION = "results/dataset_v2_relation_repair_20260821/audit/asset_depth_calibration.json"
OLD_PLAN_PATHS = (
    "protocol/dataset_v2_pilot_capture_plan.json",
    "protocol/dataset_v2_development_capture_plan.json",
)


class RepairGenerationError(RuntimeError):
    """Raised when the preregistered repair composition cannot be realized."""


def templates() -> list[dict[str, Any]]:
    """Preregistered relation/asset cells; order defines family identity."""

    rows = [
        # front_of: includes the tall/round stress pair.
        ("front_of", "ycb_mustard_bottle", ["ycb_orange"]),
        ("front_of", "ycb_tomato_soup_can", ["purple_cube"]),
        ("front_of", "ycb_apple", ["green_cube"]),
        ("front_of", "ycb_lemon", ["blue_cube"]),
        # behind: the first row is the reverse physical order stress case.
        ("behind", "ycb_mustard_bottle", ["ycb_orange"]),
        ("behind", "ycb_tomato_soup_can", ["ycb_apple"]),
        ("behind", "ycb_orange", ["yellow_cube"]),
        ("behind", "green_cube", ["ycb_lemon"]),
        ("nearer_than", "ycb_tomato_soup_can", ["ycb_apple"]),
        ("nearer_than", "ycb_mustard_bottle", ["purple_cube"]),
        ("nearer_than", "ycb_orange", ["blue_cube"]),
        ("nearer_than", "ycb_mug", ["yellow_cube"]),
        ("farther_than", "ycb_tomato_soup_can", ["ycb_apple"]),
        ("farther_than", "ycb_mustard_bottle", ["green_cube"]),
        ("farther_than", "ycb_orange", ["yellow_cube"]),
        ("farther_than", "purple_cube", ["ycb_lemon"]),
        # Six multi-anchor interval cases, including two soup-can cells.
        ("between_in_depth", "ycb_tomato_soup_can", ["purple_cube", "ycb_orange"]),
        ("between_in_depth", "ycb_tomato_soup_can", ["blue_cube", "ycb_apple"]),
        ("between_in_depth", "ycb_apple", ["yellow_cube", "ycb_lemon"]),
        ("between_in_depth", "ycb_orange", ["purple_cube", "green_cube"]),
        ("between_in_depth", "ycb_mug", ["ycb_mustard_bottle", "blue_cube"]),
        ("between_in_depth", "ycb_lemon", ["ycb_tomato_soup_can", "pink_cube"]),
        ("nearer_than_both", "ycb_tomato_soup_can", ["ycb_orange", "purple_cube"]),
        ("nearer_than_both", "ycb_mustard_bottle", ["ycb_apple", "green_cube"]),
        ("nearer_than_both", "ycb_apple", ["ycb_tomato_soup_can", "yellow_cube"]),
        ("nearer_than_both", "ycb_lemon", ["ycb_mustard_bottle", "blue_cube"]),
        ("left_of", "ycb_mustard_bottle", ["ycb_orange"]),
        ("left_of", "ycb_tomato_soup_can", ["ycb_apple"]),
        ("right_of", "ycb_orange", ["ycb_mustard_bottle"]),
        ("right_of", "purple_cube", ["ycb_tomato_soup_can"]),
    ]
    return [
        {"relation": relation, "target_model": target, "anchor_models": anchors}
        for relation, target, anchors in rows
    ]


def instruction(relation: str, target: dict[str, Any], anchors: list[dict[str, Any]]) -> str:
    semantic = target["semantic_class"]
    names = [value["semantic_class"] for value in anchors]
    if relation in {"left_of", "right_of", "front_of", "behind", "nearer_than", "farther_than"}:
        return f"Locate the {semantic} that is {relation.replace('_', ' ')} the {names[0]}."
    if relation == "between_in_depth":
        return f"Locate the {semantic} between the {names[0]} and {names[1]} in camera depth."
    if relation == "nearer_than_both":
        return f"Locate the {semantic} nearer to the camera than both the {names[0]} and {names[1]}."
    raise RepairGenerationError(f"unsupported instruction relation: {relation}")


def collect_seed_values(value: Any, key: str = "") -> set[int]:
    values: set[int] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            values.update(collect_seed_values(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            values.update(collect_seed_values(child, key))
    elif isinstance(value, int) and "seed" in key.lower():
        values.add(int(value))
    return values


def seed_for(family_index: int, candidate_index: int) -> int:
    digest = hashlib.sha256(
        f"{PROTOCOL_ID}|{MASTER_SEED}|{family_index}|{candidate_index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def rank_hash(family_index: int, candidate_seed: int) -> str:
    return hashlib.sha256(
        f"{PROTOCOL_ID}|rank|{family_index}|{candidate_seed}".encode()
    ).hexdigest()


def candidate_positions(seed: int, count: int) -> list[list[float]]:
    generator = random.Random(seed)
    return [
        [
            round(generator.uniform(-0.40, -0.14), 6),
            round(generator.uniform(0.08, 0.58), 6),
            round(generator.uniform(-math.pi, math.pi), 6),
        ]
        for _ in range(count)
    ]


def projection(
    pose: list[float], row: dict[str, Any], camera_tf: dict[str, Any], intrinsics: dict[str, float]
) -> dict[str, float]:
    point = project_base_point_to_camera(
        [float(pose[0]), float(pose[1]), float(row["z"])], camera_tf
    )
    if point[2] <= 0:
        return {"visible": False, "depth_m": float(point[2])}
    u = float(intrinsics["fx"] * point[0] / point[2] + intrinsics["cx"])
    v = float(intrinsics["fy"] * point[1] / point[2] + intrinsics["cy"])
    half_height = float(row["z"])
    radius = float(row["footprint_radius_m"])
    pad = float(intrinsics["fx"] * max(radius, half_height) / point[2] + 18.0)
    visible = point[2] >= 0.18 and pad < u < 640.0 - pad and pad < v < 480.0 - pad
    return {"visible": bool(visible), "depth_m": float(point[2]), "u_px": u, "v_px": v, "pad_px": pad}


def bin_clear(pose: list[float], row: dict[str, Any]) -> bool:
    nearest_x = min(max(float(pose[0]), -0.10), 0.10)
    nearest_y = min(max(float(pose[1]), 0.255), 0.505)
    distance = math.hypot(float(pose[0]) - nearest_x, float(pose[1]) - nearest_y)
    return distance >= float(row["footprint_radius_m"]) + 0.005


def collision_free(
    poses: list[list[float]], rows: list[dict[str, Any]], clearance_m: float = 0.020
) -> bool:
    for index, pose in enumerate(poses):
        if not bin_clear(pose, rows[index]):
            return False
        for other_index in range(index + 1, len(poses)):
            distance = math.hypot(
                float(pose[0]) - float(poses[other_index][0]),
                float(pose[1]) - float(poses[other_index][1]),
            )
            required = (
                float(rows[index]["footprint_radius_m"])
                + float(rows[other_index]["footprint_radius_m"])
                + clearance_m
            )
            if distance < required:
                return False
    return True


def layout_fingerprint(active: dict[str, list[float]]) -> str:
    return canonical_json_sha256({key: active[key] for key in sorted(active)})


def select_layout(
    family_index: int,
    template: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    calibration: dict[str, Any],
    camera_tf: dict[str, Any],
    intrinsics: dict[str, float],
    denied_seeds: set[int],
    used_fingerprints: set[str],
) -> tuple[dict[str, list[float]], int, dict[str, Any], list[dict[str, Any]]]:
    names = [template["target_model"], *template["anchor_models"]]
    rows = [registry[name] for name in names]
    calibrations = [calibration[name] for name in names]
    ranked_pool = sorted(
        (
            (rank_hash(family_index, seed_for(family_index, index)), seed_for(family_index, index))
            for index in range(CANDIDATE_POOL_SIZE)
        ),
        key=lambda item: item[0],
    )
    audit: list[dict[str, Any]] = []
    for order, (ranking, candidate_seed) in enumerate(ranked_pool, start=1):
        record: dict[str, Any] = {
            "evaluation_order": order,
            "rank_sha256": ranking,
            "candidate_seed": candidate_seed,
        }
        if candidate_seed in denied_seeds:
            record.update({"accepted": False, "reason": "SEED_DENYLIST"})
            audit.append(record)
            continue
        poses = candidate_positions(candidate_seed, len(names))
        if not collision_free(poses, rows):
            record.update({"accepted": False, "reason": "COLLISION_OR_BIN"})
            audit.append(record)
            continue
        projections = [projection(pose, row, camera_tf, intrinsics) for pose, row in zip(poses, rows)]
        if not all(value["visible"] for value in projections):
            record.update({"accepted": False, "reason": "VISIBILITY_ENVELOPE"})
            audit.append(record)
            continue
        evidence = predicted_relation(
            template["relation"], poses[0], rows[0], calibrations[0], poses[1:], rows[1:],
            calibrations[1:], camera_tf, intrinsics,
            depth_margin_m=(
                BETWEEN_DEPTH_SCREENING_MARGIN_M
                if template["relation"] == "between_in_depth"
                else PREDICTOR_SCREENING_MARGIN_M
            ),
        )
        if not evidence["passed"]:
            record.update({
                "accepted": False,
                "reason": "PREDICTED_RELATION_FALSE",
                "signed_screening_margin": evidence["signed_screening_margin"],
            })
            audit.append(record)
            continue
        active = {name: pose for name, pose in zip(names, poses)}
        fingerprint = layout_fingerprint(active)
        if fingerprint in used_fingerprints:
            record.update({"accepted": False, "reason": "DUPLICATE_LAYOUT"})
            audit.append(record)
            continue
        record.update({
            "accepted": True,
            "reason": "FIRST_PASSING_CANDIDATE",
            "active_layout": active,
            "projection": projections,
            "predictor_evidence": evidence,
            "layout_fingerprint_sha256": fingerprint,
        })
        audit.append(record)
        used_fingerprints.add(fingerprint)
        return active, candidate_seed, evidence, audit
    raise RepairGenerationError(
        f"candidate pool exhausted for family index {family_index}: {template}"
    )


def build_artifacts(workspace: Path) -> tuple[dict, dict, dict, dict]:
    registry = object_registry()
    calibration_artifact = read_json(workspace / CALIBRATION)
    calibration = calibration_artifact["asset_calibration"]
    camera_lock = read_json(workspace / CAMERA_LOCK)
    repair_view = camera_lock["repair_view"]
    camera_tf = repair_view["fk_transform_base_to_camera"]
    intrinsics = camera_lock["intrinsics"]
    denied_seeds: set[int] = set()
    denied_sources = []
    for relative in OLD_PLAN_PATHS:
        artifact = read_json(workspace / relative)
        denied_seeds.update(collect_seed_values(artifact))
        denied_sources.append({"path": relative, "sha256": sha256_file(workspace / relative)})

    registry_by_id = {value["id"]: {**value, "model_name": name} for name, value in registry.items()}
    families = []
    captures = []
    candidate_rows = []
    selected_seeds: set[int] = set()
    used_fingerprints: set[str] = set()
    for family_index, template in enumerate(templates(), start=1):
        family_id = f"v21repair_family_{family_index:06d}"
        target_model = template["target_model"]
        anchor_models = list(template["anchor_models"])
        active, selected_seed, predictor, evaluated = select_layout(
            family_index, template, registry, calibration, camera_tf, intrinsics,
            denied_seeds | selected_seeds, used_fingerprints,
        )
        selected_seeds.add(selected_seed)
        capture_id = f"{family_id}__clean_capture"
        layout = {
            name: list(value["storage_pose_xyyaw"])
            for name, value in registry.items()
        }
        layout.update(active)
        target = registry[target_model]
        anchors = [registry[name] for name in anchor_models]
        family = {
            "family_id": family_id,
            "protocol_role": "REPAIR_PILOT_ENGINEERING_ONLY_NOT_OFFICIAL_DATA",
            "split": "repair_pilot_only",
            "official_dataset_eligible": False,
            "answerability_state": "FOUND",
            "relation": template["relation"],
            "target_id": target["id"],
            "anchor_ids": [value["id"] for value in anchors],
            "target_model": target_model,
            "anchor_models": anchor_models,
            "target_shape_group": asset_shape_group(target),
            "anchor_shape_groups": [asset_shape_group(value) for value in anchors],
            "instruction": instruction(template["relation"], target, anchors),
            "capture_id": capture_id,
            "layout_seed": selected_seed,
            "layout_fingerprint_sha256": layout_fingerprint(active),
            "predictor_evidence": predictor,
            "exclusion_reason": "REPAIR_PILOT_INSPECTED_ENGINEERING_DATA",
        }
        families.append(family)
        captures.append({
            "capture_id": capture_id,
            "family_id": family_id,
            "condition": "clean",
            "split": "repair_pilot_only",
            "seed": selected_seed,
            "ood_axis": "none",
            "camera_bin_id": repair_view["camera_bin_id"],
            "view_joint_pose": list(repair_view["requested_view_joint_pose"]),
            "layout": layout,
            "relation": template["relation"],
            "target_id": target["id"],
            "anchor_ids": [value["id"] for value in anchors],
            "required_visible_label_ids": [int(target["label"]), *[int(value["label"]) for value in anchors]],
        })
        candidate_rows.append({
            "family_id": family_id,
            "relation": template["relation"],
            "target_model": target_model,
            "anchor_models": anchor_models,
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "evaluated_candidate_count": len(evaluated),
            "selected_seed": selected_seed,
            "evaluated_candidates": evaluated,
        })

    relation_counts = dict(sorted(Counter(value["relation"] for value in families).items()))
    shape_counts = Counter()
    for family in families:
        shape_counts[family["target_shape_group"]] += 1
        shape_counts.update(family["anchor_shape_groups"])
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "protocol_role": "REPAIR_PILOT_ENGINEERING_ONLY_NOT_OFFICIAL_DATA",
        "family_count": FAMILY_COUNT,
        "capture_count": FAMILY_COUNT,
        "condition_counts": {"clean": FAMILY_COUNT},
        "relation_counts": relation_counts,
        "participating_shape_group_counts": dict(sorted(shape_counts.items())),
        "all_families_permanently_excluded_from_official_dataset": True,
        "families": families,
    }
    seed_lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "master_seed": MASTER_SEED,
        "candidate_pool_size_per_family": CANDIDATE_POOL_SIZE,
        "candidate_mapping": "SHA256(protocol_id|master_seed|family_index|candidate_index) first 63 bits",
        "candidate_ranking": "SHA256(protocol_id|rank|family_index|candidate_seed)",
        "selection_rule": "first candidate passing locked collision, projection and predictor filters",
        "selected_seed_count": len(selected_seeds),
        "selected_seeds": sorted(selected_seeds),
        "denied_seed_count": len(denied_seeds),
        "denied_source_artifacts": denied_sources,
        "repair_family_namespace": "v21repair_family_*",
        "official_reuse_forbidden": True,
    }
    plan = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "manifest_content_sha256": canonical_json_sha256(manifest),
        "seed_lock_content_sha256": canonical_json_sha256(seed_lock),
        "capture_output_root": OUTPUT_ROOT,
        "world_file": WORLD_FILE,
        "family_count": FAMILY_COUNT,
        "capture_count": FAMILY_COUNT,
        "captures_per_family": 1,
        "workspace_roi_base_link": {"x_m": [-0.40, -0.14], "y_m": [0.08, 0.58]},
        "camera_poses": {repair_view["camera_bin_id"]: repair_view["requested_view_joint_pose"]},
        "camera_predictor_transform": camera_tf,
        "camera_intrinsics": intrinsics,
        "sensor_contract": {
            "resolution_hw": [480, 640],
            "max_timestamp_spread_sec": 0.08,
            "required_artifacts": [
                "rgb_original.png", "depth_metric.npy", "semantic_instance_labels.png",
                "camera_info.json", "tf_snapshot.json",
            ],
        },
        "observable_relation_contract": {
            "amendment": "protocol/DATASET_V2_RELATION_GEOMETRY_AMENDMENT_02.md",
            "runtime_depth_margin_m": 0.020,
            "predictor_screening_margin_m": PREDICTOR_SCREENING_MARGIN_M,
            "between_depth_predictor_screening_margin_m": BETWEEN_DEPTH_SCREENING_MARGIN_M,
            "horizontal_margin_px": 12,
            "label_source": "synchronized clean capture mask centroid and robust median metric depth",
        },
        "object_registry": registry,
        "inference_payload_template": {
            "rgb_model_input": "CAPTURED_RGB",
            "depth_model_input": "CAPTURED_METRIC_DEPTH",
            "prompt": "FAMILY_INSTRUCTION_ONLY",
        },
        "captures": captures,
    }
    candidate_audit = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "selection_performed_before_repair_capture": True,
        "repair_capture_observations_used": 0,
        "family_count": len(candidate_rows),
        "families": candidate_rows,
    }
    return manifest, plan, seed_lock, candidate_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    manifest, plan, seed_lock, audit = build_artifacts(workspace)
    protocol = workspace / "protocol"
    write_json(protocol / "dataset_v2_relation_repair_manifest.json", manifest)
    write_json(protocol / "dataset_v2_relation_repair_seed_lock.json", seed_lock)
    write_json(protocol / "dataset_v2_relation_repair_capture_plan.json", plan)
    write_json(protocol / "dataset_v2_relation_repair_candidate_audit.json", audit)
    print(
        "DATASET_V2_RELATION_REPAIR_GENERATED "
        f"families={manifest['family_count']} captures={plan['capture_count']} "
        f"candidate_evaluations={sum(v['evaluated_candidate_count'] for v in audit['families'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
