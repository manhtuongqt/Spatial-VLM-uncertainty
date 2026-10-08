#!/usr/bin/env python3
"""Materialize the new, deterministic 400-family Dataset V2.1 design.

This is a static generator only: it never starts ROS/Gazebo and never trains a
model.  The V2 manifest is used only as a locked quota/composition template;
all V2.1 family/capture IDs, seeds, instructions and layouts are regenerated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2_pilot_generator import COORDINATE_SUFFIX, object_registry  # noqa: E402
from dataset_v2_relation_geometry_v2_1 import (  # noqa: E402
    GEOMETRY_VERSION,
    PREDICTOR_SCREENING_MARGIN_M,
    predict_median_depth,
    predicted_relation,
)
from dataset_v2_relation_repair_generator import (  # noqa: E402
    BETWEEN_DEPTH_SCREENING_MARGIN_M,
    collision_free,
    projection,
)
from validate_dataset_expansion_v2 import VARIANTS  # noqa: E402
from wp2_common import canonical_json_sha256, read_json, sha256_file, write_json  # noqa: E402


PROTOCOL_ID = "roborefer_dataset_v2_1_1_development_capture_400"
GENERATOR_VERSION = "dataset_v2_1_development_generator_v1.1.0"
MASTER_SEED = 240820261226
FAMILY_PREFIX = "v211dev_family_"
FAMILY_COUNT = 400
OUTPUT_ROOT = "datasets/roborefer_dataset_v2_1_1_development_400_20260824"
WORLD_FILE = "ur3/ur_simulation_gz/worlds/ur3_pick_place_inventory_v2.sdf"
CAMERA_LOCK = "protocol/dataset_v2_relation_camera_lock_v2_1.json"
CALIBRATION = "results/dataset_v2_relation_repair_20260821/audit/asset_depth_calibration.json"
V2_MANIFEST = "protocol/dataset_v2_development_manifest.json"
V2_PLAN = "protocol/dataset_v2_development_capture_plan.json"
PARTITION = "protocol/dataset_expansion_v2_asset_partition.json"
LAYOUT_CANDIDATE_LIMIT = 65536

DENYLIST_JSON = (
    "protocol/dataset_v2_pilot_manifest.json",
    "protocol/dataset_v2_pilot_capture_plan.json",
    "protocol/dataset_v2_development_manifest.json",
    "protocol/dataset_v2_development_capture_plan.json",
    "protocol/dataset_v2_relation_repair_manifest.json",
    "protocol/dataset_v2_relation_repair_capture_plan.json",
    "protocol/dataset_v2_1_shutdown_gate_capture_plan.json",
    "datasets/roborefer_dataset_v2_pilot_only_20260821/raw/raw_capture_manifest.json",
    "datasets/roborefer_dataset_v2_development_400_20260821/raw/raw_capture_manifest.json",
    "datasets/roborefer_dataset_v2_1_relation_repair_pilot_only_20260821/raw/raw_capture_manifest.json",
    "datasets/roborefer_dataset_v2_1_relation_repair_pilot_round2_only_20260821/raw/raw_capture_manifest.json",
    "datasets/roborefer_dataset_v2_1_shutdown_gate_canary_20260824/raw/raw_capture_manifest.json",
    "results/dataset_v2_1_development_failed_visibility_20260824/protocol_lock_snapshot/dataset_v2_1_development_manifest.json",
    "results/dataset_v2_1_development_failed_visibility_20260824/protocol_lock_snapshot/dataset_v2_1_development_capture_plan.json",
    "datasets/roborefer_dataset_v2_1_development_400_failed_visibility_20260824/raw/raw_capture_manifest.json",
)

PREFIXES = (
    "Using only this synchronized RGB-D observation,",
    "In the present wrist-camera evidence,",
    "Within the currently observed tabletop scene,",
    "From the registered color and depth frame,",
    "Considering the visible scene evidence alone,",
    "For the current robot-camera observation,",
    "In this metric RGB-D workspace view,",
    "From the scene now visible to the wrist camera,",
    "Using the current color-depth evidence,",
    "Within this observed manipulation scene,",
    "In the camera frame available for this query,",
    "For this synchronized workspace observation,",
    "Based on the visible RGB-D frame,",
    "In the present registered sensor view,",
    "Using the observed tabletop configuration,",
    "From this evidence-limited camera view,",
    "Within the current metric-depth scene,",
    "Using the visual evidence in this frame,",
    "For the scene currently projected in the image,",
    "In this robot-mounted camera observation,",
    "From the current image and depth pair,",
    "Using this registered workspace frame,",
    "Within the camera evidence now available,",
    "For the presently visible object arrangement,",
)
SUFFIXES = (
    "Return the supported image-coordinate location.",
    "Report the corresponding point in image coordinates.",
    "Give the pixel location justified by the observation.",
    "Indicate its evidence-supported position in the image.",
    "Provide the target point in the camera frame.",
    "Respond with the localized image point.",
    "Mark the corresponding pixel position.",
    "Return one point supported by the visible evidence.",
    "State the requested referent's image location.",
    "Provide its projected position in the current frame.",
    "Output the matching point in image coordinates.",
    "Give the location supported by color and depth.",
    "Return the appropriate pixel in this observation.",
    "Report where the referent projects in the image.",
    "Indicate the localized point in the registered frame.",
    "Respond with the target's camera-image position.",
    "Provide a single evidence-backed image point.",
    "State the pixel that localizes the requested object.",
    "Return the image location warranted by the sensors.",
    "Give the corresponding location in the color frame.",
    "Mark its position using image coordinates.",
    "Output the referent location supported by this view.",
    "Report the localized point in the observed frame.",
    "Indicate one matching point in the camera image.",
)
MIDDLES = (
    "Do not assume evidence outside the frame.",
    "Use the spatial relation exactly as stated.",
    "Rely on the observed geometry when resolving the query.",
    "Treat this frame as the complete current observation.",
    "Use visible object and relation evidence.",
    "Resolve the referent from this scene only.",
    "Use the registered spatial evidence.",
    "Base the localization on the current frame.",
)


class V21GenerationError(RuntimeError):
    """Raised when the locked V2.1 design cannot be materialized exactly."""


def collect_values(value: Any, field_names: set[str]) -> set[Any]:
    found: set[Any] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in field_names and isinstance(child, (str, int)):
                found.add(child)
            found.update(collect_values(child, field_names))
    elif isinstance(value, list):
        for child in value:
            found.update(collect_values(child, field_names))
    return found


def collect_seed_values(value: Any, key: str = "") -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.update(collect_seed_values(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            found.update(collect_seed_values(child, key))
    elif isinstance(value, int) and "seed" in key.lower():
        found.add(int(value))
    return found


def load_denylists(workspace: Path) -> dict[str, set[Any]]:
    result: dict[str, set[Any]] = {
        "family_ids": set(), "capture_ids": set(), "seeds": set(),
        "instructions": set(), "layout_fingerprints": set(),
    }
    for relative in DENYLIST_JSON:
        path = workspace / relative
        if not path.is_file():
            continue
        payload = read_json(path)
        result["family_ids"].update(collect_values(payload, {"family_id"}))
        result["capture_ids"].update(collect_values(payload, {"capture_id"}))
        result["seeds"].update(collect_seed_values(payload))
        result["instructions"].update(collect_values(payload, {"instruction"}))
        result["layout_fingerprints"].update(
            collect_values(payload, {"layout_instance_fingerprint_sha256", "layout_fingerprint_sha256"})
        )
    return result


def derived_seed(namespace: str, index: int, denied: set[int], used: set[int]) -> int:
    nonce = 0
    while True:
        digest = hashlib.sha256(
            f"{PROTOCOL_ID}|{MASTER_SEED}|{namespace}|{index}|{nonce}".encode()
        ).digest()
        value = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        if value not in denied and value not in used:
            used.add(value)
            return value
        nonce += 1


def semantic_phrase(spec: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str:
    classes = [by_id[value]["semantic_class"] for value in spec["candidate_target_ids"]]
    unique = set(classes)
    if len(classes) > 1 and unique.issubset({"apple", "orange", "lemon", "mango"}):
        return "fruit"
    if len(unique) == 1:
        return classes[0]
    return "object"


def query_core(spec: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str:
    semantic = semantic_phrase(spec, by_id)
    relation = spec["relations"][0]
    anchors = [by_id[value]["semantic_class"] for value in spec["anchor_ids"]]
    if relation == "direct" or not anchors:
        return f"locate the {semantic}."
    if relation == "between_in_depth" and len(anchors) >= 2:
        return f"locate the {semantic} between the {anchors[0]} and {anchors[1]} in camera depth."
    if relation in {"nearer_than_both", "farther_than_both"} and len(anchors) >= 2:
        comparison = "nearer to" if relation == "nearer_than_both" else "farther from"
        return f"locate the {semantic} {comparison} the camera than both the {anchors[0]} and {anchors[1]}."
    return f"locate the {semantic} that is {relation.replace('_', ' ')} the {anchors[0]}."


def new_instruction(
    spec: dict[str, Any], by_id: dict[str, dict[str, Any]], sample_key: str,
    denied: set[str], used: set[str],
) -> tuple[str, str]:
    core = query_core(spec, by_id)
    total = len(PREFIXES) * len(MIDDLES) * len(SUFFIXES)
    start = int(hashlib.sha256(f"{MASTER_SEED}|{sample_key}".encode()).hexdigest(), 16) % total
    for offset in range(total):
        value = (start + offset) % total
        prefix = PREFIXES[value % len(PREFIXES)]
        value //= len(PREFIXES)
        middle = MIDDLES[value % len(MIDDLES)]
        suffix = SUFFIXES[(value // len(MIDDLES)) % len(SUFFIXES)]
        instruction = f"{prefix} {core} {middle} {suffix}"
        if instruction not in denied and instruction not in used:
            used.add(instruction)
            digest = hashlib.sha256(instruction.encode()).hexdigest()
            return instruction, f"v21_development_template_{digest[:20]}"
    raise V21GenerationError(f"instruction namespace exhausted for {sample_key}")


def margin_for(relation: str) -> float:
    return BETWEEN_DEPTH_SCREENING_MARGIN_M if relation == "between_in_depth" else PREDICTOR_SCREENING_MARGIN_M


def evaluate_spec_geometry(
    spec: dict[str, Any], poses_by_id: dict[str, list[float]], by_id: dict[str, dict[str, Any]],
    id_to_model: dict[str, str], calibration: dict[str, Any], camera_tf: dict[str, Any],
    intrinsics: dict[str, float],
) -> list[dict[str, Any]]:
    relation = spec["relations"][0]
    if relation == "direct":
        return []
    anchors = list(spec["anchor_ids"])
    if not anchors or not all(value in poses_by_id for value in anchors):
        return []
    targets = (
        list(spec["candidate_target_ids"])
        if spec["answerability_state"] == "ABSENT" and spec["state_submode"] == "unsatisfied_relation"
        else list(spec["valid_target_ids"])
    )
    if not targets or not all(value in poses_by_id for value in targets):
        return []
    anchor_models = [id_to_model[value] for value in anchors]
    results = []
    for target_id in targets:
        target_model = id_to_model[target_id]
        evidence = predicted_relation(
            relation,
            poses_by_id[target_id], by_id[target_id], calibration[target_model],
            [poses_by_id[value] for value in anchors],
            [by_id[value] for value in anchors],
            [calibration[value] for value in anchor_models],
            camera_tf, intrinsics, depth_margin_m=margin_for(relation),
        )
        expected = "UNSATISFIED" if spec["state_submode"] == "unsatisfied_relation" else "SATISFIED"
        results.append({
            "variant": spec["variant"], "target_id": target_id, "relation": relation,
            "expected": expected, "passed": bool(evidence["passed"]), "evidence": evidence,
        })
    return results


def geometry_constraints_pass(results: list[dict[str, Any]]) -> bool:
    for row in results:
        if row["expected"] == "SATISFIED" and not row["passed"]:
            return False
        if row["expected"] == "UNSATISFIED":
            if row["passed"] or float(row["evidence"]["signed_screening_margin"]) > -0.005:
                return False
    return True


def candidate_pose_set(seed: int, object_ids: list[str]) -> dict[str, list[float]]:
    rng = random.Random(seed)
    return {
        object_id: [
            round(rng.uniform(-0.40, -0.14), 6),
            round(rng.uniform(0.08, 0.58), 6),
            round(rng.uniform(-math.pi, math.pi), 6),
        ]
        for object_id in object_ids
    }


def predicted_required_visibility_clear(
    poses_by_id: dict[str, list[float]], by_id: dict[str, dict[str, Any]],
    camera_tf: dict[str, Any], intrinsics: dict[str, float],
) -> bool:
    """Reject deterministic candidates where a nearer asset covers a required asset.

    The original V2.1 predictor checked field-of-view and physical collision but
    not image-plane inter-object occlusion.  This conservative static test uses
    only preregistered geometry and camera calibration; it never reads a capture.
    """

    projected = {
        object_id: projection(pose, by_id[object_id], camera_tf, intrinsics)
        for object_id, pose in poses_by_id.items()
    }
    for far_id, far in projected.items():
        far_radius = max(4.0, 0.25 * max(float(far.get("pad_px", 18.0)) - 18.0, 0.0))
        for near_id, near in projected.items():
            if near_id == far_id or float(near["depth_m"]) >= float(far["depth_m"]) - 0.010:
                continue
            near_radius = max(4.0, 0.78 * max(float(near.get("pad_px", 18.0)) - 18.0, 0.0))
            separation = math.hypot(
                float(near.get("u_px", 0.0)) - float(far.get("u_px", 0.0)),
                float(near.get("v_px", 0.0)) - float(far.get("v_px", 0.0)),
            )
            if separation + far_radius <= near_radius:
                return False
    return True


def bias_primary_depth_geometry(
    poses: dict[str, list[float]], family: dict[str, Any], seed: int,
) -> None:
    """Increase acceptance rate without observing any captured evidence."""

    scene = family["primary_scene"]
    relation = scene["relation"]
    targets = [value for value in scene["candidate_target_ids"] if value in poses]
    anchors = [value for value in scene["anchor_ids"] if value in poses]
    if not targets or not anchors or relation in {"direct", "left_of", "right_of"}:
        return
    unsatisfied = family["state_submode"] == "unsatisfied_relation"
    rng = random.Random(seed ^ 0x21D3)

    def assign(ids: list[str], levels: list[float]) -> None:
        for position, object_id in enumerate(ids):
            base = levels[min(position, len(levels) - 1)]
            poses[object_id][1] = round(base + rng.uniform(-0.012, 0.012), 6)

    if relation == "between_in_depth":
        if unsatisfied:
            assign(anchors, [0.31, 0.43])
            assign(targets, [0.56, 0.57])
        else:
            assign(anchors, [0.30, 0.58])
            assign(targets, [0.41, 0.47])
    elif relation == "nearer_than_both":
        if unsatisfied:
            assign(targets, [0.56, 0.57])
            assign(anchors, [0.31, 0.42])
        else:
            assign(targets, [0.31, 0.36])
            assign(anchors, [0.51, 0.58])
    elif relation in {"nearer_than", "front_of"}:
        assign(targets, [0.56] if unsatisfied else [0.32, 0.35])
        assign(anchors, [0.32] if unsatisfied else [0.56])
    elif relation in {"farther_than", "behind"}:
        assign(targets, [0.32] if unsatisfied else [0.54, 0.58])
        assign(anchors, [0.56] if unsatisfied else [0.32])


def calibrate_between_targets(
    poses: dict[str, list[float]], family: dict[str, Any],
    by_id: dict[str, dict[str, Any]], id_to_model: dict[str, str],
    calibration: dict[str, Any], camera_tf: dict[str, Any],
) -> None:
    scene = family["primary_scene"]
    if scene["relation"] != "between_in_depth" or family["state_submode"] == "unsatisfied_relation":
        return
    anchors = [value for value in scene["anchor_ids"] if value in poses]
    targets = [value for value in scene["candidate_target_ids"] if value in poses]
    if len(anchors) < 2 or not targets:
        return
    depths = [
        predict_median_depth(
            poses[value], by_id[value], camera_tf, calibration[id_to_model[value]]
        )
        for value in anchors
    ]
    desired = 0.5 * (min(depths) + max(depths))
    for value in targets:
        pose = poses[value]
        current = predict_median_depth(
            pose, by_id[value], camera_tf, calibration[id_to_model[value]]
        )
        shifted = [pose[0], pose[1] + 0.01, pose[2]]
        slope = (
            predict_median_depth(
                shifted, by_id[value], camera_tf, calibration[id_to_model[value]]
            ) - current
        ) / 0.01
        if abs(slope) > 1e-9:
            pose[1] = round(float(pose[1]) + (desired - current) / slope, 6)


def place_hidden_objects(
    seed: int, hidden_ids: list[str], placed: dict[str, list[float]], by_id: dict[str, dict[str, Any]],
    id_to_model: dict[str, str], camera_tf: dict[str, Any], intrinsics: dict[str, float],
) -> dict[str, list[float]] | None:
    output = dict(placed)
    rng = random.Random(seed)
    for object_id in hidden_ids:
        row = by_id[object_id]
        accepted = None
        for _ in range(4096):
            pose = [
                round(rng.uniform(0.28, 0.56), 6),
                round(rng.uniform(0.08, 0.62), 6),
                round(rng.uniform(-math.pi, math.pi), 6),
            ]
            trial_ids = [*output, object_id]
            trial_poses = [*output.values(), pose]
            trial_rows = [by_id[value] for value in output] + [row]
            if not collision_free(trial_poses, trial_rows):
                continue
            if projection(pose, row, camera_tf, intrinsics)["visible"]:
                continue
            accepted = pose
            break
        if accepted is None:
            return None
        output[object_id] = accepted
    return output


def full_storage_layout(registry: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    return {name: list(row["storage_pose_xyyaw"]) for name, row in registry.items()}


def occluder_pose(
    target_pose: list[float], target_row: dict[str, Any], occluder_row: dict[str, Any],
    camera_tf: dict[str, Any], yaw_seed: int,
) -> list[float]:
    camera_x, camera_y = [float(value) for value in camera_tf["position"][:2]]
    dx, dy = camera_x - float(target_pose[0]), camera_y - float(target_pose[1])
    norm = max(math.hypot(dx, dy), 1e-9)
    separation = (
        float(target_row["footprint_radius_m"])
        + float(occluder_row["footprint_radius_m"]) + 0.022
    )
    return [
        round(float(target_pose[0]) + separation * dx / norm, 6),
        round(float(target_pose[1]) + separation * dy / norm, 6),
        round(((yaw_seed % 6283185) / 1_000_000.0) - math.pi, 6),
    ]


def place_occluder_layout(
    target_id: str, occluder_id: str, placed: dict[str, list[float]],
    by_id: dict[str, dict[str, Any]], camera_tf: dict[str, Any],
    intrinsics: dict[str, float], seed: int,
) -> list[float] | None:
    """Choose a deterministic, visible, collision-free pose near the target ray."""

    target = placed[target_id]
    target_row, occluder_row = by_id[target_id], by_id[occluder_id]
    base = occluder_pose(target, target_row, occluder_row, camera_tf, seed)
    camera_x, camera_y = [float(value) for value in camera_tf["position"][:2]]
    angle = math.atan2(camera_y - float(target[1]), camera_x - float(target[0]))
    minimum = (
        float(target_row["footprint_radius_m"])
        + float(occluder_row["footprint_radius_m"]) + 0.022
    )
    offsets = (0.0, 0.10, -0.10, 0.20, -0.20, 0.32, -0.32, 0.48, -0.48)
    separations = (minimum, minimum + 0.015, minimum + 0.030, minimum + 0.050)
    other_ids = [value for value in placed if value != occluder_id]
    other_poses = [placed[value] for value in other_ids]
    other_rows = [by_id[value] for value in other_ids]
    for separation in separations:
        for offset in offsets:
            proposed = [
                round(float(target[0]) + separation * math.cos(angle + offset), 6),
                round(float(target[1]) + separation * math.sin(angle + offset), 6),
                base[2],
            ]
            if not collision_free([*other_poses, proposed], [*other_rows, occluder_row]):
                continue
            if not projection(proposed, occluder_row, camera_tf, intrinsics)["visible"]:
                continue
            trial = {value: placed[value] for value in other_ids}
            trial[occluder_id] = proposed
            if not predicted_required_visibility_clear(trial, by_id, camera_tf, intrinsics):
                continue
            return proposed
    rng = random.Random(seed ^ 0x5A21)
    for _ in range(4096):
        proposed = [
            round(rng.uniform(-0.40, -0.14), 6),
            round(rng.uniform(0.08, 0.58), 6),
            round(rng.uniform(-math.pi, math.pi), 6),
        ]
        trial = {value: placed[value] for value in other_ids}
        trial[occluder_id] = proposed
        if (
            collision_free([*other_poses, proposed], [*other_rows, occluder_row])
            and projection(proposed, occluder_row, camera_tf, intrinsics)["visible"]
            and predicted_required_visibility_clear(trial, by_id, camera_tf, intrinsics)
        ):
            return proposed
    return None


def select_family_layouts(
    index: int, family: dict[str, Any], specs: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]], calibration: dict[str, Any],
    camera_tf: dict[str, Any], intrinsics: dict[str, float], denied_seeds: set[int],
    used_seeds: set[int], used_fingerprints: set[str],
) -> tuple[dict[str, list[float]], dict[str, list[float]], int, list[dict[str, Any]]]:
    by_id = {row["id"]: row for row in registry.values()}
    id_to_model = {row["id"]: name for name, row in registry.items()}
    scene = family["primary_scene"]
    active_ids = list(scene["active_scene_ids"])
    hidden_ids = (
        list(scene["candidate_target_ids"])
        if family["state_submode"] == "too_small_or_out_of_view" else []
    )
    visible_ids = sorted(value for value in active_ids if value not in set(hidden_ids))
    clean_specs = [value for value in specs if value["capture_id"].endswith("__clean_capture")]

    chosen: tuple[dict[str, list[float]], int, list[dict[str, Any]]] | None = None
    for attempt in range(LAYOUT_CANDIDATE_LIMIT):
        candidate_seed = derived_seed(f"layout_candidate_{attempt}", index, denied_seeds, used_seeds)
        visible = candidate_pose_set(candidate_seed, visible_ids)
        bias_primary_depth_geometry(visible, family, candidate_seed)
        calibrate_between_targets(
            visible, family, by_id, id_to_model, calibration, camera_tf
        )
        poses = list(visible.values())
        rows = [by_id[value] for value in visible_ids]
        if not collision_free(poses, rows):
            continue
        if not all(
            projection(visible[value], by_id[value], camera_tf, intrinsics)["visible"]
            for value in visible_ids
        ):
            continue
        if not predicted_required_visibility_clear(visible, by_id, camera_tf, intrinsics):
            continue
        placed = place_hidden_objects(
            candidate_seed ^ MASTER_SEED, hidden_ids, visible, by_id, id_to_model,
            camera_tf, intrinsics,
        )
        if placed is None:
            continue
        geometry_rows = [
            row for spec in clean_specs
            for row in evaluate_spec_geometry(
                spec, visible, by_id, id_to_model, calibration, camera_tf, intrinsics
            )
        ]
        if not geometry_constraints_pass(geometry_rows):
            continue
        fingerprint = canonical_json_sha256({key: placed[key] for key in sorted(placed)})
        if fingerprint in used_fingerprints:
            continue
        used_fingerprints.add(fingerprint)
        chosen = (placed, candidate_seed, geometry_rows)
        break
    if chosen is None:
        raise V21GenerationError(f"candidate pool exhausted for {family['family_id']}")

    placed, layout_seed, geometry_rows = chosen
    clean = full_storage_layout(registry)
    for object_id, pose in placed.items():
        clean[id_to_model[object_id]] = pose

    occlusion = copy.deepcopy(clean)
    target_ids = [value for value in scene["candidate_target_ids"] if value in placed]
    if not target_ids:
        semantic = [value for value in placed if value not in set(scene["anchor_ids"])]
        target_ids = semantic[:1]
    occluder_id = scene["occluder_id"]
    if target_ids and occluder_id not in set(scene["candidate_target_ids"] + scene["anchor_ids"]):
        target_id = target_ids[0]
        target_model, occluder_model = id_to_model[target_id], id_to_model[occluder_id]
        proposed = place_occluder_layout(
            target_id, occluder_id, placed, by_id, camera_tf, intrinsics, layout_seed
        )
        if proposed is not None:
            occlusion_trial = dict(placed)
            occlusion_trial[occluder_id] = proposed
            if not predicted_required_visibility_clear(
                occlusion_trial, by_id, camera_tf, intrinsics
            ):
                proposed = None
        if proposed is not None:
            occlusion[occluder_model] = proposed
            if family["state_submode"] == "occlusion":
                clean[occluder_model] = proposed
                occlusion[occluder_model] = [
                    proposed[0], proposed[1], round(float(proposed[2]) + 0.07, 6)
                ]

    clean_active = {name: pose for name, pose in clean.items() if float(pose[1]) <= .65}
    occlusion_active = {name: pose for name, pose in occlusion.items() if float(pose[1]) <= .65}
    for active in (clean_active, occlusion_active):
        fingerprint = canonical_json_sha256({key: active[key] for key in sorted(active)})
        if fingerprint in used_fingerprints and active is occlusion_active:
            # The clean fingerprint was already reserved above; only exact reuse
            # between the two capture conditions is forbidden here.
            if canonical_json_sha256(clean_active) == fingerprint:
                raise V21GenerationError(f"clean/occlusion layout identical: {family['family_id']}")
        used_fingerprints.add(fingerprint)
    return clean, occlusion, layout_seed, geometry_rows


def required_visible_ids(family: dict[str, Any], specs: list[dict[str, Any]], condition: str) -> list[str]:
    active = set(family["primary_scene"]["active_scene_ids"])
    if family["state_submode"] == "too_small_or_out_of_view":
        active.difference_update(family["primary_scene"]["candidate_target_ids"])
    if condition == "occlusion":
        active.add(family["primary_scene"]["occluder_id"])
    return sorted(active)


def allocate_batches(families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canary_targets = {
        ("train", "FOUND"): 7, ("train", "INSUFFICIENT_EVIDENCE"): 3,
        ("train", "AMBIGUOUS"): 3, ("train", "ABSENT"): 3,
        ("dev", "FOUND"): 1, ("dev", "INSUFFICIENT_EVIDENCE"): 1,
        ("dev", "AMBIGUOUS"): 1, ("dev", "ABSENT"): 1,
    }
    selected: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    for key, amount in canary_targets.items():
        split, state = key
        candidates = [
            item for item in families
            if item["split"] == split and item["primary_answerability_stratum"] == state
        ]
        for _ in range(amount):
            available = [item for item in candidates if item not in selected]
            available.sort(key=lambda item: (
                categories[item["family_category"]],
                hashlib.sha256(f"{MASTER_SEED}|canary|{item['family_id']}".encode()).hexdigest(),
            ))
            if not available:
                raise V21GenerationError(f"canary allocation exhausted for {key}")
            selected.append(available[0])
            categories[available[0]["family_category"]] += 1
    if len(selected) != 20 or len(categories) != 6:
        raise V21GenerationError(f"canary does not cover all six categories: {dict(categories)}")
    selected_ids = {item["family_id"] for item in selected}
    remainder = sorted(
        [item for item in families if item["family_id"] not in selected_ids],
        key=lambda item: hashlib.sha256(
            f"{MASTER_SEED}|post_canary|{item['family_id']}".encode()
        ).hexdigest(),
    )
    chunks = [selected] + [remainder[index:index + 95] for index in range(0, 380, 95)]
    if [len(value) for value in chunks] != [20, 95, 95, 95, 95]:
        raise V21GenerationError("batch sizes differ from 20 + 95 x 4")
    output = []
    for order, chunk in enumerate(chunks):
        batch_id = "canary_000" if order == 0 else f"batch_{order:03d}"
        output.append({
            "batch_id": batch_id,
            "order": order,
            "family_count": len(chunk),
            "planned_capture_count": 2 * len(chunk),
            "family_ids": [item["family_id"] for item in chunk],
            "split_counts": dict(sorted(Counter(item["split"] for item in chunk).items())),
            "primary_answerability_counts": dict(sorted(Counter(
                item["primary_answerability_stratum"] for item in chunk
            ).items())),
            "family_category_counts": dict(sorted(Counter(
                item["family_category"] for item in chunk
            ).items())),
            "prerequisite_batch": None if order == 0 else (
                "canary_000" if order == 1 else f"batch_{order - 1:03d}"
            ),
            "next_batch_requires_raw_qc_pass": True,
        })
    return output


def counts_by_split(families: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    return {
        split: dict(sorted(Counter(str(item[field]) for item in families if item["split"] == split).items()))
        for split in ("train", "dev")
    }


def build_artifacts(workspace: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    old_manifest_path = workspace / V2_MANIFEST
    old_plan_path = workspace / V2_PLAN
    old_manifest = read_json(old_manifest_path)
    old_plan = read_json(old_plan_path)
    partition = read_json(workspace / PARTITION)
    camera_lock = read_json(workspace / CAMERA_LOCK)
    calibration_artifact = read_json(workspace / CALIBRATION)
    repair_view = camera_lock["repair_view"]
    camera_tf = repair_view["fk_transform_base_to_camera"]
    intrinsics = camera_lock["intrinsics"]
    calibration = calibration_artifact["asset_calibration"]
    registry = object_registry()
    by_id = {row["id"]: row for row in registry.values()}
    if old_manifest.get("family_count") != FAMILY_COUNT:
        raise V21GenerationError("V2 quota template is not the locked 400-family composition")

    denylists = load_denylists(workspace)
    used_seeds: set[int] = set()
    used_instructions: set[str] = set()
    used_fingerprints: set[str] = set(denylists["layout_fingerprints"])
    families: list[dict[str, Any]] = []
    captures_by_family: dict[str, list[dict[str, Any]]] = {}

    for index, old in enumerate(old_manifest["families"], start=1):
        family_id = f"{FAMILY_PREFIX}{index:06d}"
        if family_id in denylists["family_ids"]:
            raise V21GenerationError(f"new family ID collides with denylist: {family_id}")
        seed_bundle = {
            key: derived_seed(key, index, denylists["seeds"], used_seeds)
            for key in ("layout", "sensor", "occlusion", "language", "counterfactual")
        }
        specs = copy.deepcopy(old["variant_specs"])
        for spec in specs:
            condition = "occlusion" if spec["variant"] == "occlusion_view_counterfactual" else "clean"
            spec["capture_id"] = f"{family_id}__{condition}_capture"
            instruction, template_id = new_instruction(
                spec, by_id, f"{family_id}__{spec['variant']}",
                denylists["instructions"], used_instructions,
            )
            spec["instruction"] = instruction
            spec["instruction_sha256"] = hashlib.sha256(instruction.encode()).hexdigest()
            spec["language_template_family_id"] = template_id
        family = {
            "family_id": family_id,
            "split": old["split"],
            "primary_answerability_stratum": old["primary_answerability_stratum"],
            "state_submode": old["state_submode"],
            "family_category": old["family_category"],
            "depth_dependent": bool(old["depth_dependent"]),
            "asset_pool": "seen",
            "ood_axis": "none",
            "layout_generator_id": "layout_v2_1_observable_relation_iid",
            "camera_bin_id": repair_view["camera_bin_id"],
            "language_template_bank_id": "language_v2_1_iid",
            "depth_noise_generator_id": "depth_noise_v2_1_iid",
            "seed_bundle": seed_bundle,
            "eligible_for_official_dataset": True,
            "capture_policy": "OFFICIAL_V2_1_DEVELOPMENT_TRAIN_OR_DEV_ONLY",
            "all_prior_capture_reuse_forbidden": True,
            "primary_scene": copy.deepcopy(old["primary_scene"]),
            "variants": list(VARIANTS),
            "variant_specs": specs,
        }
        family["primary_scene"]["instruction"] = specs[0]["instruction"]
        clean, occlusion, layout_seed, geometry_rows = select_family_layouts(
            index, family, specs, registry, calibration, camera_tf, intrinsics,
            denylists["seeds"], used_seeds, used_fingerprints,
        )
        family["selected_layout_seed"] = layout_seed
        family["observable_geometry_screening"] = {
            "geometry_version": GEOMETRY_VERSION,
            "camera_bin_id": repair_view["camera_bin_id"],
            "selection_policy": "FIRST_DETERMINISTIC_CANDIDATE_PASSING_V2_1_RELATION_AND_VISIBILITY_PREDICTOR",
            "predictor_rows": geometry_rows,
        }
        families.append(family)
        family_captures = []
        for condition, layout, seed_key in (
            ("clean", clean, "sensor"), ("occlusion", occlusion, "occlusion")
        ):
            capture_id = f"{family_id}__{condition}_capture"
            if capture_id in denylists["capture_ids"]:
                raise V21GenerationError(f"new capture ID collides with denylist: {capture_id}")
            required = required_visible_ids(family, specs, condition)
            active = {name: pose for name, pose in layout.items() if float(pose[1]) <= .65}
            family_captures.append({
                "capture_id": capture_id,
                "family_id": family_id,
                "split": family["split"],
                "condition": condition,
                "seed": int(seed_bundle[seed_key]),
                "layout_seed": int(layout_seed),
                "ood_axis": "none",
                "camera_bin_id": repair_view["camera_bin_id"],
                "view_joint_pose": list(repair_view["requested_view_joint_pose"]),
                "layout_generator_id": family["layout_generator_id"],
                "layout_instance_fingerprint_sha256": canonical_json_sha256(
                    {key: active[key] for key in sorted(active)}
                ),
                "layout": layout,
                "required_visible_object_ids": required,
                "required_visible_label_ids": sorted(int(by_id[value]["label"]) for value in required),
            })
        captures_by_family[family_id] = family_captures

    batches = allocate_batches(families)
    batch_by_family = {
        family_id: batch["batch_id"] for batch in batches for family_id in batch["family_ids"]
    }
    captures = []
    for batch in batches:
        for family_id in batch["family_ids"]:
            for capture in captures_by_family[family_id]:
                capture["batch_id"] = batch_by_family[family_id]
                captures.append(capture)

    assignment_rows = [
        {
            "family_id": item["family_id"], "split": item["split"],
            "state": item["primary_answerability_stratum"],
            "category": item["family_category"], "submode": item["state_submode"],
            "seed_bundle": item["seed_bundle"],
        }
        for item in families
    ]
    counts = {
        "split": dict(sorted(Counter(item["split"] for item in families).items())),
        "primary_answerability_by_split": counts_by_split(families, "primary_answerability_stratum"),
        "family_category_by_split": counts_by_split(families, "family_category"),
        "ood_axis_by_split": counts_by_split(families, "ood_axis"),
        "state_submode_by_split": counts_by_split(families, "state_submode"),
    }
    source_hashes = {
        relative: sha256_file(workspace / relative)
        for relative in (V2_MANIFEST, V2_PLAN, PARTITION, CAMERA_LOCK, CALIBRATION)
    }
    denylist_hashes = {
        relative: sha256_file(workspace / relative)
        for relative in DENYLIST_JSON if (workspace / relative).is_file()
    }
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "status": "STATIC_V2_1_DEVELOPMENT_PLAN_NOT_CAPTURED",
        "scientific_status": "PLANNED_NOT_CAPTURED",
        "master_seed": MASTER_SEED,
        "eligible_for_official_development": True,
        "allowed_splits": ["train", "dev"],
        "sealed_splits_not_created": ["calibration", "test_iid", "test_ood"],
        "family_count": len(families),
        "sample_count": 5 * len(families),
        "variant_count_per_family": 5,
        "v2_1_assignment_commitment_sha256": canonical_json_sha256(assignment_rows),
        "v2_quota_template_only": True,
        "all_v2_id_seed_instruction_layout_capture_reuse_forbidden": True,
        "source_artifact_sha256": source_hashes,
        "denylist_artifact_sha256": denylist_hashes,
        "asset_partition_sha256": sha256_file(workspace / PARTITION),
        "counts": counts,
        "batch_assignments": batches,
        "training_performed": False,
        "calibration_or_test_created": False,
        "families": families,
    }
    world = workspace / WORLD_FILE
    plan = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "status": "STATIC_PLAN_CAPTURE_FORBIDDEN_UNTIL_PREFLIGHT_PASS",
        "scientific_status": "PLANNED_NOT_CAPTURED",
        "capture_output_root": OUTPUT_ROOT,
        "world_file": WORLD_FILE,
        "world_sha256": sha256_file(world),
        "family_count": len(families),
        "sample_count_after_materialization": 5 * len(families),
        "capture_count": len(captures),
        "captures_per_family": 2,
        "batch_execution_order": [item["batch_id"] for item in batches],
        "batches": batches,
        "resume_contract": {
            "identity_key": "capture_id",
            "write_mode": "ATOMIC_CREATE_ONLY",
            "existing_complete_capture": "VERIFY_ALL_DECLARED_HASHES_THEN_SKIP",
            "existing_partial_or_corrupt_capture": "FAIL_NEVER_OVERWRITE",
            "checkpoint_after_each_family": True,
            "checkpoint_at_batch_boundary": True,
            "next_batch_requires_previous_raw_qc_pass": True,
        },
        "sensor_contract": {
            "rgb_topic": "/wrist_camera/color/image_raw",
            "depth_topic": "/wrist_camera/depth/image_raw",
            "semantic_label_topic": "/wrist_camera/evaluation_labels/labels_map",
            "max_timestamp_spread_sec": 0.035,
            "metric_depth_unit": "metre",
            "required_capture_artifacts": [
                "rgb_original.png", "depth_metric.npy", "semantic_instance_labels.png",
                "camera_info.json", "tf_snapshot.json", "capture_meta.json",
            ],
        },
        "camera_poses": {
            repair_view["camera_bin_id"]: list(repair_view["requested_view_joint_pose"])
        },
        "camera_predictor_transform": camera_tf,
        "camera_intrinsics": intrinsics,
        "observable_relation_contract": {
            "geometry_version": GEOMETRY_VERSION,
            "label_source": "synchronized clean-capture semantic-mask centroid and robust median metric depth",
            "runtime_depth_margin_m": 0.020,
            "predictor_screening_margin_m": PREDICTOR_SCREENING_MARGIN_M,
            "between_depth_predictor_screening_margin_m": BETWEEN_DEPTH_SCREENING_MARGIN_M,
            "horizontal_runtime_margin_px": 12.0,
            "simulator_object_center_role": "SECONDARY_STATIC_SCREENING_ONLY",
        },
        "ordered_shutdown_contract": {
            "qualified_gate_decision": "PASS_V2_1_SHUTDOWN_GATE",
            "launcher": "protocol/dataset_v2_1_development_capture.launch.py",
            "coordinator": "protocol/dataset_v2_relation_repair_shutdown.py",
            "sequence": ["capture_node", "action_server_and_servo", "move_group", "gazebo"],
            "baseline_launch_files_modified": False,
        },
        "workspace_roi_base_link": {"x_m": [-0.60, 0.60], "y_m": [0.06, 0.65]},
        "inference_payload_template": {
            "rgb_model_input": "TO_BE_CAPTURED",
            "depth_relative_model_input": "TO_BE_DERIVED",
            "enable_depth": True,
            "prompt": "{instruction} " + COORDINATE_SUFFIX,
            "coordinate_suffix": COORDINATE_SUFFIX,
        },
        "object_registry": registry,
        "training_authorized": False,
        "calibration_test_authorized": False,
        "manifest_sha256_pending": True,
        "captures": captures,
    }
    return manifest, plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    protocol = workspace / "protocol"
    manifest, plan = build_artifacts(workspace)
    manifest_path = protocol / "dataset_v2_1_development_manifest.json"
    plan_path = protocol / "dataset_v2_1_development_capture_plan.json"
    write_json(manifest_path, manifest)
    plan["manifest_sha256"] = sha256_file(manifest_path)
    plan.pop("manifest_sha256_pending", None)
    write_json(plan_path, plan)
    print(
        "DATASET_V2_1_DEVELOPMENT_STATIC_MATERIALIZED "
        f"families={manifest['family_count']} samples={manifest['sample_count']} "
        f"captures={plan['capture_count']} batches="
        f"{[item['family_count'] for item in plan['batches']]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
