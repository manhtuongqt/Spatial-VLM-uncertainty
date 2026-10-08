#!/usr/bin/env python3
"""Materialize the locked 400-family Dataset V2 development capture design.

No Gazebo process, sensor, model or training code is invoked here.  The scene
builder is imported from the already qualified pilot implementation, but every
development identity, seed, instruction, layout instance and capture is new.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2_pilot_generator import (  # noqa: E402
    COORDINATE_SUFFIX,
    IID_VIEW,
    PilotGenerationError,
    object_registry,
    place_safe_occluder,
    primary_scene,
    variant_specs,
)
from validate_dataset_expansion_v2 import (  # noqa: E402
    VARIANTS,
    assignment_commitment,
    build_development_assignments,
)
from wp2_common import canonical_json_sha256, read_json, sha256_file, write_json  # noqa: E402


PROTOCOL_ID = "roborefer_dataset_v2_development_capture_400"
GENERATOR_VERSION = "dataset_v2_development_generator_v1.0.0"
EXPECTED_DEVELOPMENT_COMMITMENT = "d6790084dab886bbca226c892256e0ee4ca86f9a51c6970b9c2c03b99bcbc2a2"
PILOT_ROOT = "datasets/roborefer_dataset_v2_pilot_only_20260821"

SUBMODE_COUNTS_BY_SPLIT = {
    "train": {
        "AMBIGUOUS": {
            "same_class_duplicate": 22,
            "attribute_tie": 14,
            "relation_tie": 11,
            "multi_anchor_conflict": 9,
        },
        "ABSENT": {"target_absent": 19, "anchor_absent": 19, "unsatisfied_relation": 18},
        "INSUFFICIENT_EVIDENCE": {
            "depth_invalid_or_corrupt": 24,
            "occlusion": 24,
            "too_small_or_out_of_view": 16,
            "cross_modal_conflict": 16,
        },
    },
    "dev": {
        "AMBIGUOUS": {
            "same_class_duplicate": 6,
            "attribute_tie": 4,
            "relation_tie": 3,
            "multi_anchor_conflict": 1,
        },
        "ABSENT": {"target_absent": 5, "anchor_absent": 5, "unsatisfied_relation": 4},
        "INSUFFICIENT_EVIDENCE": {
            "depth_invalid_or_corrupt": 6,
            "occlusion": 6,
            "too_small_or_out_of_view": 4,
            "cross_modal_conflict": 4,
        },
    },
}

FRUIT_IDS = (
    "ycb_apple_01", "ycb_apple_02", "ycb_apple_03",
    "ycb_orange_01", "ycb_orange_02", "ycb_orange_03",
    "ycb_lemon_01", "mango_01",
)
FRUIT_CLASSES = {"apple", "orange", "lemon", "mango"}

PREFIXES = (
    "In the current camera view,", "Using the present RGB-D observation,",
    "Within the visible tabletop scene,", "From this wrist-camera frame,",
    "Considering only the current observation,", "In this registered color-depth frame,",
    "For the object arrangement now visible,", "Using the scene shown by the wrist camera,",
    "In the observed manipulation workspace,", "From the robot's current visual frame,",
    "Within this synchronized RGB-D view,", "For the tabletop configuration in view,",
    "In the camera evidence currently available,", "Using the displayed workspace observation,",
    "For this particular object arrangement,", "Within the camera's present field of view,",
)
SUFFIXES = (
    "Return its image location.", "Report the corresponding pixel location.",
    "Give the target point in the image.", "Respond with the target's pixel position.",
    "Indicate where it projects in the image.", "Provide one image-coordinate target point.",
    "Mark the location supported by the observation.", "Output the matching image point.",
    "Identify its location in image coordinates.", "Return the appropriate point in the frame.",
    "State the pixel that localizes the requested referent.", "Give the point supported by the visual evidence.",
    "Respond using an image-coordinate point.", "Provide the localized point in the camera frame.",
    "Indicate the requested referent's image position.", "Return the evidence-supported target pixel.",
)


class DevelopmentGenerationError(RuntimeError):
    """Raised when the locked development artifacts cannot be generated exactly."""


def ranked(items: list[dict[str, Any]], seed: int, namespace: str) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            hashlib.sha256(f"{seed}|{namespace}|{item['family_id']}".encode()).hexdigest(),
            item["family_id"],
        ),
    )


def compatible_submode(assignment: dict[str, Any], submode: str) -> bool:
    category = assignment["family_category"]
    if submode == "multi_anchor_conflict":
        return category == "multi_anchor_depth_order"
    if submode in {"relation_tie", "anchor_absent", "unsatisfied_relation"}:
        return category != "direct_grounding"
    return True


def allocate_submodes(assignments: list[dict[str, Any]], master_seed: int) -> dict[str, str]:
    result: dict[str, str] = {}
    priority = {
        "AMBIGUOUS": ("multi_anchor_conflict", "relation_tie", "attribute_tie", "same_class_duplicate"),
        "ABSENT": ("unsatisfied_relation", "anchor_absent", "target_absent"),
        "INSUFFICIENT_EVIDENCE": (
            "depth_invalid_or_corrupt", "occlusion", "too_small_or_out_of_view", "cross_modal_conflict"
        ),
    }
    for split, state_plans in SUBMODE_COUNTS_BY_SPLIT.items():
        for state, counts in state_plans.items():
            available = [
                item for item in assignments
                if item["split"] == split and item["primary_answerability_stratum"] == state
            ]
            if sum(counts.values()) != len(available):
                raise DevelopmentGenerationError(f"submode total differs for {split}/{state}")
            unassigned = {item["family_id"]: item for item in available}
            for submode in priority[state]:
                eligible = [item for item in unassigned.values() if compatible_submode(item, submode)]
                ordered = ranked(eligible, master_seed, f"development_submode|{split}|{state}|{submode}")
                take = int(counts[submode])
                if len(ordered) < take:
                    raise DevelopmentGenerationError(
                        f"not enough compatible families for {split}/{state}/{submode}: {len(ordered)} < {take}"
                    )
                for item in ordered[:take]:
                    result[item["family_id"]] = submode
                    del unassigned[item["family_id"]]
            if unassigned:
                raise DevelopmentGenerationError(f"unassigned submodes: {split}/{state}/{sorted(unassigned)}")
    for item in assignments:
        if item["primary_answerability_stratum"] == "FOUND":
            result[item["family_id"]] = "unique_visible_referent"
    return result


def normalize_ambiguous_scene(
    scene: dict[str, Any], submode: str, family_index: int, registry: dict[str, dict[str, Any]]
) -> None:
    """Ensure every ambiguous candidate truly matches one shared semantic query."""

    if submode == "same_class_duplicate":
        return
    by_id = {value["id"]: value for value in registry.values()}
    id_to_model = {value["id"]: name for name, value in registry.items()}
    old_candidates = list(scene["candidate_target_ids"])
    used = set(scene["anchor_ids"]) | {scene["semantic_cf_id"], scene["occluder_id"]}
    ordered_fruit = sorted(
        [value for value in FRUIT_IDS if value not in used],
        key=lambda value: hashlib.sha256(f"ambiguous-fruit|{family_index}|{value}".encode()).hexdigest(),
    )
    new_candidates = ordered_fruit[:2]
    if len(new_candidates) != 2:
        raise DevelopmentGenerationError("cannot allocate two semantically compatible ambiguous candidates")

    for layout_name in ("clean_layout", "occlusion_layout"):
        layout = scene[layout_name]
        old_poses = [copy.deepcopy(layout[id_to_model[value]]) for value in old_candidates]
        for value in old_candidates:
            if value not in new_candidates:
                layout[id_to_model[value]] = list(by_id[value]["storage_pose_xyyaw"])
        for index, value in enumerate(new_candidates):
            layout[id_to_model[value]] = old_poses[min(index, len(old_poses) - 1)]

    active = set(scene["active_scene_ids"])
    active.difference_update(old_candidates)
    active.update(new_candidates)
    scene["candidate_target_ids"] = list(new_candidates)
    scene["valid_target_ids"] = list(new_candidates)
    scene["active_scene_ids"] = sorted(active)


def recalibrate_task_geometry(
    scene: dict[str, Any], state: str, submode: str,
    registry: dict[str, dict[str, Any]],
) -> None:
    """Spread task evidence while preserving the locked relation semantics.

    The 30-family pilot happened to use mostly small adjacent objects. At 400
    families the same fixed y slots can pair larger seen objects and violate
    conservative footprints. Four calibrated depth levels provide at least
    120 mm separation; physical occluders are then recomputed collision-free.
    """

    id_to_model = {value["id"]: name for name, value in registry.items()}
    relation = scene["relation"]
    active = set(scene["active_scene_ids"])
    candidates = [value for value in scene["candidate_target_ids"] if value in active]
    anchors = [value for value in scene["anchor_ids"] if value in active]
    near_y, mid_near_y, mid_far_y, far_y = .13, .25, .37, .49

    if relation == "direct":
        target_y, anchor_y = [near_y, far_y], []
    elif relation == "right_of":
        target_y, anchor_y = [mid_far_y, far_y], [near_y]
    elif relation == "left_of":
        target_y, anchor_y = [near_y, mid_near_y], [far_y]
    elif relation in {"nearer_than", "front_of"}:
        target_y, anchor_y = [near_y, mid_near_y], [far_y]
    elif relation in {"farther_than", "behind"}:
        target_y, anchor_y = [mid_far_y, far_y], [near_y]
    elif relation == "between_in_depth":
        target_y, anchor_y = [mid_near_y, mid_far_y], [near_y, far_y]
    elif relation == "nearer_than_both":
        target_y, anchor_y = [near_y, mid_near_y], [mid_far_y, far_y]
    else:
        raise DevelopmentGenerationError(f"unsupported relation during recalibration: {relation}")

    if state == "ABSENT" and submode == "unsatisfied_relation" and candidates and anchors:
        if relation in {"right_of", "farther_than", "behind"}:
            target_y, anchor_y = [near_y], [far_y]
        elif relation in {"left_of", "nearer_than", "front_of"}:
            target_y, anchor_y = [far_y], [near_y]
        elif relation == "between_in_depth":
            target_y, anchor_y = [far_y], [near_y, mid_far_y]
        elif relation == "nearer_than_both":
            target_y, anchor_y = [far_y], [near_y, mid_near_y]

    occluder_model = id_to_model[scene["occluder_id"]]
    for layout_name in ("clean_layout", "occlusion_layout"):
        layout = scene[layout_name]
        layout[occluder_model] = list(registry[occluder_model]["storage_pose_xyyaw"])
        for index, object_id in enumerate(candidates):
            model_name = id_to_model[object_id]
            x_value = float(layout[model_name][0])
            layout[model_name] = [x_value, target_y[min(index, len(target_y) - 1)], float(layout[model_name][2])]
        for index, object_id in enumerate(anchors):
            model_name = id_to_model[object_id]
            x_value = float(layout[model_name][0])
            layout[model_name] = [x_value, anchor_y[min(index, len(anchor_y) - 1)], float(layout[model_name][2])]

    if state == "INSUFFICIENT_EVIDENCE" and submode == "occlusion" and candidates:
        place_safe_occluder(
            scene["clean_layout"], id_to_model[candidates[0]], occluder_model, registry
        )
    occlusion_target = candidates[0] if candidates and state != "ABSENT" else scene["semantic_cf_id"]
    place_safe_occluder(
        scene["occlusion_layout"], id_to_model[occlusion_target], occluder_model, registry
    )


def apply_development_layout_nonce(scene: dict[str, Any], layout_seed: int) -> str:
    """Apply a seed-specific yaw nonce without changing relation geometry."""

    # Retain nanoradian serialization precision so 400 independently derived
    # seeds do not collapse onto the same finite set of layout fingerprints.
    yaw_delta = 0.001 + (int(layout_seed) % 19_000_001) / 1_000_000_000.0
    for layout_name in ("clean_layout", "occlusion_layout"):
        layout = scene[layout_name]
        for model_name, pose in layout.items():
            if float(pose[1]) <= .65:
                layout[model_name] = [float(pose[0]), float(pose[1]), round(float(pose[2]) + yaw_delta, 9)]
    return f"yaw_nonce_rad={yaw_delta:.9f}"


def semantic_phrase(spec: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str:
    classes = [by_id[value]["semantic_class"] for value in spec["candidate_target_ids"]]
    unique = set(classes)
    if len(classes) > 1 and unique.issubset(FRUIT_CLASSES):
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
    phrase = relation.replace("_", " ")
    if relation in {"between_in_depth"} and len(anchors) >= 2:
        return f"locate the {semantic} between the {anchors[0]} and {anchors[1]} in camera depth."
    if relation in {"nearer_than_both", "farther_than_both"} and len(anchors) >= 2:
        comparison = "nearer to" if relation == "nearer_than_both" else "farther from"
        return f"locate the {semantic} {comparison} the camera than both the {anchors[0]} and {anchors[1]}."
    return f"locate the {semantic} that is {phrase} the {anchors[0]}."


def unique_instruction(
    spec: dict[str, Any], by_id: dict[str, dict[str, Any]], language_seed: int,
    used: set[str], forbidden: set[str], sample_key: str,
) -> tuple[str, str]:
    core = query_core(spec, by_id)
    for attempt in range(len(PREFIXES) * len(SUFFIXES)):
        digest = hashlib.sha256(f"{language_seed}|{sample_key}|{attempt}".encode()).hexdigest()
        prefix = PREFIXES[int(digest[:8], 16) % len(PREFIXES)]
        suffix = SUFFIXES[int(digest[8:16], 16) % len(SUFFIXES)]
        instruction = f"{prefix} {core} {suffix}"
        if instruction not in used and instruction not in forbidden:
            used.add(instruction)
            template_id = "development_template_" + hashlib.sha256(instruction.encode()).hexdigest()[:20]
            return instruction, template_id
    raise DevelopmentGenerationError(f"cannot allocate a unique instruction: {sample_key}")


def required_visible_ids(specs: list[dict[str, Any]], capture_id: str) -> list[str]:
    required: set[str] = set()
    for spec in (item for item in specs if item["capture_id"] == capture_id):
        state = spec["answerability_state"]
        if state in {"FOUND", "AMBIGUOUS"}:
            required.update(spec["valid_target_ids"])
            required.update(spec["anchor_ids"])
        elif state == "INSUFFICIENT_EVIDENCE":
            inverse_out_of_view = (
                spec["variant"] == "relation_counterfactual"
                and spec["state_submode"] == "too_small_or_out_of_view"
            )
            if not inverse_out_of_view:
                required.update(spec["anchor_ids"])
            if spec["state_submode"] != "too_small_or_out_of_view" or inverse_out_of_view:
                required.update(spec["valid_target_ids"])
        elif state == "ABSENT" and spec["state_submode"] == "unsatisfied_relation":
            required.update(spec["candidate_target_ids"])
            required.update(spec["anchor_ids"])
    return sorted(required)


def allocate_batches(families: list[dict[str, Any]], master_seed: int) -> list[dict[str, Any]]:
    canary_targets = {
        ("train", "FOUND"): 7,
        ("train", "INSUFFICIENT_EVIDENCE"): 3,
        ("train", "AMBIGUOUS"): 3,
        ("train", "ABSENT"): 3,
        ("dev", "FOUND"): 1,
        ("dev", "INSUFFICIENT_EVIDENCE"): 1,
        ("dev", "AMBIGUOUS"): 1,
        ("dev", "ABSENT"): 1,
    }
    selected: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for (split, state), amount in canary_targets.items():
        candidates = [
            item for item in families
            if item["split"] == split and item["primary_answerability_stratum"] == state
        ]
        for _ in range(amount):
            available = [item for item in candidates if item not in selected]
            available.sort(key=lambda item: (
                category_counts[item["family_category"]],
                hashlib.sha256(
                    f"{master_seed}|development_canary|{item['family_id']}".encode()
                ).hexdigest(),
            ))
            if not available:
                raise DevelopmentGenerationError(f"canary allocation exhausted: {split}/{state}")
            chosen = available[0]
            selected.append(chosen)
            category_counts[chosen["family_category"]] += 1
    if len(selected) != 20 or len(category_counts) != 6:
        raise DevelopmentGenerationError(f"canary coverage failed: {dict(category_counts)}")

    selected_ids = {item["family_id"] for item in selected}
    remainder = ranked(
        [item for item in families if item["family_id"] not in selected_ids],
        master_seed,
        "development_post_canary_batch_order",
    )
    chunks = [selected] + [remainder[index:index + 95] for index in range(0, len(remainder), 95)]
    if [len(value) for value in chunks] != [20, 95, 95, 95, 95]:
        raise DevelopmentGenerationError("batch sizes differ from 20 + 95 x 4")
    batches = []
    for order, chunk in enumerate(chunks):
        batch_id = "canary_000" if order == 0 else f"batch_{order:03d}"
        batches.append({
            "batch_id": batch_id,
            "order": order,
            "family_count": len(chunk),
            "planned_capture_count": 2 * len(chunk),
            "family_ids": [item["family_id"] for item in chunk],
            "split_counts": dict(sorted(Counter(item["split"] for item in chunk).items())),
            "primary_answerability_counts": dict(sorted(Counter(
                item["primary_answerability_stratum"] for item in chunk
            ).items())),
            "family_category_counts": dict(sorted(Counter(item["family_category"] for item in chunk).items())),
            "prerequisite_batch": None if order == 0 else ("canary_000" if order == 1 else f"batch_{order - 1:03d}"),
            "next_batch_requires_raw_qc_pass": True,
        })
    return batches


def count_by_split(families: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    return {
        split: dict(sorted(Counter(str(item[field]) for item in families if item["split"] == split).items()))
        for split in ("train", "dev")
    }


def build_artifacts(workspace: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = workspace / "protocol"
    seed_path = protocol / "dataset_expansion_v2_seed_lock.json"
    partition_path = protocol / "dataset_expansion_v2_asset_partition.json"
    pilot_manifest_path = protocol / "dataset_v2_pilot_manifest.json"
    pilot_plan_path = protocol / "dataset_v2_pilot_capture_plan.json"
    seed_lock = read_json(seed_path)
    partition = read_json(partition_path)
    pilot_manifest = read_json(pilot_manifest_path)
    pilot_plan = read_json(pilot_plan_path)
    assignments = build_development_assignments(seed_lock, partition)
    commitment = assignment_commitment(assignments)
    if commitment != EXPECTED_DEVELOPMENT_COMMITMENT or commitment != seed_lock["development"]["assignment_commitment_sha256"]:
        raise DevelopmentGenerationError(f"development assignment commitment mismatch: {commitment}")

    registry = object_registry()
    by_id = {value["id"]: value for value in registry.values()}
    if len(by_id) != len(registry) or len({int(value["label"]) for value in registry.values()}) != len(registry):
        raise DevelopmentGenerationError("object IDs or semantic-instance labels are not unique")
    submodes = allocate_submodes(assignments, int(seed_lock["master_seed"]))
    pilot_instructions = {
        spec["instruction"] for family in pilot_manifest["families"] for spec in family["variant_specs"]
    }
    used_instructions: set[str] = set()
    families: list[dict[str, Any]] = []
    capture_by_family: dict[str, list[dict[str, Any]]] = {}

    for index, assignment in enumerate(assignments):
        family_id = assignment["family_id"]
        submode = submodes[family_id]
        try:
            scene = primary_scene(assignment, submode, index, registry)
        except PilotGenerationError as exc:
            raise DevelopmentGenerationError(f"{family_id}: {exc}") from exc
        if assignment["primary_answerability_stratum"] == "AMBIGUOUS":
            normalize_ambiguous_scene(scene, submode, index, registry)
        recalibrate_task_geometry(
            scene, assignment["primary_answerability_stratum"], submode, registry
        )
        layout_nonce = apply_development_layout_nonce(scene, int(assignment["seed_bundle"]["layout"]))
        specs = variant_specs(assignment, scene, registry, index)
        template_ids = []
        for spec in specs:
            sample_key = f"{family_id}__{spec['variant']}"
            instruction, template_id = unique_instruction(
                spec, by_id, int(assignment["seed_bundle"]["language"]),
                used_instructions, pilot_instructions, sample_key,
            )
            spec["instruction"] = instruction
            spec["language_template_family_id"] = template_id
            spec["instruction_sha256"] = hashlib.sha256(instruction.encode()).hexdigest()
            template_ids.append(template_id)
        scene["instruction"] = specs[0]["instruction"]
        family = {
            **assignment,
            "development_capture_policy": "OFFICIAL_DEVELOPMENT_TRAIN_OR_DEV_ONLY",
            "pilot_reuse_forbidden": True,
            "state_submode": scene["state_submode"],
            "layout_instance_nonce": layout_nonce,
            "language_template_family_ids": template_ids,
            "primary_scene": {
                key: scene[key] for key in (
                    "candidate_target_ids", "valid_target_ids", "anchor_ids", "active_scene_ids",
                    "relation", "reference_frame", "instruction", "occluder_id",
                )
            },
            "variant_specs": specs,
        }
        families.append(family)
        family_captures = []
        for condition, layout, seed_key in (
            ("clean", scene["clean_layout"], "sensor"),
            ("occlusion", scene["occlusion_layout"], "occlusion"),
        ):
            capture_id = f"{family_id}__{condition}_capture"
            visible_ids = required_visible_ids(specs, capture_id)
            active_layout = {
                name: pose for name, pose in layout.items() if float(pose[1]) <= .65
            }
            family_captures.append({
                "capture_id": capture_id,
                "family_id": family_id,
                "split": assignment["split"],
                "condition": condition,
                "seed": int(assignment["seed_bundle"][seed_key]),
                "layout_seed": int(assignment["seed_bundle"]["layout"]),
                "ood_axis": "none",
                "camera_bin_id": assignment["camera_bin_id"],
                "view_joint_pose": list(IID_VIEW),
                "layout_generator_id": assignment["layout_generator_id"],
                "layout_instance_fingerprint_sha256": canonical_json_sha256(active_layout),
                "layout": layout,
                "required_visible_object_ids": visible_ids,
                "required_visible_label_ids": sorted(int(by_id[value]["label"]) for value in visible_ids),
            })
        capture_by_family[family_id] = family_captures

    batches = allocate_batches(families, int(seed_lock["master_seed"]))
    batch_for_family = {
        family_id: batch["batch_id"] for batch in batches for family_id in batch["family_ids"]
    }
    family_by_id = {item["family_id"]: item for item in families}
    ordered_family_ids = [family_id for batch in batches for family_id in batch["family_ids"]]
    captures: list[dict[str, Any]] = []
    for family_id in ordered_family_ids:
        for capture in capture_by_family[family_id]:
            capture["batch_id"] = batch_for_family[family_id]
            captures.append(capture)

    counts = {
        "split": dict(sorted(Counter(item["split"] for item in families).items())),
        "primary_answerability_by_split": count_by_split(families, "primary_answerability_stratum"),
        "family_category_by_split": count_by_split(families, "family_category"),
        "ood_axis_by_split": count_by_split(families, "ood_axis"),
        "state_submode_by_split": count_by_split(families, "state_submode"),
    }
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "status": "STATIC_DEVELOPMENT_PLAN_NOT_CAPTURED",
        "scientific_status": "PLANNED_NOT_CAPTURED",
        "eligible_for_official_development": True,
        "allowed_splits": ["train", "dev"],
        "sealed_splits_not_created": ["calibration", "test_iid", "test_ood"],
        "family_count": len(families),
        "sample_count": sum(len(item["variant_specs"]) for item in families),
        "variant_count_per_family": len(VARIANTS),
        "design_assignment_commitment_sha256": commitment,
        "design_seed_lock_sha256": sha256_file(seed_path),
        "asset_partition_sha256": sha256_file(partition_path),
        "pilot_denylist": {
            "manifest_sha256": sha256_file(pilot_manifest_path),
            "capture_plan_sha256": sha256_file(pilot_plan_path),
            "family_ids_sha256": canonical_json_sha256(sorted(item["family_id"] for item in pilot_manifest["families"])),
            "all_seed_values_sha256": canonical_json_sha256(sorted(
                value for item in pilot_manifest["families"] for value in item["seed_bundle"].values()
            )),
            "exact_instructions_sha256": canonical_json_sha256(sorted(pilot_instructions)),
        },
        "counts": counts,
        "batch_assignments": batches,
        "families": families,
    }

    world = workspace / "ur3/ur_simulation_gz/worlds/ur3_pick_place_inventory_v2.sdf"
    capture_plan = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "STATIC_PLAN_CAPTURE_FORBIDDEN_UNTIL_PREFLIGHT_PASS",
        "scientific_status": "PLANNED_NOT_CAPTURED",
        "world_file": str(world.relative_to(workspace)),
        "world_sha256": sha256_file(world),
        "coordinate_suffix": COORDINATE_SUFFIX,
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
            "max_timestamp_spread_sec": .035,
            "metric_depth_unit": "metre",
        },
        "camera_poses": {"camera_v2_iid": IID_VIEW},
        "workspace_roi_base_link": {"x_m": [-.60, .60], "y_m": [.06, .65]},
        "inference_payload_template": {
            "rgb_model_input": "TO_BE_CAPTURED",
            "depth_relative_model_input": "TO_BE_DERIVED",
            "enable_depth": True,
            "prompt": "{instruction} " + COORDINATE_SUFFIX,
            "coordinate_suffix": COORDINATE_SUFFIX,
        },
        "object_registry": registry,
        "manifest_sha256_pending": True,
        "captures": captures,
    }
    if set(family_by_id) != set(batch_for_family):
        raise DevelopmentGenerationError("batch assignment does not cover every family exactly once")
    return manifest, capture_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    protocol = workspace / "protocol"
    manifest, capture_plan = build_artifacts(workspace)
    manifest_path = protocol / "dataset_v2_development_manifest.json"
    plan_path = protocol / "dataset_v2_development_capture_plan.json"
    write_json(manifest_path, manifest)
    capture_plan["manifest_sha256"] = sha256_file(manifest_path)
    capture_plan.pop("manifest_sha256_pending", None)
    write_json(plan_path, capture_plan)
    print(
        "DATASET_V2_DEVELOPMENT_STATIC_MATERIALIZED "
        f"families={manifest['family_count']} samples={manifest['sample_count']} "
        f"captures={capture_plan['capture_count']} batches="
        f"{[item['family_count'] for item in capture_plan['batches']]} "
        f"commitment={manifest['design_assignment_commitment_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
