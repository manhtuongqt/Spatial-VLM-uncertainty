#!/usr/bin/env python3
"""Generate the locked 50-family WP2 manifest and Gazebo capture plan."""

from __future__ import annotations

import argparse
import copy
import random
from collections import Counter
from pathlib import Path

from wp2_common import (
    GENERATOR_VERSION,
    PROTOCOL_ID,
    RANDOM_SEED,
    SCHEMA_VERSION,
    VARIANTS,
    canonical_json_sha256,
    ensure_new_directory,
    sha256_file,
    utc_now,
    workspace_root,
    write_json,
)


COORDINATE_SUFFIX = (
    "Your answer must be exactly one list containing one normalized image point, "
    "formatted as [(x, y)], where x and y are between 0 and 1. Return no words."
)

# Storage poses are copied from the already validated pilot scene configuration,
# but no pilot RGB/depth/frame is used. Every WP2 family receives a new seed and
# a new Gazebo capture.
OBJECTS = {
    "red_cube": {"id": "cube_red_01", "label": 1, "category": "red cube", "z": 0.030000, "storage": [-0.65, 1.40, 0.0]},
    "blue_cube": {"id": "cube_blue_01", "label": 2, "category": "blue cube", "z": 0.025000, "storage": [-0.45, 1.40, 0.0]},
    "green_cube": {"id": "cube_green_01", "label": 3, "category": "green cube", "z": 0.025000, "storage": [-0.25, 1.40, 0.0]},
    "yellow_cube": {"id": "cube_yellow_01", "label": 4, "category": "yellow cube", "z": 0.025000, "storage": [-0.05, 1.40, 0.0]},
    "orange_cube": {"id": "cube_orange_01", "label": 5, "category": "orange cube", "z": 0.025000, "storage": [0.15, 1.40, 0.0]},
    "purple_cube": {"id": "cube_purple_01", "label": 6, "category": "purple cube", "z": 0.025000, "storage": [0.35, 1.40, 0.0]},
    "pink_cube": {"id": "cube_pink_01", "label": 7, "category": "pink cube", "z": 0.025000, "storage": [0.55, 1.40, 0.0]},
    "ycb_cracker_box": {"id": "ycb_cracker_box_01", "label": 21, "category": "red cracker box", "z": 0.106710, "storage": [-0.02, 0.85, 0.0]},
    "ycb_sugar_box": {"id": "ycb_sugar_box_01", "label": 22, "category": "yellow sugar box", "z": 0.088107, "storage": [0.20, 0.85, 0.0]},
    "ycb_tomato_soup_can": {"id": "ycb_tomato_soup_can_01", "label": 23, "category": "red tomato soup can", "z": 0.050948, "storage": [-0.55, 1.10, 0.0]},
    "ycb_mustard_bottle": {"id": "ycb_mustard_bottle_01", "label": 24, "category": "yellow mustard bottle", "z": 0.095695, "storage": [-0.38, 1.10, 0.0]},
    "ycb_banana": {"id": "ycb_banana_01", "label": 25, "category": "long curved yellow banana", "z": 0.018376, "storage": [-0.30, 0.85, 0.25]},
    "ycb_apple": {"id": "ycb_apple_01", "label": 26, "category": "round red apple", "z": 0.037710, "storage": [-0.21, 1.10, 0.0]},
    "ycb_orange": {"id": "ycb_orange_01", "label": 27, "category": "round orange fruit", "z": 0.037010, "storage": [-0.04, 1.10, 0.0]},
    "ycb_power_drill": {"id": "ycb_power_drill_01", "label": 28, "category": "blue handheld power drill", "z": 0.028689, "storage": [-0.62, 0.85, -0.25]},
    "mango": {"id": "mango_01", "label": 29, "category": "orange-colored mango", "z": 0.030000, "storage": [0.38, 0.85, 0.35]},
}
ID_TO_MODEL = {value["id"]: key for key, value in OBJECTS.items()}

CATEGORY_COUNTS = (
    ("nearer_farther", 12),
    ("front_behind_camera", 8),
    ("multi_anchor_depth_order", 5),
    ("occlusion_depth_evidence", 5),
    ("direct_grounding", 5),
    ("relation_2d", 5),
    ("ambiguous_absent", 4),
    ("shape_comparison", 3),
    ("metric_comparison", 3),
)
DEPTH_CATEGORIES = {
    "nearer_farther", "front_behind_camera", "multi_anchor_depth_order",
    "occlusion_depth_evidence",
}

DEPTH_TRIPLES = (
    ("ycb_apple_01", "ycb_orange_01", "ycb_tomato_soup_can_01"),
    ("ycb_orange_01", "ycb_apple_01", "ycb_mustard_bottle_01"),
    ("ycb_tomato_soup_can_01", "ycb_mustard_bottle_01", "ycb_apple_01"),
    ("ycb_sugar_box_01", "ycb_cracker_box_01", "ycb_banana_01"),
    ("ycb_mustard_bottle_01", "ycb_tomato_soup_can_01", "ycb_orange_01"),
)
DIRECT_TRIPLES = (
    ("ycb_apple_01", "ycb_orange_01", "ycb_mustard_bottle_01"),
    ("ycb_orange_01", "ycb_apple_01", "ycb_tomato_soup_can_01"),
    ("ycb_banana_01", "ycb_apple_01", "ycb_orange_01"),
    ("ycb_mustard_bottle_01", "ycb_tomato_soup_can_01", "ycb_apple_01"),
    ("ycb_power_drill_01", "ycb_sugar_box_01", "ycb_banana_01"),
)


def object_name(object_id: str) -> str:
    return str(OBJECTS[ID_TO_MODEL[object_id]]["category"])


def storage_layout() -> dict[str, list[float]]:
    return {name: list(item["storage"]) for name, item in OBJECTS.items()}


def jitter(rng: random.Random, x_value: float, y_value: float, yaw: float = 0.0) -> list[float]:
    return [
        round(x_value + rng.uniform(-0.012, 0.012), 6),
        round(y_value + rng.uniform(-0.010, 0.010), 6),
        round(yaw + rng.uniform(-0.12, 0.12), 6),
    ]


def place(layout: dict, object_id: str, pose: list[float]) -> None:
    layout[ID_TO_MODEL[object_id]] = pose


def split_assignments() -> list[str]:
    rng = random.Random(RANDOM_SEED + 91)
    depth = ["train"] * 18 + ["dev"] * 5 + ["calibration"] * 4 + ["test"] * 3
    other = ["train"] * 12 + ["dev"] * 3 + ["calibration"] * 2 + ["test"] * 3
    rng.shuffle(depth)
    rng.shuffle(other)
    return depth + other


def base_variant(
    variant: str,
    capture_id: str,
    instruction: str,
    target_ids: list[str],
    anchor_ids: list[str],
    relations: list[str],
    reference_frame: str,
    answerable: bool,
    state: str,
    sources: list[str],
    severity: int,
    intervention: str,
    perturbation: dict,
) -> dict:
    return {
        "variant": variant,
        "capture_id": capture_id,
        "instruction": instruction,
        "target_ids": target_ids,
        "anchor_ids": anchor_ids,
        "valid_target_ids": target_ids if answerable else [],
        "relations": relations,
        "reference_frame": reference_frame,
        "answerable": answerable,
        "answerability_state": state,
        "uncertainty_sources": sources,
        "severity": severity,
        "expected_intervention": intervention,
        "perturbation": perturbation,
    }


def make_family(index: int, category: str, split: str, category_offset: int) -> dict:
    family_id = f"wp2_family_{index:04d}"
    family_seed = RANDOM_SEED + index * 1009
    rng = random.Random(family_seed)
    target_id, anchor_id, semantic_id = (
        DEPTH_TRIPLES[category_offset % len(DEPTH_TRIPLES)]
        if category in DEPTH_CATEGORIES
        else DIRECT_TRIPLES[category_offset % len(DIRECT_TRIPLES)]
    )
    if category == "ambiguous_absent":
        if category_offset < 2:
            target_id, semantic_id, anchor_id = (
                "ycb_orange_01", "mango_01", "ycb_apple_01"
            )
        else:
            target_id, semantic_id, anchor_id = (
                "ycb_power_drill_01", "ycb_apple_01", "ycb_orange_01"
            )
    elif category == "shape_comparison":
        target_id, anchor_id, semantic_id = (
            "ycb_banana_01", "ycb_apple_01", "ycb_orange_01"
        )
    elif category == "metric_comparison":
        target_id, anchor_id, semantic_id = (
            "ycb_mustard_bottle_01", "ycb_apple_01", "ycb_orange_01"
        )
    occupied_ids = {target_id, anchor_id, semantic_id}
    second_anchor_id = next(
        candidate for candidate in (
            "ycb_banana_01", "ycb_sugar_box_01", "ycb_tomato_soup_can_01",
            "ycb_apple_01", "ycb_orange_01", "ycb_cracker_box_01",
        ) if candidate not in occupied_ids
    )
    occluder_id = next(
        candidate for candidate in (
            "ycb_mustard_bottle_01", "ycb_tomato_soup_can_01",
            "ycb_sugar_box_01", "ycb_cracker_box_01", "ycb_power_drill_01",
        ) if candidate not in occupied_ids | {second_anchor_id}
    )

    clean = storage_layout()
    occluded = storage_layout()
    near_pose = jitter(rng, -0.20, 0.145)
    mid_pose = jitter(rng, -0.22, 0.285)
    far_pose = jitter(rng, -0.20, 0.440)
    # At the locked oblique wrist view, changing tabletop +Y traverses the
    # visible image horizontally. Keeping X close to -0.20 avoids the robot
    # self-occlusion band observed during smoke attempt 1.
    side_pose = jitter(rng, -0.20, 0.285)
    other_pose = jitter(rng, -0.20, 0.535, 0.25)

    relation = ""
    relation_cf_relations: list[str] = []
    reference_frame = "object_semantics"
    anchors = [anchor_id]
    relation_cf_anchor_ids = [target_id]
    clean_state = "FOUND"
    clean_answerable = True
    clean_sources: list[str] = []
    clean_intervention = "EXECUTE"

    if category in {"nearer_farther", "front_behind_camera"}:
        relation = (
            ("nearer_than" if category == "nearer_farther" else "front_of")
            if category_offset % 2 == 0
            else ("farther_than" if category == "nearer_farther" else "behind")
        )
        target_pose, anchor_pose = (
            (near_pose, far_pose)
            if relation in {"nearer_than", "front_of"}
            else (far_pose, near_pose)
        )
        place(clean, target_id, target_pose)
        place(clean, anchor_id, anchor_pose)
        place(clean, semantic_id, side_pose)
        place(clean, second_anchor_id, other_pose)
        relation_words = {
            "nearer_than": "closer to the camera than",
            "farther_than": "farther from the camera than",
            "front_of": "in front of",
            "behind": "behind",
        }
        instruction = (
            f"Point to the {object_name(target_id)} that is {relation_words[relation]} "
            f"the {object_name(anchor_id)}."
        )
        opposite = {
            "nearer_than": "farther_than", "farther_than": "nearer_than",
            "front_of": "behind", "behind": "front_of",
        }[relation]
        relation_cf_instruction = (
            f"Point to the {object_name(anchor_id)} that is "
            f"{relation_words[opposite]} the {object_name(target_id)}."
        )
        relation_cf_relations = [opposite]
    elif category == "multi_anchor_depth_order":
        anchors = [anchor_id, second_anchor_id]
        place(clean, anchor_id, near_pose)
        place(clean, target_id, mid_pose)
        place(clean, second_anchor_id, far_pose)
        place(clean, semantic_id, other_pose)
        relation = "nearer_than_both" if category_offset == 4 else "between_in_depth"
        reference_frame = "camera_color_optical_frame"
        if relation == "between_in_depth":
            instruction = (
                f"Point to the {object_name(target_id)} whose camera depth is between "
                f"the nearer {object_name(anchor_id)} and farther {object_name(second_anchor_id)}."
            )
            relation_cf_instruction = (
                f"Point to the nearer {object_name(anchor_id)} in front of the "
                f"{object_name(target_id)}."
            )
            relation_cf_relations = ["nearer_than"]
        else:
            instruction = (
                f"Point to the {object_name(target_id)} that is closer to the camera than "
                f"both the {object_name(anchor_id)} and {object_name(second_anchor_id)}."
            )
            relation_cf_instruction = (
                f"Point to the {object_name(anchor_id)} farther from the camera than the "
                f"{object_name(target_id)}."
            )
            relation_cf_relations = ["farther_than"]
    elif category == "occlusion_depth_evidence":
        anchors = [anchor_id]
        place(clean, target_id, far_pose)
        place(clean, anchor_id, near_pose)
        place(clean, semantic_id, side_pose)
        relation = "behind"
        reference_frame = "camera_color_optical_frame"
        instruction = (
            f"Point to the {object_name(target_id)} behind the {object_name(anchor_id)} "
            "using the visible surface and depth ordering."
        )
        relation_cf_instruction = (
            f"Point to the {object_name(anchor_id)} in front of the {object_name(target_id)}."
        )
        relation_cf_relations = ["front_of"]
    elif category == "direct_grounding":
        anchors = [anchor_id]
        place(clean, target_id, mid_pose)
        place(clean, anchor_id, near_pose)
        place(clean, semantic_id, far_pose)
        instruction = f"Point to one visible point well inside the {object_name(target_id)}."
        relation = "direct"
        direct_cf_relation = "farther_than" if category_offset == 3 else "nearer_than"
        direct_cf_words = "farther from the camera than" if direct_cf_relation == "farther_than" else "closer to the camera than"
        relation_cf_instruction = (
            f"Point to the {object_name(anchor_id)} {direct_cf_words} the "
            f"{object_name(target_id)}."
        )
        relation_cf_relations = [direct_cf_relation]
    elif category == "relation_2d":
        anchors = [anchor_id]
        # This mirrors the empirically verified pilot placement that produces
        # a horizontal image relation at the locked wrist view.
        place(clean, anchor_id, jitter(rng, -0.15, 0.16))
        place(clean, target_id, jitter(rng, -0.18, 0.35))
        place(clean, semantic_id, other_pose)
        relation = "right_of"
        reference_frame = "image"
        instruction = (
            f"Point to the {object_name(target_id)} that is {relation.replace('_', ' ')} "
            f"the {object_name(anchor_id)} in the image."
        )
        inverse = "left_of" if relation == "right_of" else "right_of"
        relation_cf_instruction = (
            f"Point to the {object_name(anchor_id)} that is {inverse.replace('_', ' ')} "
            f"the {object_name(target_id)} in the image."
        )
        relation_cf_relations = [inverse]
    elif category == "ambiguous_absent":
        if category_offset < 2:
            anchors = [anchor_id]
            place(clean, target_id, near_pose)
            place(clean, semantic_id, mid_pose)
            place(clean, anchor_id, far_pose)
            instruction = "Point to the orange-colored fruit on the tabletop."
            clean_answerable, clean_state = False, "AMBIGUOUS"
            clean_sources, clean_intervention = ["semantic"], "ASK_USER"
        else:
            anchors = [anchor_id]
            place(clean, semantic_id, near_pose)
            place(clean, anchor_id, far_pose)
            instruction = "Point to the blue handheld power drill on the tabletop."
            clean_answerable, clean_state = False, "ABSENT"
            clean_sources, clean_intervention = ["semantic"], "ABSTAIN"
        relation = "semantic_reference"
        relation_cf_instruction = (
            f"Point to the {object_name(anchor_id)} farther from the camera than the "
            f"{object_name(semantic_id)}."
        )
        relation_cf_anchor_ids = [semantic_id]
        relation_cf_relations = ["farther_than"]
    elif category == "shape_comparison":
        anchors = [anchor_id, semantic_id]
        place(clean, target_id, mid_pose)
        place(clean, anchor_id, near_pose)
        place(clean, semantic_id, far_pose)
        relation = "more_elongated_than"
        reference_frame = "base_link"
        instruction = "Point to the long curved fruit that is more elongated than both round fruits."
        relation_cf_instruction = "Point to the round red apple closer to the camera than the banana."
        relation_cf_relations = ["nearer_than"]
    elif category == "metric_comparison":
        anchors = [anchor_id, semantic_id]
        place(clean, target_id, mid_pose)
        place(clean, anchor_id, near_pose)
        place(clean, semantic_id, far_pose)
        relation = "taller_than"
        reference_frame = "base_link"
        instruction = "Point to the upright yellow bottle physically taller than both round fruits."
        relation_cf_instruction = "Point to the round red apple farther from the camera than the bottle."
        relation_cf_relations = ["farther_than"]
    else:
        raise ValueError(category)

    # Occlusion capture preserves the family semantics but aligns a dedicated
    # occluder with the target. It is separately rendered and never synthesized
    # by painting over RGB pixels.
    occluded = copy.deepcopy(clean)
    target_model = ID_TO_MODEL[target_id]
    target_clean_pose = clean[target_model]
    place(
        occluded,
        occluder_id,
        [target_clean_pose[0], round(target_clean_pose[1] - 0.055, 6), 0.0],
    )

    clean_capture = f"{family_id}__clean_capture"
    occlusion_capture = f"{family_id}__occlusion_capture"
    semantic_instruction = f"Point to the {object_name(semantic_id)}."
    semantic_target = semantic_id
    if category == "ambiguous_absent":
        semantic_instruction = f"Point specifically to the {object_name(semantic_id)}."

    clean_targets = [target_id]
    if clean_state == "AMBIGUOUS":
        clean_targets = [target_id, semantic_id]
    clean_valid = clean_targets if clean_answerable else []

    variants = [
        base_variant(
            "clean", clean_capture, instruction, clean_targets, anchors, [relation],
            reference_frame, clean_answerable, clean_state, clean_sources, 0,
            clean_intervention, {"kind": "none", "severity": 0},
        ),
        base_variant(
            "semantic_counterfactual", clean_capture, semantic_instruction,
            [semantic_target], [], ["direct"], "object_semantics", True, "FOUND",
            ["semantic"], 1, "EXECUTE",
            {"kind": "semantic_query_substitution", "from": target_id, "to": semantic_target, "severity": 1},
        ),
        base_variant(
            "relation_counterfactual", clean_capture, relation_cf_instruction,
            [anchor_id], relation_cf_anchor_ids, relation_cf_relations,
            "camera_color_optical_frame" if category != "relation_2d" else "image",
            True, "FOUND", ["relation"], 1, "EXECUTE",
            {"kind": "relation_query_inversion", "base_relation": relation, "severity": 1},
        ),
        base_variant(
            "depth_corruption", clean_capture, instruction, clean_targets, anchors,
            [relation], reference_frame,
            False if clean_state == "FOUND" else clean_answerable,
            "INSUFFICIENT_EVIDENCE" if clean_state == "FOUND" else clean_state,
            sorted(set(clean_sources + ["depth"])), 1 + (index % 3),
            "REOBSERVE" if clean_state == "FOUND" else clean_intervention,
            {"kind": ("bias_noise", "localized_holes_edges", "inversion_shift")[index % 3], "severity": 1 + (index % 3)},
        ),
        base_variant(
            "occlusion_view_counterfactual", occlusion_capture, instruction,
            clean_targets, anchors, [relation], reference_frame,
            False if clean_state == "FOUND" else clean_answerable,
            "INSUFFICIENT_EVIDENCE" if clean_state == "FOUND" else clean_state,
            sorted(set(clean_sources + ["occlusion", "spatial"])), 2,
            "REOBSERVE" if clean_state == "FOUND" else clean_intervention,
            {"kind": "gazebo_object_occlusion", "occluder_id": occluder_id, "severity": 2},
        ),
    ]
    variants[0]["valid_target_ids"] = clean_valid
    variants[3]["valid_target_ids"] = clean_valid
    variants[4]["valid_target_ids"] = clean_valid

    return {
        "family_id": family_id,
        "family_index": index,
        "split": split,
        "category": category,
        "depth_dependent": category in DEPTH_CATEGORIES,
        "seed": family_seed,
        "target_id": target_id,
        "anchor_ids": anchors,
        "semantic_counterfactual_target_id": semantic_target,
        "clean_layout": clean,
        "occlusion_layout": occluded,
        "variant_specs": variants,
    }


def build_manifest() -> tuple[dict, dict]:
    category_values = []
    for category, count in CATEGORY_COUNTS:
        category_values.extend([category] * count)
    splits = split_assignments()
    offsets = Counter()
    families = []
    for index, (category, split) in enumerate(zip(category_values, splits), start=1):
        families.append(make_family(index, category, split, offsets[category]))
        offsets[category] += 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "seed": RANDOM_SEED,
        "split_policy": {
            "unit": "family_id",
            "assignments_locked_before_capture": True,
            "target_family_counts": {"train": 30, "dev": 8, "calibration": 6, "test": 6},
        },
        "family_count": len(families),
        "sample_count": len(families) * len(VARIANTS),
        "depth_dependent_family_count": sum(item["depth_dependent"] for item in families),
        "variants": list(VARIANTS),
        "families": families,
    }
    capture_records = []
    for family in families:
        for suffix, layout in (("clean_capture", family["clean_layout"]), ("occlusion_capture", family["occlusion_layout"])):
            capture_records.append({
                "capture_id": f"{family['family_id']}__{suffix}",
                "family_id": family["family_id"],
                "split": family["split"],
                "seed": family["seed"],
                "condition": suffix.replace("_capture", ""),
                "layout": layout,
            })
    capture_plan = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "seed": RANDOM_SEED,
        "object_registry": OBJECTS,
        "coordinate_suffix": COORDINATE_SUFFIX,
        "capture_count": len(capture_records),
        "captures": capture_records,
    }
    return manifest, capture_plan


def validate_internal(manifest: dict, capture_plan: dict) -> None:
    assert manifest["family_count"] == 50
    assert manifest["sample_count"] == 250
    assert manifest["depth_dependent_family_count"] == 30
    assert Counter(item["split"] for item in manifest["families"]) == {
        "train": 30, "dev": 8, "calibration": 6, "test": 6,
    }
    family_ids = [item["family_id"] for item in manifest["families"]]
    assert len(family_ids) == len(set(family_ids))
    for family in manifest["families"]:
        assert [item["variant"] for item in family["variant_specs"]] == list(VARIANTS)
        assert all(len(layout) == len(OBJECTS) for layout in (family["clean_layout"], family["occlusion_layout"]))
        clean_visible = [
            (name, pose) for name, pose in family["clean_layout"].items()
            if float(pose[1]) <= 0.65
        ]
        for left_index, (left_name, left_pose) in enumerate(clean_visible):
            for right_name, right_pose in clean_visible[left_index + 1:]:
                distance_xy = (
                    (float(left_pose[0]) - float(right_pose[0])) ** 2
                    + (float(left_pose[1]) - float(right_pose[1])) ** 2
                ) ** 0.5
                assert distance_xy >= 0.045, (
                    family["family_id"], left_name, right_name, distance_xy
                )
    assert capture_plan["capture_count"] == 100
    assert len({item["capture_id"] for item in capture_plan["captures"]}) == 100


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allow-existing-empty", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root).expanduser().resolve()
    if args.allow_existing_empty:
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise RuntimeError(f"output root is not empty: {root}")
    else:
        ensure_new_directory(root)
    manifest, capture_plan = build_manifest()
    validate_internal(manifest, capture_plan)
    write_json(root / "family_manifest.json", manifest)
    write_json(root / "capture_plan.json", capture_plan)
    smoke_ids = {"wp2_family_0001", "wp2_family_0031", "wp2_family_0041"}
    smoke = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "gate": "3 representative families x 5 variants",
        "family_ids": sorted(smoke_ids),
        "sample_count": 15,
        "capture_ids": [
            item["capture_id"] for item in capture_plan["captures"]
            if item["family_id"] in smoke_ids
        ],
    }
    write_json(root / "smoke_selection.json", smoke)
    write_json(root / "object_registry.json", OBJECTS)
    checkpoint = {
        "schema_version": 1,
        "checkpoint": "SCHEMA_AND_FAMILIES_LOCKED_BEFORE_CAPTURE",
        "created_at_utc": utc_now(),
        "family_manifest_sha256": sha256_file(root / "family_manifest.json"),
        "capture_plan_sha256": sha256_file(root / "capture_plan.json"),
        "family_manifest_canonical_sha256": canonical_json_sha256(manifest),
        "generator_source_sha256": sha256_file(Path(__file__).resolve()),
        "workspace": str(workspace_root()),
    }
    write_json(root / "report_assets" / "checkpoints" / "00_schema_family_lock.json", checkpoint)
    print(root)
    print(f"families=50 samples=250 depth_dependent=30 captures=100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
