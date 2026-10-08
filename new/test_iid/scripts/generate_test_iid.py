#!/usr/bin/env python3
"""Generate the sealed 200-family Test-IID design without model inference."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OLD_PROTOCOL = ROOT / "old" / "protocol"
sys.path.insert(0, str(OLD_PROTOCOL))

import dataset_v2_1_calibration_generator as base  # noqa: E402
from wp2_common import sha256_file, write_json  # noqa: E402


PROTOCOL_ID = "roborefer_dataset_v2_1_test_iid_capture_200"
MASTER_SEED = 300920260701
FAMILY_PREFIX = "v211iid_family_"
OUTPUT_ROOT = "new/test_iid/dataset"
FREEZE_LOCK = "new/test_iid/contracts/best_v2_freeze_lock.json"
PROTOCOL_DIR = ROOT / "new" / "test_iid" / "protocol"

TARGET_QUOTAS = {
    "fruit": 110,
    "mug": 30,
    "box": 20,
    "container": 30,
    "cube": 10,
}

POOLS = {
    "fruit": [
        "ycb_apple_01", "ycb_apple_02", "ycb_apple_03",
        "ycb_orange_01", "ycb_orange_02", "ycb_orange_03",
        "ycb_lemon_01", "mango_01",
    ],
    "mug": ["ycb_mug_01"],
    "box": ["ycb_cracker_box_01", "ycb_sugar_box_01"],
    "container": ["ycb_tomato_soup_can_01", "ycb_mustard_bottle_01"],
    "cube": [
        "cube_blue_01", "cube_green_01", "cube_yellow_01",
        "cube_orange_01", "cube_purple_01", "cube_pink_01",
    ],
}

FRUIT_IDS = set(POOLS["fruit"]) | {"ycb_banana_01", "ycb_banana_02", "ycb_banana_03"}


def recursive_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            found.update(recursive_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(recursive_ids(child))
    elif isinstance(value, str) and (
        value.startswith("ycb_") or value.startswith("cube_") or value == "mango_01"
    ):
        found.add(value)
    return found


def replace_id(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: replace_id(child, old, new) for key, child in value.items()}
    if isinstance(value, list):
        return [replace_id(child, old, new) for child in value]
    return new if value == old else value


def primary_group(family: dict[str, Any]) -> str:
    ids = family["variant_specs"][0]["candidate_target_ids"]
    if len(ids) > 1 or all(value in FRUIT_IDS for value in ids):
        return "fruit"
    value = ids[0]
    if value == "ycb_mug_01":
        return "mug"
    if "box" in value:
        return "box"
    if value.startswith("cube_"):
        return "cube"
    return "container"


def remap_primary_target(family: dict[str, Any], group: str) -> dict[str, Any] | None:
    candidate_ids = family["variant_specs"][0]["candidate_target_ids"]
    if len(candidate_ids) != 1:
        return copy.deepcopy(family) if group == "fruit" else None
    old = candidate_ids[0]
    used = recursive_ids(family)
    choices = [value for value in POOLS[group] if value == old or value not in used]
    if not choices:
        return None
    choices.sort(key=lambda value: hashlib.sha256(
        f"{PROTOCOL_ID}|{family['family_id']}|{group}|{value}".encode()
    ).hexdigest())
    return replace_id(copy.deepcopy(family), old, choices[0])


def prioritized_templates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    original_selector = ORIGINAL_SELECT_TEMPLATES
    selected = original_selector(payload)
    ambiguous = [
        copy.deepcopy(row) for row in selected
        if row["primary_answerability_stratum"] == "AMBIGUOUS"
    ]
    if len(ambiguous) != 35 or any(primary_group(row) != "fruit" for row in ambiguous):
        raise RuntimeError("Expected the locked 35 ambiguous fruit families")

    remaining = [row for row in selected if row["primary_answerability_stratum"] != "AMBIGUOUS"]
    assigned = list(ambiguous)
    requirements = {
        "box": TARGET_QUOTAS["box"],
        "mug": TARGET_QUOTAS["mug"],
        "container": TARGET_QUOTAS["container"],
        "cube": TARGET_QUOTAS["cube"],
        "fruit": TARGET_QUOTAS["fruit"] - len(ambiguous),
    }
    for group in ("box", "mug", "container", "cube", "fruit"):
        need = requirements[group]
        candidates = sorted(remaining, key=lambda row: hashlib.sha256(
            f"{PROTOCOL_ID}|assign|{group}|{row['family_id']}".encode()
        ).hexdigest())
        chosen: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for row in candidates:
            if group == "box" and (
                row["family_category"] != "direct_grounding"
                or row["primary_answerability_stratum"] == "AMBIGUOUS"
            ):
                continue
            mapped = remap_primary_target(row, group)
            if mapped is not None:
                chosen.append((row, mapped))
            if len(chosen) == need:
                break
        if len(chosen) != need:
            raise RuntimeError(f"Cannot allocate target group {group}: {len(chosen)}/{need}")
        chosen_ids = {id(source) for source, _ in chosen}
        remaining = [row for row in remaining if id(row) not in chosen_ids]
        assigned.extend(mapped for _, mapped in chosen)
    if remaining or len(assigned) != 200:
        raise RuntimeError(f"Target allocation incomplete: remaining={len(remaining)}")
    counts = Counter(primary_group(row) for row in assigned)
    if dict(counts) != TARGET_QUOTAS:
        raise RuntimeError(f"Target quota drift: {dict(counts)}")

    # Keep cubes as a minority even among primary relation anchors.  Replacing
    # an anchor globally inside its family preserves graph consistency across
    # the five variants while the layout is regenerated afterward.
    anchor_pool = POOLS["fruit"] + POOLS["mug"] + POOLS["container"]
    cube_anchor_count = sum(
        value.startswith("cube_")
        for row in assigned for value in row["variant_specs"][0]["anchor_ids"]
    )
    target_cube_anchors = 40
    for index, row in enumerate(list(assigned)):
        if cube_anchor_count <= target_cube_anchors:
            break
        clean_anchors = list(row["variant_specs"][0]["anchor_ids"])
        for old_anchor in clean_anchors:
            if cube_anchor_count <= target_cube_anchors:
                break
            if not old_anchor.startswith("cube_"):
                continue
            used = recursive_ids(row)
            choices = [value for value in anchor_pool if value not in used]
            if not choices:
                continue
            choices.sort(key=lambda value: hashlib.sha256(
                f"{PROTOCOL_ID}|anchor|{row['family_id']}|{old_anchor}|{value}".encode()
            ).hexdigest())
            row = replace_id(row, old_anchor, choices[0])
            assigned[index] = row
            cube_anchor_count -= 1
    if cube_anchor_count != target_cube_anchors:
        raise RuntimeError(f"Cube-anchor quota drift: {cube_anchor_count}")
    assigned.sort(key=lambda row: hashlib.sha256(
        f"{PROTOCOL_ID}|final-order|{row['family_id']}".encode()
    ).hexdigest())
    return assigned


def rewrite_for_test(manifest: dict[str, Any], plan: dict[str, Any]) -> None:
    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: walk(child) for key, child in value.items()}
        if isinstance(value, list):
            return [walk(child) for child in value]
        if isinstance(value, str):
            return value.replace("calibration", "test_iid").replace("CALIBRATION", "TEST_IID")
        return value

    rewritten_manifest = walk(manifest)
    rewritten_plan = walk(plan)
    manifest.clear(); manifest.update(rewritten_manifest)
    plan.clear(); plan.update(rewritten_plan)
    manifest.update({
        "status": "STATIC_TEST_IID_PLAN_NOT_CAPTURED",
        "scientific_status": "SEALED_TEST_IID_NOT_CAPTURED",
        "eligible_for_calibration": False,
        "eligible_for_training_or_dev": False,
        "eligible_for_test": True,
        "allowed_splits": ["test_iid"],
        "sealed_splits_not_created": ["test_ood"],
        "test_opened": False,
        "target_object_group_quota": TARGET_QUOTAS,
        "cube_policy": "PRIMARY_TARGET_5_PERCENT; PRIMARY_CUBE_ANCHORS_CAPPED_AT_40_OF_200",
    })
    manifest["counts"]["split"] = {"test_iid": 200}
    for family in manifest["families"]:
        family.update({
            "split": "test_iid",
            "eligible_for_calibration": False,
            "eligible_for_training_or_dev": False,
            "eligible_for_test": True,
            "capture_policy": "OFFICIAL_TEST_IID_ONE_SHOT_ONLY",
            "target_object_group": primary_group(family),
        })
    plan.update({
        "status": "STATIC_TEST_IID_CAPTURE_FORBIDDEN_UNTIL_PREFLIGHT_PASS",
        "scientific_status": "SEALED_TEST_IID_NOT_CAPTURED",
        "capture_output_root": OUTPUT_ROOT,
        "training_authorized": False,
        "model_inference_authorized": False,
        "calibrator_fit_authorized": False,
        "test_authorized": False,
    })
    for batch in plan["batches"]:
        batch["split_counts"] = {"test_iid": batch["family_count"]}
    for capture in plan["captures"]:
        capture["split"] = "test_iid"


def configure_base() -> None:
    base.PROTOCOL_ID = PROTOCOL_ID
    base.MASTER_SEED = MASTER_SEED
    base.FAMILY_PREFIX = FAMILY_PREFIX
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.ARCHITECTURE_FREEZE_LOCK = FREEZE_LOCK
    base.DEVELOPMENT_TEMPLATE = "old/protocol/dataset_v2_1_development_manifest.json"
    base.PARTITION = "old/protocol/dataset_expansion_v2_asset_partition.json"
    base.CAMERA_LOCK = "old/protocol/dataset_v2_relation_camera_lock_v2_1.json"
    base.DEPTH_CALIBRATION = "old/results/dataset_v2_relation_repair_20260821/audit/asset_depth_calibration.json"
    historical = tuple(
        f"old/{value}" if (ROOT / "old" / value).is_file() else value
        for value in base.DENYLIST_JSON
    )
    base.DENYLIST_JSON = tuple(dict.fromkeys((*historical,
        "old/protocol/dataset_v2_1_calibration_manifest.json",
        "old/protocol/dataset_v2_1_calibration_capture_plan.json",
        "old/datasets/roborefer_dataset_v2_1_calibration_200_20260824/family_manifest.json",
        "old/datasets/roborefer_dataset_v2_1_calibration_200_20260824/capture_plan.json",
    )))
    base.select_templates = prioritized_templates


ORIGINAL_SELECT_TEMPLATES = base.select_templates


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    configure_base()
    manifest, plan = base.build_artifacts(ROOT)
    rewrite_for_test(manifest, plan)
    return manifest, plan


def main() -> None:
    manifest, plan = build()
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PROTOCOL_DIR / "test_iid_manifest.json"
    plan_path = PROTOCOL_DIR / "test_iid_capture_plan.json"
    write_json(manifest_path, manifest)
    plan["manifest_sha256"] = sha256_file(manifest_path)
    write_json(plan_path, plan)
    print(json.dumps({
        "status": "TEST_IID_STATIC_DESIGN_CREATED",
        "families": len(manifest["families"]),
        "samples": manifest["sample_count"],
        "captures": len(plan["captures"]),
        "target_groups": dict(Counter(row["target_object_group"] for row in manifest["families"])),
        "manifest": str(manifest_path),
        "capture_plan": str(plan_path),
    }, indent=2))


if __name__ == "__main__":
    main()
