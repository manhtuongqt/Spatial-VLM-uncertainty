#!/usr/bin/env python3
"""Prepare the append-only Day-8 Gazebo Development-R2 capture.

The script has two explicit stages.  ``canary`` creates exactly one family for
each answerability x relation cell.  ``bulk`` refuses to create the 512-family
configuration unless the independently produced canary QC is PASS.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workspace.mh_pcrau_v3.day6_left_ycb_prepare import (
    BOXES, BOX_LABELS, CLUTTER, FRUIT_LABELS, ROOT, VIEW_POSE, add_inventory,
    rank_index, tie_indices,
)


REPORT = ROOT / "ketquangay/ngay_08"
RAW = REPORT / "du_lieu"
WORLD = ROOT / "workspace/mh_pcrau_v3/generated/day6_left_ycb_v5_contact_exact/left_ycb_top30_contact_exact.sdf"
GEOMETRY = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v13_sugar_apple_near/SELECTED_15_PAIR_GEOMETRY.json"
CONFIG = ROOT / "ur3/ur3_perception/config"
SOURCE_SCENES = CONFIG / "gazebo_train_uq_v2_full_r3_scenes.yaml"
SOURCE_ANNOTATIONS = CONFIG / "gazebo_train_uq_v2_full_r3_annotations.yaml"
SOURCE_GATE = CONFIG / "gazebo_train_uq_v2_full_r3_gate.yaml"
CAPTURE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
LAUNCH = ROOT / "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py"
PROTOCOL = "mh_pcrau_v3_gazebo_development_r2_v1"
NAMESPACE = "mh_pcrau_v3/gazebo_development_r2_v1"
SEED = 29092026
STATES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")
RELATIONS = ("leftmost", "rightmost", "second_from_left", "second_from_right")
FRUITS = ("ycb_apple", "ycb_orange", "mango", "ycb_pear", "ycb_plum")
NAMES = {"ycb_apple": "apple", "ycb_orange": "orange", "mango": "mango", "ycb_pear": "pear", "ycb_plum": "plum"}
CAMERAS = (
    VIEW_POSE,
    [VIEW_POSE[0] - .018, VIEW_POSE[1] + .012, VIEW_POSE[2], VIEW_POSE[3] - .008, VIEW_POSE[4], VIEW_POSE[5] + .018],
    [VIEW_POSE[0] + .018, VIEW_POSE[1] - .012, VIEW_POSE[2], VIEW_POSE[3] + .008, VIEW_POSE[4], VIEW_POSE[5] - .018],
    [VIEW_POSE[0], VIEW_POSE[1] + .016, VIEW_POSE[2] - .012, VIEW_POSE[3] - .004, VIEW_POSE[4], VIEW_POSE[5] + .010],
)
PHRASES = {
    "leftmost": ("identify the leftmost item", "find the item furthest to the left", "select the leftmost object", "return the object on the far left"),
    "rightmost": ("identify the rightmost item", "find the item furthest to the right", "select the rightmost object", "return the object on the far right"),
    "second_from_left": ("identify the second item from the left", "find the object second from left to right", "select the second-left item", "return the item in second position from the left"),
    "second_from_right": ("identify the second item from the right", "find the object second from right to left", "select the second-right item", "return the item in second position from the right"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def required_sources() -> set[str]:
    tree = ast.parse(CAPTURE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REQUIRED_SOURCE_ARTIFACTS" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("capture source inventory missing")


def token_for(stage: str, state: str, relation: str, repeat: int) -> str:
    value = f"{NAMESPACE}|{SEED}|{stage}|{state}|{relation}|{repeat}"
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def common_design(stage: str, relation: str, repeat: int) -> dict:
    """State-independent design variables prevent text/layout state shortcuts."""
    key = int(hashlib.sha256(f"{NAMESPACE}|{SEED}|{stage}|{relation}|{repeat}|common".encode()).hexdigest()[:16], 16)
    rng = random.Random(key)
    fruit_pool = list(FRUITS)
    rng.shuffle(fruit_pool)
    fruits = fruit_pool[:3]
    decoy = fruit_pool[3]
    # Wide cluster translation and scale intentionally break the relation-centroid shortcut.
    center_y = rng.uniform(.38, .76)
    spacing = rng.uniform(.115, .205)
    base_x = rng.uniform(-.34, -.11)
    return {
        "fruits": fruits,
        "decoy": decoy,
        "center_y": center_y,
        "spacing": spacing,
        "base_x": base_x,
        "prompt_variant": repeat % len(PHRASES[relation]),
        "occluder": BOXES[(repeat + RELATIONS.index(relation)) % len(BOXES)],
        "camera_stratum": repeat % len(CAMERAS),
    }


def ordered_positions(state: str, relation: str, design: dict, rng: random.Random) -> list[list[float]]:
    c, s, x = design["center_y"], design["spacing"], design["base_x"]
    if state != "AMBIGUOUS":
        ys = (c + s, c, c - s)
        xs = (x + rng.uniform(-.035, .035), x + rng.uniform(-.035, .035), x + rng.uniform(-.035, .035))
    elif relation == "leftmost":
        ys, xs = ((c + s, c + s, c - s), (x - .060, x + .060, x))
    elif relation == "rightmost":
        ys, xs = ((c + s, c - s, c - s), (x, x - .060, x + .060))
    elif relation == "second_from_left":
        ys, xs = ((c + s, c, c), (x, x - .060, x + .060))
    else:
        ys, xs = ((c, c, c - s), (x - .060, x + .060, x))
    return [[float(px), float(py), rng.uniform(-.45, .45)] for px, py in zip(xs, ys)]


def prompt(relation: str, fruits: list[str], variant: int) -> str:
    names = ", ".join(NAMES[x] for x in fruits[:-1]) + " and " + NAMES[fruits[-1]]
    return f"Among the {names}, {PHRASES[relation][variant]}."


def make_scene(stage: str, state: str, relation: str, repeat: int, geometry: dict) -> tuple[dict, dict, dict]:
    token = token_for(stage, state, relation, repeat)
    rng = random.Random(int(token, 16))
    design = common_design(stage, relation, repeat)
    fruits = design["fruits"]
    target_index = rank_index(relation)
    target = fruits[target_index]
    positions = ordered_positions(state, relation, design, rng)
    poses: dict[str, list[float]] = {}
    for index, (fruit, pose) in enumerate(zip(fruits, positions)):
        if state != "ABSENT" or index != target_index:
            poses[fruit] = pose

    # Keep three fruit-like objects and one calibrated box occlusion in every state.
    if state == "INSUFFICIENT_EVIDENCE":
        covered = target
    elif state == "FOUND":
        covered = fruits[(target_index + 1) % 3]
    elif state == "AMBIGUOUS":
        tied = {fruits[i] for i in tie_indices(relation)}
        covered = next(x for x in fruits if x not in tied)
    else:
        covered = design["decoy"]
        poses[covered] = positions[target_index]
    occluder = design["occluder"]
    calibration = geometry["pairs"][occluder][covered]
    # Projection calibration is translationally equivariant over the tabletop.
    tx, ty, _ = poses[covered]
    poses[covered] = [tx, ty, 0.0]
    poses[occluder] = [
        tx - float(calibration["forward_separation_m"]),
        ty + float(calibration["signed_lateral_offset_m"]),
        float(calibration.get("occluder_yaw_rad", 0.0)),
    ]

    # State-independent support distribution; small independent jitter only.
    support = [(-.49, .86), (-.39, .27), (-.28, .92), (-.16, .26), (-.08, .78), (-.46, .54)]
    common_rng = random.Random(int(hashlib.sha256(f"{NAMESPACE}|{stage}|{relation}|{repeat}|support".encode()).hexdigest()[:16], 16))
    common_rng.shuffle(support)
    for name, (x, y) in zip(CLUTTER, support):
        poses[name] = [x + rng.uniform(-.025, .025), y + rng.uniform(-.025, .025), rng.uniform(-.8, .8)]
    for idx, box in enumerate(BOXES):
        if box != occluder:
            poses[box] = [-.52 + .22 * idx, 1.04, 0.0]
    poses["ycb_lemon"] = [5.34, 3.0, 0.0]

    split = "canary" if stage == "canary" else ("train" if repeat < 24 else "development_validation")
    sid = f"d8r2_{stage}_{token}"
    family = f"{NAMESPACE}/{stage}/{token}"
    signature = hashlib.sha256(json.dumps(poses, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    scene = {
        "scene_id": sid,
        "scene_family_id": family,
        "layout_id": sid,
        "layout_signature_sha256": signature,
        "task_type": "development_r2_ordinal_grounding",
        "instruction": prompt(relation, fruits, design["prompt_variant"]),
        "poses": poses,
    }
    ann = {
        "state": state,
        "relation_variant": relation,
        "rank_from": "left" if relation in ("leftmost", "second_from_left") else "right",
        "rank": 1 if relation in ("leftmost", "rightmost") else 2,
        "family_id": family,
        "layout_id": sid,
        "layout_signature_sha256": signature,
        "split": split,
        "seed": int(token[:12], 16),
        "camera_stratum": design["camera_stratum"],
        "prompt_variant": design["prompt_variant"],
        "candidate_ids": [x for x in fruits if x in poses],
        "candidate_labels": [FRUIT_LABELS[x] for x in fruits if x in poses],
        "failure_tags": ["fresh_family", "development_r2", "world_v5", "lemon_excluded"],
    }
    if state == "FOUND":
        ann.update({"target_id": target, "target_label": FRUIT_LABELS[target]})
    elif state == "AMBIGUOUS":
        valid = [fruits[i] for i in tie_indices(relation)]
        ann.update({"target_id": None, "target_label": None, "valid_target_ids": valid, "valid_target_labels": [FRUIT_LABELS[x] for x in valid]})
    elif state == "ABSENT":
        ann.update({"target_id": None, "target_label": None, "requested_absent_target_id": target, "requested_absent_target_label": FRUIT_LABELS[target]})
    else:
        ann.update({
            "target_id": target,
            "target_label": FRUIT_LABELS[target],
            "occluder_ids": [occluder],
            "occluder_labels": [BOX_LABELS[occluder]],
            "insufficient_reason": "target_visible_pixels_1_to_119_due_to_controlled_ycb_occlusion",
            "projection_calibration": {
                "forward_separation_m": float(calibration["forward_separation_m"]),
                "signed_lateral_offset_m": float(calibration["signed_lateral_offset_m"]),
                "occluder_yaw_rad": float(calibration.get("occluder_yaw_rad", 0.0)),
            },
        })
    index = {
        "scene_id": sid, "family_id": family, "state": state, "relation": relation,
        "split": split, "seed": ann["seed"], "camera_stratum": ann["camera_stratum"],
        "prompt_variant": ann["prompt_variant"], "layout_signature_sha256": signature,
    }
    return scene, ann, index


def source_lock(folder: Path, protocol: str, view_pose: list[float], count: int) -> None:
    relative = lambda p: str(Path(p).resolve().relative_to(ROOT))
    sources = required_sources() | {
        relative(WORLD), relative(GEOMETRY), relative(Path(__file__)), relative(LAUNCH),
        relative(folder / "scenes.yaml"), relative(folder / "annotations.yaml"), relative(folder / "gate.yaml"),
        "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
    }
    lock = {
        "schema_version": 1,
        "protocol_id": protocol,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "source_artifact_sha256": {x: sha256(ROOT / x) for x in sorted(sources)},
        "model_inventory_sha256": "NO_MODEL_INFERENCE_DAY8_DATA_CAPTURE",
        "planned_scene_count": count,
        "view_joint_pose": view_pose,
        "world_file": relative(WORLD),
    }
    (folder / "CAPTURE_SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n")


def write_batch(folder: Path, label: str, view_pose: list[float], rows: list[dict], annotations: dict, src_scenes: dict, src_ann: dict, src_gate: dict) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    protocol = f"{PROTOCOL}_{label}"
    scenes = deepcopy(src_scenes)
    scenes.update({
        "protocol_id": protocol, "expected_scene_count": len(rows), "random_seed": SEED,
        "view_joint_pose": view_pose, "camera_frame": "top_table_camera_optical_frame", "scenes": rows,
    })
    scenes["objects"] = add_inventory(scenes["objects"])
    scenes["objects"]["ycb_lemon"]["storage_pose"] = [5.34, 3.0, 0.0]
    oracle = deepcopy(src_ann)
    oracle.update({"protocol_id": protocol, "oracle_usage": "evaluator_only_after_capture", "scenes": annotations})
    gate = deepcopy(src_gate)
    gate.update({
        "protocol_id": protocol, "parent_family_count": len(rows),
        "camera": {"frame": "top_table_camera_optical_frame", "resolution": [640, 480], "view_joint_pose": view_pose, "depth_unit": "metre", "valid_depth_range_m": [.1, 3.]},
        "policies": {**src_gate.get("policies", {}), "no_training": True, "no_model_inference": True, "no_materialization_before_qc": True},
    })
    for name, payload in (("scenes.yaml", scenes), ("annotations.yaml", oracle), ("gate.yaml", gate)):
        (folder / name).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=180))
    source_lock(folder, protocol, view_pose, len(rows))


def write_contract() -> None:
    contract = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED_BEFORE_CAPTURE",
        "day": 8,
        "objective": "Create Gazebo Development-R2 only; Candidate R3 training is prohibited on Day 8.",
        "namespace": NAMESPACE,
        "seed_root": SEED,
        "world_sha256": sha256(WORLD),
        "geometry_sha256": sha256(GEOMETRY),
        "required_support": {"families": 512, "per_state": 128, "per_cell": 32, "train": 384, "development_validation": 128},
        "hard_shortcut_gates": {"text_only_answerability_macro_f1_max": .35, "rgb_thumbnail_answerability_macro_f1_max": .70, "depth_thumbnail_answerability_macro_f1_max": .70, "relation_centroid_hit_at_0_05_max": .60},
        "canary_policy": "Exactly 16 families, one per answerability x relation cell; bulk preparation forbidden until CANARY_QC status PASS.",
        "prohibited": ["Candidate R3 training", "OOF/S1b", "G3", "calibration", "Test", "robot final"],
    }
    (REPORT / "DEVELOPMENT_R2_CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")


def prepare(stage: str) -> None:
    geometry = json.loads(GEOMETRY.read_text())
    if geometry.get("status") != "LOCKED":
        raise RuntimeError("projection geometry is not locked")
    src_scenes = yaml.safe_load(SOURCE_SCENES.read_text())
    src_ann = yaml.safe_load(SOURCE_ANNOTATIONS.read_text())
    src_gate = yaml.safe_load(SOURCE_GATE.read_text())
    if stage == "canary":
        if REPORT.exists():
            raise FileExistsError(f"append-only Day-8 report already exists: {REPORT}")
        (REPORT / "bang").mkdir(parents=True)
        (REPORT / "anh").mkdir()
        RAW.mkdir()
        write_contract()
        rows, anns, index = [], {}, []
        for si, state in enumerate(STATES):
            for ri, relation in enumerate(RELATIONS):
                row, ann, item = make_scene("canary", state, relation, ri, geometry)
                rows.append(row); anns[row["scene_id"]] = ann; index.append(item)
        folder = RAW / "canary"
        write_batch(folder, "canary", CAMERAS[0], rows, anns, src_scenes, src_ann, src_gate)
        (folder / "DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in index))
        print(json.dumps({"status": "CANARY_READY", "families": len(index), "path": str(folder.relative_to(ROOT))}, indent=2))
        return

    canary_qc = REPORT / "CANARY_QC.json"
    if not canary_qc.is_file() or json.loads(canary_qc.read_text()).get("status") != "PASS":
        raise RuntimeError("bulk is sealed until CANARY_QC.json has status PASS")
    bulk = RAW / "bulk"
    if bulk.exists():
        raise FileExistsError(f"append-only bulk design already exists: {bulk}")
    bulk.mkdir()
    batches = {i: ([], {}, []) for i in range(4)}
    all_index = []
    for state in STATES:
        for relation in RELATIONS:
            for repeat in range(32):
                row, ann, item = make_scene("bulk", state, relation, repeat, geometry)
                batch = repeat % 4
                batches[batch][0].append(row); batches[batch][1][row["scene_id"]] = ann; batches[batch][2].append(item)
                all_index.append(item)
    for batch, (rows, anns, index) in batches.items():
        random.Random(SEED + batch).shuffle(rows)
        folder = bulk / f"batch_{batch}"
        write_batch(folder, f"bulk_b{batch}", CAMERAS[batch], rows, anns, src_scenes, src_ann, src_gate)
        (folder / "DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in index))
    (bulk / "DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in all_index))
    cells = Counter((x["state"], x["relation"]) for x in all_index)
    summary = {
        "schema_version": 1, "status": "PASS_CAPTURE_NOT_STARTED", "families": len(all_index),
        "states": dict(Counter(x["state"] for x in all_index)),
        "relations": dict(Counter(x["relation"] for x in all_index)),
        "splits": dict(Counter(x["split"] for x in all_index)),
        "cells": {f"{s}|{r}": cells[(s, r)] for s in STATES for r in RELATIONS},
        "unique_family_ids": len({x["family_id"] for x in all_index}),
        "unique_seeds": len({x["seed"] for x in all_index}),
        "unique_layouts": len({x["layout_signature_sha256"] for x in all_index}),
    }
    (bulk / "DESIGN_STATIC_QC.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("canary", "bulk"))
    prepare(parser.parse_args().stage)
