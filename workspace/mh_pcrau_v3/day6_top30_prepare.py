#!/usr/bin/env python3
"""Prepare a fresh top-oblique 30-degree Anti-Shortcut Val capture.

The occlusion geometry is physical: a YCB box is closer to the camera ray than
the fruit, and their collision footprints are separated along the viewing
direction.  The lateral offset controls visible evidence; no two objects are
co-located.  This script creates worlds/configs/locks only and never opens a
model checkpoint.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import re

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ur3/ur3_perception/config"
SOURCE_WORLD = ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf"
OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_top30_v1"
GENERATED = ROOT / "workspace/mh_pcrau_v3/generated/day6_top30"
PARENT_LOCK = ROOT / "protocol/MH_PCRAU_V3_ANTI_SHORTCUT_VAL_V1_LOCK.json"
SOURCE_SCENES = CONFIG / "gazebo_train_uq_v2_full_r3_scenes.yaml"
SOURCE_ANNOTATIONS = CONFIG / "gazebo_train_uq_v2_full_r3_annotations.yaml"
SOURCE_GATE = CONFIG / "gazebo_train_uq_v2_full_r3_gate.yaml"
CAPTURE_NODE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
LAUNCH = ROOT / "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py"
PROTOCOL = "mh_pcrau_v3_anti_shortcut_top30_v1"
SEED = 25092040

STATES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")
RELATIONS = ("leftmost", "rightmost", "second_from_left", "second_from_right")
FRUITS = ("ycb_apple", "ycb_orange", "mango")
LABELS = {"ycb_apple": 26, "ycb_orange": 27, "mango": 29}
FRUIT_NAMES = {"ycb_apple": "apple", "ycb_orange": "orange", "mango": "mango"}
BOXES = ("ycb_cracker_box", "ycb_sugar_box")
BOX_FORWARD_HALF = {"ycb_cracker_box": 0.071720 / 2, "ycb_sugar_box": 0.049450 / 2}
BOX_SEPARATION = {"ycb_cracker_box": 0.090, "ycb_sugar_box": 0.080}
BOX_LATERAL = {"ycb_cracker_box": 0.099, "ycb_sugar_box": 0.064}
FRUIT_RADIUS = {"ycb_apple": 0.03771, "ycb_orange": 0.03701, "mango": 0.030}
BASE_JOINT_POSE = [1.3815, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601]
AZIMUTHS = (0.0, math.pi / 2, math.pi, -math.pi / 2)
CAMERA_CENTER = (0.0, 0.45)
CAMERA_HEIGHT = 1.80
CAMERA_PITCH = math.radians(60.0)  # optical axis is 30 degrees from vertical
CAMERA_RADIUS = CAMERA_HEIGHT / math.tan(CAMERA_PITCH)
PROMPTS = {
    "leftmost": (
        "Among the apple, orange, and mango, identify the leftmost fruit in the image.",
        "Point to the fruit furthest left among the apple, orange, and mango.",
        "Select the far-left fruit from the apple, orange, and mango.",
        "Which fruit is leftmost: the apple, orange, or mango?",
    ),
    "rightmost": (
        "Among the apple, orange, and mango, identify the rightmost fruit in the image.",
        "Point to the fruit furthest right among the apple, orange, and mango.",
        "Select the far-right fruit from the apple, orange, and mango.",
        "Which fruit is rightmost: the apple, orange, or mango?",
    ),
    "second_from_left": (
        "Identify the second fruit from the left among the apple, orange, and mango.",
        "Point to the fruit in the middle when ordered left to right.",
        "Select the second-from-left fruit from the apple, orange, and mango.",
        "Which fruit ranks second from the left: apple, orange, or mango?",
    ),
    "second_from_right": (
        "Identify the second fruit from the right among the apple, orange, and mango.",
        "Point to the fruit in the middle when ordered right to left.",
        "Select the second-from-right fruit from the apple, orange, and mango.",
        "Which fruit ranks second from the right: apple, orange, or mango?",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def required_capture_sources() -> set[str]:
    tree = ast.parse(CAPTURE_NODE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_SOURCE_ARTIFACTS"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("capture source inventory not found")


def camera_pose(azimuth: float) -> tuple[float, float, float, float, float, float]:
    hx, hy = math.cos(azimuth), math.sin(azimuth)
    return (
        CAMERA_CENTER[0] - CAMERA_RADIUS * hx,
        CAMERA_CENTER[1] - CAMERA_RADIUS * hy,
        CAMERA_HEIGHT,
        0.0,
        CAMERA_PITCH,
        azimuth,
    )


def make_worlds() -> list[Path]:
    text = SOURCE_WORLD.read_text(encoding="utf-8")
    old_pose = "<pose>0 0.45 1.80 0 1.57079632679 0</pose>"
    if text.count(old_pose) != 1:
        raise RuntimeError("top camera pose marker drifted")
    sensor_end = """        </sensor>
      </link>
    </model>

    <model name=\"red_cube\">"""
    if text.count(sensor_end) != 1:
        raise RuntimeError("top camera sensor marker drifted")
    semantic = """        </sensor>
        <sensor name=\"top_table_evaluation_labels\" type=\"segmentation\">
          <camera>
            <segmentation_type>semantic</segmentation_type>
            <horizontal_fov>1.10</horizontal_fov>
            <image><width>640</width><height>480</height></image>
            <clip><near>0.10</near><far>3.0</far></clip>
            <optical_frame_id>top_table_camera_optical_frame</optical_frame_id>
          </camera>
          <always_on>true</always_on><update_rate>30</update_rate><visualize>false</visualize>
          <topic>/top_table_camera/evaluation_labels</topic>
          <gz_frame_id>top_table_camera_optical_frame</gz_frame_id>
        </sensor>
      </link>
    </model>

    <model name=\"red_cube\">"""
    GENERATED.mkdir(parents=True, exist_ok=True)
    worlds = []
    for batch, azimuth in enumerate(AZIMUTHS):
        pose = camera_pose(azimuth)
        pose_text = "<pose>" + " ".join(f"{value:.12f}" for value in pose) + "</pose>"
        output = text.replace(old_pose, pose_text).replace(sensor_end, semantic)
        path = GENERATED / f"ur3_pick_place_uq_top30_az{batch}.sdf"
        path.write_text(output, encoding="utf-8")
        worlds.append(path)
    return worlds


def xy(center: tuple[float, float], h: tuple[float, float], p: tuple[float, float],
       forward: float, lateral: float) -> tuple[float, float]:
    return (
        center[0] + forward * h[0] + lateral * p[0],
        center[1] + forward * h[1] + lateral * p[1],
    )


def storage_objects(source: dict) -> dict:
    objects = deepcopy(source["objects"])
    for index, item in enumerate(objects.values()):
        item["storage_pose"] = [3.0 + 0.15 * index, 3.0, 0.0]
    return objects


def ordered_positions(state: str, relation: str) -> list[tuple[float, float]]:
    # Tuple = (forward along camera ray, lateral; positive lateral is image-left).
    if state != "AMBIGUOUS":
        return [(-0.035, 0.28), (0.025, 0.0), (0.075, -0.28)]
    if relation == "leftmost":
        return [(-0.10, 0.19), (0.12, 0.19), (0.02, -0.28)]
    if relation == "rightmost":
        return [(0.02, 0.28), (-0.10, -0.19), (0.12, -0.19)]
    if relation == "second_from_left":
        return [(0.02, 0.28), (-0.10, -0.10), (0.12, -0.10)]
    return [(-0.10, 0.10), (0.12, 0.10), (0.02, -0.28)]


def target_index(relation: str) -> int:
    return {"leftmost": 0, "rightmost": 2, "second_from_left": 1,
            "second_from_right": 1}[relation]


def tie_indices(relation: str) -> tuple[int, int]:
    return {"leftmost": (0, 1), "rightmost": (1, 2),
            "second_from_left": (1, 2), "second_from_right": (0, 1)}[relation]


def scene_geometry(batch: int, state: str, relation: str, token: str, repetition: int):
    rng = random.Random(int(token, 16))
    azimuth = AZIMUTHS[batch]
    h = (math.cos(azimuth), math.sin(azimuth))
    p = (-h[1], h[0])
    center = (
        CAMERA_CENTER[0] + rng.uniform(-0.035, 0.035),
        CAMERA_CENTER[1] + rng.uniform(-0.035, 0.035),
    )
    fruits = list(FRUITS)
    rng.shuffle(fruits)
    positions = ordered_positions(state, relation)
    poses: dict[str, list[float]] = {}
    for fruit, (forward, lateral) in zip(fruits, positions):
        if state == "ABSENT" and fruits.index(fruit) == target_index(relation):
            continue
        x_value, y_value = xy(center, h, p, forward, lateral)
        poses[fruit] = [x_value, y_value, rng.uniform(-0.35, 0.35)]

    target = fruits[target_index(relation)]
    occluder = BOXES[(repetition + batch) % 2]
    other_box = BOXES[1 - BOXES.index(occluder)]
    if state == "INSUFFICIENT_EVIDENCE":
        tx, ty, _ = poses[target]
        separation = BOX_SEPARATION[occluder]
        lateral = BOX_LATERAL[occluder] * (-1.0 if repetition % 2 else 1.0)
        poses[occluder] = [
            tx - separation * h[0] + lateral * p[0],
            ty - separation * h[1] + lateral * p[1],
            azimuth,
        ]
        clearance = separation - (BOX_FORWARD_HALF[occluder] + FRUIT_RADIUS[target])
        if clearance < 0.008:
            raise RuntimeError(f"occluder/fruit physical clearance too small: {clearance}")
    else:
        clearance = None
        x_value, y_value = xy(center, h, p, 0.33, 0.34)
        poses[occluder] = [x_value, y_value, azimuth]
    # Dense but separated background clutter: second box, can, bottle, banana.
    for name, forward, lateral, yaw_delta in (
        (other_box, 0.34, -0.34, math.pi / 2),
        ("ycb_tomato_soup_can", -0.31, 0.31, 0.0),
        ("ycb_mustard_bottle", -0.33, -0.30, 0.0),
        ("ycb_banana", 0.31, 0.02, math.pi / 2),
    ):
        x_value, y_value = xy(center, h, p, forward, lateral)
        poses[name] = [x_value, y_value, azimuth + yaw_delta]
    return poses, fruits, target, occluder, clearance


def annotation(state: str, relation: str, fruits: list[str], target: str,
               occluder: str, family_id: str, scene_id: str, seed: int,
               repetition: int) -> dict:
    rank_from = "left" if relation in ("leftmost", "second_from_left") else "right"
    rank = 1 if relation in ("leftmost", "rightmost") else 2
    row = {
        "state": state,
        "target_id": target if state != "AMBIGUOUS" else None,
        "target_label": LABELS[target] if state != "AMBIGUOUS" else None,
        "candidate_ids": fruits if state != "ABSENT" else [],
        "candidate_labels": [LABELS[name] for name in fruits] if state != "ABSENT" else [],
        "rank_from": rank_from,
        "rank": rank,
        "relation_variant": relation,
        "target_category": FRUIT_NAMES[target],
        "failure_tags": ["fruit_identity", "top30_oblique", "dense_ycb_clutter"],
        "family_id": family_id,
        "layout_id": scene_id,
        "split": "anti_shortcut_top30_v1",
        "seed": seed,
        "template_provenance": "preregistered_top30_geometry_only_no_model_access",
        "cell_repetition": repetition,
    }
    if state == "AMBIGUOUS":
        indices = tie_indices(relation)
        valid = [fruits[index] for index in indices]
        row.update({
            "valid_target_ids": valid,
            "valid_target_labels": [LABELS[name] for name in valid],
            "target_category": "fruit_horizontal_tie",
            "failure_tags": row["failure_tags"] + ["horizontal_tie"],
        })
    elif state == "ABSENT":
        context = [name for name in fruits if name != target]
        row.update({"context_ids": context, "context_labels": [LABELS[name] for name in context],
                    "failure_tags": row["failure_tags"] + ["absent_target"]})
    elif state == "INSUFFICIENT_EVIDENCE":
        row.update({"occluder_ids": [occluder], "occluder_labels": [21 if occluder == "ycb_cracker_box" else 22],
                    "failure_tags": row["failure_tags"] + ["physical_front_back_occlusion", "non_intersecting_3d"]})
    return row


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only output exists: {OUT}")
    parent = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    if parent.get("status") != "FROZEN_BEFORE_CAPTURE_AND_MODEL_INFERENCE":
        raise RuntimeError("parent anti-shortcut lock is not frozen")
    source_scenes = yaml.safe_load(SOURCE_SCENES.read_text(encoding="utf-8"))
    source_annotations = yaml.safe_load(SOURCE_ANNOTATIONS.read_text(encoding="utf-8"))
    source_gate = yaml.safe_load(SOURCE_GATE.read_text(encoding="utf-8"))
    worlds = make_worlds()
    objects = storage_objects(source_scenes)
    OUT.mkdir(parents=True)
    design_rows = []
    signatures = set()
    minimum_clearance = float("inf")
    for batch in range(4):
        folder = OUT / f"batch_{batch}"
        folder.mkdir()
        rows = []
        annotations = {}
        cells = Counter()
        for state in STATES:
            for relation in RELATIONS:
                for repetition in range(4):
                    opaque = hashlib.sha256(
                        f"{PROTOCOL}|{SEED}|{batch}|{state}|{relation}|{repetition}".encode()
                    ).hexdigest()[:16]
                    scene_id = f"top30_{opaque}"
                    family_id = f"mh_pcrau_v3/anti_shortcut_top30_v1/{opaque}"
                    poses, fruits, target, occluder, clearance = scene_geometry(
                        batch, state, relation, opaque, repetition
                    )
                    signature = digest({name: poses.get(name, item["storage_pose"])
                                        for name, item in sorted(objects.items())})
                    if signature in signatures:
                        raise RuntimeError("duplicate layout signature")
                    signatures.add(signature)
                    if clearance is not None:
                        minimum_clearance = min(minimum_clearance, clearance)
                    instruction = PROMPTS[relation][(repetition + batch) % 4]
                    if re.search(r"FOUND|AMBIGUOUS|ABSENT|INSUFFICIENT", instruction, re.I):
                        raise RuntimeError("answerability leakage in prompt")
                    ann = annotation(state, relation, fruits, target, occluder,
                                     family_id, scene_id, int(opaque[:8], 16), repetition)
                    ann.update({
                        "layout_signature_sha256": signature,
                        "camera_stratum": batch,
                        "camera_azimuth_deg": round(math.degrees(AZIMUTHS[batch])),
                        "camera_elevation_from_table_deg": 30,
                        "object_composition_stratum": (2 * repetition + batch) % 8,
                    })
                    rows.append({
                        "scene_id": scene_id,
                        "scene_family_id": family_id,
                        "layout_id": scene_id,
                        "layout_signature_sha256": signature,
                        "task_type": "horizontal_ordinal_ranking_answerability",
                        "instruction": instruction,
                        "poses": poses,
                    })
                    annotations[scene_id] = ann
                    cells[(state, relation)] += 1
                    design_rows.append({
                        "scene_id": scene_id, "family_id": family_id, "batch": batch,
                        "state": state, "relation": relation, "occluder": occluder if state == "INSUFFICIENT_EVIDENCE" else None,
                        "layout_signature_sha256": signature,
                    })
        if len(rows) != 64 or set(cells.values()) != {4}:
            raise RuntimeError(f"unbalanced batch {batch}: {cells}")
        random.Random(SEED + batch).shuffle(rows)
        pid = f"{PROTOCOL}_b{batch}"
        scene_config = deepcopy(source_scenes)
        scene_config.update({
            "protocol_id": pid, "expected_scene_count": 64, "random_seed": SEED + batch,
            "view_joint_pose": BASE_JOINT_POSE, "camera_frame": "top_table_camera_optical_frame",
            "objects": objects, "scenes": rows,
        })
        oracle = deepcopy(source_annotations)
        oracle.update({"protocol_id": pid, "oracle_usage": "evaluation_only_after_prediction_lock",
                       "scenes": annotations})
        gate = deepcopy(source_gate)
        gate.update({
            "protocol_id": pid, "parent_family_count": 64,
            "split_parent_family_count": {"anti_shortcut_top30_v1": 64},
            "state_quota": {"anti_shortcut_top30_v1": {state: 16 for state in STATES}},
            "relation_variant_quota": {"anti_shortcut_top30_v1": {relation: 16 for relation in RELATIONS}},
            "state_relation_cell_quota": {"anti_shortcut_top30_v1": 4},
            "camera": {"frame": "top_table_camera_optical_frame", "base_frame": "base_link",
                       "resolution": [640, 480], "top_oblique_degrees_from_vertical": 30,
                       "azimuth_degrees": round(math.degrees(AZIMUTHS[batch])),
                       "view_joint_pose": BASE_JOINT_POSE, "depth_unit": "metre",
                       "valid_depth_range_m": [0.1, 3.0]},
            "physical_occlusion_rule": {"occluder_types": list(BOXES), "target_types": list(FRUITS),
                                         "minimum_3d_footprint_clearance_m": 0.008,
                                         "require_occluder_closer_along_camera_ray": True,
                                         "co_location_forbidden": True},
            "policies": {**source_gate["policies"], "no_training": True,
                         "no_model_inference": True, "no_materialization_before_qc": True},
        })
        for name, payload in (("scenes.yaml", scene_config), ("annotations.yaml", oracle), ("gate.yaml", gate)):
            (folder / name).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140), encoding="utf-8")
        common = required_capture_sources() | {
            str(SOURCE_WORLD.relative_to(ROOT)), str(worlds[batch].relative_to(ROOT)),
            str(Path(__file__).relative_to(ROOT)), str(LAUNCH.relative_to(ROOT)),
            "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
            str(PARENT_LOCK.relative_to(ROOT)),
            str((folder / "scenes.yaml").relative_to(ROOT)),
            str((folder / "annotations.yaml").relative_to(ROOT)),
            str((folder / "gate.yaml").relative_to(ROOT)),
        }
        lock = {
            "schema_version": 1, "protocol_id": pid,
            "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "workspace_root": str(ROOT),
            "source_artifact_sha256": {name: sha256(ROOT / name) for name in sorted(common)},
            "model_inventory_sha256": "NO_MODEL_INFERENCE_ANTI_SHORTCUT_TOP30_CAPTURE",
            "capture_role": "INDEPENDENT_DEVELOPMENT_CHALLENGE",
            "planned_scene_count": 64, "view_joint_pose": BASE_JOINT_POSE,
            "world_file": str(worlds[batch].relative_to(ROOT)),
            "camera_design": {"degrees_from_vertical": 30,
                              "azimuth_degrees": round(math.degrees(AZIMUTHS[batch]))},
            "seals": {"calibration": True, "test_iid": True, "test_ood": True},
        }
        (folder / "CAPTURE_SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    index = OUT / "DESIGN_INDEX.jsonl"
    index.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in design_rows), encoding="utf-8")
    states = Counter(row["state"] for row in design_rows)
    relations = Counter(row["relation"] for row in design_rows)
    result = {
        "status": "DESIGN_STATIC_QC_PASS_CAPTURE_NOT_STARTED",
        "protocol_id": PROTOCOL, "families": len(design_rows),
        "state_counts": dict(states), "relation_counts": dict(relations),
        "camera_strata": 4, "camera_degrees_from_vertical": 30,
        "dense_active_objects_per_scene": 7,
        "physical_occlusion": "YCB box in front of fruit along camera ray; no co-location",
        "minimum_preregistered_occluder_target_clearance_m": minimum_clearance,
        "model_opened": False, "capture_started": False,
        "design_index_sha256": sha256(index),
        "world_sha256": {str(i): sha256(path) for i, path in enumerate(worlds)},
    }
    (OUT / "DESIGN_STATIC_QC.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
