#!/usr/bin/env python3
"""Prepare fresh left-of-UR3 top-oblique Anti-Shortcut validation captures.

All active objects live at x < 0 (robot-left half of the bench).  The
INSUFFICIENT_EVIDENCE state uses a physically separated YCB box-like object
in front of the requested fruit along the fixed camera ray.  Other YCB items
are deliberately retained as dense clutter, never silently used as oracle
inputs or primary occluders.
"""

from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE_WORLD = ROOT / "workspace/mh_pcrau_v3/generated/day6_top30_canary_v3/low_top30_canary.sdf"
OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_v3"
WORLD = ROOT / "workspace/mh_pcrau_v3/generated/day6_left_ycb_v3/left_ycb_top30.sdf"
CONFIG = ROOT / "ur3/ur3_perception/config"
SOURCE_SCENES = CONFIG / "gazebo_train_uq_v2_full_r3_scenes.yaml"
SOURCE_ANNOTATIONS = CONFIG / "gazebo_train_uq_v2_full_r3_annotations.yaml"
SOURCE_GATE = CONFIG / "gazebo_train_uq_v2_full_r3_gate.yaml"
CAPTURE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
LAUNCH = ROOT / "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py"
PROTOCOL = "mh_pcrau_v3_anti_shortcut_left_ycb_v3"
SEED = 25092046
VIEW_POSE = [1.3815, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601]
STATES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")
RELATIONS = ("leftmost", "rightmost", "second_from_left", "second_from_right")

# Primary occluders are the YCB box-like containers.  Bleach cleaner has a
# box collision footprint in this world revision; non-box items stay clutter.
BOXES = ("ycb_cracker_box", "ycb_sugar_box", "ycb_bleach_cleanser")
BOX_LABELS = {"ycb_cracker_box": 21, "ycb_sugar_box": 22, "ycb_bleach_cleanser": 35}
BOX_X_HALF = {"ycb_cracker_box": .03586, "ycb_sugar_box": .024725, "ycb_bleach_cleanser": .040}
FRUITS = ("ycb_apple", "ycb_orange", "mango", "ycb_lemon", "ycb_pear", "ycb_plum")
FRUIT_LABELS = {"ycb_apple": 26, "ycb_orange": 27, "mango": 29, "ycb_lemon": 32, "ycb_pear": 33, "ycb_plum": 34}
FRUIT_NAMES = {"ycb_apple": "apple", "ycb_orange": "orange", "mango": "mango", "ycb_lemon": "lemon", "ycb_pear": "pear", "ycb_plum": "plum"}
FRUIT_RADIUS = {"ycb_apple": .03771, "ycb_orange": .03701, "mango": .030, "ycb_lemon": .034, "ycb_pear": .040, "ycb_plum": .028}
CLUTTER = ("ycb_tomato_soup_can", "ycb_mustard_bottle", "ycb_tuna_fish_can", "ycb_banana", "ycb_power_drill", "ycb_mug")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def required_sources() -> set[str]:
    tree = ast.parse(CAPTURE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REQUIRED_SOURCE_ARTIFACTS" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("required capture inventory missing")


def ycb_model(name: str, label: int, mesh: str, collision: str, z: float) -> str:
    root = (ROOT / "ur3/ur_simulation_gz/models/ycb" / mesh / f"{mesh}.obj").resolve().as_uri()
    return f'''    <model name="{name}">
      <pose>3 3 {z:.6f} 0 0 0</pose>
      <link name="object_link">
        <inertial><mass>0.14</mass><inertia><ixx>0.0005</ixx><iyy>0.0005</iyy><izz>0.0005</izz></inertia></inertial>
        <collision name="collision"><geometry>{collision}</geometry><surface><friction><ode><mu>1.1</mu><mu2>1.1</mu2></ode></friction></surface></collision>
        <visual name="ycb_visual"><geometry><mesh><uri>{root}</uri></mesh></geometry></visual>
      </link>
      <plugin filename="ignition-gazebo-pose-publisher-system" name="gz::sim::systems::PosePublisher"><publish_link_pose>false</publish_link_pose><publish_collision_pose>false</publish_collision_pose><publish_visual_pose>false</publish_visual_pose><publish_nested_model_pose>false</publish_nested_model_pose><use_pose_vector_msg>true</use_pose_vector_msg><static_publisher>false</static_publisher><update_frequency>15</update_frequency></plugin>
      <plugin filename="ignition-gazebo-label-system" name="ignition::gazebo::systems::Label"><label>{label}</label></plugin>
    </model>
'''


def make_world() -> None:
    text = BASE_WORLD.read_text(encoding="utf-8")
    marker = '    <model name="mango">'
    if text.count(marker) != 1:
        raise RuntimeError("world insertion marker drifted")
    # Extra YCB inventory: fruit diversity, one box-like bleach container, and
    # harmless tabletop clutter. All mesh URIs are absolute (base world fix).
    additions = "".join((
        ycb_model("ycb_tuna_fish_can", 31, "007_tuna_fish_can", "<cylinder><radius>0.043</radius><length>0.035</length></cylinder>", .018),
        ycb_model("ycb_lemon", 32, "014_lemon", "<sphere><radius>0.034</radius></sphere>", .034),
        ycb_model("ycb_pear", 33, "016_pear", "<sphere><radius>0.040</radius></sphere>", .040),
        ycb_model("ycb_plum", 34, "018_plum", "<sphere><radius>0.028</radius></sphere>", .028),
        ycb_model("ycb_bleach_cleanser", 35, "021_bleach_cleanser", "<box><size>0.080 0.090 0.240</size></box>", .120),
        ycb_model("ycb_mug", 36, "025_mug", "<cylinder><radius>0.045</radius><length>0.090</length></cylinder>", .045),
    ))
    WORLD.parent.mkdir(parents=True, exist_ok=True)
    WORLD.write_text(text.replace(marker, additions + marker), encoding="utf-8")


def add_inventory(objects: dict) -> dict:
    data = deepcopy(objects)
    extra = {
        "ycb_tuna_fish_can": .018, "ycb_lemon": .034, "ycb_pear": .040,
        "ycb_plum": .028, "ycb_bleach_cleanser": .120, "ycb_mug": .045,
    }
    for idx, (name, z) in enumerate(extra.items()):
        data[name] = {"z": z, "storage_pose": [3.5 + .15 * idx, 3.0, 0.0]}
    for idx, value in enumerate(data.values()):
        value["storage_pose"] = [3.0 + .13 * idx, 3.0, 0.0]
    return data


def rank_index(relation: str) -> int:
    return {"leftmost": 0, "rightmost": 2, "second_from_left": 1, "second_from_right": 1}[relation]


def tie_indices(relation: str) -> tuple[int, int]:
    return {"leftmost": (0, 1), "rightmost": (1, 2), "second_from_left": (1, 2), "second_from_right": (0, 1)}[relation]


def relation_positions(state: str, relation: str) -> list[tuple[float, float]]:
    # (x along camera ray, y screen-lateral).  Larger y maps image-left here.
    if state != "AMBIGUOUS":
        return [(-.28, .82), (-.22, .56), (-.16, .30)]
    if relation == "leftmost": return [(-.30, .69), (-.18, .69), (-.22, .31)]
    if relation == "rightmost": return [(-.22, .83), (-.30, .43), (-.18, .43)]
    if relation == "second_from_left": return [(-.22, .83), (-.30, .56), (-.18, .56)]
    return [(-.30, .56), (-.18, .56), (-.22, .30)]


def scene_layout(batch: int, state: str, relation: str, repetition: int, token: str) -> tuple[dict, list[str], str, str | None, float | None]:
    rng = random.Random(int(token[:12], 16))
    fruits = list(FRUITS)
    rng.shuffle(fruits)
    fruits = fruits[:3]
    target = fruits[rank_index(relation)]
    poses: dict[str, list[float]] = {}
    for index, (name, (x, y)) in enumerate(zip(fruits, relation_positions(state, relation))):
        if state == "ABSENT" and index == rank_index(relation):
            continue
        poses[name] = [x + rng.uniform(-.012, .012), y + rng.uniform(-.010, .010), rng.uniform(-.35, .35)]
    occluder = None
    clearance = None
    if state == "INSUFFICIENT_EVIDENCE":
        occluder = BOXES[(batch + repetition) % len(BOXES)]
        tx, ty, _ = poses[target]
        dx = .102 if occluder == "ycb_cracker_box" else (.092 if occluder == "ycb_bleach_cleanser" else .086)
        # The small but nonzero lateral displacement avoids co-location while
        # retaining a projected overlap. Positive clearance is explicit.
        poses[occluder] = [tx - dx, ty + (.006 if repetition % 2 else -.006), 0.0]
        clearance = dx - (BOX_X_HALF[occluder] + FRUIT_RADIUS[target])
        if clearance < .008:
            raise RuntimeError("occluder/fruit clearance below preregistered minimum")
    # Spread unused YCB items around the left half. No clutter item intersects
    # a candidate; box-like clutter stays at least 0.18m from fruit centers.
    fixed = [(-.65, .25), (-.63, .91), (-.48, .16), (-.46, 1.01), (-.08, .90), (-.06, .18)]
    for name, (x, y) in zip(CLUTTER, fixed):
        poses[name] = [x + rng.uniform(-.012, .012), y + rng.uniform(-.010, .010), rng.uniform(-.3, .3)]
    for idx, box in enumerate(BOXES):
        if box != occluder:
            poses[box] = [[-.62, -.47, -.11][idx], [.53, .73, .98][idx], 0.0]
    return poses, fruits, target, occluder, clearance


def prompt(relation: str, fruits: list[str]) -> str:
    names = ", ".join(FRUIT_NAMES[x] for x in fruits[:-1]) + " and " + FRUIT_NAMES[fruits[-1]]
    phrases = {"leftmost": "leftmost", "rightmost": "rightmost", "second_from_left": "second fruit from the left", "second_from_right": "second fruit from the right"}
    return f"Among the {names}, identify the {phrases[relation]} fruit in the image."


def main() -> None:
    if OUT.exists(): raise FileExistsError(f"append-only output exists: {OUT}")
    make_world()
    src_scenes = yaml.safe_load(SOURCE_SCENES.read_text(encoding="utf-8"))
    src_ann = yaml.safe_load(SOURCE_ANNOTATIONS.read_text(encoding="utf-8"))
    src_gate = yaml.safe_load(SOURCE_GATE.read_text(encoding="utf-8"))
    objects = add_inventory(src_scenes["objects"])
    OUT.mkdir(parents=True)
    all_rows, signatures = [], set()
    min_clearance = float("inf")
    for batch in range(4):
        folder = OUT / f"batch_{batch}"; folder.mkdir()
        rows, annotation, cells = [], {}, Counter()
        for state in STATES:
            for relation in RELATIONS:
                for repeat in range(4):
                    token = hashlib.sha256(f"{PROTOCOL}|{SEED}|{batch}|{state}|{relation}|{repeat}".encode()).hexdigest()[:16]
                    sid = f"leftycb_{token}"; family = f"mh_pcrau_v3/anti_shortcut_left_ycb_v1/{token}"
                    poses, fruits, target, box, clearance = scene_layout(batch, state, relation, repeat, token)
                    signature = hashlib.sha256(json.dumps(poses, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    if signature in signatures: raise RuntimeError("duplicate layout")
                    signatures.add(signature)
                    if clearance is not None: min_clearance = min(min_clearance, clearance)
                    rows.append({"scene_id": sid, "scene_family_id": family, "layout_id": sid, "layout_signature_sha256": signature, "task_type": "left_ycb_ordinal_grounding", "instruction": prompt(relation, fruits), "poses": poses})
                    rank = 1 if relation in ("leftmost", "rightmost") else 2
                    ann = {"state": state, "target_id": target if state != "AMBIGUOUS" else None, "target_label": FRUIT_LABELS[target] if state != "AMBIGUOUS" else None, "candidate_ids": fruits if state != "ABSENT" else [], "candidate_labels": [FRUIT_LABELS[x] for x in fruits] if state != "ABSENT" else [], "rank_from": "left" if relation in ("leftmost", "second_from_left") else "right", "rank": rank, "relation_variant": relation, "family_id": family, "layout_id": sid, "layout_signature_sha256": signature, "split": "anti_shortcut_left_ycb_v1", "camera_stratum": batch, "failure_tags": ["left_of_robot", "top30_oblique", "diverse_ycb_clutter"]}
                    if state == "AMBIGUOUS":
                        valid = [fruits[i] for i in tie_indices(relation)]
                        ann.update({"valid_target_ids": valid, "valid_target_labels": [FRUIT_LABELS[x] for x in valid], "failure_tags": ann["failure_tags"] + ["horizontal_tie"]})
                    elif state == "ABSENT":
                        context = [x for x in fruits if x != target]
                        ann.update({"context_ids": context, "context_labels": [FRUIT_LABELS[x] for x in context], "failure_tags": ann["failure_tags"] + ["absent_target"]})
                    elif state == "INSUFFICIENT_EVIDENCE":
                        ann.update({"occluder_ids": [box], "occluder_labels": [BOX_LABELS[box]], "failure_tags": ann["failure_tags"] + ["physical_front_back_ycb_box_occlusion", "non_intersecting_3d"]})
                    annotation[sid] = ann; cells[(state, relation)] += 1
                    all_rows.append({"scene_id": sid, "family_id": family, "batch": batch, "state": state, "relation": relation, "primary_occluder": box, "layout_signature_sha256": signature})
        if len(rows) != 64 or set(cells.values()) != {4}: raise RuntimeError(f"unbalanced batch {batch}")
        random.Random(SEED + batch).shuffle(rows)
        scenes = deepcopy(src_scenes); scenes.update({"protocol_id": f"{PROTOCOL}_b{batch}", "expected_scene_count": 64, "random_seed": SEED + batch, "view_joint_pose": VIEW_POSE, "camera_frame": "top_table_camera_optical_frame", "objects": objects, "scenes": rows})
        oracle = deepcopy(src_ann); oracle.update({"protocol_id": scenes["protocol_id"], "oracle_usage": "evaluator_only_after_prediction_lock", "scenes": annotation})
        gate = deepcopy(src_gate); gate.update({"protocol_id": scenes["protocol_id"], "parent_family_count": 64, "split_parent_family_count": {"anti_shortcut_left_ycb_v1": 64}, "state_quota": {"anti_shortcut_left_ycb_v1": {s: 16 for s in STATES}}, "relation_variant_quota": {"anti_shortcut_left_ycb_v1": {r: 16 for r in RELATIONS}}, "state_relation_cell_quota": {"anti_shortcut_left_ycb_v1": 4}, "camera": {"frame": "top_table_camera_optical_frame", "base_frame": "base_link", "resolution": [640,480], "height_above_table_m": 1.10, "degrees_from_vertical": 30, "view_joint_pose": VIEW_POSE, "depth_unit":"metre", "valid_depth_range_m":[.1,3.]}, "physical_occlusion_rule": {"primary_occluders": list(BOXES), "target_types": list(FRUITS), "co_location_forbidden": True, "collision_intersection_forbidden": True, "require_occluder_closer_along_camera_ray": True, "minimum_3d_footprint_clearance_m": .008}, "policies": {**src_gate["policies"], "no_training": True, "no_model_inference": True, "no_materialization_before_qc": True}})
        for name, data in (("scenes.yaml",scenes),("annotations.yaml",oracle),("gate.yaml",gate)):
            (folder/name).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=160), encoding="utf-8")
        relative = lambda p: str(p.relative_to(ROOT))
        paths = required_sources() | {relative(BASE_WORLD), relative(WORLD), relative(Path(__file__)), relative(LAUNCH), "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py", relative(folder/"scenes.yaml"), relative(folder/"annotations.yaml"), relative(folder/"gate.yaml")}
        lock = {"schema_version":1,"protocol_id":scenes["protocol_id"],"status":"LOCKED_BEFORE_CAPTURE_AND_INFERENCE","locked_at_utc":datetime.now(timezone.utc).isoformat(),"workspace_root":str(ROOT),"source_artifact_sha256":{x:sha256(ROOT/x) for x in sorted(paths)},"model_inventory_sha256":"NO_MODEL_INFERENCE_LEFT_YCB_CAPTURE","capture_role":"INDEPENDENT_DEVELOPMENT_CHALLENGE","planned_scene_count":64,"view_joint_pose":VIEW_POSE,"world_file":relative(WORLD),"camera_design":{"height_above_table_m":1.10,"degrees_from_vertical":30},"seals":{"calibration":True,"test_iid":True,"test_ood":True}}
        (folder/"CAPTURE_SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2)+"\n", encoding="utf-8")
    index = OUT/"DESIGN_INDEX.jsonl"; index.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in all_rows),encoding="utf-8")
    report = {"status":"DESIGN_STATIC_QC_PASS_CAPTURE_NOT_STARTED","protocol_id":PROTOCOL,"families":len(all_rows),"state_counts":dict(Counter(x["state"] for x in all_rows)),"relation_counts":dict(Counter(x["relation"] for x in all_rows)),"primary_occluder_counts":dict(Counter(x["primary_occluder"] for x in all_rows if x["primary_occluder"])),"all_active_objects_left_of_robot":True,"camera":{"height_above_table_m":1.10,"degrees_from_vertical":30},"minimum_preregistered_occluder_target_clearance_m":min_clearance,"model_opened":False,"design_index_sha256":sha256(index),"world_sha256":sha256(WORLD)}
    (OUT/"DESIGN_STATIC_QC.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
