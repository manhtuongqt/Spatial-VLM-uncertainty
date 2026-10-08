#!/usr/bin/env python3
"""Create one corrected low top-oblique Gazebo canary for visual approval.

This revision fixes YCB mesh URI resolution, lowers the fixed camera to the
UR3 working height, moves the objects away from the arm, and places a cracker
box physically in front of (but not intersecting) an apple.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE_WORLD = ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf"
SOURCE_SCENES = ROOT / "ur3/ur3_perception/config/gazebo_train_uq_v2_full_r3_scenes.yaml"
SOURCE_ANNOTATIONS = ROOT / "ur3/ur3_perception/config/gazebo_train_uq_v2_full_r3_annotations.yaml"
SOURCE_GATE = ROOT / "ur3/ur3_perception/config/gazebo_train_uq_v2_full_r3_gate.yaml"
CAPTURE_NODE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_top30_canary_v3"
WORLD = ROOT / "workspace/mh_pcrau_v3/generated/day6_top30_canary_v3/low_top30_canary.sdf"
PROTOCOL = "mh_pcrau_v3_low_top30_canary_v3"
VIEW_POSE = [1.3815, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def required_sources() -> set[str]:
    tree = ast.parse(CAPTURE_NODE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_SOURCE_ARTIFACTS"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("capture source inventory not found")


def build_world() -> None:
    text = SOURCE_WORLD.read_text(encoding="utf-8")
    old_pose = "<pose>0 0.45 1.80 0 1.57079632679 0</pose>"
    # Camera is 1.10 m above the table and 30 degrees away from vertical.
    target_x, target_y, camera_z = -0.34, 0.55, 1.10
    radius = camera_z / math.tan(math.radians(60.0))
    new_pose = f"<pose>{target_x-radius:.12f} {target_y:.12f} {camera_z:.12f} 0 1.047197551197 0</pose>"
    if text.count(old_pose) != 1:
        raise RuntimeError("camera pose marker drifted")
    text = text.replace(old_pose, new_pose)
    # The generated world lives outside the original worlds directory, so all
    # YCB mesh references are made absolute instead of silently disappearing.
    ycb_root = (ROOT / "ur3/ur_simulation_gz/models/ycb").resolve().as_uri()
    text = text.replace("../models/ycb", ycb_root)
    marker = """        </sensor>
      </link>
    </model>

    <model name=\"red_cube\">"""
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
    if text.count(marker) != 1:
        raise RuntimeError("camera sensor marker drifted")
    text = text.replace(marker, semantic)
    WORLD.parent.mkdir(parents=True, exist_ok=True)
    WORLD.write_text(text, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only output exists: {OUT}")
    build_world()
    source_scenes = yaml.safe_load(SOURCE_SCENES.read_text(encoding="utf-8"))
    source_annotations = yaml.safe_load(SOURCE_ANNOTATIONS.read_text(encoding="utf-8"))
    source_gate = yaml.safe_load(SOURCE_GATE.read_text(encoding="utf-8"))
    objects = deepcopy(source_scenes["objects"])
    for index, obj in enumerate(objects.values()):
        obj["storage_pose"] = [3.0 + 0.15 * index, 3.0, 0.0]

    # The camera looks along +X.  The cracker box has smaller X than the apple,
    # therefore it is genuinely in front.  dx=0.11 m and dy=0.08 m leave a
    # positive footprint gap; the two collision bodies never overlap.
    scene_id = "low_top30_ie_canary_001"
    poses = {
        "ycb_apple": [-0.24, 0.55, 0.0],
        "ycb_cracker_box": [-0.35, 0.63, 0.0],
        "ycb_orange": [-0.30, 0.31, 0.0],
        "mango": [-0.22, 0.82, 0.0],
        "ycb_sugar_box": [-0.04, 0.27, 0.0],
        "ycb_tomato_soup_can": [-0.52, 0.84, 0.0],
        "ycb_mustard_bottle": [-0.55, 0.25, 0.0],
        "ycb_banana": [-0.05, 0.91, math.pi / 2],
        "ycb_power_drill": [-0.02, 0.57, math.pi / 2],
    }
    signature = hashlib.sha256(json.dumps(poses, sort_keys=True).encode()).hexdigest()
    scene = {
        "scene_id": scene_id,
        "scene_family_id": "mh_pcrau_v3/low_top30_canary_v2/001",
        "layout_id": scene_id,
        "layout_signature_sha256": signature,
        "task_type": "physical_front_back_occlusion_canary",
        "instruction": "Point to the apple among the visible tabletop fruits.",
        "poses": poses,
    }
    scenes = deepcopy(source_scenes)
    scenes.update({
        "protocol_id": PROTOCOL,
        "expected_scene_count": 1,
        "random_seed": 25092041,
        "view_joint_pose": VIEW_POSE,
        "camera_frame": "top_table_camera_optical_frame",
        "objects": objects,
        "scenes": [scene],
    })
    annotation = deepcopy(source_annotations)
    annotation.update({
        "protocol_id": PROTOCOL,
        "oracle_usage": "visual_canary_geometry_qc_only",
        "scenes": {
            scene_id: {
                "state": "INSUFFICIENT_EVIDENCE",
                "target_id": "ycb_apple",
                "target_label": 26,
                "candidate_ids": ["ycb_apple", "ycb_orange", "mango"],
                "candidate_labels": [26, 27, 29],
                "occluder_ids": ["ycb_cracker_box"],
                "occluder_labels": [21],
                "rank_from": "left",
                "rank": 2,
                "relation_variant": "second_from_left",
                "family_id": scene["scene_family_id"],
                "layout_id": scene_id,
                "layout_signature_sha256": signature,
                "split": "visual_canary_only",
                "failure_tags": ["physical_front_back_occlusion", "non_intersecting_3d"],
            }
        },
    })
    gate = deepcopy(source_gate)
    gate.update({
        "protocol_id": PROTOCOL,
        "parent_family_count": 1,
        "split_parent_family_count": {"visual_canary_only": 1},
        "state_quota": {"visual_canary_only": {"INSUFFICIENT_EVIDENCE": 1}},
        "camera": {
            "frame": "top_table_camera_optical_frame",
            "base_frame": "base_link",
            "resolution": [640, 480],
            "height_above_table_m": 1.10,
            "degrees_from_vertical": 30,
            "view_joint_pose": VIEW_POSE,
            "depth_unit": "metre",
            "valid_depth_range_m": [0.1, 3.0],
        },
        "physical_occlusion_rule": {
            "occluder": "ycb_cracker_box",
            "target": "ycb_apple",
            "front_back_axis": "camera_ray_positive_x",
            "center_separation_m": [0.11, 0.08],
            "co_location_forbidden": True,
            "collision_intersection_forbidden": True,
        },
        "policies": {**source_gate["policies"], "no_training": True, "no_model_inference": True},
    })
    OUT.mkdir(parents=True)
    for name, payload in (("scenes.yaml", scenes), ("annotations.yaml", annotation), ("gate.yaml", gate)):
        (OUT / name).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140), encoding="utf-8")

    relative = lambda path: str(path.relative_to(ROOT))
    sources = required_sources() | {
        relative(SOURCE_WORLD), relative(WORLD), relative(Path(__file__)),
        relative(OUT / "scenes.yaml"), relative(OUT / "annotations.yaml"), relative(OUT / "gate.yaml"),
        "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py",
    }
    lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "source_artifact_sha256": {name: sha256(ROOT / name) for name in sorted(sources)},
        "model_inventory_sha256": "NO_MODEL_INFERENCE_LOW_TOP30_CANARY",
        "capture_role": "VISUAL_GEOMETRY_CANARY_ONLY",
        "planned_scene_count": 1,
        "view_joint_pose": VIEW_POSE,
        "world_file": relative(WORLD),
        "seals": {"calibration": True, "test_iid": True, "test_ood": True},
    }
    (OUT / "CAPTURE_SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "READY", "output": str(OUT), "world": str(WORLD)}, indent=2))


if __name__ == "__main__":
    main()
