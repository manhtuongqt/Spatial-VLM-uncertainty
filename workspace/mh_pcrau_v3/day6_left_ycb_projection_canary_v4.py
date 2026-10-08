#!/usr/bin/env python3
"""Build an append-only projection-calibration canary for every box/fruit pair.

The canary is evaluator-only and never opens a model checkpoint.  It sweeps a
signed screen-lateral offset for all 3 x 6 primary-occluder/fruit pairs while
keeping the camera, forward clearance, world and sensor stack identical to the
rejected left-YCB v3 capture.  Semantic masks are used only after capture to
select an offset with 1--119 visible target pixels for each pair.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml

from workspace.mh_pcrau_v3.day6_left_ycb_prepare import (
    BOXES,
    BOX_LABELS,
    CLUTTER,
    FRUITS,
    FRUIT_LABELS,
    ROOT,
    VIEW_POSE,
    add_inventory,
)


OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_projection_canary_v4"
WORLD = ROOT / "workspace/mh_pcrau_v3/generated/day6_left_ycb_v3/left_ycb_top30.sdf"
CONFIG = ROOT / "ur3/ur3_perception/config"
SOURCE_SCENES = CONFIG / "gazebo_train_uq_v2_full_r3_scenes.yaml"
SOURCE_ANNOTATIONS = CONFIG / "gazebo_train_uq_v2_full_r3_annotations.yaml"
SOURCE_GATE = CONFIG / "gazebo_train_uq_v2_full_r3_gate.yaml"
CAPTURE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
LAUNCH = ROOT / "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py"
PROTOCOL = "mh_pcrau_v3_anti_shortcut_left_ycb_projection_canary_v4"

# Preserve the v3 forward separation/clearance design.  Only projected lateral
# offset is calibrated.  A 10 mm coarse grid spans complete occlusion through
# fully separated silhouettes for every observed pair.
FORWARD_SEPARATION = {
    "ycb_cracker_box": 0.102,
    "ycb_sugar_box": 0.086,
    "ycb_bleach_cleanser": 0.092,
}
OFFSETS_M = tuple(round(-0.20 + 0.01 * index, 3) for index in range(41))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_sources() -> set[str]:
    tree = ast.parse(CAPTURE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_SOURCE_ARTIFACTS"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("required capture inventory missing")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only output exists: {OUT}")
    if not WORLD.is_file():
        raise FileNotFoundError(WORLD)

    source_scenes = yaml.safe_load(SOURCE_SCENES.read_text(encoding="utf-8"))
    source_annotations = yaml.safe_load(SOURCE_ANNOTATIONS.read_text(encoding="utf-8"))
    source_gate = yaml.safe_load(SOURCE_GATE.read_text(encoding="utf-8"))
    objects = add_inventory(source_scenes["objects"])

    rows = []
    annotations = {}
    design_rows = []
    for box_index, box in enumerate(BOXES):
        for fruit_index, fruit in enumerate(FRUITS):
            other_fruits = [name for name in FRUITS if name != fruit][:2]
            for offset_index, offset in enumerate(OFFSETS_M):
                scene_id = f"pcal_{box_index}_{fruit_index}_{offset_index:02d}"
                family_id = f"mh_pcrau_v3/projection_canary_v4/{box}/{fruit}/{offset_index:02d}"
                poses = {
                    fruit: [-0.22, 0.56, 0.0],
                    other_fruits[0]: [-0.28, 0.82, 0.0],
                    other_fruits[1]: [-0.16, 0.30, 0.0],
                    box: [-0.22 - FORWARD_SEPARATION[box], 0.56 + offset, 0.0],
                }
                fixed = [(-0.65, 0.25), (-0.63, 0.91), (-0.48, 0.16),
                         (-0.46, 1.01), (-0.08, 0.90), (-0.06, 0.18)]
                for name, (x_value, y_value) in zip(CLUTTER, fixed):
                    poses[name] = [x_value, y_value, 0.0]
                for index, other_box in enumerate(BOXES):
                    if other_box != box:
                        poses[other_box] = [(-0.62, -0.47, -0.11)[index],
                                            (0.53, 0.73, 0.98)[index], 0.0]
                signature = hashlib.sha256(
                    json.dumps(poses, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                rows.append({
                    "scene_id": scene_id,
                    "scene_family_id": family_id,
                    "layout_id": scene_id,
                    "layout_signature_sha256": signature,
                    "task_type": "projection_calibration_canary_only",
                    "instruction": f"Projection calibration for {fruit} behind {box}.",
                    "poses": poses,
                })
                annotations[scene_id] = {
                    "state": "INSUFFICIENT_EVIDENCE",
                    "target_id": fruit,
                    "target_label": FRUIT_LABELS[fruit],
                    "candidate_ids": [fruit, *other_fruits],
                    "candidate_labels": [FRUIT_LABELS[name] for name in [fruit, *other_fruits]],
                    "occluder_ids": [box],
                    "occluder_labels": [BOX_LABELS[box]],
                    "family_id": family_id,
                    "layout_id": scene_id,
                    "layout_signature_sha256": signature,
                    "split": "projection_canary_only",
                    "projection_calibration": {
                        "box": box,
                        "fruit": fruit,
                        "forward_separation_m": FORWARD_SEPARATION[box],
                        "signed_lateral_offset_m": offset,
                    },
                    "failure_tags": ["projection_canary_only", "no_model_access"],
                }
                design_rows.append({
                    "scene_id": scene_id,
                    "box": box,
                    "fruit": fruit,
                    "signed_lateral_offset_m": offset,
                    "layout_signature_sha256": signature,
                })

    expected = len(BOXES) * len(FRUITS) * len(OFFSETS_M)
    if len(rows) != expected:
        raise RuntimeError("canary cardinality mismatch")

    scenes = deepcopy(source_scenes)
    scenes.update({
        "protocol_id": PROTOCOL,
        "expected_scene_count": expected,
        "random_seed": 25092047,
        "view_joint_pose": VIEW_POSE,
        "camera_frame": "top_table_camera_optical_frame",
        "objects": objects,
        "scenes": rows,
    })
    oracle = deepcopy(source_annotations)
    oracle.update({
        "protocol_id": PROTOCOL,
        "oracle_usage": "projection_calibration_qc_only_before_final_dataset_lock",
        "scenes": annotations,
    })
    gate = deepcopy(source_gate)
    gate.update({
        "protocol_id": PROTOCOL,
        "parent_family_count": expected,
        "split_parent_family_count": {"projection_canary_only": expected},
        "state_quota": {"projection_canary_only": {"INSUFFICIENT_EVIDENCE": expected}},
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
        "projection_calibration_gate": {
            "pairs_exact": 18,
            "offset_grid_m": list(OFFSETS_M),
            "pass_rule": "each box/fruit pair has at least one captured scene with 1<=target_visible_pixels<120",
            "selection_rule": "choose visible-pixel count nearest 60, ties by smallest absolute offset then signed offset",
        },
        "policies": {
            **source_gate["policies"],
            "no_training": True,
            "no_model_inference": True,
            "canary_not_eligible_for_final_evaluation": True,
        },
    })

    OUT.mkdir(parents=True)
    for name, payload in (("scenes.yaml", scenes), ("annotations.yaml", oracle), ("gate.yaml", gate)):
        (OUT / name).write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=160),
            encoding="utf-8",
        )
    index_path = OUT / "DESIGN_INDEX.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in design_rows),
        encoding="utf-8",
    )

    relative = lambda path: str(path.relative_to(ROOT))
    sources = required_sources() | {
        relative(WORLD), relative(Path(__file__)), relative(LAUNCH),
        relative(OUT / "scenes.yaml"), relative(OUT / "annotations.yaml"),
        relative(OUT / "gate.yaml"), relative(index_path),
        "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
    }
    lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "source_artifact_sha256": {name: sha256(ROOT / name) for name in sorted(sources)},
        "model_inventory_sha256": "NO_MODEL_INFERENCE_PROJECTION_CANARY",
        "capture_role": "PAIRWISE_PROJECTION_CALIBRATION_CANARY_ONLY",
        "planned_scene_count": expected,
        "view_joint_pose": VIEW_POSE,
        "world_file": relative(WORLD),
        "seals": {"calibration": True, "test_iid": True, "test_ood": True, "robot": True},
    }
    (OUT / "CAPTURE_SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PROJECTION_CANARY_READY",
        "pairs": len(BOXES) * len(FRUITS),
        "offsets_per_pair": len(OFFSETS_M),
        "scenes": expected,
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
