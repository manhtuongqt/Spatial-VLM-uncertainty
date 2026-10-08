#!/usr/bin/env python3
"""Create a preregistered repair batch for the semantic failures in fresh v11."""
from __future__ import annotations

import ast
import hashlib
import json
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

from workspace.mh_pcrau_v3.day6_left_ycb_prepare import (
    BOXES, CLUTTER, ROOT, add_inventory, tie_indices,
)

OLD = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_final_v11"
OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_repair_v12"
WORLD = ROOT / "workspace/mh_pcrau_v3/generated/day6_left_ycb_v5_contact_exact/left_ycb_top30_contact_exact.sdf"
GEO = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v13_sugar_apple_near/SELECTED_15_PAIR_GEOMETRY.json"
CAPTURE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
LAUNCH = ROOT / "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py"
CHECKPOINT = ROOT / "workspace/mh_pcrau_v3/checkpoints/day06_robust_candidate_r2/s1a_robust_fixed_epoch54.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def required_sources() -> set[str]:
    tree = ast.parse(CAPTURE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "REQUIRED_SOURCE_ARTIFACTS"
            for t in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("capture source inventory missing")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    failed = {
        row.split(":", 1)[0]
        for row in json.loads((OLD / "CAPTURE_QC.json").read_text())["semantic_failures"]
    }
    old_scenes, old_anns = {}, {}
    template_scenes = template_anns = template_gate = None
    for batch in range(4):
        folder = OLD / f"batch_{batch}"
        scenes = yaml.safe_load((folder / "scenes.yaml").read_text())
        anns = yaml.safe_load((folder / "annotations.yaml").read_text())
        gate = yaml.safe_load((folder / "gate.yaml").read_text())
        template_scenes = template_scenes or scenes
        template_anns = template_anns or anns
        template_gate = template_gate or gate
        old_scenes.update({x["scene_id"]: x for x in scenes["scenes"]})
        old_anns.update(anns["scenes"])
    geometry = json.loads(GEO.read_text())["pairs"]
    objects = add_inventory(template_scenes["objects"])
    objects["ycb_lemon"]["storage_pose"] = [5.34, 3.0, 0.0]
    rows, anns_out, repair_map = [], {}, {}
    for old_id in sorted(failed):
        old_scene, ann = old_scenes[old_id], deepcopy(old_anns[old_id])
        state, relation = ann["state"], ann["relation_variant"]
        token = hashlib.sha256(f"repair-v12|{old_id}".encode()).hexdigest()[:16]
        scene_id = f"repair_{token}"
        family_id = f"mh_pcrau_v3/anti_shortcut_v12/{token}"
        rng = random.Random(int(token, 16))
        poses = {}
        if state == "FOUND":
            fruits = list(ann["candidate_ids"])
            target = ann["target_id"]
            covered = next(x for x in fruits if x != target)
            slots = [0.58, 0.38, 0.18]
            ordered = fruits if relation in ("leftmost", "second_from_left") else list(reversed(fruits))
            for fruit, y in zip(ordered, slots):
                poses[fruit] = [-0.24 + rng.uniform(-0.07, 0.07), y, 0.0]
        elif state == "AMBIGUOUS":
            fruits = list(ann["candidate_ids"])
            valid = list(ann["valid_target_ids"])
            covered = next(x for x in fruits if x not in valid)
            poses[valid[0]] = [-0.36, 0.50, 0.0]
            poses[valid[1]] = [-0.12, 0.50, 0.0]
            poses[covered] = [-0.24, 0.18, 0.0]
        elif state == "INSUFFICIENT_EVIDENCE":
            fruits = list(ann["candidate_ids"])
            target = ann["target_id"]
            covered = target
            others = [x for x in fruits if x != target]
            poses[others[0]] = [-0.34, 0.18, 0.0]
            poses[others[1]] = [-0.14, 0.62, 0.0]
            poses[target] = [-0.24, 0.38, 0.0]
        else:
            raise RuntimeError(f"unexpected repair state {state}")
        occluder = ann.get("occluder_ids", [BOXES[int(token, 16) % len(BOXES)]])[0]
        g = geometry[occluder][covered]
        covered_x = -0.10 if occluder == "ycb_sugar_box" and covered == "ycb_apple" else poses[covered][0]
        covered_y = poses[covered][1]
        poses[covered] = [covered_x, covered_y, 0.0]
        poses[occluder] = [
            covered_x - float(g["forward_separation_m"]),
            covered_y + float(g["signed_lateral_offset_m"]),
            float(g.get("occluder_yaw_rad", 0.0)),
        ]
        fixed = [(-0.48, 0.74), (-0.36, 0.88), (-0.22, 0.76), (-0.08, 0.90), (-0.47, 1.00), (-0.09, 0.68)]
        for name, (x, y) in zip(CLUTTER, fixed):
            poses[name] = [x, y, rng.uniform(-0.4, 0.4)]
        for i, box in enumerate(BOXES):
            if box != occluder:
                poses[box] = [-0.48 + 0.22 * i, 1.08, 0.0]
        poses["ycb_lemon"] = [5.34, 3.0, 0.0]
        signature = hashlib.sha256(json.dumps(poses, sort_keys=True).encode()).hexdigest()
        row = deepcopy(old_scene)
        row.update({"scene_id": scene_id, "scene_family_id": family_id, "layout_id": scene_id, "layout_signature_sha256": signature, "poses": poses})
        ann.update({"family_id": family_id, "layout_id": scene_id, "layout_signature_sha256": signature})
        rows.append(row)
        anns_out[scene_id] = ann
        repair_map[old_id] = scene_id
    scenes = deepcopy(template_scenes)
    scenes.update({"protocol_id": "mh_pcrau_v3_anti_shortcut_left_ycb_repair_v12", "expected_scene_count": len(rows), "random_seed": 25092052, "objects": objects, "scenes": rows})
    annotations = deepcopy(template_anns)
    annotations.update({"protocol_id": scenes["protocol_id"], "scenes": anns_out})
    gate = deepcopy(template_gate)
    gate.update({"protocol_id": scenes["protocol_id"], "parent_family_count": len(rows)})
    OUT.mkdir(parents=True)
    for name, payload in (("scenes.yaml", scenes), ("annotations.yaml", annotations), ("gate.yaml", gate)):
        (OUT / name).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=180))
    (OUT / "REPAIR_MAP.json").write_text(json.dumps({"status": "LOCKED", "repairs": repair_map}, indent=2) + "\n")
    rel = lambda p: str(Path(p).relative_to(ROOT))
    sources = required_sources() | {
        rel(WORLD), rel(Path(__file__)), rel(LAUNCH), rel(GEO),
        rel(OUT / "scenes.yaml"), rel(OUT / "annotations.yaml"), rel(OUT / "gate.yaml"),
        "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
    }
    lock = {
        "schema_version": 1,
        "protocol_id": scenes["protocol_id"],
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "source_artifact_sha256": {x: sha256(ROOT / x) for x in sorted(sources)},
        "model_inventory_sha256": sha256(CHECKPOINT),
        "frozen_model": rel(CHECKPOINT),
        "planned_scene_count": len(rows),
        "view_joint_pose": scenes["view_joint_pose"],
        "world_file": rel(WORLD),
    }
    (OUT / "CAPTURE_SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"status": "PASS_CAPTURE_NOT_STARTED", "repair_scenes": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
