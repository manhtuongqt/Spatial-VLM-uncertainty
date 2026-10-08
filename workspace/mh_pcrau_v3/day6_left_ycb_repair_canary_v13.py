#!/usr/bin/env python3
"""Lock a small geometry sweep for the seven remaining Day-6 IE failures."""
from __future__ import annotations

import ast
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

from workspace.mh_pcrau_v3.day6_left_ycb_prepare import ROOT

SRC = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_repair_v12"
OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_repair_canary_v13"
WORLD = ROOT / "workspace/mh_pcrau_v3/generated/day6_left_ycb_v5_contact_exact/left_ycb_top30_contact_exact.sdf"
CAPTURE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
LAUNCH = ROOT / "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py"

FAILED = {
    "repair_101ffd3c813a7c34",
    "repair_dfdf987cbad760e1",
    "repair_7d3d34c0c723926b",
    "repair_3cd49fcf81a0a53b",
    "repair_1602ad1cab7bc2fb",
    "repair_6545ce43d05926c2",
    "repair_7854036868c291b0",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
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
    source = yaml.safe_load((SRC / "scenes.yaml").read_text())
    annotations = yaml.safe_load((SRC / "annotations.yaml").read_text())
    gate = yaml.safe_load((SRC / "gate.yaml").read_text())
    source_rows = {row["scene_id"]: row for row in source["scenes"]}
    rows, ann_out, candidates = [], {}, {}
    for old_id in sorted(FAILED):
        old = source_rows[old_id]
        ann = annotations["scenes"][old_id]
        target = ann["target_id"]
        occluder = ann["occluder_ids"][0]
        if (occluder, target) == ("ycb_sugar_box", "mango"):
            offsets = (-0.014, -0.010, -0.006, -0.002, 0.002)
        elif (occluder, target) == ("ycb_sugar_box", "ycb_pear"):
            offsets = (-0.003, 0.003, 0.009, 0.015, 0.021)
        elif (occluder, target) == ("ycb_cracker_box", "ycb_pear"):
            offsets = (-0.050, -0.040, -0.020, -0.010, 0.010)
        else:
            raise RuntimeError(f"unexpected pair: {occluder}, {target}")
        candidates[old_id] = []
        for index, offset in enumerate(offsets):
            scene_id = f"v13_{old_id.removeprefix('repair_')}_{index}"
            row = deepcopy(old)
            row["scene_id"] = scene_id
            row["scene_family_id"] = f"mh_pcrau_v3/repair_canary_v13/{scene_id}"
            row["layout_id"] = scene_id
            row["poses"][occluder][1] = row["poses"][target][1] + offset
            signature = hashlib.sha256(json.dumps(row["poses"], sort_keys=True).encode()).hexdigest()
            row["layout_signature_sha256"] = signature
            new_ann = deepcopy(ann)
            new_ann["family_id"] = row["scene_family_id"]
            new_ann["layout_id"] = scene_id
            new_ann["layout_signature_sha256"] = signature
            rows.append(row)
            ann_out[scene_id] = new_ann
            candidates[old_id].append(scene_id)
    protocol = "mh_pcrau_v3_anti_shortcut_left_ycb_repair_canary_v13"
    scenes = deepcopy(source)
    scenes.update({"protocol_id": protocol, "expected_scene_count": len(rows), "scenes": rows})
    oracle = deepcopy(annotations)
    oracle.update({"protocol_id": protocol, "scenes": ann_out})
    gate.update({"protocol_id": protocol, "parent_family_count": len(rows)})
    OUT.mkdir(parents=True)
    for name, payload in (("scenes.yaml", scenes), ("annotations.yaml", oracle), ("gate.yaml", gate)):
        (OUT / name).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=180))
    (OUT / "CANDIDATE_MAP.json").write_text(json.dumps({"candidates": candidates}, indent=2) + "\n")
    rel = lambda path: str(Path(path).relative_to(ROOT))
    sources = required_sources() | {
        rel(WORLD), rel(Path(__file__)), rel(LAUNCH), rel(OUT / "scenes.yaml"),
        rel(OUT / "annotations.yaml"), rel(OUT / "gate.yaml"),
        "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
    }
    lock = {
        "schema_version": 1,
        "protocol_id": protocol,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "source_artifact_sha256": {name: sha256(ROOT / name) for name in sorted(sources)},
        "model_inventory_sha256": "NO_MODEL_INFERENCE_PROJECTION_REPAIR_CANARY",
        "planned_scene_count": len(rows),
        "view_joint_pose": scenes["view_joint_pose"],
        "world_file": rel(WORLD),
    }
    (OUT / "CAPTURE_SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"status": "PASS_CAPTURE_NOT_STARTED", "scenes": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
