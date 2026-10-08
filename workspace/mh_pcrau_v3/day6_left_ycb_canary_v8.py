#!/usr/bin/env python3
"""Prepare the final 15-pair projection canary on grounded world v5.

This is an evaluator-only geometry calibration.  Lemon is intentionally out of
scope.  Historical v4 offsets are used only as scan centres; every acceptance
pixel count is measured again from a fresh v5 semantic frame.
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
    BOXES, BOX_LABELS, CLUTTER, FRUIT_LABELS, ROOT, VIEW_POSE, add_inventory,
)

OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v8"
WORLD = ROOT / "workspace/mh_pcrau_v3/generated/day6_left_ycb_v5_contact_exact/left_ycb_top30_contact_exact.sdf"
CONFIG = ROOT / "ur3/ur3_perception/config"
SOURCE_SCENES = CONFIG / "gazebo_train_uq_v2_full_r3_scenes.yaml"
SOURCE_ANNOTATIONS = CONFIG / "gazebo_train_uq_v2_full_r3_annotations.yaml"
SOURCE_GATE = CONFIG / "gazebo_train_uq_v2_full_r3_gate.yaml"
CAPTURE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
LAUNCH = ROOT / "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py"
PROTOCOL = "mh_pcrau_v3_anti_shortcut_left_ycb_canary_v8"
FRUITS = ("ycb_apple", "ycb_orange", "mango", "ycb_pear", "ycb_plum")

# Diagnostic v4 centres only.  They do not contribute observations to v8.
CENTRES = {
    "ycb_bleach_cleanser": {"mango": 0.00, "ycb_apple": -0.02, "ycb_orange": -0.02, "ycb_pear": -0.03, "ycb_plum": 0.02},
    "ycb_cracker_box": {"mango": 0.06, "ycb_apple": -0.06, "ycb_orange": -0.06, "ycb_pear": -0.06, "ycb_plum": 0.08},
    "ycb_sugar_box": {"mango": -0.03, "ycb_apple": 0.00, "ycb_orange": 0.00, "ycb_pear": -0.02, "ycb_plum": -0.01},
}
FORWARD = {"ycb_cracker_box": 0.102, "ycb_sugar_box": 0.086, "ycb_bleach_cleanser": 0.092}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def required_sources() -> set[str]:
    tree = ast.parse(CAPTURE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REQUIRED_SOURCE_ARTIFACTS" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("capture source inventory missing")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only output exists: {OUT}")
    src_scenes = yaml.safe_load(SOURCE_SCENES.read_text(encoding="utf-8"))
    src_ann = yaml.safe_load(SOURCE_ANNOTATIONS.read_text(encoding="utf-8"))
    src_gate = yaml.safe_load(SOURCE_GATE.read_text(encoding="utf-8"))
    objects = add_inventory(src_scenes["objects"])
    rows, annotations, index = [], {}, []
    for bi, box in enumerate(BOXES):
        for fi, fruit in enumerate(FRUITS):
            centre = CENTRES[box][fruit]
            # A slightly larger ray separation makes the narrow sugar box
            # cover the apple sufficiently; other pairs retain v4 geometry.
            forward = 0.135 if (box, fruit) == ("ycb_sugar_box", "ycb_apple") else FORWARD[box]
            offsets = [round(centre + delta, 3) for delta in (-0.012, -0.006, 0.0, 0.006, 0.012)]
            for oi, offset in enumerate(offsets):
                sid = f"v8can_{bi}_{fi}_{oi}"
                family = f"mh_pcrau_v3/canary_v8/{box}/{fruit}/{oi}"
                others = [x for x in FRUITS if x != fruit][:2]
                poses = {
                    fruit: [-0.24, 0.38, 0.0],
                    others[0]: [-0.12, 0.22, 0.0],
                    others[1]: [-0.38, 0.23, 0.0],
                    box: [-0.24 - forward, 0.38 + offset, 0.0],
                }
                fixed = [(-0.46, .66), (-0.34, .76), (-0.20, .70), (-0.08, .82), (-0.48, .88), (-0.10, .60)]
                for name, xy in zip(CLUTTER, fixed):
                    poses[name] = [xy[0], xy[1], 0.0]
                for bj, other_box in enumerate(BOXES):
                    if other_box != box:
                        poses[other_box] = [-0.48 + .22 * bj, .98, 0.0]
                poses["ycb_lemon"] = [5.34, 3.0, 0.0]
                sig = hashlib.sha256(json.dumps(poses, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                rows.append({"scene_id": sid, "scene_family_id": family, "layout_id": sid, "layout_signature_sha256": sig,
                             "task_type": "projection_calibration_canary_only", "instruction": f"Locate {fruit} behind {box}.", "poses": poses})
                annotations[sid] = {"state": "INSUFFICIENT_EVIDENCE", "target_id": fruit, "target_label": FRUIT_LABELS[fruit],
                                    "candidate_ids": [fruit, *others], "candidate_labels": [FRUIT_LABELS[x] for x in [fruit, *others]],
                                    "occluder_ids": [box], "occluder_labels": [BOX_LABELS[box]], "family_id": family,
                                    "layout_id": sid, "layout_signature_sha256": sig, "split": "projection_canary_only",
                                    "projection_calibration": {"box": box, "fruit": fruit, "forward_separation_m": forward,
                                                               "signed_lateral_offset_m": offset}}
                index.append({"scene_id": sid, "box": box, "fruit": fruit, "forward_separation_m": forward,
                              "signed_lateral_offset_m": offset, "layout_signature_sha256": sig})
    assert len(rows) == 75
    scenes = deepcopy(src_scenes); scenes.update({"protocol_id": PROTOCOL, "expected_scene_count": 75, "random_seed": 25092048,
        "view_joint_pose": VIEW_POSE, "camera_frame": "top_table_camera_optical_frame", "objects": objects, "scenes": rows})
    oracle = deepcopy(src_ann); oracle.update({"protocol_id": PROTOCOL, "oracle_usage": "projection_calibration_qc_only", "scenes": annotations})
    gate = deepcopy(src_gate); gate.update({"protocol_id": PROTOCOL, "parent_family_count": 75,
        "projection_calibration_gate": {"pairs_exact": 15, "fresh_observations_per_pair": 5,
          "pass_rule": "each pair has a fresh v5 observation with 1<=target_visible_pixels<120"},
        "policies": {**src_gate["policies"], "no_training": True, "no_model_inference": True, "canary_not_eligible_for_final_evaluation": True}})
    OUT.mkdir(parents=True)
    for name, payload in (("scenes.yaml", scenes), ("annotations.yaml", oracle), ("gate.yaml", gate)):
        (OUT / name).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=160), encoding="utf-8")
    idx = OUT / "DESIGN_INDEX.jsonl"
    idx.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in index), encoding="utf-8")
    rel = lambda p: str(p.relative_to(ROOT))
    sources = required_sources() | {rel(WORLD), rel(Path(__file__)), rel(LAUNCH), rel(OUT/"scenes.yaml"), rel(OUT/"annotations.yaml"), rel(OUT/"gate.yaml"), rel(idx), "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py"}
    lock = {"schema_version": 1, "protocol_id": PROTOCOL, "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
            "locked_at_utc": datetime.now(timezone.utc).isoformat(), "workspace_root": str(ROOT),
            "source_artifact_sha256": {x: sha256(ROOT/x) for x in sorted(sources)},
            "model_inventory_sha256": "NO_MODEL_INFERENCE_PROJECTION_CANARY", "planned_scene_count": 75,
            "view_joint_pose": VIEW_POSE, "world_file": rel(WORLD), "lemon_excluded": True}
    (OUT/"CAPTURE_SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "READY", "pairs": 15, "scenes": 75, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
