#!/usr/bin/env python3
"""Generate fresh full-v2-r2 identities with stabilized IE geometry."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

import generate_gazebo_train_uq_v2_full_contract as v1

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ur3/ur3_perception/config"
PROTOCOL_ID = "gazebo_train_uq_v2_full_r2"
SCENES_PATH = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS_PATH = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE_PATH = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
FAMILY_MANIFEST_PATH = ROOT / "protocol/gazebo_train_uq_v2_full_r2_family_manifest.jsonl"
SPLIT_MANIFEST_PATH = ROOT / "protocol/gazebo_train_uq_v2_full_r2_split_manifest.json"
OUTPUT_PATHS = (SCENES_PATH, ANNOTATIONS_PATH, GATE_PATH)
STATES, RELATIONS, SPLITS = v1.STATES, v1.RELATIONS, v1.SPLITS
PANEL = "uq_neutral_occluder"

# Chosen before r2 capture from the immutable full-attempt-01 empirical bands.
# Every listed band remained inside the official 1–119 px gate. No evaluator
# feedback is consumed during capture.
STABLE_IE = {
    "ycb_apple": {"yaw": 0.35, "panel_y": 0.26180},
    "ycb_orange": {"yaw": 0.35, "panel_y": 0.25820},
    "mango": {"yaw": 0.35, "panel_y": 0.25110},
}


def seed_for(scene_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{PROTOCOL_ID}:{scene_id}".encode()).digest()[:4], "big")


def build():
    scenes, annotations, gate, _, _ = copy.deepcopy(v1.build())
    scenes["protocol_id"] = annotations["protocol_id"] = gate["protocol_id"] = PROTOCOL_ID
    scenes["random_seed"] = 14092039
    # Apple's collision sphere radius is 0.03771 m. The old mesh-derived
    # 0.035953 m support height injected a settling impulse.
    scenes["objects"]["ycb_apple"]["z"] = 0.03771
    oracle = annotations["scenes"]
    family_rows = []
    for index, row in enumerate(scenes["scenes"]):
        old_sid = row["scene_id"]
        item = oracle.pop(old_sid)
        split = item["split"]
        sid = f"gazebo_uq_v2_full_r2_{split}_{index:03d}"
        family = f"spatial_vlm_uq_v2_full_r2/{split}/parent_{index:03d}"
        layout = f"uq_v2_full_r2_layout_{index:03d}"
        poses = copy.deepcopy(row["poses"])
        if item["state"] == "INSUFFICIENT_EVIDENCE":
            target = item["target_id"]
            stable = STABLE_IE[target]
            poses[target] = [-0.220, 0.300, stable["yaw"]]
            poses[PANEL] = [-0.2168, stable["panel_y"], 1.5707963267948966]
            # Fresh full signatures come from preregistered context variation,
            # while the safety-critical target/screen pair is held fixed.
            for j, name in enumerate(sorted(set(poses) - {target, PANEL})):
                poses[name][0] += 0.020 + 0.0011 * ((index + j) % 9)
                poses[name][1] += 0.0013 * (((index + 3 * j) % 9) - 4)
                poses[name][2] += 0.017 * ((index + j) % 11)
            item["geometry_provenance"] = "attempt_01_safe_object_yaw_band_support_height_repair"
        else:
            dx = 0.0060 + 0.00031 * (index % 11)
            dy = -0.0050 + 0.00037 * (index % 13)
            for j, pose in enumerate(poses.values()):
                pose[0] += dx + 0.00017 * j
                pose[1] += dy
                pose[2] += 0.011 + 0.003 * ((index + j) % 5)
        signature = v1.geometry.layout_signature(poses)
        row.update({"scene_id": sid, "scene_family_id": family, "layout_id": layout,
                    "layout_signature_sha256": signature, "poses": poses})
        item.update({"family_id": family, "layout_id": layout,
                     "layout_signature_sha256": signature, "seed": seed_for(sid),
                     "template_provenance": "full_attempt_01_geometry_class_only_no_sample_reuse"})
        oracle[sid] = item
        family_rows.append({"scene_id": sid, "family_id": family, "layout_id": layout,
                            "layout_signature_sha256": signature, "split": split,
                            "state": item["state"], "relation_variant": item["relation_variant"],
                            "deterministic_seed": seed_for(sid)})
    gate["full_revision"] = {
        "predecessor": "gazebo_train_uq_v2_full",
        "predecessor_decision": "REJECT_317_OF_320_NO_MATERIALIZATION",
        "repair_scope": "geometry_stability_only",
        "required_acceptance": "320/320_in_one_attempt",
        "no_sample_reuse": True,
    }
    split_manifest = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PREREGISTERED",
                      "train_uq": {"families": 256, "cell_quota": 16},
                      "val_uq": {"families": 64, "cell_quota": 4},
                      "family_disjoint": True, "pilot_families_excluded": True,
                      "predecessor_families_excluded": True}
    return scenes, annotations, gate, family_rows, split_manifest


def serialized(payload):
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--write", action="store_true"); p.add_argument("--check", action="store_true"); a = p.parse_args()
    if a.write == a.check: p.error("choose exactly one of --write or --check")
    values = build(); expected = [serialized(x) for x in values[:3]]
    family_text = "".join(json.dumps(x, sort_keys=True) + "\n" for x in values[3])
    split_text = json.dumps(values[4], indent=2, sort_keys=True) + "\n"
    paths = (*OUTPUT_PATHS, FAMILY_MANIFEST_PATH, SPLIT_MANIFEST_PATH)
    texts = (*expected, family_text, split_text)
    if a.write:
        if any(x.exists() for x in paths): raise FileExistsError("refusing to overwrite full-r2 contract")
        for path, text in zip(paths, texts): path.write_text(text, encoding="utf-8")
        print(json.dumps({"status": "GENERATED", "families": len(values[3])}, indent=2))
    else:
        drift = [str(path) for path, text in zip(paths, texts) if not path.is_file() or path.read_text() != text]
        if drift: raise ValueError(f"contract drift: {drift}")
        print("PASS: deterministic fresh 320-family full-v2-r2 contract")


if __name__ == "__main__": main()
