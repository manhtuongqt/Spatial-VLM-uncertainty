#!/usr/bin/env python3
"""Generate fresh r6 identities for the collision-free WP5 geometry pilot."""
from __future__ import annotations

import argparse
import copy
import hashlib

import yaml

import generate_gazebo_train_uq_v2_pilot_r5_contract as parent

ROOT = parent.ROOT
CONFIG = parent.CONFIG
PROTOCOL_ID = "gazebo_train_uq_v2_pilot_r6"
SCENES_PATH = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS_PATH = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE_PATH = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
OUTPUT_PATHS = (SCENES_PATH, ANNOTATIONS_PATH, GATE_PATH)


def seed_for(scene_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{PROTOCOL_ID}:{scene_id}".encode()).digest()[:4], "big")


def build():
    old_scenes, old_oracle, old_gate = parent.build()
    scenes, oracle, gate = copy.deepcopy(old_scenes), copy.deepcopy(old_oracle), copy.deepcopy(old_gate)
    scenes["protocol_id"] = PROTOCOL_ID; scenes["random_seed"] = 14092037
    oracle["protocol_id"] = PROTOCOL_ID; gate["protocol_id"] = PROTOCOL_ID
    new_annotations = {}
    for index, row in enumerate(scenes["scenes"]):
        old_sid = row["scene_id"]; item = oracle["scenes"][old_sid]; state = item["state"]
        if state == "INSUFFICIENT_EVIDENCE":
            target = item["target_id"]
            for name, pose in row["poses"].items():
                if name not in (target, parent.PANEL):
                    pose[0] += 0.008; pose[2] += 0.007
        else:
            for pose in row["poses"].values():
                pose[0] += 0.006; pose[1] += 0.004 if index % 2 == 0 else -0.004; pose[2] += 0.008
        sid = f"gazebo_uq_v2_pilot_r6_{index:03d}"
        family = f"spatial_vlm_uq_v2_pilot_r6/pilot/parent_{index:03d}"
        layout = f"layout_r6_{index:03d}"; signature = parent.layout_signature(row["poses"])
        row.update({"scene_id": sid, "scene_family_id": family, "layout_id": layout,
                    "layout_signature_sha256": signature})
        item.update({"family_id": family, "layout_id": layout, "layout_signature_sha256": signature,
                     "seed": seed_for(sid)})
        new_annotations[sid] = item
    oracle["scenes"] = new_annotations
    return scenes, oracle, gate


def serialized(payload):
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--write", action="store_true"); parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write == args.check: parser.error("choose exactly one of --write or --check")
    outputs = dict(zip(OUTPUT_PATHS, build()))
    if args.write:
        existing = [str(path) for path in outputs if path.exists()]
        if existing: raise FileExistsError(f"refusing overwrite: {existing}")
        for path, payload in outputs.items(): path.write_text(serialized(payload), encoding="utf-8"); print(path)
    else:
        mismatches = [str(path) for path,payload in outputs.items() if not path.is_file() or path.read_text()!=serialized(payload)]
        if mismatches: raise ValueError(f"generated inputs differ: {mismatches}")
        print("PASS: deterministic Gazebo Train-UQ v2 pilot r6 contract inputs")


if __name__ == "__main__": main()
