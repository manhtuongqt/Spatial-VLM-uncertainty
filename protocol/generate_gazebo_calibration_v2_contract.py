#!/usr/bin/env python3
"""Generate fresh Calibration-v2 identities after invalid v1 capture ordering."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json

import yaml

import generate_gazebo_calibration_v1_contract as v1

# Compatibility alias used by the shared immutable-overlap validator.
source = v1.source


ROOT = v1.ROOT
CONFIG = v1.CONFIG
PROTOCOL_ID = "gazebo_calibration_v2"
SCENES_PATH = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS_PATH = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE_PATH = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
FAMILY_MANIFEST_PATH = ROOT / "protocol/gazebo_calibration_v2_family_manifest.jsonl"
SPLIT_MANIFEST_PATH = ROOT / "protocol/gazebo_calibration_v2_split_manifest.json"
OUTPUT_PATHS = (SCENES_PATH, ANNOTATIONS_PATH, GATE_PATH)
STATES, RELATIONS, REPETITIONS = v1.STATES, v1.RELATIONS, v1.REPETITIONS


def seed_for(scene_id):
    return int.from_bytes(hashlib.sha256(f"{PROTOCOL_ID}:{scene_id}".encode()).digest()[:4], "big")


def build():
    scenes, annotations, gate, _, split = copy.deepcopy(v1.build())
    old_oracle, oracle, families = annotations["scenes"], {}, []
    for index, row in enumerate(scenes["scenes"]):
        old_sid = row["scene_id"]
        item = old_oracle[old_sid]
        state, relation = item["state"], item["relation_variant"]
        repetition = item["cell_repetition"]
        poses = copy.deepcopy(row["poses"])
        if state == "INSUFFICIENT_EVIDENCE":
            for offset, name in enumerate(sorted(set(poses) - {v1.TARGET_IE, v1.PANEL})):
                poses[name][0] += 0.0032 + 0.0005 * ((index + offset) % 4)
                poses[name][1] += 0.0010 * (((index + 2 * offset) % 5) - 2)
                poses[name][2] += 0.023 + 0.006 * ((index + offset) % 4)
        else:
            dx = 0.0035
            dy = -0.0050 + 0.0010 * (repetition % 3)
            for offset, pose in enumerate(poses.values()):
                pose[0] += dx; pose[1] += dy; pose[2] += 0.019 + 0.005 * ((index + offset) % 5)
        sid = f"gazebo_calibration_v2_{index:03d}"
        family = f"spatial_vlm_gazebo_calibration_v2/calibration/parent_{index:03d}"
        layout = f"gazebo_calibration_v2_layout_{index:03d}"
        signature = v1.source.v1.geometry.layout_signature(poses)
        row.update(scene_id=sid, scene_family_id=family, layout_id=layout,
                   layout_signature_sha256=signature, poses=poses)
        item.update(family_id=family, layout_id=layout, layout_signature_sha256=signature,
                    seed=seed_for(sid), geometry_provenance="fresh_v2_revision_after_invalid_v1_execution_order")
        oracle[sid] = item
        families.append({"scene_id": sid, "family_id": family, "layout_id": layout,
                         "layout_signature_sha256": signature, "split": "calibration",
                         "state": state, "relation_variant": relation,
                         "cell_repetition": repetition, "deterministic_seed": seed_for(sid)})
    scenes.update(protocol_id=PROTOCOL_ID, random_seed=14092042)
    annotations.update(protocol_id=PROTOCOL_ID, scenes=oracle)
    gate.update(protocol_id=PROTOCOL_ID,
                revision={"parent": "gazebo_calibration_v1", "parent_decision": "INVALID_PREFLIGHT_ORDER",
                          "repair_scope": "execution_order_and_fresh_identities_only", "no_sample_reuse": True})
    split.update(protocol_id=PROTOCOL_ID)
    return scenes, annotations, gate, families, split


def serialized(payload):
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--write", action="store_true"); parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write == args.check: parser.error("choose exactly one")
    values = build(); texts = [serialized(x) for x in values[:3]] + [
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in values[3]), json.dumps(values[4], indent=2, sort_keys=True) + "\n"]
    paths = [*OUTPUT_PATHS, FAMILY_MANIFEST_PATH, SPLIT_MANIFEST_PATH]
    if args.write:
        if any(p.exists() for p in paths): raise FileExistsError("refusing overwrite Calibration-v2")
        for path, text in zip(paths, texts): path.write_text(text)
        print(json.dumps({"status": "GENERATED", "families": len(values[3])}, indent=2))
    else:
        drift = [str(p) for p, text in zip(paths, texts) if not p.is_file() or p.read_text() != text]
        if drift: raise ValueError(f"Calibration-v2 drift: {drift}")
        print("PASS: deterministic fresh 128-family Calibration-v2 contract")


if __name__ == "__main__": main()
