#!/usr/bin/env python3
"""Revision 6: geometry-spaced Day-8 canary, then sealed 512-family design.

The v5 canary established that the second-ordinal layouts were physically
overlapping in the top camera.  This revision restores the certified ordinal
separation used by the Day-6 generator while retaining a fresh namespace,
seeds, prompt variants and four immutable top-camera strata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from workspace.mh_pcrau_v3 import day8_development_r2_prepare as base
from workspace.mh_pcrau_v3.day6_left_ycb_prepare import relation_positions


ROOT = base.ROOT
REPORT = base.REPORT
RAW = base.RAW
OUT = RAW / "canary_v6"
WORLDS = ROOT / "workspace/mh_pcrau_v3/generated/day8_development_r2_worlds_v1"


def world_paths() -> list[Path]:
    paths = [WORLDS / f"world_camera_{i}.sdf" for i in range(4)]
    if not all(p.is_file() for p in paths):
        raise RuntimeError("four locked top-camera worlds are required")
    return paths


def build(stage: str, state: str, relation: str, repeat: int, geometry: dict):
    scene, ann, index = base.make_scene(stage, state, relation, repeat, geometry)
    design = base.common_design(stage, relation, repeat)
    fruits = design["fruits"]
    target_index = base.rank_index(relation)
    target = fruits[target_index]
    rng = random.Random(int(hashlib.sha256(
        f"{base.NAMESPACE}|{stage}|{state}|{relation}|{repeat}|v6".encode()
    ).hexdigest()[:16], 16))

    # A common lateral shift preserves ordinal spacing and changes target UV
    # across families.  It is deliberately independent of answerability.
    shift_y = rng.uniform(-0.09, 0.09)
    positions = [[x, y + shift_y, rng.uniform(-0.25, 0.25)]
                 for x, y in relation_positions(state, relation)]
    for i, fruit in enumerate(fruits):
        if state == "ABSENT" and i == target_index:
            scene["poses"].pop(fruit, None)
        else:
            scene["poses"][fruit] = positions[i]

    # Use the locked projection calibration only for the deliberately
    # occluded object.  FOUND/AMBIGUOUS targets remain spatially separated.
    covered = target if state == "INSUFFICIENT_EVIDENCE" else design["decoy"]
    occ = design["occluder"]
    calibration = geometry["pairs"][occ][covered]
    # The inherited ABSENT template may park a state-neutral decoy outside
    # the explicit three-fruit set.  Give it a deterministic tabletop pose
    # before applying the calibrated cover.
    if covered not in scene["poses"]:
        px, py, _ = positions[target_index]
        scene["poses"][covered] = [px, py, 0.0]
    tx, ty, _ = scene["poses"][covered]
    scene["poses"][covered] = [tx, ty, 0.0]
    scene["poses"][occ] = [
        tx - float(calibration["forward_separation_m"]),
        ty + float(calibration["signed_lateral_offset_m"]),
        float(calibration.get("occluder_yaw_rad", 0.0)),
    ]

    # Keep non-target clutter outside the three ordinal positions.
    safe = [(-.49, .88), (-.39, .27), (-.28, .94), (-.16, .24), (-.08, .80), (-.46, .52)]
    for name, (x, y) in zip(base.CLUTTER, safe):
        scene["poses"][name] = [x + rng.uniform(-.018, .018), y + rng.uniform(-.018, .018), rng.uniform(-.45, .45)]

    sig = hashlib.sha256(json.dumps(scene["poses"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    scene["layout_signature_sha256"] = sig
    ann["layout_signature_sha256"] = sig
    index["layout_signature_sha256"] = sig
    return scene, ann, index


def lock_world(folder: Path, world: Path) -> None:
    path = folder / "CAPTURE_SOURCE_LOCK.json"
    lock = json.loads(path.read_text())
    lock["source_artifact_sha256"][str(Path(__file__).relative_to(ROOT))] = base.sha256(Path(__file__))
    lock["source_artifact_sha256"][str(world.relative_to(ROOT))] = base.sha256(world)
    lock["world_file"] = str(world.relative_to(ROOT))
    lock["camera_randomization"] = "four immutable top-table camera pose strata"
    lock["layout_revision"] = "v6 ordinal spacing; common state-independent lateral shift"
    path.write_text(json.dumps(lock, indent=2) + "\n")


def write(stage: str) -> None:
    geometry = json.loads(base.GEOMETRY.read_text())
    source_scenes = yaml.safe_load(base.SOURCE_SCENES.read_text())
    source_ann = yaml.safe_load(base.SOURCE_ANNOTATIONS.read_text())
    source_gate = yaml.safe_load(base.SOURCE_GATE.read_text())
    worlds = world_paths()
    if stage == "canary":
        if OUT.exists() and any(OUT.iterdir()):
            raise FileExistsError(OUT)
        if json.loads((REPORT / "CANARY_QC.json").read_text()).get("status") != "FAIL":
            raise RuntimeError("v6 must follow a preserved failed canary")
        OUT.mkdir(exist_ok=True)
        all_index = []
        for camera, relation in enumerate(base.RELATIONS):
            rows, annotations, index = [], {}, []
            for state in base.STATES:
                row, ann, item = build("canary_v6", state, relation, camera, geometry)
                rows.append(row); annotations[row["scene_id"]] = ann; index.append(item); all_index.append(item)
            folder = OUT / f"camera_{camera}"
            base.write_batch(folder, f"canary_v6_c{camera}", base.CAMERAS[0], rows, annotations, source_scenes, source_ann, source_gate)
            lock_world(folder, worlds[camera])
            (folder / "DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in index))
        (OUT / "DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in all_index))
        print(json.dumps({"status": "CANARY_V6_READY", "families": 16}, indent=2)); return

    if json.loads((REPORT / "CANARY_QC.json").read_text()).get("status") != "PASS":
        raise RuntimeError("bulk sealed until v6 canary PASS")
    bulk = RAW / "bulk_v6"
    if bulk.exists():
        raise FileExistsError(bulk)
    bulk.mkdir(); batches = {i: ([], {}, []) for i in range(4)}; all_index = []
    for state in base.STATES:
        for relation in base.RELATIONS:
            for repeat in range(32):
                row, ann, item = build("bulk_v6", state, relation, repeat, geometry)
                batch = repeat % 4
                batches[batch][0].append(row); batches[batch][1][row["scene_id"]] = ann; batches[batch][2].append(item); all_index.append(item)
    for batch, (rows, annotations, index) in batches.items():
        random.Random(base.SEED + batch).shuffle(rows)
        folder = bulk / f"batch_{batch}"
        base.write_batch(folder, f"bulk_v6_b{batch}", base.CAMERAS[0], rows, annotations, source_scenes, source_ann, source_gate)
        lock_world(folder, worlds[batch])
        (folder / "DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in index))
    (bulk / "DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in all_index))
    cells = Counter((x["state"], x["relation"]) for x in all_index)
    (bulk / "DESIGN_STATIC_QC.json").write_text(json.dumps({"families": len(all_index), "cells": {f"{s}|{r}": cells[(s,r)] for s in base.STATES for r in base.RELATIONS}}, indent=2) + "\n")
    print(json.dumps({"status": "BULK_V6_READY", "families": len(all_index)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("canary", "bulk")); write(parser.parse_args().stage)
