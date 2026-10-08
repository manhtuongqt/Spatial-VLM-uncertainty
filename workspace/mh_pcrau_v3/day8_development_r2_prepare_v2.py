#!/usr/bin/env python3
"""Day-8 canary revision 2 after the preserved out-of-view canary failure."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workspace.mh_pcrau_v3.day8_development_r2_prepare as base


OUT = base.RAW / "canary_v2"


def revised_positions(state: str, relation: str, design: dict, rng: random.Random) -> list[list[float]]:
    # This is the empirically visible world-v5 region used by the successful
    # Day-6 canary, while retaining a broad state-independent translation.
    c = min(.50, max(.32, design["center_y"] * .52 + .10))
    s = min(.13, max(.075, design["spacing"] * .66))
    dx = rng.uniform(-.055, .055)
    if state != "AMBIGUOUS":
        ys, xs = ((c + s, c, c - s), (-.31 + dx, -.24 + dx, -.17 + dx))
    elif relation == "leftmost":
        ys, xs = ((c + s, c + s, c - s), (-.31, -.17, -.24))
    elif relation == "rightmost":
        ys, xs = ((c + s, c - s, c - s), (-.24, -.31, -.17))
    elif relation == "second_from_left":
        ys, xs = ((c + s, c, c), (-.24, -.31, -.17))
    else:
        ys, xs = ((c, c, c - s), (-.31, -.17, -.24))
    return [[float(x), float(y), rng.uniform(-.35, .35)] for x, y in zip(xs, ys)]


def revised_scene(stage: str, state: str, relation: str, repeat: int, geometry: dict):
    original = base.ordered_positions
    base.ordered_positions = revised_positions
    try:
        scene, ann, index = base.make_scene(stage, state, relation, repeat, geometry)
    finally:
        base.ordered_positions = original
    design = base.common_design(stage, relation, repeat)
    fruits = design["fruits"]
    target_index = base.rank_index(relation)
    if state == "INSUFFICIENT_EVIDENCE": covered = fruits[target_index]
    elif state == "FOUND": covered = fruits[(target_index + 1) % 3]
    elif state == "AMBIGUOUS":
        tied = {fruits[i] for i in base.tie_indices(relation)}
        covered = next(x for x in fruits if x not in tied)
    else: covered = design["decoy"]
    occluder = design["occluder"]
    calibration = geometry["pairs"][occluder][covered]
    _, ty, _ = scene["poses"][covered]
    tx = -.10 if occluder == "ycb_sugar_box" and covered == "ycb_apple" else -.24
    scene["poses"][covered] = [tx, ty, 0.0]
    scene["poses"][occluder] = [tx - float(calibration["forward_separation_m"]), ty + float(calibration["signed_lateral_offset_m"]), float(calibration.get("occluder_yaw_rad", 0.0))]
    signature = hashlib.sha256(json.dumps(scene["poses"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    scene["layout_signature_sha256"] = signature
    ann["layout_signature_sha256"] = signature
    index["layout_signature_sha256"] = signature
    return scene, ann, index


def add_revision_to_lock(folder: Path, reason: str) -> None:
    lock_path = folder / "CAPTURE_SOURCE_LOCK.json"
    lock = json.loads(lock_path.read_text())
    rel = str(Path(__file__).relative_to(ROOT))
    lock["source_artifact_sha256"][rel] = base.sha256(Path(__file__))
    lock["revision_reason"] = reason
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")


def main(stage: str) -> None:
    geometry = json.loads(base.GEOMETRY.read_text())
    src_scenes = yaml.safe_load(base.SOURCE_SCENES.read_text())
    src_ann = yaml.safe_load(base.SOURCE_ANNOTATIONS.read_text())
    src_gate = yaml.safe_load(base.SOURCE_GATE.read_text())
    if stage == "canary":
        if OUT.exists():
            raise FileExistsError(f"append-only canary revision exists: {OUT}")
        first = base.REPORT / "CANARY_QC.json"
        if not first.is_file() or json.loads(first.read_text()).get("status") != "FAIL":
            raise RuntimeError("revision 2 requires a preserved canary-attempt-1 FAIL")
        rows, anns, index = [], {}, []
        for state in base.STATES:
            for ri, relation in enumerate(base.RELATIONS):
                row, ann, item = revised_scene("canary_v2", state, relation, ri, geometry)
                rows.append(row); anns[row["scene_id"]] = ann; index.append(item)
        base.write_batch(OUT, "canary_v2", base.CAMERAS[0], rows, anns, src_scenes, src_ann, src_gate)
        (OUT / "DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in index))
        add_revision_to_lock(OUT, "Attempt 1 preserved FAIL: layouts outside visible world-v5 domain and occlusion calibration translated off its validated x coordinate.")
        print(json.dumps({"status": "CANARY_V2_READY", "families": len(index), "path": str(OUT.relative_to(ROOT))}, indent=2))
        return

    canary_qc = base.REPORT / "CANARY_QC.json"
    if not canary_qc.is_file() or json.loads(canary_qc.read_text()).get("status") != "PASS":
        raise RuntimeError("bulk is sealed until revised CANARY_QC is PASS")
    bulk = base.RAW / "bulk"
    if bulk.exists():
        raise FileExistsError(f"append-only bulk design exists: {bulk}")
    bulk.mkdir()
    batches = {i: ([], {}, []) for i in range(4)}; all_index = []
    for state in base.STATES:
        for relation in base.RELATIONS:
            for repeat in range(32):
                row, ann, item = revised_scene("bulk", state, relation, repeat, geometry)
                batch = repeat % 4
                batches[batch][0].append(row); batches[batch][1][row["scene_id"]] = ann; batches[batch][2].append(item); all_index.append(item)
    for batch, (rows, anns, index) in batches.items():
        random.Random(base.SEED + batch).shuffle(rows)
        folder = bulk / f"batch_{batch}"
        base.write_batch(folder, f"bulk_v2_b{batch}", base.CAMERAS[batch], rows, anns, src_scenes, src_ann, src_gate)
        add_revision_to_lock(folder, "Bulk uses canary-v2 visible-domain and projection-calibrated geometry.")
        (folder / "DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in index))
    (bulk / "DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in all_index))
    cells = Counter((x["state"], x["relation"]) for x in all_index)
    summary = {"schema_version": 1, "status": "PASS_CAPTURE_NOT_STARTED", "families": len(all_index), "states": dict(Counter(x["state"] for x in all_index)), "relations": dict(Counter(x["relation"] for x in all_index)), "splits": dict(Counter(x["split"] for x in all_index)), "cells": {f"{s}|{r}": cells[(s, r)] for s in base.STATES for r in base.RELATIONS}, "unique_family_ids": len({x["family_id"] for x in all_index}), "unique_seeds": len({x["seed"] for x in all_index}), "unique_layouts": len({x["layout_signature_sha256"] for x in all_index})}
    (bulk / "DESIGN_STATIC_QC.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("canary", "bulk"))
    main(parser.parse_args().stage)
