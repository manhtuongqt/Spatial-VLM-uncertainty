#!/usr/bin/env python3
"""Prepare only the two failed Day-8 canary cells for a targeted repair.

The other fourteen v10 captures are immutable accepted observations.  Each
replacement reuses a geometry that already passed the same semantic gate at
the same camera stratum, but receives a fresh scene/family/seed namespace.
"""
from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workspace.mh_pcrau_v3 import day8_development_r2_prepare_v7 as v7

OUT = v7.RAW / "canary_repair_v11"
REPAIRS = (
    # (camera stratum, state, relation, previously passing source revision)
    (0, "INSUFFICIENT_EVIDENCE", "leftmost", "canary_v8"),
    (3, "AMBIGUOUS", "second_from_right", "canary_v7"),
)


def source_scene(revision: str, camera: int, state: str, relation: str) -> tuple[dict, dict]:
    folder = v7.RAW / revision / f"camera_{camera}"
    scenes = yaml.safe_load((folder / "scenes.yaml").read_text())["scenes"]
    annotations = yaml.safe_load((folder / "annotations.yaml").read_text())["scenes"]
    for scene in scenes:
        annotation = annotations[scene["scene_id"]]
        if annotation["state"] == state and annotation["relation_variant"] == relation:
            return deepcopy(scene), deepcopy(annotation)
    raise RuntimeError(f"no passing source configuration for {revision}/{camera}/{state}/{relation}")


def replacement(camera: int, state: str, relation: str, source_revision: str) -> tuple[dict, dict, dict]:
    scene, annotation = source_scene(source_revision, camera, state, relation)
    token = hashlib.sha256(
        f"{v7.base.NAMESPACE}|canary_repair_v11|{camera}|{state}|{relation}|{source_revision}".encode()
    ).hexdigest()[:20]
    scene_id = f"d8r2_canary_repair_v11_{token}"
    family_id = f"{v7.base.NAMESPACE}/canary_repair_v11/{token}"
    signature = hashlib.sha256(
        json.dumps(scene["poses"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    scene.update({
        "scene_id": scene_id,
        "scene_family_id": family_id,
        "layout_id": scene_id,
        "layout_signature_sha256": signature,
    })
    annotation.update({
        "family_id": family_id,
        "layout_id": scene_id,
        "layout_signature_sha256": signature,
        "split": "canary",
        "seed": int(token[:12], 16),
        "camera_stratum": camera,
        "failure_tags": list(annotation.get("failure_tags", [])) + [
            "targeted_repair_of_v10_only",
            f"projection_geometry_reused_from_{source_revision}_pass",
        ],
    })
    index = {
        "scene_id": scene_id,
        "family_id": family_id,
        "state": state,
        "relation": relation,
        "split": "canary",
        "seed": annotation["seed"],
        "camera_stratum": camera,
        "prompt_variant": annotation["prompt_variant"],
        "layout_signature_sha256": signature,
        "repair_of": {
            "capture_revision": "canary_v10",
            "reason": "semantic_gate_failure",
            "source_passing_revision": source_revision,
        },
    }
    return scene, annotation, index


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    previous = json.loads((v7.REPORT / "CANARY_QC.json").read_text())
    if previous.get("capture_revision") != "canary_v10" or previous.get("status") != "FAIL":
        raise RuntimeError("targeted repair requires the failed canary_v10 QC as its immutable base")
    worlds = v7.worlds()
    src_scenes = yaml.safe_load(v7.base.SOURCE_SCENES.read_text())
    src_annotations = yaml.safe_load(v7.base.SOURCE_ANNOTATIONS.read_text())
    src_gate = yaml.safe_load(v7.base.SOURCE_GATE.read_text())
    OUT.mkdir(parents=True)
    all_index = []
    for camera, state, relation, revision in REPAIRS:
        scene, annotation, index = replacement(camera, state, relation, revision)
        folder = OUT / f"camera_{camera}"
        v7.base.write_batch(
            folder, f"canary_repair_v11_c{camera}", v7.base.CAMERAS[0],
            [scene], {scene["scene_id"]: annotation}, src_scenes, src_annotations, src_gate,
        )
        v7.lock(folder, worlds[camera])
        (folder / "DESIGN_INDEX.jsonl").write_text(json.dumps(index, sort_keys=True) + "\n")
        all_index.append(index)
    (OUT / "DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in all_index))
    (OUT / "REPAIR_SCOPE.json").write_text(json.dumps({
        "status": "LOCKED_BEFORE_CAPTURE",
        "accepted_v10_observations_retained": 14,
        "replacement_observations": 2,
        "replaced_v10_scene_ids": [
            "d8r2_canary_v10_f81a2e34aa3b0912fbbe",
            "d8r2_canary_v10_1aa6e6c0d57607d4ba77",
        ],
        "repair_cells": [f"{state}|{relation}" for _, state, relation, _ in REPAIRS],
        "no_bulk_capture": True,
    }, indent=2) + "\n")
    print(json.dumps({"status": "TARGETED_REPAIR_READY", "families": 2, "retained": 14}, indent=2))


if __name__ == "__main__":
    main()
