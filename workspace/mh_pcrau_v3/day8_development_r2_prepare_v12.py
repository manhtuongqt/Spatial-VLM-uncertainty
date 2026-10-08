#!/usr/bin/env python3
"""Second and final targeted repair: one ambiguous canary scene only."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workspace.mh_pcrau_v3 import day8_development_r2_prepare_v7 as v7
from workspace.mh_pcrau_v3 import day8_development_r2_prepare_v11 as v11

OUT = v7.RAW / "canary_repair_v12"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    # The prior repair was captured and measured; only this one cell is
    # replaced.  +3.3 cm follows the measured camera-3 projection slope.
    scene, annotation, index = v11.replacement(3, "AMBIGUOUS", "second_from_right", "canary_v7")
    token = hashlib.sha256(f"{v7.base.NAMESPACE}|canary_repair_v12|ambiguous|second_from_right".encode()).hexdigest()[:20]
    sid = f"d8r2_canary_repair_v12_{token}"
    family = f"{v7.base.NAMESPACE}/canary_repair_v12/{token}"
    scene["poses"]["ycb_pear"][1] += .033
    signature = hashlib.sha256(json.dumps(scene["poses"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    scene.update({"scene_id": sid, "scene_family_id": family, "layout_id": sid, "layout_signature_sha256": signature})
    annotation.update({
        "family_id": family, "layout_id": sid, "layout_signature_sha256": signature,
        "seed": int(token[:12], 16), "camera_stratum": 3,
        "failure_tags": list(annotation.get("failure_tags", [])) + [
            "targeted_repair_of_repair_v11_only", "pear_projection_offset_plus_0p033m",
        ],
    })
    index.update({"scene_id": sid, "family_id": family, "seed": annotation["seed"], "layout_signature_sha256": signature})
    worlds = v7.worlds()
    src_scenes = yaml.safe_load(v7.base.SOURCE_SCENES.read_text())
    src_annotations = yaml.safe_load(v7.base.SOURCE_ANNOTATIONS.read_text())
    src_gate = yaml.safe_load(v7.base.SOURCE_GATE.read_text())
    OUT.mkdir(parents=True)
    folder = OUT / "camera_3"
    v7.base.write_batch(folder, "canary_repair_v12_c3", v7.base.CAMERAS[0], [scene], {sid: annotation}, src_scenes, src_annotations, src_gate)
    v7.lock(folder, worlds[3])
    (folder / "DESIGN_INDEX.jsonl").write_text(json.dumps(index, sort_keys=True) + "\n")
    (OUT / "REPAIR_SCOPE.json").write_text(json.dumps({
        "status": "LOCKED_BEFORE_CAPTURE", "accepted_v10_observations_retained": 14,
        "accepted_v11_ie_repair_retained": 1, "replacement_observations": 1,
        "replaced_repair_v11_scene_id": "d8r2_canary_repair_v11_2e91a6b7a2dd407ebbb0",
        "repair_cell": "AMBIGUOUS|second_from_right", "pear_y_offset_m": .033,
        "no_bulk_capture": True,
    }, indent=2) + "\n")
    print(json.dumps({"status": "TARGETED_REPAIR_V12_READY", "families": 1, "retained": 15}, indent=2))


if __name__ == "__main__":
    main()
