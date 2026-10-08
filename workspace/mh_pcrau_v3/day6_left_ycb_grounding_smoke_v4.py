#!/usr/bin/env python3
"""Preregister one no-model capture that visually verifies grounded YCB meshes."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_projection_canary_v4"
OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_grounding_smoke_v4"
WORLD = ROOT / "workspace/mh_pcrau_v3/generated/day6_left_ycb_v4_grounded/left_ycb_top30_grounded.sdf"
CAPTURE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
PROTOCOL = "mh_pcrau_v3_left_ycb_grounding_smoke_v4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_sources() -> set[str]:
    tree = ast.parse(CAPTURE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_SOURCE_ARTIFACTS"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("required capture inventory missing")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only output exists: {OUT}")
    source_scenes = yaml.safe_load((SOURCE_DIR / "scenes.yaml").read_text(encoding="utf-8"))
    source_annotations = yaml.safe_load((SOURCE_DIR / "annotations.yaml").read_text(encoding="utf-8"))
    source_gate = yaml.safe_load((SOURCE_DIR / "gate.yaml").read_text(encoding="utf-8"))
    original = next(row for row in source_scenes["scenes"] if row["scene_id"] == "pcal_2_4_20")
    scene = deepcopy(original)
    scene.update({
        "scene_id": "grounding_smoke_v4_000",
        "scene_family_id": "mh_pcrau_v3/grounding_smoke_v4/000",
        "layout_id": "grounding_smoke_v4_000",
        "task_type": "grounding_visual_qc_only",
        "instruction": "Visual grounding smoke check; no model inference.",
    })
    # Keep the bleach/pear pair central and expose the other newly corrected
    # non-lemon meshes elsewhere on the same table for a single-frame audit.
    scene["poses"]["ycb_plum"] = [-0.08, 0.76, 0.0]
    scene["poses"]["ycb_mug"] = [-0.06, 0.18, 0.0]
    scene["poses"]["ycb_lemon"] = [5.34, 3.0, 0.0]
    signature = hashlib.sha256(
        json.dumps(scene["poses"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    scene["layout_signature_sha256"] = signature

    scenes = deepcopy(source_scenes)
    scenes.update({"protocol_id": PROTOCOL, "expected_scene_count": 1, "scenes": [scene]})
    annotation = deepcopy(source_annotations["scenes"]["pcal_2_4_20"])
    annotation.update({
        "family_id": scene["scene_family_id"],
        "layout_id": scene["layout_id"],
        "layout_signature_sha256": signature,
        "split": "grounding_smoke_only",
        "failure_tags": ["visual_grounding_qc_only", "no_model_access", "lemon_excluded"],
    })
    annotations = deepcopy(source_annotations)
    annotations.update({"protocol_id": PROTOCOL, "scenes": {scene["scene_id"]: annotation}})
    gate = deepcopy(source_gate)
    gate.update({
        "protocol_id": PROTOCOL,
        "parent_family_count": 1,
        "split_parent_family_count": {"grounding_smoke_only": 1},
        "policies": {
            **source_gate["policies"],
            "no_training": True,
            "no_model_inference": True,
            "lemon_excluded_from_next_protocol": True,
        },
    })

    OUT.mkdir(parents=True)
    for name, payload in (("scenes.yaml", scenes), ("annotations.yaml", annotations), ("gate.yaml", gate)):
        (OUT / name).write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=160),
            encoding="utf-8",
        )

    relative = lambda path: str(path.relative_to(ROOT))
    artifacts = required_sources() | {
        relative(WORLD),
        relative(Path(__file__)),
        relative(ROOT / "workspace/mh_pcrau_v3/day6_left_ycb_grounded_world_v4.py"),
        relative(OUT / "scenes.yaml"),
        relative(OUT / "annotations.yaml"),
        relative(OUT / "gate.yaml"),
        "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py",
    }
    lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "source_artifact_sha256": {
            name: sha256(ROOT / name) for name in sorted(artifacts)
        },
        "model_inventory_sha256": "NO_MODEL_INFERENCE_GROUNDING_SMOKE",
        "capture_role": "VISUAL_GROUNDING_QC_ONLY",
        "planned_scene_count": 1,
        "world_file": relative(WORLD),
        "lemon_policy": "EXCLUDED_FROM_NEXT_CANARY_AND_FINAL_FAMILY",
    }
    (OUT / "CAPTURE_SOURCE_LOCK.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    print(OUT)


if __name__ == "__main__":
    main()
