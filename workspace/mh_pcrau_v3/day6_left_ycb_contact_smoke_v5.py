#!/usr/bin/env python3
"""Preregister a repeat of the v4 layout against contact-exact world v5."""

from __future__ import annotations

from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_grounding_smoke_v4"
OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_contact_smoke_v5"
WORLD = ROOT / "workspace/mh_pcrau_v3/generated/day6_left_ycb_v5_contact_exact/left_ycb_top30_contact_exact.sdf"
CAPTURE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
PROTOCOL = "mh_pcrau_v3_left_ycb_contact_smoke_v5"


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
    payloads = {}
    for filename in ("scenes.yaml", "annotations.yaml", "gate.yaml"):
        payload = yaml.safe_load((SOURCE / filename).read_text(encoding="utf-8"))
        payload["protocol_id"] = PROTOCOL
        payloads[filename] = payload
    scene = payloads["scenes.yaml"]["scenes"][0]
    scene["scene_id"] = "contact_smoke_v5_000"
    scene["scene_family_id"] = "mh_pcrau_v3/contact_smoke_v5/000"
    scene["layout_id"] = scene["scene_id"]
    old_annotation = next(iter(payloads["annotations.yaml"]["scenes"].values()))
    old_annotation["family_id"] = scene["scene_family_id"]
    old_annotation["layout_id"] = scene["layout_id"]
    old_annotation["failure_tags"] = [
        "contact_exact_visual_qc_only", "no_model_access", "lemon_excluded"
    ]
    payloads["annotations.yaml"]["scenes"] = {scene["scene_id"]: old_annotation}

    OUT.mkdir(parents=True)
    for filename, payload in payloads.items():
        (OUT / filename).write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=160),
            encoding="utf-8",
        )
    relative = lambda path: str(path.relative_to(ROOT))
    artifacts = required_sources() | {
        relative(WORLD),
        relative(Path(__file__)),
        relative(ROOT / "workspace/mh_pcrau_v3/day6_left_ycb_grounded_world_v5.py"),
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
        "source_artifact_sha256": {name: sha256(ROOT / name) for name in sorted(artifacts)},
        "model_inventory_sha256": "NO_MODEL_INFERENCE_CONTACT_SMOKE",
        "capture_role": "CONTACT_EXACT_VISUAL_QC_ONLY",
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
