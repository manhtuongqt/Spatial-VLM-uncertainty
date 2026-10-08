"""Create immutable source/config locks for the four pilot capture batches."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json

from .audit_development import ROOT, sha256
from .generate_gazebo_pilot_512 import OUT


def frozen_required_sources() -> set[str]:
    path = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_SOURCE_ARTIFACTS"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise ValueError("Capture node required source list missing")


def main() -> None:
    plan = json.loads((OUT / "CAPTURE_PLAN.json").read_text())
    common = frozen_required_sources() | {
        "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
        "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
        "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py",
        "workspace/mh_pcrau_v3/g1/generate_gazebo_pilot_512.py",
        "workspace/mh_pcrau_v3/g1/lock_gazebo_pilot_512.py",
    }
    for batch in plan["batches"]:
        folder = OUT / f"batch_{batch['batch']}"
        lock_path = folder / "CAPTURE_SOURCE_LOCK.json"
        if lock_path.exists():
            raise FileExistsError(lock_path)
        paths = common | {str((folder / name).relative_to(ROOT))
                          for name in ("scenes.yaml", "annotations.yaml", "gate.yaml")}
        hashes = {name: sha256(ROOT / name) for name in sorted(paths)}
        lock = {
            "schema_version": 1, "protocol_id": batch["protocol_id"],
            "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "workspace_root": str(ROOT),
            "source_artifact_sha256": hashes,
            "model_inventory_sha256": "NO_MODEL_INFERENCE_V3_PILOT_CAPTURE",
            "capture_role": "PILOT_DEVELOPMENT_ONLY",
            "authorization": "User requested fresh Gazebo pilot images and 512-image pilot on 2026-09-24; append-only deviation from Day 1 no-capture boundary",
            "planned_scene_count": 64,
            "view_joint_pose": batch["view_joint_pose"],
            "seals": {"calibration": True, "test_iid": True, "test_ood": True},
        }
        lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "LOCKED", "batch_locks": 4}))


if __name__ == "__main__":
    main()
