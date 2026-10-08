#!/usr/bin/env python3
"""Create the legacy-capture-compatible execution lock for Train-UQ v1.

The ROS capture node has a fixed historical status literal.  This adapter lock
does not relax the primary contract: it inherits its SHA-256 and repeats every
legacy source hash required by the capture node before any scene is reset.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "protocol/gazebo_train_uq_v1_contract_lock.json"
OUT = ROOT / "protocol/gazebo_train_uq_v1_capture_compatibility_lock.json"
PROTOCOL_ID = "gazebo_train_uq_v1"
LEGACY_REQUIRED = {
    "RoboRefer/API/api.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_capture.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_runner.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_validate.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_evaluator.py",
    "ur3/ur3_perception/scripts/roborefer_grounder.py",
    "ur3/ur3_perception/scripts/spatial_point_utils.py",
    "ur3/ur3_perception/scripts/move_camera_to_view.py",
    "ur3/ur3_perception/launch/roborefer_pilot_capture.launch.py",
    "ur3/ur3_perception/config/roborefer_pilot_v0_scenes.yaml",
    "ur3/ur3_perception/config/roborefer_pilot_v0_gate.yaml",
    "ur3/ur3_perception/config/roborefer_pilot_v0_annotations.yaml",
    "ur3/ur3_perception/schemas/roborefer_pilot_prediction.schema.json",
    "ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro",
    "ur3/ur3_moveit_control/launch/ur3_susgrip_sim.launch.py",
    "ur3/ur_simulation_gz/worlds/ur3_pick_place.sdf",
    "ur3/ur_simulation_gz/launch/ur_sim_control.launch.py",
}
TRAIN_UQ_CONFIGS = {
    "ur3/ur3_perception/config/gazebo_train_uq_v1_scenes.yaml",
    "ur3/ur3_perception/config/gazebo_train_uq_v1_annotations.yaml",
    "ur3/ur3_perception/config/gazebo_train_uq_v1_gate.yaml",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite execution lock: {OUT}")
    primary = json.loads(PRIMARY.read_text(encoding="utf-8"))
    if primary.get("protocol_id") != PROTOCOL_ID or primary.get("status") != "LOCKED_BEFORE_CAPTURE_AND_TRAINING":
        raise ValueError("primary Train-UQ contract lock is invalid")
    sources = {}
    for relative in sorted(LEGACY_REQUIRED | TRAIN_UQ_CONFIGS):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[relative] = sha256(path)
    for relative, expected in primary["source_artifact_sha256"].items():
        if relative in sources and sources[relative] != expected:
            raise ValueError(f"primary source drift: {relative}")
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "parent_contract_lock": str(PRIMARY.relative_to(ROOT)),
        "parent_contract_lock_sha256": sha256(PRIMARY),
        "purpose": "Compatibility adapter for the frozen legacy ROS capture-node status check; does not authorize training or evaluation.",
        "source_artifact_sha256": sources,
        "model_inventory_sha256": primary["model_inventory_sha256"],
        "policies_inherited": primary["policies"],
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "LOCKED", "path": str(OUT), "sha256": sha256(OUT), "parent_sha256": payload["parent_contract_lock_sha256"], "sources": len(sources)}, indent=2))


if __name__ == "__main__":
    main()
