"""Append-only eight-scene Gazebo geometry canary for G1 remediation.

This is development characterization, never a training or Test release.
It intentionally preserves the original object identities and the relative
target/occluder geometry while translating a complete layout together.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

from .audit_development import ROOT, sha256
from .generate_gazebo_pilot_512 import BASE_POSE
from .lock_gazebo_pilot_512 import frozen_required_sources
from .qc_gazebo_pilot_512 import check_state


OUT = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03/gazebo_canary_revision_v2"
CONFIG = ROOT / "ur3/ur3_perception/config"
PROTOCOL = "mh_pcrau_v3_g1_canary_revision_v2"
INDICES = (0, 1, 4, 5, 8, 9, 12, 13)


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    sys.path.insert(0, str(ROOT / "protocol"))
    import generate_gazebo_train_uq_v2_pilot_r5_contract as geometry

    original = yaml.safe_load((CONFIG / "gazebo_train_uq_v2_full_r3_scenes.yaml").read_text())
    oracle = yaml.safe_load((CONFIG / "gazebo_train_uq_v2_full_r3_annotations.yaml").read_text())
    gate = yaml.safe_load((CONFIG / "gazebo_train_uq_v2_full_r3_gate.yaml").read_text())
    rows, annotations = [], {}
    # Keep all relative offsets, especially the millimetre-scale IE occluder gap.
    # The 6-mm translation is a development probe, not proof of novelty.
    for local, index in enumerate(INDICES):
        source = original["scenes"][index]
        sid = f"g1_canary_v2_{local:02d}"
        poses = {name: [pose[0] + 0.006, pose[1] + 0.006, pose[2]]
                 for name, pose in source["poses"].items()}
        signature = geometry.layout_signature(poses)
        row = dict(source)
        row.update(scene_id=sid, scene_family_id=f"g1_canary_v2/parent_{local:02d}",
                   layout_id=sid, layout_signature_sha256=signature, poses=poses)
        ann = dict(oracle["scenes"][source["scene_id"]])
        ann.update(family_id=row["scene_family_id"], layout_id=sid,
                   layout_signature_sha256=signature, split="development_canary_only",
                   view_id="original_view", template_provenance=source["scene_id"])
        rows.append(row)
        annotations[sid] = ann
    original.update(protocol_id=PROTOCOL, expected_scene_count=len(rows),
                    random_seed=24092027, view_joint_pose=BASE_POSE, scenes=rows)
    oracle.update(protocol_id=PROTOCOL, oracle_usage="development_geometry_qc_only",
                  scenes=annotations)
    gate.update(protocol_id=PROTOCOL, parent_family_count=len(rows),
                split_parent_family_count={"development_canary_only": len(rows)},
                state_quota={state: 2 for state in geometry.STATES},
                relation_variant_quota={"leftmost": 4, "rightmost": 4},
                state_relation_cell_quota=1,
                camera={**gate["camera"], "view_joint_pose": BASE_POSE},
                policies={**gate["policies"], "no_training": True,
                          "no_model_inference": True, "no_materialization": True})
    OUT.mkdir(parents=True)
    for name, payload in (("scenes.yaml", original), ("annotations.yaml", oracle),
                          ("gate.yaml", gate)):
        (OUT / name).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True,
                                              width=140), encoding="utf-8")
    common = frozen_required_sources() | {
        "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
        "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
        "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py",
        "workspace/mh_pcrau_v3/g1/canary_scene_revision.py",
    }
    common |= {str((OUT / name).relative_to(ROOT)) for name in
               ("scenes.yaml", "annotations.yaml", "gate.yaml")}
    lock = {
        "schema_version": 1, "protocol_id": PROTOCOL,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "source_artifact_sha256": {name: sha256(ROOT / name) for name in sorted(common)},
        "model_inventory_sha256": "NO_MODEL_INFERENCE_V3_PILOT_CAPTURE",
        "capture_role": "DEVELOPMENT_CANARY_ONLY",
        "authorization": "User requested scene/view redesign and canary before补 capture on 2026-09-24",
        "planned_scene_count": len(rows), "view_joint_pose": BASE_POSE,
        "seals": {"calibration": True, "test_iid": True, "test_ood": True},
    }
    (OUT / "CAPTURE_SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "LOCKED", "canary_scenes": len(rows), "folder": str(OUT)}))


def qc() -> None:
    destination = OUT / "CANARY_QC.json"
    if destination.exists():
        raise FileExistsError(destination)
    capture = OUT / "capture_attempt_01"
    manifest = json.loads((capture / "capture_manifest.json").read_text())
    scene_config = yaml.safe_load((OUT / "scenes.yaml").read_text())
    annotations = yaml.safe_load((OUT / "annotations.yaml").read_text())
    gate = yaml.safe_load((OUT / "gate.yaml").read_text())
    inputs = [json.loads(line) for line in (capture / "input_manifest.jsonl").read_text().splitlines()]
    scenes = {row["scene_id"]: row for row in scene_config["scenes"]}
    rows = []
    for item in inputs:
        sid = item["scene_id"]
        label_map = cv2.imread(str(capture / sid / "evaluator/semantic_labels.png"), cv2.IMREAD_UNCHANGED)
        if label_map is None:
            raise FileNotFoundError(sid)
        if label_map.ndim == 3:
            label_map = label_map[:, :, 0]
        valid, detail = check_state(annotations["scenes"][sid], scenes[sid], label_map,
                                    capture, item, gate)
        rows.append({"scene_id": sid, "state": annotations["scenes"][sid]["state"],
                     "verified": valid, "geometry": detail})
    integrity = (manifest.get("status") == "COMPLETE" and manifest.get("scene_count") == 8
                 and len(inputs) == 8 and manifest.get("input_only_sensor_qc", {}).get("passed") is True
                 and manifest.get("pretrial_source_lock_sha256") == sha256(OUT / "CAPTURE_SOURCE_LOCK.json"))
    result = {"status": "PASS" if integrity and all(row["verified"] for row in rows) else "FAIL",
              "integrity": integrity, "verified": sum(row["verified"] for row in rows),
              "total": len(rows), "rows": rows,
              "rule": "Do not capture bulk unless all eight canary states verify and novelty/leakage review follows."}
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "verified": result["verified"], "total": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "qc"))
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else qc)()
