"""Append-only targeted IE occluder-gap sweep after full revision-v2 QC.

Each failed family gets four preregistered panel-offset candidates. A repaired
view is selected only by geometry QC, never by model prediction or Test score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import sys

import yaml

from .audit_development import ROOT, sha256
from .capture_revision_v2 import OUT as PILOT, BASE_POSE, CONFIG, frozen_required_sources
from .qc_gazebo_pilot_512 import check_state


OUT = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03/gazebo_ie_repair_revision_v2"
PROTOCOL = "mh_pcrau_v3_g1_ie_repair_v2"
DELTAS = (-0.0020, -0.0010, 0.0010, 0.0020)


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    from protocol.gazebo_train_uq_v2_pilot_r4_pipeline import labels
    sys.path.insert(0, str(ROOT / "protocol"))
    original = yaml.safe_load((CONFIG / "gazebo_train_uq_v2_full_r3_scenes.yaml").read_text())
    oracle = yaml.safe_load((CONFIG / "gazebo_train_uq_v2_full_r3_annotations.yaml").read_text())
    gate = yaml.safe_load((CONFIG / "gazebo_train_uq_v2_full_r3_gate.yaml").read_text())
    import generate_gazebo_train_uq_v2_pilot_r5_contract as geometry
    failed = []
    for batch in range(4):
        folder = PILOT / f"batch_{batch}"
        capture = folder / "capture_attempt_01"
        scene_config = yaml.safe_load((folder / "scenes.yaml").read_text())
        annotation = yaml.safe_load((folder / "annotations.yaml").read_text())
        source_gate = yaml.safe_load((folder / "gate.yaml").read_text())
        scenes = {row["scene_id"]: row for row in scene_config["scenes"]}
        inputs = [json.loads(line) for line in (capture / "input_manifest.jsonl").read_text().splitlines()]
        if len(inputs) != 64:
            raise ValueError(f"Incomplete batch {batch}")
        for item in inputs:
            sid = item["scene_id"]
            ann = annotation["scenes"][sid]
            valid, _ = check_state(ann, scenes[sid],
                                   labels(capture / sid / "evaluator/semantic_labels.png"),
                                   capture, item, source_gate)
            if not valid:
                if ann["state"] != "INSUFFICIENT_EVIDENCE":
                    raise ValueError(f"Non-IE failure {sid}; do not auto-repair")
                failed.append((scenes[sid], ann))
    rows, annotations, mapping = [], {}, []
    for family_index, (scene, ann) in enumerate(failed):
        for variant, delta in enumerate(DELTAS):
            sid = f"g1_ie_repair_v2_{family_index:03d}_{variant}"
            poses = {name: list(pose) for name, pose in scene["poses"].items()}
            poses["uq_neutral_occluder"][1] += delta
            signature = geometry.layout_signature(poses)
            row = dict(scene)
            row.update(scene_id=sid, layout_id=sid, layout_signature_sha256=signature,
                       poses=poses)
            new_ann = dict(ann)
            new_ann.update(layout_id=sid, layout_signature_sha256=signature,
                           panel_y_delta_m=delta, split="development_ie_repair_only")
            rows.append(row)
            annotations[sid] = new_ann
            mapping.append({"scene_id": sid, "original_failed_scene": scene["scene_id"],
                            "parent_family_id": ann["family_id"], "panel_y_delta_m": delta})
    OUT.mkdir(parents=True)
    original.update(protocol_id=PROTOCOL, expected_scene_count=len(rows),
                    random_seed=24092031, view_joint_pose=BASE_POSE, scenes=rows)
    oracle.update(protocol_id=PROTOCOL, oracle_usage="development_geometry_qc_only",
                  scenes=annotations)
    gate.update(protocol_id=PROTOCOL, parent_family_count=len(failed),
                split_parent_family_count={"development_ie_repair_only": len(failed)},
                state_quota={"INSUFFICIENT_EVIDENCE": len(rows)},
                relation_variant_quota=dict(Counter(ann["relation_variant"] for ann in annotations.values())),
                state_relation_cell_quota=None,
                camera={**gate["camera"], "view_joint_pose": BASE_POSE},
                policies={**gate["policies"], "no_training": True,
                          "no_model_inference": True, "no_materialization": True})
    for name, payload in (("scenes.yaml", original), ("annotations.yaml", oracle),
                          ("gate.yaml", gate)):
        (OUT / name).write_text(yaml.safe_dump(payload, sort_keys=False,
                                               allow_unicode=True, width=140))
    common = frozen_required_sources() | {
        "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
        "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
        "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py",
        "workspace/mh_pcrau_v3/g1/repair_ie_revision_v2.py",
    }
    common |= {str((OUT / name).relative_to(ROOT)) for name in
               ("scenes.yaml", "annotations.yaml", "gate.yaml")}
    lock = {"schema_version": 1, "protocol_id": PROTOCOL,
            "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "workspace_root": str(ROOT),
            "source_artifact_sha256": {name: sha256(ROOT / name) for name in sorted(common)},
            "model_inventory_sha256": "NO_MODEL_INFERENCE_V3_PILOT_CAPTURE",
            "capture_role": "DEVELOPMENT_IE_REPAIR_ONLY",
            "authorization": "User requested canary then QC-directed supplementary capture on 2026-09-24",
            "planned_scene_count": len(rows), "view_joint_pose": BASE_POSE,
            "seals": {"calibration": True, "test_iid": True, "test_ood": True}}
    (OUT / "CAPTURE_SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n")
    (OUT / "REPAIR_PLAN.json").write_text(json.dumps(mapping, indent=2) + "\n")
    print(json.dumps({"failed_families": len(failed), "planned_variants": len(rows)}))


def qc() -> None:
    if (OUT / "REPAIR_QC.json").exists():
        raise FileExistsError(OUT / "REPAIR_QC.json")
    from protocol.gazebo_train_uq_v2_pilot_r4_pipeline import labels
    capture = OUT / "capture_attempt_01"
    manifest = json.loads((capture / "capture_manifest.json").read_text())
    scene_config = yaml.safe_load((OUT / "scenes.yaml").read_text())
    annotation = yaml.safe_load((OUT / "annotations.yaml").read_text())
    gate = yaml.safe_load((OUT / "gate.yaml").read_text())
    scenes = {row["scene_id"]: row for row in scene_config["scenes"]}
    inputs = [json.loads(line) for line in (capture / "input_manifest.jsonl").read_text().splitlines()]
    plan = json.loads((OUT / "REPAIR_PLAN.json").read_text())
    by_sid = {row["scene_id"]: row for row in plan}
    rows = []
    for item in inputs:
        sid = item["scene_id"]
        valid, detail = check_state(annotation["scenes"][sid], scenes[sid],
                                    labels(capture / sid / "evaluator/semantic_labels.png"),
                                    capture, item, gate)
        rows.append({**by_sid[sid], "geometry_verified": valid, "geometry": detail,
                     "rgb_path": str((capture / item["input_files"]["rgb"]).relative_to(ROOT))})
    groups = {}
    for row in rows:
        if row["geometry_verified"]:
            groups.setdefault(row["original_failed_scene"], row)
    result = {"status": "PASS" if manifest.get("status") == "COMPLETE"
              and len(inputs) == len(plan) and len(groups) == len(plan) // len(DELTAS) else "FAIL",
              "captured": len(rows), "verified_variants": sum(row["geometry_verified"] for row in rows),
              "failed_family_count": len(plan) // len(DELTAS),
              "repaired_family_count": len(groups), "selected_one_per_family": list(groups.values()),
              "all_variants": rows}
    (OUT / "REPAIR_QC.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "repaired": len(groups),
                      "needed": result["failed_family_count"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "qc"))
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else qc)()
