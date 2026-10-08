"""Generate/QC an append-only 256-scene revised Gazebo development pilot.

The same-family original view and translated view are co-located: 512 images,
256 independent parent families. This is not a new Test/Calibration split.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import sys

import numpy as np
import yaml

from .audit_development import GAZEBO, ROOT, read_jsonl, sha256
from .canary_scene_revision import BASE_POSE, CONFIG, frozen_required_sources
from .qc_gazebo_pilot_512 import check_state


OUT = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03/gazebo_pilot_512_revision_v2"
PROTOCOL = "mh_pcrau_v3_g1_pilot_512_revision_v2"


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    sys.path.insert(0, str(ROOT / "protocol"))
    import generate_gazebo_train_uq_v2_pilot_r5_contract as geometry

    original = yaml.safe_load((CONFIG / "gazebo_train_uq_v2_full_r3_scenes.yaml").read_text())
    oracle = yaml.safe_load((CONFIG / "gazebo_train_uq_v2_full_r3_annotations.yaml").read_text())
    gate = yaml.safe_load((CONFIG / "gazebo_train_uq_v2_full_r3_gate.yaml").read_text())
    old_rows = [row for row in read_jsonl(GAZEBO / "inference_manifest.jsonl")
                if row["split"] == "train_uq"]
    if len(old_rows) != 256 or len(original["scenes"]) < 256:
        raise ValueError("Historical train_uq inventory drift")
    OUT.mkdir(parents=True)
    index = []
    for batch in range(4):
        folder = OUT / f"batch_{batch}"
        folder.mkdir()
        rows, annotations = [], {}
        for local in range(64):
            parent_index = batch * 64 + local
            source = original["scenes"][parent_index]
            old = old_rows[parent_index]
            sid = f"g1_pilot_v2_b{batch}_{local:03d}"
            # Tested on eight scenes covering all states in canary v2.
            poses = {name: [pose[0] + 0.006, pose[1] + 0.006, pose[2]]
                     for name, pose in source["poses"].items()}
            signature = geometry.layout_signature(poses)
            scene = dict(source)
            scene.update(scene_id=sid, scene_family_id=old["family_id"], layout_id=sid,
                         layout_signature_sha256=signature, poses=poses)
            ann = dict(oracle["scenes"][source["scene_id"]])
            ann.update(family_id=old["family_id"], layout_id=sid,
                       layout_signature_sha256=signature, split="development_pilot_only",
                       view_id="translated_view_v2", template_provenance=source["scene_id"])
            rows.append(scene)
            annotations[sid] = ann
            index.append({"scene_id": sid, "parent_family_id": old["family_id"],
                          "old_sample_id": old["sample_id"], "state": ann["state"],
                          "relation": ann["relation_variant"], "batch": batch})
        pid = f"{PROTOCOL}_b{batch}"
        scene_config = dict(original)
        scene_config.update(protocol_id=pid, expected_scene_count=64,
                            random_seed=24092030 + batch, view_joint_pose=BASE_POSE, scenes=rows)
        annotation = dict(oracle)
        annotation.update(protocol_id=pid, oracle_usage="development_geometry_qc_only", scenes=annotations)
        g = dict(gate)
        g.update(protocol_id=pid, parent_family_count=64,
                 split_parent_family_count={"development_pilot_only": 64},
                 state_quota={state: 16 for state in geometry.STATES},
                 relation_variant_quota={relation: 16 for relation in geometry.RELATIONS},
                 state_relation_cell_quota=4,
                 camera={**gate["camera"], "view_joint_pose": BASE_POSE},
                 policies={**gate["policies"], "no_training": True,
                           "no_model_inference": True, "no_materialization": True})
        for name, payload in (("scenes.yaml", scene_config), ("annotations.yaml", annotation),
                              ("gate.yaml", g)):
            (folder / name).write_text(yaml.safe_dump(payload, sort_keys=False,
                                                      allow_unicode=True, width=140))
        common = frozen_required_sources() | {
            "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
            "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
            "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py",
            "workspace/mh_pcrau_v3/g1/capture_revision_v2.py",
        }
        common |= {str((folder / name).relative_to(ROOT)) for name in
                   ("scenes.yaml", "annotations.yaml", "gate.yaml")}
        lock = {
            "schema_version": 1, "protocol_id": pid,
            "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "workspace_root": str(ROOT),
            "source_artifact_sha256": {name: sha256(ROOT / name) for name in sorted(common)},
            "model_inventory_sha256": "NO_MODEL_INFERENCE_V3_PILOT_CAPTURE",
            "capture_role": "DEVELOPMENT_PILOT_ONLY",
            "authorization": "User requested G1 canary then supplement Gazebo pilot on 2026-09-24",
            "planned_scene_count": 64, "view_joint_pose": BASE_POSE,
            "seals": {"calibration": True, "test_iid": True, "test_ood": True},
        }
        (folder / "CAPTURE_SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n")
    (OUT / "PARENT_FAMILY_INDEX.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"status": "LOCKED", "batches": 4, "scenes": 256,
                      "independent_parent_families": 256}))


def qc() -> None:
    destination = OUT / "REVISION_V2_QC.json"
    if destination.exists():
        raise FileExistsError(destination)
    reports = []
    all_rows = []
    for batch in range(4):
        folder = OUT / f"batch_{batch}"
        capture = folder / "capture_attempt_01"
        manifest = json.loads((capture / "capture_manifest.json").read_text())
        inputs = [json.loads(line) for line in (capture / "input_manifest.jsonl").read_text().splitlines()]
        scene_config = yaml.safe_load((folder / "scenes.yaml").read_text())
        annotation = yaml.safe_load((folder / "annotations.yaml").read_text())
        gate = yaml.safe_load((folder / "gate.yaml").read_text())
        scenes = {row["scene_id"]: row for row in scene_config["scenes"]}
        integrity = (manifest.get("status") == "COMPLETE" and manifest.get("scene_count") == 64
                     and len(inputs) == 64
                     and manifest.get("input_only_sensor_qc", {}).get("passed") is True
                     and manifest.get("pretrial_source_lock_sha256") == sha256(folder / "CAPTURE_SOURCE_LOCK.json")
                     and manifest.get("input_manifest_sha256") == sha256(capture / "input_manifest.jsonl"))
        rows = []
        for item in inputs:
            sid = item["scene_id"]
            from protocol.gazebo_train_uq_v2_pilot_r4_pipeline import labels
            label_map = labels(capture / sid / "evaluator/semantic_labels.png")
            valid, detail = check_state(annotation["scenes"][sid], scenes[sid], label_map,
                                        capture, item, gate)
            rgb = capture / item["input_files"]["rgb"]
            hashes_match = all(sha256(capture / path) == item["input_sha256"][kind]
                               for kind, path in item["input_files"].items())
            row = {"scene_id": sid, "family_id": annotation["scenes"][sid]["family_id"],
                   "state": annotation["scenes"][sid]["state"],
                   "relation": annotation["scenes"][sid]["relation_variant"],
                   "geometry_verified": valid, "input_hashes_match": hashes_match,
                   "geometry": detail, "rgb_path": str(rgb.relative_to(ROOT)),
                   "rgb_sha256": sha256(rgb)}
            rows.append(row)
            all_rows.append(row)
        reports.append({"batch": batch, "integrity": integrity,
                        "verified": sum(row["geometry_verified"] for row in rows),
                        "rows": rows})
    counts = Counter(row["state"] for row in all_rows if row["geometry_verified"] and row["input_hashes_match"])
    status = "PASS" if (all(item["integrity"] for item in reports)
                        and all(row["geometry_verified"] and row["input_hashes_match"] for row in all_rows)) else "FAIL"
    result = {"status": status, "raw_images": len(all_rows),
              "valid_labels": sum(counts.values()), "valid_by_state": dict(sorted(counts.items())),
              "independent_parent_families": len({row["family_id"] for row in all_rows}),
              "same_family_as_old_view": True,
              "note": "QC PASS is necessary but not sufficient for G1_PASS; full historical leakage and power locks remain separate.",
              "batches": reports}
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": status, "raw": len(all_rows), "valid": sum(counts.values()),
                      "valid_by_state": dict(counts)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "qc"))
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else qc)()
