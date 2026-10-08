#!/usr/bin/env python3
"""Lock and preflight the immutable 320-family full-v2-r2 capture."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import generate_gazebo_train_uq_v2_full_r2_contract as generator
import gazebo_train_uq_v2_full_pipeline as base

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = generator.PROTOCOL_ID
LOCK = ROOT / "protocol/gazebo_train_uq_v2_full_r2_contract_lock.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_train_uq_v2_full_r2_capture_compatibility_lock.json"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full_r2"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, value): Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def configure():
    values = {"generator": generator, "PROTOCOL_ID": PROTOCOL_ID,
              "SCENES": generator.SCENES_PATH, "ANNOTATIONS": generator.ANNOTATIONS_PATH,
              "GATE": generator.GATE_PATH, "FAMILY_MANIFEST": generator.FAMILY_MANIFEST_PATH,
              "SPLIT_MANIFEST": generator.SPLIT_MANIFEST_PATH, "LOCK": LOCK,
              "CAPTURE_LOCK": CAPTURE_LOCK, "RESULT": RESULT,
              "STATIC_PREFLIGHT": RESULT/"PREFLIGHT_STATIC.json",
              "LIVE_PREFLIGHT": RESULT/"PREFLIGHT_LIVE.json"}
    for name, value in values.items(): setattr(base, name, value)


def lock():
    configure(); scenes, _, gate = base.validate_contract()
    if LOCK.exists() or CAPTURE_LOCK.exists(): raise FileExistsError("refusing to overwrite full-r2 locks")
    fixed = [base.GROUNDING_LOCK, base.HYPOTHESIS_LOCK, base.R6_LOCK, base.R6_CAPTURE_LOCK,
             base.R6_QC, base.R6_DECISION]
    sources = [Path(__file__).resolve(), Path(generator.__file__).resolve(),
               generator.SCENES_PATH, generator.ANNOTATIONS_PATH, generator.GATE_PATH,
               generator.FAMILY_MANIFEST_PATH, generator.SPLIT_MANIFEST_PATH,
               ROOT/"protocol/gazebo_train_uq_v2_full_attempt_01_decision.md",
               ROOT/"protocol/gazebo_train_uq_v2_full_contract_lock.json",
               ROOT/"results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full/GAZEBO_TRAIN_UQ_V2_FULL_GEOMETRY_QC.json",
               ROOT/"ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
               ROOT/"ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf", *fixed]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID,
               "status": "LOCKED_BEFORE_CAPTURE_INFERENCE_AND_TRAINING",
               "locked_at_utc": datetime.now(timezone.utc).isoformat(),
               "parent_family_count": 320, "split_parent_family_count": {"train_uq":256,"val_uq":64},
               "state_relation_cell_quota": {"train_uq":16,"val_uq":4},
               "family_manifest_sha256": sha(generator.FAMILY_MANIFEST_PATH),
               "split_manifest_sha256": sha(generator.SPLIT_MANIFEST_PATH),
               "scene_layout_signature_set_sha256": hashlib.sha256(json.dumps(sorted(x["layout_signature_sha256"] for x in scenes["scenes"])).encode()).hexdigest(),
               "source_artifact_sha256": hashes,
               "b0_grounding_lock_sha256": sha(base.GROUNDING_LOCK),
               "hypothesis_lock_sha256": sha(base.HYPOTHESIS_LOCK),
               "pilot_r6_geometry_qc_sha256": sha(base.R6_QC),
               "world_file": "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
               "camera": gate["camera"], "inference_lock": gate["inference_lock"],
               "risk_feature_schema": gate["risk_feature_schema"], "proposed_method": gate["proposed_method"],
               "selection_rule": gate["selection_rule"], "statistical_protocol": gate["statistical_protocol"],
               "sealed": gate["sealed"]}
    dump(LOCK, payload)
    capture_sources = set(base.capture_common.LEGACY_CAPTURE_SOURCES) | set(hashes)
    cap = {"schema_version":1,"protocol_id":PROTOCOL_ID,"status":"LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
           "locked_at_utc":datetime.now(timezone.utc).isoformat(),"parent_contract_lock":str(LOCK.relative_to(ROOT)),
           "parent_contract_lock_sha256":sha(LOCK),"purpose":"fresh 320-family no-manipulation capture adapter",
           "source_artifact_sha256":{name:sha(ROOT/name) for name in sorted(capture_sources)},
           "model_inventory_sha256":"NO_MODEL_INFERENCE_FULL_UQ_V2_R2_CAPTURE",
           "policies":{"no_model_inference_during_capture":True,"no_robot_manipulation":True,
                       "semantic_labels_evaluator_only":True,"calibration_and_test_sealed":True}}
    dump(CAPTURE_LOCK, cap)
    print(json.dumps({"status":"LOCKED","contract_sha256":sha(LOCK),"capture_lock_sha256":sha(CAPTURE_LOCK)},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("validate","lock","preflight-static","preflight-live"));a=p.parse_args();configure()
    if a.command=="validate": base.validate_contract(); print("PASS: full-v2-r2 contract")
    elif a.command=="lock": lock()
    elif a.command=="preflight-static": base.static_preflight()
    else: base.live_preflight()


if __name__=="__main__": main()
