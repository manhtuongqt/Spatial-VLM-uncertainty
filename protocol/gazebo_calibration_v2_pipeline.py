#!/usr/bin/env python3
"""Lock/preflight fresh Calibration-v2 while preserving invalid v1."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import gazebo_calibration_v1_pipeline as base
import generate_gazebo_calibration_v2_contract as generator
ROOT=Path(__file__).resolve().parents[1];PROTOCOL_ID=generator.PROTOCOL_ID;RESULT=ROOT/"results/spatial_vlm_refspatial_v1/gazebo_calibration_v2"
LOCK=ROOT/"protocol/GAZEBO_CALIBRATION_V2_CONTRACT_LOCK.json";CAPTURE_LOCK=ROOT/"protocol/gazebo_calibration_v2_capture_compatibility_lock.json"

def configure():
 values={"generator":generator,"PROTOCOL_ID":PROTOCOL_ID,"SCENES":generator.SCENES_PATH,"ANNOTATIONS":generator.ANNOTATIONS_PATH,"GATE":generator.GATE_PATH,"FAMILY_MANIFEST":generator.FAMILY_MANIFEST_PATH,"SPLIT_MANIFEST":generator.SPLIT_MANIFEST_PATH,"LOCK":LOCK,"CAPTURE_LOCK":CAPTURE_LOCK,"RESULT":RESULT,"STATIC_PREFLIGHT":RESULT/"PREFLIGHT_STATIC.json","LIVE_PREFLIGHT":RESULT/"PREFLIGHT_LIVE.json"}
 for name,value in values.items():setattr(base,name,value)

def lock():
 configure();base.lock_contract();decision=ROOT/"results/spatial_vlm_refspatial_v1/gazebo_calibration_v1/CAPTURE_ATTEMPT_01_DECISION.md";amendment=ROOT/"protocol/gazebo_calibration_v2_amendment.json";extra=[Path(__file__).resolve(),decision,amendment,ROOT/"protocol/gazebo_calibration_v2_infer.py",ROOT/"protocol/materialize_gazebo_calibration_v2.py",ROOT/"protocol/gazebo_calibration_v2_fit.py"]
 value=json.loads(LOCK.read_text());value["source_artifact_sha256"].update({str(p.relative_to(ROOT)):base.sha(p) for p in extra});value["parent_invalid_attempt_decision_sha256"]=base.sha(decision);value["amendment_sha256"]=base.sha(amendment);base.dump(LOCK,value)
 cap=json.loads(CAPTURE_LOCK.read_text());cap["parent_contract_lock_sha256"]=base.sha(LOCK);cap["source_artifact_sha256"].update({str(p.relative_to(ROOT)):base.sha(p) for p in extra});base.dump(CAPTURE_LOCK,cap);print(json.dumps({"status":"LOCKED_FINAL_V2","contract_sha256":base.sha(LOCK),"capture_lock_sha256":base.sha(CAPTURE_LOCK)},indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=("validate","lock","preflight-static","preflight-live"));a=p.parse_args();configure()
 if a.command=="validate":base.validate_contract();print("PASS: calibration-v2 contract")
 elif a.command=="lock":lock()
 elif a.command=="preflight-static":base.static_preflight()
 else:base.live_preflight()
if __name__=="__main__":main()
