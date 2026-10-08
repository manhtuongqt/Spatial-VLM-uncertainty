#!/usr/bin/env python3
"""Calibration-v2 wrapper preserving the invalid v1 attempt."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import materialize_gazebo_calibration_v1 as impl

ROOT=Path(__file__).resolve().parents[1];PROTOCOL="gazebo_calibration_v2";CONFIG=ROOT/"ur3/ur3_perception/config"
SCENES=CONFIG/f"{PROTOCOL}_scenes.yaml";ANNOTATIONS=CONFIG/f"{PROTOCOL}_annotations.yaml";GATE=CONFIG/f"{PROTOCOL}_gate.yaml"
PRIMARY_LOCK=ROOT/"protocol/GAZEBO_CALIBRATION_V2_CONTRACT_LOCK.json";CAPTURE_LOCK=ROOT/"protocol/gazebo_calibration_v2_capture_compatibility_lock.json"
CAPTURE=ROOT/"results/spatial_vlm_refspatial_v1/gazebo_calibration_v2/capture_attempt_01";OUT=ROOT/"datasets/Gazebo_calibration_v2";RESULT=ROOT/"results/spatial_vlm_refspatial_v1/gazebo_calibration_v2"
LOCK=ROOT/"protocol/gazebo_calibration_v2_materialization_lock.json";QC=RESULT/"GAZEBO_CALIBRATION_V2_GEOMETRY_QC.json";DUPLICATE_QC=RESULT/"GAZEBO_CALIBRATION_V2_RGB_DUPLICATE_QC.json";CROSS_DUPLICATE_QC=RESULT/"GAZEBO_CALIBRATION_V2_CROSS_SPLIT_RGB_QC.json"

def configure():
 for name,value in {"PROTOCOL":PROTOCOL,"SCENES":SCENES,"ANNOTATIONS":ANNOTATIONS,"GATE":GATE,"PRIMARY_LOCK":PRIMARY_LOCK,"CAPTURE_LOCK":CAPTURE_LOCK,"CAPTURE":CAPTURE,"OUT":OUT,"RESULT":RESULT,"LOCK":LOCK,"QC":QC,"DUPLICATE_QC":DUPLICATE_QC,"CROSS_DUPLICATE_QC":CROSS_DUPLICATE_QC}.items():setattr(impl,name,value)
 impl.configure=configure

def lock():
 configure();impl.lock_materialization();value=json.loads(LOCK.read_text());value.update(v2_wrapper_source_sha256=impl.base.sha(Path(__file__).resolve()),parent_invalid_attempt_decision_sha256=impl.base.sha(ROOT/"results/spatial_vlm_refspatial_v1/gazebo_calibration_v1/CAPTURE_ATTEMPT_01_DECISION.md"),locked_at_utc_v2=datetime.now(timezone.utc).isoformat());impl.base.dump(LOCK,value);print(json.dumps({"status":"LOCKED_FINAL_V2","sha256":impl.base.sha(LOCK)},indent=2))

def materialize(output):
 configure();impl.materialize(output);path=output/"manifest.json";value=json.loads(path.read_text());value["dataset"]="Gazebo_calibration_v2";value["protocol_id"]=PROTOCOL;impl.base.dump(path,value)

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="command",required=True);s.add_parser("lock-materialization");m=s.add_parser("materialize");m.add_argument("--output",type=Path,default=OUT);a=p.parse_args();lock() if a.command=="lock-materialization" else materialize(a.output)
if __name__=="__main__":main()
