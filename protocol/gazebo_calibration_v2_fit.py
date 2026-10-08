#!/usr/bin/env python3
"""Calibration-v2 fit wrapper; algorithm and gates are unchanged from v1."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import gazebo_calibration_v1_fit as impl
ROOT=Path(__file__).resolve().parents[1];RESULT=ROOT/"results/spatial_vlm_refspatial_v1/gazebo_calibration_v2"

def configure():
 values={"DATA":ROOT/"datasets/Gazebo_calibration_v2","RESULT":RESULT,"PREDICTIONS":RESULT/"b0_predictions/predictions.jsonl","PREDICTION_RUN":RESULT/"b0_predictions/run.json","CONTRACT":ROOT/"protocol/GAZEBO_CALIBRATION_V2_CONTRACT_LOCK.json","MATERIALIZATION_LOCK":ROOT/"protocol/gazebo_calibration_v2_materialization_lock.json","FIT_LOCK":ROOT/"protocol/GAZEBO_CALIBRATION_V2_FIT_INPUT_LOCK.json","CALIBRATOR":RESULT/"temperature_calibrator.json","METRICS":RESULT/"GAZEBO_CALIBRATION_METRICS.json","REPORT":RESULT/"GAZEBO_CALIBRATION_RESULT.md","FINAL_LOCK":RESULT/"CALIBRATOR_THRESHOLD_LOCK.json"}
 for name,value in values.items():setattr(impl,name,value)

def lock_inputs():
 configure();impl.lock_fit_inputs();path=impl.FIT_LOCK;value=json.loads(path.read_text());value["protocol_id"]="gazebo_calibration_v2";value["source_artifact_sha256"][str(Path(__file__).resolve().relative_to(ROOT))]=impl.sha(Path(__file__).resolve());impl.dump(path,value)

def fit():
 configure();impl.fit()
 for path in (impl.CALIBRATOR,impl.METRICS):
  value=json.loads(path.read_text());value["protocol_id"]="gazebo_calibration_v2";impl.dump(path,value)
 impl.REPORT.write_text(impl.REPORT.read_text().replace("Calibration v1","Calibration v2").replace("Calibration-v1","Calibration-v2"))

def freeze():
 configure();impl.freeze();value=json.loads(impl.FINAL_LOCK.read_text());value["protocol_id"]="gazebo_calibration_v2";value["fit_wrapper_sha256"]=impl.sha(Path(__file__).resolve());impl.dump(impl.FINAL_LOCK,value)

def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=("lock-fit-inputs","fit","freeze"));a=p.parse_args();lock_inputs() if a.command=="lock-fit-inputs" else fit() if a.command=="fit" else freeze()
if __name__=="__main__":main()
