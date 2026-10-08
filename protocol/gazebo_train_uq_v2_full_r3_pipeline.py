#!/usr/bin/env python3
"""Lock and preflight immutable full-v2-r3 without changing older attempts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_gazebo_train_uq_v2_full_r3_contract as generator
import gazebo_train_uq_v2_full_r2_pipeline as impl

ROOT=Path(__file__).resolve().parents[1];PROTOCOL_ID=generator.PROTOCOL_ID
LOCK=ROOT/"protocol/gazebo_train_uq_v2_full_r3_contract_lock.json";CAPTURE_LOCK=ROOT/"protocol/gazebo_train_uq_v2_full_r3_capture_compatibility_lock.json"
RESULT=ROOT/"results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full_r3"


def configure():
    impl.generator=generator;impl.PROTOCOL_ID=PROTOCOL_ID;impl.LOCK=LOCK;impl.CAPTURE_LOCK=CAPTURE_LOCK;impl.RESULT=RESULT;impl.configure()


def lock():
    configure();impl.lock()
    additions=[ROOT/"protocol/gazebo_train_uq_v2_full_r2_attempt_01_decision.md",
      ROOT/"protocol/gazebo_train_uq_v2_full_r2_contract_lock.json",
      ROOT/"protocol/gazebo_train_uq_v2_full_r2_materialization_lock.json",
      ROOT/"results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full_r2/GAZEBO_TRAIN_UQ_V2_FULL_R2_GEOMETRY_QC.json",
      Path(__file__).resolve()]
    value=json.loads(LOCK.read_text())
    for path in additions:value["source_artifact_sha256"][str(path.relative_to(ROOT))]=impl.sha(path)
    value["predecessor_full_r2_geometry_qc_sha256"]=impl.sha(additions[-2]);impl.dump(LOCK,value)
    cap=json.loads(CAPTURE_LOCK.read_text());cap["parent_contract_lock_sha256"]=impl.sha(LOCK)
    for path in additions:cap["source_artifact_sha256"][str(path.relative_to(ROOT))]=impl.sha(path)
    impl.dump(CAPTURE_LOCK,cap)
    print(json.dumps({"status":"LOCKED_FINAL","contract_sha256":impl.sha(LOCK),"capture_lock_sha256":impl.sha(CAPTURE_LOCK)},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("validate","lock","preflight-static","preflight-live"));a=p.parse_args();configure()
    if a.command=="validate":impl.base.validate_contract();print("PASS: full-v2-r3 contract")
    elif a.command=="lock":lock()
    elif a.command=="preflight-static":impl.base.static_preflight()
    else:impl.base.live_preflight()


if __name__=="__main__":main()
