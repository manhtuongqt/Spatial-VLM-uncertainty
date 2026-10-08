#!/usr/bin/env python3
"""All-or-nothing geometry/duplicate QC and materialization for full-v2-r2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path

import materialize_gazebo_train_uq_v1 as base
import materialize_gazebo_train_uq_v2_full as full

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "gazebo_train_uq_v2_full_r2"
CONFIG = ROOT / "ur3/ur3_perception/config"
SCENES = CONFIG/f"{PROTOCOL}_scenes.yaml"; ANNOTATIONS = CONFIG/f"{PROTOCOL}_annotations.yaml"; GATE = CONFIG/f"{PROTOCOL}_gate.yaml"
PRIMARY_LOCK = ROOT/"protocol/gazebo_train_uq_v2_full_r2_contract_lock.json"
CAPTURE_LOCK = ROOT/"protocol/gazebo_train_uq_v2_full_r2_capture_compatibility_lock.json"
CAPTURE = ROOT/"results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full_r2/capture_attempt_01"
OUT = ROOT/"datasets/Gazebo_train_uq_v2_full_r2"
RESULT = ROOT/"results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full_r2"
LOCK = ROOT/"protocol/gazebo_train_uq_v2_full_r2_materialization_lock.json"
QC = RESULT/"GAZEBO_TRAIN_UQ_V2_FULL_R2_GEOMETRY_QC.json"
DUPLICATE_QC = RESULT/"GAZEBO_TRAIN_UQ_V2_FULL_R2_RGB_DUPLICATE_QC.json"


def configure():
    values={"PROTOCOL":PROTOCOL,"SCENES":SCENES,"ANNOTATIONS":ANNOTATIONS,"GATE":GATE,
            "PRIMARY_LOCK":PRIMARY_LOCK,"CAPTURE_LOCK":CAPTURE_LOCK,"CAPTURE":CAPTURE,
            "OUT":OUT,"RESULT":RESULT,"LOCK":LOCK,"QC":QC}
    for name,value in values.items(): setattr(base,name,value)
    values.update({"DUPLICATE_QC":DUPLICATE_QC})
    for name,value in values.items(): setattr(full,name,value)


def lock_materialization():
    configure(); base.lock_materialization()
    value=json.loads(LOCK.read_text()); value.update({
        "wrapper_source_sha256":base.sha(Path(__file__).resolve()),
        "contract_generator_sha256":base.sha(ROOT/"protocol/generate_gazebo_train_uq_v2_full_r2_contract.py"),
        "pipeline_source_sha256":base.sha(ROOT/"protocol/gazebo_train_uq_v2_full_r2_pipeline.py"),
        "predecessor_rejected_geometry_qc_sha256":base.sha(ROOT/"results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full/GAZEBO_TRAIN_UQ_V2_FULL_GEOMETRY_QC.json"),
        "locked_at_utc_final":datetime.now(timezone.utc).isoformat()})
    base.dump(LOCK,value); print(json.dumps({"status":"LOCKED_FINAL","sha256":base.sha(LOCK)},indent=2))


def materialize(output):
    configure(); lock=json.loads(LOCK.read_text())
    if lock.get("wrapper_source_sha256")!=base.sha(Path(__file__).resolve()): raise ValueError("wrapper changed after lock")
    duplicate=full.duplicate_audit()
    source=inspect.getsource(base.materialize).replace('"Gazebo_train_uq_v1"','"Gazebo_train_uq_v2_full_r2"')
    scope=dict(base.__dict__); scope.update(globals()); exec(source,scope); scope["materialize"](output.resolve())
    qc=json.loads(QC.read_text()); qc.update({"rgb_duplicate_status":duplicate["status"],
        "rgb_duplicate_qc_sha256":base.sha(DUPLICATE_QC),
        "full_geometry_and_duplicate_gate_passed":qc.get("status")=="PASS" and duplicate["status"]=="PASS"})
    base.dump(QC,qc)
    manifest=output/"manifest.json"; value=json.loads(manifest.read_text()); value.update({
        "qc_report_sha256":base.sha(QC),"rgb_duplicate_qc_sha256":base.sha(DUPLICATE_QC)})
    base.dump(manifest,value)
    print(json.dumps({"status":"PASS","records":qc["records"],"failed":qc["failed_scene_count"],"duplicate_status":duplicate["status"]},indent=2))


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest="command",required=True);s.add_parser("lock-materialization");m=s.add_parser("materialize");m.add_argument("--output",type=Path,default=OUT);a=p.parse_args()
    if a.command=="lock-materialization":lock_materialization()
    else:materialize(a.output)


if __name__=="__main__":main()
