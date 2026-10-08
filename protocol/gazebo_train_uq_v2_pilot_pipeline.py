#!/usr/bin/env python3
"""Validate and lock the geometry-only Gazebo Train-UQ v2 pilot before capture."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]; PROTOCOL="gazebo_train_uq_v2_pilot"; CONFIG=ROOT/"ur3/ur3_perception/config"
SCENES=CONFIG/"gazebo_train_uq_v2_pilot_scenes.yaml"; ANN=CONFIG/"gazebo_train_uq_v2_pilot_annotations.yaml"; GATE=CONFIG/"gazebo_train_uq_v2_pilot_gate.yaml"
LOCK=ROOT/"protocol/gazebo_train_uq_v2_pilot_contract_lock.json"; CAPLOCK=ROOT/"protocol/gazebo_train_uq_v2_pilot_capture_compatibility_lock.json"; RESULT=ROOT/"results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot"
V1_QC=ROOT/"results/spatial_vlm_refspatial_v1/gazebo_train_uq_v1/GAZEBO_TRAIN_UQ_V1_GEOMETRY_QC.json"; DEV=ROOT/"datasets/Gazebo_dev_answerability_v2/evaluator_ground_truth.jsonl"
LEGACY=("RoboRefer/API/api.py","ur3/ur3_perception/scripts/roborefer_pilot_capture.py","ur3/ur3_perception/scripts/roborefer_pilot_runner.py","ur3/ur3_perception/scripts/roborefer_pilot_validate.py","ur3/ur3_perception/scripts/roborefer_pilot_evaluator.py","ur3/ur3_perception/scripts/roborefer_grounder.py","ur3/ur3_perception/scripts/spatial_point_utils.py","ur3/ur3_perception/scripts/move_camera_to_view.py","ur3/ur3_perception/launch/roborefer_pilot_capture.launch.py","ur3/ur3_perception/config/roborefer_pilot_v0_scenes.yaml","ur3/ur3_perception/config/roborefer_pilot_v0_gate.yaml","ur3/ur3_perception/config/roborefer_pilot_v0_annotations.yaml","ur3/ur3_perception/schemas/roborefer_pilot_prediction.schema.json","ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro","ur3/ur3_moveit_control/launch/ur3_susgrip_sim.launch.py","ur3/ur_simulation_gz/worlds/ur3_pick_place.sdf","ur3/ur_simulation_gz/launch/ur_sim_control.launch.py")

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p:Path,x:dict)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def load(p:Path)->dict:return yaml.safe_load(p.read_text())

def validate()->tuple[dict,dict,dict]:
    from generate_gazebo_train_uq_v2_pilot_contract import build,text
    generated=build()
    for p,v in zip((SCENES,ANN,GATE),generated):
        if not p.is_file() or p.read_text()!=text(v):raise ValueError(f"non-deterministic pilot input: {p}")
    scenes,ann,gate=map(load,(SCENES,ANN,GATE)); rows=scenes["scenes"]; oracle=ann["scenes"]
    if any(x.get("protocol_id")!=PROTOCOL for x in (scenes,ann,gate)) or len(rows)!=32 or len(oracle)!=32:raise ValueError("bad pilot protocol/count")
    families=[x["scene_family_id"] for x in rows]
    if len(set(families))!=32 or set(x["scene_id"] for x in rows)!=set(oracle):raise ValueError("family/annotation alignment failed")
    if {x["state"] for x in oracle.values()}!={"FOUND","AMBIGUOUS","ABSENT","INSUFFICIENT_EVIDENCE"}:raise ValueError("missing state")
    if any(sum(x["state"]==state for x in oracle.values())!=8 for state in ("FOUND","AMBIGUOUS","ABSENT","INSUFFICIENT_EVIDENCE")):raise ValueError("state quota")
    if any(sum(x["relation_variant"]==rel for x in oracle.values())!=8 for rel in ("leftmost","rightmost","second_from_left","second_from_right")):raise ValueError("relation quota")
    prior={json.loads(line)["family_id"] for line in DEV.read_text().splitlines() if line}
    if set(families)&prior:raise ValueError("Dev-v2 family leakage")
    if json.loads(V1_QC.read_text()).get("status")!="BLOCKED":raise ValueError("v1 failed-audit evidence missing")
    if gate["policies"]!={"no_sam2":True,"no_training":True,"no_b2":True,"no_test_access":True,"no_dev_v2_fit_or_selection":True}:raise ValueError("pilot safety policy changed")
    return scenes,ann,gate

def lock()->None:
    if LOCK.exists() or CAPLOCK.exists():raise FileExistsError("refusing to overwrite pilot locks")
    validate(); own=(Path("protocol/generate_gazebo_train_uq_v2_pilot_contract.py"),Path("protocol/gazebo_train_uq_v2_pilot_pipeline.py"),Path("ur3/ur3_perception/config/gazebo_train_uq_v2_pilot_scenes.yaml"),Path("ur3/ur3_perception/config/gazebo_train_uq_v2_pilot_annotations.yaml"),Path("ur3/ur3_perception/config/gazebo_train_uq_v2_pilot_gate.yaml"),Path("results/spatial_vlm_refspatial_v1/gazebo_train_uq_v1/GAZEBO_TRAIN_UQ_V1_GEOMETRY_QC.json"))
    sources={str(x):sha(ROOT/x) for x in own}; payload={"schema_version":1,"protocol_id":PROTOCOL,"status":"LOCKED_BEFORE_PILOT_CAPTURE","locked_at_utc":datetime.now(timezone.utc).isoformat(),"parent_families":32,"state_relation_cell_quota":2,"purpose":"geometry validation only; no materialization, training, or model inference","source_artifact_sha256":sources,"v1_failed_qc_sha256":sha(V1_QC),"frozen_dev_v2_sha256":sha(DEV),"policies":{"no_sam2":True,"no_training":True,"no_b2":True,"no_test_access":True,"no_dev_v2_fit_or_selection":True}}
    dump(LOCK,payload)
    compat={"schema_version":1,"protocol_id":PROTOCOL,"status":"LOCKED_BEFORE_CAPTURE_AND_INFERENCE","locked_at_utc":datetime.now(timezone.utc).isoformat(),"workspace_root":str(ROOT),"parent_contract_lock":str(LOCK.relative_to(ROOT)),"parent_contract_lock_sha256":sha(LOCK),"purpose":"legacy ROS-capture execution adapter; geometry pilot only","source_artifact_sha256":{x:sha(ROOT/x) for x in sorted((*LEGACY,"ur3/ur3_perception/config/gazebo_train_uq_v2_pilot_scenes.yaml","ur3/ur3_perception/config/gazebo_train_uq_v2_pilot_annotations.yaml","ur3/ur3_perception/config/gazebo_train_uq_v2_pilot_gate.yaml"))},"model_inventory_sha256":"NO_MODEL_INFERENCE_GEOMETRY_PILOT","policies_inherited":payload["policies"]}
    dump(CAPLOCK,compat); print(json.dumps({"status":"LOCKED","contract_sha256":sha(LOCK),"capture_lock_sha256":sha(CAPLOCK)},indent=2))

def verify()->dict:
    if not LOCK.is_file() or not CAPLOCK.is_file():raise FileNotFoundError("pilot locks missing")
    lock=json.loads(LOCK.read_text()); validate()
    for rel,want in lock["source_artifact_sha256"].items():
        if sha(ROOT/rel)!=want:raise ValueError(f"source changed after lock: {rel}")
    cap=json.loads(CAPLOCK.read_text())
    if cap.get("parent_contract_lock_sha256")!=sha(LOCK):raise ValueError("capture lock parent mismatch")
    for rel,want in cap["source_artifact_sha256"].items():
        if sha(ROOT/rel)!=want:raise ValueError(f"capture source changed: {rel}")
    return lock

def preflight()->None:
    verify(); out=subprocess.run(["ros2","topic","list","-t"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout
    req={"/wrist_camera/color/image_raw":"sensor_msgs/msg/Image","/wrist_camera/depth/image_raw":"sensor_msgs/msg/Image","/wrist_camera/evaluation_labels/labels_map":"sensor_msgs/msg/Image","/wrist_camera/color/camera_info":"sensor_msgs/msg/CameraInfo","/tf":"tf2_msgs/msg/TFMessage"}; checks={x:f"{x} [{t}]" in out for x,t in req.items()}; free=shutil.disk_usage(ROOT).free/2**30; status="PASS" if all(checks.values()) and free>2 else "BLOCKED"; dump(RESULT/"PREFLIGHT_LIVE.json",{"protocol_id":PROTOCOL,"status":status,"contract_lock_sha256":sha(LOCK),"checks":checks,"free_gib":free,"capture_authorized":status=="PASS"}); print(json.dumps({"status":status,"checks":checks,"free_gib":free},indent=2));
    if status!="PASS":raise SystemExit(2)

def main()->None:
    p=argparse.ArgumentParser();s=p.add_subparsers(dest="cmd",required=True);s.add_parser("validate");s.add_parser("lock");s.add_parser("preflight-live");a=p.parse_args()
    if a.cmd=="validate":validate();print("PASS: v2 geometry pilot contract")
    elif a.cmd=="lock":lock()
    else:preflight()
if __name__=="__main__":main()
