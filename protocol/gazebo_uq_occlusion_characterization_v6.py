#!/usr/bin/env python3
"""Collision-free visual-only neutral-panel characterization for WP5."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json,math
import cv2
import numpy as np
import yaml
import gazebo_uq_occlusion_characterization_v1 as common
from generate_gazebo_train_uq_v1_contract import CAMERA,LABELS,OBJECTS
ROOT=common.ROOT;CONFIG=common.CONFIG;PROTOCOL_ID="gazebo_uq_occlusion_characterization_v6"
SCENES=CONFIG/f"{PROTOCOL_ID}_scenes.yaml";ANNOTATIONS=CONFIG/f"{PROTOCOL_ID}_annotations.yaml";GATE=CONFIG/f"{PROTOCOL_ID}_gate.yaml";LOCK=ROOT/"protocol/gazebo_uq_occlusion_characterization_v6_capture_lock.json"
RESULT=ROOT/"results/spatial_vlm_refspatial_v1/gazebo_uq_occlusion_characterization_v6";CAPTURE=RESULT/"capture_attempt_01";REPORT=RESULT/"OCCLUSION_CHARACTERIZATION_REPORT.json";REPORT_MD=RESULT/"OCCLUSION_CHARACTERIZATION_REPORT.md"
TARGET_XY=(-.220,.300);PANEL_ALIGNED_XY=(-.2168,.2947);OFFSETS=tuple(round(.004+.001*i,3) for i in range(22));PANEL="uq_neutral_occluder";PANEL_LABEL=39
OBJECTS_V6={**OBJECTS,PANEL:{"z":.060,"storage_pose":[-.60,1.20,0.]}}
def build():
 rows=[];oracle={};i=0
 for target in common.TARGETS:
  for yaw in common.TARGET_YAWS:
   for offset in OFFSETS:
    sid=f"uq_occ_char_v6_{i:03d}";poses={target:[TARGET_XY[0],TARGET_XY[1],yaw],PANEL:[PANEL_ALIGNED_XY[0],round(PANEL_ALIGNED_XY[1]-offset,6),math.pi/2]};sig=common.layout_signature(poses)
    rows.append({"scene_id":sid,"scene_family_id":f"engineering_occlusion_characterization_v6/family_{i:03d}","layout_id":f"neutral_panel_v6_{i:03d}","layout_signature_sha256":sig,"task_type":"engineering_geometry_characterization_no_inference","instruction":"ENGINEERING ONLY: do not run model inference on this capture.","poses":poses})
    oracle[sid]={"usage":"evaluator_only_geometry_characterization","target_id":target,"target_label":LABELS[target],"target_yaw_rad":yaw,"occluder_id":PANEL,"occluder_label":PANEL_LABEL,"lateral_offset_m":offset,"layout_signature_sha256":sig};i+=1
 scenes={"schema_version":1,"protocol_id":PROTOCOL_ID,"expected_scene_count":len(rows),"capture_mode":"engineering_geometry_only_no_manipulation","random_seed":14092032,"view_joint_pose":CAMERA["view_joint_pose"],"camera_frame":CAMERA["frame"],"base_frame":CAMERA["base_frame"],"coordinate_suffix":"ENGINEERING ONLY; MODEL INFERENCE FORBIDDEN.","objects":OBJECTS_V6,"scenes":rows}
 annotations={"schema_version":1,"protocol_id":PROTOCOL_ID,"oracle_usage":"geometry_characterization_only_never_train_or_val","reason":"r4 exposed collision/order sensitivity in cracker-box mechanism","occluder":{"id":PANEL,"label":PANEL_LABEL,"static":True,"visual_only_no_collision":True,"candidate":False},"sweep":{"offsets_m":OFFSETS,"panel_yaw_rad":math.pi/2},"scenes":oracle}
 gate={"schema_version":1,"protocol_id":PROTOCOL_ID,"official_pilot_gate_unchanged":{"min_inclusive":1,"max_exclusive":120},"internal_design_margin":{"min_inclusive":20,"max_inclusive":100},"selection_rule":{"current_and_adjacent_offsets_inside_internal_margin":True,"prefer_maximum_minimum_distance_to_20_or_100":True,"per_target_and_yaw":True},"policies":{"engineering_only":True,"no_model_inference":True,"no_training":True,"never_materialize_into_train_or_val":True,"semantic_labels_evaluator_only":True,"no_dev_calibration_or_test_access":True,"no_robot_manipulation":True,"collision_free_visual_only_occluder":True}}
 return scenes,annotations,gate
def generate():
 if any(p.exists() for p in (SCENES,ANNOTATIONS,GATE,LOCK)):raise FileExistsError("refusing overwrite")
 for p,x in zip((SCENES,ANNOTATIONS,GATE),build()):p.write_text(common.serialized(x))
 print(json.dumps({"status":"GENERATED","scenes":len(build()[0]["scenes"])},indent=2))
def lock():
 if LOCK.exists():raise FileExistsError("refusing overwrite lock")
 for p,x in zip((SCENES,ANNOTATIONS,GATE),build()):
  if p.read_text()!=common.serialized(x):raise ValueError(f"edited {p}")
 own={"protocol/gazebo_uq_occlusion_characterization_v1.py","protocol/gazebo_uq_occlusion_characterization_v6.py",str(SCENES.relative_to(ROOT)),str(ANNOTATIONS.relative_to(ROOT)),str(GATE.relative_to(ROOT)),"ur3/ur3_perception/launch/roborefer_uq_capture.launch.py","ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion.sdf","protocol/gazebo_train_uq_v2_pilot_r4_contract_lock.json","results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r4/GAZEBO_TRAIN_UQ_V2_PILOT_R4_GEOMETRY_QC.json"};sources=sorted(set(common.REQUIRED_CAPTURE_SOURCES)|own)
 payload={"schema_version":1,"protocol_id":PROTOCOL_ID,"status":"LOCKED_BEFORE_CAPTURE_AND_INFERENCE","locked_at_utc":datetime.now(timezone.utc).isoformat(),"workspace_root":str(ROOT),"purpose":"collision-free visual-only neutral occlusion characterization","scene_count":len(build()[0]["scenes"]),"world_file":"ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion.sdf","model_inventory_sha256":"NO_MODEL_INFERENCE_ENGINEERING_CHARACTERIZATION_V6","source_artifact_sha256":{n:common.sha256(ROOT/n) for n in sources},"prohibitions":build()[2]["policies"]};common.write_json(LOCK,payload);print(json.dumps({"status":"LOCKED","sha256":common.sha256(LOCK)},indent=2))
def qc():
 lk=json.loads(LOCK.read_text())
 for n,d in lk["source_artifact_sha256"].items():
  if common.sha256(ROOT/n)!=d:raise ValueError(f"locked source changed {n}")
 m=json.loads((CAPTURE/"capture_manifest.json").read_text());inputs=[json.loads(x) for x in (CAPTURE/"input_manifest.jsonl").read_text().splitlines() if x];oracle=yaml.safe_load(ANNOTATIONS.read_text())["scenes"]
 if m.get("status")!="COMPLETE" or len(inputs)!=len(oracle):raise ValueError("capture incomplete")
 rows=[];groups={}
 for rec in inputs:
  sid=rec["scene_id"];it=oracle[sid];lab=cv2.imread(str(CAPTURE/sid/"evaluator/semantic_labels.png"),cv2.IMREAD_UNCHANGED);lab=lab[...,0] if lab.ndim==3 else lab;px=int(np.count_nonzero(lab==int(it["target_label"])));opx=int(np.count_nonzero(lab==PANEL_LABEL));row={"scene_id":sid,**it,"target_visible_pixels":px,"occluder_visible_pixels":opx,"inside_official_gate":1<=px<120,"inside_internal_margin":20<=px<=100};rows.append(row);groups.setdefault((it["target_id"],it["target_yaw_rad"]),[]).append(row)
 selected=[]
 for key,g in sorted(groups.items()):
  g.sort(key=lambda x:x["lateral_offset_m"]);cand=[]
  for i in range(1,len(g)-1):
   hood=g[i-1:i+2]
   if all(x["inside_internal_margin"] for x in hood):
    margin=min(min(x["target_visible_pixels"]-20,100-x["target_visible_pixels"]) for x in hood);cand.append((margin,g[i]))
  if cand:
   margin,row=max(cand,key=lambda x:x[0]);selected.append({**row,"three_point_safety_margin_pixels":margin})
 required={(t,y) for t in common.TARGETS for y in common.TARGET_YAWS};qualified={(x["target_id"],x["target_yaw_rad"]) for x in selected};status="PASS" if required<=qualified else "REJECT"
 report={"schema_version":1,"protocol_id":PROTOCOL_ID,"status":status,"checked_at_utc":datetime.now(timezone.utc).isoformat(),"engineering_only_never_train_or_val":True,"collision_free_visual_only_occluder":True,"model_inference_performed":False,"records":len(rows),"selected_geometry":selected,"required_target_yaw_pairs":len(required),"qualified_target_yaw_pairs":len(qualified),"rows":rows,"capture_manifest_sha256":common.sha256(CAPTURE/"capture_manifest.json")};common.write_json(REPORT,report)
 lines=["# Collision-free neutral-panel characterization v6","",f"Decision: **{status}**","","Engineering-only; never eligible for Train-UQ/Val-UQ/evaluation.","",f"Qualified: `{len(qualified)}/{len(required)}`.",""]
 for x in selected:lines.append(f"- `{x['target_id']}` yaw `{x['target_yaw_rad']}`, offset `{x['lateral_offset_m']}` m: `{x['target_visible_pixels']}` px; margin `{x['three_point_safety_margin_pixels']}` px.")
 REPORT_MD.parent.mkdir(parents=True,exist_ok=True);REPORT_MD.write_text("\n".join(lines)+"\n");print(json.dumps({"status":status,"qualified":len(qualified),"required":len(required)},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=("generate","lock","qc"));a=p.parse_args();{"generate":generate,"lock":lock,"qc":qc}[a.command]()
if __name__=="__main__":main()
