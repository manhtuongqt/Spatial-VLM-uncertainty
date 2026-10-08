#!/usr/bin/env python3
"""Append-only completion sweep for apple/orange after v3 qualified mango."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import cv2
import numpy as np
import yaml
import gazebo_uq_occlusion_characterization_v1 as common
import gazebo_uq_occlusion_characterization_v2 as geom
import gazebo_uq_occlusion_characterization_v3 as parent
from generate_gazebo_train_uq_v1_contract import CAMERA,LABELS,OBJECTS

ROOT=common.ROOT;CONFIG=common.CONFIG;PROTOCOL_ID="gazebo_uq_occlusion_characterization_v4"
SCENES=CONFIG/f"{PROTOCOL_ID}_scenes.yaml";ANNOTATIONS=CONFIG/f"{PROTOCOL_ID}_annotations.yaml";GATE=CONFIG/f"{PROTOCOL_ID}_gate.yaml"
LOCK=ROOT/"protocol/gazebo_uq_occlusion_characterization_v4_capture_lock.json"
RESULT=ROOT/"results/spatial_vlm_refspatial_v1/gazebo_uq_occlusion_characterization_v4";CAPTURE=RESULT/"capture_attempt_01"
REPORT=RESULT/"OCCLUSION_CHARACTERIZATION_REPORT.json";REPORT_MD=RESULT/"OCCLUSION_CHARACTERIZATION_REPORT.md"
RANGES={"ycb_apple":(0.0805,0.0830),"ycb_orange":(0.0838,0.0852)};STEP=0.0001
def gaps(t):
 lo,hi=RANGES[t];return tuple(round(lo+STEP*i,4) for i in range(round((hi-lo)/STEP)+1))
def build():
 rows=[];oracle={};i=0
 for target in RANGES:
  for yaw in common.TARGET_YAWS:
   for gap in gaps(target):
    sid=f"uq_occ_char_v4_{i:03d}";poses={target:[geom.TARGET_XY[0],geom.TARGET_XY[1],yaw],geom.OCCLUDER:[geom.ALIGNED_OCCLUDER_XY[0],round(geom.ALIGNED_OCCLUDER_XY[1]-gap,6),0.]};sig=common.layout_signature(poses)
    rows.append({"scene_id":sid,"scene_family_id":f"engineering_occlusion_characterization_v4/family_{i:03d}","layout_id":f"occ_fine_v4_{i:03d}","layout_signature_sha256":sig,"task_type":"engineering_geometry_characterization_no_inference","instruction":"ENGINEERING ONLY: do not run model inference on this capture.","poses":poses})
    oracle[sid]={"usage":"evaluator_only_geometry_characterization","target_id":target,"target_label":LABELS[target],"target_yaw_rad":yaw,"occluder_id":geom.OCCLUDER,"occluder_label":geom.OCCLUDER_LABEL,"side":-1,"center_gap_m":gap,"layout_signature_sha256":sig};i+=1
 scenes={"schema_version":1,"protocol_id":PROTOCOL_ID,"expected_scene_count":len(rows),"capture_mode":"engineering_geometry_only_no_manipulation","random_seed":14092029,"view_joint_pose":CAMERA["view_joint_pose"],"camera_frame":CAMERA["frame"],"base_frame":CAMERA["base_frame"],"coordinate_suffix":"ENGINEERING ONLY; MODEL INFERENCE FORBIDDEN.","objects":OBJECTS,"scenes":rows}
 annotations={"schema_version":1,"protocol_id":PROTOCOL_ID,"oracle_usage":"geometry_characterization_only_never_train_or_val","parent_negative_attempt":{"protocol_id":parent.PROTOCOL_ID,"report_sha256":common.sha256(parent.REPORT),"decision":"REJECT_WITH_MANGO_QUALIFIED"},"preregistered_ranges_m":RANGES,"step_m":STEP,"scenes":oracle}
 gate={"schema_version":1,"protocol_id":PROTOCOL_ID,"official_pilot_gate_unchanged":{"min_inclusive":1,"max_exclusive":120},"internal_design_margin":{"min_inclusive":20,"max_inclusive":100},"selection_rule":{"current_and_adjacent_rows_inside_internal_margin":True,"prefer_maximum_minimum_distance_to_20_or_100":True,"combine_mango_from_frozen_v3":True},"policies":{"engineering_only":True,"no_model_inference":True,"no_training":True,"never_materialize_into_train_or_val":True,"semantic_labels_evaluator_only":True,"no_dev_calibration_or_test_access":True,"no_robot_manipulation":True}}
 return scenes,annotations,gate
def generate():
 if any(p.exists() for p in (SCENES,ANNOTATIONS,GATE,LOCK)):raise FileExistsError("refusing overwrite")
 for p,x in zip((SCENES,ANNOTATIONS,GATE),build()):p.write_text(common.serialized(x))
 print(json.dumps({"status":"GENERATED","scenes":len(build()[0]["scenes"])},indent=2))
def lock():
 if LOCK.exists():raise FileExistsError("refusing overwrite lock")
 for p,x in zip((SCENES,ANNOTATIONS,GATE),build()):
  if p.read_text()!=common.serialized(x):raise ValueError(f"edited {p}")
 own={"protocol/gazebo_uq_occlusion_characterization_v1.py","protocol/gazebo_uq_occlusion_characterization_v2.py","protocol/gazebo_uq_occlusion_characterization_v3.py","protocol/gazebo_uq_occlusion_characterization_v3_capture_lock.json",str(parent.REPORT.relative_to(ROOT)),"protocol/gazebo_uq_occlusion_characterization_v4.py",str(SCENES.relative_to(ROOT)),str(ANNOTATIONS.relative_to(ROOT)),str(GATE.relative_to(ROOT))}
 sources=sorted(set(common.REQUIRED_CAPTURE_SOURCES)|own);payload={"schema_version":1,"protocol_id":PROTOCOL_ID,"status":"LOCKED_BEFORE_CAPTURE_AND_INFERENCE","locked_at_utc":datetime.now(timezone.utc).isoformat(),"workspace_root":str(ROOT),"purpose":"append-only completion characterization for apple/orange","scene_count":len(build()[0]["scenes"]),"model_inventory_sha256":"NO_MODEL_INFERENCE_ENGINEERING_CHARACTERIZATION_V4","source_artifact_sha256":{n:common.sha256(ROOT/n) for n in sources},"prohibitions":build()[2]["policies"]};common.write_json(LOCK,payload);print(json.dumps({"status":"LOCKED","sha256":common.sha256(LOCK)},indent=2))
def qc():
 lk=json.loads(LOCK.read_text())
 for n,d in lk["source_artifact_sha256"].items():
  if common.sha256(ROOT/n)!=d:raise ValueError(f"locked source changed {n}")
 manifest=json.loads((CAPTURE/"capture_manifest.json").read_text());inputs=[json.loads(x) for x in (CAPTURE/"input_manifest.jsonl").read_text().splitlines() if x];oracle=yaml.safe_load(ANNOTATIONS.read_text())["scenes"]
 if manifest.get("status")!="COMPLETE" or len(inputs)!=len(oracle):raise ValueError("capture incomplete")
 rows=[];groups={}
 for rec in inputs:
  sid=rec["scene_id"];it=oracle[sid];lab=cv2.imread(str(CAPTURE/sid/"evaluator/semantic_labels.png"),cv2.IMREAD_UNCHANGED);lab=lab[...,0] if lab.ndim==3 else lab;px=int(np.count_nonzero(lab==int(it["target_label"])));row={"scene_id":sid,**it,"target_visible_pixels":px,"inside_official_gate":1<=px<120,"inside_internal_margin":20<=px<=100};rows.append(row);groups.setdefault((it["target_id"],it["target_yaw_rad"]),[]).append(row)
 selected=[]
 for key,g in sorted(groups.items()):
  g.sort(key=lambda x:x["center_gap_m"]);cand=[]
  for i in range(1,len(g)-1):
   hood=g[i-1:i+2]
   if all(x["inside_internal_margin"] for x in hood):
    margin=min(min(x["target_visible_pixels"]-20,100-x["target_visible_pixels"]) for x in hood);cand.append((margin,g[i]))
  if cand:
   margin,row=max(cand,key=lambda x:x[0]);selected.append({**row,"three_point_safety_margin_pixels":margin})
 p3=json.loads(parent.REPORT.read_text());mango=[x for x in p3["selected_geometry"] if x["target_id"]=="mango"]
 combined=selected+mango;required={(t,y) for t in common.TARGETS for y in common.TARGET_YAWS};qualified={(x["target_id"],x["target_yaw_rad"]) for x in combined};status="PASS" if required<=qualified else "REJECT"
 report={"schema_version":1,"protocol_id":PROTOCOL_ID,"status":status,"checked_at_utc":datetime.now(timezone.utc).isoformat(),"engineering_only_never_train_or_val":True,"model_inference_performed":False,"records":len(rows),"selected_geometry_current":selected,"selected_geometry_combined":combined,"required_target_yaw_pairs":len(required),"qualified_target_yaw_pairs":len(qualified),"rows":rows,"capture_manifest_sha256":common.sha256(CAPTURE/"capture_manifest.json"),"frozen_v3_report_sha256":common.sha256(parent.REPORT)};common.write_json(REPORT,report)
 lines=["# Controlled occlusion characterization v4","",f"Decision: **{status}**","","Engineering-only evidence; never eligible for Train-UQ/Val-UQ/evaluation.","",f"Combined qualified target/yaw pairs: `{len(qualified)}/{len(required)}`.",""]
 for x in combined:lines.append(f"- `{x['target_id']}` yaw `{x['target_yaw_rad']}`, gap `{x['center_gap_m']}` m: `{x['target_visible_pixels']}` px; 3-point margin `{x['three_point_safety_margin_pixels']}` px.")
 REPORT_MD.parent.mkdir(parents=True,exist_ok=True);REPORT_MD.write_text("\n".join(lines)+"\n");print(json.dumps({"status":status,"qualified":len(qualified),"required":len(required)},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=("generate","lock","qc"));a=p.parse_args();{"generate":generate,"lock":lock,"qc":qc}[a.command]()
if __name__=="__main__":main()
