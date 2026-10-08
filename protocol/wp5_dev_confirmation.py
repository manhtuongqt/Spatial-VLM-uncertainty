#!/usr/bin/env python3
"""Lock, then run exactly one paired Dev-v2 confirmation after Val freeze."""
from __future__ import annotations
import argparse,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import wp5_spatial_uncertainty_estimator as est
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/spatial_vlm_refspatial_v1/wp5_spatial_uncertainty";DEV=ROOT/"datasets/Gazebo_dev_answerability_v2";PRED=ROOT/"results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2/evaluation/b0/predictions.jsonl";LOCK=OUT/"wp5_dev_confirmation_lock.json";REPORT=OUT/"wp5_dev_confirmation_report.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")
def relation(text):
 text=text.lower()
 if "second object from left" in text or "second from left" in text:return "second_from_left"
 if "second object from right" in text or "second from right" in text:return "second_from_right"
 if "leftmost" in text:return "leftmost"
 if "rightmost" in text:return "rightmost"
 raise ValueError("relation not recoverable from instruction")
def source_paths():return [Path(__file__).resolve(),OUT/"wp5_method_lock.json",OUT/"wp5_checkpoint_or_estimator_lock.json",OUT/"selected_estimator.json",DEV/"manifest.json",DEV/"inference_manifest.jsonl",PRED,PRED.parent/"run.json"]
def lock():
 if LOCK.exists():raise FileExistsError("confirmation lock exists")
 value={"schema_version":1,"status":"LOCKED_AFTER_VAL_FREEZE_BEFORE_SINGLE_DEV_CONFIRMATION","locked_at_utc":datetime.now(timezone.utc).isoformat(),"source_artifact_sha256":{str(p.relative_to(ROOT)):sha(p) for p in source_paths()},"confirmation_count_allowed":1,"no_refit_no_reselection":True,"calibration_and_test_sealed":True};dump(LOCK,value);print(json.dumps({"status":"LOCKED","sha256":sha(LOCK)},indent=2))
def confirm():
 if REPORT.exists():raise FileExistsError("Dev confirmation already exists")
 lockv=json.loads(LOCK.read_text())
 for name,digest in lockv["source_artifact_sha256"].items():
  if sha(ROOT/name)!=digest:raise ValueError(f"confirmation input drift: {name}")
 inf={x["sample_id"]:x for x in est.rows(DEV/"inference_manifest.jsonl")};gt={x["sample_id"]:x for x in est.rows(DEV/"evaluator_ground_truth.jsonl")};pr={x["sample_id"]:x for x in est.rows(PRED)}
 if set(inf)!=set(gt) or set(inf)!=set(pr) or len(inf)!=64:raise ValueError("Dev paired IDs mismatch")
 data=[]
 for sid,i in inf.items():
  i=dict(i);i["relation_variant"]=relation(i["instruction"]);g,p=gt[sid],pr[sid];distance=math.dist(p["prediction_xy"],g["target_xy"]) if g["answerability_state"]=="FOUND" and p.get("action")=="POINT" and p.get("prediction_xy") else math.inf;hit=g["answerability_state"]=="FOUND" and distance<=est.HIT_RADIUS
  data.append({"sample_id":sid,"family_id":g["family_id"],"split":"dev","state":g["answerability_state"],"relation":g["relation_variant"],"target_category":g.get("target_category"),"distance":distance if math.isfinite(distance) else None,"hit":hit,"unsafe":g["answerability_state"]!="FOUND" or not hit,"feature":est.feature(i,p,DEV),"b0":p})
 model=json.loads((OUT/"selected_estimator.json").read_text());X=np.stack([x["feature"] for x in data]);z=(X-np.asarray(model["scaler_mean"]))/np.asarray(model["scaler_scale"]);risk=1/(1+np.exp(-(z@np.asarray(model["coefficient"])+model["intercept"])));b0risk=est.baseline_risk(data);b0m=est.metrics(data,b0risk,1.01);methodm=est.metrics(data,risk,model["threshold"]);boot=est.bootstrap(data,b0risk,risk,1.01,model["threshold"]);b0d=est.decisions(data,b0risk,1.01);md=est.decisions(data,risk,model["threshold"])
 by={}
 for field in ["state","relation","target_category"]:
  by[field]={}
  for value in sorted({str(x.get(field)) for x in data}):
   ix=[j for j,x in enumerate(data) if str(x.get(field))==value];part=[data[j] for j in ix];by[field][value]={"b0":est.metrics(part,b0risk[ix],1.01),"method":est.metrics(part,risk[ix],model["threshold"])}
 value={"schema_version":1,"status":"COMPLETED_ONE_CONFIRMATION_NO_REFIT_NO_RESELECTION","confirmed_at_utc":datetime.now(timezone.utc).isoformat(),"n":64,"estimator_lock_sha256":sha(OUT/"wp5_checkpoint_or_estimator_lock.json"),"b0":b0m,"method":methodm,"paired_family_bootstrap_10000":boot,"mcnemar":est.mcnemar([x["correct"] for x in b0d],[x["correct"] for x in md]),"stratified":by,"selection_changed_after_confirmation":False}
 dump(REPORT,value)
 with (OUT/"wp5_dev_confirmation_predictions.jsonl").open("w") as f:
  for x,r,d in zip(data,risk,md):f.write(json.dumps({"sample_id":x["sample_id"],"family_id":x["family_id"],"answerability_state":x["state"],"relation_variant":x["relation"],"target_category":x["target_category"],"unsafe":x["unsafe"],"b0_action":x["b0"]["action"],"b0_risk_proxy":x["b0"].get("predictive_uncertainty",1.),"estimated_unsafe_probability":float(r),"proposed_action":d["action"],"grounding_hit_at_008":x["hit"]},sort_keys=True)+"\n")
 mdtext=f"# WP5 single Dev-v2 confirmation\n\nStatus: **{value['status']}**. No refit or reselection occurred.\n\n- B0 safe-task accuracy: {b0m['safe_task_accuracy']:.4f}\n- Method safe-task accuracy: {methodm['safe_task_accuracy']:.4f}\n- B0 AURC: {b0m['aurc']:.4f}\n- Method AURC: {methodm['aurc']:.4f}\n- B0 false accept: {b0m['nonfound_false_accept']:.4f}\n- Method false accept: {methodm['nonfound_false_accept']:.4f}\n\nDev-v2 is confirmation-only and is not a final test. Calibration/Test remain sealed.\n";(OUT/"wp5_dev_confirmation_report.md").write_text(mdtext);print(json.dumps(value,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=("lock","confirm"));a=p.parse_args();lock() if a.command=="lock" else confirm()
if __name__=="__main__":main()
