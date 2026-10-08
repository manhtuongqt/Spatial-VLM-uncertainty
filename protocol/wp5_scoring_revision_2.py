#!/usr/bin/env python3
"""Correct safe-task/McNemar scoring without changing frozen risks/decisions."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/spatial_vlm_refspatial_v1/wp5_spatial_uncertainty";SEED=13092026;DRAWS=10000
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exact_p(n01,n10):
 n=n01+n10
 return 1. if n==0 else min(1.,2*sum(math.comb(n,i) for i in range(min(n01,n10)+1))/(2**n))
def score(pred):
 b0=[];method=[]
 for x in pred:
  if x["answerability_state"]=="FOUND":
   b0.append(x["b0_action"]=="POINT" and x["grounding_hit_at_008"]);method.append(x["proposed_action"]=="POINT" and x["grounding_hit_at_008"])
  else:
   b0.append(x["b0_action"]=="ABSTAIN");method.append(x["proposed_action"]=="ABSTAIN")
 return np.asarray(b0,bool),np.asarray(method,bool)
def paired(pred):
 b,m=score(pred);n01=int(np.sum(~b&m));n10=int(np.sum(b&~m));rng=np.random.default_rng(SEED);n=len(b);draw=np.empty(DRAWS)
 for i in range(DRAWS):
  ix=rng.integers(0,n,n);draw[i]=m[ix].mean()-b[ix].mean()
 return {"b0_safe_task_accuracy":float(b.mean()),"method_safe_task_accuracy":float(m.mean()),"delta":float(m.mean()-b.mean()),"family_bootstrap_draws":DRAWS,"family_bootstrap_seed":SEED,"family_bootstrap_ci95":[float(np.quantile(draw,.025)),float(np.quantile(draw,.975))],"mcnemar":{"b0_wrong_method_right":n01,"b0_right_method_wrong":n10,"exact_two_sided_p":exact_p(n01,n10)}}
def clean(v):
 if isinstance(v,float) and not math.isfinite(v):return None
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,list):return [clean(x) for x in v]
 return v
def main():
 frozen=json.loads((OUT/"wp5_checkpoint_or_estimator_lock.json").read_text());selection=json.loads((OUT/"wp5_val_selection.json").read_text());full=rows(OUT/"wp5_scored_predictions.jsonl");dev=rows(OUT/"wp5_dev_confirmation_predictions.jsonl")
 parts={split:[x for x in full if x["split"]==split] for split in ("train_uq","val_uq")};corrected={k:paired(v) for k,v in parts.items()};corrected["dev_confirmation"]=paired(dev)
 # Preserve the already-frozen selection inputs; only the non-gating derived
 # safe-task field and paired test are superseded.
 authoritative=clean(selection);authoritative["status"]="VAL_ELIGIBLE";authoritative["score_schema_revision"]=2;authoritative["safe_task_scoring_rule"]="FOUND requires proposed POINT and Hit@0.08; non-FOUND requires exact ABSTAIN; INVALID is never ABSTAIN";authoritative["corrected_paired_safe_task"]=corrected["val_uq"]
 authoritative["metrics"]["b0_train"]["safe_task_accuracy"]=corrected["train_uq"]["b0_safe_task_accuracy"];authoritative["metrics"]["method_train"]["safe_task_accuracy"]=corrected["train_uq"]["method_safe_task_accuracy"];authoritative["metrics"]["b0_val"]["safe_task_accuracy"]=corrected["val_uq"]["b0_safe_task_accuracy"];authoritative["metrics"]["method_val"]["safe_task_accuracy"]=corrected["val_uq"]["method_safe_task_accuracy"]
 dump(OUT/"wp5_val_selection_decision.json",authoritative)
 oldboot=json.loads((OUT/"wp5_val_family_bootstrap_10000_uncorrected_attempt_01.json").read_text());oldboot["score_schema_revision"]=2;oldboot["effects"]["safe_task_accuracy"]={"delta":corrected["val_uq"]["delta"],"ci95":corrected["val_uq"]["family_bootstrap_ci95"]};dump(OUT/"wp5_val_family_bootstrap_10000.json",oldboot);dump(OUT/"wp5_val_mcnemar.json",corrected["val_uq"]["mcnemar"])
 raw=clean(json.loads((OUT/"wp5_dev_confirmation_report_uncorrected_attempt_01.json").read_text()));raw["score_schema_revision"]=2;raw["safe_task_scoring_rule"]="FOUND requires proposed POINT and Hit@0.08; non-FOUND requires exact ABSTAIN; INVALID is never ABSTAIN";raw["b0"]["safe_task_accuracy"]=corrected["dev_confirmation"]["b0_safe_task_accuracy"];raw["method"]["safe_task_accuracy"]=corrected["dev_confirmation"]["method_safe_task_accuracy"];raw["paired_family_bootstrap_10000"]["safe_task_accuracy"]={"delta":corrected["dev_confirmation"]["delta"],"ci95":corrected["dev_confirmation"]["family_bootstrap_ci95"]};raw["mcnemar"]=corrected["dev_confirmation"]["mcnemar"];raw["selection_changed_after_confirmation"]=False;dump(OUT/"wp5_dev_confirmation_report.json",raw)
 md=f"# WP5 single Dev-v2 confirmation\n\nStatus: **{raw['status']}**. Score schema revision 2 corrects safe-task/McNemar only; no inference, refit, threshold change, or reselection occurred.\n\n- B0 safe-task accuracy: {raw['b0']['safe_task_accuracy']:.4f}\n- Method safe-task accuracy: {raw['method']['safe_task_accuracy']:.4f}\n- B0 AURC: {raw['b0']['aurc']:.4f}\n- Method AURC: {raw['method']['aurc']:.4f}\n- B0 false accept: {raw['b0']['nonfound_false_accept']:.4f}\n- Method false accept: {raw['method']['nonfound_false_accept']:.4f}\n- B0 FOUND Hit@0.08: {raw['b0']['found_hit_at_008']:.4f}\n- Method FOUND Hit@0.08: {raw['method']['found_hit_at_008']:.4f}\n\nDev-v2 is confirmation-only and is not a final test. Calibration/Test remain sealed.\n";(OUT/"wp5_dev_confirmation_report.md").write_text(md)
 dump(OUT/"wp5_scoring_revision_2.json",{"status":"APPLIED_DERIVED_SCORING_CORRECTION_ONLY","frozen_estimator_lock_sha256":sha(OUT/"wp5_checkpoint_or_estimator_lock.json"),"selection_or_model_changed":False,"corrected":corrected});print(json.dumps(corrected,indent=2))
if __name__=="__main__":main()
