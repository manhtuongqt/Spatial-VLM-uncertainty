#!/usr/bin/env python3
"""Fit/select one preregistered L2 logistic risk estimator and evaluate it."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1];RELATIONS=["leftmost","rightmost","second_from_left","second_from_right"]
FEATURE_NAMES=["action_POINT","action_ABSTAIN","action_INVALID","exact_contract","self_consistency","point_dispersion","predicted_x","predicted_y","boundary_distance","local_depth_valid_fraction","local_depth_median","local_depth_mad",*[f"relation_{x}" for x in RELATIONS]]
CS=[0.01,0.1,1.0,10.0];THRESHOLDS=np.linspace(0,1,101);RADIUS=8;HIT_RADIUS=.08;BOOTSTRAP_DRAWS=10000;BOOTSTRAP_SEED=13092026

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def dump(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")
def point_dispersion(pred):
 pts=np.asarray([x["prediction_xy"] for x in pred.get("stochastic_draws",[]) if x.get("action")=="POINT" and x.get("prediction_xy") is not None],float)
 return 1.0 if len(pts)<2 else float(np.mean(np.linalg.norm(pts-pts.mean(0),axis=1)))
def depth_features(path,pred):
 if pred.get("action")!="POINT" or pred.get("prediction_xy") is None:return [0.,0.,0.]
 d=np.load(path,allow_pickle=False);x=int(round(pred["prediction_xy"][0]*(d.shape[1]-1)));y=int(round(pred["prediction_xy"][1]*(d.shape[0]-1)))
 q=d[max(0,y-RADIUS):min(d.shape[0],y+RADIUS+1),max(0,x-RADIUS):min(d.shape[1],x+RADIUS+1)];valid=np.isfinite(q)&(q>=.05)&(q<=2.)
 if not valid.any():return [0.,0.,0.]
 v=q[valid];med=float(np.median(v));return [float(valid.mean()),med,float(np.median(np.abs(v-med)))]
def feature(infer,pred,dataset):
 action=pred.get("action","INVALID");xy=pred.get("prediction_xy") if action=="POINT" else None;x,y=(xy if xy else [-1.,-1.]);boundary=min(x,y,1-x,1-y) if xy else -1.
 relation=infer["relation_variant"]
 return np.asarray([float(action=="POINT"),float(action=="ABSTAIN"),float(action=="INVALID"),float(pred.get("exact_contract",False)),float(pred.get("self_consistency_confidence",0)),point_dispersion(pred),x,y,boundary,*depth_features(dataset/infer["metric_depth"],pred),*[float(relation==r) for r in RELATIONS]],float)
def joined(dataset,predictions):
 inf={x["sample_id"]:x for x in rows(dataset/"inference_manifest.jsonl")};gt={x["sample_id"]:x for x in rows(dataset/"evaluator_ground_truth.jsonl")};pr={x["sample_id"]:x for x in rows(predictions)}
 if set(inf)!=set(gt) or set(inf)!=set(pr):raise ValueError("inference/prediction/oracle ID sets differ")
 out=[]
 for sid in inf:
  g,p=gt[sid],pr[sid];dist=math.dist(p["prediction_xy"],g["target_xy"]) if g["answerability_state"]=="FOUND" and p.get("action")=="POINT" and p.get("prediction_xy") else math.inf
  hit=g["answerability_state"]=="FOUND" and dist<=HIT_RADIUS;unsafe=g["answerability_state"]!="FOUND" or not hit
  out.append({"sample_id":sid,"family_id":g["family_id"],"split":g["split"],"state":g["answerability_state"],"relation":g["relation_variant"],"distance":dist if math.isfinite(dist) else None,"hit":hit,"unsafe":unsafe,"feature":feature(inf[sid],p,dataset),"b0":p})
 return out
def aurc(y,risk):
 order=np.argsort(np.asarray(risk),kind="stable");z=np.asarray(y,float)[order];return float(np.mean(np.cumsum(z)/np.arange(1,len(z)+1)))
def curve(y,risk):
 order=np.argsort(np.asarray(risk),kind="stable");z=np.asarray(y,float)[order];return [{"coverage":float((i+1)/len(z)),"risk":float(z[:i+1].mean()),"threshold":float(np.asarray(risk)[order[i]])} for i in range(len(z))]
def ece(y,risk):
 y=np.asarray(y,float);risk=np.asarray(risk,float);total=0.
 for i in range(10):
  lo,hi=i/10,(i+1)/10;m=(risk>=lo)&((risk<hi) if i<9 else (risk<=hi))
  if m.any():total+=m.mean()*abs(float(risk[m].mean()-y[m].mean()))
 return float(total)
def decisions(data,risk,threshold):
 out=[]
 for x,r in zip(data,risk):
  abstain=r>=threshold;action="ABSTAIN" if abstain else x["b0"].get("action","INVALID")
  exact=True if abstain else bool(x["b0"].get("exact_contract",False));correct=x["hit"] if x["state"]=="FOUND" else action=="ABSTAIN"
  out.append({**x,"risk":float(r),"action":action,"exact":exact,"correct":bool(correct)})
 return out
def metrics(data,risk,threshold):
 d=decisions(data,risk,threshold);y=np.asarray([x["unsafe"] for x in data],int);r=np.asarray(risk,float);found=[x for x in d if x["state"]=="FOUND"];non=[x for x in d if x["state"]!="FOUND"]
 return {"n":len(d),"unsafe_rate":float(y.mean()),"aurc":aurc(y,r),"auroc_error":float(roc_auc_score(y,r)) if len(set(y))==2 else None,"ece_10bin":ece(y,r),"brier":float(brier_score_loss(y,r)),"coverage":float(np.mean([x["action"]=="POINT" for x in d])),"found_hit_at_008":float(np.mean([x["action"]=="POINT" and x["hit"] for x in found])),"nonfound_abstain_recall":float(np.mean([x["action"]=="ABSTAIN" for x in non])),"nonfound_false_accept":float(np.mean([x["action"]=="POINT" for x in non])),"exact_contract":float(np.mean([x["exact"] for x in d])),"safe_task_accuracy":float(np.mean([x["correct"] for x in d])),"threshold":float(threshold)}
def baseline_risk(data):return np.asarray([float(x["b0"].get("predictive_uncertainty",1.)) for x in data])
def eligible(m,b):return m["found_hit_at_008"]-b["found_hit_at_008"]>=-.125-1e-12 and m["nonfound_abstain_recall"]>b["nonfound_abstain_recall"] and m["nonfound_false_accept"]<b["nonfound_false_accept"] and m["exact_contract"]>=b["exact_contract"] and m["aurc"]<=b["aurc"]+1e-12
def mcnemar(a,b):
 n01=sum((not x) and y for x,y in zip(a,b));n10=sum(x and (not y) for x,y in zip(a,b));n=n01+n10
 if not n:return {"b0_wrong_method_right":n01,"b0_right_method_wrong":n10,"exact_two_sided_p":1.}
 k=min(n01,n10);p=min(1.,2*sum(math.comb(n,i) for i in range(k+1))/(2**n));return {"b0_wrong_method_right":n01,"b0_right_method_wrong":n10,"exact_two_sided_p":p}
def bootstrap(data,r0,r1,t0,t1):
 rng=np.random.default_rng(BOOTSTRAP_SEED);n=len(data);names=["safe_task_accuracy","found_hit_at_008","nonfound_abstain_recall","nonfound_false_accept","aurc"];vals={x:[] for x in names}
 for _ in range(BOOTSTRAP_DRAWS):
  ix=rng.integers(0,n,n);sample=[data[i] for i in ix];a=metrics(sample,np.asarray(r0)[ix],t0);b=metrics(sample,np.asarray(r1)[ix],t1)
  for name in names:vals[name].append(b[name]-a[name])
 return {name:{"delta":metrics(data,r1,t1)[name]-metrics(data,r0,t0)[name],"ci95":[float(np.quantile(v,.025)),float(np.quantile(v,.975))]} for name,v in vals.items()}
def save_curves(out,split,data,risks):
 path=out/f"risk_coverage_{split}.csv"
 with path.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=["method","coverage","selective_risk","risk_threshold"]);w.writeheader()
  for method,risk in risks.items():
   for p in curve([x["unsafe"] for x in data],risk):w.writerow({"method":method,"coverage":p["coverage"],"selective_risk":p["risk"],"risk_threshold":p["threshold"]})
def figures(out,all_metrics,split_data,split_risks):
 fig,ax=plt.subplots(figsize=(6,4))
 for name,risk in split_risks.items():
  c=curve([x["unsafe"] for x in split_data],risk);ax.plot([x["coverage"] for x in c],[x["risk"] for x in c],label=f"{name} AURC={aurc([x['unsafe'] for x in split_data],risk):.3f}")
 ax.set(xlabel="Coverage",ylabel="Selective risk",title="Val-UQ risk–coverage");ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/"01_val_risk_coverage.png",dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(6,4));names=["found_hit_at_008","nonfound_abstain_recall","nonfound_false_accept","exact_contract"]
 x=np.arange(len(names));ax.bar(x-.18,[all_metrics["b0_val"][n] for n in names],.36,label="B0");ax.bar(x+.18,[all_metrics["method_val"][n] for n in names],.36,label="risk estimator");ax.set_xticks(x,names,rotation=20,ha="right");ax.set_ylim(0,1.05);ax.legend();fig.tight_layout();fig.savefig(out/"02_val_decision_metrics.png",dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(5,5));y=np.asarray([z["unsafe"] for z in split_data],float);r=np.asarray(split_risks["method"]);xs=[];ys=[];sizes=[]
 for i in range(10):
  m=(r>=i/10)&((r<(i+1)/10) if i<9 else (r<=1));
  if m.any():xs.append(r[m].mean());ys.append(y[m].mean());sizes.append(m.sum())
 ax.plot([0,1],[0,1],"--",color="gray");ax.scatter(xs,ys,s=np.asarray(sizes)*5);ax.set(xlabel="Predicted unsafe probability",ylabel="Observed unsafe frequency",title="Val-UQ reliability",xlim=(0,1),ylim=(0,1));fig.tight_layout();fig.savefig(out/"03_val_reliability.png",dpi=180);plt.close(fig)
def fit(args):
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);data=joined(args.dataset.resolve(),args.predictions.resolve());train=[x for x in data if x["split"]=="train_uq"];val=[x for x in data if x["split"]=="val_uq"]
 X=np.stack([x["feature"] for x in train]);y=np.asarray([x["unsafe"] for x in train],int);Xv=np.stack([x["feature"] for x in val]);yv=np.asarray([x["unsafe"] for x in val],int)
 scaler=StandardScaler().fit(X);b0v=baseline_risk(val);b0m=metrics(val,b0v,1.01);candidates=[];models={}
 with (out/"wp5_train_log.jsonl").open("w") as log:
  for C in CS:
   model=LogisticRegression(C=C,penalty="l2",solver="liblinear",random_state=13092026,max_iter=2000).fit(scaler.transform(X),y);risk=model.predict_proba(scaler.transform(Xv))[:,1];models[C]=model
   for threshold in THRESHOLDS:
    m=metrics(val,risk,threshold);ok=eligible(m,b0m);candidates.append((ok,m["aurc"],m["nonfound_false_accept"],-m["found_hit_at_008"],C,float(threshold),m,risk))
   best=min([x for x in candidates if x[4]==C],key=lambda z:(not z[0],z[1],z[2],z[3],z[5]));log.write(json.dumps({"C":C,"eligible_threshold_exists":any(x[0] for x in candidates if x[4]==C),"best_threshold":best[5],"val_metrics":best[6]})+"\n")
 eligible_rows=[x for x in candidates if x[0]];selected=min(eligible_rows or candidates,key=lambda z:(z[1],z[2],z[3],CS.index(z[4]),z[5]));ok,_,_,_,C,threshold,valm,_=selected;model=models[C]
 risk_all=model.predict_proba(scaler.transform(np.stack([x["feature"] for x in data])))[:,1];risk_train=risk_all[:len(train)];risk_val=risk_all[len(train):]
 # joined preserves manifest order: all 256 Train rows precede 64 Val rows.
 if [x["split"] for x in data]!=["train_uq"]*256+["val_uq"]*64:raise ValueError("unexpected split ordering")
 allm={"b0_train":metrics(train,baseline_risk(train),1.01),"method_train":metrics(train,risk_train,threshold),"b0_val":b0m,"method_val":metrics(val,risk_val,threshold)}
 decision={"schema_version":1,"status":"VAL_ELIGIBLE" if ok else "VAL_NOT_ELIGIBLE","selected_C":C,"selected_threshold":threshold,"feature_names":FEATURE_NAMES,"hit_radius":HIT_RADIUS,"selection_split":"val_uq_only","eligible_candidate_count":len(eligible_rows),"metrics":allm}
 model_art={"schema_version":1,"method":"StandardScaler_then_L2_LogisticRegression","C":C,"threshold":threshold,"feature_names":FEATURE_NAMES,"scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),"coefficient":model.coef_[0].tolist(),"intercept":float(model.intercept_[0]),"val_eligible":bool(ok)};dump(out/"selected_estimator.json",model_art);joblib.dump({"scaler":scaler,"model":model,"threshold":threshold},out/"selected_estimator.joblib")
 dump(out/"wp5_val_selection.json",decision);dump(out/"wp5_metrics.json",allm)
 scored=[]
 for x,r in zip(data,risk_all):
  action="ABSTAIN" if r>=threshold else x["b0"]["action"];scored.append({"sample_id":x["sample_id"],"family_id":x["family_id"],"split":x["split"],"answerability_state":x["state"],"relation_variant":x["relation"],"unsafe":x["unsafe"],"b0_action":x["b0"]["action"],"b0_risk_proxy":x["b0"].get("predictive_uncertainty",1.),"estimated_unsafe_probability":float(r),"proposed_action":action,"grounding_hit_at_008":x["hit"]})
 with (out/"wp5_scored_predictions.jsonl").open("w") as f:
  for x in scored:f.write(json.dumps(x,sort_keys=True)+"\n")
 for split,part,rr in [("train_uq",train,risk_train),("val_uq",val,risk_val)]:save_curves(out,split,part,{"b0":baseline_risk(part),"method":rr})
 boot=bootstrap(val,b0v,risk_val,1.01,threshold);dump(out/"wp5_val_family_bootstrap_10000.json",{"draws":BOOTSTRAP_DRAWS,"seed":BOOTSTRAP_SEED,"paired":True,"unit":"parent_family","effects":boot})
 b0d=decisions(val,b0v,1.01);md=decisions(val,risk_val,threshold);dump(out/"wp5_val_mcnemar.json",mcnemar([x["correct"] for x in b0d],[x["correct"] for x in md]))
 figures(out,allm,val,{"b0":b0v,"method":risk_val});print(json.dumps(decision,indent=2))
def lock_selected(args):
 out=args.output.resolve();model=out/"selected_estimator.json";binary=out/"selected_estimator.joblib";selection=out/"wp5_val_selection.json"
 value={"schema_version":1,"status":"FROZEN_BY_VAL_UQ_BEFORE_DEV_CONFIRMATION","locked_at_utc":datetime.now(timezone.utc).isoformat(),"selected_estimator_sha256":sha(model),"selected_estimator_joblib_sha256":sha(binary),"val_selection_sha256":sha(selection),"estimator_source_sha256":sha(Path(__file__).resolve()),"dev_confirmation_count_allowed":1,"calibration_and_test_sealed":True};dump(out/"wp5_checkpoint_or_estimator_lock.json",value);print(json.dumps(value,indent=2))
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="cmd",required=True);f=s.add_parser("fit-evaluate");f.add_argument("--dataset",type=Path,required=True);f.add_argument("--predictions",type=Path,required=True);f.add_argument("--output",type=Path,required=True);l=s.add_parser("lock-selected");l.add_argument("--output",type=Path,required=True);a=p.parse_args();fit(a) if a.cmd=="fit-evaluate" else lock_selected(a)
if __name__=="__main__":main()
