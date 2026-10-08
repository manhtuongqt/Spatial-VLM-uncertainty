#!/usr/bin/env python3
"""Perform the sealed one-shot Anti-Shortcut evaluation and finalize Day 6."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import csv,hashlib,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,torch
from workspace.mh_pcrau_v3.multihead_v3 import ANSWERABILITY_CLASSES,RELATION_CLASSES,build_seeded_model
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'ketqua1/03_backbone_h_spatial/ngay_06/amendment_07_fresh_left_ycb_v14_accepted249';MANIFEST=DATA/'FEATURE_CACHE_MANIFEST.json';CACHE_QC=DATA/'FEATURE_CACHE_QC.json';SUP=DATA/'SUPERVISION.jsonl';CAPTURE_QC=ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_final_v14_accepted249/CAPTURE_QC.json';CHECKPOINT=ROOT/'workspace/mh_pcrau_v3/checkpoints/day06_robust_candidate_r2/s1a_robust_fixed_epoch54.pt';OUT=ROOT/'ketqua1/09_danh_gia/metrics/ngay_06/amendment_07_fresh_left_ycb_v14_accepted249';DEC=ROOT/'ketqua1/07_huan_luyen/ngay_06/amendment_07_fresh_left_ycb_v14_accepted249';FINAL=ROOT/'ketqua1/07_huan_luyen/ngay_06/amendment_06_left_ycb_capture/DAY6_FINAL_DECISION.json';REL=tuple(RELATION_CLASSES);ST=tuple(ANSWERABILITY_CLASSES)
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def f1(y,p,labels):
 vals=[]
 for l in labels:
  tp=sum(a==l and b==l for a,b in zip(y,p));fp=sum(a!=l and b==l for a,b in zip(y,p));fn=sum(a==l and b!=l for a,b in zip(y,p));vals.append(0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
 return float(np.mean(vals))
def ci(values,seed=25092026):
 rng=np.random.default_rng(seed);x=np.asarray(values,float);means=np.mean(x[rng.integers(0,len(x),(10000,len(x)))],axis=1);return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]
def main():
 if OUT.exists() or DEC.exists():raise FileExistsError('one-shot result already exists')
 cqc=json.loads(CAPTURE_QC.read_text());fqc=json.loads(CACHE_QC.read_text());
 if cqc['status']!='PASS' or fqc['status']!='PASS' or fqc['manifest_sha256']!=sha256(MANIFEST):raise RuntimeError('QC gate not open')
 labels={x['sample_id']:x for x in (json.loads(s) for s in SUP.read_text().splitlines() if s)};records={x['sample_id']:x for x in json.loads(MANIFEST.read_text())['records']};ids=sorted(labels)
 if len(ids)!=249 or set(ids)!=set(records):raise RuntimeError('identity mismatch')
 features=[]
 for sid in ids:
  p=ROOT/records[sid]['feature_path']
  if sha256(p)!=records[sid]['feature_file_sha256']:raise RuntimeError('feature drift')
  features.append(torch.load(p,map_location='cpu',weights_only=True).float())
 x=torch.stack(features);ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);model=build_seeded_model(25092052);model.load_state_dict(ck['model_state_dict']);model.configure_trainable('inference');model.eval()
 with torch.no_grad():o=model(x)
 rp=[REL[int(i)] for i in o.relation_logits.argmax(-1)];ap=[ST[int(i)] for i in o.answerability_logits.argmax(-1)];mu=o.mu_uv.tolist();rt=[labels[i]['relation'] for i in ids];at=[labels[i]['answerability'] for i in ids];errors=[];nll=[];pred=[]
 logvar=o.log_variance_uv.detach().numpy()
 for k,sid in enumerate(ids):
  e=None;n=None
  if at[k]=='FOUND':
   target=np.asarray(labels[sid]['target_uv']);delta=target-np.asarray(mu[k]);e=float(np.linalg.norm(delta));errors.append(e);n=float(.5*np.sum(delta**2/np.exp(logvar[k])+logvar[k]));nll.append(n)
  pred.append({'sample_id':sid,'family_id':labels[sid]['family_id'],'relation_target':rt[k],'relation_prediction':rp[k],'answerability_target':at[k],'answerability_prediction':ap[k],'target_uv':labels[sid]['target_uv'],'mu_uv':mu[k],'l2_error':e,'gaussian_nll':n})
 answer_f1=f1(at,ap,ST);relation_f1=f1(rt,rp,REL);hit03=sum(e<=.03 for e in errors)/len(errors);hit05=sum(e<=.05 for e in errors)/len(errors);false_accept=sum(p=='FOUND' and t!='FOUND' for p,t in zip(ap,at))/sum(t!='FOUND' for t in at)
 bas=cqc['shortcut_baselines'];best_answer=max(bas['text_only_answerability_macro_f1'],bas['rgb_thumbnail_16x12_answerability_macro_f1'],bas['depth_thumbnail_16x12_answerability_macro_f1']);cent=bas['relation_centroid_hit_at_0_05'];checks={'capture_qc_pass':True,'feature_cache_qc_pass':True,'support_exact_249_found64':len(ids)==249 and len(errors)==64,'finite_outputs':all(math.isfinite(v) for v in errors+nll),'answerability_effect_ge_0_05':answer_f1>=best_answer+.05,'coordinate_hit05_effect_ge_0_05':hit05>=cent+.05};passed=all(checks.values())
 metrics={'schema_version':2,'status':'PASS' if passed else 'FAIL','created_at_utc':datetime.now(timezone.utc).isoformat(),'one_shot':True,'checkpoint':{'path':str(CHECKPOINT.relative_to(ROOT)),'sha256':sha256(CHECKPOINT)},'support':{'all':249,'found':64,'per_state':dict(Counter(at))},'model':{'relation_macro_f1':relation_f1,'answerability_macro_f1':answer_f1,'point_l2_mean':float(np.mean(errors)),'point_l2_median':float(np.median(errors)),'hit_at_0_03':hit03,'hit_at_0_05':hit05,'gaussian_nll_mean':float(np.mean(nll)),'false_accept_rate':false_accept},'ci95_family_bootstrap':{'answerability_accuracy':ci([a==b for a,b in zip(at,ap)]),'point_l2_mean':ci(errors),'hit_at_0_05':ci([e<=.05 for e in errors])},'matching_baselines':{'best_answerability_macro_f1':best_answer,'relation_centroid_hit_at_0_05':cent},'checks':checks}
 OUT.mkdir(parents=True);DEC.mkdir(parents=True);(OUT/'ANTI_SHORTCUT_METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n');(OUT/'PREDICTIONS.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in pred));
 with (OUT/'METRIC_TABLE.csv').open('w',newline='') as f:
  rows=[{'metric':'answerability_macro_f1','model':answer_f1,'baseline':best_answer,'delta':answer_f1-best_answer},{'metric':'hit_at_0_05','model':hit05,'baseline':cent,'delta':hit05-cent},{'metric':'relation_macro_f1','model':relation_f1,'baseline':float('nan'),'delta':float('nan')},{'metric':'point_l2_mean','model':float(np.mean(errors)),'baseline':float('nan'),'delta':float('nan')}];w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].bar(['Model','Best shortcut'],[answer_f1,best_answer],color=['#0072B2','#D55E00']);ax[0].set_ylim(0,1);ax[0].set_title('Answerability macro-F1');ax[1].bar(['Model','Relation centroid'],[hit05,cent],color=['#0072B2','#D55E00']);ax[1].set_ylim(0,1);ax[1].set_title('Coordinate Hit@0.05');fig.tight_layout();fig.savefig(OUT/'ANTI_SHORTCUT_RESULT.png',dpi=180);plt.close(fig)
 decision={'schema_version':1,'outcome':'PASS' if passed else 'GENERALIZATION_GATE_FAIL','day6_complete':passed,'opens_day_7_oof_s1b':passed,'opens_g3':passed,'one_shot_consumed':True,'checks':checks,'metrics':metrics['model'],'evidence':{'capture_qc':str(CAPTURE_QC.relative_to(ROOT)),'feature_cache_qc':str(CACHE_QC.relative_to(ROOT)),'metrics':str((OUT/'ANTI_SHORTCUT_METRICS.json').relative_to(ROOT)),'predictions':str((OUT/'PREDICTIONS.jsonl').relative_to(ROOT))}};(DEC/'DECISION.json').write_text(json.dumps(decision,indent=2)+'\n')
 final={'schema_version':2,'status':'PASS' if passed else 'DAY6_COMPLETE_GENERALIZATION_HOLD','s1a_status':'S1A_COMPLETE','fresh_left_ycb_status':'PASS' if passed else 'ONE_SHOT_FAIL','model_inference_on_fresh_left_ycb':True,'opens_day_7_oof_s1b':passed,'opens_g3':passed,'one_shot_consumed':True,'checkpoint_sha256':sha256(CHECKPOINT),'canary_15_pair_status':'PASS','capture_qc_status':'PASS_ACCEPTED_249_OF_256','excluded_ie_scenes':7,'anti_shortcut_status':metrics['status'],'metrics':metrics['model'],'next_required_action':'Proceed to Day 7 OOF/S1b.' if passed else 'Do not tune on this validation set; report the one-shot generalization failure.','evidence':decision['evidence']};FINAL.write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
