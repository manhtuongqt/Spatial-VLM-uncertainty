#!/usr/bin/env python3
"""Mesh-offset projection calibration and collision-isolated Day-8 revision 4."""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from collections import Counter
from pathlib import Path
import cv2, numpy as np, yaml

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import workspace.mh_pcrau_v3.day8_development_r2_prepare as base
import workspace.mh_pcrau_v3.day8_development_r2_prepare_v3 as v3

OUT=base.RAW/'canary_v4'; CAL=base.REPORT/'PROJECTION_CALIBRATION_V4.json'; FRUITS=base.FRUITS

def fit_projection()->dict:
 root=base.RAW/'canary_v3';anns=yaml.safe_load((root/'annotations.yaml').read_text())['scenes'];scenes={x['scene_id']:x for x in yaml.safe_load((root/'scenes.yaml').read_text())['scenes']};cap=root/'capture_attempt_01'; rows=[];obs=[]
 for sid in anns:
  lab=cv2.imread(str(cap/sid/'evaluator/semantic_labels.png'),cv2.IMREAD_UNCHANGED)
  for fruit in FRUITS:
   if fruit not in scenes[sid]['poses']:continue
   yy,xx=np.where(lab==base.FRUIT_LABELS[fruit])
   if len(xx)<120:continue
   x,y,_=scenes[sid]['poses'][fruit];row=[x,y,1.]+[1. if fruit==f else 0. for f in FRUITS[1:]];rows.append(row);obs.append(float(xx.mean()))
 X=np.asarray(rows,float);Y=np.asarray(obs,float);ridge=np.diag([1e-8,1e-8,1e-8]+[1e-3]*4);w=np.linalg.solve(X.T@X+ridge,X.T@Y);pred=X@w
 return {'schema_version':1,'source':'canary_v3 semantic centroids','observations':len(Y),'coefficients':{'x':float(w[0]),'y':float(w[1]),'constant':float(w[2]),'fruit_offsets':{FRUITS[0]:0.,**{f:float(w[3+i]) for i,f in enumerate(FRUITS[1:])}}},'rmse_pixels':float(np.sqrt(np.mean((pred-Y)**2)))}

def y_for(cal:dict,fruit:str,x:float,u:float)->float:
 c=cal['coefficients'];return float((u-c['x']*x-c['constant']-c['fruit_offsets'][fruit])/c['y'])

def build(stage,state,relation,repeat,geometry,cal):
 scene,ann,index=v3.build_scene(stage,state,relation,repeat,geometry);design=base.common_design(stage,relation,repeat);fruits=design['fruits'];target_i=base.rank_index(relation);target=fruits[target_i];us=v3.desired_u(stage,relation,repeat)
 xs=[-.08,-.05,-.02]
 if state=='AMBIGUOUS':
  ties=base.tie_indices(relation);tie=us[target_i];us[ties[0]]=tie;us[ties[1]]=tie;xs[ties[0]],xs[ties[1]]=-.12,.03
 for i,fruit in enumerate(fruits):
  if state=='ABSENT' and i==target_i:scene['poses'].pop(fruit,None)
  else:scene['poses'][fruit]=[xs[i],y_for(cal,fruit,xs[i],us[i]),random.Random(int(hashlib.sha256(f'{stage}|{state}|{relation}|{repeat}|{fruit}'.encode()).hexdigest()[:12],16)).uniform(-.3,.3)]
 covered=target if state=='INSUFFICIENT_EVIDENCE' else design['decoy'];occluder=design['occluder'];g=geometry['pairs'][occluder][covered];tx=-.10 if occluder=='ycb_sugar_box' and covered=='ycb_apple' else -.24
 scene['poses'][covered]=[tx,.38,0.];scene['poses'][occluder]=[tx-float(g['forward_separation_m']),.38+float(g['signed_lateral_offset_m']),float(g.get('occluder_yaw_rad',0.))]
 # Keep all other active objects at least 0.14 m from the calibrated pair.
 safe=[(-.50,.82),(-.43,.22),(-.32,.88),(-.18,.18),(-.04,.78),(-.49,.52)]
 for name,(x,y) in zip(base.CLUTTER,safe):scene['poses'][name]=[x,y,0.]
 sig=hashlib.sha256(json.dumps(scene['poses'],sort_keys=True,separators=(',',':')).encode()).hexdigest();scene['layout_signature_sha256']=sig;ann['layout_signature_sha256']=sig;index['layout_signature_sha256']=sig
 return scene,ann,index

def add_lock(folder,cal):
 p=folder/'CAPTURE_SOURCE_LOCK.json';lock=json.loads(p.read_text());rel=str(Path(__file__).relative_to(ROOT));lock['source_artifact_sha256'][rel]=base.sha256(Path(__file__));lock['source_artifact_sha256'][str(CAL.relative_to(ROOT))]=base.sha256(CAL);lock['revision_reason']='Mesh-specific image-plane offsets plus collision isolation after preserved canary-v1/v2/v3 failures.';lock['projection_calibration_sha256']=base.sha256(CAL);p.write_text(json.dumps(lock,indent=2)+'\n')

def main(stage):
 cal=fit_projection()
 if not CAL.exists():CAL.write_text(json.dumps(cal,indent=2)+'\n')
 elif json.loads(CAL.read_text())!=cal:raise RuntimeError('projection calibration drift')
 geometry=json.loads(base.GEOMETRY.read_text());ss=yaml.safe_load(base.SOURCE_SCENES.read_text());sa=yaml.safe_load(base.SOURCE_ANNOTATIONS.read_text());sg=yaml.safe_load(base.SOURCE_GATE.read_text())
 if stage=='canary':
  if OUT.exists():raise FileExistsError(OUT)
  if json.loads((base.REPORT/'CANARY_QC.json').read_text()).get('status')!='FAIL':raise RuntimeError('v4 requires v3 FAIL')
  rows=[];anns={};index=[]
  for state in base.STATES:
   for ri,relation in enumerate(base.RELATIONS):
    row,ann,item=build('canary_v4',state,relation,ri,geometry,cal);rows.append(row);anns[row['scene_id']]=ann;index.append(item)
  base.write_batch(OUT,'canary_v4',base.CAMERAS[0],rows,anns,ss,sa,sg);add_lock(OUT,cal);(OUT/'DESIGN_INDEX.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in index));print(json.dumps({'status':'CANARY_V4_READY','families':16,'calibration_rmse_pixels':cal['rmse_pixels']},indent=2));return
 if json.loads((base.REPORT/'CANARY_QC.json').read_text()).get('status')!='PASS':raise RuntimeError('bulk sealed')
 bulk=base.RAW/'bulk'
 if bulk.exists():raise FileExistsError(bulk)
 bulk.mkdir();batches={i:([],{},[]) for i in range(4)};all_index=[]
 for state in base.STATES:
  for relation in base.RELATIONS:
   for repeat in range(32):
    row,ann,item=build('bulk',state,relation,repeat,geometry,cal);b=repeat%4;batches[b][0].append(row);batches[b][1][row['scene_id']]=ann;batches[b][2].append(item);all_index.append(item)
 for b,(rows,anns,index) in batches.items():
  random.Random(base.SEED+b).shuffle(rows);folder=bulk/f'batch_{b}';base.write_batch(folder,f'bulk_v4_b{b}',base.CAMERAS[b],rows,anns,ss,sa,sg);add_lock(folder,cal);(folder/'DESIGN_INDEX.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in index))
 (bulk/'DESIGN_INDEX.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in all_index));cells=Counter((x['state'],x['relation']) for x in all_index);summary={'schema_version':1,'status':'PASS_CAPTURE_NOT_STARTED','families':len(all_index),'states':dict(Counter(x['state'] for x in all_index)),'relations':dict(Counter(x['relation'] for x in all_index)),'splits':dict(Counter(x['split'] for x in all_index)),'cells':{f'{s}|{r}':cells[(s,r)] for s in base.STATES for r in base.RELATIONS},'unique_family_ids':len({x['family_id'] for x in all_index}),'unique_seeds':len({x['seed'] for x in all_index}),'unique_layouts':len({x['layout_signature_sha256'] for x in all_index})};(bulk/'DESIGN_STATIC_QC.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('stage',choices=('canary','bulk'));main(p.parse_args().stage)
