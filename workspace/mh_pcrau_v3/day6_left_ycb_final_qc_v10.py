#!/usr/bin/env python3
"""QC/materialize the accepted 249-scene fresh Day-6 set."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import cv2,numpy as np,yaml,re

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_final_v11';REPAIR=ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_repair_v12';OUT=ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_final_v14_accepted249';MAT=ROOT/'ketqua1/03_backbone_h_spatial/ngay_06/amendment_07_fresh_left_ycb_v14_accepted249';STATES=('FOUND','AMBIGUOUS','ABSENT','INSUFFICIENT_EVIDENCE');RELATIONS=('leftmost','rightmost','second_from_left','second_from_right')
REPAIR_REJECTED={'repair_101ffd3c813a7c34','repair_dfdf987cbad760e1','repair_7d3d34c0c723926b','repair_3cd49fcf81a0a53b','repair_1602ad1cab7bc2fb','repair_6545ce43d05926c2','repair_7854036868c291b0'}
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def dview(x):
 x=np.asarray(x,np.float32);valid=np.isfinite(x)&(x>=.1)&(x<=3.);near,far=np.percentile(x[valid],[2,98]);inv=1/np.maximum(np.clip(x,near,far),1e-6);g=np.clip((inv-1/far)/max(1/near-1/far,1e-6)*255,0,255).astype(np.uint8);g[~valid]=0;return np.repeat(g[...,None],3,axis=2)
def thumb(path,depth=False):
 x=np.load(path).astype(np.float32) if depth else cv2.imread(str(path),cv2.IMREAD_COLOR).astype(np.float32)/255
 if depth:
  valid=np.isfinite(x);fill=float(np.median(x[valid]));x=np.where(valid,x,fill);x=(x-x.min())/max(float(x.max()-x.min()),1e-6);x=np.repeat(x[...,None],3,axis=2)
 return cv2.resize(x,(16,12),interpolation=cv2.INTER_AREA).reshape(-1)
def phash(path):
 x=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE);x=cv2.resize(x,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32);d=cv2.dct(x)[:8,:8];return (d>d.mean()).reshape(-1)
def macro(y,p):
 values=[]
 for label in STATES:
  tp=sum(a==label and b==label for a,b in zip(y,p));fp=sum(a!=label and b==label for a,b in zip(y,p));fn=sum(a==label and b!=label for a,b in zip(y,p));values.append(0. if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
 return float(np.mean(values))
def ridge_predict(xtr,ytr,xv):
 mean=xtr.mean(0);std=xtr.std(0)+1e-6;xtr=(xtr-mean)/std;xv=(xv-mean)/std;classes=list(STATES);Y=np.eye(len(classes))[[classes.index(x) for x in ytr]];w=np.linalg.solve(xtr.T@xtr+10*np.eye(xtr.shape[1]),xtr.T@Y);return [classes[i] for i in (xv@w).argmax(1)]
def text_vector(rows,vocab=None):
 toks=[[x for x in re.findall(r'[a-z]+',s.lower())] for s in rows]
 if vocab is None:vocab={x:i for i,x in enumerate(sorted({t for row in toks for t in row}))}
 out=np.zeros((len(rows),len(vocab)),np.float64)
 for i,row in enumerate(toks):
  for t in row:
   if t in vocab:out[i,vocab[t]]+=1
 return out,vocab
def main():
 if MAT.exists() or (OUT/'CAPTURE_QC.json').exists():raise FileExistsError('append-only QC/materialization exists')
 OUT.mkdir(parents=True,exist_ok=True)
 records=[];state=Counter();relation=Counter();cells=Counter();rgb_hash=[];ph=[];semantic_fail=[]
 failed_base={x.split(':',1)[0] for x in json.loads((BASE/'CAPTURE_QC.json').read_text())['semantic_failures']}
 sources=[]
 for batch in range(4):
  b=BASE/f'batch_{batch}';cap=b/'capture_attempt_01';anns=yaml.safe_load((b/'annotations.yaml').read_text())['scenes'];manifest=[json.loads(x) for x in (cap/'input_manifest.jsonl').read_text().splitlines() if x]
  if len(manifest)!=64:raise RuntimeError(f'batch {batch}: {len(manifest)} captures')
  for row in manifest:
   if row['scene_id'] not in failed_base:sources.append((f'base_batch_{batch}',cap,anns,row))
 rcap=REPAIR/'capture_attempt_01';ranns=yaml.safe_load((REPAIR/'annotations.yaml').read_text())['scenes'];rmanifest=[json.loads(x) for x in (rcap/'input_manifest.jsonl').read_text().splitlines() if x]
 for row in rmanifest:
  if row['scene_id'] not in REPAIR_REJECTED:sources.append(('repair_v12',rcap,ranns,row))
 if len(sources)!=249:raise RuntimeError(f'expected 249 accepted captures, got {len(sources)}')
 for batch,cap,anns,row in sources:
   sid=row['scene_id'];a=anns[sid];rgb=cap/sid/'input/rgb.png';depth=cap/sid/'input/depth_m.npy';labels_path=cap/sid/'evaluator/semantic_labels.png';im=cv2.imread(str(rgb));dep=np.load(depth);lab=cv2.imread(str(labels_path),cv2.IMREAD_UNCHANGED)
   if im is None or lab is None or im.shape[:2]!=(480,640) or dep.shape!=(480,640) or int((np.isfinite(dep)&(dep>=.1)&(dep<=3.)).sum())<250000:semantic_fail.append(f'{sid}:sensor')
   st=a['state'];rel=a['relation_variant'];target_uv=None;visible=None
   if st=='FOUND':
    yy,xx=np.where(lab==int(a['target_label']));visible=len(xx)
    if visible<120:semantic_fail.append(f'{sid}:found_pixels={visible}')
    else:target_uv=[float(xx.mean()/639),float(yy.mean()/479)]
   elif st=='ABSENT':
    visible=int((lab==int(a['target_label'])).sum())
    if visible!=0:semantic_fail.append(f'{sid}:absent_pixels={visible}')
   elif st=='INSUFFICIENT_EVIDENCE':
    visible=int((lab==int(a['target_label'])).sum());occ=int((lab==int(a['occluder_labels'][0])).sum())
    if not (1<=visible<120 and occ>0):semantic_fail.append(f'{sid}:ie_pixels={visible},occ={occ}')
   else:
    centers=[]
    for label in a['valid_target_labels']:
     yy,xx=np.where(lab==int(label));
     if len(xx):centers.append(float(xx.mean()))
    if len(centers)!=2 or abs(centers[0]-centers[1])>8:semantic_fail.append(f'{sid}:ambiguous_centers={centers}')
   rec={'sample_id':sid,'family_id':a['family_id'],'batch':batch,'state':st,'relation':rel,'instruction':row['instruction'],'rgb':rgb,'depth':depth,'labels':labels_path,'target_uv':target_uv,'target_visible_pixels':visible}
   records.append(rec);state[st]+=1;relation[rel]+=1;cells[(st,rel)]+=1;rgb_hash.append(sha256(rgb));ph.append(phash(rgb))
 # historical exact-hash freshness
 historical=set()
 for p in ROOT.glob('ketqua1/**/**/*INPUT_MANIFEST*.jsonl'):
  try:
   for line in p.read_text().splitlines():
    x=json.loads(line)
    if 'rgb_sha256' in x:historical.add(x['rgb_sha256'])
  except Exception:pass
 overlap=len(set(rgb_hash)&historical);min_phash=min(int(np.count_nonzero(ph[i]!=ph[j])) for i in range(len(ph)) for j in range(i))
 # Deterministic approximately 3/1 split within each available state/relation cell.
 train=[];test=[]
 for st in STATES:
  for rel in RELATIONS:
   ids=sorted([i for i,r in enumerate(records) if r['state']==st and r['relation']==rel],key=lambda i:hashlib.sha256(records[i]['sample_id'].encode()).hexdigest());cut=max(1,int(round(.75*len(ids))));train+=ids[:cut];test+=ids[cut:]
 yt=[records[i]['state'] for i in train];yv=[records[i]['state'] for i in test]
 xt,vocab=text_vector([records[i]['instruction'] for i in train]);xv,_=text_vector([records[i]['instruction'] for i in test],vocab);text_f1=macro(yv,ridge_predict(xt,yt,xv))
 def media_probe(key,depth=False):
  xtr=np.stack([thumb(records[i][key],depth) for i in train]);xv=np.stack([thumb(records[i][key],depth) for i in test]);return macro(yv,ridge_predict(xtr,yt,xv))
 rgb_f1=media_probe('rgb');depth_f1=media_probe('depth',True)
 found=[i for i,r in enumerate(records) if r['state']=='FOUND' and r['target_uv'] is not None];centers={rel:np.mean([records[i]['target_uv'] for i in found[:48] if records[i]['relation']==rel],axis=0) for rel in RELATIONS};cent_errors=[math.dist(records[i]['target_uv'],centers[records[i]['relation']]) for i in found[48:] if np.all(np.isfinite(centers[records[i]['relation']]))];cent_hit=sum(e<=.05 for e in cent_errors)/max(len(cent_errors),1)
 checks={'families_exact_249':len(records)==249 and len({r['family_id'] for r in records})==249,'accepted_state_support':state==Counter({'FOUND':64,'AMBIGUOUS':64,'ABSENT':64,'INSUFFICIENT_EVIDENCE':57}),'semantic_and_sensor_qc_pass_all_accepted':not semantic_fail,'rgb_hash_unique_249':len(set(rgb_hash))==249,'historical_rgb_overlap_zero':overlap==0,'text_only_f1_le_0_35':text_f1<=.35,'rgb_thumbnail_f1_le_0_70':rgb_f1<=.70,'depth_thumbnail_f1_le_0_70':depth_f1<=.70}
 qc={'schema_version':2,'status':'PASS' if all(checks.values()) else 'FAIL','created_at_utc':datetime.now(timezone.utc).isoformat(),'acceptance_policy':'User-approved 249/256 semantic-QC subset; seven remaining IE failures excluded before model inference. The centroid shortcut score is retained as the one-shot comparison baseline, not a capture rejection gate.','excluded_scene_ids':sorted(REPAIR_REJECTED),'checks':checks,'warnings':{'relation_centroid_hit05_above_0_60':cent_hit>.60},'counts':{'families':len(records),'states':dict(state),'relations':dict(relation),'cells':{f'{s}|{r}':cells[(s,r)] for s in STATES for r in RELATIONS}},'semantic_failures':semantic_fail,'freshness':{'historical_rgb_overlap':overlap,'unique_rgb':len(set(rgb_hash)),'minimum_phash_distance':min_phash},'shortcut_baselines':{'text_only_answerability_macro_f1':text_f1,'rgb_thumbnail_16x12_answerability_macro_f1':rgb_f1,'depth_thumbnail_16x12_answerability_macro_f1':depth_f1,'relation_centroid_hit_at_0_05':cent_hit}}
 q=OUT/'CAPTURE_QC.json';q.write_text(json.dumps(qc,indent=2)+'\n')
 if qc['status']!='PASS':print(json.dumps(qc,indent=2));raise SystemExit(2)
 MAT.mkdir(parents=True);dv=MAT/'depth_view';dv.mkdir();inputs=[];sup=[]
 for r in sorted(records,key=lambda x:x['sample_id']):
  view=dv/f"{r['sample_id']}.png";cv2.imwrite(str(view),dview(np.load(r['depth'])));inputs.append({'sample_id':r['sample_id'],'family_id':r['family_id'],'split':'anti_shortcut_val_v1_accepted249','rgb_path':str(r['rgb'].relative_to(ROOT)),'rgb_sha256':sha256(r['rgb']),'depth_view_path':str(view.relative_to(ROOT)),'depth_view_sha256':sha256(view),'metric_depth_path':str(r['depth'].relative_to(ROOT)),'metric_depth_sha256':sha256(r['depth']),'instruction':r['instruction'],'instruction_sha256':digest(r['instruction'])});sup.append({'sample_id':r['sample_id'],'family_id':r['family_id'],'relation':r['relation'],'answerability':r['state'],'target_uv':r['target_uv'],'head_mask':{'relation':True,'answerability':True,'coordinate':r['state']=='FOUND','log_variance':r['state']=='FOUND','reasoning':False,'source':False,'confidence':False},'label_source':'fresh world-v5 semantic evaluator; user-approved 249/256 QC subset'})
 ip=MAT/'INPUT_MANIFEST.jsonl';sp=MAT/'SUPERVISION.jsonl';ip.write_text(''.join(json.dumps(x)+'\n' for x in inputs));sp.write_text(''.join(json.dumps(x)+'\n' for x in sup));lock={'schema_version':2,'status':'FROZEN_BEFORE_ONE_SHOT_MODEL_INFERENCE','created_at_utc':datetime.now(timezone.utc).isoformat(),'samples':249,'excluded_after_qc':7,'capture_qc':{'path':str(q.relative_to(ROOT)),'sha256':sha256(q)},'input_manifest':{'path':str(ip.relative_to(ROOT)),'sha256':sha256(ip)},'supervision':{'path':str(sp.relative_to(ROOT)),'sha256':sha256(sp)},'model_input_allowlist':['rgb_path','depth_view_path','instruction'],'oracle_passed_to_model':False};(MAT/'ONE_SHOT_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n');print(json.dumps({'status':'PASS','families':249,'baselines':qc['shortcut_baselines']},indent=2))
if __name__=='__main__':main()
