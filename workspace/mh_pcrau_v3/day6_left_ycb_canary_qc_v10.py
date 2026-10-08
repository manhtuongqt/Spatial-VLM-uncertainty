#!/usr/bin/env python3
"""Merge v8/v9/v10 into the definitive 15-pair geometry lock."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import cv2,yaml
ROOT=Path(__file__).resolve().parents[2];DIRS=[ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v8',ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v9_targeted',ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v10_sugar',ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v13_sugar_apple_near'];OUT=DIRS[-1]
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 obs=defaultdict(list)
 for root in DIRS:
  ann=yaml.safe_load((root/'annotations.yaml').read_text())['scenes'];cap=root/'capture_attempt_01'
  for line in (cap/'input_manifest.jsonl').read_text().splitlines():
   row=json.loads(line);a=ann[row['scene_id']];c=a['projection_calibration'];p=cap/row['scene_id']/'evaluator/semantic_labels.png';im=cv2.imread(str(p),-1);n=int((im==int(a['target_label'])).sum());obs[(c['box'],c['fruit'])].append({'source':root.name,'scene_id':row['scene_id'],'signed_lateral_offset_m':float(c['signed_lateral_offset_m']),'forward_separation_m':float(c['forward_separation_m']),'occluder_yaw_rad':float(c.get('occluder_yaw_rad',0.)),'target_visible_pixels':n,'pass_1_to_119':1<=n<120,'semantic_labels_sha256':sha256(p)})
 pairs={};results=[]
 for (box,fruit),vals in sorted(obs.items()):
  valid=[v for v in vals if v['pass_1_to_119']];choice=min(valid,key=lambda v:(abs(v['target_visible_pixels']-60),-DIRS.index(ROOT/'ketqua1/09_danh_gia'/v['source']))) if valid else None
  if choice:pairs.setdefault(box,{})[fruit]=choice
  results.append({'box':box,'fruit':fruit,'observations':len(vals),'valid_count':len(valid),'selected':choice})
 passed=len(results)==15 and all(r['valid_count'] for r in results);qc={'schema_version':1,'status':'PASS' if passed else 'FAIL','created_at_utc':datetime.now(timezone.utc).isoformat(),'world_revision':'v5_contact_exact','lemon_excluded':True,'pair_count':len(results),'passing_pair_count':sum(bool(r['valid_count']) for r in results),'pass_rule':'all 15 pairs have 1<=target_visible_pixels<120','pair_results':results,'capture_manifest_hashes':{d.name:sha256(d/'capture_attempt_01/capture_manifest.json') for d in DIRS}}
 q=OUT/'FINAL_15_PAIR_CANARY_QC.json';q.write_text(json.dumps(qc,indent=2)+'\n');(OUT/'SELECTED_15_PAIR_GEOMETRY.json').write_text(json.dumps({'schema_version':1,'status':'LOCKED' if passed else 'INCOMPLETE','pairs':pairs,'canary_qc_sha256':sha256(q)},indent=2)+'\n');print(json.dumps({'status':qc['status'],'passing_pairs':qc['passing_pair_count']},indent=2));raise SystemExit(0 if passed else 2)
if __name__=='__main__':main()
