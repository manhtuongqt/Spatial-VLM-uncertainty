#!/usr/bin/env python3
"""Combine v8 and targeted v9 measurements into the final 15-pair lock."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib,json
from pathlib import Path
import cv2,yaml
ROOT=Path(__file__).resolve().parents[2]; V8=ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v8'; V9=ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v9_targeted'
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def collect(root,attempt):
 ann=yaml.safe_load((root/'annotations.yaml').read_text())['scenes']; cap=root/attempt; out=defaultdict(list)
 for line in (cap/'input_manifest.jsonl').read_text().splitlines():
  row=json.loads(line);sid=row['scene_id'];a=ann[sid];c=a['projection_calibration'];p=cap/sid/'evaluator/semantic_labels.png';im=cv2.imread(str(p),-1);n=int((im==int(a['target_label'])).sum())
  out[(c['box'],c['fruit'])].append({'source':root.name,'scene_id':sid,'signed_lateral_offset_m':float(c['signed_lateral_offset_m']),'forward_separation_m':float(c['forward_separation_m']),'target_visible_pixels':n,'pass_1_to_119':1<=n<120,'semantic_labels_sha256':sha256(p)})
 return out
def main():
 allobs=collect(V8,'capture_attempt_01'); newer=collect(V9,'capture_attempt_01')
 for k,v in newer.items():allobs[k].extend(v)
 results=[];pairs={}
 for (box,fruit),vals in sorted(allobs.items()):
  valid=[x for x in vals if x['pass_1_to_119']]; choice=min(valid,key=lambda x:(abs(x['target_visible_pixels']-60),x['source']!='anti_shortcut_left_ycb_canary_v9_targeted',abs(x['signed_lateral_offset_m']))) if valid else None
  if choice:pairs.setdefault(box,{})[fruit]=choice
  results.append({'box':box,'fruit':fruit,'observations':len(vals),'valid_count':len(valid),'selected':choice})
 passed=len(results)==15 and all(x['valid_count'] for x in results); qc={'schema_version':1,'status':'PASS' if passed else 'FAIL','created_at_utc':datetime.now(timezone.utc).isoformat(),'world_revision':'v5_contact_exact','lemon_excluded':True,'pair_count':len(results),'passing_pair_count':sum(bool(x['valid_count']) for x in results),'pass_rule':'all 15 pairs have fresh v5 observation with 1<=target_visible_pixels<120','pair_results':results,'source_qc_v8_sha256':sha256(V8/'CANARY_QC.json'),'capture_v9_sha256':sha256(V9/'capture_attempt_01/capture_manifest.json')}
 q=V9/'COMBINED_15_PAIR_CANARY_QC.json';q.write_text(json.dumps(qc,indent=2)+'\n');(V9/'SELECTED_15_PAIR_GEOMETRY.json').write_text(json.dumps({'schema_version':1,'status':'LOCKED' if passed else 'INCOMPLETE','pairs':pairs,'combined_qc_sha256':sha256(q)},indent=2)+'\n');print(json.dumps({'status':qc['status'],'passing_pairs':qc['passing_pair_count'],'pairs':len(results)},indent=2));raise SystemExit(0 if passed else 2)
if __name__=='__main__':main()
