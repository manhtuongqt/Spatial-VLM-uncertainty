#!/usr/bin/env python3
"""Extract sealed h_spatial features for fresh v10 exactly once."""
from __future__ import annotations
from datetime import datetime,timezone
import csv,json,os,statistics,time
from pathlib import Path
from workspace.mh_pcrau_v3.day6_paired_view_cache import digest,load_model,sha256,tensor_hash
ROOT=Path(__file__).resolve().parents[2];VLM=ROOT/'RoboRefer';OUT=ROOT/'ketqua1/03_backbone_h_spatial/ngay_06/amendment_07_fresh_left_ycb_v14_accepted249';INPUT=OUT/'INPUT_MANIFEST.jsonl';LOCK=OUT/'ONE_SHOT_LOCK.json';CACHE=OUT/'cache';MANIFEST=OUT/'FEATURE_CACHE_MANIFEST.json';QC=OUT/'FEATURE_CACHE_QC.json';TIMING=OUT/'FEATURE_CACHE_TIMING.csv';CANON=ROOT/'ketqua1/03_backbone_h_spatial/ngay_06/S1A_FEATURE_CACHE_MANIFEST.json';CHECKPOINT=ROOT/'workspace/mh_pcrau_v3/checkpoints/day06_robust_candidate_r2/s1a_robust_fixed_epoch54.pt'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
def main():
 if any(p.exists() for p in (CACHE,MANIFEST,QC,TIMING)):raise FileExistsError('append-only feature cache exists')
 if os.environ.get('PYTHONNOUSERSITE')!='1':raise RuntimeError('use isolated environment')
 import torch
 from workspace.mh_pcrau_v3.hidden_state_adapter import extract_h_spatial,prepare_rgbd_prompt
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 lock=json.loads(LOCK.read_text());
 if lock['status']!='FROZEN_BEFORE_ONE_SHOT_MODEL_INFERENCE' or sha256(INPUT)!=lock['input_manifest']['sha256']:raise RuntimeError('one-shot lock invalid')
 rows=[json.loads(x) for x in INPUT.read_text().splitlines() if x]
 if len(rows)!=249:raise RuntimeError('expected 249 accepted inputs')
 for r in rows:
  for key in ('rgb','depth_view','metric_depth'):
   if sha256(ROOT/r[f'{key}_path'])!=r[f'{key}_sha256']:raise RuntimeError(f"media drift {r['sample_id']}/{key}")
  if digest(r['instruction'])!=r['instruction_sha256']:raise RuntimeError('instruction drift')
 torch.manual_seed(25092052);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 started=time.perf_counter();model=load_model(torch);load_seconds=time.perf_counter()-started;canonical=json.loads(CANON.read_text());
 if digest(model.tokenizer.get_vocab())!=canonical['tokenizer_vocab_sha256'] or digest(model.tokenizer.chat_template)!=canonical['chat_template_sha256']:raise RuntimeError('tokenizer drift')
 CACHE.mkdir();records=[];timings=[];seconds=[];torch.cuda.reset_peak_memory_stats()
 for i,r in enumerate(rows,1):
  prompt=prepare_rgbd_prompt(model,rgb_path=ROOT/r['rgb_path'],depth_path=ROOT/r['depth_view_path'],instruction=r['instruction']);mask=torch.ones_like(prompt.input_ids,dtype=torch.bool);torch.cuda.synchronize();t=time.perf_counter();result=extract_h_spatial(model,prompt.input_ids.unsqueeze(0),prompt.media,mask.unsqueeze(0),prompt.media_config);torch.cuda.synchronize();elapsed=time.perf_counter()-t;feature=result.h_spatial[0].detach().cpu().contiguous()
  if feature.shape!=(1536,) or feature.dtype!=torch.float32 or not bool(torch.isfinite(feature).all()):raise RuntimeError('feature contract')
  p=CACHE/f"{r['sample_id']}.pt";torch.save(feature,p);restored=torch.load(p,map_location='cpu',weights_only=True);records.append({'sample_id':r['sample_id'],'family_id':r['family_id'],'feature_path':str(p.relative_to(ROOT)),'feature_file_sha256':sha256(p),'feature_raw_sha256':tensor_hash(feature),'shape':[1536],'dtype':'torch.float32','finite_values':1536,'roundtrip_max_abs':float((restored-feature).abs().max()),'prompt_render_sha256':digest(prompt.prompt_text),'input_token_ids_sha256':digest(prompt.input_ids.tolist()),'last_prompt_index':int(result.last_prompt_indices[0])});timings.append({'sample_id':r['sample_id'],'seconds':f'{elapsed:.9f}'});seconds.append(elapsed)
  if i%8==0:print(f'FRESH_ACCEPTED249_CACHE_PROGRESS {i}/249',flush=True)
 with TIMING.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['sample_id','seconds']);w.writeheader();w.writerows(timings)
 man={'schema_version':1,'status':'COMPLETE','created_at_utc':datetime.now(timezone.utc).isoformat(),'one_shot_lock':{'path':str(LOCK.relative_to(ROOT)),'sha256':sha256(LOCK)},'frozen_checkpoint':{'path':str(CHECKPOINT.relative_to(ROOT)),'sha256':sha256(CHECKPOINT)},'model_checkpoint':'RoboRefer/models/RoboRefer-2B-SFT','model_inventory_sha256':canonical['model_inventory_sha256'],'model_load_seconds':load_seconds,'records':records,'labels_in_feature_shards':False,'oracle_fields_passed_to_model':False};MANIFEST.write_text(json.dumps(man,indent=2)+'\n');checks={'records_249':len(records)==249,'families_249':len({r['family_id'] for r in records})==249,'hashes_unique_249':len({r['feature_raw_sha256'] for r in records})==249,'shape_dtype_finite':all(r['shape']==[1536] and r['finite_values']==1536 for r in records),'roundtrip_exact':all(r['roundtrip_max_abs']==0 for r in records)};qc={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'seconds_median':statistics.median(seconds),'seconds_mean':statistics.mean(seconds),'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'manifest_sha256':sha256(MANIFEST)};QC.write_text(json.dumps(qc,indent=2)+'\n');print(json.dumps(qc,indent=2));raise SystemExit(0 if qc['status']=='PASS' else 2)
if __name__=='__main__':main()
