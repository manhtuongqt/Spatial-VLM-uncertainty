#!/usr/bin/env python3
"""Compute real five-source MC estimates on train/dev; labels only in audit."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from safetensors.torch import save_file
from pcrau.anchor_shadow_experiment import state_digest
from pcrau.calibration import expected_calibration_error
from pcrau.dataset import ArchivedPCRAUDataset,SOURCE_CLASSES
from pcrau.source_mc import (VERSION,FIELDS,FiveSourceHeadMC,predict_with_source_mc,source_statistics)
from pcrau.unified_inference import UnifiedInference,load_features
from pcrau.utils import read_json,atomic_json,sha256_file,workspace_path,seed_everything,runtime_info


def write_rows(path,rows):
    with path.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def binary_metrics(scores,labels):
    p=np.asarray(scores);y=np.asarray(labels,dtype=bool);pred=p>=.5
    tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum())
    c=p.clip(1e-7,1-1e-7)
    return {'samples':len(y),'positive_labels':int(y.sum()),'threshold_fixed':.5,
        'accuracy':float((pred==y).mean()),'f1':2*tp/max(1,2*tp+fp+fn),'tp':tp,'fp':fp,'fn':fn,
        'brier':float(np.square(p-y).mean()),
        'nll':float(-(y*np.log(c)+(1-y)*np.log(1-c)).mean()),
        'ece10':expected_calibration_error(p,y)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='new/outputs/pcrau_source_mc_train_dev_20261007')
    args=parser.parse_args();out=workspace_path(args.output)
    if workspace_path('new/outputs') not in out.parents:raise ValueError('Output must stay under new/outputs')
    protected={}
    for base in ('new/outputs','new/src/pcrau','new/scripts','new/tests','new/configs'):
        for p in workspace_path(base).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:protected[str(p.resolve())]=sha256_file(p)
    out.mkdir(exist_ok=False);atomic_json(out/'protected_files_before.json',protected)
    def desc(path,expected=None):
        p=Path(path).resolve();h=sha256_file(p)
        if expected is not None and h!=expected:raise ValueError(f'Hash mismatch: {p}')
        protected[str(p)]=h;return {'path':str(p),'sha256':h}
    torch.set_num_threads(4);seed_everything(24082026)
    checks=[]
    for name,fn in runpy.run_path(str(workspace_path('new/tests/test_source_mc.py'))).items():
        if name.startswith('test_') and callable(fn):fn();checks.append(name)
    atomic_json(out/'unit_checks.json',{'pass':True,'tests':checks})
    # Tests may use RNG; runtime starts from its separately locked seed.
    seed_everything(24082026)
    bp=workspace_path('new/outputs/pcrau_unified_spatial_20261005_r2/bundle.json')
    desc(bp,'0aacaedf3fa27a2686f936893f5e694b7bea88eeb0675da6503238eecd126446')
    pipeline=UnifiedInference.from_bundle(bp)
    sampler=FiveSourceHeadMC(pipeline.shadow.baseline.source_head,torch.bfloat16)
    assert sampler.dropout_p==.1
    state=state_digest(pipeline.shadow.state_dict());mcstate=state_digest(sampler.head.state_dict())
    assert mcstate==state_digest(pipeline.shadow.baseline.source_head.state_dict())
    sources=[workspace_path(p) for p in ('new/src/pcrau/source_mc.py','new/scripts/audit_five_sources_mc.py',
                                        'new/tests/test_source_mc.py')]
    protocol=workspace_path('plan/S5_FIVE_SOURCE_MC_PROTOCOL_20261007.md')
    lock={'version':VERSION,'original_bundle':desc(bp),'protocol':desc(protocol),
        'source_files':{str(p):desc(p) for p in sources},'original_neural_state_sha256':state,
        'source_head_state_sha256':mcstate,'source_order':SOURCE_CLASSES,'fields':FIELDS,
        'passes':20,'dropout_p':.1,'sampling_scope':'source_head_only_fixed_global_feature',
        'seed_schedule':'24082026 + batch_index, continuous train→dev','batch_size':20,
        'train_samples':1600,'dev_samples':400,'no_train_fit_calibration_IID':True,
        'runtime':runtime_info(),'original_outputs_or_profile_changed':False}
    atomic_json(out/'mc_execution_lock.json',lock)
    frozen=out/'frozen';frozen.mkdir()
    for i,p in enumerate(sources):shutil.copyfile(p,frozen/f'{i:02d}_{p.name}')
    summary={'version':VERSION,'scope':lock['sampling_scope'],'splits':{},'passes':20,
        'source_head_dropout_p':.1,'source_labels_define_semantics':True,'weak_spatial_label':True,
        'MC_source_scores_calibrated':False,'causal_decomposition_claimed':False,
        'aleatoric_identification_claimed':False,'original_outputs_profile_preserved':True,
        'calibration_or_IID_opened':False,'optimizer_steps':0,'tests_passed':checks}
    batch_index=0
    for split in ('train','dev'):
        ds=ArchivedPCRAUDataset(pipeline.config,split,profile='development')
        desc(ds.layout.manifest);desc(ds.layout.feature_index)
        directory=out/split;directory.mkdir()
        @lru_cache(maxsize=24)
        def cached(key):
            d=ds.features[key];p=ds.layout.feature_root/d['path'];desc(p,d['sha256']);return load_features(p)
        observed=[];baseline=[];contexts=[];detlogits=[];probabilities=[];latencies=[]
        replay_check=None
        for start in range(0,len(ds.entries),20):
            es=ds.entries[start:start+20];features=[];prompts=[]
            for e in es:
                key=ds.sample_to_feature[e['sample_id']];d=ds.features[key]
                assert all(d[k]==e['feature_input'][k] for k in ('rgb_sha256','depth_sha256'))
                features.append(cached(key));prompts.append(e['feature_input']['prompt'])
            f={k:torch.stack([x[k] for x in features]) for k in features[0]}
            seed=24082026+batch_index;batch_index+=1
            torch.cuda.synchronize();started=time.perf_counter()
            result=predict_with_source_mc(pipeline,f,prompts,sampler,20,seed)
            torch.cuda.synchronize();latencies.append((time.perf_counter()-started)*1000)
            assert result['deterministic_copy_exact']
            if split=='dev' and start==0:
                ctx=result['context'].to(pipeline.device)
                same=sampler.sample(ctx,20,seed);other=sampler.sample(ctx,20,seed+10000);reference=sampler.sample(ctx,100,seed)
                assert np.array_equal(same['samples'],result['mc_result']['samples'])
                assert not np.array_equal(same['samples'],other['samples'])
                replay_check={'samples':len(es),'same_seed_exact':True,'different_seed_changes_masks':True,
                    'T20_vs_T100_max_mean_delta':float(np.abs(same['statistics'][FIELDS[0]]-reference['statistics'][FIELDS[0]]).max()),
                    'T20_vs_T100_max_MI_delta':float(np.abs(same['statistics'][FIELDS[-1]]-reference['statistics'][FIELDS[-1]]).max()),
                    'T20_vs_other_seed_max_mean_delta':float(np.abs(same['statistics'][FIELDS[0]]-other['statistics'][FIELDS[0]]).max()),
                    'T100_not_used_to_select_T_or_replace_primary':True}
                np.savez_compressed(directory/'sampling_diagnostic.npz',T20=same['samples'],
                    other_seed_T20=other['samples'],T100=reference['samples'])
            contexts.append(result['context']);detlogits.append(result['deterministic_logits'])
            probabilities.append(result['mc_result']['samples'])
            for e,r,b in zip(es,result['source_mc_rows'],result['baseline_predictions']):
                observed.append({'sample_id':e['sample_id'],'family_id':e['family_id'],
                    'prompt':e['feature_input']['prompt'],**r,
                    'deterministic_source_scores':b['source_probabilities'],
                    'deterministic_decision':b['decision'],'MC_statistics_fed_into_risk':False})
                baseline.append({'sample_id':e['sample_id'],'family_id':e['family_id'],**b})
            if start%200==0:print(f'MC {split} {min(start+20,len(ds.entries))}/{len(ds.entries)}',flush=True)
        write_rows(directory/'observable_mc_sources.jsonl',observed)
        write_rows(directory/'deterministic_runtime_predictions.jsonl',baseline)
        save_file({'source_head_input':torch.cat(contexts).contiguous(),
                   'deterministic_source_logits':torch.cat(detlogits).contiguous()},str(directory/'source_context.safetensors'))
        allp=np.concatenate(probabilities,axis=1)
        np.savez_compressed(directory/'mc_samples.npz',sample_ids=np.asarray([r['sample_id'] for r in observed]),probabilities=allp)
        # Source supervision enters only now, after observable MC and runtime export.
        joined=[];labels=[]
        for e,r in zip(ds.entries,observed):
            y=[name in e['supervision']['source_labels'] for name in SOURCE_CLASSES]
            y[SOURCE_CLASSES.index('spatial')]=e['supervision']['answerability_state']=='AMBIGUOUS'
            labels.append(y);joined.append({**r,'variant':e['variant'],
                'evaluation':{'source_multihot':y,'answerability_state':e['supervision']['answerability_state']}})
        write_rows(directory/'evaluator_sources.jsonl',joined)
        y=np.asarray(labels);stats=source_statistics(allp);per_source={}
        deterministic=np.asarray([[r['deterministic_source_scores'][k] for k in SOURCE_CLASSES] for r in observed])
        for k,name in enumerate(SOURCE_CLASSES):
            p=stats[FIELDS[0]][:,k];mi=stats[FIELDS[-1]][:,k]
            error=(p>=.5)!=y[:,k]
            per_source[name]={'deterministic':binary_metrics(deterministic[:,k],y[:,k]),
                'MC_mean':binary_metrics(p,y[:,k]),
                'mean_source_score':float(p.mean()),'mean_predictive_entropy':float(stats[FIELDS[2]][:,k].mean()),
                'mean_expected_entropy':float(stats[FIELDS[3]][:,k].mean()),
                'MI_mean':float(mi.mean()),'MI_max':float(mi.max()),
                'mean_std':float(stats[FIELDS[1]][:,k].mean()),
                'MI_mean_when_correct':float(mi[~error].mean()) if (~error).any() else None,
                'MI_mean_when_wrong':float(mi[error].mean()) if error.any() else None,
                'confident_wrong_MC_mean_ge_0p9_or_le_0p1':int((error & ((p>=.9)|(p<=.1))).sum()),
                'labels_are_weak':name=='spatial'}
        summary['splits'][split]={'samples':len(observed),'families':len({r['family_id'] for r in observed}),
            'per_source':per_source,'deterministic_copy_exact_every_batch':True,
            'same_seed_replay_and_convergence_diagnostic':replay_check,
            'batch20_pipeline_plus_source_MC_latency_median_ms':float(np.median(latencies)),
            'correctness_comparison_is_descriptive_not_heldout_confirmation':True}
        atomic_json(directory/'summary.json',summary['splits'][split])
    assert state_digest(pipeline.shadow.state_dict())==state
    assert state_digest(sampler.head.state_dict())==mcstate
    assert all(p.grad is None and not p.requires_grad for p in pipeline.shadow.parameters())
    pipeline.shadow._check_frozen()
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    summary['neural_states_unchanged']=True;summary['original_protected_files_unchanged']=len(protected)
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'protected_sha256':protected,'execution_lock_sha256':sha256_file(out/'mc_execution_lock.json'),
        'source_files':lock['source_files'],'all_original_hashes_unchanged':True})
    atomic_json(out/'RUN_STATUS.json',{'status':'COMPLETE_REAL_FIVE_SOURCE_HEAD_MC_TRAIN_DEV'})
    print(json.dumps({'status':'COMPLETE','output':str(out),'dev':summary['splits']['dev']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
