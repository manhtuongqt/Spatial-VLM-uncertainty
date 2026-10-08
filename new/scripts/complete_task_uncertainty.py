#!/usr/bin/env python3
"""Incremental recovery of S6, preserving complete r2 train/dev and incomplete files."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys,runpy,time,shutil
from pathlib import Path
from functools import lru_cache
import numpy as np,torch
from safetensors.torch import load_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcrau.task_uncertainty import TaskHeads
from pcrau.task_inference import TaskInference,apply_task_calibrator
from pcrau.unified_inference import UnifiedInference,load_features
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.live_iid_analysis import model_metrics,paired_changes,paired_family_bootstrap
from pcrau.utils import atomic_json,sha256_file,seed_everything

ROOT=Path('new/outputs/pcrau_task_uncertainty_20261007')
OUT=ROOT/'evaluation_r3'
helpers=runpy.run_path('new/scripts/evaluate_task_uncertainty.py')

def write(path,rows):
    helpers['write_rows'](path,rows)
    assert path.stat().st_size>0

def loadrows(path):return [json.loads(l) for l in path.read_text().splitlines()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['calibration','finish_calibration','test_iid'],required=True);a=p.parse_args()
    torch.set_num_threads(4);seed_everything(24082026)
    frozen=json.loads((ROOT/'evaluation_r2/neural_freeze.json').read_text())
    for desc in frozen['source_files'].values():assert sha256_file(Path(desc['path']))==desc['sha256']
    if a.stage in ('calibration','finish_calibration'):
        if a.stage=='calibration':
            assert not OUT.exists();OUT.mkdir()
            atomic_json(OUT/'recovery_protocol.json',{'reason':'r2 calibration outputs incomplete; retain all r2 files',
            'same_checkpoint_and_producer':True,'no_neural_retraining':True,'no_IID_tuning':True,
            'r2_freeze_sha256':sha256_file(ROOT/'evaluation_r2/neural_freeze.json'),
            'complete_train_dev_from':'evaluation_r2','calibration_seed_start':24082126,
                'source_sha256':sha256_file(Path(__file__))})
        else:
            assert (OUT/'calibration/observable_predictions.jsonl').exists() and not (OUT/'calibrator.json').exists()
            atomic_json(OUT/'recovery_completion_revision.json',{'reason':'compare canonical serialized observable values; parser anchors tuple becomes JSON list',
                'new_source_sha256':sha256_file(Path(__file__)),'original_recovery_protocol_preserved':True,
                'same_neural_producer_checkpoint_and_raw_samples':True,'no_new_MC_forward_or_training':True})
        pipeline=UnifiedInference.from_bundle(Path(frozen['baseline_bundle']['path']))
        heads=TaskHeads(len(frozen['classes'])).cuda();heads.load_state_dict(load_file(frozen['task_checkpoint']['path']))
        model=TaskInference(pipeline,heads,frozen['classes'])
        split='calibration';profile='calibration';seedstart=24082126
    else:
        model=TaskInference.from_bundle(OUT/'bundle.json');pipeline=model.pipeline
        split='test_iid';profile='test_iid';seedstart=24082176
        atomic_json(OUT/'pre_IID_lock.json',{'bundle_sha256':sha256_file(OUT/'bundle.json'),
            'threshold':model.profile['threshold'],'previously_observed_IID':True,'no_fit_on_IID':True})
    ds=ArchivedPCRAUDataset(pipeline.config,split,profile=profile)
    d=OUT/split
    if a.stage!='finish_calibration':d.mkdir(exist_ok=False)
    chunks=d/'raw_batches'
    if a.stage!='finish_calibration':chunks.mkdir()
    @lru_cache(maxsize=24)
    def feature(key):
        descriptor=ds.features[key];path=ds.layout.feature_root/descriptor['path'];assert sha256_file(path)==descriptor['sha256'];return load_features(path)
    rows=[];times=[]
    for start in range(0,len(ds) if a.stage!='finish_calibration' else 0,20):
        es=ds.entries[start:start+20];fs=[feature(ds.sample_to_feature[e['sample_id']]) for e in es]
        f={k:torch.stack([v[k] for v in fs]) for k in fs[0]};prompts=[e['feature_input']['prompt'] for e in es]
        tick=time.perf_counter();rr,raw=model.predict(f,prompts,seedstart+start//20) if model.calibrator else model.observe(f,prompts,seedstart+start//20)
        torch.cuda.synchronize();times.append((time.perf_counter()-tick)*1000)
        rows.extend({'sample_id':e['sample_id'],'family_id':e['family_id'],**r} for e,r in zip(es,rr))
        file=chunks/f'{start//20:03d}.npz';np.savez_compressed(file,**raw);assert file.stat().st_size>0
        if start%200==0:print(f'RECOVER {split} {start+len(es)}/{len(ds)}',flush=True)
    if a.stage=='finish_calibration':rows=loadrows(d/'observable_predictions.jsonl')
    else:write(d/'observable_predictions.jsonl',rows)
    arrays=[]
    for file in sorted(chunks.glob('*.npz')):
        with np.load(file) as arr:arrays.append({k:arr[k] for k in arr.files})
    raw={k:np.concatenate([b[k] for b in arrays],axis=1) for k in arrays[0]};del arrays
    if a.stage in ('calibration','finish_calibration'):
        prior=loadrows(ROOT/'evaluation_r2/calibration/observable_predictions.jsonl')
        assert json.loads(json.dumps(rows))==prior
        atomic_json(d/'r2_replay_check.json',{'samples':len(rows),'all_observable_outputs_exact':True})
    evaluated,tasks=helpers['join_evaluator'](ds,rows,raw,frozen['classes'])
    write(d/'task_evaluation.jsonl',tasks)
    if a.stage in ('calibration','finish_calibration'):
        cal,profile,oof=helpers['fit'](evaluated)
        atomic_json(OUT/'calibrator.json',cal);atomic_json(OUT/'profile.json',profile)
        bundle={**frozen,'calibrator':{'path':str((OUT/'calibrator.json').resolve()),'sha256':sha256_file(OUT/'calibrator.json')},
            'profile':{'path':str((OUT/'profile.json').resolve()),'sha256':sha256_file(OUT/'profile.json')},
            'recovery_source_sha256':sha256_file(Path(__file__))}
        atomic_json(OUT/'bundle.json',bundle);np.save(d/'oof_risk.npy',oof)
        model.calibrator=cal;model.profile=profile
        evaluated=apply_task_calibrator(evaluated,cal,profile)
    write(d/'evaluator_predictions.jsonl',evaluated)
    reference=[{**r,'decision':r.get('reference_live44_decision',r['decision'])} for r in evaluated]
    summary={'tasks':helpers['summarize_task_evaluation'](tasks),
        'primary':model_metrics(evaluated,[r['decision']['risk'] for r in evaluated],model.profile['threshold'] if model.profile['threshold'] is not None else -1),
        'reference_live44':model_metrics(reference,[r['decision']['risk'] for r in reference],pipeline.profile['threshold']),
        'batch20_feature_cache_pipeline_MC_latency_median_ms':float(np.median(times)) if times else None}
    summary['paired'],changes=paired_changes(evaluated,reference);write(d/'changed_decisions.jsonl',changes) if changes else atomic_json(d/'changed_decisions_empty.json',{'changed_actions':0})
    summary['family_bootstrap']=paired_family_bootstrap(evaluated,reference)
    atomic_json(d/'summary.json',summary)
    for file in ['calibrator.json','profile.json','bundle.json']:
        assert (OUT/file).stat().st_size>0;json.loads((OUT/file).read_text())
    if a.stage=='test_iid':
        protected=json.loads((ROOT/'training_lock.json').read_text())['protected_files']
        assert all(sha256_file(Path(path))==h for path,h in protected.items())
        aggregate={'status':'COMPLETE','splits':{s:json.loads((ROOT/'evaluation_r2'/s/'summary.json').read_text()) for s in ('train','dev')}}
        aggregate['splits'].update({s:json.loads((OUT/s/'summary.json').read_text()) for s in ('calibration','test_iid')})
        aggregate.update(active_profile_changed=False,original_protected_files_unchanged=len(protected),
            MC_task_features_actually_fed_into_risk=True,full_amodal_or_causal_decomposition_claimed=False)
        atomic_json(OUT/'summary.json',aggregate);atomic_json(OUT/'RUN_STATUS.json',{'status':'COMPLETE'})
    atomic_json(d/'stage_status.json',{'status':'COMPLETE','samples':len(rows)})
    print(json.dumps({'status':'COMPLETE','stage':a.stage,'summary':summary},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
