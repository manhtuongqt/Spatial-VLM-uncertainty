#!/usr/bin/env python3
"""Frozen live bundle on old IID: current forward evidence, then evaluator join."""
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

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import torch
from pcrau.anchor_shadow_experiment import state_digest
from pcrau.calibration import read_jsonl, predict_risk
from pcrau.dataset import ArchivedPCRAUDataset, ANSWER_CLASSES, VARIANT_ORDER
from pcrau.live_iid_analysis import model_metrics, paired_changes, paired_family_bootstrap
from pcrau.selective_experiment import apply_risk, predicted_answer, experimental_action
from pcrau.unified_inference import UnifiedInference, load_features, attach_decisions
from pcrau.utils import atomic_json, read_json, sha256_file, workspace_path, seed_everything, runtime_info
from pcrau.verifier_risk import risk_vector


def write_rows(path, rows):
    with path.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', default='new/outputs/pcrau_unified_spatial_20261005_r2/bundle.json')
    parser.add_argument('--output', default='new/outputs/pcrau_unified_spatial_iid_20261007')
    args=parser.parse_args();out=workspace_path(args.output)
    if workspace_path('new/outputs') not in out.parents: raise ValueError('Output must be under new/outputs')
    protected={}
    # Durable task-start snapshot, excluding only new output; no historical files overwritten.
    for base in ('new/outputs','new/src/pcrau','new/scripts','new/tests','new/configs'):
        for p in workspace_path(base).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                protected[str(p.resolve())]=sha256_file(p)
    out.mkdir(exist_ok=False);atomic_json(out/'protected_files_before.json',protected)
    def descriptor(path, expected=None):
        p=Path(path).resolve();h=sha256_file(p)
        if expected is not None and h!=expected:raise ValueError(f'Hash mismatch: {p}')
        protected[str(p)]=h;return {'path':str(p),'sha256':h}
    bundle_path=workspace_path(args.bundle);manifest=read_json(bundle_path)
    if sha256_file(bundle_path)!='0aacaedf3fa27a2686f936893f5e694b7bea88eeb0675da6503238eecd126446':
        raise ValueError('This execution protocol locks the selected r2 bundle')
    lock_path=bundle_path.parent/manifest['freeze_lock']['path']
    lock=read_json(lock_path);lock_id=sha256_file(lock_path)
    assert lock_id=='5478ac842cab60f8a2bff5e7c8a193b7f806f374ee63b72e38a38f68fa1da81a'
    active_path=workspace_path('new/outputs/active_experimental_profile.json');active=read_json(active_path)
    for key in ('config','checkpoint','calibrator','profiles'):
        descriptor(active[key],active[key+'_sha256'])
    descriptor(active_path)
    source_paths=[workspace_path(p) for p in ('new/scripts/reevaluate_unified_spatial_iid.py',
        'new/scripts/report_unified_spatial_iid.py','new/src/pcrau/live_iid_analysis.py',
        'new/tests/test_live_iid_analysis.py')]
    torch.set_num_threads(4);seed_everything(24082026)
    checks=[]
    for relative in ('new/tests/test_live_iid_analysis.py','new/tests/test_unified_inference.py',
                     'new/tests/test_horizontal_verifier.py'):
        for name,fn in runpy.run_path(str(workspace_path(relative))).items():
            if name.startswith('test_') and callable(fn):fn();checks.append(name)
    atomic_json(out/'unit_checks.json',{'pass':True,'tests':checks})
    pipeline=UnifiedInference.from_bundle(bundle_path)
    assert pipeline.profile['threshold']==0.2889643687106893
    assert all(not p.requires_grad and p.grad is None for p in pipeline.shadow.parameters())
    state_before=state_digest(pipeline.shadow.state_dict())
    baseline_root=Path(active['checkpoint']).parents[3]
    baseline_source=baseline_root/'test_iid_7p5/inference/predictions.jsonl'
    descriptor(baseline_source)
    frozen=out/'frozen';frozen.mkdir()
    for name in ('bundle.json','calibrator.json','profile.json'):
        shutil.copyfile(bundle_path.parent/name,frozen/name)
    shutil.copyfile(lock_path,frozen/'original_freeze_lock.json')
    execution={'date_local':'2026-10-07','bundle':descriptor(bundle_path),
        'original_freeze':descriptor(lock_path),'calibrator':descriptor(bundle_path.parent/'calibrator.json'),
        'profile':descriptor(bundle_path.parent/'profile.json'),
        'threshold':pipeline.profile['threshold'],'baseline_threshold':active['risk_threshold'],
        'protocol':descriptor(workspace_path('plan/S4_LIVE_IID_PROTOCOL_20261007.md')),
        'source_files':{str(p):descriptor(p) for p in source_paths},'runtime':runtime_info(),
        'neural_state_sha256':state_before,'batch_size':20,'order':'IID manifest entries',
        'seed':24082026,'family_bootstrap':{'resamples':5000,'seed':24082026},
        'model_calibrator_threshold_changed':False,'fit_or_training_authorized':False,
        'IID_history':'previously_observed_reevaluation','runtime_evidence':'current_forward_only'}
    atomic_json(out/'pre_iid_execution_lock.json',execution)
    snapshot=frozen/'source_snapshot';snapshot.mkdir()
    for i,p in enumerate(source_paths):shutil.copyfile(p,snapshot/f'{i:02d}_{p.name}')
    # Open IID only after validating the original immutable runtime bundle.
    ds=ArchivedPCRAUDataset(pipeline.config,'test_iid',profile='test_iid')
    assert len(ds.entries)==1000 and len({e['family_id'] for e in ds.entries})==200
    assert len({e['sample_id'] for e in ds.entries})==1000
    for family in {e['family_id'] for e in ds.entries}:
        assert Counter(e['variant'] for e in ds.entries if e['family_id']==family)==Counter(VARIANT_ORDER)
    descriptor(ds.layout.manifest);descriptor(ds.layout.feature_index)
    @lru_cache(maxsize=24)
    def cached(key):
        desc=ds.features[key];path=ds.layout.feature_root/desc['path']
        descriptor(path,desc['sha256']);return load_features(path)
    observed=[];anchor_observable=[];latencies=[];baseline_capture={}
    def old_anchor_hook(module,args,output):
        baseline_capture['old_anchor_logits']=output['anchor_logits'].detach().float().cpu()
    hook=pipeline.shadow.baseline.register_forward_hook(old_anchor_hook)
    try:
        for start in range(0,1000,20):
            entries=ds.entries[start:start+20];features=[];prompts=[]
            for e in entries:
                key=ds.sample_to_feature[e['sample_id']];desc=ds.features[key]
                assert all(desc[k]==e['feature_input'][k] for k in ('rgb_sha256','depth_sha256'))
                features.append(cached(key));prompts.append(e['feature_input']['prompt'])
            tensors={k:torch.stack([f[k] for f in features]) for k in features[0]}
            torch.cuda.synchronize();started=time.perf_counter()
            baseline_capture.clear();rows=pipeline.predict(tensors,prompts)
            torch.cuda.synchronize();latencies.append((time.perf_counter()-started)*1000)
            assert 'old_anchor_logits' in baseline_capture
            for i,(e,row) in enumerate(zip(entries,rows)):
                assert 'evaluation' not in row and 'sample_id' not in row
                observed.append({'sample_id':e['sample_id'],'family_id':e['family_id'],**row})
                old=baseline_capture['old_anchor_logits'][i,0]
                yy,xx=divmod(int(old.flatten().argmax()),32)
                anchor_observable.append({'sample_id':e['sample_id'],'family_id':e['family_id'],
                    'old_anchor_peak_xy':[20*xx+10,20*yy+10],
                    'old_anchor_max_logit':float(old.max()),
                    'old_anchor_sigmoid_max':float(old.sigmoid().max()),
                    'new_anchor':row['verifiers']['P1']})
            if start%100==0:print(f'LIVE IID {start+20}/1000',flush=True)
    finally:hook.remove()
    write_rows(out/'runtime_predictions.jsonl',observed)
    write_rows(out/'anchor_observable.jsonl',anchor_observable)
    # Oracle masks are read only now, after all live decisions are written.
    evaluated=[];anchors=[]
    for e,row,a in zip(ds.entries,observed,anchor_observable):
        sup=e['supervision'];path=ds.layout.dataset_path(sup['target_mask_path'])
        descriptor(path,sup['target_mask_sha256']);mask=cv2.imread(str(path),0)
        assert mask is not None and mask.shape==(480,640)
        x,y=row['spatial']['map_pixel_xy'];inside=bool(mask[y,x]>0)
        truth=sup['answerability_state']
        ev={'answerability_state':truth,'map_inside_target':inside,'target_exists':bool(mask.any()),
            'error_event':truth!='FOUND' or not inside}
        evaluated.append({**row,'variant':e['variant'],'evaluation':ev})
        v=row['verifiers']['P1']
        if v['scope_status']=='SUPPORTED_DIAGNOSTIC':
            assert len(sup['anchor_masks'])==1
            ad=sup['anchor_masks'][0];ap=ds.layout.dataset_path(ad['path'])
            descriptor(ap,ad['sha256']);am=cv2.imread(str(ap),0)
            assert am is not None and am.shape==(480,640)
            area=int((am>0).sum());ox,oy=a['old_anchor_peak_xy'];nx,ny=v['anchor_peak_xy']
            anchors.append({'sample_id':e['sample_id'],'family_id':e['family_id'],'variant':e['variant'],
                'truth':truth,'pixels':area,'target_hit':inside,
                'M0_hit':bool(am[oy,ox]>0) if area else None,
                'P1_hit':bool(am[ny,nx]>0) if area else None,
                'M0_peak_xy':[ox,oy],'P1_peak_xy':[nx,ny],
                'M0_sigmoid_max':a['old_anchor_sigmoid_max'],
                'P1_sigmoid_max':v['features']['anchor_sigmoid_max']})
    write_rows(out/'evaluator_predictions.jsonl',evaluated);write_rows(out/'anchor_evaluation.jsonl',anchors)
    # Baseline artifact read occurs after primary forward; never used as primary evidence.
    legacy={r['sample_id']:r for r in read_jsonl(baseline_source)}
    assert set(legacy)=={r['sample_id'] for r in observed}
    reference=[legacy[r['sample_id']] for r in observed]
    baseline_risk=apply_risk(reference,read_json(Path(active['calibrator'])))
    baseline=[];drift=[]
    for e,live,old,risk in zip(ds.entries,evaluated,reference,baseline_risk):
        assert old['model_checkpoint_sha256']==active['checkpoint_sha256']
        assert old['family_id']==live['family_id']
        assert old['evaluation']['answerability_state']==live['evaluation']['answerability_state']
        path=ds.layout.dataset_path(e['supervision']['target_mask_path']);m=cv2.imread(str(path),0)
        xx,yy=old['spatial']['map_pixel_xy'];old_inside=bool(m[yy,xx]>0)
        assert old_inside==old['evaluation']['map_inside_target']
        assert old['evaluation']['error_event']==(old['evaluation']['answerability_state']!='FOUND' or not old_inside)
        decision={'action':experimental_action(old,float(risk),'hard_found',active['risk_threshold']),
            'risk':float(risk),'threshold':active['risk_threshold'],'policy':'hard_found',
            'predicted_answerability':predicted_answer(old),'decision_level':'perception_only',
            'robot_motion_commanded':False}
        baseline.append({**old,'variant':e['variant'],'decision':decision})
        td=float(np.abs(np.asarray(live['spatial']['probability_grid'])-np.asarray(old['spatial']['probability_grid'])).max())
        ad=max(abs(live['answerability_probabilities'][k]-old['answerability_probabilities'][k]) for k in ANSWER_CLASSES)
        sd=max(abs(live['source_probabilities'][k]-old['source_probabilities'][k]) for k in old['source_probabilities'])
        if td or ad or sd:drift.append({'sample_id':live['sample_id'],'target_grid_delta':td,
            'answer_probability_delta':ad,'source_delta':sd,
            'answer_argmax_changed':predicted_answer(live)!=predicted_answer(old),
            'target_MAP_changed':live['spatial']['map_pixel_xy']!=old['spatial']['map_pixel_xy']})
    write_rows(out/'baseline_evaluator_predictions.jsonl',baseline)
    baseline_runtime=[{'sample_id':r['sample_id'],'family_id':r['family_id'],'decision':r['decision']} for r in baseline]
    write_rows(out/'baseline_runtime_decisions.jsonl',baseline_runtime)
    risk=np.asarray([r['decision']['risk'] for r in observed])
    metrics={'baseline':model_metrics(baseline,baseline_risk,active['risk_threshold']),
             'live':model_metrics(evaluated,risk,pipeline.profile['threshold'])}
    assert metrics['baseline']['policy']['accepted']==361
    assert metrics['baseline']['policy']['correct_acceptances']==327 and metrics['baseline']['policy']['errors']==34
    paired,changes=paired_changes(evaluated,baseline);write_rows(out/'paired_decision_changes.jsonl',changes)
    bootstrap=paired_family_bootstrap(evaluated,baseline)
    x=np.stack([risk_vector(r,'P1_G44') for r in observed]);neutral=x.copy();neutral[:,-3:]=0.
    c=pipeline.calibrator
    nr=predict_risk((neutral-np.asarray(c['mean']))/np.asarray(c['scale']),np.asarray(c['coefficients']),c['intercept'])
    geometry_changes=[]
    for r,n in zip(evaluated,nr):
        action=experimental_action(r,float(n),'hard_found',pipeline.profile['threshold'])
        if action!=r['decision']['action']:
            geometry_changes.append({'sample_id':r['sample_id'],'family_id':r['family_id'],
                'current_action':r['decision']['action'],'geometry_zero_action':action,
                'current_risk':r['decision']['risk'],'geometry_zero_risk':float(n),
                'error_event':r['evaluation']['error_event'],'truth':r['evaluation']['answerability_state']})
    write_rows(out/'geometry_zero_diagnostic.jsonl',geometry_changes)
    scope=dict(Counter(r['verifiers']['P1']['scope_status'] for r in observed))
    visible=[a for a in anchors if a['pixels']];empty=[a for a in anchors if not a['pixels']]
    anchor_metrics={'horizontal':len(anchors),'visible':len(visible),'empty':len(empty),
        'M0_hits':sum(a['M0_hit'] for a in visible),'P1_hits':sum(a['P1_hit'] for a in visible),
        'fixed':sum(a['P1_hit'] and not a['M0_hit'] for a in visible),
        'broken':sum(a['M0_hit'] and not a['P1_hit'] for a in visible),
        'found_both':sum(a['truth']=='FOUND' and a['target_hit'] and a['P1_hit'] is True for a in anchors),
        'empty_cases':empty}
    stratified={}
    for status in scope:
        ids=[i for i,r in enumerate(observed) if r['verifiers']['P1']['scope_status']==status]
        stratified[status]={'live':model_metrics([evaluated[i] for i in ids],risk[ids],pipeline.profile['threshold']),
            'baseline':model_metrics([baseline[i] for i in ids],baseline_risk[ids],active['risk_threshold'])}
    unchanged=state_digest(pipeline.shadow.state_dict())==state_before
    assert unchanged and all(p.grad is None and not p.requires_grad for p in pipeline.shadow.parameters())
    assert attach_decisions([dict(r) for r in observed],pipeline.calibrator,pipeline.profile)==observed
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    assert sha256_file(bundle_path)==execution['bundle']['sha256']
    summary={'status':'COMPLETE_FROZEN_LIVE_OLD_IID_REEVALUATION','samples':1000,'families':200,
        'bundle_sha256':sha256_file(bundle_path),'freeze_sha256':lock_id,'live_threshold':pipeline.profile['threshold'],
        'baseline_threshold':active['risk_threshold'],'metrics':metrics,'scope':scope,
        'anchor_metrics':anchor_metrics,'paired_changes':paired,'paired_family_bootstrap':bootstrap,
        'scope_stratified':stratified,'geometry_zero_diagnostic_changed_actions':len(geometry_changes),
        'geometry_zero_diagnostic_is_retrained_ablation':False,
        'numerical_diagnostic':{'drift_cases':drift,'target_MAP_changed':sum(r['target_MAP_changed'] for r in drift),
            'answer_argmax_changed':sum(r['answer_argmax_changed'] for r in drift),
            'max_answer_probability_delta':max((r['answer_probability_delta'] for r in drift),default=0.),
            'historical_parity_gate_waived_or_relabelled':False,'cached_primary_evidence_used':False},
        'primary_empirical_budget_met':metrics['live']['policy']['empirical_risk']<=.075,
        'primary_Wilson_upper_budget_met':metrics['live']['policy']['wilson_upper_95']<=.075,
        'neural_states_unchanged':unchanged,'neural_optimizer_steps':0,'calibration_fit_calls':0,
        'selected_or_tuned_on_IID':False,'previously_observed_IID':True,'active_profile_changed':False,
        'robot_motion_commanded':False,'runtime_input_masks_or_truth':False,
        'batch20_live_latency_median_ms':float(np.median(latencies)),
        'latency_includes_forward_boundary_export_risk_and_diagnostic_hook':True,
        'original_protected_files_unchanged':len(protected),'tests_passed':checks}
    np.savez_compressed(out/'risk_scores.npz',sample_ids=np.asarray([r['sample_id'] for r in observed]),
                        live=risk,baseline=baseline_risk,geometry_zero_diagnostic=nr)
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'protected_sha256':protected,'source_files':execution['source_files'],
        'execution_lock_sha256':sha256_file(out/'pre_iid_execution_lock.json'),
        'bundle_and_profile_unchanged':True,'all_original_hashes_verified':True})
    atomic_json(out/'RUN_STATUS.json',{'status':summary['status']})
    print(json.dumps({'complete':str(out),'live':metrics['live'],'paired':paired,'anchor':anchor_metrics},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
