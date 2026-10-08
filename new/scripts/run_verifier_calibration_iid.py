#!/usr/bin/env python3
"""Authorized evidence-only verifier, frozen calibration, then old-IID reevaluation."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import runpy
import sys

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcrau.anchor_shadow import AnchorShadow
from pcrau.anchor_shadow_experiment import state_digest
from pcrau.calibration import read_jsonl
from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset,ANSWER_CLASSES,SOURCE_CLASSES
from pcrau.engine import autocast_context
from pcrau.horizontal_verifier import verify,VERSION
from pcrau.model import PCRAUTargetV2
from pcrau.selective_experiment import apply_risk,choose_policy_threshold,policy_result,predicted_answer,probability_metrics
from pcrau.text import tokenize,relation_ids,prompt_anchor_mask
from pcrau.utils import atomic_json,read_json,runtime_info,seed_everything,sha256_file,workspace_path
from pcrau.verifier_risk import MODELS,fit_verifier_risk,apply_verifier_risk


def write_rows(path,rows):
    with path.open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='new/outputs/pcrau_horizontal_verifier_gate_override_20261005')
    parser.add_argument('--dev-probe',action='store_true',help='Dev-only numerical parity diagnostic; no calibration/IID')
    args=parser.parse_args();output=workspace_path(args.output)
    if workspace_path('new/outputs') not in output.parents:raise ValueError('Output must stay under new/outputs')
    output.mkdir(exist_ok=False)
    protected={}
    def protect(path,expected=None):
        path=Path(path).resolve();digest=sha256_file(path)
        if expected is not None and digest!=expected:raise ValueError(f'Protected hash mismatch: {path}')
        if str(path) in protected and protected[str(path)]!=digest:raise ValueError(f'Changed protected file: {path}')
        protected[str(path)]=digest;return digest
    # Historical checkpoint/data/preflight artifacts: no IID samples opened here.
    for path,digest in read_json(Path('/tmp/pcrau_verifier_pre_task_protected_20261005.json')).items():protect(path,digest)
    active_path=workspace_path('new/outputs/active_experimental_profile.json');active=read_json(active_path);protect(active_path)
    for key in ('config','checkpoint','calibrator','profiles'):protect(active[key],active[key+'_sha256'])
    cfg=read_json(Path(active['config']));candidate_path=workspace_path('new/configs/horizontal_verifier_evidence_20261005.json')
    candidate=read_json(candidate_path);candidate_hash=protect(candidate_path)
    assert candidate['version']==VERSION and candidate['primary']=='P1_G44' and candidate['risk_target']==.075
    protocol_path=workspace_path('plan/S2_VERIFIER_GATE_OVERRIDE_PROTOCOL_20261005.md');protocol_hash=protect(protocol_path)
    numerical_path=workspace_path('plan/S2_VERIFIER_NUMERICAL_PREFLIGHT_20261005.md');numerical_hash=protect(numerical_path)
    pilot=workspace_path('new/outputs/pcrau_s1_anchor_peak_pilot_v2_20261005')
    pilot_summary=read_json(pilot/'summary.json');assert not pilot_summary['upgrade_gates_pass']
    freeze=read_json(pilot/'freeze_lock.json')
    residuals={a:pilot/a/'checkpoints/best/mlp.safetensors' for a in ('C1','P1')}
    for a,p in residuals.items():protect(p,freeze['checkpoints'][a])
    source_paths=list(workspace_path('new/src/pcrau').glob('*.py'))+[
        Path(__file__).resolve(),workspace_path('new/tests/test_horizontal_verifier.py')]
    source_hashes={str(p):protect(p) for p in source_paths}
    torch.set_num_threads(4);seed_everything(24082026)
    checks=[]
    for name,fn in runpy.run_path(str(workspace_path('new/tests/test_horizontal_verifier.py'))).items():
        if name.startswith('test_') and callable(fn):fn();checks.append(name)
    atomic_json(output/'unit_checks.json',{'pass':True,'tests':checks})
    device=torch.device('cuda');model=PCRAUTargetV2(cfg).to(device).eval()
    load_model_checkpoint(model,active['checkpoint'],active['config_sha256']);model.requires_grad_(False)
    branches={a:AnchorShadow(model,'phrase',24082026).eval() for a in residuals}
    for a,p in residuals.items():branches[a].residual.load_state_dict(load_file(str(p),device='cpu'),strict=True);branches[a].residual.requires_grad_(False)
    model_state=state_digest(model.state_dict());branch_states={a:state_digest(b.residual.state_dict()) for a,b in branches.items()}
    assert model_state==pilot_summary['baseline_state_sha256_before_after']
    runtime_lock={'version':VERSION,'config':candidate,'config_sha256':candidate_hash,'protocol_sha256':protocol_hash,
        'baseline_bundle':active,'residual_checkpoints':{a:{'path':str(p),'sha256':sha256_file(p)} for a,p in residuals.items()},
        'source_sha256':source_hashes,'model_state_sha256':model_state,'residual_state_sha256':branch_states,
        'binding_gate_override_by_user':True,'binding_gate_actually_passed':False,
        'calibration_and_iid_authorized':True,'primary_fixed_before_calibration':'P1_G44',
        'target_map_changed':False,'numerical_preflight_sha256':numerical_hash,'runtime':runtime_info()}
    frozen=output/'frozen';frozen.mkdir();atomic_json(frozen/'freeze_lock.json',runtime_lock)
    bundle_hash=sha256_file(frozen/'freeze_lock.json')
    snapshot=frozen/'source_snapshot';snapshot.mkdir()
    snapshot_manifest={}
    for i,(p,h) in enumerate(source_hashes.items()):
        name=f'{i:03d}_{Path(p).name}';(snapshot/name).write_bytes(Path(p).read_bytes());snapshot_manifest[p]={'sha256':h,'snapshot':name}
    atomic_json(snapshot/'manifest.json',snapshot_manifest)
    cache={};parity={};all_metrics={}
    def batch(dataset,entries):
        samples=[]
        for e in entries:
            key=dataset.sample_to_feature[e['sample_id']];desc=dataset.features[key]
            assert desc['rgb_sha256']==e['feature_input']['rgb_sha256'] and desc['depth_sha256']==e['feature_input']['depth_sha256']
            ck=(dataset.profile,key)
            if ck not in cache:
                protect(dataset.layout.feature_root/desc['path'],desc['sha256']);cache[ck]=dataset._feature(e)
            feature=cache[ck];prompt=e['feature_input']['prompt']
            ids,mask=tokenize(prompt,72,8192);rid,rm=relation_ids(prompt,3)
            samples.append({'r0':feature['R0_GRID'].float(),'d0':feature['D0_GRID'].float(),
                'r_thumb':feature['R0_THUMB'].float(),'d_thumb':feature['D0_THUMB'].float(),
                'token_ids':torch.tensor(ids,dtype=torch.long),'token_mask':torch.tensor(mask,dtype=torch.bool),
                'relation_ids':torch.tensor(rid,dtype=torch.long),'relation_mask':torch.tensor(rm,dtype=torch.bool),
                'anchor_mask':torch.tensor(prompt_anchor_mask(prompt,3),dtype=torch.bool)})
        return {k:torch.stack([s[k] for s in samples]).to(device) for k in samples[0]}

    def export(split,source_file):
        profile='development' if split=='dev' else split
        dataset=ArchivedPCRAUDataset(cfg,split,profile=profile,verify_feature_hash=True)
        protect(dataset.layout.manifest);protect(dataset.layout.feature_index);protect(source_file)
        original=read_jsonl(source_file);entries={e['sample_id']:e for e in dataset.entries}
        assert set(entries)=={r['sample_id'] for r in original}
        # Historical dev export predates the per-row checkpoint field; full
        # observable parity below verifies it. Calibration/IID require the field.
        assert split=='dev' or all(r['model_checkpoint_sha256']==active['checkpoint_sha256'] for r in original)
        directory=output/split;directory.mkdir()
        traces=[];raw={};max_target_error=0.;max_answer_error=0.;max_source_error=0.;n_bypass=0;drift_cases=[]
        # Fixed20-sample observable batches reproduce original baseline export.
        for start in range(0,len(original),20):
            rows=original[start:start+20];es=[entries[r['sample_id']] for r in rows]
            inputs=batch(dataset,es);prompts=[e['feature_input']['prompt'] for e in es]
            with torch.no_grad(),autocast_context(device,cfg['optimization']):
                cap=branches['P1'].capture(inputs,prompts);plain=model(inputs)
                assert all(torch.equal(v,cap.baseline[k]) for k,v in plain.items())
                maps={'M0':cap.baseline['anchor_logits'],**{a:b.predict(cap) for a,b in branches.items()}}
                inactive=~cap.active_slots
                assert all(torch.equal(v[inactive],cap.baseline['anchor_logits'][inactive]) for v in maps.values())
            target=cap.baseline['target_logits'].float().cpu()
            answer=cap.baseline['answerability_logits'].float().softmax(-1).cpu().numpy()
            source=cap.baseline['source_logits'].float().sigmoid().cpu().numpy()
            anchor={a:v.float().cpu().numpy() for a,v in maps.items()}
            for i,row in enumerate(rows):
                probability=target[i].flatten().softmax(0).reshape(24,32).numpy()
                te=float(np.abs(probability-np.asarray(row['spatial']['probability_grid'])).max())
                ae=float(np.abs(answer[i]-np.asarray([row['answerability_probabilities'][a] for a in ANSWER_CLASSES])).max())
                se=float(np.abs(source[i]-np.asarray([row['source_probabilities'][a] for a in SOURCE_CLASSES])).max())
                max_target_error=max(max_target_error,te);max_answer_error=max(max_answer_error,ae);max_source_error=max(max_source_error,se)
                if te or ae or se:drift_cases.append({'sample_id':row['sample_id'],'target_error':te,'answer_error':ae,'source_error':se})
                assert int(answer[i].argmax())==ANSWER_CLASSES.index(predicted_answer(row))
                if not args.dev_probe:
                    tolerance=candidate['historical_adapter_probability_atol']
                    assert te==se==0. and ae<=tolerance,(split,row['sample_id'],te,ae,se)
                vs={a:verify(prompts[i],probability,anchor[a][i,0],anchor['M0'][i,0]) for a in maps}
                n_bypass+=int(vs['P1']['scope_status']!='SUPPORTED_DIAGNOSTIC')
                traces.append({'sample_id':row['sample_id'],'family_id':row['family_id'],'prompt':prompts[i],
                    'verifier_bundle_sha256':bundle_hash,'verifiers':vs})
                raw[row['sample_id']]={a:anchor[a][i,0] for a in maps}
        write_rows(directory/'observable_traces.jsonl',traces)
        # Ground-truth masks/truth join only after observable traces are saved.
        augmented=[];diagnostics=[]
        for row,trace in zip(original,traces):
            e=entries[row['sample_id']];desc=e['supervision'];path=dataset.layout.dataset_path(desc['target_mask_path'])
            protect(path,desc['target_mask_sha256']);mask=cv2.imread(str(path),0)
            assert mask is not None and mask.shape==(480,640)
            x,y=row['spatial']['map_pixel_xy'];inside=bool(mask[y,x]>0)
            truth=desc['answerability_state'];event=truth!='FOUND' or not inside
            assert inside==row['evaluation']['map_inside_target'] and event==row['evaluation']['error_event']
            assert truth==row['evaluation']['answerability_state']
            augmented.append({**row,'verifier_bundle_sha256':bundle_hash,'verifiers':trace['verifiers']})
            if trace['verifiers']['P1']['scope_status']=='SUPPORTED_DIAGNOSTIC':
                assert len(desc['anchor_masks'])==1
                ad=desc['anchor_masks'][0];ap=dataset.layout.dataset_path(ad['path']);protect(ap,ad['sha256'])
                am=cv2.imread(str(ap),0);assert am is not None and am.shape==(480,640)
                pixels=int((am>0).sum());models={}
                for a,v in trace['verifiers'].items():
                    ax,ay=v['anchor_peak_xy']
                    models[a]={'anchor_hit':bool(am[ay,ax]>0) if pixels else None,
                        'peak_compatible':v['features']['peak_compatible'],
                        'sigmoid_max':v['features']['anchor_sigmoid_max']}
                diagnostics.append({'sample_id':row['sample_id'],'family_id':row['family_id'],'truth':truth,
                    'anchor_pixels':pixels,'target_hit':inside,'models':models})
        write_rows(directory/'predictions.jsonl',augmented);write_rows(directory/'anchor_diagnostics.jsonl',diagnostics)
        np.savez_compressed(directory/'anchor_logits.npz',sample_ids=np.asarray([r['sample_id'] for r in original]),
            **{a:np.stack([raw[r['sample_id']][a] for r in original]) for a in maps})
        parity[split]={'samples':len(original),'families':len({r['family_id'] for r in original}),
            'baseline_target_answer_source_exact':max_target_error==max_answer_error==max_source_error==0.,'max_target_error':max_target_error,
            'max_answer_error':max_answer_error,'max_source_error':max_source_error,'bypass_samples':n_bypass,
            'target_map_changed':False,'drift_cases':drift_cases,'scope':dict(Counter(t['verifiers']['P1']['scope_status'] for t in traces))}
        all_metrics[split]={'target_present':sum(r['evaluation']['target_exists'] for r in original),
            'target_present_hits':sum(r['evaluation']['target_exists'] and r['evaluation']['map_inside_target'] for r in original),
            'truth_found':sum(r['evaluation']['answerability_state']=='FOUND' for r in original),
            'valid_found':sum(not r['evaluation']['error_event'] for r in original),
            'anchor_horizontal':{a:{'visible':sum(x['anchor_pixels']>0 for x in diagnostics),
                'hits':sum(x['models'][a]['anchor_hit'] is True for x in diagnostics),
                'empty':sum(x['anchor_pixels']==0 for x in diagnostics)} for a in maps}}
        atomic_json(directory/'verification.json',parity[split]);atomic_json(directory/'grounding.json',all_metrics[split])
        print(f'EXPORTED {split}: {len(original)}samples; max drift target={max_target_error}, answer={max_answer_error}, source={max_source_error}',flush=True)
        return augmented

    # Active checkpoint path is root/detail_cost/checkpoints/best/model.safetensors.
    baseline_root=Path(active['checkpoint']).parents[3]
    dev=export('dev',baseline_root/'detail_cost/verified_dev_predictions.jsonl')
    assert all_metrics['dev']['anchor_horizontal']['P1']['hits']==48
    if args.dev_probe:
        atomic_json(output/'summary.json',{'status':'DEV_NUMERICAL_PARITY_PROBE','dev':parity['dev'],
            'calibration_or_iid_opened_by_runner':False,'optimizer_steps':0,'primary_binding_hits':48})
        print(json.dumps(parity['dev']),flush=True);return
    cal=export('calibration',baseline_root/'calibration_full_fit/predictions.jsonl')
    assert parity['calibration']['samples']==1000 and parity['calibration']['families']==200
    assert all_metrics['calibration']['valid_found']==411
    caldir=output/'calibration';cals=caldir/'calibrators';cals.mkdir()
    error=np.asarray([r['evaluation']['error_event'] for r in cal]);families=[r['family_id'] for r in cal]
    eligible=np.asarray([predicted_answer(r)=='FOUND' for r in cal]);profiles={};calibrators={};scores={}
    active_cal=read_json(Path(active['calibrator']))
    for name in MODELS:
        calibrator,oof=fit_verifier_risk(cal,name,bundle_hash);fit_all=apply_verifier_risk(cal,calibrator)
        if name=='B33':
            for field in ('mean','scale','coefficients','intercept'):
                np.testing.assert_allclose(calibrator[field],active_cal[field],rtol=0,atol=1e-12)
            assert calibrator['family_folds']==active_cal['family_folds']
        threshold=choose_policy_threshold(oof,error,families,eligible,.075,60)
        oof_result,_=policy_result(cal,oof,'hard_found',threshold['threshold'])
        fit_result,_=policy_result(cal,fit_all,'hard_found',threshold['threshold'])
        if name=='B33':assert abs(threshold['threshold']-active['risk_threshold'])<1e-12
        atomic_json(cals/f'{name}.json',calibrator);calibrators[name]=calibrator
        profiles[name]={'threshold_fit':threshold,'crossfit_calibration':oof_result,
            'resubstitution_fit_all':fit_result,'probability_crossfit':calibrator['crossfit_metrics'],
            'calibrator_sha256':sha256_file(cals/f'{name}.json')}
        scores[name+'_oof']=oof;scores[name+'_fit_all']=fit_all
    np.savez_compressed(caldir/'risk_scores.npz',**scores)
    profile_lock={'status':'FROZEN_CALIBRATION_PROFILES_BEFORE_IID','primary':'P1_G44','profiles':profiles,
        'bundle_sha256':bundle_hash,'policy':'hard_found','risk_target':.075,
        'iid_opened_before_profile_freeze':False,'selection_on_iid':False}
    atomic_json(caldir/'profiles.json',profile_lock);profiles_hash=sha256_file(caldir/'profiles.json')
    atomic_json(output/'pre_iid_lock.json',{'bundle_sha256':bundle_hash,'profiles_sha256':profiles_hash,
        'calibrator_sha256':{k:v['calibrator_sha256'] for k,v in profiles.items()},'primary':'P1_G44',
        'iid_predictions_not_opened_by_this_runner_yet':True,'binding_gate_override_by_user':True,
        'old_iid_already_observed_and_schema_inspected':True})
    print('CALIBRATORS/PROFILES FROZEN; now opening existing IID',flush=True)
    iid=export('test_iid',baseline_root/'test_iid_7p5/inference/predictions.jsonl')
    assert parity['test_iid']['samples']==1000 and parity['test_iid']['families']==200
    assert all_metrics['test_iid']['valid_found']==405 and all_metrics['test_iid']['target_present_hits']==775
    assert {r['family_id'] for r in cal}.isdisjoint(r['family_id'] for r in iid)
    assert sha256_file(caldir/'profiles.json')==profiles_hash
    results={};decisions={};iid_risks={}
    for name,calibrator in calibrators.items():
        assert sha256_file(cals/f'{name}.json')==profiles[name]['calibrator_sha256']
        risk=apply_verifier_risk([{k:v for k,v in r.items() if k not in {'evaluation','variant'}} for r in iid],calibrator)
        metrics,records=policy_result(iid,risk,'hard_found',profiles[name]['threshold_fit']['threshold'])
        results[name]={'policy':metrics,'probability':probability_metrics(risk,[r['evaluation']['error_event'] for r in iid])}
        decisions[name]=records;iid_risks[name]=risk
        write_rows(output/'test_iid'/f'{name}_evaluation_decisions.jsonl',records)
        write_rows(output/'test_iid'/f'{name}_runtime_decisions.jsonl',[
            {k:v for k,v in r.items() if k!='evaluation'}|{'robot_motion_commanded':False,
                'selected_point_xy':iid[i]['spatial']['map_pixel_xy'] if r['action']=='EXECUTE' else None,
                'binding_status':'UNVERIFIED','status':'GATE_OVERRIDE_IID_REEVALUATION'} for i,r in enumerate(records)])
    official=apply_risk(iid,active_cal);official_metrics,_=policy_result(iid,official,'hard_found',active['risk_threshold'])
    assert (official_metrics['correct_acceptances'],official_metrics['errors'],official_metrics['accepted'])==(327,34,361)
    assert results['B33']['policy']==official_metrics
    np.savez_compressed(output/'test_iid/risk_scores.npz',**iid_risks)
    # Paired family bootstrap of fixed-profile utility; no test threshold choice.
    grouped=defaultdict(list)
    for i,row in enumerate(iid):grouped[row['family_id']].append(i)
    order=sorted(grouped);rng=np.random.default_rng(24082026)
    indices=rng.integers(0,len(order),size=(5000,len(order)))
    paired={};changed=[]
    for ref in ('B33','P1_A41','C1_G44'):
        tables=[]
        for family in order:
            ids=grouped[family];values=[len(ids),sum(not iid[i]['evaluation']['error_event'] for i in ids)]
            for name in ('P1_G44',ref):
                keep=[i for i in ids if decisions[name][i]['action']=='EXECUTE']
                err=sum(iid[i]['evaluation']['error_event'] for i in keep)
                values.extend([len(keep),err,len(keep)-err])
            tables.append(values)
        table=np.asarray(tables,float);sums=table[indices].sum(1);point=table.sum(0);entry={}
        for metric,num,den in [('coverage',2,0),('selective_risk',3,2),('valid_recall',4,1),('correct_acceptance_rate',4,0)]:
            rnum=num+3;rden=den+3 if metric=='selective_risk' else den
            valid=(sums[:,den]>0)&(sums[:,rden]>0)
            delta=sums[valid,num]/sums[valid,den]-sums[valid,rnum]/sums[valid,rden]
            entry[metric]={'point':float(point[num]/point[den]-point[rnum]/point[rden]) if point[den]>0 and point[rden]>0 else None,
                'ci95':np.quantile(delta,[.025,.975]).tolist() if len(delta) else None,'valid_resamples':int(valid.sum())}
        paired['P1_G44_vs_'+ref]={'families':200,'resamples':5000,'seed':24082026,'metrics':entry}
        for i,row in enumerate(iid):
            a,b=(decisions[n][i]['action']=='EXECUTE' for n in ('P1_G44',ref))
            if a!=b:changed.append({'sample_id':row['sample_id'],'family_id':row['family_id'],'reference':ref,
                'primary_accepted':a,'reference_accepted':b,'error_event':row['evaluation']['error_event'],
                'truth':row['evaluation']['answerability_state'],'variant':row['variant']})
    write_rows(output/'test_iid/paired_decision_changes.jsonl',changed)
    assert sha256_file(frozen/'freeze_lock.json')==bundle_hash
    assert state_digest(model.state_dict())==model_state and all(p.grad is None for p in model.parameters())
    assert all(state_digest(b.residual.state_dict())==branch_states[a] for a,b in branches.items())
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    primary=results['P1_G44']['policy'];budget_met=primary['empirical_risk'] is not None and primary['empirical_risk']<=.075
    summary={'status':'VERIFIER_CALIBRATION_IID_REEVALUATION_COMPLETE','primary':'P1_G44',
        'binding_gate_override_by_user':True,'binding_gate_actually_passed':False,
        'calibration_profiles':profiles,'iid':results,'official_baseline_reproduced':official_metrics,
        'primary_iid_empirical_budget_met':budget_met,'paired_family_bootstrap':paired,
        'parity':parity,'grounding':all_metrics,'target_map_changed':False,'active_profile_changed':False,
        'verifier_binding_status':'UNVERIFIED','iid_previously_observed':True,'test_selected_threshold_or_model':False,
        'new_neural_training':False,'optimizer_steps':0,'robot_motion_commanded':False,
        'profiles_frozen_before_iid':True,'bundle_sha256':bundle_hash,'profiles_sha256':profiles_hash,
        'protected_files':len(protected),'unit_checks_passed':len(checks),'runtime':runtime_info()}
    atomic_json(output/'summary.json',summary)
    atomic_json(output/'provenance.json',{'protected_sha256':protected,'source_sha256':source_hashes,
        'bundle_sha256':bundle_hash,'argv':sys.argv,'baseline_residual_states_unchanged':True})
    print(json.dumps({'status':summary['status'],'primary_budget_met':budget_met,'iid':results}),flush=True)


if __name__=='__main__':main()
