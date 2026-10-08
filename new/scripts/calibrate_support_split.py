#!/usr/bin/env python3
"""Freeze calibrator on fit families, select on threshold families, audit once."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from pcrau.calibration import fit_logistic
from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset, FamilyBatchSampler, collate_samples
from torch.utils.data import DataLoader
from pcrau.engine import evaluate
from pcrau.losses import class_weights
from pcrau.model import PCRAUTargetV2
from pcrau.support_evidence import load_support_scorer
from pcrau.selective_experiment import apply_risk, choose_policy_threshold, policy_result, predicted_answer, probability_metrics
from pcrau.utils import atomic_json, load_config, sha256_file, seed_everything


def write_rows(path, rows):
    with path.open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',default='new/outputs/pcrau_support_split_dev_20261003');args=p.parse_args()
    root=Path(args.run_root).resolve(); lock=json.loads((root/'freeze_lock.json').read_text())
    for key in ['source_checkpoint','config','ranker_checkpoint']:
        assert sha256_file(Path(lock[key]))==lock[{'source_checkpoint':'source_checkpoint_sha256','config':'config_sha256','ranker_checkpoint':'ranker_sha256'}[key]]
    out=root/'split_calibration'
    if out.exists():
        if {p.name for p in out.iterdir()}!={'model_freeze_lock.json'} or json.loads((out/'model_freeze_lock.json').read_text())!=lock:
            raise FileExistsError('Calibration outputs already exist; do not overwrite')
    else:out.mkdir()
    atomic_json(out/'model_freeze_lock.json',lock)
    config=load_config(lock['config']);seed_everything(24082026);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PCRAUTargetV2(config).to(device).eval();load_model_checkpoint(model,lock['source_checkpoint'],lock['config_sha256'])
    model=load_support_scorer(model,lock['ranker_checkpoint'],lock['source_checkpoint_sha256'])
    config['runtime_checkpoint_sha256']=lock['source_checkpoint_sha256']
    data=ArchivedPCRAUDataset(config,'calibration',profile='calibration');families=sorted({r['family_id'] for r in data.entries},key=lambda f:hashlib.sha256(('20261003|'+f).encode()).hexdigest())
    partitions={f:('fit' if i<100 else 'threshold' if i<160 else 'audit') for i,f in enumerate(families)}
    assert len(partitions)==200
    atomic_json(out/'partitions.json',{'seed':20261003,'counts':{'fit':100,'threshold':60,'audit':40},'families':partitions,'historically_observed_calibration':True})
    sampler=FamilyBatchSampler(data,4,24082026,False);loader=DataLoader(data,batch_sampler=sampler,num_workers=0,collate_fn=collate_samples)
    train=ArchivedPCRAUDataset(config,'train');weights,source_weights=class_weights(train,device)
    print('FROZEN support scorer; extracting calibration predictions',flush=True)
    metrics,rows=evaluate(model,loader,device,config,weights,source_weights)
    write_rows(out/'predictions.jsonl',rows);atomic_json(out/'model_metrics.json',metrics)
    original={r['sample_id']:r for r in [json.loads(l) for l in Path('new/outputs/pcrau_answerability_language_dev_20261003/calibration_full_fit/predictions.jsonl').open()]}
    for r in rows:
        for k in ['spatial','evaluation','answerability_probabilities','source_probabilities','relation_edge_probabilities']:
            assert r[k]==original[r['sample_id']][k],(r['sample_id'],k)
    fit=[r for r in rows if partitions[r['family_id']]=='fit']
    x=np.array([[r['ranker_error_logit']] for r in fit]); y=np.array([r['evaluation']['error_event'] for r in fit],float)
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1
    coef,intercept=fit_logistic((x-mean)/scale,y,.01)
    cal={'method':'logistic_support_scalar_split','extended':False,'ranker_feature':True,'feature_names':['ranker_error_logit'],
        'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':coef.tolist(),'intercept':intercept,'l2':.01,
        'fit_families':100,'fit_samples':500,'ranker_sha256':lock['ranker_sha256'],'source_checkpoint_sha256':lock['source_checkpoint_sha256']}
    atomic_json(out/'selected_calibrator.json',cal)
    risk=apply_risk(rows,cal)
    threshold_rows=[r for r in rows if partitions[r['family_id']]=='threshold']
    threshold_scores=np.array([v for r,v in zip(rows,risk) if partitions[r['family_id']]=='threshold'])
    error=np.array([r['evaluation']['error_event'] for r in threshold_rows]);eligible=np.array([predicted_answer(r)=='FOUND' for r in threshold_rows])
    threshold=choose_policy_threshold(threshold_scores,error,[r['family_id'] for r in threshold_rows],eligible,.075,20)
    threshold_lock={**lock,'calibrator_sha256':sha256_file(out/'selected_calibrator.json'),'partition_sha256':sha256_file(out/'partitions.json'),
        'threshold_fit':threshold,'refit_all_calibration':False,'audit_opened_before_threshold_lock':False}
    atomic_json(out/'threshold_lock.json',threshold_lock)
    results={}
    for name in ['fit','threshold','audit']:
        rr=[r for r in rows if partitions[r['family_id']]==name];ss=np.array([v for r,v in zip(rows,risk) if partitions[r['family_id']]==name])
        m,dec=policy_result(rr,ss,'hard_found',threshold['threshold'])
        m['valid_acceptance_recall']=m['correct_acceptances']/m['correct_found_total']
        m['probability_metrics']=probability_metrics(ss,[r['evaluation']['error_event'] for r in rr])
        m['by_variant']={v:{'accepted':sum(r['action']=='EXECUTE' for r,b in zip(dec,rr) if b['variant']==v),
            'errors':sum(r['action']=='EXECUTE' and r['evaluation']['error_event'] for r,b in zip(dec,rr) if b['variant']==v)} for v in sorted({r['variant'] for r in rr})}
        results[name]=m;write_rows(out/f'{name}_decisions.jsonl',dec)
    audit_pass=results['audit']['wilson_upper_95'] is not None and results['audit']['wilson_upper_95']<=.075
    results['audit']['budget_met_by_empirical_risk']=results['audit']['empirical_risk'] is not None and results['audit']['empirical_risk']<=.075
    # The audit is historically seen data; even PASS would not be independent confirmation.
    profile={'status':'SPLIT_CALIBRATION_AUDIT_PASS' if audit_pass else 'SPLIT_CALIBRATION_AUDIT_NOT_CONFIRMED',
        'profiles':[{'model':'selected_model','policy':'hard_found','risk_target':.075,'threshold_fit':threshold}],
        'default_policy':'hard_found','default_risk_target':.075,'prediction_checkpoint_sha256':lock['source_checkpoint_sha256'],
        'calibrator_sha256':{'selected_model':sha256_file(out/'selected_calibrator.json')},'ranker_checkpoint_sha256':lock['ranker_sha256'],
        'partitions':results,'same_calibrator_for_threshold_audit_and_runtime':True,'refit_all_calibration':False,
        'independent_iid_confirmation_required':True,'historical_test_not_reopened':True,'robot_motion_commanded':False}
    atomic_json(out/'profiles.json',profile)
    atomic_json(out/'active_experimental_profile.json',{'status':profile['status'],'checkpoint':lock['source_checkpoint'],
        'config':lock['config'],'support_ranker':lock['ranker_checkpoint'],'calibrator':str(out/'selected_calibrator.json'),
        'profiles':str(out/'profiles.json'),'policy':'hard_found','risk_target':.075,'risk_threshold':threshold['threshold'],
        'official_v2_replaced':False,'robot_motion_commanded':False})
    for f,h in json.loads((root/'run.json').read_text())['protected_files'].items():assert sha256_file(Path(f))==h
    for f,k in [('selected_calibrator.json','calibrator_sha256'),('partitions.json','partition_sha256')]:assert sha256_file(out/f)==threshold_lock[k]
    print(json.dumps({'status':profile['status'],'threshold':threshold,'partitions':results},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
