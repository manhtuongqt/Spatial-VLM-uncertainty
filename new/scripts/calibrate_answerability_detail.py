#!/usr/bin/env python3
"""Post-freeze fit-only calibration of a train/dev selected answerability head."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset, ANSWER_CLASSES
from pcrau.engine import evaluate
from pcrau.losses import class_weights
from pcrau.model import PCRAUTargetV2
from pcrau.selective_experiment import apply_risk, choose_policy_threshold, fit_risk, policy_result, predicted_answer
from pcrau.utils import atomic_json, load_config, seed_everything, sha256_file, workspace_path
from experiment_answerability_head import make_loader
from experiment_hard_cases_selective import write_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',required=True)
    args=parser.parse_args();root=Path(args.run_root).resolve()
    lock=json.loads((root/'freeze_lock.json').read_text())
    checkpoint=Path(lock['checkpoint']);config_path=Path(lock['config'])
    assert sha256_file(checkpoint)==lock['checkpoint_sha256'] and sha256_file(config_path)==lock['config_sha256']
    assert not lock['calibration_opened_before_freeze']
    output=root/'calibration_full_fit';output.mkdir(exist_ok=False)
    atomic_json(output/'freeze_lock.json',{**lock,'calibration_is_fit_only_not_independent_audit':True,
                 'script_sha256':sha256_file(Path(__file__))})
    source=workspace_path('new/outputs/pcrau_target_v2_full_seed_24082026')
    protected=[source/'config.json',source/'checkpoints/best/model.safetensors',source/'evaluation/calibration/calibrator.json',
               workspace_path('new/hinhanh/19_answerability_confusion/fig19_answerability_confusion_test_iid.png')]
    hashes={str(p):sha256_file(p) for p in protected}
    config=load_config(config_path)
    seed_everything(24082026);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PCRAUTargetV2(config).to(device).eval()
    load_model_checkpoint(model,checkpoint,lock['config_sha256'])
    model.requires_grad_(False)
    config['runtime_checkpoint_sha256']=lock['checkpoint_sha256']
    train=ArchivedPCRAUDataset(config,'train');calibration=ArchivedPCRAUDataset(config,'calibration',profile='calibration')
    assert len(calibration)==1000 and len({e['family_id'] for e in calibration.entries})==200
    assert {e['family_id'] for e in train.entries}.isdisjoint(e['family_id'] for e in calibration.entries)
    loader,_=make_loader(calibration,config,False);weights,source_weights=class_weights(train,device)
    print('FROZEN answerability head: calibration fit/crossfit only',flush=True)
    metrics,rows=evaluate(model,loader,device,config,weights,source_weights)
    write_rows(output/'predictions.jsonl',rows);atomic_json(output/'model_metrics.json',metrics)
    reference_root=workspace_path('new/outputs/pcrau_acceptance_residual_dev_20261003/calibration_full_fit')
    reference_rows=[json.loads(s) for s in (reference_root/'ranker_predictions.jsonl').read_text().splitlines()]
    assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in reference_rows]
    for row,before in zip(rows,reference_rows):
        assert row['spatial']['map_pixel_xy']==before['spatial']['map_pixel_xy']
        assert row['source_probabilities']==before['source_probabilities']
        assert row['relation_edge_probabilities']==before['relation_edge_probabilities']
        assert row['evaluation']==before['evaluation']
    baseline_truth=[r['evaluation']['answerability_index'] for r in reference_rows]
    from experiment_answerability_head import answer_metrics
    before=answer_metrics(baseline_truth,[ANSWER_CLASSES.index(predicted_answer(r)) for r in reference_rows])
    after=answer_metrics([r['evaluation']['answerability_index'] for r in rows],
                        [ANSWER_CLASSES.index(predicted_answer(r)) for r in rows])
    candidates={};oof_scores={}
    for name,extended,l2 in [('logistic18',False,.001),('logistic33',True,.01)]:
        calibrator,scores=fit_risk(rows,extended,l2)
        calibrator['model_checkpoint_sha256']=lock['checkpoint_sha256']
        candidates[name]=calibrator;oof_scores[name]=scores
        atomic_json(output/f'{name}.json',calibrator)
    chosen=min(candidates,key=lambda n:(candidates[n]['crossfit_metrics']['brier'],len(candidates[n]['feature_names'])))
    calibrator=candidates[chosen];scores=oof_scores[chosen];fit_all=apply_risk(rows,calibrator)
    atomic_json(output/'selected_calibrator.json',calibrator)
    error=np.array([r['evaluation']['error_event'] for r in rows]);families=[r['family_id'] for r in rows]
    with (output/'risk_scores.jsonl').open('x') as handle:
        for row,value,fit in zip(rows,scores,fit_all):
            handle.write(json.dumps({'sample_id':row['sample_id'],'family_id':row['family_id'],'crossfit_risk':float(value),
                                    'fit_all_risk':float(fit),'error_event':row['evaluation']['error_event']})+'\n')
    profiles=[]
    for policy in ['hard_found','risk_only']:
        eligible=np.ones(1000,bool) if policy=='risk_only' else np.array([predicted_answer(r)=='FOUND' for r in rows])
        for budget in [.05,.075,.1]:
            threshold=choose_policy_threshold(scores,error,families,eligible,budget,60)
            result,decisions=policy_result(rows,scores,policy,threshold['threshold'])
            result['valid_acceptance_recall']=result['correct_acceptances']/result['correct_found_total']
            resub,_=policy_result(rows,fit_all,policy,threshold['threshold'])
            profiles.append({'model':'selected_model','policy':policy,'risk_target':budget,'threshold_fit':threshold,
                             'crossfit_calibration':result,'resubstitution_fit_all':resub})
            write_rows(output/f'{policy}_{budget:.3f}_oof_decisions.jsonl',decisions)
    safe=[r for r in rows if not r['evaluation']['error_event']]
    missed_before=sum(predicted_answer(r)!='FOUND' for r in reference_rows if not r['evaluation']['error_event'])
    missed_after=sum(predicted_answer(r)!='FOUND' for r in safe)
    previous=json.loads((reference_root.parent/'risk_tradeoffs/profiles.json').read_text())
    comparisons={}
    for p in profiles:
        old=next(x for x in previous['profiles'] if x['model']=='selected_model' and x['policy']==p['policy'] and x['risk_target']==p['risk_target'])
        a,b=old['crossfit_calibration'],p['crossfit_calibration']
        comparisons[f"{p['policy']}_{p['risk_target']}"]={'old_correct':a['correct_acceptances'],'new_correct':b['correct_acceptances'],
            'old_errors':a['errors'],'new_errors':b['errors'],'correct_gain':b['correct_acceptances']-a['correct_acceptances'],
            'error_gain':b['errors']-a['errors']}
    result={'status':'CALIBRATION_FIT_ONLY','answerability_before':before,'answerability_after':after,
            'valid_cases_misclassified_before':missed_before,'valid_cases_misclassified_after':missed_after,
            'grounding_source_edge_unchanged':True,'selected_calibrator':chosen,
            'calibrator_cv_metrics':{n:c['crossfit_metrics'] for n,c in candidates.items()},'profiles':profiles,
            'comparisons_to_previous_ranker34':comparisons,'default_risk_target':.075,'default_policy':'hard_found',
            'operating_choice_after_calibration_fit':True,
            'operating_choice_reason':'7.5% already improves accepted valid cases and errors over previous 10% profile; retain 10% comparator',
            'calibrator_sha256':{'selected_model':sha256_file(output/'selected_calibrator.json')},
            'prediction_checkpoint_sha256':lock['checkpoint_sha256'],'requires_new_independent_test':True,
            'test_splits_accessed':False,'robot_motion_commanded':False,'official_v2_replaced':False}
    atomic_json(output/'profiles.json',result)
    active=next(p for p in profiles if p['policy']=='hard_found' and p['risk_target']==.075)
    atomic_json(output/'active_experimental_profile.json',{'config':str(config_path),'checkpoint':str(checkpoint),
                'calibrator':str(output/'selected_calibrator.json'),'profiles':str(output/'profiles.json'),
                'policy':'hard_found','risk_target':.075,'risk_threshold':active['threshold_fit']['threshold'],
                'status':'CALIBRATION_FIT_ONLY','official_v2_replaced':False,'robot_motion_commanded':False})
    assert hashes=={str(p):sha256_file(p) for p in protected}
    assert sha256_file(checkpoint)==lock['checkpoint_sha256']
    lines=['# Answerability detail + augmentation: calibration fit-only','','Không phải test hoặc audit độc lập.', '',
           '| Answerability | Macro-F1 | Accuracy | FOUND recall | False FOUND | ABSENT→FOUND |',
           '|---|---:|---:|---:|---:|---:|']
    for name,m in [('V2',before),('Adapter mới',after)]:
        lines.append(f"| {name} | {m['macro_f1']:.6f} | {m['accuracy']:.2%} | {m['per_class'][0]['recall']:.2%} | {m['false_found']} | {m['confusion_matrix'][2][0]} |")
    lines += ['',f'Ca hợp lệ bị sai answerability: {missed_before} → {missed_after} trên411 ca.', '',
              '| Policy | Budget | Nhận đúng / 411 | Recall hợp lệ | Nhận | Lỗi | Risk | Bypass đúng/sai |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for p in profiles:
        m=p['crossfit_calibration'];rate='N/A' if m['empirical_risk'] is None else f"{m['empirical_risk']:.2%}"
        lines.append(f"| {p['policy']} | {p['risk_target']:.1%} | {m['correct_acceptances']}/411 | {m['valid_acceptance_recall']:.2%} | {m['accepted']} | {m['errors']} | {rate} | {m['bypass_correct']}/{m['bypass_errors']} |")
    lines += ['',f'Calibrator chọn theo calibration CV Brier: {chosen}. Không chọn lại head/epoch.',
              'V2 grounding/source/edge và mọi annotation evaluator bất biến đã đối chiếu.',
              'Default vận hành thử nghiệm hard_found/budget7.5% sau đối chiếu calibration; giữ10% comparator. Không chọn lại head/epoch.',
              'Không thay model/calibrator chính, hình hoặc LaTeX. Các lỗi còn lại không được tuyên bố đã giải quyết hết.']
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'answer_before':before,'answer_after':after,
                      'valid_misclassified_before':missed_before,'valid_misclassified_after':missed_after,'profiles':profiles}),flush=True)


if __name__=='__main__':main()
