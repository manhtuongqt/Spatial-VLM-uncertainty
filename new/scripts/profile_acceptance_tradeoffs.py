#!/usr/bin/env python3
"""Lock 5/7.5/10% offline profiles from previously frozen calibration scores."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from pcrau.selective_experiment import choose_policy_threshold, policy_result, predicted_answer
from pcrau.utils import atomic_json, sha256_file


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def compare(reference_records, candidate_records):
    old = {r['sample_id']:r for r in reference_records if r['action']=='EXECUTE'}
    new = {r['sample_id']:r for r in candidate_records if r['action']=='EXECUTE'}
    added, removed = set(new)-set(old), set(old)-set(new)
    return {'added':len(added),'added_correct':sum(not new[k]['evaluation']['error_event'] for k in added),
            'added_errors':sum(new[k]['evaluation']['error_event'] for k in added),
            'removed':len(removed),'removed_correct':sum(not old[k]['evaluation']['error_event'] for k in removed),
            'removed_errors':sum(old[k]['evaluation']['error_event'] for k in removed)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True)
    args=parser.parse_args()
    root=Path(args.run_root).resolve()
    source=root/'calibration_full_fit'
    lock=json.loads((source/'freeze_lock.json').read_text())
    if sha256_file(Path(lock['ranker_checkpoint']))!=lock['ranker_sha256']:
        raise ValueError('Frozen ranker changed')
    output=root/'risk_tradeoffs'
    output.mkdir(exist_ok=False)
    rows=read_rows(source/'ranker_predictions.jsonl')
    assert len(rows)==1000 and len({r['family_id'] for r in rows})==200
    error=np.array([r['evaluation']['error_event'] for r in rows])
    profiles=[];counts={};decisions={};score_metrics={};hashes={}
    protected={str(p):sha256_file(p) for p in [source/'ranker_predictions.jsonl',source/'freeze_lock.json',Path(lock['ranker_checkpoint'])]}
    for model,filename in [('v2_reference','reference_logistic33'),('selected_model','ranker_logistic34')]:
        cal_path=source/f'{filename}_calibrator.json'
        cal=json.loads(cal_path.read_text())
        if model=='selected_model' and cal['ranker_sha256']!=lock['ranker_sha256']:
            raise ValueError('Calibrator ranker mismatch')
        destination=output/f'{model}_calibrator.json'
        shutil.copyfile(cal_path,destination)
        hashes[model]=sha256_file(destination)
        score_rows=read_rows(source/f'{filename}_scores.jsonl')
        assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in score_rows]
        crossfit=np.array([r['crossfit_risk'] for r in score_rows])
        fit_all=np.array([r['fit_all_risk'] for r in score_rows])
        score_metrics[model]=cal['crossfit_metrics']
        for policy in ['hard_found','risk_only']:
            eligible=np.ones(1000,bool) if policy=='risk_only' else np.array([predicted_answer(r)=='FOUND' for r in rows])
            for budget in [.05,.075,.1]:
                threshold=choose_policy_threshold(crossfit,error,[r['family_id'] for r in rows],eligible,budget,60)
                metrics,records=policy_result(rows,crossfit,policy,threshold['threshold'])
                metrics['valid_acceptance_recall']=metrics['correct_acceptances']/metrics['correct_found_total']
                accepted=[r for r in records if r['action']=='EXECUTE']
                metrics['accepted_error_truth_states']=dict(Counter(r['evaluation']['answerability_state'] for r in accepted if r['evaluation']['error_event']))
                metrics['accepted_point_hits']=sum(r['evaluation']['map_inside_target'] for r in accepted)
                resub,_=policy_result(rows,fit_all,policy,threshold['threshold'])
                profile={'model':model,'policy':policy,'risk_target':budget,'threshold_fit':threshold,
                         'crossfit_calibration':metrics,'resubstitution_fit_all':resub}
                profiles.append(profile)
                counts[model,policy,budget]=metrics
                decisions[model,policy,budget]=records
                with (output/f'{model}_{policy}_{budget:.3f}_oof_decisions.jsonl').open('x') as handle:
                    for row in records:handle.write(json.dumps(row)+'\n')
    comparisons={}
    for budget in [.05,.075,.1]:
        old,new=counts['v2_reference','risk_only',budget],counts['selected_model','risk_only',budget]
        comparisons[f'new_vs_reference_{budget}']={**compare(decisions['v2_reference','risk_only',budget],decisions['selected_model','risk_only',budget]),
                    'net_correct_gain':new['correct_acceptances']-old['correct_acceptances'],'net_error_gain':new['errors']-old['errors']}
    for low,high in [(.05,.075),(.075,.1),(.05,.1)]:
        old,new=counts['selected_model','risk_only',low],counts['selected_model','risk_only',high]
        extra_accepted=new['accepted']-old['accepted'];extra_error=new['errors']-old['errors']
        comparisons[f'new_budget_{low}_to_{high}']={'extra_correct':new['correct_acceptances']-old['correct_acceptances'],
                    'extra_errors':extra_error,'extra_accepted':extra_accepted,'marginal_error_rate':extra_error/extra_accepted if extra_accepted else None}
    result={'status':'CALIBRATION_FIT_ONLY','profiles':profiles,'default_risk_target':.1,'default_policy':'risk_only',
            'default_is_experimental_not_robot_deployment':True,'user_authorized_budgets':[.05,.075,.1],
            'checkpoint_sha256':lock['source_v2_sha256'],'ranker_checkpoint_sha256':lock['ranker_sha256'],
            'calibrator_sha256':hashes,'calibrator_cv_metrics':score_metrics,'comparisons':comparisons,
            'model_descriptions':{'v2_reference':'Frozen V2 + logistic33 (same calibration folds)',
                                  'selected_model':'Frozen V2 + train-only residual ranker + logistic34'},
            'prior_audit_reused_as_independent_validation':False,'test_splits_accessed':False,
            'requires_new_independent_test':True,'robot_motion_commanded':False,'script_sha256':sha256_file(Path(__file__))}
    atomic_json(output/'profiles.json',result)
    with (output/'inference_predictions.jsonl').open('x') as handle:
        for row in rows:
            observed={k:v for k,v in row.items() if k not in {'evaluation','variant'}}
            observed.update({'ranker_sha256':lock['ranker_sha256'],'source_v2_sha256':lock['source_v2_sha256']})
            handle.write(json.dumps(observed)+'\n')
    selected_profile=next(p for p in profiles if p['model']=='selected_model' and p['policy']=='risk_only' and p['risk_target']==.1)
    atomic_json(output/'active_experimental_profile.json',{
        'status':'EXPERIMENTAL_CALIBRATION_FIT_ONLY','risk_target':.1,'policy':'risk_only',
        'risk_threshold':selected_profile['threshold_fit']['threshold'],
        'ranker_checkpoint':lock['ranker_checkpoint'],'ranker_sha256':lock['ranker_sha256'],
        'calibrator':str(output/'selected_model_calibrator.json'),'profiles':str(output/'profiles.json'),
        'source_v2_sha256':lock['source_v2_sha256'],'robot_motion_commanded':False,'official_v2_replaced':False})
    assert protected=={p:sha256_file(Path(p)) for p in protected}
    lines=['# Đánh đổi chấp nhận của acceptance residual', '',
           'Calibration crossfit 1000 sample/200 family; threshold chọn trên cùng OOF, chưa test độc lập.', '',
           '| Model | Budget | Ngưỡng risk | Nhận đúng / 411 | Recall hợp lệ | Nhận | Lỗi | Risk thực nghiệm | Wilson upper |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in ['v2_reference','selected_model']:
        for budget in [.05,.075,.1]:
            m=counts[model,'risk_only',budget]
            p=next(p for p in profiles if p['model']==model and p['policy']=='risk_only' and p['risk_target']==budget)
            lines.append(f"| {model} | {budget:.1%} | {p['threshold_fit']['threshold']:.6f} | {m['correct_acceptances']}/411 | {m['valid_acceptance_recall']:.2%} | {m['accepted']} | {m['errors']} | {m['empirical_risk']:.2%} | {m['wilson_upper_95']:.2%} |")
    lines += ['', '## Tăng ngưỡng trên nhánh mới', '', '| Chuyển budget | Thêm đúng | Thêm lỗi | Lỗi trong nhóm nhận thêm |', '|---|---:|---:|---:|']
    for low,high in [(.05,.075),(.075,.1),(.05,.1)]:
        m=comparisons[f'new_budget_{low}_to_{high}']
        lines.append(f"| {low:.1%} → {high:.1%} | {m['extra_correct']} | {m['extra_errors']} | {m['marginal_error_rate']:.2%} |")
    lines += ['', 'Default thử nghiệm: budget 10%, theo việc người dùng chấp nhận đánh đổi tới 10%.',
              'Không phải thay V2 chính thức hoặc cho phép robot chuyển động. Macro-F1/grounding V2 không đổi.',
              'Risk tổng hợp là non-FOUND thật hoặc MAP sai; không phải mọi lỗi đều là sai điểm hoặc robot thất bại.',
              'Hard-FOUND và risk-only phải được đọc riêng; hiện các budget chưa bypass cổng answerability.',
              'Fit-all là resubstitution, không dùng làm số liệu khái quát. Cần đánh giá độc lập mới.', '',
              'AURC reference/new: '+str(score_metrics['v2_reference']['aurc'])+' / '+str(score_metrics['selected_model']['aurc'])+'.']
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'default_budget':.1,'default_threshold':selected_profile['threshold_fit']['threshold'],
                      'metrics':selected_profile['crossfit_calibration']}),flush=True)


if __name__=='__main__':main()
