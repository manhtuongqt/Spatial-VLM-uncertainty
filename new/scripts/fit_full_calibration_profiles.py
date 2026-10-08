#!/usr/bin/env python3
"""Optional post-freeze fit-only profiles; never reuse prior audit as validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pcrau.calibration import read_jsonl
from pcrau.selective_experiment import apply_risk, choose_policy_threshold, fit_risk, policy_result, predicted_answer
from pcrau.utils import atomic_json, sha256_file, workspace_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True)
    args=parser.parse_args()
    root=Path(args.run_root).resolve()
    lock=json.loads((root/'frozen/freeze_lock.json').read_text())
    assert sha256_file(Path(lock['checkpoint']))==lock['checkpoint_sha256']
    chosen=json.loads((root/'frozen/calibrator_selection.json').read_text())['selected']
    assert chosen in {'logistic18','logistic33'}
    output=root/'calibration_full_fit';output.mkdir(exist_ok=False)
    names=['v2_reference','selected_model']
    paths=[root/'frozen/calibration_reference_predictions.jsonl',root/'frozen/calibration_selected_predictions.jsonl']
    profiles=[];calibrators={}
    for name,path in zip(names,paths):
        rows=read_jsonl(path)
        assert len(rows)==1000 and len({r['family_id'] for r in rows})==200
        extended=name=='selected_model' and chosen=='logistic33'
        calibrator,crossfit=fit_risk(rows,extended,0.01 if extended else 0.001)
        calibrators[name]=calibrator
        atomic_json(output/f'{name}_calibrator.json',calibrator)
        final_risk=apply_risk(rows,calibrator)
        error=np.asarray([r['evaluation']['error_event'] for r in rows])
        with (output/f'{name}_risk_scores.jsonl').open('x',encoding='utf-8') as handle:
            for i,row in enumerate(rows):
                handle.write(json.dumps({'sample_id':row['sample_id'],'family_id':row['family_id'],
                    'crossfit_risk':float(crossfit[i]),'fit_all_risk':float(final_risk[i]),
                    'error_event':bool(error[i])})+'\n')
        for policy in ['hard_found','risk_only']:
            eligible=np.ones(1000,bool) if policy=='risk_only' else np.asarray([predicted_answer(r)=='FOUND' for r in rows])
            for target in [0.05,0.075,0.1]:
                threshold=choose_policy_threshold(crossfit,error,[r['family_id'] for r in rows],eligible,target,60)
                oof,_=policy_result(rows,crossfit,policy,threshold['threshold'])
                fit_all,_=policy_result(rows,final_risk,policy,threshold['threshold'])
                profiles.append({'model':name,'policy':policy,'risk_target':target,'threshold_fit':threshold,
                                 'crossfit_calibration':oof,'resubstitution_fit_all':fit_all})
    result={'status':'CALIBRATION_FIT_ONLY','profiles':profiles,'checkpoint_sha256':lock['checkpoint_sha256'],
            'calibrator_sha256':{name:sha256_file(output/f'{name}_calibrator.json') for name in names},
            'calibrator_cv_metrics':{name:c['crossfit_metrics'] for name,c in calibrators.items()},
            'prior_audit_reused_as_independent_validation':False,'test_splits_accessed':False,
            'protocol_sha256':sha256_file(workspace_path('new/docs/EXPERIMENT_FULL_CALIBRATION_PROFILES.md')),
            'requires_new_independent_test':True}
    atomic_json(output/'profiles.json',result)
    lines=['# Profile toàn bộ calibration — chỉ fit/crossfit, chưa có test độc lập','',
           '| Model | Policy | Budget | Ngưỡng OOF | Nhận OOF | Coverage OOF | Lỗi OOF | Risk OOF | Bypass đúng/sai OOF | Nhận fit-all | Lỗi fit-all |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in profiles:
        m=p['crossfit_calibration'];f=p['resubstitution_fit_all'];t=p['threshold_fit']['threshold']
        tau='None' if t is None else f'{t:.6f}'
        rate='N/A' if m['empirical_risk'] is None else f"{m['empirical_risk']:.2%}"
        lines.append(f"| {p['model']} | {p['policy']} | {p['risk_target']:.1%} | {tau} | {m['accepted']} | {m['coverage']:.2%} | {m['errors']} | {rate} | {m['bypass_correct']}/{m['bypass_errors']} | {f['accepted']} | {f['errors']} |")
    lines += ['', 'Mẫu số 1000 calibration sample/200 family. Ngưỡng chọn trên risk OOF theo family; counts OOF chịu việc chọn ngưỡng trên cùng dữ liệu.',
              'Fit-all counts là resubstitution, không phải ước lượng generalization. Không lấy audit cũ đánh giá profile mới.',
              'Không thay chính thức model/calibrator/policy V2 hoặc số liệu Test-IID. Các profile cần test độc lập mới và không phải risk robot.']
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'profiles':len(profiles),'status':'CALIBRATION_FIT_ONLY'}),flush=True)


if __name__=='__main__':main()
