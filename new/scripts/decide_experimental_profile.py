#!/usr/bin/env python3
"""Offline experimental decision labels; no robot interface or evaluator input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcrau.calibration import read_jsonl
from pcrau.selective_experiment import apply_risk, experimental_action, predicted_answer
from pcrau.utils import sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--calibrator',required=True)
    parser.add_argument('--profiles',required=True)
    parser.add_argument('--model',default='selected_model',choices=['selected_model','v2_reference'])
    parser.add_argument('--policy',default=None,choices=['hard_found','risk_only'])
    parser.add_argument('--risk-target',type=float,default=None,choices=[0.05,0.075,0.1])
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    calibrator_path=Path(args.calibrator).resolve()
    calibrator=json.loads(calibrator_path.read_text())
    profiles=json.loads(Path(args.profiles).read_text())
    risk_target=args.risk_target if args.risk_target is not None else profiles.get('default_risk_target',0.05)
    policy=args.policy if args.policy is not None else profiles.get('default_policy','risk_only')
    if sha256_file(calibrator_path)!=profiles['calibrator_sha256'][args.model]:
        raise ValueError('Calibrator does not match the locked model profile')
    selected=[p for p in profiles['profiles'] if p['model']==args.model and p['policy']==policy
              and abs(p['risk_target']-risk_target)<1e-12]
    if len(selected)!=1:
        raise ValueError('Expected exactly one locked profile')
    threshold=selected[0]['threshold_fit']['threshold']
    rows=read_jsonl(args.predictions)
    # Drop evaluator/variant fields before risk or decision. Model outputs only.
    inputs=[{key:value for key,value in row.items() if key not in {'evaluation','variant'}} for row in rows]
    expected_checkpoint=profiles.get('prediction_checkpoint_sha256')
    if expected_checkpoint and any(row.get('model_checkpoint_sha256')!=expected_checkpoint for row in inputs):
        raise ValueError('Predictions do not match the locked answerability checkpoint')
    if calibrator.get('ranker_feature',False):
        expected=profiles.get('ranker_checkpoint_sha256')
        if not expected or calibrator.get('ranker_sha256')!=expected:
            raise ValueError('Ranker does not match the locked calibrator/profile')
        source_key = ('ranker_source_checkpoint_sha256' if calibrator.get('method') == 'logistic_support_scalar_split'
                      else 'source_v2_sha256')
        expected_source = calibrator.get('source_checkpoint_sha256') if source_key == 'ranker_source_checkpoint_sha256' else calibrator.get('source_v2_sha256')
        if any(row.get('ranker_sha256')!=expected or row.get(source_key)!=expected_source for row in inputs):
            raise ValueError('Predictions were not exported by the matching frozen V2/ranker')
    risk=apply_risk(inputs,calibrator)
    output=Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as handle:
        for row,value in zip(inputs,risk):
            action=experimental_action(row,float(value),policy,threshold)
            decision={'sample_id':row['sample_id'],'action':action,'risk':float(value),
                      'risk_threshold':threshold,'risk_target':risk_target,'policy':policy,
                      'predicted_answerability':predicted_answer(row),
                      'selected_point_xy':row['spatial']['map_pixel_xy'] if action=='EXECUTE' else None,
                      'status':'EXPERIMENTAL_CALIBRATION_FIT_ONLY','robot_motion_commanded':False}
            handle.write(json.dumps(decision,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(output),'decisions':len(rows),'robot_motion_commanded':False}))


if __name__=='__main__':main()
