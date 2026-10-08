#!/usr/bin/env python3
"""Read-only artifact audit and saved-row risk replay; no fit or model forward."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from pcrau.calibration import predict_risk
from pcrau.selective_experiment import experimental_action, policy_result, probability_metrics
from pcrau.unified_inference import attach_decisions, VERSION
from pcrau.utils import atomic_json, read_json, sha256_file
from pcrau.verifier_risk import apply_verifier_risk, risk_vector


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    output = root/'final_verification.json'
    if output.exists(): raise FileExistsError(output)
    before = {str(p): sha256_file(p) for p in root.rglob('*') if p.is_file()}
    def check(desc, base=root):
        path = Path(desc['path']); path = path if path.is_absolute() else base/path
        assert sha256_file(path) == desc['sha256'], path
        return path
    manifest = read_json(root/'bundle.json')
    lock_path = check(manifest['freeze_lock']); lock = read_json(lock_path)
    bundle_id = sha256_file(lock_path)
    for key in ('config','baseline_checkpoint','baseline_metadata','anchor_residual',
                'residual_source','protocol','feature_manifest'):
        check(lock[key])
    for desc in lock['source_files'].values(): check(desc)
    for desc in read_json(root/'data_provenance.json').values(): check(desc)
    calibrator = read_json(check(manifest['calibrator'])); profile = read_json(check(manifest['profile']))
    assert manifest['version']==VERSION and calibrator['bundle_sha256']==bundle_id
    assert profile['bundle_sha256']==bundle_id and calibrator['evidence_producer']==VERSION
    summary = read_json(root/'summary.json')
    tests = []; torch.set_num_threads(4)
    workspace = Path(__file__).resolve().parents[2]
    for relative in ('new/tests/test_unified_inference.py','new/tests/test_horizontal_verifier.py'):
        for name, fn in runpy.run_path(str(workspace/relative)).items():
            if name.startswith('test_') and callable(fn): fn(); tests.append(name)
    geometry = {}; counts = {}
    for split, expected in (('train',1600),('dev',400),('calibration',1000)):
        observable = read_rows(root/split/'observable.jsonl')
        runtime = read_rows(root/split/'runtime_predictions.jsonl')
        evaluated = read_rows(root/split/'evaluator_join.jsonl')
        assert len(observable)==len(runtime)==len(evaluated)==expected
        assert all('evaluation' not in r and 'variant' not in r for r in runtime)
        replay = attach_decisions(observable, calibrator, profile)
        assert all(a['decision']==b['decision'] and a['risk_trace']==b['risk_trace']
                   for a,b in zip(replay,runtime))
        x = np.stack([risk_vector(r,'P1_G44') for r in runtime])
        assert np.isfinite(x).all() and x.shape==(expected,44)
        for r in runtime:
            v = r['verifiers']['P1']
            assert v['binding_status']=='UNVERIFIED' and v['presence_status']=='UNVERIFIED'
            assert not v['target_map_changed'] and not r['decision']['robot_motion_commanded']
            if v['scope_status']!='SUPPORTED_DIAGNOSTIC':
                assert v['pair_compatibility'] is None and v['anchor_peak_xy'] is None
                assert (risk_vector(r,'P1_G44')[-3:]==0).all()
        risk = apply_verifier_risk(observable,calibrator)
        metrics,_ = policy_result(evaluated,risk,'hard_found',profile['threshold'])
        assert metrics==summary['split_policy'][split]
        neutral = x.copy(); neutral[:,-3:]=0.
        neutral_risk = predict_risk((neutral-np.asarray(calibrator['mean']))/np.asarray(calibrator['scale']),
                                    np.asarray(calibrator['coefficients']),calibrator['intercept'])
        changed = []
        for r, n in zip(runtime,neutral_risk):
            action = experimental_action(r,float(n),'hard_found',profile['threshold'])
            if action!=r['decision']['action']:
                changed.append({'sample_id':r['sample_id'],'current_action':r['decision']['action'],
                                'geometry_zero_action':action,'current_risk':r['decision']['risk'],
                                'geometry_zero_risk':float(n)})
        geometry[split]={'raw_geometry_zero_changed_decisions':len(changed),'cases':changed,
                         'is_retrained_ablation':False,'causal_benefit_claimed':False}
        counts[split]=len(runtime)
    oof=np.load(root/'calibration/crossfit_risk.npz')['risk']
    err=np.asarray([r['evaluation']['error_event'] for r in read_rows(root/'calibration/evaluator_join.jsonl')])
    assert probability_metrics(oof,err)==calibrator['crossfit_metrics']
    smoke=read_json(root/'smoke/comparison.json')
    assert smoke['decision_equal'] and smoke['risk_vector44_max_abs_delta']==0.
    assert smoke['target_probability_max_abs_delta']==0. and smoke['risk_abs_delta']==0.
    for name,h in smoke['input_hashes'].items(): assert sha256_file(root/'smoke'/name)==h
    historical_snapshot = Path('/tmp/pcrau_unified_pre_task_20261005.json')
    if historical_snapshot.is_file():
        protected = read_json(historical_snapshot)
        protected_origin = str(historical_snapshot)
    else:
        # /tmp is transient across sessions. Reuse recorded historical hashes,
        # never manufacture a task-start snapshot from today's file contents.
        historical_snapshot = workspace/'new/outputs/pcrau_horizontal_verifier_iid_cached_20261005/provenance.json'
        protected = read_json(historical_snapshot)['protected_sha256']
        protected_origin = str(historical_snapshot)
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    for p,h in read_json(root/'report/provenance.json')['read_only_inputs'].items():
        assert sha256_file(Path(p))==h
    docs=[workspace/'plan/S3_UNIFIED_SPATIAL_INFERENCE_20261005.md',
          workspace/'plan/S3_UNIFIED_INFERENCE_PROTOCOL_20261005.md',
          workspace/'new/docs/UNIFIED_SPATIAL_INFERENCE.md',root/'README.md']
    links=0
    for p in docs:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not target.startswith(('http:','https:','#')):
                assert (p.parent/target.split('#')[0]).exists(),(p,target);links+=1
    new_source = [workspace/p for p in ('new/src/pcrau/unified_inference.py','new/src/pcrau/frozen_rgbd.py',
        'new/scripts/infer_spatial_variant.py','new/scripts/build_unified_spatial_bundle.py',
        'new/scripts/report_unified_spatial.py','new/scripts/verify_unified_spatial_bundle.py',
        'new/tests/test_unified_inference.py')]
    for p in new_source: ast.parse(p.read_text())
    assert all(sha256_file(Path(p))==h for p,h in before.items())
    atomic_json(output,{'status':'PASS_TECHNICAL_AND_ARTIFACT_VERIFICATION',
        'verified_on_local_date':'2026-10-07','bundle_sha256':sha256_file(root/'bundle.json'),
        'freeze_sha256':bundle_id,'tests_passed':tests,'source_AST_checks':len(new_source),
        'markdown_links_checked':links,'runtime_rows_replayed':counts,
        'geometry_zero_diagnostic':geometry,'historical_protected_files_unchanged':len(protected),
        'protected_hash_snapshot_origin':protected_origin,
        'original_1193_task_snapshot_available':Path('/tmp/pcrau_unified_pre_task_20261005.json').is_file(),
        'current_bundle_files_unchanged':len(before),'RGBD_one_case_smoke_pass':True,
        'neural_training_steps':0,'calibration_fit_in_verification':False,'IID_opened':False,
        'performance_gate_pass_claimed':False,'robot_motion_commanded':False,
        'artifacts_sha256':before,'documentation_sha256':{str(p):sha256_file(p) for p in docs}})
    print(json.dumps({'status':'PASS','tests':len(tests),'rows':counts,
        'geometry_zero_changed_decisions':{s:g['raw_geometry_zero_changed_decisions'] for s,g in geometry.items()},
        'protected':len(protected),'output':str(output)},ensure_ascii=False))


if __name__=='__main__': main()
