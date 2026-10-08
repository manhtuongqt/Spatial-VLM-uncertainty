#!/usr/bin/env python3
"""Recompute frozen saved-row IID analysis and verify provenance; no fit/forward."""
import argparse
import ast
import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.live_iid_analysis import model_metrics,paired_changes,paired_family_bootstrap
from pcrau.selective_experiment import apply_risk
from pcrau.unified_inference import attach_decisions
from pcrau.utils import atomic_json,read_json,sha256_file,workspace_path


def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('new/outputs/pcrau_unified_spatial_iid_20261007'))
    args=parser.parse_args();root=args.root.resolve();receipt=root/'final_verification.json'
    if receipt.exists():raise FileExistsError(receipt)
    before={str(p):sha256_file(p) for p in root.rglob('*') if p.is_file()}
    provenance=read_json(root/'provenance.json');execution=read_json(root/'pre_iid_execution_lock.json')
    assert sha256_file(root/'pre_iid_execution_lock.json')==provenance['execution_lock_sha256']
    for p,h in provenance['protected_sha256'].items():assert sha256_file(Path(p))==h,p
    for name in ('bundle','original_freeze','calibrator','profile','protocol'):
        d=execution[name];assert sha256_file(Path(d['path']))==d['sha256'],name
    original_bundle=Path(execution['bundle']['path']).parent
    calibrator=read_json(original_bundle/'calibrator.json');profile=read_json(original_bundle/'profile.json')
    for name in ('bundle.json','calibrator.json','profile.json'):
        assert (root/'frozen'/name).read_bytes()==(original_bundle/name).read_bytes()
    assert (root/'frozen/original_freeze_lock.json').read_bytes()==Path(execution['original_freeze']['path']).read_bytes()
    observable=rows(root/'runtime_predictions.jsonl');evaluated=rows(root/'evaluator_predictions.jsonl')
    baseline=rows(root/'baseline_evaluator_predictions.jsonl')
    anchors=rows(root/'anchor_evaluation.jsonl');summary=read_json(root/'summary.json')
    assert len(observable)==len(evaluated)==len(baseline)==1000
    assert len({r['sample_id'] for r in observable})==1000 and len({r['family_id'] for r in observable})==200
    assert all('evaluation' not in r and 'variant' not in r for r in observable)
    assert attach_decisions(observable,calibrator,profile)==observable
    saved=np.load(root/'risk_scores.npz')
    live_risk=np.asarray([r['decision']['risk'] for r in observable])
    assert np.array_equal(saved['sample_ids'],np.asarray([r['sample_id'] for r in observable]))
    assert np.array_equal(saved['live'],live_risk)
    active=read_json(workspace_path('new/outputs/active_experimental_profile.json'))
    baseline_risk=apply_risk(baseline,read_json(Path(active['calibrator'])))
    assert np.array_equal(saved['baseline'],baseline_risk)
    assert model_metrics(evaluated,live_risk,profile['threshold'])==summary['metrics']['live']
    assert model_metrics(baseline,baseline_risk,active['risk_threshold'])==summary['metrics']['baseline']
    paired,changed=paired_changes(evaluated,baseline)
    assert paired==summary['paired_changes'] and changed==rows(root/'paired_decision_changes.jsonl')
    assert paired_family_bootstrap(evaluated,baseline)==summary['paired_family_bootstrap']
    lock=read_json(Path(execution['original_freeze']['path']))
    for d in lock['source_files'].values():assert sha256_file(Path(d['path']))==d['sha256']
    cfg=read_json(Path(lock['config']['path']));ds=ArchivedPCRAUDataset(cfg,'test_iid',profile='test_iid')
    entries={e['sample_id']:e for e in ds.entries}
    for r in evaluated:
        e=entries[r['sample_id']];sup=e['supervision'];m=cv2.imread(str(ds.layout.dataset_path(sup['target_mask_path'])),0)
        assert m is not None and m.shape==(480,640)
        x,y=r['spatial']['map_pixel_xy'];ev=r['evaluation']
        assert ev['answerability_state']==sup['answerability_state'] and ev['map_inside_target']==bool(m[y,x]>0)
        assert ev['error_event']==(ev['answerability_state']!='FOUND' or not ev['map_inside_target'])
        v=r['verifiers']['P1']
        assert v['binding_status']=='UNVERIFIED' and v['presence_status']=='UNVERIFIED' and not v['target_map_changed']
        assert not r['decision']['robot_motion_commanded']
        if v['scope_status']!='SUPPORTED_DIAGNOSTIC':
            assert v['pair_compatibility'] is None and v['anchor_peak_xy'] is None
    for a in anchors:
        sup=entries[a['sample_id']]['supervision'];m=cv2.imread(str(ds.layout.dataset_path(sup['anchor_masks'][0]['path'])),0)
        assert m is not None and a['pixels']==int((m>0).sum())
        for arm in ('M0','P1'):
            x,y=a[arm+'_peak_xy'];assert a[arm+'_hit']==(bool(m[y,x]>0) if a['pixels'] else None)
    assert sum(a['P1_hit'] is True for a in anchors)==summary['anchor_metrics']['P1_hits']
    report=root/'report_r2'
    for p,h in read_json(report/'provenance.json')['original_run_files_sha256'].items():
        assert sha256_file(Path(p))==h
    cases=list(report.glob('*_case_*.json'))
    for p in cases:
        case=read_json(p);assert sha256_file(Path(case['rgb_path']))==case['rgb_sha256']
        assert case['runtime_prediction']==next(r for r in observable if r['sample_id']==case['sample_id'])
    with (report/'comparison.csv').open() as f:
        for row in csv.DictReader(f):
            m=summary['metrics'][row['model']]
            assert int(row['correct'])==m['policy']['correct_acceptances']
            assert int(row['errors'])==m['policy']['errors'] and float(row['selective_risk'])==m['policy']['empirical_risk']
    docs=[workspace_path('plan/SPATIAL_VARIANT_FINAL_SUMMARY_20261007.md'),
          workspace_path('plan/S4_LIVE_IID_PROTOCOL_20261007.md'),root/'README.md']
    links=0
    for p in docs:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not target.startswith(('http:','https:','#')):
                assert (p.parent/target.split('#')[0]).exists(),(p,target);links+=1
    source=[workspace_path(p) for p in ('new/src/pcrau/live_iid_analysis.py',
        'new/tests/test_live_iid_analysis.py','new/scripts/reevaluate_unified_spatial_iid.py',
        'new/scripts/report_unified_spatial_iid.py','new/scripts/report_unified_spatial_iid_v2.py',
        'new/scripts/verify_live_iid_artifacts.py')]
    for p in source:ast.parse(p.read_text())
    assert all(sha256_file(Path(p))==h for p,h in before.items())
    atomic_json(receipt,{'status':'PASS_FROZEN_LIVE_IID_ARTIFACT_VERIFICATION','date_local':'2026-10-07',
        'original_bundle_manifest_sha256':execution['bundle']['sha256'],
        'threshold_unchanged':profile['threshold'],'runtime_risk_decisions_replayed':1000,
        'target_masks_independently_checked':1000,'anchor_masks_independently_checked':len(anchors),
        'paired_bootstrap_recomputed':True,'family_bootstrap_resamples':5000,
        'tests_passed_before_forward':read_json(root/'unit_checks.json')['tests'],
        'source_AST_checks':len(source),'markdown_links_checked':links,'real_cases_checked':len(cases),
        'figures_visually_inspected':['changed_decisions.png','binding_and_scope_limits.png','risk_coverage.png'],
        'protected_files_unchanged':len(provenance['protected_sha256']),
        'current_run_files_unchanged':len(before),'artifacts_sha256':before,
        'source_sha256':{str(p):sha256_file(p) for p in source},
        'documentation_sha256':{str(p):sha256_file(p) for p in docs},
        'no_forward_or_fit_in_final_check':True,'model_or_threshold_selected_on_IID':False,
        'historically_observed_IID':True,'primary_empirical_budget_met':summary['primary_empirical_budget_met'],
        'active_profile_changed':False,'robot_motion_commanded':False})
    print(json.dumps({'status':'PASS','runtime_rows':1000,'family_bootstrap':5000,'protected':len(provenance['protected_sha256']),
                      'real_cases':len(cases),'receipt':str(receipt)},ensure_ascii=False))


if __name__=='__main__':main()
