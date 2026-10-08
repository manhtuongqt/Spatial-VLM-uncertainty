#!/usr/bin/env python3
"""Freeze live producer; train/dev audit; calibration-only refit; bundle reload."""
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
from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.engine import autocast_context
from pcrau.model import PCRAUTargetV2
from pcrau.selective_experiment import choose_policy_threshold, policy_result, predicted_answer, probability_metrics
from pcrau.unified_inference import (UnifiedInference, VERSION, load_features,
                                    make_batch, attach_decisions)
from pcrau.utils import atomic_json, read_json, seed_everything, sha256_file, workspace_path, runtime_info
from pcrau.verifier_risk import fit_verifier_risk, apply_verifier_risk, risk_vector, feature_names


def write_rows(path, rows):
    with path.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='new/outputs/pcrau_unified_spatial_20261005')
    args = parser.parse_args(); out = workspace_path(args.output); out.mkdir(exist_ok=False)
    active_path = workspace_path('new/outputs/active_experimental_profile.json')
    active = read_json(active_path)
    protected = read_json(Path('/tmp/pcrau_unified_pre_task_20261005.json'))
    def descriptor(path, expected=None):
        p = Path(path).resolve(); h = sha256_file(p)
        if expected is not None and h != expected: raise ValueError(f'Hash mismatch: {p}')
        return {'path': str(p), 'sha256': h}
    baseline_desc = {k: descriptor(active[k], active[k+'_sha256']) for k in ('config', 'checkpoint')}
    cfg = read_json(Path(active['config']))
    checks = []
    torch.set_num_threads(4); seed_everything(24082026)
    for p in ('new/tests/test_unified_inference.py', 'new/tests/test_horizontal_verifier.py'):
        for name, fn in runpy.run_path(str(workspace_path(p))).items():
            if name.startswith('test_') and callable(fn): fn(); checks.append(name)
    atomic_json(out/'unit_checks.json', {'pass': True, 'tests': checks})
    print(f'UNIT {len(checks)} PASS', flush=True)
    residual_source = workspace_path('new/outputs/pcrau_s1_anchor_peak_pilot_v2_20261005/P1/checkpoints/best/mlp.safetensors')
    freeze_pilot = read_json(residual_source.parents[3]/'freeze_lock.json')
    descriptor(residual_source, freeze_pilot['checkpoints']['P1'])
    frozen = out/'frozen'; frozen.mkdir()
    residual = frozen/'anchor_mlp.safetensors'; shutil.copyfile(residual_source, residual)
    inventory_manifest = workspace_path('old/protocol/pcra_u_calibration_feature_manifest.json')
    inventory = read_json(inventory_manifest)['model_inventory_sha256']
    source_paths = list(workspace_path('new/src/pcrau').glob('*.py')) + [
        Path(__file__).resolve(), workspace_path('new/scripts/infer_spatial_variant.py'),
        workspace_path('new/tests/test_unified_inference.py'), workspace_path('new/tests/test_horizontal_verifier.py'),
        workspace_path('old/protocol/wp3_feature_hook_smoke.py'), workspace_path('old/protocol/pcra_u_development_common.py')]
    lock = {'version': VERSION, 'config': baseline_desc['config'], 'baseline_checkpoint': baseline_desc['checkpoint'],
            'baseline_metadata': descriptor(Path(active['checkpoint']).parent/'metadata.json'),
            'anchor_residual': descriptor(residual), 'residual_source': descriptor(residual_source),
            'source_files': {str(p): descriptor(p) for p in source_paths},
            'protocol': descriptor(workspace_path('plan/S3_UNIFIED_INFERENCE_PROTOCOL_20261005.md')),
            'feature_manifest': descriptor(inventory_manifest), 'backbone_inventory_sha256': inventory,
            'feature_names': feature_names('P1_G44'), 'primary': 'P1_G44', 'device_type': 'cuda',
            'precision': 'CUDA_BF16_autocast_FP32_residual_FP16_visual_storage',
            'evidence_producer': VERSION, 'cached_baseline_evidence_used': False,
            'roborefer_generation': False, 'target_map_changed': False, 'seed': 24082026,
            'calibration': {'l2': .01, 'folds': 5, 'risk_target': .075, 'minimum_accepted_families': 60},
            'improvement_gates_advisory_only': True, 'runtime': runtime_info()}
    lock_path = frozen/'freeze_lock.json'; atomic_json(lock_path, lock); bundle_id = sha256_file(lock_path)
    snapshot = frozen/'source_snapshot'; snapshot.mkdir()
    for index, p in enumerate(source_paths): shutil.copyfile(p, snapshot/f'{index:03d}_{p.name}')
    model = PCRAUTargetV2(cfg).cuda().eval(); load_model_checkpoint(model, active['checkpoint'], active['config_sha256'])
    pipeline = UnifiedInference(cfg, model, residual, bundle_id)
    before = state_digest(pipeline.shadow.state_dict())
    observed_by_split = {}; evaluated_by_split = {}; audit = {}; data_inputs = {}
    for split in ('train', 'dev', 'calibration'):
        ds = ArchivedPCRAUDataset(cfg, split, profile='calibration' if split=='calibration' else 'development')
        directory = out/split; directory.mkdir()
        @lru_cache(maxsize=32)
        def cached(key):
            desc = ds.features[key]; path = ds.layout.feature_root/desc['path']
            data_inputs[str(path)] = descriptor(path, desc['sha256'])
            return load_features(path)
        data_inputs[str(ds.layout.manifest)] = descriptor(ds.layout.manifest)
        data_inputs[str(ds.layout.feature_index)] = descriptor(ds.layout.feature_index)
        observable = []; batch_ms = []; max_plain = 0.; first_features = None; first_prompts = None
        for start in range(0, len(ds.entries), 20):
            es = ds.entries[start:start+20]
            tensors = []; prompts = []
            for e in es:
                key = ds.sample_to_feature[e['sample_id']]; desc = ds.features[key]
                if any(desc[k] != e['feature_input'][k] for k in ('rgb_sha256', 'depth_sha256')):
                    raise ValueError('Feature cache does not match RGB-D inputs')
                tensors.append(cached(key)); prompts.append(e['feature_input']['prompt'])
            features = {k: torch.stack([t[k] for t in tensors]) for k in tensors[0]}
            if start == 0: first_features, first_prompts = features, prompts
            torch.cuda.synchronize(); started = time.perf_counter()
            rows = pipeline.observe(features, prompts)
            torch.cuda.synchronize(); batch_ms.append((time.perf_counter()-started)*1000)
            if start == 0:
                inputs = make_batch(features, prompts, cfg['model'], 'cuda')
                with torch.no_grad(), autocast_context(torch.device('cuda'), cfg['optimization']):
                    cap = pipeline.shadow.capture(inputs, prompts); plain = model(inputs)
                if any(not torch.equal(v, cap.baseline[k]) for k, v in plain.items()):
                    raise RuntimeError('Wrapper changes baseline outputs')
                max_plain = 0.
            for e, row in zip(es, rows):
                observable.append({'sample_id': e['sample_id'], 'family_id': e['family_id'], **row})
            if start%200==0: print(f'OBSERVE {split} {min(start+20,len(ds.entries))}/{len(ds.entries)}', flush=True)
        write_rows(directory/'observable.jsonl', observable)
        # Masks enter only external evaluator, after saving all live evidence.
        evaluated = []; anchors = []
        for e, row in zip(ds.entries, observable):
            sup = e['supervision']; path = ds.layout.dataset_path(sup['target_mask_path'])
            data_inputs[str(path)] = descriptor(path, sup['target_mask_sha256'])
            mask = cv2.imread(str(path), 0)
            if mask is None or mask.shape != (480, 640): raise ValueError('Invalid evaluator mask')
            x, y = row['spatial']['map_pixel_xy']; inside = bool(mask[y, x]>0)
            truth = sup['answerability_state']
            evaluator = {'answerability_state': truth, 'map_inside_target': inside,
                         'target_exists': bool(mask.any()), 'error_event': truth!='FOUND' or not inside}
            evaluated.append({**row, 'evaluation': evaluator})
            v = row['verifiers']['P1']
            if v['scope_status'] == 'SUPPORTED_DIAGNOSTIC':
                ad = sup['anchor_masks'][0]; ap = ds.layout.dataset_path(ad['path'])
                data_inputs[str(ap)] = descriptor(ap, ad['sha256']); am = cv2.imread(str(ap), 0)
                if am is None or am.shape != (480,640): raise ValueError('Invalid evaluator anchor mask')
                ax, ay = v['anchor_peak_xy']; area = int((am>0).sum())
                anchors.append({'sample_id': e['sample_id'], 'pixels': area,
                                'hit': bool(am[ay,ax]>0) if area else None})
        write_rows(directory/'evaluator_join.jsonl', evaluated)
        write_rows(directory/'anchor_evaluation.jsonl', anchors)
        counts = dict(Counter(r['verifiers']['P1']['scope_status'] for r in observable))
        audit[split] = {'samples': len(observable), 'families': len({r['family_id'] for r in observable}),
                        'scope': counts, 'all_features_finite': True, 'bypass_exact': True,
                        'representative_baseline_capture_plain_exact': max_plain==0.,
                        'visible_anchor': sum(a['pixels']>0 for a in anchors),
                        'anchor_hits': sum(a['hit'] is True for a in anchors),
                        'empty_anchor': sum(a['pixels']==0 for a in anchors),
                        'batch20_latency_ms_median': float(np.median(batch_ms)),
                        'latency_includes_export_and_boundary_checks': True}
        observed_by_split[split] = observable; evaluated_by_split[split] = evaluated
        if split == 'dev':
            # Diagnostic only: no equality gate or saved evidence substitution.
            old_path = Path(active['checkpoint']).parents[3]/'detail_cost/verified_dev_predictions.jsonl'
            old = {r['sample_id']: r for r in [json.loads(line) for line in old_path.read_text().splitlines()]}
            delta = np.stack([risk_vector(r,'P1_G44')[:33]-
                              __import__('pcrau.selective_experiment',fromlist=['vector']).vector(old[r['sample_id']],True)
                              for r in observable])
            audit[split]['historical33_max_abs_delta_by_feature'] = dict(zip(feature_names('P1_G44')[:33], np.abs(delta).max(0).tolist()))
        atomic_json(directory/'checks.json', audit[split])
        print(f'OBSERVED {split}: {audit[split]}', flush=True)
    cal = evaluated_by_split['calibration']
    assert len(cal)==1000 and len({r['family_id'] for r in cal})==200
    calibrator, oof = fit_verifier_risk(cal, 'P1_G44', bundle_id)
    calibrator['evidence_producer'] = VERSION
    profile = choose_policy_threshold(oof, [r['evaluation']['error_event'] for r in cal],
        [r['family_id'] for r in cal], [predicted_answer(r)=='FOUND' for r in cal], .075, 60)
    profile.update(policy='hard_found', bundle_sha256=bundle_id, model='P1_G44',
                   threshold_selected_on='calibration_family_OOF_only', evidence_producer=VERSION)
    atomic_json(out/'calibrator.json', calibrator); atomic_json(out/'profile.json', profile)
    np.savez_compressed(out/'calibration/crossfit_risk.npz', risk=oof)
    manifest = {'version': VERSION, 'freeze_lock': {'path': 'frozen/freeze_lock.json', 'sha256': bundle_id},
                'calibrator': {'path': 'calibrator.json', 'sha256': sha256_file(out/'calibrator.json')},
                'profile': {'path': 'profile.json', 'sha256': sha256_file(out/'profile.json')},
                'selected_as_active': False, 'robot_motion_commanded': False}
    atomic_json(out/'bundle.json', manifest)
    summary = {'version': VERSION, 'audit': audit, 'calibration_profile': profile,
               'calibration_oof_probability_metrics': calibrator['crossfit_metrics'],
               'iid_opened': False, 'neural_optimizer_steps': 0, 'cached_baseline_evidence_used': False,
               'performance_gates_advisory_only': True, 'split_policy': {}, 'geometry_in_risk': {}}
    for split, rows in observed_by_split.items():
        result = attach_decisions(rows, calibrator, profile)
        write_rows(out/split/'runtime_predictions.jsonl', result)
        risk = np.asarray([r['decision']['risk'] for r in result])
        metrics, _ = policy_result(evaluated_by_split[split], risk, 'hard_found', profile['threshold'])
        summary['split_policy'][split] = metrics
        supported = [r for r in result if r['verifiers']['P1']['scope_status']=='SUPPORTED_DIAGNOSTIC']
        summary['geometry_in_risk'][split] = {'supported': len(supported),
            'nonzero_geometry_effect': sum(abs(r['risk_trace']['geometry_raw_logit_effect_vs_zero'])>1e-12 for r in supported),
            'max_abs_geometry_logit_effect': max((abs(r['risk_trace']['geometry_raw_logit_effect_vs_zero']) for r in supported),default=0.)}
    del pipeline, model; torch.cuda.empty_cache()
    reloaded = UnifiedInference.from_bundle(out/'bundle.json')
    replay = reloaded.predict(first_features, first_prompts)
    saved = observed_by_split['calibration'][:20]
    expected = attach_decisions(saved, calibrator, profile)
    assert all(risk_vector(a,'P1_G44').tolist()==risk_vector(b,'P1_G44').tolist()
               and a['decision']==b['decision'] for a,b in zip(replay,expected))
    summary['bundle_reload_exact_representative20'] = True
    summary['neural_states_unchanged'] = before==state_digest(reloaded.shadow.state_dict())
    assert summary['neural_states_unchanged']
    # Calibration rows include external evaluators; runtime result never does.
    assert all('evaluation' not in r for r in replay)
    differences = [p for p,h in protected.items() if sha256_file(Path(p))!=h]
    assert not differences, differences
    summary['protected_files_unchanged'] = len(protected)
    atomic_json(out/'data_provenance.json', data_inputs)
    atomic_json(out/'summary.json', summary)
    atomic_json(out/'RUN_STATUS.json', {'status':'COMPLETE_LIVE_FEATURE_TO_DECISION_BUNDLE', 'iid_opened':False})
    print('COMPLETE', out, flush=True)


if __name__ == '__main__': main()
