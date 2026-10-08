"""Runtime boundary and observable evidence/decision tests; standalone runner."""
import copy
import numpy as np
import torch

from pcrau.answerability_evidence import EVIDENCE_NAMES
from pcrau.unified_inference import (VERSION, FEATURE_SHAPES, make_batch,
                                    export_observable, attach_decisions)
from pcrau.verifier_risk import feature_names, risk_vector

CFG = {'max_tokens': 72, 'vocab_size': 8192, 'max_relations': 3, 'max_anchors': 3}


def fixture():
    prompts = ['Locate the apple that is right of the purple cube.',
               'Locate the apple.', 'Locate the apple that is behind the cube.']
    features = {k: torch.zeros(3, *shape) for k, shape in FEATURE_SHAPES.items()}
    batch = make_batch(features, prompts, CFG, 'cpu')
    target = torch.full((3, 24, 32), -12.); target[:, 10, 22] = 12.
    anchor = torch.full((3, 3, 24, 32), -12.); anchor[:, :, 10, 5] = 12.
    obs = torch.zeros(3, len(EVIDENCE_NAMES)); obs[:, EVIDENCE_NAMES.index('base_found')] = .2
    output = {'target_logits': target, 'anchor_logits': anchor,
              'answerability_logits': torch.tensor([[4., 0., 0., 0.]]*3),
              'source_logits': torch.zeros(3, 5), 'relation_edge_logits': torch.zeros(3, 3),
              'observable_answerability_evidence': obs, 'fusion_gate_mean': torch.zeros(3)}
    rows = export_observable(output, batch, prompts, anchor, 'test-bundle')
    return rows, features, prompts, batch, output


def rejects(fn, exception=ValueError):
    try: fn()
    except exception: return
    raise AssertionError('Forbidden input was accepted')


def test_visual_boundary_rejects_oracle_nonfinite_and_shape():
    _, f, p, _, _ = fixture()
    rejects(lambda: make_batch({**f, 'target_mask': torch.zeros(1)}, p, CFG, 'cpu'))
    bad = dict(f); bad['r_thumb'] = f['r_thumb'].clone(); bad['r_thumb'][0, 0] = float('nan')
    rejects(lambda: make_batch(bad, p, CFG, 'cpu'))
    rejects(lambda: make_batch(f, p[:1], CFG, 'cpu'))
    rejects(lambda: make_batch(f, ['', *p[1:]], CFG, 'cpu'))


def test_export_preserves_pre_adapter_and_prompt_edge_mask():
    rows, _, _, batch, output = fixture()
    assert rows[0]['observable_answerability_evidence']['base_found'] == float(torch.tensor(.2))
    assert rows[0]['answerability_probabilities']['FOUND'] > .9
    assert rows[0]['relation_consistency'] == .5
    assert rows[1]['relation_consistency'] == 1.
    assert all('evaluation' not in r and 'family_id' not in r for r in rows)
    rejects(lambda: export_observable(output, {**batch, 'edge_mask': torch.zeros(3, 3)},
                                      ['x']*3, output['anchor_logits'], 'test-bundle'))


def test_scope_geometry_and_inputs_not_mutated():
    rows, _, _, _, _ = fixture()
    h, d, u = [r['verifiers']['P1'] for r in rows]
    assert h['scope_status'] == 'SUPPORTED_DIAGNOSTIC' and h['features']['peak_compatible'] == 1.
    assert d['scope_status'] == 'DIRECT_BYPASS' and u['scope_status'] == 'PARSE_UNSUPPORTED'
    assert d['pair_compatibility'] is None and u['anchor_peak_xy'] is None
    assert all(r['verifiers']['P1']['binding_status'] == 'UNVERIFIED' for r in rows)
    before = copy.deepcopy(rows)
    for r in rows: assert np.isfinite(risk_vector(r, 'P1_G44')).all()
    assert rows == before


def test_geometry_changes_risk_and_policy_with_exact_trace():
    rows, _, _, _, _ = fixture()
    c = {'evidence_producer': VERSION, 'model': 'P1_G44', 'bundle_sha256': 'test-bundle',
         'feature_names': feature_names('P1_G44'), 'mean': [0.]*44, 'scale': [1.]*44,
         'coefficients': [0.]*43+[-4.], 'intercept': 0.}
    profile = {'policy': 'hard_found', 'threshold': .1, 'bundle_sha256': 'test-bundle'}
    result = attach_decisions(rows, c, profile)
    assert result[0]['decision']['action'] == 'EXECUTE'
    changed = copy.deepcopy(rows)
    changed[0]['verifiers']['P1']['features']['pair_compatibility'] = 0.
    alternative = attach_decisions(changed, c, profile)
    assert alternative[0]['decision']['risk'] == .5
    assert alternative[0]['decision']['action'] != 'EXECUTE'
    for r in result:
        trace = r['risk_trace']
        assert abs(sum(trace['standardized_logit_contributions'])+trace['intercept']-trace['risk_logit']) < 1e-12
        assert not trace['target_map_changed'] and not r['decision']['robot_motion_commanded']


def test_live_calibrator_and_bundle_guard():
    rows, _, _, _, _ = fixture()
    c = {'evidence_producer': 'cached', 'model': 'P1_G44'}
    rejects(lambda: attach_decisions(rows, c, {}))
    c = {'evidence_producer': VERSION, 'model': 'P1_G44', 'bundle_sha256': 'other'}
    rejects(lambda: attach_decisions(rows, c, {'policy': 'hard_found', 'bundle_sha256': 'mismatch'}))
