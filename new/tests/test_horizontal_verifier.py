"""Observable geometry, bypass, binding claims and calibration boundary tests."""
from copy import deepcopy
import numpy as np

from pcrau.horizontal_verifier import verify,EXTRA_FEATURES
from pcrau.verifier_risk import risk_vector,feature_names,apply_verifier_risk


def maps(tx=20,ax=5):
    target=np.zeros((24,32));target[10,tx]=1
    anchor=np.full((24,32),-100.);anchor[10,ax]=10
    return target,anchor


def test_right_left_sign_and_pair_mass():
    target,anchor=maps()
    right=verify('locate the apple that is right of the cube.',target,anchor,anchor)
    left=verify('locate the apple that is left of the cube.',target,anchor,anchor)
    assert right['signed_margin_px']==288 and left['signed_margin_px']==-312
    assert right['features']['peak_compatible']==1 and left['features']['peak_compatible']==0
    assert right['pair_compatibility']>1-1e-10 and left['pair_compatibility']<1e-10


def test_same_grid_peak_fails_margin():
    target,anchor=maps(5,5);r=verify('locate the apple that is right of the cube.',target,anchor,anchor)
    assert r['signed_margin_px']==-12 and not r['features']['peak_compatible']


def test_diffuse_map_score_is_not_presence_or_binding_probability():
    target,anchor=maps();anchor[:]=-100
    r=verify('locate the apple that is right of the cube.',target,anchor,anchor)
    assert np.isclose(r['pair_compatibility'],20/32)
    assert r['presence_status']==r['binding_status']=='UNVERIFIED'
    assert r['features']['anchor_sigmoid_max']<1e-30
    assert 'nonpositive_anchor_peak_logit' in r['failure_flags']


def test_direct_and_unsupported_have_no_fabricated_geometry():
    target,_=maps()
    for prompt,status,key in [('locate the apple.','DIRECT_BYPASS','direct_bypass'),
        ('locate the apple that is behind the cube.','PARSE_UNSUPPORTED','parse_unsupported')]:
        r=verify(prompt,target)
        assert r['scope_status']==status and r['features'][key]==1
        assert r['signed_margin_px'] is None and r['anchor_peak_xy'] is None
        assert all(r['features'][k]==0 for k in EXTRA_FEATURES if k!=key)


def test_inputs_unchanged_and_oracle_kwargs_rejected():
    t,a=maps();before=t.copy(),a.copy();verify('locate the apple that is right of the cube.',t,a,a)
    assert np.array_equal(t,before[0]) and np.array_equal(a,before[1])
    try:verify('locate the apple.',t,anchor_mask=np.ones((24,32)))
    except TypeError:pass
    else:raise AssertionError('Oracle argument accepted')


def test_nonfinite_or_invalid_distribution_rejected():
    t,a=maps()
    for invalid in (np.full_like(t,np.nan),t*2,-t):
        try:verify('locate the apple.',invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid target distribution accepted')
    try:verify('locate the apple that is right of the cube.',t,np.full_like(a,np.inf),a)
    except ValueError:pass
    else:raise AssertionError('Invalid anchor logits accepted')


def test_risk_feature_vector_ignores_all_evaluator_fields():
    import json
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'outputs/pcrau_answerability_language_dev_20261003/detail_cost/verified_dev_predictions.jsonl'
    row=json.loads(source.open().readline());t,a=maps()
    row['verifiers']={k:verify('locate the apple that is right of the cube.',t,a,a) for k in ('M0','C1','P1')}
    other=deepcopy(row);other['evaluation']={'error_event':not row['evaluation']['error_event']};other['variant']='changed';other['family_id']='never_feature'
    for model,count in [('B33',33),('P1_A41',41),('P1_G44',44)]:
        assert len(feature_names(model))==count
        assert np.array_equal(risk_vector(row,model),risk_vector(other,model))


def test_calibrator_bundle_mismatch_rejected_before_prediction():
    try:apply_verifier_risk([{'verifier_bundle_sha256':'wrong'}],{'bundle_sha256':'right'})
    except ValueError:pass
    else:raise AssertionError('Wrong verifier bundle accepted')
