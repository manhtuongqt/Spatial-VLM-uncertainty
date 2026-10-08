"""Primary point boundary: coordinates, GT independence and calibration provenance."""
from copy import deepcopy

import pytest

from pcrau.roborefer_primary import compose_prediction, decode_point


@pytest.mark.parametrize('answer', ['garbage', '[(0.1, 0.2), (0.3, 0.4)]',
                                   '[(True, 0.2)]', '[(1.2, 0.2)]', '[(1.0, 0.2)]',
                                   '[(0.2, 0.3, 0.4, 0.5)]'])
def test_invalid_points_never_fall_back_to_sidecar(answer):
    row = compose_prediction('Locate the apple.', answer)
    assert row['target']['pixel_xy'] is None
    assert row['decision']['action'] == 'REOBSERVE_INVALID_POINT'
    assert row['decision']['risk'] is None


def test_coordinates_use_normalized_point_without_grid_quantization():
    assert decode_point('[(0.847, 0.639)]', (640, 475))['pixel_xy'] == [542, 303]
    row = compose_prediction('Locate the apple.', '[(0.847, 0.639)]')
    assert row['target']['point_xy_inference640'] == [0.847*640, 0.639*480]
    assert row['target']['probability_grid'] is None


def test_auxiliary_prediction_cannot_override_primary_or_supply_its_risk():
    prompt = 'Locate the apple.'
    auxiliary = {'prompt': prompt, 'spatial': {'map_pixel_xy': [10, 10]},
                 'decision': {'action': 'REOBSERVE', 'risk': 0.999},
                 'evaluation': {'bbox': [0, 0, 640, 480]}, 'family_id': 'oracle'}
    before = deepcopy(auxiliary)
    row = compose_prediction(prompt, '[(0.5, 0.5)]', sidecar_observation=auxiliary)
    assert row['target']['pixel_xy'] == [320, 240]
    assert row['decision']['risk'] is None
    assert row['decision']['action'] == 'REVIEW_UNCALIBRATED'
    assert 'evaluation' not in row['sidecar_diagnostic']
    assert 'family_id' not in row['sidecar_diagnostic']
    auxiliary['evaluation']['bbox'] = [0, 0, 1, 1]
    assert row == compose_prediction(prompt, '[(0.5, 0.5)]', sidecar_observation=auxiliary)
    assert before['spatial'] == auxiliary['spatial']
    with pytest.raises(TypeError):
        compose_prediction(prompt, '[(0.5, 0.5)]', target_mask=[[1]])


def test_geometric_sign_uses_primary_point_not_sidecar_peak():
    prompt = 'Locate the apple that is right of the cube.'
    auxiliary = {'prompt': prompt, 'spatial': {'map_pixel_xy': [10, 10]},
                 'verifiers': {'P1': {'scope_status': 'SUPPORTED_DIAGNOSTIC',
                                      'anchor_peak_xy': [200, 240]}}}
    r = compose_prediction(prompt, '[(0.75, 0.5)]', sidecar_observation=auxiliary)
    assert r['geometry']['signed_margin_px'] == 268
    assert r['geometry']['peak_compatible']
    assert r['geometry']['relation_probability'] is None
    r = compose_prediction(prompt, '[(0.25, 0.5)]', sidecar_observation=auxiliary)
    assert not r['geometry']['peak_compatible']


def test_missing_anchor_unsupported_and_prompt_alignment():
    r = compose_prediction('Locate the apple that is right of the cube.', '[(0.5, 0.5)]')
    assert r['geometry']['scope_status'] == 'ANCHOR_EVIDENCE_UNAVAILABLE'
    r = compose_prediction('Locate the apple that is behind the cube.', '[(0.5, 0.5)]')
    assert r['geometry']['scope_status'] == 'PARSE_UNSUPPORTED'
    with pytest.raises(ValueError):
        compose_prediction('Locate the apple.', '[(0.5, 0.5)]',
                           sidecar_observation={'prompt': 'Locate the cube.'})
