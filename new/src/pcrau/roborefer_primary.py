"""Opt-in RoboRefer target point with explicitly separate Sidecar diagnostics.

No synthetic heatmap, no GT input, and no reuse of Sidecar calibration for
the new target. The frozen Tasks60 implementation and bundle are untouched.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import math

from .query_parser import parse_query

VERSION = 'pcrau_roborefer_primary_v1'
POINT_SUFFIX = ('Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], '
                'where the coordinates are between 0 and 1, indicating the normalized '
                'pixel location of one point satisfying the conditions above.')


def generation_prompt(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Expected a nonempty instruction')
    return prompt.strip() + ' ' + POINT_SUFFIX


def decode_point(answer: str, image_size=(640, 480)) -> dict:
    """Strict normalized [(x,y)]; no clamp, fallback or GT-based point choice."""
    if (len(image_size) != 2 or
            any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in image_size)):
        raise ValueError('Image size must be positive integer width,height')
    result = {'status': 'INVALID_FORMAT', 'normalized_xy': None, 'pixel_xy': None,
              'image_size': list(image_size), 'coordinate_contract': 'normalized_xy_then_floor'}
    if not isinstance(answer, str):
        return result
    text = answer.strip()
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return result
    if not isinstance(value, list) or len(value) != 1:
        result['status'] = 'NOT_EXACTLY_ONE_POINT'
        return result
    point = value[0]
    if (not isinstance(point, tuple) or len(point) != 2 or
            not text.startswith('[(') or not text.endswith(')]') or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in point)):
        return result
    xy = [float(x) for x in point]
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in xy):
        result['status'] = 'OUT_OF_RANGE'
        return result
    result['normalized_xy'] = xy
    pixels = [int(x*s) for x, s in zip(xy, image_size)]
    # A normalized 1.0 lies at the image edge, not at an addressable pixel.
    if any(p >= s for p, s in zip(pixels, image_size)):
        result['status'] = 'OUTSIDE_IMAGE'
        return result
    result.update(status='VALID', pixel_xy=pixels)
    return result


def _geometry(prompt, point, sidecar):
    query = parse_query(prompt)
    trace = {'query': query.to_dict(), 'target_source': 'RoboRefer_generation',
             'target_point_xy': None, 'anchor_source': 'Sidecar_P1_predicted_peak',
             'anchor_point_xy': None, 'signed_margin_px': None, 'margin_px': 12,
             'peak_compatible': None, 'relation_probability': None,
             'binding_status': 'UNVERIFIED', 'presence_status': 'UNVERIFIED',
             'target_distribution_available': False}
    if point['status'] != 'VALID':
        trace['scope_status'] = 'INVALID_TARGET_POINT'
    elif not query.supported:
        trace['scope_status'] = 'PARSE_UNSUPPORTED'
    elif query.predicate == 'direct':
        trace['scope_status'] = 'DIRECT_BYPASS'
    else:
        evidence = (sidecar or {}).get('verifiers', {}).get('P1', {})
        anchor = evidence.get('anchor_peak_xy')
        if evidence.get('scope_status') != 'SUPPORTED_DIAGNOSTIC' or anchor is None:
            trace['scope_status'] = 'ANCHOR_EVIDENCE_UNAVAILABLE'
        else:
            if len(anchor) != 2 or any(not math.isfinite(float(v)) for v in anchor):
                raise ValueError('Nonfinite anchor prediction')
            if not (0 <= anchor[0] < 640 and 0 <= anchor[1] < 480):
                raise ValueError('Anchor prediction outside inference image')
            target = [point['normalized_xy'][0]*640, point['normalized_xy'][1]*480]
            signed = (1 if query.predicate == 'right_of' else -1)*(target[0]-anchor[0])-12
            trace.update(scope_status='SUPPORTED_POINT_DIAGNOSTIC', target_point_xy=target,
                         anchor_point_xy=list(anchor), signed_margin_px=signed,
                         peak_compatible=bool(signed > 0))
    return trace


def compose_prediction(prompt: str, answer: str, *, image_size=(640, 480),
                       sidecar_observation=None) -> dict:
    """Runtime inputs only. Sidecar annotation/metadata fields are never copied."""
    generation_prompt(prompt)  # validate even if the caller supplied cached text
    if sidecar_observation is not None and sidecar_observation.get('prompt') != prompt:
        raise ValueError('Sidecar observation belongs to a different instruction')
    point = decode_point(answer, image_size)
    geometry = _geometry(prompt, point, sidecar_observation)
    diagnostic = None
    if sidecar_observation is not None:
        diagnostic = {k: deepcopy(sidecar_observation[k]) for k in
                      ('spatial', 'answerability_probabilities', 'source_probabilities',
                       'task_uncertainty', 'decision') if k in sidecar_observation}
        diagnostic['calibration_applies_to'] = 'original_Sidecar_target_MAP_only'
        diagnostic['MC_distributions_apply_to'] = 'Sidecar_auxiliary_tasks_not_RoboRefer_generation'
        old_point = diagnostic.get('spatial', {}).get('map_pixel_xy')
    else:
        old_point = None
    target640 = ([v*s for v, s in zip(point['normalized_xy'], (640, 480))]
                 if point['status'] == 'VALID' else None)
    disagreement = (math.hypot(target640[0]-old_point[0], target640[1]-old_point[1])/800
                    if target640 is not None and old_point is not None else None)
    return {'version': VERSION, 'prompt': prompt,
            'target': {'source': 'RoboRefer-2B-SFT_RGBD_generation', 'answer': answer, **point,
                       'point_xy_inference640': target640, 'probability_grid': None},
            'geometry': geometry, 'sidecar_diagnostic': diagnostic,
            'cross_model_point_disagreement_normalized': disagreement,
            'decision': {'action': ('REVIEW_UNCALIBRATED' if point['status'] == 'VALID'
                                    else 'REOBSERVE_INVALID_POINT'),
                         'risk': None, 'threshold': None,
                         'calibration_status': 'REQUIRES_PRIMARY_POINT_CALIBRATION',
                         'decision_level': 'perception_only', 'robot_motion_commanded': False},
            'trace': {'target_selected_without_GT': True, 'sidecar_target_overrides_primary': False,
                      'sidecar_answerability_vetoes_primary': False,
                      'heatmap_fabricated_from_point': False,
                      'old_calibrator_reused_for_primary': False}}
