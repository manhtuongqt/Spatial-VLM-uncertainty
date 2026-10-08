#!/usr/bin/env python3
"""Prepared normalized 2D point evaluator. No model calls or default test access.

Targets: JSONL {sample_id, family_id, split, target_xy}; predictions: JSONL
{sample_id, prediction_xy}. Radius must be explicitly supplied from locked config.
CLI accepts RefSpatial dev/diagnostic only; Gazebo mask/3D evaluation is separate.
"""
import argparse
import json
import math
from pathlib import Path


def valid_xy(xy):
    return (isinstance(xy, (tuple, list)) and len(xy) == 2
            and all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in xy))


def evaluate(targets, predictions, radius):
    if not 0 < radius <= math.sqrt(2):
        raise ValueError('Radius must be explicitly locked in (0, sqrt(2)].')
    truth, pred = {}, {}
    for row in targets:
        if row['sample_id'] in truth or not valid_xy(row['target_xy']):
            raise ValueError('Duplicate target ID or invalid ground truth.')
        if row.get('split') not in {'dev', 'diagnostic'}:
            raise ValueError('Prepared CLI permits RefSpatial dev/diagnostic only.')
        if not row.get('family_id'):
            raise ValueError('Missing family ID.')
        truth[row['sample_id']] = row
    if not truth:
        raise ValueError('Empty target manifest.')
    for row in predictions:
        if row['sample_id'] in pred or row['sample_id'] not in truth:
            raise ValueError('Duplicate or unknown prediction ID.')
        pred[row['sample_id']] = row.get('prediction_xy')
    per_sample = []
    for sid, row in truth.items():
        xy = pred.get(sid)
        valid = valid_xy(xy)
        error = math.dist(row['target_xy'], xy) if valid else None
        per_sample.append({'sample_id': sid, 'family_id': row['family_id'], 'valid_prediction': valid,
                           'normalized_point_error': error, 'hit': valid and error <= radius})
    errors = [r['normalized_point_error'] for r in per_sample if r['valid_prediction']]
    return {'scope': 'REFSPATIAL_INTERNAL_DIAGNOSTIC_NOT_GENERALIZATION', 'radius': radius,
        'samples': len(truth), 'families': len({r['family_id'] for r in targets}),
        'missing_predictions': len(truth.keys()-pred.keys()), 'invalid_or_missing': len(truth)-len(errors),
        'point_hit_all_samples': sum(r['hit'] for r in per_sample)/len(truth),
        'mean_point_error_valid_only': sum(errors)/len(errors) if errors else None,
        'point_error_denominator': len(errors), 'per_sample': per_sample,
        'confidence_metrics': 'NOT_IMPLEMENTED_NO_SCORE_ASSUMED'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--radius', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    load = lambda p: [json.loads(s) for s in p.read_text().splitlines()]
    result = evaluate(load(a.targets), load(a.predictions), a.radius)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
