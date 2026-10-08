#!/usr/bin/env python3
"""Validate attributed audit decisions and report precision/Wilson CI."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(success, n):
    if not n:
        return None
    z = 1.959963984540054
    p = success/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0, center-half), min(1, center+half)]


def summarize(queue, decisions):
    known = {r['sample_id']: r for r in queue}
    errors, completed, seen = [], [], set()
    for r in decisions:
        sid = r['sample_id']
        if sid not in known or sid in seen:
            errors.append({'sample_id': sid, 'reason': 'unknown_or_duplicate_id'})
            continue
        seen.add(sid)
        if not r.get('decision'):
            continue
        issues = []
        if r['decision'] not in {'accept', 'reject', 'unsure'}:
            issues.append('invalid_decision')
        if not r.get('reviewer', '').strip() or not r.get('reviewed_at', '').strip():
            issues.append('missing_reviewer_or_date')
        if r.get('review_origin', 'unspecified') not in {'human', 'ai_visual', 'unspecified'}:
            issues.append('invalid_review_origin')
        for field in ['target_correct', 'tabletop_appropriate']:
            if r.get(field) not in {'yes', 'no', 'unsure'}:
                issues.append('invalid_' + field)
        for field in ['relation_correct', 'anchor_correct']:
            if r.get(field) not in {'yes', 'no', 'unsure', 'na'}:
                issues.append('invalid_' + field)
        if known[sid]['relation_candidates'] and r.get('relation_correct') == 'na':
            issues.append('relation_candidate_cannot_be_na')
        if known[sid]['anchor_candidates_offline'] and r.get('anchor_correct') == 'na':
            issues.append('anchor_candidate_cannot_be_na')
        if r['decision'] == 'accept':
            if any(r.get(k) != 'yes' for k in ['target_correct', 'tabletop_appropriate']):
                issues.append('accept_requires_correct_target_and_tabletop')
            if any(r.get(k) in {'no', 'unsure'} for k in ['relation_correct', 'anchor_correct']):
                issues.append('accept_requires_correct_applicable_labels')
            if not r.get('reference_frame') or r.get('reasoning_depth') not in {'0', '1', '2', '3'}:
                issues.append('accepted_sample_missing_frame_or_reasoning')
            frame = r.get('reference_frame', '').lower()
            if 'unknown' in frame or 'unverified' in frame:
                issues.append('accepted_sample_unverified_frame')
        if issues:
            errors.append({'sample_id': sid, 'reasons': issues})
        else:
            completed.append(r)
    def precision(rows):
        result = {}
        for field in ['target_correct', 'relation_correct', 'anchor_correct']:
            values = [r[field] for r in rows if r[field] != 'na']
            yes = values.count('yes')
            result[field] = {'yes': yes, 'no': values.count('no'), 'n': len(values),
                             'unsure': values.count('unsure'),
                             'precision': yes/len(values) if values else None,
                             'wilson_ci95': wilson(yes, len(values))}
        return result
    metrics = precision(completed)
    strata = collections.Counter(tag for r in completed for tag in known[r['sample_id']]['audit_strata'])
    relations = sorted({tag for q in queue for tag in q['audit_strata'] if tag != 'object_grounding_general'})
    per_stratum = {tag: precision([r for r in completed if tag in known[r['sample_id']]['audit_strata']])
                   for tag in sorted(strata)}
    origins = collections.Counter(r.get('review_origin', 'unspecified') for r in completed)
    pass_gate = (not errors and len(completed) >= 300 and all(strata[r] >= 50 for r in relations)
                 and all(m['precision'] is not None and m['precision'] >= .95 for m in metrics.values())
                 and all(m['precision'] >= .95 for tag in relations
                         for m in per_stratum[tag].values() if m['n']))
    return {'status': 'MANUAL_SAMPLE_GATE_PASS' if pass_gate else 'MANUAL_SAMPLE_GATE_NOT_MET',
            'queue_count': len(queue), 'completed_reviews': len(completed), 'pending_reviews': len(queue)-len(completed),
            'validation_errors': errors, 'precision': metrics, 'reviewed_per_stratum': dict(strata),
            'precision_per_stratum': per_stratum, 'review_origins': dict(origins),
            'decision_counts': dict(collections.Counter(r['decision'] for r in completed)),
            'independent_human_review_complete': len(completed) >= 300 and origins.get('human', 0) == len(completed),
            'training_eligible': False,
            'note': 'This gate only audits the review sample; full relation-family support, annotation extraction, provenance and dataset release gates remain separate.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir', type=Path, default=ROOT/'results/spatial_vlm_refspatial_v1/wp1_source_audit')
    p.add_argument('--queue-file', type=Path, help='Optional JSONL queue path.')
    p.add_argument('--decisions-file', type=Path, help='Optional CSV decision path.')
    p.add_argument('--output', type=Path, help='Optional JSON report path.')
    a = p.parse_args()
    queue_path = a.queue_file or a.audit_dir/'manual_audit_queue.jsonl'
    decisions_path = a.decisions_file or a.audit_dir/'manual_audit_decisions.csv'
    output_path = a.output or a.audit_dir/'manual_review_report.json'
    queue = [json.loads(s) for s in queue_path.read_text().splitlines()]
    with decisions_path.open() as f:
        result = summarize(queue, list(csv.DictReader(f)))
    output_path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
