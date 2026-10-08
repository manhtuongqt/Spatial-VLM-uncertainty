#!/usr/bin/env python3
"""Apply scope guards to the immutable candidate pool and prepare a fresh audit.

Never exports a clean training dataset. Existing review decisions are preserved.
"""
import collections
import csv
import hashlib
import html
import json
from pathlib import Path
from refspatial_manual_review import summarize
from refspatial_wp1_rules import semantic_review_requirements

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/spatial_vlm_refspatial_v1'
DIRECT = ['leftmost_ranking', 'rightmost_ranking', 'horizontal_ordinal_ranking']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')


def main():
    source, out = BASE/'wp1_source_audit', BASE/'wp2_visual_review'
    out.mkdir(parents=True, exist_ok=True)
    queue = [json.loads(s) for s in (source/'manual_audit_queue.jsonl').read_text().splitlines()]
    with (source/'manual_audit_decisions.csv').open() as f:
        decisions = list(csv.DictReader(f))
    report = summarize(queue, decisions)
    if report['validation_errors']:
        raise ValueError(report['validation_errors'])
    write(source/'manual_review_report.json', report)
    reviewed = {r['sample_id']: r for r in decisions if r['decision']}
    old_families = {r['family_id'] for r in queue}
    scopes, blockers = collections.Counter(), collections.Counter()
    counts = collections.Counter()
    families = collections.defaultdict(set)
    # Keep only the 50 smallest hashes per relation/split while streaming.
    pools = collections.defaultdict(dict)
    with (source/'object_candidates.jsonl').open() as f, (out/'semantic_scope_overlay.jsonl').open('w') as overlay:
        for line in f:
            row = json.loads(line)
            guard = semantic_review_requirements(row['instruction'], row['target_kind'], row['canonical_type'])
            scopes[guard['scope']] += 1
            blockers.update(guard['blockers'])
            rel = guard['direct_relation']
            overlay.write(json.dumps({'sample_id': row['sample_id'], **guard,
                'review_decision': reviewed.get(row['sample_id'], {}).get('decision', 'pending')}, separators=(',', ':'))+'\n')
            if not rel or guard['blockers']:
                continue
            key = (rel, row['split'])
            counts[key] += 1
            families[key].add(row['family_id'])
            if row['family_id'] in old_families:
                continue
            pool = pools[key]
            old = pool.get(row['family_id'])
            if old is None or row['sample_id'] < old['sample_id']:
                pool[row['family_id']] = row
            if len(pool) > 50:
                del pool[max(pool, key=lambda k: pool[k]['sample_id'])]
    coverage = []
    fresh, used = [], set()
    for rel in DIRECT:
        for split, quota in [('train', 30), ('dev', 10), ('diagnostic', 10)]:
            chosen = []
            for row in sorted(pools[(rel, split)].values(), key=lambda r: r['sample_id']):
                if row['family_id'] in used:
                    continue
                chosen.append(dict(row, audit_strata=[rel], label_qc='FRESH_REVIEW_PENDING',
                                   semantic_review=semantic_review_requirements(row['instruction'], row['target_kind'], row['canonical_type'])))
                used.add(row['family_id'])
                if len(chosen) == quota:
                    break
            fresh.extend(chosen)
            coverage.append({'direct_relation': rel, 'split': split, 'candidate_qa': counts[(rel, split)],
                             'candidate_families': len(families[(rel, split)]),
                             'next_audit_quota': quota, 'next_audit_queued': len(chosen),
                             'clean_certified_families': 0, 'training_eligible': False})
    with (out/'direct_relation_coverage.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(coverage[0])); w.writeheader(); w.writerows(coverage)
    with (out/'next_audit_queue.jsonl').open('w') as f:
        for row in fresh: f.write(json.dumps(row, separators=(',', ':'))+'\n')
    next_csv = out/'next_audit_decisions.csv'
    if not next_csv.exists():
        fields = list(decisions[0])
        with next_csv.open('w') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for row in fresh:
                record = {k: '' for k in fields}
                record.update({k: row[k] for k in ['sample_id', 'scene_id', 'split']})
                record['audit_strata'] = '|'.join(row['audit_strata'])
                w.writerow(record)
    page = ['<!doctype html><meta charset="utf-8"><title>Fresh direct-ranking audit</title>',
            '<style>body{font:16px sans-serif;max-width:1100px;margin:auto}article{border-bottom:2px solid #777;padding:20px}img{max-width:960px}pre{white-space:pre-wrap}</style>',
            '<h1>150 fresh direct-ranking candidates — NOT REVIEWED</h1>',
            '<p>Read instruction and RGB first. Target coordinates below are source answers, not verified truth. '
            'No distance-from-anchor queries; no family from the original 300-query audit.</p>']
    for i, row in enumerate(fresh, 1):
        page.append(f'<article><h2>{i}. {html.escape(row["sample_id"])}</h2><p>{html.escape(row["instruction"])}</p>'
                    f'<img loading="lazy" src="../../../{html.escape(row["image"], quote=True)}">'
                    f'<details><summary>Source target (offline)</summary><pre>{html.escape(json.dumps(row["target_xy"]))}</pre></details></article>')
    (out/'NEXT_AUDIT.html').write_text('\n'.join(page))
    with (out/'followup_review_required.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id', 'decision', 'notes']); w.writeheader()
        for row in decisions:
            if row['decision'] != 'accept': w.writerow({k: row[k] for k in w.fieldnames})
    write(out/'semantic_scope_report.json', {'scope_counts': dict(scopes), 'blocker_counts': dict(blockers),
          'source_sha256': sha(source/'object_candidates.jsonl'), 'rules_sha256': sha(ROOT/'protocol/refspatial_wp1_rules.py'),
          'fresh_queue_count': len(fresh), 'fresh_queue_family_count': len(used),
          'fresh_queue_excludes_all_previous_audit_families': not (used & old_families),
          'note': 'New audit is prospective; not reviewed. Candidate support is not certified clean support.'})
    ontology = {'ontology_version': 'refspatial_review_v1', 'status': 'DEFINITIONS_LOCKED_RETENTION_NOT_APPROVED',
       'direct_horizontal': {'leftmost_ranking': 'argmin image horizontal object-center coordinate within the named candidate class',
                             'rightmost_ranking': 'argmax image horizontal object-center coordinate within the named candidate class',
                             'horizontal_ordinal_ranking': 'explicit ordinal and direction over named class; preserve ordinal and direction'},
       'required_annotation': ['unique object ID and candidate class', 'object center definition validated against source',
                               'target point on the selected object', 'explicit reference frame', 'visually established reasoning depth'],
       'uncertainty_policy': 'Close horizontal centers, color/class ambiguity, occlusion and malformed syntax remain unsure/reject. Do not auto-relabel.',
       'distance_policy': 'Distance ranking and distance-from-anchor are separate; require documented metric, frame, candidate membership and tie rule. PNG proxy depth is not metric GT.',
       'table_edge_policy': 'Camera-relative table edge is distinct from pairwise front/behind and distance ranking.',
       'size_policy': 'Physical height, projected height and scale are distinct; unspecified size definition remains unverified.',
       'anchor_policy': 'Ranking in an anchor clause cannot certify direct target-ranking coverage.',
       'retained_main_relations': [], 'pairwise_relations_promoted_from_ranking': [], 'training_eligible': False}
    write(out/'ontology_v1.json', ontology)
    gate = {'release': 'D_tabletop_clean_v1', 'created': False, 'training_eligible': False,
            'sample_gate_status': report['status'], 'review_origins': report['review_origins'],
            'blockers': ['target/relation/anchor precision below 95%', 'retained relations not empirically approved',
                         'reference frame and object-center/metric conventions unresolved',
                         '500 certified train families per retained relation not established',
                         'split manifests and annotation extraction not certified'],
            'ontology_sha256': sha(out/'ontology_v1.json'), 'review_csv_sha256': sha(source/'manual_audit_decisions.csv'),
            'next_action': 'Resolve source semantics and adjudicate flagged samples; audit the fresh direct-ranking pool before retention.'}
    write(out/'clean_release_gate.json', gate)
    print(json.dumps({'reviews': report['completed_reviews'], 'decisions': report['decision_counts'],
                      'scopes': dict(scopes), 'fresh_queue': len(fresh), 'release_created': False}, indent=2))


if __name__ == '__main__':
    main()
