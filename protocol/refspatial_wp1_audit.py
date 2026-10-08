#!/usr/bin/env python3
"""Read-only full local media/metadata audit; generates candidates, NOT clean labels.

Run: python3 protocol/refspatial_wp1_audit.py
No source edits, downloads, training or final-test prediction access.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import html
import itertools
import json
import os
from pathlib import Path
import time

from PIL import Image, ImageChops
from refspatial_wp1_rules import (QUESTION_TYPES, RELATIONS, anchor_candidates,
                                  corrected_categories, normalized, point,
                                  relation_tags, semantic_review_requirements, target_kind)

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['RefSpatial-Tabletop-Large', 'RefSpatial-Tabletop', 'RefSpatial-Handle-Occlusion',
         'RefSpatial-Cup-Occlusion', 'RefSpatial-WristLike-Cup-Fruit-Occlusion']
SPLITS = {'train': 'train', 'validation': 'dev', 'test': 'diagnostic'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def jsonl(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def dump_line(f, value):
    f.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def csv_write(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inspect_media(task):
    folder, kind, p, refs = task
    row = {'dataset': folder, 'kind': kind, 'path': str(p.relative_to(ROOT)),
           'referenced_by_scenes': refs, 'bytes': p.stat().st_size, 'sha256': sha(p)}
    try:
        with Image.open(p) as im:
            im.load()
            row.update(size=list(im.size), mode=im.mode, format=im.format,
                       pixel_sha256=hashlib.sha256(im.mode.encode() + str(im.size).encode() + im.tobytes()).hexdigest(),
                       decode_ok=True)
            if kind == 'depth':
                row['extrema'] = im.getextrema()
                bands = im.split()
                row['channels_equal'] = (len(bands) == 1 or
                    all(ImageChops.difference(bands[0], band).getbbox() is None for band in bands[1:]))
    except Exception as e:
        row.update(decode_ok=False, error=f'{type(e).__name__}: {e}')
    return row


def run(out, workers=4):
    started = time.time()
    decisions = out / 'manual_audit_decisions.csv'
    if decisions.exists():
        with decisions.open() as f:
            if any(row.get('decision', '').strip() for row in csv.DictReader(f)):
                raise ValueError('Completed reviews exist; use a new --out directory to preserve audit provenance.')
    out.mkdir(parents=True, exist_ok=True)
    scenes, source_hashes, problems, tasks = {}, [], [], []
    for name in NAMES:
        folder = ROOT / 'datasets' / name
        scenes[name] = {}
        for s in jsonl(folder / 'scenes.jsonl'):
            if s['id'] in scenes[name]:
                problems.append({'dataset': name, 'scene_id': s['id'], 'reason': 'duplicate_scene_id'})
            scenes[name][s['id']] = s
        for p in sorted(folder.iterdir()):
            if p.is_file():
                source_hashes.append({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)})
        for kind, subdir in [('rgb', 'image'), ('depth', 'depth')]:
            references = collections.defaultdict(list)
            for sid, s in scenes[name].items():
                key = s['image' if kind == 'rgb' else 'depth']
                references[key].append(sid)
                if not (folder / key).is_file():
                    problems.append({'dataset': name, 'scene_id': sid, 'path': key, 'reason': 'missing_media'})
            for p in sorted((folder / subdir).rglob('*')):
                if p.is_file():
                    tasks.append((name, kind, p, references.get(p.relative_to(folder).as_posix(), [])))
    write_json(out / 'source_inventory.json', source_hashes)
    print(f'Media audit: {len(tasks)} files; workers={workers}', flush=True)
    media, content, counts = {}, collections.defaultdict(list), collections.Counter()
    with (out / 'media_manifest.jsonl').open('w') as f, ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(inspect_media, tasks), 1):
            media[row['path']] = row
            dump_line(f, row)
            counts['files'] += 1
            counts['bytes'] += row['bytes']
            counts['decode_ok' if row['decode_ok'] else 'decode_failed'] += 1
            if not row['referenced_by_scenes']:
                counts['extra_files'] += 1
            for method in ['sha256', 'pixel_sha256']:
                if method in row:
                    content[(row['kind'], method, row[method])].append(row['path'])
            if i % 2000 == 0:
                print(f'Decoded {i}/{len(tasks)} files', flush=True)
    hash_duplicates, leaks, unsafe_scenes = [], [], set()
    for (kind, method, digest), paths in content.items():
        if len(paths) < 2:
            continue
        refs = []
        for path in paths:
            row = media[path]
            refs.extend({'dataset': row['dataset'], 'scene_id': sid,
                         'split': scenes[row['dataset']][sid]['split']} for sid in row['referenced_by_scenes'])
        entry = {'kind': kind, 'method': method, 'hash': digest, 'paths': paths,
                 'scene_ids': sorted({r['scene_id'] for r in refs}), 'splits': sorted({r['split'] for r in refs})}
        hash_duplicates.append(entry)
        if len(entry['splits']) > 1:
            leaks.append(entry)
            unsafe_scenes.update(entry['scene_ids'])
    overlaps = []
    for a, b in itertools.combinations(NAMES, 2):
        common = scenes[a].keys() & scenes[b].keys()
        conflicts = sorted(s for s in common if scenes[a][s]['split'] != scenes[b][s]['split'])
        unsafe_scenes.update(conflicts)
        overlaps.append({'a': a, 'b': b, 'shared_scene_ids': len(common), 'split_conflict_scene_ids': conflicts})
    for name, rows in scenes.items():
        for sid, s in rows.items():
            paths = [str(Path('datasets') / name / s[k]) for k in ['image', 'depth']]
            pair = [media.get(p) for p in paths]
            if (not all(r and r['decode_ok'] for r in pair)
                    or (pair[0]['size'] != pair[1]['size'])
                    or Path(s['image']).stem != sid or Path(s['depth']).stem != sid):
                unsafe_scenes.add(sid)
                problems.append({'dataset': name, 'scene_id': sid, 'reason': 'invalid_manifest_media_pair'})
    extras = []
    for row in media.values():
        if not row['referenced_by_scenes']:
            matches = content[(row['kind'], 'sha256', row['sha256'])]
            extras.append({'path': row['path'], 'sha256': row['sha256'],
                           'identical_referenced_files': [p for p in matches if media[p]['referenced_by_scenes']],
                           'action': 'EXCLUDED_FROM_CANDIDATES_SOURCE_UNCHANGED'})
    write_json(out / 'media_qc_report.json', {'counts': dict(counts), 'extra_files': extras,
        'structural_problems': problems, 'cross_split_hash_leak_groups': len(leaks),
        'duplicate_content_groups': len(hash_duplicates), 'unsafe_scene_ids': sorted(unsafe_scenes),
        'formats': dict(collections.Counter(f"{r['kind']}:{r.get('mode')}:{r.get('size')}" for r in media.values())),
        'depth_channels_not_equal': sum(r.get('channels_equal') is False for r in media.values())})
    write_json(out / 'overlap_report.json', {'union_scene_count': len(set().union(*(set(s) for s in scenes.values()))),
                                           'folder_overlaps': overlaps, 'cross_split_content_leaks': leaks})
    with (out / 'duplicate_content_groups.jsonl').open('w') as f:
        for row in hash_duplicates:
            dump_line(f, row)

    # Correct known taxonomy mistakes in a DERIVED table; never edit source labels.
    taxonomy = []
    with (out / 'taxonomy_derived_points.jsonl').open('w') as f:
        for name, ss in scenes.items():
            for sid, s in ss.items():
                for p in s.get('points', []):
                    updated, evidence = corrected_categories(p['labels'], p.get('categories', []))
                    if updated != sorted(p.get('categories', [])) or any('orange' in l.lower() or 'juice' in l.lower() for l in p['labels']):
                        row = {'dataset': name, 'scene_id': sid, 'xy': p['xy'], 'labels': p['labels'],
                               'original': p.get('categories', []), 'derived': updated, 'evidence': evidence,
                               'visual_review': 'PENDING'}
                        dump_line(f, row)
                        taxonomy.append({'dataset': name, 'scene_id': sid, 'xy': str(p['xy']),
                            'original_categories': '|'.join(sorted(p.get('categories', []))),
                            'derived_categories': '|'.join(updated), 'changed': updated != sorted(p.get('categories', [])),
                            'labels': ' | '.join(p['labels'])})
    csv_write(out / 'taxonomy_audit.csv', taxonomy,
              ['dataset', 'scene_id', 'xy', 'original_categories', 'derived_categories', 'changed', 'labels'])

    report, candidates, conflicts = {}, [], set()
    with (out / 'quarantine.jsonl').open('w') as quarantine:
        for name in NAMES:
            print('Metadata audit:', name, flush=True)
            c, types, seen, answers = collections.Counter(), collections.Counter(), set(), {}
            rec_counts, qa_counts = collections.Counter(), collections.Counter()
            for ri, r in enumerate(jsonl(ROOT / 'datasets' / name / 'metadata.jsonl'), 1):
                c['records'] += 1
                images, depths = r.get('image', []), r.get('depth', [])
                if not (len(images) == len(depths) == 1):
                    c['bad_media_reference_shape'] += 1
                    dump_line(quarantine, {'dataset': name, 'record_line': ri, 'reason': 'bad_media_reference_shape'})
                    continue
                sid = Path(images[0]).stem
                s = scenes[name].get(sid)
                conv, thinking = r.get('conversations', []), r.get('think', [])
                if s is None or sid != Path(depths[0]).stem or len(conv) != 2 * len(thinking):
                    c['bad_scene_or_qa_alignment'] += 1
                    dump_line(quarantine, {'dataset': name, 'record_line': ri, 'reason': 'bad_scene_or_qa_alignment'})
                    continue
                rec_counts[sid] += 1
                c['record_id_differs_from_scene_id'] += r.get('id') != sid
                for qi, t in enumerate(thinking):
                    q, a = conv[2 * qi:2 * qi + 2]
                    c['qa'] += 1
                    qa_counts[sid] += 1
                    kind, canonical = target_kind(t.get('question_type'))
                    types[canonical] += 1
                    c[kind + '_qa'] += 1
                    question, answer = q.get('value', ''), a.get('value', '')
                    xy = point(answer)
                    key = (sid, normalized(question), normalized(answer))
                    query_key = key[:2]
                    canonical_answer = xy if xy is not None else normalized(answer)
                    duplicate = key in seen
                    seen.add(key)
                    if duplicate:
                        c[kind + '_duplicate_occurrences'] += 1
                    if query_key in answers and answers[query_key] != canonical_answer:
                        c[kind + '_answer_conflict_occurrences'] += 1
                        if name == NAMES[0] and kind == 'object':
                            conflicts.add(query_key)
                    answers[query_key] = canonical_answer
                    reason = []
                    if xy is None:
                        reason.append('invalid_target_point')
                    if q.get('from') != 'human' or a.get('from') != 'gpt':
                        reason.append('invalid_role_order')
                    if kind == 'unknown':
                        reason.append('unknown_question_type')
                    if sid in unsafe_scenes:
                        reason.append('unsafe_scene_media_or_split')
                    recovered = [p for p in s.get('points', []) if xy is not None and tuple(p['xy']) == xy]
                    if kind == 'object' and not recovered:
                        reason.append('target_not_exact_recovered_scene_point')
                    if reason:
                        c['quarantined_qa'] += 1
                        dump_line(quarantine, {'dataset': name, 'record_line': ri, 'qa_index': qi,
                            'scene_id': sid, 'question': question, 'answer': answer, 'reasons': reason})
                    if name != NAMES[0] or kind != 'object' or duplicate or reason:
                        continue
                    uid = hashlib.sha256(json.dumps([sid, normalized(question), xy]).encode()).hexdigest()[:24]
                    anchors = anchor_candidates(question, t.get('thinking', ''), xy)
                    candidates.append({'sample_id': uid, 'scene_id': sid, 'family_id': f'RefSpatialSimulator:scene:{sid}',
                        'split': SPLITS[s['split']], 'source_record_line': ri, 'source_qa_index': qi,
                        'source_sets': [n for n in NAMES if sid in scenes[n]],
                        'stress_tags_proxy_only': [n for n in NAMES[2:] if sid in scenes[n]],
                        'question_type_raw': t['question_type'], 'target_kind': kind, 'canonical_type': canonical,
                        'instruction': question, 'target_xy': list(xy),
                        'image': str(Path('datasets') / name / s['image']), 'depth': str(Path('datasets') / name / s['depth']),
                        'relation_candidates': relation_tags(question, kind), 'target_labels_offline': recovered[0]['labels'],
                        'anchor_candidates_offline': anchors, 'reference_frame': 'UNVERIFIED',
                        'semantic_review': semantic_review_requirements(question, kind, canonical),
                        'reasoning_depth': None, 'label_qc': 'MANUAL_REVIEW_PENDING', 'training_eligible': False})
            mismatch = [sid for sid, s in scenes[name].items()
                        if rec_counts[sid] != s['source_record_count'] or qa_counts[sid] != s['qa_count']]
            c['scene_record_or_qa_count_mismatches'] = len(mismatch)
            report[name] = {'counts': dict(c), 'question_types': dict(types), 'count_mismatch_scenes': mismatch,
                            'split_scenes': dict(collections.Counter(s['split'] for s in scenes[name].values()))}
        remaining = []
        for row in candidates:
            if (row['scene_id'], normalized(row['instruction'])) in conflicts:
                dump_line(quarantine, {**row, 'reason': 'conflicting_object_answers'})
            else:
                remaining.append(row)
        candidates = remaining
    write_json(out / 'qa_type_report.json', {'mapping': QUESTION_TYPES, 'datasets': report})
    with (out / 'object_candidates.jsonl').open('w') as f:
        for row in candidates:
            dump_line(f, row)
    # Keep stress-only scenes outside Large separate from the adaptation pool.
    with (out / 'qualitative_only_scenes.jsonl').open('w') as f:
        outside = set().union(*(set(scenes[n]) for n in NAMES[2:])) - set(scenes[NAMES[0]])
        for sid in sorted(outside):
            dump_line(f, {'scene_id': sid, 'source_sets': [n for n in NAMES[2:] if sid in scenes[n]],
                          'role': 'QUALITATIVE_ONLY_MANUAL_QC_PENDING', 'training_eligible': False})
    feasibility = []
    for rel in RELATIONS:
        rows = [r for r in candidates if rel in r['relation_candidates']]
        train = {r['family_id'] for r in rows if r['split'] == 'train'}
        anchors = {r['family_id'] for r in rows if r['split'] == 'train' and len(r['anchor_candidates_offline']) == 1}
        feasibility.append({'relation_candidate': rel, 'qa_candidates': len(rows), 'train_scene_families': len(train),
            'dev_scene_families': len({r['family_id'] for r in rows if r['split'] == 'dev'}),
            'diagnostic_scene_families': len({r['family_id'] for r in rows if r['split'] == 'diagnostic'}),
            'train_families_with_single_offline_anchor': len(anchors), 'audited_clean_train_families': 0,
            'candidate_support_ge_500': len(train) >= 500, 'main_experiment_eligible': False,
            'status': 'MANUAL_QC_FRAME_REASONING_PENDING' if len(rows) else 'NO_LEXICAL_CANDIDATES_NOT_PROOF_OF_SEMANTIC_ABSENCE'})
    csv_write(out / 'relation_feasibility.csv', feasibility, list(feasibility[0]))

    # Fixed-hash ordering, 50 unique scenes per relation across all three splits.
    # This is an audit queue, NEVER completed human annotation.
    selected, quota = {}, {}
    for rel in RELATIONS:
        pool = sorted((r for r in candidates if rel in r['relation_candidates']), key=lambda r: r['sample_id'])
        family_seen, chosen = set(), []
        for split, want in [('train', 30), ('dev', 10), ('diagnostic', 10)]:
            for row in (r for r in pool if r['split'] == split):
                if row['family_id'] in family_seen:
                    continue
                family_seen.add(row['family_id'])
                chosen.append(row)
                if sum(r['split'] == split for r in chosen) >= want:
                    break
        for row in pool:
            if len(chosen) >= 50:
                break
            if row['family_id'] not in family_seen:
                chosen.append(row)
                family_seen.add(row['family_id'])
        quota[rel] = {'queued': len(chosen), 'required_before_retention': 50, 'shortfall': max(0, 50-len(chosen))}
        for row in chosen:
            selected.setdefault(row['sample_id'], dict(row, audit_strata=[]))['audit_strata'].append(rel)
    # Add broad object-QA coverage up to at least 300 distinct scene-query items.
    families = {r['family_id'] for r in selected.values()}
    for row in sorted(candidates, key=lambda r: r['sample_id']):
        if len(selected) >= 300:
            break
        if row['family_id'] in families:
            continue
        families.add(row['family_id'])
        selected[row['sample_id']] = dict(row, audit_strata=['object_grounding_general'])
    queue = sorted(selected.values(), key=lambda r: (r['audit_strata'][0], r['split'], r['sample_id']))
    with (out / 'manual_audit_queue.jsonl').open('w') as f:
        for row in queue:
            dump_line(f, row)
    fields = ['sample_id', 'scene_id', 'split', 'audit_strata', 'reviewer', 'reviewed_at',
              'target_correct', 'relation_correct', 'anchor_correct', 'tabletop_appropriate',
              'reference_frame', 'reasoning_depth', 'decision', 'notes']
    decisions = out / 'manual_audit_decisions.csv'
    if not decisions.exists():
        csv_write(decisions, [dict(sample_id=r['sample_id'], scene_id=r['scene_id'], split=r['split'],
                                  audit_strata='|'.join(r['audit_strata'])) for r in queue], fields)
    else:
        existing = {r['sample_id'] for r in csv.DictReader(decisions.open())}
        if existing != set(selected):
            raise ValueError('Existing manual decisions refer to a different queue; use a new output version.')
    body = ['<!doctype html><html lang="vi"><meta charset="utf-8"><title>RefSpatial manual audit</title>',
            '<style>body{font:16px sans-serif;max-width:1250px;margin:24px auto}article{border-top:1px solid #bbb;padding:20px 0}.pair{display:flex;gap:12px}.im{position:relative;width:49%}.im img{width:100%;display:block}.dot{position:absolute;width:12px;height:12px;border:2px solid white;background:red;border-radius:50%;transform:translate(-50%,-50%)}small{overflow-wrap:anywhere}</style>',
            '<h1>RefSpatial — hàng đợi manual audit</h1><p>Chấm đỏ: target nguồn, chưa xác nhận đúng. Nhãn relation/anchor bên dưới chỉ là ứng viên offline. Không dùng thông tin oracle này cho inference.</p>',
            '<p>Điền phiếu <a href="manual_audit_decisions.csv">CSV</a> theo <a href="MANUAL_AUDIT_GUIDE.md">hướng dẫn</a>. Review chưa hoàn tất; không có nhãn được tự duyệt.</p>']
    for i, row in enumerate(queue, 1):
        x, y = row['target_xy']
        body.append(f'<article id="{row["sample_id"]}"><h2>{i}. {html.escape(" / ".join(row["audit_strata"]))} · {row["split"]}</h2><small>{row["sample_id"]} · {row["scene_id"]}</small><p>{html.escape(row["instruction"])}</p><div class="pair">')
        for key in ['image', 'depth']:
            src = os.path.relpath(ROOT / row[key], out)
            body.append(f'<div class="im"><img loading="lazy" src="{html.escape(src, quote=True)}"><span class="dot" style="left:{100*x}%;top:{100*y}%"></span></div>')
        body.append('</div><p>Target labels (offline): ' + html.escape(' | '.join(row['target_labels_offline'])) + '</p><p>Anchor candidates (offline): ' + html.escape(json.dumps(row['anchor_candidates_offline'], ensure_ascii=False)) + '</p></article>')
    (out / 'MANUAL_AUDIT.html').write_text('\n'.join(body) + '</html>\n')
    write_json(out / 'manual_audit_status.json', {'status': 'PENDING_HUMAN_REVIEW', 'queue_count': len(queue),
        'unique_scenes': len({r['scene_id'] for r in queue}), 'minimum_total': 300, 'relation_quotas': quota,
        'completed_reviews': None, 'note': 'Run review validator to count submitted decisions. Queue generation does not certify reviews.',
        'sampling': 'SHA256 sample_id order; 30/10/10 train/dev/diagnostic per lexical relation, unique scenes within each stratum; fill total to 300'})
    status = {'status': 'AUTOMATED_AUDIT_COMPLETE_MANUAL_QC_PENDING', 'media': dict(counts),
        'candidate_qa_after_structural_gate': len(candidates), 'candidate_scene_families': len({r['family_id'] for r in candidates}),
        'taxonomy_changed_points_all_folders': sum(r['changed'] for r in taxonomy),
        'cross_split_content_leak_groups': len(leaks), 'manual_audit_queue': len(queue),
        'training_eligible': False, 'D_tabletop_clean_created': False, 'elapsed_seconds': round(time.time()-started, 2),
        'source_mutations': False, 'raw_source_provenance': 'NOT_RECOVERED',
        'script_sha256': sha(Path(__file__)), 'rules_sha256': sha(Path(__file__).with_name('refspatial_wp1_rules.py'))}
    write_json(out / 'audit_summary.json', status)
    write_json(out / 'provenance_status.json', {'derived_source_files': 'SHA256_LOCKED',
        'original_raw_metadata_exists': Path('/home/manhtuong/ros2_ws/datasets/RefSpatial/Simulator/metadata.json').exists(),
        'original_filter_reproduced': False, 'original_source_revision_verified': False, 'license_verified': False,
        'roborefer_pretraining_lineage': 'POTENTIALLY_SEEN', 'scope': 'AUDIT_EXISTING_DERIVED_OUTPUTS_ONLY'})
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/spatial_vlm_refspatial_v1/wp1_source_audit')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.out.resolve(), args.workers)
