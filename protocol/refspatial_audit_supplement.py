#!/usr/bin/env python3
"""Validate downloaded media and materialize candidate manifests + manual queue.

No human label decisions, semantic correctness or training pass are fabricated.
"""
import collections
import csv
import hashlib
import html
import json
from pathlib import Path

from PIL import Image

from refspatial_build_supplement import write_json
from refspatial_fetch import DEFAULT_ROOT, REPO, REVISION, digest


def rows(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def audit(root):
    out = root / 'audit'
    out.mkdir(parents=True, exist_ok=True)
    media, errors, dimensions = {}, [], collections.Counter()
    with (out / 'media_manifest.jsonl').open('w') as manifest:
        for i, p in enumerate(sorted((root / 'media').rglob('*'))):
            if not p.is_file() or p.suffix not in ['.jpg', '.png']:
                continue
            relative = p.relative_to(root / 'media').as_posix()
            try:
                with Image.open(p) as im:
                    im.load()
                    row = {'path': relative, 'bytes': p.stat().st_size, 'sha256': digest(p),
                           'size': list(im.size), 'mode': im.mode, 'format': im.format}
                    if '/depth/' in relative:
                        row['extrema'] = im.getextrema()
                media[relative] = row
                dimensions[(relative.split('/')[0], relative.split('/')[1], tuple(row['size']), row['mode'])] += 1
                manifest.write(json.dumps(row) + '\n')
            except Exception as e:
                errors.append({'path': relative, 'error': str(e)})
            if i % 2000 == 0:
                print(f'Decoded media={len(media)} errors={len(errors)}', flush=True)
    selected = json.loads((root / 'candidate_3d/selected_frames.json').read_text())
    counts, rels = collections.Counter(), collections.Counter()
    family_by_relation = collections.defaultdict(set)
    train_family_by_relation = collections.defaultdict(set)
    families_by_split, query_answers = collections.defaultdict(set), collections.defaultdict(set)
    sources = [('3D', root / 'candidate_3d/samples_all_available_metadata.jsonl'),
               ('Simulator', root / 'simulator_restored/object_samples.jsonl')]
    candidate_rows = []
    for partition, source in sources:
        for row in rows(source):
            if partition == '3D' and Path(row['image']).name not in selected:
                continue
            if row['image'] not in media or row['depth'] not in media:
                counts[f'{partition}_missing_media_samples'] += 1
                continue
            if media[row['image']]['size'] != media[row['depth']]['size']:
                counts[f'{partition}_rgb_depth_dimension_mismatch_samples'] += 1
                continue
            qkey = (row['image'], ' '.join(row['question'].lower().split()))
            query_answers[qkey].add(tuple(row['target_xy']))
            candidate_rows.append(row)
    conflicts = {k for k, v in query_answers.items() if len(v) > 1}
    write_json(out / 'query_answer_conflicts.json', [
        {'image': k[0], 'question_normalized': k[1], 'points': sorted(query_answers[k])} for k in sorted(conflicts)])
    eligible = []
    with (out / 'samples_with_verified_media.jsonl').open('w') as f, (out / 'quarantined_conflicts.jsonl').open('w') as quarantine:
        for row in candidate_rows:
            part = row['source_partition']
            if (row['image'], ' '.join(row['question'].lower().split())) in conflicts:
                quarantine.write(json.dumps(row, ensure_ascii=False) + '\n')
                counts[f'{part}_conflicting_samples_quarantined'] += 1
                continue
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            counts[f'{part}_samples_with_verified_media'] += 1
            counts[f'{part}_{row["split"]}'] += 1
            families_by_split[row['split']].add(row['family_id'])
            for rel in row.get('relation_tags_candidate', []):
                rels[rel] += 1
                family_by_relation[rel].add(row['family_id'])
                if row['split'] == 'adaptation_train_candidate':
                    train_family_by_relation[rel].add(row['family_id'])
            eligible.append(row)
    split_overlap = {}
    splits = sorted(families_by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            split_overlap[f'{a} vs {b}'] = len(families_by_split[a] & families_by_split[b])
    # 50 different scene-query families per 3D relation where available, plus
    # 50 Simulator examples. Review remains pending; this is a queue, not a pass.
    queue, chosen = [], set()
    for rel in sorted(rels):
        candidates = sorted((r for r in eligible if rel in r.get('relation_tags_candidate', [])), key=lambda r: r['sample_id'])
        seen_families = set()
        n = 0
        for row in candidates:
            if row['family_id'] in seen_families or row['sample_id'] in chosen:
                continue
            queue.append(dict(row, audit_stratum=rel))
            chosen.add(row['sample_id'])
            seen_families.add(row['family_id'])
            n += 1
            if n >= 50:
                break
    sim = sorted((r for r in eligible if r['source_partition'] == 'Simulator'), key=lambda r: r['sample_id'])
    sim_families = set()
    for row in sim:
        if row['family_id'] in sim_families:
            continue
        queue.append(dict(row, audit_stratum='Simulator'))
        sim_families.add(row['family_id'])
        if len(sim_families) >= 50:
            break
    with (out / 'manual_audit_queue.jsonl').open('w') as f:
        for row in queue:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    review = out / 'manual_audit_decisions.csv'
    # Never replace manual work when rerunning the mechanical audit.
    if not review.exists():
        fields = ['sample_id', 'audit_stratum', 'family_id', 'split', 'target_correct', 'relation_correct',
                  'anchor_correct', 'tabletop_appropriate', 'reference_frame', 'reasoning_depth', 'notes']
        with review.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in queue:
                writer.writerow({k: row.get(k, '') for k in fields[:4]})
    body = ['<!doctype html><meta charset="utf-8"><title>RefSpatial candidate audit</title>',
            '<style>body{font:16px sans-serif;max-width:1200px;margin:24px auto}article{border-top:1px solid #bbb;padding:20px 0}.pair{display:flex;gap:12px}.im{position:relative;width:48%}.im img{width:100%;display:block}.dot{position:absolute;width:12px;height:12px;border:2px solid white;background:red;border-radius:50%;transform:translate(-50%,-50%)}small{overflow-wrap:anywhere}</style>',
            '<h1>RefSpatial — hàng đợi kiểm tra nhãn</h1><p>Chấm đỏ là target do nguồn cung cấp. Chưa xác nhận target nằm trên vật, anchor đúng, hay cảnh phù hợp tabletop. Điền quyết định vào manual_audit_decisions.csv.</p>']
    for row in queue:
        x, y = row['target_xy']
        body.append(f'<article><b>{html.escape(row["audit_stratum"])}</b> · {html.escape(row["split"])}<p>{html.escape(row["question"])}</p><div class="pair">')
        for key in ['image', 'depth']:
            body.append(f'<div class="im"><img loading="lazy" src="../media/{html.escape(row[key], quote=True)}"><span class="dot" style="left:{100*x}%;top:{100*y}%"></span></div>')
        body.append(f'</div><p>Target: {x}, {y}</p><small>{row["sample_id"]} · {html.escape(row["family_id"])}</small></article>')
    (out / 'MANUAL_AUDIT.html').write_text('\n'.join(body))
    result = {'status': 'MEDIA_VERIFIED_CANDIDATES_MANUAL_QC_PENDING', 'repo': REPO, 'revision': REVISION,
              'counts': dict(counts), 'decoded_media_files': len(media), 'decode_errors': errors,
              'media_shapes': [{'partition': k[0], 'kind': k[1], 'size': k[2], 'mode': k[3], 'count': n} for k, n in dimensions.items()],
              '3d_relation_candidate_qa': dict(rels),
              '3d_relation_candidate_families': {k: len(v) for k, v in family_by_relation.items()},
              '3d_relation_candidate_train_families': {k: len(v) for k, v in train_family_by_relation.items()},
              'family_split_overlap': split_overlap, 'conflicting_scene_queries_quarantined': len(conflicts),
              'manual_audit_queue_count': len(queue), 'manual_audit_completed': False,
              'calibration_or_final_test_eligible': False, 'training_executed': False,
              'dataset_disk_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}
    write_json(out / 'audit_report.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    audit(DEFAULT_ROOT)
