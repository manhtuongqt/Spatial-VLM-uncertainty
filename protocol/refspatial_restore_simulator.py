#!/usr/bin/env python3
"""Restore the union of user Drive scene selections from pinned Simulator.

Requires the five Drive manifests/metadata downloaded by the earlier audit.
It reproduces a selection, not the missing original tabletop filtering code.
"""
import argparse
import collections
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from refspatial_build_supplement import json_array, point, write_json
from refspatial_fetch import DEFAULT_ROOT, REVISION, digest

NAMES = ['large', 'tabletop', 'cup', 'handle', 'wrist']
OBJECT_TYPES = {
    'Identifying key object features, ranking objects based on those features, and then determining which specific object is being referenced.': 'object_features_ranking',
    'Identifying and selecting objects positioned along the edges of a tabletop.': 'object_at_table_edge',
    'Identifying object distances from a reference object.': 'object_distance_from_anchor',
}


def fingerprint(r):
    return hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def scan(root, drive_dir):
    out = root / 'simulator_restored'
    out.mkdir(parents=True, exist_ok=True)
    scenes, wanted, provenance = {}, set(), {}
    for name in NAMES:
        for kind in ['scenes', 'metadata']:
            src = drive_dir / f'refspatial_{name}_{kind}'
            provenance[f'{name}/{kind}'] = {'bytes': src.stat().st_size, 'sha256': digest(src)}
            if kind == 'scenes':
                # Preserve small complete manifests so the selection is rebuildable.
                dst = out / 'drive_manifests' / f'{name}.scenes.jsonl'
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                with src.open() as f:
                    for line in f:
                        r = json.loads(line)
                        if r['id'] in scenes and scenes[r['id']]['drive_split'] != r['split']:
                            raise ValueError('Conflicting Drive scene splits')
                        scenes.setdefault(r['id'], {'scene_id': r['id'], 'drive_split': r['split'], 'source_sets': []})
                        scenes[r['id']]['source_sets'].append(name)
            else:
                with src.open() as f:
                    for line in f:
                        wanted.add(fingerprint(json.loads(line)))
    write_json(out / 'drive_input_hashes.json', provenance)
    counts, types, matched, seen = collections.Counter(), collections.Counter(), set(), set()
    matched_scenes, raw_scenes = set(), set()
    with (out / 'source_records.jsonl').open('w') as records, (out / 'object_samples.jsonl').open('w') as samples:
        for ri, r in enumerate(json_array(root / 'source/Simulator/metadata.json')):
            counts['source_records_total'] += 1
            sid = Path(r['image'][0]).stem
            raw_scenes.add(sid)
            types.update(t['question_type'] for t in r.get('think', []))
            if sid not in scenes:
                continue
            counts['source_records_in_drive_scene_union'] += 1
            matched_scenes.add(sid)
            fp = fingerprint(r)
            if fp in wanted:
                matched.add(fp)
            else:
                counts['source_records_in_selected_scenes_not_in_drive_metadata'] += 1
                # Preserve user filtering: do not silently add omitted source QA.
                continue
            records.write(json.dumps({'source_record_index_zero_based': ri, 'record': r}, ensure_ascii=False) + '\n')
            counts['restored_source_records'] += 1
            conv, thinking = r['conversations'], r.get('think', [])
            if len(conv) != 2 * len(thinking):
                counts['bad_qa_think_alignment'] += 1
                continue
            for qi, t in enumerate(thinking):
                typ = OBJECT_TYPES.get(t['question_type'])
                if typ is None:
                    counts['placement_or_unmapped_qa_excluded'] += 1
                    continue
                q, a = conv[2 * qi], conv[2 * qi + 1]
                xy = point(a['value'])
                if xy is None or q['from'] != 'human' or a['from'] != 'gpt':
                    counts['invalid_object_qa'] += 1
                    continue
                key = (sid, ' '.join(q['value'].split()), tuple(xy))
                uid = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
                if uid in seen:
                    counts['duplicate_object_qa_removed'] += 1
                    continue
                seen.add(uid)
                scene = scenes[sid]
                row = {'sample_id': uid, 'source_partition': 'Simulator', 'source_revision': REVISION,
                       'source_file': 'Simulator/metadata.json', 'source_record_index_zero_based': ri,
                       'source_record_id': r['id'], 'source_qa_index_zero_based': qi,
                       'family_id': f'RefSpatialSimulator:scene:{sid}', 'scene_id': sid,
                       'split': {'train': 'adaptation_train_candidate', 'validation': 'internal_dev_candidate',
                                 'test': 'internal_diagnostic_test_candidate'}[scene['drive_split']],
                       'drive_split': scene['drive_split'], 'source_sets': scene['source_sets'],
                       'image': f'Simulator/image/{sid}.jpg', 'depth': f'Simulator/depth/{sid}.png',
                       'question': q['value'], 'answer_original': a['value'], 'target_xy': xy,
                       'question_type': typ, 'task': 'native_object_point_grounding_candidate',
                       'qc_status': 'NEEDS_MANUAL_TARGET_RELATION_ANCHOR_AND_TABLETOP_QC',
                       'answerability': None, 'anchor_ids': None, 'reasoning_depth': None,
                       'thinking_available_in_source_only': True}
                samples.write(json.dumps(row, ensure_ascii=False) + '\n')
                counts['object_qa_candidates'] += 1
            if ri % 25000 == 0:
                print(f'Simulator scan records={ri} matched={len(matched)}', flush=True)
    with (out / 'scenes.jsonl').open('w') as f:
        for sid, scene in sorted(scenes.items()):
            f.write(json.dumps(scene) + '\n')
    report = {'counts': dict(counts), 'source_unique_image_scenes': len(raw_scenes),
              'source_question_type_counts': dict(types), 'drive_union_scenes': len(scenes),
              'matched_source_scenes': len(matched_scenes), 'missing_source_scenes': sorted(set(scenes) - matched_scenes),
              'unique_drive_record_fingerprints': len(wanted), 'exact_canonical_record_matches': len(matched),
              'missing_drive_record_fingerprints': sorted(wanted - matched),
              'original_filter_code_recovered': False, 'selection_replay_from_saved_manifests': True,
              'status': 'SOURCE_LINEAGE_AUDIT_NOT_MANUAL_TRAINING_QC'}
    write_json(out / 'source_lineage_audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['source_question_type_counts', 'missing_drive_record_fingerprints']}), flush=True)


def extract(root):
    out = root / 'simulator_restored'
    scenes = {json.loads(s)['scene_id'] for s in (out / 'scenes.jsonl').read_text().splitlines()}
    results = {}
    for kind, ext in [('image', '.jpg'), ('depth', '.png')]:
        dst = root / 'media/Simulator' / kind
        dst.mkdir(parents=True, exist_ok=True)
        found = set()
        with tarfile.open(root / f'source/Simulator/{kind}/{kind}.tar.gz', 'r|gz') as archive:
            for m in archive:
                name = Path(m.name).name
                sid = Path(name).stem
                if m.isfile() and sid in scenes and name.endswith(ext):
                    if m.size > 32 * 1024**2 or shutil.disk_usage(root).free < 3 * 1024**3 + m.size:
                        raise OSError('Unsafe member size or free-space reserve')
                    with archive.extractfile(m) as source, (dst / name).open('wb') as target:
                        shutil.copyfileobj(source, target)
                    found.add(sid)
        results[kind] = {'requested': len(scenes), 'found': len(found), 'missing': sorted(scenes - found)}
        print(f'Simulator {kind}: {len(found)}/{len(scenes)}', flush=True)
    write_json(out / 'media_extract.json', results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['scan', 'extract'])
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--drive-dir', type=Path, default=Path('/tmp'))
    a = p.parse_args()
    scan(a.root, a.drive_dir) if a.action == 'scan' else extract(a.root)


if __name__ == '__main__':
    main()
