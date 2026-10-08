#!/usr/bin/env python3
"""Build source-traceable RefSpatial candidates; does not certify training QC.

Python stdlib except requests (stream-images) and Pillow (audit-media).
Candidate relation tags are query-template matches, not audited truth labels.
"""
import argparse
import ast
import collections
import hashlib
import json
import math
import re
import shutil
import tarfile
import time
from pathlib import Path

from refspatial_fetch import DEFAULT_ROOT, REPO, REVISION, digest

RELATIONS = {
    'left': r'\b(?:left of|left side of)\b',
    'right': r'\b(?:right of|right side of)\b',
    'front': r'\b(?:in front of|front of)\b',
    'behind': r'\bbehind\b',
    'between': r'\bbetween\b',
    'nearest': r'\b(?:nearest|closest)\b',
    'farthest': r'\b(?:farthest|furthest)\b',
}
SMALL_OBJECT = re.compile(
    r'\b(?:cups?|mugs?|bottles?|bowls?|plates?|(?:soda|tin|beverage) cans?|apples?|bananas?|'
    r'pears?|plums?|fruits?|boxes|box|books?|toys?|sponges?|remotes?|phones?|'
    r'forks?|spoons?|knives|knife|scissors|tapes?|balls?|jars?|cartons?|'
    r'containers?|kettles?|packets?|packages?|pens?|pencils?|erasers?|'
    r'cups?|tumblers?|(?:wine|drinking) glasses|(?:wine|drinking) glass|dishes|dish|pans?|pots?)\b', re.I)


def json_array(path):
    """Incremental top-level JSON array parser, bounded by one record + 1 MiB."""
    dec, buf, pos, started, ended = json.JSONDecoder(), '', 0, False, False
    with Path(path).open(encoding='utf-8') as f:
        while True:
            chunk = f.read(1024**2)
            buf = buf[pos:] + chunk
            pos = 0
            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if not started:
                    if pos == len(buf):
                        break
                    if buf[pos] != '[':
                        raise ValueError('Expected top-level JSON array')
                    started, pos = True, pos + 1
                    continue
                if pos < len(buf) and buf[pos] == ',':
                    pos += 1
                    continue
                if pos < len(buf) and buf[pos] == ']':
                    ended = True
                    pos += 1
                    break
                try:
                    record, end = dec.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break
                yield record
                pos = end
            if ended:
                if buf[pos:].strip() or f.read().strip():
                    raise ValueError('Trailing data after JSON array')
                return
            if not chunk:
                raise ValueError('Truncated JSON array')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def point(answer):
    try:
        a = ast.literal_eval(answer)
        if (isinstance(a, (list, tuple)) and len(a) == 1
                and isinstance(a[0], (list, tuple)) and len(a[0]) == 2
                and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in a[0])):
            return list(a[0])
    except (ValueError, SyntaxError, TypeError):
        pass
    return None


def tags(question):
    # Strip generic output instructions: "between 0 and 1" is not a relation.
    query = question.split('Your answer should')[0]
    return [name for name, rx in RELATIONS.items() if re.search(rx, query, re.I)]


def candidate_target(question):
    query = question.split('Your answer should')[0]
    # Some templates introduce an anchor first, then say "Pinpoint the ...".
    # Restrict the noun test to the actual referring clause when it is explicit.
    commands = list(re.finditer(r'\b(?:point to|point out|pinpoint|locate|identify|select|find)\b', query, re.I))
    if commands:
        query = query[commands[-1].end():]
    # Require a small-object noun BEFORE the first relation, so an anchor alone
    # cannot turn a free-space / furniture target into a manipulation target.
    matches = [m for rx in RELATIONS.values() if (m := re.search(rx, query, re.I))]
    prefix = query[:min(m.start() for m in matches)] if matches else query
    m = SMALL_OBJECT.search(prefix)
    return m.group().lower() if m else None


def scan_3d(root):
    src = root / 'source/3D/reasoning_template_qa.json'
    out = root / 'candidate_3d'
    out.mkdir(parents=True, exist_ok=True)
    counts, rels, examples = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    frames, seen = {}, set()
    with (out / 'samples_all_available_metadata.jsonl').open('w') as f:
        for ri, record in enumerate(json_array(src)):
            counts['source_records'] += 1
            conv = record.get('conversations', [])
            if len(record.get('image', [])) != 1 or len(record.get('depth', [])) != 1:
                counts['non_single_view_records'] += 1
                continue
            image, depth = record['image'][0], record['depth'][0]
            video = Path(image).name.split('_')[0]
            for qi in range(0, len(conv) - 1, 2):
                q, a = conv[qi], conv[qi + 1]
                counts['source_qa'] += 1
                if q.get('from') != 'human' or a.get('from') != 'gpt':
                    counts['bad_roles'] += 1
                    continue
                xy, rt = point(a['value']), tags(q['value'])
                if xy is not None:
                    counts['single_point_qa'] += 1
                if xy is None or not rt:
                    continue
                counts['single_point_relation_qa_all_targets'] += 1
                target = candidate_target(q['value'])
                if not target:
                    continue
                # Ordinal/index is the source locator; source IDs are not unique.
                key = (image, ' '.join(q['value'].split()), tuple(xy))
                if key in seen:
                    counts['exact_duplicate_qa_removed'] += 1
                    continue
                seen.add(key)
                uid = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
                family = f'RefSpatial3D:video:{video}'
                bucket = int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 100
                sample = {'sample_id': uid, 'source_partition': '3D',
                          'source_file': '3D/reasoning_template_qa.json', 'source_revision': REVISION,
                          'source_record_index_zero_based': ri, 'source_record_id': record['id'],
                          'source_qa_index_zero_based': qi // 2,
                          'family_id': family, 'split': 'adaptation_train_candidate' if bucket < 80 else 'internal_dev_candidate',
                          'image': f'3D/image/{Path(image).name}', 'depth': f'3D/depth/{Path(depth).name}',
                          'question': q['value'], 'answer_original': a['value'], 'target_xy': xy,
                          'relation_tags_candidate': rt, 'target_keyword_candidate': target,
                          'reference_frame': 'source_query_semantics_NOT_robot_frame',
                          'task': 'native_object_point_grounding_candidate',
                          'qc_status': 'NEEDS_MANUAL_TARGET_RELATION_ANCHOR_AND_TABLETOP_QC',
                          'anchor_ids': None, 'reasoning_depth': None, 'answerability': None}
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                frames.setdefault(Path(image).name, {'depth': Path(depth).name, 'family_id': family,
                                                    'relation_tags_candidate': [], 'qa_count': 0})
                fr = frames[Path(image).name]
                fr['relation_tags_candidate'] = sorted(set(fr['relation_tags_candidate']) | set(rt))
                fr['qa_count'] += 1
                counts['small_object_relation_point_candidates'] += 1
                rels.update(rt)
                for rel in rt:
                    if len(examples[rel]) < 8:
                        examples[rel].append(sample)
            if ri % 25000 == 0:
                print(f'SCAN 3D records={ri} candidates={counts["small_object_relation_point_candidates"]}', flush=True)
    write_json(out / 'frames_available_metadata.json', frames)
    write_json(out / 'metadata_scan.json', {'counts': dict(counts), 'relations_candidate': dict(rels),
                                          'frames': len(frames), 'video_families': len({x['family_id'] for x in frames.values()}),
                                          'examples': dict(examples), 'status': 'LEXICAL_CANDIDATES_NOT_QC_PASS'})
    print(json.dumps({'counts': dict(counts), 'relations': dict(rels), 'frames': len(frames)}), flush=True)


def stream_images(root, max_frames, per_family, max_compressed_gib):
    import requests
    frames = json.loads((root / 'candidate_3d/frames_available_metadata.json').read_text())
    out = root / 'media/3D/image'
    out.mkdir(parents=True, exist_ok=True)
    selected, families = {}, collections.Counter()
    for p in sorted(out.glob('*.png')):
        if p.name in frames:
            selected[p.name] = frames[p.name]
            families[frames[p.name]['family_id']] += 1
    catalog = json.loads((root / 'official_tree.json').read_text())
    entry = next(x for x in catalog if x['path'] == '3D/image/image.tar.gz')
    stats = {'archive_path': entry['path'], 'revision': REVISION, 'source_archive_sha256': entry['lfs']['oid'],
             'full_archive_sha256_verified': False, 'selection': 'archive_order_prefix_then_keyword_filter_and_video_cap',
             'representative_random_sample': False, 'max_frames': max_frames, 'per_family_cap': per_family,
             'max_compressed_gib': max_compressed_gib, 'members_scanned': 0}
    tick = time.monotonic()
    url = f'{REPO}/resolve/{REVISION}/3D/image/image.tar.gz?download=true&stream_subset=v1'
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with tarfile.open(fileobj=response.raw, mode='r|gz') as tf:
            for member in tf:
                stats['members_scanned'] += 1
                if len(selected) >= max_frames or response.raw.tell() >= max_compressed_gib * 1024**3:
                    break
                name = Path(member.name).name
                if (member.isfile() and name in frames and name not in selected
                        and families[frames[name]['family_id']] < per_family):
                    if member.size > 32 * 1024**2:
                        raise ValueError('Unexpectedly large image member')
                    if shutil.disk_usage(root).free < member.size + 3 * 1024**3:
                        raise OSError('Free-space reserve reached')
                    # Copy only bytes of selected regular members, never extractall.
                    dst = out / name
                    temporary = dst.with_suffix('.partial')
                    with tf.extractfile(member) as source, temporary.open('wb') as target:
                        shutil.copyfileobj(source, target)
                    if temporary.stat().st_size != member.size:
                        raise ValueError('Truncated tar member')
                    temporary.rename(dst)
                    selected[name] = frames[name]
                    families[frames[name]['family_id']] += 1
                if time.monotonic() - tick > 20:
                    print(f'3D RGB scanned={stats["members_scanned"]} selected={len(selected)} families={len(families)} compressed_MiB={response.raw.tell()/1024**2:.0f}', flush=True)
                    tick = time.monotonic()
        stats['compressed_bytes_read'] = response.raw.tell()
    stats.update(selected_frames=len(selected), video_families=len(families))
    write_json(root / 'candidate_3d/selected_frames.json', selected)
    write_json(root / 'candidate_3d/stream_download.json', stats)
    print(json.dumps(stats), flush=True)


def extract_depth(root):
    selected = json.loads((root / 'candidate_3d/selected_frames.json').read_text())
    names = {x['depth'] for x in selected.values()}
    out = root / 'media/3D/depth'
    out.mkdir(parents=True, exist_ok=True)
    found = set()
    with tarfile.open(root / 'source/3D/depth/depth.tar.gz', 'r|gz') as tf:
        for member in tf:
            name = Path(member.name).name
            if member.isfile() and name in names:
                if member.size > 32 * 1024**2:
                    raise ValueError('Unexpectedly large depth member')
                with tf.extractfile(member) as src, (out / name).open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                found.add(name)
    write_json(root / 'candidate_3d/depth_extract.json', {'requested': len(names), 'found': len(found), 'missing': sorted(names - found)})
    print(f'Depth: {len(found)}/{len(names)}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['scan-3d', 'stream-images', 'extract-depth'])
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--max-frames', type=int, default=2500)
    p.add_argument('--per-family', type=int, default=8)
    p.add_argument('--max-compressed-gib', type=float, default=8)
    a = p.parse_args()
    if a.action == 'scan-3d':
        scan_3d(a.root)
    elif a.action == 'stream-images':
        stream_images(a.root, a.max_frames, a.per_family, a.max_compressed_gib)
    elif a.action == 'extract-depth':
        extract_depth(a.root)


if __name__ == '__main__':
    main()
