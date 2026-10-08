#!/usr/bin/env python3
"""Fetch pinned RefSpatial files with resume, SHA256 and a free-space reserve.

Usage: python3 protocol/refspatial_fetch.py Simulator/metadata.json ...
Only downloads files explicitly named on the command line; never extracts code.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from urllib.parse import quote

import requests

REVISION = '519a6fd43aee2d0ed1366776e50f9456d212f94f'
REPO = 'https://huggingface.co/datasets/JingkunAn/RefSpatial'
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'datasets/refspatial_supplement_v1_20260907'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def fetch(root, entry, reserve_gib=3):
    relative = Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe source path')
    dst = root / 'source' / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    expected = entry['size']
    expected_sha = entry.get('lfs', {}).get('oid')
    if dst.exists():
        if dst.stat().st_size != expected or (expected_sha and digest(dst) != expected_sha):
            raise ValueError(f'Existing file has wrong size/hash: {dst}')
        print(f'VERIFIED existing {relative}', flush=True)
        return
    part = dst.with_name(dst.name + '.partial')
    for attempt in range(6):
        start = part.stat().st_size if part.exists() else 0
        if start > expected:
            raise ValueError(f'Partial too large: {part}')
        if start == expected:
            break
        if shutil.disk_usage(root).free < expected - start + reserve_gib * 1024**3:
            raise OSError('Insufficient space with configured reserve')
        url = f'{REPO}/resolve/{REVISION}/{quote(relative.as_posix())}?download=true&resume_offset={start}'
        try:
            with requests.get(url, headers={'Range': f'bytes={start}-'}, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                if r.status_code == 206:
                    content_range = r.headers.get('Content-Range', '')
                    if not content_range.startswith(f'bytes {start}-') or not content_range.endswith(f'/{expected}'):
                        raise ValueError(f'Unexpected Content-Range: {content_range}')
                elif start:
                    raise ValueError('Server ignored resume range; refusing to append')
                n, tick = start, time.monotonic()
                with part.open('ab') as f:
                    for b in r.iter_content(4 * 1024**2):
                        if n + len(b) > expected:
                            raise ValueError('Response exceeds source size')
                        if shutil.disk_usage(root).free < reserve_gib * 1024**3 + len(b):
                            raise OSError('Stopping at free-space reserve')
                        f.write(b)
                        n += len(b)
                        if time.monotonic() - tick > 20:
                            print(f'{relative}: {n / 1024**2:.0f}/{expected / 1024**2:.0f} MiB', flush=True)
                            tick = time.monotonic()
            if part.stat().st_size == expected:
                break
        except requests.RequestException as e:
            print(f'Retry {attempt + 1}: {type(e).__name__}', flush=True)
            time.sleep(min(2**attempt, 15))
    if not part.exists() or part.stat().st_size != expected:
        raise RuntimeError(f'Incomplete download: {relative}')
    sha = digest(part)
    if expected_sha and sha != expected_sha:
        raise ValueError(f'SHA256 mismatch: {relative}; partial retained for inspection')
    part.rename(dst)
    record = {'repo': REPO, 'revision': REVISION, 'path': relative.as_posix(),
              'bytes': expected, 'sha256': sha, 'official_lfs_sha256': expected_sha,
              'sha256_verified_against_lfs': bool(expected_sha)}
    dst.with_name(dst.name + '.provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'COMPLETE {relative} {sha}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', nargs='+')
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--reserve-gib', type=float, default=3)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    catalog_path = a.root / 'official_tree.json'
    if not catalog_path.exists():
        r = requests.get(f'https://huggingface.co/api/datasets/JingkunAn/RefSpatial/tree/{REVISION}',
                         params={'recursive': 'true', 'expand': 'false', 'limit': 1000}, timeout=60)
        r.raise_for_status()
        catalog_path.write_text(json.dumps(r.json(), indent=2) + '\n')
    entries = {e['path']: e for e in json.loads(catalog_path.read_text()) if e['type'] == 'file'}
    for name in a.files:
        fetch(a.root, entries[name], a.reserve_gib)


if __name__ == '__main__':
    main()
