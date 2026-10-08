"""Verify a cloned code snapshot and restored payloads without loading models."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hash-critical', action='store_true', help='also read/check model weights')
    parser.add_argument('--code-only', action='store_true', help='check GitHub snapshot before restoring Drive data')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    meta = root / 'migration'
    failures = []
    rows = json.loads((meta / 'CODE_COPY_MANIFEST.json').read_text())
    for row in rows:
        p = root / row['destination']
        if 'symlink' in row:
            if not p.is_symlink() or not p.exists():
                failures.append(f'Broken/missing code symlink: {row["destination"]}')
        elif not p.is_file():
            failures.append(f'Missing code: {row["destination"]}')
        elif sha(p) != row['sha256']:
            failures.append(f'Code hash differs: {row["destination"]}')
    if args.code_only:
        print(json.dumps({'project': str(root), 'code_files_checked': len(rows),
                          'failures': failures, 'data_checked': False,
                          'models_executed': False}, ensure_ascii=False, indent=2))
        raise SystemExit(1 if failures else 0)
    missing_payloads = []
    for rel in (meta / 'DATA_REQUIRED_PATHS.txt').read_text().splitlines():
        if rel and not (root / rel).exists():
            missing_payloads.append(rel)
    for rel in missing_payloads:
        failures.append(f'Missing Drive payload: {rel}')
    critical_count = 0
    lab = Path('/home/dhcn/ur_ws/src/myproject')
    for line in (meta / 'CRITICAL_FILES.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        raw = Path(name)
        if not raw.is_relative_to(lab):
            failures.append(f'Unmapped critical path: {name}'); continue
        p = root / raw.relative_to(lab)
        # SAM2 is retained in the historical inventory, but is not used by
        # the selected inference pipeline or required in the Drive payload.
        if raw.relative_to(lab).as_posix().startswith('sam2/checkpoints/'):
            continue
        if not p.is_file():
            failures.append(f'Missing critical file: {p.relative_to(root)}')
        elif args.hash_critical:
            critical_count += 1
            if sha(p) != expected:
                failures.append(f'Critical hash differs: {p.relative_to(root)}')
    for profile in json.loads((meta / 'SCOPED_DATASET_MANIFEST.json').read_text())['profiles']:
        manifest = root / profile['manifest']
        if manifest.is_file():
            entries = json.loads(manifest.read_text())['entries']
            if len(entries) != profile['samples'] or sha(manifest) != profile['manifest_sha256']:
                failures.append(f'Dataset manifest differs: {profile["name"]}')
        index = root / profile['feature_index']
        if index.is_file() and sha(index) != profile['feature_index_sha256']:
            failures.append(f'Feature index differs: {profile["name"]}')
    print(json.dumps({'project': str(root), 'code_files_checked': len(rows),
                      'critical_files_hashed': critical_count, 'failures': len(failures),
                      'missing_drive_roots': missing_payloads, 'errors_first_50': failures[:50],
                      'models_executed': False, 'robot_commanded': False}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
