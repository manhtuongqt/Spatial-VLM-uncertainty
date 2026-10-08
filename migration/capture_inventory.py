"""Read-only migration inventory; writes only next to this script.

No model forward, training, ROS launch, copy, deletion or network request.
File inventory is metadata, not a checksum verification of a completed backup.
"""
from __future__ import annotations

import collections
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
HOME = Path('/home/dhcn')
ENV = PROJECT / '.conda-roborefer'


def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def command(name, args, env=None, timeout=60):
    try:
        r = subprocess.run(args, text=True, capture_output=True, env=env, timeout=timeout)
        (OUT / name).write_text(r.stdout)
        if r.stderr:
            (OUT / (name + '.stderr.txt')).write_text(r.stderr)
        return {'file': name, 'returncode': r.returncode}
    except (OSError, subprocess.TimeoutExpired) as e:
        (OUT / (name + '.stderr.txt')).write_text(str(e))
        return {'file': name, 'error': str(e)}


def disk_bytes(path):
    r = subprocess.run(['du', '-s', '-B1', str(path)], capture_output=True, text=True)
    return int(r.stdout.split()[0]) if r.stdout else None


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    roots = [HOME / 'ur_ws', HOME / 'workspace/ros_ur_driver',
             HOME / '.local/lib/python3.10', HOME / 'Downloads',
             HOME / '.ignition', HOME / '.gazebo', HOME / '.rviz2', HOME / '.sdformat',
             HOME / '.codex/attachments', Path('/opt/ros/humble/share/ur_description'),
             Path('/opt/ros/humble/share/realsense2_description')]
    roots = [p for p in roots if p.exists()]
    links, errors, counts = [], [], collections.Counter()
    with gzip.open(OUT / 'FILE_INVENTORY.tsv.gz', 'wt', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['absolute_path', 'type', 'size_bytes', 'mtime_ns', 'mode_octal',
                         'device', 'inode', 'hardlink_count', 'symlink_target'])
        for root in roots:
            for base, ds, fs in os.walk(root, followlinks=False,
                                        onerror=lambda e: errors.append(str(e))):
                ds.sort(); fs.sort()
                ds[:] = [n for n in ds if (Path(base) / n) != OUT]
                for name in ds + fs:
                    p = Path(base) / name
                    try:
                        st = p.lstat()
                        is_link = p.is_symlink()
                        kind = 'symlink' if is_link else 'directory' if p.is_dir() else 'file'
                        target = os.readlink(p) if is_link else ''
                        writer.writerow([str(p), kind, st.st_size, st.st_mtime_ns,
                                         oct(st.st_mode & 0o7777), st.st_dev, st.st_ino,
                                         st.st_nlink, target])
                        counts[kind] += 1
                        if is_link:
                            resolved = p.resolve()
                            links.append({'path': str(p), 'link': target,
                                          'target': str(resolved), 'exists': p.exists(),
                                          'outside_project': not resolved.is_relative_to(PROJECT)})
                    except OSError as e:
                        errors.append(f'{p}: {e}')
    save_json('SYMLINK_INVENTORY.json', links)
    save_json('INVENTORY_ERRORS.json', errors)

    size_paths = [PROJECT] + sorted(p for p in PROJECT.iterdir() if p.is_dir() and p != OUT)
    size_paths += roots + [HOME / 'shawn_ws', HOME / 'miniforge3',
                          HOME / '.cache/tectonic', HOME / '.cache/Tectonic',
                          HOME / '.ros', Path('/opt/ros/humble')]
    sizes = []
    for p in dict.fromkeys(size_paths):
        if p.exists():
            n = disk_bytes(p)
            sizes.append({'path': str(p), 'allocated_bytes_du': n,
                          'GiB': round(n / 1024**3, 3) if n is not None else None})
    save_json('DIRECTORY_SIZES.json', sizes)

    repositories = [PROJECT, PROJECT / 'RoboRefer', PROJECT / 'sam2', PROJECT / 'ur3',
                    HOME / 'ur_ws/src/uet_ur3']
    repositories += [p for p in (HOME / 'workspace/ros_ur_driver/src').iterdir()
                     if (p / '.git').exists()]
    repos = []
    for i, p in enumerate(repositories):
        if not (p / '.git').exists():
            continue
        result = {'path': str(p)}
        for label, args in [('head', ['rev-parse', 'HEAD']),
                            ('status', ['status', '--porcelain=v1', '--untracked-files=all']),
                            ('diff_stat', ['diff', '--stat'])]:
            result[label] = command(f'git_{i:02d}_{label}.txt', ['git', '-C', str(p), *args])
        repos.append(result)
    save_json('GIT_REPOSITORIES.json', repos)

    active = json.loads((PROJECT / 'new/outputs/active_experimental_profile.json').read_text())
    critical = {PROJECT / 'new/outputs/active_experimental_profile.json': None}
    for k in ('checkpoint', 'config', 'calibrator', 'profiles'):
        critical[Path(active[k])] = active[k + '_sha256']
    for relative in ['new/outputs/pcrau_target_v2_full_seed_24082026/checkpoints/best/model.safetensors',
                     'new/outputs/pcrau_unified_spatial_20261005_r2/bundle.json',
                     'new/outputs/pcrau_unified_spatial_20261005_r2/frozen/anchor_mlp.safetensors',
                     'new/outputs/pcrau_unified_spatial_20261005_r2/calibrator.json',
                     'new/outputs/pcrau_unified_spatial_20261005_r2/profile.json',
                     'new/outputs/pcrau_task_uncertainty_20261007/best_task_heads.safetensors',
                     'new/outputs/pcrau_task_uncertainty_20261007/evaluation_r3/bundle_release.json',
                     'new/outputs/pcrau_task_uncertainty_20261007/evaluation_r3/calibrator.json',
                     'new/outputs/pcrau_task_uncertainty_20261007/evaluation_r3/profile.json']:
        critical[PROJECT / relative] = None
    # Preserve complete, local model payloads; these are byte hashes, not model evaluations.
    for base in [PROJECT / 'RoboRefer/models', PROJECT / 'sam2/checkpoints']:
        for p in base.rglob('*'):
            if p.is_file() and not p.is_symlink() and '.cache' not in p.parts:
                critical[p] = None
    hashes = []
    with (OUT / 'CRITICAL_FILES.sha256').open('w') as f:
        for p, expected in sorted(critical.items()):
            if not p.is_file():
                hashes.append({'path': str(p), 'exists': False}); continue
            sha = digest(p)
            f.write(f'{sha}  {p}\n')
            hashes.append({'path': str(p), 'sha256': sha, 'expected': expected,
                           'matches_expected': sha == expected if expected else None})
    save_json('CRITICAL_FILES.json', hashes)

    clean = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    exports = []
    exports.append(command('conda_explicit.txt', [str(HOME / 'miniforge3/bin/conda'),
                                                'list', '-p', str(ENV), '--explicit']))
    exports.append(command('conda_environment.yml', [str(HOME / 'miniforge3/bin/conda'),
                                                    'env', 'export', '-p', str(ENV), '--no-builds']))
    exports.append(command('pip_conda_clean.txt', [str(ENV / 'bin/python3.10'), '-B', '-m',
                                                 'pip', 'freeze'], env=clean))
    user_site = HOME / '.local/lib/python3.10/site-packages'
    user_packages = sorted((d.metadata.get('Name', 'UNKNOWN'), d.version)
                           for d in importlib.metadata.distributions(path=[str(user_site)]))
    (OUT / 'pip_user_python310.txt').write_text(''.join(f'{n}=={v}\n' for n, v in user_packages))
    exports.append(command('dpkg_installed.tsv', ['dpkg-query', '-W',
                                                 '-f=${binary:Package}\t${Version}\t${db:Status-Status}\n']))
    exports.append(command('gpu.txt', ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                      '--format=csv,noheader']))
    exports.append(command('gazebo_versions.txt', ['ign', 'gazebo', '--versions']))
    exports.append(command('storage.txt', ['lsblk', '-o', 'NAME,SIZE,FSTYPE,MOUNTPOINTS']))
    exports.append(command('os_release.txt', ['cat', '/etc/os-release']))
    config_lines = []
    for p in [HOME / '.bashrc', HOME / '.profile']:
        for line in p.read_text().splitlines():
            if re.search(r'ros|gazebo|ignition|ur_ws|shawn_ws|GAZEBO|PYTHON|LD_LIBRARY|HF_|CUDA|conda', line):
                config_lines.append(f'{p}: {line}')
    (OUT / 'shell_dependency_lines.txt').write_text('\n'.join(config_lines) + '\n')
    save_json('EXPORT_STATUS.json', exports)
    for name in ['pcrau_migration_external_references.json']:
        source = Path('/tmp') / name
        if source.exists():
            (OUT / 'EXTERNAL_TEXT_REFERENCES.json').write_bytes(source.read_bytes())

    summary = {'captured_at': datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),
               'project': str(PROJECT), 'roots': [str(p) for p in roots],
               'counts': dict(counts), 'errors': len(errors),
               'symlinks': len(links), 'broken_symlinks': sum(not x['exists'] for x in links),
               'project_symlinks': sum(Path(x['path']).is_relative_to(PROJECT) for x in links),
               'critical_files_hashed': len(hashes),
               'selected_profile_hashes_match': all(x.get('matches_expected') is not False for x in hashes),
               'backup_was_created': False, 'inventory_is_not_content_verification': True,
               'excluded_from_file_inventory': str(OUT),
               'private_account_authentication_files_not_collected': True}
    save_json('SNAPSHOT_SUMMARY.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
