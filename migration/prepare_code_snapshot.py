"""Prepare a separate code-only snapshot. Never commits/pushes or edits source.

Large data/model/environment/history remain in the Google Drive backup.
Every omitted item is listed; this snapshot is not a full backup.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

SOURCE = Path(__file__).resolve().parents[2]
DEST = Path('/home/dhcn/ur_ws/pcrau_code_snapshot_20261008')
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', '.cache', 'node_modules',
             'build', 'install', 'log', 'logs', 'runs',
             'MIGRATION_BACKUP_20261008'}
SKIP_SUFFIXES = {'.safetensors', '.pt', '.pth', '.ckpt', '.bin', '.npy', '.npz',
                 '.bag', '.db3', '.mp4', '.avi', '.mkv', '.zip', '.7z', '.gz',
                 '.tar', '.aux', '.toc', '.out', '.log', '.fls', '.fdb_latexmk',
                 '.pyc', '.pyo', '.pem', '.key', '.p12', '.credentials'}
TREES = ['new/src', 'new/scripts', 'new/configs', 'new/docs', 'new/tests',
         'new/demo_gazebo', 'new/hinhanh', 'new/test_iid/scripts',
         'new/test_iid/contracts', 'new/test_iid/protocol',
         'protocol', 'old/protocol', 'ur3', 'RoboRefer',
         'sam2', 'workspace', 'plan', 'latex', 'baocao', 'baocao1']
LIMIT = 20 * 1024 * 1024


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    if DEST.exists():
        raise SystemExit(f'Refusing to overwrite existing snapshot: {DEST}')
    DEST.mkdir()
    copied, omitted = [], []

    def copy_file(p, rel=None):
        rel = rel or p.relative_to(SOURCE)
        d = DEST / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink():
            target = p.resolve()
            if not target.is_relative_to(SOURCE):
                omitted.append({'path': str(p.relative_to(SOURCE)), 'reason': 'external symlink'})
                return
            d.symlink_to(os.readlink(p))
            copied.append({'source': str(p.relative_to(SOURCE)), 'destination': str(rel),
                           'symlink': os.readlink(p)})
        else:
            shutil.copy2(p, d)
            source_sha = sha(p)
            if sha(d) != source_sha:
                raise RuntimeError(f'Copy differs: {p}')
            copied.append({'source': str(p.relative_to(SOURCE)), 'destination': str(rel),
                           'sha256': source_sha, 'size_bytes': d.stat().st_size})

    for tree in TREES:
        root = SOURCE / tree
        if not root.exists():
            omitted.append({'path': tree, 'reason': 'not present'}); continue
        for base, ds, fs in os.walk(root, followlinks=False):
            ds.sort(); fs.sort()
            keep = []
            for n in ds:
                q = Path(base) / n
                relative = str(q.relative_to(SOURCE))
                if n in SKIP_DIRS or relative.startswith('sam2/notebooks/videos') or relative in {'RoboRefer/models', 'sam2/checkpoints'}:
                    omitted.append({'path': str(q.relative_to(SOURCE)), 'reason': 'data/cache/history/generated directory'})
                elif q.is_symlink():
                    omitted.append({'path': str(q.relative_to(SOURCE)), 'reason': 'directory symlink; preserve in full backup'})
                else:
                    keep.append(n)
            ds[:] = keep
            for n in fs:
                p = Path(base) / n
                reason = None
                if n == '.env' or n.startswith('.env.'):
                    reason = 'machine configuration'
                elif p.suffix.lower() in SKIP_SUFFIXES:
                    reason = 'data/model/media/build product or private key suffix'
                elif not p.is_symlink() and p.stat().st_size > LIMIT:
                    reason = 'file exceeds 20 MiB code snapshot limit'
                if reason:
                    omitted.append({'path': str(p.relative_to(SOURCE)), 'reason': reason})
                else:
                    copy_file(p)
    for n in ['DEPENDENCIES.md', 'README.md']:
        copy_file(SOURCE / n, Path('README_HISTORICAL_ROOT.md') if n == 'README.md' else None)
    migration = SOURCE / 'plan/MIGRATION_BACKUP_20261008'
    for n in ['README.md', 'BACKUP_ROOTS.txt', 'conda_environment.yml', 'conda_explicit.txt',
              'pip_conda_clean.txt', 'pip_user_python310.txt', 'dpkg_installed.tsv',
              'gpu.txt', 'gazebo_versions.txt', 'os_release.txt', 'shell_dependency_lines.txt',
              'CRITICAL_FILES.sha256', 'capture_inventory.py', 'prepare_code_snapshot.py',
              'GITHUB_DRIVE_GUIDE.md', 'SCOPED_DATASET_MANIFEST.json',
              'DATA_REQUIRED_PATHS.txt', 'LAPTOP_REPRODUCTION.md', 'verify_restore.py']:
        if (migration / n).exists():
            copy_file(migration / n, Path('migration') / n)
    meta = DEST / 'migration'; meta.mkdir(exist_ok=True)
    # Keep the current runtime identity, selected neural metadata/calibration and
    # lock files, without shipping tensor payloads or large prediction JSONL.
    selected = ['pcrau_target_v2_full_seed_24082026',
                'pcrau_answerability_language_dev_20261003',
                'pcrau_s1_anchor_peak_pilot_v2_20261005',
                'pcrau_unified_spatial_20261005_r2',
                'pcrau_task_uncertainty_20261007']
    copy_file(SOURCE / 'new/outputs/active_experimental_profile.json',
              Path('reproducibility/locked/new/outputs/active_experimental_profile.json'))
    for run in selected:
        for p in sorted((SOURCE / 'new/outputs' / run).rglob('*')):
            if p.is_file() and p.suffix in {'.json', '.md', '.yaml', '.yml'} and p.stat().st_size <= 2 * 1024 * 1024:
                copy_file(p, Path('reproducibility/locked') / p.relative_to(SOURCE))
    # Standard description packages are also snapshotted: exported Gazebo worlds
    # refer to meshes here by absolute /opt paths. These are assets, not ROS binaries.
    for name in ['ur_description', 'realsense2_description']:
        base = Path('/opt/ros/humble/share') / name
        for p in sorted(base.rglob('*')):
            if p.is_file() and not p.is_symlink():
                relative = Path('robot_assets/system_description') / name / p.relative_to(base)
                d = DEST / relative
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d)
                h = sha(p)
                if sha(d) != h: raise RuntimeError(f'Asset copy differs: {p}')
                copied.append({'source_external': str(p), 'destination': str(relative),
                               'sha256': h, 'size_bytes': d.stat().st_size})
    (meta / 'CODE_COPY_MANIFEST.json').write_text(json.dumps(copied, ensure_ascii=False, indent=2)+'\n')
    (meta / 'OMITTED_ITEMS.json').write_text(json.dumps(omitted, ensure_ascii=False, indent=2)+'\n')
    repos = []
    for rel in ['.', 'RoboRefer', 'sam2', 'ur3']:
        p = SOURCE / rel
        r = subprocess.run(['git', '-C', str(p), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        repos.append({'source': rel, 'head': r.stdout.strip(), 'local_changes_included': True,
                      'git_history_included': False})
    (meta / 'SOURCE_REPOSITORIES.json').write_text(json.dumps(repos, indent=2)+'\n')
    (DEST / '.gitignore').write_text('''__pycache__/
*.py[cod]
.pytest_cache/
.conda-roborefer/
.venv/
build/
install/
log/
new/outputs/
new/test_iid/dataset/
new/test_iid/feature_cache/
new/demo_gazebo/runs/
datasets/
RoboRefer/models/
sam2/checkpoints/
*.safetensors
*.pt
*.pth
*.ckpt
*.npy
*.npz
*.bag
*.db3
.env
.env.*
*.pem
*.key
*.p12
''')
    (DEST / '.gitattributes').write_text('# Preserve source bytes used by frozen hashes.\n* -text\n')
    (DEST / 'README.md').write_text('''# P-CRA-U — snapshot code ngày 08/10/2026

Snapshot code riêng của đồ án: RGB-D spatial referring, phrase-conditioned
anchor, verifier trái/phải và các nhánh uncertainty/calibration.

Đường RoboRefer primary là biến thể opt-in. Registry baseline Adapter và các
bundle live44/tasks60 giữ riêng trong backup data; xem `new/docs/` và `plan/`
để phân biệt scope/kết quả của từng bundle.

## Khôi phục

Repository này chỉ chứa code, configs, tài liệu và assets robot nhỏ. **Clone
riêng repo chưa đủ để inference/train.** Khôi phục data/model/artifacts từ
Google Drive vào đúng relative paths trước; xem `migration/README.md` và
`migration/GITHUB_DRIVE_GUIDE.md`. Drive folder link sẽ bổ sung sau khi upload
được kiểm chứng; hiện chưa có upload hoặc policy promotion.

Giữ các dataset/cache được chỉ rõ ở `migration/SCOPED_DATASET_MANIFEST.json`:
1.600 train, 400 dev, 1.000 calibration và 1.000 Test-IID; không cần copy mọi
dataset lịch sử chỉ để chạy model hiện tại. Các paths này còn nằm dưới `old/`.
`RoboRefer/`, `sam2/`, `ur3/` chứa snapshot source local, giữ licenses, gồm các
sửa đổi và file source untracked đã copy. Nested Git history không nằm trong
snapshot này; source trên máy lab giữ nguyên. Licenses gốc được giữ trong cây.

URDF/Xacro, YCB/gripper meshes, textures/worlds nằm trong `ur3/`; camera/UR
description assets bên hệ thống được lưu tại `robot_assets/system_description/`.
Metadata/calibration khóa nằm ở `reproducibility/locked/`, để phục hồi vào
paths gốc cùng neural weights từ Drive. Binaries ROS và Python env không
đưa lên GitHub; exports/version inventory nằm trong `migration/`.

Exports môi trường nằm ở `migration/`; backbone dùng Conda sạch, Sidecar còn
dùng user packages Python3.10. Các paths máy lab phải được xử lý có provenance
khi chuyển máy, không sửa hàng loạt bundles khóa.

Danh sách file đã copy và SHA256 nằm ở `migration/CODE_COPY_MANIFEST.json`.
Items bỏ khỏi snapshot nằm ở `migration/OMITTED_ITEMS.json` và vẫn cần giữ
trong full backup. Đây không phải bản thay thế toàn workspace.
''')
    missing = []
    for p in DEST.rglob('*'):
        if p.is_symlink() and not p.exists():
            missing.append(str(p.relative_to(DEST)))
    summary = {'path': str(DEST), 'copied_files': len(copied), 'omitted_items': len(omitted),
               'copied_bytes': sum(x.get('size_bytes', 0) for x in copied),
               'broken_snapshot_symlinks': missing, 'committed': False, 'pushed': False}
    (meta / 'PREPARATION_SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
