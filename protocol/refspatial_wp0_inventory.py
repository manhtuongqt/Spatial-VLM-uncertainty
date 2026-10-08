#!/usr/bin/env python3
"""Inventory local runtime and verify existing weights/cache against manifests.

Run with .conda-roborefer/bin/python -s protocol/refspatial_wp0_inventory.py
Does not run training, load old predictions or read final test examples.
"""
import collections
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/spatial_vlm_refspatial_v1/wp0_inventory'
LEGACY_ROOT = ROOT / 'old'
LEGACY_RESULTS = LEGACY_ROOT / 'results'
LEGACY_PROTOCOL = LEGACY_ROOT / 'protocol'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def command(args, cwd=ROOT):
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=60)
        return {'argv': args, 'exit_code': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'argv': args, 'error': str(e)}


def tensor_header(p):
    try:
        from safetensors import safe_open
        with safe_open(str(p), framework='pt', device='cpu') as f:
            return {'status': 'READABLE_HEADER', 'tensor_count': len(f.keys()),
                    'tensor_shapes': {k: f.get_slice(k).get_shape() for k in f.keys()}}
    except Exception as e:
        return {'status': 'FAILED', 'error': f'{type(e).__name__}: {e}'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git = {}
    for name in ['.', 'RoboRefer', 'sam2', 'ur3']:
        cwd = ROOT / name
        git[name] = {'commit': command(['git', 'rev-parse', 'HEAD'], cwd),
                     'status': command(['git', 'status', '--porcelain=v1'], cwd),
                     'diff_stat': command(['git', 'diff', '--stat'], cwd)}
    write('git_baseline.json', git)
    versions = {}
    for name in ['torch', 'torchvision', 'transformers', 'safetensors', 'numpy', 'Pillow',
                 'opencv-python', 'opencv-python-headless', 'accelerate', 'peft', 'flash-attn']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'NOT_INSTALLED_IN_THIS_INTERPRETER'
    runtime = {'timestamp_utc': start, 'python': sys.version, 'executable': sys.executable,
               'platform': platform.platform(), 'packages': versions,
               'disk': dict(zip(['total', 'used', 'free'], shutil.disk_usage(ROOT))),
               'memory': command(['free', '-b']),
               'gpu': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv']),
               'ros_installations': [str(p) for p in Path('/opt/ros').glob('*')],
               'gazebo': command(['gazebo', '--version']),
               'ros_humble_import': command(['bash', '-c', 'source /opt/ros/humble/setup.bash && /usr/bin/python3 -c "import rclpy; print(rclpy.__file__)"']),
               'model_inference_run': False, 'training_run': False}
    try:
        import torch
        runtime['torch_runtime'] = {'version': torch.__version__, 'cuda_build': torch.version.cuda,
                                    'cuda_available': torch.cuda.is_available()}
        if torch.cuda.is_available():
            torch.manual_seed(20260908)
            a = torch.randn(64, 64, device='cuda', requires_grad=True)
            loss = (a @ a.T).square().mean()
            loss.backward()
            torch.cuda.synchronize()
            runtime['cuda_forward_backward_smoke'] = {'passed': bool(torch.isfinite(loss) and torch.isfinite(a.grad).all()),
                'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(), 'scope': 'SMALL_TENSOR_ONLY_NOT_ROBOREFER_TRAINING'}
            del a, loss
            torch.cuda.empty_cache()
    except Exception as e:
        runtime['torch_error'] = f'{type(e).__name__}: {e}'
    write('environment.json', runtime)
    weights = []
    for p in sorted((ROOT / 'RoboRefer/models').rglob('*')):
        if p.is_file() and p.suffix in {'.safetensors', '.pth', '.json'}:
            print('Weight/config:', p.relative_to(ROOT), flush=True)
            row = {'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
            if p.suffix == '.safetensors':
                row['header'] = tensor_header(p)
            elif p.suffix == '.pth':
                try:
                    data = torch.load(p, map_location='cpu', weights_only=True)
                    row['safe_load'] = {'status': 'PASS', 'top_level_items': len(data)}
                    del data
                except Exception as e:
                    row['safe_load'] = {'status': 'FAILED', 'error': str(e)}
            weights.append(row)
    required = ['RoboRefer/models/RoboRefer-2B-SFT/' + x + '/model.safetensors'
                for x in ['llm', 'vision_tower', 'depth_tower', 'mm_projector', 'depth_projector']]
    required += ['RoboRefer/models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth']
    write('model_weight_inventory.json', {'files': weights, 'required_files': {p: (ROOT / p).is_file() for p in required},
        'verification_scope': 'Local SHA256 + safetensors headers + weights_only depth load; no official source checksum or full RoboRefer inference claimed.'})
    checkpoints = []
    for manifest in sorted((LEGACY_RESULTS / 'pcra_u_runs').glob('*/checkpoints/**/manifest.json')):
        data = json.loads(manifest.read_text())
        checks = []
        for item in data.get('files', []):
            p = manifest.parent / item['path']
            row = {'path': str(p.relative_to(ROOT)), 'exists': p.is_file()}
            if p.is_file():
                row['sha256'] = sha(p)
                row['matches_manifest'] = row['sha256'] == item['sha256'] and p.stat().st_size == item['bytes']
                if p.suffix == '.safetensors':
                    row['header_status'] = tensor_header(p)['status']
            checks.append(row)
        checkpoints.append({'manifest': str(manifest.relative_to(ROOT)), 'checks': checks,
                             'pass': bool(checks) and all(r.get('matches_manifest') and r.get('header_status', 'READABLE_HEADER') == 'READABLE_HEADER' for r in checks)})
    write('checkpoint_integrity.json', checkpoints)
    cache_reports = []
    for index in sorted((LEGACY_RESULTS / 'pcra_u_feature_cache').glob('*/indexes/index_full.json')):
        data = json.loads(index.read_text())
        failures, total = [], 0
        print('Cache:', index.relative_to(ROOT), flush=True)
        for key, item in data['features'].items():
            p = index.parent.parent / item['path']
            total += 1
            ok = p.is_file() and p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
            if ok:
                header = tensor_header(p)
                ok = header['status'] == 'READABLE_HEADER' and header['tensor_shapes'] == item['tensor_shapes']
            if not ok:
                failures.append({'key': key, 'path': str(p.relative_to(ROOT))})
        cache_reports.append({'index': str(index.relative_to(ROOT)), 'index_sha256': sha(index),
            'checked_features': total, 'failures': failures, 'all_features_hash_shape_pass': not failures,
            'reuse': 'OLD_MODEL_PREPROCESSING_ONLY; SOURCE_DATASET_AND_NEW_TASK_COMPATIBILITY_NOT_CERTIFIED'})
    write('feature_cache_integrity.json', cache_reports)
    tracked = command(['git', 'ls-files', 'datasets'])['stdout'].splitlines()
    missing = [p for p in tracked if not (ROOT / p).exists()]
    lock_path = LEGACY_PROTOCOL / 'pcra_u_development_v1_1_decision_lock.json'
    lock = json.loads(lock_path.read_text()) if lock_path.is_file() else {}
    write('artifact_reuse_matrix.json', {'tracked_dataset_artifacts': len(tracked), 'missing_tracked_dataset_artifacts': missing,
        'current_dataset_directories': [p.name for p in (ROOT / 'datasets').iterdir() if p.is_dir()],
        'retained_legacy_model': lock.get('retained_model'),
        'legacy_checkpoints_verified': sum(x['pass'] for x in checkpoints), 'legacy_checkpoint_manifests': len(checkpoints),
        'cache_integrity': cache_reports,
        'legacy_raw_dataset_status': 'MISSING_IN_DATASETS_ROOT_REQUIRES_RESTORE_OR_RECAPTURE',
        'new_grounding_or_risk_compatibility': 'NOT_CERTIFIED', 'final_test_predictions_read': False})
    sizes = command(['du', '-sb', 'datasets', 'RoboRefer/models', 'old/results/pcra_u_feature_cache', 'old/results/pcra_u_runs', '.conda-roborefer'])
    write('storage_inventory.json', sizes)
    write('workspace_inventory.json', {'status': 'INVENTORY_COMPLETE', 'timestamp_utc': start,
        'runtime': runtime, 'model_weight_files': len(weights), 'checkpoint_manifests': len(checkpoints),
        'checkpoint_pass': sum(c['pass'] for c in checkpoints), 'feature_caches': cache_reports,
        'disk_reserve_note': 'Only small manifests/reports generated; no dataset copies or cleanup performed.',
        'inference_readiness': 'NOT_YET_TESTED_END_TO_END', 'script_sha256': sha(Path(__file__))})
    print('WP0 inventory complete', flush=True)


if __name__ == '__main__':
    main()
