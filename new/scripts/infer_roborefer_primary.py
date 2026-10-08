#!/usr/bin/env python3
"""Live RoboRefer primary point; frozen Tasks60 diagnostics, new calibration pending."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(ROOT/'new/src'))


def main():
    import torch
    from safetensors.torch import load_file
    from pcrau.task_inference import TaskInference
    from pcrau.roborefer_primary import compose_prediction
    from pcrau.utils import sha256_file, seed_everything, atomic_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rgb', type=Path, required=True)
    p.add_argument('--depth', type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bundle', type=Path, default=ROOT/'new/outputs/pcrau_task_uncertainty_20261007/evaluation_r3/bundle_release.json')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = json.loads(a.bundle.read_text())
    def checked(desc, base):
        path = Path(desc['path'])
        if not path.is_absolute():
            path = base/path
        if sha256_file(path) != desc['sha256']:
            raise ValueError('Bundle input hash mismatch: '+str(path))
        return path
    baseline_path = checked(manifest['baseline_bundle'], a.bundle.resolve().parent)
    baseline = json.loads(baseline_path.read_text())
    lock = json.loads(checked(baseline['freeze_lock'], baseline_path.parent).read_text())
    env = dict(os.environ, PYTHONNOUSERSITE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    worker = ROOT/'new/scripts/roborefer_primary_worker.py'
    with tempfile.TemporaryDirectory(prefix='pcrau_primary_') as directory:
        command = [str(ROOT/'.conda-roborefer/bin/python3.10'), '-s', '-B', str(worker),
                   '--rgb', str(a.rgb.resolve()), '--depth', str(a.depth.resolve()),
                   '--prompt', a.prompt, '--inventory', lock['backbone_inventory_sha256'],
                   '--output-dir', directory]
        subprocess.run(command, env=env, check=True)
        provenance = json.loads((Path(directory)/'backbone.json').read_text())
        feature_path = Path(directory)/'features.safetensors'
        if sha256_file(feature_path) != provenance['feature_sha256']:
            raise ValueError('Worker feature checksum mismatch')
        features = load_file(str(feature_path))
    torch.set_num_threads(4)
    seed_everything(24082026)
    auxiliary = TaskInference.from_bundle(a.bundle)
    rows, _ = auxiliary.predict({k: v.float()[None] for k, v in features.items()}, [a.prompt], 24082026)
    result = compose_prediction(a.prompt, provenance['answer'], sidecar_observation=rows[0])
    result['runtime'] = {'mode': 'LIVE_RGBD_AND_GENERATION', 'backbone': provenance,
                         'auxiliary_bundle_sha256': sha256_file(a.bundle),
                         'primary_module_sha256': sha256_file(ROOT/'new/src/pcrau/roborefer_primary.py'),
                         'worker_sha256': sha256_file(worker), 'train_steps': 0, 'fit_calls': 0}
    atomic_json(a.output, result)
    print(json.dumps({'target': result['target']['pixel_xy'], 'decision': result['decision'],
                      'output': str(a.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
