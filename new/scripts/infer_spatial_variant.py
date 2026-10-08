#!/usr/bin/env python3
"""One observation → live evidence → verifier → calibrated perception decision."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from pcrau.frozen_rgbd import extract_rgbd
from pcrau.unified_inference import UnifiedInference, load_features
from pcrau.utils import atomic_json, read_json, seed_everything, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--features', type=Path)
    inputs.add_argument('--rgb', type=Path)
    parser.add_argument('--depth', type=Path, help='640x480 uint8 relative-depth model input')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.rgb is None) != (args.depth is None):
        parser.error('--rgb and --depth must be used together')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4); seed_everything(24082026)
    if args.features:
        features = load_features(args.features)
        provenance = {'feature_file': str(args.features.resolve()), 'sha256': sha256_file(args.features)}
    else:
        manifest = read_json(args.bundle.resolve())
        lock_desc = manifest['freeze_lock']; lock_path = Path(lock_desc['path'])
        if not lock_path.is_absolute(): lock_path = args.bundle.resolve().parent/lock_path
        if sha256_file(lock_path) != lock_desc['sha256']: raise ValueError('Freeze lock hash mismatch')
        lock = read_json(lock_path)
        features, provenance = extract_rgbd(args.rgb, args.depth, lock['backbone_inventory_sha256'])
    pipeline = UnifiedInference.from_bundle(args.bundle)
    started = time.perf_counter()
    result = pipeline.predict({k: v[None] for k, v in features.items()}, [args.prompt])[0]
    torch.cuda.synchronize()
    result['runtime'] = {'elapsed_ms': (time.perf_counter()-started)*1000,
                         'visual_input': provenance, 'bundle_sha256': sha256_file(args.bundle)}
    atomic_json(args.output, result)
    print(f"{result['decision']['action']} risk={result['decision']['risk']:.6f} → {args.output}")


if __name__ == '__main__': main()
