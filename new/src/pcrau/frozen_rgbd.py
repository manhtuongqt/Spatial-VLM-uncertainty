"""Lazy bridge to the existing frozen RoboRefer visual extractor; no generation."""
from __future__ import annotations

import sys
import time
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import torch

from .utils import WORKSPACE_ROOT, read_json, sha256_file


def _extract_in_process(rgb_path: str | Path, depth_path: str | Path, expected_inventory: str):
    rgb_path, depth_path = Path(rgb_path).resolve(), Path(depth_path).resolve()
    for path in (rgb_path, depth_path):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != (480, 640) or image.dtype.name != 'uint8':
            raise ValueError('Inputs must be existing 640x480 uint8 RGB and relative-depth model images')
    # The original helper's archive ROOT precedes relocation. Insert the real
    # RoboRefer path explicitly; do not modify archived files.
    sys.path.insert(0, str(WORKSPACE_ROOT/'RoboRefer'))
    sys.path.insert(0, str(WORKSPACE_ROOT/'old/protocol'))
    from wp3_feature_hook_smoke import extract_features, load_model, model_inventory_sha256
    from pcra_u_development_common import pool_raw_feature, seed_runtime
    root = WORKSPACE_ROOT/'RoboRefer/models/RoboRefer-2B-SFT'
    inventory = model_inventory_sha256(root)
    if inventory != expected_inventory:
        raise ValueError('Frozen backbone inventory mismatch')
    seed_runtime(24082026, strict=False)
    started = time.perf_counter()
    backbone = load_model(root)
    try:
        extracted = extract_features(backbone, rgb_path, depth_path)
        r, rt = pool_raw_feature(extracted['r0'])
        d, dt = pool_raw_feature(extracted['d0'])
        features = {k: v.detach().cpu().half().float() for k, v in
                    zip(('r0', 'd0', 'r_thumb', 'd_thumb'), (r, d, rt, dt))}
    finally:
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return features, {'rgb_sha256': sha256_file(rgb_path), 'depth_sha256': sha256_file(depth_path),
                      'model_inventory_sha256': inventory, 'feature_storage_quantization': 'fp16',
                      'preprocess_latency_ms': extracted['preprocess_latency_ms'],
                      'feature_latency_ms': extracted['feature_latency_ms'],
                      'backbone_load_extract_elapsed_s': time.perf_counter()-started}


def extract_rgbd(rgb_path: str | Path, depth_path: str | Path, expected_inventory: str):
    """Isolate legacy torch/torchvision from the calibrated Sidecar interpreter.

    Sidecar torch is in user site; RoboRefer torchvision is paired with conda
    torch. No environment mutation or module swapping. Only four visual tensors
    cross the process boundary.
    """
    from safetensors.torch import load_file
    with tempfile.TemporaryDirectory(prefix='pcrau_rgbd_') as temporary:
        directory = Path(temporary)
        env = dict(os.environ)
        env['PYTHONNOUSERSITE'] = '1'
        env['PYTHONPATH'] = str(WORKSPACE_ROOT/'new/src')
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        command = [str(WORKSPACE_ROOT/'.conda-roborefer/bin/python3.10'), '-s', '-B',
                   '-m', 'pcrau.frozen_rgbd', str(Path(rgb_path).resolve()),
                   str(Path(depth_path).resolve()), expected_inventory, str(directory)]
        subprocess.run(command, env=env, check=True)
        features = load_file(str(directory/'features.safetensors'), device='cpu')
        if set(features) != {'r0', 'd0', 'r_thumb', 'd_thumb'}:
            raise ValueError('Worker feature boundary mismatch')
        provenance = read_json(directory/'provenance.json')
        provenance['backbone_process_user_site_disabled'] = True
        return features, provenance


if __name__ == '__main__':
    from safetensors.torch import save_file
    rgb, depth, inventory, directory = sys.argv[1:]
    features, provenance = _extract_in_process(rgb, depth, inventory)
    provenance['extractor_torch_version'] = torch.__version__
    save_file({k: v.half().contiguous() for k, v in features.items()}, str(Path(directory)/'features.safetensors'))
    (Path(directory)/'provenance.json').write_text(json.dumps(provenance))
