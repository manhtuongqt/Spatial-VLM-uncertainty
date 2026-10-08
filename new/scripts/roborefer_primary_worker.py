#!/usr/bin/env python3
"""Clean legacy interpreter: same RGB-D observation -> features + generated point."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path[:0] = [str(ROOT/'RoboRefer'), str(ROOT/'old/protocol'), str(ROOT/'new/src')]


def main():
    import cv2
    import torch
    from safetensors.torch import save_file
    from wp3_feature_hook_smoke import load_model, extract_features, generate_answer, model_inventory_sha256
    from pcra_u_development_common import pool_raw_feature, seed_runtime
    from pcrau.roborefer_primary import generation_prompt
    from pcrau.utils import sha256_file
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rgb', type=Path, required=True)
    p.add_argument('--depth', type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--inventory', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    for path in (a.rgb, a.depth):
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None or im.shape != (480, 640, 3) or im.dtype.name != 'uint8':
            raise ValueError('Expected 640x480 uint8 three-channel RGB/relative-depth inputs')
    model_root = ROOT/'RoboRefer/models/RoboRefer-2B-SFT'
    inventory = model_inventory_sha256(model_root)
    if inventory != a.inventory:
        raise ValueError('Backbone inventory mismatch')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if (a.output_dir/'backbone.json').exists():
        raise FileExistsError(a.output_dir/'backbone.json')
    seed_runtime(24082026, strict=False)
    model = load_model(model_root)
    raw = extract_features(model, a.rgb, a.depth)
    r, rt = pool_raw_feature(raw['r0'])
    d, dt = pool_raw_feature(raw['d0'])
    save_file({k: v.detach().cpu().half().contiguous() for k, v in
               zip(('r0', 'd0', 'r_thumb', 'd_thumb'), (r, d, rt, dt))},
              str(a.output_dir/'features.safetensors'))
    del raw, r, d, rt, dt
    seed_runtime(24082026, strict=False)
    answer, ms = generate_answer(model, a.rgb, a.depth, generation_prompt(a.prompt), {'max_new_tokens': 128})
    value = {'answer': answer, 'prompt': a.prompt, 'generation_prompt': generation_prompt(a.prompt),
             'generation_ms': ms, 'backbone_inventory_sha256': inventory,
             'rgb_sha256': sha256_file(a.rgb), 'depth_sha256': sha256_file(a.depth),
             'feature_sha256': sha256_file(a.output_dir/'features.safetensors'),
             'feature_storage_quantization': 'fp16', 'torch': torch.__version__,
             'GT_input': False, 'train_steps': 0}
    (a.output_dir/'backbone.json').write_text(json.dumps(value, indent=2)+'\n')


if __name__ == '__main__':
    main()
