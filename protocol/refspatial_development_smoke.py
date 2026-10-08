#!/usr/bin/env python3
"""One local development RGB-D inference; input manifest must exclude oracle labels."""
import argparse
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    row = json.loads(a.input.read_text())
    if set(row) != {'sample_id', 'split', 'image', 'depth', 'instruction'} or row['split'] != 'dev':
        raise ValueError('Require development-only input allowlist; no offline annotation fields')
    result = {'status': 'STARTED', 'sample_id': row['sample_id'], 'split': 'dev',
              'mode': 'RGB-D', 'depth_semantics': 'source 8-bit proxy, not metric ground truth',
              'input_manifest_sha256': hashlib.sha256(a.input.read_bytes()).hexdigest(),
              'seed': 9092026, 'model': 'RoboRefer/models/RoboRefer-2B-SFT',
              'training_performed': False, 'accuracy_evaluation_performed': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    try:
        sys.path.insert(0, str(ROOT/'RoboRefer'))
        import torch
        import llava
        from llava import conversation as clib
        from llava.media import Image, Depth
        torch.manual_seed(result['seed'])
        torch.cuda.manual_seed_all(result['seed'])
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model = llava.load(str(ROOT/result['model']))
        clib.default_conversation = clib.conv_templates['auto'].copy()
        torch.cuda.synchronize()
        result['load_seconds'] = time.perf_counter()-start
        config = copy.deepcopy(model.default_generation_config)
        config.do_sample = False
        config.temperature = config.top_p = config.top_k = None
        config.max_new_tokens = 256
        result['max_new_tokens'] = 256
        before = time.perf_counter()
        with torch.inference_mode():
            answer = model.generate_content([Image(str(ROOT/row['image'])), Depth(str(ROOT/row['depth'])), row['instruction']], generation_config=config)
        torch.cuda.synchronize()
        result.update(status='INFERENCE_COMPLETED', latency_seconds=time.perf_counter()-before, answer=answer,
                      cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                      gpu_name=torch.cuda.get_device_name())
        from refspatial_wp1_rules import point
        parsed = point(answer)
        result['normalized_point_xy'] = parsed
        result['point_format_pass'] = parsed is not None
    except Exception:
        result.update(status='FAILED', error=traceback.format_exc())
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'INFERENCE_COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
