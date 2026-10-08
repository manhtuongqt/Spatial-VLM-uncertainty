#!/usr/bin/env python3
"""Run a blinded RoboRefer agreement triage over a RefSpatial audit queue.

This is an AI-assistance artifact, not a label audit: RoboRefer receives only an
RGB image, its paired depth image, and the question.  The source point is read
only after generation in order to measure coordinate agreement.  A prediction
near or far from the source point does not prove that the source label is right
or wrong, since multiple points on a selected object may be valid.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / 'results/spatial_vlm_refspatial_v1/wp2_visual_review/next_audit_queue.jsonl'
DEFAULT_OUT = ROOT / 'results/spatial_vlm_refspatial_v1/wp2_visual_review/ai_triage_roborefer_2b_sft'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(distance, threshold, uncertain_threshold):
    if distance is None:
        return 'parse_failed'
    if distance <= threshold:
        return 'high_coordinate_agreement'
    if distance <= uncertain_threshold:
        return 'moderate_coordinate_agreement'
    return 'low_coordinate_agreement'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, default=DEFAULT_QUEUE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Use only for a resumable development slice; omitted means the full queue.')
    parser.add_argument('--agreement-threshold', type=float, default=.08)
    parser.add_argument('--moderate-threshold', type=float, default=.20)
    args = parser.parse_args()
    if not 0 < args.agreement_threshold < args.moderate_threshold:
        raise ValueError('Require 0 < agreement-threshold < moderate-threshold')

    queue = [json.loads(line) for line in args.queue.read_text().splitlines() if line]
    if args.max_samples is not None:
        queue = queue[:args.max_samples]
    if not queue:
        raise ValueError('Queue is empty')
    required = {'sample_id', 'split', 'image', 'depth', 'instruction', 'target_xy', 'audit_strata'}
    if any(required - row.keys() for row in queue):
        raise ValueError('Queue is missing fields needed for blinded triage')

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    run = {
        'status': 'STARTED',
        'kind': 'AI_MODEL_COORDINATE_AGREEMENT_TRIAGE',
        'review_origin': 'ai_model_agreement',
        'model': 'RoboRefer/models/RoboRefer-2B-SFT',
        'mode': 'RGB-D',
        'queue': str(args.queue.relative_to(ROOT)),
        'queue_sha256': sha256(args.queue),
        'queue_count': len(queue),
        'agreement_threshold_normalized_l2': args.agreement_threshold,
        'moderate_threshold_normalized_l2': args.moderate_threshold,
        'source_target_hidden_from_model': True,
        'training_performed': False,
        'label_validity_evaluation_performed': False,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    (out / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    predictions = []
    try:
        sys.path.insert(0, str(ROOT / 'RoboRefer'))
        import torch
        import llava
        from llava import conversation as clib
        from llava.media import Depth, Image
        from refspatial_wp1_rules import point

        seed = 9092026
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = llava.load(str(ROOT / run['model']))
        clib.default_conversation = clib.conv_templates['auto'].copy()
        torch.cuda.synchronize()
        run['load_seconds'] = time.perf_counter() - started
        run['seed'] = seed
        run['gpu_name'] = torch.cuda.get_device_name()
        config = copy.deepcopy(model.default_generation_config)
        config.do_sample = False
        config.temperature = config.top_p = config.top_k = None
        config.max_new_tokens = 256

        for index, row in enumerate(queue, 1):
            start = time.perf_counter()
            record = {
                'sample_id': row['sample_id'], 'scene_id': row.get('scene_id'), 'split': row['split'],
                'audit_strata': row['audit_strata'], 'source_target_xy': row['target_xy'],
                'input_allowlist': ['image', 'depth', 'instruction'],
            }
            try:
                with torch.inference_mode():
                    answer = model.generate_content(
                        [Image(str(ROOT / row['image'])), Depth(str(ROOT / row['depth'])), row['instruction']],
                        generation_config=config)
                torch.cuda.synchronize()
                prediction = point(answer)
                record.update(answer=answer, predicted_xy=prediction,
                              latency_seconds=time.perf_counter() - start)
                if prediction is None:
                    distance = None
                else:
                    distance = math.dist(prediction, row['target_xy'])
                record['normalized_l2_distance_to_source_target'] = distance
                record['triage'] = classify(distance, args.agreement_threshold, args.moderate_threshold)
            except Exception:
                record.update(triage='inference_failed', error=traceback.format_exc(),
                              latency_seconds=time.perf_counter() - start)
            predictions.append(record)
            print(f'[{index}/{len(queue)}] {row["sample_id"]}: {record["triage"]}', flush=True)
        run.update(status='COMPLETED', completed_count=len(predictions),
                   cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                   cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    except Exception:
        run.update(status='FAILED', error=traceback.format_exc(), completed_count=len(predictions))

    with (out / 'predictions.jsonl').open('w') as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    fields = ['sample_id', 'scene_id', 'split', 'audit_strata', 'triage', 'source_target_xy', 'predicted_xy',
              'normalized_l2_distance_to_source_target', 'latency_seconds', 'answer', 'error']
    with (out / 'triage.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(predictions)
    priority = sorted(predictions, key=lambda row: (
        {'inference_failed': 0, 'parse_failed': 1, 'low_coordinate_agreement': 2,
         'moderate_coordinate_agreement': 3, 'high_coordinate_agreement': 4}.get(row['triage'], 5),
        -(row.get('normalized_l2_distance_to_source_target') or -1)))
    with (out / 'human_priority.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(priority)
    counts = Counter(row['triage'] for row in predictions)
    run.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), triage_counts=dict(counts))
    (out / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    report = [
        '# AI coordinate-agreement triage', '',
        f"- Queue: `{run['queue']}` ({len(queue)} samples)",
        '- Model input: RGB, paired depth proxy, and instruction only; source target was hidden until after generation.',
        f"- Model: `{run['model']}`; RGB-D, greedy seed {run.get('seed', 'not_reached')}",
        f"- Coordinate agreement: high ≤ {args.agreement_threshold:.2f}; moderate ≤ {args.moderate_threshold:.2f}; low above that.",
        f"- Status: **{run['status']}**. Counts: {dict(counts)}.", '',
        '## Interpretation', '',
        'This is a work-prioritization signal only. Agreement does not validate a label, and disagreement does not establish an error: the model may select another valid point on the same object or may be wrong. Do not copy these outputs into the human decision CSV and do not use them to pass the clean-data gate.', '',
        'Review `human_priority.csv` from the top, then confirm every final decision in the independent human audit record.'
    ]
    (out / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(run, indent=2))
    return 0 if run['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
