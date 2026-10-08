#!/usr/bin/env python3
"""Resumable blinded RoboRefer RGB-D agreement for the machine LoRA pilot."""
import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "results/spatial_vlm_refspatial_v1/wp3_machine_lora_pilot/agreement_queue.jsonl"
DEFAULT_OUT = ROOT / "results/spatial_vlm_refspatial_v1/wp3_machine_lora_pilot/roborefer_agreement"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_point(text):
    import ast
    try:
        value = ast.literal_eval(text)
        if len(value) == 1 and len(value[0]) == 2:
            x, y = map(float, value[0])
            return (x, y) if 0 <= x <= 1 and 0 <= y <= 1 else None
    except (SyntaxError, ValueError, TypeError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    queue = [json.loads(line) for line in args.queue.read_text().splitlines() if line]
    if args.limit:
        queue = queue[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "predictions.jsonl"
    done = {json.loads(line)["sample_id"] for line in output.read_text().splitlines()} if output.exists() else set()
    pending = [row for row in queue if row["sample_id"] not in done]
    run = {"kind": "MACHINE_FILTER_ROBOREFER_AGREEMENT", "queue": str(args.queue.relative_to(ROOT)),
           "queue_sha256": sha(args.queue), "queue_count": len(queue), "already_completed": len(done & {r['sample_id'] for r in queue}),
           "pending": len(pending), "source_target_hidden_from_model": True, "threshold_normalized_l2": .08,
           "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING"}
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    if not pending:
        run.update(status="COMPLETED", finished_at_utc=datetime.now(timezone.utc).isoformat())
        (args.output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n"); print(json.dumps(run, indent=2)); return
    sys.path.insert(0, str(ROOT / "RoboRefer"))
    import torch
    import llava
    from llava import conversation as clib
    from llava.media import Depth, Image
    torch.manual_seed(9092026); torch.cuda.manual_seed_all(9092026)
    model = llava.load(str(ROOT / "RoboRefer/models/RoboRefer-2B-SFT"))
    clib.default_conversation = clib.conv_templates["auto"].copy()
    config = model.default_generation_config
    config.do_sample = False; config.temperature = config.top_p = config.top_k = None; config.max_new_tokens = 256
    with output.open("a") as handle:
        for index, row in enumerate(pending, 1):
            record = {"sample_id": row["sample_id"], "scene_id": row["scene_id"], "split": row["split"],
                      "relation": row["relation"], "source_target_xy": row["target_xy"],
                      "input_allowlist": ["image", "depth", "instruction"]}
            start = time.perf_counter()
            try:
                with torch.inference_mode():
                    answer = model.generate_content([Image(str(ROOT / row["image"])), Depth(str(ROOT / row["depth"])), row["instruction"]], generation_config=config)
                prediction = parse_point(answer)
                distance = math.dist(prediction, row["target_xy"]) if prediction else None
                triage = "high_coordinate_agreement" if distance is not None and distance <= .08 else "low_coordinate_agreement" if distance is not None else "parse_failed"
                record.update(answer=answer, predicted_xy=prediction, normalized_l2_distance_to_source_target=distance, triage=triage)
            except Exception:
                record.update(triage="inference_failed", error=traceback.format_exc())
            record["latency_seconds"] = time.perf_counter() - start
            handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush()
            if index % 25 == 0 or index == len(pending):
                print(f"[{index}/{len(pending)}] {record['sample_id']} {record['triage']}", flush=True)
    run.update(status="COMPLETED", finished_at_utc=datetime.now(timezone.utc).isoformat())
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
