#!/usr/bin/env python3
"""Run resumable blinded B0/B1 RGB-D point and self-consistency evaluation."""
import argparse
import copy
import hashlib
import json
import math
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "datasets/D_tabletop_machine_v1/provenance.jsonl"
BASE = ROOT / "RoboRefer/models/RoboRefer-2B-SFT"
ADAPTER_ROOT = ROOT / "results/spatial_vlm_refspatial_v1/wp4_b1_lora_pilot"
OUT_ROOT = ROOT / "results/spatial_vlm_refspatial_v1/wp5_b0_b1_machine_eval"
POINT_RE = re.compile(r"\[?\s*\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)\s*\]?")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_first_point(text):
    match = POINT_RE.search(text or "")
    if not match:
        return None
    xy = [float(match.group(1)), float(match.group(2))]
    return xy if all(math.isfinite(v) and 0 <= v <= 1 for v in xy) else None


def stable_seed(sample_id, model_id, draw):
    payload = f"9092026:{sample_id}:{model_id}:{draw}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def load_rows():
    rows = [json.loads(line) for line in PROVENANCE.read_text().splitlines() if line]
    rows = [row for row in rows if row["split"] in {"dev", "diagnostic"}]
    rows.sort(key=lambda row: (row["split"], row["relation"], row["sample_id"]))
    if len(rows) != 180:
        raise ValueError(f"Expected 180 dev+diagnostic records, got {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=["b0", "b1"], required=True)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--radius", type=float, default=0.08)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-count", type=int, default=180,
                        help="Fail if the evaluation manifest does not contain this many dev/diagnostic rows.")
    parser.add_argument("--provenance", type=Path, default=PROVENANCE,
                        help="JSONL evaluation manifest; defaults to the immutable WP5 pilot provenance")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT,
                        help="Directory holding model prediction subdirectories")
    parser.add_argument("--adapter-root", type=Path, default=ADAPTER_ROOT,
                        help="B1 run root containing model/; ignored for B0")
    args = parser.parse_args()
    if args.draws < 1 or not 0 < args.radius < 1:
        raise ValueError("draws must be positive and radius must lie in (0, 1)")

    provenance = args.provenance.resolve()
    adapter_path = None
    adapter_sha256 = None
    if args.model_id == "b1":
        adapter_path = args.adapter_root.resolve() / "model" / "adapter_model.safetensors"
        if not adapter_path.is_file():
            raise FileNotFoundError(f"Missing B1 adapter: {adapter_path}")
        adapter_sha256 = sha256(adapter_path)
    rows = [json.loads(line) for line in provenance.read_text().splitlines() if line]
    rows = [row for row in rows if row["split"] in {"dev", "diagnostic"}]
    rows.sort(key=lambda row: (row["split"], row["relation"], row["sample_id"]))
    if len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} dev+diagnostic records, got {len(rows)} from {provenance}")
    if args.limit:
        rows = rows[: args.limit]
    output_dir = args.out_root.resolve() / args.model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.jsonl"
    run_path = output_dir / "run.json"
    previous_run = json.loads(run_path.read_text()) if run_path.exists() else None
    provenance_sha256 = sha256(provenance)
    if output_path.exists():
        if previous_run is None:
            raise ValueError(f"Cannot validate resumable predictions without {run_path}")
        expected_contract = {
            "model_id": args.model_id,
            "provenance_sha256": provenance_sha256,
            "uncertainty_draws": args.draws,
            "agreement_radius": args.radius,
            "max_new_tokens": args.max_new_tokens,
            "adapter_sha256": adapter_sha256,
        }
        mismatches = {
            key: {"expected": value, "found": previous_run.get(key)}
            for key, value in expected_contract.items()
            if previous_run.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Refusing to mix incompatible resumed predictions: {mismatches}")
    completed = {}
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line:
                row = json.loads(line)
                if row.get("uncertainty_draws_requested") == args.draws:
                    completed[row["sample_id"]] = row
    pending = [row for row in rows if row["sample_id"] not in completed]
    run = {
        "status": "RUNNING",
        "model_id": args.model_id,
        "model_input_allowlist": ["rgb", "depth", "instruction"],
        "targets_hidden_until_scoring": True,
        "provenance": str(provenance.relative_to(ROOT)),
        "provenance_sha256": provenance_sha256,
        "samples_requested": len(rows),
        "expected_manifest_count": args.expected_count,
        "already_completed": len(rows) - len(pending),
        "pending": len(pending),
        "greedy_decoding": True,
        "uncertainty_method": "five-draw stochastic self-consistency around greedy point",
        "uncertainty_draws": args.draws,
        "sampling_temperature": 0.7,
        "sampling_top_p": 0.9,
        "agreement_radius": args.radius,
        "max_new_tokens": args.max_new_tokens,
        "seed_scheme": "sha256(9092026:sample_id:model_id:draw)",
        "evaluator_sha256": sha256(Path(__file__)),
        "adapter_sha256": adapter_sha256,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    if not pending:
        for key in ("model_source", "load_seconds", "gpu_name", "cuda_peak_allocated_mib", "cuda_peak_reserved_mib"):
            if previous_run and key in previous_run:
                run[key] = previous_run[key]
        run.update(status="COMPLETED", completed_count=len(rows),
                   finished_at_utc=datetime.now(timezone.utc).isoformat())
        run_path.write_text(json.dumps(run, indent=2) + "\n")
        print(json.dumps(run, indent=2))
        return

    sys.path.insert(0, str(ROOT / "RoboRefer"))
    import torch
    import llava
    from llava import conversation as conversation_lib
    from llava.media import Depth, Image

    torch.manual_seed(9092026)
    torch.cuda.manual_seed_all(9092026)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    if args.model_id == "b0":
        model = llava.load(str(BASE))
        model_source = str(BASE)
    else:
        adapter_root = args.adapter_root.resolve()
        model = llava.load(str(adapter_root), model_base=str(BASE))
        model_source = str(adapter_root / "model")
    conversation_lib.default_conversation = conversation_lib.conv_templates["auto"].copy()
    torch.cuda.synchronize()
    run.update(model_source=model_source, load_seconds=time.perf_counter() - load_started,
               gpu_name=torch.cuda.get_device_name())
    greedy_config = copy.deepcopy(model.default_generation_config)
    greedy_config.do_sample = False
    greedy_config.temperature = greedy_config.top_p = greedy_config.top_k = None
    greedy_config.max_new_tokens = args.max_new_tokens
    sample_config = copy.deepcopy(model.default_generation_config)
    sample_config.do_sample = True
    sample_config.temperature = 0.7
    sample_config.top_p = 0.9
    sample_config.top_k = 50
    sample_config.max_new_tokens = args.max_new_tokens

    mode = "a" if output_path.exists() else "w"
    try:
        with output_path.open(mode) as handle:
            for index, row in enumerate(pending, 1):
                started = time.perf_counter()
                prompt = [Image(str(ROOT / row["image"])), Depth(str(ROOT / row["depth"])), row["instruction"]]
                record = {
                    "sample_id": row["sample_id"], "scene_id": row["scene_id"],
                    "family_id": row["family_id"], "split": row["split"], "relation": row["relation"],
                    "challenge_stratum": row.get("challenge_stratum"),
                    "uncertainty_draws_requested": args.draws,
                }
                try:
                    with torch.inference_mode():
                        greedy_answer = model.generate_content(prompt, generation_config=copy.deepcopy(greedy_config))
                    greedy_xy = parse_first_point(greedy_answer)
                    draws = []
                    for draw in range(args.draws):
                        seed = stable_seed(row["sample_id"], args.model_id, draw)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        with torch.inference_mode():
                            answer = model.generate_content(prompt, generation_config=copy.deepcopy(sample_config))
                        draws.append({"draw": draw, "seed": seed, "answer": answer,
                                      "prediction_xy": parse_first_point(answer)})
                    torch.cuda.synchronize()
                    target = row["target_xy"]
                    error = math.dist(greedy_xy, target) if greedy_xy else None
                    consistent = sum(
                        draw["prediction_xy"] is not None and greedy_xy is not None
                        and math.dist(draw["prediction_xy"], greedy_xy) <= args.radius
                        for draw in draws
                    )
                    valid_draws = [draw["prediction_xy"] for draw in draws if draw["prediction_xy"] is not None]
                    dispersion = (sum(math.dist(xy, greedy_xy) for xy in valid_draws) / len(valid_draws)
                                  if greedy_xy is not None and valid_draws else None)
                    record.update(
                        greedy_answer=greedy_answer, prediction_xy=greedy_xy,
                        target_xy=target, normalized_point_error=error,
                        hit_at_008=error is not None and error <= args.radius,
                        self_consistency_confidence=consistent / args.draws,
                        predictive_uncertainty=1 - consistent / args.draws,
                        valid_stochastic_draws=len(valid_draws),
                        stochastic_dispersion_from_greedy=dispersion,
                        stochastic_draws=draws,
                        latency_seconds=time.perf_counter() - started,
                    )
                except Exception:
                    record.update(error=traceback.format_exc(), prediction_xy=None, target_xy=row["target_xy"],
                                  normalized_point_error=None, hit_at_008=False,
                                  self_consistency_confidence=0.0, predictive_uncertainty=1.0,
                                  valid_stochastic_draws=0, stochastic_draws=[],
                                  latency_seconds=time.perf_counter() - started)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if index % 10 == 0 or index == len(pending):
                    print(f"[{args.model_id} {index}/{len(pending)}] {row['split']} {row['relation']} "
                          f"hit={record['hit_at_008']} conf={record['self_consistency_confidence']:.2f}", flush=True)
        run.update(status="COMPLETED", completed_count=len(rows),
                   cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                   cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    except Exception:
        run.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_path.write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
