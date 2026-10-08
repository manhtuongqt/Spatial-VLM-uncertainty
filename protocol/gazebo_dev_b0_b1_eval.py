#!/usr/bin/env python3
"""Run blinded B0/clean-B1 inference and score both on Gazebo_dev."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/Gazebo_dev"
OUT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_v1/evaluation"
BASE = ROOT / "RoboRefer/models/RoboRefer-2B-SFT"
B1 = ROOT / "results/spatial_vlm_refspatial_v1/wp7_b1_clean_grounding"
LOCK = ROOT / "protocol/gazebo_dev_v1_contract_lock.json"
POINT_RE = re.compile(r"\[?\s*\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)\s*\]?")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_point(text: str):
    matches = POINT_RE.findall(text or "")
    if len(matches) != 1:
        return None
    values = [float(value) for value in matches[0]]
    return values if all(math.isfinite(value) and 0 <= value <= 1 for value in values) else None


def stable_seed(sample_id: str, model_id: str, draw: int) -> int:
    return int.from_bytes(hashlib.sha256(f"gazebo-dev-v1:{sample_id}:{model_id}:{draw}".encode()).digest()[:4], "big")


def inference(model_id: str, draws: int, max_new_tokens: int) -> None:
    manifest = json.loads((DATASET / "manifest.json").read_text())
    if manifest.get("status") != "PASS" or manifest.get("records") != 16:
        raise ValueError("Gazebo_dev is not a PASS/16-record materialization")
    rows = [json.loads(line) for line in (DATASET / "inference_manifest.jsonl").read_text().splitlines() if line]
    if len(rows) != 16:
        raise ValueError("inference manifest must contain exactly 16 rows")
    forbidden = {"target_xy", "target_id", "target_mask", "answerability_state", "candidate_set", "semantic_label"}
    leaks = [(row["sample_id"], key) for row in rows for key in forbidden if key in row]
    if leaks:
        raise ValueError(f"oracle fields leaked into inference manifest: {leaks}")
    adapter = B1 / "model/adapter_model.safetensors" if model_id == "b1" else None
    if adapter is not None and not adapter.is_file():
        raise FileNotFoundError(adapter)
    output_dir = OUT / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    run_path = output_dir / "run.json"
    contract = {
        "model_id": model_id,
        "dataset_manifest_sha256": sha256(DATASET / "manifest.json"),
        "inference_manifest_sha256": sha256(DATASET / "inference_manifest.jsonl"),
        "adapter_sha256": sha256(adapter) if adapter else None,
        "draws": draws,
        "max_new_tokens": max_new_tokens,
    }
    if predictions_path.exists():
        previous = json.loads(run_path.read_text()) if run_path.exists() else {}
        for key, value in contract.items():
            if previous.get(key) != value:
                raise ValueError(f"refusing incompatible resume: {key}")
    completed = {}
    if predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            if line:
                record = json.loads(line)
                completed[record["sample_id"]] = record
    pending = [row for row in rows if row["sample_id"] not in completed]
    run = {
        **contract, "status": "RUNNING", "protocol_id": "gazebo_dev_v1", "samples_requested": 16,
        "already_completed": len(completed), "pending": len(pending),
        "model_input_allowlist": ["rgb", "depth_view", "instruction"], "oracle_opened_by_runner": False,
        "greedy_decoding": True, "stochastic_self_consistency_draws": draws,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_path, run)
    if not pending:
        run.update(status="COMPLETED", completed_count=16, finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(run_path, run)
        return
    sys.path.insert(0, str(ROOT / "RoboRefer"))
    import torch
    import llava
    from llava import conversation as conversation_lib
    from llava.media import Depth, Image

    torch.manual_seed(8132026)
    torch.cuda.manual_seed_all(8132026)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if model_id == "b0":
        model = llava.load(str(BASE))
        model_source = str(BASE)
    else:
        model = llava.load(str(B1), model_base=str(BASE))
        model_source = str(B1 / "model")
    conversation_lib.default_conversation = conversation_lib.conv_templates["auto"].copy()
    torch.cuda.synchronize()
    greedy = copy.deepcopy(model.default_generation_config)
    greedy.do_sample = False
    greedy.temperature = greedy.top_p = greedy.top_k = None
    greedy.max_new_tokens = max_new_tokens
    stochastic = copy.deepcopy(model.default_generation_config)
    stochastic.do_sample = True
    stochastic.temperature, stochastic.top_p, stochastic.top_k = 0.7, 0.9, 50
    stochastic.max_new_tokens = max_new_tokens
    run.update(model_source=model_source, load_seconds=time.perf_counter() - started, gpu_name=torch.cuda.get_device_name())
    write_json(run_path, run)
    mode = "a" if predictions_path.exists() else "w"
    try:
        with predictions_path.open(mode, encoding="utf-8") as stream:
            for index, row in enumerate(pending, 1):
                item_started = time.perf_counter()
                prompt = [Image(str(DATASET / row["image"])), Depth(str(DATASET / row["depth"])), row["instruction"]]
                record = {key: row[key] for key in ("sample_id", "scene_id", "family_id", "split", "relation")}
                try:
                    with torch.inference_mode():
                        answer = model.generate_content(prompt, generation_config=copy.deepcopy(greedy))
                    point = parse_point(answer)
                    sample_rows = []
                    for draw in range(draws):
                        seed = stable_seed(row["sample_id"], model_id, draw)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        with torch.inference_mode():
                            sample_answer = model.generate_content(prompt, generation_config=copy.deepcopy(stochastic))
                        sample_rows.append({"draw": draw, "seed": seed, "answer": sample_answer, "prediction_xy": parse_point(sample_answer)})
                    valid = [item["prediction_xy"] for item in sample_rows if item["prediction_xy"] is not None]
                    consistent = sum(point is not None and value is not None and math.dist(point, value) <= 0.08 for value in valid)
                    record.update(answer=answer, prediction_xy=point, action="POINT" if point else "ABSTAIN_OR_INVALID", stochastic_draws=sample_rows, self_consistency_confidence=consistent / draws, predictive_uncertainty=1 - consistent / draws)
                except Exception:
                    record.update(answer=None, prediction_xy=None, action="ABSTAIN_OR_INVALID", error=traceback.format_exc(), stochastic_draws=[], self_consistency_confidence=0.0, predictive_uncertainty=1.0)
                record["latency_seconds"] = time.perf_counter() - item_started
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"[{model_id} {index}/{len(pending)}] {row['sample_id']} {record['action']}", flush=True)
        run.update(status="COMPLETED", completed_count=16, cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    except Exception:
        run.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run_path, run)


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def score() -> None:
    ground_truth = {row["sample_id"]: row for row in (json.loads(line) for line in (DATASET / "evaluator_ground_truth.jsonl").read_text().splitlines() if line)}
    if len(ground_truth) != 16 or not all(row["answerability_verified"] for row in ground_truth.values()):
        raise ValueError("evaluator GT is incomplete or unverified")
    results = {}
    enriched = {}
    for model_id in ("b0", "b1"):
        run = json.loads((OUT / model_id / "run.json").read_text())
        if run.get("status") != "COMPLETED" or run.get("oracle_opened_by_runner") is not False:
            raise ValueError(f"{model_id} inference is not complete/blinded")
        predictions = [json.loads(line) for line in (OUT / model_id / "predictions.jsonl").read_text().splitlines() if line]
        if len(predictions) != 16:
            raise ValueError(f"{model_id} must have 16 predictions")
        rows = []
        for prediction in predictions:
            gt = ground_truth[prediction["sample_id"]]
            point = prediction.get("prediction_xy")
            should_point = gt["answerability_state"] == "FOUND"
            action_point = point is not None
            error = math.dist(point, gt["target_xy"]) if should_point and action_point and gt["target_xy"] else None
            row = {**prediction, "answerability_state": gt["answerability_state"], "should_point": should_point, "decision_correct": action_point == should_point, "normalized_point_error": error, "hit_at_008": error is not None and error <= 0.08}
            rows.append(row)
        enriched[model_id] = rows
        found = [row for row in rows if row["should_point"]]
        selective = [row for row in rows if not row["should_point"]]
        results[model_id] = {
            "records": 16, "found_records": len(found), "selective_records": len(selective),
            "found_hit_at_008": _ratio(sum(row["hit_at_008"] for row in found), len(found)),
            "found_mean_normalized_error": sum(row["normalized_point_error"] for row in found if row["normalized_point_error"] is not None) / max(1, sum(row["normalized_point_error"] is not None for row in found)),
            "answerability_decision_accuracy": _ratio(sum(row["decision_correct"] for row in rows), len(rows)),
            "false_accept_rate_nonfound": _ratio(sum(row["action"] == "POINT" for row in selective), len(selective)),
            "parse_rate": _ratio(sum(row["prediction_xy"] is not None for row in rows), len(rows)),
            "mean_self_consistency_confidence": sum(row["self_consistency_confidence"] for row in rows) / len(rows),
        }
    b0 = {row["sample_id"]: row for row in enriched["b0"]}
    b1 = {row["sample_id"]: row for row in enriched["b1"]}
    found_ids = [sample_id for sample_id, gt in ground_truth.items() if gt["answerability_state"] == "FOUND"]
    fixed = sum(not b0[sid]["hit_at_008"] and b1[sid]["hit_at_008"] for sid in found_ids)
    introduced = sum(b0[sid]["hit_at_008"] and not b1[sid]["hit_at_008"] for sid in found_ids)
    comparison = {
        "found_hit_delta_b1_minus_b0": results["b1"]["found_hit_at_008"] - results["b0"]["found_hit_at_008"],
        "answerability_accuracy_delta_b1_minus_b0": results["b1"]["answerability_decision_accuracy"] - results["b0"]["answerability_decision_accuracy"],
        "b0_wrong_b1_right_found": fixed, "b0_right_b1_wrong_found": introduced,
        "promotion_decision": "PROMOTE_B1" if results["b1"]["found_hit_at_008"] > results["b0"]["found_hit_at_008"] and fixed > introduced else "KEEP_B0_AND_ANALYZE_ERRORS",
        "statistical_note": "N=4 FOUND families is a pipeline pilot, not a powered generalization claim.",
    }
    payload = {"schema_version": 1, "protocol_id": "gazebo_dev_v1", "scope": "same 16-family Gazebo_dev pilot", "metrics": results, "paired": comparison, "no_sam2": True, "b2_opened": False, "test_iid_ood_access": False}
    write_json(OUT / "B0_B1_GAZEBO_DEV_METRICS.json", payload)
    lines = ["# B0 vs clean B1 on Gazebo_dev", "", "This is a 16-family development pilot (4 per answerability state), not final Test-IID/OOD evidence.", "", "| Metric | B0 | clean B1 |", "|---|---:|---:|", f"| FOUND Hit@.08 (n=4) | {results['b0']['found_hit_at_008']:.2%} | {results['b1']['found_hit_at_008']:.2%} |", f"| Answerability decision accuracy (n=16) | {results['b0']['answerability_decision_accuracy']:.2%} | {results['b1']['answerability_decision_accuracy']:.2%} |", f"| False accept on non-FOUND (n=12) | {results['b0']['false_accept_rate_nonfound']:.2%} | {results['b1']['false_accept_rate_nonfound']:.2%} |", "", f"Decision: **{comparison['promotion_decision']}**.", "", comparison["statistical_note"], "", "No SAM2 was used; B2 remains closed and Test-IID/OOD remains sealed."]
    (OUT / "B0_B1_GAZEBO_DEV_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--model-id", choices=["b0", "b1"], required=True)
    run_parser.add_argument("--draws", type=int, default=3)
    run_parser.add_argument("--max-new-tokens", type=int, default=32)
    sub.add_parser("score")
    args = parser.parse_args()
    if args.command == "run":
        inference(args.model_id, args.draws, args.max_new_tokens)
    else:
        score()


if __name__ == "__main__":
    main()
