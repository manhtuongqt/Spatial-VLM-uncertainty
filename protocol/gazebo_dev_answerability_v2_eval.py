#!/usr/bin/env python3
"""Run blinded B0/B1 inference and score Gazebo answerability-v2."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
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
PROTOCOL_ID = "gazebo_dev_answerability_v2"
N = 64
DATASET = ROOT / "datasets/Gazebo_dev_answerability_v2"
OUT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2/evaluation"
BASE = ROOT / "RoboRefer/models/RoboRefer-2B-SFT"
B1 = ROOT / "results/spatial_vlm_refspatial_v1/wp7_b1_clean_grounding"
LOCK = ROOT / "protocol/gazebo_dev_answerability_v2_contract_lock.json"
GATE = ROOT / "ur3/ur3_perception/config/gazebo_dev_answerability_v2_gate.yaml"
TOLERANT_POINT_RE = re.compile(r"\[?\s*\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)\s*\]?")
EXACT_POINT_RE = re.compile(r"^POINT \[\(([0-9]*\.?[0-9]+), ([0-9]*\.?[0-9]+)\)\]$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_response(text: str | None) -> dict:
    value = (text or "").strip()
    matches = TOLERANT_POINT_RE.findall(value)
    abstain = bool(re.search(r"\bABSTAIN\b", value, re.IGNORECASE))
    if len(matches) == 1 and not abstain:
        point = [float(item) for item in matches[0]]
        if all(math.isfinite(item) and 0 <= item <= 1 for item in point):
            return {"action": "POINT", "prediction_xy": point, "exact_contract": bool(EXACT_POINT_RE.fullmatch(value))}
    if not matches and abstain:
        return {"action": "ABSTAIN", "prediction_xy": None, "exact_contract": value == "ABSTAIN"}
    return {"action": "INVALID", "prediction_xy": None, "exact_contract": False}


def stable_seed(sample_id: str, model_id: str, draw: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{PROTOCOL_ID}:{sample_id}:{model_id}:{draw}".encode()).digest()[:4], "big")


def inference(model_id: str, draws: int, max_new_tokens: int) -> None:
    manifest = json.loads((DATASET / "manifest.json").read_text())
    if manifest.get("status") != "PASS" or manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("records") != N:
        raise ValueError("answerability-v2 is not a PASS/64-record materialization")
    rows = load_jsonl(DATASET / "inference_manifest.jsonl")
    if len(rows) != N:
        raise ValueError("inference manifest must contain exactly 64 rows")
    forbidden = {"target_xy", "target_id", "target_mask", "answerability_state", "candidate_set", "semantic_label", "failure_tags"}
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
        "model_id": model_id, "protocol_id": PROTOCOL_ID,
        "contract_lock_sha256": sha256(LOCK),
        "dataset_manifest_sha256": sha256(DATASET / "manifest.json"),
        "inference_manifest_sha256": sha256(DATASET / "inference_manifest.jsonl"),
        "adapter_sha256": sha256(adapter) if adapter else None,
        "draws": draws, "max_new_tokens": max_new_tokens,
    }
    if predictions_path.exists():
        previous = json.loads(run_path.read_text()) if run_path.exists() else {}
        for key, value in contract.items():
            if previous.get(key) != value:
                raise ValueError(f"refusing incompatible resume: {key}")
    completed = {row["sample_id"]: row for row in load_jsonl(predictions_path)} if predictions_path.exists() else {}
    pending = [row for row in rows if row["sample_id"] not in completed]
    run = {
        **contract, "status": "RUNNING", "samples_requested": N, "already_completed": len(completed), "pending": len(pending),
        "model_input_allowlist": ["rgb", "depth_view", "instruction"], "oracle_opened_by_runner": False,
        "greedy_decoding": True, "stochastic_self_consistency_draws": draws,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_path, run)
    if not pending:
        run.update(status="COMPLETED", completed_count=N, finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(run_path, run)
        return
    sys.path.insert(0, str(ROOT / "RoboRefer"))
    import torch
    import llava
    from llava import conversation as conversation_lib
    from llava.media import Depth, Image

    torch.manual_seed(12092026)
    torch.cuda.manual_seed_all(12092026)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if model_id == "b0":
        model = llava.load(str(BASE)); model_source = str(BASE)
    else:
        model = llava.load(str(B1), model_base=str(BASE)); model_source = str(B1 / "model")
    conversation_lib.default_conversation = conversation_lib.conv_templates["auto"].copy()
    torch.cuda.synchronize()
    greedy = copy.deepcopy(model.default_generation_config)
    greedy.do_sample = False; greedy.temperature = greedy.top_p = greedy.top_k = None; greedy.max_new_tokens = max_new_tokens
    stochastic = copy.deepcopy(model.default_generation_config)
    stochastic.do_sample = True; stochastic.temperature = 0.7; stochastic.top_p = 0.9; stochastic.top_k = 50; stochastic.max_new_tokens = max_new_tokens
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
                    parsed = parse_response(answer)
                    sample_rows = []
                    for draw in range(draws):
                        seed = stable_seed(row["sample_id"], model_id, draw)
                        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                        with torch.inference_mode():
                            sample_answer = model.generate_content(prompt, generation_config=copy.deepcopy(stochastic))
                        sample_rows.append({"draw": draw, "seed": seed, "answer": sample_answer, **parse_response(sample_answer)})
                    if parsed["action"] == "POINT":
                        consistent = sum(item["action"] == "POINT" and math.dist(parsed["prediction_xy"], item["prediction_xy"]) <= 0.08 for item in sample_rows)
                    elif parsed["action"] == "ABSTAIN":
                        consistent = sum(item["action"] == "ABSTAIN" for item in sample_rows)
                    else:
                        consistent = 0
                    record.update(answer=answer, **parsed, stochastic_draws=sample_rows, self_consistency_confidence=consistent / draws, predictive_uncertainty=1 - consistent / draws)
                except Exception:
                    record.update(answer=None, action="INVALID", prediction_xy=None, exact_contract=False, error=traceback.format_exc(), stochastic_draws=[], self_consistency_confidence=0.0, predictive_uncertainty=1.0)
                record["latency_seconds"] = time.perf_counter() - item_started
                stream.write(json.dumps(record, ensure_ascii=False) + "\n"); stream.flush()
                print(f"[{model_id} {index}/{len(pending)}] {row['sample_id']} {record['action']}", flush=True)
        run.update(status="COMPLETED", completed_count=N, cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    except Exception:
        run.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run_path, run)


def ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if not total:
        return None
    p = successes / total; denominator = 1 + z*z/total
    center = (p + z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return [center-half, center+half]


def error_auroc(rows: list[dict]) -> float | None:
    positives = [row for row in rows if not row["task_correct"]]
    negatives = [row for row in rows if row["task_correct"]]
    if not positives or not negatives:
        return None
    score = 0.0
    for positive in positives:
        for negative in negatives:
            ps = positive["predictive_uncertainty"]; ns = negative["predictive_uncertainty"]
            score += 1.0 if ps > ns else 0.5 if ps == ns else 0.0
    return score / (len(positives) * len(negatives))


def risk_curve(rows: list[dict]) -> tuple[list[dict], float]:
    ordered = sorted(rows, key=lambda row: (-row["self_consistency_confidence"], row["sample_id"]))
    curve = []
    mistakes = 0
    for index, row in enumerate(ordered, 1):
        mistakes += int(not row["task_correct"])
        curve.append({"coverage": index/len(rows), "risk": mistakes/index, "accuracy": 1-mistakes/index})
    return curve, sum(point["risk"] for point in curve) / len(curve)


def mcnemar(a: int, b: int) -> float:
    n = a+b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a, b)+1)) / 2**n
    return min(1.0, 2*tail)


def bootstrap_delta(b0: list[bool], b1: list[bool], seed: int = 12092026, draws: int = 10000) -> list[float]:
    import numpy as np
    rng = np.random.default_rng(seed); a = np.asarray(b0, dtype=float); b = np.asarray(b1, dtype=float)
    indices = rng.integers(0, len(a), size=(draws, len(a)))
    deltas = b[indices].mean(axis=1) - a[indices].mean(axis=1)
    return [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]


def render_plots(metrics: dict, enriched: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})
    colors = {"b0": "#276FBF", "b1": "#F28E2B"}
    labels = ["FOUND\nHit@.08", "Answerability\naction accuracy", "Safe-task\naccuracy", "Exact output\ncontract"]
    x = np.arange(4); width = 0.34
    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, model in ((-width/2, "b0"), (width/2, "b1")):
        m = metrics[model]; values = [m["found_hit_at_008"], m["answerability_action_accuracy"], m["safe_task_accuracy"], m["exact_contract_rate"]]
        bars = ax.bar(x+offset, values, width, label=model.upper(), color=colors[model])
        for bar, value in zip(bars, values): ax.text(bar.get_x()+bar.get_width()/2, value+.02, f"{value:.1%}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.12); ax.set_xticks(x, labels); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.set_title("Gazebo answerability-v2 — B0 vs clean B1", loc="left", fontweight="bold"); ax.legend(); ax.set_axisbelow(True)
    fig.savefig(OUT / "01_b0_b1_summary.png", dpi=300); plt.close(fig)
    states = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
    matrix = np.array([[metrics[model]["by_state"][state]["correct_action_rate"] for state in states] for model in ("b0", "b1")])
    fig, ax = plt.subplots(figsize=(11, 4.5)); im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    for i in range(2):
        for j in range(4): ax.text(j, i, f"{matrix[i,j]:.1%}", ha="center", va="center", fontweight="bold")
    ax.set_xticks(range(4), ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT\nEVIDENCE"]); ax.set_yticks(range(2), ["B0", "Clean B1"]); ax.set_title("Correct action rate theo answerability state", loc="left", fontweight="bold"); fig.colorbar(im, ax=ax, label="Correct action rate")
    fig.savefig(OUT / "02_action_accuracy_by_state.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    for model in ("b0", "b1"):
        curve = metrics[model]["risk_coverage"]
        ax.step([0]+[p["coverage"] for p in curve], [curve[0]["risk"]]+[p["risk"] for p in curve], where="post", color=colors[model], lw=2, label=f"{model.upper()} (AURC={metrics[model]['aurc']:.3f})")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.set_xlabel("Coverage"); ax.set_ylabel("Risk"); ax.set_title("Risk–coverage từ self-consistency proxy", loc="left", fontweight="bold"); ax.legend(); ax.set_axisbelow(True)
    fig.savefig(OUT / "03_risk_coverage.png", dpi=300); plt.close(fig)


def score() -> None:
    import yaml
    ground_truth = {row["sample_id"]: row for row in load_jsonl(DATASET / "evaluator_ground_truth.jsonl")}
    if len(ground_truth) != N or not all(row["answerability_verified"] for row in ground_truth.values()):
        raise ValueError("evaluator GT is incomplete or unverified")
    gate = yaml.safe_load(GATE.read_text())
    results = {}; enriched = {}
    for model_id in ("b0", "b1"):
        run = json.loads((OUT / model_id / "run.json").read_text())
        if run.get("status") != "COMPLETED" or run.get("completed_count") != N or run.get("oracle_opened_by_runner") is not False:
            raise ValueError(f"{model_id} inference is not complete/blinded")
        predictions = load_jsonl(OUT / model_id / "predictions.jsonl")
        if len(predictions) != N or len({row["sample_id"] for row in predictions}) != N:
            raise ValueError(f"{model_id} predictions are incomplete/non-unique")
        rows = []
        for prediction in predictions:
            gt = ground_truth[prediction["sample_id"]]; state = gt["answerability_state"]; point = prediction.get("prediction_xy")
            should_point = state == "FOUND"; action_correct = prediction["action"] == ("POINT" if should_point else "ABSTAIN")
            error = math.dist(point, gt["target_xy"]) if should_point and point and gt.get("target_xy") else None
            hit = error is not None and error <= 0.08
            rows.append({**prediction, "answerability_state": state, "relation_variant": gt["relation_variant"], "target_category": gt["target_category"], "failure_tags": gt["failure_tags"], "should_point": should_point, "action_correct": action_correct, "normalized_point_error": error, "hit_at_008": hit, "task_correct": hit if should_point else prediction["action"] == "ABSTAIN"})
        enriched[model_id] = rows
        found = [row for row in rows if row["should_point"]]; nonfound = [row for row in rows if not row["should_point"]]
        curve, aurc = risk_curve(rows)
        by_state = {}
        for state in gate["state_quota"]:
            group = [row for row in rows if row["answerability_state"] == state]
            correct = sum(row["action_correct"] for row in group)
            by_state[state] = {"records": len(group), "correct_action_rate": ratio(correct, len(group)), "correct_action_wilson_95": wilson(correct, len(group)), "point_rate": ratio(sum(row["action"] == "POINT" for row in group), len(group)), "abstain_rate": ratio(sum(row["action"] == "ABSTAIN" for row in group), len(group)), "invalid_rate": ratio(sum(row["action"] == "INVALID" for row in group), len(group))}
        found_hits = sum(row["hit_at_008"] for row in found); safe = sum(row["task_correct"] for row in rows); action_correct = sum(row["action_correct"] for row in rows)
        results[model_id] = {
            "records": N, "found_records": len(found), "nonfound_records": len(nonfound),
            "found_hit_at_008": ratio(found_hits, len(found)), "found_hit_wilson_95": wilson(found_hits, len(found)),
            "found_mean_normalized_error": sum(row["normalized_point_error"] for row in found if row["normalized_point_error"] is not None) / max(1, sum(row["normalized_point_error"] is not None for row in found)),
            "answerability_action_accuracy": ratio(action_correct, N), "answerability_action_wilson_95": wilson(action_correct, N),
            "safe_task_accuracy": ratio(safe, N), "safe_task_wilson_95": wilson(safe, N),
            "nonfound_abstain_recall": ratio(sum(row["action"] == "ABSTAIN" for row in nonfound), len(nonfound)),
            "false_accept_rate_nonfound": ratio(sum(row["action"] == "POINT" for row in nonfound), len(nonfound)),
            "invalid_rate": ratio(sum(row["action"] == "INVALID" for row in rows), N),
            "exact_contract_rate": ratio(sum(row["exact_contract"] for row in rows), N),
            "mean_self_consistency_confidence": sum(row["self_consistency_confidence"] for row in rows)/N,
            "high_confidence_error_count_ge_2_3": sum((not row["task_correct"]) and row["self_consistency_confidence"] >= 2/3 for row in rows),
            "error_detection_auroc": error_auroc(rows), "aurc": aurc, "risk_coverage": curve, "by_state": by_state,
        }
    b0 = {row["sample_id"]: row for row in enriched["b0"]}; b1 = {row["sample_id"]: row for row in enriched["b1"]}; ids = sorted(ground_truth)
    b0_task = [b0[sid]["task_correct"] for sid in ids]; b1_task = [b1[sid]["task_correct"] for sid in ids]
    b0_found = [b0[sid]["hit_at_008"] for sid in ids if b0[sid]["should_point"]]; b1_found = [b1[sid]["hit_at_008"] for sid in ids if b1[sid]["should_point"]]
    fixes = sum(not a and b for a,b in zip(b0_task,b1_task)); introduced = sum(a and not b for a,b in zip(b0_task,b1_task))
    exact_regression = results["b1"]["exact_contract_rate"] < results["b0"]["exact_contract_rate"]
    promote = results["b1"]["found_hit_at_008"] > results["b0"]["found_hit_at_008"] and results["b1"]["safe_task_accuracy"] > results["b0"]["safe_task_accuracy"] and not exact_regression
    paired = {
        "safe_task_delta_b1_minus_b0": results["b1"]["safe_task_accuracy"]-results["b0"]["safe_task_accuracy"],
        "safe_task_family_bootstrap_delta_95": bootstrap_delta(b0_task,b1_task),
        "found_hit_delta_b1_minus_b0": results["b1"]["found_hit_at_008"]-results["b0"]["found_hit_at_008"],
        "found_hit_family_bootstrap_delta_95": bootstrap_delta(b0_found,b1_found,seed=12092027),
        "b0_wrong_b1_right": fixes, "b0_right_b1_wrong": introduced, "mcnemar_exact_two_sided_p": mcnemar(fixes,introduced),
        "promotion_rule": gate["promotion_rule"], "promotion_decision": "PROMOTE_B1" if promote else "KEEP_B0_AND_ANALYZE_ERRORS",
    }
    payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "scope": "64 independent Gazebo Dev answerability families", "metrics": results, "paired": paired, "no_sam2": True, "b2_opened": False, "test_iid_ood_access": False, "statistical_note": "Development data for method selection/failure analysis; not final generalization evidence."}
    write_json(OUT / "B0_B1_ANSWERABILITY_V2_METRICS.json", payload)
    for model_id, rows in enriched.items():
        with (OUT / f"{model_id}/scored_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows: stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n")
    render_plots(results,enriched)
    lines = [
        "# B0 vs clean B1 — Gazebo_dev_answerability_v2", "", "64 independent Dev families; 16 per answerability state. This is not Test-IID/OOD evidence.", "",
        "| Metric | B0 | clean B1 |", "|---|---:|---:|",
        f"| FOUND Hit@.08 (n=16) | {results['b0']['found_hit_at_008']:.2%} | {results['b1']['found_hit_at_008']:.2%} |",
        f"| Answerability action accuracy (n=64) | {results['b0']['answerability_action_accuracy']:.2%} | {results['b1']['answerability_action_accuracy']:.2%} |",
        f"| Safe-task accuracy (n=64) | {results['b0']['safe_task_accuracy']:.2%} | {results['b1']['safe_task_accuracy']:.2%} |",
        f"| Non-FOUND abstain recall (n=48) | {results['b0']['nonfound_abstain_recall']:.2%} | {results['b1']['nonfound_abstain_recall']:.2%} |",
        f"| False accept on non-FOUND | {results['b0']['false_accept_rate_nonfound']:.2%} | {results['b1']['false_accept_rate_nonfound']:.2%} |",
        f"| Exact output contract | {results['b0']['exact_contract_rate']:.2%} | {results['b1']['exact_contract_rate']:.2%} |",
        f"| Error-detection AUROC | {results['b0']['error_detection_auroc']} | {results['b1']['error_detection_auroc']} |",
        f"| AURC | {results['b0']['aurc']:.4f} | {results['b1']['aurc']:.4f} |", "",
        f"Decision: **{paired['promotion_decision']}**.", "", "No SAM2 was used; B2 remained closed and Test-IID/OOD remained sealed.",
    ]
    (OUT / "B0_B1_ANSWERABILITY_V2_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run"); run_parser.add_argument("--model-id", choices=["b0","b1"], required=True); run_parser.add_argument("--draws", type=int, default=3); run_parser.add_argument("--max-new-tokens", type=int, default=40)
    sub.add_parser("score")
    args = parser.parse_args()
    if args.command == "run": inference(args.model_id,args.draws,args.max_new_tokens)
    else: score()


if __name__ == "__main__":
    main()
