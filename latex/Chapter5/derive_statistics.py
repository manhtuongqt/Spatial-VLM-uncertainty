"""Read-only supplementary analysis for Chapter 5; prints JSON to stdout."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "new/test_iid/evaluation/best_v2/predictions.jsonl"
DECISIONS = ROOT / "new/test_iid/evaluation/best_v2/decisions.jsonl"
MANIFEST = ROOT / "new/test_iid/protocol/test_iid_eval_manifest.json"
SOURCES = ["semantic", "relation", "spatial", "depth", "occlusion"]
ANSWERS = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]


def ranks(scores):
    order = np.argsort(scores, kind="mergesort")
    ranked = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranked


def score_metrics(truth, score):
    positive, negative = int(truth.sum()), int((1 - truth).sum())
    rank = ranks(score)
    roc = (rank[truth == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
    order = np.argsort(-score, kind="mergesort")
    ordered = truth[order]
    precision = np.cumsum(ordered) / np.arange(1, len(truth) + 1)
    ap = float(precision[ordered == 1].sum() / positive)
    p = np.clip(score, 1e-7, 1 - 1e-7)
    ece = 0.0
    for i in range(10):
        low, high = i / 10, (i + 1) / 10
        selected = (score >= low) & ((score < high) if i < 9 else (score <= high))
        if selected.any():
            ece += selected.mean() * abs(score[selected].mean() - truth[selected].mean())
    order = np.argsort(score, kind="mergesort")
    return {
        "roc_auc": float(roc),
        "average_precision": ap,
        "brier": float(np.mean((score - truth) ** 2)),
        "nll": float(-np.mean(truth * np.log(p) + (1 - truth) * np.log(1 - p))),
        "ece_10": float(ece),
        "aurc": float(np.mean(np.cumsum(truth[order]) / np.arange(1, len(truth) + 1))),
    }


def main():
    predictions = [json.loads(line) for line in PREDICTIONS.read_text().splitlines()]
    decisions = {r["sample_id"]: r for r in map(json.loads, DECISIONS.read_text().splitlines())}
    entries = {r["sample_id"]: r for r in json.loads(MANIFEST.read_text())["entries"]}
    assert len(predictions) == len(decisions) == len(entries) == 1000
    assert set(entries) == set(decisions) == {r["sample_id"] for r in predictions}
    truth = np.asarray([r["evaluation"]["error_event"] for r in predictions], dtype=float)
    raw = np.asarray([1 - r["answerability_probabilities"]["FOUND"] for r in predictions])
    risk = np.asarray([decisions[r["sample_id"]]["calibrated_grounding_risk"] for r in predictions])
    metrics = json.loads((ROOT / "new/test_iid/evaluation/best_v2/full_metrics.json").read_text())
    calibrated = score_metrics(truth, risk)
    for k, v in calibrated.items():
        assert np.isclose(v, metrics["risk_calibration_and_ranking"][k], rtol=0, atol=1e-12)
    matrix = np.asarray(metrics["answerability"]["confusion_matrix"])
    balanced = float(np.mean(np.diag(matrix) / matrix.sum(axis=1)))
    source_macro4 = float(np.mean([
        metrics["source_uncertainty"]["per_source"][k]["f1"]
        for k in SOURCES if k != "spatial"
    ]))
    family = defaultdict(dict)
    for row in predictions:
        family[row["family_id"]][row["variant"]] = row
    paired = {}
    for variant in ["semantic_counterfactual", "relation_counterfactual",
                    "depth_corruption", "occlusion_view_counterfactual"]:
        deltas = np.asarray([
            [members[variant]["source_probabilities"][s] -
             members["clean"]["source_probabilities"][s] for s in SOURCES]
            for members in family.values()
        ])
        assert len(deltas) == 200
        paired[variant] = {
            "family_pairs": len(deltas),
            "mean_delta": dict(zip(SOURCES, deltas.mean(0).tolist())),
            "positive_delta_fraction": dict(zip(SOURCES, (deltas > 0).mean(0).tolist())),
        }
    action_order = ["EXECUTE", "REOBSERVE", "ASK_USER", "ABSTAIN"]
    actual_expected = [[0] * 4 for _ in range(4)]
    overlap = Counter()
    nonfound_executed = Counter()
    for row in predictions:
        action = decisions[row["sample_id"]]["action"]
        expected = entries[row["sample_id"]]["audit_only"]["expected_intervention"]
        actual_expected[action_order.index(expected)][action_order.index(action)] += 1
        if action == "EXECUTE":
            nonfound = row["evaluation"]["answerability_state"] != "FOUND"
            outside = not row["evaluation"]["map_inside_target"]
            overlap[f"nonfound={nonfound},outside={outside}"] += 1
            if nonfound:
                nonfound_executed[row["evaluation"]["answerability_state"]] += 1
    selected = ["000127__clean", "000003__depth_corruption", "000031__clean",
                "000005__clean", "000127__semantic_counterfactual",
                "000002__clean", "000165__relation_counterfactual"]
    by_id = {r["sample_id"]: r for r in predictions}
    cases = {}
    for short in selected:
        sid = "v211iid_family_" + short
        r, d = by_id[sid], decisions[sid]
        cases[sid] = {
            "truth": r["evaluation"]["answerability_state"],
            "predicted": max(r["answerability_probabilities"], key=r["answerability_probabilities"].get),
            "map": r["spatial"]["map_pixel_xy"],
            "hit": r["evaluation"]["map_inside_target"],
            "risk": d["calibrated_grounding_risk"],
            "action": d["action"],
            "source": r["source_probabilities"],
            "truth_source": dict(zip(SOURCES, r["evaluation"]["source_multihot_5"])),
        }
    v3 = {}
    for seed in [24082026, 24082027, 24082028]:
        history = ROOT / f"new/outputs/pcrau_target_v3_seed_{seed}/history.jsonl"
        records = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
        best = max(records, key=lambda x: x["dev"]["grounding_accuracy"])
        v3[str(seed)] = {"max_observed_dev_grounding": best["dev"]["grounding_accuracy"],
                         "epoch_at_max": best["epoch"], "history_epochs": len(records),
                         "best_json_exists": (history.parent / "best.json").is_file()}
    result = {
        "provenance_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [PREDICTIONS, DECISIONS, MANIFEST]},
        "no_training_or_threshold_fitting": True,
        "raw_risk_definition": "1 - predicted FOUND probability; no logistic/scalar fitting",
        "raw_risk_metrics": score_metrics(truth, raw),
        "calibrated_risk_metrics_verified": calibrated,
        "answerability_balanced_accuracy": balanced,
        "source_macro_f1_excluding_weak_spatial": source_macro4,
        "counterfactual_paired_source_changes": paired,
        "expected_action_order": action_order,
        "expected_vs_predicted_action_confusion": actual_expected,
        "execute_error_overlap": dict(overlap),
        "nonfound_executed_by_truth": dict(nonfound_executed),
        "cases": cases,
        "v3_observed_history": v3,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
