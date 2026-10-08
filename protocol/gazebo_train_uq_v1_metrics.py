#!/usr/bin/env python3
"""Score frozen POINT|ABSTAIN metrics for Gazebo_train_uq_v1 materializations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from gazebo_train_uq_v1_infer import PROTOCOL_ID, load_jsonl, parse_response, sha256


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def risk_coverage(rows: list[dict]) -> tuple[list[dict], float | None]:
    if not rows or any(row.get("self_consistency_confidence") is None for row in rows):
        return [], None
    ordered = sorted(rows, key=lambda row: (-float(row["self_consistency_confidence"]), row["sample_id"]))
    mistakes = 0
    curve = []
    for index, row in enumerate(ordered, 1):
        mistakes += int(not row["task_correct"])
        curve.append({"coverage": index / len(ordered), "risk": mistakes / index, "accuracy": 1 - mistakes / index})
    return curve, sum(point["risk"] for point in curve) / len(curve)


def validate(dataset: Path, predictions_path: Path) -> tuple[dict, dict[str, dict], list[dict]]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("status") != "PASS":
        raise ValueError("dataset is not a PASS Gazebo_train_uq_v1 materialization")
    gt_rows = load_jsonl(dataset / "evaluator_ground_truth.jsonl")
    gt = {row["sample_id"]: row for row in gt_rows}
    predictions = load_jsonl(predictions_path)
    if len(gt) != int(manifest.get("records", 0)) or len(predictions) != len(gt) or {row.get("sample_id") for row in predictions} != set(gt):
        raise ValueError("prediction and evaluator-GT sample IDs must align exactly")
    if not all(row.get("answerability_verified") for row in gt.values()):
        raise ValueError("geometry answerability verification is incomplete")
    return manifest, gt, predictions


def score(dataset: Path, predictions_path: Path, output: Path, model_id: str) -> None:
    manifest, gt, predictions = validate(dataset, predictions_path)
    rows = []
    for prediction in predictions:
        truth = gt[prediction["sample_id"]]
        parsed = parse_response(prediction.get("answer") if "answer" in prediction else None)
        # Stored action is accepted only if it agrees with the frozen parser.
        if prediction.get("action") != parsed["action"] or prediction.get("prediction_xy") != parsed["prediction_xy"]:
            raise ValueError(f"prediction parser mismatch: {prediction['sample_id']}")
        state = truth["answerability_state"]
        should_point = state == "FOUND"
        action_correct = parsed["action"] == ("POINT" if should_point else "ABSTAIN")
        error = math.dist(parsed["prediction_xy"], truth["target_xy"]) if should_point and parsed["action"] == "POINT" and truth.get("target_xy") else None
        hit = error is not None and error <= 0.08
        rows.append({**prediction, "answerability_state": state, "relation_variant": truth["relation_variant"], "target_category": truth["target_category"], "failure_tags": truth["failure_tags"], "should_point": should_point, "action_correct": action_correct, "normalized_point_error": error, "hit_at_008": hit, "task_correct": hit if should_point else parsed["action"] == "ABSTAIN"})
    found = [row for row in rows if row["should_point"]]
    nonfound = [row for row in rows if not row["should_point"]]
    by_state = {}
    for state, count in manifest["state_counts"].items():
        group = [row for row in rows if row["answerability_state"] == state]
        correct = sum(row["action_correct"] for row in group)
        by_state[state] = {"records": len(group), "expected_records": count, "correct_action_rate": ratio(correct, len(group)), "correct_action_wilson_95": wilson(correct, len(group)), "point_rate": ratio(sum(row["action"] == "POINT" for row in group), len(group)), "abstain_rate": ratio(sum(row["action"] == "ABSTAIN" for row in group), len(group)), "invalid_rate": ratio(sum(row["action"] == "INVALID" for row in group), len(group))}
    curve, aurc = risk_coverage(rows)
    found_hits = sum(row["hit_at_008"] for row in found)
    metric = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "model_id": model_id,
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"), "prediction_sha256": sha256(predictions_path),
        "records": len(rows), "split_counts": dict(Counter(row["split"] for row in rows)),
        "found_records": len(found), "nonfound_records": len(nonfound),
        "found_hit_at_008": ratio(found_hits, len(found)), "found_hit_wilson_95": wilson(found_hits, len(found)),
        "found_mean_normalized_error": sum(row["normalized_point_error"] for row in found if row["normalized_point_error"] is not None) / max(1, sum(row["normalized_point_error"] is not None for row in found)),
        "answerability_action_accuracy": ratio(sum(row["action_correct"] for row in rows), len(rows)),
        "safe_task_accuracy": ratio(sum(row["task_correct"] for row in rows), len(rows)),
        "nonfound_abstain_recall": ratio(sum(row["action"] == "ABSTAIN" for row in nonfound), len(nonfound)),
        "false_accept_rate_nonfound": ratio(sum(row["action"] == "POINT" for row in nonfound), len(nonfound)),
        "invalid_rate": ratio(sum(row["action"] == "INVALID" for row in rows), len(rows)),
        "exact_contract_rate": ratio(sum(row["exact_contract"] for row in rows), len(rows)),
        "high_confidence_error_count_ge_2_3": sum(not row["task_correct"] and float(row.get("self_consistency_confidence", 0.0)) >= 2 / 3 for row in rows),
        "aurc": aurc, "risk_coverage": curve, "by_state": by_state,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scored_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "metrics.json").write_text(json.dumps(metric, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "records": len(rows), "model_id": model_id}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", choices=["b0", "clean_b1", "b1_uq"], required=True)
    args = parser.parse_args()
    score(args.dataset.resolve(), args.predictions.resolve(), args.output.resolve(), args.model_id)


if __name__ == "__main__":
    main()
