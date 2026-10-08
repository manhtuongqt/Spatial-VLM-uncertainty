#!/usr/bin/env python3
"""Paired dev comparison: original RoboRefer B1 versus best P-CRA-U V2."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_B1 = WORKSPACE / "new" / "outputs" / "roborefer_original_b1_rerun_seed_8132026" / "predictions.jsonl"
DEFAULT_V2 = WORKSPACE / "new" / "outputs" / "pcrau_target_v2_full_seed_24082026" / "evaluation" / "dev" / "predictions.jsonl"
DEFAULT_TRUTH = WORKSPACE / "old" / "protocol" / "pcra_u_development_train_manifest.json"
DEFAULT_ARCHIVED_B1 = WORKSPACE / "old" / "results" / "pcra_u_runs" / "pcra_u_development_failure_audit_20260824" / "predictions" / "roborefer_baselines_dev.jsonl"
DEFAULT_OUTPUT = WORKSPACE / "new" / "outputs" / "roborefer_original_b1_rerun_seed_8132026"
DATASET_ROOT = WORKSPACE / "old" / "roborefer_dataset_v2_1_1_development_400_20260824"
FORMER_DATASET_NAME = "roborefer_dataset_v2_1_1_development_400_20260824"
CLASSES = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def resolve_dataset_path(raw_path: str) -> Path:
    raw = Path(raw_path)
    if FORMER_DATASET_NAME not in raw.parts:
        raise ValueError(f"Unexpected archived dataset path: {raw_path}")
    suffix = raw.parts[raw.parts.index(FORMER_DATASET_NAME) + 1 :]
    result = DATASET_ROOT.joinpath(*suffix).resolve()
    if DATASET_ROOT.resolve() not in result.parents:
        raise ValueError(f"Path escapes development archive: {raw_path}")
    return result


def point_in_mask(point: Iterable[int], mask_path: str) -> bool:
    x, y = (int(value) for value in point)
    mask = cv2.imread(str(resolve_dataset_path(mask_path)), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (480, 640):
        raise RuntimeError(f"Invalid evaluator mask: {mask_path}")
    return bool(mask[y, x] > 0)


def classification_metrics(truth: list[str], prediction: list[str]) -> dict:
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    lookup = {name: index for index, name in enumerate(CLASSES)}
    for expected, observed in zip(truth, prediction):
        matrix[lookup[expected], lookup[observed]] += 1
    per_class, f1s = {}, []
    for index, name in enumerate(CLASSES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[index].sum())}
        f1s.append(f1)
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix_truth_rows_prediction_columns": matrix.tolist(),
        "class_order": CLASSES,
        "per_class": per_class,
    }


def ratio(success: int, total: int) -> dict:
    return {"success": int(success), "total": int(total), "rate": success / total if total else None}


def grouped_rates(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        name: {
            "samples": len(items),
            "original_b1_point_in_target": float(np.mean([item["b1_hit"] for item in items])),
            "best_v2_point_in_target": float(np.mean([item["v2_hit"] for item in items])),
            "delta_v2_minus_b1": float(np.mean([item["v2_hit"] - item["b1_hit"] for item in items])),
        }
        for name, items in sorted(groups.items())
    }


def cluster_bootstrap(rows: list[dict], seed: int = 24082027, replicates: int = 5000) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["family_id"]].append(row)
    families = sorted(groups)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(families, len(families), replace=True)
        draw = [row for family in sampled for row in groups[str(family)]]
        deltas[index] = np.mean([row["v2_hit"] - row["b1_hit"] for row in draw])
    return {
        "unit": "family",
        "families": len(families),
        "replicates": replicates,
        "seed": seed,
        "estimate": float(np.mean([row["v2_hit"] - row["b1_hit"] for row in rows])),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
    }


def percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def run(args: argparse.Namespace) -> None:
    b1_path, v2_path, truth_path = map(lambda value: Path(value).resolve(), (args.b1, args.v2, args.truth))
    output_dir = Path(args.output_dir).resolve()
    b1_rows, v2_rows = read_jsonl(b1_path), read_jsonl(v2_path)
    truth_manifest = json.loads(truth_path.read_text(encoding="utf-8"))
    truth_rows = [row for row in truth_manifest["entries"] if row["split"] == "dev"]
    if not (len(b1_rows) == len(v2_rows) == len(truth_rows) == 400):
        raise RuntimeError(f"Expected 400 paired rows, got B1={len(b1_rows)}, V2={len(v2_rows)}, truth={len(truth_rows)}")
    b1 = {row["sample_id"]: row for row in b1_rows}
    v2 = {row["sample_id"]: row for row in v2_rows}
    truth = {row["sample_id"]: row for row in truth_rows}
    if not (set(b1) == set(v2) == set(truth)):
        raise RuntimeError("B1, V2, and truth sample IDs are not identical")

    paired_found = []
    answer_truth, answer_b1, answer_v2 = [], [], []
    for sample_id in sorted(truth):
        expected = truth[sample_id]
        supervision = expected["supervision"]
        state = supervision["answerability_state"]
        b1_point = b1[sample_id].get("pixel_points_xy", [])
        if len(b1_point) != 1:
            b1_prediction = "INSUFFICIENT_EVIDENCE"
        else:
            b1_prediction = "FOUND"
        probabilities = v2[sample_id]["answerability_probabilities"]
        v2_prediction = max(CLASSES, key=lambda name: probabilities[name])
        answer_truth.append(state)
        answer_b1.append(b1_prediction)
        answer_v2.append(v2_prediction)
        if state == "FOUND":
            if len(b1_point) != 1:
                b1_hit = b1_interior = False
            else:
                b1_hit = point_in_mask(b1_point[0], supervision["target_mask_path"])
                b1_interior = point_in_mask(b1_point[0], supervision["target_interior_mask_path"])
            v2_point = v2[sample_id]["spatial"]["map_pixel_xy"]
            v2_hit = point_in_mask(v2_point, supervision["target_mask_path"])
            v2_interior = point_in_mask(v2_point, supervision["target_interior_mask_path"])
            paired_found.append(
                {
                    "sample_id": sample_id,
                    "family_id": expected["family_id"],
                    "variant": expected["variant"],
                    "relation": expected["audit_only"]["relation"],
                    "b1_hit": int(b1_hit),
                    "v2_hit": int(v2_hit),
                    "b1_interior": int(b1_interior),
                    "v2_interior": int(v2_interior),
                    "v2_answer_prediction": v2_prediction,
                }
            )

    original_answer = classification_metrics(answer_truth, answer_b1)
    v2_answer = classification_metrics(answer_truth, answer_v2)
    nonfound_indices = [index for index, state in enumerate(answer_truth) if state != "FOUND"]
    legacy_negative_indices = [index for index, state in enumerate(answer_truth) if state in {"AMBIGUOUS", "ABSENT"}]
    b1_target = sum(row["b1_hit"] for row in paired_found)
    v2_target = sum(row["v2_hit"] for row in paired_found)
    b1_interior = sum(row["b1_interior"] for row in paired_found)
    v2_interior = sum(row["v2_interior"] for row in paired_found)
    both = sum(row["b1_hit"] and row["v2_hit"] for row in paired_found)
    b1_only = sum(row["b1_hit"] and not row["v2_hit"] for row in paired_found)
    v2_only = sum(not row["b1_hit"] and row["v2_hit"] for row in paired_found)
    neither = len(paired_found) - both - b1_only - v2_only
    joint_v2 = sum(row["v2_hit"] and row["v2_answer_prediction"] == "FOUND" for row in paired_found)

    archived_path = Path(args.archived_b1).resolve()
    archived = [row for row in read_jsonl(archived_path) if row.get("method") == "B1"]
    archived_by_id = {row["sample_id"]: row for row in archived}
    reproducibility = {
        "archived_samples": len(archived),
        "rerun_samples": len(b1_rows),
        "same_sample_ids": set(archived_by_id) == set(b1),
        "exact_raw_answer_matches": sum(archived_by_id[sid]["raw_answer"] == b1[sid]["raw_answer"] for sid in b1),
        "exact_pixel_point_matches": sum(archived_by_id[sid]["pixel_points_xy"] == b1[sid]["pixel_points_xy"] for sid in b1),
    }

    results = {
        "schema_version": 1,
        "scientific_status": "EXPLORATORY_DEV_ONLY",
        "paired_samples": 400,
        "found_samples_for_grounding": len(paired_found),
        "original_b1": {
            "description": "Frozen RoboRefer-2B-SFT with registered RGB-D, greedy decoding, forced point output",
            "answerability": original_answer,
            "found_point_in_target": ratio(b1_target, len(paired_found)),
            "found_point_in_interior": ratio(b1_interior, len(paired_found)),
            "false_found_ambiguous_or_absent": ratio(sum(answer_b1[i] == "FOUND" for i in legacy_negative_indices), len(legacy_negative_indices)),
            "false_found_all_nonfound": ratio(sum(answer_b1[i] == "FOUND" for i in nonfound_indices), len(nonfound_indices)),
            "mean_latency_ms": float(np.mean([row["latency_ms"] for row in b1_rows])),
            "parse_status_counts": dict(Counter(row["parse_status"] for row in b1_rows)),
        },
        "best_v2": {
            "description": "Best P-CRA-U V2 checkpoint, epoch 13, sidecar over frozen features",
            "answerability": v2_answer,
            "found_point_in_target": ratio(v2_target, len(paired_found)),
            "found_point_in_interior": ratio(v2_interior, len(paired_found)),
            "false_found_ambiguous_or_absent": ratio(sum(answer_v2[i] == "FOUND" for i in legacy_negative_indices), len(legacy_negative_indices)),
            "false_found_all_nonfound": ratio(sum(answer_v2[i] == "FOUND" for i in nonfound_indices), len(nonfound_indices)),
            "joint_true_found_prediction_and_point_in_target": ratio(joint_v2, len(paired_found)),
            "checkpoint_model_sha256": "1505152fca7b725d7db701c1341ad1a57049d6b5ac450a9f5ce0c8d78705ba26",
        },
        "delta_v2_minus_original": {
            "answerability_accuracy": v2_answer["accuracy"] - original_answer["accuracy"],
            "answerability_macro_f1": v2_answer["macro_f1"] - original_answer["macro_f1"],
            "found_point_in_target": v2_target / len(paired_found) - b1_target / len(paired_found),
            "found_point_in_interior": v2_interior / len(paired_found) - b1_interior / len(paired_found),
            "false_found_ambiguous_or_absent": (
                sum(answer_v2[i] == "FOUND" for i in legacy_negative_indices)
                - sum(answer_b1[i] == "FOUND" for i in legacy_negative_indices)
            ) / len(legacy_negative_indices),
            "false_found_all_nonfound": (
                sum(answer_v2[i] == "FOUND" for i in nonfound_indices)
                - sum(answer_b1[i] == "FOUND" for i in nonfound_indices)
            ) / len(nonfound_indices),
        },
        "paired_grounding_outcomes": {"both_hit": both, "original_only": b1_only, "v2_only": v2_only, "neither": neither},
        "found_grounding_by_variant": grouped_rates(paired_found, "variant"),
        "found_grounding_by_relation": grouped_rates(paired_found, "relation"),
        "family_cluster_bootstrap_target_delta": cluster_bootstrap(paired_found),
        "rerun_reproducibility_against_archived_b1": reproducibility,
        "scope_notes": [
            "Grounding comparison is restricted to the 168 truth-FOUND samples and scores the predicted point independently of answerability.",
            "Original RoboRefer has no answerability head; a successfully parsed point is therefore counted as FOUND.",
            "V2 latency is not compared because its saved evaluation uses cached frozen features and is not an end-to-end timing equivalent.",
            "Source uncertainty, relation-edge, calibration, and selective policy have no original-RoboRefer counterpart.",
            "Development-set results are exploratory and point-in-mask is not physical robot success.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison_b1_vs_best_v2.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ob, nv, delta = results["original_b1"], results["best_v2"], results["delta_v2_minus_original"]
    lines = [
        "# Original RoboRefer B1 vs best P-CRA-U V2",
        "",
        "Paired exploratory evaluation on the same 400 dev samples; grounding uses the 168 truth-FOUND samples.",
        "",
        "| Metric | Original B1 RGB-D | Best V2 | V2 - B1 |",
        "|---|---:|---:|---:|",
        f"| Answerability accuracy | {percentage(ob['answerability']['accuracy'])} | {percentage(nv['answerability']['accuracy'])} | {100*delta['answerability_accuracy']:+.2f} pp |",
        f"| Answerability macro-F1 | {percentage(ob['answerability']['macro_f1'])} | {percentage(nv['answerability']['macro_f1'])} | {100*delta['answerability_macro_f1']:+.2f} pp |",
        f"| Point in target (truth FOUND) | {percentage(ob['found_point_in_target']['rate'])} ({b1_target}/168) | {percentage(nv['found_point_in_target']['rate'])} ({v2_target}/168) | {100*delta['found_point_in_target']:+.2f} pp |",
        f"| Point in interior (truth FOUND) | {percentage(ob['found_point_in_interior']['rate'])} ({b1_interior}/168) | {percentage(nv['found_point_in_interior']['rate'])} ({v2_interior}/168) | {100*delta['found_point_in_interior']:+.2f} pp |",
        f"| False FOUND: AMBIGUOUS + ABSENT | {percentage(ob['false_found_ambiguous_or_absent']['rate'])} | {percentage(nv['false_found_ambiguous_or_absent']['rate'])} | {100*delta['false_found_ambiguous_or_absent']:+.2f} pp |",
        f"| False FOUND: all non-FOUND | {percentage(ob['false_found_all_nonfound']['rate'])} | {percentage(nv['false_found_all_nonfound']['rate'])} | {100*delta['false_found_all_nonfound']:+.2f} pp |",
        "",
        f"Paired grounding: both hit {both}, original-only {b1_only}, V2-only {v2_only}, neither {neither}.",
        f"Family-cluster bootstrap for V2-B1 target-hit delta: {100*results['family_cluster_bootstrap_target_delta']['estimate']:+.2f} pp "
        f"(95% CI {100*results['family_cluster_bootstrap_target_delta']['ci95_low']:+.2f} to {100*results['family_cluster_bootstrap_target_delta']['ci95_high']:+.2f} pp).",
        f"V2 joint true-FOUND classification and target hit: {joint_v2}/168 ({percentage(joint_v2/168)}).",
        "",
        "The original model always emits a point, so it preserves FOUND recall but cannot reject ambiguous, absent, or insufficient-evidence requests. V2's main gain is selective answerability and uncertainty handling; its localization gain is smaller. The tradeoff is that V2 abstains or rejects some genuinely answerable samples.",
        "",
        "These are exploratory dev results, not a held-out test or robot-success measurement.",
    ]
    markdown_path = output_dir / "comparison_b1_vs_best_v2.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "summary": results}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1", default=str(DEFAULT_B1))
    parser.add_argument("--v2", default=str(DEFAULT_V2))
    parser.add_argument("--truth", default=str(DEFAULT_TRUTH))
    parser.add_argument("--archived-b1", default=str(DEFAULT_ARCHIVED_B1))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
