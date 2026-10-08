#!/usr/bin/env python3
"""Summarize paired B0/B1 machine-pilot point and self-consistency results."""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results/spatial_vlm_refspatial_v1/wp5_b0_b1_machine_eval"
RADII = (0.05, 0.08, 0.10)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def median(values):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def wilson(success, total, z=1.959963984540054):
    if not total:
        return [None, None]
    p = success / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def error_auroc(rows):
    positive = [r for r in rows if not r["hit"]]
    negative = [r for r in rows if r["hit"]]
    if not positive or not negative:
        return None
    score = 0.0
    for pos in positive:
        for neg in negative:
            a, b = pos["uncertainty"], neg["uncertainty"]
            score += 1 if a > b else 0.5 if a == b else 0
    return score / (len(positive) * len(negative))


def metrics(rows):
    normalized = []
    for row in rows:
        error = row.get("normalized_point_error")
        confidence = float(row.get("self_consistency_confidence", 0))
        normalized.append({"error": error, "hit": error is not None and error <= 0.08,
                           "confidence": confidence, "uncertainty": 1 - confidence})
    n = len(normalized)
    errors = [r["error"] for r in normalized if r["error"] is not None]
    hit_count = sum(r["hit"] for r in normalized)
    bins = []
    ece = 0.0
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        bucket = [r for r in normalized if low <= r["confidence"] < high or index == 9 and r["confidence"] == 1]
        if bucket:
            conf = mean(r["confidence"] for r in bucket)
            accuracy = mean(r["hit"] for r in bucket)
            ece += len(bucket) / n * abs(conf - accuracy)
            bins.append({"low": low, "high": high, "count": len(bucket),
                         "mean_confidence": conf, "accuracy": accuracy})
    ranked = sorted(normalized, key=lambda r: (-r["confidence"], r["uncertainty"]))
    cumulative_errors = 0
    risk_curve = []
    risk_sum = 0.0
    for index, row in enumerate(ranked, 1):
        cumulative_errors += not row["hit"]
        risk = cumulative_errors / index
        risk_sum += risk
        if index in {max(1, round(n * x)) for x in (0.25, 0.5, 0.75, 1.0)}:
            risk_curve.append({"coverage": index / n, "risk": risk})
    return {
        "samples": n,
        "valid_predictions": len(errors),
        "parse_rate": len(errors) / n,
        "hit_at_005": sum(e <= 0.05 for e in errors) / n,
        "hit_at_008": hit_count / n,
        "hit_at_008_wilson_95": wilson(hit_count, n),
        "hit_at_010": sum(e <= 0.10 for e in errors) / n,
        "mean_point_error_all_invalid_as_sqrt2": mean(e if e is not None else math.sqrt(2) for e in (r["error"] for r in normalized)),
        "mean_point_error_valid": mean(errors),
        "median_point_error_valid": median(errors),
        "mean_confidence": mean(r["confidence"] for r in normalized),
        "mean_predictive_uncertainty": mean(r["uncertainty"] for r in normalized),
        "ece_10bin": ece,
        "brier": mean((r["confidence"] - float(r["hit"])) ** 2 for r in normalized),
        "error_detection_auroc": error_auroc(normalized),
        "aurc_discrete": risk_sum / n,
        "risk_coverage": risk_curve,
        "calibration_bins": bins,
        "mean_stochastic_dispersion": mean(r.get("stochastic_dispersion_from_greedy") for r in rows
                                              if r.get("stochastic_dispersion_from_greedy") is not None),
    }


def paired_stats(b0, b1):
    ids = sorted(set(b0) & set(b1))
    pairs = [(b0[sid], b1[sid]) for sid in ids]
    b = sum((a.get("normalized_point_error") is None or a["normalized_point_error"] > .08)
            and z.get("normalized_point_error") is not None and z["normalized_point_error"] <= .08 for a, z in pairs)
    c = sum(a.get("normalized_point_error") is not None and a["normalized_point_error"] <= .08
            and (z.get("normalized_point_error") is None or z["normalized_point_error"] > .08) for a, z in pairs)
    discordant = b + c
    if discordant:
        lower_tail = sum(math.comb(discordant, k) for k in range(0, min(b, c) + 1)) / 2**discordant
        mcnemar_p = min(1.0, 2 * lower_tail)
    else:
        mcnemar_p = 1.0
    families = defaultdict(list)
    for a, z in pairs:
        families[a["family_id"]].append((a, z))
    family_ids = sorted(families)
    rng = random.Random(9092026)
    deltas = []
    error_deltas = []
    for _ in range(10000):
        sampled = [families[rng.choice(family_ids)] for _ in family_ids]
        flat = [pair for group in sampled for pair in group]
        deltas.append(mean(float(z.get("normalized_point_error") is not None and z["normalized_point_error"] <= .08)
                           - float(a.get("normalized_point_error") is not None and a["normalized_point_error"] <= .08)
                           for a, z in flat))
        valid_flat = [(a, z) for a, z in flat if a.get("normalized_point_error") is not None
                      and z.get("normalized_point_error") is not None]
        error_deltas.append(mean(z["normalized_point_error"] - a["normalized_point_error"]
                                 for a, z in valid_flat))
    deltas.sort()
    error_deltas.sort()
    both_valid = [(a["normalized_point_error"], z["normalized_point_error"]) for a, z in pairs
                  if a.get("normalized_point_error") is not None and z.get("normalized_point_error") is not None]
    return {
        "paired_samples": len(pairs), "families": len(family_ids),
        "b0_wrong_b1_right": b, "b0_right_b1_wrong": c,
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "hit_at_008_delta_b1_minus_b0": mean(float(z.get("normalized_point_error") is not None and z["normalized_point_error"] <= .08)
                                               - float(a.get("normalized_point_error") is not None and a["normalized_point_error"] <= .08)
                                               for a, z in pairs),
        "family_cluster_bootstrap_delta_95": [deltas[249], deltas[9749]],
        "mean_error_delta_b1_minus_b0_both_valid": mean(z - a for a, z in both_valid),
        "family_cluster_bootstrap_mean_error_delta_95": [error_deltas[249], error_deltas[9749]],
        "mean_confidence_delta_b1_minus_b0": mean(z["self_consistency_confidence"] - a["self_consistency_confidence"]
                                                   for a, z in pairs),
        "mean_dispersion_delta_b1_minus_b0": mean(
            z["stochastic_dispersion_from_greedy"] - a["stochastic_dispersion_from_greedy"]
            for a, z in pairs if a.get("stochastic_dispersion_from_greedy") is not None
            and z.get("stochastic_dispersion_from_greedy") is not None),
    }


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def main():
    models = {}
    for model_id in ("b0", "b1"):
        path = EVAL / model_id / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        by_id = {row["sample_id"]: row for row in rows}
        if len(rows) != 180 or len(by_id) != 180:
            raise ValueError(f"{model_id} needs exactly 180 unique predictions; got {len(rows)}/{len(by_id)}")
        sections = {"all": metrics(rows)}
        for split in ("dev", "diagnostic"):
            sections[split] = metrics([row for row in rows if row["split"] == split])
        sections["by_relation"] = {}
        for relation in sorted({row["relation"] for row in rows}):
            sections["by_relation"][relation] = metrics([row for row in rows if row["relation"] == relation])
        sections["by_split_relation"] = {}
        for split in ("dev", "diagnostic"):
            for relation in sorted({row["relation"] for row in rows}):
                sections["by_split_relation"][f"{split}:{relation}"] = metrics(
                    [row for row in rows if row["split"] == split and row["relation"] == relation])
        models[model_id] = {"rows": by_id, "metrics": sections}
    paired = paired_stats(models["b0"]["rows"], models["b1"]["rows"])
    paired_by_split = {}
    for split in ("dev", "diagnostic"):
        paired_by_split[split] = paired_stats(
            {sid: row for sid, row in models["b0"]["rows"].items() if row["split"] == split},
            {sid: row for sid, row in models["b1"]["rows"].items() if row["split"] == split})
    paired_by_relation = {}
    for relation in models["b0"]["metrics"]["by_relation"]:
        paired_by_relation[relation] = paired_stats(
            {sid: row for sid, row in models["b0"]["rows"].items() if row["relation"] == relation},
            {sid: row for sid, row in models["b1"]["rows"].items() if row["relation"] == relation})
    result = {
        "status": "COMPLETED_MACHINE_FILTERED_INTERNAL_EVALUATION",
        "scope": "REFSPATIAL_MACHINE_PILOT_DEV_DIAGNOSTIC",
        "not_for_b2": True,
        "gazebo_promotion_gate": "HOLD_FOR_UNBIASED_CHALLENGE_EVALUATION",
        "b1_improvement_claim_supported": False,
        "primary_radius": 0.08,
        "confidence_definition": "fraction of 5 stochastic predictions within normalized L2 0.08 of greedy prediction",
        "uncertainty_definition": "1 - self-consistency confidence",
        "calibration_target": "greedy hit at normalized L2 <= 0.08 against machine-filtered source point",
        "models": {key: value["metrics"] for key, value in models.items()},
        "paired": paired,
        "paired_by_split": paired_by_split,
        "paired_by_relation": paired_by_relation,
        "limitations": [
            "Machine-filtered RefSpatial labels are not human-certified ground truth.",
            "Self-consistency is a predictive uncertainty proxy and is not PCRAU calibration.",
            "This internal result does not establish Gazebo transfer or B2 eligibility.",
        ],
    }
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "B0_B1_METRICS.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# B0 versus B1 machine-pilot evaluation", "",
             "Status: **completed internal machine-filtered evaluation**. B2 remains closed. Gazebo promotion gate: **HOLD** pending an evaluation set that was not selected using B0 agreement.", "",
             "Primary correctness is normalized point error ≤ 0.08. Confidence is the fraction of five stochastic outputs within 0.08 of the greedy output; uncertainty is one minus this confidence.", "",
             "## Overall and split results", "",
             "| Scope | Model | N | Parse | Hit@.05 | Hit@.08 | Hit@.10 | Mean error | ECE | Brier | Error AUROC | AURC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scope in ("all", "dev", "diagnostic"):
        for model_id in ("b0", "b1"):
            m = models[model_id]["metrics"][scope]
            lines.append(f"| {scope} | {model_id.upper()} | {m['samples']} | {fmt(m['parse_rate'])} | {fmt(m['hit_at_005'])} | {fmt(m['hit_at_008'])} | {fmt(m['hit_at_010'])} | {fmt(m['mean_point_error_all_invalid_as_sqrt2'])} | {fmt(m['ece_10bin'])} | {fmt(m['brier'])} | {fmt(m['error_detection_auroc'])} | {fmt(m['aurc_discrete'])} |")
    lines += ["", "## Relation results", "",
              "| Relation | Model | N | Hit@.08 | Mean error | ECE | Brier | Error AUROC |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for relation in models["b0"]["metrics"]["by_relation"]:
        for model_id in ("b0", "b1"):
            m = models[model_id]["metrics"]["by_relation"][relation]
            lines.append(f"| {relation} | {model_id.upper()} | {m['samples']} | {fmt(m['hit_at_008'])} | {fmt(m['mean_point_error_all_invalid_as_sqrt2'])} | {fmt(m['ece_10bin'])} | {fmt(m['brier'])} | {fmt(m['error_detection_auroc'])} |")
    lines += ["", "## Paired comparison", "",
              f"- Hit@.08 delta B1−B0: **{fmt(paired['hit_at_008_delta_b1_minus_b0'])}**.",
              f"- Family-cluster bootstrap 95% interval: [{fmt(paired['family_cluster_bootstrap_delta_95'][0])}, {fmt(paired['family_cluster_bootstrap_delta_95'][1])}].",
              f"- B0 wrong/B1 right: {paired['b0_wrong_b1_right']}; B0 right/B1 wrong: {paired['b0_right_b1_wrong']}; exact McNemar p={fmt(paired['mcnemar_exact_two_sided_p'], 4)}.",
              f"- Mean point-error delta on pairs valid for both: {fmt(paired['mean_error_delta_b1_minus_b0_both_valid'], 6)}.",
              f"- Family-cluster bootstrap 95% interval for mean-error delta: [{fmt(paired['family_cluster_bootstrap_mean_error_delta_95'][0], 6)}, {fmt(paired['family_cluster_bootstrap_mean_error_delta_95'][1], 6)}].",
              f"- Mean confidence delta B1−B0: {fmt(paired['mean_confidence_delta_b1_minus_b0'], 6)}; mean stochastic-dispersion delta: {fmt(paired['mean_dispersion_delta_b1_minus_b0'], 6)}.",
              "", "## Paired deltas by split", "",
              "| Split | Hit@.08 delta | Mean-error delta | Error-delta bootstrap 95% | Confidence delta | Dispersion delta |",
              "|---|---:|---:|---:|---:|---:|"]
    for split, values in paired_by_split.items():
        interval = values["family_cluster_bootstrap_mean_error_delta_95"]
        lines.append(f"| {split} | {fmt(values['hit_at_008_delta_b1_minus_b0'])} | {fmt(values['mean_error_delta_b1_minus_b0_both_valid'], 6)} | [{fmt(interval[0], 6)}, {fmt(interval[1], 6)}] | {fmt(values['mean_confidence_delta_b1_minus_b0'], 6)} | {fmt(values['mean_dispersion_delta_b1_minus_b0'], 6)} |")
    lines += ["", "## Paired deltas by relation", "",
              "| Relation | Hit@.08 delta | Mean-error delta | Error-delta bootstrap 95% | Confidence delta | Dispersion delta |",
              "|---|---:|---:|---:|---:|---:|"]
    for relation, values in paired_by_relation.items():
        interval = values["family_cluster_bootstrap_mean_error_delta_95"]
        lines.append(f"| {relation} | {fmt(values['hit_at_008_delta_b1_minus_b0'])} | {fmt(values['mean_error_delta_b1_minus_b0_both_valid'], 6)} | [{fmt(interval[0], 6)}, {fmt(interval[1], 6)}] | {fmt(values['mean_confidence_delta_b1_minus_b0'], 6)} | {fmt(values['mean_dispersion_delta_b1_minus_b0'], 6)} |")
    lines += [
              "", "## Boundary", "",
              "These labels were machine-filtered and were selected partly using B0 agreement, so the absolute B0 result is optimistically biased. The confidence score is stochastic self-consistency, not a fitted PCRAU calibrator. Error-detection AUROC is unavailable because neither model made a hit@.08 error. Lower ECE/Brier here only means confidence moved closer to the saturated 100% accuracy; it does not demonstrate useful uncertainty discrimination.", "",
              "The current evidence does not support a B1 improvement claim. Keep B1 as an artifact and evaluate it on an untouched challenge set containing errors and ambiguous cases before choosing it for Gazebo. This result cannot open B2 or establish Gazebo transfer."]
    (EVAL / "B0_B1_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
