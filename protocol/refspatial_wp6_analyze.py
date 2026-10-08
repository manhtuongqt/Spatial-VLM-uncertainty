#!/usr/bin/env python3
"""Analyze the fixed WP6 primary challenge and its post-hoc B0-low-agreement subgroup."""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = ROOT / "results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge/evaluation"
RADIUS = 0.08


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def wilson(success, total, z=1.959963984540054):
    if not total:
        return [None, None]
    p = success / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    h = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [c - h, c + h]


def auroc(rows):
    positives = [r for r in rows if not r["hit"]]
    negatives = [r for r in rows if r["hit"]]
    if not positives or not negatives:
        return None
    return sum(1 if p["uncertainty"] > n["uncertainty"] else .5 if p["uncertainty"] == n["uncertainty"] else 0
               for p in positives for n in negatives) / (len(positives) * len(negatives))


def metrics(rows):
    normalized = []
    for row in rows:
        error = row.get("normalized_point_error")
        conf = float(row.get("self_consistency_confidence", 0))
        normalized.append({"error": error, "confidence": conf, "uncertainty": 1 - conf,
                           "hit": error is not None and error <= RADIUS})
    n = len(normalized)
    if not n:
        return {"samples": 0}
    valid = [r["error"] for r in normalized if r["error"] is not None]
    hits = sum(r["hit"] for r in normalized)
    bins, ece = [], 0.0
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        bucket = [r for r in normalized if lo <= r["confidence"] < hi or (i == 9 and r["confidence"] == 1)]
        if bucket:
            conf, accuracy = mean(r["confidence"] for r in bucket), mean(r["hit"] for r in bucket)
            ece += len(bucket) / n * abs(conf - accuracy)
            bins.append({"low": lo, "high": hi, "count": len(bucket), "mean_confidence": conf, "accuracy": accuracy})
    return {"samples": n, "valid_predictions": len(valid), "parse_rate": len(valid) / n,
            "hit_at_008": hits / n, "hit_at_008_wilson_95": wilson(hits, n),
            "mean_point_error_all_invalid_as_sqrt2": mean(r["error"] if r["error"] is not None else math.sqrt(2) for r in normalized),
            "mean_confidence": mean(r["confidence"] for r in normalized),
            "mean_predictive_uncertainty": mean(r["uncertainty"] for r in normalized),
            "ece_10bin": ece, "brier": mean((r["confidence"] - float(r["hit"])) ** 2 for r in normalized),
            "error_detection_auroc": auroc(normalized), "calibration_bins": bins}


def paired(b0, b1):
    ids = sorted(set(b0) & set(b1))
    pairs = [(b0[i], b1[i]) for i in ids]
    if not pairs:
        return {"paired_samples": 0}
    families = defaultdict(list)
    for x, y in pairs:
        families[x["family_id"]].append((x, y))
    rng = random.Random(9092026)
    draws, errors = [], []
    family_ids = sorted(families)
    for _ in range(10000):
        flat = [p for family in (families[rng.choice(family_ids)] for _ in family_ids) for p in family]
        draws.append(mean(float(y.get("normalized_point_error") is not None and y["normalized_point_error"] <= RADIUS)
                          - float(x.get("normalized_point_error") is not None and x["normalized_point_error"] <= RADIUS)
                          for x, y in flat))
        valid = [(x, y) for x, y in flat if x.get("normalized_point_error") is not None and y.get("normalized_point_error") is not None]
        errors.append(mean(y["normalized_point_error"] - x["normalized_point_error"] for x, y in valid) if valid else None)
    draws.sort()
    errors = sorted(x for x in errors if x is not None)
    valid = [(x, y) for x, y in pairs if x.get("normalized_point_error") is not None and y.get("normalized_point_error") is not None]
    b = sum((x.get("normalized_point_error") is None or x["normalized_point_error"] > RADIUS) and y.get("normalized_point_error") is not None and y["normalized_point_error"] <= RADIUS for x, y in pairs)
    c = sum(x.get("normalized_point_error") is not None and x["normalized_point_error"] <= RADIUS and (y.get("normalized_point_error") is None or y["normalized_point_error"] > RADIUS) for x, y in pairs)
    d = b + c
    p = min(1.0, 2 * sum(math.comb(d, k) for k in range(min(b, c) + 1)) / 2 ** d) if d else 1.0
    return {"paired_samples": len(pairs), "families": len(family_ids), "b0_wrong_b1_right": b, "b0_right_b1_wrong": c,
            "mcnemar_exact_two_sided_p": p,
            "hit_at_008_delta_b1_minus_b0": mean(float(y.get("normalized_point_error") is not None and y["normalized_point_error"] <= RADIUS) - float(x.get("normalized_point_error") is not None and x["normalized_point_error"] <= RADIUS) for x, y in pairs),
            "family_cluster_bootstrap_hit_delta_95": [draws[249], draws[9749]],
            "mean_error_delta_b1_minus_b0_both_valid": mean(y["normalized_point_error"] - x["normalized_point_error"] for x, y in valid),
            "family_cluster_bootstrap_error_delta_95": [errors[249], errors[9749]],
            "mean_confidence_delta_b1_minus_b0": mean(y["self_consistency_confidence"] - x["self_consistency_confidence"] for x, y in pairs)}


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    args = parser.parse_args()
    root = args.eval_root.resolve()
    model_rows = {}
    for model in ("b0", "b1"):
        rows = [json.loads(x) for x in (root / model / "predictions.jsonl").read_text().splitlines() if x]
        ids = {x["sample_id"] for x in rows}
        if len(rows) != 180 or len(ids) != 180:
            raise ValueError(f"{model} needs 180 unique WP6 predictions; got {len(rows)}/{len(ids)}")
        model_rows[model] = {x["sample_id"]: x for x in rows}
    # Defined after B0 is available: confidence <= .60 means at most 3/5 draws agree with B0 greedy point.
    low_ids = {sid for sid, row in model_rows["b0"].items() if row["self_consistency_confidence"] <= .60}
    groups = {"all_primary": set(model_rows["b0"]), "posthoc_b0_low_agreement_conf_le_060": low_ids}
    for split in ("dev", "diagnostic"):
        groups[split] = {sid for sid, row in model_rows["b0"].items() if row["split"] == split}
    for relation in sorted({x["relation"] for x in model_rows["b0"].values()}):
        groups[f"relation:{relation}"] = {sid for sid, row in model_rows["b0"].items() if row["relation"] == relation}
    for stratum in sorted({x.get("challenge_stratum", "unknown") for x in model_rows["b0"].values()}):
        groups[f"stratum:{stratum}"] = {sid for sid, row in model_rows["b0"].items() if row.get("challenge_stratum", "unknown") == stratum}
    report_groups = {}
    for name, ids in groups.items():
        a = {sid: model_rows["b0"][sid] for sid in ids}
        b = {sid: model_rows["b1"][sid] for sid in ids}
        report_groups[name] = {"b0": metrics(a.values()), "b1": metrics(b.values()), "paired": paired(a, b)}
    result = {"status": "COMPLETED_UNBIASED_SELECTION_MACHINE_SOURCE_EVALUATION", "scope": "WP6 fixed primary challenge",
              "not_for_training": True, "not_for_b2": True, "not_human_certified": True, "primary_radius": RADIUS,
              "confidence_definition": "fraction of five stochastic predictions within normalized L2 .08 of greedy point",
              "posthoc_stress_definition": "B0 self-consistency confidence <= .60; selected after B0 inference, not a primary comparison",
              "groups": report_groups,
              "limitations": ["Primary source labels are not human-certified.", "The low-agreement group is post-hoc and cannot establish an overall B1 benefit.", "Self-consistency is a predictive uncertainty proxy, not PCRAU calibration."]}
    (root / "B0_B1_CHALLENGE_METRICS.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# B0 versus B1 on WP6 fixed primary challenge", "", "The 180 primary cases were selected without RoboRefer outputs and are excluded from both the prior B0 agreement queue and B1 dataset. Labels remain machine-source-only; this is not B2 or Gazebo-transfer evidence.", "", "Primary metrics use normalized point error ≤ 0.08. Confidence is five-draw stochastic self-consistency around greedy output.", "", "| Scope | Model | N | Parse | Hit@.08 | Mean error | Mean conf. | ECE | Brier | Error AUROC |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["all_primary", "dev", "diagnostic"] + [x for x in groups if x.startswith("relation:")] + [x for x in groups if x.startswith("stratum:")]:
        for model in ("b0", "b1"):
            x = report_groups[name][model]
            lines.append(f"| {name} | {model.upper()} | {x['samples']} | {fmt(x['parse_rate'])} | {fmt(x['hit_at_008'])} | {fmt(x['mean_point_error_all_invalid_as_sqrt2'])} | {fmt(x['mean_confidence'])} | {fmt(x['ece_10bin'])} | {fmt(x['brier'])} | {fmt(x['error_detection_auroc'])} |")
    lines += ["", "## Primary paired delta", ""]
    x = report_groups["all_primary"]["paired"]
    lines += [f"- Hit@.08 delta B1−B0: **{fmt(x['hit_at_008_delta_b1_minus_b0'])}**, cluster-bootstrap 95% CI [{fmt(x['family_cluster_bootstrap_hit_delta_95'][0])}, {fmt(x['family_cluster_bootstrap_hit_delta_95'][1])}].",
              f"- B0 wrong/B1 right: {x['b0_wrong_b1_right']}; B0 right/B1 wrong: {x['b0_right_b1_wrong']}; McNemar exact p={fmt(x['mcnemar_exact_two_sided_p'], 4)}.",
              f"- Mean-error delta B1−B0: {fmt(x['mean_error_delta_b1_minus_b0_both_valid'], 6)}, cluster-bootstrap 95% CI [{fmt(x['family_cluster_bootstrap_error_delta_95'][0], 6)}, {fmt(x['family_cluster_bootstrap_error_delta_95'][1], 6)}].",
              f"- Confidence delta B1−B0: {fmt(x['mean_confidence_delta_b1_minus_b0'], 6)}.", "", "## Post-hoc B0 low-agreement stress subgroup", "", "This subgroup is defined only after B0 inference as confidence ≤0.60. It is descriptive and does not replace the full primary paired result.", ""]
    for model in ("b0", "b1"):
        x = report_groups["posthoc_b0_low_agreement_conf_le_060"][model]
        lines.append(f"- {model.upper()}: N={x['samples']}, Hit@.08={fmt(x.get('hit_at_008'))}, ECE={fmt(x.get('ece_10bin'))}, Brier={fmt(x.get('brier'))}.")
    (root / "B0_B1_CHALLENGE_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"primary": report_groups["all_primary"]["paired"], "low_agreement_n": len(low_ids)}, indent=2))


if __name__ == "__main__":
    main()
