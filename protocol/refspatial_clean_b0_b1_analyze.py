#!/usr/bin/env python3
"""Analyze paired B0/B1 results on D_tabletop_clean_v1 held-out families."""

import argparse
import json
from pathlib import Path

from refspatial_wp6_analyze import fmt, metrics, paired


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets/D_tabletop_clean_v1/provenance.jsonl"
DEFAULT_EVAL = ROOT / "results/spatial_vlm_refspatial_v1/wp7_b1_clean_grounding/clean_eval"
DEFAULT_TRAINING_REPORT = ROOT / "results/spatial_vlm_refspatial_v1/wp7_b1_clean_grounding/TRAINING_REPORT.json"


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--training-report", type=Path, default=DEFAULT_TRAINING_REPORT)
    args = parser.parse_args()

    source = [row for row in load(args.data) if row["split"] in {"dev", "diagnostic"}]
    source_ids = {row["sample_id"] for row in source}
    if not source or len(source) != len(source_ids):
        raise ValueError("Clean held-out manifest is empty or has duplicate IDs")
    if any(row.get("label_certification") != "INDIVIDUAL_HUMAN_REVIEW_ACCEPT" for row in source):
        raise ValueError("Held-out manifest contains a non-human-certified label")
    if any(row.get("family_id") is None or row.get("b2_eligible") is not False for row in source):
        raise ValueError("Held-out manifest violates family/B2 scope")

    training = json.loads(args.training_report.read_text())
    if training.get("status") != "CLEAN_B1_GROUNDING_TRAINING_PASS":
        raise ValueError("B1 training artifact has not passed its release checks")
    expected_adapter_sha256 = training["artifacts"]["adapter_model.safetensors"]["sha256"]

    model_rows = {}
    runs = {}
    for model in ("b0", "b1"):
        rows = load(args.eval_root / model / "predictions.jsonl")
        ids = {row["sample_id"] for row in rows}
        if len(rows) != len(source) or ids != source_ids:
            raise ValueError(f"{model}: predictions do not exactly match held-out manifest")
        run = json.loads((args.eval_root / model / "run.json").read_text())
        if run.get("status") != "COMPLETED" or run.get("targets_hidden_until_scoring") is not True:
            raise ValueError(f"{model}: run incomplete or not blinded")
        runs[model] = run
        model_rows[model] = {row["sample_id"]: row for row in rows}

    controlled_keys = [
        "provenance_sha256", "samples_requested", "uncertainty_draws",
        "sampling_temperature", "sampling_top_p", "agreement_radius",
        "max_new_tokens", "seed_scheme", "evaluator_sha256",
    ]
    mismatch = [key for key in controlled_keys if runs["b0"].get(key) != runs["b1"].get(key)]
    if mismatch:
        raise ValueError(f"B0/B1 evaluation contract mismatch: {mismatch}")
    if runs["b0"].get("adapter_sha256") is not None:
        raise ValueError("B0 unexpectedly records an adapter")
    if runs["b1"].get("adapter_sha256") != expected_adapter_sha256:
        raise ValueError("B1 evaluation does not use the released clean adapter")

    groups = {"all": source_ids}
    for split in ("dev", "diagnostic"):
        groups[f"split:{split}"] = {row["sample_id"] for row in source if row["split"] == split}
    for relation in sorted({row["relation"] for row in source}):
        groups[f"relation:{relation}"] = {row["sample_id"] for row in source if row["relation"] == relation}

    results = {}
    for name, ids in groups.items():
        b0 = {sid: model_rows["b0"][sid] for sid in ids}
        b1 = {sid: model_rows["b1"][sid] for sid in ids}
        results[name] = {"b0": metrics(b0.values()), "b1": metrics(b1.values()), "paired": paired(b0, b1)}
    primary = results["all"]
    delta = primary["paired"]
    hit_ci = delta["family_cluster_bootstrap_hit_delta_95"]
    error_ci = delta["family_cluster_bootstrap_error_delta_95"]
    supported = hit_ci[0] > 0 or error_ci[1] < 0
    conclusion = (
        "B1 shows an internal held-out improvement signal; Gazebo evidence is still required."
        if supported else
        "B1 improvement is not supported on this small internal human-certified held-out set."
    )
    error_ids = {
        model: sorted(sid for sid, row in rows.items() if not row["hit_at_008"])
        for model, rows in model_rows.items()
    }
    report = {
        "status": "COMPLETED_CLEAN_B1_INTERNAL_PAIRED_EVALUATION",
        "scope": "D_tabletop_clean_v1 individually human-reviewed dev+diagnostic families",
        "samples": len(source),
        "groups": results,
        "error_sample_ids": error_ids,
        "b1_internal_improvement_supported": supported,
        "conclusion": conclusion,
        "gazebo_generalization_evidence": False,
        "b2_open": False,
        "sam2_used": False,
        "limitations": [
            "The held-out set is small and was used for dataset audit, so it is internal diagnostic evidence.",
            "RefSpatial may have been seen during RoboRefer pretraining.",
            "Self-consistency is an uncertainty proxy, not a calibrated PCRAU score.",
            "Only direct horizontal ranking is in scope; ranking is not pairwise left/right.",
        ],
    }
    args.eval_root.mkdir(parents=True, exist_ok=True)
    (args.eval_root / "B0_B1_CLEAN_METRICS.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# B0 versus clean-grounding B1 — internal paired evaluation", "",
        f"Status: **{report['status']}**. {conclusion}", "",
        "All held-out labels are individually accepted human reviews and family-disjoint from train and `D_tabletop_human_eval_v1`. Primary correctness is normalized point error ≤ 0.08.", "",
        "| Scope | Model | N | Parse | Hit@.08 | Mean error | Mean confidence | ECE | Brier | Error AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["all", "split:dev", "split:diagnostic"] + sorted(name for name in groups if name.startswith("relation:"))
    for name in order:
        for model in ("b0", "b1"):
            value = results[name][model]
            lines.append(
                f"| {name} | {model.upper()} | {value['samples']} | {fmt(value.get('parse_rate'))} | "
                f"{fmt(value.get('hit_at_008'))} | {fmt(value.get('mean_point_error_all_invalid_as_sqrt2'), 6)} | "
                f"{fmt(value.get('mean_confidence'))} | {fmt(value.get('ece_10bin'))} | "
                f"{fmt(value.get('brier'))} | {fmt(value.get('error_detection_auroc'))} |"
            )
    lines += [
        "", "## Paired result", "",
        f"- Hit@.08 delta B1−B0: {delta['hit_at_008_delta_b1_minus_b0']:.4f}; family-bootstrap 95% CI [{hit_ci[0]:.4f}, {hit_ci[1]:.4f}].",
        f"- B0 wrong/B1 right: {delta['b0_wrong_b1_right']}; B0 right/B1 wrong: {delta['b0_right_b1_wrong']}; McNemar exact p={delta['mcnemar_exact_two_sided_p']:.4f}.",
        f"- Mean-error delta B1−B0: {delta['mean_error_delta_b1_minus_b0_both_valid']:.6f}; bootstrap 95% CI [{error_ci[0]:.6f}, {error_ci[1]:.6f}].",
        "", "## Boundary", "",
        "This is internal RefSpatial evidence only. It does not open B2, validate calibrated uncertainty, select a model on final tests, or establish Gazebo transfer. No SAM2 was used.", "",
    ]
    (args.eval_root / "B0_B1_CLEAN_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"status": report["status"], "primary": primary, "errors": error_ids, "conclusion": conclusion}, indent=2))


if __name__ == "__main__":
    main()
