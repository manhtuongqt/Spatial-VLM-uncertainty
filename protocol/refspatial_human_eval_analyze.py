#!/usr/bin/env python3
"""Analyze B0/B1 on the selection-free human-certified RefSpatial stress set."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from refspatial_wp6_analyze import metrics, paired, fmt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets/D_tabletop_human_eval_v1/primary_evaluation.jsonl"
DEFAULT_EVAL = ROOT / "results/spatial_vlm_refspatial_v1/wp6_human_certified_eval/evaluation"


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--training-report", type=Path,
                        help="If set, require B1 to match this clean-adapter training report.")
    args = parser.parse_args()
    source = load(args.data)
    source_ids = {row["sample_id"] for row in source}
    if len(source) != len(source_ids) or not source:
        raise ValueError("Primary human evaluation manifest is empty or has duplicate IDs")
    if any(row.get("evaluation_role") != "PRIMARY_SELECTION_FREE_HUMAN_CERTIFIED_STRESS" for row in source):
        raise ValueError("Outcome-selected diagnostic examples leaked into primary evaluation")
    if any(row.get("label_status") != "HUMAN_CERTIFIED_SINGLE_REVIEWER" for row in source):
        raise ValueError("Primary manifest contains a non-certified label")

    model_rows = {}
    runs = {}
    for model in ("b0", "b1"):
        rows = load(args.eval_root / model / "predictions.jsonl")
        ids = {row["sample_id"] for row in rows}
        if len(rows) != len(source) or ids != source_ids:
            raise ValueError(f"{model}: prediction IDs do not exactly match the human primary manifest")
        run = json.loads((args.eval_root / model / "run.json").read_text())
        if run.get("status") != "COMPLETED" or run.get("targets_hidden_until_scoring") is not True:
            raise ValueError(f"{model}: run is incomplete or not blinded")
        runs[model] = run
        model_rows[model] = {row["sample_id"]: row for row in rows}

    if args.training_report:
        training = json.loads(args.training_report.read_text())
        if training.get("status") != "CLEAN_B1_GROUNDING_TRAINING_PASS":
            raise ValueError("B1 training artifact has not passed its release checks")
        expected_adapter_sha256 = training["artifacts"]["adapter_model.safetensors"]["sha256"]
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

    groups = {"all_primary": source_ids}
    for split in sorted({row["split"] for row in source}):
        groups[f"split:{split}"] = {row["sample_id"] for row in source if row["split"] == split}
    for relation in sorted({row["relation"] for row in source}):
        groups[f"relation:{relation}"] = {row["sample_id"] for row in source if row["relation"] == relation}
    for stratum in sorted({row["challenge_stratum"] for row in source}):
        groups[f"stratum:{stratum}"] = {row["sample_id"] for row in source if row["challenge_stratum"] == stratum}

    results = {}
    for name, ids in groups.items():
        b0 = {sid: model_rows["b0"][sid] for sid in ids}
        b1 = {sid: model_rows["b1"][sid] for sid in ids}
        results[name] = {"b0": metrics(b0.values()), "b1": metrics(b1.values()), "paired": paired(b0, b1)}
    primary = results["all_primary"]
    delta = primary["paired"]
    hit_ci = delta["family_cluster_bootstrap_hit_delta_95"]
    error_ci = delta["family_cluster_bootstrap_error_delta_95"]
    supported = hit_ci[0] > 0 or error_ci[1] < 0
    fixed = delta["b0_wrong_b1_right"]
    introduced = delta["b0_right_b1_wrong"]
    fixed_word = "error" if fixed == 1 else "errors"
    introduced_word = "error" if introduced == 1 else "errors"
    if supported:
        conclusion = (
            f"B1 shows a supported improvement signal on this stress subset "
            f"({fixed} B0 {fixed_word} fixed, {introduced} {introduced_word} introduced)."
        )
    else:
        conclusion = (
            f"B1 improvement is not supported on this stress subset "
            f"({fixed} B0 {fixed_word} fixed, {introduced} {introduced_word} introduced)."
        )
    error_ids = {
        model: sorted(sid for sid, row in rows.items() if not row["hit_at_008"])
        for model, rows in model_rows.items()
    }
    report = {
        "status": "COMPLETED_HUMAN_CERTIFIED_SELECTION_FREE_REFSPATIAL_STRESS_EVALUATION",
        "scope": "99-case RefSpatial source-quarantine stress subset reviewed before B0/B1 inference",
        "human_certification": "single_reviewer_target_horizontal_relation_image_frame",
        "primary_radius": 0.08,
        "groups": results, "error_sample_ids": error_ids,
        "training_eligible": False, "b2_eligible": False, "sam2_used": False,
        "gazebo_generalization_evidence": False,
        "b1_improvement_supported": supported,
        "conclusion": conclusion,
        "limitations": [
            "This is a deliberately source-quarantined RefSpatial stress subset, not a random population benchmark.",
            "Certification is from one reviewer; inter-annotator agreement is unavailable.",
            "Self-consistency is an uncertainty proxy, not a fitted calibrator.",
            "RefSpatial results cannot establish Gazebo transfer.",
        ],
    }
    metrics_path = args.eval_root / "B0_B1_HUMAN_EVAL_METRICS.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# B0 versus B1 on human-certified RefSpatial stress evaluation", "",
        "The 99 primary cases were source-quarantined without reading B0/B1 outputs, reviewed by a human, and frozen before this inference run. Known WP6 error-union cases are excluded from this primary metric.", "",
        "| Scope | Model | N | Parse | Hit@.08 | Mean error | Mean confidence | ECE | Brier | Error AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["all_primary"] + sorted(name for name in groups if name.startswith("split:")) + sorted(name for name in groups if name.startswith("relation:")) + sorted(name for name in groups if name.startswith("stratum:"))
    for name in order:
        for model in ("b0", "b1"):
            value = results[name][model]
            lines.append(f"| {name} | {model.upper()} | {value['samples']} | {fmt(value.get('parse_rate'))} | {fmt(value.get('hit_at_008'))} | {fmt(value.get('mean_point_error_all_invalid_as_sqrt2'), 6)} | {fmt(value.get('mean_confidence'))} | {fmt(value.get('ece_10bin'))} | {fmt(value.get('brier'))} | {fmt(value.get('error_detection_auroc'))} |")
    lines.extend([
        "", "## Paired result", "",
        f"- Hit@.08: B0 {primary['b0']['hit_at_008']:.2%} ({round(primary['b0']['hit_at_008']*len(source))}/{len(source)}), B1 {primary['b1']['hit_at_008']:.2%} ({round(primary['b1']['hit_at_008']*len(source))}/{len(source)}).",
        f"- Delta B1−B0: {delta['hit_at_008_delta_b1_minus_b0']:.4f}; family bootstrap 95% CI [{delta['family_cluster_bootstrap_hit_delta_95'][0]:.4f}, {delta['family_cluster_bootstrap_hit_delta_95'][1]:.4f}].",
        f"- B0 wrong/B1 right: {delta['b0_wrong_b1_right']}; B0 right/B1 wrong: {delta['b0_right_b1_wrong']}; McNemar exact p={delta['mcnemar_exact_two_sided_p']:.4f}.",
        f"- Mean-error delta B1−B0: {delta['mean_error_delta_b1_minus_b0_both_valid']:.6f}; bootstrap 95% CI [{delta['family_cluster_bootstrap_error_delta_95'][0]:.6f}, {delta['family_cluster_bootstrap_error_delta_95'][1]:.6f}].",
        "", conclusion, "",
        "## Boundaries", "",
        "This release is evaluation-only. The result does not satisfy G1, open B2, constitute calibrated uncertainty, or establish Gazebo generalization.", "",
    ])
    (args.eval_root / "B0_B1_HUMAN_EVAL_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"primary": primary, "paired": delta, "errors": error_ids}, indent=2))


if __name__ == "__main__":
    main()
