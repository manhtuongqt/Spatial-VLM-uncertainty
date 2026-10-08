#!/usr/bin/env python3
"""Materialize the reviewed WP6 cases as a human-certified eval-only subset.

Accepted source targets and explicit human replacement clicks become final
evaluation labels. Ambiguous cases are quarantined. No output is eligible for
training or B2 supervision.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WP6 = ROOT / "results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge"
ERROR_DIR = WP6 / "error_review"
DEFAULT_OUT = ROOT / "datasets/D_tabletop_human_eval_v1"


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(success, total):
    if not total:
        return None
    z = 1.959963984540054
    p = success / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1-p) / total + z*z / (4*total*total)) / denominator
    return [max(0, center-half), min(1, center+half)]


def review_payload(decision):
    return {
        "reviewer": decision["reviewer"], "reviewed_at": decision["reviewed_at"],
        "review_origin": decision["review_origin"], "decision": decision["decision"],
        "correction_action": decision["correction_action"],
        "reference_frame": decision["reference_frame"], "notes": decision.get("notes", ""),
    }


def target_from(source_target, decision):
    action = decision["correction_action"]
    if action == "keep_source_target":
        return [float(source_target[0]), float(source_target[1])], False
    if action == "replace_source_target":
        if not decision.get("human_target_x") or not decision.get("human_target_y"):
            raise ValueError(f"{decision['sample_id']}: replacement target has no human click")
        point = [float(decision["human_target_x"]), float(decision["human_target_y"])]
        if not all(0 <= value <= 1 for value in point):
            raise ValueError(f"{decision['sample_id']}: human target out of bounds")
        return point, True
    raise ValueError(f"{decision['sample_id']}: cannot materialize action {action}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manual_queue_path = WP6 / "manual_tie_ambiguity_queue.jsonl"
    manual_decisions_path = WP6 / "manual_tie_ambiguity_decisions.csv"
    error_queue_path = ERROR_DIR / "wp6_error_review_queue.jsonl"
    error_decisions_path = ERROR_DIR / "wp6_error_review_decisions.csv"
    pilot_provenance_path = ROOT / "datasets/D_tabletop_machine_v1/provenance.jsonl"
    agreement_path = ROOT / "results/spatial_vlm_refspatial_v1/wp3_machine_lora_pilot/agreement_queue.jsonl"

    manual_queue = {row["sample_id"]: row for row in jsonl(manual_queue_path)}
    manual_decisions = {row["sample_id"]: row for row in csv_rows(manual_decisions_path)}
    error_queue = {row["sample_id"]: row for row in jsonl(error_queue_path)}
    error_decisions = {row["sample_id"]: row for row in csv_rows(error_decisions_path)}
    validation_errors = []
    if set(manual_queue) != set(manual_decisions):
        validation_errors.append("manual queue/decision IDs differ")
    if set(error_queue) != set(error_decisions):
        validation_errors.append("error queue/decision IDs differ")
    for group, decisions in (("manual", manual_decisions), ("error", error_decisions)):
        for sid, row in decisions.items():
            if row.get("completed") != "yes" or row.get("review_origin") != "human":
                validation_errors.append(f"{group}:{sid}: review is not completed human review")
            if row.get("reference_frame") != "image_viewer_left_to_right":
                validation_errors.append(f"{group}:{sid}: reference frame is not certified image-viewer frame")
    if validation_errors:
        raise RuntimeError("; ".join(validation_errors))

    evaluation, excluded, source_correct = [], [], 0
    for sid, decision in manual_decisions.items():
        source = manual_queue[sid]
        if decision["correction_action"] == "exclude_from_eval":
            excluded.append({**source, "exclusion_reason": decision["tie_or_ambiguity"],
                             "human_review": review_payload(decision)})
            continue
        target, corrected = target_from(source["target_xy"], decision)
        source_correct += not corrected
        evaluation.append({
            **source, "source_pool": "wp6_manual_tie_ambiguity_queue",
            "evaluation_role": "PRIMARY_SELECTION_FREE_HUMAN_CERTIFIED_STRESS",
            "source_target_xy": source["target_xy"], "target_xy": target,
            "target_provenance": "human_click" if corrected else "source_target_human_accepted",
            "human_corrected": corrected, "human_review": review_payload(decision),
            "challenge_role": "HUMAN_CERTIFIED_EVALUATION_ONLY",
            "label_status": "HUMAN_CERTIFIED_SINGLE_REVIEWER",
            "certification_scope": ["target_point", "horizontal_relation", "image_reference_frame"],
            "training_eligible": False, "b2_eligible": False,
        })

    for sid, decision in error_decisions.items():
        source = error_queue[sid]
        if decision["correction_action"] == "exclude_from_eval":
            excluded.append({**source, "exclusion_reason": decision["label_ambiguity"],
                             "human_review": review_payload(decision)})
            continue
        target, corrected = target_from(source["source_target_xy"], decision)
        source_correct += not corrected
        evaluation.append({
            **source, "source_pool": "wp6_primary_error_union",
            "evaluation_role": "DIAGNOSTIC_OUTCOME_SELECTED_ERROR_REVIEW",
            "source_target_xy": source["source_target_xy"], "target_xy": target,
            "target_provenance": "human_click" if corrected else "source_target_human_accepted",
            "human_corrected": corrected, "human_review": review_payload(decision),
            "challenge_role": "HUMAN_CERTIFIED_EVALUATION_ONLY",
            "label_status": "HUMAN_CERTIFIED_SINGLE_REVIEWER",
            "certification_scope": ["target_point", "horizontal_relation", "image_reference_frame"],
            "training_eligible": False, "b2_eligible": False,
        })

    evaluation.sort(key=lambda row: (row["split"], row["relation"], row["sample_id"]))
    excluded.sort(key=lambda row: row["sample_id"])
    families = [row["family_id"] for row in evaluation]
    if len(families) != len(set(families)):
        raise RuntimeError("Human evaluation subset contains duplicate families")
    pilot_families = {row["family_id"] for row in jsonl(pilot_provenance_path)}
    agreement_families = {row["family_id"] for row in jsonl(agreement_path)}
    pilot_overlap = sorted(set(families) & pilot_families)
    agreement_overlap = sorted(set(families) & agreement_families)
    if pilot_overlap or agreement_overlap:
        raise RuntimeError("Evaluation-family leakage into B1 dataset or B0 agreement queue")

    primary = [row for row in evaluation if row["evaluation_role"] == "PRIMARY_SELECTION_FREE_HUMAN_CERTIFIED_STRESS"]
    diagnostic = [row for row in evaluation if row["evaluation_role"] == "DIAGNOSTIC_OUTCOME_SELECTED_ERROR_REVIEW"]
    evaluation_path = args.output / "evaluation.jsonl"
    primary_path = args.output / "primary_evaluation.jsonl"
    diagnostic_path = args.output / "diagnostic_error_review.jsonl"
    excluded_path = args.output / "excluded.jsonl"
    evaluation_path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in evaluation))
    primary_path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in primary))
    diagnostic_path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in diagnostic))
    excluded_path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in excluded))

    reviewed_total = len(manual_decisions) + len(error_decisions)
    relation_counts = Counter(row["relation"] for row in evaluation)
    split_counts = Counter(row["split"] for row in evaluation)
    source_total = len(evaluation)
    precision = source_correct / source_total
    report = {
        "dataset": "D_tabletop_human_eval_v1",
        "status": "HUMAN_CERTIFIED_EVALUATION_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewed_total": reviewed_total, "evaluation_count": len(evaluation),
        "primary_selection_free_count": len(primary),
        "diagnostic_outcome_selected_count": len(diagnostic),
        "excluded_count": len(excluded), "unique_families": len(set(families)),
        "human_corrected_targets": sum(row["human_corrected"] for row in evaluation),
        "source_targets_human_accepted": source_correct,
        "source_target_precision_on_decidable_reviewed_cases": precision,
        "source_target_precision_wilson_ci95": wilson(source_correct, source_total),
        "by_relation": dict(relation_counts), "by_split": dict(split_counts),
        "reviewers": sorted({row["human_review"]["reviewer"] for row in evaluation}),
        "family_overlap_b1_pilot": len(pilot_overlap),
        "family_overlap_b0_agreement_queue": len(agreement_overlap),
        "sam2_used": False, "training_eligible": False, "b2_eligible": False,
        "g1_gate": {
            "status": "NOT_MET", "minimum_manual_reviews": 300,
            "actual_manual_reviews": reviewed_total, "minimum_reviews_per_kept_relation": 50,
            "actual_by_relation": dict(relation_counts),
            "anchor_precision": "N/A_NO_ANCHOR_LABELS_IN_THIS_RANKING_SUBSET",
            "reasons": ["107 < 300 reviewed scene-query quota", "leftmost/rightmost each have fewer than 50 retained reviews", "no audited anchor/reasoning supervision"],
        },
        "limitations": [
            "Single-reviewer certification; no inter-annotator agreement measurement.",
            "This subset certifies only target point, horizontal ranking relation and image-viewer reference frame.",
            "The subset is evaluation-only and cannot be used to train B1/B2.",
            "Only primary_evaluation.jsonl may produce the main B0/B1 stress metric; diagnostic_error_review.jsonl was selected from known errors and must be reported separately.",
        ],
    }
    report_path = args.output / "HUMAN_EVAL_AUDIT_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    md_path = args.output / "HUMAN_EVAL_AUDIT_REPORT.md"
    md_path.write_text("\n".join([
        "# D_tabletop_human_eval_v1", "",
        "Status: **human-certified, single-reviewer, evaluation-only**.", "",
        f"The release retains {len(evaluation)}/{reviewed_total} reviewed scene-query families, including "
        f"{sum(row['human_corrected'] for row in evaluation)} human-corrected target; {len(excluded)} ambiguous cases are quarantined.", "",
        f"For model comparison, `primary_evaluation.jsonl` contains {len(primary)} source-selection-free reviewed cases. "
        f"The {len(diagnostic)} known-error cases are isolated in `diagnostic_error_review.jsonl` and must not be mixed into the primary accuracy.", "",
        "| Relation | Retained |", "|---|---:|",
        *[f"| {name} | {count} |" for name, count in sorted(relation_counts.items())], "",
        f"Original SOURCE target precision among the {source_total} decidable retained reviews is "
        f"{precision:.2%} ({source_correct}/{source_total}); Wilson 95% CI "
        f"[{report['source_target_precision_wilson_ci95'][0]:.2%}, {report['source_target_precision_wilson_ci95'][1]:.2%}].", "",
        "There is zero family overlap with `D_tabletop_machine_v1` and the former B0 agreement queue. "
        "The release remains excluded from all training and B2 supervision.", "",
        "## Gate decision", "",
        "G1 is **not met**: only 107/300 required reviews are complete; leftmost/rightmost coverage is below 50 each; no anchor or reasoning supervision was certified. Do not create `D_tabletop_clean_v1` and do not open B2 from this release.", "",
    ]))

    manifest = {
        **report,
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in (
            manual_queue_path, manual_decisions_path, error_queue_path, error_decisions_path,
            pilot_provenance_path, agreement_path)},
        "outputs": {path.name: sha256(path) for path in (evaluation_path, primary_path, diagnostic_path, excluded_path, report_path, md_path)},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
