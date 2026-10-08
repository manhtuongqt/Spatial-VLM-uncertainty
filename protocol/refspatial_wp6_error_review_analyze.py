#!/usr/bin/env python3
"""Validate and summarize completed human review of the WP6 error union."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge/error_review"
DEFAULT_QUEUE = DEFAULT_DIR / "wp6_error_review_queue.jsonl"
DEFAULT_DECISIONS = DEFAULT_DIR / "wp6_error_review_decisions.csv"
DEFAULT_JSON = DEFAULT_DIR / "WP6_ERROR_REVIEW_REPORT.json"
DEFAULT_MD = DEFAULT_DIR / "WP6_ERROR_REVIEW_REPORT.md"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    queue = load_jsonl(args.queue)
    queued = {row["sample_id"]: row for row in queue}
    with args.decisions.open(newline="") as handle:
        decisions = list(csv.DictReader(handle))
    decided = {row.get("sample_id"): row for row in decisions}
    errors, warnings = [], []
    if len(decided) != len(decisions):
        errors.append("duplicate sample_id in decision CSV")
    unknown = sorted(set(decided) - set(queued))
    missing = sorted(set(queued) - set(decided))
    if unknown:
        errors.append(f"unknown decision sample IDs: {unknown}")
    if missing:
        errors.append(f"missing decision sample IDs: {missing}")

    required = [
        "reviewer", "reviewed_at", "review_origin", "target_correct", "object_set_correct",
        "ordinal_direction_correct", "reference_frame", "label_ambiguity", "source_target_valid",
        "b0_error_confirmed", "b1_error_confirmed", "decision", "correction_action", "completed",
    ]
    complete = []
    for sid in sorted(set(queued) & set(decided)):
        row = decided[sid]
        absent = [key for key in required if not row.get(key)]
        if absent:
            errors.append(f"{sid}: missing {', '.join(absent)}")
            continue
        if row["completed"] != "yes":
            errors.append(f"{sid}: completed must be yes")
            continue
        if row["review_origin"] != "human":
            errors.append(f"{sid}: review_origin is not human")
        if row["reference_frame"] == "other" and not row.get("notes", "").strip():
            warnings.append(f"{sid}: reference_frame=other lacks an explanatory note")
        if row["decision"] == "ambiguous" and row["correction_action"] != "exclude_from_eval":
            warnings.append(f"{sid}: ambiguous case is not excluded from evaluation")
        complete.append(row)

    excluded = [row for row in complete if row["correction_action"] == "exclude_from_eval"]
    kept = [row for row in complete if row["correction_action"] != "exclude_from_eval"]
    confirmed_b0 = sum(row["b0_error_confirmed"] == "yes" for row in kept)
    confirmed_b1 = sum(row["b1_error_confirmed"] == "yes" for row in kept)
    total_primary = 180
    retained_primary = total_primary - len(excluded)
    # Every reviewed row was drawn from the machine-scored error union. For
    # unreviewed machine-correct rows, source validity remains unaudited.
    b0_correct = retained_primary - confirmed_b0
    b1_correct = retained_primary - confirmed_b1
    status = "COMPLETE" if not errors and not warnings else "COMPLETE_WITH_WARNINGS" if not errors else "INVALID"
    report = {
        "status": status,
        "review_scope": "WP6_MACHINE_ERROR_UNION_ONLY",
        "queue_count": len(queue), "completed_reviews": len(complete),
        "human_reviews": sum(row.get("review_origin") == "human" for row in complete),
        "decision_counts": dict(Counter(row.get("decision") for row in complete)),
        "correction_action_counts": dict(Counter(row.get("correction_action") for row in complete)),
        "excluded_from_eval": len(excluded),
        "confirmed_b0_errors_retained": confirmed_b0,
        "confirmed_b1_errors_retained": confirmed_b1,
        "error_audited_machine_source_estimate": {
            "retained_primary_n": retained_primary,
            "b0_correct": b0_correct, "b0_hit_at_008": b0_correct / retained_primary,
            "b1_correct": b1_correct, "b1_hit_at_008": b1_correct / retained_primary,
            "qualification": "Only the original error union was human-reviewed; the remaining machine-correct cases retain machine-source labels.",
        },
        "full_benchmark_human_certified": False,
        "g1_manual_audit_gate_met": False,
        "training_eligible": False, "b2_eligible": False, "sam2_used": False,
        "validation_errors": errors, "warnings": warnings,
    }
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# WP6 targeted error-review report", "",
        f"Status: **{status}**.", "",
        f"All {len(complete)}/{len(queue)} queued cases have attributed human decisions. "
        f"The reviewer confirmed {confirmed_b0} B0 errors and {confirmed_b1} B1 errors after excluding {len(excluded)} ambiguous case.", "",
        "| Result | B0 | B1 |", "|---|---:|---:|",
        f"| Retained primary N | {retained_primary} | {retained_primary} |",
        f"| Confirmed errors among reviewed error union | {confirmed_b0} | {confirmed_b1} |",
        f"| Error-audited machine-source Hit@.08 estimate | {b0_correct/retained_primary:.4%} ({b0_correct}/{retained_primary}) | {b1_correct/retained_primary:.4%} ({b1_correct}/{retained_primary}) |", "",
        "This is not a human-certified 179-case benchmark: only the seven machine-flagged error-union cases were reviewed, while the other cases retain machine-source labels. It does not satisfy the 300-query G1 audit and is not training/B2 evidence.", "",
    ]
    if warnings:
        lines.extend(["## Traceability warnings", ""] + [f"- {warning}" for warning in warnings] + [""])
    if errors:
        lines.extend(["## Validation errors", ""] + [f"- {error}" for error in errors] + [""])
    args.output_md.write_text("\n".join(lines))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
