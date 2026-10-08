"""Validate a returned blind G1 audit CSV without certifying the G1 gate."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .audit_development import ROOT, STATES, sha256
from .build_label_audit import DEFAULT_OUTPUT, SOURCE_CLASSES


RELATION_LABELS = {"leftmost", "rightmost", "second_from_left", "second_from_right", "OUT_OF_SCOPE", "UNSURE"}
SOURCE_LABELS = set(SOURCE_CLASSES) | {"NONE", "MULTIPLE", "UNSURE"}
SOURCE_CANONICAL = dict(zip(SOURCE_CLASSES, ("SEMANTIC", "RELATION", "SPATIAL", "DEPTH", "OCCLUSION")))
FIELDS = {
    "review_relation": RELATION_LABELS,
    "review_answerability": set(STATES) | {"UNSURE"},
    "review_target_correct": {"yes", "no", "unsure", "not_visible"},
    "review_reasoning_depth": {"0", "1", "2", "UNSURE"},
    "review_source": SOURCE_LABELS,
    "review_single_source": {"yes", "no", "unsure"},
    "review_status": {"NOT_REVIEWED", "IN_PROGRESS", "DONE", "UNSURE"},
}


def load_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_rows(queue_rows, reviewed_rows):
    queue = {r["queue_id"]: r for r in queue_rows}
    if len(queue) != len(queue_rows):
        raise ValueError("Duplicate queue IDs")
    seen = set()
    issues = []
    done = []
    for line, row in enumerate(reviewed_rows, 2):
        qid = row.get("queue_id", "")
        if qid in seen:
            issues.append(f"line {line}: duplicate queue_id {qid}")
            continue
        seen.add(qid)
        expected = queue.get(qid)
        if expected is None:
            issues.append(f"line {line}: unknown queue_id {qid}")
            continue
        for key in ("dataset", "sample_id", "family_id"):
            if row.get(key) != expected[key]:
                issues.append(f"line {line}: {key} differs from locked queue")
        for key, allowed in FIELDS.items():
            value = (row.get(key) or "").strip()
            if value and value not in allowed:
                issues.append(f"line {line}: invalid {key}={value!r}")
        status = (row.get("review_status") or "").strip()
        if status == "DONE":
            if not (row.get("reviewer_id") or "").strip():
                issues.append(f"line {line}: DONE without reviewer_id")
            for key in ("review_relation", "review_answerability", "review_source", "review_single_source"):
                if not (row.get(key) or "").strip():
                    issues.append(f"line {line}: DONE missing {key}")
            if expected["candidate_target_uv"] and not (row.get("review_target_correct") or "").strip():
                issues.append(f"line {line}: DONE missing target correctness")
            source = row.get("review_source", "")
            single = row.get("review_single_source", "")
            if source in SOURCE_CLASSES and single != "yes":
                issues.append(f"line {line}: named source requires single_source=yes")
            if source in SOURCE_CLASSES and not (row.get("review_notes") or "").strip():
                issues.append(f"line {line}: named source requires evidence in review_notes")
            if source == "depth_invalid_noisy" and expected["dataset"] == "D_tabletop_clean_v1":
                issues.append(f"line {line}: Tabletop relative depth cannot certify metric DEPTH source")
            if row.get("review_answerability") != "FOUND" and row.get("review_target_correct") == "yes":
                issues.append(f"line {line}: non-FOUND cannot confirm one target point as correct")
            if source in {"NONE", "MULTIPLE"} and single == "yes":
                issues.append(f"line {line}: {source} conflicts with single_source=yes")
            done.append((row, expected))
    missing = sorted(set(queue) - seen)
    if missing:
        issues.append(f"Missing {len(missing)} queue IDs, first: {missing[0]}")
    relation_compared = [(r["review_relation"], q["candidate_relation"])
                         for r, q in done if r["review_relation"] in RELATION_LABELS - {"UNSURE", "OUT_OF_SCOPE"}
                         and q["candidate_relation"]]
    answer_compared = [(r["review_answerability"], q["candidate_answerability"])
                       for r, q in done if r["review_answerability"] in STATES and q["candidate_answerability"]]
    return {
        "rows_in_queue": len(queue), "rows_returned": len(reviewed_rows),
        "done_count": len(done), "issues": issues,
        "relation_candidate_agreement": {"agree": sum(a == b for a, b in relation_compared),
                                         "compared": len(relation_compared)},
        "answerability_candidate_agreement": {"agree": sum(a == b for a, b in answer_compared),
                                              "compared": len(answer_compared)},
        "human_source_counts_done_only": dict(sorted(Counter(r["review_source"] for r, _ in done).items())),
        "canonical_single_source_counts_done_only": dict(sorted(Counter(
            SOURCE_CANONICAL[r["review_source"]] for r, _ in done
            if r["review_source"] in SOURCE_CANONICAL).items())),
        "note": "Agreement is not ground-truth precision; one reviewer and an unlocked pilot sample cannot establish G1_PASS.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-csv", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    queue_path = args.packet_dir / "HUMAN_AUDIT_QUEUE.csv"
    reviewed = load_csv(args.reviewed_csv)
    result = validate_rows(load_csv(queue_path), reviewed)
    report = {
        "status": "RUN_COMPLETE" if not result["issues"] else "SMOKE_FAIL",
        "gate_state": "G1_IN_PROGRESS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue_sha256": sha256(queue_path), "reviewed_sha256": sha256(args.reviewed_csv),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "done": result["done_count"],
                      "issues": len(result["issues"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
