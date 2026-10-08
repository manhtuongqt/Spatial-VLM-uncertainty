"""Prepare a blind, development-only review worklist without altering progress."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .audit_development import ROOT, ref
from .build_label_audit import DEFAULT_OUTPUT
from .validate_human_review import load_csv, validate_rows


DAY3 = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03"
DEFAULT_PREP = DAY3 / "label_audit_preparation_v1"
BLIND_COLUMNS = ("queue_id", "dataset", "sample_id", "family_id", "split",
                 "rgb_path", "depth_view_path", "instruction", "saved_status",
                 "missing_review_fields")
REVIEW_FIELDS = ("review_relation", "review_answerability", "review_target_correct",
                 "review_reasoning_depth", "review_source", "review_single_source")


def prepare(queue: list[dict], saved: list[dict]) -> tuple[list[dict], dict]:
    review = validate_rows(queue, saved)
    if review["rows_in_queue"] != review["rows_returned"] or review["issues"]:
        raise ValueError(f"Saved review identity/label validation failed: {review['issues'][:3]}")
    by_id = {row["queue_id"]: row for row in saved}
    rows = []
    status = Counter()
    filled = Counter()
    dataset = Counter()
    prior_yes = 0
    for item in queue:
        row = by_id[item["queue_id"]]
        dataset[item["dataset"]] += 1
        status[row["review_status"]] += 1
        missing = []
        for field in REVIEW_FIELDS:
            if row.get(field, "").strip():
                filled[field] += 1
            elif field != "review_target_correct" or item.get("candidate_target_uv"):
                missing.append(field)
        if row.get("review_target_correct") == "yes":
            prior_yes += 1
        rows.append({
            "queue_id": item["queue_id"], "dataset": item["dataset"],
            "sample_id": item["sample_id"], "family_id": item["family_id"],
            "split": item["split"], "rgb_path": item["rgb_path"],
            "depth_view_path": item["depth_view_path"], "instruction": item["instruction"],
            "saved_status": row["review_status"],
            "missing_review_fields": ";".join(missing),
        })
    if len({row["family_id"] for row in rows}) != len(rows):
        raise ValueError("Queue families must be distinct")
    return rows, {
        "queue_rows": len(rows), "unique_families": len(rows),
        "by_dataset": dict(sorted(dataset.items())),
        "saved_status_counts": dict(sorted(status.items())),
        "filled_field_counts": dict(sorted(filled.items())),
        "prior_target_yes_unverifiable_autopreset_risk": prior_yes,
        "done_count_under_existing_validator": review["done_count"],
        "relation_candidate_agreement": review["relation_candidate_agreement"],
        "answerability_candidate_agreement": review["answerability_candidate_agreement"],
        "source_counts": review["canonical_single_source_counts_done_only"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREP)
    args = parser.parse_args()
    queue_path = args.packet_dir / "HUMAN_AUDIT_QUEUE.csv"
    saved_path = args.packet_dir / "HUMAN_AUDIT_FILLED.csv"
    if not saved_path.is_file():
        raise FileNotFoundError(saved_path)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    queue, saved = load_csv(queue_path), load_csv(saved_path)
    rows, counts = prepare(queue, saved)
    report = {
        "status": "PREPARED_NOT_HUMAN_CERTIFIED",
        "gate_state": "G1_IN_PROGRESS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue": ref(queue_path), "saved_review": ref(saved_path),
        "ontology": ref(DAY3 / "ONTOLOGY_AND_SOURCE_RULES.md"),
        "counts": counts,
        "instructions": [
            "Use the existing GUI queue; do not overwrite or reset HUMAN_AUDIT_FILLED.csv",
            "Read ontology before labeling; candidate point shown only after independent decision",
            "A prior GUI preset could set target_correct=yes when FOUND was selected; those values are not independent human target confirmations unless manually rechecked",
            "Do not treat this 128-case pilot as source precision certification or full G1 audit",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "HUMAN_AUDIT_REMAINING_BLIND.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=BLIND_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "AUDIT_PREP_STATUS.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "counts": counts,
                      "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
