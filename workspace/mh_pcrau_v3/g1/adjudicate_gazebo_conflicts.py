"""Append-only adjudication record for the 36 confirmed pilot/Gazebo conflicts.

The 36 RGB images and prompts were inspected manually on 2026-09-24. This
script binds the decision to the frozen input, reviewer CSV, and evaluator
metadata; it does not modify any original label or authorize training.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .audit_development import GAZEBO, ROOT, read_jsonl, sha256


DAY3 = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03"
PACKET = DAY3 / "label_audit_packet_v1"
OUTPUT = DAY3 / "GAZEBO_36_CONFLICT_ADJUDICATION.csv"
SUMMARY = DAY3 / "GAZEBO_36_CONFLICT_ADJUDICATION.json"
NOTES = {
    "AMBIGUOUS": (
        "RGB shows apple/orange overlapping in horizontal rank; distinct visible "
        "objects can both satisfy the rank. Evaluator records two valid target "
        "IDs and a horizontal-tie failure tag. No unique point is justified."
    ),
    "ABSENT": (
        "The named fruit in the prompt is absent from the visible objects. "
        "Evaluator candidate_set is empty and absent_target is flagged."
    ),
    "INSUFFICIENT_EVIDENCE": (
        "The requested fruit is occluded or clipped in RGB. Evaluator visible "
        "pixels are below the preregistered sufficient-visibility threshold; "
        "a confident unique point is not justified."
    ),
}
FIELDS = (
    "sample_id", "family_id", "queue_id", "prompt", "rgb_path",
    "reviewer_state", "original_train_state", "evaluator_state",
    "evaluator_verified", "valid_target_count", "candidate_count",
    "visible_pixels", "minimum_sufficient_pixels", "failure_tags",
    "adjudicated_state", "adjudication_decision", "reason",
    "reviewed_rgb_20260924", "training_label_mutated",
)


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError("Refusing to overwrite adjudication artifacts")
    review_path = PACKET / "HUMAN_AUDIT_FILLED.csv"
    queue_path = PACKET / "HUMAN_AUDIT_QUEUE.csv"
    review = csv_rows(review_path)
    queue = {row["sample_id"]: row for row in csv_rows(queue_path)}
    infer_path = GAZEBO / "inference_manifest.jsonl"
    train_path = GAZEBO / "train_supervision.jsonl"
    eval_path = GAZEBO / "evaluator_ground_truth.jsonl"
    infer = {row["sample_id"]: row for row in read_jsonl(infer_path)}
    train = {row["sample_id"]: row for row in read_jsonl(train_path)}
    evaluator = {row["sample_id"]: row for row in read_jsonl(eval_path)}
    out = []
    for row in review:
        sid = row["sample_id"]
        if sid not in train:
            continue
        original = train[sid]["supervision"]["answerability_state"]
        reviewer = row["review_answerability"]
        if reviewer == original:
            continue
        source = infer[sid]
        oracle = evaluator[sid]
        visibility = oracle.get("visibility") or {}
        state = oracle["answerability_state"]
        assert reviewer == "FOUND" and original == state and state in NOTES, sid
        assert oracle["answerability_verified"] is True, sid
        assert row["review_status"] == "DONE" and queue[sid]["family_id"] == row["family_id"]
        if state == "AMBIGUOUS":
            assert len(oracle["valid_target_ids"]) >= 2 and "horizontal_tie" in oracle["failure_tags"], sid
        elif state == "ABSENT":
            assert not oracle["candidate_set"] and "absent_target" in oracle["failure_tags"], sid
        else:
            assert visibility["visible_pixels"] < visibility["minimum_sufficient_pixels"], sid
            assert "insufficient_visibility" in oracle["failure_tags"], sid
        out.append({
            "sample_id": sid, "family_id": row["family_id"], "queue_id": row["queue_id"],
            "prompt": source["instruction"],
            "rgb_path": str((GAZEBO / source["image"]).relative_to(ROOT)),
            "reviewer_state": reviewer, "original_train_state": original,
            "evaluator_state": state, "evaluator_verified": "true",
            "valid_target_count": len(oracle.get("valid_target_ids") or []),
            "candidate_count": len(oracle.get("candidate_set") or []),
            "visible_pixels": visibility.get("visible_pixels", ""),
            "minimum_sufficient_pixels": visibility.get("minimum_sufficient_pixels", ""),
            "failure_tags": "|".join(oracle.get("failure_tags") or []),
            "adjudicated_state": state, "adjudication_decision": "KEEP_ORIGINAL",
            "reason": NOTES[state], "reviewed_rgb_20260924": "yes",
            "training_label_mutated": "no",
        })
    counts = Counter(row["adjudicated_state"] for row in out)
    assert len(out) == 36 and counts == {
        "AMBIGUOUS": 12, "ABSENT": 12, "INSUFFICIENT_EVIDENCE": 12,
    }, counts
    with OUTPUT.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)
    summary = {
        "status": "ADJUDICATION_COMPLETE", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "36 Gazebo pilot cases with reviewer FOUND vs original non-FOUND",
        "decision": "KEEP_ORIGINAL_36_OF_36", "counts": dict(sorted(counts.items())),
        "visual_review": "All 36 RGB images plus prompts inspected; evaluator metadata used as corroboration",
        "provenance": {str(p.relative_to(ROOT)): sha256(p) for p in
            (review_path, queue_path, infer_path, train_path, eval_path)},
        "rows_path": str(OUTPUT.relative_to(ROOT)), "rows_sha256": sha256(OUTPUT),
        "source_data_modified": False,
        "limitations": [
            "Reviewer-attested FOUND responses remain preserved as a separate conflicting opinion",
            "This adjudication does not turn historical Gazebo training data into a new Test set",
            "No five-class uncertainty-source label is inferred from failure_tags",
        ],
    }
    with SUMMARY.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"decision": summary["decision"], "counts": summary["counts"]}))


if __name__ == "__main__":
    main()
