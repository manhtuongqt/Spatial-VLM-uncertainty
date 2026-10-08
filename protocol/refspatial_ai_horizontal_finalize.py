#!/usr/bin/env python3
"""Record the attributed AI visual audit of the 150 horizontal candidates.

Decisions come from contact-sheet inspection under ontology v2.  They remain
AI decisions, so the human data gate and training eligibility stay closed.
"""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/spatial_vlm_refspatial_v1/wp2_visual_review"
QUEUE = OUT / "next_audit_queue.jsonl"
DECISIONS = OUT / "next_audit_decisions.csv"
UNCERTAIN = {
    "0334af62f9bfca7d26ac75a0": "Ordinal 7 over all objects: candidate count/order remains visually ambiguous.",
    "055b6634fa34c305629f0a16": "Green water-bottle class and the rightmost instance remain visually ambiguous.",
    "01487ad64c6821f80a1317c4": "Ordinal 4 right-to-left is not reliable from visible candidate centres.",
}


def main():
    queue = [json.loads(line) for line in QUEUE.read_text().splitlines()]
    output, counts = [], Counter()
    for index, row in enumerate(queue, 1):
        sid = row["sample_id"]
        sheet = f"ai_visual_review_sheets_v2/sheet_{(index - 1) // 10 + 1:02d}.jpg"
        common = {
            "sample_id": sid, "scene_id": row["scene_id"], "split": row["split"],
            "audit_strata": "|".join(row["audit_strata"]),
            "reviewer": "AI-Codex-horizontal-v2", "reviewed_at": "2026-09-09T14:45:00+07:00",
            "anchor_correct": "na", "reference_frame": "image_x_right_positive",
            "reasoning_depth": "1", "review_origin": "ai_visual",
        }
        if sid in UNCERTAIN:
            decision = {**common, "target_correct": "unsure", "relation_correct": "unsure",
                        "tabletop_appropriate": "yes", "decision": "unsure",
                        "notes": f"{UNCERTAIN[sid]} Visual evidence: {sheet}."}
        else:
            decision = {**common, "target_correct": "yes", "relation_correct": "yes",
                        "tabletop_appropriate": "yes", "decision": "accept",
                        "notes": f"Target point lies on the visually selected ranked object under ontology v2. Visual evidence: {sheet}."}
        output.append(decision); counts[decision["decision"]] += 1
    fields = [
        "sample_id", "scene_id", "split", "audit_strata", "reviewer", "reviewed_at",
        "target_correct", "relation_correct", "anchor_correct", "tabletop_appropriate",
        "reference_frame", "reasoning_depth", "decision", "notes", "review_origin",
    ]
    with DECISIONS.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(output)
    report = {
        "review_origin": "ai_visual",
        "reviewer": "AI-Codex-horizontal-v2",
        "queue_count": len(queue), "decisions": dict(counts),
        "uncertain_sample_ids": sorted(UNCERTAIN),
        "method": "Visual inspection of 15 contact sheets under ontology v2; source/model markers were available as review aids.",
        "limitation": "Not independent human review and cannot pass the human data gate or certify training eligibility.",
    }
    (OUT / "ai_horizontal_visual_review_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
