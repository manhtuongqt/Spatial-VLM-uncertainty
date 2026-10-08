"""Build a 512-image pilot index with per-head label masks and provenance.

This index is exploratory development evidence, never a train/Test release.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json

from .audit_development import GAZEBO, ROOT, read_jsonl, sha256
from .generate_gazebo_pilot_512 import OUT


def main() -> None:
    qc_path = OUT / "PILOT_512_CAPTURE_QC.json"
    index_path = OUT / "PILOT_512_INDEX.jsonl"
    summary_path = OUT / "PILOT_512_INDEX_SUMMARY.json"
    if index_path.exists() or summary_path.exists():
        raise FileExistsError("Refusing overwrite pilot index")
    qc = json.loads(qc_path.read_text())
    if qc["new_captured_images"] != 256:
        raise ValueError("Need 256 actually captured new pilot images")
    old_infer = [row for row in read_jsonl(GAZEBO / "inference_manifest.jsonl")
                 if row["split"] == "train_uq"]
    old_supervision = {row["sample_id"]: row for row in read_jsonl(GAZEBO / "train_supervision.jsonl")}
    old_rows = []
    for row in old_infer:
        original = old_supervision[row["sample_id"]]["supervision"]
        state = original["answerability_state"]
        old_rows.append({
            "sample_id": row["sample_id"], "family_id": row["family_id"],
            "origin": "EXISTING_GAZEBO_TRAIN_UQ_REUSED_AS_PILOT",
            "rgb_path": str((GAZEBO / row["image"]).relative_to(ROOT)),
            "depth_path": str((GAZEBO / row["depth"]).relative_to(ROOT)),
            "metric_depth_path": str((GAZEBO / row["metric_depth"]).relative_to(ROOT)),
            "instruction": row["instruction"],
            "relation_candidate": row["relation_variant"],
            "answerability_candidate": state,
            "target_uv_candidate": original.get("target_xy") if state == "FOUND" else None,
            "head_label_mask": {"relation": True, "answerability": True,
                                "coordinate": state == "FOUND", "variance": state == "FOUND",
                                "reasoning": False, "source": False, "confidence": False},
            "label_provenance": "existing train_supervision; 36 conflicts resolved KEEP_ORIGINAL",
            "pilot_qc": "HISTORICAL_DATASET_QC_PASS",
            "v3_training_eligible": False,
        })
    new_rows = []
    for batch in range(4):
        folder = OUT / f"batch_{batch}"
        capture = folder / "capture_attempt_01"
        qc_report = json.loads((folder / "CAPTURE_QC.json").read_text())
        verified = {row["scene_id"]: row for row in qc_report["rows"]}
        for row in read_jsonl(capture / "input_manifest.jsonl"):
            sid = row["scene_id"]
            check = verified[sid]
            usable = qc_report["integrity"] and not check["reasons"]
            new_rows.append({
                "sample_id": sid, "family_id": check["family_id"],
                "origin": "NEW_V3_GAZEBO_PILOT_CAPTURE",
                "rgb_path": str((capture / row["input_files"]["rgb"]).relative_to(ROOT)),
                "depth_path": None,
                "metric_depth_path": str((capture / row["input_files"]["depth_m"]).relative_to(ROOT)),
                "instruction": row["instruction"],
                "relation_candidate": check["relation"] if usable else None,
                "answerability_candidate": check["requested_state"] if usable else None,
                "target_uv_candidate": None,
                "head_label_mask": {"relation": bool(usable), "answerability": bool(usable),
                                    "coordinate": False, "variance": False,
                                    "reasoning": False, "source": False, "confidence": False},
                "label_provenance": "preregistered pilot scene + evaluator-only geometry QC" if usable
                                    else "REQUESTED_STATE_QC_FAILED_MASK_ALL_HEADS",
                "pilot_qc": "PASS" if usable else "REJECT",
                "v3_training_eligible": False,
            })
    rows = old_rows + new_rows
    assert len(rows) == 512 and len({row["sample_id"] for row in rows}) == 512
    assert len({row["family_id"] for row in rows}) == 512
    with index_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    status_counts = Counter(row["pilot_qc"] for row in rows)
    masks = {head: sum(row["head_label_mask"][head] for row in rows)
             for head in rows[0]["head_label_mask"]}
    answer_counts = Counter(row["answerability_candidate"] for row in rows
                            if row["answerability_candidate"])
    summary = {
        "status": "PILOT_INDEX_COMPLETE_NOT_TRAIN_RELEASE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": 512, "parent_families": 512,
        "origin_counts": dict(Counter(row["origin"] for row in rows)),
        "qc_status_counts": dict(status_counts), "head_label_mask_counts": masks,
        "answerability_candidate_counts": dict(sorted(answer_counts.items())),
        "index_sha256": sha256(index_path), "capture_qc_sha256": sha256(qc_path),
        "v3_training_eligible": 0,
        "limitations": [
            "Existing 256 images are reused, not newly captured",
            "New failed-QC images remain indexed but every head label is masked",
            "No reasoning-depth or five-class uncertainty-source labels were certified",
            "No confidence OOF correctness label exists",
            "This exploratory pilot must not be treated as sealed Test or Calibration",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(rows), "head_masks": masks,
                      "answerability": summary["answerability_candidate_counts"]}))


if __name__ == "__main__":
    main()
