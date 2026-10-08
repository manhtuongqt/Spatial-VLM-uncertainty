"""Verify every clean Tabletop training QA and export scoped label candidates.

The output is an audit artifact, not a v3 training release. No evaluator,
Calibration, or Test data is read. Source QA is never modified.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from .audit_development import ROOT, TABLETOP, map_relation, read_json, read_jsonl, ref, valid_point
from .build_label_audit import verify_prerequisites


DEFAULT_OUTPUT = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03/tabletop_qa_full_1500_v1"


def parse_single_point(answer: str) -> list[float]:
    """Parse the RefSpatial one-point answer without executing source text."""
    try:
        value = ast.literal_eval(answer)
    except (ValueError, SyntaxError, TypeError, RecursionError) as exc:
        raise ValueError(f"Invalid QA answer syntax: {answer!r}") from exc
    if not isinstance(value, (tuple, list)) or len(value) != 1:
        raise ValueError(f"Expected one point: {answer!r}")
    point = value[0]
    if not valid_point(point) or any(isinstance(v, bool) or not math.isfinite(v) for v in point):
        raise ValueError(f"Invalid normalized point: {answer!r}")
    return [float(point[0]), float(point[1])]


def extract_rows(provenance: list[dict], train: list[dict]) -> tuple[list[dict], dict]:
    train_prov = [row for row in provenance if row["split"] == "train"]
    train_by_id = {row["source_sample_id"]: row for row in train}
    if len(train_by_id) != len(train) or len(train_prov) != len({r["sample_id"] for r in train_prov}):
        raise ValueError("Duplicate sample ID in train or provenance")
    if set(train_by_id) != {r["sample_id"] for r in train_prov}:
        raise ValueError("Train and provenance sample IDs differ")

    output = []
    relation_counts = Counter()
    raw_relation_counts = Counter()
    for source in train_prov:
        sample_id = source["sample_id"]
        qa = train_by_id[sample_id]
        turns = qa.get("conversations")
        if not isinstance(turns, list) or len(turns) != 2 or [t.get("from") for t in turns] != ["human", "gpt"]:
            raise ValueError(f"{sample_id}: unexpected QA structure")
        question, answer = turns[0]["value"], turns[1]["value"]
        if question != source["instruction"] or answer != source["answer_original"]:
            raise ValueError(f"{sample_id}: QA differs from locked provenance")
        if Path(qa["image"]).name != Path(source["image"]).name or Path(qa["depth"]).name != Path(source["depth"]).name:
            raise ValueError(f"{sample_id}: media differs from provenance")
        point = parse_single_point(answer)
        if not valid_point(source.get("target_xy")) or any(abs(a - b) > 1e-9 for a, b in zip(point, source["target_xy"])):
            raise ValueError(f"{sample_id}: parsed point differs from provenance")
        if source["relation"] == "leftmost_ranking" and not re.search(r"\bleftmost\b", question, re.IGNORECASE):
            raise ValueError(f"{sample_id}: leftmost relation disagrees with question")
        if source["relation"] == "rightmost_ranking" and not re.search(r"\brightmost\b", question, re.IGNORECASE):
            raise ValueError(f"{sample_id}: rightmost relation disagrees with question")
        candidate = map_relation(source["relation"], question)
        raw_relation_counts[source["relation"]] += 1
        relation_counts[candidate or "UNMAPPED"] += 1
        output.append({
            "sample_id": sample_id,
            "family_id": source["family_id"],
            "split": "train",
            "rgb_path": source["image"],
            "relative_depth_path": source["depth"],
            "question": question,
            "answer_original": answer,
            "target_uv_b1": point,
            "target_label_source": "RefSpatial QA answer; B1 target grounding only",
            "relation_raw": source["relation"],
            "relation_v3_candidate": candidate,
            "relation_candidate_status": "CANDIDATE_NOT_V3_CERTIFIED" if candidate else "UNMAPPED",
            "reasoning_depth_v3": None,
            "answerability_v3": None,
            "primary_uncertainty_source_v3": None,
            "metric_depth_path": None,
            "camera_intrinsics": None,
            "tf_provenance": None,
            "v3_training_eligible": False,
        })
    if len({row["family_id"] for row in output}) != len(output):
        raise ValueError("Train family IDs are not unique")
    return output, {
        "qa_rows_verified": len(output),
        "unique_train_families": len(output),
        "b1_target_valid": len(output),
        "relation_raw_counts": dict(sorted(raw_relation_counts.items())),
        "relation_v3_candidate_counts": dict(sorted(relation_counts.items())),
        "v3_reasoning_labels": 0,
        "v3_answerability_labels": 0,
        "v3_uncertainty_source_labels": 0,
        "v3_train_ready_rows": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    verify_prerequisites()
    manifest = read_json(TABLETOP / "manifest.json")
    provenance = list(read_jsonl(TABLETOP / "provenance.jsonl"))
    train_path = TABLETOP / "train.json"
    train = read_json(train_path)
    if len(train) != manifest["data_counts"]["train"]:
        raise ValueError("Train count differs from clean manifest")
    rows, counts = extract_rows(provenance, train)
    if counts["qa_rows_verified"] != manifest["data_counts"]["train"]:
        raise ValueError("Verified count differs from manifest")
    summary = {
        "status": "RUN_COMPLETE",
        "gate_state": "G1_IN_PROGRESS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All 1500 D_tabletop_clean_v1 training QA; B1 grounding labels and v3 relation candidates only",
        "inputs": {
            "manifest": ref(TABLETOP / "manifest.json"),
            "provenance": ref(TABLETOP / "provenance.jsonl"),
            "train_qa": ref(train_path),
        },
        "counts": counts,
        "limitations": [
            "No individual human recertification of all 1500 QA is claimed",
            "Prior 300-case B1 audit is preserved; this run is exact automated QA/provenance reconciliation",
            "QA target is B1 point grounding, not v3 metric coordinate or a final evaluation label",
            "Relation mapping is a candidate pending G1 ontology and independent audit",
            "No answerability, uncertainty-source, reasoning-depth, or metric geometry is inferred",
            "These 1500 training rows cannot be used as a final Test set",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "QA_LABEL_CANDIDATES.jsonl"
    summary_path = args.output_dir / "QA_FULL_AUDIT_SUMMARY.json"
    if rows_path.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite an existing QA audit artifact")
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": summary["status"], "counts": counts,
                      "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
