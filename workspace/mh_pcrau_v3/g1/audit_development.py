"""Read-only G1 inventory of already authorized development datasets.

This is an input/schema audit, not a v3 split release or a G1 PASS decision.
Evaluator rows are read only for aggregate label support and never emitted as
model input records.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
DAY1 = ROOT / "ketqua1/00_quan_tri_khoa/ngay_01"
DAY2 = ROOT / "ketqua1/03_backbone_h_spatial/ngay_02"
TABLETOP = ROOT / "datasets/D_tabletop_clean_v1"
GAZEBO = ROOT / "datasets/Gazebo_train_uq_v2_full_r3"
RELATIONS = ("leftmost", "rightmost", "second_from_left", "second_from_right")
STATES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc


def ref(path: Path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def map_relation(value: str, instruction: str) -> str | None:
    """Candidate mapping only; no oracle labels and no implied certification."""
    if value == "leftmost_ranking":
        return "leftmost"
    if value == "rightmost_ranking":
        return "rightmost"
    if value == "horizontal_ordinal_ranking":
        text = " ".join(instruction.lower().split())
        if re.search(r"\bsecond\b.*\bfrom left to right\b", text):
            return "second_from_left"
        if re.search(r"\bsecond\b.*\bfrom right to left\b", text):
            return "second_from_right"
        return None
    if value in RELATIONS:
        return value
    return None


def split_overlap(rows):
    families = defaultdict(set)
    for row in rows:
        families[row["split"]].add(row["family_id"])
    splits = sorted(families)
    return {
        f"{left}|{right}": len(families[left] & families[right])
        for i, left in enumerate(splits)
        for right in splits[i + 1 :]
    }


def valid_point(point):
    return (
        isinstance(point, (list, tuple))
        and len(point) == 2
        and all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in point)
    )


def audit_tabletop(hash_rgb: bool):
    manifest = read_json(TABLETOP / "manifest.json")
    rows = list(read_jsonl(TABLETOP / "provenance.jsonl"))
    split_counts = Counter(row["split"] for row in rows)
    relation_candidates = Counter()
    fields = Counter()
    missing_media = Counter()
    rgb_hashes = defaultdict(set)
    duplicate_ids = len(rows) - len({row["sample_id"] for row in rows})
    for row in rows:
        candidate = map_relation(row.get("relation", ""), row.get("instruction", ""))
        relation_candidates[candidate or "UNMAPPED"] += 1
        fields["candidate_relation_mapped"] += candidate is not None
        fields["target_uv_valid"] += valid_point(row.get("target_xy"))
        fields["reasoning_depth_present"] += row.get("reasoning_depth") is not None
        fields["answerability_state_present"] += row.get("answerability_state") in STATES
        fields["source_label_present"] += row.get("primary_uncertainty_source") is not None
        fields["metric_depth_present"] += bool(row.get("metric_depth_path"))
        fields["intrinsics_present"] += row.get("camera_intrinsics") is not None
        fields["tf_provenance_present"] += row.get("tf_provenance") is not None
        for name in ("image", "depth"):
            path = ROOT / row[name]
            if not path.is_file():
                missing_media[name] += 1
            elif name == "image" and hash_rgb:
                rgb_hashes[sha256(path)].add(row["split"])
    assert dict(split_counts) == manifest["data_counts"], "Tabletop split count drift"
    return {
        "source": ref(TABLETOP / "manifest.json"),
        "provenance": ref(TABLETOP / "provenance.jsonl"),
        "dataset_role": "B1_TARGET_GROUNDING_ONLY; v3 eligibility not yet certified",
        "records": len(rows),
        "families": len({row["family_id"] for row in rows}),
        "split_counts": dict(sorted(split_counts.items())),
        "family_overlap": split_overlap(rows),
        "duplicate_sample_ids": duplicate_ids,
        "candidate_relation_counts": dict(sorted(relation_candidates.items())),
        "field_counts": dict(sorted(fields.items())),
        "missing_media": dict(sorted(missing_media.items())),
        "depth_semantics": "relative_visual_depth; not certified metric depth",
        "rgb_exact_duplicates_across_splits": sum(len(splits) > 1 for splits in rgb_hashes.values())
        if hash_rgb else None,
        "_rgb_hashes": rgb_hashes,
    }


def audit_gazebo(hash_rgb: bool):
    manifest = read_json(GAZEBO / "manifest.json")
    infer = list(read_jsonl(GAZEBO / "inference_manifest.jsonl"))
    supervise = list(read_jsonl(GAZEBO / "train_supervision.jsonl"))
    evaluator = list(read_jsonl(GAZEBO / "evaluator_ground_truth.jsonl"))
    by_id = {row["sample_id"]: row for row in infer}
    supervised_ids = {row["sample_id"] for row in supervise}
    evaluator_ids = {row["sample_id"] for row in evaluator}
    if len(by_id) != len(infer) or len(supervised_ids) != len(supervise):
        raise ValueError("Gazebo duplicate sample IDs")
    if not supervised_ids.issubset(by_id) or evaluator_ids != set(by_id):
        raise ValueError("Gazebo inference/supervision/evaluator ID mismatch")
    if any("supervision" in row for row in infer):
        raise ValueError("Oracle supervision embedded in inference manifest")
    split_counts = Counter(row["split"] for row in infer)
    assert dict(split_counts) == manifest["split_counts"], "Gazebo split count drift"
    fields = Counter()
    missing_media = Counter()
    relation_candidates = Counter()
    rgb_hashes = defaultdict(set)
    for row in infer:
        candidate = map_relation(row.get("relation_variant", ""), row.get("instruction", ""))
        relation_candidates[candidate or "UNMAPPED"] += 1
        fields["candidate_relation_mapped"] += candidate is not None
        fields["metric_depth_present"] += bool(row.get("metric_depth"))
        fields["intrinsics_present"] += row.get("camera_intrinsics") is not None
        fields["tf_provenance_present"] += row.get("tf_provenance") is not None
        fields["reasoning_depth_present"] += row.get("reasoning_depth") is not None
        fields["source_label_present"] += row.get("primary_uncertainty_source") is not None
        for name in ("image", "depth", "metric_depth"):
            path = GAZEBO / row[name]
            if not path.is_file():
                missing_media[name] += 1
            elif name == "image" and hash_rgb:
                rgb_hashes[sha256(path)].add(row["split"])
    states = Counter(row["answerability_state"] for row in evaluator)
    supervised_states = Counter(row["supervision"]["answerability_state"] for row in supervise)
    fields["evaluator_answerability_present"] = sum(states.values())
    fields["train_supervision_answerability_present"] = sum(supervised_states.values())
    fields["train_supervision_target_uv_valid"] = sum(
        valid_point(row["supervision"].get("target_xy")) for row in supervise
    )
    return {
        "source": ref(GAZEBO / "manifest.json"),
        "inference_manifest": ref(GAZEBO / "inference_manifest.jsonl"),
        "supervision_manifest": ref(GAZEBO / "train_supervision.jsonl"),
        "evaluator_manifest": ref(GAZEBO / "evaluator_ground_truth.jsonl"),
        "dataset_role": "development schema audit; train_uq supervision only",
        "records": len(infer),
        "families": len({row["family_id"] for row in infer}),
        "split_counts": dict(sorted(split_counts.items())),
        "family_overlap": split_overlap(infer),
        "candidate_relation_counts": dict(sorted(relation_candidates.items())),
        "answerability_evaluator_counts": dict(sorted(states.items())),
        "answerability_train_supervision_counts": dict(sorted(supervised_states.items())),
        "field_counts": dict(sorted(fields.items())),
        "missing_media": dict(sorted(missing_media.items())),
        "rgb_exact_duplicates_across_splits": sum(len(splits) > 1 for splits in rgb_hashes.values())
        if hash_rgb else None,
        "_rgb_hashes": rgb_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-rgb-hash", action="store_true")
    args = parser.parse_args()
    decision = read_json(DAY2 / "G0_DECISION.json")
    if decision["outcome"] != "G0_PASS":
        raise ValueError("G0 PASS prerequisite missing")
    boundary_path = DAY1 / "DATA_ACCESS_BOUNDARY.json"
    boundary = read_json(boundary_path)
    expected = {d["dataset"]: d["manifest_sha256"] for d in boundary["development_data"]}
    if sha256(TABLETOP / "manifest.json") != expected["D_tabletop_clean_v1"]:
        raise ValueError("Tabletop manifest differs from Day 1 lock")
    if sha256(GAZEBO / "manifest.json") != expected["Gazebo_train_uq_v2_full_r3"]:
        raise ValueError("Gazebo manifest differs from Day 1 lock")
    hash_rgb = not args.skip_rgb_hash
    tabletop = audit_tabletop(hash_rgb)
    gazebo = audit_gazebo(hash_rgb)
    cross_rgb = (
        len(set(tabletop["_rgb_hashes"]) & set(gazebo["_rgb_hashes"]))
        if hash_rgb else None
    )
    tabletop.pop("_rgb_hashes")
    gazebo.pop("_rgb_hashes")
    checks = {
        "g0_pass": True,
        "day1_dataset_manifest_hashes_match": True,
        "tabletop_family_splits_disjoint": all(v == 0 for v in tabletop["family_overlap"].values()),
        "gazebo_family_splits_disjoint": all(v == 0 for v in gazebo["family_overlap"].values()),
        "tabletop_media_exist": not tabletop["missing_media"],
        "gazebo_media_exist": not gazebo["missing_media"],
        "gazebo_inference_manifest_has_no_supervision": True,
        "exact_rgb_cross_split_or_dataset_duplicates_absent": (
            tabletop["rgb_exact_duplicates_across_splits"] == 0
            and gazebo["rgb_exact_duplicates_across_splits"] == 0
            and cross_rgb == 0
        ) if hash_rgb else None,
    }
    report = {
        "schema_version": "1.0",
        "method_id": "MH-PCRA-U-v3",
        "namespace": "mh_pcrau_v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUN_COMPLETE",
        "gate_state": "G1_IN_PROGRESS",
        "scope": "Read-only inventory of two authorized development datasets; no new split, train, capture, Calibration, Test or robot access",
        "audit_source": ref(Path(__file__)),
        "g0_decision": ref(DAY2 / "G0_DECISION.json"),
        "day1_boundary": ref(boundary_path),
        "tabletop": tabletop,
        "gazebo": gazebo,
        "exact_rgb_hash_overlap_between_datasets": cross_rgb,
        "checks": checks,
        "not_yet_checked": [
            "near-duplicate RGB/layout/signature across historical attempts",
            "independent v3 human certification of relation, reasoning and source labels",
            "power/precision calculations and preregistered numerical margins",
            "new v3 train/val/calibration/test exclusion registry",
            "metric depth, camera intrinsics and TF completeness for all candidate rows",
        ],
        "gate_conclusion": "NOT_DECIDED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "status": report["status"],
        "gate_state": report["gate_state"],
        "tabletop_records": tabletop["records"],
        "gazebo_records": gazebo["records"],
        "checks": checks,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
