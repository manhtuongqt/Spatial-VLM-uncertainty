#!/usr/bin/env python3
"""Build a sealed 1,500-family clean-release candidate; never authorize training."""

import collections
import hashlib
import json
from datetime import datetime
from pathlib import Path

from refspatial_wp1_rules import semantic_review_requirements


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/spatial_vlm_refspatial_v1"
AUDIT = BASE / "wp1_source_audit"
OUT = BASE / "wp1_clean_candidate"
SOURCE = AUDIT / "object_candidates.jsonl"
AUDIT_QUEUE = AUDIT / "manual_audit_queue.jsonl"
HUMAN_GATE = AUDIT / "manual_audit_human_gate.json"
PINNED_LINEAGE = AUDIT / "pinned_source_lineage.json"
HUMAN_EVAL = ROOT / "datasets/D_tabletop_human_eval_v1/evaluation.jsonl"
OFFICIAL_README_PROVENANCE = ROOT / "datasets/refspatial_supplement_v1_20260907/source/README.md.provenance.json"
RELATIONS = ("leftmost_ranking", "rightmost_ranking", "horizontal_ordinal_ranking")
QUOTA = 500


def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    gate = json.loads(HUMAN_GATE.read_text())
    if gate.get("status") != "HUMAN_SAMPLE_GATE_PASS_RELEASE_BLOCKED":
        raise RuntimeError("Independent human sample gate has not passed")
    lineage = json.loads(PINNED_LINEAGE.read_text())
    if lineage.get("status") != "PINNED_METADATA_LINEAGE_PASS":
        raise RuntimeError("Pinned official metadata lineage has not passed")
    audit_families = {row["family_id"] for row in jsonl(AUDIT_QUEUE)}
    eval_families = {row["family_id"] for row in jsonl(HUMAN_EVAL)}
    excluded_families = audit_families | eval_families

    per_relation = {relation: {} for relation in RELATIONS}
    with SOURCE.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "train" or row["family_id"] in excluded_families:
                continue
            guard = semantic_review_requirements(
                row["instruction"], row["target_kind"], row["canonical_type"]
            )
            relation = guard.get("direct_relation")
            if relation not in RELATIONS or guard.get("blockers"):
                continue
            old = per_relation[relation].get(row["family_id"])
            if old is None or row["sample_id"] < old["sample_id"]:
                per_relation[relation][row["family_id"]] = row

    selected = []
    used_families = set()
    available = {}
    for relation in RELATIONS:
        pool = sorted(per_relation[relation].values(), key=lambda row: row["sample_id"])
        available[relation] = len(pool)
        chosen = []
        for row in pool:
            if row["family_id"] in used_families:
                continue
            chosen.append(row)
            used_families.add(row["family_id"])
            if len(chosen) == QUOTA:
                break
        if len(chosen) != QUOTA:
            raise RuntimeError(f"Insufficient disjoint families for {relation}: {len(chosen)}/{QUOTA}")
        for row in chosen:
            selected.append({
                "sample_id": row["sample_id"],
                "scene_id": row["scene_id"],
                "family_id": row["family_id"],
                "source_split": row["split"],
                "proposed_split": "train",
                "image": row["image"],
                "depth": row["depth"],
                "instruction": row["instruction"],
                "target_xy": row["target_xy"],
                "target_kind": "object",
                "relation_candidate": relation,
                "reference_frame_candidate": "image_viewer_left_to_right",
                "reasoning_depth": None,
                "reasoning_depth_status": "NOT_CERTIFIED_FOR_B2",
                "label_qc": "EXTRACTION_CANDIDATE_RELEASE_GATE_BLOCKED",
                "training_eligible": False,
                "sam2_used": False,
            })

    OUT.mkdir(parents=True, exist_ok=True)
    samples_path = OUT / "samples.jsonl"
    with samples_path.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    counts = collections.Counter(row["relation_candidate"] for row in selected)
    sample_ids = {row["sample_id"] for row in selected}
    family_ids = {row["family_id"] for row in selected}
    checks = {
        "human_sample_gate_passed": gate["validation_report"]["status"] == "MANUAL_SAMPLE_GATE_PASS",
        "selected_1500": len(selected) == QUOTA * len(RELATIONS),
        "quota_500_each": all(counts[relation] == QUOTA for relation in RELATIONS),
        "unique_sample_ids": len(sample_ids) == len(selected),
        "unique_families_globally": len(family_ids) == len(selected),
        "no_human_eval_family_overlap": not (family_ids & eval_families),
        "no_manual_audit_family_overlap": not (family_ids & audit_families),
        "all_points_in_bounds": all(
            len(row["target_xy"]) == 2 and all(0 <= float(value) <= 1 for value in row["target_xy"])
            for row in selected
        ),
        "all_media_exist": all((ROOT / row[key]).is_file() for row in selected for key in ("image", "depth")),
        "no_privileged_rationale_fields": all("think" not in row and "thinking" not in row for row in selected),
        "training_disabled": all(row["training_eligible"] is False for row in selected),
        "sam2_not_used": all(row["sam2_used"] is False for row in selected),
    }
    official = json.loads(OFFICIAL_README_PROVENANCE.read_text())
    manifest = {
        "name": "D_tabletop_clean_candidate_v1",
        "status": "CANDIDATE_ONLY_RELEASE_GATE_BLOCKED",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sample_count": len(selected),
        "family_count": len(family_ids),
        "relation_counts": dict(counts),
        "available_train_families_after_exclusions": available,
        "selection": "lowest sample_id per family, 500 globally family-disjoint rows per direct horizontal ranking",
        "excluded_manual_audit_families": len(audit_families),
        "excluded_human_eval_families": len(eval_families),
        "checks": checks,
        "checks_passed": all(checks.values()),
        "source": {
            "derived_object_candidates_sha256": sha256(SOURCE),
            "official_repository": official["repo"],
            "official_revision_target": official["revision"],
            "official_license_from_pinned_card": "apache-2.0",
            "official_readme_sha256": official["sha256"],
            "derived_copy_matched_to_official_raw_metadata": True,
            "derived_unique_records_matched": lineage["derived_union"]["matched_unique_record_fingerprints"],
            "derived_unique_records_missing": lineage["derived_union"]["missing_unique_record_fingerprints"],
            "original_filter_reproduced": False,
        },
        "release_gate": {
            "D_tabletop_clean_v1_created": False,
            "training_eligible": False,
            "b2_open": False,
            "blockers": [
                "original tabletop filtering code/revision has not been reproduced",
                "the 1,500 selected labels are pipeline candidates, not individually human-certified",
                "reasoning-depth labels are intentionally absent",
            ],
        },
        "samples_sha256": sha256(samples_path),
        "sam2_used": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "sample_count": manifest["sample_count"],
        "family_count": manifest["family_count"],
        "relation_counts": manifest["relation_counts"],
        "checks_passed": manifest["checks_passed"],
        "training_eligible": False,
        "b2_open": False,
    }, indent=2))


if __name__ == "__main__":
    main()
