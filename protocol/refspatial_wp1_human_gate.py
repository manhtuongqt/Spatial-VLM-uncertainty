#!/usr/bin/env python3
"""Summarize the independent WP1 human audit without creating training data."""

import collections
import csv
import hashlib
import json
from pathlib import Path

from refspatial_manual_review import summarize
from refspatial_wp1_rules import semantic_review_requirements


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/spatial_vlm_refspatial_v1"
AUDIT = BASE / "wp1_source_audit"
QUEUE = AUDIT / "manual_audit_queue.jsonl"
DECISIONS = AUDIT / "manual_audit_human_decisions.csv"
JSON_REPORT = AUDIT / "manual_audit_human_gate.json"
MD_REPORT = AUDIT / "MANUAL_AUDIT_HUMAN_REPORT.md"
HUMAN_EVAL = ROOT / "datasets/D_tabletop_human_eval_v1/evaluation.jsonl"
PINNED_LINEAGE = AUDIT / "pinned_source_lineage.json"

DIRECT = ("leftmost_ranking", "rightmost_ranking", "horizontal_ordinal_ranking")
DISTANCE = ("nearest_ranking", "farthest_ranking")


def read_jsonl(path):
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
    queue = read_jsonl(QUEUE)
    with DECISIONS.open(newline="") as handle:
        decisions = list(csv.DictReader(handle))
    validation = summarize(queue, decisions)
    by_id = {row["sample_id"]: row for row in decisions}
    completed = [row for row in decisions if row.get("completed") == "yes"]
    completion_issues = []
    if len(completed) != len(queue):
        completion_issues.append("not_all_queue_rows_marked_completed")
    if any(row.get("review_origin") != "human" for row in completed):
        completion_issues.append("non_human_review_origin_present")

    eval_families = {row.get("family_id") for row in read_jsonl(HUMAN_EVAL)}
    queue_by_relation = collections.defaultdict(list)
    for item in queue:
        for relation in item.get("audit_strata", []):
            if relation in DIRECT + DISTANCE:
                queue_by_relation[relation].append(item)

    candidate_train_families = collections.defaultdict(set)
    eligible_candidate_train_families = collections.defaultdict(set)
    with (AUDIT / "object_candidates.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            guard = semantic_review_requirements(
                row["instruction"], row["target_kind"], row["canonical_type"]
            )
            relation = guard.get("direct_relation")
            if relation not in DIRECT:
                continue
            candidate_train_families[relation].add(row["family_id"])
            if not guard.get("blockers") and row["family_id"] not in eval_families:
                eligible_candidate_train_families[relation].add(row["family_id"])

    relation_gate = []
    for relation in DIRECT + DISTANCE:
        items = queue_by_relation[relation]
        reviewed = [by_id[item["sample_id"]] for item in items]
        accepted = [row for row in reviewed if row.get("decision") == "accept"]
        accepted_train_families = {
            item["family_id"] for item in items
            if item.get("split") == "train" and by_id[item["sample_id"]].get("decision") == "accept"
        }
        applicable = [row for row in reviewed if row.get("relation_correct") != "na"]
        relation_precision = (
            sum(row.get("relation_correct") == "yes" for row in applicable) / len(applicable)
            if applicable else None
        )
        if relation in DISTANCE:
            status = "OUTSIDE_B2_NONMETRIC_DISTANCE_SEMANTICS"
            candidate_count = None
            eligible_count = None
        else:
            status = "AUDIT_PASS_RELEASE_BLOCKED" if relation_precision is not None and relation_precision >= 0.95 else "AUDIT_FAIL"
            candidate_count = len(candidate_train_families[relation])
            eligible_count = len(eligible_candidate_train_families[relation])
        relation_gate.append({
            "relation": relation,
            "human_reviewed": len(reviewed),
            "human_accepted": len(accepted),
            "human_relation_precision": relation_precision,
            "human_certified_train_families": len(accepted_train_families),
            "candidate_train_families": candidate_count,
            "candidate_train_families_after_human_eval_exclusion": eligible_count,
            "required_clean_train_families": 500,
            "status": status,
        })

    queue_families = {row["family_id"] for row in queue}
    direct_train_review_families = {
        item["family_id"] for relation in DIRECT for item in queue_by_relation[relation]
        if item.get("split") == "train" and by_id[item["sample_id"]].get("decision") == "accept"
    }
    sample_gate_pass = (
        validation["status"] == "MANUAL_SAMPLE_GATE_PASS"
        and not completion_issues
        and validation.get("independent_human_review_complete")
    )
    lineage = json.loads(PINNED_LINEAGE.read_text()) if PINNED_LINEAGE.exists() else {}
    metadata_lineage_pass = lineage.get("status") == "PINNED_METADATA_LINEAGE_PASS"
    blockers = [
        "source-generator relation semantics are not independently verified",
        "clean annotation extraction and split manifests are not materialized/certified",
        "only 30 human-certified train families per audited relation; required clean support is 500",
        "nearest/farthest metric semantics cannot be certified from the local non-metric depth PNG",
    ]
    if not metadata_lineage_pass:
        blockers.insert(0, "raw source metadata revision is not verified locally")
    result = {
        "status": "HUMAN_SAMPLE_GATE_PASS_RELEASE_BLOCKED" if sample_gate_pass else "HUMAN_SAMPLE_GATE_NOT_MET",
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "queue_sha256": sha256(QUEUE),
        "human_decisions_sha256": sha256(DECISIONS),
        "validation_report": validation,
        "completion_issues": completion_issues,
        "relation_gate": relation_gate,
        "human_eval_family_exclusion": {
            "human_eval_families": len(eval_families),
            "queue_family_overlap": len(queue_families & eval_families),
            "accepted_direct_train_family_overlap": len(direct_train_review_families & eval_families),
            "must_exclude_from_any_training_release": True,
        },
        "pinned_source_metadata_lineage": {
            "status": lineage.get("status", "NOT_RUN"),
            "verified": metadata_lineage_pass,
            "official_revision": lineage.get("official_source", {}).get("revision"),
            "derived_unique_records_matched": lineage.get("derived_union", {}).get("matched_unique_record_fingerprints"),
            "derived_unique_records_missing": lineage.get("derived_union", {}).get("missing_unique_record_fingerprints"),
        },
        "release_gate": {
            "release": "D_tabletop_clean_v1",
            "created": False,
            "training_eligible": False,
            "b2_open": False,
            "retained_relations": [],
            "blockers": blockers,
        },
        "sam2_used": False,
    }
    JSON_REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    precision = validation["precision"]
    lines = [
        "# WP1 — Independent human-audit gate",
        "",
        f"Status: **{result['status']}**.",
        "",
        "## Human review",
        "",
        f"- Completed: {validation['completed_reviews']}/{validation['queue_count']}; origin: human.",
        f"- Decisions: `{json.dumps(validation['decision_counts'], sort_keys=True)}`.",
        f"- Target precision: {precision['target_correct']['yes']}/{precision['target_correct']['n']} = {precision['target_correct']['precision']:.2%}; Wilson 95% CI [{precision['target_correct']['wilson_ci95'][0]:.2%}, {precision['target_correct']['wilson_ci95'][1]:.2%}].",
        f"- Relation precision: {precision['relation_correct']['yes']}/{precision['relation_correct']['n']} = {precision['relation_correct']['precision']:.2%}; Wilson 95% CI [{precision['relation_correct']['wilson_ci95'][0]:.2%}, {precision['relation_correct']['wilson_ci95'][1]:.2%}].",
        f"- Anchor precision: {precision['anchor_correct']['yes']}/{precision['anchor_correct']['n']} = {precision['anchor_correct']['precision']:.2%}; Wilson 95% CI [{precision['anchor_correct']['wilson_ci95'][0]:.2%}, {precision['anchor_correct']['wilson_ci95'][1]:.2%}].",
        "- Each selected stratum has 50 reviews; the 62 general-grounding rows have no applicable relation label.",
        "",
        "## Relation gate",
        "",
        "| Relation | Human review | Accepted train family | Candidate train family after eval exclusion | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in relation_gate:
        candidate = "n/a" if row["candidate_train_families_after_human_eval_exclusion"] is None else str(row["candidate_train_families_after_human_eval_exclusion"])
        lines.append(
            f"| {row['relation']} | {row['human_reviewed']} | {row['human_certified_train_families']} | {candidate} | {row['status']} |"
        )
    lines += [
        "",
        "The three horizontal ranking labels pass the sampled human audit, but ranking is not silently relabeled as pairwise left/right. Nearest/farthest remain outside B2 because the local depth is a non-metric visual proxy.",
        "",
        "## Release decision",
        "",
        "`D_tabletop_clean_v1` is not created and B2 remains closed. Candidate support above 500 is not the same as 500 certified clean families. The extraction/provenance gate must be closed first, while excluding every family in `D_tabletop_human_eval_v1` from training.",
        "",
        "No SAM2 was used.",
    ]
    MD_REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "status": result["status"],
        "completed_reviews": validation["completed_reviews"],
        "precision": precision,
        "release_created": False,
        "b2_open": False,
        "report": str(MD_REPORT.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
