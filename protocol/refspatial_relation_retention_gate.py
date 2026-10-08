#!/usr/bin/env python3
"""Evaluate whether reviewed RefSpatial relations may enter B2 or clean release.

Candidate-family counts are deliberately kept separate from independently
certified clean family support.  This program never creates a dataset.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/spatial_vlm_refspatial_v1/wp2_visual_review"
REQUIRED_FAMILIES = 500
REQUIRED_AUDIT_PER_RELATION = 50


def main():
    queue = [json.loads(line) for line in (OUT / "next_audit_queue.jsonl").read_text().splitlines()]
    decisions = {row["sample_id"]: row for row in csv.DictReader((OUT / "next_audit_decisions.csv").open())}
    candidate_coverage = list(csv.DictReader((OUT / "direct_relation_coverage.csv").open()))
    ontology = json.loads((OUT / "ontology_v1.json").read_text())
    by_relation = defaultdict(list)
    for row in queue:
        by_relation[row["audit_strata"][0]].append((row, decisions.get(row["sample_id"], {})))
    candidate_train = {
        row["direct_relation"]: int(row["candidate_families"])
        for row in candidate_coverage if row["split"] == "train"
    }
    rows = []
    for relation in sorted(by_relation):
        entries = by_relation[relation]
        reviewed = [decision for _, decision in entries if decision.get("decision")]
        human = [decision for decision in reviewed if decision.get("review_origin") == "human"]
        accepted_human_train_families = {
            source["family_id"] for source, decision in entries
            if source["split"] == "train" and decision.get("review_origin") == "human" and decision.get("decision") == "accept"
        }
        field_values = [d.get("relation_correct") for d in reviewed if d.get("relation_correct") not in {"", "na"}]
        precision = field_values.count("yes") / len(field_values) if field_values else None
        pass_relation = (
            len(human) >= REQUIRED_AUDIT_PER_RELATION
            and precision is not None and precision >= .95
            and len(accepted_human_train_families) >= REQUIRED_FAMILIES
            and ontology["source_semantics"]["not_documented"] == []
        )
        rows.append({
            "relation": relation,
            "candidate_train_families": candidate_train.get(relation, 0),
            "ai_reviewed": len(reviewed) - len(human),
            "human_reviewed": len(human),
            "ai_relation_precision": precision,
            "certified_human_train_families": len(accepted_human_train_families),
            "required_train_families": REQUIRED_FAMILIES,
            "status": "RETAIN_FOR_B2" if pass_relation else "OUTSIDE_B2",
        })
    with (OUT / "relation_retention_gate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {
        "release": "D_tabletop_clean_v1",
        "created": False,
        "training_eligible": False,
        "ontology": ontology["ontology_version"],
        "source_generator_semantics_verified": not bool(ontology["source_semantics"]["not_documented"]),
        "relation_gate": rows,
        "retained_relations": [row["relation"] for row in rows if row["status"] == "RETAIN_FOR_B2"],
        "next_action": "No relation has passed. Keep all three rankings outside B2; do not materialize D_tabletop_clean_v1.",
    }
    (OUT / "relation_retention_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    gate_path = OUT / "clean_release_gate.json"
    gate = json.loads(gate_path.read_text())
    gate["relation_retention_gate"] = "NO_RELATION_RETAINED_FOR_B2"
    gate["relation_retention_gate_path"] = "relation_retention_gate.json"
    gate["next_action"] = report["next_action"]
    gate_path.write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
