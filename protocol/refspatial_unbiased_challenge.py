#!/usr/bin/env python3
"""Build a B0-selection-free RefSpatial RGB-D challenge set without SAM2.

The scored primary set uses source structure only.  It never reads or uses a
RoboRefer agreement prediction.  Tie and missing-source-sequence examples are
kept in an unscored manual-audit queue because their labels are not reliable
enough to be automatic point-evaluation ground truth.
"""
import csv
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results/spatial_vlm_refspatial_v1/wp3_machine_lora_pilot"
SOURCE = ROOT / "results/spatial_vlm_refspatial_v1/wp1_source_audit"
VISUAL = ROOT / "results/spatial_vlm_refspatial_v1/wp2_visual_review"
DATASET = ROOT / "datasets/D_tabletop_machine_v1"
OUT = ROOT / "results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge"
ORDINAL = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", re.I)
ORDINAL_VALUE = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                 "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_key(row):
    return hashlib.sha256(("challenge-v1:" + row["sample_id"]).encode()).hexdigest()


def ordinal_value(instruction):
    match = ORDINAL.search(instruction)
    return ORDINAL_VALUE.get(match.group(1).casefold(), 0) if match else 0


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def choose(pool, count, selected_families):
    chosen = []
    for row in sorted(pool, key=rank_key):
        if row["family_id"] in selected_families:
            continue
        selected_families.add(row["family_id"])
        chosen.append(row)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(f"Challenge quota shortfall: required {count}, selected {len(chosen)}")
    return chosen


def normalize_primary(row, stratum):
    gap = row["source_evidence"]["min_source_point_x_gap"]
    return {
        **row,
        "challenge_role": "PRIMARY_SCORED_MACHINE_SOURCE_ONLY",
        "challenge_stratum": stratum,
        "ordinal_value": ordinal_value(row["instruction"]),
        "source_gap_proxy": gap,
        "b0_selection_free": True,
        "never_in_b0_agreement_queue": True,
        "never_in_b1_train_dev_diagnostic": True,
        "training_eligible": False,
        "b2_eligible": False,
        "label_status": "SOURCE_STRUCTURE_ONLY_HUMAN_CERTIFICATION_PENDING",
    }


def normalize_manual(row, reason):
    return {
        "sample_id": row["sample_id"], "scene_id": row["scene_id"], "family_id": row["family_id"],
        "split": row["split"], "relation": row["relation"], "instruction": row["instruction"],
        "target_xy": row["target_xy"], "image": row["image"], "depth": row["depth"],
        "challenge_role": "UNSCORED_MANUAL_TIE_OR_AMBIGUITY_AUDIT",
        "challenge_stratum": reason,
        "b0_selection_free": True, "never_in_b0_agreement_queue": True,
        "never_in_b1_train_dev_diagnostic": True, "training_eligible": False,
        "b2_eligible": False, "label_status": "UNVERIFIED_SOURCE_AMBIGUITY",
    }


def write_manual_gallery(rows):
    parts = ["<!doctype html><meta charset='utf-8'><title>Challenge tie and ambiguity audit</title>",
             "<style>body{font:16px sans-serif;max-width:1100px;margin:auto}article{border-bottom:2px solid #777;padding:20px}img{max-width:960px}pre{white-space:pre-wrap}</style>",
             "<h1>100 tie/ambiguity challenge cases — unscored pending human audit</h1>",
             "<p>Read RGB and instruction before opening the source target. These cases are excluded from automatic B0/B1 point metrics. Record decisions in <code>manual_tie_ambiguity_decisions.csv</code>.</p>"]
    for index, row in enumerate(rows, 1):
        image = "../../../" + row["image"]
        parts.append(
            f"<article><h2>{index}. {html.escape(row['sample_id'])}</h2>"
            f"<p><b>Stratum:</b> {html.escape(row['challenge_stratum'])}; <b>relation:</b> {html.escape(row['relation'])}</p>"
            f"<p>{html.escape(row['instruction'])}</p><img loading='lazy' src='{html.escape(image)}'>"
            f"<details><summary>Unverified source target</summary><pre>{html.escape(json.dumps(row['target_xy']))}</pre></details></article>"
        )
    (OUT / "MANUAL_TIE_AMBIGUITY_AUDIT.html").write_text("\n".join(parts) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    screen_path = PILOT / "machine_screen_all.jsonl"
    agreement_path = PILOT / "agreement_queue.jsonl"
    provenance_path = DATASET / "provenance.jsonl"
    candidates_path = SOURCE / "object_candidates.jsonl"
    quarantine_path = PILOT / "machine_quarantine_source.jsonl"
    overlay_path = VISUAL / "semantic_scope_overlay.jsonl"

    screen = load_jsonl(screen_path)
    agreement = load_jsonl(agreement_path)
    used = load_jsonl(provenance_path)
    agreement_ids = {row["sample_id"] for row in agreement}
    excluded_families = {row["family_id"] for row in agreement} | {row["family_id"] for row in used}
    free = [row for row in screen if row["split"] in {"dev", "diagnostic"}
            and row["sample_id"] not in agreement_ids and row["family_id"] not in excluded_families]

    selected_families = set()
    primary = []
    quotas = [
        ("dev_extremum_left_scarce_source_pass", 5, lambda r: r["split"] == "dev" and r["relation"] == "leftmost_ranking"),
        ("dev_extremum_right_scarce_source_pass", 7, lambda r: r["split"] == "dev" and r["relation"] == "rightmost_ranking"),
        ("diagnostic_extremum_left_scarce_source_pass", 6, lambda r: r["split"] == "diagnostic" and r["relation"] == "leftmost_ranking"),
        ("dev_ordinal_rank_ge3_gap_lt008", 76, lambda r: r["split"] == "dev" and r["relation"] == "horizontal_ordinal_ranking"
         and ordinal_value(r["instruction"]) >= 3 and r["source_evidence"]["min_source_point_x_gap"] < .08),
        ("diagnostic_ordinal_rank_ge2_gap_lt008", 86, lambda r: r["split"] == "diagnostic" and r["relation"] == "horizontal_ordinal_ranking"
         and ordinal_value(r["instruction"]) >= 2 and r["source_evidence"]["min_source_point_x_gap"] < .08),
    ]
    quota_report = []
    for stratum, wanted, predicate in quotas:
        pool = [row for row in free if predicate(row)]
        chosen = choose(pool, wanted, selected_families)
        primary.extend(normalize_primary(row, stratum) for row in chosen)
        quota_report.append({"stratum": stratum, "requested": wanted, "candidate_rows": len(pool),
                             "selected": len(chosen), "candidate_families": len({r['family_id'] for r in pool})})
    primary.sort(key=lambda row: (row["split"], row["challenge_stratum"], rank_key(row)))

    candidate_by_id = {row["sample_id"]: row for row in load_jsonl(candidates_path)}
    overlay = {row["sample_id"]: row.get("direct_relation") for row in load_jsonl(overlay_path)}
    quarantine = {row["sample_id"]: row["reason"] for row in load_jsonl(quarantine_path)}
    manual_pool = []
    for sample_id, reason in quarantine.items():
        if reason not in {"source_point_x_tie_proxy", "no_matching_source_rank_sequence"}:
            continue
        row = candidate_by_id.get(sample_id)
        relation = overlay.get(sample_id)
        if not row or relation not in {"leftmost_ranking", "rightmost_ranking", "horizontal_ordinal_ranking"}:
            continue
        if row["split"] not in {"dev", "diagnostic"} or row["family_id"] in excluded_families:
            continue
        manual_pool.append({"sample_id": row["sample_id"], "scene_id": row["scene_id"], "family_id": row["family_id"],
                            "split": row["split"], "relation": relation, "instruction": row["instruction"],
                            "target_xy": row["target_xy"], "image": row["image"], "depth": row["depth"],
                            "source_quarantine_reason": reason})
    manual_families = set(selected_families)
    manual = []
    for reason in ("source_point_x_tie_proxy", "no_matching_source_rank_sequence"):
        for split in ("dev", "diagnostic"):
            pool = [row for row in manual_pool if row["source_quarantine_reason"] == reason and row["split"] == split]
            manual.extend(normalize_manual(row, reason) for row in choose(pool, 25, manual_families))
    manual.sort(key=lambda row: (row["challenge_stratum"], row["split"], rank_key(row)))
    write_manual_gallery(manual)

    with (OUT / "primary_challenge.jsonl").open("w") as handle:
        for row in primary:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (OUT / "manual_tie_ambiguity_queue.jsonl").open("w") as handle:
        for row in manual:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    fields = ["sample_id", "scene_id", "split", "relation", "challenge_stratum", "reviewer", "reviewed_at",
              "review_origin", "target_correct", "selected_instance_correct", "tie_or_ambiguity",
              "reference_frame", "decision", "human_target_x", "human_target_y",
              "correction_action", "completed", "notes"]
    decisions_path = OUT / "manual_tie_ambiguity_decisions.csv"
    existing = {}
    if decisions_path.exists():
        with decisions_path.open(newline="") as handle:
            existing = {row["sample_id"]: row for row in csv.DictReader(handle)}
    with decisions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in manual:
            preserved = existing.get(row["sample_id"], {})
            writer.writerow({key: preserved.get(key, row.get(key, "")) for key in fields})

    protocol_path = OUT / "CHALLENGE_PROTOCOL.md"
    protocol_path.write_text("""# RefSpatial unbiased challenge v1

`primary_challenge.jsonl` has 180 RGB-D source-structure examples that were selected without reading any RoboRefer/B0 prediction. Every primary family is excluded from the B1 train/dev/diagnostic dataset and the former B0 agreement queue.

The primary set is source-labelled only. Report its result as an internal machine-source stress evaluation, never as human-certified accuracy, B2 evidence, or Gazebo transfer.

`manual_tie_ambiguity_queue.jsonl` contains 100 source-quarantined tie or missing-rank-sequence cases. It is unscored until independent human decisions are complete in `manual_tie_ambiguity_decisions.csv`.

The evaluation order is fixed: first lock this manifest, then run B0 and B1 with the same RGB-D prompt and decoding protocol. B0-low-agreement cases may be reported only as a post-hoc stress subgroup; they must not replace the full primary paired comparison.
""")
    manifest = {
        "dataset": "D_tabletop_unbiased_challenge_v1",
        "status": "PRIMARY_MACHINE_SOURCE_ONLY_MANUAL_AMBIGUITY_AUDIT_PENDING",
        "purpose": "B0/B1 challenge evaluation before any Gazebo promotion decision",
        "sam2_used": False,
        "primary_count": len(primary), "manual_tie_ambiguity_count": len(manual),
        "primary_by_split": dict(Counter(row["split"] for row in primary)),
        "primary_by_relation": dict(Counter(row["relation"] for row in primary)),
        "primary_by_stratum": dict(Counter(row["challenge_stratum"] for row in primary)),
        "manual_by_reason": dict(Counter(row["challenge_stratum"] for row in manual)),
        "primary_unique_families": len({row["family_id"] for row in primary}),
        "manual_unique_families": len({row["family_id"] for row in manual}),
        "exclusions": {
            "b0_agreement_queue_samples": len(agreement_ids),
            "b0_agreement_or_b1_dataset_families": len(excluded_families),
            "primary_overlap_b0_agreement_queue": len({r['sample_id'] for r in primary} & agreement_ids),
            "primary_overlap_b1_dataset_families": len({r['family_id'] for r in primary} & {r['family_id'] for r in used}),
        },
        "selection": "fixed SHA-256 ordering over source-only strata; no RoboRefer outputs read",
        "not_for_training": True, "not_for_b2": True,
        "label_policy": "Primary labels are source-structure-only and must be reported as machine-source evaluation; tie/ambiguity queue is unscored until human review.",
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in [screen_path, agreement_path, provenance_path, candidates_path, quarantine_path, overlay_path]},
        "outputs": {}, "quotas": quota_report,
    }
    for path in [OUT / "primary_challenge.jsonl", OUT / "manual_tie_ambiguity_queue.jsonl", OUT / "manual_tie_ambiguity_decisions.csv", OUT / "MANUAL_TIE_AMBIGUITY_AUDIT.html", protocol_path]:
        manifest["outputs"][path.name] = sha256(path)
    (OUT / "challenge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
