#!/usr/bin/env python3
"""Build/materialize the RefSpatial machine-filtered B1 LoRA pilot.

This intentionally excludes SAM2 and uses only source-structure checks plus
blinded RoboRefer RGB-D coordinate agreement.  It is a machine-filtered pilot,
not D_tabletop_clean_v1 and not B2 supervision.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/spatial_vlm_refspatial_v1"
SOURCE = BASE / "wp1_source_audit"
OUT = BASE / "wp3_machine_lora_pilot"
DATASET = ROOT / "datasets/D_tabletop_machine_v1"
DIRECT = ("leftmost_ranking", "rightmost_ranking", "horizontal_ordinal_ranking")
POSITION = re.compile(r"\[Position\]\s*\[([^\]]+)\]:\s*\[\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)\]")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def query(text):
    return text.split("Your answer should")[0].casefold()


def label_matches(relation, label):
    label = label.casefold()
    if relation == "leftmost_ranking":
        return "leftmost" in label
    if relation == "rightmost_ranking":
        return "rightmost" in label
    return "from left to right" in label or "from right to left" in label


def close(a, b):
    return math.dist(a, b) <= 1e-6


def source_evidence(record, row):
    """Conservative proxy checks; points are not assumed to be object centres."""
    relation, target = row["relation_candidates"][0], tuple(row["target_xy"])
    if relation not in DIRECT:
        return None, "not_direct_horizontal"
    if not any(label_matches(relation, label) for label in row.get("target_labels_offline", [])):
        return None, "target_label_missing_direct_rank"
    best = None
    for think in record.get("think", []):
        positions = [(label, (float(x), float(y))) for label, x, y in POSITION.findall(think.get("thinking", ""))]
        target_labels = [label for label, xy in positions if close(xy, target) and label_matches(relation, label)]
        if not target_labels:
            continue
        horizontal = [xy for label, xy in positions if label_matches(relation, label)]
        unique = sorted({(round(x, 6), round(y, 6)) for x, y in horizontal})
        if len(unique) < 2:
            continue
        gap = min(abs(target[0] - xy[0]) for xy in unique if not close(xy, target))
        candidate = {"source_rank_label": target_labels[0], "source_rank_positions": len(unique),
                     "min_source_point_x_gap": gap}
        if best is None or candidate["min_source_point_x_gap"] > best["min_source_point_x_gap"]:
            best = candidate
    if best is None:
        return None, "no_matching_source_rank_sequence"
    if best["min_source_point_x_gap"] < .03:
        return None, "source_point_x_tie_proxy"
    return best, None


def selected_queue(passed, train_per_relation, eval_per_relation):
    quotas = {"train": train_per_relation, "dev": eval_per_relation, "diagnostic": eval_per_relation}
    grouped = defaultdict(list)
    for row in passed:
        grouped[(row["relation"], row["split"])].append(row)
    chosen = []
    for relation in DIRECT:
        for split in ("train", "dev", "diagnostic"):
            target = quotas[split]
            used_within_relation = set()
            for row in sorted(grouped[(relation, split)], key=lambda item: item["sample_id"]):
                # At most one query per family within a relation. The same
                # scene may contribute one query to another relation, but all
                # of its variants already stay in the same split.
                if row["family_id"] in used_within_relation:
                    continue
                chosen.append(row); used_within_relation.add(row["family_id"])
                if sum(1 for item in chosen if item["relation"] == relation and item["split"] == split) == target:
                    break
            actual = sum(1 for item in chosen if item["relation"] == relation and item["split"] == split)
            if actual < target:
                raise RuntimeError(f"Only {actual}/{target} source-passing families for {relation}/{split}")
    return chosen


def build(args):
    OUT.mkdir(parents=True, exist_ok=True)
    candidates_path = SOURCE / "object_candidates.jsonl"
    overlay = {}
    with (BASE / "wp2_visual_review/semantic_scope_overlay.jsonl").open() as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("direct_relation") in DIRECT and not item.get("blockers"):
                overlay[item["sample_id"]] = item["direct_relation"]
    records = {record["id"]: record for record in json.loads((ROOT / "datasets/RefSpatial-Tabletop-Large/metadata.json").read_text())}
    counts, rejected, passed = Counter(), [], []
    with candidates_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            relation = overlay.get(row["sample_id"])
            if relation is None:
                continue
            counts["direct_total"] += 1
            record = records.get(row["scene_id"])
            if record is None:
                rejected.append({"sample_id": row["sample_id"], "reason": "missing_source_record"}); continue
            evidence, reason = source_evidence(record, row)
            if reason:
                rejected.append({"sample_id": row["sample_id"], "reason": reason}); counts[reason] += 1; continue
            item = {
                "sample_id": row["sample_id"], "scene_id": row["scene_id"], "family_id": row["family_id"],
                "split": row["split"], "relation": relation, "instruction": row["instruction"],
                "target_xy": row["target_xy"], "image": row["image"], "depth": row["depth"],
                "source_evidence": evidence, "machine_filter_stage": "SOURCE_STRUCTURE_PASS_PENDING_ROBOREFER_AGREEMENT",
            }
            passed.append(item); counts["source_structure_pass"] += 1
    with (OUT / "machine_screen_all.jsonl").open("w") as handle:
        for row in passed: handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (OUT / "machine_quarantine_source.jsonl").open("w") as handle:
        for row in rejected: handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    queue = selected_queue(passed, args.train_per_relation, args.eval_per_relation)
    with (OUT / "agreement_queue.jsonl").open("w") as handle:
        for row in queue: handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "dataset": "D_tabletop_machine_v1", "status": "PENDING_ROBOREFER_AGREEMENT",
        "kind": "machine_filtered_b1_pilot_not_clean_release", "sam2_used": False,
        "screened_direct_horizontal_candidates": counts["direct_total"],
        "source_structure_pass": counts["source_structure_pass"],
        "source_quarantine": len(rejected), "source_rejection_counts": dict(counts),
        "agreement_queue_count": len(queue),
        "agreement_queue_by_relation_split": dict(Counter(f"{r['relation']}/{r['split']}" for r in queue)),
        "source_candidates_sha256": digest(candidates_path),
        "selection": {"surplus_train_per_relation": args.train_per_relation, "surplus_eval_per_relation": args.eval_per_relation},
        "constraints": ["family-disjoint across train/dev/diagnostic", "one selected query per family per relation", "no think in model input", "B2 excluded"],
    }
    (OUT / "machine_build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def conversation(row):
    x, y = row["target_xy"]
    return [{"from": "human", "value": row["instruction"]}, {"from": "gpt", "value": f"[({x:.3f}, {y:.3f})]"}]


def materialize(args):
    queue = [json.loads(line) for line in (OUT / "agreement_queue.jsonl").read_text().splitlines()]
    prediction_path = OUT / "roborefer_agreement/predictions.jsonl"
    if not prediction_path.exists():
        raise FileNotFoundError("Run refspatial_machine_agreement.py before materialization")
    predictions = {row["sample_id"]: row for row in (json.loads(line) for line in prediction_path.read_text().splitlines())}
    completed = [predictions.get(row["sample_id"]) for row in queue]
    if any(row is None or row.get("triage") not in {"high_coordinate_agreement", "moderate_coordinate_agreement", "low_coordinate_agreement", "parse_failed", "inference_failed"} for row in completed):
        raise RuntimeError("Agreement queue is incomplete")
    eligible = [row for row in queue if predictions[row["sample_id"]].get("triage") == "high_coordinate_agreement"]
    final = selected_queue(eligible, args.final_train_per_relation, args.final_eval_per_relation)
    DATASET.mkdir(parents=True, exist_ok=True)
    data_rows = defaultdict(list)
    provenance = []
    for row in final:
        output = {"image": Path(row["image"]).name, "depth": Path(row["depth"]).name,
                  "conversations": conversation(row), "source_sample_id": row["sample_id"]}
        data_rows[row["split"]].append(output)
        provenance.append({**row, "roborefer_agreement": predictions[row["sample_id"]],
                           "machine_filter_stage": "MACHINE_FILTERED_PASS"})
    for split in ("train", "dev", "diagnostic"):
        (DATASET / f"{split}.json").write_text(json.dumps(data_rows[split], indent=2) + "\n")
    with (DATASET / "provenance.jsonl").open("w") as handle:
        for row in provenance: handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {
        "dataset": "D_tabletop_machine_v1", "status": "MACHINE_FILTERED_PILOT", "training_eligible": True,
        "not_clean_release": True, "not_for_b2": True, "sam2_used": False,
        "filter": "source rank structure + source point-gap proxy >=0.03 + RoboRefer RGB-D L2 agreement <=0.08",
        "model_input": ["RGB", "depth", "instruction"], "offline_only": ["source think", "source target for agreement"],
        "data_counts": {split: len(data_rows[split]) for split in data_rows},
        "family_counts": {split: len({row["family_id"] for row in provenance if row["split"] == split}) for split in data_rows},
        "raw_media_roots": {"image": "datasets/RefSpatial-Tabletop-Large/image", "depth": "datasets/RefSpatial-Tabletop-Large/depth"},
        "files": {f"{split}.json": digest(DATASET / f"{split}.json") for split in ("train", "dev", "diagnostic")},
    }
    (DATASET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (DATASET / "README.md").write_text("# D_tabletop_machine_v1\n\nMachine-filtered B1 LoRA pilot. It is not D_tabletop_clean_v1, has no independent human label certification, and must not be used for B2 reasoning supervision or final claims.\n")
    report = {"status": "MATERIALIZED", **manifest, "agreement_queue": len(queue), "high_agreement_candidates": len(eligible)}
    (OUT / "materialization_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "materialize"))
    parser.add_argument("--train-per-relation", type=int, default=300)
    parser.add_argument("--eval-per-relation", type=int, default=40)
    parser.add_argument("--final-train-per-relation", type=int, default=250)
    parser.add_argument("--final-eval-per-relation", type=int, default=30)
    args = parser.parse_args()
    build(args) if args.command == "build" else materialize(args)
