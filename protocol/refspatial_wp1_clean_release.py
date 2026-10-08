#!/usr/bin/env python3
"""Reproducibly materialize the B1-only D_tabletop_clean_v1 release.

The extractor starts from pinned official RefSpatial Simulator metadata and a
SHA-locked local scene selection. It never exposes source rationale to training,
never emits B2 supervision, and excludes all human-evaluation families.
"""

import argparse
import collections
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from refspatial_build_supplement import json_array
from refspatial_wp1_rules import normalized, point, semantic_review_requirements, target_kind


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/spatial_vlm_refspatial_v1"
AUDIT = BASE / "wp1_source_audit"
RAW = ROOT / "datasets/refspatial_supplement_v1_20260907/source/Simulator/metadata.json"
RAW_PROVENANCE = RAW.with_name(RAW.name + ".provenance.json")
SCENES = ROOT / "datasets/RefSpatial-Tabletop-Large/scenes.jsonl"
DERIVED_METADATA = ROOT / "datasets/RefSpatial-Tabletop-Large/metadata.jsonl"
HUMAN_DECISIONS = AUDIT / "manual_audit_human_decisions.csv"
HUMAN_GATE = AUDIT / "manual_audit_human_gate.json"
LINEAGE = AUDIT / "pinned_source_lineage.json"
AUDIT_QUEUE = AUDIT / "manual_audit_queue.jsonl"
HUMAN_EVAL = ROOT / "datasets/D_tabletop_human_eval_v1/evaluation.jsonl"
CANDIDATE = BASE / "wp1_clean_candidate/samples.jsonl"
OUT = ROOT / "datasets/D_tabletop_clean_v1"

REVISION = "519a6fd43aee2d0ed1366776e50f9456d212f94f"
RELATIONS = ("leftmost_ranking", "rightmost_ranking", "horizontal_ordinal_ranking")
SPLITS = {"train": "train", "validation": "dev", "test": "diagnostic"}
TRAIN_QUOTA = 500


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


def record_fingerprint(record):
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def sample_id(scene_id, instruction, target_xy):
    payload = json.dumps([scene_id, normalized(instruction), tuple(target_xy)]).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def select_disjoint(pools, quota, relation_order=RELATIONS):
    selected = []
    used = set()
    for relation in relation_order:
        chosen = []
        for row in sorted(pools[relation], key=lambda item: item["sample_id"]):
            if row["family_id"] in used:
                continue
            chosen.append(row)
            used.add(row["family_id"])
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise ValueError(f"{relation}: only {len(chosen)}/{quota} disjoint families")
        selected.extend(chosen)
    return selected


def training_record(row):
    return {
        "image": Path(row["image"]).name,
        "depth": Path(row["depth"]).name,
        "conversations": [
            {"from": "human", "value": row["instruction"]},
            {"from": "gpt", "value": row["answer_original"]},
        ],
        "source_sample_id": row["sample_id"],
    }


def build():
    official = json.loads(RAW_PROVENANCE.read_text())
    lineage = json.loads(LINEAGE.read_text())
    human_gate = json.loads(HUMAN_GATE.read_text())
    if official.get("revision") != REVISION or not official.get("sha256_verified_against_lfs"):
        raise RuntimeError("Pinned official metadata hash/revision is not verified")
    if lineage.get("status") != "PINNED_METADATA_LINEAGE_PASS":
        raise RuntimeError("Derived-to-official metadata lineage has not passed")
    if human_gate.get("validation_report", {}).get("status") != "MANUAL_SAMPLE_GATE_PASS":
        raise RuntimeError("Independent human audit sample gate has not passed")

    scenes = {row["id"]: row for row in read_jsonl(SCENES)}
    if len(scenes) != 10323:
        raise RuntimeError(f"Expected 10,323 locked scenes, got {len(scenes)}")
    derived_fingerprints = set()
    with DERIVED_METADATA.open() as handle:
        for line in handle:
            if line.strip():
                derived_fingerprints.add(record_fingerprint(json.loads(line)))

    recovered_points = {
        sid: {tuple(item["xy"]) for item in scene.get("points", [])}
        for sid, scene in scenes.items()
    }
    candidates = []
    seen_qa = set()
    selected_raw_fingerprints = set()
    counters = collections.Counter()
    for raw_index, record in enumerate(json_array(RAW)):
        images = record.get("image", [])
        if len(images) != 1:
            continue
        sid = Path(images[0]).stem
        if sid not in scenes:
            continue
        counters["raw_records_in_locked_scenes"] += 1
        fp = record_fingerprint(record)
        selected_raw_fingerprints.add(fp)
        if fp not in derived_fingerprints:
            counters["raw_selected_record_missing_from_derived"] += 1
            continue
        conv = record.get("conversations", [])
        thinking = record.get("think", [])
        if len(conv) != 2 * len(thinking):
            counters["bad_qa_think_alignment"] += 1
            continue
        for qa_index, thought in enumerate(thinking):
            question_row, answer_row = conv[2 * qa_index:2 * qa_index + 2]
            kind, canonical = target_kind(thought.get("question_type"))
            if kind != "object":
                continue
            instruction = question_row.get("value", "")
            answer_original = answer_row.get("value", "")
            xy = point(answer_original)
            if (xy is None or question_row.get("from") != "human" or answer_row.get("from") != "gpt"
                    or tuple(xy) not in recovered_points[sid]):
                counters["invalid_object_qa"] += 1
                continue
            dedupe_key = (sid, normalized(instruction), normalized(answer_original))
            if dedupe_key in seen_qa:
                counters["duplicate_qa_removed"] += 1
                continue
            seen_qa.add(dedupe_key)
            guard = semantic_review_requirements(instruction, kind, canonical)
            relation = guard.get("direct_relation")
            if relation not in RELATIONS or guard.get("blockers"):
                continue
            split = SPLITS[scenes[sid]["split"]]
            candidates.append({
                "sample_id": sample_id(sid, instruction, xy),
                "scene_id": sid,
                "family_id": f"RefSpatialSimulator:scene:{sid}",
                "split": split,
                "relation": relation,
                "image": f"datasets/RefSpatial-Tabletop-Large/image/{sid}.jpg",
                "depth": f"datasets/RefSpatial-Tabletop-Large/depth/{sid}.png",
                "instruction": instruction,
                "answer_original": answer_original,
                "target_xy": list(xy),
                "official_source_record_index_zero_based": raw_index,
                "source_qa_index_zero_based": qa_index,
                "reference_frame": "image_viewer_left_to_right",
            })
            counters["direct_horizontal_candidates"] += 1

    if selected_raw_fingerprints != derived_fingerprints:
        raise RuntimeError(
            "Locked-scene extraction does not exactly reproduce derived metadata: "
            f"raw-only={len(selected_raw_fingerprints-derived_fingerprints)}, "
            f"derived-only={len(derived_fingerprints-selected_raw_fingerprints)}"
        )

    audit_rows = read_jsonl(AUDIT_QUEUE)
    audit_by_id = {row["sample_id"]: row for row in audit_rows}
    audit_families = {row["family_id"] for row in audit_rows}
    eval_families = {row["family_id"] for row in read_jsonl(HUMAN_EVAL)}
    with HUMAN_DECISIONS.open(newline="") as handle:
        decisions = {row["sample_id"]: row for row in csv.DictReader(handle)}

    pools = {relation: [] for relation in RELATIONS}
    by_id = {row["sample_id"]: row for row in candidates}
    for row in candidates:
        if row["split"] != "train" or row["family_id"] in audit_families | eval_families:
            continue
        pools[row["relation"]].append(row)
    train = select_disjoint(pools, TRAIN_QUOTA)

    # Dev/diagnostic are the accepted human-reviewed direct-ranking samples,
    # minus any family reserved for the independent human evaluation dataset.
    held_out = {"dev": [], "diagnostic": []}
    for sid, decision in decisions.items():
        item = audit_by_id.get(sid)
        row = by_id.get(sid)
        if not item or not row or row["relation"] not in RELATIONS:
            continue
        if row["split"] not in held_out or row["family_id"] in eval_families:
            continue
        if decision.get("completed") != "yes" or decision.get("review_origin") != "human":
            continue
        if decision.get("decision") != "accept" or decision.get("target_correct") != "yes" or decision.get("relation_correct") != "yes":
            continue
        held_out[row["split"]].append(row)

    expected_candidate_ids = {row["sample_id"] for row in read_jsonl(CANDIDATE)}
    train_ids = {row["sample_id"] for row in train}
    if train_ids != expected_candidate_ids:
        raise RuntimeError(
            f"Pinned-raw extraction disagrees with sealed candidate manifest: "
            f"new-only={len(train_ids-expected_candidate_ids)}, candidate-only={len(expected_candidate_ids-train_ids)}"
        )

    staging = OUT.with_name(OUT.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    split_rows = {"train": train, **held_out}
    for split, rows in split_rows.items():
        (staging / f"{split}.json").write_text(json.dumps(
            [training_record(row) for row in rows], ensure_ascii=False, indent=2
        ) + "\n")

    provenance_path = staging / "provenance.jsonl"
    with provenance_path.open("w") as handle:
        for split in ("train", "dev", "diagnostic"):
            for row in split_rows[split]:
                record = {
                    **row,
                    "split": split,
                    "source_repository": official["repo"],
                    "source_revision": REVISION,
                    "source_metadata_sha256": official["sha256"],
                    "source_scene_manifest_sha256": sha256(SCENES),
                    "label_certification": (
                        "STRATIFIED_HUMAN_AUDIT_CERTIFIED_EXTRACTION"
                        if split == "train" else "INDIVIDUAL_HUMAN_REVIEW_ACCEPT"
                    ),
                    "human_reviewed_individually": split != "train",
                    "supervision_scope": "B1_TARGET_GROUNDING_ONLY",
                    "reasoning_depth": None,
                    "b2_eligible": False,
                    "sam2_used": False,
                }
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    split_counts = {split: len(rows) for split, rows in split_rows.items()}
    family_sets = {split: {row["family_id"] for row in rows} for split, rows in split_rows.items()}
    relation_counts = {
        split: dict(collections.Counter(row["relation"] for row in rows))
        for split, rows in split_rows.items()
    }
    checks = {
        "pinned_raw_sha256_verified": official["sha256_verified_against_lfs"],
        "pinned_metadata_lineage_pass": lineage["status"] == "PINNED_METADATA_LINEAGE_PASS",
        "locked_scene_metadata_exactly_reproduced": selected_raw_fingerprints == derived_fingerprints,
        "human_sample_gate_pass": human_gate["validation_report"]["status"] == "MANUAL_SAMPLE_GATE_PASS",
        "train_1500_families": len(family_sets["train"]) == 1500 == len(train),
        "train_500_families_per_relation": all(relation_counts["train"].get(rel) == 500 for rel in RELATIONS),
        "split_family_disjoint": all(
            not (family_sets[a] & family_sets[b])
            for index, a in enumerate(family_sets) for b in list(family_sets)[index + 1:]
        ),
        "human_eval_family_disjoint": not (set().union(*family_sets.values()) & eval_families),
        "train_excludes_manual_audit_families": not (family_sets["train"] & audit_families),
        "heldout_individually_human_accepted": all(
            decisions[row["sample_id"]]["decision"] == "accept" for split in ("dev", "diagnostic") for row in split_rows[split]
        ),
        "all_media_exist": all((ROOT / row[key]).is_file() for rows in split_rows.values() for row in rows for key in ("image", "depth")),
        "all_targets_in_bounds": all(len(row["target_xy"]) == 2 and all(0 <= value <= 1 for value in row["target_xy"]) for rows in split_rows.values() for row in rows),
        "no_reasoning_supervision": True,
        "sam2_not_used": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Release checks failed: {[key for key, value in checks.items() if not value]}")

    files = {
        name: sha256(staging / name)
        for name in ("train.json", "dev.json", "diagnostic.json", "provenance.jsonl")
    }
    manifest = {
        "dataset": "D_tabletop_clean_v1",
        "status": "CLEAN_B1_GROUNDING_RELEASE_PASS",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "training_eligible": True,
        "training_scope": "B1_TARGET_GROUNDING_ONLY",
        "not_for_b2": True,
        "retained_relations": list(RELATIONS),
        "relation_semantics": {
            "leftmost_ranking": "argmin horizontal object-center coordinate in the named candidate class",
            "rightmost_ranking": "argmax horizontal object-center coordinate in the named candidate class",
            "horizontal_ordinal_ranking": "explicit ordinal over the stated image-viewer direction",
        },
        "reference_frame": "image_viewer_left_to_right",
        "data_counts": split_counts,
        "family_counts": {split: len(families) for split, families in family_sets.items()},
        "relation_counts": relation_counts,
        "model_input": ["RGB", "relative_visual_depth", "instruction"],
        "training_label": "normalized target point",
        "offline_only": ["source target", "source metadata indices", "audit decisions"],
        "forbidden_model_input": ["think", "source rationale", "target_xy", "relation label", "audit decision"],
        "source": {
            "repository": official["repo"],
            "revision": REVISION,
            "metadata_sha256": official["sha256"],
            "license": "apache-2.0",
            "scene_manifest_sha256": sha256(SCENES),
            "derived_metadata_sha256": sha256(DERIVED_METADATA),
            "metadata_records_exactly_reproduced": len(derived_fingerprints),
            "original_tabletop_filter_reproduced": False,
            "scope_limitation": "Clean extraction from the locked 10,323-scene derived selection; not an exhaustive rebuild of RefSpatial Simulator.",
        },
        "audit": {
            "human_decisions_sha256": sha256(HUMAN_DECISIONS),
            "human_gate_sha256": sha256(HUMAN_GATE),
            "completed_reviews": human_gate["validation_report"]["completed_reviews"],
            "precision": human_gate["validation_report"]["precision"],
        },
        "exclusions": {
            "human_eval_families": len(eval_families),
            "manual_audit_families_excluded_from_train": len(audit_families),
            "distance_relations": "excluded_nonmetric_local_depth",
            "pairwise_relations": "not present as certified object-target supervision",
        },
        "checks": checks,
        "checks_passed": True,
        "extractor": {
            "path": "protocol/refspatial_wp1_clean_release.py",
            "sha256": sha256(Path(__file__)),
            "rules_path": "protocol/refspatial_wp1_rules.py",
            "rules_sha256": sha256(Path(__file__).with_name("refspatial_wp1_rules.py")),
            "selection": "lowest SHA-derived sample_id, one sample per family, global family disjointness",
        },
        "files": files,
        "sam2_used": False,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (staging / "README.md").write_text(
        "# D_tabletop_clean_v1\n\n"
        "Human-audit-certified RefSpatial horizontal-ranking extraction for B1 target grounding only. "
        "It is not B2 reasoning supervision and is not a final generalization benchmark. "
        "All human-evaluation families are excluded. Depth is a non-metric visual input. No SAM2 was used.\n"
    )
    if OUT.exists():
        old_manifest = OUT / "manifest.json"
        if not old_manifest.exists() or json.loads(old_manifest.read_text()).get("dataset") != "D_tabletop_clean_v1":
            raise RuntimeError(f"Refusing to replace unexpected directory: {OUT}")
        shutil.rmtree(OUT)
    staging.replace(OUT)
    return manifest


def validate_release():
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    errors = []
    if manifest.get("status") != "CLEAN_B1_GROUNDING_RELEASE_PASS":
        errors.append("invalid_release_status")
    for name, expected in manifest.get("files", {}).items():
        path = OUT / name
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"file_hash_mismatch:{name}")
    split_rows = {split: json.loads((OUT / f"{split}.json").read_text()) for split in ("train", "dev", "diagnostic")}
    provenance = read_jsonl(OUT / "provenance.jsonl")
    if sum(map(len, split_rows.values())) != len(provenance):
        errors.append("training_provenance_count_mismatch")
    forbidden = {"think", "thinking", "target_xy", "relation", "audit_decision", "reasoning_depth"}
    for split, rows in split_rows.items():
        for row in rows:
            if forbidden & set(row):
                errors.append(f"privileged_training_field:{split}:{row.get('source_sample_id')}")
            if set(row) != {"image", "depth", "conversations", "source_sample_id"}:
                errors.append(f"unexpected_training_schema:{split}:{row.get('source_sample_id')}")
            if len(row.get("conversations", [])) != 2 or point(row["conversations"][1].get("value", "")) is None:
                errors.append(f"invalid_training_conversation:{split}:{row.get('source_sample_id')}")
    families = collections.defaultdict(set)
    relation_counts = collections.Counter()
    eval_families = {row["family_id"] for row in read_jsonl(HUMAN_EVAL)}
    for row in provenance:
        families[row["split"]].add(row["family_id"])
        if row["split"] == "train":
            relation_counts[row["relation"]] += 1
        if row["family_id"] in eval_families:
            errors.append(f"human_eval_leak:{row['sample_id']}")
        if row.get("b2_eligible") or row.get("reasoning_depth") is not None or row.get("sam2_used"):
            errors.append(f"scope_violation:{row['sample_id']}")
        for key in ("image", "depth"):
            if not (ROOT / row[key]).is_file():
                errors.append(f"missing_media:{row['sample_id']}:{key}")
    names = list(families)
    if any(families[a] & families[b] for index, a in enumerate(names) for b in names[index + 1:]):
        errors.append("cross_split_family_overlap")
    if any(relation_counts[relation] != TRAIN_QUOTA for relation in RELATIONS):
        errors.append("train_relation_quota_mismatch")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "dataset": manifest.get("dataset"),
        "data_counts": {split: len(rows) for split, rows in split_rows.items()},
        "family_counts": {split: len(values) for split, values in families.items()},
        "train_relation_counts": dict(relation_counts),
        "errors": errors[:100],
        "training_eligible": not errors and manifest.get("training_eligible") is True,
        "training_scope": "B1_TARGET_GROUNDING_ONLY",
        "b2_open": False,
        "sam2_used": False,
    }
    (OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate an existing release without rebuilding it")
    args = parser.parse_args()
    if args.check:
        result = validate_release()
    else:
        manifest = build()
        result = validate_release()
        result["built_status"] = manifest["status"]
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
