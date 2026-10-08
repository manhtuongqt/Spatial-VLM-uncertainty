#!/usr/bin/env python3
"""Build audited provisional Day-9 manifests without relabelling source data."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ketquangay/ngay_09/du_lieu/provisional_candidate_r3_v1"
REF = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03/tabletop_qa_full_1500_v1/QA_LABEL_CANDIDATES.jsonl"
DAY7 = ROOT / "ketquangay/ngay_07/du_lieu/REMEDIATION_MANIFEST.jsonl"
DAY8 = ROOT / "ketquangay/ngay_08/du_lieu"

HEADS = ("relation", "coordinate", "log_variance", "answerability", "source", "confidence", "reasoning")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    return path.relative_to(ROOT).as_posix()


def absolute(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blank_mask() -> dict[str, bool]:
    return {head: False for head in HEADS}


def refspatial_rows() -> list[dict]:
    result = []
    for source in load_jsonl(REF):
        rgb = absolute(source["rgb_path"])
        depth = absolute(source["relative_depth_path"])
        mask = blank_mask()
        mask["coordinate"] = True
        result.append({
            "schema_version": 1,
            "record_id": "refspatial/" + source["sample_id"],
            "source_dataset": "RefSpatial-Tabletop-Large",
            "source_role": "b1_point_grounding_train",
            "split": "train",
            "sample_id": source["sample_id"],
            "family_id": source["family_id"],
            "instruction": source["question"],
            "relation": None,
            "relation_candidate": source.get("relation_v3_candidate"),
            "relation_candidate_status": source.get("relation_candidate_status"),
            "answerability": None,
            "target_uv": source["target_uv_b1"],
            "rgb_path": rel(rgb),
            "rgb_sha256": sha256(rgb),
            "metric_depth_path": None,
            "relative_depth_path": rel(depth),
            "relative_depth_sha256": sha256(depth),
            "supervision_mask": mask,
            "label_provenance": "human-accepted RefSpatial QA answer; B1 point grounding only",
            "provisional": True,
        })
    return result


def day7_rows() -> tuple[list[dict], list[str]]:
    source_rows = load_jsonl(DAY7)
    excluded = [x["family_id"] for x in source_rows if x["remediation_split"] == "remediation_diagnostic"]
    result = []
    for source in source_rows:
        if source["remediation_split"] != "remediation_train":
            continue
        mask = {head: bool(source["training_permissions"].get(head, False)) for head in HEADS}
        result.append({
            "schema_version": 1,
            "record_id": "day7-remediation/" + source["sample_id"],
            "source_dataset": "Day7-remediation-accepted249",
            "source_role": "development_remediation_train",
            "source_role_original": source["source_role_original"],
            "split": "train",
            "sample_id": source["sample_id"],
            "family_id": source["family_id"],
            "instruction": source["instruction"],
            "relation": source["relation"],
            "answerability": source["answerability"],
            "target_uv": source["target_uv"],
            "rgb_path": source["rgb_path"],
            "rgb_sha256": source["rgb_sha256"],
            "metric_depth_path": source["metric_depth_path"],
            "metric_depth_sha256": source["metric_depth_sha256"],
            "depth_view_path": source.get("depth_view_path"),
            "supervision_mask": mask,
            "label_provenance": "Day-7 remediation manifest; development-only reuse authorized",
            "provisional": True,
        })
    return result, excluded


def collect_and_inspect(folder_name: str) -> list[dict]:
    sys.path.insert(0, str(ROOT / "workspace/mh_pcrau_v3"))
    from day8_development_r2_qc import collect, inspect
    records = []
    for batch in sorted((DAY8 / folder_name).glob("batch_*")):
        records.extend(collect(batch))
    inspect(records, False)
    return records


def day8_rows() -> tuple[list[dict], list[dict]]:
    original = collect_and_inspect("pilot_r2_v2")
    repair = collect_and_inspect("pilot_r2_v2_repair")
    repair2 = collect_and_inspect("pilot_r2_v2_repair2")
    passed = [x for x in original + repair if not x["qc_errors"]]
    failed = [x for x in original + repair + repair2 if x["qc_errors"]]
    if len(passed) != 61:
        raise RuntimeError(f"Expected exactly 61 semantic-qualified pilot families, got {len(passed)}")
    result = []
    for source in passed:
        found = source["state"] == "FOUND"
        mask = blank_mask()
        mask.update({"relation": True, "answerability": True, "coordinate": found, "log_variance": found})
        split = "train" if source["split"] == "pilot_train" else "validation"
        result.append({
            "schema_version": 1,
            "record_id": "day8-pilot/" + source["scene_id"],
            "source_dataset": "Gazebo-development-r2-v2-pilot",
            "source_role": "semantic_qualified_pilot_" + split,
            "split": split,
            "sample_id": source["scene_id"],
            "family_id": source["family_id"],
            "seed": source["seed"],
            "instruction": source["instruction"],
            "relation": source["relation"],
            "answerability": source["state"],
            "target_uv": source["target_uv"] if found else None,
            "target_visible_pixels": source["target_visible_pixels"],
            "rgb_path": rel(source["rgb"]),
            "rgb_sha256": source["rgb_sha256"],
            "metric_depth_path": rel(source["depth"]),
            "metric_depth_sha256": source["depth_sha256"],
            "semantic_labels_path": rel(source["labels"]),
            "semantic_labels_sha256": source["labels_sha256"],
            "supervision_mask": mask,
            "label_provenance": "Gazebo oracle plus semantic/sensor/floating-object QC",
            "qc_errors": [],
            "provisional": True,
        })
    failed_summary = [{
        "family_id": x["family_id"], "scene_id": x["scene_id"], "split": x["split"],
        "state": x["state"], "qc_errors": x["qc_errors"],
    } for x in failed]
    return result, failed_summary


def duplicates(rows: list[dict], key: str) -> list[str]:
    values = defaultdict(list)
    for row in rows:
        if row.get(key):
            values[str(row[key])].append(row["record_id"])
    return sorted(value for value, records in values.items() if len(records) > 1)


def verify_assets_and_masks(rows: list[dict]) -> tuple[list[str], list[str]]:
    asset_errors, mask_errors = [], []
    for row in rows:
        rid = row["record_id"]
        for path_key, hash_key in (
            ("rgb_path", "rgb_sha256"),
            ("metric_depth_path", "metric_depth_sha256"),
            ("relative_depth_path", "relative_depth_sha256"),
            ("semantic_labels_path", "semantic_labels_sha256"),
        ):
            if not row.get(path_key):
                continue
            path = absolute(row[path_key])
            if not path.is_file():
                asset_errors.append(f"{rid}:{path_key}:missing")
            elif row.get(hash_key) and sha256(path) != row[hash_key]:
                asset_errors.append(f"{rid}:{hash_key}:mismatch")
        mask = row["supervision_mask"]
        if set(mask) != set(HEADS):
            mask_errors.append(f"{rid}:head_set")
        if mask["coordinate"] and row.get("target_uv") is None:
            mask_errors.append(f"{rid}:coordinate_without_target_uv")
        if mask["log_variance"] and not mask["coordinate"]:
            mask_errors.append(f"{rid}:log_variance_without_coordinate")
        if mask["answerability"] and row.get("answerability") is None:
            mask_errors.append(f"{rid}:answerability_without_label")
        if mask["relation"] and row.get("relation") is None:
            mask_errors.append(f"{rid}:relation_without_label")
        if row["source_dataset"] == "RefSpatial-Tabletop-Large" and mask["relation"]:
            mask_errors.append(f"{rid}:uncertified_refspatial_relation_enabled")
    return asset_errors, mask_errors


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref_rows = refspatial_rows()
    remediation_rows, diagnostic_families = day7_rows()
    pilot_rows, failed_pilot = day8_rows()
    rows = ref_rows + remediation_rows + pilot_rows
    train = sorted((x for x in rows if x["split"] == "train"), key=lambda x: x["record_id"])
    val = sorted((x for x in rows if x["split"] == "validation"), key=lambda x: x["record_id"])

    train_families = {x["family_id"] for x in train}
    val_families = {x["family_id"] for x in val}
    train_rgb = {x["rgb_sha256"] for x in train}
    val_rgb = {x["rgb_sha256"] for x in val}
    asset_errors, mask_errors = verify_assets_and_masks(rows)
    excluded_present = sorted(train_families.intersection(diagnostic_families))
    failed_ids = {x["family_id"] for x in failed_pilot}
    failed_present = sorted({x["family_id"] for x in rows}.intersection(failed_ids))

    audit = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "assembled provisional Candidate-R3 train/validation manifests",
        "counts": {"train": len(train), "validation": len(val), "total": len(rows)},
        "train_validation": {
            "family_overlap_count": len(train_families & val_families),
            "family_overlap": sorted(train_families & val_families),
            "exact_rgb_overlap_count": len(train_rgb & val_rgb),
            "exact_rgb_overlap": sorted(train_rgb & val_rgb),
        },
        "within_manifest": {
            "duplicate_family_ids": duplicates(rows, "family_id"),
            "duplicate_rgb_sha256": duplicates(rows, "rgb_sha256"),
            "duplicate_record_ids": duplicates(rows, "record_id"),
        },
        "exclusion_checks": {
            "day7_diagnostic_total": len(diagnostic_families),
            "day7_diagnostic_included": excluded_present,
            "pilot_failed_attempt_rows": len(failed_pilot),
            "pilot_failed_family_included": failed_present,
            "old_249_evaluation_role_used_directly": False,
            "note": "The 201 explicitly authorized Day-7 remediation_train rows retain original one-shot provenance but are used only in their current development-remediation role.",
        },
        "asset_hash_qc": {"passed": not asset_errors, "errors": asset_errors},
        "supervision_mask_qc": {"passed": not mask_errors, "errors": mask_errors},
    }
    hard_fail = (
        audit["train_validation"]["family_overlap_count"]
        or audit["train_validation"]["exact_rgb_overlap_count"]
        or any(audit["within_manifest"].values())
        or excluded_present or failed_present or asset_errors or mask_errors
    )
    audit["status"] = "FAIL" if hard_fail else "PASS"

    supervision = [{
        "record_id": x["record_id"], "split": x["split"], "family_id": x["family_id"],
        "source_dataset": x["source_dataset"], "supervision_mask": x["supervision_mask"],
    } for x in train + val]
    head_counts = {split: {head: sum(x["supervision_mask"][head] for x in subset) for head in HEADS}
                   for split, subset in (("train", train), ("validation", val))}
    summary = {
        "schema_version": 1,
        "status": "READY_PROVISIONAL" if audit["status"] == "PASS" else "DATA_HOLD",
        "development_only": True,
        "day8_status": "DATA_HOLD",
        "counts_by_split": dict(Counter(x["split"] for x in rows)),
        "counts_by_source_and_split": {
            f"{source}|{split}": count for (source, split), count in sorted(Counter((x["source_dataset"], x["split"]) for x in rows).items())
        },
        "pilot_state_by_split": {
            f"{split}|{state}": count for (split, state), count in sorted(Counter((x["split"], x["answerability"]) for x in pilot_rows).items())
        },
        "supervision_counts": head_counts,
        "refspatial_label_policy": {
            "coordinate": "enabled for all 1500 human-accepted B1 point targets",
            "relation": "disabled; relation_v3_candidate is retained as metadata only because it is not V3-certified",
            "answerability_uncertainty": "disabled; source has no four-state or uncertainty labels",
        },
        "limitations": [
            "Validation contains only the 13 semantic-qualified pilot_validation families and is development-only.",
            "Three INSUFFICIENT_EVIDENCE pilot validation families failed semantic QC and are excluded.",
            "This manifest does not open OOF/S1b, G3, calibration, Test, or robot-final evaluation.",
        ],
    }

    dump_jsonl(OUT / "TRAIN_MANIFEST.jsonl", train)
    dump_jsonl(OUT / "VAL_MANIFEST.jsonl", val)
    dump_jsonl(OUT / "SUPERVISION.jsonl", supervision)
    dump_json(OUT / "FAMILY_RGB_LEAKAGE_AUDIT.json", audit)
    dump_json(OUT / "DATASET_SUMMARY.json", summary)
    dump_json(OUT / "EXCLUDED_PILOT_FAILURES.json", failed_pilot)
    print(json.dumps({"output": rel(OUT), "summary": summary, "audit": audit}, indent=2, ensure_ascii=False))
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
