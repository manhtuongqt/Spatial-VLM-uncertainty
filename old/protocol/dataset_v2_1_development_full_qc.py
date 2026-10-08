#!/usr/bin/env python3
"""Full scientific QC gate for the materialized Dataset V2.1 development set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset_v2_1_development_materialize import materialize
from wp2_common import (
    find_forbidden_inference_keys,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)


PROTOCOL_ID = "roborefer_dataset_v2_1_1_development_capture_400"
BATCH_IDS = ["canary_000", "batch_001", "batch_002", "batch_003", "batch_004"]
EXPECTED_STATE = {
    "train": {"FOUND": 128, "INSUFFICIENT_EVIDENCE": 80, "AMBIGUOUS": 56, "ABSENT": 56},
    "dev": {"FOUND": 32, "INSUFFICIENT_EVIDENCE": 20, "AMBIGUOUS": 14, "ABSENT": 14},
}


class V21FullQCError(RuntimeError):
    pass


def collect_refs(value: Any) -> list[dict[str, Any]]:
    refs = []
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"}.issubset(value):
            refs.append(value)
        for child in value.values():
            refs.extend(collect_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.extend(collect_refs(child))
    return refs


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise V21FullQCError(f"artifact path escapes dataset root: {relative}")
    return path


def validate_raw_and_batches(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    raw = read_json(root / "raw/raw_capture_manifest.json")
    if (
        raw.get("protocol_id") != PROTOCOL_ID or raw.get("capture_count") != 800
        or raw.get("expected_capture_count") != 800 or raw.get("complete") is not True
    ):
        raise V21FullQCError("raw capture manifest is not complete at 800")
    raw_ids = [item["capture_id"] for item in raw["captures"]]
    planned_ids = [item["capture_id"] for item in plan["captures"]]
    if len(raw_ids) != len(set(raw_ids)) or set(raw_ids) != set(planned_ids):
        raise V21FullQCError("raw/planned capture identity set differs")
    reports = []
    for batch_id in BATCH_IDS:
        path = root / "report_assets/checkpoints/batch_qc" / f"{batch_id}.json"
        report = read_json(path)
        if not report.get("passed") or report.get("batch_id") != batch_id:
            raise V21FullQCError(f"batch QC is not PASS: {batch_id}")
        reports.append(report)
    shutdown = read_json(root / "report_assets/checkpoints/shutdown_sequence.json")
    if not shutdown.get("ordered_shutdown_complete"):
        raise V21FullQCError("final ordered shutdown is incomplete")
    return {
        "passed": True, "raw_capture_count": 800,
        "batch_reports": [{
            "batch_id": value["batch_id"],
            "valid_capture_count": value["valid_capture_count"],
            "relation_checks": value["relation_checks"],
            "relation_not_evaluable_allowed": value["relation_not_evaluable_allowed"],
            "max_timestamp_spread_sec": value["max_timestamp_spread_sec"],
        } for value in reports],
        "relation_checks": sum(int(value["relation_checks"]) for value in reports),
        "relation_failures": sum(len(value["relation_failures"]) for value in reports),
        "max_timestamp_spread_sec": max(float(value["max_timestamp_spread_sec"]) for value in reports),
        "ordered_shutdown_complete": True,
    }


def validate_records(root: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = read_json(root / "dataset_index.json")
    if (
        index.get("protocol_id") != PROTOCOL_ID or index.get("record_count") != 2000
        or index.get("family_count") != 400
    ):
        raise V21FullQCError("dataset index count/protocol differs")
    records = []
    hash_errors, leakage = [], []
    family_counts: Counter[str] = Counter()
    split_family: defaultdict[str, set[str]] = defaultdict(set)
    state_by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    category_by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    relation_by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    asset_by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    mask_stats = Counter()
    clean_seen: set[str] = set()

    family_by_id = {item["family_id"]: item for item in manifest["families"]}
    for item in index["records"]:
        record_path = safe_path(root, item["record_path"])
        if not record_path.is_file() or sha256_file(record_path) != item["record_sha256"]:
            hash_errors.append(f"record:{item['sample_id']}")
            continue
        record = read_json(record_path)
        records.append(record)
        if record.get("protocol_id") != PROTOCOL_ID or record.get("sample_id") != item["sample_id"]:
            hash_errors.append(f"identity:{item['sample_id']}")
        if record["split"] not in {"train", "dev"} or record["split"] != item["split"]:
            hash_errors.append(f"split:{item['sample_id']}")
        for ref in collect_refs(record):
            path = safe_path(root, str(ref["path"]))
            if (
                not path.is_file() or path.stat().st_size != int(ref["bytes"])
                or sha256_file(path) != ref["sha256"]
            ):
                hash_errors.append(f"artifact:{item['sample_id']}:{ref['path']}")
        findings = find_forbidden_inference_keys(record["inference_payload"])
        if findings:
            leakage.append({"sample_id": item["sample_id"], "paths": findings})

        family_counts[record["family_id"]] += 1
        split_family[record["split"]].add(record["family_id"])
        if record["variant"] == "clean":
            clean_seen.add(record["family_id"])
            state = record["evaluator_only"]["uncertainty_label"]["state"]
            state_by_split[record["split"]][state] += 1
            category_by_split[record["split"]][record["family_category"]] += 1
            relation_by_split[record["split"]][
                record["evaluator_only"]["spatial_label"]["relations"][0]
            ] += 1
            family = family_by_id[record["family_id"]]
            for object_id in family["primary_scene"]["active_scene_ids"]:
                asset_by_split[record["split"]][object_id] += 1

        target_ref = record["evaluator_only"]["masks"]["target"]
        target = cv2.imread(str(safe_path(root, target_ref["path"])), cv2.IMREAD_UNCHANGED)
        interior_ref = record["evaluator_only"]["masks"]["target_interior"]
        interior = cv2.imread(str(safe_path(root, interior_ref["path"])), cv2.IMREAD_UNCHANGED)
        if target is None or interior is None:
            hash_errors.append(f"mask-decode:{item['sample_id']}")
            continue
        target_pixels = int(np.count_nonzero(target))
        interior_pixels = int(np.count_nonzero(interior))
        state = record["evaluator_only"]["uncertainty_label"]["state"]
        if state in {"FOUND", "AMBIGUOUS"} and target_pixels == 0:
            hash_errors.append(f"empty-positive-target:{item['sample_id']}")
        if state == "FOUND" and interior_pixels == 0:
            hash_errors.append(f"empty-found-interior:{item['sample_id']}")
        if state == "ABSENT" and target_pixels != 0:
            hash_errors.append(f"nonempty-absent-target:{item['sample_id']}")
        mask_stats[f"{state}_records"] += 1
        mask_stats[f"{state}_target_pixels"] += target_pixels

    if hash_errors:
        raise V21FullQCError(f"record/artifact/mask validation failed: {hash_errors[:20]}")
    if leakage:
        raise V21FullQCError(f"oracle leakage found: {leakage[:10]}")
    if len(records) != 2000 or len(family_counts) != 400 or any(value != 5 for value in family_counts.values()):
        raise V21FullQCError("five variants per independent family invariant failed")
    if {key: len(value) for key, value in split_family.items()} != {"train": 320, "dev": 80}:
        raise V21FullQCError("train/dev family counts differ")
    if split_family["train"] & split_family["dev"]:
        raise V21FullQCError("family leakage across train/dev")
    if {key: dict(value) for key, value in state_by_split.items()} != EXPECTED_STATE:
        raise V21FullQCError(f"primary answerability distribution differs: {state_by_split}")
    return ({
        "passed": True, "record_count": 2000, "family_count": 400,
        "split_family_counts": {key: len(value) for key, value in sorted(split_family.items())},
        "primary_answerability_by_split": {
            key: dict(sorted(value.items())) for key, value in sorted(state_by_split.items())
        },
        "family_category_by_split": {
            key: dict(sorted(value.items())) for key, value in sorted(category_by_split.items())
        },
        "relation_by_split": {
            key: dict(sorted(value.items())) for key, value in sorted(relation_by_split.items())
        },
        "asset_by_split": {
            key: dict(sorted(value.items())) for key, value in sorted(asset_by_split.items())
        },
        "mask_statistics": dict(sorted(mask_stats.items())),
        "oracle_leakage_findings": [],
    }, records)


def validate_split_leakage(manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    families = manifest["families"]
    family_split = {item["family_id"]: item["split"] for item in families}
    instruction_splits: defaultdict[str, set[str]] = defaultdict(set)
    template_splits: defaultdict[str, set[str]] = defaultdict(set)
    layout_splits: defaultdict[str, set[str]] = defaultdict(set)
    seed_splits: defaultdict[int, set[str]] = defaultdict(set)
    capture_splits: defaultdict[str, set[str]] = defaultdict(set)
    for family in families:
        split = family["split"]
        for spec in family["variant_specs"]:
            instruction_splits[spec["instruction"]].add(split)
            template_splits[spec["language_template_family_id"]].add(split)
        for value in family["seed_bundle"].values():
            seed_splits[int(value)].add(split)
        seed_splits[int(family["selected_layout_seed"])].add(split)
    for capture in plan["captures"]:
        split = capture["split"]
        layout_splits[capture["layout_instance_fingerprint_sha256"]].add(split)
        capture_splits[capture["capture_id"]].add(split)
        if family_split[capture["family_id"]] != split:
            raise V21FullQCError(f"capture family split differs: {capture['capture_id']}")
    overlaps = {
        "instruction": sum(len(value) > 1 for value in instruction_splits.values()),
        "template": sum(len(value) > 1 for value in template_splits.values()),
        "layout": sum(len(value) > 1 for value in layout_splits.values()),
        "seed": sum(len(value) > 1 for value in seed_splits.values()),
        "capture": sum(len(value) > 1 for value in capture_splits.values()),
    }
    if any(overlaps.values()):
        raise V21FullQCError(f"train/dev leakage detected: {overlaps}")
    return {"passed": True, "cross_split_overlap_counts": overlaps}


def validate_assets(workspace: Path, manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    partition = read_json(workspace / "protocol/dataset_expansion_v2_asset_partition.json")
    heldout = set(partition["test_ood_asset_holdout"]["asset_ids"])
    excluded = set(partition["excluded_assets"]["asset_ids"])
    registry = plan["object_registry"]
    active = {
        row["id"] for capture in plan["captures"] for name, row in registry.items()
        if float(capture["layout"][name][1]) <= 0.65
    }
    bad = active & (heldout | excluded)
    if bad:
        raise V21FullQCError(f"heldout/excluded assets active in development: {sorted(bad)}")
    return {
        "passed": True, "active_asset_ids": sorted(active),
        "heldout_active": sorted(active & heldout), "excluded_active": sorted(active & excluded),
    }


def deterministic_replay(root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dataset_v2_1_development_replay_") as value:
        replay_root = Path(value)
        materialize(root, replay_root)
        tops = (
            "family_manifest.json", "capture_plan.json", "object_registry.json",
            "execution_lock.json", "dataset_index.json", "derived", "media", "evaluator", "records",
        )
        relative_files = []
        for top in tops:
            source = root / top
            if source.is_file():
                relative_files.append(Path(top))
            elif source.is_dir():
                relative_files.extend(
                    path.relative_to(root) for path in source.rglob("*") if path.is_file()
                )
        mismatches = []
        for relative in sorted(relative_files):
            replay = replay_root / relative
            if not replay.is_file() or sha256_file(root / relative) != sha256_file(replay):
                mismatches.append(str(relative))
        if mismatches:
            raise V21FullQCError(f"deterministic materialization replay differs: {mismatches[:20]}")
        return {"passed": True, "files_compared": len(relative_files), "mismatches": []}


def write_tables(root: Path, records_summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, str]:
    tables = root / "report_assets/tables"
    tables.mkdir(parents=True, exist_ok=True)
    table1 = tables / "table_01_dataset_independence.csv"
    with table1.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "split", "independent_families", "dependent_samples", "FOUND", "AMBIGUOUS",
            "ABSENT", "INSUFFICIENT_EVIDENCE", "depth_dependent", "OOD", "scientific_status",
        ])
        for split in ("train", "dev"):
            families = [item for item in manifest["families"] if item["split"] == split]
            states = Counter(item["primary_answerability_stratum"] for item in families)
            writer.writerow([
                split, len(families), 5 * len(families), states["FOUND"], states["AMBIGUOUS"],
                states["ABSENT"], states["INSUFFICIENT_EVIDENCE"],
                sum(bool(item["depth_dependent"]) for item in families), 0,
                "OBSERVED_DEVELOPMENT_FULL_QC_PASS",
            ])
        for split in ("calibration", "test_iid", "test_ood"):
            writer.writerow([split, 0, 0, 0, 0, 0, 0, 0, 0, "NOT_CREATED"])

    relation = tables / "table_01b_relation_family_balance.csv"
    with relation.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "relation", "independent_families"])
        for split, values in records_summary["relation_by_split"].items():
            for name, count in values.items():
                writer.writerow([split, name, count])

    category = tables / "table_01c_family_category_balance.csv"
    with category.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "family_category", "independent_families"])
        for split, values in records_summary["family_category_by_split"].items():
            for name, count in values.items():
                writer.writerow([split, name, count])
    status = tables / "TABLES_02_TO_06_STATUS.json"
    write_json(status, {
        "schema_version": 1, "status": "NOT_RUN",
        "reason": "Development dataset QC does not authorize model training or scientific evaluation.",
        "tables": {str(value): "NOT_RUN" for value in range(2, 7)},
    })
    return {
        "table_01": str(table1.relative_to(root)),
        "table_01b": str(relation.relative_to(root)),
        "table_01c": str(category.relative_to(root)),
        "tables_02_to_06": str(status.relative_to(root)),
    }


def write_markdown(root: Path, report: dict[str, Any]) -> None:
    batches = "\n".join(
        f"- `{item['batch_id']}`: {item['valid_capture_count']} capture; "
        f"{item['relation_checks']} relation checks"
        for item in report["raw_and_batch_qc"]["batch_reports"]
    )
    text = f"""# Dataset V2.1 development full-QC report

- Decision: `{report['decision']}`
- Independent families: {report['family_count']}
- Materialized samples: {report['sample_count']}
- Real RGB-D/semantic captures: {report['raw_capture_count']}
- Observable relation checks: {report['raw_and_batch_qc']['relation_checks']}
- Relation failures: {report['raw_and_batch_qc']['relation_failures']}
- Deterministic replay files compared: {report['deterministic_replay']['files_compared']}
- Training performed: `false`
- Calibration/Test created: `false`

## Batch gates

{batches}

Only real sensor QC images are stored under `report_assets/real_capture_qc/`.
Table 1 contains observed dataset statistics. Tables 2–6 remain `NOT_RUN`.
This PASS authorizes the next gate, `GO_DEVELOPMENT_TRAIN`, using 320 train
families and 80 dev families; it does not authorize Calibration or Test access.
"""
    (root / "DATASET_V2_1_DEVELOPMENT_FULL_QC_REPORT.md").write_text(text, encoding="utf-8")


def run_full_qc(root: Path, run_replay: bool) -> dict[str, Any]:
    workspace = Path(__file__).resolve().parents[1]
    manifest = read_json(workspace / "protocol/dataset_v2_1_development_manifest.json")
    plan = read_json(workspace / "protocol/dataset_v2_1_development_capture_plan.json")
    lock = read_json(workspace / "protocol/dataset_v2_1_development_execution_lock.json")
    if lock.get("training_authorized") or lock.get("calibration_test_authorized"):
        raise V21FullQCError("pre-training lock boundary was violated")
    for relative, expected in lock["locked_artifact_sha256"].items():
        path = workspace / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise V21FullQCError(f"execution-locked artifact changed: {relative}")
    raw_summary = validate_raw_and_batches(root, plan)
    records_summary, _ = validate_records(root, manifest)
    leakage = validate_split_leakage(manifest, plan)
    assets = validate_assets(workspace, manifest, plan)
    replay = deterministic_replay(root) if run_replay else {
        "passed": None, "files_compared": 0, "mismatches": [], "skipped": True,
    }
    tables = write_tables(root, records_summary, manifest)
    return {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(), "passed": True,
        "decision": "GO_DEVELOPMENT_TRAIN",
        "scientific_status": "OFFICIAL_DEVELOPMENT_DATASET_QC_PASS_TESTS_STILL_SEALED",
        "family_count": 400, "sample_count": 2000, "raw_capture_count": 800,
        "raw_and_batch_qc": raw_summary,
        "record_mask_ontology_oracle_qc": records_summary,
        "train_dev_leakage_qc": leakage,
        "asset_partition_qc": assets,
        "deterministic_replay": replay,
        "tables": tables,
        "figures_policy": "REAL_CAPTURE_QC_IMAGES_ONLY_NO_PLANNED_INFOGRAPHIC",
        "training_performed": False,
        "training_authorized_as_next_gate": True,
        "calibration_or_test_created": False,
        "tables_02_to_06": "NOT_RUN",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    try:
        report = run_full_qc(root, run_replay=not args.skip_replay)
    except Exception as exc:
        report = {
            "schema_version": 1, "protocol_id": PROTOCOL_ID,
            "generated_at_utc": utc_now(), "passed": False,
            "decision": "FIX_V2_1_DATASET_BEFORE_TRAINING",
            "error_type": type(exc).__name__, "error": str(exc),
            "training_performed": False, "training_authorized_as_next_gate": False,
            "calibration_or_test_created": False, "tables_02_to_06": "NOT_RUN",
        }
        write_json(root / "DATASET_V2_1_DEVELOPMENT_FULL_QC_REPORT.json", report)
        print(f"DATASET_V2_1_DEVELOPMENT_FULL_QC_FAIL {type(exc).__name__}: {exc}")
        return 2
    report_path = root / "DATASET_V2_1_DEVELOPMENT_FULL_QC_REPORT.json"
    write_json(report_path, report)
    write_markdown(root, report)
    write_json(root / "report_assets/checkpoints/04_full_qc.json", {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "checkpoint": "V2_1_DEVELOPMENT_FULL_QC_PASS",
        "created_at_utc": utc_now(), "decision": report["decision"],
        "report_sha256": sha256_file(report_path),
        "dataset_index_sha256": sha256_file(root / "dataset_index.json"),
        "training_performed": False,
    })
    print(
        "DATASET_V2_1_DEVELOPMENT_FULL_QC_PASS "
        "families=400 captures=800 samples=2000 decision=GO_DEVELOPMENT_TRAIN"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
