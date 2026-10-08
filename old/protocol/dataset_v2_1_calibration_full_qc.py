#!/usr/bin/env python3
"""Full scientific QC for the independent V2.1 Calibration dataset."""

from __future__ import annotations

import argparse
import csv
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset_v2_1_calibration_generator import (
    CAPTURE_COUNT,
    EXPECTED_CATEGORY,
    EXPECTED_STATE,
    FAMILY_COUNT,
    PROTOCOL_ID,
    SAMPLE_COUNT,
    load_calibration_denylists,
)
from dataset_v2_1_calibration_materialize import materialize
from wp2_common import find_forbidden_inference_keys, read_json, sha256_file, utc_now, write_json


BATCH_IDS = ["canary_000", "batch_001", "batch_002"]


class CalibrationFullQCError(RuntimeError):
    pass


def collect_refs(value: Any) -> list[dict[str, Any]]:
    output = []
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"}.issubset(value):
            output.append(value)
        for child in value.values():
            output.extend(collect_refs(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(collect_refs(child))
    return output


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise CalibrationFullQCError(f"artifact path escapes dataset root: {relative}")
    return path


def validate_raw_and_batches(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    raw = read_json(root / "raw/raw_capture_manifest.json")
    if (
        raw.get("protocol_id") != PROTOCOL_ID
        or raw.get("capture_count") != CAPTURE_COUNT
        or raw.get("expected_capture_count") != CAPTURE_COUNT
        or raw.get("complete") is not True
    ):
        raise CalibrationFullQCError("raw capture manifest is not complete at 400")
    raw_ids = [row["capture_id"] for row in raw["captures"]]
    planned_ids = [row["capture_id"] for row in plan["captures"]]
    if len(raw_ids) != len(set(raw_ids)) or set(raw_ids) != set(planned_ids):
        raise CalibrationFullQCError("raw/planned capture identity set differs")
    reports = []
    for batch_id in BATCH_IDS:
        report = read_json(root / "report_assets/checkpoints/batch_qc" / f"{batch_id}.json")
        if not report.get("passed") or report.get("batch_id") != batch_id:
            raise CalibrationFullQCError(f"batch QC is not PASS: {batch_id}")
        reports.append(report)
    shutdown = read_json(root / "report_assets/checkpoints/shutdown_sequence.json")
    if not shutdown.get("ordered_shutdown_complete"):
        raise CalibrationFullQCError("final ordered shutdown is incomplete")
    return {
        "passed": True,
        "raw_capture_count": CAPTURE_COUNT,
        "relation_checks": sum(int(row["relation_checks"]) for row in reports),
        "relation_failures": sum(len(row["relation_failures"]) for row in reports),
        "max_timestamp_spread_sec": max(float(row["max_timestamp_spread_sec"]) for row in reports),
        "ordered_shutdown_complete": True,
        "batch_reports": [{
            "batch_id": row["batch_id"],
            "valid_capture_count": row["valid_capture_count"],
            "relation_checks": row["relation_checks"],
            "relation_not_evaluable_allowed": row["relation_not_evaluable_allowed"],
        } for row in reports],
    }


def validate_records(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    index = read_json(root / "dataset_index.json")
    if (
        index.get("protocol_id") != PROTOCOL_ID
        or index.get("record_count") != SAMPLE_COUNT
        or index.get("family_count") != FAMILY_COUNT
        or index.get("split_family_counts") != {"calibration": FAMILY_COUNT}
        or index.get("eligible_for_training_or_dev") is not False
        or index.get("eligible_for_test") is not False
        or index.get("test_opened") is not False
    ):
        raise CalibrationFullQCError("dataset index identity/count/boundary differs")
    family_by_id = {row["family_id"]: row for row in manifest["families"]}
    family_counts: Counter[str] = Counter()
    states: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    hash_errors, leakage = [], []
    mask_stats: Counter[str] = Counter()
    for item in index["records"]:
        path = safe_path(root, item["record_path"])
        if not path.is_file() or sha256_file(path) != item["record_sha256"]:
            hash_errors.append(f"record:{item['sample_id']}")
            continue
        record = read_json(path)
        if (
            record.get("protocol_id") != PROTOCOL_ID
            or record.get("sample_id") != item["sample_id"]
            or record.get("split") != "calibration"
            or item.get("split") != "calibration"
        ):
            hash_errors.append(f"identity:{item['sample_id']}")
        for ref in collect_refs(record):
            artifact = safe_path(root, str(ref["path"]))
            if (
                not artifact.is_file() or artifact.stat().st_size != int(ref["bytes"])
                or sha256_file(artifact) != ref["sha256"]
            ):
                hash_errors.append(f"artifact:{item['sample_id']}:{ref['path']}")
        findings = find_forbidden_inference_keys(record["inference_payload"])
        if findings:
            leakage.append({"sample_id": item["sample_id"], "paths": findings})
        family_counts[record["family_id"]] += 1
        label = record["evaluator_only"]["uncertainty_label"]
        if record["variant"] == "clean":
            states[label["state"]] += 1
            categories[record["family_category"]] += 1
            relations[record["evaluator_only"]["spatial_label"]["relations"][0]] += 1
        target_ref = record["evaluator_only"]["masks"]["target"]
        interior_ref = record["evaluator_only"]["masks"]["target_interior"]
        target = cv2.imread(str(safe_path(root, target_ref["path"])), cv2.IMREAD_UNCHANGED)
        interior = cv2.imread(str(safe_path(root, interior_ref["path"])), cv2.IMREAD_UNCHANGED)
        if target is None or interior is None:
            hash_errors.append(f"mask-decode:{item['sample_id']}")
            continue
        target_pixels, interior_pixels = int(np.count_nonzero(target)), int(np.count_nonzero(interior))
        state = label["state"]
        if state in {"FOUND", "AMBIGUOUS"} and target_pixels == 0:
            hash_errors.append(f"empty-positive-target:{item['sample_id']}")
        if state == "FOUND" and interior_pixels == 0:
            hash_errors.append(f"empty-found-interior:{item['sample_id']}")
        if state == "ABSENT" and target_pixels != 0:
            hash_errors.append(f"nonempty-absent-target:{item['sample_id']}")
        mask_stats[f"{state}_records"] += 1
        mask_stats[f"{state}_target_pixels"] += target_pixels
    if hash_errors:
        raise CalibrationFullQCError(f"record/artifact/mask validation failed: {hash_errors[:20]}")
    if leakage:
        raise CalibrationFullQCError(f"oracle leakage found: {leakage[:10]}")
    if (
        len(index["records"]) != SAMPLE_COUNT
        or len(family_counts) != FAMILY_COUNT
        or any(value != 5 for value in family_counts.values())
        or set(family_counts) != set(family_by_id)
        or dict(states) != EXPECTED_STATE
        or dict(categories) != EXPECTED_CATEGORY
    ):
        raise CalibrationFullQCError(
            f"family/state/category invariant differs: states={states} categories={categories}"
        )
    return {
        "passed": True,
        "record_count": SAMPLE_COUNT,
        "family_count": FAMILY_COUNT,
        "primary_answerability": dict(sorted(states.items())),
        "family_category": dict(sorted(categories.items())),
        "relation": dict(sorted(relations.items())),
        "mask_statistics": dict(sorted(mask_stats.items())),
        "oracle_leakage_findings": [],
    }


def validate_independence(workspace: Path, manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    denied = load_calibration_denylists(workspace)
    family_ids = {row["family_id"] for row in manifest["families"]}
    capture_ids = {row["capture_id"] for row in plan["captures"]}
    instructions = {
        spec["instruction"] for family in manifest["families"] for spec in family["variant_specs"]
    }
    seeds = {
        int(value) for family in manifest["families"]
        for value in (*family["seed_bundle"].values(), family["selected_layout_seed"])
    }
    layouts = {row["layout_instance_fingerprint_sha256"] for row in plan["captures"]}
    overlaps = {
        "family_id": len(family_ids & denied["family_ids"]),
        "capture_id": len(capture_ids & denied["capture_ids"]),
        "instruction": len(instructions & denied["instructions"]),
        "seed": len(seeds & denied["seeds"]),
        "layout": len(layouts & denied["layout_fingerprints"]),
    }
    if any(overlaps.values()):
        raise CalibrationFullQCError(f"calibration overlaps historical data: {overlaps}")
    return {"passed": True, "historical_overlap_counts": overlaps}


def validate_assets(workspace: Path, plan: dict[str, Any]) -> dict[str, Any]:
    partition = read_json(workspace / "protocol/dataset_expansion_v2_asset_partition.json")
    heldout = set(partition["test_ood_asset_holdout"]["asset_ids"])
    excluded = set(partition["excluded_assets"]["asset_ids"])
    registry = plan["object_registry"]
    active = {
        row["id"] for capture in plan["captures"] for name, row in registry.items()
        if float(capture["layout"][name][1]) <= .65
    }
    bad = active & (heldout | excluded)
    if bad:
        raise CalibrationFullQCError(f"heldout/excluded assets active: {sorted(bad)}")
    return {"passed": True, "active_asset_ids": sorted(active), "heldout_active": []}


def deterministic_replay(root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dataset_v2_1_calibration_replay_") as value:
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
                relative_files.extend(path.relative_to(root) for path in source.rglob("*") if path.is_file())
        mismatches = [
            str(relative) for relative in sorted(relative_files)
            if not (replay_root / relative).is_file()
            or sha256_file(root / relative) != sha256_file(replay_root / relative)
        ]
        if mismatches:
            raise CalibrationFullQCError(f"materialization replay differs: {mismatches[:20]}")
        return {"passed": True, "files_compared": len(relative_files), "mismatches": []}


def write_tables(root: Path, records: dict[str, Any]) -> dict[str, str]:
    tables = root / "report_assets/tables"
    tables.mkdir(parents=True, exist_ok=True)
    population = tables / "table_calibration_dataset_independence.csv"
    with population.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "split", "independent_families", "dependent_samples", "FOUND", "AMBIGUOUS",
            "ABSENT", "INSUFFICIENT_EVIDENCE", "OOD", "scientific_status",
        ])
        values = records["primary_answerability"]
        writer.writerow([
            "calibration", FAMILY_COUNT, SAMPLE_COUNT, values["FOUND"], values["AMBIGUOUS"],
            values["ABSENT"], values["INSUFFICIENT_EVIDENCE"], 0,
            "OBSERVED_INDEPENDENT_CALIBRATION_FULL_QC_PASS",
        ])
        writer.writerow(["test_iid", 0, 0, 0, 0, 0, 0, 0, "SEALED_NOT_CREATED"])
        writer.writerow(["test_ood", 0, 0, 0, 0, 0, 0, 0, "SEALED_NOT_CREATED"])
    categories = tables / "table_calibration_family_category_balance.csv"
    with categories.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["family_category", "independent_families"])
        writer.writerows(sorted(records["family_category"].items()))
    return {
        "population": str(population.relative_to(root)),
        "category": str(categories.relative_to(root)),
    }


def run_full_qc(root: Path, run_replay: bool) -> dict[str, Any]:
    workspace = Path(__file__).resolve().parents[1]
    manifest = read_json(workspace / "protocol/dataset_v2_1_calibration_manifest.json")
    plan = read_json(workspace / "protocol/dataset_v2_1_calibration_capture_plan.json")
    lock = read_json(workspace / "protocol/dataset_v2_1_calibration_execution_lock.json")
    if (
        lock.get("training_authorized")
        or lock.get("test_authorized")
        or not lock.get("calibration_capture_authorized")
    ):
        raise CalibrationFullQCError("calibration-only execution boundary was violated")
    for relative, expected in lock["locked_artifact_sha256"].items():
        path = workspace / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise CalibrationFullQCError(f"execution-locked artifact changed: {relative}")
    raw = validate_raw_and_batches(root, plan)
    records = validate_records(root, manifest)
    independence = validate_independence(workspace, manifest, plan)
    assets = validate_assets(workspace, plan)
    replay = deterministic_replay(root) if run_replay else {
        "passed": None, "files_compared": 0, "mismatches": [], "skipped": True,
    }
    tables = write_tables(root, records)
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "passed": True,
        "decision": "GO_CALIBRATION_INFERENCE_AND_FIT",
        "scientific_status": "OFFICIAL_INDEPENDENT_CALIBRATION_QC_PASS_TESTS_STILL_SEALED",
        "family_count": FAMILY_COUNT,
        "sample_count": SAMPLE_COUNT,
        "raw_capture_count": CAPTURE_COUNT,
        "raw_and_batch_qc": raw,
        "record_mask_ontology_oracle_qc": records,
        "historical_independence_qc": independence,
        "asset_partition_qc": assets,
        "deterministic_replay": replay,
        "tables": tables,
        "figures_policy": "REAL_CAPTURE_QC_IMAGES_ONLY_NO_PLANNED_INFOGRAPHIC",
        "training_performed": False,
        "model_inference_performed": False,
        "calibrator_fit_performed": False,
        "calibration_inference_and_fit_authorized_as_next_gate": True,
        "test_authorized": False,
        "test_opened": False,
    }


def write_markdown(root: Path, report: dict[str, Any]) -> None:
    batches = "\n".join(
        f"- `{row['batch_id']}`: {row['valid_capture_count']} capture; {row['relation_checks']} relation checks"
        for row in report["raw_and_batch_qc"]["batch_reports"]
    )
    text = f"""# Dataset V2.1 Calibration full-QC report

- Decision: `{report['decision']}`
- Independent families: {FAMILY_COUNT}
- Materialized samples: {SAMPLE_COUNT}
- Real RGB-D/semantic captures: {CAPTURE_COUNT}
- Relation failures: {report['raw_and_batch_qc']['relation_failures']}
- Historical overlaps: {report['historical_independence_qc']['historical_overlap_counts']}
- Training performed: `false`
- Model inference performed: `false`
- Calibrator fit performed: `false`
- Test-IID/Test-OOD opened: `false`

## Batch gates

{batches}

Only real capture QC images are stored. This PASS authorizes inference with the
already frozen checkpoint and calibrator fitting on Calibration only. Test-IID
and Test-OOD remain sealed.
"""
    (root / "DATASET_V2_1_CALIBRATION_FULL_QC_REPORT.md").write_text(text, encoding="utf-8")


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
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "generated_at_utc": utc_now(),
            "passed": False,
            "decision": "FIX_CALIBRATION_DATASET_BEFORE_INFERENCE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "training_performed": False,
            "model_inference_performed": False,
            "calibrator_fit_performed": False,
            "test_opened": False,
        }
        write_json(root / "DATASET_V2_1_CALIBRATION_FULL_QC_REPORT.json", report)
        print(f"DATASET_V2_1_CALIBRATION_FULL_QC_FAIL {type(exc).__name__}: {exc}")
        return 2
    report_path = root / "DATASET_V2_1_CALIBRATION_FULL_QC_REPORT.json"
    write_json(report_path, report)
    write_markdown(root, report)
    write_json(root / "report_assets/checkpoints/04_full_qc.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "checkpoint": "V2_1_CALIBRATION_FULL_QC_PASS",
        "created_at_utc": utc_now(),
        "decision": report["decision"],
        "report_sha256": sha256_file(report_path),
        "dataset_index_sha256": sha256_file(root / "dataset_index.json"),
        "training_performed": False,
        "test_opened": False,
    })
    print(
        "DATASET_V2_1_CALIBRATION_FULL_QC_PASS "
        "families=200 captures=400 samples=1000 decision=GO_CALIBRATION_INFERENCE_AND_FIT"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
