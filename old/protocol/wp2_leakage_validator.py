#!/usr/bin/env python3
"""Validate WP2 family-level split isolation and oracle-free payloads."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp2_common import (  # noqa: E402
    PILOT_RELATIVE,
    WP1_RELATIVE,
    VARIANTS,
    WP2Error,
    find_forbidden_inference_keys,
    read_json,
    write_json,
)


def validate_leakage(dataset_root: Path, selection: str = "all") -> dict:
    index_name = "smoke_dataset_index.json" if selection == "smoke" else "dataset_index.json"
    index = read_json(dataset_root / index_name)
    manifest = read_json(dataset_root / "family_manifest.json")
    expected_splits = {item["family_id"]: item["split"] for item in manifest["families"]}
    family_splits: dict[str, set[str]] = defaultdict(set)
    family_variants: dict[str, set[str]] = defaultdict(set)
    findings = []
    forbidden_source_ids = []
    records = []
    for item in index["records"]:
        record = read_json(dataset_root / item["record_path"])
        records.append(record)
        family_id = record["family_id"]
        family_splits[family_id].add(record["split"])
        family_variants[family_id].add(record["variant"])
        if record["split"] != expected_splits.get(family_id):
            findings.append(f"{family_id}: record/manifest split mismatch")
        payload_findings = find_forbidden_inference_keys(record["inference_payload"])
        findings.extend(f"{record['sample_id']}: {path}" for path in payload_findings)
        source_ids = record["provenance"].get("source_dataset_ids", [])
        if source_ids:
            forbidden_source_ids.append({"sample_id": record["sample_id"], "ids": source_ids})
        for value in record["inference_payload"].values():
            if isinstance(value, dict) and "path" in value:
                lower = str(value["path"]).lower()
                if "evaluator" in lower or "oracle" in lower or "mask" in lower:
                    findings.append(f"{record['sample_id']}: unsafe inference path {lower}")
                if PILOT_RELATIVE.lower() in lower or WP1_RELATIVE.lower() in lower:
                    findings.append(f"{record['sample_id']}: pilot/WP1 reuse {lower}")
    leaked_families = {
        family_id: sorted(splits)
        for family_id, splits in family_splits.items() if len(splits) != 1
    }
    incomplete_families = {
        family_id: sorted(set(VARIANTS) - variants)
        for family_id, variants in family_variants.items() if variants != set(VARIANTS)
    }
    passed = not findings and not leaked_families and not incomplete_families and not forbidden_source_ids
    return {
        "passed": passed,
        "selection": selection,
        "record_count": len(records),
        "family_count": len(family_splits),
        "split_unit": "family_id",
        "family_split_overlap": leaked_families,
        "incomplete_family_variants": incomplete_families,
        "inference_oracle_findings": findings,
        "forbidden_source_dataset_ids": forbidden_source_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--selection", choices=("smoke", "all"), default="all")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = validate_leakage(Path(args.dataset_root).expanduser().resolve(), args.selection)
    if args.output:
        write_json(Path(args.output).expanduser().resolve(), result)
    print(
        f"WP2_LEAKAGE_{'PASS' if result['passed'] else 'FAIL'} "
        f"families={result['family_count']} records={result['record_count']}"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
