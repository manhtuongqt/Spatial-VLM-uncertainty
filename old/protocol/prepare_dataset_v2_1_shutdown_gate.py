#!/usr/bin/env python3
"""Materialize an isolated one-canary execution lock for the shutdown gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "roborefer_dataset_v2_1_relation_repair"
OUTPUT_ROOT = "datasets/roborefer_dataset_v2_1_shutdown_gate_canary_20260824"
PLAN_PATH = WORKSPACE / "protocol/dataset_v2_1_shutdown_gate_capture_plan.json"
LOCK_PATH = WORKSPACE / "protocol/dataset_v2_1_shutdown_gate_execution_lock.json"
BASELINE_PREFIXES = (
    "ur3/",
    "protocol/dataset_v2_development_execution_lock.json",
    "protocol/dataset_v2_pilot_execution_lock.json",
    "datasets/roborefer_dataset_v2_development_400_20260821/raw/raw_capture_manifest.json",
)
GATE_RUNTIME = (
    "protocol/dataset_v2_relation_repair_capture.py",
    "protocol/dataset_v2_relation_repair_capture.launch.py",
    "protocol/dataset_v2_relation_repair_shutdown.py",
    "protocol/dataset_v2_1_shutdown_gate.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> int:
    source_plan_path = WORKSPACE / "protocol/dataset_v2_relation_repair_capture_plan.json"
    source_lock_path = WORKSPACE / "protocol/dataset_v2_relation_repair_execution_lock.json"
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    plan = dict(source_plan)
    plan["capture_output_root"] = OUTPUT_ROOT
    plan["shutdown_gate_role"] = "ONE_REAL_CANARY_ONLY_NOT_OFFICIAL_DATA"
    write_json(PLAN_PATH, plan)

    baseline_expected = {
        relative: expected
        for relative, expected in source_lock["locked_artifact_sha256"].items()
        if relative.startswith(BASELINE_PREFIXES)
    }
    baseline_mismatches = [
        relative for relative, expected in baseline_expected.items()
        if not (WORKSPACE / relative).is_file() or sha256(WORKSPACE / relative) != expected
    ]
    if baseline_mismatches:
        raise RuntimeError(f"baseline hash mismatch before gate: {baseline_mismatches}")
    runtime_hashes = {relative: sha256(WORKSPACE / relative) for relative in GATE_RUNTIME}
    lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "GO_RELATION_REPAIR_CAPTURE_30",
        "gate_id": "FIX_V2_1_SHUTDOWN_GATE",
        "capture_authorized": True,
        "capture_output_root": OUTPUT_ROOT,
        "family_count": 30,
        "capture_count": 30,
        "gate_expected_capture_count": 1,
        "capture_plan_sha256": sha256(PLAN_PATH),
        "locked_artifact_sha256": {**baseline_expected, **runtime_hashes},
        "baseline_expected_sha256": baseline_expected,
        "training_authorized": False,
        "official_dataset_eligible": False,
    }
    write_json(LOCK_PATH, lock)
    print(
        "DATASET_V2_1_SHUTDOWN_GATE_PREPARED "
        f"baseline={len(baseline_expected)} runtime={len(runtime_hashes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
