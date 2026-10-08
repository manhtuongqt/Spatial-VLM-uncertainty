#!/usr/bin/python3
"""Resume-safe, batch-gated Gazebo capture for Dataset V2 development data.

This file is locked by static preflight but is not executed during design lock.
The low-level ROS/Gazebo sensor implementation is reused from the qualified
pilot runner; protocol identity, authorization, batching and completeness are
strictly development-specific.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import rclpy

import dataset_v2_pilot_capture as shared
from wp2_common import WP2Error, read_json, sha256_file, utc_now


PROTOCOL_ID = "roborefer_dataset_v2_development_capture_400"
DECISION = "GO_DEVELOPMENT_CAPTURE_400"
EXPECTED_CAPTURES = 800
CAPTURE_PATTERN = re.compile(r"^v2_family_[0-9]{6}__(?:clean|occlusion)_capture$")

# The inherited save path reads these module globals. Rebinding them changes
# protocol identity and ID validation only; it never imports pilot data.
shared.PROTOCOL_ID = PROTOCOL_ID
shared.CAPTURE_ID = CAPTURE_PATTERN


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_execution_lock(
    workspace: Path, plan_path: Path, lock_path: Path, output_root: Path
) -> dict[str, Any]:
    lock = read_json(lock_path)
    if lock.get("protocol_id") != PROTOCOL_ID or not lock.get("capture_authorized"):
        raise WP2Error("development execution lock does not authorize capture")
    if lock.get("decision") != DECISION:
        raise WP2Error(f"unexpected development decision: {lock.get('decision')}")
    expected_output = (workspace / str(lock["capture_output_root"])).resolve()
    if output_root != expected_output:
        raise WP2Error(f"output root differs from execution lock: {output_root} != {expected_output}")
    if sha256_file(plan_path) != lock.get("capture_plan_sha256"):
        raise WP2Error("capture plan hash differs from execution lock")
    for relative, expected in lock["locked_artifact_sha256"].items():
        path = workspace / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise WP2Error(f"execution-locked artifact changed: {relative}")
    return lock


def inventory_raw(output_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    capture_root = output_root / "raw" / "captures"
    expected_by_id = {item["capture_id"]: item for item in plan["captures"]}
    records = []
    if capture_root.is_dir():
        for directory in sorted(path for path in capture_root.iterdir() if path.is_dir()):
            capture_id = directory.name
            if capture_id not in expected_by_id or CAPTURE_PATTERN.fullmatch(capture_id) is None:
                raise WP2Error(f"unknown capture directory in development root: {capture_id}")
            meta_path = directory / "capture_meta.json"
            meta = read_json(meta_path)
            if meta.get("protocol_id") != PROTOCOL_ID or meta.get("capture_id") != capture_id:
                raise WP2Error(f"capture metadata identity differs: {capture_id}")
            for relative, expected in meta["artifact_sha256"].items():
                artifact = directory / relative
                if not artifact.is_file() or sha256_file(artifact) != expected:
                    raise WP2Error(f"capture artifact hash differs: {capture_id}/{relative}")
            expected_capture = expected_by_id[capture_id]
            records.append({
                "capture_id": capture_id,
                "family_id": meta["family_id"],
                "split": expected_capture["split"],
                "batch_id": expected_capture["batch_id"],
                "condition": meta["condition"],
                "capture_meta_sha256": sha256_file(meta_path),
                "artifact_sha256": meta["artifact_sha256"],
            })
    captured_ids = {item["capture_id"] for item in records}
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "capture_count": len(records),
        "expected_capture_count": EXPECTED_CAPTURES,
        "complete": len(records) == EXPECTED_CAPTURES and captured_ids == set(expected_by_id),
        "batch_capture_counts": dict(sorted(Counter(item["batch_id"] for item in records).items())),
        "split_capture_counts": dict(sorted(Counter(item["split"] for item in records).items())),
        "captures": records,
    }


def write_progress(
    output_root: Path, plan_path: Path, lock_path: Path, plan: dict[str, Any],
    active_batch: str, last_family_id: str,
) -> None:
    inventory = inventory_raw(output_root, plan)
    inventory["capture_plan_sha256"] = sha256_file(plan_path)
    inventory["execution_lock_sha256"] = sha256_file(lock_path)
    inventory_path = output_root / "raw" / "raw_capture_manifest.json"
    atomic_json(inventory_path, inventory)
    completed_by_family = Counter(item["family_id"] for item in inventory["captures"])
    checkpoint = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "checkpoint": "FAMILY_PROGRESS",
        "created_at_utc": utc_now(),
        "active_batch": active_batch,
        "last_completed_family_id": last_family_id,
        "completed_family_count": sum(value == 2 for value in completed_by_family.values()),
        "completed_capture_count": inventory["capture_count"],
        "raw_capture_complete": inventory["complete"],
        "raw_capture_manifest_sha256": sha256_file(inventory_path),
        "capture_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output_root / "report_assets" / "checkpoints" / "capture_progress.json", checkpoint)


def require_previous_batch_qc(
    output_root: Path, plan_path: Path, lock_path: Path,
    batch: dict[str, Any], batches: dict[str, dict[str, Any]],
) -> None:
    prerequisite = batch["prerequisite_batch"]
    if prerequisite is None:
        return
    if prerequisite not in batches:
        raise WP2Error(f"unknown prerequisite batch: {prerequisite}")
    gate_path = output_root / "report_assets" / "checkpoints" / "batch_qc" / f"{prerequisite}.json"
    if not gate_path.is_file():
        raise WP2Error(f"previous batch QC checkpoint missing: {gate_path}")
    gate = read_json(gate_path)
    if (
        not gate.get("passed")
        or gate.get("batch_id") != prerequisite
        or gate.get("capture_plan_sha256") != sha256_file(plan_path)
        or gate.get("execution_lock_sha256") != sha256_file(lock_path)
    ):
        raise WP2Error(f"previous batch QC checkpoint invalid: {prerequisite}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-plan", required=True)
    parser.add_argument("--execution-lock", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--settle-sec", type=float, default=.8)
    args = parser.parse_args()
    plan_path = Path(args.capture_plan).expanduser().resolve()
    lock_path = Path(args.execution_lock).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    workspace = Path(__file__).resolve().parents[1]
    plan = read_json(plan_path)
    if (
        plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("family_count") != 400
        or plan.get("capture_count") != EXPECTED_CAPTURES
    ):
        raise WP2Error("capture plan does not match Dataset V2 development contract")
    verify_execution_lock(workspace, plan_path, lock_path, output_root)
    batches = {item["batch_id"]: item for item in plan["batches"]}
    if args.batch_id not in batches:
        raise WP2Error(f"unknown batch ID: {args.batch_id}")
    batch = batches[args.batch_id]
    require_previous_batch_qc(output_root, plan_path, lock_path, batch, batches)
    family_ids = list(batch["family_ids"])
    captures_by_family = {
        family_id: [item for item in plan["captures"] if item["family_id"] == family_id]
        for family_id in family_ids
    }
    if any(len(value) != 2 for value in captures_by_family.values()):
        raise WP2Error("selected batch does not have exactly two captures per family")

    # Refuse evidence from a future batch even if it was placed manually.
    initial = inventory_raw(output_root, plan)
    allowed_orders = {item["batch_id"]: int(item["order"]) for item in plan["batches"]}
    if any(allowed_orders[item["batch_id"]] > int(batch["order"]) for item in initial["captures"]):
        raise WP2Error("future-batch capture exists before its gate")

    output_root.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = shared.PilotCapture(plan, output_root, args.settle_sec)
    try:
        for family_id in family_ids:
            node.run(captures_by_family[family_id])
            write_progress(output_root, plan_path, lock_path, plan, args.batch_id, family_id)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    final_inventory = inventory_raw(output_root, plan)
    batch_count = int(final_inventory["batch_capture_counts"].get(args.batch_id, 0))
    expected_batch_count = int(batch["planned_capture_count"])
    if batch_count != expected_batch_count:
        raise WP2Error(f"batch capture incomplete: {batch_count} != {expected_batch_count}")
    atomic_json(output_root / "report_assets" / "checkpoints" / f"capture_{args.batch_id}.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "batch_id": args.batch_id,
        "status": "CAPTURE_COMPLETE_AWAITING_BATCH_QC",
        "created_at_utc": utc_now(),
        "family_count": int(batch["family_count"]),
        "capture_count": batch_count,
        "capture_plan_sha256": sha256_file(plan_path),
        "execution_lock_sha256": sha256_file(lock_path),
        "training_performed": False,
    })
    print(
        "DATASET_V2_DEVELOPMENT_BATCH_CAPTURE_COMPLETE "
        f"batch={args.batch_id} families={batch['family_count']} captures={batch_count} "
        "next=RUN_BATCH_QC"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
