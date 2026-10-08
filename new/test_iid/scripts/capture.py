#!/usr/bin/python3
"""Resume-safe Gazebo capture for the locked Test-IID batches."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "old" / "protocol"))

import rclpy  # noqa: E402
import dataset_v2_development_capture as runtime  # noqa: E402
from dataset_v2_relation_repair_capture import RepairCapture  # noqa: E402
from wp2_common import WP2Error, read_json, sha256_file, utc_now  # noqa: E402


PROTOCOL_ID = "roborefer_dataset_v2_1_test_iid_capture_200"
DECISION = "GO_TEST_IID_CAPTURE_CANARY"
EXPECTED_CAPTURES = 400
PATTERN = re.compile(r"^v211iid_family_[0-9]{6}__(?:clean|occlusion)_capture$")

runtime.PROTOCOL_ID = PROTOCOL_ID
runtime.DECISION = DECISION
runtime.EXPECTED_CAPTURES = EXPECTED_CAPTURES
runtime.CAPTURE_PATTERN = PATTERN
runtime.shared.PROTOCOL_ID = PROTOCOL_ID
runtime.shared.CAPTURE_ID = PATTERN


def verify(plan_path: Path, lock_path: Path, output_root: Path) -> tuple[dict, dict]:
    plan, lock = read_json(plan_path), read_json(lock_path)
    if plan.get("protocol_id") != PROTOCOL_ID or lock.get("protocol_id") != PROTOCOL_ID:
        raise WP2Error("Test-IID protocol identity mismatch")
    if lock.get("decision") != DECISION or not lock.get("capture_authorized"):
        raise WP2Error("Test-IID execution lock does not authorize capture")
    if lock.get("test_inference_authorized") or lock.get("training_authorized"):
        raise WP2Error("Inference/training must remain sealed during capture")
    if output_root != (ROOT / lock["capture_output_root"]).resolve():
        raise WP2Error("Output root differs from execution lock")
    if sha256_file(plan_path) != lock["capture_plan_sha256"]:
        raise WP2Error("Capture plan hash differs from execution lock")
    for relative, expected in lock["locked_artifact_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise WP2Error(f"Locked artifact changed: {relative}")
    return plan, lock


def write_progress(output_root: Path, plan_path: Path, lock_path: Path, plan: dict, batch_id: str, family_id: str) -> None:
    inventory = runtime.inventory_raw(output_root, plan)
    inventory["capture_plan_sha256"] = sha256_file(plan_path)
    inventory["execution_lock_sha256"] = sha256_file(lock_path)
    manifest_path = output_root / "raw/raw_capture_manifest.json"
    runtime.atomic_json(manifest_path, inventory)
    completed = Counter(row["family_id"] for row in inventory["captures"])
    runtime.atomic_json(output_root / "report_assets/checkpoints/capture_progress.json", {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "created_at_utc": utc_now(),
        "active_batch": batch_id, "last_completed_family_id": family_id,
        "completed_family_count": sum(value == 2 for value in completed.values()),
        "completed_capture_count": inventory["capture_count"],
        "raw_capture_complete": inventory["complete"],
        "raw_capture_manifest_sha256": sha256_file(manifest_path),
        "test_inference_performed": False,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-plan", required=True)
    parser.add_argument("--execution-lock", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--settle-sec", type=float, default=.8)
    args = parser.parse_args()
    plan_path, lock_path = Path(args.capture_plan).resolve(), Path(args.execution_lock).resolve()
    output_root = Path(args.output_root).resolve()
    plan, _ = verify(plan_path, lock_path, output_root)
    batches = {row["batch_id"]: row for row in plan["batches"]}
    if args.batch_id not in batches:
        raise WP2Error(f"Unknown batch: {args.batch_id}")
    batch = batches[args.batch_id]
    runtime.require_previous_batch_qc(output_root, plan_path, lock_path, batch, batches)
    rows_by_family = {
        family_id: [row for row in plan["captures"] if row["family_id"] == family_id]
        for family_id in batch["family_ids"]
    }
    if any(len(rows) != 2 for rows in rows_by_family.values()):
        raise WP2Error("Every Test-IID family must have exactly two captures")
    output_root.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = RepairCapture(plan, output_root, args.settle_sec)
    try:
        for family_id in batch["family_ids"]:
            node.run(rows_by_family[family_id])
            write_progress(output_root, plan_path, lock_path, plan, args.batch_id, family_id)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    inventory = runtime.inventory_raw(output_root, plan)
    actual = int(inventory["batch_capture_counts"].get(args.batch_id, 0))
    expected = int(batch["planned_capture_count"])
    if actual != expected:
        raise WP2Error(f"Batch capture incomplete: {actual}/{expected}")
    runtime.atomic_json(output_root / f"report_assets/checkpoints/capture_{args.batch_id}.json", {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "batch_id": args.batch_id,
        "status": "CAPTURE_COMPLETE_AWAITING_QC", "created_at_utc": utc_now(),
        "family_count": batch["family_count"], "capture_count": actual,
        "capture_plan_sha256": sha256_file(plan_path),
        "execution_lock_sha256": sha256_file(lock_path),
        "test_inference_performed": False,
    })
    print(json.dumps({"status": "CAPTURE_COMPLETE_AWAITING_QC", "batch": args.batch_id, "captures": actual}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
