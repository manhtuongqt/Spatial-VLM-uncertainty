#!/usr/bin/python3
"""Resume-safe Dataset V2.1 calibration capture entry point.

Only the qualified V2.1 sensor writer and camera motion are reused.  Protocol
policy is calibration-specific; no development record is read or overwritten.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import rclpy

import dataset_v2_development_capture as runtime
from dataset_v2_relation_repair_capture import RepairCapture
from wp2_common import WP2Error, read_json, sha256_file, utc_now


PROTOCOL_ID = "roborefer_dataset_v2_1_calibration_capture_200"
DECISION = "GO_CALIBRATION_CAPTURE_200"
EXPECTED_FAMILIES = 200
EXPECTED_CAPTURES = 400
CAPTURE_PATTERN = re.compile(r"^v211cal_family_[0-9]{6}__(?:clean|occlusion)_capture$")

# Rebind only protocol policy in the shared capture implementation.  Baseline
# launch files, URDF, controllers, Gazebo world and sensor code remain intact.
runtime.PROTOCOL_ID = PROTOCOL_ID
runtime.DECISION = DECISION
runtime.EXPECTED_CAPTURES = EXPECTED_CAPTURES
runtime.CAPTURE_PATTERN = CAPTURE_PATTERN
runtime.shared.PROTOCOL_ID = PROTOCOL_ID
runtime.shared.CAPTURE_ID = CAPTURE_PATTERN
runtime.shared.PilotCapture = RepairCapture


def write_progress(
    output_root: Path,
    plan_path: Path,
    lock_path: Path,
    plan: dict,
    active_batch: str,
    last_family_id: str,
) -> None:
    inventory = runtime.inventory_raw(output_root, plan)
    inventory["capture_plan_sha256"] = sha256_file(plan_path)
    inventory["execution_lock_sha256"] = sha256_file(lock_path)
    inventory_path = output_root / "raw/raw_capture_manifest.json"
    runtime.atomic_json(inventory_path, inventory)
    completed = Counter(row["family_id"] for row in inventory["captures"])
    runtime.atomic_json(output_root / "report_assets/checkpoints/capture_progress.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "checkpoint": "CALIBRATION_FAMILY_PROGRESS",
        "created_at_utc": utc_now(),
        "active_batch": active_batch,
        "last_completed_family_id": last_family_id,
        "completed_family_count": sum(value == 2 for value in completed.values()),
        "completed_capture_count": inventory["capture_count"],
        "raw_capture_complete": inventory["complete"],
        "raw_capture_manifest_sha256": sha256_file(inventory_path),
        "capture_source_sha256": sha256_file(Path(__file__).resolve()),
        "training_performed": False,
        "test_opened": False,
    })


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
        or plan.get("family_count") != EXPECTED_FAMILIES
        or plan.get("capture_count") != EXPECTED_CAPTURES
        or plan.get("training_authorized") is not False
        or plan.get("test_authorized") is not False
    ):
        raise WP2Error("capture plan does not match locked calibration contract")
    lock = runtime.verify_execution_lock(workspace, plan_path, lock_path, output_root)
    if not lock.get("calibration_capture_authorized") or lock.get("test_authorized"):
        raise WP2Error("execution lock does not authorize calibration-only capture")
    batches = {row["batch_id"]: row for row in plan["batches"]}
    if args.batch_id not in batches:
        raise WP2Error(f"unknown batch ID: {args.batch_id}")
    batch = batches[args.batch_id]
    runtime.require_previous_batch_qc(output_root, plan_path, lock_path, batch, batches)
    family_ids = list(batch["family_ids"])
    captures_by_family = {
        family_id: [row for row in plan["captures"] if row["family_id"] == family_id]
        for family_id in family_ids
    }
    if any(len(rows) != 2 for rows in captures_by_family.values()):
        raise WP2Error("selected calibration batch lacks exactly two captures per family")

    initial = runtime.inventory_raw(output_root, plan)
    orders = {row["batch_id"]: int(row["order"]) for row in plan["batches"]}
    if any(orders[row["batch_id"]] > int(batch["order"]) for row in initial["captures"]):
        raise WP2Error("future-batch capture exists before its QC gate")

    output_root.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = runtime.shared.PilotCapture(plan, output_root, args.settle_sec)
    try:
        for family_id in family_ids:
            node.run(captures_by_family[family_id])
            write_progress(output_root, plan_path, lock_path, plan, args.batch_id, family_id)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    final_inventory = runtime.inventory_raw(output_root, plan)
    batch_count = int(final_inventory["batch_capture_counts"].get(args.batch_id, 0))
    expected = int(batch["planned_capture_count"])
    if batch_count != expected:
        raise WP2Error(f"batch capture incomplete: {batch_count} != {expected}")
    runtime.atomic_json(output_root / "report_assets/checkpoints" / f"capture_{args.batch_id}.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "batch_id": args.batch_id,
        "status": "CALIBRATION_CAPTURE_COMPLETE_AWAITING_BATCH_QC",
        "created_at_utc": utc_now(),
        "family_count": int(batch["family_count"]),
        "capture_count": batch_count,
        "capture_plan_sha256": sha256_file(plan_path),
        "execution_lock_sha256": sha256_file(lock_path),
        "training_performed": False,
        "model_inference_performed": False,
        "test_opened": False,
    })
    print(
        "DATASET_V2_1_CALIBRATION_BATCH_CAPTURE_COMPLETE "
        f"batch={args.batch_id} families={batch['family_count']} captures={batch_count} "
        "next=RUN_CALIBRATION_BATCH_QC"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
