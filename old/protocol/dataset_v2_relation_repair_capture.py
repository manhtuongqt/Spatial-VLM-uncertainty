#!/usr/bin/python3
"""Resume-safe Gazebo capture for the V2.1 relation-repair pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import rclpy
import dataset_v2_pilot_capture as shared
from wp2_common import WP2Error, read_json, sha256_file, utc_now


PROTOCOL_ID = "roborefer_dataset_v2_1_relation_repair"
DECISION = "GO_RELATION_REPAIR_CAPTURE_30"
EXPECTED_CAPTURES = 30
CAPTURE_PATTERN = re.compile(r"^v21repair_family_[0-9]{6}__clean_capture$")
QUALIFIED_STAGING_VIEW = [1.3815, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601]

# The inherited save method validates these module globals.
shared.PROTOCOL_ID = PROTOCOL_ID
shared.CAPTURE_ID = CAPTURE_PATTERN


class RepairCapture(shared.PilotCapture):
    """Qualified MoveIt capture staged through the proven V2 camera view."""

    def _send_joint_goal(self, requested: list[float], label: str) -> None:
        if len(requested) != 6 or not all(math.isfinite(float(value)) for value in requested):
            raise WP2Error("view_joint_pose must contain six finite values")
        if self.view_ready(requested):
            return
        if not self.motion_client.wait_for_server(timeout_sec=60.0):
            raise WP2Error("UR3 motion action server unavailable")
        goal = shared.UR3Control.Goal()
        goal.command_type = shared.UR3Control.Goal.MOVE_JOINT
        goal.joint_goal.position = [float(value) for value in requested]
        future = self.motion_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        handle = future.result() if future.done() else None
        if handle is None or not handle.accepted:
            raise WP2Error("camera joint motion rejected")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=90.0)
        result = result_future.result() if result_future.done() else None
        if result is None or not result.result.success:
            raise WP2Error(f"camera joint motion failed: {label}")
        if not self.spin_until(lambda: self.view_ready(requested), 20.0):
            raise WP2Error(f"camera joint pose did not reach tolerance: {label}")

    def move_camera(self, requested: list[float]) -> None:
        if self.view_ready(requested):
            return
        self._send_joint_goal(QUALIFIED_STAGING_VIEW, "qualified_v2_staging_view")
        self._send_joint_goal(requested, "v2_1_relation_view")


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
        raise WP2Error("repair execution lock does not authorize capture")
    if lock.get("decision") != DECISION:
        raise WP2Error(f"unexpected repair decision: {lock.get('decision')}")
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


def raw_inventory(output_root: Path, plan: dict) -> dict[str, Any]:
    expected = {item["capture_id"]: item for item in plan["captures"]}
    records = []
    root = output_root / "raw" / "captures"
    if root.is_dir():
        for directory in sorted(value for value in root.iterdir() if value.is_dir()):
            if directory.name not in expected or CAPTURE_PATTERN.fullmatch(directory.name) is None:
                raise WP2Error(f"unknown repair capture directory: {directory.name}")
            meta_path = directory / "capture_meta.json"
            meta = read_json(meta_path)
            if meta.get("protocol_id") != PROTOCOL_ID or meta.get("capture_id") != directory.name:
                raise WP2Error(f"capture identity mismatch: {directory.name}")
            for relative, digest in meta["artifact_sha256"].items():
                if sha256_file(directory / relative) != digest:
                    raise WP2Error(f"capture artifact hash mismatch: {directory.name}/{relative}")
            records.append({
                "capture_id": directory.name,
                "family_id": meta["family_id"],
                "capture_meta_sha256": sha256_file(meta_path),
                "artifact_sha256": meta["artifact_sha256"],
            })
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "capture_count": len(records),
        "expected_capture_count": EXPECTED_CAPTURES,
        "complete": len(records) == EXPECTED_CAPTURES and {v["capture_id"] for v in records} == set(expected),
        "captures": records,
    }


def write_progress(
    output_root: Path, plan_path: Path, lock_path: Path, plan: dict, last_family_id: str
) -> None:
    inventory = raw_inventory(output_root, plan)
    inventory["capture_plan_sha256"] = sha256_file(plan_path)
    inventory["execution_lock_sha256"] = sha256_file(lock_path)
    manifest_path = output_root / "raw" / "raw_capture_manifest.json"
    atomic_json(manifest_path, inventory)
    atomic_json(output_root / "report_assets" / "checkpoints" / "capture_progress.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": utc_now(),
        "status": "COMPLETE_AWAITING_QC" if inventory["complete"] else "PARTIAL_RESUMABLE",
        "last_completed_family_id": last_family_id,
        "completed_capture_count": inventory["capture_count"],
        "raw_capture_manifest_sha256": sha256_file(manifest_path),
        "capture_source_sha256": sha256_file(Path(__file__).resolve()),
        "moveit_used": True,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-plan", required=True)
    parser.add_argument("--execution-lock", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--settle-sec", type=float, default=0.8)
    parser.add_argument("--max-captures", type=int, default=0)
    parser.add_argument("--family-ids", default="")
    args = parser.parse_args()
    plan_path = Path(args.capture_plan).expanduser().resolve()
    lock_path = Path(args.execution_lock).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    workspace = Path(__file__).resolve().parents[1]
    plan = read_json(plan_path)
    if (
        plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("family_count") != EXPECTED_CAPTURES
        or plan.get("capture_count") != EXPECTED_CAPTURES
    ):
        raise WP2Error("capture plan does not match relation-repair contract")
    verify_execution_lock(workspace, plan_path, lock_path, output_root)
    captures = list(plan["captures"])
    selected = {value.strip() for value in args.family_ids.split(",") if value.strip()}
    if selected and args.max_captures:
        raise WP2Error("--family-ids and --max-captures are mutually exclusive")
    if selected:
        known = {value["family_id"] for value in captures}
        if selected - known:
            raise WP2Error(f"unknown repair family IDs: {sorted(selected - known)}")
        captures = [value for value in captures if value["family_id"] in selected]
    if args.max_captures:
        if not 1 <= args.max_captures <= EXPECTED_CAPTURES:
            raise WP2Error("--max-captures must be 1..30")
        captures = captures[: args.max_captures]

    output_root.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = RepairCapture(plan, output_root, args.settle_sec)
    try:
        for index, capture in enumerate(captures, start=1):
            node.run([capture])
            write_progress(output_root, plan_path, lock_path, plan, capture["family_id"])
            node.get_logger().info(
                f"V21_REPAIR_CAPTURE_PROGRESS {index}/{len(captures)} {capture['capture_id']}"
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    final = raw_inventory(output_root, plan)
    print(
        "DATASET_V2_RELATION_REPAIR_CAPTURE_COMPLETE "
        f"captures={final['capture_count']} complete={final['complete']} moveit_used=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
