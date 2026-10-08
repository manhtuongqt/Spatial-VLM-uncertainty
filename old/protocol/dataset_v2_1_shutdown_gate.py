#!/usr/bin/env python3
"""Evaluate the isolated Dataset V2.1 ordered-shutdown canary gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
GATE_ID = "FIX_V2_1_SHUTDOWN_GATE"
PARTITION = "ur3_roborefer_dataset_v2_1_relation_repair_local"
PROCESS_MARKERS = ("ign gazebo", "/move_group", "/servo_node_main", "/ur3_control_node")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def partition_orphans() -> list[dict[str, Any]]:
    findings = []
    expected = f"IGN_PARTITION={PARTITION}".encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            environment = (entry / "environ").read_bytes().split(b"\0")
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if expected in environment and any(marker in cmdline for marker in PROCESS_MARKERS):
            findings.append({"pid": int(entry.name), "cmdline": cmdline})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ros-log-root", required=True)
    parser.add_argument(
        "--result-root",
        default="results/dataset_v2_1_shutdown_gate_20260824",
    )
    args = parser.parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    ros_log_root = Path(args.ros_log_root).expanduser().resolve()
    result_root = (WORKSPACE / args.result_root).resolve()
    lock = read_json(WORKSPACE / "protocol/dataset_v2_1_shutdown_gate_execution_lock.json")
    plan = read_json(WORKSPACE / "protocol/dataset_v2_1_shutdown_gate_capture_plan.json")
    failures: list[str] = []
    checks: dict[str, Any] = {}

    shutdown_path = output_root / "report_assets/checkpoints/shutdown_sequence.json"
    shutdown = read_json(shutdown_path) if shutdown_path.is_file() else {}
    shutdown_ok = (
        shutdown.get("gate_id") == GATE_ID
        and shutdown.get("ordered_shutdown_complete") is True
        and shutdown.get("remaining_target_pids") == []
        and all(not stage.get("remaining_pids") for stage in shutdown.get("stages", []))
        and all(shutdown.get("initial_target_pids", {}).get(name) for name in (
            "action_server", "servo", "move_group", "gazebo"
        ))
    )
    checks["ordered_shutdown"] = {
        "passed": shutdown_ok,
        "sequence_report": str(shutdown_path),
        "initial_target_pids": shutdown.get("initial_target_pids", {}),
        "remaining_target_pids": shutdown.get("remaining_target_pids"),
    }
    if not shutdown_ok:
        failures.append("ordered shutdown sequence incomplete")

    log_findings = []
    for path in sorted(ros_log_root.rglob("*")) if ros_log_root.is_dir() else []:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for marker in ("Segmentation fault", "exit code -11"):
            if marker.lower() in text.lower():
                log_findings.append({"path": str(path), "marker": marker})
    no_segfault = not log_findings
    checks["no_move_group_segmentation_fault"] = {
        "passed": no_segfault,
        "ros_log_root": str(ros_log_root),
        "findings": log_findings,
    }
    if not no_segfault:
        failures.append("segmentation fault remains in shutdown logs")

    orphans = partition_orphans()
    checks["no_scoped_orphan_process"] = {
        "passed": not orphans,
        "partition": PARTITION,
        "orphans": orphans,
    }
    if orphans:
        failures.append("scoped ROS/Gazebo process remains after launch exit")

    capture_root = output_root / "raw/captures"
    capture_dirs = sorted(value for value in capture_root.iterdir() if value.is_dir()) \
        if capture_root.is_dir() else []
    data_failures = []
    expected_id = plan["captures"][0]["capture_id"]
    if len(capture_dirs) != 1 or capture_dirs[0].name != expected_id:
        data_failures.append(f"capture inventory is not exactly the first canary: {[v.name for v in capture_dirs]}")
    for directory in capture_dirs:
        meta_path = directory / "capture_meta.json"
        if not meta_path.is_file():
            data_failures.append(f"missing capture_meta.json: {directory.name}")
            continue
        meta = read_json(meta_path)
        if meta.get("sensor_qc", {}).get("passed") is not True:
            data_failures.append(f"sensor QC not passed: {directory.name}")
        for relative, expected in meta.get("artifact_sha256", {}).items():
            path = directory / relative
            if not path.is_file() or sha256(path) != expected:
                data_failures.append(f"artifact hash mismatch: {directory.name}/{relative}")
    raw_manifest_path = output_root / "raw/raw_capture_manifest.json"
    raw_manifest = read_json(raw_manifest_path) if raw_manifest_path.is_file() else {}
    if raw_manifest.get("capture_count") != 1:
        data_failures.append("raw manifest does not contain exactly one capture")
    checks["canary_data_and_hash"] = {
        "passed": not data_failures,
        "capture_count": len(capture_dirs),
        "raw_manifest_capture_count": raw_manifest.get("capture_count"),
        "failures": data_failures,
    }
    if data_failures:
        failures.append("canary capture or artifact hash failed")

    baseline_mismatches = []
    for relative, expected in lock.get("baseline_expected_sha256", {}).items():
        path = WORKSPACE / relative
        if not path.is_file() or sha256(path) != expected:
            baseline_mismatches.append(relative)
    checks["baseline_hash_unchanged"] = {
        "passed": not baseline_mismatches,
        "checked_count": len(lock.get("baseline_expected_sha256", {})),
        "mismatches": baseline_mismatches,
    }
    if baseline_mismatches:
        failures.append("protected baseline hash changed")

    decision = "PASS_V2_1_SHUTDOWN_GATE" if not failures else "FIX_V2_1_SHUTDOWN_FIRST"
    report = {
        "schema_version": 1,
        "gate_id": GATE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "passed": not failures,
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "training_performed": False,
        "official_400_capture_started": False,
        "baseline_modified": False,
    }
    write_json(result_root / "DATASET_V2_1_SHUTDOWN_GATE_REPORT.json", report)
    markdown = [
        "# Dataset V2.1 shutdown gate", "",
        f"- Decision: `{decision}`.",
        f"- Real canary capture: `{len(capture_dirs)}/1`.",
        f"- Segmentation-fault findings: `{len(log_findings)}`.",
        f"- Scoped orphan processes: `{len(orphans)}`.",
        f"- Protected baseline hashes: `{len(lock.get('baseline_expected_sha256', {})) - len(baseline_mismatches)}/{len(lock.get('baseline_expected_sha256', {}))}` matched.",
        "- Baseline modified: `False`.",
        "- Training performed: `False`.", "",
        "The V2.1 wrapper stops action-server/Servo clients first, contains the known installed MoveIt destructor crash, and stops Gazebo last.",
    ]
    (result_root / "DATASET_V2_1_SHUTDOWN_GATE_REPORT.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(
        "DATASET_V2_1_SHUTDOWN_GATE "
        f"decision={decision} failures={len(failures)} captures={len(capture_dirs)}"
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
