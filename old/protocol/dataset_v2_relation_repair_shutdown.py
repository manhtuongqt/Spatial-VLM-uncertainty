#!/usr/bin/env python3
"""Ordered process shutdown for the Dataset V2.1 capture wrapper only.

The qualified baseline launch files are included unchanged.  This coordinator
runs only after the capture process has exited and contains the known MoveIt
shutdown crash by terminating move_group after its clients have stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any


TARGET_MARKERS = {
    "action_server": ("/ur3_control_node",),
    "servo": ("/servo_node_main",),
    "move_group": ("/move_group",),
    "gazebo": ("/usr/bin/ign gazebo", "ign gazebo"),
}


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


def process_table() -> dict[int, dict[str, Any]]:
    table: dict[int, dict[str, Any]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            remainder = stat[stat.rfind(")") + 2 :].split()
            state, ppid = remainder[0], int(remainder[1])
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        table[pid] = {"pid": pid, "ppid": ppid, "state": state, "cmdline": cmdline}
    return table


def descendants(root_pid: int, table: dict[int, dict[str, Any]]) -> set[int]:
    selected: set[int] = set()
    frontier = {root_pid}
    while frontier:
        children = {
            pid for pid, row in table.items()
            if row["ppid"] in frontier and pid not in selected
        }
        selected.update(children)
        frontier = children
    return selected


def classify(root_pid: int) -> dict[str, list[int]]:
    table = process_table()
    scoped = descendants(root_pid, table)
    groups = {name: [] for name in TARGET_MARKERS}
    for pid in sorted(scoped):
        if pid == os.getpid():
            continue
        cmdline = table[pid]["cmdline"]
        for name, markers in TARGET_MARKERS.items():
            if any(marker in cmdline for marker in markers):
                groups[name].append(pid)
                break
    return groups


def alive(pids: list[int]) -> list[int]:
    table = process_table()
    return [pid for pid in pids if pid in table and table[pid]["state"] != "Z"]


def wait_gone(pids: list[int], timeout_sec: float) -> list[int]:
    deadline = time.monotonic() + timeout_sec
    remaining = alive(pids)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.05)
        remaining = alive(pids)
    return remaining


def send(pids: list[int], value: signal.Signals) -> list[int]:
    sent = []
    for pid in pids:
        try:
            os.kill(pid, value)
            sent.append(pid)
        except ProcessLookupError:
            pass
    return sent


def graceful_stage(root_pid: int, names: tuple[str, ...], timeout_sec: float = 8.0) -> dict[str, Any]:
    groups = classify(root_pid)
    requested = sorted({pid for name in names for pid in groups[name]}, reverse=True)
    sent_int = send(requested, signal.SIGINT)
    remaining = wait_gone(requested, timeout_sec)
    sent_kill = send(remaining, signal.SIGKILL)
    final_remaining = wait_gone(remaining, 3.0)
    return {
        "targets": list(names),
        "requested_pids": requested,
        "sigint_pids": sent_int,
        "fallback_sigkill_pids": sent_kill,
        "remaining_pids": final_remaining,
    }


def contained_move_group_stage(root_pid: int) -> dict[str, Any]:
    groups = classify(root_pid)
    requested = sorted(groups["move_group"], reverse=True)
    # The installed MoveIt process consistently crashes in its SIGINT destructor.
    # Capture is already committed, and all MoveIt clients are stopped first, so
    # bypass that destructor without changing the baseline executable.
    sent_kill = send(requested, signal.SIGKILL)
    remaining = wait_gone(requested, 5.0)
    return {
        "targets": ["move_group"],
        "policy": "SIGKILL_AFTER_CLIENTS_STOPPED_AVOIDS_KNOWN_SIGINT_DESTRUCTOR_CRASH",
        "requested_pids": requested,
        "sigkill_pids": sent_kill,
        "remaining_pids": remaining,
    }


def gazebo_stage(root_pid: int) -> dict[str, Any]:
    groups = classify(root_pid)
    requested = sorted(groups["gazebo"], reverse=True)
    sent_int = send(requested, signal.SIGINT)
    remaining = wait_gone(requested, 5.0)
    sent_term = send(remaining, signal.SIGTERM)
    remaining = wait_gone(remaining, 3.0)
    sent_kill = send(remaining, signal.SIGKILL)
    final_remaining = wait_gone(remaining, 3.0)
    return {
        "targets": ["gazebo"],
        "requested_pids": requested,
        "sigint_pids": sent_int,
        "fallback_sigterm_pids": sent_term,
        "fallback_sigkill_pids": sent_kill,
        "remaining_pids": final_remaining,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    launch_pid = os.getppid()
    initial = classify(launch_pid)
    stages = [
        graceful_stage(launch_pid, ("action_server", "servo")),
        contained_move_group_stage(launch_pid),
        gazebo_stage(launch_pid),
    ]
    final = classify(launch_pid)
    remaining = sorted({pid for values in final.values() for pid in values})
    report = {
        "schema_version": 1,
        "gate_id": "FIX_V2_1_SHUTDOWN_GATE",
        "launch_pid": launch_pid,
        "coordinator_pid": os.getpid(),
        "initial_target_pids": initial,
        "stages": stages,
        "final_target_pids": final,
        "remaining_target_pids": remaining,
        "ordered_shutdown_complete": not remaining,
        "baseline_files_modified": False,
    }
    atomic_json(
        output_root / "report_assets/checkpoints/shutdown_sequence.json", report
    )
    print(
        "DATASET_V2_1_ORDERED_SHUTDOWN "
        f"complete={str(not remaining).lower()} remaining={len(remaining)}"
    )
    return 0 if not remaining else 2


if __name__ == "__main__":
    raise SystemExit(main())
