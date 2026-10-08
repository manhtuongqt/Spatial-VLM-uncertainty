#!/usr/bin/env python3
"""Create the WP0 hash inventory without modifying either source repository."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "protocol" / "source_inventory.json"
REPOSITORIES = ("RoboRefer", "ur3")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str, binary: bool = False):
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def dirty_entries(repo: Path) -> list[dict]:
    raw = git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all", binary=True)
    fields = raw.split(b"\0")
    entries = []
    index = 0
    while index < len(fields) and fields[index]:
        field = fields[index]
        status = field[:2].decode("ascii", errors="replace")
        relative = field[3:].decode("utf-8", errors="surrogateescape")
        index += 1
        source_path = None
        if "R" in status or "C" in status:
            source_path = relative
            relative = fields[index].decode("utf-8", errors="surrogateescape")
            index += 1
        path = repo / relative
        item = {
            "git_status": status,
            "path": f"{repo.name}/{relative}",
            "source_path": f"{repo.name}/{source_path}" if source_path else None,
            "exists": path.is_file(),
        }
        if path.is_file():
            item.update(size_bytes=path.stat().st_size, sha256=sha256_file(path))
        else:
            item.update(size_bytes=None, sha256=None)
        entries.append(item)
    return entries


def file_record(relative: str) -> dict:
    path = ROOT / relative
    return {
        "path": relative,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def aggregate_directory(relative: str) -> dict:
    root = ROOT / relative
    entries = []
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        file_hash = sha256_file(path)
        rel = str(path.relative_to(root))
        line = f"{file_hash}  {rel}\n".encode("utf-8")
        digest.update(line)
        entries.append({"path": rel, "size_bytes": path.stat().st_size, "sha256": file_hash})
    return {
        "path": relative,
        "file_count": len(entries),
        "total_size_bytes": sum(item["size_bytes"] for item in entries),
        "canonical_manifest_sha256": digest.hexdigest(),
        "canonical_rule": "SHA256(concatenated '<file_sha256>  <relative_path>\\n' sorted by relative path)",
    }


def main() -> None:
    repos = []
    all_entries = []
    for name in REPOSITORIES:
        repo = ROOT / name
        entries = dirty_entries(repo)
        statuses = Counter(item["git_status"] for item in entries)
        repos.append(
            {
                "name": name,
                "root": str(repo),
                "branch": git(repo, "branch", "--show-current"),
                "head": git(repo, "rev-parse", "HEAD"),
                "dirty_entry_count": len(entries),
                "status_counts": dict(sorted(statuses.items())),
                "entries": entries,
            }
        )
        all_entries.extend(entries)

    critical_paths = [
        "plan/ke_hoach_moi_roborefer_llm_vlm_agent_ur3.md",
        "RoboRefer/API/api.py",
        "RoboRefer/API/query_model.py",
        "RoboRefer/models/RoboRefer-2B-SFT/config.json",
        "ur3/ur3_perception/CMakeLists.txt",
        "ur3/ur3_perception/package.xml",
        "ur3/ur3_perception/config/roborefer_pilot_v0_gate.yaml",
        "ur3/ur3_perception/config/roborefer_pilot_v0_scenes.yaml",
        "ur3/ur3_perception/scripts/depth_component_gate_v2.py",
        "ur3/ur3_perception/scripts/roborefer_grounder.py",
        "ur3/ur3_perception/scripts/rgbd_object_pose.py",
        "ur3/ur3_moveit_control/scripts/fixed_pick_place.py",
        "ur3/ur3_moveit_control/test/test_pose_generation.py",
    ]
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "scope": "Every modified/untracked file reported by Git in RoboRefer and ur3, plus critical WP0 source/config and the locked pilot aggregate.",
        "repositories": repos,
        "totals": {
            "dirty_entry_count": len(all_entries),
            "hashed_existing_entry_count": sum(item["sha256"] is not None for item in all_entries),
        },
        "critical_source_and_config": [file_record(path) for path in critical_paths],
        "locked_pilot": aggregate_directory("results/roborefer_pilot_v0_20260813_173305"),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(all_entries)} dirty entries")


if __name__ == "__main__":
    main()

