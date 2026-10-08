#!/usr/bin/env python3
"""Replay WP2 derivation from locked raw captures and compare SHA-256 bytes."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp2_common import file_inventory, write_json  # noqa: E402
from wp2_materialize import materialize  # noqa: E402


def replay_validate(dataset_root: Path, selection: str = "all") -> dict:
    temporary = Path(tempfile.mkdtemp(prefix="wp2_replay_"))
    try:
        materialize(dataset_root, temporary, selection, validate_schema=True)
        generated = file_inventory(temporary)
        mismatches = []
        for item in generated:
            original = dataset_root / item["path"]
            if not original.is_file():
                mismatches.append({"path": item["path"], "reason": "missing_original"})
                continue
            from wp2_common import sha256_file
            observed = sha256_file(original)
            if observed != item["sha256"]:
                mismatches.append({
                    "path": item["path"], "reason": "sha256_mismatch",
                    "expected": item["sha256"], "observed": observed,
                })
        return {
            "passed": not mismatches,
            "selection": selection,
            "record_count": 15 if selection == "smoke" else 250,
            "generated_files_compared": len(generated),
            "mismatches": mismatches,
            "replay_definition": (
                "bitwise offline rematerialization from locked Gazebo raw captures; "
                "does not claim Gazebo rerender bitwise determinism"
            ),
        }
    finally:
        shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--selection", choices=("smoke", "all"), default="all")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = replay_validate(Path(args.dataset_root).expanduser().resolve(), args.selection)
    if args.output:
        write_json(Path(args.output).expanduser().resolve(), result)
    print(
        f"WP2_REPLAY_{'PASS' if result['passed'] else 'FAIL'} "
        f"files={result['generated_files_compared']}"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
