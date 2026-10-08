#!/usr/bin/env python3
"""Assemble the auditable WP0 JSON report from generated evidence."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol"
PILOT = ROOT / "results/roborefer_pilot_v0_20260813_173305"


def load(name: str) -> dict:
    return json.loads((PROTOCOL / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_directory(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(
            f"{sha256_file(path)}  {path.relative_to(root)}\n".encode("utf-8")
        )
    return digest.hexdigest()


def parse_pytest_log(name: str) -> dict:
    text = (PROTOCOL / "logs" / name).read_text(encoding="utf-8")
    match = re.search(r"(\d+) passed(?:, (\d+) warnings?)? in ([0-9.]+)s", text)
    return {
        "passed": bool(match),
        "passed_count": int(match.group(1)) if match else None,
        "warning_count": int(match.group(2) or 0) if match else None,
        "duration_s": float(match.group(3)) if match else None,
        "log": f"protocol/logs/{name}",
    }


def main() -> None:
    core = load("core_check_result.json")
    inventory = load("source_inventory.json")
    api = load("api_smoke_result.json")
    demo = load("reason_code_demo.json")
    baseline = parse_pytest_log("wp0_baseline_54_pytest.log")
    extension = parse_pytest_log("wp0_extension_18_pytest.log")
    current_pilot_digest = aggregate_directory(PILOT)
    inventory_pilot_digest = inventory["locked_pilot"]["canonical_manifest_sha256"]
    critical = inventory["critical_source_and_config"]
    installed_gate = ROOT / "install/ur3_perception/lib/ur3_perception/depth_component_gate_v2.py"

    gates = {
        "clean_ros_build": bool(core["build"]["passed"]),
        "unit_tests_one_reproducible_command": bool(core["pytest"]["passed"]),
        "baseline_54_tests": baseline["passed"] and baseline["passed_count"] == 54,
        "moveit_and_new_contract_tests": extension["passed"] and extension["passed_count"] == 18,
        "ur3_perception_interfaces_importable": bool(core["interface_import"]["passed"]),
        "reason_code_demo_6_of_6": bool(demo["all_cases_passed"] and len(demo["cases"]) == 6),
        "roborefer_rgb_and_rgbd_api_smoke": bool(api["all_passed"]),
        "locked_pilot_unchanged_during_checks": bool(
            core["pilot_lock"]["unchanged"]
            and current_pilot_digest == inventory_pilot_digest
        ),
        "dirty_source_inventory_complete_and_hashed": bool(
            inventory["totals"]["dirty_entry_count"]
            == inventory["totals"]["hashed_existing_entry_count"]
        ),
        "critical_source_and_config_hashed": all(
            item["exists"] and bool(item["sha256"]) for item in critical
        ),
        "gate_v2_installed": installed_gate.is_file(),
    }
    build_log = (PROTOCOL / "logs/wp0_build.log").read_text(encoding="utf-8")
    stderr_match = re.search(r"\d+ packages had stderr output: (.+)", build_log)
    payload = {
        "schema_version": 1,
        "protocol_id": "WP0_STARTING_POINT_LOCK",
        "project_title_vi": "Nghiên cứu phương pháp LLM/VLM Agent lập kế hoạch vòng kín với hiệu chuẩn độ bất định không gian và can thiệp thích ứng cho thao tác gắp đặt bằng tay máy UR3.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_passed": all(gates.values()),
        "gates": gates,
        "build": {
            **core["build"],
            "scope": "ROS dependency closure selected by --packages-up-to ur3_perception (9 packages)",
            "cmake_clean_cache": True,
            "python_executable": "/usr/bin/python3 (3.10)",
            "stderr_warning_packages": stderr_match.group(1).split() if stderr_match else [],
            "stderr_warnings_are_non_fatal": True,
            "log": "protocol/logs/wp0_build.log",
        },
        "tests": {
            "one_command": "./protocol/run_wp0_checks.sh",
            "environment": core["environment"],
            "combined": core["pytest"],
            "baseline_perception_and_pilot": {
                **baseline,
                "expected_count": 54,
                "files": [
                    "ur3/ur3_perception/test/test_roborefer_adapter.py",
                    "ur3/ur3_perception/test/test_roborefer_dimension_comparator.py",
                    "ur3/ur3_perception/test/test_roborefer_pilot.py",
                ],
            },
            "extension_moveit_geometry_camera": {
                **extension,
                "expected_count": 18,
                "files": [
                    "ur3/ur3_perception/test/test_projection.py",
                    "ur3/ur3_perception/test/test_depth_component_gate_v2.py",
                    "ur3/ur3_moveit_control/test/test_pose_generation.py",
                    "ur3/ur3_moveit_control/test/test_d435i_description.py",
                ],
            },
            "warning_note": "Five PendingDeprecationWarning instances originate in ROS image_geometry using numpy.matrix; no test failed.",
        },
        "interface_and_install": {
            "object_observation_import": core["interface_import"],
            "gate_v2_installed_path": str(installed_gate.relative_to(ROOT)),
            "gate_v2_installed_sha256": sha256_file(installed_gate) if installed_gate.is_file() else None,
        },
        "reason_codes": {
            "implemented": [
                "AREA_TOO_LARGE",
                "AREA_TOO_SMALL",
                "NO_SEED_SUPPORT",
                "PLANE_MERGE",
                "BORDER_TRUNCATED",
                "UNSTABLE_3D",
            ],
            "demo_all_passed": demo["all_cases_passed"],
            "cases": [
                {
                    "expected": item["expected_reason_code"],
                    "observed": item["reason_code"],
                    "passed": item["demo_passed"],
                }
                for item in demo["cases"]
            ],
            "pilot_scene_0007_post_hoc": {
                "reason_code": demo["locked_pilot_post_hoc_diagnostic"].get("reason_code"),
                "interpretation": "POST_HOC_DIAGNOSTIC_ONLY",
            },
            "json_evidence": "protocol/reason_code_demo.json",
            "visual_demo": "protocol/WP0_DEMO.html",
        },
        "api_smoke": {
            "all_passed": api["all_passed"],
            "scene_id": api["scene_id"],
            "queries": api["queries"],
            "checkpoint_inventory_sha256": api["queries"][0]["model_fingerprint"]["inventory_sha256"],
            "evidence": "protocol/api_smoke_result.json",
        },
        "source_inventory": {
            "dirty_entry_count": inventory["totals"]["dirty_entry_count"],
            "hashed_existing_entry_count": inventory["totals"]["hashed_existing_entry_count"],
            "repositories": [
                {
                    "name": repo["name"],
                    "branch": repo["branch"],
                    "head": repo["head"],
                    "dirty_entry_count": repo["dirty_entry_count"],
                }
                for repo in inventory["repositories"]
            ],
            "critical_file_count": len(critical),
            "evidence": "protocol/source_inventory.json",
        },
        "pilot_lock": {
            **core["pilot_lock"],
            "file_count": inventory["locked_pilot"]["file_count"],
            "stable_relative_manifest_sha256": inventory_pilot_digest,
            "current_relative_manifest_sha256": current_pilot_digest,
            "current_matches_inventory": current_pilot_digest == inventory_pilot_digest,
            "note": "The before/after digest uses paths relative to the workspace; the second canonical digest omits the artifact-root prefix and is portable within the artifact tree.",
        },
        "limitations": [
            "Reason-code branch demos are deterministic contract tests; thresholds still require calibration on real RGB-D failure data.",
            "PLANE_MERGE is an observable geometry heuristic (oversize component reaching ROI boundary), not semantic ground truth.",
            "BORDER_TRUNCATED is implemented as a fail-closed option but defaults to disabled until operational calibration.",
            "API smoke validates real model loading, transport, decoding policy and output contract; it is not a new semantic-accuracy experiment.",
            "WP0 does not execute physical or simulated robot manipulation; closed-loop manipulation belongs to later work packages.",
        ],
    }
    (PROTOCOL / "test_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"WP0_REPORT {'PASS' if payload['overall_passed'] else 'FAIL'}")
    if not payload["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
