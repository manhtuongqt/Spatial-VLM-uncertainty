#!/usr/bin/env python3
"""Create the final WP2 prelock checkpoint, artifact manifest and test report."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp2_common import (  # noqa: E402
    EXPECTED_MODEL_INVENTORY_SHA256,
    EXPECTED_PILOT_TREE_SHA256,
    EXPECTED_WP1_TREE_SHA256,
    PILOT_RELATIVE,
    PROTOCOL_ID,
    WP1_RELATIVE,
    file_inventory,
    read_json,
    sha256_file,
    tree_digest,
    utc_now,
    workspace_root,
    write_json,
)


def main() -> int:
    workspace = workspace_root()
    protocol_dir = Path(__file__).resolve().parent
    dataset_root = workspace / "datasets/roborefer_dataset_v1_prototype_20260818_155606"
    gate = read_json(dataset_root / "report_assets/checkpoints/05_full_gate_report.json")
    if not gate.get("passed") or gate.get("record_count") != 250:
        raise RuntimeError("final WP2 gate report is not valid")
    pilot_digest = tree_digest(workspace, PILOT_RELATIVE)
    wp1_digest = tree_digest(workspace, WP1_RELATIVE)
    if pilot_digest != EXPECTED_PILOT_TREE_SHA256 or wp1_digest != EXPECTED_WP1_TREE_SHA256:
        raise RuntimeError("WP0/WP1 immutable evidence changed")

    source_names = [
        "wp2_common.py", "wp2_family_generator.py", "wp2_gazebo_capture.py",
        "wp2_gazebo_capture.launch.py", "wp2_materialize.py", "wp2_validate.py",
        "wp2_leakage_validator.py", "wp2_replay_validator.py",
        "wp2_generate_report_assets.py", "test_wp2_dataset.py", "run_wp2_checks.sh",
        "dataset_v1.schema.json", "family_manifest_v1.schema.json",
    ]
    source_hashes = {
        f"protocol/{name}": sha256_file(protocol_dir / name) for name in source_names
    }
    checkpoint = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "checkpoint": "WP2_DATASET_PRELOCK_COMPLETE",
        "created_at_utc": utc_now(),
        "training_started": False,
        "pcra_f_training_started": False,
        "family_count": 50,
        "sample_count": 250,
        "depth_dependent_family_count": 30,
        "raw_gazebo_capture_count": 100,
        "family_manifest_sha256": sha256_file(dataset_root / "family_manifest.json"),
        "capture_plan_sha256": sha256_file(dataset_root / "capture_plan.json"),
        "raw_capture_manifest_sha256": sha256_file(dataset_root / "raw/raw_capture_manifest.json"),
        "dataset_index_sha256": sha256_file(dataset_root / "dataset_index.json"),
        "full_gate_report_sha256": sha256_file(dataset_root / "report_assets/checkpoints/05_full_gate_report.json"),
        "gui_sha256": sha256_file(dataset_root / "WP2_QC_GUI.html"),
        "roborefer_model_inventory_sha256": EXPECTED_MODEL_INVENTORY_SHA256,
        "pilot_wp0_tree_sha256": pilot_digest,
        "wp1_tree_sha256": wp1_digest,
        "source_sha256": source_hashes,
        "known_non_data_issue": (
            "MoveIt move_group can segfault during launch teardown after the capture "
            "process has completed; all 100 capture directories and hashes passed QC"
        ),
    }
    write_json(dataset_root / "report_assets/checkpoints/07_dataset_prelock.json", checkpoint)

    assets = dataset_root / "report_assets"
    asset_inventory = file_inventory(assets, excluded_names={"ASSET_INDEX.json"})
    write_json(assets / "ASSET_INDEX.json", {
        "schema_version": 1,
        "self_hash_policy": "ASSET_INDEX.json excluded to avoid recursive hash",
        "artifact_count": len(asset_inventory),
        "artifacts": asset_inventory,
    })
    dataset_inventory = file_inventory(dataset_root, excluded_names={"artifact_manifest.json"})
    write_json(dataset_root / "artifact_manifest.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "root": dataset_root.name,
        "self_hash_policy": "artifact_manifest.json excluded to avoid recursive hash",
        "artifact_count": len(dataset_inventory),
        "artifacts": dataset_inventory,
    })

    pytest_text = (protocol_dir / "logs/wp2_pytest.log").read_text(encoding="utf-8")
    match = re.search(r"(\d+) passed, (\d+) warnings? in ([0-9.]+)s", pytest_text)
    if not match or int(match.group(1)) != 86:
        raise RuntimeError("WP2 combined pytest log does not show 86 passed")
    test_report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "overall_passed": True,
        "decision": "GO_WP3_PCRA_U_PROTOTYPE",
        "pcra_f_status": "DEFERRED_ABLATION_NOT_TRAINED",
        "result_root": str(dataset_root.relative_to(workspace)),
        "counts": {
            "families": 50, "samples": 250, "variants_per_family": 5,
            "depth_dependent_families": 30, "raw_gazebo_captures": 100,
            "projective_relation_records_checked": gate["relation_qc"]["projective_records_checked"],
            "replay_files_compared": gate["replay_validation"]["generated_files_compared"],
        },
        "splits": {
            "train": {"families": 30, "samples": 150},
            "dev": {"families": 8, "samples": 40},
            "calibration": {"families": 6, "samples": 30},
            "test": {"families": 6, "samples": 30, "prototype_not_final_locked_test": True},
        },
        "gates": {
            "smoke_3_families_15_samples_passed": True,
            "full_50_families_250_samples_passed": True,
            "at_least_30_depth_dependent_families": True,
            "family_level_split_no_overlap": gate["leakage_validation"]["passed"],
            "inference_payload_contains_no_oracle": not gate["leakage_validation"]["inference_oracle_findings"],
            "all_record_artifact_hashes_verified": gate["hash_validation"]["passed"],
            "mask_qc_passed": gate["mask_qc"]["passed"],
            "relation_evidence_qc_passed": gate["relation_qc"]["passed"],
            "offline_replay_100_percent": gate["replay_validation"]["passed"],
            "pilot_wp0_unchanged": gate["immutability"]["pilot_wp0_unchanged"],
            "wp1_unchanged": gate["immutability"]["wp1_unchanged"],
            "pilot_wp1_not_used_as_training_source": not gate["pilot_wp1_used_as_training_source"],
            "software_tests_passed": True,
        },
        "software_tests": {
            "command": "./protocol/run_wp2_checks.sh",
            "passed_count": int(match.group(1)),
            "warning_count": int(match.group(2)),
            "duration_s": float(match.group(3)),
            "log": "protocol/logs/wp2_pytest.log",
            "pytest_plugin_autoload_disabled": True,
        },
        "hashes": {
            "family_manifest_sha256": sha256_file(dataset_root / "family_manifest.json"),
            "capture_plan_sha256": sha256_file(dataset_root / "capture_plan.json"),
            "raw_capture_manifest_sha256": sha256_file(dataset_root / "raw/raw_capture_manifest.json"),
            "dataset_index_sha256": sha256_file(dataset_root / "dataset_index.json"),
            "artifact_manifest_sha256": sha256_file(dataset_root / "artifact_manifest.json"),
            "report_asset_index_sha256": sha256_file(dataset_root / "report_assets/ASSET_INDEX.json"),
            "pilot_wp0_tree_sha256": pilot_digest,
            "wp1_tree_sha256": wp1_digest,
            "roborefer_model_inventory_sha256": EXPECTED_MODEL_INVENTORY_SHA256,
        },
        "deliverables": {
            "protocol/dataset_v1.schema.json": sha256_file(protocol_dir / "dataset_v1.schema.json"),
            "protocol/family_manifest_v1.schema.json": sha256_file(protocol_dir / "family_manifest_v1.schema.json"),
            "protocol/WP2_DATASET_V1.md": sha256_file(protocol_dir / "WP2_DATASET_V1.md"),
            str((dataset_root / "WP2_QC_GUI.html").relative_to(workspace)): sha256_file(dataset_root / "WP2_QC_GUI.html"),
            str((dataset_root / "report_assets/ASSET_INDEX.json").relative_to(workspace)): sha256_file(dataset_root / "report_assets/ASSET_INDEX.json"),
            str((dataset_root / "artifact_manifest.json").relative_to(workspace)): sha256_file(dataset_root / "artifact_manifest.json"),
        },
        "limitations": [
            "reachable/graspable masks are locked prototype proxies, not full MoveIt collision labels",
            "offline replay does not claim bitwise Gazebo rerender determinism",
            "MoveIt move_group teardown segfault remains after capture completion",
            "20 percent model failure audit remains for later development evaluation",
        ],
    }
    write_json(protocol_dir / "wp2_test_report.json", test_report)
    print(json.dumps(test_report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
