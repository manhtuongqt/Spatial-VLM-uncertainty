#!/usr/bin/env python3
"""Create the preregistered WP1 manifest before any model query."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from wp1_common import (
    CONDITIONS,
    EXPECTED_MODEL_INVENTORY_SHA256,
    PROTOCOL_ID,
    pilot_tree_digest,
    sha256_file,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "protocol/wp1_manifest.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace locked WP1 manifest: {output}")
    result_root = Path(args.result_root)
    if result_root.is_absolute():
        result_relative = str(result_root.resolve().relative_to(ROOT.resolve()))
    else:
        result_relative = str(result_root)
    dataset_relative = "results/roborefer_pilot_v0_20260813_173305/dataset_attempt_02"
    pilot_relative = "results/roborefer_pilot_v0_20260813_173305"
    source_paths = [
        "protocol/wp1_common.py",
        "protocol/wp1_runner.py",
        "protocol/reason_code_registry_v1.json",
        "protocol/reason_event_v1.schema.json",
        "RoboRefer/API/api.py",
    ]
    evaluator_paths = [
        "protocol/wp1_evaluate.py",
        "protocol/test_wp1_depth_sensitivity.py",
        "ur3/ur3_perception/config/roborefer_pilot_v0_annotations.yaml",
        "ur3/ur3_perception/config/roborefer_pilot_v0_gate.yaml",
    ]
    missing = [path for path in source_paths + evaluator_paths if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"WP1 sources must exist before preregistration: {missing}")
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": dataset_relative,
        "result_root": result_relative,
        "scene_count": 10,
        "condition_count": 6,
        "expected_record_count": 60,
        "query_attempts_per_scene_condition": 1,
        "condition_order": list(CONDITIONS),
        "conditions": {
            "rgb_only": {"enable_depth": False, "transform": "none"},
            "correct_depth": {"enable_depth": True, "transform": "locked inverse_depth_percentile_v1"},
            "flat_depth": {"enable_depth": True, "transform": "all pixels uint8=127"},
            "shuffled_depth": {"enable_depth": True, "transform": "cyclic next-scene derangement"},
            "inverted_depth": {"enable_depth": True, "transform": "255-correct relative depth"},
            "localized_holes_edge": {
                "enable_depth": True,
                "transform": "zero Canny(24,72) depth edges dilated 7x7 plus three fixed 10% image patches",
                "uses_rgb_or_oracle_to_choose_holes": False,
            },
        },
        "request_lock": {
            "rgb": "source RGB decoded then JPEG quality=90",
            "prompt": "instruction + single space + coordinate_suffix",
            "generation_mode": "greedy",
            "random_seed": 8132026,
            "retry_count": 0,
        },
        "model": {
            "name": "RoboRefer-2B-SFT",
            "inventory_sha256": EXPECTED_MODEL_INVENTORY_SHA256,
        },
        "evaluation": {
            "oracle_access": "ONLY_AFTER_prediction_lock_verifies",
            "safe_eroded_margin_px": 4,
            "point_displacement": "Euclidean distance in normalized image coordinates and pixels",
            "instance_switch": "semantic label at counterfactual point differs from semantic label at correct-depth point",
            "task_groups": {
                "direct": ["direct_grounding"],
                "relation_or_comparison": [
                    "relative_2d",
                    "depth_relation",
                    "multi_reference",
                    "metric_comparison",
                    "shape_comparison",
                    "occlusion"
                ],
                "selective": ["ambiguous_command", "target_absent"]
            },
            "limitations_locked_before_run": [
                "Checkpoint emits one point and no heatmap, so WP1 cannot measure heatmap change.",
                "Ten-scene WP1 is diagnostic and cannot support significance claims."
            ]
        },
        "decision_thresholds": {
            "displacement_sensitive_normalized": 0.02,
            "required_correct_depth_parse_rate": 1.0,
            "minimum_correct_depth_point_hit_rate": 0.75,
            "maximum_correct_depth_hit_deficit_vs_rgb": 1,
            "go_pcra_f_max_sensitive_pair_rate": 0.20,
            "go_pcra_f_max_instance_switch_rate": 0.10
        },
        "decision_rule": [
            "FIX_DEPTH_PIPELINE_FIRST if correct-depth parse rate <1.0, correct-depth single-target hit rate <0.75, or correct depth trails RGB by more than one hit.",
            "Otherwise GO_PCRA_F if corrupted-pair displacement-sensitive rate <=0.20 and instance-switch rate <=0.10.",
            "Otherwise NO_GO_PCRA_F because the existing checkpoint measurably responds to depth; prioritize P-CRA-U/calibration/intervention."
        ],
        "pilot_lock": {
            "relative_path": pilot_relative,
            "tree_sha256": pilot_tree_digest(ROOT, pilot_relative),
            "file_count": sum(1 for item in (ROOT / pilot_relative).rglob("*") if item.is_file()),
        },
        "input_lock": {
            "capture_manifest_sha256": sha256_file(ROOT / dataset_relative / "capture_manifest.json"),
            "input_manifest_sha256": sha256_file(ROOT / dataset_relative / "input_manifest.jsonl"),
        },
        "runner_source_sha256": {path: sha256_file(ROOT / path) for path in source_paths},
        "evaluator_source_sha256": {path: sha256_file(ROOT / path) for path in evaluator_paths},
        "runner_oracle_access": False,
        "robot_manipulation_performed": False,
    }
    write_json(output, payload)
    print(f"WP1_PROTOCOL_LOCKED: {output}")


if __name__ == "__main__":
    main()

