#!/usr/bin/env python3
"""Append-only cracker-box occlusion characterization after v1 rejection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

import gazebo_uq_occlusion_characterization_v1 as base
from generate_gazebo_train_uq_v1_contract import CAMERA, LABELS, OBJECTS


ROOT = base.ROOT
CONFIG = base.CONFIG
PROTOCOL_ID = "gazebo_uq_occlusion_characterization_v2"
SCENES = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
LOCK = ROOT / "protocol/gazebo_uq_occlusion_characterization_v2_capture_lock.json"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_uq_occlusion_characterization_v2"
CAPTURE = RESULT / "capture_attempt_01"
REPORT = RESULT / "OCCLUSION_CHARACTERIZATION_REPORT.json"
REPORT_MD = RESULT / "OCCLUSION_CHARACTERIZATION_REPORT.md"
OCCLUDER = "ycb_cracker_box"
OCCLUDER_LABEL = 21
TARGET_XY = (-0.220, 0.300)
# Projection-aligned box center for target center under the frozen camera TF.
ALIGNED_OCCLUDER_XY = (-0.210121, 0.283509)
GAPS_M = tuple(round(0.100 + 0.002 * i, 3) for i in range(21))
SIDES = (-1, 1)


def build() -> tuple[dict, dict, dict]:
    rows, oracle = [], {}
    index = 0
    for target in base.TARGETS:
        for target_yaw in base.TARGET_YAWS:
            for side in SIDES:
                for gap in GAPS_M:
                    sid = f"uq_occ_char_v2_{index:03d}"
                    poses = {
                        target: [TARGET_XY[0], TARGET_XY[1], target_yaw],
                        OCCLUDER: [ALIGNED_OCCLUDER_XY[0], round(ALIGNED_OCCLUDER_XY[1] + side * gap, 6), 0.0],
                    }
                    signature = base.layout_signature(poses)
                    rows.append({
                        "scene_id": sid,
                        "scene_family_id": f"engineering_occlusion_characterization_v2/family_{index:03d}",
                        "layout_id": f"occ_sweep_v2_{index:03d}",
                        "layout_signature_sha256": signature,
                        "task_type": "engineering_geometry_characterization_no_inference",
                        "instruction": "ENGINEERING ONLY: do not run model inference on this capture.",
                        "poses": poses,
                    })
                    oracle[sid] = {
                        "usage": "evaluator_only_geometry_characterization",
                        "target_id": target, "target_label": LABELS[target], "target_yaw_rad": target_yaw,
                        "occluder_id": OCCLUDER, "occluder_label": OCCLUDER_LABEL,
                        "side": side, "center_gap_m": gap, "layout_signature_sha256": signature,
                    }
                    index += 1
    scenes = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "expected_scene_count": len(rows),
        "capture_mode": "engineering_geometry_only_no_manipulation", "random_seed": 14092027,
        "view_joint_pose": CAMERA["view_joint_pose"], "camera_frame": CAMERA["frame"], "base_frame": CAMERA["base_frame"],
        "coordinate_suffix": "ENGINEERING ONLY; MODEL INFERENCE FORBIDDEN.", "objects": OBJECTS, "scenes": rows,
    }
    annotations = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "oracle_usage": "geometry_characterization_only_never_train_or_val",
        "parent_negative_attempt": {"protocol_id": base.PROTOCOL_ID, "report_sha256": base.sha256(base.REPORT), "decision": "REJECT"},
        "occluder": {"id": OCCLUDER, "semantic_label": OCCLUDER_LABEL, "candidate": False,
                     "reason": "tall opaque non-candidate object; blue cube v1 produced zero overlap"},
        "scenes": oracle,
    }
    gate = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "official_pilot_gate_unchanged": {"min_inclusive": 1, "max_exclusive": 120},
        "internal_design_margin": {"min_inclusive": 20, "max_inclusive": 100},
        "selection_rule": {"current_and_adjacent_gap_rows_inside_internal_margin": True,
                           "prefer_maximum_minimum_distance_to_20_or_100": True, "per_target_and_yaw": True},
        "policies": {"engineering_only": True, "no_model_inference": True, "no_training": True,
                     "never_materialize_into_train_or_val": True, "semantic_labels_evaluator_only": True,
                     "no_dev_calibration_or_test_access": True, "no_robot_manipulation": True},
    }
    return scenes, annotations, gate


def generate() -> None:
    if any(p.exists() for p in (SCENES, ANNOTATIONS, GATE, LOCK)):
        raise FileExistsError("refusing to overwrite v2 characterization inputs")
    for path, payload in zip((SCENES, ANNOTATIONS, GATE), build()):
        path.write_text(base.serialized(payload), encoding="utf-8")
    print(json.dumps({"status": "GENERATED", "scenes": len(build()[0]["scenes"])}, indent=2))


def lock() -> None:
    if LOCK.exists():
        raise FileExistsError("refusing to overwrite v2 lock")
    for path, payload in zip((SCENES, ANNOTATIONS, GATE), build()):
        if path.read_text(encoding="utf-8") != base.serialized(payload):
            raise ValueError(f"edited input: {path}")
    own = {
        "protocol/gazebo_uq_occlusion_characterization_v1.py",
        "protocol/gazebo_uq_occlusion_characterization_v1_capture_lock.json",
        str(base.REPORT.relative_to(ROOT)),
        "protocol/gazebo_uq_occlusion_characterization_v2.py",
        str(SCENES.relative_to(ROOT)), str(ANNOTATIONS.relative_to(ROOT)), str(GATE.relative_to(ROOT)),
    }
    sources = sorted(set(base.REQUIRED_CAPTURE_SOURCES) | own)
    payload = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(), "workspace_root": str(ROOT),
        "purpose": "append-only engineering characterization after immutable v1 rejection",
        "scene_count": len(build()[0]["scenes"]), "model_inventory_sha256": "NO_MODEL_INFERENCE_ENGINEERING_CHARACTERIZATION_V2",
        "source_artifact_sha256": {name: base.sha256(ROOT / name) for name in sources},
        "prohibitions": build()[2]["policies"],
    }
    base.write_json(LOCK, payload)
    print(json.dumps({"status": "LOCKED", "sha256": base.sha256(LOCK)}, indent=2))


def qc() -> None:
    lock_data = json.loads(LOCK.read_text())
    for name, digest in lock_data["source_artifact_sha256"].items():
        if base.sha256(ROOT / name) != digest:
            raise ValueError(f"locked source changed: {name}")
    manifest = json.loads((CAPTURE / "capture_manifest.json").read_text())
    inputs = [json.loads(line) for line in (CAPTURE / "input_manifest.jsonl").read_text().splitlines() if line]
    oracle = yaml.safe_load(ANNOTATIONS.read_text())["scenes"]
    if manifest.get("status") != "COMPLETE" or len(inputs) != len(oracle):
        raise ValueError("v2 characterization capture incomplete")
    rows, groups = [], {}
    for record in inputs:
        sid = record["scene_id"]; item = oracle[sid]
        labels = cv2.imread(str(CAPTURE / sid / "evaluator/semantic_labels.png"), cv2.IMREAD_UNCHANGED)
        if labels.ndim == 3: labels = labels[..., 0]
        pixels = int(np.count_nonzero(labels == int(item["target_label"])))
        row = {"scene_id": sid, **item, "target_visible_pixels": pixels,
               "inside_official_gate": 1 <= pixels < 120, "inside_internal_margin": 20 <= pixels <= 100}
        rows.append(row); groups.setdefault((item["target_id"], item["target_yaw_rad"], item["side"]), []).append(row)
    selected = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda x: x["center_gap_m"]); candidates = []
        for i in range(1, len(group)-1):
            hood = group[i-1:i+2]
            if all(x["inside_internal_margin"] for x in hood):
                margin = min(min(x["target_visible_pixels"]-20, 100-x["target_visible_pixels"]) for x in hood)
                candidates.append((margin, group[i]))
        if candidates:
            margin, row = max(candidates, key=lambda x: x[0]); selected.append({**row, "three_point_safety_margin_pixels": margin})
    required = {(t,y) for t in base.TARGETS for y in base.TARGET_YAWS}
    qualified = {(x["target_id"],x["target_yaw_rad"]) for x in selected}
    status = "PASS" if required <= qualified else "REJECT"
    report = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": status,
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "engineering_only_never_train_or_val": True,
              "model_inference_performed": False, "records": len(rows), "selected_geometry": selected,
              "required_target_yaw_pairs": len(required), "qualified_target_yaw_pairs": len(qualified), "rows": rows,
              "capture_manifest_sha256": base.sha256(CAPTURE / "capture_manifest.json")}
    base.write_json(REPORT, report)
    lines = ["# Controlled occlusion characterization v2", "", f"Decision: **{status}**", "",
             "Append-only engineering sweep; all rows are forbidden from Train-UQ/Val-UQ/evaluation.", "",
             f"Qualified target/yaw pairs: `{len(qualified)}/{len(required)}`; rows: `{len(rows)}`.", ""]
    for x in selected:
        lines.append(f"- `{x['target_id']}` yaw `{x['target_yaw_rad']}`, side `{x['side']}`, gap `{x['center_gap_m']}` m: `{x['target_visible_pixels']}` px, neighborhood margin `{x['three_point_safety_margin_pixels']}` px.")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True); REPORT_MD.write_text("\n".join(lines)+"\n")
    print(json.dumps({"status": status, "qualified": len(qualified), "required": len(required)}, indent=2))


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("command", choices=("generate","lock","qc")); a=p.parse_args()
    {"generate":generate,"lock":lock,"qc":qc}[a.command]()


if __name__ == "__main__": main()
