#!/usr/bin/env python3
"""Transition-only, collision-free neutral-screen characterization for WP5.

Revision v7 is immutable and rejected because fully occluded scenes produced
byte-identical RGB.  Its evaluator-only diagnostic locates the transition
bands used here; no model inference is permitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math

import cv2
import numpy as np
import yaml

import gazebo_uq_occlusion_characterization_v1 as common
from generate_gazebo_train_uq_v1_contract import CAMERA, LABELS, OBJECTS

ROOT = common.ROOT
CONFIG = common.CONFIG
PROTOCOL_ID = "gazebo_uq_occlusion_characterization_v8"
SCENES = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
LOCK = ROOT / "protocol/gazebo_uq_occlusion_characterization_v8_capture_lock.json"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_uq_occlusion_characterization_v8"
CAPTURE = RESULT / "capture_attempt_01"
REPORT = RESULT / "OCCLUSION_CHARACTERIZATION_REPORT.json"
REPORT_MD = RESULT / "OCCLUSION_CHARACTERIZATION_REPORT.md"
V7_LOCK = ROOT / "protocol/gazebo_uq_occlusion_characterization_v7_capture_lock.json"
V7_CAPTURE = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_uq_occlusion_characterization_v7/capture_attempt_01"
TARGET_XY = (-0.220, 0.300)
PANEL_ALIGNED_XY = (-0.2168, 0.2947)
PANEL = "uq_neutral_occluder"
PANEL_LABEL = 39
OBJECTS_V8 = {**OBJECTS, PANEL: {"z": 0.060, "storage_pose": [-0.60, 1.20, 0.0]}}

# Every offset is globally unique.  Fine transition bands come from v7's
# evaluator-only labels; mango receives a wider, disjoint half-mm grid.
SWEEPS = {
    ("ycb_apple", 0.0): [0.03255, 0.03280, 0.03305, 0.03330, 0.03355],
    ("ycb_apple", 0.35): [0.03195, 0.03220, 0.03245, 0.03270, 0.03295],
    ("ycb_orange", 0.0): [0.03565, 0.03590, 0.03615, 0.03640, 0.03665],
    ("ycb_orange", 0.35): [0.03575, 0.03600, 0.03625, 0.03650, 0.03675],
    ("mango", 0.0): [round(0.0400 + 0.001 * i, 5) for i in range(26)],
    ("mango", 0.35): [round(0.0405 + 0.001 * i, 5) for i in range(26)],
}


def build():
    rows, oracle = [], {}
    for i, ((target, yaw), offsets) in enumerate(SWEEPS.items()):
        for offset in offsets:
            scene_index = len(rows)
            sid = f"uq_occ_char_v8_{scene_index:03d}"
            poses = {
                target: [TARGET_XY[0], TARGET_XY[1], yaw],
                PANEL: [PANEL_ALIGNED_XY[0], round(PANEL_ALIGNED_XY[1] - offset, 6), math.pi / 2],
            }
            signature = common.layout_signature(poses)
            rows.append({
                "scene_id": sid,
                "scene_family_id": f"engineering_occlusion_characterization_v8/family_{scene_index:03d}",
                "layout_id": f"neutral_screen_transition_v8_{scene_index:03d}",
                "layout_signature_sha256": signature,
                "task_type": "engineering_geometry_characterization_no_inference",
                "instruction": "ENGINEERING ONLY: do not run model inference on this capture.",
                "poses": poses,
            })
            oracle[sid] = {
                "usage": "evaluator_only_geometry_characterization",
                "target_id": target,
                "target_label": LABELS[target],
                "target_yaw_rad": yaw,
                "occluder_id": PANEL,
                "occluder_label": PANEL_LABEL,
                "lateral_offset_m": offset,
                "sweep_group": i,
                "layout_signature_sha256": signature,
            }
    scenes = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "expected_scene_count": len(rows),
        "capture_mode": "engineering_geometry_only_no_manipulation",
        "random_seed": 14092034,
        "view_joint_pose": CAMERA["view_joint_pose"],
        "camera_frame": CAMERA["frame"],
        "base_frame": CAMERA["base_frame"],
        "coordinate_suffix": "ENGINEERING ONLY; MODEL INFERENCE FORBIDDEN.",
        "objects": OBJECTS_V8,
        "scenes": rows,
    }
    annotations = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "oracle_usage": "geometry_characterization_only_never_train_or_val",
        "parent_attempt": {
            "protocol_id": "gazebo_uq_occlusion_characterization_v7",
            "status": "REJECT",
            "reason": "capture finalizer rejected 126 scenes because only 36 RGB hashes were unique; full-occlusion plateaus were non-informative",
        },
        "occluder": {"id": PANEL, "label": PANEL_LABEL, "static": True, "visual_only_no_collision": True, "candidate": False},
        "sweeps_m": {f"{target}|{yaw}": values for (target, yaw), values in SWEEPS.items()},
        "scenes": oracle,
    }
    gate = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "official_pilot_gate_unchanged": {"min_inclusive": 1, "max_exclusive": 120},
        "internal_design_margin": {"min_inclusive": 20, "max_inclusive": 100},
        "selection_rule": {
            "at_least_three_samples_inside_internal_margin": True,
            "prefer_maximum_distance_to_nearest_internal_bound": True,
            "per_target_and_yaw": True,
        },
        "capture_acceptance": {"all_scene_rgb_sha256_unique": True},
        "policies": {
            "engineering_only": True,
            "no_model_inference": True,
            "no_training": True,
            "never_materialize_into_train_or_val": True,
            "semantic_labels_evaluator_only": True,
            "no_dev_calibration_or_test_access": True,
            "no_robot_manipulation": True,
            "collision_free_visual_only_occluder": True,
        },
    }
    return scenes, annotations, gate


def generate():
    if any(path.exists() for path in (SCENES, ANNOTATIONS, GATE, LOCK)):
        raise FileExistsError("refusing overwrite")
    for path, payload in zip((SCENES, ANNOTATIONS, GATE), build()):
        path.write_text(common.serialized(payload), encoding="utf-8")
    print(json.dumps({"status": "GENERATED", "scenes": len(build()[0]["scenes"])}, indent=2))


def lock():
    if LOCK.exists():
        raise FileExistsError("refusing overwrite lock")
    for path, payload in zip((SCENES, ANNOTATIONS, GATE), build()):
        if path.read_text(encoding="utf-8") != common.serialized(payload):
            raise ValueError(f"edited {path}")
    v7_rows = [json.loads(line) for line in (V7_CAPTURE / "input_manifest.jsonl").read_text().splitlines() if line]
    unique_v7_rgb = len({row["input_sha256"]["rgb"] for row in v7_rows})
    if len(v7_rows) != 126 or unique_v7_rgb != 36:
        raise ValueError("v7 rejection evidence changed")
    sources = sorted(set(common.REQUIRED_CAPTURE_SOURCES) | {
        "protocol/gazebo_uq_occlusion_characterization_v7_capture_lock.json",
        "protocol/gazebo_uq_occlusion_characterization_v8.py",
        str(SCENES.relative_to(ROOT)), str(ANNOTATIONS.relative_to(ROOT)), str(GATE.relative_to(ROOT)),
        "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
        "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
    })
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "transition-only collision-free screen characterization",
        "scene_count": len(build()[0]["scenes"]),
        "world_file": "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
        "model_inventory_sha256": "NO_MODEL_INFERENCE_ENGINEERING_CHARACTERIZATION_V8",
        "parent_v7_evidence": {
            "lock_sha256": common.sha256(V7_LOCK),
            "capture_manifest_sha256": common.sha256(V7_CAPTURE / "capture_manifest.json"),
            "input_manifest_sha256": common.sha256(V7_CAPTURE / "input_manifest.jsonl"),
            "captured_rows": len(v7_rows),
            "unique_rgb_sha256": unique_v7_rgb,
            "capture_status": "REJECT_DUPLICATE_RGB",
        },
        "source_artifact_sha256": {name: common.sha256(ROOT / name) for name in sources},
        "prohibitions": build()[2]["policies"],
    }
    common.write_json(LOCK, payload)
    print(json.dumps({"status": "LOCKED", "sha256": common.sha256(LOCK)}, indent=2))


def qc():
    locked = json.loads(LOCK.read_text())
    for name, digest in locked["source_artifact_sha256"].items():
        if common.sha256(ROOT / name) != digest:
            raise ValueError(f"locked source changed: {name}")
    manifest = json.loads((CAPTURE / "capture_manifest.json").read_text())
    inputs = [json.loads(line) for line in (CAPTURE / "input_manifest.jsonl").read_text().splitlines() if line]
    oracle = yaml.safe_load(ANNOTATIONS.read_text())["scenes"]
    if manifest.get("status") != "COMPLETE" or len(inputs) != len(oracle):
        raise ValueError("capture incomplete")
    rows, groups = [], {}
    for record in inputs:
        sid = record["scene_id"]
        item = oracle[sid]
        labels = cv2.imread(str(CAPTURE / sid / "evaluator/semantic_labels.png"), cv2.IMREAD_UNCHANGED)
        labels = labels[..., 0] if labels.ndim == 3 else labels
        pixels = int(np.count_nonzero(labels == int(item["target_label"])))
        row = {**item, "scene_id": sid, "target_visible_pixels": pixels,
               "inside_official_gate": 1 <= pixels < 120, "inside_internal_margin": 20 <= pixels <= 100}
        rows.append(row)
        groups.setdefault((item["target_id"], item["target_yaw_rad"]), []).append(row)
    selected = []
    for (target, yaw), group in sorted(groups.items()):
        candidates = [row for row in group if row["inside_internal_margin"]]
        if len(candidates) >= 3:
            choice = max(candidates, key=lambda row: min(row["target_visible_pixels"] - 20, 100 - row["target_visible_pixels"]))
            selected.append({**choice, "qualified_internal_samples": len(candidates),
                             "safety_margin_pixels": min(choice["target_visible_pixels"] - 20, 100 - choice["target_visible_pixels"])})
    required = {(target, yaw) for target in common.TARGETS for yaw in common.TARGET_YAWS}
    qualified = {(row["target_id"], row["target_yaw_rad"]) for row in selected}
    status = "PASS" if required <= qualified else "REJECT"
    report = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": status,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "engineering_only_never_train_or_val": True, "model_inference_performed": False,
        "records": len(rows), "unique_rgb_sha256": len({r["input_sha256"]["rgb"] for r in inputs}),
        "selected_geometry": selected, "required_target_yaw_pairs": len(required),
        "qualified_target_yaw_pairs": len(qualified), "rows": rows,
        "capture_manifest_sha256": common.sha256(CAPTURE / "capture_manifest.json"),
    }
    common.write_json(REPORT, report)
    lines = ["# Collision-free neutral-screen characterization v8", "", f"Decision: **{status}**", "",
             "Engineering-only; never eligible for Train-UQ/Val-UQ/evaluation.", "",
             f"Qualified: `{len(qualified)}/{len(required)}`; unique RGB: `{report['unique_rgb_sha256']}/{len(rows)}`.", ""]
    for row in selected:
        lines.append(f"- `{row['target_id']}` yaw `{row['target_yaw_rad']}`, offset `{row['lateral_offset_m']}` m: `{row['target_visible_pixels']}` px; margin `{row['safety_margin_pixels']}` px.")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "qualified": len(qualified), "required": len(required)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "lock", "qc"))
    command = parser.parse_args().command
    {"generate": generate, "lock": lock, "qc": qc}[command]()


if __name__ == "__main__":
    main()
