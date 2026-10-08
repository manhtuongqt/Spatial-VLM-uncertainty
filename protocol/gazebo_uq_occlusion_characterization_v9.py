#!/usr/bin/env python3
"""Immutable mango transition refinement completing WP5 characterization."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math

import cv2
import numpy as np
import yaml

import gazebo_uq_occlusion_characterization_v1 as common
import gazebo_uq_occlusion_characterization_v8 as parent
from generate_gazebo_train_uq_v1_contract import CAMERA, LABELS, OBJECTS

ROOT = common.ROOT
CONFIG = common.CONFIG
PROTOCOL_ID = "gazebo_uq_occlusion_characterization_v9"
SCENES = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
LOCK = ROOT / "protocol/gazebo_uq_occlusion_characterization_v9_capture_lock.json"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_uq_occlusion_characterization_v9"
CAPTURE = RESULT / "capture_attempt_01"
REPORT = RESULT / "OCCLUSION_CHARACTERIZATION_REPORT.json"
REPORT_MD = RESULT / "OCCLUSION_CHARACTERIZATION_REPORT.md"
PANEL = "uq_neutral_occluder"
PANEL_LABEL = 39
OBJECTS_V9 = {**OBJECTS, PANEL: {"z": 0.060, "storage_pose": [-0.60, 1.20, 0.0]}}
SWEEPS = {
    0.0: [0.04265, 0.04285, 0.04305, 0.04325, 0.04345, 0.04365, 0.04385],
    0.35: [0.04270, 0.04290, 0.04310, 0.04330, 0.04350, 0.04370, 0.04390],
}


def build():
    rows, oracle = [], {}
    for yaw, offsets in SWEEPS.items():
        for offset in offsets:
            index = len(rows)
            sid = f"uq_occ_char_v9_{index:03d}"
            poses = {
                "mango": [-0.220, 0.300, yaw],
                PANEL: [-0.2168, round(0.2947 - offset, 6), math.pi / 2],
            }
            signature = common.layout_signature(poses)
            rows.append({
                "scene_id": sid,
                "scene_family_id": f"engineering_occlusion_characterization_v9/family_{index:03d}",
                "layout_id": f"neutral_screen_mango_v9_{index:03d}",
                "layout_signature_sha256": signature,
                "task_type": "engineering_geometry_characterization_no_inference",
                "instruction": "ENGINEERING ONLY: do not run model inference on this capture.",
                "poses": poses,
            })
            oracle[sid] = {
                "usage": "evaluator_only_geometry_characterization",
                "target_id": "mango", "target_label": LABELS["mango"], "target_yaw_rad": yaw,
                "occluder_id": PANEL, "occluder_label": PANEL_LABEL,
                "lateral_offset_m": offset, "layout_signature_sha256": signature,
            }
    scenes = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "expected_scene_count": len(rows),
        "capture_mode": "engineering_geometry_only_no_manipulation", "random_seed": 14092035,
        "view_joint_pose": CAMERA["view_joint_pose"], "camera_frame": CAMERA["frame"],
        "base_frame": CAMERA["base_frame"], "coordinate_suffix": "ENGINEERING ONLY; MODEL INFERENCE FORBIDDEN.",
        "objects": OBJECTS_V9, "scenes": rows,
    }
    annotations = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "oracle_usage": "geometry_characterization_only_never_train_or_val",
        "parent_v8_status": "REJECT_4_OF_6_QUALIFIED",
        "scope": "mango-only transition refinement; apple/orange geometry frozen from v8",
        "occluder": {"id": PANEL, "label": PANEL_LABEL, "static": True, "visual_only_no_collision": True, "candidate": False},
        "sweeps_m": {str(yaw): values for yaw, values in SWEEPS.items()}, "scenes": oracle,
    }
    gate = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "official_pilot_gate_unchanged": {"min_inclusive": 1, "max_exclusive": 120},
        "internal_design_margin": {"min_inclusive": 20, "max_inclusive": 100},
        "selection_rule": {"at_least_three_samples_inside_internal_margin": True,
                           "prefer_maximum_distance_to_nearest_internal_bound": True,
                           "per_mango_yaw": True},
        "capture_acceptance": {"all_scene_rgb_sha256_unique": True},
        "policies": parent.build()[2]["policies"],
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
    v8 = json.loads(parent.REPORT.read_text())
    if v8.get("status") != "REJECT" or v8.get("qualified_target_yaw_pairs") != 4:
        raise ValueError("v8 parent evidence mismatch")
    sources = sorted(set(common.REQUIRED_CAPTURE_SOURCES) | {
        "protocol/gazebo_uq_occlusion_characterization_v8.py",
        "protocol/gazebo_uq_occlusion_characterization_v8_capture_lock.json",
        "results/spatial_vlm_refspatial_v1/gazebo_uq_occlusion_characterization_v8/OCCLUSION_CHARACTERIZATION_REPORT.json",
        "protocol/gazebo_uq_occlusion_characterization_v9.py",
        str(SCENES.relative_to(ROOT)), str(ANNOTATIONS.relative_to(ROOT)), str(GATE.relative_to(ROOT)),
        "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py",
        "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
    })
    payload = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE", "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "mango transition refinement completing six target-yaw geometries",
        "scene_count": len(build()[0]["scenes"]),
        "model_inventory_sha256": "NO_MODEL_INFERENCE_ENGINEERING_CHARACTERIZATION_V9",
        "parent_v8_report_sha256": common.sha256(parent.REPORT),
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
        rows.append(row); groups.setdefault(item["target_yaw_rad"], []).append(row)
    selected = []
    for yaw, group in sorted(groups.items()):
        candidates = [row for row in group if row["inside_internal_margin"]]
        if len(candidates) >= 3:
            choice = max(candidates, key=lambda row: min(row["target_visible_pixels"] - 20, 100 - row["target_visible_pixels"]))
            selected.append({**choice, "qualified_internal_samples": len(candidates),
                             "safety_margin_pixels": min(choice["target_visible_pixels"] - 20, 100 - choice["target_visible_pixels"])})
    status = "PASS" if len(selected) == 2 else "REJECT"
    v8 = json.loads(parent.REPORT.read_text())
    report = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": status,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "engineering_only_never_train_or_val": True,
        "model_inference_performed": False, "records": len(rows),
        "unique_rgb_sha256": len({r["input_sha256"]["rgb"] for r in inputs}),
        "selected_geometry": selected, "required_mango_yaw_pairs": 2, "qualified_mango_yaw_pairs": len(selected),
        "combined_v8_v9_required_target_yaw_pairs": 6,
        "combined_v8_v9_qualified_target_yaw_pairs": v8["qualified_target_yaw_pairs"] + len(selected),
        "combined_characterization_status": "PASS" if status == "PASS" else "REJECT",
        "parent_v8_report_sha256": common.sha256(parent.REPORT), "rows": rows,
        "capture_manifest_sha256": common.sha256(CAPTURE / "capture_manifest.json"),
    }
    common.write_json(REPORT, report)
    lines = ["# Collision-free neutral-screen characterization v9", "", f"Decision: **{status}**", "",
             f"Combined v8+v9: **{report['combined_characterization_status']}** (`{report['combined_v8_v9_qualified_target_yaw_pairs']}/6`).", "",
             "Engineering-only; never eligible for Train-UQ/Val-UQ/evaluation.", ""]
    for row in selected:
        lines.append(f"- `mango` yaw `{row['target_yaw_rad']}`, offset `{row['lateral_offset_m']}` m: `{row['target_visible_pixels']}` px; margin `{row['safety_margin_pixels']}` px.")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "combined": report["combined_characterization_status"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("generate", "lock", "qc"))
    {"generate": generate, "lock": lock, "qc": qc}[parser.parse_args().command]()


if __name__ == "__main__":
    main()
