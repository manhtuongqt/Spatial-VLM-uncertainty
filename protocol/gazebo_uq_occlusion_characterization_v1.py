#!/usr/bin/env python3
"""Engineering-only controlled-occlusion sweep for WP5 geometry.

This artifact is intentionally outside every learning/evaluation split.  It
captures RGB-D plus evaluator-only semantic labels, never invokes B0, and is
used only to select robust target/occluder geometry before pilot r4 is locked.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from generate_gazebo_train_uq_v1_contract import CAMERA, LABELS, OBJECTS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ur3/ur3_perception/config"
PROTOCOL_ID = "gazebo_uq_occlusion_characterization_v1"
SCENES = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
LOCK = ROOT / "protocol/gazebo_uq_occlusion_characterization_v1_capture_lock.json"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_uq_occlusion_characterization_v1"
CAPTURE = RESULT / "capture_attempt_01"
REPORT = RESULT / "OCCLUSION_CHARACTERIZATION_REPORT.json"
REPORT_MD = RESULT / "OCCLUSION_CHARACTERIZATION_REPORT.md"

TARGETS = ("ycb_apple", "ycb_orange", "mango")
TARGET_YAWS = (0.0, 0.35)
LONGITUDINAL_M = (0.080, 0.095)
LATERAL_M = tuple(round(-0.040 + 0.008 * i, 3) for i in range(11))
OCCLUDER = "blue_cube"
OCCLUDER_LABEL = 2
TARGET_XY = (-0.300, 0.300)
CAMERA_XY = (-0.1681077769, 0.2133762724)

REQUIRED_CAPTURE_SOURCES = (
    "RoboRefer/API/api.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_capture.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_runner.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_validate.py",
    "ur3/ur3_perception/scripts/roborefer_pilot_evaluator.py",
    "ur3/ur3_perception/scripts/roborefer_grounder.py",
    "ur3/ur3_perception/scripts/spatial_point_utils.py",
    "ur3/ur3_perception/scripts/move_camera_to_view.py",
    "ur3/ur3_perception/launch/roborefer_pilot_capture.launch.py",
    "ur3/ur3_perception/config/roborefer_pilot_v0_scenes.yaml",
    "ur3/ur3_perception/config/roborefer_pilot_v0_gate.yaml",
    "ur3/ur3_perception/config/roborefer_pilot_v0_annotations.yaml",
    "ur3/ur3_perception/schemas/roborefer_pilot_prediction.schema.json",
    "ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro",
    "ur3/ur3_moveit_control/launch/ur3_susgrip_sim.launch.py",
    "ur3/ur_simulation_gz/worlds/ur3_pick_place.sdf",
    "ur3/ur_simulation_gz/launch/ur_sim_control.launch.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialized(payload: dict) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def layout_signature(poses: dict) -> str:
    canonical = {key: [round(float(v), 9) for v in value] for key, value in sorted(poses.items())}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build() -> tuple[dict, dict, dict]:
    dx = CAMERA_XY[0] - TARGET_XY[0]
    dy = CAMERA_XY[1] - TARGET_XY[1]
    norm = math.hypot(dx, dy)
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    rows, oracle = [], {}
    index = 0
    for target in TARGETS:
        for target_yaw in TARGET_YAWS:
            for longitudinal in LONGITUDINAL_M:
                for lateral in LATERAL_M:
                    sid = f"uq_occ_char_v1_{index:03d}"
                    ox = TARGET_XY[0] + longitudinal * ux + lateral * px
                    oy = TARGET_XY[1] + longitudinal * uy + lateral * py
                    poses = {
                        target: [TARGET_XY[0], TARGET_XY[1], target_yaw],
                        OCCLUDER: [round(ox, 6), round(oy, 6), math.pi / 4.0],
                    }
                    signature = layout_signature(poses)
                    rows.append({
                        "scene_id": sid,
                        "scene_family_id": f"engineering_occlusion_characterization_v1/family_{index:03d}",
                        "layout_id": f"occ_sweep_{index:03d}",
                        "layout_signature_sha256": signature,
                        "task_type": "engineering_geometry_characterization_no_inference",
                        "instruction": "ENGINEERING ONLY: do not run model inference on this capture.",
                        "poses": poses,
                    })
                    oracle[sid] = {
                        "usage": "evaluator_only_geometry_characterization",
                        "target_id": target,
                        "target_label": LABELS[target],
                        "target_yaw_rad": target_yaw,
                        "occluder_id": OCCLUDER,
                        "occluder_label": OCCLUDER_LABEL,
                        "longitudinal_offset_m": longitudinal,
                        "lateral_offset_m": lateral,
                        "layout_signature_sha256": signature,
                    }
                    index += 1
    scenes = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "expected_scene_count": len(rows),
        "capture_mode": "engineering_geometry_only_no_manipulation",
        "random_seed": 14092026,
        "view_joint_pose": CAMERA["view_joint_pose"],
        "camera_frame": CAMERA["frame"],
        "base_frame": CAMERA["base_frame"],
        "coordinate_suffix": "ENGINEERING ONLY; MODEL INFERENCE FORBIDDEN.",
        "objects": OBJECTS,
        "scenes": rows,
    }
    annotations = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "oracle_usage": "geometry_characterization_only_never_train_or_val",
        "occluder": {"id": OCCLUDER, "semantic_label": OCCLUDER_LABEL, "candidate": False},
        "scenes": oracle,
    }
    gate = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "official_pilot_gate_unchanged": {"min_inclusive": 1, "max_exclusive": 120},
        "internal_design_margin": {"min_inclusive": 20, "max_inclusive": 100},
        "selection_rule": {
            "eligible_if_current_and_adjacent_lateral_offsets_inside_internal_margin": True,
            "prefer_maximum_minimum_distance_to_20_or_100": True,
            "per_target_and_yaw": True,
        },
        "policies": {
            "engineering_only": True,
            "no_model_inference": True,
            "no_training": True,
            "never_materialize_into_train_or_val": True,
            "semantic_labels_evaluator_only": True,
            "no_dev_calibration_or_test_access": True,
            "no_robot_manipulation": True,
        },
    }
    return scenes, annotations, gate


def generate() -> None:
    if any(path.exists() for path in (SCENES, ANNOTATIONS, GATE, LOCK)):
        raise FileExistsError("refusing to overwrite characterization inputs/lock")
    for path, payload in zip((SCENES, ANNOTATIONS, GATE), build()):
        path.write_text(serialized(payload), encoding="utf-8")
    print(json.dumps({"status": "GENERATED", "scenes": len(build()[0]["scenes"])}, indent=2))


def lock() -> None:
    if LOCK.exists():
        raise FileExistsError("refusing to overwrite characterization lock")
    expected = build()
    for path, payload in zip((SCENES, ANNOTATIONS, GATE), expected):
        if path.read_text(encoding="utf-8") != serialized(payload):
            raise ValueError(f"non-deterministic or edited input: {path}")
    own = (
        "protocol/gazebo_uq_occlusion_characterization_v1.py",
        str(SCENES.relative_to(ROOT)), str(ANNOTATIONS.relative_to(ROOT)), str(GATE.relative_to(ROOT)),
        "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/grounding_model_lock.json",
        "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/hypothesis_lock.json",
        "protocol/gazebo_train_uq_v2_pilot_r2_contract_lock.json",
        "protocol/gazebo_train_uq_v2_pilot_r3_contract_lock.json",
        "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r2/GAZEBO_TRAIN_UQ_V2_PILOT_R2_GEOMETRY_QC.json",
        "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r3/GAZEBO_TRAIN_UQ_V2_PILOT_R3_GEOMETRY_QC.json",
    )
    sources = sorted(set(REQUIRED_CAPTURE_SOURCES) | set(own))
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "purpose": "engineering-only controlled partial-occlusion characterization",
        "scene_count": len(expected[0]["scenes"]),
        "model_inventory_sha256": "NO_MODEL_INFERENCE_ENGINEERING_CHARACTERIZATION",
        "source_artifact_sha256": {name: sha256(ROOT / name) for name in sources},
        "prohibitions": expected[2]["policies"],
    }
    write_json(LOCK, payload)
    print(json.dumps({"status": "LOCKED", "sha256": sha256(LOCK)}, indent=2))


def verify_capture() -> tuple[dict, dict, list[dict]]:
    lock_data = json.loads(LOCK.read_text(encoding="utf-8"))
    for name, digest in lock_data["source_artifact_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"locked source changed: {name}")
    manifest = json.loads((CAPTURE / "capture_manifest.json").read_text(encoding="utf-8"))
    inputs = [json.loads(line) for line in (CAPTURE / "input_manifest.jsonl").read_text().splitlines() if line]
    annotations = yaml.safe_load(ANNOTATIONS.read_text(encoding="utf-8"))["scenes"]
    if manifest.get("status") != "COMPLETE" or len(inputs) != len(annotations):
        raise ValueError("characterization capture incomplete")
    if manifest.get("model_inventory_sha256_preregistered") != "NO_MODEL_INFERENCE_ENGINEERING_CHARACTERIZATION":
        raise ValueError("model-inference prohibition provenance mismatch")
    return manifest, annotations, inputs


def qc() -> None:
    manifest, annotations, inputs = verify_capture()
    rows = []
    by_key: dict[tuple, list[dict]] = {}
    for record in inputs:
        sid = record["scene_id"]
        item = annotations[sid]
        label_path = CAPTURE / sid / "evaluator/semantic_labels.png"
        labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if labels is None:
            raise ValueError(f"cannot decode {label_path}")
        if labels.ndim == 3:
            labels = labels[..., 0]
        pixels = int(np.count_nonzero(labels == int(item["target_label"])))
        occluder_pixels = int(np.count_nonzero(labels == int(item["occluder_label"])))
        row = {"scene_id": sid, **item, "target_visible_pixels": pixels, "occluder_visible_pixels": occluder_pixels,
               "inside_official_gate": 1 <= pixels < 120, "inside_internal_margin": 20 <= pixels <= 100}
        rows.append(row)
        by_key.setdefault((item["target_id"], float(item["target_yaw_rad"]), float(item["longitudinal_offset_m"])), []).append(row)
    selected = []
    for key, group in sorted(by_key.items()):
        group.sort(key=lambda row: row["lateral_offset_m"])
        candidates = []
        for i in range(1, len(group) - 1):
            neighborhood = group[i - 1:i + 2]
            if all(row["inside_internal_margin"] for row in neighborhood):
                margin = min(min(row["target_visible_pixels"] - 20, 100 - row["target_visible_pixels"]) for row in neighborhood)
                candidates.append((margin, group[i]))
        if candidates:
            best = max(candidates, key=lambda pair: (pair[0], -abs(pair[1]["lateral_offset_m"])))
            selected.append({**best[1], "three_point_safety_margin_pixels": int(best[0])})
    complete_keys = {(row["target_id"], row["target_yaw_rad"]) for row in selected}
    required_keys = {(target, yaw) for target in TARGETS for yaw in TARGET_YAWS}
    status = "PASS" if required_keys <= complete_keys else "REJECT"
    report = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": status,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "engineering_only_never_train_or_val": True, "model_inference_performed": False,
        "capture_manifest_sha256": sha256(CAPTURE / "capture_manifest.json"),
        "records": len(rows), "selected_geometry": selected,
        "required_target_yaw_pairs": len(required_keys), "qualified_target_yaw_pairs": len(complete_keys),
        "rows": rows,
    }
    write_json(REPORT, report)
    lines = ["# Controlled occlusion characterization v1", "", f"Decision: **{status}**", "",
             "Engineering-only sweep; these rows are forbidden from Train-UQ, Val-UQ and all evaluation splits.", "",
             f"Qualified target/yaw pairs: `{len(complete_keys)}/{len(required_keys)}`; captured rows: `{len(rows)}`.", "", "## Selected geometry", ""]
    for row in selected:
        lines.append(f"- `{row['target_id']}` yaw `{row['target_yaw_rad']}`: longitudinal `{row['longitudinal_offset_m']}` m, lateral `{row['lateral_offset_m']}` m, `{row['target_visible_pixels']}` px, 3-point margin `{row['three_point_safety_margin_pixels']}` px.")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "qualified": len(complete_keys), "required": len(required_keys)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "lock", "qc"))
    args = parser.parse_args()
    {"generate": generate, "lock": lock, "qc": qc}[args.command]()


if __name__ == "__main__":
    main()
