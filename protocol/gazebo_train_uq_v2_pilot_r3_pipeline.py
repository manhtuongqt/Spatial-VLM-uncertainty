#!/usr/bin/env python3
"""Lock, preflight and geometry-QC Gazebo Train-UQ v2 pilot revision r3.

This pipeline is geometry-only.  It must never run model inference, materialize
a training dataset, train a model, command manipulation, or access final Test.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_train_uq_v2_pilot_r3"
CONFIG = ROOT / "ur3/ur3_perception/config"
SCENES = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
LOCK = ROOT / "protocol/gazebo_train_uq_v2_pilot_r3_contract_lock.json"
CAPTURE_LOCK = (
    ROOT / "protocol/gazebo_train_uq_v2_pilot_r3_capture_compatibility_lock.json"
)
RESULT = (
    ROOT
    / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r3"
)
CAPTURE = RESULT / "capture_attempt_01"
STATIC_PREFLIGHT = RESULT / "PREFLIGHT_STATIC.json"
LIVE_PREFLIGHT = RESULT / "PREFLIGHT_LIVE.json"
QC_REPORT = RESULT / "GAZEBO_TRAIN_UQ_V2_PILOT_R3_GEOMETRY_QC.json"
DECISION = RESULT / "CAPTURE_ATTEMPT_01_DECISION.md"

GROUNDING_LOCK = (
    ROOT
    / "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/grounding_model_lock.json"
)
HYPOTHESIS_LOCK = (
    ROOT
    / "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/hypothesis_lock.json"
)
AMENDMENT = ROOT / "protocol/gazebo_train_uq_v2_pilot_r3_amendment.json"
R2_CONTRACT_LOCK = ROOT / "protocol/gazebo_train_uq_v2_pilot_r2_contract_lock.json"
R2_CAPTURE_LOCK = (
    ROOT / "protocol/gazebo_train_uq_v2_pilot_r2_capture_compatibility_lock.json"
)
R2_QC = (
    ROOT
    / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r2/GAZEBO_TRAIN_UQ_V2_PILOT_R2_GEOMETRY_QC.json"
)
R2_DECISION = (
    ROOT
    / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r2/CAPTURE_ATTEMPT_01_DECISION.md"
)
V1_QC = (
    ROOT
    / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v1/GAZEBO_TRAIN_UQ_V1_GEOMETRY_QC.json"
)
V2_DECISION = (
    ROOT
    / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot/CAPTURE_ATTEMPT_01_DECISION.md"
)
DEV_MANIFESTS = (
    ROOT / "datasets/Gazebo_dev/evaluator_ground_truth.jsonl",
    ROOT / "datasets/Gazebo_dev_answerability_v2/evaluator_ground_truth.jsonl",
)
PRIOR_SCENE_CONFIGS = (
    CONFIG / "gazebo_train_uq_v1_scenes.yaml",
    CONFIG / "gazebo_train_uq_v2_pilot_scenes.yaml",
    CONFIG / "gazebo_train_uq_v2_pilot_r2_scenes.yaml",
)
PRIOR_ANNOTATION_CONFIGS = (
    CONFIG / "gazebo_train_uq_v1_annotations.yaml",
    CONFIG / "gazebo_train_uq_v2_pilot_annotations.yaml",
    CONFIG / "gazebo_train_uq_v2_pilot_r2_annotations.yaml",
)
STATES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")
RELATIONS = ("leftmost", "rightmost", "second_from_left", "second_from_right")

# The generic capture node requires these entries in its pretrial lock.
LEGACY_CAPTURE_SOURCES = (
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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def family_ids_from_annotations(path: Path) -> set[str]:
    data = load_yaml(path)
    return {
        str(item["family_id"])
        for item in data.get("scenes", {}).values()
        if item.get("family_id")
    }


def verify_amendment_and_r2_evidence() -> dict:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if amendment.get("status") != "LOCKED_BEFORE_R3_DESIGN_AND_CAPTURE":
        raise ValueError("r3 amendment is not preregistered")
    if sha256(HYPOTHESIS_LOCK) != "077f65f3db1b9562bc589ba54ac47d19c5a6ee1c1a00d3a178228b5822e8c78a":
        raise ValueError("the original hypothesis lock changed")
    expected = amendment["r2_frozen_negative_evidence"]["artifact_sha256"]
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"frozen r2 evidence changed: {relative}")
    r2_qc = json.loads(R2_QC.read_text(encoding="utf-8"))
    failed = {
        row["scene_id"]: row["target_visible_pixels"]
        for row in r2_qc.get("scenes", [])
        if not row.get("state_verified")
    }
    if (
        r2_qc.get("status") != "REJECT"
        or r2_qc.get("records") != 32
        or r2_qc.get("failed_scene_count") != 2
        or failed
        != {
            "gazebo_uq_v2_pilot_r2_012": 0,
            "gazebo_uq_v2_pilot_r2_014": 154,
        }
        or "Decision: **REJECT**" not in R2_DECISION.read_text(encoding="utf-8")
    ):
        raise ValueError("r2 rejection evidence does not match the amendment")
    return amendment


def validate_contract() -> tuple[dict, dict, dict]:
    from generate_gazebo_train_uq_v2_pilot_r3_contract import (
        build,
        layout_signature,
        serialized,
    )

    generated = build()
    for path, payload in zip((SCENES, ANNOTATIONS, GATE), generated):
        if not path.is_file() or path.read_text(encoding="utf-8") != serialized(
            payload
        ):
            raise ValueError(f"non-deterministic r3 input: {path}")

    scenes, annotations, gate = map(
        load_yaml, (SCENES, ANNOTATIONS, GATE)
    )
    if any(
        item.get("protocol_id") != PROTOCOL_ID
        for item in (scenes, annotations, gate)
    ):
        raise ValueError("protocol ID mismatch")
    rows = scenes.get("scenes", [])
    oracle = annotations.get("scenes", {})
    if len(rows) != 32 or len(oracle) != 32:
        raise ValueError("r3 pilot must contain exactly 32 rows")

    scene_ids = [row["scene_id"] for row in rows]
    families = [row["scene_family_id"] for row in rows]
    layout_ids = [row["layout_id"] for row in rows]
    layout_signatures = [row["layout_signature_sha256"] for row in rows]
    if any(
        len(set(values)) != 32
        for values in (scene_ids, families, layout_ids, layout_signatures)
    ):
        raise ValueError(
            "scene, family, layout ID and full pose signature must each be unique"
        )
    if set(scene_ids) != set(oracle):
        raise ValueError("scene/annotation alignment failed")
    for row in rows:
        sid = row["scene_id"]
        item = oracle[sid]
        actual_signature = layout_signature(row["poses"])
        if row["layout_signature_sha256"] != actual_signature:
            raise ValueError(f"layout signature mismatch: {sid}")
        if item.get("layout_signature_sha256") != actual_signature:
            raise ValueError(f"oracle layout signature mismatch: {sid}")
        if item.get("family_id") != row["scene_family_id"]:
            raise ValueError(f"family mismatch: {sid}")

    state_counts = Counter(item["state"] for item in oracle.values())
    relation_counts = Counter(
        item["relation_variant"] for item in oracle.values()
    )
    cell_counts = Counter(
        (item["state"], item["relation_variant"])
        for item in oracle.values()
    )
    if state_counts != Counter({state: 8 for state in STATES}):
        raise ValueError(f"state quota mismatch: {state_counts}")
    if relation_counts != Counter({relation: 8 for relation in RELATIONS}):
        raise ValueError(f"relation quota mismatch: {relation_counts}")
    if set(cell_counts.values()) != {2} or len(cell_counts) != 16:
        raise ValueError(f"state/relation cell quota mismatch: {cell_counts}")

    prior_families: set[str] = set()
    prior_scene_ids: set[str] = set()
    for path in DEV_MANIFESTS:
        prior_rows = load_jsonl(path)
        prior_families.update(str(item["family_id"]) for item in prior_rows)
        prior_scene_ids.update(str(item["scene_id"]) for item in prior_rows)
    for path in PRIOR_ANNOTATION_CONFIGS:
        prior_families.update(family_ids_from_annotations(path))
    prior_signatures: set[str] = set()
    for path in PRIOR_SCENE_CONFIGS:
        prior_config = load_yaml(path)
        for row in prior_config.get("scenes", []):
            prior_scene_ids.add(str(row["scene_id"]))
            prior_signatures.add(layout_signature(row["poses"]))
    if set(families) & prior_families:
        raise ValueError("r3 family overlap with Dev or prior UQ attempt")
    if set(scene_ids) & prior_scene_ids:
        raise ValueError("r3 scene ID overlap with Dev or prior UQ attempt")
    if set(layout_signatures) & prior_signatures:
        raise ValueError("r3 full pose signature reuses a prior UQ layout")

    if json.loads(V1_QC.read_text(encoding="utf-8")).get("status") != "BLOCKED":
        raise ValueError("frozen v1 failed-QC evidence is missing")
    if "INVALID_FOR_GEOMETRY_QC" not in V2_DECISION.read_text(encoding="utf-8"):
        raise ValueError("frozen v2 invalid-attempt evidence is missing")
    amendment = verify_amendment_and_r2_evidence()

    grounding_lock = json.loads(GROUNDING_LOCK.read_text(encoding="utf-8"))
    hypothesis_lock = json.loads(HYPOTHESIS_LOCK.read_text(encoding="utf-8"))
    if grounding_lock.get("status") != "LOCKED_BEFORE_WP5_UQ_DEVELOPMENT":
        raise ValueError("WP4 grounding model is not locked")
    if hypothesis_lock.get("status") != "LOCKED_AFTER_DEV_FAILURE_ANALYSIS_BEFORE_UQ_V2_R2":
        raise ValueError("WP5 hypothesis is not locked")
    if hypothesis_lock.get("grounding_model_lock", {}).get("sha256") != sha256(
        GROUNDING_LOCK
    ):
        raise ValueError("hypothesis/grounding lock linkage failed")

    required_policies = {
        "no_sam2": True,
        "no_training": True,
        "no_model_inference": True,
        "no_materialization": True,
        "no_b2": True,
        "no_test_access": True,
        "no_dev_v2_fit_or_selection": True,
        "no_robot_manipulation": True,
        "full_train_uq_only_after_pilot_pass": True,
    }
    if gate.get("policies") != required_policies:
        raise ValueError("r3 geometry-only safety policy mismatch")
    if gate.get("insufficient_evidence_rule") != {
        "min_visible_pixels_inclusive": 1,
        "max_visible_pixels_exclusive": 120,
        "projected_center_outer_margin_fraction": 0.20,
    }:
        raise ValueError("r3 changed the official insufficient-evidence gate")
    if amendment["r3_design_contract"].get("capture_attempts") != 1:
        raise ValueError("r3 amendment does not freeze one capture attempt")
    uniqueness = gate.get("layout_uniqueness", {})
    if not all(
        uniqueness.get(key) is True
        for key in (
            "full_pose_signatures_unique",
            "no_pose_signature_overlap_with_prior_uq_attempts",
            "byte_identical_rgb_forbidden",
        )
    ):
        raise ValueError("layout uniqueness policy is incomplete")
    return scenes, annotations, gate


def lock_contract() -> None:
    if LOCK.exists() or CAPTURE_LOCK.exists():
        raise FileExistsError("refusing to overwrite r3 pilot locks")
    scenes, annotations, gate = validate_contract()
    own_sources = (
        Path("protocol/generate_gazebo_train_uq_v2_pilot_r3_contract.py"),
        Path("protocol/gazebo_train_uq_v2_pilot_r3_pipeline.py"),
        Path("ur3/ur3_perception/config/gazebo_train_uq_v2_pilot_r3_scenes.yaml"),
        Path("ur3/ur3_perception/config/gazebo_train_uq_v2_pilot_r3_annotations.yaml"),
        Path("ur3/ur3_perception/config/gazebo_train_uq_v2_pilot_r3_gate.yaml"),
        AMENDMENT.relative_to(ROOT),
        GROUNDING_LOCK.relative_to(ROOT),
        HYPOTHESIS_LOCK.relative_to(ROOT),
        V1_QC.relative_to(ROOT),
        V2_DECISION.relative_to(ROOT),
        R2_CONTRACT_LOCK.relative_to(ROOT),
        R2_CAPTURE_LOCK.relative_to(ROOT),
        R2_QC.relative_to(ROOT),
        R2_DECISION.relative_to(ROOT),
    )
    source_hashes = {str(path): sha256(ROOT / path) for path in own_sources}
    rows = scenes["scenes"]
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_PILOT_CAPTURE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "purpose": "geometry validation only; no materialization, training, model inference, manipulation, calibration, or Test access",
        "parent_families": 32,
        "state_relation_cell_quota": 2,
        "layout_signature_sha256": {
            row["scene_id"]: row["layout_signature_sha256"] for row in rows
        },
        "layout_signature_set_sha256": hashlib.sha256(
            json.dumps(
                sorted(row["layout_signature_sha256"] for row in rows)
            ).encode("utf-8")
        ).hexdigest(),
        "source_artifact_sha256": source_hashes,
        "grounding_model_lock_sha256": sha256(GROUNDING_LOCK),
        "hypothesis_lock_sha256": sha256(HYPOTHESIS_LOCK),
        "r3_amendment_sha256": sha256(AMENDMENT),
        "frozen_dev_manifest_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in DEV_MANIFESTS
        },
        "prior_failed_evidence_sha256": {
            str(V1_QC.relative_to(ROOT)): sha256(V1_QC),
            str(V2_DECISION.relative_to(ROOT)): sha256(V2_DECISION),
            str(R2_CONTRACT_LOCK.relative_to(ROOT)): sha256(R2_CONTRACT_LOCK),
            str(R2_CAPTURE_LOCK.relative_to(ROOT)): sha256(R2_CAPTURE_LOCK),
            str(R2_QC.relative_to(ROOT)): sha256(R2_QC),
            str(R2_DECISION.relative_to(ROOT)): sha256(R2_DECISION),
        },
        "geometry_gate": {
            "tie_margin_normalized": gate["tie_margin_normalized"],
            "min_visible_evidence_px": gate["min_visible_evidence_px"],
            "insufficient_evidence_rule": gate["insufficient_evidence_rule"],
            "layout_uniqueness": gate["layout_uniqueness"],
        },
        "policies": gate["policies"],
    }
    write_json(LOCK, payload)

    capture_sources = set(LEGACY_CAPTURE_SOURCES)
    capture_sources.update(str(path) for path in own_sources)
    compatibility = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "parent_contract_lock": str(LOCK.relative_to(ROOT)),
        "parent_contract_lock_sha256": sha256(LOCK),
        "purpose": "generic ROS capture compatibility adapter; geometry pilot only",
        "source_artifact_sha256": {
            relative: sha256(ROOT / relative)
            for relative in sorted(capture_sources)
        },
        "model_inventory_sha256": "NO_MODEL_INFERENCE_GEOMETRY_PILOT_R3",
        "policies_inherited": gate["policies"],
    }
    write_json(CAPTURE_LOCK, compatibility)
    print(
        json.dumps(
            {
                "status": "LOCKED",
                "contract_lock": str(LOCK),
                "contract_sha256": sha256(LOCK),
                "capture_compatibility_lock": str(CAPTURE_LOCK),
                "capture_lock_sha256": sha256(CAPTURE_LOCK),
            },
            indent=2,
        )
    )


def verify_locks() -> tuple[dict, dict]:
    if not LOCK.is_file() or not CAPTURE_LOCK.is_file():
        raise FileNotFoundError("r3 pilot locks are missing")
    validate_contract()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    capture_lock = json.loads(CAPTURE_LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_PILOT_CAPTURE":
        raise ValueError("invalid r3 primary lock")
    for relative, expected in lock.get("source_artifact_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"primary locked source changed: {relative}")
    if capture_lock.get("parent_contract_lock_sha256") != sha256(LOCK):
        raise ValueError("capture lock parent mismatch")
    for relative, expected in capture_lock.get(
        "source_artifact_sha256", {}
    ).items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"capture locked source changed: {relative}")
    return lock, capture_lock


def preflight_static() -> None:
    scenes, annotations, gate = validate_contract()
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "deterministic_generator": True,
            "scene_count_32": len(scenes["scenes"]) == 32,
            "annotation_count_32": len(annotations["scenes"]) == 32,
            "unique_scene_family_layout_signatures": True,
            "no_prior_dev_or_uq_family_overlap": True,
            "no_prior_uq_pose_signature_overlap": True,
            "state_relation_quotas": True,
            "wp4_grounding_and_hypothesis_locks": True,
            "geometry_only_policy": gate["policies"],
        },
        "contract_lock_sha256": sha256(LOCK) if LOCK.is_file() else None,
        "capture_authorized": LOCK.is_file() and CAPTURE_LOCK.is_file(),
    }
    if LOCK.is_file() or CAPTURE_LOCK.is_file():
        verify_locks()
        report["locks_verified"] = True
    write_json(STATIC_PREFLIGHT, report)
    print(json.dumps(report, indent=2))


def command_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout


def preflight_live() -> None:
    verify_locks()
    topics = command_output(["ros2", "topic", "list", "-t"])
    services = command_output(["ros2", "service", "list", "-t"])
    required_topics = {
        "/wrist_camera/color/image_raw": "sensor_msgs/msg/Image",
        "/wrist_camera/depth/image_raw": "sensor_msgs/msg/Image",
        "/wrist_camera/evaluation_labels/labels_map": "sensor_msgs/msg/Image",
        "/wrist_camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
        "/tf": "tf2_msgs/msg/TFMessage",
        "/joint_states": "sensor_msgs/msg/JointState",
    }
    required_services = {
        "/world/ur3_pick_place/set_pose": "ros_gz_interfaces/srv/SetEntityPose"
    }
    topic_checks = {
        name: f"{name} [{type_name}]" in topics
        for name, type_name in required_topics.items()
    }
    service_checks = {
        name: f"{name} [{type_name}]" in services
        for name, type_name in required_services.items()
    }
    free_gib = shutil.disk_usage(ROOT).free / 2**30
    passed = all(topic_checks.values()) and all(service_checks.values()) and free_gib > 2
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if passed else "BLOCKED",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_lock_sha256": sha256(LOCK),
        "capture_compatibility_lock_sha256": sha256(CAPTURE_LOCK),
        "topic_checks": topic_checks,
        "service_checks": service_checks,
        "free_gib": free_gib,
        "capture_authorized": passed,
    }
    write_json(LIVE_PREFLIGHT, report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


def labels(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode semantic labels: {path}")
    if image.ndim == 3:
        if not (
            np.array_equal(image[..., 0], image[..., 1])
            and np.array_equal(image[..., 1], image[..., 2])
        ):
            raise ValueError(f"semantic-label channels disagree: {path}")
        image = image[..., 0]
    return image


def geometry(label_map: np.ndarray, label: int) -> dict:
    ys, xs = np.nonzero(label_map == int(label))
    if not xs.size:
        return {
            "semantic_label": int(label),
            "visible_pixels": 0,
            "centroid_normalized_xy": None,
            "bbox_xyxy": None,
        }
    return {
        "semantic_label": int(label),
        "visible_pixels": int(xs.size),
        "centroid_normalized_xy": [
            float(xs.mean() / label_map.shape[1]),
            float(ys.mean() / label_map.shape[0]),
        ],
        "bbox_xyxy": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ],
    }


def rotation_matrix(quaternion: list[float]) -> np.ndarray:
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def project_base_point(point: list[float], transform: dict, camera: dict) -> dict:
    camera_point = rotation_matrix(transform["orientation_xyzw"]).T @ (
        np.asarray(point, dtype=np.float64)
        - np.asarray(transform["position"], dtype=np.float64)
    )
    z = float(camera_point[2])
    if z <= 0:
        return {"camera_xyz": camera_point.tolist(), "pixel_xy": None}
    intrinsics = camera["k"]
    u = intrinsics[0] * float(camera_point[0]) / z + intrinsics[2]
    v = intrinsics[4] * float(camera_point[1]) / z + intrinsics[5]
    return {"camera_xyz": camera_point.tolist(), "pixel_xy": [u, v]}


def forbidden_input_paths(value, prefix: str = "$", findings=None) -> list[str]:
    findings = [] if findings is None else findings
    forbidden = {
        "scene_family_id",
        "task_type",
        "target_model",
        "target_id",
        "target_pose",
        "target_pixel",
        "target_bbox",
        "target_mask",
        "semantic_label",
        "answerability_state",
        "candidate_set",
        "failure_tags",
        "oracle",
        "ground_truth",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if str(key).strip().lower() in forbidden:
                findings.append(child_path)
            forbidden_input_paths(child, child_path, findings)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            forbidden_input_paths(child, f"{prefix}[{index}]", findings)
    return findings


def validate_capture() -> tuple[dict, dict, dict, dict, list[dict]]:
    verify_locks()
    scenes, annotations, gate = map(
        load_yaml, (SCENES, ANNOTATIONS, GATE)
    )
    manifest_path = CAPTURE / "capture_manifest.json"
    input_manifest_path = CAPTURE / "input_manifest.jsonl"
    if not manifest_path.is_file() or not input_manifest_path.is_file():
        raise FileNotFoundError("r3 capture manifest or input manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = load_jsonl(input_manifest_path)
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("protocol_id") != PROTOCOL_ID
        or len(inputs) != 32
        or len({row["scene_id"] for row in inputs}) != 32
    ):
        raise ValueError("r3 capture is incomplete or has a bad record count")
    expected = {
        "scene_config_sha256": sha256(SCENES),
        "annotation_file_sha256": sha256(ANNOTATIONS),
        "gate_config_file_sha256": sha256(GATE),
        "pretrial_source_lock_sha256": sha256(CAPTURE_LOCK),
        "input_manifest_sha256": sha256(input_manifest_path),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"capture provenance mismatch: {key}")
    if (
        manifest.get("source_artifacts_verified_before_capture") is not True
        or manifest.get("semantic_labels_are_evaluator_only") is not True
        or manifest.get("target_handoff_published") is not False
        or manifest.get("robot_manipulation_performed") is not False
        or manifest.get("model_inventory_sha256_preregistered")
        != "NO_MODEL_INFERENCE_GEOMETRY_PILOT_R3"
    ):
        raise ValueError("capture safety/provenance contract failed")
    leaks = [
        (row.get("scene_id"), path)
        for row in inputs
        for path in forbidden_input_paths(row)
    ]
    if leaks:
        raise ValueError(f"oracle fields leaked into captured input records: {leaks}")
    return scenes, annotations, gate, manifest, inputs


def rgb_duplicate_audit(inputs: list[dict], gate: dict) -> dict:
    rows = []
    for record in inputs:
        path = CAPTURE / record["input_files"]["rgb"]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"cannot decode RGB for duplicate audit: {path}")
        rows.append((record["scene_id"], path, image))
    hashes = {sid: sha256(path) for sid, path, _ in rows}
    exact_groups: dict[str, list[str]] = {}
    for sid, value in hashes.items():
        exact_groups.setdefault(value, []).append(sid)
    exact_duplicates = [group for group in exact_groups.values() if len(group) > 1]

    rule = gate["layout_uniqueness"]["perceptual_near_duplicate_rule"]
    mad_limit = float(rule["near_duplicate_if_gray_mad_below"])
    changed_limit = float(rule["and_changed_pixel_fraction_below"])
    pixel_threshold = int(rule["gray_absdiff_threshold_for_changed_pixel"])
    pairs = []
    near_duplicates = []
    for left, right in itertools.combinations(rows, 2):
        difference = cv2.absdiff(left[2], right[2])
        mad = float(np.mean(difference))
        changed = float(np.mean(difference >= pixel_threshold))
        item = {
            "left": left[0],
            "right": right[0],
            "gray_mad": mad,
            "changed_pixel_fraction": changed,
        }
        pairs.append(item)
        if mad < mad_limit and changed < changed_limit:
            near_duplicates.append(item)
    closest = sorted(
        pairs, key=lambda item: (item["gray_mad"], item["changed_pixel_fraction"])
    )[:10]
    return {
        "status": "PASS" if not exact_duplicates and not near_duplicates else "FAIL",
        "rgb_count": len(rows),
        "unique_rgb_sha256": len(set(hashes.values())),
        "exact_duplicate_groups": exact_duplicates,
        "perceptual_near_duplicate_pairs": near_duplicates,
        "closest_pairs": closest,
        "rule": rule,
    }


def geometry_qc() -> None:
    scenes, annotations, gate, manifest, inputs = validate_capture()
    scene_map = {row["scene_id"]: row for row in scenes["scenes"]}
    oracle = annotations["scenes"]
    duplicate_audit = rgb_duplicate_audit(inputs, gate)
    minimum_pixels = int(gate["min_visible_evidence_px"])
    tie_margin = float(gate["tie_margin_normalized"])
    qc_rows = []

    for record in inputs:
        sid = record["scene_id"]
        item = oracle[sid]
        scene = scene_map[sid]
        reasons: list[str] = []
        label_map = labels(CAPTURE / sid / "evaluator/semantic_labels.png")
        depth = np.load(
            CAPTURE / record["input_files"]["depth_m"], allow_pickle=False
        )
        camera = json.loads(
            (CAPTURE / record["input_files"]["camera_info"]).read_text(
                encoding="utf-8"
            )
        )
        transforms = json.loads(
            (CAPTURE / record["input_files"]["tf_snapshot"]).read_text(
                encoding="utf-8"
            )
        )
        transform = transforms.get("camera_color_optical_frame", {})
        if depth.shape != label_map.shape or depth.dtype not in (
            np.float32,
            np.float64,
        ):
            reasons.append("DEPTH_NOT_REGISTERED_METRIC_ARRAY")
        if (
            camera.get("width") != 640
            or camera.get("height") != 480
            or camera.get("frame_id") != "camera_color_optical_frame"
        ):
            reasons.append("CAMERA_CONTRACT_MISMATCH")
        if "error" in transform or not transform.get("position"):
            reasons.append("CAMERA_TF_MISSING")

        candidates = [
            geometry(label_map, label)
            for label in item.get("candidate_labels", [])
        ]
        context = [
            geometry(label_map, label)
            for label in item.get("context_labels", [])
        ]
        target = (
            geometry(label_map, item["target_label"])
            if item.get("target_label") is not None
            else None
        )
        state = item["state"]
        verified = False
        details: dict = {}

        if state == "FOUND":
            visible = all(
                candidate["visible_pixels"] >= minimum_pixels
                for candidate in candidates
            )
            order = (
                sorted(
                    candidates,
                    key=lambda candidate: candidate["centroid_normalized_xy"][0],
                )
                if visible
                else []
            )
            if item["rank_from"] == "right":
                order.reverse()
            gaps = [
                abs(
                    order[index]["centroid_normalized_xy"][0]
                    - order[index + 1]["centroid_normalized_xy"][0]
                )
                for index in range(max(0, len(order) - 1))
            ]
            verified = bool(
                visible
                and item["rank"] <= len(order)
                and order[item["rank"] - 1]["semantic_label"]
                == item["target_label"]
                and all(gap > tie_margin for gap in gaps)
            )
            details = {
                "ordered_semantic_labels": [
                    entry["semantic_label"] for entry in order
                ],
                "adjacent_normalized_x_gaps": gaps,
            }
        elif state == "AMBIGUOUS":
            visible = all(
                candidate["visible_pixels"] >= minimum_pixels
                for candidate in candidates
            )
            valid = [
                geometry(label_map, label)
                for label in item["valid_target_labels"]
            ]
            order = (
                sorted(
                    candidates,
                    key=lambda candidate: candidate["centroid_normalized_xy"][0],
                )
                if visible
                else []
            )
            if item["rank_from"] == "right":
                order.reverse()
            tied_positions = [
                index
                for index, entry in enumerate(order)
                if entry["semantic_label"] in set(item["valid_target_labels"])
            ]
            xs = [
                entry["centroid_normalized_xy"][0]
                for entry in valid
                if entry["centroid_normalized_xy"] is not None
            ]
            rendered_gap = max(xs) - min(xs) if len(xs) == 2 else None
            verified = bool(
                visible
                and rendered_gap is not None
                and rendered_gap <= tie_margin
                and item["rank"] - 1 in tied_positions
            )
            details = {
                "rendered_tie_gap_normalized_x": rendered_gap,
                "tied_positions_zero_based": tied_positions,
                "ordered_semantic_labels": [
                    entry["semantic_label"] for entry in order
                ],
            }
        elif state == "ABSENT":
            verified = bool(
                target
                and target["visible_pixels"] == 0
                and item["target_id"] not in scene["poses"]
                and all(
                    entry["visible_pixels"] >= minimum_pixels for entry in context
                )
            )
            details = {
                "target_visible_pixels": target["visible_pixels"] if target else None,
                "context_visible_pixels": [
                    entry["visible_pixels"] for entry in context
                ],
            }
        else:
            projected = None
            if transform.get("position"):
                capture_oracle = json.loads(
                    (
                        CAPTURE / sid / "evaluator/capture_oracle.json"
                    ).read_text(encoding="utf-8")
                )
                layout = capture_oracle["requested_scene_layout_base_link"]
                projected = project_base_point(
                    layout[item["target_id"]][:3], transform, camera
                )
            rule = gate["insufficient_evidence_rule"]
            outer = float(rule["projected_center_outer_margin_fraction"])
            pixel = projected.get("pixel_xy") if projected else None
            near_sensor = bool(
                pixel
                and -640 * outer <= pixel[0] < 640 * (1 + outer)
                and -480 * outer <= pixel[1] < 480 * (1 + outer)
                and 0.1 <= projected["camera_xyz"][2] <= 2.0
            )
            others = [
                entry
                for entry in candidates
                if entry["semantic_label"] != item["target_label"]
            ]
            verified = bool(
                target
                and int(rule["min_visible_pixels_inclusive"])
                <= target["visible_pixels"]
                < int(rule["max_visible_pixels_exclusive"])
                and near_sensor
                and all(
                    entry["visible_pixels"] >= minimum_pixels for entry in others
                )
            )
            details = {
                "target_visible_pixels": target["visible_pixels"] if target else None,
                "other_visible_pixels": [
                    entry["visible_pixels"] for entry in others
                ],
                "target_center_projection_from_geometry": projected,
                "near_sensor_fov": near_sensor,
            }

        if not verified:
            reasons.append(f"REQUESTED_STATE_NOT_VERIFIED:{state}")
        qc_rows.append(
            {
                "scene_id": sid,
                "family_id": item["family_id"],
                "layout_id": item["layout_id"],
                "layout_signature_sha256": item["layout_signature_sha256"],
                "requested_state": state,
                "relation_variant": item["relation_variant"],
                "state_verified": verified,
                "reasons": reasons,
                "target_visible_pixels": target["visible_pixels"] if target else 0,
                "geometry": details,
            }
        )

    prior_families = set()
    for path in DEV_MANIFESTS:
        prior_families.update(row["family_id"] for row in load_jsonl(path))
    for path in PRIOR_ANNOTATION_CONFIGS:
        prior_families.update(family_ids_from_annotations(path))
    families = [row["family_id"] for row in qc_rows]
    leakage_checks = {
        "unique_parent_families_32": len(set(families)) == 32,
        "no_dev_or_prior_uq_family_overlap": not bool(
            set(families) & prior_families
        ),
        "input_manifest_oracle_free": all(
            not forbidden_input_paths(record) for record in inputs
        ),
        "semantic_labels_evaluator_only": manifest.get(
            "semantic_labels_are_evaluator_only"
        )
        is True,
        "no_model_inference": manifest.get("model_inventory_sha256_preregistered")
        == "NO_MODEL_INFERENCE_GEOMETRY_PILOT_R3",
        "no_target_handoff": manifest.get("target_handoff_published") is False,
        "no_robot_manipulation": manifest.get("robot_manipulation_performed")
        is False,
    }
    passed = (
        duplicate_audit["status"] == "PASS"
        and all(not row["reasons"] for row in qc_rows)
        and all(leakage_checks.values())
    )
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if passed else "REJECT",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_lock_sha256": sha256(LOCK),
        "capture_compatibility_lock_sha256": sha256(CAPTURE_LOCK),
        "capture_manifest_sha256": sha256(CAPTURE / "capture_manifest.json"),
        "capture_input_manifest_sha256": sha256(
            CAPTURE / "input_manifest.jsonl"
        ),
        "records": len(qc_rows),
        "state_counts": dict(Counter(row["requested_state"] for row in qc_rows)),
        "verified_state_counts": dict(
            Counter(
                row["requested_state"]
                for row in qc_rows
                if row["state_verified"]
            )
        ),
        "relation_variant_counts": dict(
            Counter(row["relation_variant"] for row in qc_rows)
        ),
        "failed_scene_count": sum(bool(row["reasons"]) for row in qc_rows),
        "rgb_duplicate_audit": duplicate_audit,
        "leakage_and_safety_checks": leakage_checks,
        "full_train_uq_design_authorized": passed,
        "model_inference_performed": False,
        "dataset_materialized": False,
        "scenes": qc_rows,
    }
    write_json(QC_REPORT, report)
    decision = "PASS" if passed else "REJECT"
    failed = [row for row in qc_rows if row["reasons"]]
    decision_text = f"""# Gazebo_train_uq_v2_pilot_r3 — capture attempt 01 decision

Decision: **{decision}**

- Contract lock: `{sha256(LOCK)}`
- Capture compatibility lock: `{sha256(CAPTURE_LOCK)}`
- Capture manifest: `{sha256(CAPTURE / 'capture_manifest.json')}`
- Geometry QC report: `{sha256(QC_REPORT)}`
- Captured rows: `{len(qc_rows)}/32`
- Geometry-verified rows: `{sum(row['state_verified'] for row in qc_rows)}/32`
- Exact RGB duplicates: `{len(duplicate_audit['exact_duplicate_groups'])}`
- Perceptual near-duplicate pairs: `{len(duplicate_audit['perceptual_near_duplicate_pairs'])}`
- Failed scenes: `{len(failed)}`
- Oracle/family/safety checks: `{'PASS' if all(leakage_checks.values()) else 'FAIL'}`

This was a geometry-only capture. No VLM inference, dataset materialization,
training, calibration, target handoff, robot manipulation, B2, or Test access
was performed.

{"A full Train-UQ/Val-UQ design may now be created as a new locked protocol; this pilot itself is never training data." if passed else "Full Train-UQ/Val-UQ design and capture remain blocked. Create a new revision only after reviewing the frozen QC failures; do not relabel or filter this attempt into training data."}
"""
    DECISION.write_text(decision_text, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": decision,
                "qc_report": str(QC_REPORT),
                "decision": str(DECISION),
                "geometry_verified": sum(
                    row["state_verified"] for row in qc_rows
                ),
                "failed_scenes": len(failed),
                "exact_duplicate_groups": len(
                    duplicate_audit["exact_duplicate_groups"]
                ),
                "perceptual_near_duplicates": len(
                    duplicate_audit["perceptual_near_duplicate_pairs"]
                ),
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-contract")
    subparsers.add_parser("lock")
    subparsers.add_parser("preflight-static")
    subparsers.add_parser("preflight-live")
    subparsers.add_parser("geometry-qc")
    args = parser.parse_args()
    if args.command == "validate-contract":
        validate_contract()
        print("PASS: Gazebo Train-UQ v2 pilot r3 contract")
    elif args.command == "lock":
        lock_contract()
    elif args.command == "preflight-static":
        preflight_static()
    elif args.command == "preflight-live":
        preflight_live()
    else:
        geometry_qc()


if __name__ == "__main__":
    main()
