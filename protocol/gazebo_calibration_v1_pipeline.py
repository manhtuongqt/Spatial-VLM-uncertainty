#!/usr/bin/env python3
"""Validate, lock, and preflight the immutable final-calibration capture."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import yaml

import generate_gazebo_calibration_v1_contract as generator
import gazebo_train_uq_v2_pilot_r4_pipeline as capture_common


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = generator.PROTOCOL_ID
SCENES, ANNOTATIONS, GATE = generator.OUTPUT_PATHS
FAMILY_MANIFEST = generator.FAMILY_MANIFEST_PATH
SPLIT_MANIFEST = generator.SPLIT_MANIFEST_PATH
LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_CONTRACT_LOCK.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_calibration_v1_capture_compatibility_lock.json"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v1"
STATIC_PREFLIGHT = RESULT / "PREFLIGHT_STATIC.json"
LIVE_PREFLIGHT = RESULT / "PREFLIGHT_LIVE.json"
GROUNDING_LOCK = ROOT / "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/grounding_model_lock.json"
HYPOTHESIS_LOCK = ROOT / "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/hypothesis_lock.json"
WP5 = ROOT / "results/spatial_vlm_refspatial_v1/wp5_spatial_uncertainty"
WP5_METHOD_LOCK = WP5 / "wp5_method_lock.json"
WP5_ESTIMATOR_LOCK = WP5 / "wp5_checkpoint_or_estimator_lock.json"
WP5_ESTIMATOR = WP5 / "selected_estimator.json"
WP5_ESTIMATOR_BINARY = WP5 / "selected_estimator.joblib"
WP5_DECISION = WP5 / "CALIBRATION_GO_NO_GO.json"
FULL_R3_QC = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full_r3/GAZEBO_TRAIN_UQ_V2_FULL_R3_V2_GEOMETRY_QC.json"
WORLD = ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf"
STATES = generator.STATES
RELATIONS = generator.RELATIONS


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate_contract():
    expected = generator.build()
    texts = [generator.serialized(x) for x in expected[:3]] + [
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in expected[3]),
        json.dumps(expected[4], indent=2, sort_keys=True) + "\n",
    ]
    paths = [SCENES, ANNOTATIONS, GATE, FAMILY_MANIFEST, SPLIT_MANIFEST]
    drift = [str(p) for p, t in zip(paths, texts) if not p.is_file() or p.read_text() != t]
    if drift:
        raise ValueError(f"deterministic contract drift: {drift}")
    scenes, annotations, gate = (yaml.safe_load(p.read_text()) for p in (SCENES, ANNOTATIONS, GATE))
    rows, oracle = scenes["scenes"], annotations["scenes"]
    if len(rows) != 128 or len(oracle) != 128:
        raise ValueError("calibration must contain exactly 128 families")
    identities = ([r["scene_id"] for r in rows], [r["scene_family_id"] for r in rows],
                  [r["layout_id"] for r in rows], [r["layout_signature_sha256"] for r in rows])
    if any(len(set(values)) != 128 for values in identities):
        raise ValueError("calibration identities/signatures are not unique")
    counts = Counter((oracle[r["scene_id"]]["state"], oracle[r["scene_id"]]["relation_variant"]) for r in rows)
    if len(counts) != 16 or set(counts.values()) != {8}:
        raise ValueError(f"state/relation quota mismatch: {counts}")
    prior_family, prior_scene, prior_signature = set(), set(), set()
    for path in sorted((ROOT / "ur3/ur3_perception/config").glob("gazebo*_scenes.yaml")):
        if path == SCENES:
            continue
        data = yaml.safe_load(path.read_text()) or {}
        for row in data.get("scenes", []):
            prior_scene.add(str(row.get("scene_id")))
            prior_family.add(str(row.get("scene_family_id")))
            prior_signature.add(generator.source.v1.geometry.layout_signature(row["poses"]))
    for path in [ROOT / "datasets/Gazebo_dev/evaluator_ground_truth.jsonl",
                 ROOT / "datasets/Gazebo_dev_answerability_v2/evaluator_ground_truth.jsonl",
                 ROOT / "datasets/Gazebo_train_uq_v2_full_r3/evaluator_ground_truth.jsonl"]:
        for row in load_jsonl(path):
            prior_scene.add(str(row["scene_id"])); prior_family.add(str(row["family_id"]))
    if set(identities[0]) & prior_scene or set(identities[1]) & prior_family or set(identities[3]) & prior_signature:
        raise ValueError("Calibration overlaps a prior family/scene/layout signature")
    expected_hashes = {
        GROUNDING_LOCK: "6e1da7e9759eacfaadf7be07467f4996e95575fd2a6225af25d87e4f74418b09",
        HYPOTHESIS_LOCK: "077f65f3db1b9562bc589ba54ac47d19c5a6ee1c1a00d3a178228b5822e8c78a",
        WP5_METHOD_LOCK: "a4c3777bb043d4fe72701355a8bf6cd7a3dd44c135cd97e610273406d415ec45",
        WP5_ESTIMATOR_LOCK: "167c05da20530f5e22a6b743043422952a258688e71721ba229086c55dc76f4b",
    }
    for path, digest in expected_hashes.items():
        if sha(path) != digest:
            raise ValueError(f"immutable upstream lock changed: {path}")
    decision = json.loads(WP5_DECISION.read_text())
    if decision.get("decision") != "GO_TO_CALIBRATION":
        raise ValueError("WP5 did not authorize calibration")
    if gate["calibration_protocol"]["calibrator_count"] != 1 or not gate["sealed"]["gazebo_test_iid"]:
        raise ValueError("calibrator-count or Test sealing contract failed")
    return scenes, annotations, gate


def lock_contract():
    if LOCK.exists() or CAPTURE_LOCK.exists():
        raise FileExistsError("refusing to overwrite calibration locks")
    scenes, _, gate = validate_contract()
    sources = [Path(__file__).resolve(), Path(generator.__file__).resolve(), SCENES, ANNOTATIONS, GATE,
               FAMILY_MANIFEST, SPLIT_MANIFEST, GROUNDING_LOCK, HYPOTHESIS_LOCK, WP5_METHOD_LOCK,
               WP5_ESTIMATOR_LOCK, WP5_ESTIMATOR, WP5_ESTIMATOR_BINARY, WP5_DECISION, FULL_R3_QC,
               ROOT / "protocol/wp5_spatial_uncertainty_estimator.py",
               ROOT / "protocol/gazebo_calibration_v1_infer.py",
               ROOT / "protocol/materialize_gazebo_calibration_v1.py",
               ROOT / "protocol/gazebo_calibration_v1_fit.py",
               ROOT / "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py", WORLD]
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_CALIBRATION_CAPTURE_OR_ACCESS",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_family_count": 128,
        "state_relation_cell_quota": 8,
        "family_manifest_sha256": sha(FAMILY_MANIFEST),
        "split_manifest_sha256": sha(SPLIT_MANIFEST),
        "scene_layout_signature_set_sha256": hashlib.sha256(
            json.dumps(sorted(r["layout_signature_sha256"] for r in scenes["scenes"])).encode()
        ).hexdigest(),
        "source_artifact_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "frozen_upstream": {
            "b0_grounding_lock_sha256": sha(GROUNDING_LOCK),
            "hypothesis_lock_sha256": sha(HYPOTHESIS_LOCK),
            "wp5_method_lock_sha256": sha(WP5_METHOD_LOCK),
            "wp5_estimator_lock_sha256": sha(WP5_ESTIMATOR_LOCK),
            "selected_estimator_sha256": sha(WP5_ESTIMATOR),
            "selected_estimator_joblib_sha256": sha(WP5_ESTIMATOR_BINARY),
            "feature_count": 16,
            "C": 0.1,
            "val_selected_threshold_reference_only": 0.72,
        },
        "calibration_protocol": gate["calibration_protocol"],
        "operating_point_rule": gate["operating_point_rule"],
        "camera": gate["camera"],
        "visibility_gates": {
            "minimum_sufficient_pixels": gate["min_visible_evidence_px"],
            "insufficient": gate["insufficient_evidence_rule"],
            "tie_margin_normalized": gate["tie_margin_normalized"],
        },
        "sealed": gate["sealed"],
        "decision_if_pass": "CALIBRATION_PASS_READY_FOR_LOCKED_TEST",
        "decision_if_fail": "CALIBRATION_NEGATIVE_NO_ROBOT_DEPLOYMENT",
    }
    dump(LOCK, payload)
    capture_sources = set(capture_common.LEGACY_CAPTURE_SOURCES) | {
        str(p.relative_to(ROOT)) for p in sources
    }
    capture = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "parent_contract_lock": str(LOCK.relative_to(ROOT)),
        "parent_contract_lock_sha256": sha(LOCK),
        "purpose": "128-family no-manipulation final-calibration capture",
        "source_artifact_sha256": {name: sha(ROOT / name) for name in sorted(capture_sources)},
        "model_inventory_sha256": "NO_MODEL_INFERENCE_DURING_CALIBRATION_CAPTURE",
        "policies": {
            "no_model_inference_during_capture": True,
            "no_robot_manipulation": True,
            "semantic_labels_evaluator_only": True,
            "dev_v2_closed": True,
            "test_iid_ood_sealed": True,
        },
    }
    dump(CAPTURE_LOCK, capture)
    print(json.dumps({"status": "LOCKED", "contract_sha256": sha(LOCK), "capture_lock_sha256": sha(CAPTURE_LOCK)}, indent=2))


def verify_locks():
    validate_contract()
    lock, capture = (json.loads(p.read_text()) for p in (LOCK, CAPTURE_LOCK))
    if lock.get("status") != "LOCKED_BEFORE_CALIBRATION_CAPTURE_OR_ACCESS":
        raise ValueError("invalid calibration contract lock")
    if capture.get("parent_contract_lock_sha256") != sha(LOCK):
        raise ValueError("capture lock linkage failed")
    for item in (lock, capture):
        for name, digest in item["source_artifact_sha256"].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"locked source changed: {name}")


def static_preflight():
    verify_locks()
    free = shutil.disk_usage(ROOT).free / 2**30
    passed = free > 2.0
    report = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PASS" if passed else "BLOCKED",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "contract_lock_sha256": sha(LOCK),
              "capture_lock_sha256": sha(CAPTURE_LOCK), "deterministic_128_family_contract": True,
              "family_disjoint_and_quota_checks": True, "dev_v2_closed": True, "test_iid_ood_sealed": True,
              "free_gib": free, "capture_authorized": passed}
    dump(STATIC_PREFLIGHT, report); print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


def output(command):
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout


def live_preflight():
    verify_locks()
    topics, services = output(["ros2", "topic", "list", "-t"]), output(["ros2", "service", "list", "-t"])
    required = {"/wrist_camera/color/image_raw": "sensor_msgs/msg/Image",
                "/wrist_camera/depth/image_raw": "sensor_msgs/msg/Image",
                "/wrist_camera/evaluation_labels/labels_map": "sensor_msgs/msg/Image",
                "/wrist_camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
                "/tf": "tf2_msgs/msg/TFMessage", "/joint_states": "sensor_msgs/msg/JointState"}
    checks = {name: f"{name} [{kind}]" in topics for name, kind in required.items()}
    service = "/world/ur3_pick_place/set_pose [ros_gz_interfaces/srv/SetEntityPose]" in services
    free = shutil.disk_usage(ROOT).free / 2**30
    passed = all(checks.values()) and service and free > 2.0
    report = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PASS" if passed else "BLOCKED",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "topic_checks": checks,
              "set_pose_service": service, "free_gib": free, "capture_authorized": passed,
              "contract_lock_sha256": sha(LOCK), "capture_lock_sha256": sha(CAPTURE_LOCK)}
    dump(LIVE_PREFLIGHT, report); print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "lock", "preflight-static", "preflight-live"))
    command = parser.parse_args().command
    if command == "validate":
        validate_contract(); print("PASS: calibration-v1 contract")
    elif command == "lock":
        lock_contract()
    elif command == "preflight-static":
        static_preflight()
    else:
        live_preflight()


if __name__ == "__main__":
    main()
