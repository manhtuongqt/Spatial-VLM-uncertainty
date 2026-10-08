#!/usr/bin/env python3
"""Validate, lock, and preflight Gazebo_train_uq_v1 before any capture or LoRA."""

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

from generate_gazebo_train_uq_v1_contract import FRUITS, RELATIONS, SPLITS, STATES, build, serialized


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_train_uq_v1"
CONFIG = ROOT / "ur3/ur3_perception/config"
SCENES = CONFIG / "gazebo_train_uq_v1_scenes.yaml"
ANNOTATIONS = CONFIG / "gazebo_train_uq_v1_annotations.yaml"
GATE = CONFIG / "gazebo_train_uq_v1_gate.yaml"
LOCK = ROOT / "protocol/gazebo_train_uq_v1_contract_lock.json"
RESULT_ROOT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v1"
STATIC_PREFLIGHT = RESULT_ROOT / "PREFLIGHT_STATIC.json"
LIVE_PREFLIGHT = RESULT_ROOT / "PREFLIGHT_LIVE.json"
BASE = ROOT / "RoboRefer/models/RoboRefer-2B-SFT"
CLEAN_B1_ADAPTER = ROOT / "results/spatial_vlm_refspatial_v1/wp7_b1_clean_grounding/model/adapter_model.safetensors"
FROZEN_DEV = {
    "protocol/gazebo_dev_answerability_v2_contract_lock.json": "93da829b80d5fa1f2bb7a00a0f79ad80717e99e846a163e572ecb309a32a08e8",
    "datasets/Gazebo_dev_answerability_v2/manifest.json": "57fe333df4570cd32093d9810fb347e91ce7af90ce596805ad3fe25d52aad28a",
    "results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2/GAZEBO_DEV_ANSWERABILITY_V2_QC.json": "3ac26194dd2410472dbc763fcc82905bf15fcad2d7f9619ace395c91d406912d",
    "results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2/evaluation/B0_B1_ANSWERABILITY_V2_METRICS.json": "161f3971f781b1b9a901667273be297764948b5512ea78fdf329e578616c0983",
}
REQUIRED_SOURCES = [
    Path("protocol/generate_gazebo_train_uq_v1_contract.py"), Path("protocol/gazebo_train_uq_v1_pipeline.py"),
    Path("protocol/gazebo_train_uq_v1_infer.py"), Path("protocol/gazebo_train_uq_v1_metrics.py"),
    Path("ur3/ur3_perception/config/gazebo_train_uq_v1_scenes.yaml"), Path("ur3/ur3_perception/config/gazebo_train_uq_v1_annotations.yaml"), Path("ur3/ur3_perception/config/gazebo_train_uq_v1_gate.yaml"),
    Path("ur3/ur3_perception/scripts/roborefer_pilot_capture.py"), Path("ur3/ur3_perception/scripts/roborefer_pilot_runner.py"),
    Path("ur3/ur3_perception/scripts/roborefer_pilot_validate.py"), Path("ur3/ur3_perception/scripts/roborefer_pilot_evaluator.py"),
    Path("ur3/ur3_perception/scripts/roborefer_grounder.py"), Path("ur3/ur3_perception/scripts/spatial_point_utils.py"),
    Path("ur3/ur3_perception/scripts/move_camera_to_view.py"), Path("ur3/ur3_perception/launch/roborefer_pilot_capture.launch.py"),
    Path("ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro"), Path("ur3/ur3_moveit_control/launch/ur3_susgrip_sim.launch.py"),
    Path("ur3/ur_simulation_gz/worlds/ur3_pick_place.sdf"), Path("ur3/ur_simulation_gz/launch/ur_sim_control.launch.py"),
]
EXPECTED_OUTPUT = {"point_regex": r"^POINT \[\(([0-9]*\.?[0-9]+), ([0-9]*\.?[0-9]+)\)\]$", "abstain_literal": "ABSTAIN", "allowed_actions": ["POINT", "ABSTAIN"], "point_state": "FOUND", "abstain_states": ["AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_frozen_dev() -> None:
    changed = []
    for relative, expected in FROZEN_DEV.items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            changed.append({"path": relative, "expected": expected, "actual": actual})
    if changed:
        raise ValueError(f"frozen Gazebo_dev_answerability_v2 artifacts changed: {changed}")


def validate_contract() -> tuple[dict, dict, dict]:
    verify_frozen_dev()
    expected_payloads = build()
    for path, payload in zip((SCENES, ANNOTATIONS, GATE), expected_payloads):
        if not path.is_file() or path.read_text(encoding="utf-8") != serialized(payload):
            raise ValueError(f"deterministic contract source mismatch: {path}")
    scenes, annotations, gate = (load_yaml(path) for path in (SCENES, ANNOTATIONS, GATE))
    rows, oracle = scenes.get("scenes"), annotations.get("scenes")
    expected_total = sum(16 * repetitions for repetitions in SPLITS.values())
    if scenes.get("protocol_id") != PROTOCOL_ID or annotations.get("protocol_id") != PROTOCOL_ID or gate.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("protocol ID mismatch")
    if len(rows or []) != expected_total or scenes.get("expected_scene_count") != expected_total or len(oracle or {}) != expected_total:
        raise ValueError("expected 320 preregistered scene-query families")
    ids = [row.get("scene_id") for row in rows]
    families = [row.get("scene_family_id") for row in rows]
    if len(set(ids)) != expected_total or set(ids) != set(oracle) or len(set(families)) != expected_total:
        raise ValueError("scene IDs/family IDs are not unique and aligned")
    prior = set()
    for path in (ROOT / "datasets/Gazebo_dev/evaluator_ground_truth.jsonl", ROOT / "datasets/Gazebo_dev_answerability_v2/evaluator_ground_truth.jsonl"):
        prior.update(row["family_id"] for row in load_jsonl(path))
    if set(families) & prior:
        raise ValueError("Train-UQ/Val-UQ overlaps a frozen Gazebo Dev family")
    by_split = {split: [row for row in rows if oracle[row["scene_id"]].get("split") == split] for split in SPLITS}
    if sum(map(len, by_split.values())) != expected_total or any(row.get("scene_family_id") != oracle[row["scene_id"]].get("family_id") for row in rows):
        raise ValueError("split/family annotation mismatch")
    for split, repetitions in SPLITS.items():
        entries = by_split[split]
        if len(entries) != 16 * repetitions:
            raise ValueError(f"{split}: parent-family quota mismatch")
        state_count = Counter(oracle[row["scene_id"]].get("state") for row in entries)
        relation_count = Counter(oracle[row["scene_id"]].get("relation_variant") for row in entries)
        cells = Counter((oracle[row["scene_id"]].get("state"), oracle[row["scene_id"]].get("relation_variant")) for row in entries)
        if set(state_count) != set(STATES) or any(state_count[state] != 4 * repetitions for state in STATES):
            raise ValueError(f"{split}: state balance mismatch: {state_count}")
        if set(relation_count) != set(RELATIONS) or any(relation_count[relation] != 4 * repetitions for relation in RELATIONS):
            raise ValueError(f"{split}: relation balance mismatch: {relation_count}")
        if any(cells[(state, relation)] != repetitions for state in STATES for relation in RELATIONS):
            raise ValueError(f"{split}: state x relation factorial balance mismatch")
        appearances = Counter(name for row in entries for name in row.get("poses", {}) if name in FRUITS)
        for name, minimum in gate["focus_object_quota_min"][split].items():
            if appearances[name] < minimum:
                raise ValueError(f"{split}: focus-object quota fails for {name}")
        tags = Counter(tag for row in entries for tag in oracle[row["scene_id"]].get("failure_tags", []))
        for tag, minimum in gate["hard_case_quota_min"][split].items():
            if tags[tag] < minimum:
                raise ValueError(f"{split}: hard-case quota fails for {tag}")
    for row in rows:
        item = oracle[row["scene_id"]]
        if row.get("task_type") != "horizontal_ordinal_ranking_answerability" or set(row.get("poses", {})) - set(scenes.get("objects", {})):
            raise ValueError(f"invalid task/assets: {row['scene_id']}")
        state = item.get("state")
        if state == "FOUND" and item.get("target_id") not in item.get("candidate_ids", []):
            raise ValueError(f"invalid FOUND target: {row['scene_id']}")
        if state == "AMBIGUOUS" and (item.get("target_id") is not None or len(item.get("valid_target_ids", [])) < 2):
            raise ValueError(f"invalid AMBIGUOUS target: {row['scene_id']}")
        if state == "ABSENT" and (not item.get("target_id") or item["target_id"] in row.get("poses", {})):
            raise ValueError(f"invalid ABSENT target: {row['scene_id']}")
        if state == "INSUFFICIENT_EVIDENCE" and (not item.get("target_id") or item["target_id"] not in row.get("poses", {}) or "edge_truncation" not in item.get("failure_tags", [])):
            raise ValueError(f"invalid INSUFFICIENT_EVIDENCE target: {row['scene_id']}")
    dev_lock = json.loads((ROOT / "protocol/gazebo_dev_answerability_v2_contract_lock.json").read_text(encoding="utf-8"))
    if gate.get("camera") != dev_lock.get("camera") or annotations.get("reference_frame") != "image_viewer_left_to_right" or gate.get("reference_frame") != "image_viewer_left_to_right":
        raise ValueError("camera or reference-frame drift from frozen answerability-v2")
    if gate.get("output_contract") != EXPECTED_OUTPUT or gate.get("tie_margin_normalized") != 0.03 or gate.get("min_visible_evidence_px") != 120:
        raise ValueError("POINT/ABSTAIN or geometry threshold contract mismatch")
    if gate.get("no_sam2") is not True or gate.get("b2_opened") is not False or gate.get("test_iid_ood_access") is not False:
        raise ValueError("SAM2/B2/test policy is not locked")
    selection = gate.get("model_selection", {})
    if selection.get("selection_split") != "val_uq" or "gazebo_dev_answerability_v2" not in selection.get("forbidden_selection_splits", []):
        raise ValueError("Val-UQ-only model selection policy is missing")
    return scenes, annotations, gate


def lock_contract() -> None:
    scenes, annotations, gate = validate_contract()
    if LOCK.exists():
        raise FileExistsError(f"refusing to overwrite existing lock: {LOCK}")
    missing = [str(path) for path in REQUIRED_SOURCES if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required sources missing: {missing}")
    if not BASE.is_dir() or not CLEAN_B1_ADAPTER.is_file():
        raise FileNotFoundError("frozen B0 base or clean-B1 adapter is missing")
    model_hashes = {"b0_base": directory_sha256(BASE), "clean_b1_adapter": sha256(CLEAN_B1_ADAPTER), "b1_uq_adapter": None}
    source_hashes = {str(path): sha256(ROOT / path) for path in REQUIRED_SOURCES}
    train_counts = gate["split_parent_family_count"]
    payload = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "LOCKED_BEFORE_CAPTURE_AND_TRAINING",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(), "workspace_root": str(ROOT),
        "parent_family_count": sum(train_counts.values()), "split_parent_family_count": train_counts,
        "state_quota": gate["state_quota"], "relation_variant_quota": gate["relation_variant_quota"], "state_relation_cell_quota": gate["state_relation_cell_quota"],
        "reference_frame": gate["reference_frame"], "camera": gate["camera"], "tie_margin_normalized": gate["tie_margin_normalized"], "min_visible_evidence_px": gate["min_visible_evidence_px"], "insufficient_evidence_rule": gate["insufficient_evidence_rule"],
        "output_contract": gate["output_contract"], "family_split_rule": gate["family_split_rule"], "model_selection": gate["model_selection"], "post_val_dev_rule": gate["post_val_dev_rule"], "future_splits": gate["future_splits"],
        "model_definition": {"b0": "frozen RoboRefer-2B-SFT", "clean_b1": "frozen D_tabletop_clean_v1 LoRA", "b1_uq": "untrained; must be a new Train-UQ-only LoRA adapter and receive a separate training lock after capture QC"},
        "inference_configuration": {"runner": "protocol/gazebo_train_uq_v1_infer.py", "metrics": "protocol/gazebo_train_uq_v1_metrics.py", "models": ["b0", "clean_b1", "b1_uq"], "greedy_decoding": True, "stochastic_draws": 3, "max_new_tokens": 64, "parser": "strict POINT [(x, y)] or exact ABSTAIN; tolerant parsing diagnostic only"},
        "dev_v2_frozen_artifact_sha256": FROZEN_DEV, "source_artifact_sha256": source_hashes, "model_sha256": model_hashes,
        "model_inventory_sha256": hashlib.sha256(json.dumps(model_hashes, sort_keys=True).encode()).hexdigest(),
        "promotion_gate": {"found_grounding": "B1-UQ must be non-inferior to B0 at frozen margin -0.125 on Dev-v2 FOUND Hit@0.08", "abstention": "strictly higher non-FOUND abstain recall and strictly lower non-FOUND false-accept rate than B0", "format": "no exact-output-contract regression", "risk": "AURC no worse than B0", "decision_if_fail": "KEEP_B0_AND_REPORT_NEGATIVE_RESULT"},
        "policies": {"no_sam2": True, "b2_closed": True, "test_iid_ood_sealed": True, "no_capture_before_lock": True, "no_training_before_geometry_qc_pass": True, "no_dev_v2_fit_or_selection": True, "no_calibration_or_test_access": True},
    }
    write_json(LOCK, payload)
    print(json.dumps({"status": "LOCKED", "path": str(LOCK), "sha256": sha256(LOCK)}, indent=2))


def verify_lock_sources(lock: dict) -> None:
    if lock.get("protocol_id") != PROTOCOL_ID or lock.get("status") != "LOCKED_BEFORE_CAPTURE_AND_TRAINING":
        raise ValueError("invalid Train-UQ lock")
    changed = []
    for relative, expected in lock.get("source_artifact_sha256", {}).items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            changed.append({"path": relative, "expected": expected, "actual": actual})
    if changed:
        raise ValueError(f"locked source artifacts changed: {changed}")
    if lock.get("dev_v2_frozen_artifact_sha256") != FROZEN_DEV:
        raise ValueError("frozen Dev-v2 hashes differ from the lock")
    verify_frozen_dev()


def preflight_static() -> None:
    validate_contract()
    if not LOCK.is_file():
        raise FileNotFoundError("contract must be locked before preflight")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    verify_lock_sources(lock)
    disk = shutil.disk_usage(ROOT)
    report = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PASS", "kind": "STATIC_PREFLIGHT", "contract_lock_sha256": sha256(LOCK), "contract_inputs_deterministic": True, "frozen_dev_v2_unchanged": True, "source_hashes_verified": len(lock["source_artifact_sha256"]), "disk_free_gib": disk.free / 2**30, "capture_authorized": False, "reason": "A live ROS2/Gazebo preflight has not yet verified topics, TF, metric depth, semantic labels, or capture throughput."}
    write_json(STATIC_PREFLIGHT, report)
    print(json.dumps(report, indent=2))


def run_command(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20, check=False)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout}
    except Exception as exc:
        return {"command": command, "returncode": None, "stdout": repr(exc)}


def preflight_live() -> None:
    validate_contract()
    if not LOCK.is_file():
        raise FileNotFoundError("contract must be locked before preflight")
    lock = json.loads(LOCK.read_text(encoding="utf-8")); verify_lock_sources(lock)
    topics = run_command(["ros2", "topic", "list", "-t"])
    services = run_command(["ros2", "service", "list", "-t"])
    required_topics = {"/wrist_camera/color/image_raw": "sensor_msgs/msg/Image", "/wrist_camera/depth/image_raw": "sensor_msgs/msg/Image", "/wrist_camera/evaluation_labels/labels_map": "sensor_msgs/msg/Image", "/wrist_camera/color/camera_info": "sensor_msgs/msg/CameraInfo", "/tf": "tf2_msgs/msg/TFMessage"}
    text = topics["stdout"]
    topic_checks = {name: topics["returncode"] == 0 and f"{name} [{type_name}]" in text for name, type_name in required_topics.items()}
    service_check = services["returncode"] == 0 and "/world/ur3_pick_place/set_pose" in services["stdout"]
    disk = shutil.disk_usage(ROOT)
    passed = all(topic_checks.values()) and service_check
    report = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PASS" if passed else "BLOCKED", "kind": "LIVE_PREFLIGHT", "contract_lock_sha256": sha256(LOCK), "required_topic_checks": topic_checks, "reset_service_present": service_check, "disk_free_gib": disk.free / 2**30, "raw": {"topics": topics, "services": services}, "capture_authorized": passed}
    write_json(LIVE_PREFLIGHT, report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-contract")
    sub.add_parser("lock")
    sub.add_parser("preflight-static")
    sub.add_parser("preflight-live")
    args = parser.parse_args()
    if args.command == "validate-contract":
        validate_contract(); print("PASS: Gazebo_train_uq_v1 contract is internally consistent")
    elif args.command == "lock":
        lock_contract()
    elif args.command == "preflight-static":
        preflight_static()
    else:
        preflight_live()


if __name__ == "__main__":
    main()
