#!/usr/bin/env python3
"""Lock, preflight and evaluator-only QC for Gazebo Train-UQ pilot r5."""
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

import gazebo_train_uq_v2_pilot_r4_pipeline as prior
import generate_gazebo_train_uq_v2_pilot_r4_contract as prior_generator
import generate_gazebo_train_uq_v2_pilot_r5_contract as generator

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_train_uq_v2_pilot_r5"
CONFIG = ROOT / "ur3/ur3_perception/config"
SCENES = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
AMENDMENT = ROOT / "protocol/gazebo_train_uq_v2_pilot_r5_amendment.json"
LOCK = ROOT / "protocol/gazebo_train_uq_v2_pilot_r5_contract_lock.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_train_uq_v2_pilot_r5_capture_compatibility_lock.json"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r5"
CAPTURE = RESULT / "capture_attempt_01"
STATIC_PREFLIGHT = RESULT / "PREFLIGHT_STATIC.json"
LIVE_PREFLIGHT = RESULT / "PREFLIGHT_LIVE.json"
QC_REPORT = RESULT / "GAZEBO_TRAIN_UQ_V2_PILOT_R5_GEOMETRY_QC.json"
DECISION = RESULT / "CAPTURE_ATTEMPT_01_DECISION.md"
GROUNDING_LOCK = ROOT / "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/grounding_model_lock.json"
HYPOTHESIS_LOCK = ROOT / "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/hypothesis_lock.json"
R4_LOCK = ROOT / "protocol/gazebo_train_uq_v2_pilot_r4_contract_lock.json"
R4_CAPTURE_LOCK = ROOT / "protocol/gazebo_train_uq_v2_pilot_r4_capture_compatibility_lock.json"
R4_QC = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r4/GAZEBO_TRAIN_UQ_V2_PILOT_R4_GEOMETRY_QC.json"
R4_DECISION = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r4/CAPTURE_ATTEMPT_01_DECISION.md"
V8_REPORT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_uq_occlusion_characterization_v8/OCCLUSION_CHARACTERIZATION_REPORT.json"
V9_REPORT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_uq_occlusion_characterization_v9/OCCLUSION_CHARACTERIZATION_REPORT.json"
DEV_MANIFESTS = prior.DEV_MANIFESTS
PRIOR_SCENE_CONFIGS = prior.PRIOR_SCENE_CONFIGS + (CONFIG / "gazebo_train_uq_v2_pilot_r4_scenes.yaml",)
PRIOR_ANNOTATION_CONFIGS = prior.PRIOR_ANNOTATION_CONFIGS + (CONFIG / "gazebo_train_uq_v2_pilot_r4_annotations.yaml",)
STATES = prior.STATES
RELATIONS = prior.RELATIONS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate_contract():
    generated = generator.build()
    for path, payload in zip((SCENES, ANNOTATIONS, GATE), generated):
        if not path.is_file() or path.read_text() != generator.serialized(payload):
            raise ValueError(f"non-deterministic r5 input: {path}")
    scenes, annotations, gate = map(load_yaml, (SCENES, ANNOTATIONS, GATE))
    rows, oracle = scenes["scenes"], annotations["scenes"]
    if len(rows) != 32 or len(oracle) != 32 or set(r["scene_id"] for r in rows) != set(oracle):
        raise ValueError("r5 requires 32 aligned rows")
    vectors = ([r["scene_id"] for r in rows], [r["scene_family_id"] for r in rows],
               [r["layout_id"] for r in rows], [r["layout_signature_sha256"] for r in rows])
    if any(len(set(values)) != 32 for values in vectors):
        raise ValueError("scene/family/layout/signature uniqueness failed")
    for row in rows:
        sid = row["scene_id"]
        signature = generator.layout_signature(row["poses"])
        if signature != row["layout_signature_sha256"] or signature != oracle[sid]["layout_signature_sha256"]:
            raise ValueError(f"signature mismatch: {sid}")
        if row["scene_family_id"] != oracle[sid]["family_id"]:
            raise ValueError(f"family mismatch: {sid}")
    states = Counter(item["state"] for item in oracle.values())
    relations = Counter(item["relation_variant"] for item in oracle.values())
    cells = Counter((item["state"], item["relation_variant"]) for item in oracle.values())
    if states != Counter({s: 8 for s in STATES}) or relations != Counter({r: 8 for r in RELATIONS}) or set(cells.values()) != {2}:
        raise ValueError("state/relation quota failed")
    old_families, old_scenes, old_signatures = set(), set(), set()
    for path in DEV_MANIFESTS:
        for item in load_jsonl(path): old_families.add(str(item["family_id"])); old_scenes.add(str(item["scene_id"]))
    for path in PRIOR_ANNOTATION_CONFIGS:
        old_families |= {str(item["family_id"]) for item in load_yaml(path).get("scenes", {}).values() if item.get("family_id")}
    for path in PRIOR_SCENE_CONFIGS:
        for item in load_yaml(path).get("scenes", []):
            old_scenes.add(str(item["scene_id"])); old_signatures.add(prior_generator.layout_signature(item["poses"]))
    if set(vectors[1]) & old_families or set(vectors[0]) & old_scenes or set(vectors[3]) & old_signatures:
        raise ValueError("r5 overlaps Dev or a prior UQ attempt")
    amendment = json.loads(AMENDMENT.read_text())
    if amendment.get("status") != "LOCKED_BEFORE_R5_DESIGN_AND_CAPTURE": raise ValueError("r5 amendment invalid")
    evidence = amendment["parent_attempt"]
    for path, key in ((R4_LOCK, "contract_lock_sha256"), (R4_CAPTURE_LOCK, "capture_lock_sha256"),
                      (R4_QC, "geometry_qc_sha256"), (R4_DECISION, "decision_sha256")):
        if sha256(path) != evidence[key]: raise ValueError(f"r4 evidence changed: {path}")
    if json.loads(R4_QC.read_text()).get("status") != "REJECT": raise ValueError("r4 is not frozen REJECT")
    if sha256(V8_REPORT) != amendment["characterization_evidence"]["v8_report_sha256"]: raise ValueError("v8 evidence changed")
    if sha256(V9_REPORT) != amendment["characterization_evidence"]["v9_report_sha256"]: raise ValueError("v9 evidence changed")
    if json.loads(V9_REPORT.read_text()).get("combined_characterization_status") != "PASS": raise ValueError("characterization not PASS")
    if sha256(GROUNDING_LOCK) != "6e1da7e9759eacfaadf7be07467f4996e95575fd2a6225af25d87e4f74418b09": raise ValueError("B0 lock changed")
    if sha256(HYPOTHESIS_LOCK) != "077f65f3db1b9562bc589ba54ac47d19c5a6ee1c1a00d3a178228b5822e8c78a": raise ValueError("hypothesis lock changed")
    expected = {"no_sam2": True, "no_training": True, "no_model_inference": True, "no_materialization": True,
                "no_b2": True, "no_test_access": True, "no_dev_v2_fit_or_selection": True,
                "no_robot_manipulation": True, "full_train_uq_only_after_pilot_pass": True}
    if gate["policies"] != expected: raise ValueError("safety policy mismatch")
    return scenes, annotations, gate


def lock_contract():
    if LOCK.exists() or CAPTURE_LOCK.exists(): raise FileExistsError("refusing overwrite r5 locks")
    scenes, _, gate = validate_contract()
    own = [Path("protocol/generate_gazebo_train_uq_v2_pilot_r5_contract.py"),
           Path("protocol/gazebo_train_uq_v2_pilot_r5_pipeline.py"), AMENDMENT.relative_to(ROOT),
           SCENES.relative_to(ROOT), ANNOTATIONS.relative_to(ROOT), GATE.relative_to(ROOT),
           GROUNDING_LOCK.relative_to(ROOT), HYPOTHESIS_LOCK.relative_to(ROOT), R4_LOCK.relative_to(ROOT),
           R4_CAPTURE_LOCK.relative_to(ROOT), R4_QC.relative_to(ROOT), R4_DECISION.relative_to(ROOT),
           V8_REPORT.relative_to(ROOT), V9_REPORT.relative_to(ROOT),
           Path("ur3/ur3_perception/launch/roborefer_uq_capture.launch.py"),
           Path("ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf")]
    source_hashes = {str(path): sha256(ROOT / path) for path in own}
    payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "LOCKED_BEFORE_PILOT_CAPTURE",
               "locked_at_utc": datetime.now(timezone.utc).isoformat(),
               "purpose": "geometry validation only; no inference, materialization, training, manipulation, calibration, or Test",
               "parent_families": 32, "state_relation_cell_quota": 2,
               "layout_signature_sha256": {r["scene_id"]: r["layout_signature_sha256"] for r in scenes["scenes"]},
               "source_artifact_sha256": source_hashes, "grounding_model_lock_sha256": sha256(GROUNDING_LOCK),
               "hypothesis_lock_sha256": sha256(HYPOTHESIS_LOCK), "r5_amendment_sha256": sha256(AMENDMENT),
               "geometry_gate": {k: gate[k] for k in ("tie_margin_normalized", "min_visible_evidence_px",
                                                       "insufficient_evidence_rule", "layout_uniqueness")},
               "world_file": "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
               "policies": gate["policies"]}
    write_json(LOCK, payload)
    capture_sources = set(prior.LEGACY_CAPTURE_SOURCES) | {str(path) for path in own}
    compatibility = {"schema_version": 1, "protocol_id": PROTOCOL_ID,
                     "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE", "locked_at_utc": datetime.now(timezone.utc).isoformat(),
                     "parent_contract_lock": str(LOCK.relative_to(ROOT)), "parent_contract_lock_sha256": sha256(LOCK),
                     "purpose": "generic ROS capture compatibility adapter; geometry pilot only",
                     "source_artifact_sha256": {name: sha256(ROOT / name) for name in sorted(capture_sources)},
                     "model_inventory_sha256": "NO_MODEL_INFERENCE_GEOMETRY_PILOT_R5",
                     "policies_inherited": gate["policies"]}
    write_json(CAPTURE_LOCK, compatibility)
    print(json.dumps({"status": "LOCKED", "contract_sha256": sha256(LOCK), "capture_lock_sha256": sha256(CAPTURE_LOCK)}, indent=2))


def verify_locks():
    validate_contract()
    primary, capture = json.loads(LOCK.read_text()), json.loads(CAPTURE_LOCK.read_text())
    if primary.get("status") != "LOCKED_BEFORE_PILOT_CAPTURE" or capture.get("parent_contract_lock_sha256") != sha256(LOCK):
        raise ValueError("r5 lock linkage failed")
    for lock in (primary, capture):
        for name, digest in lock["source_artifact_sha256"].items():
            if not (ROOT / name).is_file() or sha256(ROOT / name) != digest: raise ValueError(f"locked source changed: {name}")
    return primary, capture


def preflight_static():
    scenes, annotations, gate = validate_contract(); verify_locks()
    report = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PASS",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "checks": {"deterministic_generator": True, "scene_count_32": len(scenes["scenes"]) == 32,
                         "annotation_count_32": len(annotations["scenes"]) == 32,
                         "unique_scene_family_layout_signatures": True, "no_prior_dev_or_uq_overlap": True,
                         "state_relation_quotas": True, "characterization_6_of_6": True,
                         "wp4_grounding_and_hypothesis_locks": True, "geometry_only_policy": gate["policies"]},
              "contract_lock_sha256": sha256(LOCK), "capture_compatibility_lock_sha256": sha256(CAPTURE_LOCK),
              "capture_authorized": True}
    write_json(STATIC_PREFLIGHT, report); print(json.dumps(report, indent=2))


def command_output(command):
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout


def preflight_live():
    verify_locks(); topics = command_output(["ros2", "topic", "list", "-t"]); services = command_output(["ros2", "service", "list", "-t"])
    required_topics = {"/wrist_camera/color/image_raw": "sensor_msgs/msg/Image",
                       "/wrist_camera/depth/image_raw": "sensor_msgs/msg/Image",
                       "/wrist_camera/evaluation_labels/labels_map": "sensor_msgs/msg/Image",
                       "/wrist_camera/color/camera_info": "sensor_msgs/msg/CameraInfo", "/tf": "tf2_msgs/msg/TFMessage",
                       "/joint_states": "sensor_msgs/msg/JointState"}
    required_services = {"/world/ur3_pick_place/set_pose": "ros_gz_interfaces/srv/SetEntityPose"}
    topic_checks = {name: f"{name} [{kind}]" in topics for name, kind in required_topics.items()}
    service_checks = {name: f"{name} [{kind}]" in services for name, kind in required_services.items()}
    free = shutil.disk_usage(ROOT).free / 2**30; passed = all(topic_checks.values()) and all(service_checks.values()) and free > 2
    report = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PASS" if passed else "BLOCKED",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "topic_checks": topic_checks,
              "service_checks": service_checks, "free_gib": free, "capture_authorized": passed,
              "contract_lock_sha256": sha256(LOCK), "capture_compatibility_lock_sha256": sha256(CAPTURE_LOCK)}
    write_json(LIVE_PREFLIGHT, report); print(json.dumps(report, indent=2))
    if not passed: raise SystemExit(2)


def rgb_audit(inputs, gate):
    images = []
    for record in inputs:
        path = CAPTURE / record["input_files"]["rgb"]
        images.append((record["scene_id"], sha256(path), cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)))
    groups = {}
    for sid, digest, _ in images: groups.setdefault(digest, []).append(sid)
    exact = [group for group in groups.values() if len(group) > 1]
    rule = gate["layout_uniqueness"]["perceptual_near_duplicate_rule"]
    near, pairs = [], []
    for left, right in itertools.combinations(images, 2):
        diff = cv2.absdiff(left[2], right[2]); mad = float(np.mean(diff)); changed = float(np.mean(diff >= int(rule["gray_absdiff_threshold_for_changed_pixel"])))
        item = {"left": left[0], "right": right[0], "gray_mad": mad, "changed_pixel_fraction": changed}; pairs.append(item)
        if mad < float(rule["near_duplicate_if_gray_mad_below"]) and changed < float(rule["and_changed_pixel_fraction_below"]): near.append(item)
    return {"status": "PASS" if not exact and not near else "FAIL", "rgb_count": len(images),
            "unique_rgb_sha256": len(groups), "exact_duplicate_groups": exact,
            "perceptual_near_duplicate_pairs": near,
            "closest_pairs": sorted(pairs, key=lambda x: (x["gray_mad"], x["changed_pixel_fraction"]))[:10], "rule": rule}


def geometry_qc():
    verify_locks(); scenes, annotations, gate = map(load_yaml, (SCENES, ANNOTATIONS, GATE))
    manifest = json.loads((CAPTURE / "capture_manifest.json").read_text()); inputs = load_jsonl(CAPTURE / "input_manifest.jsonl")
    if manifest.get("status") != "COMPLETE" or manifest.get("protocol_id") != PROTOCOL_ID or len(inputs) != 32:
        raise ValueError("capture incomplete")
    expected = {"scene_config_sha256": sha256(SCENES), "annotation_file_sha256": sha256(ANNOTATIONS),
                "gate_config_file_sha256": sha256(GATE), "pretrial_source_lock_sha256": sha256(CAPTURE_LOCK),
                "input_manifest_sha256": sha256(CAPTURE / "input_manifest.jsonl")}
    if any(manifest.get(k) != v for k, v in expected.items()): raise ValueError("capture provenance mismatch")
    if manifest.get("model_inventory_sha256_preregistered") != "NO_MODEL_INFERENCE_GEOMETRY_PILOT_R5": raise ValueError("model-inventory token mismatch")
    scene_map = {row["scene_id"]: row for row in scenes["scenes"]}; oracle = annotations["scenes"]
    audit = rgb_audit(inputs, gate); qc_rows = []
    for record in inputs:
        sid = record["scene_id"]; item = oracle[sid]; scene = scene_map[sid]; reasons = []
        label_map = prior.labels(CAPTURE / sid / "evaluator/semantic_labels.png")
        candidates = [prior.geometry(label_map, label) for label in item.get("candidate_labels", [])]
        context = [prior.geometry(label_map, label) for label in item.get("context_labels", [])]
        target = prior.geometry(label_map, item["target_label"]) if item.get("target_label") is not None else None
        state = item["state"]; minimum = int(gate["min_visible_evidence_px"]); tie = float(gate["tie_margin_normalized"]); details = {}; verified = False
        if state == "FOUND":
            visible = all(c["visible_pixels"] >= minimum for c in candidates)
            order = sorted(candidates, key=lambda c: c["centroid_normalized_xy"][0]) if visible else []
            if item["rank_from"] == "right": order.reverse()
            gaps = [abs(order[i]["centroid_normalized_xy"][0] - order[i+1]["centroid_normalized_xy"][0]) for i in range(max(0, len(order)-1))]
            verified = bool(visible and item["rank"] <= len(order) and order[item["rank"]-1]["semantic_label"] == item["target_label"] and all(g > tie for g in gaps))
            details = {"ordered_semantic_labels": [c["semantic_label"] for c in order], "adjacent_normalized_x_gaps": gaps}
        elif state == "AMBIGUOUS":
            visible = all(c["visible_pixels"] >= minimum for c in candidates)
            valid = [prior.geometry(label_map, label) for label in item["valid_target_labels"]]
            order = sorted(candidates, key=lambda c: c["centroid_normalized_xy"][0]) if visible else []
            if item["rank_from"] == "right": order.reverse()
            positions = [i for i,c in enumerate(order) if c["semantic_label"] in set(item["valid_target_labels"])]
            xs = [c["centroid_normalized_xy"][0] for c in valid if c["centroid_normalized_xy"] is not None]
            gap = max(xs)-min(xs) if len(xs)==2 else None
            verified = bool(visible and gap is not None and gap <= tie and item["rank"]-1 in positions)
            details = {"rendered_tie_gap_normalized_x": gap, "tied_positions_zero_based": positions,
                       "ordered_semantic_labels": [c["semantic_label"] for c in order]}
        elif state == "ABSENT":
            verified = bool(target and target["visible_pixels"] == 0 and item["target_id"] not in scene["poses"] and all(c["visible_pixels"] >= minimum for c in context))
            details = {"target_visible_pixels": target["visible_pixels"] if target else None,
                       "context_visible_pixels": [c["visible_pixels"] for c in context]}
        else:
            camera = json.loads((CAPTURE / record["input_files"]["camera_info"]).read_text())
            transforms = json.loads((CAPTURE / record["input_files"]["tf_snapshot"]).read_text()); transform = transforms.get("camera_color_optical_frame", {})
            capture_oracle = json.loads((CAPTURE / sid / "evaluator/capture_oracle.json").read_text())
            projected = prior.project_base_point(capture_oracle["requested_scene_layout_base_link"][item["target_id"]][:3], transform, camera)
            pixel = projected.get("pixel_xy"); outer = float(gate["insufficient_evidence_rule"]["projected_center_outer_margin_fraction"])
            near_sensor = bool(pixel and -640*outer <= pixel[0] < 640*(1+outer) and -480*outer <= pixel[1] < 480*(1+outer) and 0.1 <= projected["camera_xyz"][2] <= 2.0)
            others = [c for c in candidates if c["semantic_label"] != item["target_label"]]
            verified = bool(target and 1 <= target["visible_pixels"] < 120 and near_sensor and all(c["visible_pixels"] >= minimum for c in others))
            details = {"target_visible_pixels": target["visible_pixels"] if target else None,
                       "other_visible_pixels": [c["visible_pixels"] for c in others],
                       "target_center_projection_from_geometry": projected, "near_sensor_fov": near_sensor}
        if not verified: reasons.append(f"REQUESTED_STATE_NOT_VERIFIED:{state}")
        qc_rows.append({"scene_id": sid, "family_id": item["family_id"], "layout_id": item["layout_id"],
                        "layout_signature_sha256": item["layout_signature_sha256"], "requested_state": state,
                        "relation_variant": item["relation_variant"], "state_verified": verified, "reasons": reasons,
                        "target_visible_pixels": target["visible_pixels"] if target else 0, "geometry": details})
    safety = {"unique_parent_families_32": len({r["family_id"] for r in qc_rows}) == 32,
              "input_manifest_oracle_free": all(not prior.forbidden_input_paths(record) for record in inputs),
              "semantic_labels_evaluator_only": manifest.get("semantic_labels_are_evaluator_only") is True,
              "no_model_inference": manifest.get("model_inventory_sha256_preregistered") == "NO_MODEL_INFERENCE_GEOMETRY_PILOT_R5",
              "no_target_handoff": manifest.get("target_handoff_published") is False,
              "no_robot_manipulation": manifest.get("robot_manipulation_performed") is False}
    passed = audit["status"] == "PASS" and all(not row["reasons"] for row in qc_rows) and all(safety.values())
    report = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PASS" if passed else "REJECT",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "contract_lock_sha256": sha256(LOCK),
              "capture_compatibility_lock_sha256": sha256(CAPTURE_LOCK),
              "capture_manifest_sha256": sha256(CAPTURE / "capture_manifest.json"),
              "capture_input_manifest_sha256": sha256(CAPTURE / "input_manifest.jsonl"), "records": len(qc_rows),
              "state_counts": dict(Counter(r["requested_state"] for r in qc_rows)),
              "verified_state_counts": dict(Counter(r["requested_state"] for r in qc_rows if r["state_verified"])),
              "relation_variant_counts": dict(Counter(r["relation_variant"] for r in qc_rows)),
              "failed_scene_count": sum(bool(r["reasons"]) for r in qc_rows), "rgb_duplicate_audit": audit,
              "leakage_and_safety_checks": safety, "full_train_uq_design_authorized": passed,
              "model_inference_performed": False, "dataset_materialized": False, "scenes": qc_rows}
    write_json(QC_REPORT, report)
    failed = [row for row in qc_rows if row["reasons"]]; decision = "PASS" if passed else "REJECT"
    DECISION.write_text(f"# Gazebo_train_uq_v2_pilot_r5 — capture attempt 01 decision\n\nDecision: **{decision}**\n\n"
                        f"- Contract lock: `{sha256(LOCK)}`\n- Capture compatibility lock: `{sha256(CAPTURE_LOCK)}`\n"
                        f"- Capture manifest: `{sha256(CAPTURE / 'capture_manifest.json')}`\n- Geometry QC report: `{sha256(QC_REPORT)}`\n"
                        f"- Captured rows: `{len(qc_rows)}/32`\n- Geometry-verified rows: `{sum(r['state_verified'] for r in qc_rows)}/32`\n"
                        f"- Exact RGB duplicates: `{len(audit['exact_duplicate_groups'])}`\n"
                        f"- Perceptual near-duplicate pairs: `{len(audit['perceptual_near_duplicate_pairs'])}`\n"
                        f"- Failed scenes: `{len(failed)}`\n\nThis was geometry-only; no inference, materialization, training, calibration, target handoff, manipulation, B2, or Test access occurred.\n\n"
                        + ("A new locked Train-UQ/Val-UQ protocol is authorized; this pilot is never training data.\n" if passed else
                           "Full Train-UQ/Val-UQ remains blocked; preserve this attempt and create only an append-only revision.\n"), encoding="utf-8")
    print(json.dumps({"status": decision, "geometry_verified": sum(r["state_verified"] for r in qc_rows),
                      "failed_scenes": len(failed), "exact_duplicate_groups": len(audit["exact_duplicate_groups"]),
                      "perceptual_near_duplicates": len(audit["perceptual_near_duplicate_pairs"])}, indent=2))
    if not passed: raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("validate-contract", "lock", "preflight-static", "preflight-live", "geometry-qc"))
    command = parser.parse_args().command
    if command == "validate-contract": validate_contract(); print("PASS: Gazebo Train-UQ v2 pilot r5 contract")
    elif command == "lock": lock_contract()
    elif command == "preflight-static": preflight_static()
    elif command == "preflight-live": preflight_live()
    else: geometry_qc()


if __name__ == "__main__":
    main()
