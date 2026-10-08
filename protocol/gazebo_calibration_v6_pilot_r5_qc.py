#!/usr/bin/env python3
"""All-or-nothing QC for the append-only Calibration-v6 R5 pilot."""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

import gazebo_calibration_v6_qc_r4 as r4_qc
import generate_gazebo_calibration_v6_design as v6
import materialize_gazebo_train_uq_v1 as frozen


ROOT = v6.ROOT
PID = "gazebo_calibration_v6_geometry_pilot_r5"
OUTPUT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot_r5"
CAPTURE = OUTPUT / "capture_attempt_01"
AUTH = ROOT / "protocol/gazebo_calibration_v6_geometry_pilot_r5_capture_compatibility_lock.json"
IMPLEMENTATION = ROOT / "protocol/GAZEBO_CALIBRATION_V6_PILOT_R5_IMPLEMENTATION_LOCK.json"
REGISTRY = OUTPUT / "R5_EXCLUSION_REGISTRY.json"
SCENES = ROOT / f"ur3/ur3_perception/config/{PID}_scenes.yaml"
ANNOTATIONS = ROOT / f"ur3/ur3_perception/config/{PID}_annotations.yaml"
GATE = ROOT / f"ur3/ur3_perception/config/{PID}_gate.yaml"
FAMILY_MANIFEST = ROOT / f"protocol/{PID}_family_manifest.jsonl"
OLD_R3 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/capture_attempt_01/input_manifest.jsonl"
OLD_R4 = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot_r4/capture_attempt_01/input_manifest.jsonl"
REPORT = OUTPUT / "GEOMETRY_QC.json"
RGB_REPORT = OUTPUT / "RGB_DUPLICATE_QC.json"
LEAKAGE_REPORT = OUTPUT / "MANIFEST_LEAKAGE_QC.json"
INPUT_LOCK = OUTPUT / "QC_INPUT_LOCK.json"
DECISION = OUTPUT / "PILOT_R5_DECISION.json"
FAILURE = OUTPUT / "PILOT_QC_FAILURE_LOCK.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_lock(path: Path, expected_status: str) -> dict[str, Any]:
    value = read_json(path)
    if value.get("status") != expected_status:
        raise RuntimeError(f"unexpected lock status: {path}")
    for name, digest in value.get("source_artifact_sha256", {}).items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"locked source drift: {name}")
    return value


def validate_capture():
    verify_lock(IMPLEMENTATION, "PILOT_R5_IMPLEMENTATION_FROZEN_BEFORE_LIVE_PREFLIGHT_AND_CAPTURE")
    authorization = verify_lock(AUTH, "LOCKED_BEFORE_CAPTURE_AND_INFERENCE")
    scenes = yaml.safe_load(SCENES.read_text(encoding="utf-8"))
    annotations = yaml.safe_load(ANNOTATIONS.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    manifest = read_json(CAPTURE / "capture_manifest.json")
    records = read_jsonl(CAPTURE / "input_manifest.jsonl")
    expected_ids = {scene["scene_id"] for scene in scenes["scenes"]}
    checks = {
        "capture_complete": manifest.get("status") == "COMPLETE",
        "protocol_id": manifest.get("protocol_id") == PID,
        "r4_sensor_implementation_reused_unchanged": manifest.get("implementation_revision") == "r4_timestamp_queue_liveness",
        "record_count_32": len(records) == 32,
        "unique_scene_ids": len({row["scene_id"] for row in records}) == 32,
        "scene_id_set_exact": {row["scene_id"] for row in records} == expected_ids,
        "authorized_output_exact": authorization.get("authorized_output") == str(CAPTURE.relative_to(ROOT)),
        "pretrial_lock_bound": manifest.get("pretrial_source_lock_sha256") == sha256(AUTH),
        "scene_config_bound": manifest.get("scene_config_sha256") == sha256(SCENES),
        "annotation_bound": manifest.get("annotation_file_sha256") == sha256(ANNOTATIONS),
        "gate_bound": manifest.get("gate_config_file_sha256") == sha256(GATE),
        "input_manifest_bound": manifest.get("input_manifest_sha256") == sha256(CAPTURE / "input_manifest.jsonl"),
        "source_closure_verified": manifest.get("source_artifacts_verified_before_capture") is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"R5 capture/provenance invalid: {checks}")
    for record in records:
        sync = record.get("capture", {}).get("r4_synchronization", {})
        if (
            sync.get("matcher") != "newest_exact_header_timestamp_from_bounded_queues"
            or sync.get("post_settle_freshness_barrier") is not True
            or any(sync.get("decode_error_counts", {}).values())
            or record["capture"].get("rgb_depth_label_spread_sec") != 0.0
        ):
            raise RuntimeError(f"R5 exact synchronization failed: {record['scene_id']}")
        for name, relative in record["input_files"].items():
            if name in record.get("input_sha256", {}) and sha256(CAPTURE / relative) != record["input_sha256"][name]:
                raise RuntimeError(f"captured input drift: {record['scene_id']}/{name}")
    return scenes, annotations, gate, records


def manifest_leakage_qc() -> dict[str, Any]:
    rows = read_jsonl(FAMILY_MANIFEST)
    calibration = read_jsonl(v6.paths(v6.CALIBRATION)[3])
    registry = read_json(REGISTRY)
    fields = ("family_id", "scene_id", "layout_id", "deterministic_seed", "layout_signature_sha256")
    internal = {field: len({row[field] for row in rows}) == 32 for field in fields}
    prior = {
        field: sorted(set(row[field] for row in rows).intersection(registry["identifiers"][field]))
        for field in fields
    }
    calibration_overlap = {
        field: sorted(set(row[field] for row in rows).intersection(row[field] for row in calibration))
        for field in fields
    }
    cells = Counter((row["state"], row["relation_variant"]) for row in rows)
    checks = {
        "internal_unique": all(internal.values()),
        "zero_prior_overlap": all(not values for values in prior.values()),
        "zero_frozen_calibration_overlap": all(not values for values in calibration_overlap.values()),
        "four_by_four_by_two": len(cells) == 16 and set(cells.values()) == {2},
        "capture_order_0_to_31": sorted(row["capture_order"] for row in rows) == list(range(32)),
    }
    return {
        "status": "PASS" if all(checks.values()) else "REJECT",
        "checks": checks,
        "internal_unique": internal,
        "prior_overlap": prior,
        "frozen_calibration_overlap": calibration_overlap,
        "registry_sha256": sha256(REGISTRY),
        "family_manifest_sha256": sha256(FAMILY_MANIFEST),
    }


def gray_item(manifest: Path, row: dict[str, Any]):
    path = manifest.parent / row["input_files"]["rgb"]
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"invalid RGB: {path}")
    thumb = cv2.resize(image, (80, 60), interpolation=cv2.INTER_AREA)
    return f"{manifest.parent.relative_to(ROOT)}:{row['scene_id']}", sha256(path), image, thumb


def duplicate_qc(records: list[dict[str, Any]]) -> dict[str, Any]:
    base_lock = read_json(ROOT / "protocol/GAZEBO_CALIBRATION_V6_IMPLEMENTATION_LOCK.json")
    prior_manifests = [ROOT / name for name in base_lock["prior_rgb_manifest_sha256"]]
    prior_manifests.extend((OLD_R3, OLD_R4))
    current_manifest = CAPTURE / "input_manifest.jsonl"
    current = [gray_item(current_manifest, row) for row in records]
    exact: list[list[str]] = []
    near: list[dict[str, Any]] = []

    def compare(first, second) -> None:
        if first[1] == second[1]:
            exact.append([first[0], second[0]])
            return
        if first[2].shape != second[2].shape:
            return
        if float(cv2.absdiff(first[3], second[3]).mean()) >= 1.0:
            return
        difference = cv2.absdiff(first[2], second[2])
        mad = float(difference.mean())
        changed = float(np.mean(difference >= 3))
        if mad < 0.05 and changed < 0.002:
            near.append({"pair": [first[0], second[0]], "gray_mad": mad, "changed_fraction": changed})

    for first, second in itertools.combinations(current, 2):
        compare(first, second)
    prior_count = 0
    sources: dict[str, str] = {}
    for manifest in prior_manifests:
        sources[str(manifest.relative_to(ROOT))] = sha256(manifest)
        for row in read_jsonl(manifest):
            previous = gray_item(manifest, row)
            prior_count += 1
            for item in current:
                compare(item, previous)
    unique = len({item[1] for item in current})
    checks = {
        "32_unique_current_rgb_sha256": unique == 32,
        "zero_exact_duplicates": not exact,
        "zero_perceptual_near_duplicates": not near,
        "r3_r4_rows_used_only_for_exclusion": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "REJECT",
        "checks": checks,
        "rgb_count": 32,
        "prior_rgb_count": prior_count,
        "unique_rgb_sha256": unique,
        "exact_duplicates": exact,
        "perceptual_near_duplicates": near,
        "rule_unchanged_from_r4": {
            "gray_mad_lt": 0.05,
            "changed_fraction_lt": 0.002,
            "changed_pixel_absdiff_ge": 3,
            "thumbnail_prefilter_mad_lt": 1.0,
        },
        "source_manifest_sha256": sources,
    }


def run() -> None:
    if any(path.exists() for path in (REPORT, RGB_REPORT, LEAKAGE_REPORT, INPUT_LOCK, DECISION, FAILURE)):
        raise FileExistsError("R5 QC already attempted")
    scenes, annotations, gate, records = validate_capture()
    capture_sources = {str(path.relative_to(ROOT)): sha256(path) for path in CAPTURE.rglob("*") if path.is_file()}
    write_new(INPUT_LOCK, {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "LOCKED_BEFORE_QC",
        "source_artifact_sha256": capture_sources,
        "qc_code_sha256": sha256(Path(__file__)),
        "implementation_lock_sha256": sha256(IMPLEMENTATION),
    })
    predicate = r4_qc.state_predicate()
    scene_lookup = {scene["scene_id"]: scene for scene in scenes["scenes"]}
    min_pixels = int(gate["min_visible_evidence_px"])
    tie = float(gate["tie_margin_normalized"])
    rows: list[dict[str, Any]] = []
    for record in records:
        scene_id = record["scene_id"]
        item = annotations["scenes"][scene_id]
        scene = scene_lookup[scene_id]
        labels = frozen.labels(CAPTURE / scene_id / "evaluator/semantic_labels.png")
        camera = read_json(CAPTURE / record["input_files"]["camera_info"])
        tf = read_json(CAPTURE / record["input_files"]["tf_snapshot"])["camera_color_optical_frame"]
        depth = np.load(CAPTURE / record["input_files"]["depth_m"], allow_pickle=False)
        reasons = frozen.sensor_reasons(labels, depth, camera, tf)
        candidates = [frozen.geom(labels, label) for label in item.get("candidate_labels", [])]
        context = [frozen.geom(labels, label) for label in item.get("context_labels", [])]
        target = frozen.geom(labels, item["target_label"]) if item.get("target_label") is not None else None
        verified, projected = predicate(
            item["state"], item, scene, candidates, context, target, labels, tf, camera,
            gate, CAPTURE, scene_id, min_pixels, tie,
        )
        target_pixels = target["visible_pixels"] if target else 0
        if item["state"] == "INSUFFICIENT_EVIDENCE" and not (1 <= target_pixels < min_pixels):
            reasons.append("INSUFFICIENT_EVIDENCE_TARGET_PIXEL_GATE_FAILED")
        if item["state"] not in ("INSUFFICIENT_EVIDENCE", "ABSENT") and any(
            candidate["visible_pixels"] < min_pixels for candidate in candidates
        ):
            reasons.append("CANDIDATE_MINIMUM_VISIBLE_PIXEL_GATE_FAILED")
        if item["state"] == "ABSENT" and any(value["visible_pixels"] < min_pixels for value in context):
            reasons.append("ABSENT_CONTEXT_MINIMUM_VISIBLE_PIXEL_GATE_FAILED")
        if not verified:
            reasons.append("REQUESTED_STATE_NOT_VERIFIED:" + item["state"])
        rows.append({
            "scene_id": scene_id,
            "family_id": item["family_id"],
            "state": item["state"],
            "relation_variant": item["relation_variant"],
            "state_verified": bool(verified),
            "target_visible_pixels": target_pixels,
            "candidate_set": candidates,
            "context_set": context,
            "target": target,
            "projected_target": projected,
            "reasons": reasons,
        })
    rgb = duplicate_qc(records)
    leakage = manifest_leakage_qc()
    write_new(RGB_REPORT, rgb)
    write_new(LEAKAGE_REPORT, leakage)
    passed = all(not row["reasons"] for row in rows) and rgb["status"] == "PASS" and leakage["status"] == "PASS"
    report = {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "PASS" if passed else "BLOCKED",
        "records": len(rows),
        "passed_scene_count": sum(not row["reasons"] for row in rows),
        "verified_state_counts": dict(Counter(row["state"] for row in rows if not row["reasons"])),
        "visibility_contract": {"insufficient_target_pixels": "1 <= pixels < 120", "all_required_candidates_or_context": "pixels >= 120"},
        "scenes": rows,
        "rgb_qc_sha256": sha256(RGB_REPORT),
        "manifest_leakage_qc_sha256": sha256(LEAKAGE_REPORT),
        "input_lock_sha256": sha256(INPUT_LOCK),
        "dataset_materialized": False,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(REPORT, report)
    decision = {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "PASS" if passed else "REJECT",
        "decision": "PILOT_R5_32_OF_32_QC_PASS" if passed else "PILOT_R5_ATTEMPT_01_REJECTED_STOP_NO_RETRY",
        "classification": "GEOMETRY_PILOT_DATA_GATE_NOT_SCIENTIFIC_CALIBRATION_RESULT",
        "captured_family_count": len(rows),
        "passed_scene_count": report["passed_scene_count"],
        "accepted_family_count": 32 if passed else 0,
        "source_artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (INPUT_LOCK, REPORT, RGB_REPORT, LEAKAGE_REPORT)
        },
        "calibration_capture_authorized": False,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(DECISION, decision)
    if not passed:
        write_new(FAILURE, {
            "schema_version": 1,
            "protocol_id": PID,
            "status": "PILOT_QC_REJECTED_STOP_NO_RETRY_UNDER_R5",
            "classification": "DATA_GATE_FAILURE_NOT_SCIENTIFIC_CALIBRATION_RESULT",
            "source_artifact_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (INPUT_LOCK, REPORT, RGB_REPORT, LEAKAGE_REPORT, DECISION)
            },
            "failed_rows_reused": False,
            "model_inference_run": False,
            "calibrator_fit_call_count": 0,
            "scientific_hypothesis": "NOT_TESTED",
            "scientific_decision": None,
            "test_iid_ood_access": False,
            "robot_access": False,
        })
    print(json.dumps({
        "status": report["status"],
        "passed_scene_count": report["passed_scene_count"],
        "rgb_status": rgb["status"],
        "leakage_status": leakage["status"],
        "decision": decision["decision"],
    }, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    run()
