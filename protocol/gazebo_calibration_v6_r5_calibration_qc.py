#!/usr/bin/env python3
"""All-or-nothing QC and downstream authorization for Calibration-v6 R5."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

import gazebo_calibration_v6_pilot_r5_qc as pilot_qc
import gazebo_calibration_v6_qc_r4 as r4_qc
import gazebo_calibration_v6_r5_calibration as execution
import generate_gazebo_calibration_v6_design as v6
import materialize_gazebo_train_uq_v1 as frozen


ROOT = v6.ROOT
PID = v6.CALIBRATION
OUTPUT = execution.OUTPUT
CAPTURE = execution.CAPTURE
AUTH = execution.AUTH
EXECUTION_LOCK = execution.EXECUTION_LOCK
PREFLIGHT = execution.PREFLIGHT
SCENES, ANNOTATIONS, GATE, FAMILY_MANIFEST, SPLIT_MANIFEST = v6.paths(PID)
REPORT = OUTPUT / "GEOMETRY_QC.json"
RGB_REPORT = OUTPUT / "RGB_DUPLICATE_QC.json"
LEAKAGE_REPORT = OUTPUT / "MANIFEST_LEAKAGE_QC.json"
INPUT_LOCK = OUTPUT / "QC_INPUT_LOCK.json"
DECISION = OUTPUT / "CALIBRATION_R5_DECISION.json"
FAILURE = OUTPUT / "CALIBRATION_QC_FAILURE_LOCK.json"
DOWNSTREAM_AUTH = ROOT / "protocol/GAZEBO_CALIBRATION_V6_R5_ONE_SHOT_INFERENCE_FIT_AUTHORIZATION_LOCK.json"
DATASET = execution.DATASET
MATERIALIZATION_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V6_MATERIALIZATION_LOCK.json"
PILOT_MANIFESTS = (
    v6.paths(v6.PILOT)[3],
    ROOT / "protocol/gazebo_calibration_v6_geometry_pilot_r5_family_manifest.jsonl",
)
PRIOR_CAPTURE_MANIFESTS = (
    ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/capture_attempt_01/input_manifest.jsonl",
    ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot_r4/capture_attempt_01/input_manifest.jsonl",
    ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot_r5/capture_attempt_01/input_manifest.jsonl",
)


def sha256(path: Path) -> str:
    return execution.sha256(path)


def read_json(path: Path) -> dict[str, Any]:
    return execution.read_json(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return execution.read_jsonl(path)


def write_new(path: Path, value: dict[str, Any]) -> None:
    execution.write_new(path, value)


def verify_lock(path: Path, status: str) -> dict[str, Any]:
    value = read_json(path)
    if value.get("status") != status:
        raise RuntimeError(f"unexpected lock status: {path}")
    execution.verify_hashes(value.get("source_artifact_sha256", {}))
    return value


def validate_capture():
    execution.validate_execution()
    authorization = execution.verify_authorization()
    if read_json(PREFLIGHT).get("status") != "PASS":
        raise RuntimeError("Calibration-v6 R5 live preflight PASS required")
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
        "record_count_128": len(records) == 128,
        "unique_scene_ids": len({row["scene_id"] for row in records}) == 128,
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
        raise RuntimeError(f"Calibration-v6 R5 capture/provenance invalid: {checks}")
    for record in records:
        sync = record.get("capture", {}).get("r4_synchronization", {})
        if (
            sync.get("matcher") != "newest_exact_header_timestamp_from_bounded_queues"
            or sync.get("post_settle_freshness_barrier") is not True
            or any(sync.get("decode_error_counts", {}).values())
            or record["capture"].get("rgb_depth_label_spread_sec") != 0.0
        ):
            raise RuntimeError(f"Calibration-v6 R5 exact synchronization failed: {record['scene_id']}")
        for name, relative in record["input_files"].items():
            if name in record.get("input_sha256", {}) and sha256(CAPTURE / relative) != record["input_sha256"][name]:
                raise RuntimeError(f"captured input drift: {record['scene_id']}/{name}")
    return scenes, annotations, gate, records


def manifest_leakage_qc() -> dict[str, Any]:
    rows = read_jsonl(FAMILY_MANIFEST)
    registry = read_json(v6.REGISTRY)
    fields = ("family_id", "scene_id", "layout_id", "deterministic_seed", "layout_signature_sha256")
    internal = {field: len({row[field] for row in rows}) == 128 for field in fields}
    prior = {
        field: sorted(set(row[field] for row in rows).intersection(registry["identifiers"][field]))
        for field in fields
    }
    pilot_overlap: dict[str, dict[str, list[Any]]] = {}
    for manifest in PILOT_MANIFESTS:
        pilot = read_jsonl(manifest)
        pilot_overlap[str(manifest.relative_to(ROOT))] = {
            field: sorted(set(row[field] for row in rows).intersection(row[field] for row in pilot))
            for field in fields
        }
    cells = Counter((row["state"], row["relation_variant"]) for row in rows)
    scenes = yaml.safe_load(SCENES.read_text(encoding="utf-8"))["scenes"]
    expected_order = [row["scene_id"] for row in sorted(rows, key=lambda row: row["capture_order"])]
    checks = {
        "internal_unique": all(internal.values()),
        "zero_historical_registry_overlap": all(not values for values in prior.values()),
        "zero_pilot_r3_r4_r5_identifier_overlap": all(
            not values for overlap in pilot_overlap.values() for values in overlap.values()
        ),
        "four_by_four_by_eight": len(cells) == 16 and set(cells.values()) == {8},
        "capture_order_0_to_127": sorted(row["capture_order"] for row in rows) == list(range(128)),
        "scene_order_matches_manifest": [scene["scene_id"] for scene in scenes] == expected_order,
    }
    return {
        "status": "PASS" if all(checks.values()) else "REJECT",
        "checks": checks,
        "internal_unique": internal,
        "historical_registry_overlap": prior,
        "pilot_manifest_overlap": pilot_overlap,
        "registry_sha256": sha256(v6.REGISTRY),
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
    prior_manifests.extend(PRIOR_CAPTURE_MANIFESTS)
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
    seen: set[Path] = set()
    for manifest in prior_manifests:
        if manifest in seen:
            continue
        seen.add(manifest)
        sources[str(manifest.relative_to(ROOT))] = sha256(manifest)
        for row in read_jsonl(manifest):
            previous = gray_item(manifest, row)
            prior_count += 1
            for item in current:
                compare(item, previous)
    unique = len({item[1] for item in current})
    checks = {
        "128_unique_current_rgb_sha256": unique == 128,
        "zero_exact_duplicates": not exact,
        "zero_perceptual_near_duplicates": not near,
        "r3_r4_r5_pilot_rows_used_only_for_exclusion": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "REJECT",
        "checks": checks,
        "rgb_count": len(current),
        "prior_rgb_count": prior_count,
        "unique_rgb_sha256": unique,
        "exact_duplicates": exact,
        "perceptual_near_duplicates": near,
        "rule_unchanged_from_r4_r5": {
            "gray_mad_lt": 0.05,
            "changed_fraction_lt": 0.002,
            "changed_pixel_absdiff_ge": 3,
            "thumbnail_prefilter_mad_lt": 1.0,
        },
        "source_manifest_sha256": sources,
    }


def run_qc() -> None:
    artifacts = (REPORT, RGB_REPORT, LEAKAGE_REPORT, INPUT_LOCK, DECISION, FAILURE)
    if any(path.exists() for path in artifacts):
        raise FileExistsError("Calibration-v6 R5 QC already attempted")
    scenes, annotations, gate, records = validate_capture()
    capture_sources = {str(path.relative_to(ROOT)): sha256(path) for path in CAPTURE.rglob("*") if path.is_file()}
    write_new(INPUT_LOCK, {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "LOCKED_BEFORE_QC",
        "source_artifact_sha256": capture_sources,
        "qc_code_sha256": sha256(Path(__file__)),
        "execution_lock_sha256": sha256(EXECUTION_LOCK),
        "preflight_sha256": sha256(PREFLIGHT),
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
    passed = (
        len(rows) == 128
        and all(not row["reasons"] for row in rows)
        and rgb["status"] == "PASS"
        and leakage["status"] == "PASS"
    )
    report = {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "PASS" if passed else "BLOCKED",
        "records": len(rows),
        "passed_scene_count": sum(not row["reasons"] for row in rows),
        "verified_state_counts": dict(Counter(row["state"] for row in rows if not row["reasons"])),
        "visibility_contract": {
            "insufficient_target_pixels": "1 <= pixels < 120",
            "all_required_candidates_or_context": "pixels >= 120",
        },
        "scenes": rows,
        "rgb_qc_sha256": sha256(RGB_REPORT),
        "manifest_leakage_qc_sha256": sha256(LEAKAGE_REPORT),
        "input_lock_sha256": sha256(INPUT_LOCK),
        "dataset_materialized": False,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(REPORT, report)
    decision = {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "PASS" if passed else "REJECT",
        "decision": "CALIBRATION_R5_128_OF_128_QC_PASS" if passed else "CALIBRATION_R5_ATTEMPT_01_REJECTED_STOP_NO_RETRY",
        "classification": "CALIBRATION_DATA_GATE_NOT_YET_SCIENTIFIC_RESULT",
        "captured_family_count": len(rows),
        "passed_scene_count": report["passed_scene_count"],
        "accepted_family_count": 128 if passed else 0,
        "source_artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (INPUT_LOCK, REPORT, RGB_REPORT, LEAKAGE_REPORT)
        },
        "downstream_inference_fit_authorized": False,
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
            "status": "CALIBRATION_R5_QC_REJECTED_STOP_NO_RETRY",
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
        "records": report["records"],
        "passed_scene_count": report["passed_scene_count"],
        "rgb_status": rgb["status"],
        "leakage_status": leakage["status"],
        "decision": decision["decision"],
    }, indent=2))
    if not passed:
        raise SystemExit(2)


def authorize_downstream() -> None:
    if DOWNSTREAM_AUTH.exists() or MATERIALIZATION_LOCK.exists() or DATASET.exists():
        raise RuntimeError("downstream authorization/materialization already exists")
    execution.validate_execution()
    decision = read_json(DECISION)
    if (
        decision.get("decision") != "CALIBRATION_R5_128_OF_128_QC_PASS"
        or decision.get("accepted_family_count") != 128
        or any(read_json(path).get("status") != "PASS" for path in (REPORT, RGB_REPORT, LEAKAGE_REPORT))
    ):
        raise RuntimeError("Calibration-v6 R5 128/128 all-QC PASS required")
    if any(path.exists() for path in execution.INFERENCE_OUTPUTS):
        raise RuntimeError("materialization, inference or fit output already exists")
    sources = [
        Path(__file__).resolve(), EXECUTION_LOCK, AUTH, PREFLIGHT,
        INPUT_LOCK, REPORT, RGB_REPORT, LEAKAGE_REPORT, DECISION,
        ROOT / "protocol/gazebo_calibration_v6_infer.py",
        ROOT / "protocol/gazebo_calibration_v6_fit.py",
        ROOT / "protocol/spatial_risk_v2_development.py",
        ROOT / "results/spatial_vlm_refspatial_v1/wp5_spatial_risk_v2_development/spatial_risk_v2_estimator.joblib",
    ]
    payload = {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "LOCKED_AFTER_CALIBRATION_R5_128_OF_128_QC_PASS_BEFORE_ONE_SHOT_INFERENCE_AND_FIT",
        "authorization_status": "ONE_MATERIALIZATION_ONE_B0_INFERENCE_ONE_SPATIAL_RISK_V2_INFERENCE_ONE_CALIBRATOR_FIT_AUTHORIZED",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_r5_decision_sha256": sha256(DECISION),
        "source_artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "execution_order": [
            "materialize blinded inference inputs after 128/128 QC PASS",
            "run and hash-lock B0 raw predictions exactly once",
            "run and hash-lock frozen 12-feature spatial-risk v2 predictions exactly once",
            "lock fit inputs before oracle join",
            "fit the preregistered affine-logit calibrator exactly once and apply the scientific gate",
        ],
        "attempts_authorized": {
            "materialization": 1,
            "b0_inference": 1,
            "spatial_risk_v2_inference": 1,
            "calibrator_fit": 1,
        },
        "policies": {
            "no_row_filter_repair_retry_or_cross_attempt_merge": True,
            "no_test_iid_ood_access": True,
            "no_robot_policy_access": True,
            "scientific_pass_or_negative_must_follow_frozen_gate": True,
        },
        "materialized": False,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(DOWNSTREAM_AUTH, payload)
    print(json.dumps({"status": payload["authorization_status"], "sha256": sha256(DOWNSTREAM_AUTH)}, indent=2))


def verify_downstream_authorization() -> dict[str, Any]:
    value = verify_lock(
        DOWNSTREAM_AUTH,
        "LOCKED_AFTER_CALIBRATION_R5_128_OF_128_QC_PASS_BEFORE_ONE_SHOT_INFERENCE_AND_FIT",
    )
    if not all(value["policies"].values()):
        raise RuntimeError("downstream authorization policy drift")
    return value


def materialize() -> None:
    verify_downstream_authorization()
    if DATASET.exists() or MATERIALIZATION_LOCK.exists():
        raise FileExistsError("Calibration-v6 materialization already attempted")
    scenes, annotations, gate, records = validate_capture()
    report = read_json(REPORT)
    if report.get("status") != "PASS" or report.get("passed_scene_count") != 128:
        raise RuntimeError("Calibration-v6 R5 128/128 QC PASS required")
    locked = read_json(INPUT_LOCK)
    execution.verify_hashes(locked["source_artifact_sha256"])
    DATASET.mkdir()
    by_id = {row["scene_id"]: row for row in report["scenes"]}
    inference_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for record in records:
        scene_id = record["scene_id"]
        annotation = annotations["scenes"][scene_id]
        row = by_id[scene_id]
        destination = DATASET / "records" / scene_id / "input"
        destination.mkdir(parents=True)
        for name in ("rgb", "depth_m", "camera_info", "tf_snapshot"):
            source = CAPTURE / record["input_files"][name]
            os.link(source, destination / source.name)
        depth = np.load(CAPTURE / record["input_files"]["depth_m"], allow_pickle=False)
        if not cv2.imwrite(str(destination / "depth_view.png"), frozen.depth_view(depth)):
            raise RuntimeError("depth-view write failed")
        common = {
            "sample_id": scene_id,
            "scene_id": scene_id,
            "family_id": annotation["family_id"],
            "split": "calibration",
            "relation_variant": annotation["relation_variant"],
            "relation": "horizontal_ordinal_ranking_answerability",
            "reference_frame": gate["reference_frame"],
        }
        inference_rows.append({
            **common,
            "image": f"records/{scene_id}/input/rgb.png",
            "depth": f"records/{scene_id}/input/depth_view.png",
            "metric_depth": f"records/{scene_id}/input/depth_m.npy",
            "instruction": record["instruction"] + " " + record["coordinate_suffix"],
        })
        truth_rows.append({
            **common,
            "answerability_state": annotation["state"],
            "answerability_verified": True,
            "target_id": annotation.get("target_id"),
            "target_xy": row["target"]["centroid_normalized_xy"] if annotation["state"] == "FOUND" else None,
        })
    for name, rows in (("inference_manifest.jsonl", inference_rows), ("evaluator_ground_truth.jsonl", truth_rows)):
        with (DATASET / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_new(DATASET / "manifest.json", {
        "schema_version": 1,
        "status": "PASS",
        "protocol_id": PID,
        "records": 128,
        "parent_families": 128,
        "capture_revision": "R5",
        "qc_report_sha256": sha256(REPORT),
        "downstream_authorization_sha256": sha256(DOWNSTREAM_AUTH),
        "test_iid_ood_access": False,
    })
    paths = [
        DOWNSTREAM_AUTH, EXECUTION_LOCK, AUTH, PREFLIGHT, INPUT_LOCK,
        REPORT, RGB_REPORT, LEAKAGE_REPORT, DECISION,
        *[path for path in DATASET.rglob("*") if path.is_file()],
    ]
    write_new(MATERIALIZATION_LOCK, {
        "schema_version": 1,
        "status": "MATERIALIZATION_FROZEN_128_OF_128_QC_PASS",
        "protocol_id": PID,
        "source_artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "test_iid_ood_access": False,
        "robot_access": False,
    })
    print(json.dumps({"status": "MATERIALIZATION_FROZEN_128_OF_128_QC_PASS", "sha256": sha256(MATERIALIZATION_LOCK)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("qc", "authorize-downstream", "materialize"))
    command = parser.parse_args().command
    if command == "qc":
        run_qc()
    elif command == "authorize-downstream":
        authorize_downstream()
    else:
        materialize()


if __name__ == "__main__":
    main()
