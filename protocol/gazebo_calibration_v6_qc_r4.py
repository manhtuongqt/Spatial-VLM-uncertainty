#!/usr/bin/env python3
"""R4 capture QC and all-or-nothing Calibration-v6 materialization."""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
import itertools
import json
import os
from pathlib import Path
import textwrap
from typing import Any

import cv2
import numpy as np
import yaml

import generate_gazebo_calibration_v6_design as design
import materialize_gazebo_train_uq_v1 as frozen


ROOT = design.ROOT
PILOT_ROOT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot_r4"
CALIBRATION_ROOT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_r4"
PILOT_CAPTURE = PILOT_ROOT / "capture_attempt_01"
CALIBRATION_CAPTURE = CALIBRATION_ROOT / "capture_attempt_01"
PILOT_AUTH = ROOT / "protocol/gazebo_calibration_v6_geometry_pilot_r4_capture_compatibility_lock.json"
CALIBRATION_AUTH = ROOT / "protocol/gazebo_calibration_v6_r4_capture_compatibility_lock.json"
R4_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V6_IMPLEMENTATION_AMENDMENT_R4_LOCK.json"
PILOT_DECISION = PILOT_ROOT / "PILOT_R4_DECISION.json"
OLD_PARTIAL = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/capture_attempt_01/input_manifest.jsonl"
DATASET = ROOT / "datasets/Gazebo_calibration_v6"
MATERIALIZATION_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V6_MATERIALIZATION_LOCK.json"


def sha(path: Path) -> str:
    return design.sha256(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_new(path: Path, value: dict[str, Any]) -> None:
    design.write_new(path, value)


def state_predicate():
    source = inspect.getsource(frozen.materialize)
    start = source.index('        if state == "FOUND":')
    stop = source.index("        if not verified: reasons.append", start)
    body = textwrap.dedent(source[start:stop])
    wrapped = "def predicate(state,item,scene,candidates,context,target,label_map,tf,camera,gate,CAPTURE,sid,min_px,tie):\n    projected=None\n"
    wrapped += textwrap.indent(body, "    ") + "\n    return verified,projected\n"
    env = dict(frozen.__dict__)
    exec(compile(wrapped, "<frozen_state_predicate>", "exec"), env)
    return env["predicate"]


def paths(pid: str) -> tuple[Path, Path, Path]:
    if pid == design.PILOT:
        return PILOT_ROOT, PILOT_CAPTURE, PILOT_AUTH
    return CALIBRATION_ROOT, CALIBRATION_CAPTURE, CALIBRATION_AUTH


def verify_lock(path: Path, status: str) -> dict[str, Any]:
    value = read_json(path)
    if value.get("status") != status:
        raise RuntimeError(f"unexpected lock status: {path}")
    for name, digest in value.get("source_artifact_sha256", {}).items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"locked source drift: {name}")
    return value


def validate_capture(pid: str):
    output, capture, authorization = paths(pid)
    verify_lock(R4_LOCK, "IMPLEMENTATION_AMENDMENT_R4_FROZEN_BEFORE_SENSOR_PREFLIGHT")
    auth = verify_lock(authorization, "LOCKED_BEFORE_CAPTURE_AND_INFERENCE")
    scenes, annotations, gate = (
        yaml.safe_load(path.read_text(encoding="utf-8")) for path in design.paths(pid)[:3]
    )
    manifest = read_json(capture / "capture_manifest.json")
    records = read_jsonl(capture / "input_manifest.jsonl")
    expected_count = 32 if pid == design.PILOT else 128
    expected_ids = {scene["scene_id"] for scene in scenes["scenes"]}
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("protocol_id") != pid
        or manifest.get("implementation_revision") != "r4_timestamp_queue_liveness"
        or len(records) != expected_count
        or {row["scene_id"] for row in records} != expected_ids
        or len({row["scene_id"] for row in records}) != expected_count
    ):
        raise RuntimeError("R4 capture incomplete, duplicated, or ID mismatched")
    expected = {
        "pretrial_source_lock_sha256": sha(authorization),
        "scene_config_sha256": sha(design.paths(pid)[0]),
        "annotation_file_sha256": sha(design.paths(pid)[1]),
        "gate_config_file_sha256": sha(design.paths(pid)[2]),
        "input_manifest_sha256": sha(capture / "input_manifest.jsonl"),
    }
    if any(manifest.get(name) != digest for name, digest in expected.items()):
        raise RuntimeError("R4 capture provenance mismatch")
    if manifest.get("source_artifacts_verified_before_capture") is not True:
        raise RuntimeError("R4 capture source closure was not verified")
    if auth.get("authorized_output") != str(capture.relative_to(ROOT)):
        raise RuntimeError("capture output differs from authorization")
    for record in records:
        sync = record.get("capture", {}).get("r4_synchronization", {})
        if (
            sync.get("matcher") != "newest_exact_header_timestamp_from_bounded_queues"
            or sync.get("post_settle_freshness_barrier") is not True
            or any(sync.get("decode_error_counts", {}).values())
            or record["capture"].get("rgb_depth_label_spread_sec") != 0.0
        ):
            raise RuntimeError(f"R4 synchronization contract failed: {record['scene_id']}")
        for name, relative in record["input_files"].items():
            if name in record.get("input_sha256", {}) and sha(capture / relative) != record["input_sha256"][name]:
                raise RuntimeError(f"captured input changed: {record['scene_id']}/{name}")
    return output, capture, scenes, annotations, gate, records


def manifest_leakage_qc(pid: str) -> dict[str, Any]:
    rows = read_jsonl(design.paths(pid)[3])
    registry = read_json(design.REGISTRY)
    fields = ("family_id", "scene_id", "layout_id", "deterministic_seed", "layout_signature_sha256")
    internal_unique = {field: len({row[field] for row in rows}) == len(rows) for field in fields}
    prior_overlap = {
        field: sorted(set(row[field] for row in rows).intersection(registry["identifiers"][field]))
        for field in fields
    }
    cross_split: dict[str, list[Any]] = {field: [] for field in fields}
    if pid == design.CALIBRATION:
        pilot = read_jsonl(design.paths(design.PILOT)[3])
        cross_split = {
            field: sorted(set(row[field] for row in rows).intersection(row[field] for row in pilot))
            for field in fields
        }
    passed = all(internal_unique.values()) and all(not values for values in prior_overlap.values()) and all(not values for values in cross_split.values())
    return {
        "status": "PASS" if passed else "REJECT",
        "row_count": len(rows),
        "internal_unique": internal_unique,
        "prior_exclusion_registry_overlap": prior_overlap,
        "pilot_calibration_cross_split_overlap": cross_split,
        "registry_sha256": sha(design.REGISTRY),
        "family_manifest_sha256": sha(design.paths(pid)[3]),
    }


def gray_item(manifest: Path, row: dict[str, Any]) -> tuple[str, str, np.ndarray, np.ndarray]:
    path = manifest.parent / row["input_files"]["rgb"]
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"invalid RGB: {path}")
    thumb = cv2.resize(image, (80, 60), interpolation=cv2.INTER_AREA)
    return f"{manifest.parent.relative_to(ROOT)}:{row['scene_id']}", sha(path), image, thumb


def duplicate_qc(pid: str, capture: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    implementation = read_json(ROOT / "protocol/GAZEBO_CALIBRATION_V6_IMPLEMENTATION_LOCK.json")
    roots = [ROOT / name for name in implementation["prior_rgb_manifest_sha256"]]
    roots.append(OLD_PARTIAL)
    if pid == design.CALIBRATION:
        roots.append(PILOT_CAPTURE / "input_manifest.jsonl")
    current_manifest = capture / "input_manifest.jsonl"
    current = [gray_item(current_manifest, row) for row in records]
    exact: list[list[str]] = []
    near: list[dict[str, Any]] = []

    def compare(a, b) -> None:
        if a[1] == b[1]:
            exact.append([a[0], b[0]])
            return
        if a[2].shape != b[2].shape:
            return
        # A permissive thumbnail prefilter reduces the ~370k full-frame
        # comparisons while retaining every candidate capable of satisfying
        # the preregistered full-frame near-duplicate rule below.
        if float(cv2.absdiff(a[3], b[3]).mean()) >= 1.0:
            return
        difference = cv2.absdiff(a[2], b[2])
        mad = float(difference.mean())
        changed = float(np.mean(difference >= 3))
        if mad < 0.05 and changed < 0.002:
            near.append({"pair": [a[0], b[0]], "gray_mad": mad, "changed_fraction": changed})

    for first, second in itertools.combinations(current, 2):
        compare(first, second)
    prior_count = 0
    source_hashes: dict[str, str] = {}
    for manifest in roots:
        source_hashes[str(manifest.relative_to(ROOT))] = sha(manifest)
        for row in read_jsonl(manifest):
            previous = gray_item(manifest, row)
            prior_count += 1
            for item in current:
                compare(item, previous)
    unique = len({item[1] for item in current})
    return {
        "status": "PASS" if not exact and not near and unique == len(current) else "REJECT",
        "rgb_count": len(current),
        "prior_rgb_count": prior_count,
        "unique_rgb_sha256": unique,
        "exact_duplicates": exact,
        "perceptual_near_duplicates": near,
        "rule": {"gray_mad_lt": 0.05, "changed_fraction_lt": 0.002, "changed_pixel_absdiff_ge": 3, "thumbnail_prefilter_mad_lt": 1.0},
        "r3_partial_attempt_used_as_data": False,
        "source_manifest_sha256": source_hashes,
    }


def failure_lock(output: Path, pid: str, artifacts: list[Path], stage: str) -> None:
    write_new(output / f"{stage}_FAILURE_LOCK.json", {
        "schema_version": 1,
        "protocol_id": pid,
        "status": f"{stage}_REJECTED_STOP_NO_RETRY_UNDER_R4",
        "classification": "DATA_GATE_FAILURE_NOT_SCIENTIFIC_CALIBRATION_RESULT",
        "source_artifact_sha256": {str(path.relative_to(ROOT)): sha(path) for path in artifacts},
        "partial_or_failed_rows_reused": False,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    })


def qc(pid: str) -> None:
    output, capture, scenes, annotations, gate, records = validate_capture(pid)
    report_path = output / "GEOMETRY_QC.json"
    duplicate_path = output / "RGB_DUPLICATE_QC.json"
    leakage_path = output / "MANIFEST_LEAKAGE_QC.json"
    input_lock = output / "QC_INPUT_LOCK.json"
    if any(path.exists() for path in (report_path, duplicate_path, leakage_path, input_lock)):
        raise FileExistsError("R4 QC already attempted")
    capture_sources = {str(path.relative_to(ROOT)): sha(path) for path in capture.rglob("*") if path.is_file()}
    write_new(input_lock, {
        "status": "LOCKED_BEFORE_QC",
        "protocol_id": pid,
        "source_artifact_sha256": capture_sources,
        "qc_code_sha256": sha(Path(__file__)),
        "r4_amendment_sha256": sha(R4_LOCK),
    })
    predicate = state_predicate()
    scene_lookup = {scene["scene_id"]: scene for scene in scenes["scenes"]}
    min_pixels = int(gate["min_visible_evidence_px"])
    tie = float(gate["tie_margin_normalized"])
    rows: list[dict[str, Any]] = []
    for record in records:
        scene_id = record["scene_id"]
        item = annotations["scenes"][scene_id]
        scene = scene_lookup[scene_id]
        labels = frozen.labels(capture / scene_id / "evaluator/semantic_labels.png")
        camera = read_json(capture / record["input_files"]["camera_info"])
        tf = read_json(capture / record["input_files"]["tf_snapshot"])["camera_color_optical_frame"]
        depth = np.load(capture / record["input_files"]["depth_m"], allow_pickle=False)
        reasons = frozen.sensor_reasons(labels, depth, camera, tf)
        candidates = [frozen.geom(labels, label) for label in item.get("candidate_labels", [])]
        context = [frozen.geom(labels, label) for label in item.get("context_labels", [])]
        target = frozen.geom(labels, item["target_label"]) if item.get("target_label") is not None else None
        passed, projected = predicate(
            item["state"], item, scene, candidates, context, target, labels, tf, camera,
            gate, capture, scene_id, min_pixels, tie,
        )
        target_pixels = target["visible_pixels"] if target else 0
        if item["state"] == "INSUFFICIENT_EVIDENCE" and not (1 <= target_pixels < min_pixels):
            reasons.append("INSUFFICIENT_EVIDENCE_TARGET_PIXEL_GATE_FAILED")
        if item["state"] != "INSUFFICIENT_EVIDENCE" and item["state"] != "ABSENT":
            if any(candidate["visible_pixels"] < min_pixels for candidate in candidates):
                reasons.append("CANDIDATE_MINIMUM_VISIBLE_PIXEL_GATE_FAILED")
        if item["state"] == "ABSENT" and any(value["visible_pixels"] < min_pixels for value in context):
            reasons.append("ABSENT_CONTEXT_MINIMUM_VISIBLE_PIXEL_GATE_FAILED")
        if not passed:
            reasons.append("REQUESTED_STATE_NOT_VERIFIED:" + item["state"])
        rows.append({
            "scene_id": scene_id,
            "family_id": item["family_id"],
            "state": item["state"],
            "relation_variant": item["relation_variant"],
            "state_verified": bool(passed),
            "target_visible_pixels": target_pixels,
            "candidate_set": candidates,
            "context_set": context,
            "target": target,
            "projected_target": projected,
            "reasons": reasons,
        })
    duplicate = duplicate_qc(pid, capture, records)
    leakage = manifest_leakage_qc(pid)
    write_new(duplicate_path, duplicate)
    write_new(leakage_path, leakage)
    passed = all(not row["reasons"] for row in rows) and duplicate["status"] == "PASS" and leakage["status"] == "PASS"
    report = {
        "status": "PASS" if passed else "BLOCKED",
        "protocol_id": pid,
        "records": len(rows),
        "passed_scene_count": sum(not row["reasons"] for row in rows),
        "verified_state_counts": dict(Counter(row["state"] for row in rows if not row["reasons"])),
        "visibility_contract": {"insufficient_target_pixels": "1 <= pixels < 120", "all_required_candidates_or_context": "pixels >= 120"},
        "scenes": rows,
        "rgb_qc_sha256": sha(duplicate_path),
        "manifest_leakage_qc_sha256": sha(leakage_path),
        "input_lock_sha256": sha(input_lock),
        "scientific_hypothesis": "NOT_TESTED",
        "dataset_materialized": False,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(report_path, report)
    artifacts = [input_lock, duplicate_path, leakage_path, report_path]
    if pid == design.PILOT:
        decision = {
            "schema_version": 1,
            "protocol_id": pid,
            "status": "PASS" if passed and len(rows) == 32 else "REJECT",
            "decision": "PILOT_R4_32_OF_32_QC_PASS" if passed and len(rows) == 32 else "PILOT_R4_ATTEMPT_01_REJECTED_STOP_NO_RETRY",
            "classification": "GEOMETRY_PILOT_DATA_GATE_NOT_SCIENTIFIC_CALIBRATION_RESULT",
            "captured_family_count": len(rows),
            "passed_scene_count": report["passed_scene_count"],
            "source_artifact_sha256": {str(path.relative_to(ROOT)): sha(path) for path in artifacts},
            "calibration_capture_authorized": False,
            "model_inference_run": False,
            "calibrator_fit_call_count": 0,
            "scientific_hypothesis": "NOT_TESTED",
            "scientific_decision": None,
            "test_iid_ood_access": False,
            "robot_access": False,
        }
        write_new(PILOT_DECISION, decision)
        artifacts.append(PILOT_DECISION)
    print(json.dumps({key: report[key] for key in ("status", "records", "passed_scene_count", "verified_state_counts")}, indent=2))
    if not passed:
        failure_lock(output, pid, artifacts, "PILOT_QC" if pid == design.PILOT else "CALIBRATION_QC")
        raise SystemExit(2)


def materialize() -> None:
    pid = design.CALIBRATION
    output, capture, scenes, annotations, gate, records = validate_capture(pid)
    qc_path = output / "GEOMETRY_QC.json"
    qc_report = read_json(qc_path)
    if qc_report.get("status") != "PASS" or qc_report.get("passed_scene_count") != 128:
        raise RuntimeError("Calibration-v6 R4 128/128 QC PASS required")
    input_lock = read_json(output / "QC_INPUT_LOCK.json")
    for name, digest in input_lock["source_artifact_sha256"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("capture drift after QC input lock")
    DATASET.mkdir()  # fail closed if canonical dataset already exists
    by_id = {row["scene_id"]: row for row in qc_report["scenes"]}
    inference: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    for record in records:
        scene_id = record["scene_id"]
        annotation = annotations["scenes"][scene_id]
        row = by_id[scene_id]
        destination = DATASET / "records" / scene_id / "input"
        destination.mkdir(parents=True)
        for name in ("rgb", "depth_m", "camera_info", "tf_snapshot"):
            source = capture / record["input_files"][name]
            os.link(source, destination / source.name)
        depth = np.load(capture / record["input_files"]["depth_m"], allow_pickle=False)
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
        inference.append({
            **common,
            "image": f"records/{scene_id}/input/rgb.png",
            "depth": f"records/{scene_id}/input/depth_view.png",
            "metric_depth": f"records/{scene_id}/input/depth_m.npy",
            "instruction": record["instruction"] + " " + record["coordinate_suffix"],
        })
        truth.append({
            **common,
            "answerability_state": annotation["state"],
            "answerability_verified": True,
            "target_id": annotation.get("target_id"),
            "target_xy": row["target"]["centroid_normalized_xy"] if annotation["state"] == "FOUND" else None,
        })
    for name, rows in (("inference_manifest.jsonl", inference), ("evaluator_ground_truth.jsonl", truth)):
        with (DATASET / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_new(DATASET / "manifest.json", {
        "status": "PASS",
        "protocol_id": pid,
        "records": 128,
        "parent_families": 128,
        "capture_revision": "R4",
        "qc_report_sha256": sha(qc_path),
        "test_iid_ood_access": False,
    })
    source_paths = [
        qc_path, output / "QC_INPUT_LOCK.json", output / "RGB_DUPLICATE_QC.json",
        output / "MANIFEST_LEAKAGE_QC.json", R4_LOCK, CALIBRATION_AUTH, *DATASET.rglob("*"),
    ]
    write_new(MATERIALIZATION_LOCK, {
        "status": "MATERIALIZATION_FROZEN_128_OF_128_QC_PASS",
        "protocol_id": pid,
        "source_artifact_sha256": {str(path.relative_to(ROOT)): sha(path) for path in source_paths if path.is_file()},
        "test_iid_ood_access": False,
        "robot_access": False,
    })
    print(json.dumps({"status": "MATERIALIZATION_FROZEN_128_OF_128_QC_PASS", "sha256": sha(MATERIALIZATION_LOCK)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("pilot", "calibration", "materialize"))
    args = parser.parse_args()
    if args.command == "materialize":
        materialize()
    else:
        qc(design.PILOT if args.command == "pilot" else design.CALIBRATION)


if __name__ == "__main__":
    main()
