#!/usr/bin/env python3
"""Geometry-QC and materialize family-disjoint Gazebo Train-UQ/Val-UQ.

This is deliberately a new, post-capture materialization stage.  It never
changes the frozen capture contract or Dev-v2.  ``lock-materialization`` must
be run once before ``materialize``; the latter refuses a stale lock, a partial
capture, a failed QC, or a non-empty destination.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "gazebo_train_uq_v1"
CONFIG = ROOT / "ur3/ur3_perception/config"
SCENES, ANNOTATIONS, GATE = (CONFIG / "gazebo_train_uq_v1_scenes.yaml", CONFIG / "gazebo_train_uq_v1_annotations.yaml", CONFIG / "gazebo_train_uq_v1_gate.yaml")
PRIMARY_LOCK = ROOT / "protocol/gazebo_train_uq_v1_contract_lock.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_train_uq_v1_capture_compatibility_lock.json"
CAPTURE = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v1/capture_attempt_01"
OUT = ROOT / "datasets/Gazebo_train_uq_v1"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v1"
LOCK = ROOT / "protocol/gazebo_train_uq_v1_materialization_lock.json"
QC = RESULT / "GAZEBO_TRAIN_UQ_V1_GEOMETRY_QC.json"
DEV_MANIFESTS = (ROOT / "datasets/Gazebo_dev/evaluator_ground_truth.jsonl", ROOT / "datasets/Gazebo_dev_answerability_v2/evaluator_ground_truth.jsonl")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def labels(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode {path}")
    if image.ndim == 3:
        if not (np.array_equal(image[..., 0], image[..., 1]) and np.array_equal(image[..., 1], image[..., 2])):
            raise ValueError(f"label channels disagree: {path}")
        image = image[..., 0]
    return image


def geom(label_map: np.ndarray, label: int) -> dict:
    ys, xs = np.nonzero(label_map == int(label))
    if not xs.size:
        return {"semantic_label": int(label), "visible_pixels": 0, "centroid_xy": None, "centroid_normalized_xy": None, "bbox_xyxy": None}
    return {"semantic_label": int(label), "visible_pixels": int(xs.size), "centroid_xy": [float(xs.mean()), float(ys.mean())], "centroid_normalized_xy": [float(xs.mean()/label_map.shape[1]), float(ys.mean()/label_map.shape[0])], "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}


def depth_view(depth: np.ndarray) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(value) & (value >= .10) & (value <= 2.0)
    if int(valid.sum()) < 16:
        raise ValueError("too few valid metric-depth pixels")
    near, far = np.percentile(value[valid], [2, 98])
    if far - near < 1e-6:
        raise ValueError("metric depth has no usable range")
    inverse = 1 / np.maximum(np.clip(value, near, far), 1e-6)
    gray = np.clip((inverse - 1/far) / max(1/near - 1/far, 1e-6) * 255, 0, 255).astype(np.uint8)
    gray[~valid] = 0
    return np.repeat(gray[..., None], 3, axis=-1)


def rotation(q: list[float]) -> np.ndarray:
    x, y, z, w = q; n = math.sqrt(x*x+y*y+z*z+w*w); x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float64)


def project(point: list[float], tf: dict, camera: dict) -> dict:
    camera_point = rotation(tf["orientation_xyzw"]).T @ (np.asarray(point, dtype=np.float64) - np.asarray(tf["position"], dtype=np.float64))
    z = float(camera_point[2])
    if z <= 0:
        return {"camera_xyz": camera_point.tolist(), "pixel_xy": None, "in_frame": False}
    u, v = camera["k"][0]*float(camera_point[0])/z + camera["k"][2], camera["k"][4]*float(camera_point[1])/z + camera["k"][5]
    return {"camera_xyz": camera_point.tolist(), "pixel_xy": [u, v], "in_frame": 0 <= u < 640 and 0 <= v < 480 and .1 <= z <= 2.0}


def sensor_reasons(label_map: np.ndarray, depth: np.ndarray, camera: dict, tf: dict) -> list[str]:
    reasons = []
    if depth.shape != label_map.shape or depth.dtype not in (np.float32, np.float64): reasons.append("DEPTH_NOT_REGISTERED_METRIC_ARRAY")
    if camera.get("width") != 640 or camera.get("height") != 480 or camera.get("frame_id") != "camera_color_optical_frame": reasons.append("CAMERA_CONTRACT_MISMATCH")
    if "error" in tf or not tf.get("position"): reasons.append("CAMERA_TF_MISSING")
    return reasons


def validate_capture() -> tuple[dict, dict, dict, dict, list[dict]]:
    scenes, ann, gate = (yaml.safe_load(path.read_text(encoding="utf-8")) for path in (SCENES, ANNOTATIONS, GATE))
    lock, cap_lock = (json.loads(path.read_text(encoding="utf-8")) for path in (PRIMARY_LOCK, CAPTURE_LOCK))
    manifest_path = CAPTURE / "capture_manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = jsonl(CAPTURE / "input_manifest.jsonl")
    if manifest.get("status") != "COMPLETE" or manifest.get("protocol_id") != PROTOCOL or len(inputs) != 320 or len({r["scene_id"] for r in inputs}) != 320:
        raise ValueError("capture is incomplete or has a bad record count")
    expected = {"pretrial_source_lock_sha256": sha(CAPTURE_LOCK), "scene_config_sha256": sha(SCENES), "annotation_file_sha256": sha(ANNOTATIONS), "gate_config_file_sha256": sha(GATE)}
    for name, value in expected.items():
        if manifest.get(name) != value: raise ValueError(f"capture provenance mismatch: {name}")
    if cap_lock.get("parent_contract_lock_sha256") != sha(PRIMARY_LOCK) or manifest.get("model_inventory_sha256_preregistered") != cap_lock.get("model_inventory_sha256"):
        raise ValueError("capture/primary-lock linkage failed")
    if manifest.get("input_manifest_sha256") != sha(CAPTURE / "input_manifest.jsonl") or manifest.get("source_artifacts_verified_before_capture") is not True:
        raise ValueError("capture inputs/source hashes are not verified")
    return scenes, ann, gate, manifest, inputs


def lock_materialization() -> None:
    if LOCK.exists(): raise FileExistsError(f"refusing to overwrite lock: {LOCK}")
    _, _, _, manifest, _ = validate_capture()
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in (Path(__file__).resolve(), SCENES, ANNOTATIONS, GATE, PRIMARY_LOCK, CAPTURE_LOCK, CAPTURE / "capture_manifest.json", CAPTURE / "input_manifest.jsonl")}
    dump(LOCK, {"schema_version": 1, "protocol_id": PROTOCOL, "status": "LOCKED_BEFORE_GEOMETRY_QC_AND_MATERIALIZATION", "locked_at_utc": datetime.now(timezone.utc).isoformat(), "capture_manifest_sha256": sha(CAPTURE / "capture_manifest.json"), "capture_input_manifest_sha256": sha(CAPTURE / "input_manifest.jsonl"), "primary_contract_lock_sha256": sha(PRIMARY_LOCK), "capture_compatibility_lock_sha256": sha(CAPTURE_LOCK), "source_artifact_sha256": source_hashes, "model_input_allowlist": ["image", "depth", "instruction"], "dev_v2_access": "forbidden_for_training_or_selection", "no_sam2": True, "capture_status": manifest["status"]})
    print(json.dumps({"status": "LOCKED", "sha256": sha(LOCK), "path": str(LOCK)}, indent=2))


def verify_materialization_lock() -> None:
    if not LOCK.is_file(): raise FileNotFoundError("run lock-materialization first")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_GEOMETRY_QC_AND_MATERIALIZATION": raise ValueError("invalid materialization lock")
    for relative, expected in lock["source_artifact_sha256"].items():
        if sha(ROOT / relative) != expected: raise ValueError(f"locked materialization input changed: {relative}")


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def materialize(output: Path) -> None:
    verify_materialization_lock()
    scenes, annotations, gate, manifest, inputs = validate_capture()
    output = output.resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing to overwrite non-empty dataset: {output}")
    scene_cfg = {s["scene_id"]: s for s in scenes["scenes"]}; min_px, tie = int(gate["min_visible_evidence_px"]), float(gate["tie_margin_normalized"])
    qc_rows, eval_rows, infer_rows, train_rows, artifacts = [], [], [], [], []
    for record in inputs:
        sid = record["scene_id"]; item = annotations["scenes"].get(sid); scene = scene_cfg.get(sid)
        if not item or not scene: raise ValueError(f"unregistered capture: {sid}")
        label_map = labels(CAPTURE / sid / "evaluator/semantic_labels.png")
        camera = json.loads((CAPTURE / record["input_files"]["camera_info"]).read_text(encoding="utf-8")); tf_all = json.loads((CAPTURE / record["input_files"]["tf_snapshot"]).read_text(encoding="utf-8")); tf = tf_all.get("camera_color_optical_frame", {})
        depth = np.load(CAPTURE / record["input_files"]["depth_m"], allow_pickle=False); reasons = sensor_reasons(label_map, depth, camera, tf)
        candidates = [geom(label_map, x) for x in item.get("candidate_labels", [])]; context = [geom(label_map, x) for x in item.get("context_labels", [])]
        target = geom(label_map, item["target_label"]) if item.get("target_label") is not None else None; state, projected = item["state"], None
        if state == "FOUND":
            visible = all(x["visible_pixels"] >= min_px for x in candidates); order = sorted(candidates, key=lambda x: x["centroid_normalized_xy"][0]) if visible else []
            if item["rank_from"] == "right": order.reverse()
            gaps = [abs(order[i]["centroid_normalized_xy"][0]-order[i+1]["centroid_normalized_xy"][0]) for i in range(len(order)-1)]
            verified = bool(visible and item["rank"] <= len(order) and order[item["rank"]-1]["semantic_label"] == item["target_label"] and all(gap > tie for gap in gaps))
        elif state == "AMBIGUOUS":
            valid = [geom(label_map, x) for x in item["valid_target_labels"]]; visible = all(x["visible_pixels"] >= min_px for x in candidates)
            order = sorted(candidates, key=lambda x: x["centroid_normalized_xy"][0]) if visible else []
            if item["rank_from"] == "right": order.reverse()
            tied = [i for i, x in enumerate(order) if x["semantic_label"] in set(item["valid_target_labels"])]
            xs = [x["centroid_normalized_xy"][0] for x in valid if x["centroid_normalized_xy"]]
            verified = bool(visible and len(xs) >= 2 and max(xs)-min(xs) <= tie and item["rank"]-1 in tied)
        elif state == "ABSENT":
            verified = bool(target and target["visible_pixels"] == 0 and item["target_id"] not in scene["poses"] and all(x["visible_pixels"] >= min_px for x in context))
        else:
            layout = json.loads((CAPTURE / sid / "evaluator/capture_oracle.json").read_text(encoding="utf-8"))["requested_scene_layout_base_link"]
            if tf.get("position"): projected = project(layout[item["target_id"]][:3], tf, camera)
            rule = gate["insufficient_evidence_rule"]; outer = float(rule["projected_center_outer_margin_fraction"]); p = projected.get("pixel_xy") if projected else None
            near = bool(p and -640*outer <= p[0] < 640*(1+outer) and -480*outer <= p[1] < 480*(1+outer) and .1 <= projected["camera_xyz"][2] <= 2.)
            others = [x for x in candidates if x["semantic_label"] != item["target_label"]]
            verified = bool(target and int(rule["min_visible_pixels_inclusive"]) <= target["visible_pixels"] < int(rule["max_visible_pixels_exclusive"]) and near and all(x["visible_pixels"] >= min_px for x in others))
        if not verified: reasons.append(f"REQUESTED_STATE_NOT_VERIFIED:{state}")
        valid_labels = item.get("valid_target_labels", [item.get("target_label")] if item.get("target_label") is not None else [])
        target_mask = np.isin(label_map, np.asarray(valid_labels if state == "AMBIGUOUS" else valid_labels[:1], dtype=label_map.dtype)).astype(np.uint8)*255 if valid_labels else np.zeros_like(label_map, dtype=np.uint8)
        anchor_labels = sorted((set(item.get("candidate_labels", [])) | set(item.get("context_labels", [])) | set(item.get("occluder_labels", []))) - ({item.get("target_label")} if item.get("target_label") is not None else set()))
        anchor = np.isin(label_map, np.asarray(anchor_labels, dtype=label_map.dtype)).astype(np.uint8)*255 if anchor_labels else np.zeros_like(label_map, dtype=np.uint8)
        answer = f"POINT [({target['centroid_normalized_xy'][0]:.6f}, {target['centroid_normalized_xy'][1]:.6f})]" if state == "FOUND" and target else "ABSTAIN"
        base = {"sample_id": sid, "scene_id": sid, "family_id": item["family_id"], "split": item["split"], "relation": "horizontal_ordinal_ranking_answerability", "relation_variant": item["relation_variant"], "reference_frame": gate["reference_frame"]}
        eval_rows.append({**base, "answerability_state": state, "answerability_verified": verified, "target_id": item.get("target_id"), "target_category": item["target_category"], "target_semantic_label": item.get("target_label"), "valid_target_ids": item.get("valid_target_ids", [item.get("target_id")] if item.get("target_id") else []), "valid_target_semantic_labels": valid_labels, "target_xy": target["centroid_normalized_xy"] if state == "FOUND" and target else None, "target_mask": f"records/{sid}/evaluator/target_mask.png", "anchor_mask_union": f"records/{sid}/evaluator/anchor_mask_union.png", "candidate_set": candidates, "context_set": context, "visibility": {"method": "visible_semantic_mask_pixels", "visible_pixels": target["visible_pixels"] if target else 0, "minimum_sufficient_pixels": min_px, "sufficient": bool(target and target["visible_pixels"] >= min_px)}, "target_center_projection_from_geometry": projected, "failure_tags": item["failure_tags"]})
        infer_rows.append({**base, "image": f"records/{sid}/input/rgb.png", "depth": f"records/{sid}/input/depth_view.png", "metric_depth": f"records/{sid}/input/depth_m.npy", "instruction": record["instruction"] + " " + record["coordinate_suffix"]})
        if item["split"] == "train_uq": train_rows.append({**infer_rows[-1], "supervision": {"response": answer, "answerability_state": state, "target_xy": target["centroid_normalized_xy"] if state == "FOUND" and target else None}})
        qc_rows.append({"scene_id": sid, "family_id": item["family_id"], "split": item["split"], "requested_state": state, "relation_variant": item["relation_variant"], "state_verified": verified, "reasons": reasons, "target_visible_pixels": target["visible_pixels"] if target else 0})
        artifacts.append((sid, target_mask, anchor, depth_view(depth), record))
    prior = set().union(*(set(row["family_id"] for row in jsonl(path)) for path in DEV_MANIFESTS))
    families = [row["family_id"] for row in eval_rows]; split_families = {split: {row["family_id"] for row in eval_rows if row["split"] == split} for split in ("train_uq", "val_uq")}
    leakage = {"unique_parent_families": len(set(families)) == 320, "split_sizes": {"train_uq": len(split_families["train_uq"]) == 256, "val_uq": len(split_families["val_uq"]) == 64}, "train_val_family_disjoint": not bool(split_families["train_uq"] & split_families["val_uq"]), "no_dev_family_overlap": not bool(set(families) & prior), "inference_manifest_oracle_free": all(not ({"answerability_state", "target_id", "target_xy", "candidate_set", "failure_tags", "supervision"} & set(row)) for row in infer_rows)}
    passed = all(not row["reasons"] for row in qc_rows) and all(leakage["split_sizes"].values()) and all(value for key, value in leakage.items() if key != "split_sizes")
    report = {"schema_version": 1, "protocol_id": PROTOCOL, "status": "PASS" if passed else "BLOCKED", "materialization_lock_sha256": sha(LOCK), "capture_manifest_sha256": sha(CAPTURE / "capture_manifest.json"), "records": len(qc_rows), "state_counts": dict(Counter(row["requested_state"] for row in qc_rows)), "split_counts": dict(Counter(row["split"] for row in qc_rows)), "relation_variant_counts": dict(Counter(row["relation_variant"] for row in qc_rows)), "verified_state_counts": dict(Counter(row["requested_state"] for row in qc_rows if row["state_verified"])), "failed_scene_count": sum(bool(row["reasons"]) for row in qc_rows), "leakage_checks": leakage, "scenes": qc_rows, "dataset_materialized": passed}
    dump(QC, report)
    if not passed:
        print(json.dumps({"status": "BLOCKED", "qc_report": str(QC), "failed": [r for r in qc_rows if r["reasons"]]}, indent=2)); raise SystemExit(2)
    output.mkdir(parents=True)
    for sid, tmask, anchor, view, record in artifacts:
        base = output / "records" / sid; ev = base / "evaluator"; ev.mkdir(parents=True)
        if not cv2.imwrite(str(ev / "target_mask.png"), tmask) or not cv2.imwrite(str(ev / "anchor_mask_union.png"), anchor) or not cv2.imwrite(str(base / "input/depth_view.png"), view): raise RuntimeError(f"write failed: {sid}")
        for key in ("rgb", "depth_m", "camera_info", "tf_snapshot"):
            src = CAPTURE / record["input_files"][key]; hardlink(src, base / "input" / src.name)
    for filename, rows in (("inference_manifest.jsonl", infer_rows), ("evaluator_ground_truth.jsonl", eval_rows), ("train_supervision.jsonl", train_rows)):
        with (output / filename).open("w", encoding="utf-8") as stream:
            for row in rows: stream.write(json.dumps(row, sort_keys=True) + "\n")
    dump(output / "manifest.json", {"schema_version": 1, "dataset": "Gazebo_train_uq_v1", "protocol_id": PROTOCOL, "status": "PASS", "records": 320, "parent_families": 320, "split_counts": report["split_counts"], "state_counts": report["state_counts"], "materialization_lock_sha256": sha(LOCK), "capture_manifest_sha256": sha(CAPTURE / "capture_manifest.json"), "qc_report_sha256": sha(QC), "model_input_allowlist": ["image", "depth", "instruction"], "supervision_split": "train_uq_only", "output_contract": gate["output_contract"], "no_sam2": True, "b2_opened": False, "test_iid_ood_access": False})
    print(json.dumps({"status": "PASS", "dataset": str(output), "qc_report": str(QC)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("lock-materialization"); p = sub.add_parser("materialize"); p.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "lock-materialization": lock_materialization()
    else: materialize(args.output)


if __name__ == "__main__": main()
