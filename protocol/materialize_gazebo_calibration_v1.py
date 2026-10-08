#!/usr/bin/env python3
"""All-or-nothing geometry QC and materialization for Calibration-v1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

import materialize_gazebo_train_uq_v1 as base
import materialize_gazebo_train_uq_v2_full as full


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "gazebo_calibration_v1"
CONFIG = ROOT / "ur3/ur3_perception/config"
SCENES = CONFIG / f"{PROTOCOL}_scenes.yaml"
ANNOTATIONS = CONFIG / f"{PROTOCOL}_annotations.yaml"
GATE = CONFIG / f"{PROTOCOL}_gate.yaml"
PRIMARY_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_CONTRACT_LOCK.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_calibration_v1_capture_compatibility_lock.json"
CAPTURE = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v1/capture_attempt_01"
OUT = ROOT / "datasets/Gazebo_calibration_v1"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v1"
LOCK = ROOT / "protocol/gazebo_calibration_v1_materialization_lock.json"
QC = RESULT / "GAZEBO_CALIBRATION_V1_GEOMETRY_QC.json"
DUPLICATE_QC = RESULT / "GAZEBO_CALIBRATION_V1_RGB_DUPLICATE_QC.json"
CROSS_DUPLICATE_QC = RESULT / "GAZEBO_CALIBRATION_V1_CROSS_SPLIT_RGB_QC.json"
PRIOR_FULL_CAPTURE = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full_r3/capture_attempt_01"
PRIOR_MANIFESTS = (
    ROOT / "datasets/Gazebo_dev/evaluator_ground_truth.jsonl",
    ROOT / "datasets/Gazebo_dev_answerability_v2/evaluator_ground_truth.jsonl",
    ROOT / "datasets/Gazebo_train_uq_v2_full_r3/evaluator_ground_truth.jsonl",
)


def configure():
    values = {"PROTOCOL": PROTOCOL, "SCENES": SCENES, "ANNOTATIONS": ANNOTATIONS,
              "GATE": GATE, "PRIMARY_LOCK": PRIMARY_LOCK, "CAPTURE_LOCK": CAPTURE_LOCK,
              "CAPTURE": CAPTURE, "OUT": OUT, "RESULT": RESULT, "LOCK": LOCK, "QC": QC,
              "DEV_MANIFESTS": PRIOR_MANIFESTS}
    for name, value in values.items():
        setattr(base, name, value)
    for name, value in {**values, "DUPLICATE_QC": DUPLICATE_QC}.items():
        setattr(full, name, value)


def calibration_validate_capture():
    configure()
    source = inspect.getsource(base.validate_capture).replace(
        "len(inputs) != 320 or len({r[\"scene_id\"] for r in inputs}) != 320",
        "len(inputs) != 128 or len({r[\"scene_id\"] for r in inputs}) != 128",
    )
    scope = dict(base.__dict__)
    exec(source, scope)
    return scope["validate_capture"]()


def lock_materialization():
    configure()
    base.validate_capture = calibration_validate_capture
    base.lock_materialization()
    payload = json.loads(LOCK.read_text())
    payload.update(
        wrapper_source_sha256=base.sha(Path(__file__).resolve()),
        calibration_contract_sha256=base.sha(PRIMARY_LOCK),
        calibration_only_no_model_training=True,
        expected_records=128,
        locked_at_utc_final=datetime.now(timezone.utc).isoformat(),
    )
    base.dump(LOCK, payload)
    print(json.dumps({"status": "LOCKED_FINAL", "sha256": base.sha(LOCK)}, indent=2))


def cross_split_duplicate_audit():
    current = base.jsonl(CAPTURE / "input_manifest.jsonl")
    previous = base.jsonl(PRIOR_FULL_CAPTURE / "input_manifest.jsonl")
    def load(root, rows):
        out = []
        for row in rows:
            path = root / row["input_files"]["rgb"]
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(path)
            out.append((row["scene_id"], row["input_sha256"]["rgb"], image,
                        cv2.resize(image, (80, 60), interpolation=cv2.INTER_AREA)))
        return out
    left, right = load(CAPTURE, current), load(PRIOR_FULL_CAPTURE, previous)
    prior_hashes = {digest: sid for sid, digest, _, _ in right}
    exact = [{"calibration": sid, "prior": prior_hashes[digest]} for sid, digest, _, _ in left if digest in prior_hashes]
    near, closest = [], []
    for sid, _, image, thumb in left:
        for old_sid, _, old_image, old_thumb in right:
            thumb_mad = float(np.mean(cv2.absdiff(thumb, old_thumb)))
            if thumb_mad < 2.0:
                diff = cv2.absdiff(image, old_image)
                item = {"calibration": sid, "prior": old_sid, "thumbnail_mad": thumb_mad,
                        "gray_mad": float(np.mean(diff)), "changed_pixel_fraction": float(np.mean(diff >= 3))}
                closest.append(item)
                if item["gray_mad"] < 0.05 and item["changed_pixel_fraction"] < 0.002:
                    near.append(item)
    report = {"schema_version": 1, "protocol_id": PROTOCOL,
              "status": "PASS" if not exact and not near else "REJECT",
              "comparison": "Calibration-v1 versus accepted Train-UQ/Val-UQ full-r3",
              "calibration_images": len(left), "prior_images": len(right),
              "exact_duplicate_count": len(exact), "perceptual_near_duplicate_count": len(near),
              "exact_duplicates": exact, "perceptual_near_duplicates": near,
              "closest_pairs": sorted(closest, key=lambda x: (x["gray_mad"], x["changed_pixel_fraction"]))[:20]}
    base.dump(CROSS_DUPLICATE_QC, report)
    return report


def materialize(output: Path):
    configure()
    base.validate_capture = calibration_validate_capture
    within = full.duplicate_audit()
    cross = cross_split_duplicate_audit()
    if within["status"] != "PASS" or cross["status"] != "PASS":
        raise SystemExit("Calibration RGB duplicate gate REJECT")
    source = inspect.getsource(base.materialize)
    old = '''prior = set().union(*(set(row["family_id"] for row in jsonl(path)) for path in DEV_MANIFESTS))
    families = [row["family_id"] for row in eval_rows]; split_families = {split: {row["family_id"] for row in eval_rows if row["split"] == split} for split in ("train_uq", "val_uq")}
    leakage = {"unique_parent_families": len(set(families)) == 320, "split_sizes": {"train_uq": len(split_families["train_uq"]) == 256, "val_uq": len(split_families["val_uq"]) == 64}, "train_val_family_disjoint": not bool(split_families["train_uq"] & split_families["val_uq"]), "no_dev_family_overlap": not bool(set(families) & prior), "inference_manifest_oracle_free": all(not ({"answerability_state", "target_id", "target_xy", "candidate_set", "failure_tags", "supervision"} & set(row)) for row in infer_rows)}
    passed = all(not row["reasons"] for row in qc_rows) and all(leakage["split_sizes"].values()) and all(value for key, value in leakage.items() if key != "split_sizes")'''
    new = '''prior = set().union(*(set(row["family_id"] for row in jsonl(path)) for path in DEV_MANIFESTS))
    families = [row["family_id"] for row in eval_rows]
    leakage = {"unique_parent_families": len(set(families)) == 128, "calibration_size": len(families) == 128, "all_rows_calibration_split": all(row["split"] == "calibration" for row in eval_rows), "no_prior_family_overlap": not bool(set(families) & prior), "inference_manifest_oracle_free": all(not ({"answerability_state", "target_id", "target_xy", "candidate_set", "failure_tags", "supervision"} & set(row)) for row in infer_rows)}
    passed = all(not row["reasons"] for row in qc_rows) and all(leakage.values())'''
    if old not in source:
        raise ValueError("upstream leakage block changed")
    source = source.replace(old, new)
    source = source.replace('ev = base / "evaluator"; ev.mkdir(parents=True)',
                            'ev = base / "evaluator"; ev.mkdir(parents=True); (base / "input").mkdir(parents=True, exist_ok=True)')
    source = source.replace('"Gazebo_train_uq_v1"', '"Gazebo_calibration_v1"')
    source = source.replace('"records": 320, "parent_families": 320', '"records": 128, "parent_families": 128')
    source = source.replace('"supervision_split": "train_uq_only"', '"supervision_split": "none_calibration_fit_only"')
    scope = dict(base.__dict__)
    scope.update(globals())
    scope["validate_capture"] = calibration_validate_capture
    exec(source, scope)
    scope["materialize"](output.resolve())
    qc = json.loads(QC.read_text())
    qc.update(within_split_rgb_qc_sha256=base.sha(DUPLICATE_QC),
              cross_split_rgb_qc_sha256=base.sha(CROSS_DUPLICATE_QC),
              all_geometry_and_duplicate_gates_passed=qc["status"] == "PASS")
    base.dump(QC, qc)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest.update(qc_report_sha256=base.sha(QC), calibration_labels_use="fit_calibrator_and_threshold_only",
                    grounding_or_risk_training_forbidden=True, test_iid_ood_access=False)
    base.dump(output / "manifest.json", manifest)
    print(json.dumps({"status": "PASS", "records": qc["records"], "failed": qc["failed_scene_count"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("lock-materialization")
    run = sub.add_parser("materialize")
    run.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "lock-materialization":
        lock_materialization()
    else:
        materialize(args.output)


if __name__ == "__main__":
    main()
