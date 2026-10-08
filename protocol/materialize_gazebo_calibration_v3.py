#!/usr/bin/env python3
"""All-or-nothing QC and materialization for Calibration-v3."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

import materialize_gazebo_calibration_v1 as implementation


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "gazebo_calibration_v3"
CONFIG = ROOT / "ur3/ur3_perception/config"
SCENES = CONFIG / f"{PROTOCOL}_scenes.yaml"
ANNOTATIONS = CONFIG / f"{PROTOCOL}_annotations.yaml"
GATE = CONFIG / f"{PROTOCOL}_gate.yaml"
PRIMARY_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json"
IMPLEMENTATION_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_IMPLEMENTATION_LOCK_R3.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_calibration_v3_capture_compatibility_lock.json"
CAPTURE = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/capture_attempt_01"
OUT = ROOT / "datasets/Gazebo_calibration_v3"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3"
INPUT_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_MATERIALIZATION_INPUT_LOCK.json"
FINAL_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V3_MATERIALIZATION_LOCK.json"
QC = RESULT / "GAZEBO_CALIBRATION_V3_GEOMETRY_QC.json"
DUPLICATE_QC = RESULT / "GAZEBO_CALIBRATION_V3_RGB_DUPLICATE_QC.json"
CROSS_DUPLICATE_QC = RESULT / "GAZEBO_CALIBRATION_V3_CROSS_SPLIT_RGB_QC.json"
PRIOR_MANIFESTS = tuple(
    path
    for path in (
        ROOT / "datasets/Gazebo_dev/evaluator_ground_truth.jsonl",
        ROOT / "datasets/Gazebo_dev_answerability_v2/evaluator_ground_truth.jsonl",
        ROOT / "datasets/Gazebo_train_uq_v2_full_r3/evaluator_ground_truth.jsonl",
        ROOT / "datasets/Gazebo_calibration_v2/evaluator_ground_truth.jsonl",
    )
    if path.is_file()
)
PRIOR_CAPTURE_ROOTS = tuple(
    path
    for path in (
        ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev/capture_attempt_01",
        ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2/capture_attempt_01",
        ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v1/capture_attempt_01",
        ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full/capture_attempt_01",
        ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full_r2/capture_attempt_01",
        ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full_r3/capture_attempt_01",
        ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v1/capture_attempt_01",
        ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v2/capture_attempt_01",
    )
    if (path / "input_manifest.jsonl").is_file()
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure() -> None:
    common = {
        "PROTOCOL": PROTOCOL,
        "SCENES": SCENES,
        "ANNOTATIONS": ANNOTATIONS,
        "GATE": GATE,
        "PRIMARY_LOCK": PRIMARY_LOCK,
        "CAPTURE_LOCK": CAPTURE_LOCK,
        "CAPTURE": CAPTURE,
        "OUT": OUT,
        "RESULT": RESULT,
        "LOCK": INPUT_LOCK,
        "QC": QC,
        "DEV_MANIFESTS": PRIOR_MANIFESTS,
    }
    for target in (implementation, implementation.base, implementation.full):
        for name, value in common.items():
            setattr(target, name, value)
    implementation.DUPLICATE_QC = DUPLICATE_QC
    implementation.full.DUPLICATE_QC = DUPLICATE_QC
    implementation.CROSS_DUPLICATE_QC = CROSS_DUPLICATE_QC
    implementation.configure = configure


def validate_capture():
    scenes, annotations, gate = (
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in (SCENES, ANNOTATIONS, GATE)
    )
    contract = json.loads(PRIMARY_LOCK.read_text(encoding="utf-8"))
    capture_lock = json.loads(CAPTURE_LOCK.read_text(encoding="utf-8"))
    manifest = json.loads((CAPTURE / "capture_manifest.json").read_text(encoding="utf-8"))
    inputs = implementation.base.jsonl(CAPTURE / "input_manifest.jsonl")
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("protocol_id") != PROTOCOL
        or len(inputs) != 128
        or len({row["scene_id"] for row in inputs}) != 128
    ):
        raise RuntimeError("Calibration-v3 capture is incomplete")
    expected = {
        "pretrial_source_lock_sha256": sha256(CAPTURE_LOCK),
        "scene_config_sha256": sha256(SCENES),
        "annotation_file_sha256": sha256(ANNOTATIONS),
        "gate_config_file_sha256": sha256(GATE),
    }
    for name, digest in expected.items():
        if manifest.get(name) != digest:
            raise RuntimeError(f"capture provenance mismatch: {name}")
    if capture_lock.get("parent_contract_lock_sha256") != sha256(PRIMARY_LOCK):
        raise RuntimeError("capture lock does not bind the contract")
    if capture_lock.get("implementation_lock_sha256") != sha256(IMPLEMENTATION_LOCK):
        raise RuntimeError("capture lock does not bind the implementation")
    if manifest.get("input_manifest_sha256") != sha256(CAPTURE / "input_manifest.jsonl"):
        raise RuntimeError("capture input manifest drift")
    if manifest.get("source_artifacts_verified_before_capture") is not True:
        raise RuntimeError("capture did not verify its source artifacts")
    return scenes, annotations, gate, manifest, inputs


def load_capture_images(root: Path) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    output = []
    for row in implementation.base.jsonl(root / "input_manifest.jsonl"):
        path = root / row["input_files"]["rgb"]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"cannot decode RGB: {path}")
        digest = row.get("input_sha256", {}).get("rgb") or sha256(path)
        thumb = cv2.resize(image, (80, 60), interpolation=cv2.INTER_AREA)
        output.append((row["scene_id"], digest, image, thumb))
    return output


def cross_split_duplicate_audit() -> dict:
    current = load_capture_images(CAPTURE)
    prior = [
        (root.name + ":" + scene_id, digest, image, thumb)
        for root in PRIOR_CAPTURE_ROOTS
        for scene_id, digest, image, thumb in load_capture_images(root)
    ]
    prior_hash = {digest: scene_id for scene_id, digest, _, _ in prior}
    exact = [
        {"calibration_v3": scene_id, "prior": prior_hash[digest]}
        for scene_id, digest, _, _ in current
        if digest in prior_hash
    ]
    near = []
    closest = []
    for scene_id, _, image, thumb in current:
        for prior_id, _, prior_image, prior_thumb in prior:
            thumbnail_mad = float(np.mean(cv2.absdiff(thumb, prior_thumb)))
            if thumbnail_mad < 2.0:
                difference = cv2.absdiff(image, prior_image)
                item = {
                    "calibration_v3": scene_id,
                    "prior": prior_id,
                    "thumbnail_mad": thumbnail_mad,
                    "gray_mad": float(np.mean(difference)),
                    "changed_pixel_fraction": float(np.mean(difference >= 3)),
                }
                closest.append(item)
                if item["gray_mad"] < 0.05 and item["changed_pixel_fraction"] < 0.002:
                    near.append(item)
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "status": "PASS" if not exact and not near else "REJECT",
        "current_images": len(current),
        "prior_images": len(prior),
        "prior_capture_input_manifest_sha256": {
            str((root / "input_manifest.jsonl").relative_to(ROOT)): sha256(root / "input_manifest.jsonl")
            for root in PRIOR_CAPTURE_ROOTS
        },
        "exact_duplicate_count": len(exact),
        "perceptual_near_duplicate_count": len(near),
        "exact_duplicates": exact,
        "perceptual_near_duplicates": near,
        "closest_pairs": sorted(
            closest, key=lambda row: (row["gray_mad"], row["changed_pixel_fraction"])
        )[:20],
    }
    write_json(CROSS_DUPLICATE_QC, report)
    return report


def lock_inputs() -> None:
    configure()
    if INPUT_LOCK.exists():
        raise FileExistsError("refusing to overwrite materialization input lock")
    implementation.base.validate_capture = validate_capture
    implementation.base.lock_materialization()
    value = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    value.update(
        status="LOCKED_BEFORE_QC_AND_MATERIALIZATION",
        protocol_id=PROTOCOL,
        implementation_lock_sha256=sha256(IMPLEMENTATION_LOCK),
        wrapper_source_sha256=sha256(Path(__file__).resolve()),
    )
    write_json(INPUT_LOCK, value)
    print(json.dumps({"status": value["status"], "sha256": sha256(INPUT_LOCK)}, indent=2))


def materialize(output: Path) -> None:
    configure()
    implementation.calibration_validate_capture = validate_capture
    implementation.base.validate_capture = validate_capture
    implementation.cross_split_duplicate_audit = cross_split_duplicate_audit
    implementation.materialize(output.resolve())
    manifest_path = output.resolve() / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value.update(
        dataset="Gazebo_calibration_v3",
        protocol_id=PROTOCOL,
        records=128,
        parent_families=128,
        supervision_split="none_calibration_fit_only",
        implementation_lock_sha256=sha256(IMPLEMENTATION_LOCK),
        test_iid_ood_access=False,
    )
    write_json(manifest_path, value)


def freeze_materialization() -> None:
    if FINAL_LOCK.exists():
        raise FileExistsError("refusing to overwrite final materialization lock")
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    geometry = json.loads(QC.read_text(encoding="utf-8"))
    within = json.loads(DUPLICATE_QC.read_text(encoding="utf-8"))
    cross = json.loads(CROSS_DUPLICATE_QC.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or any(
        report.get("status") != "PASS" for report in (geometry, within, cross)
    ):
        raise RuntimeError("materialization cannot be frozen without all QC gates")
    paths = [
        Path(__file__).resolve(), INPUT_LOCK, PRIMARY_LOCK, IMPLEMENTATION_LOCK,
        CAPTURE_LOCK, CAPTURE / "capture_manifest.json", CAPTURE / "input_manifest.jsonl",
        OUT / "manifest.json", OUT / "inference_manifest.jsonl",
        OUT / "evaluator_ground_truth.jsonl", QC, DUPLICATE_QC, CROSS_DUPLICATE_QC,
    ]
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "status": "MATERIALIZATION_FROZEN_128_OF_128_QC_PASS",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": 128,
        "source_artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in paths
        },
        "test_iid_ood_access": False,
    }
    write_json(FINAL_LOCK, payload)
    print(json.dumps({"status": payload["status"], "sha256": sha256(FINAL_LOCK)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("lock-inputs")
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--output", type=Path, default=OUT)
    subparsers.add_parser("freeze-materialization")
    args = parser.parse_args()
    if args.command == "lock-inputs":
        lock_inputs()
    elif args.command == "materialize":
        materialize(args.output)
    else:
        freeze_materialization()


if __name__ == "__main__":
    main()
