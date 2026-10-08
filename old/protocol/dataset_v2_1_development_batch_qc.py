#!/usr/bin/env python3
"""Observed RGB-D/semantic QC for one Dataset V2.1 development batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset_v2_relation_geometry_v2_1 import (
    GEOMETRY_VERSION,
    RelationGeometryError,
    evaluate_observed_relation,
)
from wp2_common import find_forbidden_inference_keys, read_json, sha256_file, utc_now


PROTOCOL_ID = "roborefer_dataset_v2_1_1_development_capture_400"
DECISION = "GO_DEVELOPMENT_V2_1_CAPTURE_400"
EXPECTED_ARTIFACTS = {
    "rgb_original.png", "depth_metric.npy", "semantic_instance_labels.png",
    "camera_info.json", "tf_snapshot.json",
}


class V21BatchQCError(RuntimeError):
    pass


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def semantic_visual(labels: np.ndarray) -> np.ndarray:
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for raw in np.unique(labels):
        label = int(raw)
        if label == 0:
            continue
        digest = hashlib.sha256(f"v21-semantic-label|{label}".encode()).digest()
        output[labels == label] = [
            64 + digest[0] % 192, 64 + digest[1] % 192, 64 + digest[2] % 192,
        ]
    return output


def depth_visual(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth >= 0.10) & (depth <= 2.0)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        near, far = np.percentile(depth[valid], [2, 98])
        gray[valid] = np.clip(
            (depth[valid] - near) * 255.0 / max(float(far - near), 1e-6), 0, 255
        ).astype(np.uint8)
    image = cv2.applyColorMap(255 - gray, cv2.COLORMAP_VIRIDIS)
    image[~valid] = 0
    return image


def mask_for_ids(labels: np.ndarray, ids: list[str], label_by_id: dict[str, int]) -> np.ndarray:
    if not ids:
        return np.zeros(labels.shape, dtype=np.uint8)
    wanted = np.asarray([label_by_id[value] for value in ids], dtype=labels.dtype)
    return np.isin(labels, wanted).astype(np.uint8) * 255


def relation_rows(
    family: dict[str, Any], labels: np.ndarray, depth: np.ndarray,
    label_by_id: dict[str, int],
) -> list[dict[str, Any]]:
    spec = family["variant_specs"][0]
    relation = spec["relations"][0]
    if relation == "direct":
        return []
    state = spec["answerability_state"]
    submode = spec["state_submode"]
    if state == "ABSENT" and submode != "unsatisfied_relation":
        return []
    targets = (
        spec["candidate_target_ids"]
        if state == "ABSENT" else spec["valid_target_ids"]
    )
    rows = []
    for target_id in targets:
        expected = "UNSATISFIED" if submode == "unsatisfied_relation" else "SATISFIED"
        allow_not_evaluable = state == "INSUFFICIENT_EVIDENCE"
        try:
            evidence = evaluate_observed_relation(
                relation, labels, depth, label_by_id[target_id],
                [label_by_id[value] for value in spec["anchor_ids"]],
            )
            observed = bool(evidence["passed"])
            passed = (not observed) if expected == "UNSATISFIED" else observed
            rows.append({
                "family_id": family["family_id"], "target_id": target_id,
                "relation": relation, "expected": expected,
                "status": "PASS" if passed else "FAIL", "gate_passed": passed,
                "evidence": evidence,
            })
        except RelationGeometryError as exc:
            rows.append({
                "family_id": family["family_id"], "target_id": target_id,
                "relation": relation, "expected": expected,
                "status": "NOT_EVALUABLE_ALLOWED" if allow_not_evaluable else "FAIL",
                "gate_passed": allow_not_evaluable,
                "error": str(exc),
            })
    return rows


def selected_evidence_families(
    batch: dict[str, Any], family_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    selected = []
    for state in ("FOUND", "INSUFFICIENT_EVIDENCE", "AMBIGUOUS", "ABSENT"):
        candidates = [
            value for value in batch["family_ids"]
            if family_by_id[value]["primary_answerability_stratum"] == state
        ]
        candidates.sort(key=lambda value: hashlib.sha256(
            f"v21-real-qc|{batch['batch_id']}|{state}|{value}".encode()
        ).hexdigest())
        if candidates:
            selected.append(candidates[0])
    return selected


def write_real_evidence(
    dataset_root: Path, batch: dict[str, Any], family_by_id: dict[str, dict[str, Any]],
    label_by_id: dict[str, int],
) -> dict[str, Any]:
    output = dataset_root / "report_assets" / "real_capture_qc" / batch["batch_id"]
    index = []
    for family_id in selected_evidence_families(batch, family_by_id):
        capture_id = f"{family_id}__clean_capture"
        source = dataset_root / "raw" / "captures" / capture_id
        destination = output / family_id
        destination.mkdir(parents=True, exist_ok=True)
        family = family_by_id[family_id]
        spec = family["variant_specs"][0]
        rgb = cv2.imread(str(source / "rgb_original.png"), cv2.IMREAD_COLOR)
        labels = cv2.imread(str(source / "semantic_instance_labels.png"), cv2.IMREAD_UNCHANGED)
        depth = np.load(source / "depth_metric.npy", allow_pickle=False)
        if rgb is None or labels is None:
            raise V21BatchQCError(f"cannot render evidence: {capture_id}")
        target_ids = spec["valid_target_ids"] or spec["candidate_target_ids"]
        target_mask = mask_for_ids(labels, target_ids, label_by_id)
        anchor_mask = mask_for_ids(labels, spec["anchor_ids"], label_by_id)
        overlay = rgb.copy()
        overlay[target_mask > 0] = (
            0.35 * overlay[target_mask > 0] + 0.65 * np.asarray([0, 0, 255])
        ).astype(np.uint8)
        overlay[anchor_mask > 0] = (
            0.35 * overlay[anchor_mask > 0] + 0.65 * np.asarray([0, 255, 255])
        ).astype(np.uint8)
        outputs = {
            "rgb.png": rgb,
            "depth_metric_visualization.png": depth_visual(depth),
            "semantic_instance_labels.png": semantic_visual(labels),
            "target_mask.png": target_mask,
            "anchor_mask_union.png": anchor_mask,
            "target_anchor_overlay.png": overlay,
        }
        hashes = {}
        for name, image in outputs.items():
            path = destination / name
            if not cv2.imwrite(str(path), image):
                raise V21BatchQCError(f"cannot write evidence image: {path}")
            hashes[name] = sha256_file(path)
        index.append({
            "family_id": family_id,
            "answerability_state": family["primary_answerability_stratum"],
            "capture_id": capture_id,
            "source_capture_meta_sha256": sha256_file(source / "capture_meta.json"),
            "image_sha256": hashes,
        })
    index_path = output / "REAL_CAPTURE_QC_INDEX.json"
    atomic_json(index_path, {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "batch_id": batch["batch_id"],
        "selection_policy": "one clean family per answerability state by locked SHA-256 rank",
        "contains_planned_or_decorative_images": False,
        "selected_family_count": len(index), "samples": index,
    })
    return {
        "path": str(output.relative_to(dataset_root)),
        "selected_family_count": len(index), "index_sha256": sha256_file(index_path),
    }


def run_qc(
    workspace: Path, dataset_root: Path, plan_path: Path, lock_path: Path,
    manifest_path: Path, batch_id: str,
) -> dict[str, Any]:
    plan, lock, manifest = read_json(plan_path), read_json(lock_path), read_json(manifest_path)
    if lock.get("protocol_id") != PROTOCOL_ID or lock.get("decision") != DECISION:
        raise V21BatchQCError("V2.1 execution lock identity/decision mismatch")
    if not lock.get("capture_authorized") or lock.get("training_authorized"):
        raise V21BatchQCError("capture is not authorized or training was incorrectly authorized")
    if dataset_root != (workspace / lock["capture_output_root"]).resolve():
        raise V21BatchQCError("dataset root differs from execution lock")
    if sha256_file(plan_path) != lock["capture_plan_sha256"] or sha256_file(manifest_path) != lock["manifest_sha256"]:
        raise V21BatchQCError("manifest/capture plan differs from execution lock")
    for relative, expected in lock["locked_artifact_sha256"].items():
        path = workspace / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise V21BatchQCError(f"execution-locked artifact changed: {relative}")
    if find_forbidden_inference_keys(plan["inference_payload_template"]):
        raise V21BatchQCError("oracle key exists in inference payload template")

    batches = {item["batch_id"]: item for item in plan["batches"]}
    if batch_id not in batches:
        raise V21BatchQCError(f"unknown batch: {batch_id}")
    batch = batches[batch_id]
    if batch["prerequisite_batch"] is not None:
        previous = dataset_root / "report_assets/checkpoints/batch_qc" / f"{batch['prerequisite_batch']}.json"
        if not previous.is_file() or not read_json(previous).get("passed"):
            raise V21BatchQCError(f"previous batch QC is not PASS: {batch['prerequisite_batch']}")

    family_by_id = {item["family_id"]: item for item in manifest["families"]}
    expected = {
        item["capture_id"]: item for item in plan["captures"] if item["batch_id"] == batch_id
    }
    if len(expected) != int(batch["planned_capture_count"]):
        raise V21BatchQCError("planned batch capture count differs")
    registry = plan["object_registry"]
    label_by_id = {row["id"]: int(row["label"]) for row in registry.values()}
    capture_root = dataset_root / "raw" / "captures"
    errors: list[str] = []
    sync_spreads, valid_depth_fractions = [], []
    raw_signatures: defaultdict[str, set[str]] = defaultdict(set)
    relation_evidence: list[dict[str, Any]] = []
    valid_capture_count = 0

    for capture_id, capture in expected.items():
        directory = capture_root / capture_id
        meta_path = directory / "capture_meta.json"
        if not meta_path.is_file():
            errors.append(f"missing capture: {capture_id}")
            continue
        try:
            meta = read_json(meta_path)
            if (
                meta.get("protocol_id") != PROTOCOL_ID
                or meta.get("capture_id") != capture_id
                or meta.get("family_id") != capture["family_id"]
                or meta.get("condition") != capture["condition"]
                or int(meta.get("seed", -1)) != int(capture["seed"])
                or meta.get("requested_layout_base_link") != capture["layout"]
            ):
                raise V21BatchQCError("capture identity, seed or layout differs")
            if not EXPECTED_ARTIFACTS.issubset(meta["artifact_sha256"]):
                raise V21BatchQCError("required RGB-D/semantic/camera/TF artifacts not declared")
            for relative, expected_hash in meta["artifact_sha256"].items():
                path = directory / relative
                if not path.is_file() or sha256_file(path) != expected_hash:
                    raise V21BatchQCError(f"artifact hash differs: {relative}")
            rgb = cv2.imread(str(directory / "rgb_original.png"), cv2.IMREAD_COLOR)
            labels = cv2.imread(str(directory / "semantic_instance_labels.png"), cv2.IMREAD_UNCHANGED)
            depth = np.load(directory / "depth_metric.npy", allow_pickle=False)
            if rgb is None or labels is None or rgb.shape[:2] != depth.shape or labels.shape != depth.shape:
                raise V21BatchQCError("registered RGB-depth-semantic shape differs")
            if depth.dtype != np.float32 or labels.dtype != np.uint8:
                raise V21BatchQCError(f"sensor dtype differs: depth={depth.dtype}, labels={labels.dtype}")
            stamps = meta["capture_timestamps"]
            spread = float(stamps.get("max_spread_sec", max(
                float(stamps[key]) for key in ("rgb_sec", "depth_sec", "labels_sec")
            ) - min(float(stamps[key]) for key in ("rgb_sec", "depth_sec", "labels_sec"))))
            if spread > float(plan["sensor_contract"]["max_timestamp_spread_sec"]) + 1e-9:
                raise V21BatchQCError(f"timestamp spread exceeds lock: {spread}")
            visible_labels = {int(value) for value in np.unique(labels)}
            missing = set(capture["required_visible_label_ids"]) - visible_labels
            if missing:
                raise V21BatchQCError(f"required semantic labels missing: {sorted(missing)}")
            valid_depth = np.isfinite(depth) & (depth >= 0.10) & (depth <= 2.0)
            valid_fraction = float(valid_depth.mean())
            if valid_fraction < 0.10 or float(np.std(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY))) < 5.0:
                raise V21BatchQCError("RGB/depth content QC failed")
            realized = meta["realized_view_joint_pose"]
            if len(realized) != 6 or any(
                value is None or not math.isfinite(float(value))
                or abs(float(value) - float(requested)) > 0.025
                for value, requested in zip(realized, capture["view_joint_pose"])
            ):
                raise V21BatchQCError("realized camera pose differs from lock")
            signature = hashlib.sha256(json.dumps(
                meta["artifact_sha256"], sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            raw_signatures[signature].add(capture["family_id"])
            if capture["condition"] == "clean":
                relation_evidence.extend(relation_rows(
                    family_by_id[capture["family_id"]], labels, depth, label_by_id
                ))
            sync_spreads.append(spread)
            valid_depth_fractions.append(valid_fraction)
            valid_capture_count += 1
        except Exception as exc:
            errors.append(f"{capture_id}: {type(exc).__name__}: {exc}")

    duplicates = [sorted(value) for value in raw_signatures.values() if len(value) > 1]
    if duplicates:
        errors.append(f"exact raw signature reused across families: {duplicates[:10]}")
    relation_failures = [item for item in relation_evidence if not item["gate_passed"]]
    if relation_failures:
        errors.append(f"observable relation V2.1 failures: {len(relation_failures)}")
    if valid_capture_count != len(expected):
        errors.append(f"valid capture count differs: {valid_capture_count} != {len(expected)}")

    shutdown_path = dataset_root / "report_assets/checkpoints/shutdown_sequence.json"
    shutdown_ok = shutdown_path.is_file() and read_json(shutdown_path).get("ordered_shutdown_complete") is True
    if not shutdown_ok:
        errors.append("ordered shutdown evidence missing or incomplete")
    evidence = None
    if not errors:
        evidence = write_real_evidence(dataset_root, batch, family_by_id, label_by_id)
    return {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "generated_at_utc": utc_now(),
        "batch_id": batch_id, "passed": not errors,
        "decision": (
            "GO_FULL_V2_1_MATERIALIZATION_QC" if not errors and batch_id == "batch_004"
            else "GO_NEXT_V2_1_DEVELOPMENT_BATCH" if not errors
            else "FIX_CURRENT_V2_1_BATCH_FIRST"
        ),
        "family_count": int(batch["family_count"]),
        "expected_capture_count": len(expected), "valid_capture_count": valid_capture_count,
        "max_timestamp_spread_sec": max(sync_spreads, default=None),
        "mean_valid_depth_fraction": float(np.mean(valid_depth_fractions)) if valid_depth_fractions else None,
        "relation_geometry_version": GEOMETRY_VERSION,
        "relation_checks": len(relation_evidence),
        "relation_not_evaluable_allowed": sum(
            item["status"] == "NOT_EVALUABLE_ALLOWED" for item in relation_evidence
        ),
        "relation_failures": relation_failures,
        "duplicate_cross_family_raw_signatures": duplicates,
        "ordered_shutdown_passed": shutdown_ok,
        "real_capture_qc_evidence": evidence,
        "errors": errors,
        "capture_plan_sha256": sha256_file(plan_path),
        "execution_lock_sha256": sha256_file(lock_path),
        "manifest_sha256": sha256_file(manifest_path),
        "training_performed": False, "calibration_or_test_created": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--capture-plan", default="protocol/dataset_v2_1_development_capture_plan.json")
    parser.add_argument("--execution-lock", default="protocol/dataset_v2_1_development_execution_lock.json")
    parser.add_argument("--manifest", default="protocol/dataset_v2_1_development_manifest.json")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    plan_path = (workspace / args.capture_plan).resolve()
    lock_path = (workspace / args.execution_lock).resolve()
    manifest_path = (workspace / args.manifest).resolve()
    try:
        report = run_qc(
            workspace, dataset_root, plan_path, lock_path, manifest_path, args.batch_id
        )
    except Exception as exc:
        report = {
            "schema_version": 1, "protocol_id": PROTOCOL_ID,
            "generated_at_utc": utc_now(), "batch_id": args.batch_id,
            "passed": False, "decision": "FIX_CURRENT_V2_1_BATCH_FIRST",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "training_performed": False, "calibration_or_test_created": False,
        }
    checkpoint = dataset_root / "report_assets/checkpoints/batch_qc" / f"{args.batch_id}.json"
    atomic_json(checkpoint, report)
    print(
        "DATASET_V2_1_DEVELOPMENT_BATCH_QC "
        f"batch={args.batch_id} passed={report['passed']} decision={report['decision']} "
        f"errors={len(report['errors'])}"
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
