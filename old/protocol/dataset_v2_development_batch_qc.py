#!/usr/bin/env python3
"""Raw RGB-D/semantic QC gate for one locked development capture batch."""

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

from dataset_v2_pilot_validate import relation_satisfied
from wp2_common import read_json, sha256_file, tree_digest, utc_now


PROTOCOL_ID = "roborefer_dataset_v2_development_capture_400"


class BatchQCError(RuntimeError):
    """Raised when observed raw evidence violates a locked batch invariant."""


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
    for value in np.unique(labels):
        label = int(value)
        if label == 0:
            continue
        digest = hashlib.sha256(f"semantic-label|{label}".encode()).digest()
        output[labels == label] = [64 + digest[0] % 192, 64 + digest[1] % 192, 64 + digest[2] % 192]
    return output


def depth_visual(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth >= .10) & (depth <= 2.0)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        lower, upper = np.percentile(depth[valid], [2, 98])
        if upper <= lower:
            upper = lower + 1e-6
        scaled[valid] = np.clip((depth[valid] - lower) * 255.0 / (upper - lower), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(255 - scaled, cv2.COLORMAP_VIRIDIS)


def relation_rows(
    family: dict[str, Any], capture: dict[str, Any], labels: np.ndarray,
    depth: np.ndarray, label_by_id: dict[str, int],
) -> list[dict[str, Any]]:
    spec = family["variant_specs"][0]
    relation = spec["relations"][0]
    if relation == "direct":
        return []
    anchors = [labels == int(label_by_id[value]) for value in spec["anchor_ids"]]
    rows = []
    if spec["answerability_state"] in {"FOUND", "AMBIGUOUS"}:
        for target_id in spec["valid_target_ids"]:
            passed, evidence = relation_satisfied(
                relation, labels == int(label_by_id[target_id]), anchors, depth
            )
            rows.append({
                "family_id": family["family_id"], "capture_id": capture["capture_id"],
                "target_id": target_id, "expected": "SATISFIED", "passed": passed, **evidence,
            })
    elif spec["state_submode"] == "unsatisfied_relation":
        for target_id in spec["candidate_target_ids"]:
            satisfied, evidence = relation_satisfied(
                relation, labels == int(label_by_id[target_id]), anchors, depth
            )
            rows.append({
                "family_id": family["family_id"], "capture_id": capture["capture_id"],
                "target_id": target_id, "expected": "UNSATISFIED", "passed": not satisfied, **evidence,
            })
    return rows


def select_evidence_families(batch: dict[str, Any], family_by_id: dict[str, dict[str, Any]]) -> list[str]:
    selected = []
    for state in ("FOUND", "INSUFFICIENT_EVIDENCE", "AMBIGUOUS", "ABSENT"):
        candidates = [
            family_id for family_id in batch["family_ids"]
            if family_by_id[family_id]["primary_answerability_stratum"] == state
        ]
        candidates.sort(key=lambda value: hashlib.sha256(
            f"development-batch-real-qc|{batch['batch_id']}|{state}|{value}".encode()
        ).hexdigest())
        if candidates:
            selected.append(candidates[0])
    return selected


def write_real_evidence(
    dataset_root: Path, batch: dict[str, Any], captures_by_id: dict[str, dict[str, Any]],
    family_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output_root = dataset_root / "report_assets" / "real_capture_qc" / batch["batch_id"]
    selected = select_evidence_families(batch, family_by_id)
    index = []
    for family_id in selected:
        capture_id = f"{family_id}__clean_capture"
        directory = dataset_root / "raw" / "captures" / capture_id
        destination = output_root / family_id
        destination.mkdir(parents=True, exist_ok=True)
        rgb = cv2.imread(str(directory / "rgb_original.png"), cv2.IMREAD_COLOR)
        labels = cv2.imread(str(directory / "semantic_instance_labels.png"), cv2.IMREAD_UNCHANGED)
        depth = np.load(directory / "depth_metric.npy", allow_pickle=False)
        if rgb is None or labels is None:
            raise BatchQCError(f"cannot build real evidence: {capture_id}")
        shutil.copy2(directory / "rgb_original.png", destination / "rgb.png")
        cv2.imwrite(str(destination / "depth_metric_visualization.png"), depth_visual(depth))
        cv2.imwrite(str(destination / "semantic_instance_labels.png"), semantic_visual(labels))
        hashes = {
            path.name: sha256_file(path) for path in sorted(destination.iterdir()) if path.is_file()
        }
        index.append({
            "family_id": family_id,
            "answerability_state": family_by_id[family_id]["primary_answerability_stratum"],
            "capture_id": capture_id,
            "image_sha256": hashes,
            "source_capture_meta_sha256": sha256_file(directory / "capture_meta.json"),
        })
    index_path = output_root / "REAL_CAPTURE_QC_INDEX.json"
    atomic_json(index_path, {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "batch_id": batch["batch_id"],
        "selection_policy": "one clean family per primary state by locked SHA-256 rank",
        "contains_planned_or_decorative_images": False,
        "selected_family_count": len(selected), "samples": index,
    })
    return {
        "path": str(output_root.relative_to(dataset_root)),
        "selected_family_count": len(selected),
        "index_sha256": sha256_file(index_path),
    }


def run_qc(
    workspace: Path, dataset_root: Path, plan_path: Path, lock_path: Path,
    manifest_path: Path, batch_id: str,
) -> dict[str, Any]:
    plan = read_json(plan_path)
    lock = read_json(lock_path)
    manifest = read_json(manifest_path)
    if lock.get("protocol_id") != PROTOCOL_ID or not lock.get("capture_authorized"):
        raise BatchQCError("execution lock does not authorize this development capture")
    if dataset_root != (workspace / lock["capture_output_root"]).resolve():
        raise BatchQCError("dataset root differs from execution lock")
    if sha256_file(plan_path) != lock["capture_plan_sha256"] or sha256_file(manifest_path) != lock["manifest_sha256"]:
        raise BatchQCError("manifest or capture plan differs from execution lock")
    for relative, expected in lock["locked_artifact_sha256"].items():
        path = workspace / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise BatchQCError(f"execution-locked artifact changed: {relative}")
    if tree_digest(workspace, "datasets/roborefer_dataset_v2_pilot_only_20260821") != lock["pilot_tree_digest"]:
        raise BatchQCError("pilot tree changed after development authorization")

    batches = {item["batch_id"]: item for item in plan["batches"]}
    if batch_id not in batches:
        raise BatchQCError(f"unknown batch ID: {batch_id}")
    batch = batches[batch_id]
    family_by_id = {item["family_id"]: item for item in manifest["families"]}
    expected = {
        item["capture_id"]: item for item in plan["captures"] if item["batch_id"] == batch_id
    }
    if len(expected) != int(batch["planned_capture_count"]):
        raise BatchQCError("batch plan count differs")
    capture_root = dataset_root / "raw" / "captures"
    captures_by_id: dict[str, dict[str, Any]] = {}
    errors = []
    sync_spreads = []
    valid_depth_fractions = []
    capture_signatures: defaultdict[str, set[str]] = defaultdict(set)
    relation_evidence = []
    registry = plan["object_registry"]
    label_by_id = {value["id"]: int(value["label"]) for value in registry.values()}

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
                raise BatchQCError("identity, seed or requested layout differs")
            for relative, declared in meta["artifact_sha256"].items():
                path = directory / relative
                if not path.is_file() or sha256_file(path) != declared:
                    raise BatchQCError(f"artifact hash differs: {relative}")
            rgb = cv2.imread(str(directory / "rgb_original.png"), cv2.IMREAD_COLOR)
            labels = cv2.imread(str(directory / "semantic_instance_labels.png"), cv2.IMREAD_UNCHANGED)
            depth = np.load(directory / "depth_metric.npy", allow_pickle=False)
            if rgb is None or labels is None or rgb.shape[:2] != depth.shape or labels.shape != depth.shape:
                raise BatchQCError("registered RGB-depth-label shape differs")
            if depth.dtype != np.float32 or labels.dtype != np.uint8:
                raise BatchQCError(f"sensor dtype differs: depth={depth.dtype}, labels={labels.dtype}")
            stamps = meta["capture_timestamps"]
            spread = max(float(stamps[key]) for key in ("rgb_sec", "depth_sec", "labels_sec")) - min(
                float(stamps[key]) for key in ("rgb_sec", "depth_sec", "labels_sec")
            )
            if spread > float(plan["sensor_contract"]["max_timestamp_spread_sec"]) + 1e-9:
                raise BatchQCError(f"timestamp spread exceeds lock: {spread}")
            visible = {int(value) for value in np.unique(labels)}
            missing_labels = set(capture["required_visible_label_ids"]) - visible
            if missing_labels:
                raise BatchQCError(f"required evidence labels absent: {sorted(missing_labels)}")
            valid_depth = np.isfinite(depth) & (depth >= .10) & (depth <= 2.0)
            valid_fraction = float(np.count_nonzero(valid_depth) / valid_depth.size)
            if valid_fraction < .10 or float(np.std(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY))) < 5.0:
                raise BatchQCError("raw RGB/depth content QC failed")
            realized = meta["realized_view_joint_pose"]
            if len(realized) != 6 or any(
                value is None or not math.isfinite(float(value))
                or abs(float(value) - float(requested)) > .025
                for value, requested in zip(realized, capture["view_joint_pose"])
            ):
                raise BatchQCError("realized wrist-camera pose differs from plan")
            signature = json.dumps(meta["artifact_sha256"], sort_keys=True, separators=(",", ":"))
            capture_signatures[hashlib.sha256(signature.encode()).hexdigest()].add(capture["family_id"])
            sync_spreads.append(spread)
            valid_depth_fractions.append(valid_fraction)
            captures_by_id[capture_id] = meta
            if capture["condition"] == "clean":
                relation_evidence.extend(relation_rows(
                    family_by_id[capture["family_id"]], capture, labels, depth, label_by_id
                ))
        except Exception as exc:
            errors.append(f"{capture_id}: {type(exc).__name__}: {exc}")

    duplicate_cross_family = [
        value for value in capture_signatures.values() if len(value) > 1
    ]
    if duplicate_cross_family:
        errors.append(f"exact raw RGB-depth-label signature reused across families: {duplicate_cross_family}")
    relation_failures = [item for item in relation_evidence if not item["passed"]]
    if relation_failures:
        errors.append(f"captured metric/projective relation failures: {len(relation_failures)}")
    if len(captures_by_id) != len(expected):
        errors.append(f"valid capture count differs: {len(captures_by_id)} != {len(expected)}")

    evidence = None
    if not errors:
        evidence = write_real_evidence(dataset_root, batch, captures_by_id, family_by_id)
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "batch_id": batch_id,
        "passed": not errors,
        "decision": "GO_NEXT_DEVELOPMENT_BATCH" if not errors and batch_id != "batch_004" else (
            "GO_FULL_DEVELOPMENT_MATERIALIZATION_QC" if not errors else "FIX_CURRENT_BATCH_FIRST"
        ),
        "family_count": int(batch["family_count"]),
        "expected_capture_count": len(expected),
        "valid_capture_count": len(captures_by_id),
        "max_timestamp_spread_sec": max(sync_spreads, default=None),
        "mean_valid_depth_fraction": float(np.mean(valid_depth_fractions)) if valid_depth_fractions else None,
        "relation_checks": len(relation_evidence),
        "relation_failures": relation_failures,
        "duplicate_cross_family_raw_signatures": duplicate_cross_family,
        "real_capture_qc_evidence": evidence,
        "errors": errors,
        "capture_plan_sha256": sha256_file(plan_path),
        "execution_lock_sha256": sha256_file(lock_path),
        "manifest_sha256": sha256_file(manifest_path),
        "training_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--capture-plan", default="protocol/dataset_v2_development_capture_plan.json")
    parser.add_argument("--execution-lock", default="protocol/dataset_v2_development_execution_lock.json")
    parser.add_argument("--manifest", default="protocol/dataset_v2_development_manifest.json")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    plan_path = (workspace / args.capture_plan).resolve()
    lock_path = (workspace / args.execution_lock).resolve()
    manifest_path = (workspace / args.manifest).resolve()
    try:
        report = run_qc(workspace, dataset_root, plan_path, lock_path, manifest_path, args.batch_id)
    except Exception as exc:
        report = {
            "schema_version": 1, "protocol_id": PROTOCOL_ID, "generated_at_utc": utc_now(),
            "batch_id": args.batch_id, "passed": False, "decision": "FIX_CURRENT_BATCH_FIRST",
            "errors": [f"{type(exc).__name__}: {exc}"], "training_performed": False,
            "capture_plan_sha256": sha256_file(plan_path) if plan_path.is_file() else None,
            "execution_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        }
    checkpoint = dataset_root / "report_assets" / "checkpoints" / "batch_qc" / f"{args.batch_id}.json"
    atomic_json(checkpoint, report)
    print(
        "DATASET_V2_DEVELOPMENT_BATCH_QC "
        f"batch={args.batch_id} passed={report['passed']} decision={report['decision']} "
        f"errors={len(report['errors'])}"
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
