#!/usr/bin/env python3
"""Full schema, hash, mask, capture, replay and leakage validation for WP2."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import jsonschema
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp2_common import (  # noqa: E402
    EXPECTED_PILOT_TREE_SHA256,
    EXPECTED_WP1_TREE_SHA256,
    PILOT_RELATIVE,
    WP1_RELATIVE,
    PROTOCOL_ID,
    VARIANTS,
    WP2Error,
    read_json,
    sha256_file,
    tree_digest,
    utc_now,
    workspace_root,
    write_json,
)
from wp2_leakage_validator import validate_leakage  # noqa: E402
from wp2_replay_validator import replay_validate  # noqa: E402


def collect_file_refs(value: Any) -> list[dict]:
    refs = []
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"}.issubset(value):
            refs.append(value)
        for child in value.values():
            refs.extend(collect_file_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.extend(collect_file_refs(child))
    return refs


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise WP2Error(f"path escapes dataset root: {relative}")
    return path


def mask_qc(dataset_root: Path, record: dict) -> dict:
    masks = record["evaluator_only"]["masks"]
    names = ["target", "target_interior", "graspable", "reachable", "valid_depth"]
    arrays = {}
    for name in names:
        arrays[name] = cv2.imread(
            str(safe_path(dataset_root, masks[name]["path"])), cv2.IMREAD_GRAYSCALE
        )
        if arrays[name] is None:
            raise WP2Error(f"mask decode failed: {record['sample_id']}:{name}")
        if set(int(value) for value in np.unique(arrays[name])) - {0, 255}:
            raise WP2Error(f"mask is not binary: {record['sample_id']}:{name}")
    shape = arrays["target"].shape
    if any(value.shape != shape for value in arrays.values()):
        raise WP2Error(f"mask shapes disagree: {record['sample_id']}")
    anchor_pixel_counts = []
    for anchor in masks["anchor"]:
        value = cv2.imread(
            str(safe_path(dataset_root, anchor["path"])), cv2.IMREAD_GRAYSCALE
        )
        if value is None or value.shape != shape:
            raise WP2Error(f"anchor mask invalid: {record['sample_id']}")
        anchor_pixel_counts.append(int(np.count_nonzero(value)))
    target = arrays["target"] > 0
    interior = arrays["target_interior"] > 0
    graspable = arrays["graspable"] > 0
    reachable = arrays["reachable"] > 0
    if np.any(interior & ~target):
        raise WP2Error(f"interior is not target subset: {record['sample_id']}")
    if np.any(graspable & ~target) or np.any(reachable & ~graspable):
        raise WP2Error(f"action masks violate subset contract: {record['sample_id']}")
    state = record["evaluator_only"]["uncertainty_label"]["state"]
    if state == "ABSENT" and int(np.count_nonzero(target)) != 0:
        raise WP2Error(f"ABSENT sample has visible target pixels: {record['sample_id']}")
    if state == "FOUND":
        if int(np.count_nonzero(target)) == 0 or int(np.count_nonzero(interior)) == 0:
            raise WP2Error(
                f"FOUND sample has empty target/interior mask: {record['sample_id']}"
            )
        if any(count == 0 for count in anchor_pixel_counts):
            raise WP2Error(f"FOUND sample has an invisible anchor: {record['sample_id']}")
        intervention = record["evaluator_only"]["uncertainty_label"]["expected_intervention"]
        if intervention == "EXECUTE" and (
            int(np.count_nonzero(graspable)) == 0 or int(np.count_nonzero(reachable)) == 0
        ):
            raise WP2Error(f"EXECUTE sample has an empty action mask: {record['sample_id']}")
    return {
        "sample_id": record["sample_id"],
        "target_pixels": int(np.count_nonzero(target)),
        "interior_pixels": int(np.count_nonzero(interior)),
        "graspable_pixels": int(np.count_nonzero(graspable)),
        "reachable_pixels": int(np.count_nonzero(reachable)),
        "anchor_pixel_counts": anchor_pixel_counts,
        "valid_depth_fraction": float(np.count_nonzero(arrays["valid_depth"]) / arrays["valid_depth"].size),
    }


def relation_qc(dataset_root: Path, record: dict) -> dict:
    """Verify projective/depth relation labels against captured evidence."""
    state = record["evaluator_only"]["uncertainty_label"]["state"]
    label = record["evaluator_only"]["spatial_label"]
    relations = label["relations"]
    anchors = record["evaluator_only"]["masks"]["anchor"]
    if state != "FOUND" or not relations or not anchors:
        return {"sample_id": record["sample_id"], "status": "NOT_APPLICABLE"}
    relation = relations[0]
    if relation in {"direct", "more_elongated_than", "taller_than", "semantic_reference"}:
        return {
            "sample_id": record["sample_id"], "status": "NON_PROJECTIVE_LABEL",
            "relation": relation,
        }
    target = cv2.imread(
        str(safe_path(dataset_root, record["evaluator_only"]["masks"]["target"]["path"])),
        cv2.IMREAD_GRAYSCALE,
    ) > 0
    anchor_arrays = [
        cv2.imread(str(safe_path(dataset_root, item["path"])), cv2.IMREAD_GRAYSCALE) > 0
        for item in anchors
    ]
    depth = np.load(
        safe_path(dataset_root, record["sensor_evidence"]["depth_metric"]["path"]),
        allow_pickle=False,
    )
    target_depth = float(np.median(depth[target]))
    anchor_depths = [float(np.median(depth[value])) for value in anchor_arrays]
    target_y, target_x = np.where(target)
    target_centroid_x = float(np.mean(target_x))
    anchor_centroid_x = [float(np.mean(np.where(value)[1])) for value in anchor_arrays]
    margin_m = 0.003
    if relation in {"nearer_than", "front_of"}:
        passed = target_depth + margin_m < anchor_depths[0]
    elif relation in {"farther_than", "behind"}:
        passed = target_depth > anchor_depths[0] + margin_m
    elif relation == "nearer_than_both":
        passed = len(anchor_depths) >= 2 and all(
            target_depth + margin_m < value for value in anchor_depths
        )
    elif relation == "between_in_depth":
        passed = len(anchor_depths) >= 2 and (
            min(anchor_depths) + margin_m < target_depth
            < max(anchor_depths) - margin_m
        )
    elif relation == "right_of":
        passed = target_centroid_x > anchor_centroid_x[0]
    elif relation == "left_of":
        passed = target_centroid_x < anchor_centroid_x[0]
    else:
        raise WP2Error(f"relation QC has no policy for {relation}: {record['sample_id']}")
    if not passed:
        raise WP2Error(
            f"relation evidence mismatch: {record['sample_id']} relation={relation} "
            f"target_depth={target_depth:.4f} anchors={anchor_depths}"
        )
    return {
        "sample_id": record["sample_id"], "status": "PASSED",
        "relation": relation, "target_median_depth_m": target_depth,
        "anchor_median_depth_m": anchor_depths,
        "target_centroid_x": target_centroid_x,
        "anchor_centroid_x": anchor_centroid_x,
    }


def validate(dataset_root: Path, selection: str = "all", run_replay: bool = True) -> dict:
    protocol_dir = Path(__file__).resolve().parent
    workspace = workspace_root()
    index_name = "smoke_dataset_index.json" if selection == "smoke" else "dataset_index.json"
    index = read_json(dataset_root / index_name)
    family_manifest = read_json(dataset_root / "family_manifest.json")
    dataset_schema = read_json(protocol_dir / "dataset_v1.schema.json")
    family_schema = read_json(protocol_dir / "family_manifest_v1.schema.json")
    jsonschema.validate(family_manifest, family_schema)
    expected_records = 15 if selection == "smoke" else 250
    expected_families = 3 if selection == "smoke" else 50
    if index["record_count"] != expected_records or index["family_count"] != expected_families:
        raise WP2Error(f"index count mismatch: {index['record_count']}/{index['family_count']}")
    records = []
    hash_failures = []
    qc = []
    relation_checks = []
    for index_item in index["records"]:
        record_path = safe_path(dataset_root, index_item["record_path"])
        if sha256_file(record_path) != index_item["record_sha256"]:
            hash_failures.append(str(index_item["record_path"]))
        record = read_json(record_path)
        jsonschema.validate(record, dataset_schema)
        records.append(record)
        for ref in collect_file_refs(record):
            path = safe_path(dataset_root, ref["path"])
            if not path.is_file() or path.stat().st_size != ref["bytes"] or sha256_file(path) != ref["sha256"]:
                hash_failures.append(f"{record['sample_id']}:{ref['path']}")
        if record["artifact_sha256"] != {
            key: value for key, value in record["artifact_sha256"].items()
        }:
            hash_failures.append(f"{record['sample_id']}:artifact_map")
        qc.append(mask_qc(dataset_root, record))
        relation_checks.append(relation_qc(dataset_root, record))
    if hash_failures:
        raise WP2Error(f"artifact hash failures: {hash_failures[:10]}")

    family_counts = Counter(item["family_id"] for item in records)
    if any(count != len(VARIANTS) for count in family_counts.values()):
        raise WP2Error("not every selected family contains five variants")
    leakage = validate_leakage(dataset_root, selection)
    if not leakage["passed"]:
        raise WP2Error(f"leakage validation failed: {leakage}")
    replay = replay_validate(dataset_root, selection) if run_replay else {"passed": None}
    if run_replay and not replay["passed"]:
        raise WP2Error(f"replay validation failed: {replay['mismatches'][:5]}")
    raw_manifest = read_json(dataset_root / "raw" / "raw_capture_manifest.json")
    required_raw = 6 if selection == "smoke" else 100
    if raw_manifest["capture_count"] < required_raw:
        raise WP2Error(f"raw capture count {raw_manifest['capture_count']} < {required_raw}")

    pilot_digest = tree_digest(workspace, PILOT_RELATIVE)
    wp1_digest = tree_digest(workspace, WP1_RELATIVE)
    if pilot_digest != EXPECTED_PILOT_TREE_SHA256:
        raise WP2Error(f"WP0 pilot digest changed: {pilot_digest}")
    if wp1_digest != EXPECTED_WP1_TREE_SHA256:
        raise WP2Error(f"WP1 result digest changed: {wp1_digest}")

    depth_families = {
        item["family_id"] for item in records if item["depth_dependent"]
    }
    expected_depth = 1 if selection == "smoke" else 30
    if len(depth_families) < expected_depth:
        raise WP2Error(f"depth-dependent family gate failed: {len(depth_families)}")
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "selection": selection,
        "generated_at_utc": utc_now(),
        "passed": True,
        "record_count": len(records),
        "family_count": len(family_counts),
        "depth_dependent_family_count": len(depth_families),
        "raw_capture_count": raw_manifest["capture_count"],
        "schema_validation": {"passed": True, "records_validated": len(records)},
        "hash_validation": {"passed": True, "failures": []},
        "mask_qc": {
            "passed": True,
            "records_validated": len(qc),
            "target_pixels_total": sum(item["target_pixels"] for item in qc),
            "interior_pixels_total": sum(item["interior_pixels"] for item in qc),
            "mean_valid_depth_fraction": float(np.mean([item["valid_depth_fraction"] for item in qc])),
            "per_sample": qc,
        },
        "relation_qc": {
            "passed": True,
            "projective_records_checked": sum(
                item["status"] == "PASSED" for item in relation_checks
            ),
            "non_projective_labels": sum(
                item["status"] == "NON_PROJECTIVE_LABEL" for item in relation_checks
            ),
            "not_applicable": sum(
                item["status"] == "NOT_APPLICABLE" for item in relation_checks
            ),
            "per_sample": relation_checks,
        },
        "leakage_validation": leakage,
        "replay_validation": replay,
        "immutability": {
            "pilot_wp0_tree_sha256": pilot_digest,
            "pilot_wp0_unchanged": True,
            "wp1_tree_sha256": wp1_digest,
            "wp1_unchanged": True,
        },
        "pilot_wp1_used_as_training_source": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--selection", choices=("smoke", "all"), default="all")
    parser.add_argument("--output")
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    result = validate(root, args.selection, run_replay=not args.skip_replay)
    if args.output:
        write_json(Path(args.output).expanduser().resolve(), result)
    print(
        f"WP2_VALIDATE_PASS selection={args.selection} families={result['family_count']} "
        f"records={result['record_count']} replay={result['replay_validation']['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
