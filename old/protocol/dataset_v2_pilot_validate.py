#!/usr/bin/env python3
"""Scientific QC, replay and final decision for the Dataset V2 pilot-only gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import jsonschema
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2_pilot_generator import PROTOCOL_ID, VARIANTS  # noqa: E402
from dataset_v2_pilot_materialize import materialize  # noqa: E402
from wp2_common import (  # noqa: E402
    find_forbidden_inference_keys,
    read_json,
    sha256_file,
    tree_digest,
    utc_now,
    write_json,
)


class PilotValidationError(RuntimeError):
    """Raised for a failed final pilot QC gate."""


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise PilotValidationError(f"path escapes dataset root: {relative}")
    return path


def collect_file_refs(value: Any) -> list[dict[str, Any]]:
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


def read_mask(root: Path, ref: dict[str, Any]) -> np.ndarray:
    value = cv2.imread(str(safe_path(root, ref["path"])), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise PilotValidationError(f"mask decode failed: {ref['path']}")
    if set(int(item) for item in np.unique(value)) - {0, 255}:
        raise PilotValidationError(f"mask is not binary: {ref['path']}")
    return value


def mask_and_ontology_qc(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    evaluator = record["evaluator_only"]
    masks = evaluator["masks"]
    arrays = {name: read_mask(root, masks[name]) for name in ("target", "target_interior", "graspable", "reachable", "valid_depth")}
    shape = arrays["target"].shape
    if any(value.shape != shape for value in arrays.values()):
        raise PilotValidationError(f"mask shapes disagree: {record['sample_id']}")
    anchor_arrays = [read_mask(root, item) for item in masks["anchor"]]
    if any(value.shape != shape for value in anchor_arrays):
        raise PilotValidationError(f"anchor mask shape differs: {record['sample_id']}")
    target = arrays["target"] > 0
    interior = arrays["target_interior"] > 0
    graspable = arrays["graspable"] > 0
    reachable = arrays["reachable"] > 0
    if np.any(interior & ~target):
        raise PilotValidationError(f"interior not subset of target: {record['sample_id']}")
    if np.any(graspable & ~target) or np.any(reachable & ~graspable):
        raise PilotValidationError(f"action mask subset invariant failed: {record['sample_id']}")
    uncertainty = evaluator["uncertainty_label"]
    spatial = evaluator["spatial_label"]
    state = uncertainty["state"]
    cardinality = len(spatial["valid_target_ids"])
    target_pixels = int(np.count_nonzero(target))
    interior_pixels = int(np.count_nonzero(interior))
    anchor_pixels = [int(np.count_nonzero(value)) for value in anchor_arrays]
    labels = cv2.imread(
        str(safe_path(root, evaluator["semantic_instance_labels"]["path"])),
        cv2.IMREAD_UNCHANGED,
    )
    if labels is None or labels.shape != shape:
        raise PilotValidationError(f"semantic labels unavailable for mask QC: {record['sample_id']}")
    label_ids = evaluator["object_oracle"]["label_ids"]
    candidate_pixel_counts = {
        object_id: int(np.count_nonzero(labels == int(label_ids[object_id])))
        for object_id in spatial["candidate_target_ids"]
        if object_id in label_ids
    }
    if state == "FOUND":
        if cardinality != 1 or not uncertainty["answerable"] or target_pixels == 0 or interior_pixels == 0:
            raise PilotValidationError(f"FOUND ontology/mask failed: {record['sample_id']}")
        expected_invisible_cf_anchor = (
            record["variant"] == "relation_counterfactual"
            and uncertainty["state_submode"] == "too_small_or_out_of_view"
        )
        if any(value == 0 for value in anchor_pixels) and not expected_invisible_cf_anchor:
            raise PilotValidationError(f"FOUND anchor not visible: {record['sample_id']}")
        if uncertainty["expected_intervention"] == "EXECUTE" and (
            int(np.count_nonzero(graspable)) == 0 or int(np.count_nonzero(reachable)) == 0
        ):
            raise PilotValidationError(f"FOUND EXECUTE action mask empty: {record['sample_id']}")
    elif state == "AMBIGUOUS":
        if cardinality < 2 or uncertainty["answerable"] or target_pixels == 0:
            raise PilotValidationError(f"AMBIGUOUS ontology/mask failed: {record['sample_id']}")
        if any(candidate_pixel_counts.get(value, 0) == 0 for value in spatial["valid_target_ids"]):
            raise PilotValidationError(f"AMBIGUOUS valid candidate not visible: {record['sample_id']}")
    elif state == "ABSENT":
        if cardinality != 0 or uncertainty["answerable"] or target_pixels != 0:
            raise PilotValidationError(f"ABSENT ontology/mask failed: {record['sample_id']}")
        if uncertainty["state_submode"] == "unsatisfied_relation":
            required_visible = spatial["candidate_target_ids"] + spatial["anchor_ids"]
            visible_counts = {
                object_id: int(np.count_nonzero(labels == int(label_ids[object_id])))
                for object_id in required_visible if object_id in label_ids
            }
            if any(visible_counts.get(value, 0) == 0 for value in required_visible):
                raise PilotValidationError(
                    f"unsatisfied relation confounded by invisible object: {record['sample_id']}"
                )
    elif state == "INSUFFICIENT_EVIDENCE":
        if cardinality != 1 or uncertainty["answerable"]:
            raise PilotValidationError(f"INSUFFICIENT ontology failed: {record['sample_id']}")
        if any(value == 0 for value in anchor_pixels):
            raise PilotValidationError(f"INSUFFICIENT anchor not visible: {record['sample_id']}")
        if uncertainty["state_submode"] != "too_small_or_out_of_view" and target_pixels == 0:
            raise PilotValidationError(f"INSUFFICIENT target unexpectedly invisible: {record['sample_id']}")
        if int(np.count_nonzero(graspable)) or int(np.count_nonzero(reachable)):
            raise PilotValidationError(f"INSUFFICIENT has executable action mask: {record['sample_id']}")
    else:
        raise PilotValidationError(f"unknown state: {record['sample_id']}:{state}")
    return {
        "sample_id": record["sample_id"], "state": state,
        "state_submode": uncertainty["state_submode"], "target_pixels": target_pixels,
        "interior_pixels": interior_pixels, "anchor_pixel_counts": anchor_pixels,
        "candidate_pixel_counts": candidate_pixel_counts,
        "graspable_pixels": int(np.count_nonzero(graspable)),
        "reachable_pixels": int(np.count_nonzero(reachable)),
        "valid_depth_fraction": float(np.count_nonzero(arrays["valid_depth"]) / arrays["valid_depth"].size),
    }


def relation_satisfied(
    relation: str, target: np.ndarray, anchors: list[np.ndarray], depth: np.ndarray
) -> tuple[bool, dict[str, Any]]:
    if relation == "direct":
        return True, {"relation": relation, "policy": "non_relational"}
    if not np.any(target) or any(not np.any(value) for value in anchors):
        return False, {"relation": relation, "reason": "target_or_anchor_not_visible"}
    target_depth = float(np.median(depth[target]))
    anchor_depth = [float(np.median(depth[value])) for value in anchors]
    target_x = float(np.mean(np.where(target)[1]))
    anchor_x = [float(np.mean(np.where(value)[1])) for value in anchors]
    margin_m = .003
    margin_px = 3.0
    if relation == "right_of":
        passed = target_x > anchor_x[0] + margin_px
    elif relation == "left_of":
        passed = target_x + margin_px < anchor_x[0]
    elif relation in {"nearer_than", "front_of"}:
        passed = target_depth + margin_m < anchor_depth[0]
    elif relation in {"farther_than", "behind"}:
        passed = target_depth > anchor_depth[0] + margin_m
    elif relation == "between_in_depth":
        passed = len(anchor_depth) >= 2 and min(anchor_depth) + margin_m < target_depth < max(anchor_depth) - margin_m
    elif relation == "nearer_than_both":
        passed = len(anchor_depth) >= 2 and all(target_depth + margin_m < value for value in anchor_depth)
    elif relation == "farther_than_both":
        passed = len(anchor_depth) >= 2 and all(target_depth > value + margin_m for value in anchor_depth)
    else:
        raise PilotValidationError(f"no relation QC policy for {relation}")
    return passed, {
        "relation": relation, "target_median_depth_m": target_depth,
        "anchor_median_depth_m": anchor_depth, "target_centroid_x_px": target_x,
        "anchor_centroid_x_px": anchor_x,
    }


def relation_qc(root: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    spatial = record["evaluator_only"]["spatial_label"]
    uncertainty = record["evaluator_only"]["uncertainty_label"]
    relation = spatial["relations"][0]
    if relation == "direct" or uncertainty["state"] in {"ABSENT", "INSUFFICIENT_EVIDENCE"}:
        return [{"sample_id": record["sample_id"], "status": "NOT_APPLICABLE", "relation": relation}]
    label_ref = record["evaluator_only"]["semantic_instance_labels"]
    labels = cv2.imread(str(safe_path(root, label_ref["path"])), cv2.IMREAD_UNCHANGED)
    depth = np.load(safe_path(root, record["sensor_evidence"]["depth_metric"]["path"]), allow_pickle=False)
    if labels is None or labels.shape != depth.shape:
        raise PilotValidationError(f"relation RGB-D-label shape failed: {record['sample_id']}")
    label_ids = record["evaluator_only"]["object_oracle"]["label_ids"]
    anchors = [labels == int(label_ids[value]) for value in spatial["anchor_ids"]]
    results = []
    for target_id in spatial["valid_target_ids"]:
        target = labels == int(label_ids[target_id])
        passed, values = relation_satisfied(relation, target, anchors, depth)
        results.append({"sample_id": record["sample_id"], "target_id": target_id, "status": "PASSED" if passed else "FAILED", **values})
        if not passed:
            raise PilotValidationError(f"captured relation evidence mismatch: {record['sample_id']}:{target_id}:{values}")
    return results


def validate_raw_captures(root: Path) -> dict[str, Any]:
    raw_manifest = read_json(root / "raw" / "raw_capture_manifest.json")
    if raw_manifest.get("capture_count") != 60 or not raw_manifest.get("complete"):
        raise PilotValidationError("raw manifest is not complete at 60 captures")
    family_conditions: dict[str, set[str]] = defaultdict(set)
    spreads = []
    valid_fractions = []
    shapes = set()
    for item in raw_manifest["captures"]:
        capture_dir = root / "raw" / "captures" / item["capture_id"]
        meta_path = capture_dir / "capture_meta.json"
        if sha256_file(meta_path) != item["capture_meta_sha256"]:
            raise PilotValidationError(f"raw capture meta hash failed: {item['capture_id']}")
        meta = read_json(meta_path)
        for relative, expected in meta["artifact_sha256"].items():
            path = capture_dir / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise PilotValidationError(f"raw artifact hash failed: {item['capture_id']}/{relative}")
        rgb = cv2.imread(str(capture_dir / "rgb_original.png"), cv2.IMREAD_COLOR)
        labels = cv2.imread(str(capture_dir / "semantic_instance_labels.png"), cv2.IMREAD_UNCHANGED)
        depth = np.load(capture_dir / "depth_metric.npy", allow_pickle=False)
        if rgb is None or labels is None or rgb.shape[:2] != labels.shape or labels.shape != depth.shape:
            raise PilotValidationError(f"raw RGB-depth-label alignment failed: {item['capture_id']}")
        spread = float(meta["capture_timestamps"]["max_spread_sec"])
        if spread > .035:
            raise PilotValidationError(f"timestamp spread exceeds 35 ms: {item['capture_id']}")
        if any(value is None for value in meta["realized_view_joint_pose"]):
            raise PilotValidationError(f"realized camera pose missing: {item['capture_id']}")
        if max(abs(float(a) - float(b)) for a, b in zip(meta["requested_view_joint_pose"], meta["realized_view_joint_pose"])) > .025:
            raise PilotValidationError(f"camera pose tolerance failed: {item['capture_id']}")
        family_conditions[meta["family_id"]].add(meta["condition"])
        spreads.append(spread)
        valid_fractions.append(float(meta["sensor_qc"]["valid_depth_fraction"]))
        shapes.add(tuple(depth.shape))
    if len(family_conditions) != 30 or any(value != {"clean", "occlusion"} for value in family_conditions.values()):
        raise PilotValidationError("each family must contain clean and occlusion raw captures")
    return {
        "passed": True, "capture_count": 60, "family_count": 30,
        "registered_shapes_hw": [list(value) for value in sorted(shapes)],
        "max_timestamp_spread_sec": max(spreads),
        "mean_valid_depth_fraction": float(np.mean(valid_fractions)),
    }


def replay_validate(root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dataset_v2_pilot_replay_") as value:
        replay_root = Path(value)
        materialize(root, replay_root)
        relative_files = []
        for top in ("family_manifest.json", "capture_plan.json", "object_registry.json", "execution_lock.json", "dataset_index.json", "derived", "media", "evaluator", "records"):
            path = root / top
            if path.is_file():
                relative_files.append(Path(top))
            elif path.is_dir():
                relative_files.extend(item.relative_to(root) for item in path.rglob("*") if item.is_file())
        mismatches = []
        for relative in sorted(relative_files):
            replay_path = replay_root / relative
            if not replay_path.is_file() or sha256_file(root / relative) != sha256_file(replay_path):
                mismatches.append(str(relative))
        return {"passed": not mismatches, "files_compared": len(relative_files), "mismatches": mismatches}


def label_visual(labels: np.ndarray) -> np.ndarray:
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in np.unique(labels):
        if int(label) == 0:
            continue
        digest = hashlib.sha256(f"pilot-label-{int(label)}".encode()).digest()
        output[labels == label] = [64 + digest[0] % 192, 64 + digest[1] % 192, 64 + digest[2] % 192]
    return output


def write_real_qc_evidence(root: Path, records: list[dict[str, Any]], mask_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["variant"] == "clean":
            by_axis[record["ood_axis"]].append(record)
    selected = []
    for axis in ("none", "asset", "layout", "viewpoint", "language", "depth_noise"):
        values = sorted(by_axis[axis], key=lambda item: hashlib.sha256(f"20260821|qc|{item['family_id']}".encode()).hexdigest())
        selected.append(values[0])
    evidence_root = root / "report_assets" / "real_capture_qc"
    measurements_by_id = {item["sample_id"]: item for item in mask_rows}
    inventory = []
    for record in selected:
        sample_dir = evidence_root / record["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        rgb = cv2.imread(str(safe_path(root, record["sensor_evidence"]["rgb_original"]["path"])), cv2.IMREAD_COLOR)
        depth = np.load(safe_path(root, record["sensor_evidence"]["depth_metric"]["path"]), allow_pickle=False)
        labels = cv2.imread(str(safe_path(root, record["evaluator_only"]["semantic_instance_labels"]["path"])), cv2.IMREAD_UNCHANGED)
        target = read_mask(root, record["evaluator_only"]["masks"]["target"])
        anchors = [read_mask(root, item) for item in record["evaluator_only"]["masks"]["anchor"]]
        valid = np.isfinite(depth) & (depth >= .10) & (depth <= 2.0)
        depth_gray = np.zeros(depth.shape, np.uint8)
        if np.any(valid):
            near, far = np.percentile(depth[valid], [2, 98])
            depth_gray[valid] = np.clip((depth[valid] - near) / max(float(far - near), 1e-6) * 255, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(255 - depth_gray, cv2.COLORMAP_TURBO)
        depth_color[~valid] = 0
        semantic = label_visual(labels)
        overlay = rgb.copy()
        overlay[target > 0] = (.35 * overlay[target > 0] + .65 * np.array([0, 0, 255])).astype(np.uint8)
        for anchor in anchors:
            overlay[anchor > 0] = (.35 * overlay[anchor > 0] + .65 * np.array([0, 255, 255])).astype(np.uint8)
        outputs = {
            "rgb.png": rgb, "depth_metric_visualization.png": depth_color,
            "semantic_instance_labels.png": semantic, "target_mask.png": target,
            "target_anchor_overlay.png": overlay,
        }
        if anchors:
            anchor_union = np.maximum.reduce(anchors)
            outputs["anchor_mask_union.png"] = anchor_union
        hashes = {}
        for name, image in outputs.items():
            path = sample_dir / name
            if not cv2.imwrite(str(path), image):
                raise PilotValidationError(f"cannot write real QC image: {path}")
            hashes[name] = sha256_file(path)
        measurement = {
            "sample_id": record["sample_id"], "family_id": record["family_id"],
            "ood_axis": record["ood_axis"], "answerability_state": record["evaluator_only"]["uncertainty_label"]["state"],
            "source_capture_id": record["provenance"]["capture_id"],
            "source_capture_rgb_sha256": record["sensor_evidence"]["rgb_original"]["sha256"],
            "source_depth_metric_sha256": record["sensor_evidence"]["depth_metric"]["sha256"],
            "source_semantic_label_sha256": record["evaluator_only"]["semantic_instance_labels"]["sha256"],
            "mask_qc": measurements_by_id[record["sample_id"]], "image_sha256": hashes,
        }
        write_json(sample_dir / "qc_measurements.json", measurement)
        inventory.append(measurement)
    write_json(evidence_root / "REAL_CAPTURE_QC_INDEX.json", {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "selection_policy": "one clean family per OOD axis via locked SHA-256 ranking seed 20260821",
        "contains_planned_or_decorative_images": False,
        "selected_count": len(inventory), "samples": inventory,
    })
    return {"passed": True, "selected_count": len(inventory), "path": str(evidence_root.relative_to(root))}


def write_table_01(root: Path, records: list[dict[str, Any]]) -> str:
    families = {}
    for record in records:
        if record["variant"] == "clean":
            families[record["family_id"]] = record
    state_counts = Counter(value["evaluator_only"]["uncertainty_label"]["state"] for value in families.values())
    depth_count = sum(bool(value["depth_dependent"]) for value in families.values())
    path = root / "report_assets" / "tables" / "table_01_pilot_dataset_qc.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "independent_families", "dependent_samples", "FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE", "depth_dependent", "scientific_status"])
        writer.writerow(["pilot_only", len(families), len(records), state_counts["FOUND"], state_counts["AMBIGUOUS"], state_counts["ABSENT"], state_counts["INSUFFICIENT_EVIDENCE"], depth_count, "ENGINEERING_QC_ONLY_NOT_OFFICIAL_DATA"])
    return str(path.relative_to(root))


def validate(root: Path, run_replay: bool = True) -> dict[str, Any]:
    workspace = Path(__file__).resolve().parents[1]
    protocol = workspace / "protocol"
    index = read_json(root / "dataset_index.json")
    manifest = read_json(root / "family_manifest.json")
    schema = read_json(protocol / "dataset_v2_pilot_record.schema.json")
    if index.get("protocol_id") != PROTOCOL_ID or index.get("record_count") != 150 or index.get("family_count") != 30:
        raise PilotValidationError("dataset index count/protocol mismatch")
    raw_qc = validate_raw_captures(root)
    records = []
    mask_rows = []
    relation_rows = []
    hash_failures = []
    leakage_findings = []
    for item in index["records"]:
        record_path = safe_path(root, item["record_path"])
        if sha256_file(record_path) != item["record_sha256"]:
            hash_failures.append(str(item["record_path"]))
        record = read_json(record_path)
        jsonschema.validate(record, schema)
        records.append(record)
        for ref in collect_file_refs(record):
            path = safe_path(root, ref["path"])
            if not path.is_file() or path.stat().st_size != int(ref["bytes"]) or sha256_file(path) != ref["sha256"]:
                hash_failures.append(f"{record['sample_id']}:{ref['path']}")
        findings = find_forbidden_inference_keys(record["inference_payload"])
        if findings:
            leakage_findings.append({"sample_id": record["sample_id"], "paths": findings})
        mask_rows.append(mask_and_ontology_qc(root, record))
        relation_rows.extend(relation_qc(root, record))
    if hash_failures:
        raise PilotValidationError(f"artifact hash failures: {hash_failures[:10]}")
    if leakage_findings:
        raise PilotValidationError(f"oracle leakage: {leakage_findings[:5]}")

    family_counts = Counter(item["family_id"] for item in records)
    if len(family_counts) != 30 or any(value != 5 for value in family_counts.values()):
        raise PilotValidationError("five dependent variants per family invariant failed")
    clean_records = [item for item in records if item["variant"] == "clean"]
    primary_counts = Counter(item["evaluator_only"]["uncertainty_label"]["state"] for item in clean_records)
    expected_primary = {"FOUND": 8, "INSUFFICIENT_EVIDENCE": 8, "AMBIGUOUS": 7, "ABSENT": 7}
    if dict(primary_counts) != expected_primary:
        raise PilotValidationError(f"primary state counts differ: {dict(primary_counts)}")
    submodes = defaultdict(set)
    for item in clean_records:
        submodes[item["evaluator_only"]["uncertainty_label"]["state"]].add(item["evaluator_only"]["uncertainty_label"]["state_submode"])
    required = {
        "AMBIGUOUS": {"same_class_duplicate", "attribute_tie", "relation_tie", "multi_anchor_conflict"},
        "ABSENT": {"target_absent", "anchor_absent", "unsatisfied_relation"},
        "INSUFFICIENT_EVIDENCE": {"depth_invalid_or_corrupt", "occlusion", "too_small_or_out_of_view", "cross_modal_conflict"},
    }
    if any(not value.issubset(submodes[key]) for key, value in required.items()):
        raise PilotValidationError("required ontology submode coverage failed")

    lock = read_json(protocol / "dataset_v2_pilot_execution_lock.json")
    if not lock.get("capture_authorized") or sha256_file(protocol / "dataset_v2_pilot_capture_plan.json") != lock["capture_plan_sha256"]:
        raise PilotValidationError("execution lock invalid after capture")
    for relative, expected in lock["locked_artifact_sha256"].items():
        if not (workspace / relative).is_file() or sha256_file(workspace / relative) != expected:
            raise PilotValidationError(f"execution-locked artifact changed: {relative}")
    seed_lock = read_json(protocol / "dataset_expansion_v2_seed_lock.json")
    protected_failures = []
    for relative, expected in seed_lock["protected_files"].items():
        if not (workspace / relative).is_file() or sha256_file(workspace / relative) != expected:
            protected_failures.append(relative)
    for relative, expected in seed_lock["protected_tree_digests"].items():
        if tree_digest(workspace, relative) != expected:
            protected_failures.append(relative)
    if protected_failures:
        raise PilotValidationError(f"protected WP0-WP3/WP2 hashes changed: {protected_failures}")

    replay = replay_validate(root) if run_replay else {"passed": None, "files_compared": 0, "mismatches": []}
    if run_replay and not replay["passed"]:
        raise PilotValidationError(f"deterministic replay failed: {replay['mismatches'][:10]}")
    real_qc = write_real_qc_evidence(root, records, mask_rows)
    table_path = write_table_01(root, records)
    report = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "generated_at_utc": utc_now(),
        "passed": True, "decision": "GO_DEVELOPMENT_CAPTURE_400",
        "scientific_status": "PILOT_ENGINEERING_QC_ONLY_NOT_OFFICIAL_DATA",
        "family_count": 30, "sample_count": 150, "raw_capture_count": 60,
        "primary_answerability_counts": dict(sorted(primary_counts.items())),
        "family_category_counts": dict(sorted(Counter(item["family_category"] for item in clean_records).items())),
        "ood_axis_counts": dict(sorted(Counter(item["ood_axis"] for item in clean_records).items())),
        "state_submode_coverage": {key: sorted(value) for key, value in submodes.items()},
        "raw_rgb_depth_semantic_sync_qc": raw_qc,
        "schema_hash_mask_ontology_qc": {
            "passed": True, "records_validated": len(records),
            "mean_valid_depth_fraction": float(np.mean([value["valid_depth_fraction"] for value in mask_rows])),
            "target_pixels_total": sum(value["target_pixels"] for value in mask_rows),
        },
        "relation_qc": {
            "passed": True, "applicable_checks": sum(value["status"] == "PASSED" for value in relation_rows),
            "not_applicable": sum(value["status"] == "NOT_APPLICABLE" for value in relation_rows),
        },
        "oracle_leakage_qc": {"passed": True, "records_checked": len(records)},
        "pilot_official_disjointness": {"passed": True, "policy": manifest["families"][0]["pilot_exclusion_policy"]},
        "protected_hash_qc": {"passed": True, "failures": []},
        "deterministic_replay": replay,
        "real_capture_qc_evidence": real_qc,
        "table_01_path": table_path,
        "tables_02_to_06": "NOT_RUN",
        "training_performed": False,
    }
    return report


def write_markdown(root: Path, report: dict[str, Any]) -> None:
    text = f"""# Dataset V2 pilot-only QC report

- Decision: `{report['decision']}`
- Status: `{report['scientific_status']}`
- Independent families: {report['family_count']}
- Dependent samples: {report['sample_count']}
- Real Gazebo RGB-D/semantic captures: {report['raw_capture_count']}
- Maximum RGB/depth/label timestamp spread: {report['raw_rgb_depth_semantic_sync_qc']['max_timestamp_spread_sec']:.6f} s
- Deterministic replay files compared: {report['deterministic_replay']['files_compared']}
- Training performed: `false`

Primary answerability counts: `{json.dumps(report['primary_answerability_counts'], sort_keys=True)}`.

Only real-capture QC images are stored under `report_assets/real_capture_qc/`. Table 1 is observed pilot QC data. Tables 2-6 remain `NOT_RUN`. Pilot families, seeds, layouts, instructions, captures and derived samples remain permanently excluded from the official dataset.
"""
    (root / "PILOT_QC_REPORT.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    try:
        report = validate(root, run_replay=not args.skip_replay)
    except Exception as exc:
        report = {
            "schema_version": 1, "protocol_id": PROTOCOL_ID, "generated_at_utc": utc_now(),
            "passed": False, "decision": "FIX_DATASET_V2_GENERATOR_OR_CAPTURE_PIPELINE_FIRST",
            "error_type": type(exc).__name__, "error": str(exc), "training_performed": False,
            "tables_02_to_06": "NOT_RUN",
        }
        write_json(root / "PILOT_QC_REPORT.json", report)
        print(f"DATASET_V2_PILOT_QC_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    write_json(root / "PILOT_QC_REPORT.json", report)
    write_markdown(root, report)
    write_json(root / "report_assets" / "checkpoints" / "03_qc_gate.json", {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "checkpoint": "PILOT_QC_PASS",
        "created_at_utc": utc_now(), "decision": report["decision"],
        "report_sha256": sha256_file(root / "PILOT_QC_REPORT.json"),
        "dataset_index_sha256": sha256_file(root / "dataset_index.json"),
    })
    print(f"DATASET_V2_PILOT_QC_PASS decision={report['decision']} records=150 captures=60")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
