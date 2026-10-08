#!/usr/bin/env python3
"""Validate materialized Test-IID and separate inference inputs from evaluator labels."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[3]
TEST_ROOT = ROOT / "new/test_iid"
DATASET_ROOT = TEST_ROOT / "dataset"
PROTOCOL_ROOT = TEST_ROOT / "protocol"
PROTOCOL_ID = "roborefer_dataset_v2_1_test_iid_capture_200"
VARIANTS = [
    "clean", "semantic_counterfactual", "relation_counterfactual",
    "depth_corruption", "occlusion_view_counterfactual",
]
ANSWER_CLASSES = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
SOURCE_CLASSES = ["semantic", "relation", "spatial", "depth", "occlusion"]

sys.path.insert(0, str(ROOT / "old/protocol"))
from pcra_u_development_common import canonical_sha256, parse_relation, sha256_text  # noqa: E402
from wp2_common import read_json, sha256_file, utc_now, write_json  # noqa: E402
from wp3_feature_hook_smoke import forbidden_cache_findings, model_inventory_sha256  # noqa: E402


class TestIIDPrepareError(RuntimeError):
    pass


def safe_dataset_path(relative: str) -> Path:
    path = (DATASET_ROOT / relative).resolve()
    if path != DATASET_ROOT.resolve() and DATASET_ROOT.resolve() not in path.parents:
        raise TestIIDPrepareError(f"path escapes dataset: {relative}")
    return path


def verify_ref(ref: dict[str, Any], label: str) -> Path:
    path = safe_dataset_path(str(ref["path"]))
    if (
        not path.is_file()
        or path.stat().st_size != int(ref["bytes"])
        or sha256_file(path) != ref["sha256"]
    ):
        raise TestIIDPrepareError(f"{label} artifact mismatch: {path}")
    return path


def collect_refs(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"}.issubset(value):
            result.append(value)
        for child in value.values():
            result.extend(collect_refs(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(collect_refs(child))
    return result


def write_immutable(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if read_json(path) != value:
            raise TestIIDPrepareError(f"refusing to replace different manifest: {path}")
        return
    write_json(path, value)


def prepare() -> dict[str, Any]:
    config_path = ROOT / "new/configs/pcrau_target_v2.json"
    config = read_json(config_path)
    index_path = DATASET_ROOT / "dataset_index.json"
    family_path = DATASET_ROOT / "family_manifest.json"
    materialized_path = DATASET_ROOT / "report_assets/checkpoints/02_materialized.json"
    index = read_json(index_path)
    family_manifest = read_json(family_path)
    materialized = read_json(materialized_path)
    if (
        index.get("protocol_id") != PROTOCOL_ID
        or index.get("selection") != "official_independent_test_iid"
        or index.get("eligible_for_test_iid") is not True
        or index.get("eligible_for_training_or_dev_or_calibration") is not False
        or index.get("record_count") != 1000
        or index.get("family_count") != 200
        or materialized.get("record_count") != 1000
    ):
        raise TestIIDPrepareError("materialized Test-IID identity/count/boundary differs")
    families = family_manifest["families"]
    by_family = {row["family_id"]: row for row in families}
    if len(families) != 200 or len(by_family) != 200:
        raise TestIIDPrepareError("expected 200 unique Test-IID families")

    freeze_path = TEST_ROOT / "contracts/best_v2_freeze_lock.json"
    freeze = read_json(freeze_path)
    checkpoint_path = ROOT / freeze["selected_checkpoint"]["path"]
    calibrator_path = ROOT / freeze["frozen_calibrator"]["path"]
    if sha256_file(checkpoint_path) != freeze["selected_checkpoint"]["model_sha256"]:
        raise TestIIDPrepareError("best V2 checkpoint hash differs")
    if sha256_file(calibrator_path) != freeze["frozen_calibrator"]["sha256"]:
        raise TestIIDPrepareError("frozen calibrator hash differs")
    model_root = ROOT / "RoboRefer/models/RoboRefer-2B-SFT"
    model_hash = model_inventory_sha256(model_root)
    calibration_feature_manifest = read_json(ROOT / "old/protocol/pcra_u_calibration_feature_manifest.json")
    if model_hash != calibration_feature_manifest["model_inventory_sha256"]:
        raise TestIIDPrepareError("frozen RoboRefer inventory differs from calibration")

    eval_entries: list[dict[str, Any]] = []
    feature_entries: list[dict[str, Any]] = []
    family_variants: dict[str, list[str]] = defaultdict(list)
    state_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    rgb_hashes, depth_hashes = set(), set()

    for item in index["records"]:
        path = safe_dataset_path(item["record_path"])
        if not path.is_file() or sha256_file(path) != item["record_sha256"]:
            raise TestIIDPrepareError(f"record hash mismatch: {item['sample_id']}")
        record = read_json(path)
        for key in ("sample_id", "family_id", "split", "variant"):
            if record.get(key) != item.get(key):
                raise TestIIDPrepareError(f"record identity mismatch: {item['sample_id']}:{key}")
        if record["split"] != "test_iid" or record["family_id"] not in by_family:
            raise TestIIDPrepareError(f"invalid test row: {item['sample_id']}")
        for ref in collect_refs(record):
            verify_ref(ref, item["sample_id"])
        inference = record["inference_payload"]
        evaluator = record["evaluator_only"]
        if evaluator.get("access_policy") != "never_export_to_inference_payload":
            raise TestIIDPrepareError(f"evaluator policy differs: {item['sample_id']}")
        if forbidden_cache_findings(inference):
            raise TestIIDPrepareError(f"oracle key in inference payload: {item['sample_id']}")
        rgb = inference["rgb_model_input"]
        depth = inference["depth_relative_model_input"]
        rgb_path = verify_ref(rgb, "RGB")
        depth_path = verify_ref(depth, "depth")
        target = evaluator["masks"]["target"]
        interior = evaluator["masks"]["target_interior"]
        target_path = verify_ref(target, "target mask")
        interior_path = verify_ref(interior, "interior mask")
        target_image = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        interior_image = cv2.imread(str(interior_path), cv2.IMREAD_GRAYSCALE)
        if target_image is None or interior_image is None or target_image.shape != (480, 640):
            raise TestIIDPrepareError(f"mask decode failure: {item['sample_id']}")
        uncertainty = evaluator["uncertainty_label"]
        state = uncertainty["state"]
        target_pixels = int((target_image > 0).sum())
        interior_pixels = int((interior_image > 0).sum())
        if state in {"FOUND", "AMBIGUOUS"} and target_pixels == 0:
            raise TestIIDPrepareError(f"empty positive target: {item['sample_id']}")
        if state == "FOUND" and interior_pixels == 0:
            raise TestIIDPrepareError(f"empty FOUND interior: {item['sample_id']}")
        if state == "ABSENT" and target_pixels != 0:
            raise TestIIDPrepareError(f"nonempty ABSENT target: {item['sample_id']}")
        relation_labels = evaluator["spatial_label"]["relations"]
        prompt = inference["prompt"]
        if len(relation_labels) != 1 or parse_relation(prompt) != relation_labels[0]:
            raise TestIIDPrepareError(f"prompt/relation mismatch: {item['sample_id']}")
        anchors = []
        for anchor in evaluator["masks"]["anchor"]:
            anchor_path = verify_ref(anchor, "anchor mask")
            anchors.append({
                "path": str(anchor_path.relative_to(DATASET_ROOT)),
                "sha256": anchor["sha256"], "bytes": anchor["bytes"],
            })
        feature_key = canonical_sha256({
            "rgb_sha256": rgb["sha256"],
            "depth_sha256": depth["sha256"],
            "model_inventory_sha256": model_hash,
            "feature_transform": read_json(ROOT / "old/protocol/pcra_u_calibration_config.json")["feature_input"],
        })
        feature_input = {
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "rgb_path": str(rgb_path.relative_to(ROOT)),
            "rgb_sha256": rgb["sha256"],
            "depth_path": str(depth_path.relative_to(ROOT)),
            "depth_sha256": depth["sha256"],
            "feature_key": feature_key,
        }
        eval_entries.append({
            "sample_id": item["sample_id"],
            "family_id": item["family_id"],
            "split": "test_iid",
            "variant": item["variant"],
            "record_path": item["record_path"],
            "record_sha256": item["record_sha256"],
            "feature_input": feature_input,
            "supervision": {
                "access_policy": "TEST_IID_EVALUATOR_ONLY_NEVER_MODEL_INPUT",
                "answerability_state": state,
                "answerability_index": ANSWER_CLASSES.index(state),
                "source_labels": uncertainty["sources"],
                "source_multihot": [int(name in uncertainty["sources"]) for name in SOURCE_CLASSES],
                "target_mask_path": str(target_path.relative_to(DATASET_ROOT)),
                "target_mask_sha256": target["sha256"],
                "target_interior_mask_path": str(interior_path.relative_to(DATASET_ROOT)),
                "target_interior_mask_sha256": interior["sha256"],
                "anchor_masks": anchors,
            },
            "audit_only": {
                "relation": relation_labels[0],
                "family_category": record["family_category"],
                "target_object_group": record.get("target_object_group"),
                "depth_dependent": record["depth_dependent"],
                "state_submode": uncertainty["state_submode"],
                "expected_intervention": uncertainty["expected_intervention"],
                "valid_target_count": len(evaluator["spatial_label"]["valid_target_ids"]),
            },
        })
        feature_entries.append({
            "sample_id": item["sample_id"], "family_id": item["family_id"],
            "split": "test_iid", "variant": item["variant"], **feature_input,
        })
        family_variants[item["family_id"]].append(item["variant"])
        state_counts[state] += 1
        source_counts.update(uncertainty["sources"])
        relation_counts[relation_labels[0]] += 1
        category_counts[record["family_category"]] += 1
        group_counts[str(record.get("target_object_group"))] += 1
        rgb_hashes.add(rgb["sha256"])
        depth_hashes.add(depth["sha256"])

    order = {name: index for index, name in enumerate(VARIANTS)}
    eval_entries.sort(key=lambda row: (row["family_id"], order[row["variant"]]))
    feature_entries.sort(key=lambda row: (row["family_id"], order[row["variant"]]))
    if len(eval_entries) != 1000 or len({row["sample_id"] for row in eval_entries}) != 1000:
        raise TestIIDPrepareError("expected 1,000 unique evaluation entries")
    if any(values != VARIANTS for values in family_variants.values()):
        raise TestIIDPrepareError("family variant coverage/order differs")
    prior_feature_entries = (
        read_json(ROOT / "old/protocol/pcra_u_development_feature_manifest.json")["entries"]
        + calibration_feature_manifest["entries"]
    )
    if rgb_hashes & {row["rgb_sha256"] for row in prior_feature_entries}:
        raise TestIIDPrepareError("Test-IID RGB content overlaps development/calibration")
    if depth_hashes & {row["depth_sha256"] for row in prior_feature_entries}:
        raise TestIIDPrepareError("Test-IID depth content overlaps development/calibration")

    protected = {
        "dataset_index_sha256": sha256_file(index_path),
        "family_manifest_sha256": sha256_file(family_path),
        "materialized_checkpoint_sha256": sha256_file(materialized_path),
        "best_v2_freeze_lock_sha256": sha256_file(freeze_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "calibrator_sha256": sha256_file(calibrator_path),
        "model_inventory_sha256": model_hash,
        "model_config_sha256": sha256_file(config_path),
    }
    summary = {
        "answerability": dict(sorted(state_counts.items())),
        "uncertainty_sources": dict(sorted(source_counts.items())),
        "relations": dict(sorted(relation_counts.items())),
        "family_categories": dict(sorted(category_counts.items())),
        "target_object_groups": dict(sorted(group_counts.items())),
        "unique_feature_pairs": len({row["feature_key"] for row in feature_entries}),
        "development_calibration_rgb_overlap": 0,
        "development_calibration_depth_overlap": 0,
    }
    eval_manifest = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "status": "TEST_IID_EVALUATOR_LABELS_SEPARATED",
        "dataset_root": str(DATASET_ROOT.relative_to(ROOT)),
        "split": "test_iid", "family_count": 200, "sample_count": 1000,
        "answerability_classes": ANSWER_CLASSES, "source_classes": SOURCE_CLASSES,
        "variant_order": VARIANTS, "family_ids": sorted(by_family),
        "summary": summary, "protected_inputs": protected, "entries": eval_entries,
        "safety": {"raw_inference_may_read_this_manifest": False, "training_performed": False,
                   "checkpoint_or_threshold_selection_performed": False},
    }
    feature_manifest = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "status": "ORACLE_FREE_TEST_IID_FEATURE_INPUT_ONLY",
        "dataset_root": str(DATASET_ROOT.relative_to(ROOT)), "split": "test_iid",
        "model_root": "RoboRefer/models/RoboRefer-2B-SFT",
        "model_inventory_sha256": model_hash,
        "feature_transform_sha256": canonical_sha256(
            read_json(ROOT / "old/protocol/pcra_u_calibration_config.json")["feature_input"]
        ),
        "sample_count": 1000, "family_count": 200,
        "unique_feature_pair_count": summary["unique_feature_pairs"],
        "entries": feature_entries,
        "safety": {"evaluator_artifacts_read_by_feature_extractor": False,
                   "oracle_labels_present": False, "training_performed": False},
    }
    if forbidden_cache_findings(feature_manifest):
        raise TestIIDPrepareError("generated feature manifest contains evaluator/oracle keys")
    eval_path = PROTOCOL_ROOT / "test_iid_eval_manifest.json"
    feature_path = PROTOCOL_ROOT / "test_iid_feature_manifest.json"
    write_immutable(eval_path, eval_manifest)
    write_immutable(feature_path, feature_manifest)
    gate = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "created_at_utc": utc_now(), "status": "FROZEN_TEST_IID_INFERENCE_AUTHORIZED",
        "decision": "GO_FROZEN_BEST_V2_TEST_IID_INFERENCE",
        "family_count": 200, "sample_count": 1000,
        "eval_manifest_sha256": sha256_file(eval_path),
        "feature_manifest_sha256": sha256_file(feature_path),
        "protected_inputs": protected,
        "checkpoint_frozen": True, "calibrator_frozen": True,
        "training_authorized": False, "threshold_tuning_authorized": False,
        "test_ood_opened": False,
    }
    write_immutable(PROTOCOL_ROOT / "frozen_inference_gate.json", gate)
    report = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "passed": True,
        "decision": gate["decision"], "family_count": 200, "sample_count": 1000,
        "summary": summary, "protected_inputs": protected,
        "eval_manifest": str(eval_path.relative_to(ROOT)),
        "feature_manifest": str(feature_path.relative_to(ROOT)),
    }
    write_json(DATASET_ROOT / "TEST_IID_MATERIALIZED_QC_REPORT.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2, sort_keys=True))
