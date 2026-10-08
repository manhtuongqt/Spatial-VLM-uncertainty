#!/usr/bin/env python3
"""Materialize deterministic P-CRA-U development feature/supervision manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from .pcra_u_development_common import (
        ANSWERABILITY_CLASSES,
        RELATION_CLASSES,
        SOURCE_CLASSES,
        VARIANT_ORDER,
        DevelopmentTrainingError,
        canonical_sha256,
        parse_relation,
        read_json,
        safe_resolve,
        sha256_file,
        sha256_text,
        write_json,
    )
    from .wp3_feature_hook_smoke import model_inventory_sha256
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pcra_u_development_common import (  # type: ignore
        ANSWERABILITY_CLASSES,
        RELATION_CLASSES,
        SOURCE_CLASSES,
        VARIANT_ORDER,
        DevelopmentTrainingError,
        canonical_sha256,
        parse_relation,
        read_json,
        safe_resolve,
        sha256_file,
        sha256_text,
        write_json,
    )
    from wp3_feature_hook_smoke import model_inventory_sha256  # type: ignore


WORKSPACE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "pcra_u_development_train_v1"
CONFIG_PATH = WORKSPACE / "protocol/pcra_u_development_train_config.json"
TRAIN_MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_development_train_manifest.json"
FEATURE_MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_development_feature_manifest.json"
DATASET_ROOT = WORKSPACE / "datasets/roborefer_dataset_v2_1_1_development_400_20260824"
DATASET_INDEX = DATASET_ROOT / "dataset_index.json"
FAMILY_MANIFEST = DATASET_ROOT / "family_manifest.json"
FULL_QC = DATASET_ROOT / "DATASET_V2_1_DEVELOPMENT_FULL_QC_REPORT.json"
WP3_LOCK = WORKSPACE / "protocol/wp3_feature_gate_lock.json"
WP3_CONTRACT = WORKSPACE / "protocol/WP3_GRAPH_FEATURE_CONTRACT.md"


def tree_digest(root: Path) -> str:
    relative_paths = [str(path.relative_to(WORKSPACE)) for path in root.rglob("*") if path.is_file()]
    sorted_bytes = subprocess.run(
        ["sort", "-z"],
        input=b"\0".join(path.encode("utf-8") for path in relative_paths) + b"\0",
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    digest = hashlib.sha256()
    for raw_path in sorted_bytes.rstrip(b"\0").split(b"\0"):
        relative = raw_path.decode("utf-8")
        digest.update(f"{sha256_file(WORKSPACE / relative)}  {relative}\n".encode("utf-8"))
    return digest.hexdigest()


def verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise DevelopmentTrainingError(f"{label} missing/hash mismatch: {path}")


def manifest_entry(dataset_root: Path, index_row: dict[str, Any], family_row: dict[str, Any], config: dict[str, Any], model_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record_path = safe_resolve(dataset_root, index_row["record_path"])
    verify_file(record_path, index_row["record_sha256"], "record")
    record = read_json(record_path)
    for key in ["sample_id", "family_id", "split", "variant"]:
        if record.get(key) != index_row.get(key):
            raise DevelopmentTrainingError(f"Index/record {key} mismatch: {index_row['sample_id']}")
    variant_rows = {row["variant"]: row for row in family_row["variant_specs"]}
    variant_row = variant_rows[index_row["variant"]]
    if record["instruction"] != variant_row["instruction"]:
        raise DevelopmentTrainingError(f"Family/record instruction mismatch: {index_row['sample_id']}")
    if sha256_text(record["instruction"]) != variant_row["instruction_sha256"]:
        raise DevelopmentTrainingError(f"Instruction hash mismatch: {index_row['sample_id']}")
    inference = record["inference_payload"]
    evaluator = record["evaluator_only"]
    if evaluator.get("access_policy") != "never_export_to_inference_payload":
        raise DevelopmentTrainingError(f"Evaluator access policy mismatch: {index_row['sample_id']}")
    rgb = inference["rgb_model_input"]
    depth = inference["depth_relative_model_input"]
    rgb_path = safe_resolve(dataset_root, rgb["path"])
    depth_path = safe_resolve(dataset_root, depth["path"])
    verify_file(rgb_path, rgb["sha256"], "RGB")
    verify_file(depth_path, depth["sha256"], "depth")
    prompt = inference["prompt"]
    observed_relation = parse_relation(prompt)
    relation_labels = evaluator["spatial_label"]["relations"]
    if len(relation_labels) != 1 or observed_relation != relation_labels[0]:
        raise DevelopmentTrainingError(
            f"Prompt-only relation mismatch for {index_row['sample_id']}: {observed_relation} != {relation_labels}"
        )
    uncertainty = evaluator["uncertainty_label"]
    if uncertainty["state"] != variant_row["answerability_state"] or uncertainty["sources"] != variant_row["uncertainty_sources"]:
        raise DevelopmentTrainingError(f"Family/record uncertainty mismatch: {index_row['sample_id']}")
    if evaluator["spatial_label"]["valid_target_ids"] != variant_row["valid_target_ids"]:
        raise DevelopmentTrainingError(f"Family/record valid targets mismatch: {index_row['sample_id']}")
    unknown_sources = sorted(set(uncertainty["sources"]) - set(SOURCE_CLASSES))
    if unknown_sources:
        raise DevelopmentTrainingError(f"Unsupported source labels {unknown_sources}: {index_row['sample_id']}")
    target = evaluator["masks"]["target"]
    interior = evaluator["masks"]["target_interior"]
    target_path = safe_resolve(dataset_root, target["path"])
    interior_path = safe_resolve(dataset_root, interior["path"])
    verify_file(target_path, target["sha256"], "target mask")
    verify_file(interior_path, interior["sha256"], "target interior mask")
    anchor_rows = []
    for anchor in evaluator["masks"]["anchor"]:
        anchor_path = safe_resolve(dataset_root, anchor["path"])
        verify_file(anchor_path, anchor["sha256"], "anchor mask")
        anchor_rows.append(
            {
                "path": str(anchor_path.relative_to(WORKSPACE)),
                "sha256": anchor["sha256"],
                "bytes": anchor["bytes"],
            }
        )
    feature_key = canonical_sha256(
        {
            "rgb_sha256": rgb["sha256"],
            "depth_sha256": depth["sha256"],
            "model_inventory_sha256": model_hash,
            "feature_transform": config["feature_input"],
        }
    )
    feature_input = {
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "rgb_path": str(rgb_path.relative_to(WORKSPACE)),
        "rgb_sha256": rgb["sha256"],
        "depth_path": str(depth_path.relative_to(WORKSPACE)),
        "depth_sha256": depth["sha256"],
        "feature_key": feature_key,
    }
    train_entry = {
        "sample_id": index_row["sample_id"],
        "family_id": index_row["family_id"],
        "split": index_row["split"],
        "variant": index_row["variant"],
        "record_path": str(record_path.relative_to(WORKSPACE)),
        "record_sha256": index_row["record_sha256"],
        "feature_input": feature_input,
        "supervision": {
            "access_policy": "TRAIN_SUPERVISION_OR_DEV_EVALUATION_ONLY_NEVER_MODEL_INPUT",
            "answerability_state": uncertainty["state"],
            "answerability_index": ANSWERABILITY_CLASSES.index(uncertainty["state"]),
            "source_labels": uncertainty["sources"],
            "source_multihot": [int(name in uncertainty["sources"]) for name in SOURCE_CLASSES],
            "target_mask_path": str(target_path.relative_to(WORKSPACE)),
            "target_mask_sha256": target["sha256"],
            "target_interior_mask_path": str(interior_path.relative_to(WORKSPACE)),
            "target_interior_mask_sha256": interior["sha256"],
            "anchor_masks": anchor_rows,
        },
        "audit_only": {
            "relation": relation_labels[0],
            "family_category": record["family_category"],
            "depth_dependent": record["depth_dependent"],
            "state_submode": uncertainty["state_submode"],
            "valid_target_count": len(evaluator["spatial_label"]["valid_target_ids"]),
        },
    }
    feature_entry = {
        "sample_id": index_row["sample_id"],
        "family_id": index_row["family_id"],
        "split": index_row["split"],
        "variant": index_row["variant"],
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "rgb_path": str(rgb_path.relative_to(WORKSPACE)),
        "rgb_sha256": rgb["sha256"],
        "depth_path": str(depth_path.relative_to(WORKSPACE)),
        "depth_sha256": depth["sha256"],
        "feature_key": feature_key,
    }
    return train_entry, feature_entry


def prepare() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    dataset_index = read_json(DATASET_INDEX)
    family_manifest = read_json(FAMILY_MANIFEST)
    full_qc = read_json(FULL_QC)
    if full_qc.get("passed") is not True or full_qc.get("decision") != "GO_DEVELOPMENT_TRAIN":
        raise DevelopmentTrainingError("V2.1.1 full-QC does not authorize development training")
    if full_qc.get("training_performed") or full_qc.get("calibration_or_test_created"):
        raise DevelopmentTrainingError("Dataset full-QC safety state changed")
    if dataset_index.get("family_count") != 400 or dataset_index.get("record_count") != 2000:
        raise DevelopmentTrainingError("Dataset index count mismatch")
    family_rows = family_manifest["families"]
    by_family = {row["family_id"]: row for row in family_rows}
    if len(by_family) != 400:
        raise DevelopmentTrainingError("Family manifest does not contain 400 unique families")
    model_root = safe_resolve(WORKSPACE, config["feature_input"]["model_root"])
    model_hash = model_inventory_sha256(model_root)
    wp3_lock = read_json(WP3_LOCK)
    for relative, expected in wp3_lock["artifacts"].items():
        candidate = safe_resolve(WORKSPACE, relative)
        verify_file(candidate, expected, "WP3 protected baseline artifact")

    train_entries: list[dict[str, Any]] = []
    feature_entries: list[dict[str, Any]] = []
    for index_row in dataset_index["records"]:
        family_id = index_row["family_id"]
        if family_id not in by_family:
            raise DevelopmentTrainingError(f"Unknown family: {family_id}")
        train_entry, feature_entry = manifest_entry(DATASET_ROOT, index_row, by_family[family_id], config, model_hash)
        train_entries.append(train_entry)
        feature_entries.append(feature_entry)

    split_families: dict[str, set[str]] = defaultdict(set)
    split_samples = Counter()
    answer_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    variants: dict[str, Counter[str]] = defaultdict(Counter)
    family_variants: dict[str, list[str]] = defaultdict(list)
    for entry in train_entries:
        split = entry["split"]
        split_families[split].add(entry["family_id"])
        split_samples[split] += 1
        answer_counts[split][entry["supervision"]["answerability_state"]] += 1
        source_counts[split].update(entry["supervision"]["source_labels"])
        variants[split][entry["variant"]] += 1
        family_variants[entry["family_id"]].append(entry["variant"])
    if split_families["train"] & split_families["dev"]:
        raise DevelopmentTrainingError("Train/dev family overlap")
    if {name: len(values) for name, values in split_families.items()} != {"train": 320, "dev": 80}:
        raise DevelopmentTrainingError("Train/dev family count mismatch")
    if dict(split_samples) != {"dev": 400, "train": 1600}:
        raise DevelopmentTrainingError(f"Train/dev sample count mismatch: {dict(split_samples)}")
    for family_id, observed in family_variants.items():
        if observed != VARIANT_ORDER:
            raise DevelopmentTrainingError(f"Variant order mismatch: {family_id}: {observed}")
    canary_row = next(row for row in family_manifest["batch_assignments"] if row["batch_id"] == "canary_000")
    canary_ids = canary_row["family_ids"]
    if len(canary_ids) != 20 or canary_row["split_counts"] != {"train": 16, "dev": 4}:
        raise DevelopmentTrainingError("Locked canary_000 family assignment mismatch")
    feature_keys = {entry["feature_key"] for entry in feature_entries}
    unique_by_split = {
        split: len({entry["feature_key"] for entry in feature_entries if entry["split"] == split})
        for split in ["train", "dev"]
    }
    summary = {
        "family_counts": {name: len(values) for name, values in sorted(split_families.items())},
        "sample_counts": dict(sorted(split_samples.items())),
        "record_answerability_counts": {name: dict(counter) for name, counter in answer_counts.items()},
        "source_positive_counts": {name: dict(counter) for name, counter in source_counts.items()},
        "variant_counts": {name: dict(counter) for name, counter in variants.items()},
        "unique_feature_pairs": len(feature_keys),
        "unique_feature_pairs_by_split": unique_by_split,
    }
    protected = {
        "dataset_index_sha256": sha256_file(DATASET_INDEX),
        "family_manifest_sha256": sha256_file(FAMILY_MANIFEST),
        "full_qc_report_sha256": sha256_file(FULL_QC),
        "dataset_tree_sha256": tree_digest(DATASET_ROOT),
        "model_inventory_sha256": model_hash,
        "wp3_feature_gate_lock_sha256": sha256_file(WP3_LOCK),
        "wp3_graph_feature_contract_sha256": sha256_file(WP3_CONTRACT),
        "wp3_locked_baseline_artifacts": wp3_lock["artifacts"],
    }
    train_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "MATERIALIZED_BEFORE_TRAINING_PREFLIGHT",
        "scientific_status": config["scientific_status"],
        "dataset_root": str(DATASET_ROOT.relative_to(WORKSPACE)),
        "dataset_protocol_id": dataset_index["protocol_id"],
        "dataset_full_qc_decision": full_qc["decision"],
        "split_policy": {
            "gradient_split": "train",
            "selection_split": "dev",
            "sealed_splits": ["calibration", "test_iid", "test_ood"],
            "family_is_independent_unit": True,
        },
        "answerability_classes": ANSWERABILITY_CLASSES,
        "source_classes": SOURCE_CLASSES,
        "deferred_source_classes": {"spatial": "zero_positive_labels"},
        "relation_classes": RELATION_CLASSES,
        "variant_order": VARIANT_ORDER,
        "train_family_ids": sorted(split_families["train"]),
        "dev_family_ids": sorted(split_families["dev"]),
        "canary_family_ids": canary_ids,
        "summary": summary,
        "protected_inputs": protected,
        "entries": train_entries,
    }
    feature_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "ORACLE_FREE_FEATURE_INPUT_ONLY",
        "dataset_root": str(DATASET_ROOT.relative_to(WORKSPACE)),
        "model_root": config["feature_input"]["model_root"],
        "model_inventory_sha256": model_hash,
        "feature_transform_sha256": canonical_sha256(config["feature_input"]),
        "sample_count": len(feature_entries),
        "unique_feature_pair_count": len(feature_keys),
        "entries": feature_entries,
        "safety": {
            "evaluator_artifacts_read_by_feature_extractor": False,
            "oracle_labels_present": False,
            "training_performed": False,
            "calibration_or_test_read": False,
        },
    }
    write_json(TRAIN_MANIFEST_PATH, train_manifest)
    write_json(FEATURE_MANIFEST_PATH, feature_manifest)
    return {
        "status": "MATERIALIZED",
        "train_manifest": str(TRAIN_MANIFEST_PATH.relative_to(WORKSPACE)),
        "train_manifest_sha256": sha256_file(TRAIN_MANIFEST_PATH),
        "feature_manifest": str(FEATURE_MANIFEST_PATH.relative_to(WORKSPACE)),
        "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
