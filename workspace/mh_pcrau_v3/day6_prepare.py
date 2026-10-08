"""Freeze the Day-6 S1a data, cache, training and selection contract."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets/Gazebo_train_uq_v2_full_r3"
OUT = ROOT / "ketqua1/07_huan_luyen/ngay_06"
CACHE_OUT = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06"
LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY6_S1A_RUN_LOCK.json"
G2 = ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_05/G2_DECISION_REVISION_V2.json"
LOSS = ROOT / "ketqua1/06_ham_mat_mat/ngay_05/LOSS_WEIGHT_DECISION.json"
DATASET_MANIFEST = DATASET / "manifest.json"
SPLIT_LOCK = ROOT / "protocol/gazebo_train_uq_v2_full_r3_split_manifest.json"
INFERENCE = DATASET / "inference_manifest.jsonl"
TRAIN_SUPERVISION = DATASET / "train_supervision.jsonl"
EVALUATOR = DATASET / "evaluator_ground_truth.jsonl"
DAY5_CACHE = ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/FEATURE_CACHE_MANIFEST.json"
DAY5_CACHE_QC = ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/CACHE_QC.json"

RELATIONS = ("leftmost", "rightmost", "second_from_left", "second_from_right")
STATES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if LOCK.exists() or OUT.exists() or CACHE_OUT.exists():
        raise FileExistsError("Day-6 preparation outputs are append-only")
    if json.loads(G2.read_text())["outcome"] != "G2_PASS":
        raise RuntimeError("G2_PASS is required")
    dataset_manifest = json.loads(DATASET_MANIFEST.read_text())
    split_lock = json.loads(SPLIT_LOCK.read_text())
    cache_qc = json.loads(DAY5_CACHE_QC.read_text())
    if dataset_manifest["status"] != "PASS" or dataset_manifest["records"] != 320:
        raise RuntimeError("Canonical Train-UQ dataset is not eligible")
    if split_lock["status"] != "PREREGISTERED" or not split_lock["family_disjoint"]:
        raise RuntimeError("Train/Val split is not preregistered family-disjoint")
    if cache_qc["status"] != "PASS":
        raise RuntimeError("Day-5 cache prerequisite failed")

    inference_rows = read_jsonl(INFERENCE)
    evaluator_rows = read_jsonl(EVALUATOR)
    train_rows = read_jsonl(TRAIN_SUPERVISION)
    if len(inference_rows) != 320 or len(evaluator_rows) != 320 or len(train_rows) != 256:
        raise RuntimeError("Dataset row counts drifted")
    inference_by_id = {row["sample_id"]: row for row in inference_rows}
    evaluator_by_id = {row["sample_id"]: row for row in evaluator_rows}
    train_by_id = {row["sample_id"]: row for row in train_rows}
    if len(inference_by_id) != 320 or set(inference_by_id) != set(evaluator_by_id):
        raise RuntimeError("Inference/evaluator identity mismatch")

    input_manifest: list[dict] = []
    supervision_store: list[dict] = []
    split_families: dict[str, set[str]] = {"train_uq": set(), "val_uq": set()}
    split_counts = Counter()
    cell_counts = Counter()
    for sample_id in sorted(inference_by_id):
        row = inference_by_id[sample_id]
        gt = evaluator_by_id[sample_id]
        if row["family_id"] != gt["family_id"] or row["split"] != gt["split"]:
            raise RuntimeError(f"Identity drift for {sample_id}")
        split = row["split"]
        if split not in split_families:
            raise RuntimeError(f"Unauthorized split: {split}")
        relation = gt["relation_variant"]
        state = gt["answerability_state"]
        if relation not in RELATIONS or state not in STATES or not gt["answerability_verified"]:
            raise RuntimeError(f"Invalid core label: {sample_id}")
        if split == "train_uq":
            train_row = train_by_id.get(sample_id)
            if train_row is None:
                raise RuntimeError(f"Missing canonical train supervision: {sample_id}")
            if train_row["relation_variant"] != relation:
                raise RuntimeError(f"Train relation mismatch: {sample_id}")
            supplied = train_row["supervision"]
            if supplied["answerability_state"] != state or supplied.get("target_xy") != gt.get("target_xy"):
                raise RuntimeError(f"Train/evaluator supervision mismatch: {sample_id}")
        image = DATASET / row["image"]
        depth_view = DATASET / row["depth"]
        metric_depth = DATASET / row["metric_depth"]
        for path in (image, depth_view, metric_depth):
            if not path.is_file():
                raise FileNotFoundError(path)
        input_manifest.append({
            "sample_id": sample_id,
            "family_id": row["family_id"],
            "split": split,
            "rgb_path": rel(image),
            "rgb_sha256": sha256(image),
            "depth_view_path": rel(depth_view),
            "depth_view_sha256": sha256(depth_view),
            "metric_depth_path": rel(metric_depth),
            "metric_depth_sha256": sha256(metric_depth),
            "instruction": row["instruction"],
            "instruction_sha256": digest(row["instruction"]),
        })
        target = gt.get("target_xy") if state == "FOUND" else None
        if state == "FOUND" and (not isinstance(target, list) or len(target) != 2 or not all(0 <= value <= 1 for value in target)):
            raise RuntimeError(f"Invalid FOUND target: {sample_id}")
        supervision_store.append({
            "sample_id": sample_id,
            "family_id": row["family_id"],
            "split": split,
            "relation": relation,
            "answerability": state,
            "target_uv": target,
            "head_mask": {
                "relation": True,
                "reasoning_depth": False,
                "coordinate": state == "FOUND",
                "log_variance": state == "FOUND",
                "uncertainty_source": False,
                "answerability": True,
                "confidence": False,
            },
            "label_source": "Gazebo evaluator geometry/QC; train canonical supervision cross-checked when split=train_uq",
        })
        split_families[split].add(row["family_id"])
        split_counts[split] += 1
        cell_counts[(split, relation, state)] += 1

    if split_counts != Counter({"train_uq": 256, "val_uq": 64}):
        raise RuntimeError(f"Split counts drift: {split_counts}")
    overlap = split_families["train_uq"] & split_families["val_uq"]
    if overlap or len(split_families["train_uq"]) != 256 or len(split_families["val_uq"]) != 64:
        raise RuntimeError("Family-disjointness violation")
    for split, quota in (("train_uq", 16), ("val_uq", 4)):
        for relation in RELATIONS:
            for state in STATES:
                if cell_counts[(split, relation, state)] != quota:
                    raise RuntimeError(f"Cell imbalance: {split}/{relation}/{state}")

    OUT.mkdir(parents=True)
    CACHE_OUT.mkdir(parents=True)
    input_path = OUT / "S1A_INPUT_MANIFEST.jsonl"
    supervision_path = OUT / "S1A_SUPERVISION_STORE.jsonl"
    input_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in input_manifest), encoding="utf-8")
    supervision_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in supervision_store), encoding="utf-8")
    audit = {
        "schema_version": "1.0",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Canonical existing Train-UQ view only; no Day-3 second-view augmentation; no Tabletop B1 candidate promotion",
        "samples": 320,
        "families": 320,
        "split_counts": dict(split_counts),
        "family_counts": {key: len(value) for key, value in split_families.items()},
        "family_overlap": len(overlap),
        "per_relation_answerability_cell": [
            {"split": split, "relation": relation, "answerability": state, "samples": cell_counts[(split, relation, state)]}
            for split in ("train_uq", "val_uq") for relation in RELATIONS for state in STATES
        ],
        "head_support": {
            "train_uq": {"relation": 256, "answerability": 256, "coordinate_logvariance": 64, "reasoning": 0, "source": 0, "confidence": 0},
            "val_uq": {"relation": 64, "answerability": 64, "coordinate_logvariance": 16, "reasoning": 0, "source": 0, "confidence": 0},
        },
        "sealed_access": {"Calibration": False, "IID_Test": False, "OOD_Test": False, "robot": False},
    }
    audit_path = OUT / "S1A_SPLIT_AUDIT.json"
    write_json(audit_path, audit)

    loss_decision = json.loads(LOSS.read_text())
    if loss_decision["selected_candidate"] != "W2_SPA050":
        raise RuntimeError("Day-5 loss selection drift")
    lock = {
        "schema_version": "1.0",
        "method_id": "MH-PCRA-U-v3",
        "status": "FROZEN_BEFORE_DAY6_CACHE_OR_TRAINING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 26092026,
        "prerequisites": [
            {"path": rel(G2), "sha256": sha256(G2)},
            {"path": rel(LOSS), "sha256": sha256(LOSS)},
            {"path": rel(DATASET_MANIFEST), "sha256": sha256(DATASET_MANIFEST)},
            {"path": rel(SPLIT_LOCK), "sha256": sha256(SPLIT_LOCK)},
            {"path": rel(INFERENCE), "sha256": sha256(INFERENCE)},
            {"path": rel(TRAIN_SUPERVISION), "sha256": sha256(TRAIN_SUPERVISION)},
            {"path": rel(EVALUATOR), "sha256": sha256(EVALUATOR)},
            {"path": rel(DAY5_CACHE), "sha256": sha256(DAY5_CACHE)},
            {"path": rel(DAY5_CACHE_QC), "sha256": sha256(DAY5_CACHE_QC)},
        ],
        "data": {
            "input_manifest": {"path": rel(input_path), "sha256": sha256(input_path)},
            "supervision_store": {"path": rel(supervision_path), "sha256": sha256(supervision_path)},
            "split_audit": {"path": rel(audit_path), "sha256": sha256(audit_path)},
            "train_samples_families": 256,
            "val_samples_families": 64,
            "unit": "parent family; one canonical existing view per family",
        },
        "cache": {
            "expected_samples": 320,
            "expected_reuse_from_day5": 32,
            "expected_new_extractions": 288,
            "shape": [1536],
            "dtype": "torch.float32",
            "precision_policy": "G0: RGB/depth towers and projectors BF16; frozen LLM and h_spatial FP32; eager attention; TF32 off",
            "model_input_allowlist": ["rgb_path", "depth_view_path", "instruction"],
        },
        "stage": {
            "name": "S1a_head_only",
            "backbone": "frozen_and_absent_from_offline_optimizer",
            "trainable_modules": ["shared_trunk", "relation_head", "reasoning_head", "coordinate_head", "log_variance_head", "source_head", "answerability_head"],
            "frozen_modules": ["RoboRefer-2B-SFT", "confidence_head"],
            "zero_support_unchanged_heads": ["reasoning_head", "source_head", "confidence_head"],
        },
        "loss_weights": loss_decision["selected_weights"],
        "optimizer": {
            "name": "AdamW",
            "main_learning_rate": 0.0005,
            "log_variance_head_learning_rate": 0.000005,
            "weight_decay": 0.0,
            "maximum_gradient_norm": 5.0,
        },
        "training": {
            "device": "cuda",
            "model_dtype": "float32",
            "batch_size": 32,
            "gradient_accumulation": 1,
            "maximum_epochs": 200,
            "minimum_epochs": 30,
            "early_stopping_patience": 25,
            "selection_min_delta": 0.0001,
            "shuffle": "deterministic torch.randperm seeded by seed+epoch",
            "checkpoint_rule": "maximum eligible Val mean(relation_macro_f1, answerability_macro_f1, hit_at_0.08); tie by lower Val total loss then earlier epoch",
            "eligibility": "all metrics finite and Val log_variance strictly inside (-7.95,1.95)",
        },
        "completion_checks": {
            "all_rows_consumed_each_epoch": True,
            "active_head_support_exact": True,
            "reasoning_source_confidence_support_zero": True,
            "best_checkpoint_eligible": True,
            "best_train_total_loss_below_epoch0": True,
            "checkpoint_reload_max_abs": 0.0,
            "backbone_inventory_unchanged": True,
            "peak_reserved_mib_max": 14742.0,
            "finite_rate": 1.0,
        },
        "next_gate_if_complete": "Day 7 OOF/S1b only; G3/Calibration/Test/robot remain closed",
        "access_boundary": "Development Train-UQ/Val-UQ only. No Calibration/Test/robot access.",
    }
    write_json(LOCK, lock)
    print(json.dumps({"status": lock["status"], "train": 256, "val": 64, "cells": 32, "lock_sha256": sha256(LOCK)}))


if __name__ == "__main__":
    main()
