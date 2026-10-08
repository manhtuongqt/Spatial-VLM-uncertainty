#!/usr/bin/env python3
"""Train a fixed-recipe multi-view/prompt candidate awaiting fresh validation."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from workspace.mh_pcrau_v3.day6_train import evaluate, state_clone, state_delta_l2, subset, tensor_hash
from workspace.mh_pcrau_v3.loss_v3 import LossWeights, SupervisionBatch, compute_multitask_loss
from workspace.mh_pcrau_v3.multihead_v3 import ANSWERABILITY_CLASSES, RELATION_CLASSES, build_seeded_model


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY6_ROBUST_CANDIDATE_LOCK.json"
OUT = ROOT / "ketqua1/07_huan_luyen/ngay_06/amendment_04_robust_candidate"
CHECKPOINT_DIR = ROOT / "workspace/mh_pcrau_v3/checkpoints/day06_robust_candidate"
CANONICAL_CACHE = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/S1A_FEATURE_CACHE_MANIFEST.json"
CANONICAL_LABELS = ROOT / "ketqua1/07_huan_luyen/ngay_06/S1A_SUPERVISION_STORE.jsonl"
SHORT_CACHE = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_02_paired_view/PAIRED_VIEW_FEATURE_CACHE_MANIFEST.json"
SHORT_QC = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_02_paired_view/PAIRED_VIEW_CACHE_QC.json"
PAIRED_LABELS = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_02_paired_view/PAIRED_VIEW_SUPERVISION.jsonl"
ALIGNED_CACHE = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_03_prompt_aligned/PROMPT_ALIGNED_FEATURE_CACHE_MANIFEST.json"
ALIGNED_QC = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_03_prompt_aligned/PROMPT_ALIGNED_CACHE_QC.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_cache(path: Path) -> dict[str, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {row["sample_id"]: row for row in manifest["records"]}


def load_feature(record: dict) -> torch.Tensor:
    path = ROOT / record["feature_path"]
    if sha256(path) != record["feature_file_sha256"]:
        raise RuntimeError(f"Feature file drift: {record['sample_id']}")
    feature = torch.load(path, map_location="cpu", weights_only=True).to(torch.float32).contiguous()
    if feature.shape != (1536,) or tensor_hash(feature) != record["feature_raw_sha256"]:
        raise RuntimeError(f"Feature payload drift: {record['sample_id']}")
    return feature


def main() -> None:
    if OUT.exists() or CHECKPOINT_DIR.exists():
        raise FileExistsError("Robust candidate training is append-only")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "FROZEN_BEFORE_ROBUST_CANDIDATE_TRAINING":
        raise RuntimeError("Invalid robust candidate lock")
    for ref in lock["bindings"]:
        if sha256(ROOT / ref["path"]) != ref["sha256"]:
            raise RuntimeError(f"Training binding drift: {ref['path']}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Deterministic cuBLAS configuration missing")
    for qc_path, manifest_path in ((SHORT_QC, SHORT_CACHE), (ALIGNED_QC, ALIGNED_CACHE)):
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if qc["status"] != "PASS" or qc["manifest_sha256"] != sha256(manifest_path):
            raise RuntimeError(f"Ineligible cache: {manifest_path}")

    canonical_labels = {row["sample_id"]: row for row in read_jsonl(CANONICAL_LABELS)}
    paired_labels = {row["sample_id"]: row for row in read_jsonl(PAIRED_LABELS)}
    canonical_cache = load_cache(CANONICAL_CACHE); short_cache = load_cache(SHORT_CACHE); aligned_cache = load_cache(ALIGNED_CACHE)
    canonical_train_ids = sorted(sample_id for sample_id, row in canonical_labels.items() if row["split"] == "train_uq")
    canonical_val_ids = sorted(sample_id for sample_id, row in canonical_labels.items() if row["split"] == "val_uq")
    paired_ids = sorted(paired_labels)
    if len(canonical_train_ids) != 256 or len(canonical_val_ids) != 64 or len(paired_ids) != 256:
        raise RuntimeError("Unexpected split size")
    if set(paired_ids) != set(short_cache) or set(paired_ids) != set(aligned_cache):
        raise RuntimeError("Paired cache identity mismatch")
    if {canonical_labels[sample_id]["family_id"] for sample_id in canonical_train_ids} != {paired_labels[sample_id]["family_id"] for sample_id in paired_ids}:
        raise RuntimeError("Paired views do not cover exactly the 256 Train-UQ families")

    examples = []
    for sample_id in canonical_train_ids:
        examples.append(("canonical_view_canonical_prompt", canonical_labels[sample_id], load_feature(canonical_cache[sample_id])))
    for sample_id in paired_ids:
        examples.append(("second_view_short_prompt", paired_labels[sample_id], load_feature(short_cache[sample_id])))
    for sample_id in paired_ids:
        examples.append(("second_view_canonical_prompt", paired_labels[sample_id], load_feature(aligned_cache[sample_id])))
    for sample_id in canonical_val_ids:
        examples.append(("old_val_monitor_only", canonical_labels[sample_id], load_feature(canonical_cache[sample_id])))
    if len(examples) != 832:
        raise RuntimeError("Combined feature cardinality drift")

    device = torch.device("cuda:0")
    features = torch.stack([item[2] for item in examples]).to(device)
    relation_map = {name: index for index, name in enumerate(RELATION_CLASSES)}
    answer_map = {name: index for index, name in enumerate(ANSWERABILITY_CLASSES)}
    target_uv = torch.full((len(examples), 2), float("nan"), device=device)
    spatial_mask = torch.tensor([item[1]["head_mask"]["coordinate"] for item in examples], dtype=torch.bool, device=device)
    for index, (_, label, _) in enumerate(examples):
        if spatial_mask[index]: target_uv[index] = torch.tensor(label["target_uv"], device=device)
    target = SupervisionBatch(
        relation=torch.tensor([relation_map[item[1]["relation"]] for item in examples], dtype=torch.long, device=device),
        relation_mask=torch.ones(len(examples), dtype=torch.bool, device=device),
        reasoning_depth=torch.full((len(examples),), -1, dtype=torch.long, device=device),
        reasoning_mask=torch.zeros(len(examples), dtype=torch.bool, device=device),
        target_uv=target_uv, spatial_mask=spatial_mask,
        uncertainty_source=torch.full((len(examples),), -1, dtype=torch.long, device=device),
        source_mask=torch.zeros(len(examples), dtype=torch.bool, device=device),
        answerability=torch.tensor([answer_map[item[1]["answerability"]] for item in examples], dtype=torch.long, device=device),
        answerability_mask=torch.ones(len(examples), dtype=torch.bool, device=device),
    )
    train_indices = torch.arange(0, 768, dtype=torch.long, device=device)
    old_val_indices = torch.arange(768, 832, dtype=torch.long, device=device)
    variant_indices = {
        "canonical_view_canonical_prompt": torch.arange(0, 256, device=device),
        "second_view_short_prompt": torch.arange(256, 512, device=device),
        "second_view_canonical_prompt": torch.arange(512, 768, device=device),
    }
    seed = lock["seed"]
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    model = build_seeded_model(seed).to(device)
    inventory = model.configure_trainable("s1a")
    confidence_initial = state_clone(model.confidence_head); reasoning_initial = state_clone(model.reasoning_head); source_initial = state_clone(model.source_head)
    logvar_parameters = list(model.log_variance_head.parameters()); logvar_ids = {id(value) for value in logvar_parameters}
    main_parameters = [value for value in model.parameters() if value.requires_grad and id(value) not in logvar_ids]
    optimizer = torch.optim.AdamW([
        {"params": main_parameters, "lr": lock["optimizer"]["main_learning_rate"]},
        {"params": logvar_parameters, "lr": lock["optimizer"]["log_variance_head_learning_rate"]},
    ], weight_decay=0.0)
    weights = LossWeights(**lock["loss_weights"])
    initial_train = evaluate(model, features, target, train_indices, weights, 0, "augmented_train")
    initial_val = evaluate(model, features, target, old_val_indices, weights, 0, "old_val_monitor_only")
    logs = []; epoch_rows = []; clipped_steps = 0; global_step = 0; all_finite = True
    started = time.perf_counter()
    for epoch in range(1, lock["training"]["fixed_epochs"] + 1):
        generator = torch.Generator(device="cpu"); generator.manual_seed(seed + epoch)
        order = torch.randperm(len(train_indices), generator=generator).to(device)
        model.train()
        for batch_number, start in enumerate(range(0, len(order), lock["training"]["batch_size"]), start=1):
            indices = train_indices[order[start:start + lock["training"]["batch_size"]]]
            optimizer.zero_grad(set_to_none=True)
            loss = compute_multitask_loss(model(features[indices]), subset(target, indices), weights, stage="s1a")
            if not bool(torch.isfinite(loss.total)): raise RuntimeError("Nonfinite training loss")
            loss.total.backward()
            norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0, error_if_nonfinite=True))
            optimizer.step(); global_step += 1; clipped_steps += int(norm > 5.0)
            logs.append({"epoch": epoch, "batch": batch_number, "global_step": global_step,
                         "total_loss": float(loss.total), "relation_loss": float(loss.components["relation"]),
                         "spatial_nll": float(loss.components["spatial"]), "answerability_loss": float(loss.components["answerability"]),
                         "gradient_norm_preclip": norm, "gradient_clipped": norm > 5.0})
        train_metrics = evaluate(model, features, target, train_indices, weights, epoch, "augmented_train")
        val_metrics = evaluate(model, features, target, old_val_indices, weights, epoch, "old_val_monitor_only")
        all_finite &= train_metrics["finite"] and val_metrics["finite"]
        epoch_rows.append({"epoch": epoch, "train_total_loss": train_metrics["total_loss"],
                           "train_relation_macro_f1": train_metrics["relation_macro_f1"],
                           "train_answerability_macro_f1": train_metrics["answerability_macro_f1"],
                           "train_hit_at_0_05": train_metrics["hit_at_0_05"],
                           "old_val_total_loss_monitor_only": val_metrics["total_loss"],
                           "old_val_relation_macro_f1_monitor_only": val_metrics["relation_macro_f1"],
                           "old_val_answerability_macro_f1_monitor_only": val_metrics["answerability_macro_f1"],
                           "old_val_hit_at_0_05_monitor_only": val_metrics["hit_at_0_05"]})
        if epoch % 5 == 0 or epoch == 1:
            print(json.dumps({"epoch": epoch, "train_loss": train_metrics["total_loss"],
                              "train_answer_f1": train_metrics["answerability_macro_f1"],
                              "old_val_answer_f1_monitor_only": val_metrics["answerability_macro_f1"]}), flush=True)
    wall_seconds = time.perf_counter() - started
    final_train = evaluate(model, features, target, train_indices, weights, 54, "augmented_train")
    old_val_monitor = evaluate(model, features, target, old_val_indices, weights, 54, "old_val_monitor_only")
    per_variant = {name: evaluate(model, features, target, indices, weights, 54, name) for name, indices in variant_indices.items()}
    OUT.mkdir(parents=True); CHECKPOINT_DIR.mkdir(parents=True)
    checkpoint_path = CHECKPOINT_DIR / "s1a_robust_fixed_epoch54.pt"
    torch.save({"schema_version": "1.0", "stage": "S1a_robust_candidate", "epoch": 54,
                "run_lock_sha256": sha256(LOCK), "model_config": model.config.to_dict(),
                "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "train_metrics": final_train, "old_val_metrics_monitor_only": old_val_monitor,
                "per_train_variant_metrics": per_variant}, checkpoint_path)
    (OUT / "ROBUST_CANDIDATE_TRAIN_LOG.jsonl").write_text("".join(json.dumps(row) + "\n" for row in logs), encoding="utf-8")
    with (OUT / "ROBUST_CANDIDATE_EPOCH_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0])); writer.writeheader(); writer.writerows(epoch_rows)
    unsupported_delta = {"reasoning": state_delta_l2(reasoning_initial, model.reasoning_head),
                         "source": state_delta_l2(source_initial, model.source_head),
                         "confidence": state_delta_l2(confidence_initial, model.confidence_head)}
    checks = {
        "all_finite": all_finite and final_train["finite"] and old_val_monitor["finite"],
        "fixed_54_epochs_exact": len(epoch_rows) == 54,
        "optimizer_steps_exact": global_step == 54 * 24,
        "train_support_exact": final_train["valid_relation"] == 768 and final_train["valid_answerability"] == 768 and final_train["valid_spatial"] == 192,
        "old_val_not_used_for_selection": True,
        "unsupported_heads_unchanged": all(value == 0.0 for value in unsupported_delta.values()),
        "final_log_variance_eligible": final_train["log_variance_min"] > -7.95 and final_train["log_variance_max"] < 1.95,
        "peak_reserved_within_14742_mib": torch.cuda.max_memory_reserved() / 2**20 <= 14742.0,
    }
    ready = all(checks.values())
    manifest = {
        "schema_version": "1.0", "status": "PASS" if ready else "FAIL",
        "checkpoint": {"path": str(checkpoint_path.relative_to(ROOT)), "sha256": sha256(checkpoint_path), "bytes": checkpoint_path.stat().st_size},
        "run_lock": {"path": str(LOCK.relative_to(ROOT)), "sha256": sha256(LOCK)},
        "data": {"train_examples": 768, "train_parent_families": 256, "examples_per_family": 3,
                 "old_val_monitor_examples": 64, "old_val_used_for_selection": False},
        "resource": {"wall_seconds": wall_seconds, "optimizer_steps": global_step, "clipped_steps": clipped_steps,
                     "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                     "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, "model_parameters": inventory},
        "initial_train": initial_train, "initial_old_val_monitor": initial_val,
        "final_train": final_train, "old_val_monitor": old_val_monitor,
        "per_train_variant": per_variant, "unsupported_head_delta_l2": unsupported_delta,
    }
    write_json(OUT / "ROBUST_CANDIDATE_CHECKPOINT_MANIFEST.json", manifest)
    decision = {
        "schema_version": "1.0", "outcome": "ROBUST_CANDIDATE_READY_AWAITING_FRESH_VAL" if ready else "ROBUST_CANDIDATE_STOP",
        "checks": checks, "checkpoint": manifest["checkpoint"],
        "generalization_claim": False, "day7_oof_s1b_authorized": False,
        "paired_metrics_are_training_fit_only": True,
        "required_next_gate": "ANTI_SHORTCUT_VAL_V1_DATA_QC_PASS_THEN_ONE_SHOT_EVALUATION",
    }
    write_json(OUT / "ROBUST_CANDIDATE_DECISION.json", decision)
    print(json.dumps({"outcome": decision["outcome"], "checkpoint": manifest["checkpoint"],
                      "train": final_train, "old_val_monitor": old_val_monitor,
                      "peak_reserved_mib": manifest["resource"]["peak_reserved_mib"]}, ensure_ascii=False), flush=True)
    if not ready: raise SystemExit(1)


if __name__ == "__main__": main()
