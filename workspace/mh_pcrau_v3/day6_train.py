"""Run the locked Day-6 S1a offline head-only training on real h_spatial."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from workspace.mh_pcrau_v3.loss_v3 import LossWeights, SupervisionBatch, compute_multitask_loss
from workspace.mh_pcrau_v3.multihead_v3 import ANSWERABILITY_CLASSES, RELATION_CLASSES, build_seeded_model


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY6_S1A_RUN_LOCK_R2.json"
CACHE_MANIFEST = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/S1A_FEATURE_CACHE_MANIFEST.json"
CACHE_QC = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/S1A_CACHE_QC.json"
OUT = ROOT / "ketqua1/07_huan_luyen/ngay_06"
CHECKPOINTS = ROOT / "workspace/mh_pcrau_v3/checkpoints/day06"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def subset(target: SupervisionBatch, indices: torch.Tensor) -> SupervisionBatch:
    return SupervisionBatch(
        relation=target.relation[indices], relation_mask=target.relation_mask[indices],
        reasoning_depth=target.reasoning_depth[indices], reasoning_mask=target.reasoning_mask[indices],
        target_uv=target.target_uv[indices], spatial_mask=target.spatial_mask[indices],
        uncertainty_source=target.uncertainty_source[indices], source_mask=target.source_mask[indices],
        answerability=target.answerability[indices], answerability_mask=target.answerability_mask[indices],
    )


def macro_f1(prediction: torch.Tensor, target: torch.Tensor, classes: int) -> tuple[float, list[float]]:
    values = []
    recalls = []
    for index in range(classes):
        tp = int(((prediction == index) & (target == index)).sum().item())
        fp = int(((prediction == index) & (target != index)).sum().item())
        fn = int(((prediction != index) & (target == index)).sum().item())
        denom = 2 * tp + fp + fn
        values.append(0.0 if denom == 0 else 2 * tp / denom)
        recalls.append(0.0 if tp + fn == 0 else tp / (tp + fn))
    return sum(values) / classes, recalls


@torch.no_grad()
def evaluate(model, features: torch.Tensor, target: SupervisionBatch, indices: torch.Tensor,
             weights: LossWeights, epoch: int, split_name: str) -> dict:
    model.eval()
    selected_features = features[indices]
    selected_target = subset(target, indices)
    output = model(selected_features)
    loss = compute_multitask_loss(output, selected_target, weights, stage="s1a")
    relation_pred = output.relation_logits.argmax(-1)
    answer_pred = output.answerability_logits.argmax(-1)
    relation_f1, relation_recall = macro_f1(relation_pred, selected_target.relation, 4)
    answer_f1, answer_recall = macro_f1(answer_pred, selected_target.answerability, 4)
    spatial = selected_target.spatial_mask
    delta = output.mu_uv[spatial] - selected_target.target_uv[spatial]
    l2 = delta.square().sum(-1).sqrt()
    logvar = output.log_variance_uv[spatial]
    tensors = [
        output.relation_logits, output.reasoning_logits, output.mu_uv, output.log_variance_uv,
        output.source_logits, output.answerability_logits, output.confidence_logit, loss.total,
    ]
    finite = all(bool(torch.isfinite(value).all().item()) for value in tensors)
    result = {
        "epoch": epoch,
        "split": split_name,
        "total_loss": float(loss.total.item()),
        "relation_loss": float(loss.components["relation"].item()),
        "reasoning_loss": float(loss.components["reasoning"].item()),
        "spatial_nll": float(loss.components["spatial"].item()),
        "source_loss": float(loss.components["source"].item()),
        "answerability_loss": float(loss.components["answerability"].item()),
        "confidence_loss": float(loss.components["confidence"].item()),
        "relation_accuracy": float((relation_pred == selected_target.relation).float().mean().item()),
        "relation_macro_f1": relation_f1,
        "relation_per_class_recall": relation_recall,
        "answerability_accuracy": float((answer_pred == selected_target.answerability).float().mean().item()),
        "answerability_macro_f1": answer_f1,
        "answerability_per_class_recall": answer_recall,
        "point_mae_uv": float(delta.abs().mean().item()),
        "point_l2_mean": float(l2.mean().item()),
        "hit_at_0_05": float((l2 <= 0.05).float().mean().item()),
        "hit_at_0_08": float((l2 <= 0.08).float().mean().item()),
        "log_variance_min": float(logvar.min().item()),
        "log_variance_max": float(logvar.max().item()),
        "log_variance_mean": float(logvar.mean().item()),
        "finite": finite,
        "valid_relation": loss.valid_counts["relation"],
        "valid_reasoning": loss.valid_counts["reasoning"],
        "valid_spatial": loss.valid_counts["spatial"],
        "valid_source": loss.valid_counts["source"],
        "valid_answerability": loss.valid_counts["answerability"],
        "valid_confidence": loss.valid_counts["confidence"],
    }
    result["selection_score"] = (relation_f1 + answer_f1 + result["hit_at_0_08"]) / 3.0
    result["eligible"] = finite and result["log_variance_min"] > -7.95 and result["log_variance_max"] < 1.95
    return result


def flatten_epoch(train: dict, val: dict, seconds: float, improved: bool) -> dict:
    row: dict[str, object] = {"epoch": train["epoch"], "epoch_seconds": seconds, "best_updated": improved}
    skip = {"epoch", "split", "relation_per_class_recall", "answerability_per_class_recall"}
    for prefix, metrics in (("train", train), ("val", val)):
        for key, value in metrics.items():
            if key not in skip:
                row[f"{prefix}_{key}"] = value
    return row


def state_clone(module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def state_delta_l2(before: dict[str, torch.Tensor], module) -> float:
    total = 0.0
    for name, after in module.state_dict().items():
        total += float((after.detach().cpu().double() - before[name].double()).square().sum().item())
    return math.sqrt(total)


def outputs_max_abs(left, right) -> float:
    names = (
        "relation_logits", "reasoning_logits", "mu_uv", "log_variance_uv",
        "source_logits", "answerability_logits", "confidence_logit", "z_spatial",
    )
    return max(float((getattr(left, name) - getattr(right, name)).abs().max().item()) for name in names)


def main() -> None:
    output_names = [
        "S1A_TRAIN_LOG.jsonl", "S1A_EPOCH_METRICS.csv", "S1A_CHECKPOINT_MANIFEST.json",
        "S1A_RESOURCE_REPORT.json", "BACKBONE_FREEZE_AUDIT.json", "S1A_DECISION.json",
    ]
    if CHECKPOINTS.exists() or any((OUT / name).exists() for name in output_names):
        raise FileExistsError("Day-6 training outputs are append-only")
    if not torch.cuda.is_available():
        raise RuntimeError("Locked Day-6 training device is CUDA")
    lock = json.loads(LOCK.read_text())
    if lock["status"] != "FROZEN_BEFORE_DAY6_TRAIN_ATTEMPT_02":
        raise RuntimeError("Invalid Day-6 lock")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != lock["deterministic_environment"]["CUBLAS_WORKSPACE_CONFIG"]:
        raise RuntimeError("Deterministic cuBLAS environment drift")
    for key in ("inherited_lock", "cache_manifest", "cache_qc", "attempt_01_failure"):
        ref = lock[key]
        if sha256(ROOT / ref["path"]) != ref["sha256"]:
            raise RuntimeError(f"Attempt-02 evidence drift: {ref['path']}")
    for ref in lock["source_identity"].values():
        if sha256(ROOT / ref["path"]) != ref["sha256"]:
            raise RuntimeError(f"Attempt-02 source drift: {ref['path']}")
    for ref in lock["prerequisites"]:
        if sha256(ROOT / ref["path"]) != ref["sha256"]:
            raise RuntimeError(f"Prerequisite drift: {ref['path']}")
    for key in ("input_manifest", "supervision_store", "split_audit"):
        ref = lock["data"][key]
        if sha256(ROOT / ref["path"]) != ref["sha256"]:
            raise RuntimeError(f"Day-6 data drift: {ref['path']}")
    cache_qc = json.loads(CACHE_QC.read_text())
    cache = json.loads(CACHE_MANIFEST.read_text())
    if cache_qc["status"] != "PASS" or sha256(CACHE_MANIFEST) != cache_qc["manifest_sha256"]:
        raise RuntimeError("S1a cache is not eligible")

    cache_by_id = {row["sample_id"]: row for row in cache["records"]}
    supervision_rows = read_jsonl(ROOT / lock["data"]["supervision_store"]["path"])
    if len(supervision_rows) != 320 or set(cache_by_id) != {row["sample_id"] for row in supervision_rows}:
        raise RuntimeError("Cache/supervision identity mismatch")
    ordered = sorted(supervision_rows, key=lambda row: row["sample_id"])
    features_cpu = []
    for row in ordered:
        item = cache_by_id[row["sample_id"]]
        path = ROOT / item["feature_path"]
        if sha256(path) != item["feature_file_sha256"]:
            raise RuntimeError(f"Feature shard drift: {row['sample_id']}")
        feature = torch.load(path, map_location="cpu", weights_only=True).contiguous()
        if tensor_hash(feature) != item["feature_raw_sha256"] or feature.shape != (1536,):
            raise RuntimeError(f"Feature payload drift: {row['sample_id']}")
        features_cpu.append(feature)
    device = torch.device("cuda:0")
    features = torch.stack(features_cpu).to(device=device, dtype=torch.float32)
    relation_map = {name: index for index, name in enumerate(RELATION_CLASSES)}
    answer_map = {name: index for index, name in enumerate(ANSWERABILITY_CLASSES)}
    target_uv = torch.full((320, 2), float("nan"), dtype=torch.float32, device=device)
    spatial_mask = torch.tensor([row["head_mask"]["coordinate"] for row in ordered], dtype=torch.bool, device=device)
    for index, row in enumerate(ordered):
        if spatial_mask[index]:
            target_uv[index] = torch.tensor(row["target_uv"], dtype=torch.float32, device=device)
    target = SupervisionBatch(
        relation=torch.tensor([relation_map[row["relation"]] for row in ordered], dtype=torch.long, device=device),
        relation_mask=torch.ones(320, dtype=torch.bool, device=device),
        reasoning_depth=torch.full((320,), -1, dtype=torch.long, device=device),
        reasoning_mask=torch.zeros(320, dtype=torch.bool, device=device),
        target_uv=target_uv,
        spatial_mask=spatial_mask,
        uncertainty_source=torch.full((320,), -1, dtype=torch.long, device=device),
        source_mask=torch.zeros(320, dtype=torch.bool, device=device),
        answerability=torch.tensor([answer_map[row["answerability"]] for row in ordered], dtype=torch.long, device=device),
        answerability_mask=torch.ones(320, dtype=torch.bool, device=device),
    )
    train_indices = torch.tensor([i for i, row in enumerate(ordered) if row["split"] == "train_uq"], dtype=torch.long, device=device)
    val_indices = torch.tensor([i for i, row in enumerate(ordered) if row["split"] == "val_uq"], dtype=torch.long, device=device)
    if len(train_indices) != 256 or len(val_indices) != 64:
        raise RuntimeError("Split cardinality drift")

    seed = lock["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    model = build_seeded_model(seed).to(device=device, dtype=torch.float32)
    inventory = model.configure_trainable("s1a")
    confidence_initial = state_clone(model.confidence_head)
    reasoning_initial = state_clone(model.reasoning_head)
    source_initial = state_clone(model.source_head)
    logvar_parameters = list(model.log_variance_head.parameters())
    logvar_ids = {id(parameter) for parameter in logvar_parameters}
    main_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in logvar_ids]
    optimizer = torch.optim.AdamW([
        {"params": main_parameters, "lr": lock["optimizer"]["main_learning_rate"]},
        {"params": logvar_parameters, "lr": lock["optimizer"]["log_variance_head_learning_rate"]},
    ], weight_decay=lock["optimizer"]["weight_decay"])
    weights = LossWeights(**lock["loss_weights"])
    initial_train = evaluate(model, features, target, train_indices, weights, 0, "train_uq")
    initial_val = evaluate(model, features, target, val_indices, weights, 0, "val_uq")

    CHECKPOINTS.mkdir(parents=True)
    best_path = CHECKPOINTS / "s1a_best.pt"
    final_path = CHECKPOINTS / "s1a_final.pt"
    optimizer_path = CHECKPOINTS / "s1a_final_optimizer.pt"
    train_log_rows: list[dict] = []
    epoch_rows: list[dict] = [flatten_epoch(initial_train, initial_val, 0.0, False)]
    all_finite = initial_train["finite"] and initial_val["finite"]
    best_epoch = None
    best_score = -math.inf
    best_val_loss = math.inf
    no_improvement = 0
    global_step = 0
    clipped_steps = 0
    stopped_reason = "maximum_epochs"
    started = time.perf_counter()
    recipe = lock["training"]
    for epoch in range(1, recipe["maximum_epochs"] + 1):
        epoch_started = time.perf_counter()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch)
        order = torch.randperm(len(train_indices), generator=generator).to(device)
        model.train()
        for batch_number, start in enumerate(range(0, len(order), recipe["batch_size"]), start=1):
            batch_indices = train_indices[order[start:start + recipe["batch_size"]]]
            optimizer.zero_grad(set_to_none=True)
            loss = compute_multitask_loss(model(features[batch_indices]), subset(target, batch_indices), weights, stage="s1a")
            if not torch.isfinite(loss.total):
                all_finite = False
                stopped_reason = "nonfinite_training_loss"
                break
            loss.total.backward()
            preclip = float(torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lock["optimizer"]["maximum_gradient_norm"], error_if_nonfinite=True,
            ))
            if not math.isfinite(preclip):
                all_finite = False
                stopped_reason = "nonfinite_gradient"
                break
            clipped_steps += int(preclip > lock["optimizer"]["maximum_gradient_norm"])
            optimizer.step()
            global_step += 1
            train_log_rows.append({
                "epoch": epoch, "batch": batch_number, "global_step": global_step,
                "total_loss": float(loss.total.item()),
                "relation_loss": float(loss.components["relation"].item()),
                "reasoning_loss": float(loss.components["reasoning"].item()),
                "spatial_nll": float(loss.components["spatial"].item()),
                "source_loss": float(loss.components["source"].item()),
                "answerability_loss": float(loss.components["answerability"].item()),
                "confidence_loss": float(loss.components["confidence"].item()),
                "gradient_norm_preclip": preclip,
                "gradient_clipped": preclip > lock["optimizer"]["maximum_gradient_norm"],
                "valid_counts": loss.valid_counts,
                "finite": True,
            })
        if not all_finite:
            break
        train_metrics = evaluate(model, features, target, train_indices, weights, epoch, "train_uq")
        val_metrics = evaluate(model, features, target, val_indices, weights, epoch, "val_uq")
        all_finite = all_finite and train_metrics["finite"] and val_metrics["finite"]
        score = val_metrics["selection_score"]
        improved = False
        if val_metrics["eligible"]:
            score_better = score > best_score + recipe["selection_min_delta"]
            tied_score_lower_loss = abs(score - best_score) <= recipe["selection_min_delta"] and val_metrics["total_loss"] < best_val_loss
            if score_better or tied_score_lower_loss:
                improved = True
                best_epoch = epoch
                best_score = score
                best_val_loss = val_metrics["total_loss"]
                no_improvement = 0
                torch.save({
                    "schema_version": "1.0", "stage": "S1a", "epoch": epoch,
                    "run_lock_sha256": sha256(LOCK), "cache_manifest_sha256": sha256(CACHE_MANIFEST),
                    "model_config": model.config.to_dict(),
                    "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "train_metrics": train_metrics, "val_metrics": val_metrics,
                }, best_path)
            else:
                no_improvement += 1
        else:
            no_improvement += 1
        epoch_rows.append(flatten_epoch(train_metrics, val_metrics, time.perf_counter() - epoch_started, improved))
        if epoch % 5 == 0 or improved:
            print(json.dumps({
                "epoch": epoch, "score": score, "val_loss": val_metrics["total_loss"],
                "rel_f1": val_metrics["relation_macro_f1"], "ans_f1": val_metrics["answerability_macro_f1"],
                "hit08": val_metrics["hit_at_0_08"], "eligible": val_metrics["eligible"],
                "best_epoch": best_epoch, "patience": no_improvement,
            }), flush=True)
        if epoch >= recipe["minimum_epochs"] and no_improvement >= recipe["early_stopping_patience"]:
            stopped_reason = "early_stopping"
            break
    wall_seconds = time.perf_counter() - started
    last_epoch = epoch_rows[-1]["epoch"]
    final_train = evaluate(model, features, target, train_indices, weights, last_epoch, "train_uq")
    final_val = evaluate(model, features, target, val_indices, weights, last_epoch, "val_uq")
    torch.save({
        "schema_version": "1.0", "stage": "S1a", "epoch": last_epoch,
        "run_lock_sha256": sha256(LOCK), "cache_manifest_sha256": sha256(CACHE_MANIFEST),
        "model_config": model.config.to_dict(),
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "train_metrics": final_train, "val_metrics": final_val,
    }, final_path)
    torch.save({"epoch": last_epoch, "global_step": global_step, "optimizer_state_dict": optimizer.state_dict()}, optimizer_path)
    if best_epoch is None or not best_path.exists():
        raise RuntimeError("No eligible best checkpoint was produced")

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    best_train = evaluate(model, features, target, train_indices, weights, best_epoch, "train_uq")
    best_val = evaluate(model, features, target, val_indices, weights, best_epoch, "val_uq")
    with torch.no_grad():
        reference_output = model(features[val_indices])
    reloaded = build_seeded_model(seed).to(device=device, dtype=torch.float32)
    reloaded.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model_state_dict"])
    reloaded.configure_trainable("inference")
    reloaded.eval()
    with torch.no_grad():
        reload_output = reloaded(features[val_indices])
    reload_max_abs = outputs_max_abs(reference_output, reload_output)

    train_log_path = OUT / "S1A_TRAIN_LOG.jsonl"
    train_log_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_log_rows), encoding="utf-8")
    metrics_path = OUT / "S1A_EPOCH_METRICS.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)

    backbone_before = cache["model_inventory_sha256"]
    backbone_after = json.loads(CACHE_MANIFEST.read_text())["model_inventory_sha256"]
    backbone_audit = {
        "schema_version": "1.0", "status": "PASS" if backbone_before == backbone_after else "FAIL",
        "training_mode": "offline h_spatial; RoboRefer modules never loaded by optimizer",
        "backbone_model_inventory_sha256_before": backbone_before,
        "backbone_model_inventory_sha256_after": backbone_after,
        "backbone_parameters_in_optimizer": 0,
        "confidence_parameter_delta_l2": state_delta_l2(confidence_initial, model.confidence_head),
        "reasoning_parameter_delta_l2": state_delta_l2(reasoning_initial, model.reasoning_head),
        "source_parameter_delta_l2": state_delta_l2(source_initial, model.source_head),
        "interpretation": "Reasoning/source have zero certified support; confidence is frozen until OOF/S1b.",
    }
    write_json(OUT / "BACKBONE_FREEZE_AUDIT.json", backbone_audit)

    resource = {
        "schema_version": "1.0", "status": "PASS",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "wall_seconds": wall_seconds,
        "epochs_completed": last_epoch,
        "optimizer_steps": global_step,
        "clipped_steps": clipped_steps,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "budget_reserved_mib": lock["completion_checks"]["peak_reserved_mib_max"],
        "features_shape": list(features.shape),
        "model_parameters": inventory,
    }
    resource["within_vram_budget"] = resource["peak_reserved_mib"] <= resource["budget_reserved_mib"]
    resource["status"] = "PASS" if resource["within_vram_budget"] else "FAIL"
    write_json(OUT / "S1A_RESOURCE_REPORT.json", resource)

    checkpoint_manifest = {
        "schema_version": "1.0", "status": "PASS" if reload_max_abs == 0.0 else "FAIL",
        "best_epoch": best_epoch, "last_epoch": last_epoch, "stopped_reason": stopped_reason,
        "selection_score": best_score, "best_train_metrics": best_train, "best_val_metrics": best_val,
        "reload_max_abs": reload_max_abs,
        "files": [
            {"role": "best_model", "path": str(best_path.relative_to(ROOT)), "sha256": sha256(best_path), "bytes": best_path.stat().st_size},
            {"role": "final_model", "path": str(final_path.relative_to(ROOT)), "sha256": sha256(final_path), "bytes": final_path.stat().st_size},
            {"role": "final_optimizer", "path": str(optimizer_path.relative_to(ROOT)), "sha256": sha256(optimizer_path), "bytes": optimizer_path.stat().st_size},
        ],
        "run_lock": {"path": str(LOCK.relative_to(ROOT)), "sha256": sha256(LOCK)},
        "cache_manifest": {"path": str(CACHE_MANIFEST.relative_to(ROOT)), "sha256": sha256(CACHE_MANIFEST)},
    }
    write_json(OUT / "S1A_CHECKPOINT_MANIFEST.json", checkpoint_manifest)

    support_exact = (
        best_train["valid_relation"] == 256 and best_train["valid_answerability"] == 256 and best_train["valid_spatial"] == 64
        and best_val["valid_relation"] == 64 and best_val["valid_answerability"] == 64 and best_val["valid_spatial"] == 16
    )
    masked_zero = all(
        metrics[key] == 0
        for metrics in (best_train, best_val)
        for key in ("valid_reasoning", "valid_source", "valid_confidence")
    )
    checks = {
        "cache_qc_pass": cache_qc["status"] == "PASS",
        "all_rows_consumed_each_epoch": len(train_log_rows) == last_epoch * (256 // recipe["batch_size"]),
        "active_head_support_exact": support_exact,
        "reasoning_source_confidence_support_zero": masked_zero,
        "unsupported_head_parameters_unchanged": backbone_audit["confidence_parameter_delta_l2"] == backbone_audit["reasoning_parameter_delta_l2"] == backbone_audit["source_parameter_delta_l2"] == 0.0,
        "best_checkpoint_eligible": best_val["eligible"],
        "best_train_total_loss_below_epoch0": best_train["total_loss"] < initial_train["total_loss"],
        "checkpoint_reload_exact": reload_max_abs == 0.0,
        "backbone_inventory_unchanged": backbone_audit["status"] == "PASS",
        "vram_within_budget": resource["within_vram_budget"],
        "all_training_and_evaluation_finite": all_finite and best_train["finite"] and best_val["finite"],
    }
    complete = all(checks.values())
    decision = {
        "schema_version": "1.0", "date_local": "2026-09-24",
        "outcome": "S1A_COMPLETE" if complete else "S1A_STOP",
        "day7_oof_s1b_authorized": complete,
        "g3_status": "NOT_OPENED",
        "calibration_test_robot": "SEALED",
        "checks": checks,
        "best_epoch": best_epoch, "last_epoch": last_epoch, "stopped_reason": stopped_reason,
        "best_train_metrics": best_train, "best_val_metrics": best_val,
        "scope_limits": {
            "reasoning": "not trained; certified support 0",
            "uncertainty_source": "not trained; certified support 0; exploratory only",
            "confidence": "not trained; frozen until Day-7 OOF/S1b",
            "val_metrics": "development model-selection evidence, not Test or confirmatory claims",
        },
        "evidence": {
            "train_log": str(train_log_path.relative_to(ROOT)),
            "epoch_metrics": str(metrics_path.relative_to(ROOT)),
            "checkpoint_manifest": "ketqua1/07_huan_luyen/ngay_06/S1A_CHECKPOINT_MANIFEST.json",
            "resource_report": "ketqua1/07_huan_luyen/ngay_06/S1A_RESOURCE_REPORT.json",
            "backbone_audit": "ketqua1/07_huan_luyen/ngay_06/BACKBONE_FREEZE_AUDIT.json",
        },
    }
    write_json(OUT / "S1A_DECISION.json", decision)
    print(json.dumps({
        "outcome": decision["outcome"], "best_epoch": best_epoch, "last_epoch": last_epoch,
        "best_val_score": best_score, "best_val": best_val,
        "peak_reserved_mib": resource["peak_reserved_mib"], "reload_max_abs": reload_max_abs,
    }, ensure_ascii=False), flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
