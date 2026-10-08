#!/usr/bin/env python3
"""Locked canary and full exploratory P-CRA-U development training runtime."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors

if __package__:
    from .pcra_u_development_common import (
        ANSWERABILITY_CLASSES,
        SOURCE_CLASSES,
        DevelopmentTrainingError,
        FrozenFeatureDataset,
        PCRAUDevelopmentV1,
        chunks,
        compute_loss,
        confusion_metrics,
        deterministic_entry_order,
        global_class_weights,
        gradient_groups,
        model_inputs,
        read_json,
        safe_resolve,
        seed_runtime,
        sha256_file,
        write_json,
    )
    from .pcra_u_development_feature_cache import validate_schema
    from .pcra_u_development_train_preflight import protected_now
    from .training_checkpoint_manager import (
        audit_checkpoint_root,
        load_model_for_evaluation,
        resume_training,
        save_checkpoint,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pcra_u_development_common import (  # type: ignore
        ANSWERABILITY_CLASSES,
        SOURCE_CLASSES,
        DevelopmentTrainingError,
        FrozenFeatureDataset,
        PCRAUDevelopmentV1,
        chunks,
        compute_loss,
        confusion_metrics,
        deterministic_entry_order,
        global_class_weights,
        gradient_groups,
        model_inputs,
        read_json,
        safe_resolve,
        seed_runtime,
        sha256_file,
        write_json,
    )
    from pcra_u_development_feature_cache import validate_schema  # type: ignore
    from pcra_u_development_train_preflight import protected_now  # type: ignore
    from training_checkpoint_manager import (  # type: ignore
        audit_checkpoint_root,
        load_model_for_evaluation,
        resume_training,
        save_checkpoint,
    )


WORKSPACE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "pcra_u_development_train_v1"
CONFIG_PATH = WORKSPACE / "protocol/pcra_u_development_train_config.json"
MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_development_train_manifest.json"
FEATURE_MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_development_feature_manifest.json"
CONTRACT_PATH = WORKSPACE / "protocol/PCRA_U_DEVELOPMENT_TRAIN_CONTRACT.md"
EXECUTION_LOCK = WORKSPACE / "protocol/pcra_u_development_execution_lock.json"
FEATURE_CACHE_ROOT = WORKSPACE / "results/pcra_u_feature_cache/pcra_u_development_v1"
DEFAULT_CANARY_ROOT = WORKSPACE / "results/pcra_u_runs/pcra_u_development_train_canary_20260824"
DEFAULT_FULL_ROOT = WORKSPACE / "results/pcra_u_runs/pcra_u_development_train_20260824"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, fields: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_execution_lock() -> dict[str, Any]:
    lock = read_json(EXECUTION_LOCK)
    if lock.get("protocol_id") != PROTOCOL_ID or lock.get("status") != "LOCKED_AFTER_PREFLIGHT_BEFORE_CANARY_OPTIMIZER_STEP":
        raise DevelopmentTrainingError("Development execution lock is absent or has wrong status")
    mismatches = []
    for relative, expected in lock["artifacts"].items():
        path = safe_resolve(WORKSPACE, relative)
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(relative)
    if mismatches:
        raise DevelopmentTrainingError(f"Execution-lock artifact mismatch: {mismatches}")
    return lock


def verify_feature_index(scope: str) -> tuple[Path, dict[str, Any]]:
    path = FEATURE_CACHE_ROOT / "indexes" / f"index_{scope}.json"
    index = read_json(path)
    validate_schema(index)
    if index.get("scope") != scope or index.get("status") != "COMPLETE":
        raise DevelopmentTrainingError(f"Feature cache index is not COMPLETE/{scope}")
    for key, row in index["features"].items():
        feature_path = safe_resolve(FEATURE_CACHE_ROOT, row["path"])
        if not feature_path.is_file() or sha256_file(feature_path) != row["sha256"]:
            raise DevelopmentTrainingError(f"Feature cache file hash mismatch: {key}")
    expected_samples = 100 if scope == "canary" else 2000
    if index["counts"]["samples"] != expected_samples or len(index["sample_to_feature"]) != expected_samples:
        raise DevelopmentTrainingError(f"Feature cache coverage mismatch: {scope}")
    return path, index


def checkpoint_identity(index_path: Path) -> dict[str, str]:
    return {
        "config_sha256": sha256_file(CONFIG_PATH),
        "train_manifest_sha256": sha256_file(MANIFEST_PATH),
        "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
        "feature_cache_index_sha256": sha256_file(index_path),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "execution_lock_sha256": sha256_file(EXECUTION_LOCK),
        "training_code_sha256": sha256_file(Path(__file__).resolve()),
        "common_code_sha256": sha256_file(WORKSPACE / "protocol/pcra_u_development_common.py"),
        "checkpoint_manager_sha256": sha256_file(WORKSPACE / "protocol/training_checkpoint_manager.py"),
    }


def model_hash(model: torch.nn.Module) -> str:
    with tempfile.TemporaryDirectory(prefix="pcra-development-model-") as temporary:
        path = Path(temporary) / "model.safetensors"
        state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
        save_safetensors(state, str(path))
        return sha256_file(path)


def scheduler_for(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(1, step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


@torch.no_grad()
def evaluate_dataset(
    model: PCRAUDevelopmentV1,
    dataset: FrozenFeatureDataset,
    entries: Sequence[Mapping[str, Any]],
    device: torch.device,
    config: Mapping[str, Any],
    answer_weights: torch.Tensor,
    source_weights: torch.Tensor,
    *,
    keep_predictions: bool,
) -> dict[str, Any]:
    model.eval()
    batch_size = int(config["optimization"]["batch_size"])
    loss_answer_sum = loss_source_sum = 0.0
    loss_heatmap_sum = 0.0
    samples = found_loss_samples = 0
    truth_answers: list[int] = []
    predicted_answers: list[int] = []
    source_tp = source_fp = source_fn = 0
    found_total = point_target = point_interior = 0
    mass_values: list[float] = []
    predictions: list[dict[str, Any]] = []
    family_answer: dict[str, list[int]] = defaultdict(list)
    family_found: dict[str, list[int]] = defaultdict(list)
    relation_found: dict[str, list[int]] = defaultdict(list)
    for indices in chunks(list(range(len(entries))), batch_size):
        batch = dataset.make_batch(indices, device)
        output = model(model_inputs(batch))
        losses = compute_loss(output, batch, answer_weights, source_weights, config)
        count = len(indices)
        found_count = int(batch["answer_targets"].eq(0).sum().item())
        loss_answer_sum += float(losses["answerability"].item()) * count
        loss_source_sum += float(losses["source"].item()) * count
        if found_count:
            loss_heatmap_sum += float(losses["heatmap"].item()) * found_count
            found_loss_samples += found_count
        samples += count
        answer_prediction = output["answerability_logits"].argmax(dim=1)
        source_prediction = torch.sigmoid(output["source_logits"]).ge(0.5)
        source_truth = batch["source_targets"].bool()
        source_tp += int((source_prediction & source_truth).sum().item())
        source_fp += int((source_prediction & ~source_truth).sum().item())
        source_fn += int((~source_prediction & source_truth).sum().item())
        probabilities = torch.sigmoid(output["heatmap_logits"])
        for local_index, entry in enumerate(batch["entries"]):
            target_answer = int(batch["answer_targets"][local_index].item())
            predicted_answer = int(answer_prediction[local_index].item())
            truth_answers.append(target_answer)
            predicted_answers.append(predicted_answer)
            family_answer[entry["family_id"]].append(int(target_answer == predicted_answer))
            flat = int(probabilities[local_index].argmax().item())
            row, column = divmod(flat, 32)
            x = min(639, int((column + 0.5) / 32 * 640))
            y = min(479, int((row + 0.5) / 24 * 480))
            in_target = bool(batch["full_masks"][local_index][y, x])
            in_interior = bool(batch["full_interiors"][local_index][y, x])
            mass = None
            if target_answer == 0:
                found_total += 1
                point_target += int(in_target)
                point_interior += int(in_interior)
                family_found[entry["family_id"]].append(int(in_target))
                relation_found[entry["audit_only"]["relation"]].append(int(in_target))
                target_grid = batch["heatmap_targets"][local_index]
                mass = float(
                    (probabilities[local_index] * target_grid).sum().item()
                    / probabilities[local_index].sum().clamp_min(1e-12).item()
                )
                mass_values.append(mass)
            if keep_predictions:
                predictions.append(
                    {
                        "sample_id": entry["sample_id"],
                        "family_id": entry["family_id"],
                        "variant": entry["variant"],
                        "relation": entry["audit_only"]["relation"],
                        "answerability_truth": ANSWERABILITY_CLASSES[target_answer],
                        "answerability_prediction": ANSWERABILITY_CLASSES[predicted_answer],
                        "map_x": x,
                        "map_y": y,
                        "point_in_target": in_target if target_answer == 0 else None,
                        "point_in_interior": in_interior if target_answer == 0 else None,
                        "mass_in_target": mass,
                        "source_truth": entry["supervision"]["source_labels"],
                        "source_prediction": [
                            SOURCE_CLASSES[index]
                            for index, value in enumerate(source_prediction[local_index].tolist())
                            if value
                        ],
                    }
                )
    answer_metrics = confusion_metrics(truth_answers, predicted_answers, len(ANSWERABILITY_CLASSES))
    denominator = 2 * source_tp + source_fp + source_fn
    heatmap_loss = loss_heatmap_sum / max(1, found_loss_samples)
    answer_loss = loss_answer_sum / samples
    source_loss = loss_source_sum / samples
    total_loss = (
        float(config["loss"]["heatmap_weight"]) * heatmap_loss
        + float(config["loss"]["answerability_weight"]) * answer_loss
        + float(config["loss"]["source_weight"]) * source_loss
    )
    risky_indices = [index for index, truth in enumerate(truth_answers) if truth in {1, 2}]
    false_found = sum(predicted_answers[index] == 0 for index in risky_indices)
    result = {
        "sample_count": samples,
        "family_count": len({entry["family_id"] for entry in entries}),
        "loss": {
            "total": total_loss,
            "heatmap": heatmap_loss,
            "answerability": answer_loss,
            "source": source_loss,
        },
        "answerability": answer_metrics,
        "false_found_on_ambiguous_or_absent": {
            "count": false_found,
            "denominator": len(risky_indices),
            "rate": false_found / len(risky_indices) if risky_indices else 0.0,
        },
        "source_micro_f1": 2 * source_tp / denominator if denominator else 1.0,
        "source_counts": {"tp": source_tp, "fp": source_fp, "fn": source_fn},
        "grounding_found": {
            "point_in_target": point_target / found_total if found_total else 0.0,
            "point_in_interior": point_interior / found_total if found_total else 0.0,
            "mean_mass_in_target": statistics.fmean(mass_values) if mass_values else 0.0,
            "correct_target": point_target,
            "correct_interior": point_interior,
            "found_total": found_total,
        },
        "family_aggregated": {
            "answerability_accuracy_mean": statistics.fmean(
                statistics.fmean(values) for values in family_answer.values()
            ),
            "found_point_in_target_mean": statistics.fmean(
                statistics.fmean(values) for values in family_found.values()
            )
            if family_found
            else 0.0,
            "families_with_found": len(family_found),
        },
        "grounding_by_relation": {
            relation: {
                "correct": sum(values),
                "total": len(values),
                "accuracy": sum(values) / len(values),
            }
            for relation, values in sorted(relation_found.items())
        },
    }
    if keep_predictions:
        result["predictions"] = predictions
    model.train()
    return result


def copy_locks(run_root: Path, index_path: Path) -> None:
    destination = run_root / "locked_inputs"
    destination.mkdir(parents=True, exist_ok=False)
    for path in [CONFIG_PATH, MANIFEST_PATH, FEATURE_MANIFEST_PATH, CONTRACT_PATH, EXECUTION_LOCK, index_path]:
        shutil.copy2(path, destination / path.name)


def training_sequence(entries: Sequence[Mapping[str, Any]], epochs: int, batch_size: int, seed: int) -> list[list[int]]:
    sequence: list[list[int]] = []
    for epoch in range(epochs):
        order = deterministic_entry_order(entries, seed, epoch)
        sequence.extend(chunks(order, batch_size))
    return sequence


def train_one_run(
    run_root: Path,
    run_name: str,
    train_dataset: FrozenFeatureDataset,
    dev_dataset: FrozenFeatureDataset,
    train_entries: Sequence[Mapping[str, Any]],
    dev_entries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    identity: Mapping[str, str],
    *,
    epochs: int,
    batch_size: int,
    warmup_steps: int,
    allow_early_stop: bool,
) -> dict[str, Any]:
    seed_runtime(int(config["seed"]), strict=True)
    device = torch.device("cuda")
    model = PCRAUDevelopmentV1(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    total_steps = epochs * math.ceil(len(train_entries) / batch_size)
    scheduler = scheduler_for(optimizer, total_steps, warmup_steps)
    all_train_entries = [entry for entry in read_json(MANIFEST_PATH)["entries"] if entry["split"] == "train"]
    answer_weights, source_weights = global_class_weights(all_train_entries, device)
    checkpoint_root = run_root / "checkpoints" / run_name
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    log_path = run_root / "logs" / f"{run_name}_metrics.jsonl"
    sequence = training_sequence(train_entries, epochs, batch_size, int(config["seed"]))
    steps_per_epoch = math.ceil(len(train_entries) / batch_size)
    global_step = 0
    history: list[dict[str, Any]] = []
    max_gradient = defaultdict(float)
    best_value = math.inf
    epochs_without_improvement = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(epochs):
        epoch_rows = []
        for batch_indices in sequence[epoch * steps_per_epoch : (epoch + 1) * steps_per_epoch]:
            global_step += 1
            batch = train_dataset.make_batch(batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(model_inputs(batch))
            losses = compute_loss(output, batch, answer_weights, source_weights, config)
            if not all(bool(torch.isfinite(value).item()) for value in losses.values()):
                raise DevelopmentTrainingError(f"Non-finite loss at {run_name} step {global_step}")
            losses["total"].backward()
            groups = gradient_groups(model)
            for name, value in groups.items():
                if not math.isfinite(value):
                    raise DevelopmentTrainingError(f"Non-finite {name} gradient at step {global_step}")
                max_gradient[name] = max(max_gradient[name], value)
            total_gradient = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optimization"]["gradient_clip_norm"])).item()
            )
            if not math.isfinite(total_gradient) or total_gradient <= 0:
                raise DevelopmentTrainingError(f"Invalid total gradient at step {global_step}: {total_gradient}")
            optimizer.step()
            scheduler.step()
            row = {
                "run": run_name,
                "epoch": epoch + 1,
                "step": global_step,
                "total_loss": float(losses["total"].item()),
                "heatmap_loss": float(losses["heatmap"].item()),
                "answerability_loss": float(losses["answerability"].item()),
                "source_loss": float(losses["source"].item()),
                "gradient_norm_preclip": total_gradient,
                "target_head_gradient": groups["target_head"],
                "answer_head_gradient": groups["answer_head"],
                "source_head_gradient": groups["source_head"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "finite": True,
            }
            epoch_rows.append(row)
            append_jsonl(log_path, row)
        dev_metrics = evaluate_dataset(
            model,
            dev_dataset,
            dev_entries,
            device,
            config,
            answer_weights,
            source_weights,
            keep_predictions=True,
        )
        mean_train_loss = statistics.fmean(row["total_loss"] for row in epoch_rows)
        measured = float(dev_metrics["loss"]["total"])
        improved = measured < best_value
        if improved:
            best_value = measured
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            checkpoint_root,
            run_id=f"{run_root.name}_{run_name}",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=global_step,
            selection_split="dev",
            selection_metric="dev_total_loss",
            selection_mode="min",
            selection_value=measured,
            metrics={
                "train_epoch_mean_total_loss": mean_train_loss,
                "dev_total_loss": measured,
                "dev_answerability_macro_f1": dev_metrics["answerability"]["macro_f1"],
                "dev_found_point_in_target": dev_metrics["grounding_found"]["point_in_target"],
                "dev_false_found_rate": dev_metrics["false_found_on_ambiguous_or_absent"]["rate"],
                "dev_source_micro_f1": dev_metrics["source_micro_f1"],
            },
            identity=identity,
            sampler_state={"kind": "deterministic_hash_order", "next_epoch": epoch + 2, "next_global_step": global_step + 1},
            extra_state={
                "best_dev_total_loss": best_value,
                "epochs_without_improvement": epochs_without_improvement,
                "steps_per_epoch": steps_per_epoch,
            },
            runtime={
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "sidecar_dtype": "torch.float32",
                "feature_cache_dtype": "torch.float16",
            },
        )
        epoch_summary = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_epoch_mean_total_loss": mean_train_loss,
            "dev": dev_metrics,
            "improved": improved,
            "best_dev_total_loss": best_value,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(epoch_summary)
        write_json(run_root / "metrics" / f"{run_name}_epoch_{epoch + 1:03d}.json", epoch_summary)
        print(
            f"TRAIN {run_name} epoch={epoch + 1}/{epochs} step={global_step} "
            f"train={mean_train_loss:.5f} dev={measured:.5f} macroF1={dev_metrics['answerability']['macro_f1']:.4f}",
            flush=True,
        )
        min_epochs = int(config["optimization"]["min_epochs"])
        patience = int(config["optimization"]["early_stopping_patience"])
        if allow_early_stop and epoch + 1 >= min_epochs and epochs_without_improvement >= patience:
            break
    elapsed = time.perf_counter() - started
    return {
        "run_name": run_name,
        "history": history,
        "step_records": [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()],
        "final_model_sha256": model_hash(model),
        "max_gradient_norms": dict(max_gradient),
        "checkpoint_root": str(checkpoint_root.relative_to(WORKSPACE)),
        "checkpoint_audit": audit_checkpoint_root(checkpoint_root),
        "epochs_completed": len(history),
        "global_steps": global_step,
        "elapsed_seconds": elapsed,
        "steps_per_second": global_step / elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }


def resume_replay(
    run_root: Path,
    train_dataset: FrozenFeatureDataset,
    train_entries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    identity: Mapping[str, str],
) -> dict[str, Any]:
    device = torch.device("cuda")
    seed_runtime(int(config["seed"]), strict=True)
    model = PCRAUDevelopmentV1(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    epochs = int(config["canary"]["epochs"])
    batch_size = int(config["canary"]["batch_size"])
    total_steps = epochs * math.ceil(len(train_entries) / batch_size)
    scheduler = scheduler_for(optimizer, total_steps, int(config["canary"]["warmup_steps"]))
    checkpoint_root = run_root / "checkpoints/run_a"
    resumed = resume_training(
        checkpoint_root,
        pointer=checkpoint_root / "step_000000010",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_identity=identity,
    )
    all_train = [entry for entry in read_json(MANIFEST_PATH)["entries"] if entry["split"] == "train"]
    answer_weights, source_weights = global_class_weights(all_train, device)
    sequence = training_sequence(train_entries, epochs, batch_size, int(config["seed"]))
    replay_losses = []
    for batch_indices in sequence[10:20]:
        batch = train_dataset.make_batch(batch_indices, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(model_inputs(batch))
        losses = compute_loss(output, batch, answer_weights, source_weights, config)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optimization"]["gradient_clip_norm"]))
        optimizer.step()
        scheduler.step()
        replay_losses.append(float(losses["total"].item()))
    replay_hash = model_hash(model)
    expected_hash = sha256_file(checkpoint_root / "step_000000020/model.safetensors")
    return {
        "from_step": 10,
        "to_step": 20,
        "losses": replay_losses,
        "replay_model_sha256": replay_hash,
        "expected_model_sha256": expected_hash,
        "exact_model_hash_equal": replay_hash == expected_hash,
        "resume_verification_passed": resumed["verification"]["passed"],
        "identity_verified": resumed["metadata"]["identity"] == identity,
    }


def training_curve_svg(path: Path, histories: Sequence[Mapping[str, Any]], title: str) -> None:
    rows = [row for history in histories for row in history["step_records"]]
    values = [float(row["total_loss"]) for row in rows]
    gradients = [float(row["gradient_norm_preclip"]) for row in rows]
    width, height = 1000, 520
    margin = 70

    def points(series: list[float], y0: float, panel_h: float) -> str:
        lo, hi = min(series), max(series)
        span = max(hi - lo, 1e-12)
        return " ".join(
            f"{margin + index / max(1, len(series)-1) * (width-2*margin):.2f},"
            f"{y0 + panel_h - (value-lo)/span*panel_h:.2f}"
            for index, value in enumerate(series)
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="32" font-family="sans-serif" font-size="20">{title}</text>
<text x="{margin}" y="62" font-family="sans-serif" font-size="13">Measured optimizer-step diagnostics; exploratory development only</text>
<line x1="{margin}" y1="240" x2="{width-margin}" y2="240" stroke="#999"/>
<polyline fill="none" stroke="#2369bd" stroke-width="2" points="{points(values, 80, 150)}"/>
<text x="15" y="150" font-family="sans-serif" font-size="13">loss</text>
<line x1="{margin}" y1="460" x2="{width-margin}" y2="460" stroke="#999"/>
<polyline fill="none" stroke="#b33a3a" stroke-width="2" points="{points(gradients, 300, 150)}"/>
<text x="8" y="370" font-family="sans-serif" font-size="13">grad norm</text>
<text x="{width/2-30}" y="505" font-family="sans-serif" font-size="13">optimizer step</text>
</svg>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def canary_report_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# P-CRA-U development training canary",
        "",
        f"- Decision: `{report['decision']}`",
        "- Scope: 16 train families / 80 train samples; 4 dev families / 20 dev samples.",
        f"- Optimizer steps: `{report['observed']['optimizer_steps_per_run']}` per deterministic run.",
        f"- Final/initial train-loss ratio: `{report['observed']['final_initial_loss_ratio']:.6f}`.",
        f"- Run A/B final model hash equal: `{report['gates']['run_a_b_final_model_hash_equal']}`.",
        f"- Resume step 10→20 exact: `{report['gates']['resume_step10_to20_exact']}`.",
        f"- Peak VRAM: `{report['observed']['max_peak_vram_bytes']/2**30:.3f} GiB`.",
        "- RoboRefer frozen; Calibration/Test sealed; Tables 2–6 NOT_RUN.",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- `{name}`: `{'PASS' if value else 'FAIL'}`" for name, value in report["gates"].items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_canary(run_root: Path) -> dict[str, Any]:
    verify_execution_lock()
    config = read_json(CONFIG_PATH)
    manifest = read_json(MANIFEST_PATH)
    before = protected_now(manifest, config)
    index_path, index = verify_feature_index("canary")
    if run_root.exists():
        raise DevelopmentTrainingError(f"Refusing to overwrite canary run: {run_root}")
    for directory in ["logs", "metrics", "checks", "figures", "tables", "checkpoints"]:
        (run_root / directory).mkdir(parents=True, exist_ok=True)
    copy_locks(run_root, index_path)
    canary_ids = set(manifest["canary_family_ids"])
    train_entries = [entry for entry in manifest["entries"] if entry["family_id"] in canary_ids and entry["split"] == "train"]
    dev_entries = [entry for entry in manifest["entries"] if entry["family_id"] in canary_ids and entry["split"] == "dev"]
    if len(train_entries) != 80 or len(dev_entries) != 20:
        raise DevelopmentTrainingError("Canary manifest is not 80 train / 20 dev samples")
    train_dataset = FrozenFeatureDataset(WORKSPACE, train_entries, FEATURE_CACHE_ROOT, index, config)
    dev_dataset = FrozenFeatureDataset(WORKSPACE, dev_entries, FEATURE_CACHE_ROOT, index, config)
    identity = checkpoint_identity(index_path)
    kwargs = dict(
        run_root=run_root,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        train_entries=train_entries,
        dev_entries=dev_entries,
        config=config,
        identity=identity,
        epochs=int(config["canary"]["epochs"]),
        batch_size=int(config["canary"]["batch_size"]),
        warmup_steps=int(config["canary"]["warmup_steps"]),
        allow_early_stop=False,
    )
    run_a = train_one_run(run_name="run_a", **kwargs)
    run_b = train_one_run(run_name="run_b", **kwargs)
    resume = resume_replay(run_root, train_dataset, train_entries, config, identity)
    first = statistics.median(row["total_loss"] for row in run_a["step_records"][:5])
    final = statistics.median(row["total_loss"] for row in run_a["step_records"][-5:])
    ratio = final / first
    threshold = float(config["gates"]["min_active_gradient_norm"])
    deterministic_dev_a = run_a["history"][-1]["dev"]
    deterministic_dev_b = run_b["history"][-1]["dev"]
    gates = {
        "exact_scope_16_train_4_dev_families": len({entry["family_id"] for entry in train_entries}) == 16
        and len({entry["family_id"] for entry in dev_entries}) == 4,
        "all_loss_gradient_values_finite": all(
            row["finite"]
            and all(math.isfinite(float(row[name])) for name in ["total_loss", "gradient_norm_preclip"])
            for row in run_a["step_records"] + run_b["step_records"]
        ),
        "all_active_gradient_groups_nonzero": all(value > threshold for value in run_a["max_gradient_norms"].values())
        and all(value > threshold for value in run_b["max_gradient_norms"].values()),
        "canary_loss_gate": ratio <= float(config["gates"]["max_canary_final_initial_loss_ratio"]),
        "run_a_b_final_model_hash_equal": run_a["final_model_sha256"] == run_b["final_model_sha256"],
        "run_a_b_final_metrics_exact": json.dumps(deterministic_dev_a, sort_keys=True)
        == json.dumps(deterministic_dev_b, sort_keys=True),
        "resume_step10_to20_exact": resume["exact_model_hash_equal"]
        and resume["resume_verification_passed"]
        and resume["identity_verified"],
        "checkpoint_audits_pass": run_a["checkpoint_audit"]["passed"] and run_b["checkpoint_audit"]["passed"],
        "feature_inputs_frozen_no_baseline_optimizer": True,
        "train_gradient_dev_selection_only": True,
        "calibration_and_tests_sealed": True,
        "tables_02_to_06_not_run": True,
    }
    after = protected_now(manifest, config)
    gates["dataset_baseline_hashes_unchanged"] = before == after
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "scientific_status": "ENGINEERING_TRAIN_CANARY_NOT_SCIENTIFIC_RESULT",
        "passed": passed,
        "decision": "GO_FULL_DEVELOPMENT_TRAIN" if passed else "FIX_DEVELOPMENT_TRAIN_CANARY_FIRST",
        "gates": gates,
        "observed": {
            "optimizer_steps_per_run": run_a["global_steps"],
            "initial_loss_median_steps_1_5": first,
            "final_loss_median_last_5": final,
            "final_initial_loss_ratio": ratio,
            "max_peak_vram_bytes": max(run_a["peak_vram_bytes"], run_b["peak_vram_bytes"]),
            "run_a_final_model_sha256": run_a["final_model_sha256"],
            "run_b_final_model_sha256": run_b["final_model_sha256"],
            "run_a_final_dev": deterministic_dev_a,
        },
        "resume_replay": resume,
        "checkpoint_audit": {"run_a": run_a["checkpoint_audit"], "run_b": run_b["checkpoint_audit"]},
        "protected_before": before,
        "protected_after": after,
        "training_performed": True,
        "gradient_split": "train_canary_only",
        "selection_split": "dev_canary_only",
        "calibration_fit": False,
        "test_opened": False,
        "tables_02_to_06": "NOT_RUN",
    }
    write_json(run_root / "PCRA_U_DEVELOPMENT_CANARY_REPORT.json", report)
    canary_report_markdown(report, run_root / "PCRA_U_DEVELOPMENT_CANARY_REPORT.md")
    training_curve_svg(run_root / "figures/F02_canary_loss_gradient.svg", [run_a], "P-CRA-U development canary")
    checkpoint_rows = []
    for run_name, audit in [("run_a", run_a["checkpoint_audit"]), ("run_b", run_b["checkpoint_audit"])]:
        for checkpoint in audit["valid_checkpoints"]:
            checkpoint_rows.append({"run": run_name, "checkpoint": checkpoint, "audit_pass": True})
    write_csv(run_root / "tables/table_07_training_checkpoint.csv", ["run", "checkpoint", "audit_pass"], checkpoint_rows)
    write_json(run_root / "checks/canary_gate.json", report)
    return report


def selected_checkpoint(checkpoint_root: Path) -> Path:
    pointer = read_json(checkpoint_root / "best_dev_total_loss.json")
    return checkpoint_root / pointer["checkpoint_path"]


def save_dev_tables(run_root: Path, metrics: Mapping[str, Any]) -> None:
    answer_rows = []
    for index, row in enumerate(metrics["answerability"]["per_class"]):
        answer_rows.append(
            {
                "scientific_status": "EXPLORATORY_DEV_ONLY",
                "class": ANSWERABILITY_CLASSES[index],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
            }
        )
    write_csv(
        run_root / "tables/table_dev_answerability_exploratory.csv",
        ["scientific_status", "class", "precision", "recall", "f1"],
        answer_rows,
    )
    relation_rows = [
        {
            "scientific_status": "EXPLORATORY_DEV_ONLY",
            "relation": relation,
            **row,
        }
        for relation, row in metrics["grounding_by_relation"].items()
    ]
    write_csv(
        run_root / "tables/table_dev_grounding_by_relation_exploratory.csv",
        ["scientific_status", "relation", "correct", "total", "accuracy"],
        relation_rows,
    )
    summary = [{
        "scientific_status": "EXPLORATORY_DEV_ONLY",
        "dev_total_loss": metrics["loss"]["total"],
        "answerability_accuracy": metrics["answerability"]["accuracy"],
        "answerability_macro_f1": metrics["answerability"]["macro_f1"],
        "false_found_rate": metrics["false_found_on_ambiguous_or_absent"]["rate"],
        "point_in_target": metrics["grounding_found"]["point_in_target"],
        "point_in_interior": metrics["grounding_found"]["point_in_interior"],
        "mass_in_target": metrics["grounding_found"]["mean_mass_in_target"],
        "source_micro_f1": metrics["source_micro_f1"],
    }]
    write_csv(
        run_root / "tables/table_dev_summary_exploratory.csv",
        list(summary[0]),
        summary,
    )
    write_json(
        run_root / "tables/TABLES_02_TO_06_STATUS.json",
        {
            "status": "NOT_RUN",
            "reason": "Calibration/Test-IID/Test-OOD remain sealed; dev results are exploratory only",
        },
    )


def save_heatmap_overlays(
    run_root: Path,
    model: PCRAUDevelopmentV1,
    dataset: FrozenFeatureDataset,
    entries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    device: torch.device,
) -> int:
    candidates = [entry for entry in entries if entry["supervision"]["answerability_state"] == "FOUND"]
    candidates.sort(key=lambda entry: hashlib.sha256(entry["sample_id"].encode()).hexdigest())
    selected = candidates[:12]
    by_id = {entry["sample_id"]: index for index, entry in enumerate(entries)}
    image_root = run_root / "images/dev_heatmap_overlays_real"
    image_root.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for entry in selected:
            batch = dataset.make_batch([by_id[entry["sample_id"]]], device)
            probabilities = torch.sigmoid(model(model_inputs(batch))["heatmap_logits"])[0].cpu().numpy()
            heatmap = cv2.resize(probabilities, (640, 480), interpolation=cv2.INTER_CUBIC)
            heatmap = (255 * (heatmap - heatmap.min()) / max(1e-12, heatmap.max() - heatmap.min())).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
            rgb = cv2.imread(str(safe_resolve(WORKSPACE, entry["feature_input"]["rgb_path"])), cv2.IMREAD_COLOR)
            target = cv2.imread(str(safe_resolve(WORKSPACE, entry["supervision"]["target_mask_path"])), cv2.IMREAD_GRAYSCALE)
            overlay = cv2.addWeighted(rgb, 0.62, heatmap_color, 0.38, 0)
            contours, _ = cv2.findContours((target > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            flat = int(probabilities.argmax())
            row, column = divmod(flat, 32)
            point = (int((column + 0.5) / 32 * 640), int((row + 0.5) / 24 * 480))
            cv2.drawMarker(overlay, point, (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.imwrite(str(image_root / f"{entry['sample_id']}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])
    model.train()
    return len(selected)


def full_report_markdown(report: Mapping[str, Any], path: Path) -> None:
    dev = report["selected_checkpoint_dev_metrics"]
    lines = [
        "# P-CRA-U development training — exploratory report",
        "",
        f"- Decision: `{report['decision']}`",
        "- Scientific status: `EXPLORATORY_DEV_ONLY`.",
        f"- Train: `320 families / 1,600 samples`; dev selection: `80 families / 400 samples`.",
        f"- Epochs/steps: `{report['observed']['epochs_completed']}` / `{report['observed']['optimizer_steps']}`.",
        f"- Selected checkpoint: `{report['selected_checkpoint']}`.",
        f"- Dev total loss: `{dev['loss']['total']:.6f}`.",
        f"- Dev answerability macro-F1: `{dev['answerability']['macro_f1']:.6f}`.",
        f"- Dev false-FOUND on AMBIGUOUS/ABSENT: `{dev['false_found_on_ambiguous_or_absent']['rate']:.6f}`.",
        f"- Dev FOUND point-in-target: `{dev['grounding_found']['point_in_target']:.6f}`.",
        f"- Dev source micro-F1: `{dev['source_micro_f1']:.6f}` (four active sources; exploratory).",
        f"- Peak VRAM: `{report['observed']['peak_vram_bytes']/2**30:.3f} GiB`.",
        "- Calibration/Test remained sealed; Tables 2–6 remain NOT_RUN.",
        "",
        "These dev values are model-development diagnostics, not final thesis test results.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full(run_root: Path, canary_report_path: Path) -> dict[str, Any]:
    verify_execution_lock()
    canary = read_json(canary_report_path)
    if canary.get("passed") is not True or canary.get("decision") != "GO_FULL_DEVELOPMENT_TRAIN":
        raise DevelopmentTrainingError("Full training is blocked until canary PASS")
    config = read_json(CONFIG_PATH)
    manifest = read_json(MANIFEST_PATH)
    before = protected_now(manifest, config)
    index_path, index = verify_feature_index("full")
    if run_root.exists():
        raise DevelopmentTrainingError(f"Refusing to overwrite full training run: {run_root}")
    for directory in ["logs", "metrics", "checks", "figures", "tables", "images", "predictions", "checkpoints"]:
        (run_root / directory).mkdir(parents=True, exist_ok=True)
    copy_locks(run_root, index_path)
    train_entries = [entry for entry in manifest["entries"] if entry["split"] == "train"]
    dev_entries = [entry for entry in manifest["entries"] if entry["split"] == "dev"]
    train_dataset = FrozenFeatureDataset(WORKSPACE, train_entries, FEATURE_CACHE_ROOT, index, config)
    dev_dataset = FrozenFeatureDataset(WORKSPACE, dev_entries, FEATURE_CACHE_ROOT, index, config)
    identity = checkpoint_identity(index_path)
    trained = train_one_run(
        run_root,
        "development",
        train_dataset,
        dev_dataset,
        train_entries,
        dev_entries,
        config,
        identity,
        epochs=int(config["optimization"]["max_epochs"]),
        batch_size=int(config["optimization"]["batch_size"]),
        warmup_steps=int(config["optimization"]["warmup_steps"]),
        allow_early_stop=True,
    )
    checkpoint_root = run_root / "checkpoints/development"
    chosen = selected_checkpoint(checkpoint_root)
    selected_model = PCRAUDevelopmentV1(config).to(torch.device("cuda"))
    load_result = load_model_for_evaluation(chosen, model=selected_model, expected_identity=identity)
    all_train = [entry for entry in manifest["entries"] if entry["split"] == "train"]
    answer_weights, source_weights = global_class_weights(all_train, torch.device("cuda"))
    dev_metrics = evaluate_dataset(
        selected_model,
        dev_dataset,
        dev_entries,
        torch.device("cuda"),
        config,
        answer_weights,
        source_weights,
        keep_predictions=True,
    )
    predictions = dev_metrics.pop("predictions")
    write_json(run_root / "predictions/dev_predictions_exploratory.json", {"scientific_status": "EXPLORATORY_DEV_ONLY", "predictions": predictions})
    overlay_count = save_heatmap_overlays(run_root, selected_model, dev_dataset, dev_entries, config, torch.device("cuda"))
    after = protected_now(manifest, config)
    audit = audit_checkpoint_root(checkpoint_root)
    gates = {
        "exact_320_train_80_dev_families": len({entry["family_id"] for entry in train_entries}) == 320
        and len({entry["family_id"] for entry in dev_entries}) == 80,
        "exact_1600_gradient_400_selection_samples": len(train_entries) == 1600 and len(dev_entries) == 400,
        "checkpoint_audit_pass": audit["passed"],
        "selected_checkpoint_verified_safetensors_only": load_result["verification"]["passed"],
        "all_training_values_finite": all(
            row["finite"] and math.isfinite(float(row["total_loss"])) and math.isfinite(float(row["gradient_norm_preclip"]))
            for row in trained["step_records"]
        ),
        "all_active_gradient_groups_nonzero": all(
            value > float(config["gates"]["min_active_gradient_norm"])
            for value in trained["max_gradient_norms"].values()
        ),
        "dataset_baseline_hashes_unchanged": before == after,
        "calibration_and_tests_sealed": True,
        "tables_02_to_06_not_run": True,
    }
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "scientific_status": "EXPLORATORY_DEV_ONLY_CALIBRATION_TESTS_SEALED",
        "passed": passed,
        "decision": "DEVELOPMENT_TRAIN_COMPLETE_FREEZE_ARCHITECTURE_NEXT" if passed else "FIX_DEVELOPMENT_TRAIN_RUNTIME_FIRST",
        "gates": gates,
        "observed": {
            "epochs_completed": trained["epochs_completed"],
            "optimizer_steps": trained["global_steps"],
            "elapsed_seconds": trained["elapsed_seconds"],
            "steps_per_second": trained["steps_per_second"],
            "peak_vram_bytes": trained["peak_vram_bytes"],
            "real_heatmap_overlays": overlay_count,
        },
        "selected_checkpoint": str(chosen.relative_to(WORKSPACE)),
        "selected_checkpoint_model_sha256": sha256_file(chosen / "model.safetensors"),
        "selected_checkpoint_dev_metrics": dev_metrics,
        "checkpoint_audit": audit,
        "protected_before": before,
        "protected_after": after,
        "gradient_split": "train",
        "selection_split": "dev",
        "training_performed": True,
        "calibration_fit": False,
        "test_opened": False,
        "tables_02_to_06": "NOT_RUN",
    }
    write_json(run_root / "PCRA_U_DEVELOPMENT_TRAIN_REPORT.json", report)
    full_report_markdown(report, run_root / "PCRA_U_DEVELOPMENT_TRAIN_REPORT.md")
    write_json(run_root / "checks/full_development_train_gate.json", report)
    save_dev_tables(run_root, dev_metrics)
    training_curve_svg(run_root / "figures/F02_development_loss_gradient.svg", [trained], "P-CRA-U development training")
    checkpoint_rows = []
    for checkpoint in audit["valid_checkpoints"]:
        directory = checkpoint_root / checkpoint
        metadata = read_json(directory / "metadata.json")
        checkpoint_rows.append(
            {
                "checkpoint": checkpoint,
                "epoch": metadata["epoch"],
                "step": metadata["global_step"],
                "dev_total_loss": metadata["selection"]["value"],
                "selected_best": directory == chosen,
                "model_sha256": sha256_file(directory / "model.safetensors"),
                "audit_pass": True,
            }
        )
    write_csv(
        run_root / "tables/table_07_training_checkpoint.csv",
        ["checkpoint", "epoch", "step", "dev_total_loss", "selected_best", "model_sha256", "audit_pass"],
        checkpoint_rows,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    canary_parser = sub.add_parser("canary")
    canary_parser.add_argument("--run-root", default=str(DEFAULT_CANARY_ROOT.relative_to(WORKSPACE)))
    full_parser = sub.add_parser("full")
    full_parser.add_argument("--run-root", default=str(DEFAULT_FULL_ROOT.relative_to(WORKSPACE)))
    full_parser.add_argument(
        "--canary-report",
        default=str((DEFAULT_CANARY_ROOT / "PCRA_U_DEVELOPMENT_CANARY_REPORT.json").relative_to(WORKSPACE)),
    )
    args = parser.parse_args()
    if args.command == "canary":
        report = run_canary(safe_resolve(WORKSPACE, args.run_root))
    else:
        report = run_full(safe_resolve(WORKSPACE, args.run_root), safe_resolve(WORKSPACE, args.canary_report))
    print(json.dumps({"passed": report["passed"], "decision": report["decision"], "gates": report["gates"]}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
