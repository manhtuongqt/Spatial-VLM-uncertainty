#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pcrau.checkpoint import load_model_checkpoint, load_training_state, save_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset, FamilyBatchSampler, collate_samples
from pcrau.engine import evaluate, train_epoch
from pcrau.losses import class_weights
from pcrau.model import PCRAUTargetV2
from pcrau.utils import atomic_json, config_path, load_config, runtime_info, seed_everything, sha256_file, workspace_path


def make_loader(dataset, config, shuffle: bool):
    sampler = FamilyBatchSampler(
        dataset, int(config["optimization"]["families_per_batch"]), int(config["seed"]), shuffle,
        sampling=config.get("sampling") if shuffle else None,
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=int(config["optimization"]["num_workers"]),
        pin_memory=torch.cuda.is_available(), collate_fn=collate_samples,
    )
    return loader, sampler


def main() -> None:
    parser = argparse.ArgumentParser(description="Train complete P-CRA-U target sidecar")
    parser.add_argument("--config")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", help="checkpoint model.safetensors from the same run")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true", help="one train and one dev batch")
    args = parser.parse_args()
    config_file = config_path(args.config)
    config = load_config(config_file)
    config_hash = sha256_file(config_file)
    run_root = workspace_path(config["paths"]["output_root"]) / args.run_name
    if run_root.exists() and not args.resume:
        raise FileExistsError(f"Run exists; choose a new --run-name or pass --resume: {run_root}")
    if not run_root.exists():
        run_root.mkdir(parents=True)
        shutil.copy2(config_file, run_root / "config.json")
        atomic_json(run_root / "run.json", {
            "schema_version": 1, "run_name": args.run_name, "config_sha256": config_hash,
            "runtime": runtime_info(), "scientific_status": "DEVELOPMENT_ONLY",
            "train_families": 320, "dev_families": 80, "sealed_test_splits_accessed": False,
        })
    elif sha256_file(run_root / "config.json") != config_hash:
        raise ValueError("Resume config differs from run config")
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ArchivedPCRAUDataset(config, "train")
    dev_dataset = ArchivedPCRAUDataset(config, "dev")
    train_loader, train_sampler = make_loader(train_dataset, config, True)
    dev_loader, _ = make_loader(dev_dataset, config, False)
    model = PCRAUTargetV2(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    epochs = args.epochs or int(config["optimization"]["epochs"])
    total_steps = max(1, len(train_loader) * epochs)
    warmup = min(int(config["optimization"]["warmup_steps"]), max(0, total_steps - 1))
    def schedule(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    use_fp16_scaler = (
        bool(config["optimization"]["amp"]) and device.type == "cuda"
        and config["optimization"].get("amp_dtype") == "float16"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    answer_weights, source_weights = class_weights(train_dataset, device)
    start_epoch, global_step, best_metric = 0, 0, float("inf")
    history_path = run_root / "history.jsonl"
    if args.resume:
        resume = Path(args.resume).resolve()
        if run_root not in resume.parents:
            raise ValueError("--resume must point inside the selected run directory")
        load_model_checkpoint(model, resume, config_hash)
        state = load_training_state(resume, optimizer, scheduler)
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_metric = float(state["best_metric"])
        if history_path.is_file():
            lines = history_path.read_text(encoding="utf-8").splitlines()
            kept = [line for line in lines if int(json.loads(line)["epoch"]) <= int(state["epoch"])]
            if len(kept) != len(lines):
                backup = run_root / f"history_interrupted_before_resume_epoch_{start_epoch:03d}.jsonl"
                if not backup.exists():
                    shutil.copy2(history_path, backup)
                temporary = history_path.parent / f".{history_path.name}.tmp-{os.getpid()}"
                temporary.write_text("\n".join(kept) + "\n", encoding="utf-8")
                os.replace(temporary, history_path)
    atomic_json(run_root / "sampler.json", train_sampler.summary())
    stale_epochs = 0
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        train_metrics, steps = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, config,
            answer_weights, source_weights, max_batches=1 if args.smoke else None,
        )
        global_step += steps
        dev_metrics, _ = evaluate(
            model, dev_loader, device, config, answer_weights, source_weights,
            include_predictions=False, max_batches=1 if args.smoke else None,
        )
        selected = float(dev_metrics["loss"]["total"])
        minimum_grounding = float(config.get("selection", {}).get("minimum_grounding_accuracy", 0.0))
        grounding = dev_metrics.get("grounding_accuracy")
        eligible = grounding is not None and float(grounding) >= minimum_grounding
        improved = eligible and selected < best_metric
        if improved:
            best_metric, stale_epochs = selected, 0
        elif best_metric < float("inf"):
            stale_epochs += 1
        record = {
            "epoch": epoch, "global_step": global_step, "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics, "dev": dev_metrics, "selected_metric": selected,
            "selection_eligible": eligible, "minimum_grounding_accuracy": minimum_grounding,
            "improved": improved,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        if improved:
            best = run_root / "checkpoints" / "best"
            metadata = save_checkpoint(
                best, model, optimizer, scheduler, epoch, global_step, best_metric,
                config_hash, record, replace=True,
            )
            atomic_json(run_root / "best.json", {"checkpoint": str(best / "model.safetensors"), **metadata})
        checkpoint_every = int(config["optimization"].get("checkpoint_every_epochs", 1))
        periodic = (epoch + 1) % checkpoint_every == 0 or epoch == epochs - 1 or args.smoke
        if periodic:
            checkpoint_dir = run_root / "checkpoints" / f"epoch_{epoch:03d}_step_{global_step:06d}"
            save_checkpoint(
                checkpoint_dir, model, optimizer, scheduler, epoch, global_step,
                best_metric, config_hash, record,
            )
        print(json.dumps({
            "epoch": epoch, "train_total": train_metrics["total"], "dev_total": selected,
            "grounding": grounding, "eligible": eligible,
            "best": None if best_metric == float("inf") else best_metric,
        }))
        if args.smoke or stale_epochs >= int(config["optimization"]["early_stopping_patience"]):
            break


if __name__ == "__main__":
    main()
