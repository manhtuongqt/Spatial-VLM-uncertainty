from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from .utils import atomic_json, read_json, sha256_file


def _atomic_safetensors(path: Path, state: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    cpu_state = {name: value.detach().contiguous().cpu() for name, value in state.items()}
    save_file(cpu_state, str(temporary))
    os.replace(temporary, path)


def save_checkpoint(
    directory: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_metric: float,
    config_sha256: str,
    metrics: Mapping[str, Any],
    replace: bool = False,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=replace)
    model_path = directory / "model.safetensors"
    _atomic_safetensors(model_path, model.state_dict())
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
        },
        directory / "trainer_state.pt",
    )
    metadata = {
        "schema_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric if math.isfinite(best_metric) else None,
        "config_sha256": config_sha256,
        "model_sha256": sha256_file(model_path),
        "metrics": dict(metrics),
    }
    atomic_json(directory / "metadata.json", metadata)
    return metadata


def load_model_checkpoint(
    model: torch.nn.Module, checkpoint: str | Path, expected_config_sha256: str | None = None
) -> dict[str, Any]:
    model_path = Path(checkpoint).resolve()
    metadata_path = model_path.parent / "metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete checkpoint: {model_path}")
    metadata = read_json(metadata_path)
    if sha256_file(model_path) != metadata["model_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {model_path}")
    if expected_config_sha256 and metadata["config_sha256"] != expected_config_sha256:
        raise ValueError("Checkpoint was produced by a different configuration")
    model.load_state_dict(load_file(str(model_path), device="cpu"), strict=True)
    return metadata


def load_training_state(
    checkpoint: str | Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, Any]:
    path = Path(checkpoint).resolve().parent / "trainer_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state
