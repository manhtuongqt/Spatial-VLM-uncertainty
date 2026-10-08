#!/usr/bin/env python3
"""Locked train-only P-CRA-U overfit smoke on 15 official WP3 feature caches."""

from __future__ import annotations

import os

# Must be set before importing torch/CUDA libraries.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

if __package__:
    from .training_checkpoint_manager import (
        audit_checkpoint_root,
        resume_training,
        save_checkpoint,
        sha256_file,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from training_checkpoint_manager import (  # type: ignore
        audit_checkpoint_root,
        resume_training,
        save_checkpoint,
        sha256_file,
    )


WORKSPACE = Path(__file__).resolve().parents[1]
CONTRACT = WORKSPACE / "protocol/PCRA_U_OVERFIT_SMOKE_CONTRACT.md"
CONFIG_PATH = WORKSPACE / "protocol/pcra_u_v0_overfit_config.json"
MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_overfit_smoke_manifest.json"
SPEC_LOCK_PATH = WORKSPACE / "protocol/pcra_u_overfit_spec_lock.json"
ONE_BATCH_JSON = WORKSPACE / "protocol/PCRA_U_ONE_BATCH_REPORT.json"
ONE_BATCH_MD = WORKSPACE / "protocol/PCRA_U_ONE_BATCH_REPORT.md"
EXECUTION_LOCK_PATH = WORKSPACE / "protocol/pcra_u_overfit_execution_lock.json"
CHECKPOINT_MANAGER = WORKSPACE / "protocol/training_checkpoint_manager.py"
CHECKPOINT_TEST = WORKSPACE / "protocol/test_training_checkpoint_manager.py"
PROTOCOL_ID = "pcra_u_overfit_smoke_v1"


class SmokeError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SmokeError(f"Cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SmokeError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_sha(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def tree_digest(relative_root: str) -> str:
    root = (WORKSPACE / relative_root).resolve()
    if not root.is_dir() or not root.is_relative_to(WORKSPACE):
        raise SmokeError(f"Protected tree missing/unsafe: {relative_root}")
    relative_paths = [
        str(path.relative_to(WORKSPACE))
        for path in root.rglob("*")
        if path.is_file()
    ]
    # WP0--WP3 locks were created with GNU `sort -z`; reproduce that byte
    # ordering exactly (Python's Unicode ordering differs for some filenames).
    sorted_bytes = subprocess.run(
        ["sort", "-z"],
        input=b"\0".join(path.encode("utf-8") for path in relative_paths) + b"\0",
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    digest = hashlib.sha256()
    for raw_relative in sorted_bytes.rstrip(b"\0").split(b"\0"):
        relative = raw_relative.decode("utf-8")
        digest.update(
            f"{sha256_file(WORKSPACE / relative)}  {relative}\n".encode("utf-8")
        )
    return digest.hexdigest()


def protected_snapshot(manifest: Mapping[str, Any]) -> dict[str, Any]:
    protected = manifest["protected_inputs"]
    return {
        "files": {path: sha256_file(WORKSPACE / path) for path in protected["files"]},
        "trees": {path: tree_digest(path) for path in protected["trees"]},
    }


def verify_protected(manifest: Mapping[str, Any]) -> dict[str, Any]:
    observed = protected_snapshot(manifest)
    expected = manifest["protected_inputs"]
    mismatches = []
    for kind in ["files", "trees"]:
        for path, digest in expected[kind].items():
            if observed[kind].get(path) != digest:
                mismatches.append(path)
    if mismatches:
        raise SmokeError(f"Protected input mismatch: {mismatches}")
    return observed


def verify_hash_lock(path: Path, required_status: str | None = None) -> dict[str, Any]:
    lock = load_json(path)
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise SmokeError(f"Wrong protocol lock: {path}")
    if required_status and lock.get("status") != required_status:
        raise SmokeError(f"Lock status mismatch: {lock.get('status')} != {required_status}")
    for relative, expected in lock.get("files", {}).items():
        candidate = WORKSPACE / relative
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise SmokeError(f"Locked file mismatch: {relative}")
    return lock


def seed_runtime(seed: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SmokeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def parse_relation(instruction: str, relations: list[str]) -> int:
    text = " ".join(re.findall(r"[a-z0-9]+", instruction.lower()))
    if "between" in text and "depth" in text:
        relation = "between_in_depth"
    elif "nearer than both" in text or "closer than both" in text:
        relation = "nearer_than_both"
    elif "right of" in text:
        relation = "right_of"
    elif "left of" in text:
        relation = "left_of"
    elif "in front of" in text:
        relation = "front_of"
    elif "behind" in text:
        relation = "behind"
    elif "closer to" in text or "nearer" in text:
        relation = "nearer_than"
    elif "farther" in text or "further" in text:
        relation = "farther_than"
    elif "more elongated" in text or "taller than" in text:
        relation = "shape_comparison"
    elif "colored fruit" in text or "power drill" in text:
        relation = "semantic_reference"
    elif any(phrase in text for phrase in ["point to", "visible point", "red apple"]):
        relation = "direct"
    else:
        relation = "unknown"
    return relations.index(relation)


def tokenize(instruction: str, max_tokens: int, vocab_size: int) -> list[int]:
    values = []
    for token in re.findall(r"[a-z0-9]+", instruction.lower())[:max_tokens]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "big") % (vocab_size - 1) + 1)
    return values or [1]


def pool_feature(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if list(tensor.shape) != [13, 1024, 1152]:
        raise SmokeError(f"Unexpected feature shape: {list(tensor.shape)}")
    local = tensor[:12].float().reshape(12, 32, 32, 1152).permute(0, 3, 1, 2)
    local = F.avg_pool2d(local, kernel_size=4, stride=4)
    global_grid = local.reshape(3, 4, 1152, 8, 8).permute(2, 0, 3, 1, 4).reshape(1152, 24, 32)
    global_grid = global_grid.permute(1, 2, 0).contiguous()
    global_grid = F.layer_norm(global_grid, (1152,))
    thumbnail = tensor[12].float().mean(dim=0)
    thumbnail = F.layer_norm(thumbnail, (1152,))
    return global_grid, thumbnail


def load_smoke_batch(device: torch.device) -> dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    config = load_json(CONFIG_PATH)
    entries = manifest["entries"]
    if len(entries) != 15 or any(item["split"] != "train" for item in entries):
        raise SmokeError("Locked 15-sample train manifest violated")
    rgb_features, depth_features, rgb_thumbnails, depth_thumbnails = [], [], [], []
    masks, full_masks, tokens, relations, answers, sources = [], [], [], [], [], []
    max_tokens = config["language"]["max_tokens"]
    vocab_size = config["language"]["vocab_size"]
    relation_names = config["language"]["relations"]
    for item in entries:
        tensor_path = WORKSPACE / item["feature_tensor_path"]
        metadata_path = WORKSPACE / item["cache_metadata_path"]
        record_path = WORKSPACE / item["record_path"]
        mask_path = WORKSPACE / item["target_mask_path"]
        for path, expected in [
            (tensor_path, item["feature_tensor_sha256"]),
            (metadata_path, item["cache_metadata_sha256"]),
            (record_path, item["record_sha256"]),
            (mask_path, item["target_mask_sha256"]),
            (WORKSPACE / item["rgb_path"], item["rgb_sha256"]),
            (WORKSPACE / item["depth_path"], item["depth_sha256"]),
        ]:
            if not path.is_file() or sha256_file(path) != expected:
                raise SmokeError(f"Locked sample artifact mismatch: {path}")
        tensors = load_safetensors(str(tensor_path), device="cpu")
        if set(tensors) != {"R0", "D0"}:
            raise SmokeError(f"Unexpected tensor keys: {tensor_path}")
        r_grid, r_thumb = pool_feature(tensors["R0"])
        d_grid, d_thumb = pool_feature(tensors["D0"])
        rgb_features.append(r_grid)
        depth_features.append(d_grid)
        rgb_thumbnails.append(r_thumb)
        depth_thumbnails.append(d_thumb)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (480, 640):
            raise SmokeError(f"Invalid target mask: {mask_path}")
        full_masks.append(mask > 0)
        masks.append(cv2.resize((mask > 0).astype(np.float32), (32, 24), interpolation=cv2.INTER_AREA))
        token_values = tokenize(item["instruction"], max_tokens, vocab_size)
        tokens.append(token_values + [0] * (max_tokens - len(token_values)))
        relations.append(parse_relation(item["instruction"], relation_names))
        answers.append(item["answerability_index"])
        sources.append(item["source_multihot"])
    batch = {
        "r0": torch.stack(rgb_features).to(device=device, non_blocking=False),
        "d0": torch.stack(depth_features).to(device=device, non_blocking=False),
        "r_thumb": torch.stack(rgb_thumbnails).to(device=device),
        "d_thumb": torch.stack(depth_thumbnails).to(device=device),
        "tokens": torch.tensor(tokens, dtype=torch.long, device=device),
        "relation_ids": torch.tensor(relations, dtype=torch.long, device=device),
        "answer_targets": torch.tensor(answers, dtype=torch.long, device=device),
        "source_targets": torch.tensor(sources, dtype=torch.float32, device=device),
        "heatmap_targets": torch.tensor(np.stack(masks), dtype=torch.float32, device=device),
        "full_masks": full_masks,
        "entries": entries,
        "manifest": manifest,
    }
    for key in ["r0", "d0", "r_thumb", "d_thumb"]:
        batch[key].requires_grad_(False)
    return batch


class PCRAUv0(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        hidden = int(config["model"]["hidden_dim"])
        vocab = int(config["language"]["vocab_size"])
        relation_count = len(config["language"]["relations"])
        self.token_embedding = nn.Embedding(vocab, hidden, padding_idx=0)
        self.relation_embedding = nn.Embedding(relation_count, hidden)
        self.rgb_projection = nn.Linear(1152, hidden, bias=False)
        self.depth_projection = nn.Linear(1152, hidden, bias=False)
        self.thumbnail_rgb = nn.Linear(1152, hidden, bias=False)
        self.thumbnail_depth = nn.Linear(1152, hidden, bias=False)
        self.gate = nn.Linear(hidden * 3, hidden)
        self.film = nn.Linear(hidden, hidden * 2)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.target_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        global_dim = hidden * 4
        self.answer_head = nn.Sequential(nn.Linear(global_dim, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, 4))
        self.source_head = nn.Sequential(nn.Linear(global_dim, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, 5))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["tokens"].ne(0).unsqueeze(-1)
        token_values = self.token_embedding(batch["tokens"])
        token_query = (token_values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        query = token_query + self.relation_embedding(batch["relation_ids"])
        rgb = self.rgb_projection(batch["r0"])
        depth = self.depth_projection(batch["d0"])
        q_grid = query[:, None, None, :].expand_as(rgb)
        gate = torch.sigmoid(self.gate(torch.cat([rgb, depth, q_grid], dim=-1)))
        fused = self.fusion_norm(rgb + gate * depth)
        gamma, beta = self.film(query).chunk(2, dim=-1)
        fused = fused * (1.0 + 0.1 * torch.tanh(gamma[:, None, None, :])) + beta[:, None, None, :]
        heatmap_logits = self.target_head(fused).squeeze(-1)
        thumbnail = self.thumbnail_rgb(batch["r_thumb"]) + self.thumbnail_depth(batch["d_thumb"])
        global_feature = torch.cat([fused.mean((1, 2)), fused.amax((1, 2)), query, thumbnail], dim=-1)
        return {
            "heatmap_logits": heatmap_logits,
            "answerability_logits": self.answer_head(global_feature),
            "source_logits": self.source_head(global_feature),
            "gate_mean": gate.mean(),
        }


def class_weights(batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    answer = batch["answer_targets"]
    counts = torch.bincount(answer, minlength=4).float()
    answer_weights = counts.reciprocal()
    answer_weights = answer_weights / answer_weights.mean()
    positives = batch["source_targets"].sum(dim=0)
    negatives = batch["source_targets"].shape[0] - positives
    source_weights = (negatives / positives.clamp_min(1)).clamp(1, 14)
    return answer_weights, source_weights


def compute_loss(model_output: Mapping[str, torch.Tensor], batch: Mapping[str, Any],
                 answer_weights: torch.Tensor, source_weights: torch.Tensor) -> dict[str, torch.Tensor]:
    found = batch["answer_targets"].eq(0)
    logits = model_output["heatmap_logits"][found]
    targets = batch["heatmap_targets"][found]
    positive_fraction = targets.mean(dim=(1, 2)).clamp_min(1e-6)
    positive_weight = ((1 - positive_fraction) / positive_fraction).clamp(1, 20)
    bce = -(positive_weight[:, None, None] * targets * F.logsigmoid(logits)
            + (1 - targets) * F.logsigmoid(-logits)).mean(dim=(1, 2)).mean()
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2))
    dice = 1 - ((2 * intersection + 1) / (probabilities.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + 1)).mean()
    heatmap = bce + dice
    answer = F.cross_entropy(model_output["answerability_logits"], batch["answer_targets"], weight=answer_weights)
    source = F.binary_cross_entropy_with_logits(model_output["source_logits"], batch["source_targets"],
                                                pos_weight=source_weights)
    total = heatmap + 0.5 * answer + 0.5 * source
    return {"total": total, "heatmap": heatmap, "heatmap_bce": bce, "heatmap_dice": dice,
            "answerability": answer, "source": source}


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


@torch.no_grad()
def evaluate(model: nn.Module, batch: Mapping[str, Any], answer_weights: torch.Tensor,
             source_weights: torch.Tensor) -> dict[str, Any]:
    model.eval()
    output = model(batch)
    losses = compute_loss(output, batch, answer_weights, source_weights)
    answer_prediction = output["answerability_logits"].argmax(dim=1)
    answer_correct = int(answer_prediction.eq(batch["answer_targets"]).sum().item())
    source_prediction = torch.sigmoid(output["source_logits"]).ge(0.5)
    source_truth = batch["source_targets"].bool()
    true_positive = int((source_prediction & source_truth).sum().item())
    false_positive = int((source_prediction & ~source_truth).sum().item())
    false_negative = int((~source_prediction & source_truth).sum().item())
    denominator = 2 * true_positive + false_positive + false_negative
    source_f1 = (2 * true_positive / denominator) if denominator else 1.0
    probabilities = torch.sigmoid(output["heatmap_logits"])
    found_indices = torch.nonzero(batch["answer_targets"].eq(0), as_tuple=False).flatten().tolist()
    point_correct = 0
    masses = []
    map_points = []
    for index in range(len(batch["entries"])):
        flat = int(probabilities[index].argmax().item())
        row, column = divmod(flat, 32)
        x = min(639, int((column + 0.5) / 32 * 640))
        y = min(479, int((row + 0.5) / 24 * 480))
        correct = bool(batch["full_masks"][index][y, x])
        if index in found_indices:
            point_correct += int(correct)
            target = batch["heatmap_targets"][index]
            masses.append(float((probabilities[index] * target).sum().item()
                                / probabilities[index].sum().clamp_min(1e-12).item()))
        map_points.append({"sample_id": batch["entries"][index]["sample_id"], "x": x, "y": y,
                           "x_norm": (x + 0.5) / 640, "y_norm": (y + 0.5) / 480,
                           "point_in_target": correct})
    result = {
        "loss": {key: float(value.item()) for key, value in losses.items()},
        "answerability_correct": answer_correct,
        "answerability_total": len(batch["entries"]),
        "answerability_accuracy": answer_correct / len(batch["entries"]),
        "source_micro_f1": source_f1,
        "source_tp": true_positive,
        "source_fp": false_positive,
        "source_fn": false_negative,
        "found_point_in_target_correct": point_correct,
        "found_total": len(found_indices),
        "found_mean_mass_in_target": statistics.fmean(masses),
        "map_points": map_points,
        "answerability_predictions": answer_prediction.cpu().tolist(),
        "source_predictions": source_prediction.int().cpu().tolist(),
        "gate_mean": float(output["gate_mean"].item()),
    }
    model.train()
    return result


def model_tensor_hash(model: nn.Module) -> str:
    with tempfile.TemporaryDirectory(prefix="pcra-model-hash-") as temp:
        path = Path(temp) / "model.safetensors"
        state = {key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}
        save_safetensors(state, str(path))
        return sha256_file(path)


def one_batch() -> dict[str, Any]:
    spec_lock = verify_hash_lock(SPEC_LOCK_PATH, "LOCKED_BEFORE_ONE_BATCH_BACKWARD_AND_OPTIMIZER_STEP")
    config = load_json(CONFIG_PATH)
    manifest = load_json(MANIFEST_PATH)
    before = verify_protected(manifest)
    checkpoint_test = subprocess.run(
        [sys.executable, str(CHECKPOINT_TEST)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False,
    )
    seed_runtime(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SmokeError("CUDA is required for the locked smoke")
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    batch = load_smoke_batch(device)
    load_seconds = time.perf_counter() - load_start
    model = PCRAUv0(config).to(device)
    answer_weights, source_weights = class_weights(batch)
    output = model(batch)
    losses = compute_loss(output, batch, answer_weights, source_weights)
    losses["total"].backward()
    gradients = {
        "target_head": grad_norm(model.target_head),
        "answer_head": grad_norm(model.answer_head),
        "source_head": grad_norm(model.source_head),
        "relation_embedding": grad_norm(model.relation_embedding),
        "fusion_gate": grad_norm(model.gate),
    }
    finite_losses = all(torch.isfinite(value).item() for value in losses.values())
    finite_outputs = all(torch.isfinite(value).all().item() for key, value in output.items() if key.endswith("logits"))
    minimum = config["gates"]["min_active_head_gradient_norm"]
    active_gradients = all(math.isfinite(value) and value > minimum for value in gradients.values())
    input_gradients_absent = all(batch[key].grad is None for key in ["r0", "d0", "r_thumb", "d_thumb"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    after = verify_protected(manifest)
    gates = {
        "spec_lock_verified": True,
        "checkpoint_manager_tests_pass": checkpoint_test.returncode == 0,
        "batch_shape_r0": list(batch["r0"].shape) == [15, 24, 32, 1152],
        "batch_shape_d0": list(batch["d0"].shape) == [15, 24, 32, 1152],
        "heatmap_shape": list(output["heatmap_logits"].shape) == [15, 24, 32],
        "answerability_shape": list(output["answerability_logits"].shape) == [15, 4],
        "source_shape": list(output["source_logits"].shape) == [15, 5],
        "finite_losses": bool(finite_losses),
        "finite_logits": bool(finite_outputs),
        "active_gradients_nonzero": bool(active_gradients),
        "feature_inputs_have_no_gradient": input_gradients_absent,
        "all_parameters_sidecar_trainable": parameter_count == trainable_count,
        "protected_inputs_unchanged": before == after,
        "no_optimizer_step": True,
        "train_only_manifest": all(item["split"] == "train" for item in manifest["entries"]),
    }
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "decision": "LOCK_EXECUTION_AND_RUN_OVERFIT" if all(gates.values()) else "FIX_LOADER_MODEL_LOSS_FIRST",
        "gates": gates,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "runtime": {"feature_load_seconds": load_seconds,
                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated())},
        "losses": {key: float(value.item()) for key, value in losses.items()},
        "gradient_norms": gradients,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "answer_class_weights": answer_weights.detach().cpu().tolist(),
        "source_positive_weights": source_weights.detach().cpu().tolist(),
        "checkpoint_test_output": checkpoint_test.stdout,
        "protected_before": before,
        "protected_after": after,
        "spec_lock_sha256": sha256_file(SPEC_LOCK_PATH),
        "training_code_sha256": sha256_file(Path(__file__)),
        "training_performed": False,
        "optimizer_step_performed": False,
        "dataset_scaled": False,
        "splits_read": ["train"],
        "oracle_graph_read_by_model": False,
    }
    write_json(ONE_BATCH_JSON, report)
    lines = ["# P-CRA-U one-batch forward/backward gate", "", f"> **{report['decision']}**", "",
             "Không có optimizer step; đây chỉ là preflight gradient.", "", "## Gates", "",
             "| Gate | Result |", "|---|---:|"]
    lines += [f"| `{key}` | {'PASS' if value else 'FAIL'} |" for key, value in gates.items()]
    lines += ["", "## Loss", "", *[f"- `{key}`: {value:.8f}" for key, value in report["losses"].items()],
              "", "## Gradient norms", "", *[f"- `{key}`: {value:.8e}" for key, value in gradients.items()]]
    ONE_BATCH_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not all(gates.values()):
        raise SmokeError(f"One-batch gate failed: {gates}")
    return report


def lock_execution() -> dict[str, Any]:
    verify_hash_lock(SPEC_LOCK_PATH, "LOCKED_BEFORE_ONE_BATCH_BACKWARD_AND_OPTIMIZER_STEP")
    report = load_json(ONE_BATCH_JSON)
    if report.get("decision") != "LOCK_EXECUTION_AND_RUN_OVERFIT" or not all(report.get("gates", {}).values()):
        raise SmokeError("One-batch report has not passed")
    files = [CONTRACT, CONFIG_PATH, MANIFEST_PATH, SPEC_LOCK_PATH, ONE_BATCH_JSON, ONE_BATCH_MD,
             Path(__file__).resolve(), CHECKPOINT_MANAGER, CHECKPOINT_TEST]
    lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "locked_at_utc": utc_now(),
        "status": "LOCKED_AFTER_ONE_BATCH_BEFORE_FIRST_OPTIMIZER_STEP",
        "files": {str(path.relative_to(WORKSPACE)): sha256_file(path) for path in files},
        "allowed_next_action": "TWO_RUN_OVERFIT_SMOKE_WITH_CHECKPOINT_RESUME",
        "training_performed": False,
        "dataset_scaled": False,
    }
    write_json(EXECUTION_LOCK_PATH, lock)
    return lock


def checkpoint_identity() -> dict[str, str]:
    return {
        "config_sha256": sha256_file(CONFIG_PATH),
        "dataset_sha256": sha256_file(WORKSPACE / "datasets/roborefer_dataset_v1_prototype_20260818_155606/artifact_manifest.json"),
        "split_manifest_sha256": sha256_file(MANIFEST_PATH),
        "code_sha256": sha256_file(Path(__file__)),
        "feature_contract_sha256": sha256_file(CONTRACT),
        "execution_lock_sha256": sha256_file(EXECUTION_LOCK_PATH),
    }


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def train_once(run_root: Path, run_name: str, batch: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    seed_runtime(config["seed"])
    device = batch["r0"].device
    model = PCRAUv0(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimization"]["learning_rate"],
                                  weight_decay=config["optimization"]["weight_decay"])
    answer_weights, source_weights = class_weights(batch)
    log_path = run_root / "logs" / f"{run_name}_metrics.jsonl"
    if log_path.exists():
        raise SmokeError(f"Refusing to overwrite log: {log_path}")
    checkpoint_root = run_root / "checkpoints" / run_name
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    records = []
    checkpoint_steps = set(config["optimization"]["checkpoint_steps"])
    for step in range(1, config["optimization"]["steps"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        losses = compute_loss(output, batch, answer_weights, source_weights)
        if not all(torch.isfinite(value).item() for value in losses.values()):
            raise SmokeError(f"Non-finite loss at {run_name} step {step}")
        losses["total"].backward()
        preclip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                            config["optimization"]["gradient_clip_norm"]).item())
        if not math.isfinite(preclip_norm) or preclip_norm <= 0:
            raise SmokeError(f"Invalid gradient norm at {run_name} step {step}: {preclip_norm}")
        optimizer.step()
        record = {
            "run": run_name,
            "step": step,
            "total_loss": float(losses["total"].item()),
            "heatmap_loss": float(losses["heatmap"].item()),
            "answerability_loss": float(losses["answerability"].item()),
            "source_loss": float(losses["source"].item()),
            "gradient_norm_preclip": preclip_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "finite": True,
        }
        records.append(record)
        append_jsonl(log_path, record)
        if step in checkpoint_steps:
            measured = evaluate(model, batch, answer_weights, source_weights)
            save_checkpoint(
                checkpoint_root,
                run_id=f"{run_root.name}_{run_name}",
                model=model,
                optimizer=optimizer,
                epoch=step,
                global_step=step,
                selection_split="train",
                selection_metric="train_total_loss",
                selection_mode="min",
                selection_value=measured["loss"]["total"],
                metrics={
                    "train_total_loss": measured["loss"]["total"],
                    "train_heatmap_loss": measured["loss"]["heatmap"],
                    "train_answerability_accuracy": measured["answerability_accuracy"],
                    "train_source_micro_f1": measured["source_micro_f1"],
                },
                identity=checkpoint_identity(),
                sampler_state={"kind": "full_batch", "next_step": step + 1},
                extra_state={"records_written": step},
                runtime={"device": str(device), "gpu": torch.cuda.get_device_name(0),
                         "torch": torch.__version__, "dtype": "torch.float32"},
            )
    elapsed = time.perf_counter() - started
    final = evaluate(model, batch, answer_weights, source_weights)
    final_hash = model_tensor_hash(model)
    checkpoint_hash = sha256_file(checkpoint_root / "step_000000600/model.safetensors")
    if final_hash != checkpoint_hash:
        raise SmokeError(f"In-memory/final checkpoint hash mismatch: {run_name}")
    return {
        "run_name": run_name,
        "records": records,
        "final_metrics": final,
        "final_model_sha256": final_hash,
        "checkpoint_root": str(checkpoint_root.relative_to(WORKSPACE)),
        "checkpoint_audit": audit_checkpoint_root(checkpoint_root),
        "elapsed_seconds": elapsed,
        "steps_per_second": config["optimization"]["steps"] / elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }


def resume_replay(run_root: Path, batch: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    seed_runtime(config["seed"])
    model = PCRAUv0(config).to(batch["r0"].device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimization"]["learning_rate"],
                                  weight_decay=config["optimization"]["weight_decay"])
    checkpoint_root = run_root / "checkpoints/run_a"
    resumed = resume_training(
        checkpoint_root,
        pointer=checkpoint_root / "step_000000250",
        model=model,
        optimizer=optimizer,
        expected_identity=checkpoint_identity(),
    )
    answer_weights, source_weights = class_weights(batch)
    losses = []
    for step in range(251, 256):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        values = compute_loss(output, batch, answer_weights, source_weights)
        values["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimization"]["gradient_clip_norm"])
        optimizer.step()
        losses.append(float(values["total"].item()))
    replay_hash = model_tensor_hash(model)
    expected_hash = sha256_file(checkpoint_root / "step_000000255/model.safetensors")
    return {
        "from_step": 250,
        "to_step": 255,
        "losses": losses,
        "replay_model_sha256": replay_hash,
        "expected_step255_model_sha256": expected_hash,
        "exact_model_hash_equal": replay_hash == expected_hash,
        "resume_verification_passed": resumed["verification"]["passed"],
        "identity_verified": resumed["metadata"]["identity"] == checkpoint_identity(),
    }


def svg_polyline(values: list[float], x: float, y: float, width: float, height: float,
                 color: str, minimum: float | None = None, maximum: float | None = None) -> str:
    low = min(values) if minimum is None else minimum
    high = max(values) if maximum is None else maximum
    if high <= low:
        high = low + 1
    points = []
    for index, value in enumerate(values):
        px = x + index / max(1, len(values) - 1) * width
        py = y + height - (value - low) / (high - low) * height
        points.append(f"{px:.2f},{py:.2f}")
    return f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'


def make_training_figure(path: Path, run_a: list[dict[str, Any]], run_b: list[dict[str, Any]]) -> None:
    panels = [
        ("Total loss", "total_loss", "#2563eb"),
        ("Heatmap loss", "heatmap_loss", "#7c3aed"),
        ("Answerability loss", "answerability_loss", "#d97706"),
        ("Source loss", "source_loss", "#16a34a"),
        ("Gradient norm (pre-clip)", "gradient_norm_preclip", "#dc2626"),
        ("Learning rate", "learning_rate", "#0891b2"),
    ]
    width, panel_h = 1100, 190
    height = 70 + len(panels) * panel_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<text x="30" y="38" font-family="sans-serif" font-size="24" font-weight="700">P-CRA-U overfit smoke — run A/B training curves</text>']
    for panel_index, (title, key, color) in enumerate(panels):
        top = 60 + panel_index * panel_h
        x, y, chart_w, chart_h = 85, top + 38, 960, 115
        a = [float(row[key]) for row in run_a]
        b = [float(row[key]) for row in run_b]
        low, high = min(a + b), max(a + b)
        parts += [f'<text x="30" y="{top + 22}" font-family="sans-serif" font-size="17" font-weight="600">{html.escape(title)}</text>',
                  f'<rect x="{x}" y="{y}" width="{chart_w}" height="{chart_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
                  svg_polyline(a, x, y, chart_w, chart_h, color, low, high),
                  svg_polyline(b, x, y, chart_w, chart_h, "#111827", low, high),
                  f'<text x="{x}" y="{y + chart_h + 22}" font-family="sans-serif" font-size="12">step 1</text>',
                  f'<text x="{x + chart_w}" y="{y + chart_h + 22}" text-anchor="end" font-family="sans-serif" font-size="12">step 600</text>',
                  f'<text x="{x + 8}" y="{y + 16}" font-family="sans-serif" font-size="11" fill="{color}">run A</text>',
                  f'<text x="{x + 70}" y="{y + 16}" font-family="sans-serif" font-size="11" fill="#111827">run B</text>',
                  f'<text x="{x + chart_w - 4}" y="{y + 16}" text-anchor="end" font-family="sans-serif" font-size="11">range {low:.4g}–{high:.4g}</text>']
    parts.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


@torch.no_grad()
def make_heatmap_images(run_root: Path, model: nn.Module, batch: Mapping[str, Any]) -> int:
    model.eval()
    output = model(batch)
    probabilities = torch.sigmoid(output["heatmap_logits"]).cpu().numpy()
    answer_predictions = output["answerability_logits"].argmax(dim=1).cpu().tolist()
    source_probabilities = torch.sigmoid(output["source_logits"]).cpu().numpy()
    answer_names = load_json(CONFIG_PATH)["model"]["answerability_classes"]
    image_dir = run_root / "images/overfit_heatmaps"
    image_dir.mkdir(parents=True, exist_ok=True)
    thumbnails = []
    for index, item in enumerate(batch["entries"]):
        rgb = cv2.imread(str(WORKSPACE / item["rgb_path"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise SmokeError(f"Cannot read RGB: {item['rgb_path']}")
        heat = cv2.resize(probabilities[index], (640, 480), interpolation=cv2.INTER_CUBIC)
        heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(heat_norm, cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(rgb, 0.58, colored, 0.42, 0)
        mask = batch["full_masks"][index].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        flat = int(probabilities[index].argmax())
        row, column = divmod(flat, 32)
        point = (min(639, int((column + 0.5) / 32 * 640)), min(479, int((row + 0.5) / 24 * 480)))
        cv2.drawMarker(overlay, point, (0, 0, 255), cv2.MARKER_CROSS, 26, 3)
        canvas = cv2.copyMakeBorder(overlay, 62, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 24, 36))
        truth = item["answerability_state"]
        predicted = answer_names[answer_predictions[index]]
        source_text = ",".join(f"{v:.2f}" for v in source_probabilities[index])
        cv2.putText(canvas, f"{item['sample_id']} | truth={truth} pred={predicted}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"raw source scores [sem,rel,spa,dep,occ]={source_text}", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        destination = image_dir / f"{item['sample_id']}.jpg"
        cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        thumbnails.append(cv2.resize(canvas, (320, 271), interpolation=cv2.INTER_AREA))
    rows = [cv2.hconcat(thumbnails[index:index + 5]) for index in range(0, 15, 5)]
    contact = cv2.vconcat(rows)
    cv2.imwrite(str(run_root / "images/overfit_heatmap_contact_sheet.jpg"), contact,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    return len(thumbnails)


def update_run_artifacts(run_root: Path, run_a: dict[str, Any], run_b: dict[str, Any],
                         resume: dict[str, Any], comparison: dict[str, Any], batch: Mapping[str, Any],
                         final_model: nn.Module) -> None:
    make_training_figure(run_root / "figures/training_curves.svg", run_a["records"], run_b["records"])
    image_count = make_heatmap_images(run_root, final_model, batch)
    table_rows = []
    for run in [run_a, run_b]:
        table_rows.append({
            "run_id": run["run_name"], "seed": load_json(CONFIG_PATH)["seed"], "status": "COMPLETE",
            "best_dev_step": "NOT_APPLICABLE_OVERFIT_TRAIN_ONLY", "selection_metric": "train_total_loss",
            "selection_value": run["final_metrics"]["loss"]["total"],
            "final_train_loss": run["final_metrics"]["loss"]["total"], "gradient_finite": True,
            "resume_exact": resume["exact_model_hash_equal"] if run["run_name"] == "run_a" else "NOT_RUN",
            "checkpoint_verified": run["checkpoint_audit"]["passed"],
            "notes": "Overfit diagnostic only; not a dev/generalization result.",
        })
    table_path = run_root / "tables/table_07_training_checkpoint.csv"
    fields = ["run_id", "seed", "status", "best_dev_step", "selection_metric", "selection_value",
              "final_train_loss", "gradient_finite", "resume_exact", "checkpoint_verified", "notes"]
    write_csv(table_path, fields, table_rows)
    metrics_index = load_json(run_root / "metrics/scientific_results_index.json")
    metrics_index["status"] = "COMPLETE" if comparison["decision"].startswith("GO_") else "FAILED"
    for item in metrics_index["tables"]:
        if item["id"] == "T07":
            item.update({"status": "COMPLETE", "reason": "Overfit diagnostic A/B + resume exact audit."})
    for item in metrics_index["figures"]:
        if item["id"] == "F02":
            item.update({"status": "COMPLETE", "path": "figures/training_curves.svg",
                         "reason": "Run A/B loss, gradient and LR curves."})
    write_json(run_root / "metrics/scientific_results_index.json", metrics_index)
    write_json(run_root / "metrics/overfit_comparison.json", comparison)
    write_json(run_root / "checks/checkpoint_audit.json", {
        "run_a": run_a["checkpoint_audit"], "run_b": run_b["checkpoint_audit"], "resume": resume,
        "passed": run_a["checkpoint_audit"]["passed"] and run_b["checkpoint_audit"]["passed"]
                  and resume["exact_model_hash_equal"],
    })
    config = load_json(run_root / "config/run_config.json")
    config.update({"status": "COMPLETE" if comparison["decision"].startswith("GO_") else "FAILED",
                   "model_config_status": "LOCKED", "training_performed": True,
                   "calibration_fit_performed": False, "test_predictions_generated": False,
                   "completed_at_utc": utc_now(), "overfit_decision": comparison["decision"],
                   "execution_lock_sha256": sha256_file(EXECUTION_LOCK_PATH)})
    write_json(run_root / "config/run_config.json", config)
    report = ["# P-CRA-U v0 — Overfit smoke report", "", f"> **{comparison['decision']}**", "",
              "Train-only diagnostic trên 15 sample/15 family; không có dev/calibration/test read.", "",
              "## Gate", "", "| Gate | Result |", "|---|---:|"]
    report += [f"| `{key}` | {'PASS' if value else 'FAIL'} |" for key, value in comparison["gates"].items()]
    report += ["", "## Final metrics", "",
               f"- loss ratio: {comparison['observed']['final_initial_loss_ratio']:.6f};",
               f"- answerability: {comparison['observed']['answerability_correct']}/15;",
               f"- source micro-F1: {comparison['observed']['source_micro_f1']:.6f};",
               f"- FOUND point-in-target: {comparison['observed']['found_point_in_target_correct']}/8;",
               f"- mean mass-in-target: {comparison['observed']['found_mean_mass_in_target']:.6f};",
               f"- final A/B model hash equal: {comparison['gates']['run_a_b_final_model_hash_equal']};",
               f"- resume 250→255 exact: {resume['exact_model_hash_equal']};",
               f"- heatmap overlays: {image_count}/15.", "",
               "Bảng 2–6 vẫn `NOT_RUN`; kết quả này không phải generalization/calibration evidence."]
    (run_root / "PCRA_U_OVERFIT_SMOKE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (run_root / "RUN_CARD.md").write_text("\n".join(report[:4] + ["", "Xem `PCRA_U_OVERFIT_SMOKE_REPORT.md` và `checks/overfit_gate.json`."]) + "\n", encoding="utf-8")


def build_final_manifest(run_root: Path) -> None:
    artifacts = []
    for path in sorted(p for p in run_root.rglob("*") if p.is_file()
                       and p.name not in {"final_manifest.json", "final_manifest_check.json"}):
        artifacts.append({"path": str(path.relative_to(run_root)), "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    write_json(run_root / "manifests/final_manifest.json", {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "generated_at_utc": utc_now(),
        "artifact_count": len(artifacts), "artifacts": artifacts,
    })
    mismatches = []
    for item in artifacts:
        path = run_root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            mismatches.append(item["path"])
    write_json(run_root / "checks/final_manifest_check.json", {
        "artifact_count": len(artifacts), "mismatches": mismatches, "passed": not mismatches,
    })


def run_overfit(run_root: Path) -> dict[str, Any]:
    verify_hash_lock(EXECUTION_LOCK_PATH, "LOCKED_AFTER_ONE_BATCH_BEFORE_FIRST_OPTIMIZER_STEP")
    if not run_root.is_dir():
        raise SmokeError(f"Scaffold run does not exist: {run_root}")
    run_config = load_json(run_root / "config/run_config.json")
    if run_config.get("phase") != "OVERFIT_SMOKE" or run_config["split_access"] != {
        "read_splits": ["train"], "selection_split": "train", "test_opened": False
    }:
        raise SmokeError("Run scaffold split policy mismatch")
    manifest = load_json(MANIFEST_PATH)
    protected_before = verify_protected(manifest)
    config = load_json(CONFIG_PATH)
    seed_runtime(config["seed"])
    device = torch.device("cuda")
    batch = load_smoke_batch(device)
    run_a = train_once(run_root, "run_a", batch, config)
    run_b = train_once(run_root, "run_b", batch, config)
    resume = resume_replay(run_root, batch, config)
    a_records, b_records = run_a["records"], run_b["records"]
    initial = statistics.median(row["total_loss"] for row in a_records[:20])
    final_window = statistics.median(row["total_loss"] for row in a_records[-20:])
    ratio = final_window / initial
    final_a = run_a["final_metrics"]
    deterministic_metrics_a = {key: value for key, value in final_a.items() if key != "map_points"}
    deterministic_metrics_b = {key: value for key, value in run_b["final_metrics"].items() if key != "map_points"}
    protected_after = verify_protected(manifest)
    feature_hashes_unchanged = all(
        sha256_file(WORKSPACE / item["feature_tensor_path"]) == item["feature_tensor_sha256"]
        and sha256_file(WORKSPACE / item["cache_metadata_path"]) == item["cache_metadata_sha256"]
        for item in manifest["entries"]
    )
    gate_config = config["gates"]
    gates = {
        "loss_ratio_pass": ratio <= gate_config["max_final_initial_loss_ratio"],
        "answerability_memorized": final_a["answerability_correct"] >= gate_config["min_answerability_correct"],
        "source_micro_f1_pass": final_a["source_micro_f1"] >= gate_config["min_source_micro_f1"],
        "found_point_in_target_pass": final_a["found_point_in_target_correct"] >= gate_config["min_found_point_in_target_correct"],
        "found_mass_in_target_pass": final_a["found_mean_mass_in_target"] >= gate_config["min_found_mean_mass_in_target"],
        "all_logged_values_finite": all(row["finite"] and all(math.isfinite(float(row[key])) for key in
            ["total_loss", "heatmap_loss", "answerability_loss", "source_loss", "gradient_norm_preclip"])
            for row in a_records + b_records),
        "run_a_b_final_model_hash_equal": run_a["final_model_sha256"] == run_b["final_model_sha256"],
        "run_a_b_final_metrics_exact": canonical_sha(deterministic_metrics_a) == canonical_sha(deterministic_metrics_b),
        "resume_250_255_exact": resume["exact_model_hash_equal"],
        "checkpoint_audits_pass": run_a["checkpoint_audit"]["passed"] and run_b["checkpoint_audit"]["passed"],
        "protected_inputs_unchanged": protected_before == protected_after,
        "feature_source_hashes_unchanged": feature_hashes_unchanged,
        "train_only_no_test": run_config["split_access"]["read_splits"] == ["train"]
                               and not run_config["split_access"]["test_opened"],
        "no_dataset_scaling": True,
        "no_robot_publish": True,
    }
    decision = "GO_DATA_EXPANSION_DESIGN" if all(gates.values()) else "FIX_PCRA_U_SMOKE_FIRST"
    comparison = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "generated_at_utc": utc_now(),
        "decision": decision, "gates": gates,
        "observed": {
            "initial_loss_median_steps_1_20": initial,
            "final_loss_median_steps_581_600": final_window,
            "final_initial_loss_ratio": ratio,
            "answerability_correct": final_a["answerability_correct"],
            "source_micro_f1": final_a["source_micro_f1"],
            "found_point_in_target_correct": final_a["found_point_in_target_correct"],
            "found_mean_mass_in_target": final_a["found_mean_mass_in_target"],
            "run_a_elapsed_seconds": run_a["elapsed_seconds"],
            "run_b_elapsed_seconds": run_b["elapsed_seconds"],
            "max_peak_vram_bytes": max(run_a["peak_vram_bytes"], run_b["peak_vram_bytes"]),
            "final_model_sha256": run_a["final_model_sha256"],
        },
        "run_a_final": final_a,
        "run_b_final": run_b["final_metrics"],
        "resume_replay": resume,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "splits_read": ["train"], "training_performed": True, "dataset_scaled": False,
        "calibration_fit": False, "test_opened": False, "robot_published": False,
    }
    write_json(run_root / "checks/overfit_gate.json", comparison)
    # Recreate final run-A model for deterministic visualization from checkpoint without unpickling training state.
    final_model = PCRAUv0(config).to(device)
    final_weights = load_safetensors(str(run_root / "checkpoints/run_a/step_000000600/model.safetensors"), device="cpu")
    final_model.load_state_dict(final_weights)
    update_run_artifacts(run_root, run_a, run_b, resume, comparison, batch, final_model)
    build_final_manifest(run_root)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("one-batch")
    sub.add_parser("lock-execution")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    if args.command == "one-batch":
        result = one_batch()
    elif args.command == "lock-execution":
        result = lock_execution()
    else:
        result = run_overfit((WORKSPACE / args.run_root).resolve())
    summary = {key: result[key] for key in ["decision", "gates"] if key in result}
    if not summary:
        summary = {"status": result.get("status"), "allowed_next_action": result.get("allowed_next_action")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
