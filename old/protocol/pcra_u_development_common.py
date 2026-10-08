#!/usr/bin/env python3
"""Shared, sidecar-only components for locked P-CRA-U development training.

This module never imports or mutates RoboRefer.  Feature extraction is isolated
in ``pcra_u_development_feature_cache.py``; the train/eval batch passed to this
module contains only frozen pooled features, prompt-derived language tensors and
supervision tensors.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors


ANSWERABILITY_CLASSES = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
SOURCE_CLASSES = ["semantic", "relation", "depth", "occlusion"]
RELATION_CLASSES = [
    "direct",
    "left_of",
    "right_of",
    "front_of",
    "behind",
    "nearer_than",
    "farther_than",
    "between_in_depth",
    "nearer_than_both",
]
VARIANT_ORDER = [
    "clean",
    "semantic_counterfactual",
    "relation_counterfactual",
    "depth_corruption",
    "occlusion_view_counterfactual",
]


class DevelopmentTrainingError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DevelopmentTrainingError(f"Cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise DevelopmentTrainingError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def safe_resolve(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise DevelopmentTrainingError(f"Path must be non-empty and relative: {relative!r}")
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise DevelopmentTrainingError(f"Path escapes root {root}: {relative}")
    return path


def seed_runtime(seed: int, *, strict: bool = True) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise DevelopmentTrainingError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before importing torch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def tokenize(prompt: str, max_tokens: int, vocab_size: int) -> list[int]:
    values: list[int] = []
    for token in re.findall(r"[a-z0-9]+", prompt.lower())[:max_tokens]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "big") % (vocab_size - 1) + 1)
    values = values or [1]
    return values + [0] * (max_tokens - len(values))


def parse_relation(prompt: str) -> str:
    """Parse only prompt-visible language; evaluator relation labels are never inputs."""
    text = " ".join(re.findall(r"[a-z0-9]+", prompt.lower()))
    if "between" in text and "depth" in text:
        return "between_in_depth"
    if ("nearer" in text or "closer" in text) and "both" in text:
        return "nearer_than_both"
    if "right of" in text:
        return "right_of"
    if "left of" in text:
        return "left_of"
    if "front of" in text:
        return "front_of"
    if "behind" in text:
        return "behind"
    if "farther" in text or "further" in text:
        return "farther_than"
    if "closer to" in text or "nearer" in text:
        return "nearer_than"
    return "direct"


def pool_raw_feature(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Locked WP3/overfit transform from tower output to a 24x32 grid."""
    if list(tensor.shape) != [13, 1024, 1152]:
        raise DevelopmentTrainingError(f"Unexpected pre-projector feature shape: {list(tensor.shape)}")
    local = tensor[:12].float().reshape(12, 32, 32, 1152).permute(0, 3, 1, 2)
    local = F.avg_pool2d(local, kernel_size=4, stride=4)
    grid = local.reshape(3, 4, 1152, 8, 8).permute(2, 0, 3, 1, 4).reshape(1152, 24, 32)
    grid = grid.permute(1, 2, 0).contiguous()
    grid = F.layer_norm(grid, (1152,))
    thumbnail = F.layer_norm(tensor[12].float().mean(dim=0), (1152,))
    return grid, thumbnail


class PCRAUDevelopmentV1(nn.Module):
    """Minimal relation-conditioned sidecar; RoboRefer remains fully frozen."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        hidden = int(config["model"]["hidden_dim"])
        vocab = int(config["language"]["vocab_size"])
        source_count = len(config["model"]["source_classes"])
        self.token_embedding = nn.Embedding(vocab, hidden, padding_idx=0)
        self.relation_embedding = nn.Embedding(len(config["language"]["relations"]), hidden)
        self.rgb_projection = nn.Linear(1152, hidden, bias=False)
        self.depth_projection = nn.Linear(1152, hidden, bias=False)
        self.thumbnail_rgb = nn.Linear(1152, hidden, bias=False)
        self.thumbnail_depth = nn.Linear(1152, hidden, bias=False)
        self.gate = nn.Linear(hidden * 3, hidden)
        self.film = nn.Linear(hidden, hidden * 2)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.target_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        global_dim = hidden * 4
        self.answer_head = nn.Sequential(
            nn.Linear(global_dim, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, len(ANSWERABILITY_CLASSES))
        )
        self.source_head = nn.Sequential(
            nn.Linear(global_dim, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, source_count)
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_mask = batch["tokens"].ne(0).unsqueeze(-1)
        token_values = self.token_embedding(batch["tokens"])
        token_query = (token_values * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        query = token_query + self.relation_embedding(batch["relation_ids"])
        rgb = self.rgb_projection(batch["r0"])
        depth = self.depth_projection(batch["d0"])
        query_grid = query[:, None, None, :].expand_as(rgb)
        gate = torch.sigmoid(self.gate(torch.cat([rgb, depth, query_grid], dim=-1)))
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


def global_class_weights(entries: Sequence[Mapping[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    answer_counts = torch.zeros(len(ANSWERABILITY_CLASSES), dtype=torch.float32, device=device)
    source_counts = torch.zeros(len(SOURCE_CLASSES), dtype=torch.float32, device=device)
    for entry in entries:
        answer_counts[int(entry["supervision"]["answerability_index"])] += 1
        source_counts += torch.tensor(entry["supervision"]["source_multihot"], device=device)
    if bool(answer_counts.eq(0).any().item()) or bool(source_counts.eq(0).any().item()):
        raise DevelopmentTrainingError(
            f"Train supervision lacks an active class: answer={answer_counts.tolist()} source={source_counts.tolist()}"
        )
    answer_weights = answer_counts.reciprocal()
    answer_weights /= answer_weights.mean()
    source_weights = ((len(entries) - source_counts) / source_counts).clamp(1.0, 20.0)
    return answer_weights, source_weights


def compute_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    answer_weights: torch.Tensor,
    source_weights: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    found = batch["answer_targets"].eq(0)
    if bool(found.any().item()):
        logits = output["heatmap_logits"][found]
        targets = batch["heatmap_targets"][found]
        positive_fraction = targets.mean(dim=(1, 2)).clamp_min(1e-6)
        positive_weight = ((1 - positive_fraction) / positive_fraction).clamp(1, 20)
        bce = -(
            positive_weight[:, None, None] * targets * F.logsigmoid(logits)
            + (1 - targets) * F.logsigmoid(-logits)
        ).mean(dim=(1, 2)).mean()
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2))
        dice = 1 - (
            (2 * intersection + 1)
            / (probabilities.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + 1)
        ).mean()
        heatmap = bce + dice
    else:
        zero = output["heatmap_logits"].sum() * 0.0
        bce = zero
        dice = zero
        heatmap = zero
    answer = F.cross_entropy(
        output["answerability_logits"], batch["answer_targets"], weight=answer_weights
    )
    source = F.binary_cross_entropy_with_logits(
        output["source_logits"], batch["source_targets"], pos_weight=source_weights
    )
    weights = config["loss"]
    total = (
        float(weights["heatmap_weight"]) * heatmap
        + float(weights["answerability_weight"]) * answer
        + float(weights["source_weight"]) * source
    )
    return {
        "total": total,
        "heatmap": heatmap,
        "heatmap_bce": bce,
        "heatmap_dice": dice,
        "answerability": answer,
        "source": source,
    }


def module_gradient_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def gradient_groups(model: PCRAUDevelopmentV1) -> dict[str, float]:
    return {
        "target_head": module_gradient_norm(model.target_head),
        "answer_head": module_gradient_norm(model.answer_head),
        "source_head": module_gradient_norm(model.source_head),
        "language_relation": math.sqrt(
            module_gradient_norm(model.token_embedding) ** 2
            + module_gradient_norm(model.relation_embedding) ** 2
        ),
        "rgb_depth_fusion": math.sqrt(
            sum(
                module_gradient_norm(module) ** 2
                for module in [
                    model.rgb_projection,
                    model.depth_projection,
                    model.thumbnail_rgb,
                    model.thumbnail_depth,
                    model.gate,
                    model.film,
                    model.fusion_norm,
                ]
            )
        ),
    }


def model_inputs(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return the strict oracle-free tensor boundary accepted by the sidecar."""
    allowed = ["r0", "d0", "r_thumb", "d_thumb", "tokens", "relation_ids"]
    return {name: batch[name] for name in allowed}


def deterministic_entry_order(entries: Sequence[Mapping[str, Any]], seed: int, epoch: int) -> list[int]:
    """Shuffle samples without ever creating a new split."""
    keyed = []
    for index, entry in enumerate(entries):
        key = sha256_text(f"{seed}:{epoch}:{entry['family_id']}:{entry['variant']}:{entry['sample_id']}")
        keyed.append((key, index))
    return [index for _, index in sorted(keyed)]


def chunks(values: Sequence[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


class FrozenFeatureDataset:
    def __init__(
        self,
        workspace: Path,
        manifest_entries: Sequence[Mapping[str, Any]],
        cache_root: Path,
        cache_index: Mapping[str, Any],
        config: Mapping[str, Any],
        memory_cache: bool = True,
    ):
        self.workspace = workspace.resolve()
        self.entries = list(manifest_entries)
        self.cache_root = cache_root.resolve()
        self.cache_index = cache_index
        self.config = config
        self.memory_cache = memory_cache
        self._loaded_features: dict[str, dict[str, torch.Tensor]] = {}
        self._feature_files = cache_index["features"]
        self._sample_map = cache_index["sample_to_feature"]

    def __len__(self) -> int:
        return len(self.entries)

    def make_batch(self, indices: Sequence[int], device: torch.device) -> dict[str, Any]:
        r0s: list[torch.Tensor] = []
        d0s: list[torch.Tensor] = []
        rthumbs: list[torch.Tensor] = []
        dthumbs: list[torch.Tensor] = []
        tokens: list[list[int]] = []
        relations: list[int] = []
        answers: list[int] = []
        sources: list[list[int]] = []
        masks: list[np.ndarray] = []
        interiors: list[np.ndarray] = []
        full_masks: list[np.ndarray] = []
        full_interiors: list[np.ndarray] = []
        selected: list[Mapping[str, Any]] = []
        language = self.config["language"]
        relation_names = language["relations"]
        for index in indices:
            entry = self.entries[index]
            sample_id = entry["sample_id"]
            feature_key = self._sample_map[sample_id]
            feature_row = self._feature_files[feature_key]
            tensor_path = safe_resolve(self.cache_root, feature_row["path"])
            if feature_key in self._loaded_features:
                tensors = self._loaded_features[feature_key]
            else:
                if sha256_file(tensor_path) != feature_row["sha256"]:
                    raise DevelopmentTrainingError(f"Feature cache hash mismatch: {tensor_path}")
                tensors = load_safetensors(str(tensor_path), device="cpu")
                if self.memory_cache:
                    self._loaded_features[feature_key] = tensors
            expected_keys = {"R0_GRID", "D0_GRID", "R0_THUMB", "D0_THUMB"}
            if set(tensors) != expected_keys:
                raise DevelopmentTrainingError(f"Unexpected cache keys for {feature_key}: {sorted(tensors)}")
            r0s.append(tensors["R0_GRID"])
            d0s.append(tensors["D0_GRID"])
            rthumbs.append(tensors["R0_THUMB"])
            dthumbs.append(tensors["D0_THUMB"])
            prompt = entry["feature_input"]["prompt"]
            relation = parse_relation(prompt)
            tokens.append(tokenize(prompt, int(language["max_tokens"]), int(language["vocab_size"])))
            relations.append(relation_names.index(relation))
            supervision = entry["supervision"]
            answers.append(int(supervision["answerability_index"]))
            sources.append(list(supervision["source_multihot"]))
            mask_path = safe_resolve(self.workspace, supervision["target_mask_path"])
            interior_path = safe_resolve(self.workspace, supervision["target_interior_mask_path"])
            target = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            interior = cv2.imread(str(interior_path), cv2.IMREAD_GRAYSCALE)
            if target is None or interior is None or target.shape != (480, 640) or interior.shape != (480, 640):
                raise DevelopmentTrainingError(f"Invalid target/interior mask: {sample_id}")
            full_masks.append(target > 0)
            full_interiors.append(interior > 0)
            masks.append(cv2.resize((target > 0).astype(np.float32), (32, 24), interpolation=cv2.INTER_AREA))
            interiors.append(
                cv2.resize((interior > 0).astype(np.float32), (32, 24), interpolation=cv2.INTER_AREA)
            )
            selected.append(entry)
        batch: dict[str, Any] = {
            "r0": torch.stack(r0s).to(device=device, dtype=torch.float32),
            "d0": torch.stack(d0s).to(device=device, dtype=torch.float32),
            "r_thumb": torch.stack(rthumbs).to(device=device, dtype=torch.float32),
            "d_thumb": torch.stack(dthumbs).to(device=device, dtype=torch.float32),
            "tokens": torch.tensor(tokens, dtype=torch.long, device=device),
            "relation_ids": torch.tensor(relations, dtype=torch.long, device=device),
            "answer_targets": torch.tensor(answers, dtype=torch.long, device=device),
            "source_targets": torch.tensor(sources, dtype=torch.float32, device=device),
            "heatmap_targets": torch.tensor(np.stack(masks), dtype=torch.float32, device=device),
            "interior_targets": torch.tensor(np.stack(interiors), dtype=torch.float32, device=device),
            "full_masks": full_masks,
            "full_interiors": full_interiors,
            "entries": selected,
        }
        for name in ["r0", "d0", "r_thumb", "d_thumb"]:
            batch[name].requires_grad_(False)
        return batch


def confusion_metrics(truth: Sequence[int], prediction: Sequence[int], class_count: int) -> dict[str, Any]:
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for target, predicted in zip(truth, prediction):
        matrix[int(target)][int(predicted)] += 1
    per_class = []
    f1_values = []
    for index in range(class_count):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(class_count) if row != index)
        fn = sum(matrix[index][column] for column in range(class_count) if column != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append({"class_index": index, "precision": precision, "recall": recall, "f1": f1})
    total = len(truth)
    return {
        "confusion_matrix": matrix,
        "accuracy": sum(matrix[i][i] for i in range(class_count)) / total if total else 0.0,
        "macro_f1": sum(f1_values) / class_count,
        "per_class": per_class,
    }
