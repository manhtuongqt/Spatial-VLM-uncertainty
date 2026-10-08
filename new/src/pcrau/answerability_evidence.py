"""Observable evidence residual for experimental answerability refinement."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


EVIDENCE_NAMES = [
    "target_sigmoid_max", "target_sigmoid_mean", "spatial_peak", "spatial_peak_margin",
    "spatial_normalized_entropy", "spatial_top5_mass", "spatial_top10_mass",
    "interior_sigmoid_max", "interior_sigmoid_mean", "target_interior_peak_distance",
    "anchor_sigmoid_max_mean", "anchor_sigmoid_max_min", "edge_probability_mean",
    "edge_probability_min", "fusion_gate_mean", "rgb_depth_thumb_cosine", "rgb_depth_thumb_mae",
    "source_semantic", "source_relation", "source_spatial", "source_depth", "source_occlusion",
    "base_found", "base_ambiguous", "base_absent", "base_insufficient_evidence",
]


def observable_answerability_evidence(
    output: Mapping[str, torch.Tensor], model_batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """No annotation, sample ID, family or variant is accepted or inspected."""
    allowed = {"r0", "d0", "r_thumb", "d_thumb", "token_ids", "token_mask",
               "relation_ids", "relation_mask", "anchor_mask"}
    if set(model_batch) - allowed:
        raise ValueError("Evidence received forbidden annotation/metadata keys")
    target = output["target_logits"].float().flatten(1)
    interior = output["interior_logits"].float().flatten(1)
    probability = target.softmax(-1)
    ordered = probability.topk(min(10, probability.shape[-1]), dim=-1).values
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
    entropy = entropy / probability.new_tensor(probability.shape[-1]).log()
    height, width = output["target_logits"].shape[-2:]
    target_index, interior_index = target.argmax(-1), interior.argmax(-1)
    distance = (((target_index % width - interior_index % width).float() / max(1, width - 1)).square()
                + ((target_index // width - interior_index // width).float() / max(1, height - 1)).square()).sqrt()
    anchor = output["anchor_logits"].float().flatten(2).sigmoid().amax(-1)
    anchor_mask = model_batch["anchor_mask"].bool()
    anchor_count = anchor_mask.sum(-1)
    anchor_mean = (anchor * anchor_mask).sum(-1) / anchor_count.clamp_min(1)
    anchor_min = anchor.masked_fill(~anchor_mask, float("inf")).amin(-1)
    anchor_min = torch.where(anchor_count > 0, anchor_min, torch.zeros_like(anchor_min))
    edge = output["relation_edge_logits"].float().sigmoid()
    # Direct queries have no relation/anchor edge, matching the dataset contract.
    edge_mask = model_batch["relation_mask"].bool() & anchor_mask[:, :edge.shape[1]]
    edge_count = edge_mask.sum(-1)
    edge_mean = (edge * edge_mask).sum(-1) / edge_count.clamp_min(1)
    edge_min = edge.masked_fill(~edge_mask, float("inf")).amin(-1)
    edge_min = torch.where(edge_count > 0, edge_min, torch.zeros_like(edge_min))
    rgb, depth = model_batch["r_thumb"].float(), model_batch["d_thumb"].float()
    columns = [
        target.sigmoid().amax(-1), target.sigmoid().mean(-1), ordered[:, 0],
        ordered[:, 0] - ordered[:, 1], entropy, ordered[:, :5].sum(-1), ordered.sum(-1),
        interior.sigmoid().amax(-1), interior.sigmoid().mean(-1), distance,
        anchor_mean, anchor_min, edge_mean, edge_min, output["fusion_gate_mean"].float(),
        F.cosine_similarity(rgb, depth, dim=-1), (rgb - depth).abs().mean(-1),
    ]
    columns.extend(output["source_logits"].float().sigmoid().unbind(-1))
    # Called before adding the residual: these are base V2 probabilities.
    columns.extend(output["answerability_logits"].float().softmax(-1).unbind(-1))
    result = torch.stack(columns, dim=-1)
    if result.shape[-1] != len(EVIDENCE_NAMES) or not bool(torch.isfinite(result).all()):
        raise ValueError("Invalid observable answerability evidence")
    return result


class AnswerabilityEvidenceAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 32, dropout: float = 0.2, context_dim: int = 0):
        super().__init__()
        self.input_dim = len(EVIDENCE_NAMES) + context_dim
        self.register_buffer("feature_mean", torch.zeros(self.input_dim))
        self.register_buffer("feature_scale", torch.ones(self.input_dim))
        self.net = nn.Sequential(nn.Linear(self.input_dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @torch.no_grad()
    def fit_standardization(self, train_evidence: torch.Tensor) -> None:
        if train_evidence.ndim != 2 or train_evidence.shape[1] != self.input_dim:
            raise ValueError("Expected train-only [samples, evidence] tensor")
        values = train_evidence.detach().float().to(self.feature_mean.device)
        if values.shape[0] == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError("Empty/non-finite standardization input")
        self.feature_mean.copy_(values.mean(0))
        self.feature_scale.copy_(values.std(0, unbiased=False).clamp_min(1e-3))

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        normalized = ((evidence.float() - self.feature_mean) / self.feature_scale).clamp(-5.0, 5.0)
        return self.net(normalized)
