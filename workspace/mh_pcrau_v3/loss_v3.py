"""Masked multi-task objectives for MH-PCRA-U-v3."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .multihead_v3 import MHMultiHeadOutput


FOUND_INDEX = 0


@dataclass
class SupervisionBatch:
    relation: torch.Tensor
    relation_mask: torch.Tensor
    reasoning_depth: torch.Tensor
    reasoning_mask: torch.Tensor
    target_uv: torch.Tensor
    spatial_mask: torch.Tensor
    uncertainty_source: torch.Tensor
    source_mask: torch.Tensor
    answerability: torch.Tensor
    answerability_mask: torch.Tensor
    safe_to_execute: torch.Tensor | None = None
    confidence_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class LossWeights:
    relation: float = 1.0
    reasoning: float = 1.0
    spatial: float = 1.0
    source: float = 1.0
    answerability: float = 1.0
    confidence: float = 0.0
    language_modeling: float = 0.0


@dataclass
class MultiTaskLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    valid_counts: dict[str, int]


def _validate_mask(mask: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    if mask.shape != (batch,):
        raise ValueError(f"{name} must have shape [B]")
    if mask.dtype != torch.bool:
        raise ValueError(f"{name} must be bool")
    return mask


def _masked_ce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
               classes: int, anchor: torch.Tensor, name: str) -> tuple[torch.Tensor, int]:
    count = int(mask.sum().item())
    if labels.shape != (logits.shape[0],) or labels.dtype != torch.long:
        raise ValueError(f"{name} labels must be int64 [B]")
    if count == 0:
        return anchor, 0
    selected = labels[mask]
    if ((selected < 0) | (selected >= classes)).any():
        raise ValueError(f"Invalid active {name} label")
    return F.cross_entropy(logits[mask], selected, reduction="mean"), count


def compute_multitask_loss(
    output: MHMultiHeadOutput,
    target: SupervisionBatch,
    weights: LossWeights | None = None,
    *,
    stage: str = "s1a",
) -> MultiTaskLoss:
    weights = weights or LossWeights()
    if stage not in {"s1a", "s1b"}:
        raise ValueError("stage must be s1a or s1b")
    if weights.language_modeling != 0:
        raise ValueError("Language-modeling loss is unauthorized in head-only stages")
    if stage == "s1a" and weights.confidence != 0:
        raise ValueError("S1a requires confidence weight exactly zero")
    batch = output.relation_logits.shape[0]
    masks = {
        "relation": _validate_mask(target.relation_mask, batch, "relation_mask"),
        "reasoning": _validate_mask(target.reasoning_mask, batch, "reasoning_mask"),
        "spatial": _validate_mask(target.spatial_mask, batch, "spatial_mask"),
        "source": _validate_mask(target.source_mask, batch, "source_mask"),
        "answerability": _validate_mask(target.answerability_mask, batch, "answerability_mask"),
    }
    if target.answerability.shape != (batch,) or target.answerability.dtype != torch.long:
        raise ValueError("answerability labels must be int64 [B]")
    if masks["spatial"].any():
        spatial_rows = masks["spatial"]
        if not torch.all(masks["answerability"][spatial_rows]):
            raise ValueError("Spatial supervision requires an active answerability label")
        if not torch.all(target.answerability[spatial_rows] == FOUND_INDEX):
            raise ValueError("Spatial supervision is FOUND-only")
        if target.target_uv.shape != (batch, 2):
            raise ValueError("target_uv must have shape [B,2]")
        selected_uv = target.target_uv[spatial_rows]
        if not torch.isfinite(selected_uv).all() or ((selected_uv < 0) | (selected_uv > 1)).any():
            raise ValueError("Active target_uv must be finite and normalized")
    anchor = output.relation_logits.sum() * 0.0
    relation, n_relation = _masked_ce(output.relation_logits, target.relation,
                                      masks["relation"], 4, anchor, "relation")
    reasoning, n_reasoning = _masked_ce(output.reasoning_logits, target.reasoning_depth,
                                        masks["reasoning"], 3, anchor, "reasoning")
    source, n_source = _masked_ce(output.source_logits, target.uncertainty_source,
                                  masks["source"], 5, anchor, "source")
    answerability, n_answerability = _masked_ce(
        output.answerability_logits, target.answerability, masks["answerability"],
        4, anchor, "answerability"
    )
    n_spatial = int(masks["spatial"].sum().item())
    if n_spatial:
        delta2 = (target.target_uv[masks["spatial"]] - output.mu_uv[masks["spatial"]]).square()
        logvar = output.log_variance_uv[masks["spatial"]]
        spatial = (0.5 * (torch.exp(-logvar) * delta2 + logvar).sum(dim=-1)).mean()
    else:
        spatial = anchor
    confidence = anchor
    n_confidence = 0
    if stage == "s1b":
        if target.safe_to_execute is None or target.confidence_mask is None:
            raise ValueError("S1b requires OOF safe labels and confidence mask")
        confidence_mask = _validate_mask(target.confidence_mask, batch, "confidence_mask")
        n_confidence = int(confidence_mask.sum().item())
        if target.safe_to_execute.shape != (batch,):
            raise ValueError("safe_to_execute must have shape [B]")
        if n_confidence:
            selected = target.safe_to_execute[confidence_mask].to(output.confidence_logit.dtype)
            if not torch.isfinite(selected).all() or ((selected < 0) | (selected > 1)).any():
                raise ValueError("Active OOF safe labels must be binary probabilities")
            confidence = F.binary_cross_entropy_with_logits(
                output.confidence_logit[confidence_mask], selected, reduction="mean"
            )
    components = {
        "relation": relation, "reasoning": reasoning, "spatial": spatial,
        "source": source, "answerability": answerability, "confidence": confidence,
        "language_modeling": anchor,
    }
    total = (
        weights.relation * relation + weights.reasoning * reasoning
        + weights.spatial * spatial + weights.source * source
        + weights.answerability * answerability + weights.confidence * confidence
    )
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite multi-task loss")
    return MultiTaskLoss(total=total, components=components, valid_counts={
        "relation": n_relation, "reasoning": n_reasoning, "spatial": n_spatial,
        "source": n_source, "answerability": n_answerability,
        "confidence": n_confidence, "language_modeling": 0,
    })
