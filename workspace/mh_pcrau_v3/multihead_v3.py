"""Exact MH-PCRA-U-v3 shared trunk and seven-head interface.

The module consumes only prompt-derived ``h_spatial``.  It has no access to
RGB/depth paths, oracle labels, split identifiers, or generated answer tokens.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn


RELATION_CLASSES = ("leftmost", "rightmost", "second_from_left", "second_from_right")
REASONING_DEPTH_CLASSES = ("direct_relation", "compositional_relation", "filter_relation_select")
SOURCE_CLASSES = ("SEMANTIC", "RELATION", "SPATIAL", "DEPTH", "OCCLUSION")
ANSWERABILITY_CLASSES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")


@dataclass(frozen=True)
class MHMultiHeadConfig:
    hidden_size: int = 1536
    trunk_size: int = 512
    spatial_hidden_size: int = 256
    dropout: float = 0.10
    relation_classes: int = 4
    reasoning_classes: int = 3
    source_classes: int = 5
    answerability_classes: int = 4
    log_variance_min: float = -8.0
    log_variance_max: float = 2.0
    architecture_id: str = "MH-PCRA-U-v3"

    def validate(self) -> None:
        expected = {
            "hidden_size": 1536, "trunk_size": 512, "spatial_hidden_size": 256,
            "dropout": 0.10, "relation_classes": 4, "reasoning_classes": 3,
            "source_classes": 5, "answerability_classes": 4,
            "log_variance_min": -8.0, "log_variance_max": 2.0,
            "architecture_id": "MH-PCRA-U-v3",
        }
        for key, value in expected.items():
            if getattr(self, key) != value:
                raise ValueError(f"Architecture identity drift: {key}={getattr(self, key)!r}, expected {value!r}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MHMultiHeadOutput:
    relation_logits: torch.Tensor
    relation_probabilities: torch.Tensor
    reasoning_logits: torch.Tensor
    reasoning_probabilities: torch.Tensor
    mu_uv: torch.Tensor
    log_variance_uv: torch.Tensor
    source_logits: torch.Tensor
    source_probabilities: torch.Tensor
    answerability_logits: torch.Tensor
    answerability_probabilities: torch.Tensor
    confidence_logit: torch.Tensor
    raw_safe_score: torch.Tensor
    z_spatial: torch.Tensor


class SpatialMLPHead(nn.Module):
    def __init__(self, trunk_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class MHMultiHeadPCRAU(nn.Module):
    """Shared trunk plus seven task heads from the locked v3 plan."""

    def __init__(self, config: MHMultiHeadConfig | None = None):
        super().__init__()
        self.config = config or MHMultiHeadConfig()
        self.config.validate()
        c = self.config
        self.shared_trunk = nn.Sequential(
            nn.LayerNorm(c.hidden_size),
            nn.Linear(c.hidden_size, c.trunk_size),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )
        self.relation_head = nn.Linear(c.trunk_size, c.relation_classes)
        self.reasoning_head = nn.Linear(c.trunk_size, c.reasoning_classes)
        self.coordinate_head = SpatialMLPHead(c.trunk_size, c.spatial_hidden_size, 2)
        self.log_variance_head = SpatialMLPHead(c.trunk_size, c.spatial_hidden_size, 2)
        self.source_head = nn.Linear(c.trunk_size, c.source_classes)
        self.answerability_head = nn.Linear(c.trunk_size, c.answerability_classes)
        self.confidence_head = nn.Linear(c.trunk_size, 1)
        self.configure_trainable("s1a")

    def configure_trainable(self, stage: str) -> dict[str, int]:
        """Enforce plan stages instead of relying only on zero loss weights."""
        if stage not in {"s1a", "s1b", "inference"}:
            raise ValueError(f"Unknown stage: {stage}")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if stage == "s1a":
            for name, module in self.named_children():
                if name != "confidence_head":
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        elif stage == "s1b":
            for parameter in self.confidence_head.parameters():
                parameter.requires_grad_(True)
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def forward(self, h_spatial: torch.Tensor) -> MHMultiHeadOutput:
        if h_spatial.ndim != 2 or h_spatial.shape[1] != self.config.hidden_size:
            raise ValueError(f"Expected h_spatial [B,{self.config.hidden_size}]")
        if h_spatial.shape[0] < 1:
            raise ValueError("Empty batches are not supported")
        if not torch.is_floating_point(h_spatial) or not torch.isfinite(h_spatial).all():
            raise ValueError("h_spatial must be a finite floating-point tensor")
        z = self.shared_trunk(h_spatial)
        relation_logits = self.relation_head(z)
        reasoning_logits = self.reasoning_head(z)
        mu_uv = torch.sigmoid(self.coordinate_head(z))
        log_variance_uv = self.log_variance_head(z).clamp(
            self.config.log_variance_min, self.config.log_variance_max
        )
        source_logits = self.source_head(z)
        answerability_logits = self.answerability_head(z)
        confidence_logit = self.confidence_head(z).squeeze(-1)
        return MHMultiHeadOutput(
            relation_logits=relation_logits,
            relation_probabilities=torch.softmax(relation_logits, dim=-1),
            reasoning_logits=reasoning_logits,
            reasoning_probabilities=torch.softmax(reasoning_logits, dim=-1),
            mu_uv=mu_uv,
            log_variance_uv=log_variance_uv,
            source_logits=source_logits,
            source_probabilities=torch.softmax(source_logits, dim=-1),
            answerability_logits=answerability_logits,
            answerability_probabilities=torch.softmax(answerability_logits, dim=-1),
            confidence_logit=confidence_logit,
            raw_safe_score=torch.sigmoid(confidence_logit),
            z_spatial=z,
        )


def build_seeded_model(seed: int, config: MHMultiHeadConfig | None = None) -> MHMultiHeadPCRAU:
    """Create deterministic parameters without leaving the caller RNG changed."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return MHMultiHeadPCRAU(config)


def _list(values: torch.Tensor) -> list:
    return values.detach().cpu().tolist()


def to_development_records(
    output: MHMultiHeadOutput,
    sample_ids: Sequence[str],
    *,
    model_id: str,
    source_hash: str,
) -> list[dict]:
    """Render prediction-only records; never accepts supervision/oracle fields."""
    batch = output.mu_uv.shape[0]
    if len(sample_ids) != batch or any(not isinstance(x, str) or not x for x in sample_ids):
        raise ValueError("sample_ids must contain one nonempty ID per output row")
    if not model_id or len(source_hash) != 64:
        raise ValueError("model_id and a SHA-256 source_hash are required")
    relation_index = output.relation_probabilities.argmax(-1)
    reasoning_index = output.reasoning_probabilities.argmax(-1)
    source_index = output.source_probabilities.argmax(-1)
    answerability_index = output.answerability_probabilities.argmax(-1)
    records = []
    for i, sample_id in enumerate(sample_ids):
        records.append({
            "schema_version": "mh_pcrau_v3.development_prediction/1.0",
            "sample_id": sample_id,
            "model_id": model_id,
            "split_role": "DEVELOPMENT",
            "source_hash": source_hash,
            "relation": {
                "label": RELATION_CLASSES[int(relation_index[i])],
                "logits": _list(output.relation_logits[i]),
                "probabilities": _list(output.relation_probabilities[i]),
            },
            "reasoning": {
                "depth": int(reasoning_index[i]),
                "label": REASONING_DEPTH_CLASSES[int(reasoning_index[i])],
                "logits": _list(output.reasoning_logits[i]),
                "probabilities": _list(output.reasoning_probabilities[i]),
                "explanation_status": "STRUCTURED_UNCERTIFIED",
            },
            "point_uv": _list(output.mu_uv[i]),
            "log_variance_uv": _list(output.log_variance_uv[i]),
            "variance_uv": _list(output.log_variance_uv[i].exp()),
            "uncertainty_source": {
                "label": SOURCE_CLASSES[int(source_index[i])],
                "logits": _list(output.source_logits[i]),
                "probabilities": _list(output.source_probabilities[i]),
                "status": "EXPLORATORY_UNCERTIFIED",
            },
            "answerability": {
                "label": ANSWERABILITY_CLASSES[int(answerability_index[i])],
                "logits": _list(output.answerability_logits[i]),
                "probabilities": _list(output.answerability_probabilities[i]),
            },
            "raw_safe_score": float(output.raw_safe_score[i].detach().cpu()),
            "calibrated_safe_probability": None,
            "action": "UNAUTHORIZED_BEFORE_CALIBRATION",
        })
    return records
