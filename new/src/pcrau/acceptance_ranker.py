"""Experimental binary acceptance scoring on frozen observable V2 features."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .answerability_evidence import observable_answerability_evidence


class AcceptanceRanker(nn.Module):
    """Output is an uncalibrated error logit, not a robot safety probability."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3, zero_output: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_scale", torch.ones(input_dim))
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        if zero_output:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    @torch.no_grad()
    def fit_standardization(self, train_features: torch.Tensor):
        if train_features.ndim != 2 or train_features.shape[1] != self.input_dim:
            raise ValueError("Invalid train feature shape")
        if len(train_features) == 0 or not torch.isfinite(train_features).all():
            raise ValueError("Empty or non-finite train features")
        self.feature_mean.copy_(train_features.float().mean(0))
        self.feature_scale.copy_(train_features.float().std(0, unbiased=False).clamp_min(1e-3))

    def forward(self, observable_features: torch.Tensor):
        normalized = ((observable_features.float()-self.feature_mean)/self.feature_scale).clamp(-5, 5)
        return self.net(normalized).squeeze(-1)


def acceptance_loss(error_logits: torch.Tensor, error_event: torch.Tensor, ranking_weight: float = 0.0):
    """BCE plus pairwise ordering: erroneous cases should have higher scores."""
    labels = error_event.float()
    bce = F.binary_cross_entropy_with_logits(error_logits, labels)
    bad, good = error_logits[labels > 0.5], error_logits[labels <= 0.5]
    ranking = F.softplus(good[:, None]-bad[None, :]).mean() if len(bad) and len(good) else error_logits.sum()*0
    return bce + ranking_weight*ranking


class FrozenAcceptanceScorer(nn.Module):
    """Inference wrapper preserving every V2 prediction, adding one raw logit."""

    def __init__(self, base: nn.Module, ranker: AcceptanceRanker, ranker_sha256: str, source_v2_sha256: str):
        super().__init__()
        if getattr(base, "answerability_adapter", None) is not None:
            raise ValueError("Acceptance scorer requires the frozen unmodified V2 answer head")
        self.base, self.ranker = base, ranker
        self.ranker_sha256, self.source_v2_sha256 = ranker_sha256, source_v2_sha256
        self.requires_grad_(False).eval()

    def forward(self, batch):
        if self.training:
            raise ValueError("Frozen acceptance wrapper is inference-only")
        context = []
        handle = self.base.answer_head.register_forward_pre_hook(lambda module, args: context.append(args[0]))
        try:
            output = self.base(batch)
        finally:
            handle.remove()
        if len(context) != 1:
            raise ValueError("Expected one V2 answer context")
        # No evaluator labels or metadata are used here; base checks its boundary.
        with torch.autocast(device_type=context[0].device.type, enabled=False):
            evidence = observable_answerability_evidence(output, batch)
            features = torch.cat([evidence, context[0].float()], dim=-1)
            if self.ranker.input_dim == evidence.shape[-1]:
                features = evidence
            logits = output["answerability_logits"].float()
            error_logit = torch.logsumexp(logits[:, 1:], dim=-1)-logits[:, 0]+self.ranker(features)
        if not torch.isfinite(error_logit).all():
            raise ValueError("Invalid acceptance score")
        return {**output, "observable_answerability_evidence": evidence, "ranker_error_logit": error_logit}


def load_acceptance_scorer(base: nn.Module, checkpoint, source_v2_sha256: str):
    from .utils import sha256_file
    device = next(base.parameters()).device
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if not saved.get("residual") or saved.get("source_v2_sha256") != source_v2_sha256:
        raise ValueError("Ranker does not match the frozen V2 checkpoint")
    ranker = AcceptanceRanker(saved["input_dim"], saved["hidden_dim"], saved["dropout"]).to(device)
    ranker.load_state_dict(saved["state_dict"])
    return FrozenAcceptanceScorer(base, ranker, sha256_file(checkpoint), source_v2_sha256)
