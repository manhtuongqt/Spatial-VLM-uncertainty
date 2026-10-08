"""Experimental observable support for a frozen answerability adapter.

These are learned-slot and 2D moment proxies, not verified physical geometry.
No sample identity, variant, mask, annotation or metric-depth/TF is an input.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .acceptance_ranker import AcceptanceRanker
from .answerability_evidence import EVIDENCE_NAMES

SUPPORT_NAMES = (["target_query_node_cosine", "active_anchor_fraction", "active_edge_fraction"]
    + [f"anchor_{i}_{name}" for i in range(3) for name in
       ["query_node_cosine", "relation_node_cosine", "target_node_cosine", "dx", "dy", "distance",
        "joint_variance", "moment_overlap_proxy", "active"]]
    + [f"anchor_pair_{i}_{j}_{name}" for i, j in [(0, 1), (0, 2), (1, 2)]
       for name in ["distance", "node_cosine", "active"]]
    + ["relation_edge_gap", "occlusion_target_peak", "found_shift"])


def support_features(evidence, probabilities, anchor_mask, relation_mask):
    if evidence.ndim != 2 or evidence.shape[1] != 2222 or probabilities.shape != (len(evidence), 4):
        raise ValueError("Expected observable evidence/detail and four answer probabilities")
    if anchor_mask.shape != (len(evidence), 3) or relation_mask.shape != anchor_mask.shape:
        raise ValueError("Expected language-declared slot masks")
    e, d = evidence[:, :26].float(), evidence[:, 26:].float()
    tq, rq, aq = d[:, 768:896], d[:, 896:1280].reshape(-1, 3, 128), d[:, 1280:1664].reshape(-1, 3, 128)
    tn, an = d[:, 1664:1792], d[:, 1792:2176].reshape(-1, 3, 128)
    tm, am = d[:, 2176:2181], d[:, 2181:2196].reshape(-1, 3, 5)
    anchors = anchor_mask.float(); edges = (anchor_mask.bool() & relation_mask.bool()).float()
    columns = [F.cosine_similarity(tq, tn, dim=-1), anchors.mean(-1), edges.mean(-1)]
    for i in range(3):
        delta = tm[:, :2] - am[:, i, :2]
        variance = (tm[:, 2:4] + am[:, i, 2:4]).clamp_min(1e-6)
        overlap = torch.exp(-.5 * (delta.square() / variance).sum(-1).clamp_max(40))
        active = anchors[:, i]
        columns.extend([F.cosine_similarity(aq[:, i], an[:, i], dim=-1)*active,
                        F.cosine_similarity(rq[:, i], an[:, i], dim=-1)*edges[:, i],
                        F.cosine_similarity(tn, an[:, i], dim=-1)*active,
                        delta[:, 0]*active, delta[:, 1]*active, delta.norm(dim=-1)*active,
                        variance.sum(-1)*active, overlap*active, active])
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        active = anchors[:, i]*anchors[:, j]
        columns.extend([(am[:, i, :2]-am[:, j, :2]).norm(dim=-1)*active,
                        F.cosine_similarity(an[:, i], an[:, j], dim=-1)*active, active])
    columns.extend([e[:, 18]*(1-e[:, 13]), e[:, 21]*e[:, 2], probabilities[:, 0]-e[:, 22]])
    result = torch.stack(columns, -1)
    if result.shape[1] != len(SUPPORT_NAMES) or not torch.isfinite(result).all():
        raise ValueError("Invalid observable support proxies")
    return result


def ranker_features(evidence, probabilities, anchors, relations, mode):
    if mode == "scalar":
        return torch.cat([evidence[:, :26].float(), probabilities.float()], -1)
    if mode == "detail":
        return torch.cat([evidence.float(), probabilities.float()], -1)
    if mode == "support":
        return torch.cat([evidence.float(), probabilities.float(),
                          support_features(evidence, probabilities, anchors, relations)], -1)
    raise ValueError("Unknown support mode")


class FrozenSupportScorer(nn.Module):
    def __init__(self, base, ranker, mode, checkpoint_sha256, source_checkpoint_sha256):
        super().__init__()
        self.base, self.ranker, self.mode = base, ranker, mode
        self.ranker_sha256, self.source_v2_sha256 = checkpoint_sha256, source_checkpoint_sha256
        self.source_checkpoint_sha256 = source_checkpoint_sha256
        self.requires_grad_(False).eval()

    def forward(self, batch):
        if self.training:
            raise ValueError("Support scorer is frozen and inference-only")
        output = self.base(batch)
        evidence = torch.cat([output['observable_answerability_evidence'].float(),
                              output['observable_answerability_detail'].float()], -1)
        logits = output['answerability_logits'].float()
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probability = logits.softmax(-1)
            features = ranker_features(evidence, probability, batch['anchor_mask'], batch['relation_mask'], self.mode)
            score = torch.logsumexp(logits[:, 1:], -1)-logits[:, 0]+self.ranker(features)
        if not torch.isfinite(score).all():
            raise ValueError("Non-finite support score")
        return {**output, 'ranker_error_logit': score}


def load_support_scorer(base, path, source_checkpoint_sha256):
    from pathlib import Path
    from .utils import sha256_file
    path = Path(path)
    device = next(base.parameters()).device
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved['source_checkpoint_sha256'] != source_checkpoint_sha256:
        raise ValueError("Support scorer source checkpoint mismatch")
    head = AcceptanceRanker(saved['input_dim'], saved['hidden_dim'], saved['dropout']).to(device)
    head.load_state_dict(saved['state_dict'])
    return FrozenSupportScorer(base, head, saved['mode'], sha256_file(path), source_checkpoint_sha256)
