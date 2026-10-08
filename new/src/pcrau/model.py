from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .answerability_evidence import AnswerabilityEvidenceAdapter, observable_answerability_evidence


class ResidualMLP(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(self.norm(value))


class ContextualQueryGraph(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        hidden = int(cfg["hidden_dim"])
        self.max_relations = int(cfg["max_relations"])
        self.max_anchors = int(cfg["max_anchors"])
        self.token_embedding = nn.Embedding(int(cfg["vocab_size"]), hidden, padding_idx=0)
        self.position_embedding = nn.Embedding(int(cfg["max_tokens"]), hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden, int(cfg["attention_heads"]), hidden * 4, float(cfg["dropout"]),
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, int(cfg["text_layers"]))
        self.target_slot = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.relation_slots = nn.Parameter(torch.randn(1, self.max_relations, hidden) * 0.02)
        self.anchor_slots = nn.Parameter(torch.randn(1, self.max_anchors, hidden) * 0.02)
        self.relation_embedding = nn.Embedding(len(cfg["relation_classes"]), hidden)
        self.cross_attention = nn.MultiheadAttention(
            hidden, int(cfg["attention_heads"]), float(cfg["dropout"]), batch_first=True
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        relation_ids: torch.Tensor,
        relation_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device)[None, :]
        text = self.token_embedding(token_ids) + self.position_embedding(positions)
        text = self.encoder(text, src_key_padding_mask=~token_mask)
        target = self.target_slot.expand(batch, -1, -1)
        relations = self.relation_slots.expand(batch, -1, -1) + self.relation_embedding(relation_ids)
        anchors = self.anchor_slots.expand(batch, -1, -1)
        query = torch.cat([target, relations, anchors], dim=1)
        attended, _ = self.cross_attention(query, text, text, key_padding_mask=~token_mask)
        query = self.norm(query + attended)
        target = query[:, 0]
        relations = query[:, 1 : 1 + self.max_relations]
        anchors = query[:, 1 + self.max_relations :]
        relation_weights = relation_mask.float().unsqueeze(-1)
        relation_context = (relations * relation_weights).sum(1) / relation_weights.sum(1).clamp_min(1.0)
        text_weights = token_mask.float().unsqueeze(-1)
        text_context = (text * text_weights).sum(1) / text_weights.sum(1).clamp_min(1.0)
        return {
            "target": target,
            "relations": relations,
            "anchors": anchors,
            "relation_context": relation_context,
            "text_context": text_context,
            "relation_mask": relation_mask,
            "anchor_mask": anchor_mask,
        }


class RelationConditionedFusion(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        feature = int(cfg["feature_dim"])
        hidden = int(cfg["hidden_dim"])
        dropout = float(cfg["dropout"])
        self.rgb = nn.Linear(feature, hidden, bias=False)
        self.depth = nn.Linear(feature, hidden, bias=False)
        self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.cross = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList([ResidualMLP(hidden, dropout) for _ in range(int(cfg["fusion_layers"]))])

    def forward(self, r0: torch.Tensor, d0: torch.Tensor, relation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = self.rgb(r0)
        depth = self.depth(d0)
        context = relation[:, None, None, :].expand_as(rgb)
        joined = torch.cat([rgb, depth, context], dim=-1)
        gate = torch.sigmoid(self.gate(joined))
        fused = self.norm(rgb + gate * depth + self.cross(joined))
        for block in self.blocks:
            fused = block(fused)
        return fused, gate


class SlotSpatialHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.visual = nn.Linear(hidden, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.bias = nn.Linear(hidden, 1)
        self.scale = math.sqrt(hidden)

    def forward(self, feature: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        visual = self.visual(feature)
        projected = self.query(query)
        if projected.ndim == 2:
            return torch.einsum("bhwd,bd->bhw", visual, projected) / self.scale + self.bias(query)[:, None]
        return torch.einsum("bhwd,bsd->bshw", visual, projected) / self.scale + self.bias(query)[:, :, :, None]


def spatial_node_features(logits: torch.Tensor, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return distribution-pooled visual features and normalized xy moments."""
    if logits.ndim == 3:
        logits = logits[:, None]
    batch, slots, height, width = logits.shape
    probabilities = torch.softmax(logits.flatten(-2), dim=-1).reshape(batch, slots, height, width)
    visual = torch.einsum("bshw,bhwd->bsd", probabilities, feature)
    ys = torch.linspace(-1.0, 1.0, height, device=logits.device, dtype=logits.dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=logits.device, dtype=logits.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    mean_x = (probabilities * grid_x).sum((-2, -1))
    mean_y = (probabilities * grid_y).sum((-2, -1))
    var_x = (probabilities * (grid_x - mean_x[:, :, None, None]).square()).sum((-2, -1))
    var_y = (probabilities * (grid_y - mean_y[:, :, None, None]).square()).sum((-2, -1))
    covariance = (
        probabilities
        * (grid_x - mean_x[:, :, None, None])
        * (grid_y - mean_y[:, :, None, None])
    ).sum((-2, -1))
    moments = torch.stack([mean_x, mean_y, var_x, var_y, covariance], dim=-1)
    return visual, moments


class PCRAUTargetV2(nn.Module):
    """Full target sidecar operating only on observable frozen features/text."""

    MODEL_INPUT_KEYS = {
        "r0", "d0", "r_thumb", "d_thumb", "token_ids", "token_mask",
        "relation_ids", "relation_mask", "anchor_mask",
    }

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        cfg = config["model"]
        hidden = int(cfg["hidden_dim"])
        feature = int(cfg["feature_dim"])
        anchors = int(cfg["max_anchors"])
        relations = int(cfg["max_relations"])
        self.cfg = dict(cfg)
        self.query_graph = ContextualQueryGraph(cfg)
        self.fusion = RelationConditionedFusion(cfg)
        self.target_head = SlotSpatialHead(hidden)
        self.interior_head = SlotSpatialHead(hidden)
        self.anchor_head = SlotSpatialHead(hidden)
        self.thumb_rgb = nn.Linear(feature, hidden)
        self.thumb_depth = nn.Linear(feature, hidden)
        node_dim = hidden + 5
        self.node_projection = nn.Linear(node_dim, hidden)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 5 + 15, hidden * 2), nn.GELU(), nn.Dropout(float(cfg["dropout"])),
            nn.Linear(hidden * 2, 1),
        )
        self.graph_projection = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        global_dim = hidden * 6
        self.answer_head = nn.Sequential(
            nn.LayerNorm(global_dim), nn.Linear(global_dim, hidden * 2), nn.GELU(),
            nn.Dropout(float(cfg["dropout"])), nn.Linear(hidden * 2, len(cfg["answerability_classes"])),
        )
        self.source_head = nn.Sequential(
            nn.LayerNorm(global_dim), nn.Linear(global_dim, hidden * 2), nn.GELU(),
            nn.Dropout(float(cfg["dropout"])), nn.Linear(hidden * 2, len(cfg["source_classes"])),
        )
        adapter_cfg = cfg.get("answerability_evidence_adapter")
        self.answerability_detail_dim = global_dim + hidden * (2 + relations + 2 * anchors) + 5 * (1 + anchors)
        if adapter_cfg and adapter_cfg.get("include_detail") and adapter_cfg.get("include_context"):
            raise ValueError("Detail features already include global context")
        self.answerability_adapter = (
            AnswerabilityEvidenceAdapter(int(adapter_cfg.get("hidden_dim", 32)), float(adapter_cfg.get("dropout", 0.2)),
                                         self.answerability_detail_dim if adapter_cfg.get("include_detail", False)
                                         else global_dim if adapter_cfg.get("include_context", False) else 0)
            if adapter_cfg is not None else None
        )
        self.relations = relations
        self.anchors = anchors

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        unexpected = set(batch) - self.MODEL_INPUT_KEYS
        if unexpected:
            raise ValueError(f"Model boundary received forbidden/unexpected keys: {sorted(unexpected)}")
        r0, d0 = batch["r0"], batch["d0"]
        r_thumb, d_thumb = batch["r_thumb"], batch["d_thumb"]
        modality_dropout = float(self.cfg.get("modality_dropout_probability", 0.0))
        dropped_rgb = torch.zeros(r0.shape[0], device=r0.device, dtype=torch.bool)
        dropped_depth = torch.zeros_like(dropped_rgb)
        if self.training and modality_dropout > 0.0:
            draw = torch.rand(r0.shape[0], device=r0.device)
            dropped_rgb = draw < modality_dropout / 2.0
            dropped_depth = (draw >= modality_dropout / 2.0) & (draw < modality_dropout)
            rgb_keep = (~dropped_rgb).to(r0.dtype)
            depth_keep = (~dropped_depth).to(d0.dtype)
            r0 = r0 * rgb_keep[:, None, None, None]
            d0 = d0 * depth_keep[:, None, None, None]
            r_thumb = r_thumb * rgb_keep[:, None]
            d_thumb = d_thumb * depth_keep[:, None]
        graph = self.query_graph(
            batch["token_ids"], batch["token_mask"], batch["relation_ids"],
            batch["relation_mask"], batch["anchor_mask"],
        )
        fused, gate = self.fusion(r0, d0, graph["relation_context"])
        target_logits = self.target_head(fused, graph["target"])
        interior_logits = self.interior_head(fused, graph["target"])
        anchor_logits = self.anchor_head(fused, graph["anchors"])
        target_visual, target_moments = spatial_node_features(target_logits, fused)
        anchor_visual, anchor_moments = spatial_node_features(anchor_logits, fused)
        target_node = self.node_projection(torch.cat([target_visual[:, 0], target_moments[:, 0]], dim=-1))
        anchor_nodes = self.node_projection(torch.cat([anchor_visual, anchor_moments], dim=-1))
        anchor_weights = graph["anchor_mask"].float().unsqueeze(-1)
        anchor_summary = (anchor_nodes * anchor_weights).sum(1) / anchor_weights.sum(1).clamp_min(1.0)
        anchor_moment_summary = (
            (anchor_moments * anchor_weights).sum(1) / anchor_weights.sum(1).clamp_min(1.0)
        )
        # A relation slot uses its corresponding anchor plus the full anchor
        # summary, so ternary relations such as nearer_than_both see both anchors.
        edge_parts = []
        for slot in range(self.relations):
            anchor_slot = min(slot, self.anchors - 1)
            geometry = torch.cat(
                [target_moments[:, 0], anchor_moments[:, anchor_slot], anchor_moment_summary], dim=-1
            )
            edge_parts.append(
                self.edge_head(
                    torch.cat(
                        [target_node, anchor_nodes[:, anchor_slot], anchor_summary, graph["relations"][:, slot],
                         graph["relation_context"], geometry], dim=-1
                    )
                ).squeeze(-1)
            )
        edge_logits = torch.stack(edge_parts, dim=1)
        relation_weights = graph["relation_mask"].float().unsqueeze(-1)
        relation_summary = (graph["relations"] * relation_weights).sum(1) / relation_weights.sum(1).clamp_min(1.0)
        predicted_graph = self.graph_projection(torch.cat([target_node, anchor_summary, relation_summary], dim=-1))
        thumbnail = self.thumb_rgb(r_thumb) + self.thumb_depth(d_thumb)
        global_feature = torch.cat(
            [fused.mean((1, 2)), fused.amax((1, 2)), graph["text_context"],
             graph["relation_context"], predicted_graph, thumbnail], dim=-1
        )
        output = {
            "target_logits": target_logits,
            "interior_logits": interior_logits,
            "anchor_logits": anchor_logits,
            "relation_edge_logits": edge_logits,
            "answerability_logits": self.answer_head(global_feature),
            "source_logits": self.source_head(global_feature),
            "predicted_graph_embedding": predicted_graph,
            "target_moments": target_moments[:, 0],
            "anchor_moments": anchor_moments,
            "fusion_gate_mean": gate.mean(dim=(1, 2, 3)),
            "modality_dropout_fraction": (dropped_rgb | dropped_depth).float().mean(),
        }
        detail = None
        if self.cfg.get("export_answerability_detail", False) or (
            self.answerability_adapter is not None and self.cfg["answerability_evidence_adapter"].get("include_detail", False)
        ):
            # All parts come from inference features, language slots and predicted
            # spatial moments. No target/anchor mask or annotation is inspected.
            detail = torch.cat([
                global_feature.float(), graph["target"].float(), graph["relations"].float().flatten(1),
                graph["anchors"].float().flatten(1), target_node.float(), anchor_nodes.float().flatten(1),
                target_moments[:, 0].float(), anchor_moments.float().flatten(1),
            ], dim=-1)
            if detail.shape[-1] != self.answerability_detail_dim:
                raise ValueError("Invalid answerability detail dimensions")
            if self.cfg.get("export_answerability_detail", False):
                output["observable_answerability_detail"] = detail
        if self.answerability_adapter is not None or self.cfg.get("export_answerability_evidence", False):
            evidence = observable_answerability_evidence(output, batch)
            if self.cfg.get("export_answerability_evidence", False):
                output["observable_answerability_evidence"] = evidence
        if self.answerability_adapter is not None:
            if self.cfg["answerability_evidence_adapter"].get("include_detail", False):
                evidence = torch.cat([evidence, detail], dim=-1)
            elif self.cfg["answerability_evidence_adapter"].get("include_context", False):
                evidence = torch.cat([evidence, global_feature.float()], dim=-1)
            output["answerability_logits"] = (
                output["answerability_logits"].float() + self.answerability_adapter(evidence).float()
            )
        return output
