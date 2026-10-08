"""Opt-in residual anchor queries. Baseline predictions never consume this branch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from .model import PCRAUTargetV2
from .query_parser import TypedQuery, parse_query
from .text import prompt_anchor_mask, relation_ids, tokenize

SHADOW_VERSION = "pcrau_anchor_shadow_s1_v1"


@dataclass(frozen=True)
class FrozenAnchorCapture:
    baseline: dict[str, torch.Tensor]
    text: torch.Tensor
    old_queries: torch.Tensor
    fused: torch.Tensor
    token_mask: torch.Tensor
    phrase_masks: torch.Tensor
    active_slots: torch.Tensor
    queries: tuple[TypedQuery, ...]
    scope_status: tuple[str, ...]
    head_autocast_dtype: torch.dtype | None


class AnchorShadow(nn.Module):
    """M1 phrase pooling or M2 whole-text pooling, sharing a frozen baseline.

    Input boundary: precisely the nine original model tensor keys plus a separate
    sequence of prompts. No supervision enters capture/forward. Captures can be
    shared between M1/M2 to ensure identical frozen features. Sequential use only:
    temporary hooks must not overlap another call on the same baseline instance.
    """

    def __init__(self, baseline: PCRAUTargetV2, conditioning: str = "phrase", seed: int = 24082026):
        super().__init__()
        if conditioning not in {"phrase", "whole_text"}:
            raise ValueError("conditioning must be phrase or whole_text")
        if int(baseline.cfg["hidden_dim"]) != 128:
            raise ValueError("Protocol v1 fixes hidden_dim=128 and 33,024 trainable parameters")
        self.baseline = baseline.eval().requires_grad_(False)
        for param in baseline.parameters():
            param.grad = None
        self.conditioning = conditioning
        self.seed = seed
        self._capturing = False
        # CPU-only initialization without changing the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.residual = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128)).float()
        nn.init.zeros_(self.residual[2].weight)
        nn.init.zeros_(self.residual[2].bias)
        self.residual.to(next(baseline.parameters()).device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def _check_frozen(self):
        if any(module.training for module in self.baseline.modules()):
            raise RuntimeError("Frozen baseline must remain entirely in eval mode")
        if any(p.requires_grad or p.grad is not None for p in self.baseline.parameters()):
            raise RuntimeError("Baseline weights must be frozen with no gradients")
        if any(p.dtype != torch.float32 for p in self.residual.parameters()):
            raise RuntimeError("Residual MLP must remain FP32 (use autocast for the frozen head)")

    def _parse_boundary(self, batch: Mapping[str, torch.Tensor], prompts: Sequence[str]):
        if set(batch) != self.baseline.MODEL_INPUT_KEYS:
            raise ValueError("Shadow boundary accepts exactly the nine baseline tensor keys; no oracle fields")
        if isinstance(prompts, str) or len(prompts) != batch["token_ids"].shape[0]:
            raise ValueError("One original full prompt is required per batch row")
        cfg = self.baseline.cfg
        length, slots = int(cfg["max_tokens"]), int(cfg["max_anchors"])
        parsed = tuple(parse_query(p, length, slots) for p in prompts)
        device = batch["token_ids"].device
        expected = {k: [] for k in ("token_ids", "token_mask", "relation_ids", "relation_mask", "anchor_mask")}
        for prompt in prompts:
            ids, mask = tokenize(prompt, length, int(cfg["vocab_size"]))
            rid, rmask = relation_ids(prompt, int(cfg["max_relations"]))
            for key, value in zip(expected, (ids, mask, rid, rmask, prompt_anchor_mask(prompt, slots))):
                expected[key].append(value)
        for key, values in expected.items():
            dtype = torch.long if key.endswith("ids") else torch.bool
            check = torch.tensor(values, dtype=dtype, device=device)
            if batch[key].dtype != dtype or not torch.equal(check, batch[key]):
                raise ValueError(f"Prompt/batch mismatch: {key}")
        phrase_masks = torch.tensor([q.token_masks()["anchor_token_masks"] for q in parsed], device=device, dtype=torch.bool)
        active = torch.tensor([q.token_masks()["anchor_slot_mask"] for q in parsed], device=device, dtype=torch.bool)
        statuses = tuple("SUPPORTED_SHADOW" if q.supported and q.anchors else
                         "DIRECT_BYPASS" if q.supported else "PARSE_UNSUPPORTED" for q in parsed)
        return parsed, phrase_masks, active, statuses

    def capture(self, batch: Mapping[str, torch.Tensor], prompts: Sequence[str]) -> FrozenAnchorCapture:
        self._check_frozen()
        parsed, phrase_masks, active, statuses = self._parse_boundary(batch, prompts)
        if self._capturing:
            raise RuntimeError("Concurrent/reentrant capture on this wrapper is unsupported")
        found = {}

        def collect(name, value):
            if name in found:
                raise RuntimeError(f"Expected exactly one {name} hook call")
            # Preserve the sliced query layout. Making [B,3,D] contiguous
            # changes Linear's kernel path and can break the FP32 identity gate.
            found[name] = value.detach()

        hooks = []
        self._capturing = True
        try:
            hooks.append(self.baseline.query_graph.encoder.register_forward_hook(lambda m, a, out: collect("text", out)))
            hooks.append(self.baseline.query_graph.register_forward_hook(lambda m, a, out: collect("query", out["anchors"])))
            hooks.append(self.baseline.fusion.register_forward_hook(lambda m, a, out: collect("fused", out[0])))
            # Even under an outer inference_mode, materialize ordinary detached
            # tensors that autograd can safely save later for the residual head.
            with torch.inference_mode(False), torch.no_grad():
                output = self.baseline(batch)
                token_mask = batch["token_mask"].detach().clone()
                phrase_masks = phrase_masks.clone()
                active = active.clone()
        finally:
            for hook in hooks:
                hook.remove()
            self._capturing = False
        b, length = batch["token_ids"].shape
        h = int(self.baseline.cfg["hidden_dim"])
        if (set(found) != {"text", "query", "fused"}
                or found["text"].shape != (b, length, h)
                or found["query"].shape != (b, self.baseline.anchors, h)
                or found["fused"].shape != (*batch["r0"].shape[:-1], h)):
            raise RuntimeError("Frozen capture shape/layer contract failed")
        dtype = torch.get_autocast_dtype(found["text"].device.type) if torch.is_autocast_enabled(found["text"].device.type) else None
        return FrozenAnchorCapture(output, found["text"], found["query"], found["fused"], token_mask,
                                   phrase_masks, active, parsed, statuses, dtype)

    def predict(self, capture: FrozenAnchorCapture) -> torch.Tensor:
        self._check_frozen()
        if not bool(capture.active_slots.any()):
            return capture.baseline["anchor_logits"]
        # MLP and pooling FP32 even when the frozen head is called under AMP.
        with torch.autocast(device_type=capture.text.device.type, enabled=False):
            masks = capture.phrase_masks if self.conditioning == "phrase" else (
                capture.token_mask[:, None, :].expand_as(capture.phrase_masks))
            weights = masks.float()
            pooled = torch.einsum("bsl,bld->bsd", weights, capture.text.float()) / weights.sum(-1, keepdim=True).clamp_min(1)
            delta = self.residual(pooled) * capture.active_slots[..., None]
        dense_query = capture.old_queries + delta.to(capture.old_queries.dtype)
        query = torch.empty_strided(capture.old_queries.shape, capture.old_queries.stride(),
                                    device=dense_query.device, dtype=dense_query.dtype)
        query.copy_(dense_query)  # Differentiable copy; reproduce baseline Linear layout.
        with torch.autocast(device_type=capture.text.device.type, enabled=capture.head_autocast_dtype is not None,
                            dtype=capture.head_autocast_dtype or torch.bfloat16):
            logits = self.baseline.anchor_head(capture.fused, query)
        # Restore all bypass rows and inactive slots exactly, also after training.
        return torch.where(capture.active_slots[..., None, None], logits, capture.baseline["anchor_logits"])

    def forward(self, batch: Mapping[str, torch.Tensor], prompts: Sequence[str]) -> dict:
        capture = self.capture(batch, prompts)
        return {"baseline": capture.baseline, "anchor_logits_shadow": self.predict(capture),
                "active_slots": capture.active_slots, "scope_status": capture.scope_status,
                "queries": capture.queries, "conditioning": self.conditioning, "version": SHADOW_VERSION}
