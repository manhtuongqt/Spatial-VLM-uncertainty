"""Opt-in v2 loss/arm preparation. Oracle tensors belong only to supervision.

No optimizer or training runner is constructed here. Baseline, v1 loss and
AnchorShadow source/checkpoint contract remain unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .anchor_shadow import AnchorShadow
from .anchor_shadow_experiment import PROTOCOL, anchor_shadow_loss

PEAK_VERSION = "pcrau_anchor_peak_v2_preflight_v1"
PEAK_MARGIN = 1.0
PEAK_WEIGHT = 0.10


@dataclass(frozen=True)
class PeakSupervision:
    """Train/evaluator-only tensors; never pass this object into capture/head."""
    area_masks: torch.Tensor          # [B,S,H,W], same INTER_AREA masks as v1
    center_support: torch.Tensor      # bool [B,S,H,W], full-mask pixel centers
    full_visible: torch.Tensor        # bool [B,S], full mask has >=1 pixel
    full_pixel_counts: torch.Tensor   # long [B,S], diagnostic; not model input

    def to(self, device):
        return PeakSupervision(*(v.to(device) for v in (
            self.area_masks, self.center_support, self.full_visible, self.full_pixel_counts)))


def supervision_from_full_masks(full_masks: np.ndarray, grid_shape=(24, 32)) -> PeakSupervision:
    """Keep v1 resize exact; sample the same pixel centers as MAP evaluation."""
    masks = np.asarray(full_masks)
    if masks.ndim != 4 or masks.dtype != np.bool_:
        raise ValueError("Expected boolean full masks [B,S,image_height,image_width]")
    gh, gw = grid_shape
    height, width = masks.shape[-2:]
    if min(gh, gw, height, width) <= 0 or gh > height or gw > width:
        raise ValueError("Invalid image/grid dimensions")
    xs = ((np.arange(gw) + .5) * width / gw).astype(np.int64)
    ys = ((np.arange(gh) + .5) * height / gh).astype(np.int64)
    centers = masks[..., ys[:, None], xs[None, :]].copy()
    area = np.empty((*masks.shape[:2], gh, gw), dtype=np.float32)
    for b in range(masks.shape[0]):
        for s in range(masks.shape[1]):
            area[b, s] = cv2.resize(masks[b, s].astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    counts = masks.sum((-2, -1), dtype=np.int64)
    return PeakSupervision(torch.from_numpy(area), torch.from_numpy(centers),
                           torch.from_numpy(counts > 0), torch.from_numpy(counts))


def _validate(logits, supervision, active):
    if logits.ndim != 4 or logits.shape != supervision.area_masks.shape or logits.shape != supervision.center_support.shape:
        raise ValueError("Expected matching logits/area/centers [B,S,H,W]")
    if active.shape != logits.shape[:2] or supervision.full_visible.shape != active.shape or supervision.full_pixel_counts.shape != active.shape:
        raise ValueError("Expected one active/visible/count flag per slot")
    if active.dtype != torch.bool or supervision.center_support.dtype != torch.bool or supervision.full_visible.dtype != torch.bool:
        raise ValueError("Active/visible/centers must be boolean")
    if supervision.full_pixel_counts.dtype != torch.long:
        raise ValueError("Pixel counts must be int64")
    tensors = (supervision.area_masks, supervision.center_support, supervision.full_visible, supervision.full_pixel_counts, active)
    if any(v.device != logits.device for v in tensors):
        raise ValueError("All loss tensors must use the same device")
    if not logits.is_floating_point() or not torch.isfinite(logits).all() or not torch.isfinite(supervision.area_masks).all():
        raise ValueError("Expected finite floating-point logits/targets")
    if bool(((supervision.area_masks < 0) | (supervision.area_masks > 1)).any()) or bool((supervision.full_pixel_counts < 0).any()):
        raise ValueError("Invalid target occupancy/count")
    if not torch.equal(supervision.full_visible, supervision.full_pixel_counts > 0):
        raise ValueError("Full mask visible/count mismatch")
    if not torch.equal(supervision.full_visible, supervision.area_masks.sum((-2, -1)) > 0):
        raise ValueError("Full mask/area occupancy visibility mismatch; do not relabel tiny masks")
    if bool((supervision.center_support.flatten(2).any(-1) & ~supervision.full_visible).any()):
        raise ValueError("An empty full mask cannot have a positive center")


def peak_auxiliary_loss(logits, supervision: PeakSupervision, active, margin=PEAK_MARGIN):
    """Mean over eligible active slots; row-major first max tie gets gradient.

    Positive full masks without center support retain dense supervision and
    skip only this term. Full-grid positives have no negative center: likewise
    skip ranking. Every active empty full mask gets softplus(max logit).
    """
    _validate(logits, supervision, active)
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("Peak margin must be positive and finite")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        values = logits.float().flatten(2)
        centers = supervision.center_support.flatten(2)
        visible = active & supervision.full_visible
        empty = active & ~supervision.full_visible
        has_positive, has_negative = centers.any(-1), (~centers).any(-1)
        ranking = visible & has_positive & has_negative
        unrepresentable = visible & ~has_positive
        no_outside = visible & has_positive & ~has_negative
        # Indexed reductions avoid evaluating -inf - (-inf) in skipped rows.
        if bool(ranking.any()):
            selected, mask = values[ranking], centers[ranking]
            positive = selected.masked_fill(~mask, -torch.inf).max(-1).values
            negative = selected.masked_fill(mask, -torch.inf).max(-1).values
            ranking_terms = F.relu(margin + negative - positive)
        else:
            ranking_terms = values.new_empty(0)
        empty_terms = F.softplus(values[empty].max(-1).values) if bool(empty.any()) else values.new_empty(0)
        eligible = ranking_terms.numel() + empty_terms.numel()
        # Multiply each element before reduction so an inactive large map can't
        # overflow a zero-connected loss. No detached or nonfinite zero term.
        zero = (values * 0.0).sum()
        loss = (ranking_terms.sum() + empty_terms.sum()) / eligible if eligible else zero
        return {"peak": loss,
            "peak_visible_mean": ranking_terms.mean() if ranking_terms.numel() else zero,
            "peak_empty_mean": empty_terms.mean() if empty_terms.numel() else zero,
            "peak_eligible_count": eligible, "peak_visible_count": int(ranking.sum()),
            "peak_empty_count": int(empty.sum()),
            "peak_skipped_unrepresentable_count": int(unrepresentable.sum()),
            "peak_skipped_no_outside_count": int(no_outside.sum())}


def anchor_peak_loss(logits, supervision: PeakSupervision, active, arm: str):
    """C1 exactly reuses v1; P1/P2 add the same FP32 peak auxiliary term."""
    if arm not in {"C1", "P1", "P2"}:
        raise ValueError("Expected locked v2 arm C1/P1/P2")
    _validate(logits, supervision, active)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        base = anchor_shadow_loss(logits, supervision.area_masks, active)
        if arm == "C1":
            return base
        peak = peak_auxiliary_loss(logits, supervision, active)
        return {**base, "base_total": base["total"], **peak,
                "total": base["total"] + PEAK_WEIGHT * peak["peak"]}


@dataclass(frozen=True)
class PeakArm:
    name: str
    branch: AnchorShadow

    def loss(self, logits, supervision, active):
        return anchor_peak_loss(logits, supervision, active, self.name)


def prepare_peak_arms(baseline) -> dict[str, PeakArm]:
    """No optimizer creation, no update; all arms share the same frozen model."""
    return {name: PeakArm(name, AnchorShadow(baseline, conditioning, PROTOCOL.seed))
            for name, conditioning in (("C1", "phrase"), ("P1", "phrase"), ("P2", "whole_text"))}


def locked_peak_config() -> dict:
    return {"version": PEAK_VERSION, "pilot": asdict(PROTOCOL),
        "arms": {"C1": {"conditioning": "phrase", "peak_weight": 0.0},
                 "P1": {"conditioning": "phrase", "peak_weight": PEAK_WEIGHT},
                 "P2": {"conditioning": "whole_text", "peak_weight": PEAK_WEIGHT}},
        "objective": {"margin": PEAK_MARGIN, "base_empty_weight": PROTOCOL.empty_weight,
            "mean_over": "eligible active slots across visible-ranking and empty-peak cases",
            "max_ties": "first flattened row-major index; torch.max(dim=-1)",
            "visible_no_center": "skip auxiliary only; preserve dense BCE+Dice, sample, family and denominator",
            "visible_no_outside": "skip auxiliary only; preserve dense BCE+Dice",
            "empty": "full-mask pixels == 0; softplus(max logit) plus existing mean BCE",
            "grid_shape": [24, 32], "image_shape": [480, 640], "loss_dtype": "FP32"},
        "selection": {"split": "dev", "tuple": ["visible_anchor_hits", "matched_swap_both_hits", "negative_empty_mean_sigmoid_max"],
            "ties": "earliest epoch", "checkpoint_for_gates": "selected dev best only"},
        "gates": {"visible_hits": 49, "visible_denominator": 61, "swap_both_hits": 8, "swap_denominator": 15,
            "found_both_hits": 12, "found_denominator": 16, "empty_each_tolerance": 1e-6,
            "phrase_extra_hits_or_pairs": 2, "phrase_other_metric_nonregression": True,
            "median_sidecar_overhead_max_fraction": .20, "bootstrap_resamples": 5000, "bootstrap_seed": PROTOCOL.seed},
        "control_comparisons": ["P1_vs_C1: objective improvement; other binding metric nonregression",
            "P1_vs_P2: phrase-specific benefit under matched objective"],
        "scope": "development train/dev only; no verifier, calibration samples, IID/OOD, MC or robot",
        "optimizer_authorized_in_preflight": False, "epoch_orchestration_implemented": False}


def validate_locked_peak_config(config: Mapping):
    # JSON normalizes tuple prefixes to a list.
    import json
    expected = json.loads(json.dumps(locked_peak_config()))
    if config != expected:
        raise ValueError("Config drift from locked peak protocol; create a new revision before training")
