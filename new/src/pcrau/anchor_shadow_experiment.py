"""Pilot preparation for M1/M2: protocol, family batches, separate loss/optimizer.

No training is invoked on import or construction. Preflight uses backward only;
the explicit optimizer methods below are for a subsequently authorized pilot.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .anchor_shadow import AnchorShadow, FrozenAnchorCapture
from .language_augmentation import PREFIXES, validate_prefix
from .losses import masked_heatmap_loss
from .query_parser import parse_query
from .text import prompt_anchor_mask, relation_ids, tokenize


@dataclass(frozen=True)
class PilotProtocol:
    version: str = "pcrau_anchor_shadow_pilot_v1"
    seed: int = 24082026
    families_per_batch: int = 4
    max_epochs: int = 15
    patience: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 1.0
    empty_weight: float = 0.25
    prefixes: tuple[str, ...] = PREFIXES


PROTOCOL = PilotProtocol()


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{key}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def pilot_presentations(entries: Sequence[Mapping], cfg: Mapping, split: str) -> list[dict]:
    if split not in {"train", "dev"}:
        raise ValueError("Pilot is restricted to development train/dev")
    selected = []
    for entry in entries:
        if entry["split"] != split:
            continue
        prompt = entry["feature_input"]["prompt"]
        q = parse_query(prompt, int(cfg["max_tokens"]), int(cfg["max_anchors"]))
        if not q.supported or q.predicate not in {"left_of", "right_of"}:
            continue
        for index, prefix in enumerate(PROTOCOL.prefixes if split == "train" else ("",)):
            selected.append({"entry": entry, "prompt": validate_prefix(prompt, prefix, cfg),
                             "family_id": entry["family_id"], "sample_id": entry["sample_id"], "prefix_index": index})
    return sorted(selected, key=lambda e: (e["family_id"], e["sample_id"], e["prefix_index"]))


def family_batches(presentations: Sequence[Mapping], epoch: int) -> list[list[int]]:
    if not 0 <= epoch < PROTOCOL.max_epochs:
        raise ValueError("Epoch outside locked pilot budget")
    groups = defaultdict(list)
    for index, item in enumerate(presentations):
        if item["entry"]["split"] != "train":
            raise ValueError("Training batches cannot contain dev samples")
        groups[item["family_id"]].append(index)
    families = sorted(groups)
    order = torch.randperm(len(families), generator=torch.Generator().manual_seed(PROTOCOL.seed + epoch)).tolist()
    shuffled = [families[i] for i in order]
    return [[i for f in shuffled[start:start + PROTOCOL.families_per_batch] for i in groups[f]]
            for start in range(0, len(shuffled), PROTOCOL.families_per_batch)]


def observable_batch(dataset, entries: Sequence[Mapping], prompts: Sequence[str], device: torch.device) -> dict:
    """Load cached observable features only; never dataset.__getitem__ masks."""
    if len(entries) != len(prompts) or not entries:
        raise ValueError("Expected nonempty matched entries/prompts")
    samples = []
    cfg = dataset.model_cfg
    for entry, prompt in zip(entries, prompts):
        if entry["split"] != dataset.split or dataset.profile != "development":
            raise ValueError("Observable pilot loader is restricted to its development split")
        feature = dataset._feature(entry)
        ids, mask = tokenize(prompt, int(cfg["max_tokens"]), int(cfg["vocab_size"]))
        rid, rmask = relation_ids(prompt, int(cfg["max_relations"]))
        samples.append({"r0": feature["R0_GRID"].float(), "d0": feature["D0_GRID"].float(),
            "r_thumb": feature["R0_THUMB"].float(), "d_thumb": feature["D0_THUMB"].float(),
            "token_ids": torch.tensor(ids, dtype=torch.long), "token_mask": torch.tensor(mask, dtype=torch.bool),
            "relation_ids": torch.tensor(rid, dtype=torch.long), "relation_mask": torch.tensor(rmask, dtype=torch.bool),
            "anchor_mask": torch.tensor(prompt_anchor_mask(prompt, int(cfg["max_anchors"])), dtype=torch.bool)})
    return {key: torch.stack([s[key] for s in samples]).to(device) for key in samples[0]}


def anchor_shadow_loss(logits: torch.Tensor, targets: torch.Tensor, active: torch.Tensor) -> dict:
    """Evaluator/train masks only; zero-mask active slots explicitly supervised."""
    if logits.ndim != 4 or logits.shape != targets.shape or active.shape != logits.shape[:2]:
        raise ValueError("Expected matching [B,S,H,W] logits/targets and [B,S] activation")
    if active.dtype != torch.bool or not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("Invalid loss inputs")
    if bool(((targets < 0) | (targets > 1)).any()):
        raise ValueError("Targets must be fractional mask occupancy in [0,1]")
    logits, targets = logits.float(), targets.float()
    nonempty = targets.sum((-2, -1)) > 0
    visible, empty = active & nonempty, active & ~nonempty
    with torch.autocast(device_type=logits.device.type, enabled=False):
        loss, bce, dice = masked_heatmap_loss(logits, targets, visible)
        negative = F.binary_cross_entropy_with_logits(logits[empty], torch.zeros_like(logits[empty])) if bool(empty.any()) else logits.sum() * 0.0
    return {"total": loss + PROTOCOL.empty_weight * negative, "visible": loss, "visible_bce": bce,
            "visible_dice": dice, "empty_bce": negative, "visible_count": int(visible.sum()), "empty_count": int(empty.sum())}


class ShadowPilotRunner:
    """One experimental arm. Optimizer is lazy and requires a passing preflight.

    run_epoch consumes external observable captures plus separate supervision;
    it enforces the same family/prefix plan and epoch budget for each arm. This
    preparation does not include model promotion, calibration or test runners.
    """
    def __init__(self, branch: AnchorShadow):
        self.branch = branch
        self.optimizer = None
        self.optimizer_steps = 0
        self.next_epoch = 0
        self.best_selection = None
        self.best_epoch = None
        self.stale_epochs = 0
        self.pending_selection_epoch = None
        self.frozen_digest = None

    def start_optimizer(self, preflight: Mapping):
        if preflight.get("status") != "S1_SHADOW_PREFLIGHT_PASS" or not preflight.get("technical_gates_pass"):
            raise ValueError("A passing technical preflight is required before optimizer creation")
        if self.optimizer is not None:
            raise RuntimeError("Optimizer already initialized")
        self.branch._check_frozen()
        arm = "M1" if self.branch.conditioning == "phrase" else "M2"
        if preflight.get("initialization_sha256", {}).get(arm) != state_digest(self.branch.residual.state_dict()):
            raise ValueError("Preflight initialization does not match this arm")
        digest = state_digest(self.branch.baseline.state_dict())
        if preflight.get("frozen_model_state_sha256") != digest:
            raise ValueError("Preflight baseline state does not match this model")
        self.frozen_digest = digest
        self.optimizer = torch.optim.AdamW(self.branch.residual.parameters(), lr=PROTOCOL.learning_rate, weight_decay=PROTOCOL.weight_decay)

    def run_epoch(self, epoch: int, presentations: Sequence[Mapping], prepare_batch):
        """prepare_batch(items) -> (capture, supervision_masks); caller chooses device/AMP."""
        if (self.optimizer is None or epoch != self.next_epoch or self.stale_epochs >= PROTOCOL.patience
                or self.pending_selection_epoch is not None):
            raise RuntimeError("Optimizer/epoch/early-stop gate not satisfied")
        if self.frozen_digest != state_digest(self.branch.baseline.state_dict()):
            raise RuntimeError("Frozen baseline state changed since preflight")
        self.branch.train()
        history = []
        for indices in family_batches(presentations, epoch):
            capture, masks = prepare_batch([presentations[i] for i in indices])
            if not isinstance(capture, FrozenAnchorCapture):
                raise TypeError("Expected observable frozen capture")
            self.optimizer.zero_grad(set_to_none=True)
            losses = anchor_shadow_loss(self.branch.predict(capture), masks, capture.active_slots)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.branch.residual.parameters(), PROTOCOL.gradient_clip_norm, error_if_nonfinite=True)
            self.optimizer.step()
            self.optimizer_steps += 1
            history.append({k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in losses.items()})
        self.next_epoch += 1
        self.pending_selection_epoch = epoch
        if self.frozen_digest != state_digest(self.branch.baseline.state_dict()):
            raise RuntimeError("Frozen baseline state changed during pilot epoch")
        return history

    def record_dev_selection(self, epoch: int, visible_hits: int, swap_both_hits: int, empty_mean_sigmoid_max: float) -> bool:
        if epoch != self.next_epoch - 1 or epoch != self.pending_selection_epoch:
            raise ValueError("Selection must follow the current epoch")
        if (not 0 <= visible_hits <= 61 or not 0 <= swap_both_hits <= 15
                or not math.isfinite(empty_mean_sigmoid_max) or not 0 <= empty_mean_sigmoid_max <= 1):
            raise ValueError("Invalid locked-scope dev metrics")
        selection = (visible_hits, swap_both_hits, -empty_mean_sigmoid_max)
        improved = self.best_selection is None or selection > self.best_selection
        if improved:
            self.best_selection, self.best_epoch, self.stale_epochs = selection, epoch, 0
        else:
            self.stale_epochs += 1
        self.pending_selection_epoch = None
        return improved

    def protocol_dict(self):
        return asdict(PROTOCOL)
