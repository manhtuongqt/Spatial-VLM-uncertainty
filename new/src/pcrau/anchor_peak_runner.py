"""Authorized v2 epoch runner; immutable preflight objective/config reused."""
from __future__ import annotations

import math
from typing import Mapping

import torch

from .anchor_peak_experiment import PeakArm, PeakSupervision, validate_locked_peak_config
from .anchor_shadow import FrozenAnchorCapture
from .anchor_shadow_experiment import PROTOCOL, ShadowPilotRunner, family_batches, state_digest


class PeakPilotRunner(ShadowPilotRunner):
    """Reuse v1 selection state machine, replace optimizer receipt and loss."""

    def __init__(self, arm: PeakArm):
        super().__init__(arm.branch)
        self.arm = arm

    def start_optimizer(self, preflight: Mapping, execution: Mapping):
        if preflight.get("status") != "S1_PEAK_V2_PREFLIGHT_PASS" or not preflight.get("technical_gates_pass"):
            raise ValueError("A passing v2 technical preflight is required")
        if not execution.get("pre_optimizer_checks_pass") or not execution.get("training_authorized"):
            raise ValueError("Checked execution lock and explicit pilot authorization required")
        validate_locked_peak_config(execution["config"])
        for field in ("config_sha256", "protocol_sha256"):
            if execution.get(field) != preflight.get(field):
                raise ValueError(f"Preflight/execution lock mismatch: {field}")
        if self.optimizer is not None:
            raise RuntimeError("Optimizer already initialized")
        self.branch._check_frozen()
        if state_digest(self.branch.residual.state_dict()) != preflight["initialization_sha256"][self.arm.name]:
            raise ValueError("Arm does not match locked zero-output initialization")
        self.frozen_digest = state_digest(self.branch.baseline.state_dict())
        if self.frozen_digest != preflight["frozen_model_state_sha256"]:
            raise ValueError("Baseline changed since preflight")
        if sum(p.numel() for p in self.branch.parameters() if p.requires_grad) != 33024:
            raise ValueError("Trainable set changed")
        self.optimizer = torch.optim.AdamW(self.branch.residual.parameters(),
            lr=PROTOCOL.learning_rate, weight_decay=PROTOCOL.weight_decay)

    def run_epoch(self, epoch, presentations, prepare_batch):
        if (self.optimizer is None or epoch != self.next_epoch or self.stale_epochs >= PROTOCOL.patience
                or self.pending_selection_epoch is not None):
            raise RuntimeError("Optimizer/epoch/early-stop gate not satisfied")
        if state_digest(self.branch.baseline.state_dict()) != self.frozen_digest:
            raise RuntimeError("Frozen baseline changed before epoch")
        self.branch.train()
        history = []
        for indices in family_batches(presentations, epoch):
            capture, supervision = prepare_batch([presentations[i] for i in indices])
            if not isinstance(capture, FrozenAnchorCapture) or not isinstance(supervision, PeakSupervision):
                raise TypeError("Observable capture and separate peak supervision required")
            self.optimizer.zero_grad(set_to_none=True)
            losses = self.arm.loss(self.branch.predict(capture), supervision, capture.active_slots)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Nonfinite v2 objective")
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(self.branch.residual.parameters(),
                PROTOCOL.gradient_clip_norm, error_if_nonfinite=True)
            self.optimizer.step()
            self.optimizer_steps += 1
            record = {k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
            if not all(math.isfinite(v) for v in record.values()):
                raise RuntimeError("Nonfinite batch diagnostic")
            history.append({**record, "gradient_norm_before_clip": float(norm)})
        self.next_epoch += 1
        self.pending_selection_epoch = epoch
        self.branch._check_frozen()
        if state_digest(self.branch.baseline.state_dict()) != self.frozen_digest:
            raise RuntimeError("Frozen baseline changed during epoch")
        return history
