"""Peak-event gradients and mask/boundary contracts; no optimizer or train."""
from dataclasses import replace
import math

import numpy as np
import torch
import torch.nn.functional as F

from pcrau.anchor_peak_experiment import PeakSupervision, anchor_peak_loss, peak_auxiliary_loss, prepare_peak_arms, supervision_from_full_masks
from pcrau.anchor_shadow_experiment import anchor_shadow_loss, state_digest


def supervision(centers, visible=None):
    centers = torch.tensor(centers, dtype=torch.bool).reshape(1, 1, 2, 3)
    visible = bool(centers.any()) if visible is None else visible
    area = centers.float() if centers.any() else torch.full_like(centers, .01, dtype=torch.float32) if visible else centers.float()
    return PeakSupervision(area, centers, torch.tensor([[visible]]), torch.tensor([[8 if visible else 0]]))


def test_wrong_winning_peak_gradient_lowers_wrong_and_raises_correct():
    logits = torch.tensor([[[[0., 4., -1.], [-2., -3., -4.]]]], requires_grad=True)
    sup = supervision([1, 0, 0, 0, 0, 0])
    loss = peak_auxiliary_loss(logits, sup, torch.tensor([[True]]))
    assert loss["peak"].item() == 5.
    loss["peak"].backward()
    assert logits.grad.flatten().tolist() == [-1., 1., 0., 0., 0., 0.]


def test_satisfied_margin_has_zero_gradient():
    logits = torch.tensor([[[[5., 2., -1.], [-2., -3., -4.]]]], requires_grad=True)
    loss = peak_auxiliary_loss(logits, supervision([1, 0, 0, 0, 0, 0]), torch.tensor([[True]]))
    loss["peak"].backward()
    assert loss["peak"].item() == 0 and logits.grad.abs().sum() == 0


def test_empty_peak_softplus_is_stable_and_only_peak_gets_aux_gradient():
    for maximum in (1000., -1000., 3.):
        logits = torch.tensor([[[[maximum, maximum-1, maximum-2], [maximum-3, maximum-4, maximum-5]]]], requires_grad=True)
        loss = peak_auxiliary_loss(logits, supervision([0]*6, False), torch.tensor([[True]]))
        torch.testing.assert_close(loss["peak"], F.softplus(torch.tensor(maximum)))
        loss["peak"].backward()
        assert torch.isfinite(loss["peak"]) and torch.isfinite(logits.grad).all()
        assert logits.grad.flatten()[1:].abs().sum() == 0
        if maximum == 1000.: assert logits.grad.flatten()[0] == 1


def test_tiny_visible_mask_is_not_empty_and_retains_dense_gradient():
    full = np.zeros((1, 1, 480, 640), dtype=bool); full[0, 0, 0, :8] = True
    sup = supervision_from_full_masks(full)
    assert sup.full_pixel_counts.item() == 8 and sup.full_visible.item() and not sup.center_support.any()
    assert sup.area_masks.sum() > 0
    logits = torch.zeros(1, 1, 24, 32, requires_grad=True); active = torch.tensor([[True]])
    losses = anchor_peak_loss(logits, sup, active, "P1")
    assert losses["peak_skipped_unrepresentable_count"] == 1 and losses["peak_empty_count"] == 0
    assert losses["visible_count"] == 1 and losses["empty_count"] == 0
    assert losses["peak"].item() == 0
    losses["total"].backward()
    assert logits.grad[0, 0, 0, 0] != 0 and torch.isfinite(logits.grad).all()


def test_real_pixel_center_is_distinct_from_cell_overlap():
    full = np.zeros((1, 1, 480, 640), dtype=bool); full[0, 0, 10, 11] = True
    a = supervision_from_full_masks(full)
    assert a.area_masks[0, 0, 0, 0] > 0 and not a.center_support[0, 0, 0, 0]
    full[0, 0, 10, 10] = True
    b = supervision_from_full_masks(full)
    assert b.center_support[0, 0, 0, 0] and b.full_pixel_counts.item() == 2


def test_inactive_and_no_eligible_batches_have_zero_auxiliary():
    logits = torch.ones(1, 1, 2, 3, requires_grad=True)
    for sup in (supervision([1, 0, 0, 0, 0, 0]), supervision([0]*6, False)):
        loss = peak_auxiliary_loss(logits, sup, torch.tensor([[False]]))
        loss["peak"].backward(); assert loss["peak"].item() == 0 and logits.grad.abs().sum() == 0
    whole = peak_auxiliary_loss(logits, supervision([1]*6), torch.tensor([[True]]))
    assert whole["peak_skipped_no_outside_count"] == 1 and whole["peak_eligible_count"] == 0
    whole["peak"].backward(); assert torch.isfinite(logits.grad).all()


def test_eligible_mean_is_over_slots_not_separate_group_means():
    logits = torch.tensor([[[[0., 4., -1.], [-2., -3., -4.]]], [[[3., -1., -2.], [-3., -4., -5.]]]], requires_grad=True)
    a, b = supervision([1, 0, 0, 0, 0, 0]), supervision([0]*6, False)
    sup = PeakSupervision(*(torch.cat((getattr(a, k), getattr(b, k))) for k in ("area_masks", "center_support", "full_visible", "full_pixel_counts")))
    loss = peak_auxiliary_loss(logits, sup, torch.ones(2, 1, dtype=torch.bool))
    torch.testing.assert_close(loss["peak"], (torch.tensor(5.)+F.softplus(torch.tensor(3.)))/2)
    assert loss["peak_eligible_count"] == 2 and loss["peak_visible_count"] == loss["peak_empty_count"] == 1
    loss["peak"].backward(); assert logits.grad[0, 0, 0, 0] == -.5


def test_tie_max_gradient_uses_first_row_major_cell():
    logits = torch.tensor([[[[4., 4., 4.], [4., -3., -4.]]]], requires_grad=True)
    loss = peak_auxiliary_loss(logits, supervision([1, 1, 0, 0, 0, 0]), torch.tensor([[True]]))
    loss["peak"].backward()
    assert logits.grad.flatten().tolist() == [-1., 0., 1., 0., 0., 0.]


def test_c1_total_and_gradient_exactly_match_v1():
    logits = torch.tensor([[[[0., 4., -1.], [-2., -3., -4.]]]], requires_grad=True)
    sup = supervision([1, 0, 0, 0, 0, 0]); active = torch.tensor([[True]])
    old = anchor_shadow_loss(logits, sup.area_masks, active)
    new = anchor_peak_loss(logits, sup, active, "C1")
    assert old.keys() == new.keys()
    for k in old: assert torch.equal(old[k], new[k]) if isinstance(old[k], torch.Tensor) else old[k] == new[k]
    torch.testing.assert_close(torch.autograd.grad(old["total"], logits)[0], torch.autograd.grad(new["total"], logits)[0], rtol=0, atol=0)
    p1, p2 = anchor_peak_loss(logits, sup, active, "P1"), anchor_peak_loss(logits, sup, active, "P2")
    assert torch.equal(p1["total"], p2["total"])
    torch.testing.assert_close(p1["total"], new["total"]+.1*p1["peak"])


def test_invalid_full_visibility_and_nonfinite_inputs_rejected():
    logits = torch.zeros(1, 1, 2, 3); good = supervision([1, 0, 0, 0, 0, 0]); active = torch.tensor([[True]])
    bad = replace(good, full_visible=torch.tensor([[False]]))
    for data, sup in ((logits, bad), (torch.full_like(logits, float('nan')), good)):
        try: peak_auxiliary_loss(data, sup, active)
        except ValueError: pass
        else: raise AssertionError("Malformed supervision/logits accepted")


def test_prepared_arms_zero_init_oracle_rejection_and_baseline_freeze():
    import runpy
    from pathlib import Path
    fixture = runpy.run_path(str(Path(__file__).with_name("test_anchor_shadow.py")))
    model, batch, prompts = fixture["setup"](); before = state_digest(model.state_dict())
    arms = prepare_peak_arms(model)
    assert len({state_digest(a.branch.residual.state_dict()) for a in arms.values()}) == 1
    for name, arm in arms.items():
        assert sum(p.numel() for p in arm.branch.parameters() if p.requires_grad) == 33024
        cap = arm.branch.capture(batch, prompts); predicted = arm.branch.predict(cap)
        torch.testing.assert_close(predicted, cap.baseline["anchor_logits"], rtol=0, atol=1e-6)
        for key in ("anchor_center_support", "full_visible", "full_pixel_counts", "anchor_full_mask"):
            try: arm.branch.capture(dict(batch, **{key: torch.zeros(1)}), prompts)
            except ValueError: pass
            else: raise AssertionError("Oracle input accepted")
        assert state_digest(model.state_dict()) == before and all(p.grad is None for p in model.parameters())
