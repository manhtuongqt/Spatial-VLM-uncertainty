from __future__ import annotations

from copy import deepcopy
import math

import torch

from pcrau.dataset import ArchivedPCRAUDataset, collate_samples, move_model_batch
from pcrau.losses import compute_losses, masked_location_mass_loss
from pcrau.model import PCRAUTargetV2
from pcrau.utils import load_config


def test_forward_shapes_and_oracle_boundary() -> None:
    config = load_config()
    dataset = ArchivedPCRAUDataset(config, "train")
    batch = collate_samples([dataset[0]])
    model = PCRAUTargetV2(config).eval()
    with torch.no_grad():
        output = model(move_model_batch(batch, torch.device("cpu")))
    assert output["target_logits"].shape == (1, 24, 32)
    assert output["anchor_logits"].shape == (1, 3, 24, 32)
    assert output["relation_edge_logits"].shape == (1, 3)
    assert output["answerability_logits"].shape == (1, 4)
    assert output["source_logits"].shape == (1, 5)
    losses = compute_losses(output, batch, config)
    assert torch.isfinite(losses["total"])
    mass_config = deepcopy(config)
    mass_config["loss"]["location_mass"] = 0.7
    with_mass = compute_losses(output, batch, mass_config)
    torch.testing.assert_close(
        with_mass["total"], losses["total"] + 0.7 * with_mass["location_mass"],
    )
    legacy_total = sum(
        config["loss"][weight] * losses[name]
        for weight, name in [
            ("target_heatmap", "target"), ("interior_heatmap", "interior"),
            ("anchor_heatmap", "anchor"), ("relation_edge", "relation_edge"),
            ("answerability", "answerability"), ("source", "source"),
            ("counterfactual_ranking", "ranking"),
        ]
    )
    assert torch.equal(losses["total"], legacy_total)
    assert losses["location_mass"].item() == 0.0
    unsafe = move_model_batch(batch, torch.device("cpu"))
    unsafe["target_heatmap"] = batch["target_heatmap"]
    try:
        model(unsafe)
    except ValueError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("Oracle/evaluator tensor crossed the model input boundary")


def test_location_mass_formula_and_gradient_ignore_empty_invalid_masks() -> None:
    logits = torch.zeros(3, 2, 2, requires_grad=True)
    masks = torch.zeros_like(logits)
    # Fractional edge cells count as mask support, not fractional probability.
    masks[0, 0, 0] = 0.05
    masks[2, 0, 0] = 1.0
    loss = masked_location_mass_loss(logits, masks, torch.tensor([True, True, False]))
    torch.testing.assert_close(loss, torch.tensor(-math.log(0.25 + 1e-8)))
    loss.backward()
    assert logits.grad[0, 0, 0] < 0
    assert bool((logits.grad[0].flatten()[1:] > 0).all())
    assert torch.equal(logits.grad[1:], torch.zeros_like(logits.grad[1:]))
    assert torch.isfinite(logits.grad).all()


def test_location_mass_prefers_target_probability_and_handles_empty_batch() -> None:
    masks = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    good = torch.tensor([[[3.0, 0.0], [0.0, 0.0]]])
    bad = torch.tensor([[[0.0, 3.0], [0.0, 0.0]]])
    valid = torch.tensor([True])
    assert masked_location_mass_loss(good, masks, valid) < masked_location_mass_loss(bad, masks, valid)
    empty_logits = torch.zeros(2, 2, 2, requires_grad=True)
    zero = masked_location_mass_loss(empty_logits, torch.zeros_like(empty_logits), torch.ones(2, dtype=torch.bool))
    assert zero.item() == 0.0
    zero.backward()
    assert torch.equal(empty_logits.grad, torch.zeros_like(empty_logits))


def test_location_mass_is_finite_in_bfloat16_and_shift_invariant() -> None:
    logits = torch.tensor([[[1000.0, -1000.0], [-1000.0, -1000.0]]], dtype=torch.bfloat16, requires_grad=True)
    mask = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    loss = masked_location_mass_loss(logits, mask, torch.tensor([True]))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    ordinary = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    torch.testing.assert_close(
        masked_location_mass_loss(ordinary, mask, torch.tensor([True])),
        masked_location_mass_loss(ordinary + 10.0, mask, torch.tensor([True])),
    )
