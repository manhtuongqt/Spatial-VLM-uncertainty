from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .dataset import SOURCE_CLASSES


def masked_location_mass_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Negative log spatial-softmax mass on nonempty target-mask support.

    The archived area-resized mask is binarized with > 0, matching the
    evaluator's target_probability_mass event. Empty/invalid targets do not
    supervise localization, including ABSENT samples. Compute in FP32 for AMP.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("location_mass_epsilon must be positive and finite")
    if logits.ndim != 3 or logits.shape != targets.shape:
        raise ValueError("location mass expects matching [batch, height, width] tensors")
    if valid.shape != logits.shape[:1]:
        raise ValueError("location mass expects one validity flag per sample")
    support = targets > 0
    active = valid.bool() & support.flatten(1).any(-1)
    if not bool(active.any()):
        return logits.float().sum() * 0.0
    probability = logits[active].float().flatten(1).softmax(-1)
    mass = (probability * support[active].flatten(1)).sum(-1)
    return -(mass + epsilon).log().mean()


def masked_heatmap_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim == 3:
        logits, targets, valid = logits[:, None], targets[:, None], valid[:, None]
    valid = valid.bool()
    if not bool(valid.any()):
        zero = logits.sum() * 0.0
        return zero, zero, zero
    chosen_logits = logits[valid]
    chosen_targets = targets[valid]
    positive_fraction = chosen_targets.mean((-2, -1)).clamp_min(1e-6)
    positive_weight = ((1.0 - positive_fraction) / positive_fraction).clamp(1.0, 20.0)
    bce = -(
        positive_weight[:, None, None] * chosen_targets * F.logsigmoid(chosen_logits)
        + (1.0 - chosen_targets) * F.logsigmoid(-chosen_logits)
    ).mean()
    probability = torch.sigmoid(chosen_logits)
    intersection = (probability * chosen_targets).sum((-2, -1))
    dice = 1.0 - (
        (2.0 * intersection + 1.0)
        / (probability.sum((-2, -1)) + chosen_targets.sum((-2, -1)) + 1.0)
    ).mean()
    return bce + dice, bce, dice


def family_ranking_loss(
    source_logits: torch.Tensor,
    families: list[str],
    variants: list[str],
    margin: float,
) -> torch.Tensor:
    by_family: dict[str, dict[str, int]] = defaultdict(dict)
    for index, (family, variant) in enumerate(zip(families, variants)):
        by_family[family][variant] = index
    source_for_variant = {
        "semantic_counterfactual": "semantic",
        "relation_counterfactual": "relation",
        "depth_corruption": "depth",
        "occlusion_view_counterfactual": "occlusion",
    }
    terms = []
    for members in by_family.values():
        if "clean" not in members:
            continue
        clean = members["clean"]
        for variant, source in source_for_variant.items():
            if variant in members:
                source_index = SOURCE_CLASSES.index(source)
                delta = source_logits[members[variant], source_index] - source_logits[clean, source_index]
                terms.append(F.relu(source_logits.new_tensor(margin) - delta))
    return torch.stack(terms).mean() if terms else source_logits.sum() * 0.0


def compute_losses(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
    answer_weights: torch.Tensor | None = None,
    source_pos_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    target, target_bce, target_dice = masked_heatmap_loss(
        output["target_logits"], batch["target_heatmap"].to(output["target_logits"].device),
        batch["target_loss_mask"].to(output["target_logits"].device),
    )
    interior, _, _ = masked_heatmap_loss(
        output["interior_logits"], batch["interior_heatmap"].to(output["interior_logits"].device),
        batch["interior_loss_mask"].to(output["interior_logits"].device),
    )
    anchor, _, _ = masked_heatmap_loss(
        output["anchor_logits"], batch["anchor_heatmaps"].to(output["anchor_logits"].device),
        batch["anchor_supervision_mask"].to(output["anchor_logits"].device),
    )
    edge_mask = batch["edge_mask"].to(output["relation_edge_logits"].device)
    if bool(edge_mask.any()):
        relation_edge = F.binary_cross_entropy_with_logits(
            output["relation_edge_logits"][edge_mask], batch["edge_target"].to(output["relation_edge_logits"].device)[edge_mask]
        )
    else:
        relation_edge = output["relation_edge_logits"].sum() * 0.0
    answer = F.cross_entropy(
        output["answerability_logits"], batch["answer_target"].to(output["answerability_logits"].device),
        weight=answer_weights,
    )
    source_targets = batch["source_target"].to(output["source_logits"].device)
    if source_pos_weights is None:
        source_pos_weights = torch.ones(len(SOURCE_CLASSES), device=output["source_logits"].device)
    positive_scale_cfg = config["loss"].get("source_positive_weight_scale", {})
    negative_weight_cfg = config["loss"].get("source_negative_weight", {})
    positive_scale = output["source_logits"].new_tensor(
        [float(positive_scale_cfg.get(name, 1.0)) for name in SOURCE_CLASSES]
    )
    negative_weight = output["source_logits"].new_tensor(
        [float(negative_weight_cfg.get(name, 1.0)) for name in SOURCE_CLASSES]
    )
    adjusted_positive = source_pos_weights * positive_scale
    source_per_element = -(
        adjusted_positive[None] * source_targets * F.logsigmoid(output["source_logits"])
        + negative_weight[None] * (1.0 - source_targets) * F.logsigmoid(-output["source_logits"])
    )
    source_weights = torch.ones_like(source_per_element)
    source_weights[:, SOURCE_CLASSES.index("spatial")] *= float(config["loss"]["spatial_weak_label_weight"])
    source = (source_per_element * source_weights).sum() / source_weights.sum().clamp_min(1.0)
    ranking = family_ranking_loss(
        output["source_logits"], batch["family_id"], batch["variant"],
        float(config["loss"]["ranking_margin"]),
    )
    weights = config["loss"]
    location_weight = float(weights.get("location_mass", 0.0))
    if not math.isfinite(location_weight) or location_weight < 0:
        raise ValueError("location_mass weight must be nonnegative and finite")
    location_mass = output["target_logits"].float().sum() * 0.0
    if location_weight > 0:
        location_mass = masked_location_mass_loss(
            output["target_logits"], batch["target_heatmap"].to(output["target_logits"].device),
            batch["target_loss_mask"].to(output["target_logits"].device),
            epsilon=float(weights.get("location_mass_epsilon", 1e-8)),
        )
    total = (
        float(weights["target_heatmap"]) * target
        + float(weights["interior_heatmap"]) * interior
        + float(weights["anchor_heatmap"]) * anchor
        + float(weights["relation_edge"]) * relation_edge
        + float(weights["answerability"]) * answer
        + float(weights["source"]) * source
        + float(weights["counterfactual_ranking"]) * ranking
    )
    # Disabled by default: preserve the historical V2 objective exactly.
    if location_weight > 0:
        total = total + location_weight * location_mass
    return {
        "total": total,
        "target": target,
        "target_bce": target_bce,
        "target_dice": target_dice,
        "location_mass": location_mass,
        "interior": interior,
        "anchor": anchor,
        "relation_edge": relation_edge,
        "answerability": answer,
        "source": source,
        "ranking": ranking,
    }


def class_weights(dataset: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    answers = torch.zeros(4, dtype=torch.float64)
    sources = torch.zeros(len(SOURCE_CLASSES), dtype=torch.float64)
    for entry in dataset.entries:
        supervision = entry["supervision"]
        answers[int(supervision["answerability_index"])] += 1
        for source in supervision["source_labels"]:
            if source in SOURCE_CLASSES:
                sources[SOURCE_CLASSES.index(source)] += 1
        if supervision["answerability_state"] == "AMBIGUOUS":
            sources[SOURCE_CLASSES.index("spatial")] += 1
    answer_weights = answers.sum() / (len(answers) * answers.clamp_min(1.0))
    source_weights = ((len(dataset) - sources) / sources.clamp_min(1.0)).clamp(1.0, 20.0)
    return answer_weights.float().to(device), source_weights.float().to(device)
