#!/usr/bin/env python3
"""Add-only P-CRA-U V1.1 model, loss, and development evaluator.

The V1 modules remain immutable inputs.  This module reuses their frozen
feature dataset contract and metric helpers, but owns every changed V1.1
behavior: real head dropout, clean-FOUND localization weighting, exact loss
aggregation, and the preregistered development-selection components.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .pcra_u_development_common import (
        ANSWERABILITY_CLASSES,
        SOURCE_CLASSES,
        DevelopmentTrainingError,
        FrozenFeatureDataset,
        chunks,
        confusion_metrics,
        model_inputs,
    )
else:
    from pcra_u_development_common import (  # type: ignore
        ANSWERABILITY_CLASSES,
        SOURCE_CLASSES,
        DevelopmentTrainingError,
        FrozenFeatureDataset,
        chunks,
        confusion_metrics,
        model_inputs,
    )


SELECTION_METRIC = "dev_scientific_score_v1_1"
SELECTION_MODE = "max"
SELECTION_WEIGHTS = {
    "clean_found_raw_point_in_target": 0.45,
    "all_found_raw_point_in_target": 0.20,
    "answerability_macro_f1": 0.20,
    "non_false_found_rate": 0.15,
}


class PCRAUDevelopmentV11(nn.Module):
    """V1 sidecar with dropout inserted after each task-head GELU.

    This is a fresh model definition rather than a mutation of V1.  Dropout
    has no parameters, so parameter initialization order remains aligned with
    the corresponding V1 layers for a given seed while train-time behavior is
    genuinely regularized.
    """

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        hidden = int(config["model"]["hidden_dim"])
        vocab = int(config["language"]["vocab_size"])
        source_count = len(config["model"]["source_classes"])
        dropout = float(config["model"]["dropout"])
        if not 0.0 < dropout < 1.0:
            raise DevelopmentTrainingError(f"V1.1 dropout must be in (0, 1), got {dropout}")

        self.token_embedding = nn.Embedding(vocab, hidden, padding_idx=0)
        self.relation_embedding = nn.Embedding(len(config["language"]["relations"]), hidden)
        self.rgb_projection = nn.Linear(1152, hidden, bias=False)
        self.depth_projection = nn.Linear(1152, hidden, bias=False)
        self.thumbnail_rgb = nn.Linear(1152, hidden, bias=False)
        self.thumbnail_depth = nn.Linear(1152, hidden, bias=False)
        self.gate = nn.Linear(hidden * 3, hidden)
        self.film = nn.Linear(hidden, hidden * 2)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.target_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        global_dim = hidden * 4
        self.answer_head = nn.Sequential(
            nn.Linear(global_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, len(ANSWERABILITY_CLASSES)),
        )
        self.source_head = nn.Sequential(
            nn.Linear(global_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, source_count),
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_mask = batch["tokens"].ne(0).unsqueeze(-1)
        token_values = self.token_embedding(batch["tokens"])
        token_query = (token_values * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        query = token_query + self.relation_embedding(batch["relation_ids"])
        rgb = self.rgb_projection(batch["r0"])
        depth = self.depth_projection(batch["d0"])
        query_grid = query[:, None, None, :].expand_as(rgb)
        gate = torch.sigmoid(self.gate(torch.cat([rgb, depth, query_grid], dim=-1)))
        fused = self.fusion_norm(rgb + gate * depth)
        gamma, beta = self.film(query).chunk(2, dim=-1)
        fused = fused * (1.0 + 0.1 * torch.tanh(gamma[:, None, None, :])) + beta[:, None, None, :]
        heatmap_logits = self.target_head(fused).squeeze(-1)
        thumbnail = self.thumbnail_rgb(batch["r_thumb"]) + self.thumbnail_depth(batch["d_thumb"])
        global_feature = torch.cat([fused.mean((1, 2)), fused.amax((1, 2)), query, thumbnail], dim=-1)
        return {
            "heatmap_logits": heatmap_logits,
            "answerability_logits": self.answer_head(global_feature),
            "source_logits": self.source_head(global_feature),
            "gate_mean": gate.mean(),
        }


def _loss_scalar(value: float, reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(float(value))


def compute_loss_v11(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    answer_weights: torch.Tensor,
    source_weights: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Compute the V1.1 objective and exact cross-batch aggregation terms.

    For each FOUND sample, ``l_i = weighted_BCE_i + soft_Dice_i``.  A clean
    FOUND sample receives weight four and every other FOUND sample receives
    weight one.  The heatmap loss is normalized by the sum of those weights,
    so the multiplier changes relative localization priority without changing
    the component's arbitrary batch scale.
    """

    answer_targets = batch["answer_targets"]
    found = answer_targets.eq(0)
    multiplier = float(config["loss"]["clean_found_heatmap_multiplier"])
    if multiplier <= 0.0:
        raise DevelopmentTrainingError("clean_found_heatmap_multiplier must be positive")

    if bool(found.any().item()):
        logits = output["heatmap_logits"][found]
        targets = batch["heatmap_targets"][found]
        positive_fraction = targets.mean(dim=(1, 2)).clamp_min(1e-6)
        positive_weight = ((1 - positive_fraction) / positive_fraction).clamp(1, 20)
        heatmap_bce_per_sample = -(
            positive_weight[:, None, None] * targets * F.logsigmoid(logits)
            + (1 - targets) * F.logsigmoid(-logits)
        ).mean(dim=(1, 2))
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2))
        heatmap_dice_per_sample = 1 - (
            (2 * intersection + 1)
            / (probabilities.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + 1)
        )
        heatmap_per_sample = heatmap_bce_per_sample + heatmap_dice_per_sample
        found_indices = found.nonzero(as_tuple=False).flatten().tolist()
        localization_weights = logits.new_tensor(
            [
                multiplier if batch["entries"][index]["variant"] == "clean" else 1.0
                for index in found_indices
            ]
        )
        heatmap_numerator = (localization_weights * heatmap_per_sample).sum()
        heatmap_denominator = localization_weights.sum()
        heatmap_unweighted_numerator = heatmap_per_sample.sum()
        heatmap_unweighted_denominator = _loss_scalar(heatmap_per_sample.numel(), heatmap_numerator)
        heatmap_bce_numerator = (localization_weights * heatmap_bce_per_sample).sum()
        heatmap_dice_numerator = (localization_weights * heatmap_dice_per_sample).sum()
        heatmap = heatmap_numerator / heatmap_denominator.clamp_min(1e-12)
        heatmap_unweighted = heatmap_unweighted_numerator / heatmap_unweighted_denominator.clamp_min(1.0)
        heatmap_bce = heatmap_bce_numerator / heatmap_denominator.clamp_min(1e-12)
        heatmap_dice = heatmap_dice_numerator / heatmap_denominator.clamp_min(1e-12)
    else:
        zero = output["heatmap_logits"].sum() * 0.0
        heatmap_numerator = zero
        heatmap_denominator = _loss_scalar(0.0, zero)
        heatmap_unweighted_numerator = zero
        heatmap_unweighted_denominator = _loss_scalar(0.0, zero)
        heatmap_bce_numerator = zero
        heatmap_dice_numerator = zero
        heatmap = zero
        heatmap_unweighted = zero
        heatmap_bce = zero
        heatmap_dice = zero

    answer_per_sample = F.cross_entropy(
        output["answerability_logits"],
        answer_targets,
        weight=answer_weights,
        reduction="none",
    )
    answer_numerator = answer_per_sample.sum()
    answer_denominator = answer_weights[answer_targets].sum()
    answer = answer_numerator / answer_denominator.clamp_min(1e-12)

    source_per_element = F.binary_cross_entropy_with_logits(
        output["source_logits"],
        batch["source_targets"],
        pos_weight=source_weights,
        reduction="none",
    )
    source_numerator = source_per_element.sum()
    source_denominator = _loss_scalar(source_per_element.numel(), source_numerator)
    source = source_numerator / source_denominator.clamp_min(1.0)

    weights = config["loss"]
    total = (
        float(weights["heatmap_weight"]) * heatmap
        + float(weights["answerability_weight"]) * answer
        + float(weights["source_weight"]) * source
    )
    return {
        "total": total,
        "heatmap": heatmap,
        "heatmap_weighted_objective": heatmap,
        "heatmap_unweighted_diagnostic": heatmap_unweighted,
        "heatmap_bce": heatmap_bce,
        "heatmap_dice": heatmap_dice,
        "answerability": answer,
        "source": source,
        "heatmap_numerator": heatmap_numerator,
        "heatmap_denominator": heatmap_denominator,
        "heatmap_unweighted_numerator": heatmap_unweighted_numerator,
        "heatmap_unweighted_denominator": heatmap_unweighted_denominator,
        "heatmap_bce_numerator": heatmap_bce_numerator,
        "heatmap_dice_numerator": heatmap_dice_numerator,
        "answerability_numerator": answer_numerator,
        "answerability_denominator": answer_denominator,
        "source_numerator": source_numerator,
        "source_denominator": source_denominator,
    }


def scientific_selection_score(metrics: Mapping[str, Any]) -> float:
    gc = float(metrics["clean_found"]["raw_point_in_target"])
    ga = float(metrics["grounding_found"]["point_in_target"])
    answerability = float(metrics["answerability"]["macro_f1"])
    false_found = float(metrics["false_found_on_ambiguous_or_absent"]["rate"])
    return (
        SELECTION_WEIGHTS["clean_found_raw_point_in_target"] * gc
        + SELECTION_WEIGHTS["all_found_raw_point_in_target"] * ga
        + SELECTION_WEIGHTS["answerability_macro_f1"] * answerability
        + SELECTION_WEIGHTS["non_false_found_rate"] * (1.0 - false_found)
    )


@torch.no_grad()
def evaluate_dataset_v11(
    model: PCRAUDevelopmentV11,
    dataset: FrozenFeatureDataset,
    entries: Sequence[Mapping[str, Any]],
    device: torch.device,
    config: Mapping[str, Any],
    answer_weights: torch.Tensor,
    source_weights: torch.Tensor,
    *,
    keep_predictions: bool,
) -> dict[str, Any]:
    """Evaluate one train/dev dataset with raw and answerability-joint grounding.

    Raw point-in-target deliberately ignores the predicted answerability class.
    ``clean_joint_success`` additionally requires a FOUND prediction.  Keeping
    these separate prevents answerability errors from silently changing the
    localization denominator.
    """

    was_training = model.training
    model.eval()
    batch_size = int(config["optimization"]["batch_size"])
    aggregate = {
        "heatmap_numerator": 0.0,
        "heatmap_denominator": 0.0,
        "heatmap_unweighted_numerator": 0.0,
        "heatmap_unweighted_denominator": 0.0,
        "heatmap_bce_numerator": 0.0,
        "heatmap_dice_numerator": 0.0,
        "answerability_numerator": 0.0,
        "answerability_denominator": 0.0,
        "source_numerator": 0.0,
        "source_denominator": 0.0,
    }
    truth_answers: list[int] = []
    predicted_answers: list[int] = []
    source_tp = source_fp = source_fn = 0
    found_total = point_target = point_interior = 0
    clean_found_total = clean_point_target = clean_point_interior = clean_joint = 0
    mass_values: list[float] = []
    clean_mass_values: list[float] = []
    predictions: list[dict[str, Any]] = []
    family_answer: dict[str, list[int]] = defaultdict(list)
    family_found: dict[str, list[int]] = defaultdict(list)
    clean_family_found: dict[str, list[int]] = defaultdict(list)
    relation_found: dict[str, list[int]] = defaultdict(list)

    for indices in chunks(list(range(len(entries))), batch_size):
        batch = dataset.make_batch(indices, device)
        output = model(model_inputs(batch))
        losses = compute_loss_v11(output, batch, answer_weights, source_weights, config)
        for key in aggregate:
            aggregate[key] += float(losses[key].item())

        answer_prediction = output["answerability_logits"].argmax(dim=1)
        source_prediction = torch.sigmoid(output["source_logits"]).ge(0.5)
        source_truth = batch["source_targets"].bool()
        source_tp += int((source_prediction & source_truth).sum().item())
        source_fp += int((source_prediction & ~source_truth).sum().item())
        source_fn += int((~source_prediction & source_truth).sum().item())
        probabilities = torch.sigmoid(output["heatmap_logits"])

        for local_index, entry in enumerate(batch["entries"]):
            target_answer = int(batch["answer_targets"][local_index].item())
            predicted_answer = int(answer_prediction[local_index].item())
            truth_answers.append(target_answer)
            predicted_answers.append(predicted_answer)
            family_answer[entry["family_id"]].append(int(target_answer == predicted_answer))

            flat = int(probabilities[local_index].argmax().item())
            row, column = divmod(flat, 32)
            x = min(639, int((column + 0.5) / 32 * 640))
            y = min(479, int((row + 0.5) / 24 * 480))
            in_target = bool(batch["full_masks"][local_index][y, x])
            in_interior = bool(batch["full_interiors"][local_index][y, x])
            mass = None
            clean_found = target_answer == 0 and entry["variant"] == "clean"
            if target_answer == 0:
                found_total += 1
                point_target += int(in_target)
                point_interior += int(in_interior)
                family_found[entry["family_id"]].append(int(in_target))
                relation_found[entry["audit_only"]["relation"]].append(int(in_target))
                target_grid = batch["heatmap_targets"][local_index]
                mass = float(
                    (probabilities[local_index] * target_grid).sum().item()
                    / probabilities[local_index].sum().clamp_min(1e-12).item()
                )
                mass_values.append(mass)
                if clean_found:
                    clean_found_total += 1
                    clean_point_target += int(in_target)
                    clean_point_interior += int(in_interior)
                    clean_joint += int(in_target and predicted_answer == 0)
                    clean_mass_values.append(mass)
                    clean_family_found[entry["family_id"]].append(int(in_target))

            if keep_predictions:
                predictions.append(
                    {
                        "sample_id": entry["sample_id"],
                        "family_id": entry["family_id"],
                        "variant": entry["variant"],
                        "relation": entry["audit_only"]["relation"],
                        "answerability_truth": ANSWERABILITY_CLASSES[target_answer],
                        "answerability_prediction": ANSWERABILITY_CLASSES[predicted_answer],
                        "map_x": x,
                        "map_y": y,
                        "point_in_target": in_target if target_answer == 0 else None,
                        "point_in_interior": in_interior if target_answer == 0 else None,
                        "mass_in_target": mass,
                        "clean_found_raw_grounding": clean_found,
                        "clean_joint_success": bool(clean_found and in_target and predicted_answer == 0),
                        "source_truth": entry["supervision"]["source_labels"],
                        "source_prediction": [
                            SOURCE_CLASSES[index]
                            for index, value in enumerate(source_prediction[local_index].tolist())
                            if value
                        ],
                    }
                )

    if not entries:
        raise DevelopmentTrainingError("Cannot evaluate an empty dataset")
    answer_metrics = confusion_metrics(truth_answers, predicted_answers, len(ANSWERABILITY_CLASSES))
    source_f1_denominator = 2 * source_tp + source_fp + source_fn
    risky_indices = [index for index, truth in enumerate(truth_answers) if truth in {1, 2}]
    false_found = sum(predicted_answers[index] == 0 for index in risky_indices)
    heatmap_denominator = aggregate["heatmap_denominator"]
    answer_denominator = aggregate["answerability_denominator"]
    source_denominator = aggregate["source_denominator"]
    heatmap_loss = aggregate["heatmap_numerator"] / max(heatmap_denominator, 1e-12)
    heatmap_unweighted_loss = aggregate["heatmap_unweighted_numerator"] / max(
        aggregate["heatmap_unweighted_denominator"], 1.0
    )
    heatmap_bce = aggregate["heatmap_bce_numerator"] / max(heatmap_denominator, 1e-12)
    heatmap_dice = aggregate["heatmap_dice_numerator"] / max(heatmap_denominator, 1e-12)
    answer_loss = aggregate["answerability_numerator"] / max(answer_denominator, 1e-12)
    source_loss = aggregate["source_numerator"] / max(source_denominator, 1.0)
    configured_weights = config["loss"]
    total_loss = (
        float(configured_weights["heatmap_weight"]) * heatmap_loss
        + float(configured_weights["answerability_weight"]) * answer_loss
        + float(configured_weights["source_weight"]) * source_loss
    )

    result: dict[str, Any] = {
        "sample_count": len(entries),
        "family_count": len({entry["family_id"] for entry in entries}),
        "loss": {
            "total": total_loss,
            "heatmap": heatmap_loss,
            "heatmap_weighted_objective": heatmap_loss,
            "heatmap_unweighted_diagnostic": heatmap_unweighted_loss,
            "heatmap_bce": heatmap_bce,
            "heatmap_dice": heatmap_dice,
            "answerability": answer_loss,
            "source": source_loss,
            "aggregation": aggregate,
        },
        "answerability": answer_metrics,
        "false_found_on_ambiguous_or_absent": {
            "count": false_found,
            "denominator": len(risky_indices),
            "rate": false_found / len(risky_indices) if risky_indices else 0.0,
        },
        "source_micro_f1": 2 * source_tp / source_f1_denominator if source_f1_denominator else 1.0,
        "source_counts": {"tp": source_tp, "fp": source_fp, "fn": source_fn},
        "grounding_found": {
            "point_in_target": point_target / found_total if found_total else 0.0,
            "point_in_interior": point_interior / found_total if found_total else 0.0,
            "mean_mass_in_target": statistics.fmean(mass_values) if mass_values else 0.0,
            "correct_target": point_target,
            "correct_interior": point_interior,
            "found_total": found_total,
        },
        "clean_found": {
            "raw_point_in_target": clean_point_target / clean_found_total if clean_found_total else 0.0,
            "raw_point_in_interior": clean_point_interior / clean_found_total if clean_found_total else 0.0,
            "mean_mass_in_target": statistics.fmean(clean_mass_values) if clean_mass_values else 0.0,
            "raw_correct_target": clean_point_target,
            "raw_correct_interior": clean_point_interior,
            "clean_found_total": clean_found_total,
            "joint_found_and_point_in_target": clean_joint / clean_found_total if clean_found_total else 0.0,
            "joint_success_count": clean_joint,
        },
        "family_aggregated": {
            "answerability_accuracy_mean": statistics.fmean(
                statistics.fmean(values) for values in family_answer.values()
            ),
            "found_point_in_target_mean": statistics.fmean(
                statistics.fmean(values) for values in family_found.values()
            )
            if family_found
            else 0.0,
            "clean_found_point_in_target_mean": statistics.fmean(
                statistics.fmean(values) for values in clean_family_found.values()
            )
            if clean_family_found
            else 0.0,
            "families_with_found": len(family_found),
            "families_with_clean_found": len(clean_family_found),
        },
        "grounding_by_relation": {
            relation: {
                "correct": sum(values),
                "total": len(values),
                "accuracy": sum(values) / len(values),
            }
            for relation, values in sorted(relation_found.items())
        },
    }
    score = scientific_selection_score(result)
    result["selection_components"] = {
        "G_c_clean_found_raw_point_in_target": result["clean_found"]["raw_point_in_target"],
        "clean_joint_success": result["clean_found"]["joint_found_and_point_in_target"],
        "G_a_all_found_raw_point_in_target": result["grounding_found"]["point_in_target"],
        "A_answerability_macro_f1": result["answerability"]["macro_f1"],
        "F_false_found_rate": result["false_found_on_ambiguous_or_absent"]["rate"],
        "S_source_micro_f1": result["source_micro_f1"],
        "C_dev_scientific_score_v1_1": score,
        "formula": "0.45*Gc + 0.20*Ga + 0.20*A + 0.15*(1-F)",
    }
    if keep_predictions:
        result["predictions"] = predictions
    model.train(was_training)
    return result


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
