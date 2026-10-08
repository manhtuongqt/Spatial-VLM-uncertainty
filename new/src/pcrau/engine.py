from __future__ import annotations

import json
import math
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .dataset import ANSWER_CLASSES, SOURCE_CLASSES, move_model_batch
from .losses import compute_losses
from .metrics import confusion_metrics, multilabel_f1
from .postprocess import summarize_heatmap


def autocast_context(device: torch.device, optimization: Mapping[str, Any]):
    if bool(optimization["amp"]) and device.type == "cuda":
        dtype_name = optimization.get("amp_dtype", "bfloat16")
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def train_epoch(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: Mapping[str, Any],
    answer_weights: torch.Tensor,
    source_weights: torch.Tensor,
    max_batches: int | None = None,
) -> tuple[dict[str, float], int]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    examples = 0
    steps = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, config["optimization"]):
            output = model(move_model_batch(batch, device))
            losses = compute_losses(output, batch, config, answer_weights, source_weights)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optimization"]["gradient_clip_norm"]))
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_stepped = scaler.get_scale() >= scale_before
        if optimizer_stepped:
            scheduler.step()
        batch_size = len(batch["sample_id"])
        examples += batch_size
        steps += int(optimizer_stepped)
        for name, value in losses.items():
            totals[name] += float(value.detach()) * batch_size
    return {name: value / max(1, examples) for name, value in totals.items()}, steps


def _load_rr_points(path: Path | None) -> dict[str, list[int]]:
    if path is None or not path.is_file():
        return {}
    result: dict[str, list[int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("method") == "B1" and len(row.get("pixel_points_xy", [])) == 1:
                result[row["sample_id"]] = [int(v) for v in row["pixel_points_xy"][0]]
    return result


def _inside(mask: torch.Tensor, point_xy: list[int]) -> bool:
    x, y = point_xy
    return 0 <= y < mask.shape[-2] and 0 <= x < mask.shape[-1] and bool(mask[y, x])


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    config: Mapping[str, Any],
    answer_weights: torch.Tensor | None = None,
    source_weights: torch.Tensor | None = None,
    rr_points_path: Path | None = None,
    source_thresholds: Mapping[str, float] | None = None,
    include_predictions: bool = True,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    answer_truth: list[int] = []
    answer_prediction: list[int] = []
    source_truth: list[list[bool]] = []
    source_prediction: list[list[bool]] = []
    grounding_inside: list[bool] = []
    edge_truth: list[bool] = []
    edge_prediction: list[bool] = []
    predictions: list[dict[str, Any]] = []
    rr_points = _load_rr_points(rr_points_path)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        with autocast_context(device, config["optimization"]):
            output = model(move_model_batch(batch, device))
            losses = compute_losses(output, batch, config, answer_weights, source_weights)
        batch_size = len(batch["sample_id"])
        count += batch_size
        for name, value in losses.items():
            totals[name] += float(value.detach()) * batch_size
        answer_prob = output["answerability_logits"].float().softmax(-1).cpu()
        source_prob = output["source_logits"].float().sigmoid().cpu()
        edge_prob = output["relation_edge_logits"].float().sigmoid().cpu()
        target_logits = output["target_logits"].float().cpu()
        gate = output["fusion_gate_mean"].float().cpu()
        batch_answer_truth = batch["answer_target"].tolist()
        batch_answer_prediction = answer_prob.argmax(-1).tolist()
        answer_truth.extend(batch_answer_truth)
        answer_prediction.extend(batch_answer_prediction)
        source_truth.extend((batch["source_target"] > 0.5).tolist())
        thresholds = source_prob.new_tensor(
            [float((source_thresholds or {}).get(name, 0.5)) for name in SOURCE_CLASSES]
        )
        source_prediction.extend((source_prob >= thresholds[None]).tolist())
        for index in range(batch_size):
            heatmap = summarize_heatmap(target_logits[index])
            inside = _inside(batch["target_full"][index], heatmap["map_pixel_xy"])
            if bool(batch["target_loss_mask"][index]):
                grounding_inside.append(inside)
            valid_edge = batch["edge_mask"][index].bool()
            if bool(valid_edge.any()):
                edge_truth.extend((batch["edge_target"][index][valid_edge] > 0.5).tolist())
                edge_prediction.extend((edge_prob[index][valid_edge] >= 0.5).tolist())
            if not include_predictions:
                continue
            r_thumb = batch["r_thumb"][index].float()
            d_thumb = batch["d_thumb"][index].float()
            cosine = float(torch.nn.functional.cosine_similarity(r_thumb[None], d_thumb[None]).item())
            mae = float((r_thumb - d_thumb).abs().mean().item())
            rr_point = rr_points.get(batch["sample_id"][index])
            disagreement = None
            if rr_point is not None:
                dx = heatmap["map_pixel_xy"][0] - rr_point[0]
                dy = heatmap["map_pixel_xy"][1] - rr_point[1]
                disagreement = float(math.hypot(dx, dy) / math.hypot(640, 480))
            probabilities = torch.softmax(target_logits[index].flatten(), 0).reshape(target_logits[index].shape)
            target_mask = batch["target_heatmap"][index] > 0
            target_mass = float(probabilities[target_mask].sum()) if bool(target_mask.any()) else 0.0
            required_hd_mass = None
            if bool(target_mask.any()):
                order = torch.argsort(probabilities.flatten(), descending=True)
                sorted_target = target_mask.flatten()[order]
                first_target_rank = int(torch.nonzero(sorted_target, as_tuple=False)[0, 0])
                required_hd_mass = float(probabilities.flatten()[order][: first_target_rank + 1].sum())
            heatmap["probability_grid"] = probabilities.tolist()
            prediction = {
                "schema_version": 1,
                "sample_id": batch["sample_id"][index],
                "family_id": batch["family_id"][index],
                "variant": batch["variant"][index],
                "spatial": heatmap,
                "answerability_probabilities": {
                    name: float(answer_prob[index, slot]) for slot, name in enumerate(ANSWER_CLASSES)
                },
                "source_probabilities": {
                    name: float(source_prob[index, slot]) for slot, name in enumerate(SOURCE_CLASSES)
                },
                "relation_edge_probabilities": edge_prob[index].tolist(),
                "relation_consistency": float(edge_prob[index][valid_edge].mean()) if bool(valid_edge.any()) else 1.0,
                "fusion_gate_mean": float(gate[index]),
                "rgb_depth_cosine": cosine,
                "rgb_depth_mae": mae,
                "roborefer_point_xy": rr_point,
                "roborefer_disagreement_normalized": disagreement,
                # Fields below are an evaluator-only join, never model input.
                "evaluation": {
                    "answerability_index": int(batch_answer_truth[index]),
                    "answerability_state": ANSWER_CLASSES[int(batch_answer_truth[index])],
                    "source_multihot_5": [int(v) for v in batch["source_target"][index].tolist()],
                    "target_exists": bool(batch["target_loss_mask"][index]),
                    "map_inside_target": inside,
                    "target_probability_mass": target_mass,
                    "target_required_hd_mass": required_hd_mass,
                    "error_event": bool(batch_answer_truth[index] != 0 or not inside),
                },
            }
            if "observable_answerability_evidence" in output:
                from .answerability_evidence import EVIDENCE_NAMES
                observed = output["observable_answerability_evidence"][index].detach().float().cpu().tolist()
                prediction["observable_answerability_evidence"] = dict(zip(EVIDENCE_NAMES, observed))
            if "ranker_error_logit" in output:
                prediction["ranker_error_logit"] = float(output["ranker_error_logit"][index].detach().cpu())
                prediction["ranker_sha256"] = model.ranker_sha256
                if hasattr(model, "source_checkpoint_sha256"):
                    prediction["ranker_source_checkpoint_sha256"] = model.source_checkpoint_sha256
                else:
                    prediction["source_v2_sha256"] = model.source_v2_sha256
            if config.get("runtime_checkpoint_sha256"):
                prediction["model_checkpoint_sha256"] = config["runtime_checkpoint_sha256"]
            predictions.append(prediction)
    answer_metrics = confusion_metrics(answer_truth, answer_prediction, len(ANSWER_CLASSES))
    source_metrics = multilabel_f1(
        np.asarray(source_truth, dtype=bool), np.asarray(source_prediction, dtype=bool)
    )
    metrics: dict[str, Any] = {
        "samples": count,
        "loss": {name: value / max(1, count) for name, value in totals.items()},
        "answerability": answer_metrics,
        "source": source_metrics,
        "grounding_accuracy": float(np.mean(grounding_inside)) if grounding_inside else None,
        "grounding_evaluated": len(grounding_inside),
        "relation_edge_accuracy": float(np.mean(np.equal(edge_truth, edge_prediction))) if edge_truth else None,
        "relation_edges_evaluated": len(edge_truth),
        "source_thresholds": {
            name: float((source_thresholds or {}).get(name, 0.5)) for name in SOURCE_CLASSES
        },
    }
    return metrics, predictions
