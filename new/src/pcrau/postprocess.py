from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def topk_modes(probability: np.ndarray, k: int = 3, relative_threshold: float = 0.25) -> list[dict[str, Any]]:
    maximum = float(probability.max())
    if maximum <= 0:
        return []
    binary = (probability >= maximum * relative_threshold).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    modes = []
    for label in range(1, count):
        mask = labels == label
        mass = float(probability[mask].sum())
        if mass <= 0:
            continue
        local = np.where(mask, probability, -np.inf)
        y, x = np.unravel_index(int(np.argmax(local)), probability.shape)
        ys, xs = np.nonzero(mask)
        weights = probability[mask] / mass
        mean_x = float((xs * weights).sum())
        mean_y = float((ys * weights).sum())
        dx, dy = xs - mean_x, ys - mean_y
        covariance = [
            [float((weights * dx * dx).sum()), float((weights * dx * dy).sum())],
            [float((weights * dx * dy).sum()), float((weights * dy * dy).sum())],
        ]
        modes.append({"grid_xy": [int(x), int(y)], "mass": mass, "covariance_grid": covariance})
    return sorted(modes, key=lambda row: row["mass"], reverse=True)[:k]


def summarize_heatmap(logits: torch.Tensor, image_size: tuple[int, int] = (640, 480)) -> dict[str, Any]:
    probability = torch.softmax(logits.detach().float().flatten(), dim=0).reshape(logits.shape).cpu().numpy()
    height, width = probability.shape
    flat = int(np.argmax(probability))
    y, x = divmod(flat, width)
    top = np.partition(probability.reshape(-1), -2)[-2:]
    positive = probability[probability > 0]
    entropy = float(-(positive * np.log(positive)).sum() / math.log(height * width))
    modes = topk_modes(probability)
    image_width, image_height = image_size
    return {
        "map_grid_xy": [x, y],
        "map_pixel_xy": [min(image_width - 1, int((x + 0.5) / width * image_width)),
                         min(image_height - 1, int((y + 0.5) / height * image_height))],
        "entropy_normalized": entropy,
        "peak_margin": float(top.max() - top.min()),
        "mode_count": len(modes),
        "topk_modes": modes,
    }


def confidence_region(logits: torch.Tensor, probability_threshold: float) -> np.ndarray:
    probability = torch.softmax(logits.detach().float().flatten(), dim=0).reshape(logits.shape)
    return (probability >= probability_threshold).cpu().numpy()


def highest_density_region(probability_grid: list[list[float]], probability_mass: float) -> dict[str, Any]:
    probability = np.asarray(probability_grid, dtype=np.float64)
    order = np.argsort(-probability, axis=None)
    cumulative = np.cumsum(probability.reshape(-1)[order])
    count = min(len(order), int(np.searchsorted(cumulative, probability_mass, side="left")) + 1)
    selected = order[:count]
    ys, xs = np.unravel_index(selected, probability.shape)
    return {
        "grid_cells_xy": [[int(x), int(y)] for x, y in zip(xs, ys)],
        "grid_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "actual_probability_mass": float(probability.reshape(-1)[selected].sum()),
        "grid_area_fraction": count / probability.size,
    }
