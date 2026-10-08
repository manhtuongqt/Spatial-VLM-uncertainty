#!/usr/bin/env python3
"""Render presentation-labelled v2 figures from frozen Calibration outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v2"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def curve(labels: np.ndarray, risks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(risks, kind="stable")
    sorted_labels = labels[order]
    coverage = np.arange(1, len(labels) + 1, dtype=float) / len(labels)
    selective_risk = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return coverage, selective_risk


def main() -> None:
    metrics = json.loads((RESULT / "GAZEBO_CALIBRATION_METRICS.json").read_text(encoding="utf-8"))
    scored = rows(RESULT / "calibration_scored_predictions.jsonl")
    if metrics.get("protocol_id") != "gazebo_calibration_v2" or len(scored) != 128:
        raise ValueError("expected frozen 128-record Calibration-v2 result")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    for key, label, marker in (
        ("raw", "raw WP5 risk", "o"),
        ("calibrated", "temperature-scaled", "s"),
    ):
        bins = metrics["reliability"][key]
        ax.plot(
            [item["mean_probability"] for item in bins],
            [item["unsafe_frequency"] for item in bins],
            marker=marker,
            label=label,
        )
    ax.set(
        xlabel="Predicted unsafe probability",
        ylabel="Observed unsafe frequency",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Calibration-v2 reliability (128 families)",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULT / "CALIBRATION_V2_RELIABILITY.png", dpi=180)
    plt.close(fig)

    labels = np.asarray([item["unsafe"] for item in scored], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("raw_risk", "raw WP5 risk"), ("calibrated_risk", "temperature-scaled")):
        coverage, selective_risk = curve(labels, np.asarray([item[key] for item in scored], dtype=float))
        ax.plot(coverage, selective_risk, label=label)
    ax.set(
        xlabel="Coverage",
        ylabel="Selective risk",
        xlim=(0, 1),
        title="Calibration-v2 risk–coverage (ranking invariant)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT / "CALIBRATION_V2_RISK_COVERAGE.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
