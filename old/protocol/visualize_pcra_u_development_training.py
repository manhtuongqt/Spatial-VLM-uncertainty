#!/usr/bin/env python3
"""Render reproducible scientific figures from the frozen development run.

This is an add-only, post-training renderer.  It reads the existing optimizer
log and the 15 existing dev-evaluation JSON files.  It never loads a model,
runs inference, changes a checkpoint, fits calibration, or opens a test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter, ScalarFormatter


WORKSPACE = Path(__file__).resolve().parents[1]
RUN_ROOT = WORKSPACE / "results/pcra_u_runs/pcra_u_development_train_20260824"
LOG_PATH = RUN_ROOT / "logs/development_metrics.jsonl"
EPOCH_ROOT = RUN_ROOT / "metrics"
CHECKPOINT_TABLE = RUN_ROOT / "tables/table_07_training_checkpoint.csv"
CONFIG_PATH = WORKSPACE / "protocol/pcra_u_development_train_config.json"
BEST_POINTER = RUN_ROOT / "checkpoints/development/best_dev_total_loss.json"

FIGURE_ROOT = RUN_ROOT / "figures"
TABLE_ROOT = RUN_ROOT / "tables"
EPOCH_TABLE = TABLE_ROOT / "table_development_epoch_trajectory_visualization.csv"
OPTIMIZER_TABLE = TABLE_ROOT / "table_development_optimizer_epoch_summary.csv"
SUPPLEMENT_REPORT = RUN_ROOT / "TRAINING_VISUALIZATION_SUPPLEMENT.md"
SUPPLEMENT_MANIFEST = RUN_ROOT / "checks/training_visualization_supplement_manifest.json"

FIGURES = {
    "F03": FIGURE_ROOT / "F03_development_convergence_checkpoint_selection",
    "F04": FIGURE_ROOT / "F04_development_optimization_health",
    "F05": FIGURE_ROOT / "F05_development_checkpoint_task_trajectory",
}

PROTECTED_ARTIFACTS = {
    "train_contract": WORKSPACE / "protocol/PCRA_U_DEVELOPMENT_TRAIN_CONTRACT.md",
    "train_config": CONFIG_PATH,
    "train_manifest": WORKSPACE / "protocol/pcra_u_development_train_manifest.json",
    "feature_manifest": WORKSPACE / "protocol/pcra_u_development_feature_manifest.json",
    "execution_lock": WORKSPACE / "protocol/pcra_u_development_execution_lock.json",
    "training_code": WORKSPACE / "protocol/pcra_u_development_train.py",
    "training_common_code": WORKSPACE / "protocol/pcra_u_development_common.py",
    "locked_feature_index": RUN_ROOT / "locked_inputs/index_full.json",
    "training_report": RUN_ROOT / "PCRA_U_DEVELOPMENT_TRAIN_REPORT.json",
    "selected_checkpoint_model": RUN_ROOT / "checkpoints/development/step_000002000/model.safetensors",
    "selected_checkpoint_manifest": RUN_ROOT / "checkpoints/development/step_000002000/manifest.json",
    "selected_checkpoint_metadata": RUN_ROOT / "checkpoints/development/step_000002000/metadata.json",
    "architecture_freeze_lock": WORKSPACE / "protocol/pcra_u_architecture_freeze_lock.json",
    "failure_audit_report": WORKSPACE
    / "results/pcra_u_runs/pcra_u_development_failure_audit_20260824/PCRA_U_DEVELOPMENT_FAILURE_AUDIT_REPORT.json",
}

STATUS = "EXPLORATORY_DEV_ONLY"
ROLLING_WINDOW = 50
COLORS = {
    "train": "#326FA8",
    "dev": "#C74440",
    "heatmap": "#287D5A",
    "answer": "#D89016",
    "source": "#7759B7",
    "selected": "#111111",
    "patience": "#E8E8E8",
    "coverage": "#3572A5",
}


class VisualizationError(RuntimeError):
    pass


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE.resolve()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.titlesize": 11.5,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "grid.color": "#D6D6D6",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "svg.hashsalt": "pcra-u-development-training-visualization-v1",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.09, 1.06, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def save_figure(fig: plt.Figure, stem: Path, title: str) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    metadata = {
        "Creator": "visualize_pcra_u_development_training.py",
        "Date": "2026-08-24",
        "Title": title,
    }
    fig.savefig(svg, format="svg", bbox_inches="tight", metadata=metadata)
    fig.savefig(png, format="png", dpi=220, bbox_inches="tight", metadata=metadata)
    plt.close(fig)
    return [svg, png]


def rolling(values: np.ndarray, window: int, statistic: str) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < window:
        raise VisualizationError(f"Need at least {window} observations for rolling statistic")
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    if statistic == "mean":
        result = windows.mean(axis=1)
    elif statistic == "median":
        result = np.median(windows, axis=1)
    elif statistic == "q10":
        result = np.quantile(windows, 0.10, axis=1)
    elif statistic == "q90":
        result = np.quantile(windows, 0.90, axis=1)
    else:
        raise VisualizationError(f"Unsupported rolling statistic: {statistic}")
    return np.arange(window, len(values) + 1), result


def load_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    required = [LOG_PATH, CHECKPOINT_TABLE, CONFIG_PATH, BEST_POINTER, *PROTECTED_ARTIFACTS.values()]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise VisualizationError(f"Missing required artifact: {missing[0]}")

    step_rows = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    epoch_paths = sorted(EPOCH_ROOT.glob("development_epoch_*.json"))
    epoch_rows = [json.loads(path.read_text(encoding="utf-8")) for path in epoch_paths]
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    best_pointer = json.loads(BEST_POINTER.read_text(encoding="utf-8"))

    if len(step_rows) != 3000 or [int(row["step"]) for row in step_rows] != list(range(1, 3001)):
        raise VisualizationError("Optimizer log is not the locked contiguous 3,000-step run")
    if len(epoch_rows) != 15 or [int(row["epoch"]) for row in epoch_rows] != list(range(1, 16)):
        raise VisualizationError("Expected exactly 15 ordered epoch evaluations")
    if any(not bool(row.get("finite")) for row in step_rows):
        raise VisualizationError("Optimizer log contains a non-finite step")

    weights = config["loss"]
    heatmap_weight = float(weights["heatmap_weight"])
    answer_weight = float(weights["answerability_weight"])
    source_weight = float(weights["source_weight"])
    step_identity_error = max(
        abs(
            float(row["total_loss"])
            - (
                heatmap_weight * float(row["heatmap_loss"])
                + answer_weight * float(row["answerability_loss"])
                + source_weight * float(row["source_loss"])
            )
        )
        for row in step_rows
    )
    if step_identity_error > 2e-5:
        raise VisualizationError(f"Step loss identity failed: max error {step_identity_error}")

    for row in epoch_rows:
        if int(row["global_step"]) != int(row["epoch"]) * 200:
            raise VisualizationError("Epoch/checkpoint step cadence changed")
        loss = row["dev"]["loss"]
        reconstructed = (
            heatmap_weight * float(loss["heatmap"])
            + answer_weight * float(loss["answerability"])
            + source_weight * float(loss["source"])
        )
        if abs(float(loss["total"]) - reconstructed) > 1e-9:
            raise VisualizationError("Dev loss components do not reconstruct total loss")
        dev = row["dev"]
        if int(dev["family_count"]) != 80 or int(dev["sample_count"]) != 400:
            raise VisualizationError("Dev denominator differs from the locked 80-family/400-variant split")
        if int(dev["grounding_found"]["found_total"]) != 168:
            raise VisualizationError("FOUND grounding denominator differs from 168")
        risky = dev["false_found_on_ambiguous_or_absent"]
        if int(risky["denominator"]) != 95:
            raise VisualizationError("False-FOUND denominator differs from 95")
        if len(dev["predictions"]) != 400:
            raise VisualizationError("Prediction count differs from 400")

    dev_losses = np.asarray([float(row["dev"]["loss"]["total"]) for row in epoch_rows])
    selected_index = int(np.argmin(dev_losses))
    selected_epoch = int(epoch_rows[selected_index]["epoch"])
    selected_step = int(epoch_rows[selected_index]["global_step"])
    pointer_text = json.dumps(best_pointer, sort_keys=True)
    if selected_epoch != 10 or selected_step != 2000 or "step_000002000" not in pointer_text:
        raise VisualizationError("Selected checkpoint is not the locked epoch-10/step-2000 minimum")

    return step_rows, epoch_rows, config, {
        "selected_epoch": selected_epoch,
        "selected_step": selected_step,
        "step_loss_identity_max_abs_error": step_identity_error,
    }


def epoch_visualization_rows(epoch_rows: Sequence[Mapping[str, Any]], selected_epoch: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in epoch_rows:
        dev = row["dev"]
        loss = dev["loss"]
        predictions = dev["predictions"]
        predicted_found_count = sum(item["answerability_prediction"] == "FOUND" for item in predictions)
        risky = dev["false_found_on_ambiguous_or_absent"]
        ground = dev["grounding_found"]
        rows.append(
            {
                "scientific_status": STATUS,
                "epoch": int(row["epoch"]),
                "step": int(row["global_step"]),
                "selected_by_min_dev_total_loss": int(row["epoch"]) == selected_epoch,
                "train_epoch_mean_total_loss": float(row["train_epoch_mean_total_loss"]),
                "dev_total_loss": float(loss["total"]),
                "dev_heatmap_weighted_contribution": float(loss["heatmap"]),
                "dev_answerability_weighted_contribution": 0.5 * float(loss["answerability"]),
                "dev_source_weighted_contribution": 0.5 * float(loss["source"]),
                "answerability_macro_f1": float(dev["answerability"]["macro_f1"]),
                "family_equal_found_point_in_target": float(dev["family_aggregated"]["found_point_in_target_mean"]),
                "sample_found_point_in_target": float(ground["point_in_target"]),
                "sample_found_mass_in_target": float(ground["mean_mass_in_target"]),
                "false_found_count": int(risky["count"]),
                "false_found_denominator": int(risky["denominator"]),
                "false_found_rate": float(risky["rate"]),
                "predicted_found_count": predicted_found_count,
                "predicted_found_denominator": len(predictions),
                "predicted_found_coverage": predicted_found_count / len(predictions),
                "dev_family_count": int(dev["family_count"]),
                "dev_variant_count": int(dev["sample_count"]),
                "found_grounding_variant_count": int(ground["found_total"]),
                "epochs_without_improvement": int(row["epochs_without_improvement"]),
            }
        )
    return rows


def optimizer_epoch_rows(step_rows: Sequence[Mapping[str, Any]], clip_norm: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in range(1, 16):
        subset = [row for row in step_rows if int(row["epoch"]) == epoch]
        if len(subset) != 200:
            raise VisualizationError(f"Epoch {epoch} does not contain 200 optimizer steps")
        grad = np.asarray([float(row["gradient_norm_preclip"]) for row in subset])
        target = np.asarray([float(row["target_head_gradient"]) for row in subset])
        answer = np.asarray([float(row["answer_head_gradient"]) for row in subset])
        source = np.asarray([float(row["source_head_gradient"]) for row in subset])
        rows.append(
            {
                "scientific_status": STATUS,
                "epoch": epoch,
                "step_start": int(subset[0]["step"]),
                "step_end": int(subset[-1]["step"]),
                "global_gradient_preclip_median": float(np.median(grad)),
                "global_gradient_preclip_q05": float(np.quantile(grad, 0.05)),
                "global_gradient_preclip_q95": float(np.quantile(grad, 0.95)),
                "steps_above_clip_threshold": int(np.sum(grad > clip_norm)),
                "steps_total": len(subset),
                "fraction_above_clip_threshold": float(np.mean(grad > clip_norm)),
                "target_head_gradient_median": float(np.median(target)),
                "answer_head_gradient_median": float(np.median(answer)),
                "source_head_gradient_median": float(np.median(source)),
                "target_head_nonzero_steps": int(np.sum(target > 0.0)),
                "answer_head_nonzero_steps": int(np.sum(answer > 0.0)),
                "source_head_nonzero_steps": int(np.sum(source > 0.0)),
            }
        )
    return rows


def mark_selection(ax: plt.Axes, selected_x: float, patience_start: float, patience_end: float) -> None:
    ax.axvspan(patience_start, patience_end, color=COLORS["patience"], alpha=0.65, zorder=0)
    ax.axvline(selected_x, color=COLORS["selected"], linestyle="--", linewidth=1.25, zorder=4)


def plot_convergence(
    step_rows: Sequence[Mapping[str, Any]], epoch_rows: Sequence[Mapping[str, Any]], selected_step: int
) -> list[Path]:
    steps = np.asarray([int(row["step"]) for row in step_rows])
    total = np.asarray([float(row["total_loss"]) for row in step_rows])
    heatmap = np.asarray([float(row["heatmap_loss"]) for row in step_rows])
    answer = 0.5 * np.asarray([float(row["answerability_loss"]) for row in step_rows])
    source = 0.5 * np.asarray([float(row["source_loss"]) for row in step_rows])
    epochs = np.asarray([int(row["epoch"]) for row in epoch_rows])
    epoch_steps = np.asarray([int(row["global_step"]) for row in epoch_rows])
    train_epoch = np.asarray([float(row["train_epoch_mean_total_loss"]) for row in epoch_rows])
    dev_total = np.asarray([float(row["dev"]["loss"]["total"]) for row in epoch_rows])

    fig = plt.figure(figsize=(14.0, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ax_total = fig.add_subplot(grid[0, :])
    ax_train = fig.add_subplot(grid[1, 0])
    ax_dev = fig.add_subplot(grid[1, 1])

    rolling_steps, rolling_total = rolling(total, ROLLING_WINDOW, "mean")
    ax_total.plot(steps, total, color=COLORS["train"], alpha=0.09, linewidth=0.65, rasterized=True)
    ax_total.plot(rolling_steps, rolling_total, color=COLORS["train"], linewidth=2.0, label="Train total, trailing 50-step mean")
    ax_total.plot(epoch_steps, train_epoch, "o", color=COLORS["train"], markersize=4.5, label="Train epoch mean (200 batches)")
    ax_total.plot(epoch_steps, dev_total, "s-", color=COLORS["dev"], linewidth=1.8, markersize=5, label="Dev total loss (epoch evaluation)")
    mark_selection(ax_total, selected_step, 2000, 3000)
    ax_total.scatter([selected_step], [dev_total[9]], marker="*", s=180, color=COLORS["selected"], zorder=6)
    ax_total.annotate(
        "selected: epoch 10 / step 2,000\nminimum dev total = 1.3516",
        xy=(selected_step, dev_total[9]),
        xytext=(2250, 2.35),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=9,
    )
    ax_total.text(2500, 0.16, "early-stopping wait: epochs 11–15", ha="center", fontsize=8.5, color="#555555")
    ax_total.set_xlim(0, 3050)
    ax_total.set_ylim(0, max(3.5, float(total.max()) * 1.03))
    ax_total.set_xlabel("Optimizer step")
    ax_total.set_ylabel("Objective loss")
    ax_total.set_title("Train convergence and locked checkpoint selection")
    ax_total.grid(axis="y")
    ax_total.legend(ncol=2, loc="upper right")
    panel_label(ax_total, "A")

    for values, color, label in (
        (heatmap, COLORS["heatmap"], "heatmap × 1.0"),
        (answer, COLORS["answer"], "answerability × 0.5"),
        (source, COLORS["source"], "source × 0.5"),
    ):
        x_roll, y_roll = rolling(values, ROLLING_WINDOW, "mean")
        ax_train.plot(x_roll, y_roll, color=color, linewidth=1.8, label=label)
    mark_selection(ax_train, selected_step, 2000, 3000)
    ax_train.set_xlim(0, 3050)
    ax_train.set_ylim(bottom=0)
    ax_train.set_xlabel("Optimizer step")
    ax_train.set_ylabel("Weighted loss contribution")
    ax_train.set_title("Train objective decomposition (trailing 50-step mean)")
    ax_train.grid(axis="y")
    ax_train.legend()
    panel_label(ax_train, "B")

    dev_heatmap = np.asarray([float(row["dev"]["loss"]["heatmap"]) for row in epoch_rows])
    dev_answer = 0.5 * np.asarray([float(row["dev"]["loss"]["answerability"]) for row in epoch_rows])
    dev_source = 0.5 * np.asarray([float(row["dev"]["loss"]["source"]) for row in epoch_rows])
    ax_dev.stackplot(
        epochs,
        dev_heatmap,
        dev_answer,
        dev_source,
        colors=[COLORS["heatmap"], COLORS["answer"], COLORS["source"]],
        alpha=0.88,
        labels=["heatmap × 1.0", "answerability × 0.5", "source × 0.5"],
    )
    mark_selection(ax_dev, 10, 10.5, 15.5)
    ax_dev.scatter([10], [dev_total[9]], marker="*", s=145, color="white", edgecolor="#111111", linewidth=1.0, zorder=6)
    ax_dev.set_xlim(1, 15)
    ax_dev.set_xticks(range(1, 16, 2))
    ax_dev.set_ylim(bottom=0)
    ax_dev.set_xlabel("Epoch")
    ax_dev.set_ylabel("Weighted loss contribution")
    ax_dev.set_title("Dev objective decomposition (components sum to total)")
    ax_dev.grid(axis="y")
    ax_dev.legend(loc="upper right")
    panel_label(ax_dev, "C")

    fig.suptitle("F03 — P-CRA-U development training: convergence and checkpoint selection", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.012,
        "EXPLORATORY_DEV_ONLY · one deterministic seed · selected strictly by minimum dev total loss · "
        "gray region is patience, not a confidence interval",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, FIGURES["F03"], "F03 — Development convergence and checkpoint selection")


def plot_optimization_health(
    step_rows: Sequence[Mapping[str, Any]], optimizer_rows: Sequence[Mapping[str, Any]], selected_step: int, clip_norm: float
) -> list[Path]:
    steps = np.asarray([int(row["step"]) for row in step_rows])
    global_grad = np.asarray([float(row["gradient_norm_preclip"]) for row in step_rows])
    target_grad = np.asarray([float(row["target_head_gradient"]) for row in step_rows])
    answer_grad = np.asarray([float(row["answer_head_gradient"]) for row in step_rows])
    source_grad = np.asarray([float(row["source_head_gradient"]) for row in step_rows])
    learning_rate = np.asarray([float(row["learning_rate"]) for row in step_rows])

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), constrained_layout=True)
    ax_global, ax_head, ax_lr, ax_clip = axes.flat

    roll_steps, median = rolling(global_grad, ROLLING_WINDOW, "median")
    _, q10 = rolling(global_grad, ROLLING_WINDOW, "q10")
    _, q90 = rolling(global_grad, ROLLING_WINDOW, "q90")
    ax_global.plot(steps, global_grad, color="#6F8294", alpha=0.09, linewidth=0.6, rasterized=True)
    ax_global.fill_between(roll_steps, q10, q90, color="#8BB6D9", alpha=0.30, label="rolling q10–q90 (not CI)")
    ax_global.plot(roll_steps, median, color="#285F8F", linewidth=1.8, label="rolling median")
    ax_global.axhline(clip_norm, color="#C74440", linestyle="--", linewidth=1.3, label="configured clip threshold = 5")
    mark_selection(ax_global, selected_step, 2000, 3000)
    ax_global.set_yscale("log")
    ax_global.set_xlim(0, 3050)
    ax_global.set_xlabel("Optimizer step")
    ax_global.set_ylabel("Global gradient norm (pre-clip, log scale)")
    ax_global.set_title("Pre-clip gradient dynamics")
    ax_global.grid(axis="y", which="both")
    ax_global.legend(loc="upper right")
    panel_label(ax_global, "A")

    for values, color, label in (
        (target_grad, COLORS["heatmap"], "target head"),
        (answer_grad, COLORS["answer"], "answerability head"),
        (source_grad, COLORS["source"], "source head"),
    ):
        x_roll, y_roll = rolling(values, ROLLING_WINDOW, "median")
        ax_head.plot(x_roll, np.maximum(y_roll, 1e-8), color=color, linewidth=1.8, label=label)
    mark_selection(ax_head, selected_step, 2000, 3000)
    ax_head.set_yscale("log")
    ax_head.set_xlim(0, 3050)
    ax_head.set_xlabel("Optimizer step")
    ax_head.set_ylabel("Head gradient norm (rolling median, log scale)")
    ax_head.set_title("Active-head gradients")
    ax_head.grid(axis="y", which="both")
    ax_head.legend()
    panel_label(ax_head, "B")

    ax_lr.plot(steps, learning_rate, color="#3D7C6F", linewidth=1.8)
    ax_lr.axvline(200, color="#777777", linestyle=":", linewidth=1.2)
    mark_selection(ax_lr, selected_step, 2000, 3000)
    ax_lr.annotate("warmup ends\nstep 200", xy=(200, learning_rate[199]), xytext=(420, 0.00025), arrowprops={"arrowstyle": "->"}, fontsize=8.5)
    ax_lr.set_xlim(0, 3050)
    ax_lr.set_ylim(0, 0.00032)
    ax_lr.set_xlabel("Optimizer step")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.set_title("Locked linear-warmup + cosine-decay schedule")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-4, -4))
    ax_lr.yaxis.set_major_formatter(formatter)
    ax_lr.grid(axis="y")
    panel_label(ax_lr, "C")

    epochs = np.asarray([int(row["epoch"]) for row in optimizer_rows])
    fractions = np.asarray([float(row["fraction_above_clip_threshold"]) for row in optimizer_rows])
    bars = ax_clip.bar(epochs, fractions, color="#769FBE", edgecolor="#315D7E", linewidth=0.6)
    bars[9].set_color("#222222")
    mark_selection(ax_clip, 10, 10.5, 15.5)
    ax_clip.set_xlim(0.3, 15.7)
    ax_clip.set_ylim(0, max(0.8, float(fractions.max()) + 0.08))
    ax_clip.set_xlabel("Epoch (200 optimizer steps each)")
    ax_clip.set_ylabel("Steps with pre-clip norm > 5")
    ax_clip.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_clip.set_title("Gradient-clipping engagement by epoch")
    ax_clip.grid(axis="y")
    panel_label(ax_clip, "D")

    above = int(np.sum(global_grad > clip_norm))
    target_zero = int(np.sum(target_grad == 0.0))
    fig.suptitle("F04 — P-CRA-U development training: optimization health", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.012,
        f"EXPLORATORY_DEV_ONLY · finite 3,000/3,000 steps · pre-clip > 5 on {above}/3,000 ({above/30:.1f}%) · "
        f"target-head zero on {target_zero} FOUND-free batches; this is not evidence of a dead head",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, FIGURES["F04"], "F04 — Development optimization health")


def plot_checkpoint_task_trajectory(epoch_rows: Sequence[Mapping[str, Any]], selected_epoch: int) -> list[Path]:
    epochs = np.asarray([int(row["epoch"]) for row in epoch_rows])
    family_grounding = np.asarray([float(row["dev"]["family_aggregated"]["found_point_in_target_mean"]) for row in epoch_rows])
    macro_f1 = np.asarray([float(row["dev"]["answerability"]["macro_f1"]) for row in epoch_rows])
    false_found = np.asarray([float(row["dev"]["false_found_on_ambiguous_or_absent"]["rate"]) for row in epoch_rows])
    false_found_count = np.asarray([int(row["dev"]["false_found_on_ambiguous_or_absent"]["count"]) for row in epoch_rows])
    coverage_count = np.asarray(
        [sum(item["answerability_prediction"] == "FOUND" for item in row["dev"]["predictions"]) for row in epoch_rows]
    )
    coverage = coverage_count / 400.0

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), constrained_layout=True)
    panels = [
        (axes[0, 0], family_grounding, COLORS["heatmap"], "Family-equal point-in-target", "80 dev families; 168 dependent FOUND variants"),
        (axes[0, 1], macro_f1, COLORS["answer"], "Answerability macro-F1", "4 states; 400 dependent variants"),
        (axes[1, 0], false_found, COLORS["dev"], "False-FOUND rate", "denominator: 95 AMBIGUOUS/ABSENT variants"),
        (axes[1, 1], coverage, COLORS["coverage"], "Predicted-FOUND coverage", "denominator: all 400 dev variants"),
    ]
    for index, (ax, values, color, title, denominator) in enumerate(panels):
        mark_selection(ax, selected_epoch, 10.5, 15.5)
        ax.plot(epochs, values, "o-", color=color, linewidth=1.8, markersize=4.5)
        ax.scatter([selected_epoch], [values[selected_epoch - 1]], marker="*", s=170, color="#111111", zorder=6)
        ax.set_xlim(1, 15)
        ax.set_xticks(range(1, 16, 2))
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Epoch / checkpoint")
        ax.set_ylabel(title)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"{title}\n{denominator}")
        ax.grid(axis="y")
        panel_label(ax, chr(ord("A") + index))

    axes[0, 0].annotate("85.8% at selected checkpoint", xy=(10, family_grounding[9]), xytext=(5.4, 0.66), arrowprops={"arrowstyle": "->"}, fontsize=8.5)
    axes[0, 1].annotate("74.2%", xy=(10, macro_f1[9]), xytext=(6.8, 0.57), arrowprops={"arrowstyle": "->"}, fontsize=8.5)
    axes[1, 0].annotate(f"{false_found_count[9]}/95 = 9.5%", xy=(10, false_found[9]), xytext=(5.0, 0.24), arrowprops={"arrowstyle": "->"}, fontsize=8.5)
    axes[1, 1].annotate(f"{coverage_count[9]}/400 = 35.5%", xy=(10, coverage[9]), xytext=(5.0, 0.56), arrowprops={"arrowstyle": "->"}, fontsize=8.5)

    fig.suptitle("F05 — Dev task metrics across checkpoint selection", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.013,
        "EXPLORATORY_DEV_ONLY · gray region is early-stopping patience · star is epoch 10 selected by dev loss, "
        "not by post-hoc best task metric · no between-run confidence interval (one seed)",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, FIGURES["F05"], "F05 — Development checkpoint task trajectory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace only this renderer's derived outputs")
    args = parser.parse_args()

    expected = [
        *(stem.with_suffix(ext) for stem in FIGURES.values() for ext in (".svg", ".png")),
        EPOCH_TABLE,
        OPTIMIZER_TABLE,
        SUPPLEMENT_REPORT,
        SUPPLEMENT_MANIFEST,
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise VisualizationError(f"Visualization output exists; pass --overwrite: {existing[0]}")

    configure_style()
    protected_before = {name: sha256_file(path) for name, path in PROTECTED_ARTIFACTS.items()}
    step_rows, epoch_rows, config, validation = load_inputs()
    selected_epoch = int(validation["selected_epoch"])
    selected_step = int(validation["selected_step"])
    clip_norm = float(config["optimization"]["gradient_clip_norm"])

    epoch_table_rows = epoch_visualization_rows(epoch_rows, selected_epoch)
    optimizer_table_rows = optimizer_epoch_rows(step_rows, clip_norm)
    write_csv(EPOCH_TABLE, list(epoch_table_rows[0]), epoch_table_rows)
    write_csv(OPTIMIZER_TABLE, list(optimizer_table_rows[0]), optimizer_table_rows)

    generated: list[Path] = [EPOCH_TABLE, OPTIMIZER_TABLE]
    generated.extend(plot_convergence(step_rows, epoch_rows, selected_step))
    generated.extend(plot_optimization_health(step_rows, optimizer_table_rows, selected_step, clip_norm))
    generated.extend(plot_checkpoint_task_trajectory(epoch_rows, selected_epoch))

    report = f"""# Development training visualization supplement

- Status: `{STATUS}` and `POST_TRAIN_RENDER_ONLY`.
- Source run: `pcra_u_development_train_20260824`.
- Selected checkpoint: epoch 10 / step 2,000, chosen strictly by minimum dev total loss (`1.351613`).
- Renderer activity: no model load, inference, optimizer step, calibration fit, or test access.

## Figures

### F03 — Convergence and checkpoint selection

![F03 convergence](figures/{FIGURES['F03'].name}.png)

Shows all 3,000 logged optimizer steps, a trailing {ROLLING_WINDOW}-step train mean, the 15 dev evaluations, the weighted objective decomposition, and the locked selection point. Train loss continued downward after epoch 10 while dev total loss worsened overall, so the patience region is useful evidence against selecting the final checkpoint automatically.

### F04 — Optimization health

![F04 optimization health](figures/{FIGURES['F04'].name}.png)

The global norm is logged before clipping. The q10–q90 ribbon is minibatch variability in a trailing {ROLLING_WINDOW}-step window, not a confidence interval. Crossing the configured threshold means clipping engaged; it is not by itself evidence of exploding gradients. Post-clip norms were not logged.

### F05 — Dev metric trajectory

![F05 checkpoint metrics](figures/{FIGURES['F05'].name}.png)

False-FOUND is displayed together with predicted-FOUND coverage, because a low false-FOUND rate can otherwise be obtained by rarely predicting `FOUND`. The star always denotes the checkpoint selected by the pre-registered dev-loss rule, not a post-hoc optimum for each panel.

## Denominators and interpretation limits

- Train: 320 families / 1,600 variants. Dev: 80 families / 400 dependent variants.
- Family-equal grounding uses 80 dev families; the supporting FOUND sample metric has 168 variants.
- False-FOUND uses 95 `AMBIGUOUS ∪ ABSENT` variants; coverage uses all 400 dev variants.
- This is one deterministic seed and repeated use of dev for checkpoint selection. There is no between-run confidence interval and these are not final test results.
- Calibration, Test-IID and Test-OOD remain sealed. Tables 2–6 remain `NOT_RUN`.
- The original `F02_development_loss_gradient.svg` is preserved as an engineering diagnostic; F03–F05 add numerical axes, dev trajectories and explicit denominators.
"""
    write_text(SUPPLEMENT_REPORT, report)
    generated.append(SUPPLEMENT_REPORT)

    protected_after = {name: sha256_file(path) for name, path in PROTECTED_ARTIFACTS.items()}
    if protected_before != protected_after:
        raise VisualizationError("A protected contract/checkpoint/freeze artifact changed during rendering")

    global_grad = np.asarray([float(row["gradient_norm_preclip"]) for row in step_rows])
    target_grad = np.asarray([float(row["target_head_gradient"]) for row in step_rows])
    source_paths = [LOG_PATH, CHECKPOINT_TABLE, CONFIG_PATH, BEST_POINTER, *sorted(EPOCH_ROOT.glob("development_epoch_*.json"))]
    manifest = {
        "schema_version": 1,
        "supplement_id": "pcra_u_development_training_visualization_v1",
        "created_at_utc": "2026-08-24T00:00:00+00:00",
        "scientific_status": f"{STATUS}_POST_TRAIN_RENDER_ONLY",
        "script": {"path": relative(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        "sources": [
            {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in source_paths
        ],
        "outputs": [
            {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(generated)
        ],
        "protected_artifacts": [
            {
                "name": name,
                "path": relative(PROTECTED_ARTIFACTS[name]),
                "sha256_before": protected_before[name],
                "sha256_after": protected_after[name],
                "unchanged": protected_before[name] == protected_after[name],
            }
            for name in sorted(PROTECTED_ARTIFACTS)
        ],
        "validation": {
            **validation,
            "optimizer_steps": len(step_rows),
            "epoch_evaluations": len(epoch_rows),
            "all_logged_steps_finite": all(bool(row["finite"]) for row in step_rows),
            "global_gradient_preclip_above_5_count": int(np.sum(global_grad > clip_norm)),
            "global_gradient_preclip_above_5_rate": float(np.mean(global_grad > clip_norm)),
            "global_gradient_preclip_max": float(global_grad.max()),
            "target_head_zero_gradient_steps": int(np.sum(target_grad == 0.0)),
        },
        "environment": {
            "python": platform.python_version(),
            "matplotlib": mpl.__version__,
            "numpy": np.__version__,
        },
        "claims": {
            "render_only": True,
            "derived_display_aggregation_only": True,
            "model_metrics_recomputed": False,
            "model_loaded": False,
            "inference": False,
            "optimizer_steps_executed": 0,
            "calibration_fit": False,
            "test_opened": False,
            "tables_02_to_06": "NOT_RUN",
            "protected_artifacts_unchanged": True,
        },
        "caveats": [
            "One deterministic training run/seed; no between-run confidence intervals.",
            "Dev was repeatedly evaluated for checkpoint selection, so all task trajectories are exploratory.",
            "The rolling q10-q90 band describes minibatch variability and is not a confidence interval.",
            "Global gradient norms are pre-clip; post-clip norms were not logged.",
            "Variants within the same family are not statistically independent observations.",
        ],
    }
    write_json(SUPPLEMENT_MANIFEST, manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "figures": len(FIGURES),
                "formats": ["png", "svg"],
                "selected_checkpoint": "epoch_010_step_000002000",
                "protected_unchanged": True,
                "calibration_test": "SEALED",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
