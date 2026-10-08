#!/usr/bin/env python3
"""Rebuild thesis figures from frozen Test-IID evidence, without model inference."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EVAL = ROOT / "new/test_iid/evaluation"
DATA = ROOT / "new/test_iid/dataset"
MODEL = ROOT / "new/outputs/pcrau_target_v2_full_seed_24082026"

BLUE = "#0072B2"  # RoboRefer, Okabe-Ito palette
ORANGE = "#D55E00"  # P-CRA-U
GREEN = "#009E73"  # ground truth
PURPLE = "#CC79A7"  # uncertainty / risk
GRAY = "#5C6770"
LIGHT = "#E6E9EC"
SOURCE_ORDER = ["semantic", "relation", "spatial", "depth", "occlusion"]
ACTION_COLORS = {"EXECUTE": GREEN, "REOBSERVE": BLUE, "ASK_USER": PURPLE, "ABSTAIN": ORANGE}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


manifest = read_json(ROOT / "new/test_iid/protocol/test_iid_eval_manifest.json")
entries = {row["sample_id"]: row for row in manifest["entries"]}
predictions = {row["sample_id"]: row for row in read_jsonl(EVAL / "best_v2/predictions.jsonl")}
decisions = {row["sample_id"]: row for row in read_jsonl(EVAL / "best_v2/decisions.jsonl")}
with (EVAL / "comparison_original_vs_best_v2/paired_samples.csv").open(newline="", encoding="utf-8") as handle:
    paired = {row["sample_id"]: row for row in csv.DictReader(handle)}
metrics = read_json(EVAL / "best_v2/full_metrics.json")
comparison = read_json(EVAL / "comparison_original_vs_best_v2/full_comparison.json")


def image_and_mask(sample_id: str):
    entry = entries[sample_id]
    rgb = np.asarray(Image.open(ROOT / entry["feature_input"]["rgb_path"]).convert("RGB"))
    mask_path = DATA / entry["supervision"]["target_mask_path"]
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    assert rgb.shape[:2] == mask.shape == (480, 640), sample_id
    return rgb, mask


def image_axis(ax, sample_id: str, title: str | None = None, mask: bool = False):
    rgb, target_mask = image_and_mask(sample_id)
    ax.imshow(rgb)
    if mask and target_mask.any():
        ax.contour(target_mask.astype(float), levels=[0.5], colors=[GREEN], linewidths=1.7)
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D1D6DA")
        spine.set_linewidth(0.6)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=6)
    return rgb, target_mask


def mark_points(ax, sample_id: str, b1=True, v2=True):
    row = paired[sample_id]
    if b1:
        ax.scatter(float(row["b1_x"]), float(row["b1_y"]), s=85, marker="o", facecolors="none", edgecolors=BLUE, linewidths=2.2, zorder=10)
        ax.scatter(float(row["b1_x"]), float(row["b1_y"]), s=14, marker="o", color=BLUE, zorder=11)
    if v2:
        ax.scatter(float(row["v2_x"]), float(row["v2_y"]), s=105, marker="x", color=ORANGE, linewidths=2.4, zorder=11)


def panel_label(ax, label: str):
    ax.text(-0.02, 1.02, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")


def save(fig, folder: str, basename: str):
    target = OUT / folder
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / f"{basename}.pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(target / f"{basename}.png", dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def legend_handles():
    return [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=BLUE, markeredgewidth=2, markersize=8, label="RoboRefer"),
        Line2D([], [], marker="x", linestyle="none", color=ORANGE, markeredgewidth=2, markersize=8, label="P-CRA-U"),
        Line2D([], [], linestyle="-", color=GREEN, linewidth=1.8, label="Target mask (GT)"),
    ]


def fig06_heatmap():
    sample_id = "v211iid_family_000127__clean"
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    image_axis(axes[0], sample_id, "(a) RGB observation")
    image_axis(axes[1], sample_id, "(b) Ground truth and points", mask=True)
    mark_points(axes[1], sample_id)
    image_axis(axes[2], sample_id, "(c) P-CRA-U target distribution", mask=True)
    grid = np.asarray(predictions[sample_id]["spatial"]["probability_grid"], dtype=float)
    assert grid.shape == (24, 32)
    resized = np.asarray(Image.fromarray(grid.astype(np.float32), mode="F").resize((640, 480), Image.Resampling.BILINEAR))
    layer = np.ma.masked_where(resized < 0.01 * grid.max(), resized)
    im = axes[2].imshow(layer, cmap="magma", norm=Normalize(vmin=0, vmax=grid.max()), alpha=0.70)
    mark_points(axes[2], sample_id, b1=False)
    fig.colorbar(im, ax=axes[2], shrink=0.78, pad=0.02, label="Probability per grid cell")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    save(fig, "06_target_heatmap", "fig06_target_heatmap_real_rgb")


def fig07_multimode():
    sample_id = "v211iid_family_000031__clean"
    pred = predictions[sample_id]
    grid = np.asarray(pred["spatial"]["probability_grid"], dtype=float)
    assert grid.shape == (24, 32)
    # Same 3x3 constant-zero neighborhood maximum, without a SciPy dependency.
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(grid, 1, mode="constant", constant_values=0), (3, 3)
    )
    peaks = grid == windows.max(axis=(-2, -1))
    yy, xx = np.where(peaks)
    ordered = sorted(zip(xx, yy), key=lambda p: grid[p[1], p[0]], reverse=True)
    selected = []
    for x, y in ordered:
        if all(abs(x - px) + abs(y - py) >= 3 for px, py in selected):
            selected.append((x, y))
        if len(selected) == 2:
            break
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    image_axis(axes[0], sample_id, "(a) Ambiguous RGB case", mask=True)
    mark_points(axes[0], sample_id, b1=False)
    ax = axes[1]
    im = ax.imshow(grid, cmap="magma", origin="upper", interpolation="nearest", vmin=0, vmax=grid.max())
    for i, (x, y) in enumerate(selected, 1):
        ax.scatter(x, y, s=115, marker="o", facecolor="none", edgecolor="white", linewidth=1.6)
    ax.set_title(f"(b) Spatial probability; reported modes = {pred['spatial']['mode_count']}", loc="left", fontweight="bold")
    ax.set_xlabel("Grid column (0–31)")
    ax.set_ylabel("Grid row (0–23)")
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(23.5, -0.5)
    fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02, label="Probability per grid cell")
    save(fig, "07_multimodal_distribution", "fig07_multimodal_spatial_distribution")


def fig08_source_uncertainty():
    cases = [
        ("v211iid_family_000127__clean", "Clean / no source label"),
        ("v211iid_family_000165__clean", "Ambiguous / semantic + relation + spatial*"),
        ("v211iid_family_000127__depth_corruption", "Depth corruption / depth"),
        ("v211iid_family_000094__occlusion_view_counterfactual", "Occlusion / occlusion"),
        ("v211iid_family_000127__semantic_counterfactual", "Semantic counterfactual / no source label"),
    ]
    fig = plt.figure(figsize=(10.4, 12.2), constrained_layout=True)
    spec = fig.add_gridspec(5, 2, width_ratios=[1.15, 1.65], hspace=0.12)
    for i, (sample_id, label) in enumerate(cases):
        ax_img = fig.add_subplot(spec[i, 0])
        image_axis(ax_img, sample_id, f"({chr(97+i)}) {label}")
        if i == 1:
            assert np.isclose(predictions[sample_id]["source_probabilities"]["spatial"],
                              0.9658995866775513, atol=1e-10, rtol=0)
            ax_img.text(0.02, 0.025, "000165 / clean", transform=ax_img.transAxes,
                        fontsize=7, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2})
        if i == 3:
            assert predictions[sample_id]["evaluation"]["source_multihot_5"] == [0, 0, 0, 0, 1]
            rgb, _ = image_and_mask(sample_id)
            # Magnify the same RGB pixels: the soup can occludes the orange cube.
            # No object is moved or composited, and archived predictions stay unchanged.
            ax_img.add_patch(Rectangle((175, 225), 170, 130, fill=False,
                                       edgecolor=BLUE, linewidth=1.2))
            zoom = ax_img.inset_axes([0.55, 0.55, 0.43, 0.43])
            zoom.imshow(rgb)
            zoom.set_xlim(175, 345)
            zoom.set_ylim(355, 225)
            zoom.set_xticks([])
            zoom.set_yticks([])
            for spine in zoom.spines.values():
                spine.set_edgecolor(BLUE)
                spine.set_linewidth(1.2)
            zoom.text(0.04, 0.94, "Zoom", va="top", transform=zoom.transAxes,
                      fontsize=6.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
            ax_img.text(0.02, 0.025, "000094 / occlusion", transform=ax_img.transAxes,
                        fontsize=7, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2})
        ax = fig.add_subplot(spec[i, 1])
        values = [predictions[sample_id]["source_probabilities"][source] for source in SOURCE_ORDER]
        truth = predictions[sample_id]["evaluation"]["source_multihot_5"]
        ypos = np.arange(5)
        bars = ax.barh(ypos, values, height=0.58, color=[PURPLE if truth[j] else GRAY for j in range(5)], alpha=0.86)
        for j, (bar, val) in enumerate(zip(bars, values)):
            ax.text(min(val + 0.02, 1.015), bar.get_y() + bar.get_height()/2, f"{val:.2f}", va="center", fontsize=8)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_yticks(ypos, [f"{name}{'  ●' if truth[j] else ''}" for j, name in enumerate(SOURCE_ORDER)])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.10)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        if i == 4:
            ax.set_xlabel("Predicted source probability")
        else:
            ax.set_xticklabels([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.text(0.55, -0.008, "● = positive source label; spatial* is a weak label derived from AMBIGUOUS. Dashed line = frozen 0.5 threshold.", ha="center", fontsize=8)
    save(fig, "08_source_uncertainty", "fig08_source_uncertainty_cases")


def test_risk_arrays():
    sample_ids = [row["sample_id"] for row in manifest["entries"]]
    risk = np.asarray([decisions[sid]["calibrated_grounding_risk"] for sid in sample_ids], dtype=float)
    error = np.asarray([predictions[sid]["evaluation"]["error_event"] for sid in sample_ids], dtype=float)
    assert len(risk) == 1000 and len(error) == 1000
    return risk, error


def fig10_reliability():
    risk, error = test_risk_arrays()
    bins = np.minimum((risk * 10).astype(int), 9)
    counts = np.array([(bins == i).sum() for i in range(10)])
    confidence = np.array([risk[bins == i].mean() for i in range(10)])
    observed = np.array([error[bins == i].mean() for i in range(10)])
    ece = np.sum(counts / len(risk) * np.abs(confidence - observed))
    assert np.isclose(ece, metrics["risk_calibration_and_ranking"]["ece_10"], atol=1e-10)
    fig = plt.figure(figsize=(6.2, 6.0), constrained_layout=True)
    spec = fig.add_gridspec(2, 1, height_ratios=[3.2, 0.85], hspace=0.06)
    ax = fig.add_subplot(spec[0])
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.2, label="Ideal calibration")
    ax.plot(confidence, observed, color=PURPLE, linewidth=1.2, alpha=0.65)
    ax.scatter(confidence, observed, s=30 + counts * 0.18, color=PURPLE, edgecolors="white", linewidths=0.8, zorder=3, label="10 equal-width bins")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel("Observed error rate")
    ax.set_title(f"Test-IID risk reliability  ·  N=1,000  ·  ECE={ece:.4f}", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax2 = fig.add_subplot(spec[1], sharex=ax)
    ax2.bar(np.arange(10)/10 + 0.05, counts, width=0.085, color=LIGHT, edgecolor=GRAY, linewidth=0.5)
    ax2.set_ylabel("Samples")
    ax2.set_xlabel("Predicted grounding risk")
    ax2.set_xlim(0, 1)
    ax2.set_xticks(np.arange(0, 1.01, 0.2))
    plt.setp(ax.get_xticklabels(), visible=False)
    save(fig, "10_reliability", "fig10_reliability_test_iid")


def fig11_risk_coverage():
    risk, error = test_risk_arrays()
    order = np.argsort(risk, kind="stable")
    coverage = np.arange(1, len(risk)+1) / len(risk)
    selective_risk = np.cumsum(error[order]) / np.arange(1, len(error)+1)
    aurc = selective_risk.mean()
    reported = metrics["risk_calibration_and_ranking"]
    assert np.isclose(aurc, reported["aurc"], atol=1e-10)
    threshold = reported["frozen_threshold"]
    accepted = risk <= threshold
    assert accepted.sum() == reported["accepted_samples"]
    x = accepted.mean()
    y = error[accepted].mean()
    fig, ax = plt.subplots(figsize=(7.0, 4.9), constrained_layout=True)
    ax.plot(coverage, selective_risk, color=PURPLE, linewidth=2.0, label="P-CRA-U calibrated risk")
    ax.axhline(error.mean(), color=GRAY, linestyle="--", linewidth=1, label=f"All-sample error = {error.mean():.1%}")
    ax.scatter([x], [y], s=75, color=ORANGE, zorder=5)
    ax.annotate(f"Frozen threshold\ncoverage {x:.1%}, risk {y:.2%}", xy=(x,y), xytext=(x+0.09, y+0.16),
                arrowprops={"arrowstyle":"-", "color":ORANGE, "lw":1}, color=ORANGE, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Coverage (accepted fraction of all 1,000 samples)")
    ax.set_ylabel("Selective error rate")
    ax.set_title(f"Test-IID risk–coverage  ·  AURC={aurc:.4f}", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "11_risk_coverage", "fig11_risk_coverage_test_iid")


def region_intersects_mask(sample_id: str) -> bool:
    _, mask = image_and_mask(sample_id)
    for x, y in decisions[sample_id]["confidence_region"]["grid_cells_xy"]:
        if mask[y*20:(y+1)*20, x*20:(x+1)*20].any():
            return True
    return False


def fig12_confidence_region():
    cases = [
        ("v211iid_family_000127__clean", "Region intersects target"),
        ("v211iid_family_000002__clean", "Region misses target"),
    ]
    assert region_intersects_mask(cases[0][0]) and not region_intersects_mask(cases[1][0])
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.3), constrained_layout=True)
    for i, (sample_id, label) in enumerate(cases):
        image_axis(axes[i, 0], sample_id, f"({chr(97+2*i)}) RGB + target mask", mask=True)
        image_axis(axes[i, 1], sample_id, f"({chr(98+2*i)}) {label}", mask=True)
        for x, y in decisions[sample_id]["confidence_region"]["grid_cells_xy"]:
            axes[i, 1].add_patch(Rectangle((x*20,y*20),20,20,facecolor=PURPLE,edgecolor="white",linewidth=0.5,alpha=0.48))
        mark_points(axes[i, 1], sample_id, b1=False)
        axes[i, 1].text(0.02, 0.03, sample_id, transform=axes[i, 1].transAxes, color="black", fontsize=7,
                        bbox={"facecolor":"white", "edgecolor":"none", "alpha":0.84, "pad":2})
    fig.legend(handles=[Patch(facecolor=PURPLE, alpha=0.48, label="Highest-density grid region"),
                        Line2D([],[],color=GREEN,linewidth=1.8,label="Target mask (GT)"),
                        Line2D([],[],marker="x",linestyle="none",color=ORANGE,markersize=8,label="P-CRA-U MAP point")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5,-0.04))
    save(fig, "12_confidence_region", "fig12_confidence_region_true_rgb")


def fig14_decisions():
    cases = [
        ("v211iid_family_000127__clean", "EXECUTE"),
        ("v211iid_family_000003__depth_corruption", "REOBSERVE"),
        ("v211iid_family_000031__clean", "ASK_USER"),
        ("v211iid_family_000005__clean", "ABSTAIN"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.7), constrained_layout=True)
    for i, ((sample_id, action), ax) in enumerate(zip(cases, axes.ravel())):
        decision = decisions[sample_id]
        pred = predictions[sample_id]
        assert decision["action"] == action
        image_axis(ax, sample_id, f"({chr(97+i)}) {action}", mask=True)
        if action == "EXECUTE":
            mark_points(ax, sample_id, b1=False)
        state = pred["evaluation"]["answerability_state"]
        predicted = max(pred["answerability_probabilities"], key=pred["answerability_probabilities"].get)
        label = f"Truth: {state}\nPred: {predicted}  ·  risk: {decision['calibrated_grounding_risk']:.3f}\n{sample_id}"
        ax.text(0.02, 0.025, label, transform=ax.transAxes, va="bottom", fontsize=7.4, linespacing=1.25,
                bbox={"facecolor":"white", "edgecolor":ACTION_COLORS[action], "alpha":0.90,"pad":4})
    fig.legend(handles=[Line2D([],[],color=GREEN,linewidth=1.8,label="Target mask (GT)"),
                        Line2D([],[],marker="x",linestyle="none",color=ORANGE,markersize=8,label="Executed P-CRA-U point")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5,-0.035))
    save(fig, "14_decision_cases", "fig14_four_policy_decisions")


def fig15_failures():
    cases = [
        ("v211iid_family_000007__clean", "P-CRA-U hit · RoboRefer miss"),
        ("v211iid_family_000002__clean", "RoboRefer hit · P-CRA-U miss"),
        ("v211iid_family_000165__relation_counterfactual", "Both miss"),
    ]
    assert [(paired[sid]["b1_hit"], paired[sid]["v2_hit"]) for sid,_ in cases] == [("0","1"),("1","0"),("0","0")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    for i, ((sample_id, label), ax) in enumerate(zip(cases, axes)):
        image_axis(ax, sample_id, f"({chr(97+i)}) {label}", mask=True)
        mark_points(ax, sample_id)
        row = paired[sample_id]
        ax.text(0.02, 0.025, f"{row['target_object_group']} · {row['relation']}\n{sample_id}",
                transform=ax.transAxes, va="bottom", fontsize=7.4,
                bbox={"facecolor":"white","edgecolor":"none","alpha":0.88,"pad":3})
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5,-0.05))
    save(fig, "15_failure_cases", "fig15_paired_success_and_failure")


def strata_figure(key: str, folder: str, basename: str, title: str, order: list[str] | None = None):
    strata = comparison["stratified_target_present_grounding"][key]
    names = order or sorted(strata)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(8.5, n * 1.4), 5.4), constrained_layout=True)
    positions = np.arange(n)
    width = 0.37
    label_map = {
        "between_in_depth": "between\nin depth",
        "nearer_than_both": "nearer than\nboth",
        "semantic_counterfactual": "semantic\ncounterfactual",
        "relation_counterfactual": "relation\ncounterfactual",
        "depth_corruption": "depth\ncorruption",
        "occlusion_view_counterfactual": "occlusion/view\ncounterfactual",
    }
    for field, offset, color, label in [
        ("b1_hit", -width / 2, BLUE, "RoboRefer"),
        ("v2_hit", width / 2, ORANGE, "P-CRA-U"),
    ]:
        estimates = []
        for name in names:
            value = strata[name]["family_cluster_bootstrap_95ci"][field]
            estimates.append(value["estimate"] * 100)
        bars = ax.bar(positions + offset, estimates, width, color=color, label=label, zorder=2)
        for bar, value in zip(bars, estimates):
            ax.text(bar.get_x() + bar.get_width() / 2, value - 3.0, f"{value:.1f}",
                    ha="center", va="top", color="white", fontsize=7.5,
                    fontweight="bold")
    labels = [f"{label_map.get(name, name.replace('_', ' '))}\n(n={strata[name]['samples']})" for name in names]
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Point-in-target accuracy (%)")
    ax.set_title(title, loc="left", fontweight="bold", pad=13)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2, frameon=False)
    ax.grid(axis="y", color=LIGHT, linewidth=0.7, zorder=0)
    ax.tick_params(axis="x", length=0)
    save(fig, folder, basename)


def fig19_answerability_confusion():
    matrix = np.asarray(metrics["answerability"]["confusion_matrix"], dtype=int)
    assert matrix.shape == (4, 4) and matrix.sum() == 1000
    support = matrix.sum(axis=1)
    assert support.tolist() == [420, 105, 132, 343]
    percentages = 100 * matrix / support[:, None]
    labels = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT\nEVIDENCE"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    fig.suptitle("P-CRA-U answerability · Test-IID", fontsize=13, fontweight="bold")
    for ax, values, normalized, title in [
        (axes[0], matrix, False, "(a) Sample counts"),
        (axes[1], percentages, True, "(b) Row-normalized percentages"),
    ]:
        maximum = 100 if normalized else int(matrix.max())
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=maximum)
        ax.set_xticks(np.arange(4), labels, fontsize=8)
        ax.set_yticks(np.arange(4), [f"{label}\n(n={n})" for label, n in zip(labels, support)],
                      fontsize=8)
        ax.set_xlabel("Predicted answerability")
        ax.set_ylabel("Ground-truth answerability")
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
        ax.tick_params(axis="both", length=0)
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        for y in range(4):
            for x in range(4):
                value = values[y, x]
                label = f"{value:.1f}%" if normalized else str(int(value))
                ax.text(x, y, label, ha="center", va="center",
                        color="white" if value > maximum * 0.5 else "#20252B",
                        fontsize=10, fontweight="bold" if x == y else "normal")
        fig.colorbar(im, ax=ax, shrink=0.80, pad=0.025,
                     label="Percentage within true class (%)" if normalized else "Samples")
    save(fig, "19_answerability_confusion", "fig19_answerability_confusion_test_iid")


def fig20_training_curves():
    history = read_jsonl(MODEL / "history.jsonl")
    selected = read_json(MODEL / "best.json")
    best_epoch = int(selected["epoch"])
    epochs = np.asarray([row["epoch"] for row in history], dtype=int)
    assert epochs.tolist() == list(range(20)) and best_epoch == 13
    train_loss = np.asarray([row["train"]["total"] for row in history], dtype=float)
    dev_loss = np.asarray([row["dev"]["loss"]["total"] for row in history], dtype=float)
    dev_grounding = np.asarray([row["dev"]["grounding_accuracy"] for row in history]) * 100
    assert int(epochs[np.argmin(dev_loss)]) == best_epoch
    assert history[best_epoch]["global_step"] == 1120
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    fig.suptitle("P-CRA-U training · checkpoint selected by dev total loss",
                 fontsize=12, fontweight="bold")
    axes[0].plot(epochs, train_loss, color=BLUE, marker="o", markersize=3,
                 linewidth=1.7, label="Train total loss")
    axes[0].plot(epochs, dev_loss, color=ORANGE, marker="s", markersize=3,
                 linewidth=1.7, label="Dev total loss")
    axes[0].set_title("(a) Multi-task loss", loc="left", fontweight="bold")
    axes[0].set_ylabel("Total loss")
    axes[0].set_ylim(0, float(max(train_loss.max(), dev_loss.max())) * 1.08)
    axes[0].annotate(f"Selected dev loss = {dev_loss[best_epoch]:.4f}",
                     xy=(best_epoch, dev_loss[best_epoch]), xytext=(7.2, 3.2),
                     arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1},
                     color=ORANGE, fontsize=9)
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].plot(epochs, dev_grounding, color=GREEN, marker="o", markersize=3,
                 linewidth=1.7, label="Dev point-in-target")
    axes[1].set_title("(b) Dev grounding · n=335 target-present samples",
                      loc="left", fontweight="bold", fontsize=9.5)
    axes[1].set_ylabel("Point-in-target accuracy (%)")
    axes[1].set_ylim(35, 103)
    axes[1].annotate(f"{dev_grounding[best_epoch]:.2f}% at epoch {best_epoch}",
                     xy=(best_epoch, dev_grounding[best_epoch]), xytext=(8.3, 70),
                     arrowprops={"arrowstyle": "-", "color": GREEN, "lw": 1},
                     color=GREEN, fontsize=9)
    axes[1].legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.axvline(best_epoch, color=GRAY, linestyle=":", linewidth=1.2, zorder=0)
        ax.set_xlabel("Epoch (zero-based; epoch 13 is the 14th training pass)")
        ax.set_xlim(-0.4, 19.4)
        ax.set_xticks([0, 3, 6, 9, 13, 16, 19])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    save(fig, "20_training_curves", "fig20_pcrau_training_curves")


def fig21_source_delta():
    """Pair all four variants with clean within each of the 200 families."""
    families = {}
    for row in predictions.values():
        families.setdefault(row["family_id"], {})[row["variant"]] = row
    assert len(families) == 200
    variants = [
        "semantic_counterfactual", "relation_counterfactual",
        "depth_corruption", "occlusion_view_counterfactual",
    ]
    corresponding = [0, 1, 3, 4]
    deltas = []
    increased = []
    for variant, source_index in zip(variants, corresponding):
        values = np.asarray([
            [members[variant]["source_probabilities"][source] -
             members["clean"]["source_probabilities"][source]
             for source in SOURCE_ORDER]
            for _, members in sorted(families.items())
        ])
        assert values.shape == (200, 5)
        deltas.append(values.mean(axis=0))
        increased.append(int((values[:, source_index] > 0).sum()))
    delta = np.asarray(deltas)
    # Cross-check the rounded values already reported in Chapter 5.
    expected = np.asarray([
        [0.2202, -0.2089, -0.1823, -0.2214, -0.3297],
        [0.0561, 0.0723, -0.1776, -0.1192, -0.1439],
        [0.0950, 0.0155, 0.0079, 0.4304, -0.0890],
        [-0.0398, -0.0417, -0.0025, -0.0104, 0.1860],
    ])
    assert np.allclose(delta, expected, atol=0.000051, rtol=0)
    assert increased == [162, 130, 194, 176]
    names = ["Semantic CF", "Relation CF", "Depth corruption", "Occlusion / view CF"]
    fig, ax = plt.subplots(figsize=(9.5, 4.9), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    im = ax.imshow(delta, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(5), ["Semantic", "Relation", "Spatial*", "Depth", "Occlusion"])
    ax.set_yticks(np.arange(4), [
        f"{name}\nmatched-source increase: {count}/200"
        for name, count in zip(names, increased)
    ])
    ax.set_xlabel("Predicted uncertainty source")
    ax.set_ylabel("Counterfactual variant paired with clean")
    ax.set_title("P-CRA-U · paired source probability changes · 200 families",
                 loc="left", fontweight="bold", pad=12)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for y in range(4):
        for x in range(5):
            value = delta[y, x]
            ax.text(x, y, f"{value:+.4f}", ha="center", va="center",
                    color="white" if abs(value) > 0.27 else "#20252B",
                    fontsize=10, fontweight="bold" if x == corresponding[y] else "normal")
        ax.add_patch(Rectangle((corresponding[y]-0.46, y-0.44), 0.92, 0.88,
                               facecolor="none", edgecolor="#20252B", linewidth=1.3))
    fig.colorbar(im, ax=ax, shrink=0.90, pad=0.025,
                 label="Mean probability difference (variant − clean)")
    fig.text(0.5, -0.045,
             "Outlined cells: source matching the variant. Spatial* is weakly supervised.\n"
             "Probability differences, not accuracy; descriptive results without confidence intervals.",
             ha="center", fontsize=8)
    save(fig, "21_source_delta", "fig21_pcrau_paired_source_delta")
    print("Source delta means:", delta.tolist())


def average_ranks(values):
    """One-based average ranks, retaining ties (including zero distances)."""
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranked = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranked


def fig22_uncertainty_localization():
    """Describe localization only on true FOUND, without refitting any score."""
    sample_ids = sorted(sid for sid, pred in predictions.items()
                        if pred["evaluation"]["answerability_state"] == "FOUND")
    assert len(sample_ids) == 420
    distance = np.asarray([float(paired[sid]["v2_distance_px"]) for sid in sample_ids])
    assert np.isfinite(distance).all() and (distance >= 0).all()
    miss = distance > 0
    assert miss.sum() == 15
    assert all(bool(miss[i]) == (paired[sid]["v2_hit"] == "0")
               for i, sid in enumerate(sample_ids))
    features = [
        ("Normalized spatial entropy",
         [predictions[sid]["spatial"]["entropy_normalized"] for sid in sample_ids]),
        ("1 − spatial peak margin",
         [1 - predictions[sid]["spatial"]["peak_margin"] for sid in sample_ids]),
        ("Calibrated risk",
         [decisions[sid]["calibrated_grounding_risk"] for sid in sample_ids]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.7), sharey=True, constrained_layout=True)
    fig.suptitle("P-CRA-U · uncertainty vs localization error · true FOUND only (n=420)",
                 fontsize=12, fontweight="bold")
    for index, (ax, (name, values)) in enumerate(zip(axes, features)):
        values = np.asarray(values, dtype=float)
        pearson = float(np.corrcoef(values, distance)[0, 1])
        spearman = float(np.corrcoef(average_ranks(values), average_ranks(distance))[0, 1])
        ax.scatter(values[~miss], distance[~miss], s=17, color=BLUE, alpha=0.30,
                   edgecolors="none", label="Hit: distance = 0 (405)")
        ax.scatter(values[miss], distance[miss], s=32, color=ORANGE, alpha=0.90,
                   marker="x", linewidths=1.1, label="Miss: distance > 0 (15)")
        ax.set_title(f"({chr(97+index)}) {name}", loc="left", fontweight="bold", fontsize=9.5)
        ax.text(0.96, 0.22, f"Pearson r = {pearson:+.3f}\nSpearman ρ = {spearman:+.3f}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 3})
        ax.set_xlim(-0.025, 1.025)
        ax.set_xlabel(name + " [0, 1]")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(-0.10, float(distance.max()) * 1.6)
        ax.set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
        ax.grid(axis="y", color=LIGHT, linewidth=0.7)
        ax.set_axisbelow(True)
        print(f"FOUND-only {name}: Pearson={pearson:.9f}, Spearman={spearman:.9f}")
    axes[0].set_ylabel("Distance from MAP to target mask (px; symlog)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.055),
               ncol=2, frameon=False, markerscale=1.2)
    fig.text(0.5, -0.105,
             "All 420 samples retained; 405 zero-distance points overlap. No jitter or outlier removal.\n"
             "Pearson uses raw pixel distance; Spearman uses average ranks for ties. Descriptive, no p-values.",
             ha="center", fontsize=8)
    save(fig, "22_uncertainty_localization", "fig22_pcrau_uncertainty_localization")


def fig23_coverage_area():
    """Sweep mass descriptively, preserving the evaluator score event."""
    sample_ids = sorted(sid for sid, pred in predictions.items()
                        if pred["evaluation"]["target_exists"])
    reported = metrics["conformal_spatial_region"]
    frozen_mass = float(reported["frozen_probability_mass"])
    masses = np.unique(np.r_[np.linspace(0.01, 0.99, 199), frozen_mass, 1.0])
    assert len(masses) == 201
    required = np.asarray([predictions[sid]["evaluation"]["target_required_hd_mass"]
                           for sid in sample_ids], dtype=float)
    areas, intersections = [], []
    tied_order_differences = []
    assert len(sample_ids) == 835
    for sid in sample_ids:
        grid = np.asarray(predictions[sid]["spatial"]["probability_grid"], dtype=float)
        assert grid.shape == (24, 32) and np.isclose(grid.sum(), 1, atol=1e-6)
        order = np.argsort(-grid, axis=None)  # Same ordering as highest_density_region.
        cumulative = np.cumsum(grid.ravel()[order])
        counts = np.minimum(grid.size, np.searchsorted(cumulative, masses, side="left") + 1)
        mask = np.asarray(Image.open(DATA / entries[sid]["supervision"]["target_mask_path"])
                          .convert("L")) > 0
        assert mask.shape == (480, 640) and mask.any()
        # Any mask pixel in each 20x20 cell: same nonzero support as INTER_AREA downsampling.
        grid_mask = mask.reshape(24, 20, 32, 20).any(axis=(1, 3))
        first_rank = int(np.flatnonzero(grid_mask.ravel()[order])[0])
        stored_score = predictions[sid]["evaluation"]["target_required_hd_mass"]
        if not np.isclose(cumulative[first_rank], stored_score, atol=1e-6, rtol=0):
            # The evaluator used torch.argsort; the policy uses NumPy argsort.
            # Equal cell probabilities can yield a different first target rank.
            # Preserve the stored score and validate that this is a tied boundary.
            candidates = np.flatnonzero(np.isclose(cumulative, stored_score, atol=1e-6, rtol=0))
            assert len(candidates) and any(
                grid.ravel()[order[k]] == grid.ravel()[order[first_rank]]
                for k in candidates
            ), sid
            tied_order_differences.append(sid)
        areas.append(counts / grid.size)
        intersections.append(counts > first_rank)
        frozen_index = int(np.flatnonzero(masses == frozen_mass)[0])
        assert int(counts[frozen_index]) == len(decisions[sid]["confidence_region"]["grid_cells_xy"])
        stored_cells = decisions[sid]["confidence_region"]["grid_cells_xy"]
        reconstructed_cells = set(int(cell) for cell in order[:counts[frozen_index]])
        assert reconstructed_cells == {y * 32 + x for x, y in stored_cells}
        stored_intersection = any(grid_mask[y, x] for x, y in stored_cells)
        assert bool(counts[frozen_index] > first_rank) == stored_intersection
    area_percent = np.mean(areas, axis=0) * 100
    direct_coverage = np.mean(intersections, axis=0) * 100
    score_coverage = np.mean(required[:, None] <= masses[None, :], axis=0) * 100
    frozen_index = int(np.flatnonzero(masses == frozen_mass)[0])
    frozen_area = area_percent[frozen_index]
    frozen_score = score_coverage[frozen_index]
    frozen_direct = direct_coverage[frozen_index]
    assert np.isclose(frozen_score / 100, reported["empirical_coverage"], atol=1e-12)
    assert np.isclose(frozen_area / 100, reported["mean_grid_area_fraction"], atol=1e-12)
    assert (np.diff(area_percent) >= -1e-10).all()
    assert (np.diff(score_coverage) >= -1e-10).all()
    assert (np.diff(direct_coverage) >= -1e-10).all()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), constrained_layout=True)
    fig.suptitle("P-CRA-U · spatial region coverage–area · 835 target-present samples",
                 fontsize=12, fontweight="bold")
    for ax, x in zip(axes, [masses, area_percent]):
        ax.plot(x, score_coverage, color=PURPLE, linewidth=1.8,
                label="Evaluator score event: S ≤ mass")
        ax.plot(x, direct_coverage, color=BLUE, linewidth=1.5, linestyle="--",
                label="Direct region–mask intersection")
        ax.axhline(90, color=GRAY, linewidth=0.9, linestyle=":", label="Nominal coverage: 90%")
        frozen_x = frozen_mass if ax is axes[0] else frozen_area
        ax.scatter([frozen_x], [frozen_score], color=ORANGE, s=55, zorder=5)
        ax.scatter([frozen_x], [frozen_direct], facecolor="white", edgecolor=BLUE,
                   s=42, linewidth=1.4, zorder=5)
        ax.set_ylim(0, 103)
        ax.set_ylabel("Empirical coverage (%)")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    axes[0].set_title("(a) Coverage as probability mass increases", loc="left", fontweight="bold")
    axes[0].set_xlabel("Requested highest-density probability mass")
    axes[0].set_xlim(0, 1.01)
    axes[0].axvline(frozen_mass, color=ORANGE, linewidth=1, linestyle=":")
    axes[0].annotate(f"Frozen mass = {frozen_mass:.4f}\n"
                     f"score: {frozen_score:.2f}%\nintersection: {frozen_direct:.2f}%",
                     xy=(frozen_mass, frozen_score), xytext=(0.67, 24),
                     arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1},
                     color=ORANGE, fontsize=9)
    axes[1].set_title("(b) Coverage versus mean region area", loc="left", fontweight="bold")
    axes[1].set_xlabel("Mean grid area (% of 768 cells; logarithmic axis)")
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.1, 110)
    axes[1].set_xticks([0.1, 0.5, 1, 5, 10, 100], ["0.1", "0.5", "1", "5", "10", "100"])
    axes[1].annotate(f"Frozen mean area = {frozen_area:.4f}%",
                     xy=(frozen_area, frozen_score), xytext=(0.85, 53),
                     arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1},
                     color=ORANGE, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.075),
               ncol=3, frameon=False, fontsize=8)
    fig.text(0.5, -0.115, "Descriptive test sweep only: the calibration-fitted mass is unchanged.",
             ha="center", fontsize=8)
    save(fig, "23_coverage_area", "fig23_pcrau_coverage_area")
    print(f"Region frozen point: mass={frozen_mass:.9f}, mean_area_percent={frozen_area:.9f}, "
          f"score_coverage_percent={frozen_score:.9f}, intersection_percent={frozen_direct:.9f}")
    print(f"Tied torch/NumPy order differences: {len(tied_order_differences)} samples; "
          "evaluator scores preserved unchanged.")


def main():
    assert len(entries) == len(predictions) == len(decisions) == len(paired) == 1000
    fig06_heatmap()
    fig07_multimode()
    fig08_source_uncertainty()
    fig10_reliability()
    fig11_risk_coverage()
    fig12_confidence_region()
    fig14_decisions()
    fig15_failures()
    strata_figure("target_object_group", "16_stratified_object", "fig16_grounding_by_object",
                  "Grounding by target object group", ["fruit","container","mug","box","cube"])
    strata_figure("relation", "17_stratified_relation", "fig17_grounding_by_relation",
                  "Grounding by relation", ["direct","left_of","right_of","front_of","behind","nearer_than","farther_than","between_in_depth","nearer_than_both"])
    strata_figure("variant", "18_stratified_variant", "fig18_grounding_by_variant",
                  "Grounding by Test-IID variant", ["clean","semantic_counterfactual","relation_counterfactual","depth_corruption","occlusion_view_counterfactual"])
    fig19_answerability_confusion()
    fig20_training_curves()
    fig21_source_delta()
    fig22_uncertainty_localization()
    fig23_coverage_area()
    print("Generated 16 figures (PNG + PDF) from frozen predictions and training history; no training or inference.")


if __name__ == "__main__":
    main()
