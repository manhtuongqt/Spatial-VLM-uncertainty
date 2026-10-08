#!/usr/bin/env python3
"""Render the focused RoboRefer + spatial-risk architecture figure."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "baocao" / "figures"
RGB_SAMPLE = ROOT / "datasets/Gazebo_calibration_v2/records/gazebo_calibration_v2_002/input/rgb.png"
DEPTH_SAMPLE = ROOT / "datasets/Gazebo_calibration_v2/records/gazebo_calibration_v2_002/input/depth_view.png"

INK = "#17202A"
MUTED = "#64717C"
BLUE = "#315F7D"
BLUE_FILL = "#EAF2F7"
GREEN = "#3F7355"
GREEN_FILL = "#EAF4ED"
ORANGE = "#A5682D"
ORANGE_FILL = "#FAF0E5"
GRAY = "#686D72"
GRAY_FILL = "#F2F3F4"
WHITE = "#FFFFFF"


def box(
    ax, x, y, w, h, title, body="", *, edge=BLUE, fill=WHITE,
    tag=None, title_size=9.0, body_size=6.6,
):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.09",
        linewidth=1.45, edgecolor=edge, facecolor=fill, zorder=3,
    )
    ax.add_patch(patch)
    if tag:
        ax.text(
            x + w - 0.10, y + h - 0.10, tag,
            ha="right", va="top", fontsize=6.0,
            fontweight="bold", color=edge, zorder=4,
        )
    title_y = y + h - (0.36 if tag else 0.24)
    ax.text(
        x + w / 2, title_y, title,
        ha="center", va="top", fontsize=title_size,
        fontweight="bold", color=INK, zorder=4,
    )
    if body:
        ax.text(
            x + w / 2, title_y - 0.42, body,
            ha="center", va="top", fontsize=body_size,
            color=INK, linespacing=1.22, zorder=4,
        )


def container(ax, x, y, w, h, title, subtitle, *, edge, fill):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.14",
        linewidth=1.7, edgecolor=edge, facecolor=fill,
        linestyle=(0, (5, 3)), alpha=0.68, zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.20, y + h - 0.20, title,
        ha="left", va="top", fontsize=11.2,
        fontweight="bold", color=edge, zorder=1,
    )
    ax.text(
        x + w - 0.20, y + h - 0.20, subtitle,
        ha="right", va="top", fontsize=7.0,
        fontweight="bold", color=edge, zorder=1,
    )


def arrow(ax, start, end, *, color=BLUE, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.5, color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2, zorder=2,
    ))


def image_card(ax, path, extent, label):
    x0, x1, y0, y1 = extent
    patch = FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        linewidth=1.25, edgecolor=BLUE, facecolor=WHITE, zorder=2,
    )
    ax.add_patch(patch)
    shown = ax.imshow(
        mpimg.imread(path),
        extent=(x0 + 0.05, x1 - 0.05, y0 + 0.24, y1 - 0.05),
        aspect="auto", zorder=3,
    )
    shown.set_clip_path(patch)
    ax.text(
        (x0 + x1) / 2, y0 + 0.08, label,
        ha="center", va="bottom", fontsize=7.0,
        fontweight="bold", color=BLUE, zorder=4,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=(17.2, 7.1))
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 7.1)
    ax.axis("off")

    ax.text(
        0.35, 6.78,
        "ROBOREFER BACKBONE WITH THE THESIS SPATIAL-RISK EXTENSION",
        fontsize=14, fontweight="bold", color=BLUE, va="center",
    )
    ax.text(
        16.85, 6.78,
        "Blue: original and frozen   ·   Green/orange: thesis extension",
        fontsize=7.8, color=MUTED, ha="right", va="center",
    )
    ax.plot([0.35, 16.85], [6.52, 6.52], color=BLUE, lw=1.05)

    # Inputs.
    ax.text(0.38, 6.14, "INPUT", fontsize=9.4, fontweight="bold", color=INK)
    image_card(ax, RGB_SAMPLE, (0.40, 2.15, 4.32, 5.82), "RGB image")
    image_card(ax, DEPTH_SAMPLE, (0.40, 2.15, 2.55, 4.05), "Depth map")
    box(
        ax, 0.40, 1.08, 1.75, 1.15,
        "Instruction", '"Select the second\nobject from the left"',
        edge=GRAY, fill=GRAY_FILL, title_size=8.4, body_size=6.4,
    )

    # Original RoboRefer backbone.
    container(
        ax, 2.62, 0.82, 6.55, 5.18,
        "ORIGINAL ROBOREFER-2B-SFT", "UNCHANGED + FROZEN",
        edge=BLUE, fill=BLUE_FILL,
    )
    box(
        ax, 2.98, 4.28, 1.62, 1.08,
        "RGB encoder", "SigLIP visual features",
        edge=BLUE, tag="FROZEN",
    )
    box(
        ax, 4.92, 4.28, 1.52, 1.08,
        "RGB projector", "features → LLM tokens",
        edge=BLUE, tag="FROZEN", title_size=8.3,
    )
    box(
        ax, 2.98, 2.63, 1.62, 1.08,
        "Depth encoder", "dedicated depth tower",
        edge=BLUE, tag="FROZEN",
    )
    box(
        ax, 4.92, 2.63, 1.52, 1.08,
        "Depth projector", "features → LLM tokens",
        edge=BLUE, tag="FROZEN", title_size=8.1,
    )
    box(
        ax, 2.98, 1.08, 3.46, 1.00,
        "Language tokens", "tokenized task instruction",
        edge=GRAY, tag="TEXT", title_size=8.5, body_size=6.2,
    )
    box(
        ax, 6.83, 2.18, 1.94, 2.48,
        "VLM / LLM", "Fuse RGB, depth and\nlanguage tokens\n\nSpatial grounding",
        edge=BLUE, fill="#DCEAF3", tag="FROZEN B0",
        title_size=9.3, body_size=6.7,
    )
    box(
        ax, 6.83, 1.18, 1.94, 0.70,
        "POINT [(x, y)]", "",
        edge=BLUE, tag="OUTPUT", title_size=8.4,
    )

    arrow(ax, (2.15, 5.06), (2.98, 4.82), color=BLUE)
    arrow(ax, (4.60, 4.82), (4.92, 4.82), color=BLUE)
    arrow(ax, (6.44, 4.82), (6.83, 4.05), color=BLUE, rad=0.08)
    arrow(ax, (2.15, 3.29), (2.98, 3.17), color=BLUE)
    arrow(ax, (4.60, 3.17), (4.92, 3.17), color=BLUE)
    arrow(ax, (6.44, 3.17), (6.83, 3.26), color=BLUE)
    arrow(ax, (2.15, 1.66), (2.98, 1.58), color=GRAY)
    arrow(ax, (6.44, 1.58), (6.83, 2.42), color=GRAY, rad=-0.08)
    arrow(ax, (7.80, 2.18), (7.80, 1.88), color=BLUE)

    # Thesis extension only: no training/evidence/status detours.
    container(
        ax, 9.60, 0.82, 7.20, 5.18,
        "THESIS EXTENSION", "ADDED AFTER ROBOREFER OUTPUT",
        edge=GREEN, fill=GREEN_FILL,
    )
    box(
        ax, 9.98, 4.08, 1.62, 1.28,
        "Inference outputs", "1 greedy result\n+ 3 stochastic draws",
        edge=GREEN, tag="ADDED", title_size=8.5,
    )
    box(
        ax, 11.98, 4.08, 1.62, 1.28,
        "Output parser", "POINT / ABSTAIN\n/ INVALID",
        edge=GREEN, tag="ADDED", title_size=8.5,
    )
    box(
        ax, 10.45, 1.82, 2.55, 1.48,
        "16-D spatial features",
        "output + stochastic behavior\npoint geometry + local depth\nrelation identity",
        edge=GREEN, tag="ADDED", title_size=8.8,
    )
    box(
        ax, 13.40, 1.82, 1.55, 1.48,
        "Risk estimator", "StandardScaler\n+ Logistic-L2",
        edge=GREEN, fill="#DCEDE2", tag="ADDED", title_size=8.4,
    )
    box(
        ax, 15.32, 1.82, 1.12, 1.48,
        "Decision", "low risk:\nPOINT\n\nhigh risk:\nABSTAIN",
        edge=ORANGE, fill=ORANGE_FILL, tag="OUTPUT", title_size=8.2, body_size=5.9,
    )
    box(
        ax, 14.02, 4.08, 2.42, 1.28,
        "Selective output", "POINT [(x, y)]  or  ABSTAIN",
        edge=ORANGE, fill=ORANGE_FILL, tag="RESULT", title_size=8.8,
    )

    arrow(ax, (8.77, 1.53), (9.98, 4.60), color=GREEN, rad=-0.27)
    arrow(ax, (11.60, 4.72), (11.98, 4.72), color=GREEN)
    arrow(ax, (12.79, 4.08), (11.76, 3.30), color=GREEN, rad=0.04)
    arrow(ax, (13.00, 2.56), (13.40, 2.56), color=GREEN)
    arrow(ax, (14.95, 2.56), (15.32, 2.56), color=ORANGE)
    arrow(ax, (15.88, 3.30), (15.35, 4.08), color=ORANGE, rad=-0.06)

    ax.text(
        16.72, 0.40,
        "Core idea: keep RoboRefer frozen and add a post-hoc risk estimator after its target-point output.",
        ha="right", va="center", fontsize=7.3,
        fontweight="bold", color=GREEN,
    )

    stem = OUT / "00_roborefer_thesis_architecture"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.10)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
