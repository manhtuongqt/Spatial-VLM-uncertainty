#!/usr/bin/env python3
"""Render the architecture-only Spatial-VLM selective manipulation pipeline."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "baocao/figures"

NAVY = "#294F68"
BLUE_FILL = "#EAF1F5"
GREEN = "#466B57"
GREEN_FILL = "#EDF4EF"
ORANGE = "#9A6536"
ORANGE_FILL = "#F7F0E9"
GRAY = "#686868"
GRAY_FILL = "#F2F2F2"
BLACK = "#202020"
WHITE = "#FFFFFF"


def box(ax, x, y, w, h, title, body, *, edge=NAVY, fill=BLUE_FILL,
        subtitle=None, linestyle="-"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.35, edgecolor=edge, facecolor=fill, linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(x + w - 0.10, y + h - 0.13, subtitle, ha="right", va="top",
                fontsize=6.8, fontweight="bold", color=edge, zorder=3)
    ax.text(x + 0.13, y + h - 0.28, title, ha="left", va="top",
            fontsize=9.4, fontweight="bold", color=BLACK, zorder=3)
    ax.text(x + 0.13, y + h - 0.65, body, ha="left", va="top",
            fontsize=7.5, color=BLACK, linespacing=1.23, zorder=3)
    return patch


def arrow(ax, start, end, *, color=NAVY, dashed=False, width=1.35,
          rad=0.0, label=None, label_xy=None):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, linewidth=width,
        color=color, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=1,
    ))
    if label and label_xy:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=7.0,
                color=color, bbox=dict(fc=WHITE, ec="none", pad=0.8), zorder=4)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(16, 8.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.8)
    ax.axis("off")

    ax.text(0.35, 8.50, "SPATIAL-VLM SELECTIVE MANIPULATION ARCHITECTURE",
            fontsize=13, fontweight="bold", color=NAVY, va="center")
    ax.text(15.65, 8.50, "RGB-D + language → grounding → risk → policy → UR3",
            fontsize=8.2, color=GRAY, va="center", ha="right")
    ax.plot([0.35, 15.65], [8.24, 8.24], color=NAVY, lw=1.0)

    # Main perception and uncertainty path.
    modules = {
        "input": (0.35, 5.55, 1.95, 1.65),
        "vlm": (2.75, 5.55, 2.05, 1.65),
        "parser": (5.25, 5.55, 2.05, 1.65),
        "features": (7.75, 5.55, 2.15, 1.65),
        "risk": (10.35, 5.55, 2.05, 1.65),
        "cal": (12.85, 5.55, 2.45, 1.65),
    }

    box(ax, *modules["input"], "Multimodal observation",
        "RGB image\nMetric depth + intrinsics\nLanguage instruction",
        edge=NAVY, fill=BLUE_FILL, subtitle="INPUT")
    box(ax, *modules["vlm"], "RoboRefer grounding",
        "Vision-language encoding\nAutoregressive decoding\nGreedy + stochastic outputs",
        edge=NAVY, fill=BLUE_FILL, subtitle="FROZEN B0")
    box(ax, *modules["parser"], "Output parser",
        "POINT $(x,y)$ / ABSTAIN / INVALID\nSemantic normalized point\n3 stochastic point hypotheses",
        edge=GRAY, fill=GRAY_FILL, subtitle="DETERMINISTIC")
    box(ax, *modules["features"], "Spatial feature encoder",
        "Output and self-consistency\nPoint / boundary geometry\nLocal RGB-D + relation context",
        edge=GREEN, fill=GREEN_FILL, subtitle="16-D VECTOR")
    box(ax, *modules["risk"], "Spatial-risk estimator",
        "StandardScaler\nLogistic regression L2\n$r_{raw}=\\sigma(\\beta_0+\\beta^Tz)$",
        edge=GREEN, fill=GREEN_FILL, subtitle="ANSWERABILITY")
    box(ax, *modules["cal"], "Probability calibrator",
        "Monotone score mapping\n$p_{unsafe}=\\sigma(\\mathrm{logit}(r_{raw})/T)$\nOutput constrained to $[0,1]$",
        edge=GREEN, fill=GREEN_FILL, subtitle="CALIBRATION")

    order = ["input", "vlm", "parser", "features", "risk", "cal"]
    for left_name, right_name in zip(order[:-1], order[1:]):
        x1, y1, w1, h1 = modules[left_name]
        x2, y2, _, h2 = modules[right_name]
        arrow(ax, (x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2))

    # Auxiliary feature inputs: metric depth and relation identity.
    arrow(ax, (1.30, 5.55), (8.25, 5.55), color=GRAY, dashed=True,
          rad=-0.23, label="depth patch + camera geometry", label_xy=(5.0, 4.72))
    arrow(ax, (1.75, 7.20), (9.15, 7.20), color=GRAY, dashed=True,
          rad=0.15, label="relation token", label_xy=(5.7, 7.76))

    ax.text(0.35, 4.22, "SELECTIVE CONTROL AND ROBOT EXECUTION",
            fontsize=10.5, fontweight="bold", color=NAVY, va="center")
    ax.plot([0.35, 15.65], [3.98, 3.98], color=NAVY, lw=1.0)

    control = {
        "policy": (6.55, 1.78, 2.65, 1.55),
        "projection": (9.70, 1.78, 2.10, 1.55),
        "planner": (12.30, 1.78, 1.55, 1.55),
        "robot": (14.35, 1.78, 1.30, 1.55),
    }
    box(ax, *control["policy"], "Selective controller",
        "$p<T_1$: ACT\n$T_1\\leq p<T_2$: REOBSERVE\n$p\\geq T_2$ or budget exhausted: ABSTAIN",
        edge=ORANGE, fill=ORANGE_FILL, subtitle="POLICY")
    box(ax, *control["projection"], "Pixel-to-3D",
        "Depth lookup / local robust depth\nCamera intrinsics\nCamera frame → robot base TF",
        edge=GRAY, fill=GRAY_FILL, subtitle="GEOMETRY")
    box(ax, *control["planner"], "Motion layer",
        "Target pose\nMoveIt planning\nGrasp / place",
        edge=GRAY, fill=GRAY_FILL, subtitle="ACT")
    box(ax, *control["robot"], "UR3",
        "Execute\nObserve\nLog outcome",
        edge=NAVY, fill=BLUE_FILL, subtitle="ROBOT")

    # Calibrated probability enters policy.
    x, y, w, _ = modules["cal"]
    px, py, pw, ph = control["policy"]
    arrow(ax, (x + w / 2, y), (px + pw / 2, py + ph), color=ORANGE,
          rad=0.12, label="$p_{unsafe}$", label_xy=(11.25, 4.08))

    # ACT branch to 3-D projection and robot.
    for left_name, right_name in (("policy", "projection"), ("projection", "planner"), ("planner", "robot")):
        x1, y1, w1, h1 = control[left_name]
        x2, y2, _, h2 = control[right_name]
        arrow(ax, (x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2),
              color=ORANGE if left_name == "policy" else NAVY)

    # Reobserve feedback loop to the sensing input.
    ix, iy, iw, _ = modules["input"]
    arrow(ax, (px + 0.35, py), (ix + iw / 2, iy), color=ORANGE, dashed=True,
          rad=-0.28, label="REOBSERVE: change camera / viewpoint and rerun perception",
          label_xy=(3.55, 0.72))

    # Abstain branch terminates without motion.
    box(ax, 4.55, 1.92, 1.45, 1.12, "ABSTAIN",
        "No robot motion\nReturn uncertainty reason",
        edge=GRAY, fill=GRAY_FILL, subtitle="SAFE STOP", linestyle="--")
    arrow(ax, (px, py + ph / 2), (6.0, 2.48), color=GRAY, dashed=True)

    # Feature composition callout.
    box(ax, 0.35, 1.52, 3.65, 1.86, "16-D feature composition",
        "Output contract: 4    Stochastic behavior: 2\nPoint geometry: 3      Local depth: 3\nRelation one-hot: 4\nTotal: $4+2+3+3+4=16$",
        edge=GREEN, fill=GREEN_FILL, subtitle="FEATURE FUSION")

    ax.text(15.65, 0.28,
            "Solid arrows: forward data/action flow  ·  Dashed arrows: auxiliary input or feedback control",
            ha="right", va="center", fontsize=7.3, color=GRAY)

    png = OUT / "00_architecture_pipeline_paper.png"
    pdf = OUT / "00_architecture_pipeline_paper.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.12,
                facecolor=WHITE)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
