#!/usr/bin/env python3
"""Rebuild code-faithful method diagrams 4, 5 and 9 as vector PDF and PNG."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "new/src"))
from pcrau.text import parse_relations, prompt_anchor_mask  # noqa: E402


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#8B5AA8"
INK = "#20252B"
MUTED = "#53606B"
LIGHT = "#D8DFE4"
PALE_BLUE = "#EDF6FA"
PALE_ORANGE = "#FCF3EC"
PALE_GREEN = "#EDF8F4"
PALE_PURPLE = "#F6F0F9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_jsonl(path: Path, sample_id: str):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] == sample_id:
                return row
    raise KeyError(sample_id)


SAMPLE_ID = "v211iid_family_000007__clean"
manifest = read_json(ROOT / "new/test_iid/protocol/test_iid_eval_manifest.json")
entry = next(row for row in manifest["entries"] if row["sample_id"] == SAMPLE_ID)
prediction = find_jsonl(ROOT / "new/test_iid/evaluation/best_v2/predictions.jsonl", SAMPLE_ID)
decision = find_jsonl(ROOT / "new/test_iid/evaluation/best_v2/decisions.jsonl", SAMPLE_ID)
calibrator = read_json(ROOT / "new/outputs/pcrau_target_v2_full_seed_24082026/evaluation/calibration/calibrator.json")
config = read_json(ROOT / "new/configs/pcrau_target_v2.json")["model"]


def canvas(title: str, subtitle: str, width=14.0, height=7.0):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.025, right=0.975, top=0.97, bottom=0.035)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(0.2, 6.75, title, fontsize=15, fontweight="bold", color=INK, va="top")
    ax.text(0.2, 6.38, subtitle, fontsize=9.2, color=MUTED, va="top")
    return fig, ax


def box(ax, x, y, w, h, title, detail="", *, edge=LIGHT, fill="white", title_size=10.0,
        detail_size=8.7, align="center"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=1.15))
    cx = x + (w / 2 if align == "center" else 0.14)
    ha = "center" if align == "center" else "left"
    if detail:
        ax.text(cx, y + h * 0.64, title, ha=ha, va="center", fontsize=title_size,
                fontweight="bold", color=INK)
        ax.text(cx, y + h * 0.28, detail, ha=ha, va="center", fontsize=detail_size,
                color=MUTED, linespacing=1.3)
    else:
        ax.text(cx, y + h / 2, title, ha=ha, va="center", fontsize=title_size,
                fontweight="bold", color=INK)


def arrow(ax, start, end, *, color=MUTED, width=1.35, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12,
                                 linewidth=width, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def save(fig, folder: str, stem: str):
    target = OUT / folder
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.10)
    fig.savefig(target / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def fig04_query_graph():
    prompt = entry["feature_input"]["prompt"]
    assert "locate the mango that is nearer than the apple" in prompt
    assert parse_relations(prompt, config["max_relations"]) == ["nearer_than"]
    assert prompt_anchor_mask(prompt, config["max_anchors"]) == [True, False, False]

    fig, ax = canvas("Language query graph in the implemented P-CRA-U sidecar",
                     "One real Test-IID instruction; colored blocks are active model components, not symbolic noun extraction.")
    box(ax, 0.35, 5.18, 13.3, 0.80,
        'Prompt excerpt: “locate the mango that is nearer than the apple”',
        f"Test-IID sample {SAMPLE_ID}", edge=INK, fill="white", title_size=11.2)
    ax.text(0.48, 4.89, "Text route", color=BLUE, fontsize=9.3, fontweight="bold")
    ax.text(7.15, 4.89, "Relation/slot route", color=ORANGE, fontsize=9.3, fontweight="bold")
    box(ax, 0.5, 3.83, 5.8, 0.83, "Hashed tokens  →  Transformer text encoder",
        f"max {config['max_tokens']} tokens · {config['text_layers']} layers · hidden {config['hidden_dim']}",
        edge=BLUE, fill=PALE_BLUE)
    box(ax, 7.0, 3.83, 6.45, 0.83, "Relation regex + active-slot masks",
        "nearer_than · 1/3 relation slots · 1/3 anchor slots", edge=ORANGE, fill=PALE_ORANGE)
    arrow(ax, (3.4, 5.18), (3.4, 4.67), color=BLUE)
    arrow(ax, (10.15, 5.18), (10.15, 4.67), color=ORANGE)
    box(ax, 3.25, 2.69, 7.5, 0.77, "Learned target / relation / anchor slots cross-attend to text",
        "Slot values are contextual embeddings; inactive relation/anchor slots are masked downstream.",
        edge=PURPLE, fill=PALE_PURPLE, title_size=10.1)
    arrow(ax, (3.4, 3.83), (5.3, 3.47), color=BLUE)
    arrow(ax, (10.15, 3.83), (8.7, 3.47), color=ORANGE)
    ax.text(0.48, 2.26, "Slot-level query structure used by spatial and relation heads", color=INK,
            fontsize=9.7, fontweight="bold")
    box(ax, 0.8, 1.27, 3.25, 0.72, r"Target slot $q_T$", "learned; active", edge=GREEN, fill=PALE_GREEN)
    box(ax, 5.35, 1.27, 3.25, 0.72, r"Relation slot $q_{R1}$", "nearer_than; active", edge=ORANGE, fill=PALE_ORANGE)
    box(ax, 9.9, 1.27, 3.25, 0.72, r"Anchor slot $q_{A1}$", "learned; active", edge=BLUE, fill=PALE_BLUE)
    arrow(ax, (4.05, 1.63), (5.35, 1.63), color=MUTED)
    arrow(ax, (8.6, 1.63), (9.9, 1.63), color=MUTED)
    arrow(ax, (7.0, 2.69), (7.0, 2.01), color=PURPLE)
    ax.text(0.48, 0.72,
            "Interpretation only: ‘mango’ and ‘apple’ appear in the prompt; P-CRA-U does not output a parsed noun-to-node assignment.",
            fontsize=8.8, color=MUTED)
    save(fig, "04_language_query_graph", "fig04_language_query_graph_v2")


def fig05_fusion():
    gate = float(prediction["fusion_gate_mean"])
    assert 0 <= gate <= 1
    fig, ax = canvas("Relation-conditioned RGB–depth fusion",
                     "Implemented P-CRA-U sidecar: the relation context modulates a spatial gate and cross-modal residual term.")
    ax.text(0.45, 5.92, "Inputs", color=MUTED, fontsize=9.3, fontweight="bold")
    ax.text(3.44, 5.92, "Learned projections", color=MUTED, fontsize=9.3, fontweight="bold")
    ax.text(6.43, 5.92, "Relation-conditioned operations", color=MUTED, fontsize=9.3, fontweight="bold")
    ax.text(11.08, 5.92, "Fused output", color=MUTED, fontsize=9.3, fontweight="bold")
    rows = [
        (4.77, r"RGB grid $R_0$", r"24×32×1152", BLUE, PALE_BLUE, r"$W_R R_0$", r"24×32×128"),
        (3.66, r"Depth grid $D_0$", r"24×32×1152", GREEN, PALE_GREEN, r"$W_D D_0$", r"24×32×128"),
        (2.55, r"Relation context $q_{rel}$", r"128-D from query slots", ORANGE, PALE_ORANGE,
         "Broadcast context", r"24×32×128"),
    ]
    for y, title, detail, color, fill, projected, projected_detail in rows:
        box(ax, 0.45, y, 2.35, 0.75, title, detail, edge=color, fill=fill)
        box(ax, 3.38, y, 2.36, 0.75, projected, projected_detail, edge=color, fill=fill)
        arrow(ax, (2.8, y + 0.375), (3.38, y + 0.375), color=color)
    box(ax, 6.32, 3.49, 1.74, 1.23, "Concatenate", "RGB + depth +\nrelation", edge=MUTED, fill="white")
    for y, color in [(4.77, BLUE), (3.66, GREEN), (2.55, ORANGE)]:
        arrow(ax, (5.74, y + 0.375), (6.32, 4.15 if y == 4.77 else 3.96 if y == 3.66 else 3.72),
              color=color)
    box(ax, 8.7, 4.49, 1.82, 0.80, "Gate MLP + sigmoid", r"$g \in [0,1]^{24×32×128}$",
        edge=PURPLE, fill=PALE_PURPLE, title_size=9.3)
    box(ax, 8.7, 3.08, 1.82, 0.80, "Cross-modal MLP", "residual term", edge=PURPLE,
        fill=PALE_PURPLE, title_size=9.3)
    arrow(ax, (8.06, 4.30), (8.7, 4.89), color=PURPLE)
    arrow(ax, (8.06, 3.78), (8.7, 3.48), color=PURPLE)
    box(ax, 11.12, 3.77, 2.43, 1.06, "LayerNorm + sum",
        r"$W_RR_0+g\odot W_DD_0+\mathrm{cross}$", edge=INK, fill="white", detail_size=8.1)
    arrow(ax, (10.52, 4.89), (11.12, 4.42), color=PURPLE)
    arrow(ax, (10.52, 3.48), (11.12, 4.08), color=PURPLE)
    box(ax, 10.96, 2.40, 2.75, 0.78, "Residual MLP × 2", "fused grid F", edge=INK, fill="white")
    arrow(ax, (12.33, 3.77), (12.33, 3.18), color=INK)
    box(ax, 9.98, 1.30, 3.99, 0.70, "Downstream P-CRA-U heads",
        "target / interior / anchor + graph / answer / source", edge=MUTED, fill="white", detail_size=8.2)
    arrow(ax, (12.33, 2.40), (12.33, 2.01), color=INK)
    ax.text(0.48, 1.50, f"Observed gate mean = {gate:.4f}", fontsize=10.4, color=PURPLE,
            fontweight="bold")
    ax.text(0.48, 1.12, f"Test-IID sample {SAMPLE_ID}", fontsize=8.9, color=MUTED)
    ax.text(0.48, 0.66,
            "The scalar is averaged over pixels and channels. No per-pixel gate map was retained in frozen predictions.",
            fontsize=8.8, color=MUTED)
    save(fig, "05_relation_conditioned_fusion", "fig05_relation_conditioned_fusion_v2")


def fig09_calibrator():
    assert len(calibrator["feature_names"]) == 18
    assert calibrator["risk_policy"]["threshold"] == decision["risk_threshold"]
    assert decision["action"] == "REOBSERVE"
    threshold = float(calibrator["risk_policy"]["threshold"])
    mass = float(calibrator["conformal"]["probability_mass"])
    risk = float(decision["calibrated_grounding_risk"])
    insufficient = float(prediction["answerability_probabilities"]["INSUFFICIENT_EVIDENCE"])
    fig, ax = canvas("Independent calibration and selective policy",
                     "Frozen P-CRA-U evidence is mapped to risk; action rules and the spatial region are separate downstream steps.")
    box(ax, 0.33, 3.90, 4.20, 2.05, "Frozen P-CRA-U prediction",
        "18 features: spatial 3 · answerability 4 · source 5\nrelation/fusion 2 · RGB–depth 2 · RoboRefer 2\n+ separate target probability grid for region",
        edge=BLUE, fill=PALE_BLUE, title_size=10.5, detail_size=9.1)
    box(ax, 5.02, 5.18, 3.18, 0.72, "Standardize features", r"$z=(x-\mu)/s$",
        edge=PURPLE, fill=PALE_PURPLE)
    box(ax, 5.02, 4.00, 3.18, 0.72, "Independent logistic fit", r"$r_{ground}=\sigma(w^\top z+b)$",
        edge=PURPLE, fill=PALE_PURPLE)
    arrow(ax, (4.53, 5.47), (5.02, 5.54), color=BLUE)
    arrow(ax, (6.61, 5.18), (6.61, 4.72), color=PURPLE)
    box(ax, 8.76, 5.05, 4.82, 0.95, "Calibrated grounding risk",
        f"r_ground; frozen threshold τ = {threshold:.4f}", edge=ORANGE, fill=PALE_ORANGE)
    arrow(ax, (8.2, 4.36), (8.76, 5.50), color=ORANGE, rad=-0.18)
    box(ax, 8.76, 3.80, 4.82, 0.82, "Spatial confidence region (parallel)",
        f"highest-density grid region; calibrated mass = {mass:.4f}",
        edge=GREEN, fill=PALE_GREEN, title_size=9.6, detail_size=8.7)
    ax.plot([4.08, 4.08, 8.86], [3.90, 3.67, 3.67], color=GREEN, linewidth=1.35)
    arrow(ax, (8.86, 3.67), (8.86, 3.80), color=GREEN)
    ax.text(0.35, 3.49,
            f"Calibration artifact: {calibrator['samples']} samples / {calibrator['families']} families; family 5-fold crossfit.",
            fontsize=8.8, color=MUTED)
    ax.plot([0.35, 13.65], [3.25, 3.25], color=LIGHT, linewidth=1.0)
    ax.text(0.35, 3.06, "Selective decision: rules are checked in this priority order", fontsize=10,
            fontweight="bold", color=INK)
    decisions = [
        (0.37, 2.13, "1  Predicted ABSENT", "ABSTAIN", ORANGE, PALE_ORANGE),
        (4.85, 2.13, "2  Predicted AMBIGUOUS", "ASK_USER", PURPLE, PALE_PURPLE),
        (9.33, 2.13, "3  INSUFFICIENT_EVIDENCE", "REOBSERVE", BLUE, PALE_BLUE),
        (0.37, 1.13, "4  FOUND and risk ≤ τ", "EXECUTE", GREEN, PALE_GREEN),
        (4.85, 1.13, "5  High risk; semantic dominates", "ASK_USER", PURPLE, PALE_PURPLE),
        (9.33, 1.13, "6  Other high-risk FOUND", "REOBSERVE", BLUE, PALE_BLUE),
    ]
    for x, y, condition, action, color, fill in decisions:
        box(ax, x, y, 4.29, 0.74, condition, action, edge=color, fill=fill,
            title_size=9.2, detail_size=9.1)
    ax.text(0.36, 0.57,
            f"Observed {SAMPLE_ID}: p(INSUFFICIENT)={insufficient:.3f}, risk={risk:.3f} → REOBSERVE (rule 3).",
            fontsize=8.8, color=INK)
    ax.text(0.36, 0.23,
            "Test-IID: RoboRefer disagreement is unavailable; its missing indicator is active. The P-CRA-U calibrator has no source-specific threshold.",
            fontsize=8.2, color=MUTED)
    save(fig, "09_independent_calibrator", "fig09_independent_calibrator_v2")


def main():
    fig04_query_graph()
    fig05_fusion()
    fig09_calibrator()
    print("Generated method figures 4, 5 and 9 (PNG + PDF) from frozen P-CRA-U code and artifacts.")


if __name__ == "__main__":
    main()
