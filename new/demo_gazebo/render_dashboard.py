#!/usr/bin/env python3
"""Render an honest four-panel Gazebo/V2 observation dashboard from one run."""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


BG = "#F4F6F8"
INK = "#15212C"
MUTED = "#51606D"
BORDER = "#CCD5DC"
BLUE = "#1479A8"
AMBER = "#9A6200"
RED = "#B43E37"
GREEN = "#24784C"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape[:2] != (480, 640):
        raise ValueError(f"Expected a real 640x480 Gazebo RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def frame_axis(fig: plt.Figure, bounds: list[float], title: str) -> plt.Axes:
    ax = fig.add_axes(bounds)
    ax.set_facecolor("#E8ECEF")
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.text(bounds[0], bounds[1] + bounds[3] + 0.008, title,
             color=INK, fontsize=11.2, weight="bold")
    return ax


def show_rgb(ax: plt.Axes, rgb: np.ndarray) -> None:
    ax.imshow(rgb, extent=(0, 640, 480, 0), interpolation="nearest")
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_aspect("equal", adjustable="box")


def draw_score(ax: plt.Axes, x: float, y: float, width: float,
               label: str, value: float, color: str) -> None:
    ax.text(x, y + 0.036, label, transform=ax.transAxes, color=INK,
            fontsize=8.3, va="bottom")
    ax.text(x + width, y + 0.036, f"{value:.3f}", transform=ax.transAxes,
            color=MUTED, fontsize=8.3, ha="right", va="bottom")
    ax.add_patch(Rectangle((x, y), width, 0.017, transform=ax.transAxes,
                           facecolor="#E1E7EC", edgecolor="none"))
    ax.add_patch(Rectangle((x, y), width * min(max(value, 0.0), 1.0), 0.017,
                           transform=ax.transAxes, facecolor=color, edgecolor="none"))


def render(run_dir: Path) -> Path:
    capture = json.loads((run_dir / "capture/capture.json").read_text(encoding="utf-8"))
    prediction = json.loads((run_dir / "prediction.json").read_text(encoding="utf-8"))
    decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if capture["rgb_topic"] != "/wrist_camera/image" or capture["overview_topic"] != "/left_oblique_table_camera/image":
        raise ValueError("Four-panel dashboard requires synchronized wrist RGB-D and left-oblique overview")
    if capture["safety"]["robot_motion_commanded"] or manifest["safety"]["robot_motion_commanded"]:
        raise ValueError("Dashboard does not accept a robot-motion run")
    for key, filename in (("rgb_original", "rgb_original.png"),
                          ("overview_original", "overview_original.png")):
        path = run_dir / "capture" / filename
        if sha256(path) != capture["artifacts_sha256"][key]:
            raise ValueError(f"Captured image hash mismatch: {path}")
    overview = read_image(run_dir / "capture/overview_original.png")
    wrist = read_image(run_dir / "capture/rgb_original.png")
    depth = cv2.imread(str(run_dir / "capture/depth_relative_model_input.png"), cv2.IMREAD_GRAYSCALE)
    if depth is None or depth.shape != (480, 640):
        raise ValueError("Captured relative-depth image missing")
    probability = np.asarray(prediction["spatial"]["probability_grid"], dtype=np.float32)
    if probability.shape != (24, 32) or not np.isfinite(probability).all():
        raise ValueError("Frozen V2 heatmap must be finite 24x32")
    upsampled = cv2.resize(probability, (640, 480), interpolation=cv2.INTER_LINEAR)
    peak = float(probability.max())
    visible = np.ma.masked_where(upsampled < peak * 0.015, upsampled)
    candidate_x, candidate_y = map(int, prediction["spatial"]["map_pixel_xy"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(12, 10), dpi=120, facecolor=BG)
    fig.text(0.025, 0.975, "Spatial-VLM / P-CRA-U  |  Gazebo observation demo",
             color=INK, fontsize=16.5, weight="bold", va="top")
    fig.text(0.975, 0.975, "BEST V2 · EPOCH 13 · NO ROBOT MOTION",
             color=RED, fontsize=10.3, weight="bold", ha="right", va="top")
    prompt = "Prompt: " + prediction["prompt"]
    prompt_lines = textwrap.wrap(prompt, width=150, max_lines=2, placeholder="…")
    fig.text(0.025, 0.932, "\n".join(prompt_lines), color=MUTED, fontsize=10.1, va="top")

    bounds = {
        "overview": [0.025, 0.515, 0.46, 0.343],
        "wrist": [0.515, 0.515, 0.46, 0.343],
        "heat": [0.025, 0.125, 0.46, 0.343],
        "policy": [0.515, 0.125, 0.46, 0.343],
    }
    ax_overview = frame_axis(fig, bounds["overview"], "1  Gazebo scene · fixed left-oblique camera")
    show_rgb(ax_overview, overview)
    ax_wrist = frame_axis(fig, bounds["wrist"], "2  D435i wrist RGB · model input view")
    show_rgb(ax_wrist, wrist)
    ax_wrist.scatter([candidate_x], [candidate_y], s=160, marker="x", c="#00D6E8",
                     linewidths=2.8, zorder=5)
    ax_wrist.text(8, 465, f"Candidate MAP ({candidate_x}, {candidate_y}) · not a grasp point",
                  color="white", fontsize=9.0, weight="bold", va="bottom",
                  bbox={"facecolor": "#15212CCC", "edgecolor": "none", "pad": 4})

    ax_heat = frame_axis(fig, bounds["heat"], "3  V2 spatial distribution · registered depth inset")
    show_rgb(ax_heat, wrist)
    ax_heat.imshow(visible, extent=(0, 640, 480, 0), cmap="magma", vmin=0,
                   vmax=peak, alpha=0.78, interpolation="nearest")
    ax_heat.scatter([candidate_x], [candidate_y], s=170, marker="x", c="#00D6E8",
                    linewidths=2.8, zorder=7)
    region = decision.get("confidence_region") or {}
    for gx, gy in region.get("grid_cells_xy", []):
        ax_heat.add_patch(Rectangle((gx * 20, gy * 20), 20, 20, fill=False,
                                    edgecolor="#37DB87", linewidth=1.2, zorder=6))
    inset = fig.add_axes([0.362, 0.144, 0.112, 0.112])
    inset.imshow(depth, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(1.2)
    fig.text(0.363, 0.132, "Relative depth input", fontsize=8.2, color=MUTED)

    ax_policy = frame_axis(fig, bounds["policy"], "4  Answerability · source scores · selective decision")
    ax_policy.set_xlim(0, 1)
    ax_policy.set_ylim(0, 1)
    action = str(decision["action"])
    action_color = {"EXECUTE": GREEN, "REOBSERVE": AMBER,
                    "ASK_USER": BLUE, "ABSTAIN": RED}.get(action, INK)
    ax_policy.add_patch(Rectangle((0.025, 0.795), 0.95, 0.166,
                                  transform=ax_policy.transAxes, facecolor="#EAF0F3",
                                  edgecolor="none"))
    ax_policy.text(0.044, 0.913, action, color=action_color, fontsize=20,
                   weight="bold", transform=ax_policy.transAxes, va="center")
    ax_policy.text(0.965, 0.922, "POLICY OUTPUT ONLY", color=RED, fontsize=9.1,
                   ha="right", transform=ax_policy.transAxes, va="center", weight="bold")
    ax_policy.text(0.044, 0.818, decision["reason"], color=MUTED,
                   fontsize=9.3, transform=ax_policy.transAxes)
    risk = float(decision["calibrated_grounding_risk"])
    threshold = decision.get("risk_threshold")
    ax_policy.text(0.044, 0.738, f"Model-reported grounding risk  {risk:.3f}",
                   color=INK, fontsize=10.1, weight="bold", transform=ax_policy.transAxes)
    ax_policy.add_patch(Rectangle((0.044, 0.688), 0.91, 0.026,
                                  transform=ax_policy.transAxes, facecolor="#DFE6EB", edgecolor="none"))
    ax_policy.add_patch(Rectangle((0.044, 0.688), 0.91 * min(max(risk, 0), 1), 0.026,
                                  transform=ax_policy.transAxes, facecolor=action_color, edgecolor="none"))
    if threshold is not None:
        x_threshold = 0.044 + 0.91 * float(threshold)
        ax_policy.plot([x_threshold, x_threshold], [0.67, 0.73], color=INK,
                       lw=1.4, transform=ax_policy.transAxes)
        ax_policy.text(0.044, 0.636, f"Frozen threshold τ = {float(threshold):.3f}  |  no actuation",
                       color=MUTED, fontsize=8.8, transform=ax_policy.transAxes)
    else:
        ax_policy.text(0.044, 0.636, "No frozen risk threshold  |  no actuation",
                       color=MUTED, fontsize=8.8, transform=ax_policy.transAxes)

    ax_policy.text(0.044, 0.550, "Answerability probabilities", color=INK,
                   fontsize=9.2, weight="bold", transform=ax_policy.transAxes)
    ax_policy.text(0.532, 0.550, "Uncertainty source scores (raw)", color=INK,
                   fontsize=9.2, weight="bold", transform=ax_policy.transAxes)
    answer = prediction["answerability_probabilities"]
    for index, label in enumerate(("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")):
        draw_score(ax_policy, 0.044, 0.462 - index * 0.098, 0.425,
                   label.replace("INSUFFICIENT_EVIDENCE", "INSUFFICIENT"),
                   float(answer[label]), BLUE)
    sources = prediction["source_probabilities"]
    for index, label in enumerate(("semantic", "relation", "spatial", "depth", "occlusion")):
        draw_score(ax_policy, 0.532, 0.462 - index * 0.079, 0.425,
                   label.capitalize(), float(sources[label]), AMBER)

    fig.text(0.025, 0.069,
             "Synchronized Gazebo snapshot · Heatmap shown relative to its own peak · Green outline = frozen region output (not validated here).",
             color=MUTED, fontsize=9.0)
    fig.text(0.025, 0.041,
             "Preview pose/layout differs from evaluated Test-IID. Risk is a frozen-model output, not validated calibration or grasp success.",
             color=RED, fontsize=9.2, weight="bold")
    output = run_dir / "dashboard.png"
    fig.savefig(output, dpi=120, facecolor=BG)
    plt.close(fig)
    manifest.setdefault("outputs_sha256", {})["dashboard.png"] = sha256(output)
    manifest["dashboard_type"] = "single_synchronized_observation_no_actuation"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.run_dir.resolve()))
