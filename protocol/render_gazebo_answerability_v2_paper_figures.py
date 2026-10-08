#!/usr/bin/env python3
"""Render traceable, paper-style figures for Gazebo_dev_answerability_v2.

This renderer only consumes existing locked/derived evaluation artifacts.  It
does not call a generative model and does not create synthetic measurements.

Usage:
  python3 protocol/render_gazebo_answerability_v2_paper_figures.py render
  python3 protocol/render_gazebo_answerability_v2_paper_figures.py check
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "ketqua"
PROTOCOL = "gazebo_dev_answerability_v2"
N = 64
SOURCE_PATHS = {
    "contract_lock": ROOT / "protocol/gazebo_dev_answerability_v2_contract_lock.json",
    "dataset_manifest": ROOT / "datasets/Gazebo_dev_answerability_v2/manifest.json",
    "ground_truth": ROOT / "datasets/Gazebo_dev_answerability_v2/evaluator_ground_truth.jsonl",
    "qc": ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2/GAZEBO_DEV_ANSWERABILITY_V2_QC.json",
    "metrics": ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2/evaluation/B0_B1_ANSWERABILITY_V2_METRICS.json",
    "b0_run": ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2/evaluation/b0/run.json",
    "b1_run": ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2/evaluation/b1/run.json",
    "b0_predictions": ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2/evaluation/b0/predictions.jsonl",
    "b1_predictions": ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2/evaluation/b1/predictions.jsonl",
    "b0_scored": ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2/evaluation/b0/scored_predictions.jsonl",
    "b1_scored": ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_answerability_v2/evaluation/b1/scored_predictions.jsonl",
}

# Okabe-Ito-style, color-blind-safe primary colors.  Shapes, direct labels, and
# hatches duplicate all categorical information so the figures also work in
# grayscale printing.
B0 = "#0072B2"
B1 = "#D55E00"
POINT = "#0072B2"
ABSTAIN = "#009E73"
INVALID = "#666666"
INK = "#202124"
MUTED = "#5F6368"
GRID = "#DADCE0"
PALE_BLUE = "#E8F1F8"
PALE_ORANGE = "#FCE9DD"
GOOD = "#009E73"
BAD = "#CC3311"
STATE_COLORS = ["#56B4E9", "#E69F00", "#CC79A7", "#999999"]
COORD_RE = re.compile(r"\(\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*,\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*\)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#9AA0A6",
        "axes.linewidth": 0.7,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=12, va="top", color=INK)


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    for suffix, kwargs in (("png", {"dpi": 300}), ("svg", {}), ("pdf", {})):
        fig.savefig(out / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def validate_and_collect() -> dict:
    missing = [str(path) for path in SOURCE_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required source artifacts missing: {missing}")
    lock = load_json(SOURCE_PATHS["contract_lock"])
    manifest = load_json(SOURCE_PATHS["dataset_manifest"])
    qc = load_json(SOURCE_PATHS["qc"])
    metrics = load_json(SOURCE_PATHS["metrics"])
    if lock.get("protocol_id") != PROTOCOL or lock.get("status") != "LOCKED_BEFORE_CAPTURE_AND_INFERENCE":
        raise ValueError("contract lock does not identify a locked answerability-v2 protocol")
    if manifest.get("protocol_id") != PROTOCOL or manifest.get("status") != "PASS" or manifest.get("records") != N:
        raise ValueError("dataset manifest is not a PASS 64-record answerability-v2 materialization")
    if qc.get("protocol_id") != PROTOCOL or qc.get("status") != "PASS" or qc.get("failed_scene_count") != 0:
        raise ValueError("geometry QC is not a clean PASS")
    if qc.get("contract_lock_sha256") != sha256(SOURCE_PATHS["contract_lock"]):
        raise ValueError("QC report does not match the current contract lock")
    if metrics.get("protocol_id") != PROTOCOL or metrics.get("paired", {}).get("promotion_decision") != "KEEP_B0_AND_ANALYZE_ERRORS":
        raise ValueError("metrics package is missing expected protocol/decision")
    gt = {row["sample_id"]: row for row in load_jsonl(SOURCE_PATHS["ground_truth"])}
    if len(gt) != N or not all(row.get("answerability_verified") for row in gt.values()):
        raise ValueError("ground truth is incomplete or has unverified answerability")
    predictions, scored, runs = {}, {}, {}
    for model in ("b0", "b1"):
        predictions[model] = {row["sample_id"]: row for row in load_jsonl(SOURCE_PATHS[f"{model}_predictions"])}
        scored[model] = {row["sample_id"]: row for row in load_jsonl(SOURCE_PATHS[f"{model}_scored"])}
        runs[model] = load_json(SOURCE_PATHS[f"{model}_run"])
        if len(predictions[model]) != N or set(predictions[model]) != set(gt):
            raise ValueError(f"{model} predictions do not align with ground truth")
        if len(scored[model]) != N or set(scored[model]) != set(gt):
            raise ValueError(f"{model} scored predictions do not align with ground truth")
        if runs[model].get("status") != "COMPLETED" or runs[model].get("completed_count") != N:
            raise ValueError(f"{model} blinded inference was not completed")
    state_order = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
    relation_order = ["leftmost", "rightmost", "second_from_left", "second_from_right"]
    state_counts = Counter(row["answerability_state"] for row in gt.values())
    relation_counts = Counter(row["relation_variant"] for row in gt.values())
    if dict(state_counts) != manifest["state_counts"] or dict(relation_counts) != manifest["relation_variant_counts"]:
        raise ValueError("ground-truth balance does not match dataset manifest")
    if any(state_counts[state] != 16 for state in state_order) or any(relation_counts[relation] != 16 for relation in relation_order):
        raise ValueError("v2 balance is not 16 per state and relation")
    return {
        "lock": lock,
        "manifest": manifest,
        "qc": qc,
        "metrics": metrics,
        "gt": gt,
        "predictions": predictions,
        "scored": scored,
        "runs": runs,
        "state_order": state_order,
        "relation_order": relation_order,
        "state_counts": state_counts,
        "relation_counts": relation_counts,
    }


def action_counts(data: dict) -> dict[str, dict[str, Counter]]:
    values: dict[str, dict[str, Counter]] = {}
    for model in ("b0", "b1"):
        values[model] = {}
        for state in data["state_order"]:
            values[model][state] = Counter(
                row["action"]
                for sample_id, row in data["predictions"][model].items()
                if data["gt"][sample_id]["answerability_state"] == state
            )
    return values


def response_styles(data: dict) -> dict[str, Counter]:
    styles: dict[str, Counter] = {}
    for model in ("b0", "b1"):
        counts = Counter()
        for row in data["predictions"][model].values():
            answer = (row.get("answer") or "").strip()
            coordinates = COORD_RE.findall(answer)
            if re.fullmatch(r"ABSTAIN", answer):
                category = "exact_ABSTAIN"
            elif len(coordinates) == 0:
                category = "no_coordinate_text"
            elif len(coordinates) == 1:
                category = "one_coordinate"
            else:
                category = "multiple_coordinates"
            counts[category] += 1
        styles[model] = counts
    return styles


def write_tables(data: dict, out: Path) -> None:
    metrics = data["metrics"]["metrics"]
    paired = data["metrics"]["paired"]
    actions = action_counts(data)
    styles = response_styles(data)
    with (out / "Table_1_protocol_qc.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["field", "value", "evidence"])
        writer.writeheader()
        rows = [
            ("protocol_id", PROTOCOL, "locked contract"),
            ("parent_families", N, "dataset manifest"),
            ("state_balance", "16 each × 4 states", "ground truth / QC"),
            ("relation_balance", "16 each × 4 variants", "ground truth / QC"),
            ("geometry_qc", "PASS; 64/64; failed=0", "QC report"),
            ("no_v1_family_overlap", data["qc"]["leakage_checks"]["no_v1_family_overlap"], "QC report"),
            ("inference_oracle_free", data["qc"]["leakage_checks"]["inference_manifest_oracle_free"], "QC report"),
            ("no_sam2", True, "contract/manifest"),
            ("b2_opened", False, "contract/manifest"),
            ("test_iid_ood_access", False, "contract/manifest"),
        ]
        for field, value, evidence in rows:
            writer.writerow({"field": field, "value": value, "evidence": evidence})
    with (out / "Table_2_primary_endpoints.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "scope", "n", "b0", "clean_b1", "b1_minus_b0", "interpretation"])
        writer.writeheader()
        rows = [
            ("FOUND Hit@0.08", "FOUND", 16, metrics["b0"]["found_hit_at_008"], metrics["b1"]["found_hit_at_008"], paired["found_hit_delta_b1_minus_b0"], "higher is better"),
            ("Semantic answerability action accuracy", "all", 64, metrics["b0"]["answerability_action_accuracy"], metrics["b1"]["answerability_action_accuracy"], metrics["b1"]["answerability_action_accuracy"] - metrics["b0"]["answerability_action_accuracy"], "diagnostic semantic parser"),
            ("Semantic safe-task accuracy", "all", 64, metrics["b0"]["safe_task_accuracy"], metrics["b1"]["safe_task_accuracy"], paired["safe_task_delta_b1_minus_b0"], "diagnostic semantic parser"),
            ("Non-FOUND abstain recall", "non-FOUND", 48, metrics["b0"]["nonfound_abstain_recall"], metrics["b1"]["nonfound_abstain_recall"], metrics["b1"]["nonfound_abstain_recall"] - metrics["b0"]["nonfound_abstain_recall"], "higher is better"),
            ("POINT-only false accept rate", "non-FOUND", 48, metrics["b0"]["false_accept_rate_nonfound"], metrics["b1"]["false_accept_rate_nonfound"], metrics["b1"]["false_accept_rate_nonfound"] - metrics["b0"]["false_accept_rate_nonfound"], "lower is better; invalid separate"),
            ("Invalid response rate", "all", 64, metrics["b0"]["invalid_rate"], metrics["b1"]["invalid_rate"], metrics["b1"]["invalid_rate"] - metrics["b0"]["invalid_rate"], "lower is better"),
            ("Exact output contract", "all", 64, metrics["b0"]["exact_contract_rate"], metrics["b1"]["exact_contract_rate"], metrics["b1"]["exact_contract_rate"] - metrics["b0"]["exact_contract_rate"], "higher is better"),
            ("AURC", "all", 64, metrics["b0"]["aurc"], metrics["b1"]["aurc"], metrics["b1"]["aurc"] - metrics["b0"]["aurc"], "lower is better"),
        ]
        for metric, scope, n, b0, b1, delta, interpretation in rows:
            writer.writerow({"metric": metric, "scope": scope, "n": n, "b0": b0, "clean_b1": b1, "b1_minus_b0": delta, "interpretation": interpretation})
    with (out / "Table_3_actions_by_state.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "state", "n", "point", "abstain", "invalid", "correct_action_rate"])
        writer.writeheader()
        for model, label in (("b0", "B0"), ("b1", "clean B1")):
            for state in data["state_order"]:
                count = actions[model][state]
                writer.writerow({
                    "model": label,
                    "state": state,
                    "n": 16,
                    "point": count["POINT"],
                    "abstain": count["ABSTAIN"],
                    "invalid": count["INVALID"],
                    "correct_action_rate": data["metrics"]["metrics"][model]["by_state"][state]["correct_action_rate"],
                })
    with (out / "Table_4_paired_effects.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["endpoint", "b1_minus_b0", "bootstrap_95_low", "bootstrap_95_high", "b0_wrong_b1_right", "b0_right_b1_wrong", "mcnemar_p"])
        writer.writeheader()
        writer.writerow({"endpoint": "FOUND Hit@0.08", "b1_minus_b0": paired["found_hit_delta_b1_minus_b0"], "bootstrap_95_low": paired["found_hit_family_bootstrap_delta_95"][0], "bootstrap_95_high": paired["found_hit_family_bootstrap_delta_95"][1], "b0_wrong_b1_right": "", "b0_right_b1_wrong": "", "mcnemar_p": ""})
        writer.writerow({"endpoint": "Semantic safe-task accuracy", "b1_minus_b0": paired["safe_task_delta_b1_minus_b0"], "bootstrap_95_low": paired["safe_task_family_bootstrap_delta_95"][0], "bootstrap_95_high": paired["safe_task_family_bootstrap_delta_95"][1], "b0_wrong_b1_right": paired["b0_wrong_b1_right"], "b0_right_b1_wrong": paired["b0_right_b1_wrong"], "mcnemar_p": paired["mcnemar_exact_two_sided_p"]})
    with (out / "Table_5_raw_response_styles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "one_coordinate", "multiple_coordinates", "no_coordinate_text", "exact_abstain", "exact_contract_count"])
        writer.writeheader()
        for model, label in (("b0", "B0"), ("b1", "clean B1")):
            writer.writerow({"model": label, "one_coordinate": styles[model]["one_coordinate"], "multiple_coordinates": styles[model]["multiple_coordinates"], "no_coordinate_text": styles[model]["no_coordinate_text"], "exact_abstain": styles[model]["exact_ABSTAIN"], "exact_contract_count": sum(row["exact_contract"] for row in data["predictions"][model].values())})


def figure_1_design_and_qc(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.65), gridspec_kw={"width_ratios": [1.05, 1.05, 1.35]})
    for ax, labels, counts, title in (
        (axes[0], ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT\nEVIDENCE"], [data["state_counts"][key] for key in data["state_order"]], "Answerability states"),
        (axes[1], ["leftmost", "rightmost", "2nd from\nleft", "2nd from\nright"], [data["relation_counts"][key] for key in data["relation_order"]], "Ordinal relation variants"),
    ):
        positions = np.arange(4)[::-1]
        bars = ax.barh(positions, counts, color=STATE_COLORS, edgecolor="white", linewidth=0.8)
        for bar, value in zip(bars, counts):
            ax.text(bar.get_width() + 0.45, bar.get_y() + bar.get_height() / 2, str(value), ha="left", va="center", fontweight="bold")
        ax.set_title(title, loc="left")
        ax.set_yticks(positions, labels, fontsize=7.5)
        ax.set_xlim(0, 20); ax.set_xlabel("Parent families"); ax.grid(axis="x"); ax.set_axisbelow(True)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    axes[2].axis("off")
    panel_label(axes[2], "c")
    checks = [
        ("Contract locked before capture", True),
        ("Geometry QC: 64/64 PASS", data["qc"]["status"] == "PASS"),
        ("Family-disjoint from v1", data["qc"]["leakage_checks"]["no_v1_family_overlap"]),
        ("Inference manifest oracle-free", data["qc"]["leakage_checks"]["inference_manifest_oracle_free"]),
        ("B2 / Test-IID/OOD sealed", not data["manifest"]["b2_opened"] and not data["manifest"]["test_iid_ood_access"]),
    ]
    axes[2].text(0.0, 1.01, "Protocol integrity", fontweight="bold", fontsize=10.5, color=INK, transform=axes[2].transAxes)
    for index, (label, passed) in enumerate(checks):
        y = 0.84 - index * 0.16
        axes[2].scatter(0.025, y, s=55, color=GOOD if passed else BAD, marker="o", transform=axes[2].transAxes)
        axes[2].text(0.085, y, label, va="center", transform=axes[2].transAxes, color=INK)
    fig.suptitle("Figure 1 | Locked Gazebo_dev_answerability_v2 study design and integrity", x=0.075, ha="left", fontweight="bold", fontsize=13, color=INK)
    fig.text(0.075, 0.025, "N=64 independent Dev parent-families. Counts are observed records, not augmented views. Contract SHA-256: " + sha256(SOURCE_PATHS["contract_lock"])[:12] + "…", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(top=0.80, bottom=0.28, wspace=0.45)
    save(fig, out, "Figure_1_study_design_and_qc")


def figure_2_primary_endpoints(data: dict, out: Path) -> None:
    metrics = data["metrics"]["metrics"]
    endpoints = [
        ("FOUND\nHit@0.08", 16, "found_hit_at_008", "found_hit_wilson_95", "semantic point parser"),
        ("Semantic\nsafe-task", 64, "safe_task_accuracy", "safe_task_wilson_95", "POINT/ABSTAIN action"),
        ("Non-FOUND\nabstain recall", 48, "nonfound_abstain_recall", None, "correct action on negative states"),
        ("Exact output\ncontract", 64, "exact_contract_rate", None, "strict syntax only"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.55))
    x = np.arange(len(endpoints)); width = 0.34
    for offset, model, color, hatch in ((-width / 2, "b0", B0, ""), (width / 2, "b1", B1, "//")):
        values = [metrics[model][field] for _, _, field, _, _ in endpoints]
        lows, highs = [], []
        for _, _, _, ci_field, _ in endpoints:
            if ci_field:
                low, high = metrics[model][ci_field]
                lows.append(metrics[model][ci_field] and metrics[model][endpoints[len(lows)][2]] - low)
                highs.append(high - metrics[model][endpoints[len(highs)][2]])
            else:
                lows.append(0.0); highs.append(0.0)
        bars = ax.bar(x + offset, values, width, label="B0" if model == "b0" else "clean B1", color=color, hatch=hatch, edgecolor="white", linewidth=0.7, yerr=np.vstack([lows, highs]), capsize=3, error_kw={"ecolor": INK, "lw": 0.8})
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.1%}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x, [f"{label}\n(n={n})" for label, n, _, _, _ in endpoints])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(loc="upper right")
    ax.set_title("Primary endpoints", loc="left")
    panel_label(ax, "a")
    fig.suptitle("Figure 2 | B0 grounds FOUND cases; neither model meets the answerability contract", x=0.095, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.095, 0.015, "Error bars are Wilson 95% intervals where defined. Semantic metrics accept a single valid coordinate even when strict output syntax is violated; exact-contract rate remains 0% for both.", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(top=0.79, bottom=0.23)
    save(fig, out, "Figure_2_primary_endpoints")


def figure_3_action_by_state(data: dict, out: Path) -> None:
    counts = action_counts(data)
    states = data["state_order"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    action_specs = (("POINT", POINT, ""), ("ABSTAIN", ABSTAIN, ".."), ("INVALID", INVALID, "//"))
    for ax, model, title in zip(axes, ("b0", "b1"), ("B0", "clean B1")):
        bottom = np.zeros(len(states))
        for action, color, hatch in action_specs:
            values = np.array([counts[model][state][action] / 16 for state in states])
            bars = ax.bar(np.arange(4), values, bottom=bottom, color=color, hatch=hatch, edgecolor="white", linewidth=0.8, label=action)
            for index, (bar, value) in enumerate(zip(bars, values)):
                if value > 0.13:
                    ax.text(bar.get_x() + bar.get_width() / 2, bottom[index] + value / 2, f"{int(value * 16)}", ha="center", va="center", color="white", fontweight="bold")
            bottom += values
        ax.set_title(title, color=B0 if model == "b0" else B1, loc="left")
        ax.set_xticks(np.arange(4), ["FOUND", "AMBIG.", "ABSENT", "INSUFF."])
        ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Response composition within state (n=16/state)")
    axes[1].legend(loc="upper right")
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    fig.suptitle("Figure 3 | State-specific response behavior: ABSTAIN was never emitted", x=0.075, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.075, 0.015, "Labels are counts. INVALID means no unique in-range coordinate and no ABSTAIN token. Hatching duplicates action identity for grayscale printing.", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(top=0.80, bottom=0.20, wspace=0.12)
    save(fig, out, "Figure_3_action_by_state")


def figure_4_paired_effects(data: dict, out: Path) -> None:
    paired = data["metrics"]["paired"]
    entries = [
        ("FOUND Hit@0.08\n(n=16)", paired["found_hit_delta_b1_minus_b0"], paired["found_hit_family_bootstrap_delta_95"]),
        ("Semantic safe-task\n(n=64)", paired["safe_task_delta_b1_minus_b0"], paired["safe_task_family_bootstrap_delta_95"]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(entries))[::-1]
    for yy, (label, delta, interval) in zip(y, entries):
        low, high = interval
        ax.errorbar(delta * 100, yy, xerr=np.array([[delta - low], [high - delta]]) * 100, fmt="o", color=B1, ecolor=INK, elinewidth=1.0, capsize=4, ms=7, zorder=3)
        ax.text(high * 100 + 2.2, yy, f"{delta * 100:+.1f} pp  [{low * 100:+.1f}, {high * 100:+.1f}]", va="center", fontsize=8)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [entry[0] for entry in entries])
    ax.set_xlim(-105, 15); ax.set_xlabel("Paired delta: clean B1 − B0 (percentage points)")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.set_title("Family bootstrap 95% intervals", loc="left")
    panel_label(ax, "a")
    fig.suptitle("Figure 4 | Paired evidence favors keeping B0", x=0.11, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.11, 0.015, f"Safe-task pairing: B1 fixed {paired['b0_wrong_b1_right']} B0 errors and introduced {paired['b0_right_b1_wrong']}; exact McNemar p={paired['mcnemar_exact_two_sided_p']:.6f}.", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(top=0.77, bottom=0.24, left=0.23)
    save(fig, out, "Figure_4_paired_effects")


def figure_5_risk_and_confidence(data: dict, out: Path) -> None:
    metrics = data["metrics"]["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(10.9, 4.25), gridspec_kw={"width_ratios": [1.18, 1]})
    for model, color, hatch, label in (("b0", B0, "", "B0"), ("b1", B1, "//", "clean B1")):
        curve = metrics[model]["risk_coverage"]
        axes[0].step([0] + [point["coverage"] for point in curve], [curve[0]["risk"]] + [point["risk"] for point in curve], where="post", color=color, lw=2.0, label=f"{label} (AURC={metrics[model]['aurc']:.3f})")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1); axes[0].xaxis.set_major_formatter(PercentFormatter(1)); axes[0].yaxis.set_major_formatter(PercentFormatter(1)); axes[0].set_xlabel("Coverage"); axes[0].set_ylabel("Risk = 1 − safe-task accuracy")
    axes[0].grid(); axes[0].set_axisbelow(True); axes[0].legend(loc="lower right")
    axes[0].set_title("Risk–coverage from self-consistency", loc="left")
    panel_label(axes[0], "a")
    rng = np.random.default_rng(12092026)
    for xbase, model, color, marker in ((0, "b0", B0, "o"), (1, "b1", B1, "s")):
        rows = list(data["scored"][model].values())
        for correct, edge, label in ((True, GOOD, "task-correct"), (False, BAD, "task-error")):
            values = [row["self_consistency_confidence"] for row in rows if row["task_correct"] == correct]
            x = xbase + rng.uniform(-0.16, 0.16, size=len(values))
            axes[1].scatter(x, values, s=30, marker=marker, facecolor=color if correct else "white", edgecolor=edge, linewidth=1.0, alpha=0.9, label=label if xbase == 0 else None)
        axes[1].plot([xbase - .20, xbase + .20], [np.mean([row["self_consistency_confidence"] for row in rows]), np.mean([row["self_consistency_confidence"] for row in rows])], color=INK, lw=1.2)
    axes[1].set_xticks([0, 1], ["B0", "clean B1"]); axes[1].set_ylim(-.04, 1.04); axes[1].yaxis.set_major_formatter(PercentFormatter(1)); axes[1].set_ylabel("Self-consistency confidence proxy")
    axes[1].grid(axis="y"); axes[1].set_axisbelow(True); axes[1].legend(loc="upper right")
    axes[1].set_title("Confidence does not ensure safety", loc="left")
    panel_label(axes[1], "b")
    fig.suptitle("Figure 5 | Current self-consistency is a diagnostic proxy, not calibrated uncertainty", x=0.075, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.075, 0.01, "AURC is lower-is-better. Error-detection AUROC is 0.523 for B0 and 0.850 for B1, but B1 has lower task accuracy and a worse AURC; no calibration claim is supported.", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(top=0.79, bottom=0.22, wspace=0.34)
    save(fig, out, "Figure_5_risk_coverage_and_confidence")


def figure_6_contract_diagnostics(data: dict, out: Path) -> None:
    styles = response_styles(data)
    metrics = data["metrics"]["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    categories = [("one_coordinate", "One coordinate", B0, ""), ("multiple_coordinates", "Multiple coordinates", B1, "//"), ("no_coordinate_text", "No-coordinate text", INVALID, "xx")]
    bottom = np.zeros(2)
    for key, label, color, hatch in categories:
        values = np.array([styles["b0"][key], styles["b1"][key]]) / N
        bars = axes[0].bar([0, 1], values, bottom=bottom, color=color, hatch=hatch, edgecolor="white", linewidth=.8, label=label)
        for index, (bar, value) in enumerate(zip(bars, values)):
            if value > .08:
                axes[0].text(bar.get_x() + bar.get_width() / 2, bottom[index] + value / 2, str(int(value * N)), ha="center", va="center", color="white", fontweight="bold")
        bottom += values
    axes[0].set_xticks([0, 1], ["B0", "clean B1"]); axes[0].set_ylim(0, 1); axes[0].yaxis.set_major_formatter(PercentFormatter(1)); axes[0].set_ylabel("Raw answer style (n=64/model)")
    axes[0].grid(axis="y"); axes[0].set_axisbelow(True); axes[0].legend(loc="upper right", fontsize=7)
    axes[0].set_title("Raw response styles", loc="left"); panel_label(axes[0], "a")
    labels = ["Exact\nPOINT/ABSTAIN", "ABSTAIN\nemitted"]
    values = np.array([
        [metrics["b0"]["exact_contract_rate"], metrics["b0"]["nonfound_abstain_recall"]],
        [metrics["b1"]["exact_contract_rate"], metrics["b1"]["nonfound_abstain_recall"]],
    ])
    width = .35
    for offset, index, color, hatch, label in ((-width/2, 0, B0, "", "B0"), (width/2, 1, B1, "//", "clean B1")):
        bars = axes[1].bar(np.arange(2) + offset, values[index], width, color=color, hatch=hatch, edgecolor="white", label=label)
        for bar, value in zip(bars, values[index]):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value + .035, f"{value:.0%}", ha="center", fontweight="bold")
    axes[1].set_xticks(np.arange(2), labels); axes[1].set_ylim(0, 1); axes[1].yaxis.set_major_formatter(PercentFormatter(1)); axes[1].set_ylabel("Rate")
    axes[1].grid(axis="y"); axes[1].set_axisbelow(True); axes[1].legend(loc="upper right")
    axes[1].set_title("Strict contract endpoints", loc="left"); panel_label(axes[1], "b")
    fig.suptitle("Figure 6 | Lower B1 false-accept rate is malformed output, not correct abstention", x=0.075, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.075, 0.01, "Exact contract requires exactly `POINT [(x, y)]` or `ABSTAIN`. Neither model emitted ABSTAIN; semantic coordinate parsing is reported separately from deployment-valid syntax.", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(top=.79, bottom=.23, wspace=.35)
    save(fig, out, "Figure_6_output_contract_diagnostics")


def figure_7_geometry_qc(data: dict, out: Path) -> None:
    gt = data["gt"]
    found = sorted(row["visibility"]["visible_pixels"] for row in gt.values() if row["answerability_state"] == "FOUND")
    insufficient = sorted(row["visibility"]["visible_pixels"] for row in gt.values() if row["answerability_state"] == "INSUFFICIENT_EVIDENCE")
    relation_state = np.zeros((4, 4), dtype=int)
    for row in gt.values():
        relation_state[data["state_order"].index(row["answerability_state"]), data["relation_order"].index(row["relation_variant"])] += 1
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.15), gridspec_kw={"width_ratios": [1.05, .95]})
    rng = np.random.default_rng(409)
    for index, (label, values, color, marker) in enumerate((("FOUND", found, B0, "o"), ("INSUFFICIENT\nEVIDENCE", insufficient, B1, "s"))):
        axes[0].scatter(index + rng.uniform(-.10, .10, len(values)), values, color=color, marker=marker, s=38, edgecolor="white", linewidth=.5, zorder=3)
        axes[0].plot([index-.18, index+.18], [np.median(values), np.median(values)], color=INK, lw=1.2)
    axes[0].axhline(120, color=BAD, linestyle="--", lw=1.2, label="Sufficient threshold = 120 px")
    axes[0].set_yscale("log"); axes[0].set_ylim(4, 40000); axes[0].set_xticks([0, 1], ["FOUND\n(n=16)", "INSUFFICIENT\n(n=16)"])
    axes[0].set_ylabel("Target visible semantic-mask pixels (log scale)"); axes[0].grid(axis="y"); axes[0].set_axisbelow(True); axes[0].legend(loc="upper left", fontsize=7)
    axes[0].set_title("Geometry-verified visibility separation", loc="left"); panel_label(axes[0], "a")
    image = axes[1].imshow(relation_state, vmin=0, vmax=4, cmap="Blues")
    for row in range(4):
        for col in range(4):
            axes[1].text(col, row, str(relation_state[row, col]), ha="center", va="center", color="white" if relation_state[row, col] >= 3 else INK, fontweight="bold")
    axes[1].set_xticks(range(4), ["leftmost", "rightmost", "2nd left", "2nd right"], rotation=25, ha="right")
    axes[1].set_yticks(range(4), ["FOUND", "AMBIG.", "ABSENT", "INSUFF."])
    axes[1].set_title("Factorial state × relation balance", loc="left"); panel_label(axes[1], "b")
    cbar = fig.colorbar(image, ax=axes[1], fraction=.046, pad=.04); cbar.set_label("Families per cell")
    fig.suptitle("Figure 7 | Geometry QC verifies the intended answerability regimes", x=0.075, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.075, .01, "FOUND target evidence is ≥120 px; INSUFFICIENT_EVIDENCE is 1–119 px and near-frame by locked geometry rule. AMBIGUOUS/ABSENT have no single target point by definition.", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(top=.79, bottom=.23, wspace=.42)
    save(fig, out, "Figure_7_geometry_qc")


def write_markdown(data: dict, out: Path) -> None:
    captions = """# Figure legends

## Figure 1 — Study design and QC

The locked Dev dataset has 64 independent parent-families, balanced across four
answerability states and four ordinal-relation variants. The integrity panel is
derived from the locked contract, materialized dataset manifest, and geometry-QC
report. It does not claim Test-IID/OOD generalization.

## Figure 2 — Primary endpoints

FOUND Hit@0.08 is assessed only where one target is geometry-verified (n=16).
The semantic parser accepts one valid normalized coordinate so that legacy model
behavior can be diagnosed separately from exact output syntax. Wilson 95%
intervals are drawn where defined. Exact contract compliance is 0% for both.

## Figure 3 — Action distribution by answerability state

The desired action is POINT only for FOUND and ABSTAIN for all other states.
Neither B0 nor clean B1 returns ABSTAIN. Counts are written inside non-zero bar
segments; action is encoded by colour and hatch.

## Figure 4 — Paired B1−B0 effects

The dots are paired effects and horizontal intervals are 10,000-draw family
bootstrap 95% intervals computed by the preregistered scorer. For semantic
safe-task accuracy, B1 fixes 0 B0 errors and introduces 12; exact McNemar
p=0.000488.

## Figure 5 — Risk and confidence diagnostic

Risk–coverage uses three stochastic self-consistency draws per family. This is
not calibrated probability: B1 has higher error AUROC but worse AURC and lower
task accuracy. No calibration claim is made.

## Figure 6 — Output-contract diagnostic

Raw answer styles are counted from raw model strings. A lower POINT-only false
accept rate for B1 is not interpreted as a safety gain because it comes from
malformed/multiple-coordinate output, not correct ABSTAIN.

## Figure 7 — Geometry QC

The left panel uses semantic-mask visibility measured by the evaluator, never
as model input. The right panel shows the observed 4×4 factorial count matrix.
"""
    (out / "FIGURE_LEGENDS.md").write_text(captions, encoding="utf-8")
    readme = """# Gazebo answerability-v2 paper figure package

This folder was rendered by `protocol/render_gazebo_answerability_v2_paper_figures.py` directly from locked workspace artifacts. It contains no generated or illustrative imagery: every figure, table, and number is traceable to the contract, QC, blinded prediction, or scored-result files listed in `MANIFEST.json`.

## Contents

- `Figure_*.png`: 300 DPI raster figures for slide/deck use.
- `Figure_*.svg` and `Figure_*.pdf`: editable vector figures for manuscript layout.
- `Table_*.csv`: exact values used for figures.
- `FIGURE_LEGENDS.md`: manuscript-ready figure legends and caveats.
- `MANIFEST.json`: source hashes and output hashes for reproducibility.

## Reproduce / verify

```bash
export LD_LIBRARY_PATH=/home/dhcn/ur_ws/src/myproject/.conda-roborefer/lib:${LD_LIBRARY_PATH:-}
./.conda-roborefer/bin/python -s protocol/render_gazebo_answerability_v2_paper_figures.py render --output ketqua/Gazebo_dev_answerability_v2_paper_figures
./.conda-roborefer/bin/python -s protocol/render_gazebo_answerability_v2_paper_figures.py check --output ketqua/Gazebo_dev_answerability_v2_paper_figures
```

The renderer fails if the 64-record dataset, QC status, prediction alignment, or source hashes are inconsistent. It refuses to overwrite a non-empty output directory unless that directory was created by this renderer and `--overwrite` is specified.

## Design policy

- Vector exports retain editable text and marks; PNG exports use 300 DPI.
- Typography and editable vector overlays follow the practical figure guidance from [Nature](https://www.nature.com/documents/natrev-figure-guidelines-v1.pdf).
- The blue/orange primary comparison and direct labels/hatches follow accessible-figure principles: do not rely only on color, explain visual encodings, and make figures readable in grayscale. See [PLOS Biology](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001161) and [PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013657).

## Scope

This is a 64-family **Dev failure-analysis** package, not final generalization evidence. No SAM2 was used; B2 and Test-IID/Test-OOD remain sealed. The scientific conclusion is `KEEP_B0_AND_ANALYZE_ERRORS`: B0 has strong diagnostic FOUND grounding, while neither B0 nor clean B1 meets the required answerability/output contract.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")


def render(output: Path, overwrite: bool) -> None:
    data = validate_and_collect()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}; pass --overwrite after review")
        previous_manifest_path = output / "MANIFEST.json"
        if not previous_manifest_path.is_file():
            raise FileExistsError(
                "refusing to overwrite a directory not previously created by this renderer; "
                "choose a new output directory instead"
            )
        previous_manifest = load_json(previous_manifest_path)
        expected_renderer = str(Path(__file__).relative_to(ROOT))
        if previous_manifest.get("renderer") != expected_renderer:
            raise FileExistsError(
                "refusing to overwrite a directory managed by another tool; choose a new output directory instead"
            )
        managed_names = set(previous_manifest.get("generated_file_sha256", {})) | {"MANIFEST.json"}
        unknown_entries = sorted(path.name for path in output.iterdir() if path.name not in managed_names)
        if unknown_entries:
            raise FileExistsError(
                "refusing to delete files not listed in the renderer manifest: " + ", ".join(unknown_entries)
            )
        for name in managed_names:
            path = output / name
            if path.exists():
                if not path.is_file():
                    raise ValueError(f"refusing to modify non-file output: {path}")
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    configure()
    write_tables(data, output)
    figure_1_design_and_qc(data, output)
    figure_2_primary_endpoints(data, output)
    figure_3_action_by_state(data, output)
    figure_4_paired_effects(data, output)
    figure_5_risk_and_confidence(data, output)
    figure_6_contract_diagnostics(data, output)
    figure_7_geometry_qc(data, output)
    write_markdown(data, output)
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_PATHS.values()}
    output_hashes = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "MANIFEST.json"}
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "renderer": str(Path(__file__).relative_to(ROOT)),
        "renderer_sha256": sha256(Path(__file__)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL,
        "source_sha256": source_hashes,
        "generated_file_sha256": output_hashes,
        "no_sam2": True,
        "b2_opened": False,
        "test_iid_ood_access": False,
        "data_policy": "All figures and tables are derived from existing locked evaluation artifacts; no synthetic or generated imagery.",
    }
    write_json(output / "MANIFEST.json", manifest)
    print(json.dumps({"status": "PASS", "output": str(output), "figures": len(list(output.glob("Figure_*.png"))), "tables": len(list(output.glob("Table_*.csv")))}, indent=2))


def check(output: Path) -> None:
    validate_and_collect()
    manifest_path = output / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"figure manifest missing: {manifest_path}")
    manifest = load_json(manifest_path)
    expected_sources = {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_PATHS.values()}
    if manifest.get("source_sha256") != expected_sources:
        raise ValueError("source artifacts changed since figures were rendered; rerender is required")
    if manifest.get("renderer_sha256") != sha256(Path(__file__)):
        raise ValueError("renderer changed since figures were rendered; rerender is required")
    expected_outputs = manifest.get("generated_file_sha256", {})
    actual_outputs = {name: sha256(output / name) for name in expected_outputs if (output / name).is_file()}
    if actual_outputs != expected_outputs or len(actual_outputs) != len(expected_outputs):
        raise ValueError("one or more generated figures/tables changed or are missing")
    for stem in [f"Figure_{index}_" for index in range(1, 8)]:
        if not any(path.name.startswith(stem) and path.suffix == ".png" for path in output.iterdir()):
            raise ValueError(f"missing expected raster output for {stem}")
    print(json.dumps({"status": "PASS", "output": str(output.resolve()), "sources_verified": len(expected_sources), "outputs_verified": len(expected_outputs)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    render_parser.add_argument("--overwrite", action="store_true")
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == "render":
        render(args.output, args.overwrite)
    else:
        check(args.output)


if __name__ == "__main__":
    main()
