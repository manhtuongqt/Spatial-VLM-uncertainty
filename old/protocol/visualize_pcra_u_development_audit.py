#!/usr/bin/env python3
"""Render a post-freeze scientific visualization supplement for the dev audit.

This tool is deliberately read-only with respect to the frozen audit contract,
manifest, report, decision lock, checkpoint, dataset, and baseline.  It reads
three already-computed exploratory CSV tables and produces deterministic SVG/
PNG figures plus provenance.  It does not run inference, recompute bootstrap
statistics, fit calibration, open tests, or train a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


WORKSPACE = Path(__file__).resolve().parents[1]
RUN_ROOT = WORKSPACE / "results/pcra_u_runs/pcra_u_development_failure_audit_20260824"
TABLE_ROOT = RUN_ROOT / "tables"
FIGURE_ROOT = RUN_ROOT / "figures"
METHOD_TABLE = TABLE_ROOT / "table_dev_method_comparison_exploratory.csv"
FAILURE_TABLE = TABLE_ROOT / "table_dev_failure_cases_exploratory.csv"
PAIRED_TABLE = TABLE_ROOT / "table_dev_paired_deltas_exploratory.csv"
DERIVED_FAILURE_TABLE = TABLE_ROOT / "table_dev_failure_visualization_counts.csv"
SUPPLEMENT_REPORT = RUN_ROOT / "VISUALIZATION_SUPPLEMENT.md"
SUPPLEMENT_MANIFEST = RUN_ROOT / "checks/visualization_supplement_manifest.json"
RUN_ARTIFACT_MANIFEST = RUN_ROOT / "checks/artifact_manifest.json"

FIGURES = {
    "F10": FIGURE_ROOT / "F10_dev_method_comparison_from_csv",
    "F11": FIGURE_ROOT / "F11_dev_failure_diagnostics_from_csv",
    "F12": FIGURE_ROOT / "F12_dev_paired_family_bootstrap_from_csv",
}

PROTECTED_ARTIFACTS = {
    "audit_contract": WORKSPACE / "protocol/PCRA_U_DEVELOPMENT_FAILURE_AUDIT_CONTRACT.md",
    "eval_manifest": WORKSPACE / "protocol/pcra_u_development_eval_manifest.json",
    "frozen_audit_code": WORKSPACE / "protocol/pcra_u_development_failure_audit.py",
    "audit_report_json": RUN_ROOT / "PCRA_U_DEVELOPMENT_FAILURE_AUDIT_REPORT.json",
    "architecture_freeze_lock": WORKSPACE / "protocol/pcra_u_architecture_freeze_lock.json",
    "selected_checkpoint_model": WORKSPACE / "results/pcra_u_runs/pcra_u_development_train_20260824/checkpoints/development/step_000002000/model.safetensors",
    "selected_checkpoint_manifest": WORKSPACE / "results/pcra_u_runs/pcra_u_development_train_20260824/checkpoints/development/step_000002000/manifest.json",
    "selected_checkpoint_metadata": WORKSPACE / "results/pcra_u_runs/pcra_u_development_train_20260824/checkpoints/development/step_000002000/metadata.json",
}

METHOD_ORDER = ["B0", "B1", "B2", "U1", "P1", "A_NO_DEPTH"]
METHOD_DISPLAY = {
    "B0": "B0",
    "B1": "B1",
    "B2": "B2*",
    "U1": "U1†",
    "P1": "P1",
    "A_NO_DEPTH": "No-depth†",
}
METHOD_COLORS = {
    "B0": "#6B7280",
    "B1": "#4C78A8",
    "B2": "#9ECAE1",
    "U1": "#F58518",
    "P1": "#2E8B57",
    "A_NO_DEPTH": "#B279A2",
}
FAILURE_COLORS = {"false_FOUND": "#D1495B", "point_outside_target": "#3973AC"}
VARIANT_ORDER = [
    "clean",
    "semantic_counterfactual",
    "relation_counterfactual",
    "depth_corruption",
    "occlusion_view_counterfactual",
]
VARIANT_DISPLAY = {
    "clean": "clean",
    "semantic_counterfactual": "semantic CF",
    "relation_counterfactual": "relation CF",
    "depth_corruption": "depth corruption",
    "occlusion_view_counterfactual": "occlusion/view CF",
}
VARIANT_MARKERS = {
    "clean": "o",
    "semantic_counterfactual": "s",
    "relation_counterfactual": "D",
    "depth_corruption": "P",
    "occlusion_view_counterfactual": "^",
}
CONTRAST_ORDER = [
    "P1_minus_U1",
    "P1_minus_A_NO_DEPTH",
    "B1_minus_B0",
    "P1_minus_B1",
    "P1_minus_B2_clean",
]
CONTRAST_DISPLAY = {
    "P1_minus_U1": "P1 − U1†",
    "P1_minus_A_NO_DEPTH": "P1 − No-depth†",
    "B1_minus_B0": "B1 − B0",
    "P1_minus_B1": "P1 − B1",
    "P1_minus_B2_clean": "P1 − B2*",
}
CONTRAST_COLORS = {
    "P1_minus_U1": "#2E8B57",
    "P1_minus_A_NO_DEPTH": "#4C78A8",
    "B1_minus_B0": "#6B7280",
    "P1_minus_B1": "#F58518",
    "P1_minus_B2_clean": "#8C6BB1",
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


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise VisualizationError(f"Missing source table: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("scientific_status") != "EXPLORATORY_DEV_ONLY" for row in rows):
        raise VisualizationError(f"Invalid or non-exploratory source table: {path}")
    return rows


def as_float(row: Mapping[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in {"", None}:
        raise VisualizationError(f"Missing numeric field {field}")
    result = float(value)
    if not np.isfinite(result):
        raise VisualizationError(f"Non-finite numeric field {field}")
    return result


def as_int(row: Mapping[str, str], field: str) -> int:
    return int(row[field])


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "svg.hashsalt": "pcra-u-development-audit-visualization-v1",
        }
    )


def save_figure(fig: plt.Figure, stem: Path, title: str) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".svg"), stem.with_suffix(".png")]
    common = {"Creator": "visualize_pcra_u_development_audit.py", "Date": "2026-08-24", "Title": title}
    fig.savefig(outputs[0], format="svg", bbox_inches="tight", metadata=common)
    fig.savefig(outputs[1], format="png", dpi=220, bbox_inches="tight", metadata=common)
    plt.close(fig)
    return outputs


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.09, 1.06, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def plot_method_comparison(rows: Sequence[Mapping[str, str]]) -> list[Path]:
    by_method = {row["method"]: row for row in rows}
    if set(by_method) != set(METHOD_ORDER):
        raise VisualizationError(f"Unexpected method set: {sorted(by_method)}")
    ordered = [by_method[name] for name in METHOD_ORDER]
    x = np.arange(len(ordered))
    colors = [METHOD_COLORS[row["method"]] for row in ordered]

    fig = plt.figure(figsize=(13.8, 8.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08])
    ax_ground = fig.add_subplot(grid[0, 0])
    ax_f1 = fig.add_subplot(grid[0, 1])
    ax_trade = fig.add_subplot(grid[1, :])

    ground = [as_float(row, "family_found_point_in_target") for row in ordered]
    ax_ground.axvspan(1.55, 2.45, color="#F0F0F0", zorder=0)
    bars = ax_ground.bar(x, ground, color=colors, edgecolor="#333333", linewidth=0.7)
    bars[2].set_hatch("///")
    for index, (bar, row, value) in enumerate(zip(bars, ordered, ground)):
        ax_ground.text(
            bar.get_x() + bar.get_width() / 2,
            min(0.985, value + 0.025),
            f"{value:.3f}\nF={as_int(row, 'families_with_found')}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax_ground.set_xticks(x, [METHOD_DISPLAY[row["method"]] for row in ordered])
    ax_ground.set_ylim(0, 1.08)
    ax_ground.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_ground.set_ylabel("Family-equal point-in-target")
    ax_ground.set_title("Grounding on ground-truth FOUND")
    ax_ground.grid(axis="y")
    panel_label(ax_ground, "A")

    macro_f1 = [as_float(row, "answerability_macro_f1") for row in ordered]
    ax_f1.axvspan(1.55, 2.45, color="#F0F0F0", zorder=0)
    bars = ax_f1.bar(x, macro_f1, color=colors, edgecolor="#333333", linewidth=0.7)
    bars[2].set_hatch("///")
    for bar, row, value in zip(bars, ordered, macro_f1):
        ax_f1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}\nn={as_int(row, 'eligible_samples')}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax_f1.set_xticks(x, [METHOD_DISPLAY[row["method"]] for row in ordered])
    ax_f1.set_ylim(0, 1.08)
    ax_f1.set_ylabel("Answerability macro-F1")
    ax_f1.set_title("Four-state answerability (proxy for forced-point baselines)")
    ax_f1.grid(axis="y")
    panel_label(ax_f1, "B")

    for row in ordered:
        method = row["method"]
        coverage = as_float(row, "coverage_predicted_found")
        false_rate = as_float(row, "false_found_rate")
        marker = {"B0": "s", "B1": "D", "B2": "o"}.get(method, "o")
        size = {"B0": 190, "B1": 125, "B2": 65}.get(method, 115)
        face = "none" if method in {"B0", "B1", "B2"} else METHOD_COLORS[method]
        ax_trade.scatter(
            coverage,
            false_rate,
            s=size,
            marker=marker,
            facecolor=face,
            edgecolor=METHOD_COLORS[method],
            linewidth=2.0,
            zorder=5,
        )
    annotations = {
        "B0": (-98, 28),
        "B1": (-18, 48),
        "B2": (42, 24),
        "U1": (12, 12),
        "P1": (12, 12),
        "A_NO_DEPTH": (12, 12),
    }
    for row in ordered:
        method = row["method"]
        coverage = as_float(row, "coverage_predicted_found")
        false_rate = as_float(row, "false_found_rate")
        count = as_int(row, "false_found_count")
        denominator = as_int(row, "false_found_denominator")
        ax_trade.annotate(
            f"{METHOD_DISPLAY[method]}: {count}/{denominator}\ncoverage={coverage:.1%}",
            (coverage, false_rate),
            xytext=annotations[method],
            textcoords="offset points",
            fontsize=8.5,
            arrowprops={"arrowstyle": "-", "color": METHOD_COLORS[method], "lw": 0.8},
        )
    ax_trade.set_xlim(0, 1.08)
    ax_trade.set_ylim(-0.04, 1.14)
    ax_trade.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax_trade.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_trade.set_xlabel("Coverage predicted FOUND  →")
    ax_trade.set_ylabel("False-FOUND rate  (lower is better)")
    ax_trade.set_title("Safety–coverage trade-off; this is not calibrated risk–coverage")
    ax_trade.grid(True)
    panel_label(ax_trade, "C")

    fig.suptitle(
        "F10 — Exploratory dev method comparison from locked CSV",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.025,
        "80 dev families / 400 variants except B2*: clean-only 80 samples (32 FOUND, 28 risky), gate accepted 80/80.  "
        "† same-checkpoint post-hoc intervention, not retrained.  No-depth false-FOUND=2/95 accompanies only 4.5% coverage.",
        ha="center",
        va="top",
        fontsize=9,
    )
    return save_figure(fig, FIGURES["F10"], "F10 — Exploratory dev method comparison")


def aggregate_failure_counts(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dimension in ("relation", "variant"):
        strata = sorted({row[dimension] for row in rows})
        for stratum in strata:
            selected = [row for row in rows if row[dimension] == stratum]
            for failure_type in ("false_FOUND", "point_outside_target"):
                typed = [row for row in selected if row["failure_type"] == failure_type]
                output.append(
                    {
                        "scientific_status": "EXPLORATORY_DEV_ONLY",
                        "dimension": dimension,
                        "stratum": stratum,
                        "failure_type": failure_type,
                        "failure_sample_count": len(typed),
                        "distinct_failure_family_count": len({row["family_id"] for row in typed}),
                        "total_failure_sample_count_in_stratum": len(selected),
                        "total_distinct_failure_family_count_in_stratum": len({row["family_id"] for row in selected}),
                        "interpretation": "failure-only count; denominator unavailable; not an error rate",
                    }
                )
    return output


def write_failure_counts(rows: Sequence[Mapping[str, Any]]) -> None:
    DERIVED_FAILURE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with DERIVED_FAILURE_TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_failure_diagnostics(rows: Sequence[Mapping[str, str]], aggregate: Sequence[Mapping[str, Any]]) -> list[Path]:
    if Counter(row["failure_type"] for row in rows) != Counter({"point_outside_target": 24, "false_FOUND": 9}):
        raise VisualizationError("Failure table no longer contains the locked 24 outside + 9 false-FOUND cases")

    fig = plt.figure(figsize=(14.2, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    ax_relation = fig.add_subplot(grid[0, 0])
    ax_variant = fig.add_subplot(grid[0, 1])
    ax_scatter = fig.add_subplot(grid[1, :])

    relation_totals = Counter(row["relation"] for row in rows)
    relation_order = [name for name, _ in sorted(relation_totals.items(), key=lambda item: (item[1], item[0]))]
    y = np.arange(len(relation_order))
    left = np.zeros(len(relation_order))
    for failure_type in ("false_FOUND", "point_outside_target"):
        values = np.array([sum(row["relation"] == relation and row["failure_type"] == failure_type for row in rows) for relation in relation_order])
        ax_relation.barh(y, values, left=left, color=FAILURE_COLORS[failure_type], label=failure_type.replace("_", " "))
        left += values
    for index, relation in enumerate(relation_order):
        selected = [row for row in rows if row["relation"] == relation]
        ax_relation.text(left[index] + 0.18, index, f"{int(left[index])} rows / {len({r['family_id'] for r in selected})} fam", va="center", fontsize=8)
    ax_relation.set_yticks(y, relation_order)
    ax_relation.set_xlim(0, max(left) * 1.35)
    ax_relation.set_xlabel("Failure rows (count)")
    ax_relation.set_title("Failure counts by relation")
    ax_relation.grid(axis="x")
    ax_relation.legend(loc="lower right")
    panel_label(ax_relation, "A")

    present_variants = [name for name in VARIANT_ORDER if any(row["variant"] == name for row in rows)]
    y = np.arange(len(present_variants))
    left = np.zeros(len(present_variants))
    for failure_type in ("false_FOUND", "point_outside_target"):
        values = np.array([sum(row["variant"] == variant and row["failure_type"] == failure_type for row in rows) for variant in present_variants])
        ax_variant.barh(y, values, left=left, color=FAILURE_COLORS[failure_type])
        left += values
    for index, variant in enumerate(present_variants):
        selected = [row for row in rows if row["variant"] == variant]
        ax_variant.text(left[index] + 0.18, index, f"{int(left[index])} rows / {len({r['family_id'] for r in selected})} fam", va="center", fontsize=8)
    ax_variant.set_yticks(y, [VARIANT_DISPLAY[name] for name in present_variants])
    ax_variant.set_xlim(0, max(left) * 1.35)
    ax_variant.set_xlabel("Failure rows (count)")
    ax_variant.set_title("Failure counts by variant")
    ax_variant.grid(axis="x")
    panel_label(ax_variant, "B")

    outside = [row for row in rows if row["failure_type"] == "point_outside_target"]
    relations = sorted({row["relation"] for row in outside})
    palette = plt.get_cmap("tab10")
    relation_colors = {relation: palette(index % 10) for index, relation in enumerate(relations)}
    for row in outside:
        ax_scatter.scatter(
            as_float(row, "target_mask_fraction") * 100.0,
            as_float(row, "normalized_centroid_error"),
            s=70,
            marker=VARIANT_MARKERS[row["variant"]],
            facecolor=relation_colors[row["relation"]],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
        )
    for row in sorted(outside, key=lambda item: as_float(item, "normalized_centroid_error"), reverse=True)[:3]:
        ax_scatter.annotate(
            row["family_id"].replace("v211dev_family_", "fam "),
            (as_float(row, "target_mask_fraction") * 100.0, as_float(row, "normalized_centroid_error")),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8,
        )
    relation_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=relation_colors[name], markeredgecolor="white", markersize=8, label=name)
        for name in relations
    ]
    variant_handles = [
        Line2D([0], [0], marker=VARIANT_MARKERS[name], color="#555555", linestyle="none", markersize=7, label=VARIANT_DISPLAY[name])
        for name in present_variants
        if any(row["variant"] == name for row in outside)
    ]
    first_legend = ax_scatter.legend(handles=relation_handles, title="Relation", loc="upper right", ncol=min(4, len(relation_handles)))
    ax_scatter.add_artist(first_legend)
    ax_scatter.legend(handles=variant_handles, title="Variant", loc="lower right", ncol=min(4, len(variant_handles)))
    ax_scatter.set_xlabel("Target-mask area (% of image)")
    ax_scatter.set_ylabel("Normalized prediction-to-target centroid error")
    ax_scatter.set_title("The 24 point-outside-target cases: mask size versus localization error")
    ax_scatter.grid(True)
    panel_label(ax_scatter, "C")

    fig.suptitle("F11 — Exploratory dev failure diagnostics from 33 observed failure rows", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.025,
        "Failure-only table: 9 false-FOUND rows (7 families) + 24 point-outside rows (22 families).  "
        "Panels A/B are counts, not relation/variant error rates; exposure denominators are not present in this CSV.",
        ha="center",
        va="top",
        fontsize=9,
    )
    return save_figure(fig, FIGURES["F11"], "F11 — Exploratory dev failure diagnostics")


def plot_paired_deltas(rows: Sequence[Mapping[str, str]]) -> list[Path]:
    lookup = {(row["contrast"], row["metric"]): row for row in rows}
    expected = {(contrast, metric) for contrast in CONTRAST_ORDER for metric in ("answerability_macro_f1", "false_found_rate", "family_found_point_in_target")}
    if set(lookup) != expected:
        raise VisualizationError("Paired-delta table does not have the expected 5 contrasts x 3 metrics")

    metrics = [
        ("answerability_macro_f1", "Δ answerability macro-F1", "higher is better →"),
        ("false_found_rate", "Δ false-FOUND rate", "← lower is better"),
        ("family_found_point_in_target", "Δ family-equal point-in-target", "higher is better →"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.4), sharey=True, constrained_layout=True)
    y = np.arange(len(CONTRAST_ORDER))
    for axis_index, (ax, (metric, title, direction)) in enumerate(zip(axes, metrics)):
        ax.axvspan(3.5, 4.5, color="#F0F0F0", zorder=0)
        ax.axvline(0.0, color="#333333", linewidth=1.1, linestyle="--", zorder=1)
        if metric != "false_found_rate":
            ax.axvline(-0.02, color="#A65E2E", linewidth=0.9, linestyle=":", zorder=1)
        for index, contrast in enumerate(CONTRAST_ORDER):
            row = lookup[(contrast, metric)]
            estimate = as_float(row, "estimate")
            low = as_float(row, "ci95_low")
            high = as_float(row, "ci95_high")
            ax.errorbar(
                estimate,
                index,
                xerr=np.array([[estimate - low], [high - estimate]]),
                fmt="o",
                color=CONTRAST_COLORS[contrast],
                ecolor=CONTRAST_COLORS[contrast],
                elinewidth=2.2,
                capsize=4,
                markersize=6.5,
                zorder=4,
            )
        values = [as_float(lookup[(contrast, metric)], field) for contrast in CONTRAST_ORDER for field in ("ci95_low", "ci95_high")]
        low_limit, high_limit = min(values), max(values)
        padding = max(0.08, (high_limit - low_limit) * 0.17)
        ax.set_xlim(low_limit - padding, high_limit + padding)
        ax.set_xlabel(direction)
        ax.set_title(title)
        ax.grid(axis="x")
        panel_label(ax, chr(ord("A") + axis_index))
    axes[0].set_yticks(y, [CONTRAST_DISPLAY[name] for name in CONTRAST_ORDER])
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)
    fig.suptitle("F12 — Paired family-cluster bootstrap deltas (5,000 resamples; 95% CI)", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.015,
        "All deltas are left minus right.  † same-checkpoint post-hoc intervention, not retrained.  "
        "Dotted −0.02 gate applies only to P1−U1† and P1−No-depth† for macro-F1/grounding.  "
        "B2* is clean-only; its row is shaded and is not a full-400 comparison.",
        ha="center",
        va="top",
        fontsize=9,
    )
    return save_figure(fig, FIGURES["F12"], "F12 — Paired family-cluster bootstrap deltas")


def rebuild_run_artifact_manifest() -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in RUN_ROOT.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        rows.append({"path": str(path.relative_to(RUN_ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"schema_version": 1, "file_count": len(rows), "files": rows, "tree_commitment_sha256": canonical_sha256(rows)}
    write_json(RUN_ARTIFACT_MANIFEST, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace only this visualization supplement's derived outputs")
    args = parser.parse_args()

    outputs_expected = [
        *(stem.with_suffix(ext) for stem in FIGURES.values() for ext in (".svg", ".png")),
        DERIVED_FAILURE_TABLE,
        SUPPLEMENT_REPORT,
        SUPPLEMENT_MANIFEST,
    ]
    existing = [path for path in outputs_expected if path.exists()]
    if existing and not args.overwrite:
        raise VisualizationError(f"Derived visualization output already exists; pass --overwrite: {existing[0]}")

    configure_style()
    protected_before = {name: sha256_file(path) for name, path in PROTECTED_ARTIFACTS.items()}
    source_tables = [METHOD_TABLE, FAILURE_TABLE, PAIRED_TABLE]
    source_hashes = {relative(path): sha256_file(path) for path in source_tables}
    method_rows = read_csv(METHOD_TABLE)
    failure_rows = read_csv(FAILURE_TABLE)
    paired_rows = read_csv(PAIRED_TABLE)

    generated: list[Path] = []
    generated.extend(plot_method_comparison(method_rows))
    aggregate = aggregate_failure_counts(failure_rows)
    write_failure_counts(aggregate)
    generated.append(DERIVED_FAILURE_TABLE)
    generated.extend(plot_failure_diagnostics(failure_rows, aggregate))
    generated.extend(plot_paired_deltas(paired_rows))

    report = """# Development audit visualization supplement

- Status: `EXPLORATORY_DEV_ONLY` and `POST_FREEZE_RENDER_ONLY`.
- Source: exactly the three corrected CSV tables requested by the user.
- Model inference/training/calibration/test access: none.

## Figures

- `F10_dev_method_comparison_from_csv`: grounding, answerability and false-FOUND/coverage trade-off.

![F10 method comparison](figures/F10_dev_method_comparison_from_csv.png)

- `F11_dev_failure_diagnostics_from_csv`: failure counts by relation/variant and the 24 point-outside diagnostic scatter.

![F11 failure diagnostics](figures/F11_dev_failure_diagnostics_from_csv.png)

- `F12_dev_paired_family_bootstrap_from_csv`: existing paired estimates and 95% family-bootstrap intervals.

![F12 paired bootstrap](figures/F12_dev_paired_family_bootstrap_from_csv.png)

## Interpretation constraints

- B2 is clean-only (80 samples; 32 FOUND; 28 risky) and accepted 80/80, so it is not directly comparable to full-400 methods as an improvement claim.
- No-depth has false-FOUND 2/95 together with only 4.5% predicted-FOUND coverage; the low failure count cannot be interpreted without coverage.
- The failure-case CSV contains failures only. F11 reports counts, not error rates by relation, variant, asset, mask size, or depth.
- U1 and no-depth are same-checkpoint post-hoc interventions, not retrained causal ablations.
- No ROC, ECE, calibrated risk–coverage, Calibration, Test-IID or Test-OOD result is plotted.
"""
    write_text(SUPPLEMENT_REPORT, report)
    generated.append(SUPPLEMENT_REPORT)

    protected_after = {name: sha256_file(path) for name, path in PROTECTED_ARTIFACTS.items()}
    if protected_before != protected_after:
        raise VisualizationError("A frozen audit/checkpoint artifact changed during rendering")

    manifest = {
        "schema_version": 1,
        "supplement_id": "pcra_u_development_audit_visualization_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "EXPLORATORY_DEV_ONLY_POST_FREEZE_RENDER_ONLY",
        "script": {"path": relative(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        "source_tables": [{"path": path, "sha256": digest} for path, digest in sorted(source_hashes.items())],
        "outputs": [
            {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(generated)
        ],
        "protected_before": protected_before,
        "protected_after": protected_after,
        "protected_unchanged": protected_before == protected_after,
        "environment": {
            "python": platform.python_version(),
            "matplotlib": mpl.__version__,
            "numpy": np.__version__,
        },
        "claims": {
            "render_only": True,
            "model_metrics_recomputed": False,
            "failure_count_aggregation_for_display": True,
            "model_inference": False,
            "training": False,
            "calibration_fit": False,
            "test_opened": False,
            "tables_02_to_06": "NOT_RUN",
        },
        "caveats": [
            "B2 is clean-only and must not be treated as a full-400 comparison.",
            "A_NO_DEPTH low false-FOUND accompanies 4.5% predicted-FOUND coverage.",
            "Failure-case counts are not subgroup error rates because exposure denominators are absent.",
            "U1 and A_NO_DEPTH are post-hoc, not retrained causal ablations.",
        ],
    }
    write_json(SUPPLEMENT_MANIFEST, manifest)
    artifact = rebuild_run_artifact_manifest()
    print(
        json.dumps(
            {
                "status": "PASS",
                "figures": 3,
                "formats": ["svg", "png"],
                "protected_unchanged": True,
                "run_artifacts": artifact["file_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
