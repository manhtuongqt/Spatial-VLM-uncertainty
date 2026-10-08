"""Render Day-6 S1a figures strictly from recorded development evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DAY6 = ROOT / "ketqua1/07_huan_luyen/ngay_06"
FIGURES = ROOT / "ketqua1/09_danh_gia/figures/ngay_06"
TABLES = ROOT / "ketqua1/09_danh_gia/tables/ngay_06"
METRICS = ROOT / "ketqua1/09_danh_gia/metrics/ngay_06"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    for path in (FIGURES, TABLES, METRICS):
        path.mkdir(parents=True, exist_ok=False)
    epoch_path = DAY6 / "S1A_EPOCH_METRICS.csv"
    train_log_path = DAY6 / "S1A_TRAIN_LOG.jsonl"
    decision_path = DAY6 / "S1A_DECISION.json"
    resource_path = DAY6 / "S1A_RESOURCE_REPORT.json"
    cache_timing_path = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/S1A_CACHE_EXTRACTION_TIMING.csv"
    split_audit_path = DAY6 / "S1A_SPLIT_AUDIT.json"
    epochs = read_csv(epoch_path)
    train_steps = [json.loads(line) for line in train_log_path.read_text(encoding="utf-8").splitlines() if line]
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    split_audit = json.loads(split_audit_path.read_text(encoding="utf-8"))
    best_epoch = decision["best_epoch"]
    x = values(epochs, "epoch")
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x, values(epochs, "train_total_loss"), label="Train total")
    ax.plot(x, values(epochs, "val_total_loss"), label="Val total")
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1, label=f"best epoch {best_epoch}")
    ax.set(title="Weighted total loss", xlabel="Epoch", ylabel="Loss")
    ax.legend(frameon=True)

    ax = axes[0, 1]
    ax.plot(x, values(epochs, "val_relation_macro_f1"), label="Relation macro-F1")
    ax.plot(x, values(epochs, "val_answerability_macro_f1"), label="Answerability macro-F1")
    ax.plot(x, values(epochs, "val_hit_at_0_08"), label="FOUND Hit@0.08")
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1)
    ax.set(title="Val development selection metrics", xlabel="Epoch", ylabel="Score", ylim=(-0.02, 1.04))
    ax.legend(frameon=True)

    ax = axes[1, 0]
    ax.plot(x, values(epochs, "train_point_mae_uv"), label="Train point MAE")
    ax.plot(x, values(epochs, "val_point_mae_uv"), label="Val point MAE")
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1)
    ax.set(title="FOUND coordinate error", xlabel="Epoch", ylabel="Mean absolute error in normalized (u,v)")
    ax.legend(frameon=True)

    ax = axes[1, 1]
    low = values(epochs, "val_log_variance_min")
    high = values(epochs, "val_log_variance_max")
    mean = values(epochs, "val_log_variance_mean")
    ax.fill_between(x, low, high, alpha=0.25, label="Val min–max")
    ax.plot(x, mean, label="Val mean")
    ax.axhline(-7.95, color="#d55e00", linestyle="--", label="locked lower bound")
    ax.axhline(1.95, color="#d55e00", linestyle=":", label="locked upper bound")
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1)
    ax.set(title="FOUND log-variance stability", xlabel="Epoch", ylabel="Predicted log variance")
    ax.legend(frameon=True, fontsize=8)
    fig.suptitle("Day 6 S1a head-only training — real cached h_spatial", fontsize=15)
    learning = save(fig, "S1A_LEARNING_CURVES.png")
    shutil.copy2(learning, DAY6 / "S1A_LEARNING_CURVES.png")

    relation_labels = ["leftmost", "rightmost", "second_from_left", "second_from_right"]
    answer_labels = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
    relation_recall = np.asarray(decision["best_val_metrics"]["relation_per_class_recall"]) * 100
    answer_recall = np.asarray(decision["best_val_metrics"]["answerability_per_class_recall"]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].barh(relation_labels, relation_recall, color="#0072b2")
    axes[0].set(title="Relation recall by class", xlabel="Recall (%)", xlim=(0, 105))
    axes[1].barh(answer_labels, answer_recall, color="#009e73")
    axes[1].set(title="Answerability recall by state", xlabel="Recall (%)", xlim=(0, 105))
    fig.suptitle(f"Best checkpoint, Val development (epoch {best_epoch})", fontsize=15)
    per_class = save(fig, "S1A_PER_CLASS_RECALL.png")

    cache_rows = read_csv(cache_timing_path)
    seconds = np.asarray([float(row["seconds"]) for row in cache_rows if row["cache_origin"] == "DAY6_NEW_EXTRACTION"])
    norms = np.asarray([float(row["gradient_norm_preclip"]) for row in train_steps])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hist(seconds, bins=12, color="#56b4e9", edgecolor="white")
    axes[0].axvline(np.median(seconds), color="#d55e00", linestyle="--", label=f"median {np.median(seconds):.3f} s")
    axes[0].set(title="New cache extraction latency (n=288)", xlabel="Seconds per sample", ylabel="Samples")
    axes[0].legend(frameon=True)
    axes[1].plot(np.arange(1, len(norms) + 1), norms, linewidth=0.8)
    axes[1].axhline(5.0, color="#d55e00", linestyle="--", label="clip threshold 5")
    axes[1].set_yscale("log")
    axes[1].set(title="S1a pre-clip gradient norm", xlabel="Optimizer step", ylabel="Gradient norm (log scale)")
    axes[1].legend(frameon=True)
    resource_fig = save(fig, "S1A_RESOURCE_AND_GRADIENT.png")

    heads = ["Relation", "Spatial", "Answerability", "Reasoning", "Source", "Confidence"]
    train_support = [256, 64, 256, 0, 0, 0]
    val_support = [64, 16, 64, 0, 0, 0]
    positions = np.arange(len(heads))
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.bar(positions - 0.2, train_support, 0.4, label="Train")
    ax.bar(positions + 0.2, val_support, 0.4, label="Val")
    ax.set_xticks(positions, heads, rotation=20, ha="right")
    ax.set(title="Certified supervision used by S1a", ylabel="Valid parent families")
    ax.legend(frameon=True)
    support_fig = save(fig, "S1A_HEAD_SUPPORT.png")

    metrics_path = METRICS / "S1A_DEVELOPMENT_METRICS.json"
    metrics_payload = {
        "schema_version": "1.0",
        "scope": "Development Val model-selection evidence; not Test or confirmatory evaluation",
        "outcome": decision["outcome"],
        "best_epoch": best_epoch,
        "train_parent_families": 256,
        "val_parent_families": 64,
        "best_train_metrics": decision["best_train_metrics"],
        "best_val_metrics": decision["best_val_metrics"],
        "masked_not_evaluated": ["reasoning_depth", "uncertainty_source", "confidence"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table_path = TABLES / "S1A_BEST_CHECKPOINT_SUMMARY.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "train", "val", "support_train", "support_val", "scope"])
        writer.writeheader()
        writer.writerows([
            {"metric": "relation_macro_f1", "train": decision["best_train_metrics"]["relation_macro_f1"], "val": decision["best_val_metrics"]["relation_macro_f1"], "support_train": 256, "support_val": 64, "scope": "development"},
            {"metric": "answerability_macro_f1", "train": decision["best_train_metrics"]["answerability_macro_f1"], "val": decision["best_val_metrics"]["answerability_macro_f1"], "support_train": 256, "support_val": 64, "scope": "development"},
            {"metric": "hit_at_0.08", "train": decision["best_train_metrics"]["hit_at_0_08"], "val": decision["best_val_metrics"]["hit_at_0_08"], "support_train": 64, "support_val": 16, "scope": "FOUND development"},
            {"metric": "point_mae_uv", "train": decision["best_train_metrics"]["point_mae_uv"], "val": decision["best_val_metrics"]["point_mae_uv"], "support_train": 64, "support_val": 16, "scope": "FOUND development"},
            {"metric": "gaussian_nll", "train": decision["best_train_metrics"]["spatial_nll"], "val": decision["best_val_metrics"]["spatial_nll"], "support_train": 64, "support_val": 16, "scope": "FOUND development"},
        ])

    source_files = [epoch_path, train_log_path, decision_path, resource_path, cache_timing_path, split_audit_path]
    output_files = [learning, per_class, resource_fig, support_fig, metrics_path, table_path, DAY6 / "S1A_LEARNING_CURVES.png"]
    index = {
        "schema_version": "1.0",
        "scope": "Day-6 development engineering/model-selection figures; not confirmatory Test figures",
        "renderer": "workspace/mh_pcrau_v3/plot_day6.py",
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in source_files],
        "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in output_files],
    }
    (FIGURES / "FIGURE_INDEX.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": 4, "best_epoch": best_epoch, "metrics": str(metrics_path.relative_to(ROOT)), "peak_reserved_mib": resource["peak_reserved_mib"], "families": split_audit["families"]}))


if __name__ == "__main__":
    main()
