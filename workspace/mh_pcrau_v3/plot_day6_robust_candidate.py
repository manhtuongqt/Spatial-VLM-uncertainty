#!/usr/bin/env python3
"""Render the robust-candidate engineering figures from locked artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ketqua1/07_huan_luyen/ngay_06/amendment_05_robust_candidate_r2"
FIGURE = ROOT / "ketqua1/09_danh_gia/figures/ngay_06/amendment_05/F06E_ROBUST_CANDIDATE_ENGINEERING.png"
TABLE = ROOT / "ketqua1/09_danh_gia/tables/ngay_06/amendment_05/T06G_ROBUST_CANDIDATE_VARIANTS.csv"


def main() -> None:
    if FIGURE.exists() or TABLE.exists(): raise FileExistsError("Candidate plots are append-only")
    FIGURE.parent.mkdir(parents=True); TABLE.parent.mkdir(parents=True)
    with (SOURCE / "ROBUST_CANDIDATE_EPOCH_METRICS.csv").open(encoding="utf-8") as handle:
        epoch_rows = list(csv.DictReader(handle))
    manifest = json.loads((SOURCE / "ROBUST_CANDIDATE_CHECKPOINT_MANIFEST.json").read_text())
    variant_rows = []
    for name, metrics in manifest["per_train_variant"].items():
        variant_rows.append({"variant": name, "role": "TRAIN_FIT_ONLY", "support": metrics["valid_relation"],
                             "found_support": metrics["valid_spatial"], "relation_macro_f1": metrics["relation_macro_f1"],
                             "answerability_macro_f1": metrics["answerability_macro_f1"],
                             "point_l2_mean": metrics["point_l2_mean"], "hit_at_0_05": metrics["hit_at_0_05"],
                             "hit_at_0_08": metrics["hit_at_0_08"]})
    with TABLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(variant_rows[0])); writer.writeheader(); writer.writerows(variant_rows)
    epochs = [int(row["epoch"]) for row in epoch_rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].plot(epochs, [float(row["train_total_loss"]) for row in epoch_rows], label="Augmented train")
    axes[0].plot(epochs, [float(row["old_val_total_loss_monitor_only"]) for row in epoch_rows], label="Old Val monitor only")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Weighted loss"); axes[0].set_title("Fixed 432-update training")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.25)
    axes[1].plot(epochs, [float(row["train_answerability_macro_f1"]) for row in epoch_rows], label="Train")
    axes[1].plot(epochs, [float(row["old_val_answerability_macro_f1_monitor_only"]) for row in epoch_rows], label="Old Val monitor")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Answerability macro-F1"); axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Monitor curve; no checkpoint selection"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.25)
    names = ["Original view\ncanonical", "Second view\nshort", "Second view\ncanonical"]
    hits = [row["hit_at_0_05"] for row in variant_rows]
    bars = axes[2].bar(np.arange(3), hits, color=["#999999", "#D55E00", "#0072B2"])
    axes[2].set_xticks(np.arange(3), names); axes[2].set_ylim(0, 1.05); axes[2].set_ylabel("Hit@0.05")
    axes[2].set_title("Training-fit only, not evaluation")
    for bar, value in zip(bars, hits): axes[2].text(bar.get_x()+bar.get_width()/2, value+.025, f"{value:.3f}", ha="center")
    axes[2].grid(axis="y", alpha=.25)
    figure.suptitle("MH-PCRA-U-v3 robust candidate r2 — engineering evidence", fontsize=14)
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print(json.dumps({"status": "PASS", "figure": str(FIGURE.relative_to(ROOT)), "table": str(TABLE.relative_to(ROOT))}))


if __name__ == "__main__": main()
