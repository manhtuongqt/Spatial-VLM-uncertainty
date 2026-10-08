#!/usr/bin/env python3
"""Render one frozen best-V2 anchor/edge case for Chapter 5.

The Test-IID prediction archive stores edge scores but not anchor logits.
This script extracts the missing anchor map with the locked checkpoint and
cached features, then checks the active edge and MAP against the archive.
No training, calibration, or metric recomputation is performed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image
import torch

from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset, collate_samples, move_model_batch
from pcrau.engine import autocast_context
from pcrau.model import PCRAUTargetV2
from pcrau.postprocess import summarize_heatmap
from pcrau.utils import sha256_file


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ID = "v211iid_family_000121__clean"
RUN = ROOT / "new/outputs/pcrau_target_v2_full_seed_24082026"
CHECKPOINT = RUN / "checkpoints/best/model.safetensors"
PREDICTIONS = ROOT / "new/test_iid/evaluation/best_v2/predictions.jsonl"
OUT = ROOT / "new/hinhanh/24_anchor_edge/fig24_anchor_edge_test_iid.png"
LATEX = ROOT / "latex/hinhanh/fig24_anchor_edge_test_iid.png"


def archived_prediction() -> dict:
    with PREDICTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] == SAMPLE_ID:
                return row
    raise ValueError(f"Sample absent from prediction archive: {SAMPLE_ID}")


def image_axes(ax, rgb: np.ndarray, title: str) -> None:
    ax.imshow(rgb)
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_color("#C8D0D7")
        spine.set_linewidth(0.7)


def main() -> None:
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    best = json.loads((RUN / "best.json").read_text(encoding="utf-8"))
    assert best["epoch"] == 13 and best["global_step"] == 1120
    assert sha256_file(CHECKPOINT) == best["model_sha256"]
    dataset = ArchivedPCRAUDataset(config, "test_iid", profile="test_iid", verify_feature_hash=True)
    index = next(i for i, entry in enumerate(dataset.entries) if entry["sample_id"] == SAMPLE_ID)
    entry = dataset.entries[index]
    assert len(entry["supervision"]["anchor_masks"]) == 1
    batch = collate_samples([dataset[index]])
    assert batch["anchor_mask"][0].tolist() == [True, False, False]
    assert batch["edge_mask"][0].tolist() == [True, False, False]
    assert batch["edge_target"][0].tolist() == [1.0, 0.0, 0.0]

    model = PCRAUTargetV2(config)
    load_model_checkpoint(model, CHECKPOINT, best["config_sha256"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    with torch.no_grad(), autocast_context(device, config["optimization"]):
        output = model(move_model_batch(batch, device))
    edge_score = float(output["relation_edge_logits"][0, 0].float().sigmoid().cpu())
    anchor_map = output["anchor_logits"][0, 0].float().sigmoid().cpu().numpy()
    target_summary = summarize_heatmap(output["target_logits"][0].float().cpu())
    stored = archived_prediction()
    assert abs(edge_score - stored["relation_edge_probabilities"][0]) < 1e-3
    assert target_summary["map_pixel_xy"] == stored["spatial"]["map_pixel_xy"]
    assert anchor_map.shape == (24, 32)
    ay, ax = np.unravel_index(np.argmax(anchor_map), anchor_map.shape)
    anchor_point = (int(ax * 20 + 10), int(ay * 20 + 10))

    rgb_path = ROOT / entry["feature_input"]["rgb_path"]
    anchor_path = ROOT / "new/test_iid/dataset" / entry["supervision"]["anchor_masks"][0]["path"]
    target_path = ROOT / "new/test_iid/dataset" / entry["supervision"]["target_mask_path"]
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    anchor_gt = np.asarray(Image.open(anchor_path).convert("L")) > 0
    target_gt = np.asarray(Image.open(target_path).convert("L")) > 0
    assert rgb.shape == (480, 640, 3) and anchor_gt.shape == target_gt.shape == (480, 640)
    assert anchor_gt[anchor_point[1], anchor_point[0]]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(14.2, 5.25), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.92], left=0.045, right=0.985,
                           top=0.84, bottom=0.20, wspace=0.15)
    rgb_ax, anchor_ax, graph_ax = [fig.add_subplot(grid[0, n]) for n in range(3)]
    fig.suptitle("Anchor và relation edge trên một mẫu Test-IID", fontsize=16,
                 fontweight="bold", y=0.965)
    fig.text(0.045, 0.89,
             "Truy vấn: tìm quả táo ở phía sau quả chanh  |  P-CRA-U  |  000121__clean",
             fontsize=10.5, color="#425466")

    image_axes(rgb_ax, rgb, "(a) Ảnh RGB và vật đối chiếu")
    rgb_ax.contour(anchor_gt.astype(float), levels=[0.5], colors=["#00B894"], linewidths=2.4)
    rgb_ax.contour(target_gt.astype(float), levels=[0.5], colors=["#E67E22"], linewidths=2.4)
    rgb_ax.scatter([anchor_point[0]], [anchor_point[1]], marker="x", s=100,
                   color="white", linewidths=2.3, zorder=10)
    rgb_ax.text(0.02, -0.10, "Xanh: chanh (anchor)   Cam: táo (target)",
                transform=rgb_ax.transAxes, fontsize=9.1, color="#425466")

    image_axes(anchor_ax, rgb, "(b) Kích hoạt anchor slot 1")
    resized = np.asarray(Image.fromarray(anchor_map.astype(np.float32), mode="F").resize(
        (640, 480), Image.Resampling.BILINEAR))
    visible = np.ma.masked_where(resized < 0.05, resized)
    layer = anchor_ax.imshow(visible, cmap="magma", norm=Normalize(0, 1), alpha=0.74)
    anchor_ax.contour(anchor_gt.astype(float), levels=[0.5], colors=["#00E5B4"], linewidths=2.0)
    anchor_ax.scatter([anchor_point[0]], [anchor_point[1]], marker="x", s=110,
                      color="white", linewidths=2.5, zorder=10)
    bar = fig.colorbar(layer, ax=anchor_ax, shrink=0.75, pad=0.015, fraction=0.045)
    bar.set_label("sigmoid(anchor logit)", fontsize=8.5)
    anchor_ax.text(0.02, -0.10, "×: cực đại dự đoán; viền xanh: mask chuẩn",
                   transform=anchor_ax.transAxes, fontsize=9.1, color="#425466")

    graph_ax.set_axis_off()
    graph_ax.set_title("(c) Cạnh quan hệ đang hoạt động", loc="left", fontweight="bold", pad=8)
    graph_ax.text(0.12, 0.76, "TÁO", ha="center", va="center", fontsize=12, fontweight="bold",
                  transform=graph_ax.transAxes, color="#9A4C0E",
                  bbox={"boxstyle": "round,pad=0.55", "fc": "#FFF1DE", "ec": "#E67E22"})
    graph_ax.text(0.86, 0.76, "CHANH", ha="center", va="center", fontsize=12, fontweight="bold",
                  transform=graph_ax.transAxes, color="#057A61",
                  bbox={"boxstyle": "round,pad=0.55", "fc": "#E0FAF3", "ec": "#00A884"})
    graph_ax.add_patch(FancyArrowPatch((0.25, 0.76), (0.72, 0.76),
                       transform=graph_ax.transAxes, arrowstyle="-|>", mutation_scale=18,
                       linewidth=2, color="#4D6477"))
    graph_ax.text(0.49, 0.86, "behind", ha="center", fontsize=10.5,
                  fontweight="bold", transform=graph_ax.transAxes, color="#34495E")
    graph_ax.text(0.05, 0.52, "Slot 1: p(edge) = %.4f" % edge_score,
                  fontsize=11.5, fontweight="bold", transform=graph_ax.transAxes)
    graph_ax.barh([0.42], [0.90 * edge_score], left=0.05, height=0.07,
                  color="#00A884", transform=graph_ax.transAxes)
    graph_ax.plot([0.50, 0.50], [0.36, 0.48], color="#C04D38", linewidth=1.5,
                  transform=graph_ax.transAxes)
    graph_ax.text(0.50, 0.30, "ngưỡng 0,5", ha="center", fontsize=9,
                  transform=graph_ax.transAxes, color="#8D4639")
    graph_ax.text(0.05, 0.18, "Dự đoán: cạnh dương  |  Nhãn proxy: dương",
                  fontsize=9.3, transform=graph_ax.transAxes, color="#425466")
    graph_ax.text(0.05, 0.09, "Slot 2–3: không hoạt động, không chấm",
                  fontsize=9.3, transform=graph_ax.transAxes, color="#6B7785")
    fig.text(0.045, 0.05,
             "Mask chuẩn chỉ dùng hậu kiểm; hình một mẫu không phải metric anchor localization hay graph exact match.",
             fontsize=9.1, color="#536273")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    LATEX.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=210)
    plt.close(fig)
    shutil.copyfile(OUT, LATEX)
    print(f"Created {OUT} and {LATEX}; sample={SAMPLE_ID}, edge={edge_score:.6f}, anchor_peak={anchor_point}")


if __name__ == "__main__":
    main()
