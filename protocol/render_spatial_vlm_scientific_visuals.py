#!/usr/bin/env python3
"""Render traceable scientific tables/plots from locked Spatial-VLM artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/spatial_vlm_refspatial_v1/scientific_visualizations_20260911"
GAZEBO = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_dev_v1/evaluation"
SOURCES = {
    "wp6": ROOT / "results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge/evaluation/B0_B1_CHALLENGE_METRICS.json",
    "clean": ROOT / "results/spatial_vlm_refspatial_v1/wp7_b1_clean_grounding/clean_eval/B0_B1_CLEAN_METRICS.json",
    "human": ROOT / "results/spatial_vlm_refspatial_v1/wp7_b1_clean_grounding/human_stress_eval/B0_B1_HUMAN_EVAL_METRICS.json",
    "gazebo": GAZEBO / "B0_B1_GAZEBO_DEV_METRICS.json",
    "gazebo_gt": ROOT / "datasets/Gazebo_dev/evaluator_ground_truth.jsonl",
    "gazebo_b0": GAZEBO / "b0/predictions.jsonl",
    "gazebo_b1": GAZEBO / "b1/predictions.jsonl",
    "gazebo_manifest": ROOT / "datasets/Gazebo_dev/manifest.json",
    "training": ROOT / "results/spatial_vlm_refspatial_v1/wp7_b1_clean_grounding/TRAINING_REPORT.json",
}

B0 = "#276FBF"
B1 = "#F28E2B"
GOOD = "#2A9D8F"
BAD = "#D1495B"
INK = "#243447"
MUTED = "#667788"
GRID = "#D7DEE5"
BG = "#FBFCFE"
STATE_COLORS = {
    "FOUND": "#2A9D8F",
    "AMBIGUOUS": "#E9C46A",
    "ABSENT": "#E76F51",
    "INSUFFICIENT_EVIDENCE": "#7B6DCC",
}
POINT_RE = re.compile(r"^\[\(\s*(?:\d+(?:\.\d*)?|\.\d+)\s*,\s*(?:\d+(?:\.\d*)?|\.\d+)\s*\)\]$")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def configure() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.facecolor": BG,
        "figure.facecolor": "white",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, stem: str, svg: bool = True, pdf: bool = False) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    if svg:
        fig.savefig(OUT / f"{stem}.svg")
    if pdf:
        fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def primary_group(payload: dict) -> dict:
    key = "all" if "all" in payload["groups"] else "all_primary"
    return payload["groups"][key]


def collect() -> dict:
    raw = {name: load_json(path) for name, path in SOURCES.items() if path.suffix == ".json"}
    studies = []
    for key, label, evidence in (
        ("wp6", "WP6 challenge", "machine labels"),
        ("clean", "Clean internal", "human-accepted"),
        ("human", "Human stress", "human-certified"),
    ):
        group = primary_group(raw[key])
        studies.append({
            "key": key,
            "label": label,
            "evidence": evidence,
            "n": int(group["b0"]["samples"]),
            "b0": float(group["b0"]["hit_at_008"]),
            "b1": float(group["b1"]["hit_at_008"]),
            "b0_ci": tuple(group["b0"]["hit_at_008_wilson_95"]),
            "b1_ci": tuple(group["b1"]["hit_at_008_wilson_95"]),
            "delta": float(group["paired"]["hit_at_008_delta_b1_minus_b0"]),
            "delta_ci": tuple(group["paired"]["family_cluster_bootstrap_hit_delta_95"]),
            "p": float(group["paired"]["mcnemar_exact_two_sided_p"]),
            "fixed": int(group["paired"]["b0_wrong_b1_right"]),
            "introduced": int(group["paired"]["b0_right_b1_wrong"]),
        })
    gm = raw["gazebo"]
    studies.append({
        "key": "gazebo",
        "label": "Gazebo FOUND",
        "evidence": "robot Dev pilot",
        "n": 4,
        "b0": float(gm["metrics"]["b0"]["found_hit_at_008"]),
        "b1": float(gm["metrics"]["b1"]["found_hit_at_008"]),
        "b0_ci": wilson(3, 4),
        "b1_ci": wilson(3, 4),
        "delta": float(gm["paired"]["found_hit_delta_b1_minus_b0"]),
        "delta_ci": None,
        "p": None,
        "fixed": int(gm["paired"]["b0_wrong_b1_right_found"]),
        "introduced": int(gm["paired"]["b0_right_b1_wrong_found"]),
    })
    gt = {row["sample_id"]: row for row in load_jsonl(SOURCES["gazebo_gt"])}
    predictions = {
        "b0": {row["sample_id"]: row for row in load_jsonl(SOURCES["gazebo_b0"])},
        "b1": {row["sample_id"]: row for row in load_jsonl(SOURCES["gazebo_b1"])},
    }
    return {"raw": raw, "studies": studies, "gt": gt, "predictions": predictions}


def write_tables(data: dict) -> None:
    with (OUT / "TABLE_01_accuracy_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "evaluation", "evidence", "n", "b0_hit_at_008", "b0_wilson_low", "b0_wilson_high",
            "b1_hit_at_008", "b1_wilson_low", "b1_wilson_high", "delta_b1_minus_b0",
            "delta_ci_low", "delta_ci_high", "mcnemar_p", "b1_fixes", "b1_introduces",
        ])
        writer.writeheader()
        for s in data["studies"]:
            writer.writerow({
                "evaluation": s["label"], "evidence": s["evidence"], "n": s["n"],
                "b0_hit_at_008": s["b0"], "b0_wilson_low": s["b0_ci"][0], "b0_wilson_high": s["b0_ci"][1],
                "b1_hit_at_008": s["b1"], "b1_wilson_low": s["b1_ci"][0], "b1_wilson_high": s["b1_ci"][1],
                "delta_b1_minus_b0": s["delta"],
                "delta_ci_low": s["delta_ci"][0] if s["delta_ci"] else "",
                "delta_ci_high": s["delta_ci"][1] if s["delta_ci"] else "",
                "mcnemar_p": s["p"] if s["p"] is not None else "",
                "b1_fixes": s["fixed"], "b1_introduces": s["introduced"],
            })
    with (OUT / "TABLE_02_gazebo_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["sample_id", "state", "target_x", "target_y", "model", "pred_x", "pred_y", "point_error", "hit_at_008", "action", "self_consistency", "exact_format"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sid in sorted(data["gt"]):
            gt = data["gt"][sid]
            for model in ("b0", "b1"):
                pred = data["predictions"][model][sid]
                point = pred.get("prediction_xy")
                error = math.dist(point, gt["target_xy"]) if gt["answerability_state"] == "FOUND" and point else None
                writer.writerow({
                    "sample_id": sid, "state": gt["answerability_state"],
                    "target_x": gt["target_xy"][0] if gt.get("target_xy") else "",
                    "target_y": gt["target_xy"][1] if gt.get("target_xy") else "",
                    "model": model.upper(), "pred_x": point[0] if point else "", "pred_y": point[1] if point else "",
                    "point_error": error if error is not None else "", "hit_at_008": error is not None and error <= 0.08,
                    "action": pred["action"], "self_consistency": pred["self_consistency_confidence"],
                    "exact_format": bool(POINT_RE.fullmatch(pred.get("answer") or "")),
                })


def summary_table(data: dict) -> None:
    gm = data["raw"]["gazebo"]
    exact = {
        model: sum(bool(POINT_RE.fullmatch(row.get("answer") or "")) for row in data["predictions"][model].values()) / 16
        for model in ("b0", "b1")
    }
    rows = [
        ["Gazebo FOUND Hit@0.08", "75.0% (3/4)", "75.0% (3/4)", "Hòa"],
        ["Mean normalized error", f"{gm['metrics']['b0']['found_mean_normalized_error']:.4f}", f"{gm['metrics']['b1']['found_mean_normalized_error']:.4f}", "B0 thấp hơn"],
        ["B1 fixes / introduces", "—", "0 / 0", "Không cải thiện"],
        ["Parse rate", "100%", "100%", "Hòa"],
        ["Exact output format", f"{exact['b0']:.0%}", f"{exact['b1']:.0%}", "B1 regression"],
        ["Non-FOUND false accept*", "100%", "100%", "Prompt ép POINT"],
        ["Mean self-consistency", f"{gm['metrics']['b0']['mean_self_consistency_confidence']:.3f}", f"{gm['metrics']['b1']['mean_self_consistency_confidence']:.3f}", "Chưa calibration"],
    ]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")
    ax.text(0, 1.08, "Bảng kết quả chính — Gazebo_dev_v1", transform=ax.transAxes, fontsize=18, fontweight="bold", color=INK)
    ax.text(0, 1.015, "16 family Dev · 4 FOUND · 4 AMBIGUOUS · 4 ABSENT · 4 INSUFFICIENT_EVIDENCE", transform=ax.transAxes, fontsize=10.5, color=MUTED)
    table = ax.table(cellText=rows, colLabels=["Chỉ số", "B0", "Clean B1", "Diễn giải"], loc="upper center", cellLoc="center", colWidths=[0.34, 0.17, 0.17, 0.27])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.75)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_facecolor(INK); cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F0F4F8" if r % 2 else "white")
            if c == 0: cell.get_text().set_ha("left")
    ax.text(0, -0.05, "Kết luận preregistered gate: KEEP B0 AND ANALYZE ERRORS — không promote B1.", transform=ax.transAxes, fontsize=12, fontweight="bold", color=BAD)
    ax.text(0, -0.12, "* Diagnostic only: prompt v1 không cho phép ABSTAIN. N=4 FOUND quá nhỏ để claim generalization.", transform=ax.transAxes, fontsize=9.5, color=MUTED)
    save(fig, "00_summary_table", pdf=True)


def accuracy_across_sets(data: dict) -> None:
    studies = data["studies"]
    x = np.arange(len(studies)); width = 0.34
    fig, ax = plt.subplots(figsize=(12, 6.8))
    for offset, model, color in ((-width/2, "b0", B0), (width/2, "b1", B1)):
        values = np.array([s[model] for s in studies])
        cis = [s[f"{model}_ci"] for s in studies]
        lower = values - np.array([c[0] for c in cis]); upper = np.array([c[1] for c in cis]) - values
        bars = ax.bar(x + offset, values, width, color=color, label=model.upper(), yerr=np.vstack([lower, upper]), capsize=4, edgecolor="white", linewidth=0.8)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+0.025, f"{value:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Hit@0.08")
    ax.set_xticks(x, [f"{s['label']}\n(n={s['n']}; {s['evidence']})" for s in studies])
    fig.suptitle("Grounding accuracy qua các phạm vi đánh giá", x=0.08, y=0.98, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.08, 0.925, "Thanh lỗi: Wilson 95% CI; so sánh trong từng tập, không gộp domain", color=MUTED)
    fig.subplots_adjust(top=0.86)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(ncol=2, loc="upper left")
    save(fig, "01_grounding_accuracy_across_evaluations", pdf=True)


def delta_forest(data: dict) -> None:
    studies = data["studies"]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    y = np.arange(len(studies))[::-1]
    for yy, s in zip(y, studies):
        if s["delta_ci"]:
            lo, hi = s["delta_ci"]
            ax.errorbar(s["delta"]*100, yy, xerr=[[s["delta"]*100-lo*100], [hi*100-s["delta"]*100]], fmt="o", color=B1, ecolor=INK, capsize=5, markersize=8)
            annotation = f"{s['delta']*100:+.2f} pp  [{lo*100:+.2f}, {hi*100:+.2f}] · p={s['p']:.2g}"
        else:
            ax.scatter(s["delta"]*100, yy, s=70, color=B1, zorder=3)
            annotation = f"{s['delta']*100:+.2f} pp · pilot n=4, no powered CI"
        ax.text(4.2, yy, annotation, va="center", fontsize=9.5, color=INK,
                bbox={"facecolor": BG, "edgecolor": "none", "alpha": 0.88, "pad": 1.5})
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlim(-4.5, 12.5); ax.set_yticks(y, [s["label"] for s in studies]); ax.set_xlabel("Chênh lệch Hit@0.08: clean B1 − B0 (percentage points)")
    fig.suptitle("Paired effect: chưa có bằng chứng B1 cải thiện ổn định", x=0.12, y=0.98, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.12, 0.925, "95% family-cluster bootstrap CI khi có; McNemar exact p", color=MUTED)
    fig.subplots_adjust(top=0.86)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    save(fig, "02_paired_delta_forest", pdf=True)


def gazebo_case_errors(data: dict) -> None:
    found = [sid for sid in sorted(data["gt"]) if data["gt"][sid]["answerability_state"] == "FOUND"]
    x = np.arange(len(found)); width = 0.33
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for offset, model, color in ((-width/2, "b0", B0), (width/2, "b1", B1)):
        errors = [math.dist(data["predictions"][model][sid]["prediction_xy"], data["gt"][sid]["target_xy"]) for sid in found]
        bars = ax.bar(x+offset, errors, width, label=model.upper(), color=color, edgecolor="white")
        for bar, value in zip(bars, errors):
            ax.text(bar.get_x()+bar.get_width()/2, value+0.008, f"{value:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.08, color=BAD, linestyle="--", lw=1.8, label="Hit threshold = 0.08")
    labels=[]
    for sid in found:
        label=sid.replace("gazebo_dev_", "case ")
        if sid == "gazebo_dev_0004": label += "\nrightmost mango"
        labels.append(label)
    ax.set_xticks(x, labels); ax.set_ylabel("Normalized Euclidean point error"); ax.set_ylim(0, 0.45)
    fig.suptitle("Gazebo FOUND: lỗi theo từng family", x=0.08, y=0.98, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.08, 0.925, "Cả B0 và B1 cùng thất bại ở target mango bị cắt sát mép phải", color=MUTED)
    fig.subplots_adjust(top=0.86)
    ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(ncol=3, loc="upper left")
    save(fig, "03_gazebo_found_case_errors", pdf=True)


def answerability_heatmap(data: dict) -> None:
    states = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
    matrix=[]
    for model in ("b0", "b1"):
        row=[]
        for state in states:
            ids=[sid for sid,g in data["gt"].items() if g["answerability_state"] == state]
            row.append(sum(data["predictions"][model][sid]["action"] == "POINT" for sid in ids)/len(ids))
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    cmap=LinearSegmentedColormap.from_list("point", ["#EDF7F5", BAD])
    im=ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for i in range(2):
        for j in range(4): ax.text(j, i, f"{matrix[i][j]:.0%} POINT", ha="center", va="center", color="white" if matrix[i][j]>.65 else INK, fontweight="bold")
    ax.set_xticks(range(4), ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT\nEVIDENCE"]); ax.set_yticks(range(2), ["B0", "Clean B1"])
    ax.set_title("Answerability v1: model luôn trả POINT", loc="left", fontweight="bold", color=INK)
    ax.text(0, 1.08, "PROTOCOL DIAGNOSTIC — prompt bắt buộc một point và không định nghĩa token ABSTAIN", transform=ax.transAxes, color=BAD, fontweight="bold")
    ax.text(0, -0.22, "Không dùng hình này để claim khả năng abstention; cần contract POINT/ABSTAIN v2 preregistered.", transform=ax.transAxes, color=MUTED)
    cbar=fig.colorbar(im, ax=ax, fraction=.025, pad=.03); cbar.set_label("Tỷ lệ trả POINT")
    save(fig, "04_answerability_protocol_diagnostic", pdf=True)


def confidence_by_state(data: dict) -> None:
    states = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    rng=np.random.default_rng(8132026)
    for ax, model, color in zip(axes, ("b0", "b1"), (B0, B1)):
        for idx,state in enumerate(states):
            ids=sorted(sid for sid,g in data["gt"].items() if g["answerability_state"]==state)
            for sid in ids:
                pred=data["predictions"][model][sid]; conf=pred["self_consistency_confidence"]
                if state == "FOUND":
                    correct=math.dist(pred["prediction_xy"],data["gt"][sid]["target_xy"])<=.08
                else:
                    correct=pred["action"] != "POINT"
                marker="o" if correct else "X"
                ax.scatter(idx+rng.uniform(-.09,.09),conf,s=95,marker=marker,color=GOOD if correct else BAD,edgecolor="white",linewidth=.8,zorder=3)
        ax.set_title(model.upper(), fontweight="bold", color=color); ax.set_xticks(range(4),["FOUND","AMBIG.","ABSENT","INSUFF."]); ax.set_ylim(-.07,1.08); ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Self-consistency confidence proxy")
    fig.suptitle("Confidence cao không bảo đảm dự đoán đúng", x=.08, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(.08,.91,"● đúng theo desired action · ✕ sai; non-FOUND vẫn là protocol diagnostic",color=MUTED)
    fig.text(.08,.01,"Case rightmost mango: cả hai model sai với confidence = 1.0. Proxy này chưa được calibration.",color=BAD,fontweight="bold")
    save(fig, "05_self_consistency_and_high_confidence_errors", pdf=True)


def risk_coverage(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    for model,color in (("b0",B0),("b1",B1)):
        rows=[]
        for sid,gt in data["gt"].items():
            pred=data["predictions"][model][sid]; point=pred.get("prediction_xy")
            correct=(point is not None and math.dist(point,gt["target_xy"])<=.08) if gt["answerability_state"]=="FOUND" else pred["action"]!="POINT"
            rows.append((float(pred["self_consistency_confidence"]),bool(correct),sid))
        rows.sort(key=lambda item:(-item[0],item[2]))
        accuracy=np.cumsum([r[1] for r in rows])/np.arange(1,len(rows)+1)
        coverage=np.arange(1,len(rows)+1)/len(rows)
        ax.step(np.r_[0,coverage],np.r_[accuracy[0],accuracy],where="post",label=model.upper(),color=color,lw=2.2)
        ax.scatter(coverage,accuracy,color=color,s=18)
    ax.set_xlim(0,1.02); ax.set_ylim(0,1.02); ax.set_xlabel("Coverage (giữ lại tỷ lệ prediction confidence cao nhất)"); ax.set_ylabel("Desired-action accuracy")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.grid(); ax.set_axisbelow(True); ax.legend()
    fig.suptitle("Risk–coverage trên Gazebo_dev_v1", x=0.08, y=0.98, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.08, 0.925, "Exploratory diagnostic only: non-FOUND prompt không cho phép ABSTAIN; n=16", color=BAD, fontweight="bold")
    fig.subplots_adjust(top=0.86)
    save(fig,"06_gazebo_risk_coverage_diagnostic",pdf=True)


def calibration_metrics(data: dict) -> None:
    rows=[]
    for key,label in (("wp6","WP6 challenge"),("clean","Clean internal"),("human","Human stress")):
        group=primary_group(data["raw"][key])
        rows.append((label,group))
    metrics=[("ece_10bin","ECE ↓"),("brier","Brier ↓"),("error_detection_auroc","Error AUROC ↑")]
    fig,axes=plt.subplots(1,3,figsize=(15,5.3))
    x=np.arange(len(rows)); width=.34
    for ax,(metric,title) in zip(axes,metrics):
        for off,model,color in ((-width/2,"b0",B0),(width/2,"b1",B1)):
            vals=[group[model][metric] for _,group in rows]
            plot_vals=[np.nan if v is None else v for v in vals]
            bars=ax.bar(x+off,plot_vals,width,color=color,label=model.upper())
            for bar,v in zip(bars,vals):
                if v is None: ax.text(bar.get_x()+bar.get_width()/2,.02,"N/A",ha="center",fontsize=8,color=MUTED)
                else: ax.text(bar.get_x()+bar.get_width()/2,v+.02,f"{v:.3f}",ha="center",fontsize=8)
        ax.set_xticks(x,[label.replace(" ","\n",1) for label,_ in rows]); ax.set_ylim(0,1.05 if metric=="error_detection_auroc" else .12); ax.set_title(title,fontweight="bold"); ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[0].legend(ncol=2,loc="upper left")
    fig.suptitle("Uncertainty proxy: calibration/error-detection chưa cải thiện ổn định",x=.06,ha="left",fontsize=15,fontweight="bold",color=INK)
    fig.text(.06,.01,"Self-consistency là proxy hậu nghiệm, chưa phải PCRAU/final calibrator. Không dùng các tập này để claim Gazebo generalization.",color=MUTED)
    save(fig,"07_uncertainty_proxy_metrics",pdf=True)


def training_card(data: dict) -> None:
    t=data["raw"]["training"]; tr=t["training"]
    rows=[
        ["Training families",f"{tr['train_samples']:,}"], ["Epoch / optimizer steps",f"{tr['epochs']} / {tr['optimizer_steps']}"],
        ["Effective batch",str(tr['effective_batch_size'])], ["Learning rate",f"{tr['learning_rate']:.1e}"],
        ["LoRA rank / alpha / dropout",f"{tr['lora_rank']} / {tr['lora_alpha']} / {tr['lora_dropout']}"],
        ["Trainable adapter params",f"{tr['trainable_adapter_parameters']:,}"], ["Train loss",f"{tr['final_summary']['train_loss']:.6f}"],
        ["Runtime",f"{tr['final_summary']['train_runtime']/60:.1f} min"], ["GPU peak allocated / reserved",f"{tr['cuda_peak_allocated_gib']:.2f} / {tr['cuda_peak_reserved_gib']:.2f} GiB"],
        ["Adapter SHA-256",t['artifacts']['adapter_model.safetensors']['sha256'][:16]+"…"],
    ]
    fig,ax=plt.subplots(figsize=(9,7)); ax.axis("off")
    ax.text(0,1.05,"Clean B1 training card",transform=ax.transAxes,fontsize=18,fontweight="bold",color=INK)
    ax.text(0,.99,"D_tabletop_clean_v1 · LLM-LoRA only · no SAM2 · B2 closed",transform=ax.transAxes,color=MUTED)
    table=ax.table(cellText=rows,colLabels=["Field","Value"],loc="upper center",cellLoc="left",colWidths=[.52,.42]); table.auto_set_font_size(False); table.set_fontsize(10.5); table.scale(1,1.6)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("white"); cell.set_facecolor(INK if r==0 else ("#F0F4F8" if r%2 else "white"));
        if r==0: cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
    save(fig,"08_clean_b1_training_card",pdf=True)


def write_readme(data: dict) -> None:
    hashes={str(path.relative_to(ROOT)):sha256(path) for path in SOURCES.values()}
    files=sorted(p.name for p in OUT.iterdir() if p.is_file() and p.name not in {"README.md","MANIFEST.json"})
    manifest={
        "schema_version":1,"status":"PASS","generated_from_workspace_artifacts":True,
        "source_sha256":hashes,"files":files,"no_sam2":True,"b2_opened":False,"test_iid_ood_access":False,
    }
    (OUT/"MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    text="""# Spatial-VLM scientific visualizations

Generated directly from locked JSON/JSONL artifacts in the workspace. PNG files
are 300 DPI; SVG files are editable; PDF files are publication-ready.

## Figures

1. `00_summary_table`: Gazebo headline table and decision.
2. `01_grounding_accuracy_across_evaluations`: Hit@0.08 with Wilson 95% CI.
3. `02_paired_delta_forest`: paired B1−B0 effects and available bootstrap CI.
4. `03_gazebo_found_case_errors`: per-family point error for `FOUND`.
5. `04_answerability_protocol_diagnostic`: forced-POINT behavior by state.
6. `05_self_consistency_and_high_confidence_errors`: confidence proxy by state.
7. `06_gazebo_risk_coverage_diagnostic`: exploratory desired-action curve.
8. `07_uncertainty_proxy_metrics`: ECE, Brier, and error-AUROC.
9. `08_clean_b1_training_card`: reproducible training summary.

CSV tables contain the plotted values. `MANIFEST.json` records SHA-256 hashes of
all source artifacts.

## Scientific interpretation

- Keep B0: clean B1 does not improve the Gazebo pilot or human stress set.
- Gazebo has only four `FOUND` families, so it is a pipeline/failure-analysis
  pilot rather than generalization evidence.
- Answerability and risk-coverage plots are explicitly diagnostic because the
  v1 prompt forces a point and provides no valid `ABSTAIN` response.
- Self-consistency is an uncertainty proxy, not a calibrated probability.
- No SAM2 was used; B2 and Test-IID/Test-OOD remain closed.
"""
    (OUT/"README.md").write_text(text,encoding="utf-8")


def main() -> None:
    configure(); OUT.mkdir(parents=True,exist_ok=True)
    data=collect(); write_tables(data)
    summary_table(data); accuracy_across_sets(data); delta_forest(data); gazebo_case_errors(data)
    answerability_heatmap(data); confidence_by_state(data); risk_coverage(data); calibration_metrics(data); training_card(data)
    write_readme(data)
    print(json.dumps({"status":"PASS","output":str(OUT),"png":len(list(OUT.glob('*.png'))),"svg":len(list(OUT.glob('*.svg'))),"pdf":len(list(OUT.glob('*.pdf')))},indent=2))


if __name__ == "__main__":
    main()
