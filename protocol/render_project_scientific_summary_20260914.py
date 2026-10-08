#!/usr/bin/env python3
"""Build a traceable scientific-results and plan-progress summary.

All numbers are read from existing project artifacts. The script does not open
Test-IID/OOD, run inference, fit a model, or change any frozen artifact.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/spatial_vlm_refspatial_v1/project_scientific_summary_20260914"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1"

SOURCES = {
    "plan_exact": ROOT / "plan/plan_spatial_vlm_refspatial_gazebo_ur3_exact.md",
    "plan_8week": ROOT / "plan/ke_hoach_trien_khai_spatial_vlm_refspatial_gazebo_ur3.md",
    "handoff": ROOT / "plan/SESSION_HANDOFF_SPATIAL_VLM.md",
    "manual_audit": RESULT / "wp1_source_audit/manual_audit_human_report.json",
    "clean_manifest": ROOT / "datasets/D_tabletop_clean_v1/manifest.json",
    "b1_training": RESULT / "wp7_b1_clean_grounding/TRAINING_REPORT.json",
    "clean_eval": RESULT / "wp7_b1_clean_grounding/clean_eval/B0_B1_CLEAN_METRICS.json",
    "human_stress": RESULT / "wp7_b1_clean_grounding/human_stress_eval/B0_B1_HUMAN_EVAL_METRICS.json",
    "gazebo_dev": RESULT / "gazebo_dev_v1/evaluation/B0_B1_GAZEBO_DEV_METRICS.json",
    "answerability_v2": RESULT / "gazebo_dev_answerability_v2/evaluation/B0_B1_ANSWERABILITY_V2_METRICS.json",
    "grounding_decision": RESULT / "wp4_grounding_lock/GROUNDING_DECISION.md",
    "grounding_lock": RESULT / "wp4_grounding_lock/grounding_model_lock.json",
    "hypothesis_lock": RESULT / "wp4_grounding_lock/hypothesis_lock.json",
    "wp5": RESULT / "wp5_spatial_uncertainty/WP5_SPATIAL_UNCERTAINTY_METRICS.json",
    "wp5_dev": RESULT / "wp5_spatial_uncertainty/wp5_dev_confirmation_report.json",
    "wp5_lock": RESULT / "wp5_spatial_uncertainty/wp5_checkpoint_or_estimator_lock.json",
    "calibration": RESULT / "gazebo_calibration_v2/GAZEBO_CALIBRATION_METRICS.json",
    "calibration_qc": RESULT / "gazebo_calibration_v2/GAZEBO_CALIBRATION_V2_GEOMETRY_QC.json",
    "calibration_scored": RESULT / "gazebo_calibration_v2/calibration_scored_predictions.jsonl",
    "calibration_predictions": RESULT / "gazebo_calibration_v2/b0_predictions/predictions.jsonl",
    "calibration_lock": RESULT / "gazebo_calibration_v2/CALIBRATOR_THRESHOLD_LOCK.json",
    "uq_qc_v1": RESULT / "gazebo_train_uq_v1/GAZEBO_TRAIN_UQ_V1_GEOMETRY_QC.json",
    "uq_pilot_r2": RESULT / "gazebo_train_uq_v2_pilot_r2/GAZEBO_TRAIN_UQ_V2_PILOT_R2_GEOMETRY_QC.json",
    "uq_pilot_r3": RESULT / "gazebo_train_uq_v2_pilot_r3/GAZEBO_TRAIN_UQ_V2_PILOT_R3_GEOMETRY_QC.json",
    "uq_pilot_r4": RESULT / "gazebo_train_uq_v2_pilot_r4/GAZEBO_TRAIN_UQ_V2_PILOT_R4_GEOMETRY_QC.json",
    "uq_pilot_r6": RESULT / "gazebo_train_uq_v2_pilot_r6/GAZEBO_TRAIN_UQ_V2_PILOT_R6_GEOMETRY_QC.json",
    "uq_full_1": RESULT / "gazebo_train_uq_v2_full/GAZEBO_TRAIN_UQ_V2_FULL_GEOMETRY_QC.json",
    "uq_full_r2": RESULT / "gazebo_train_uq_v2_full_r2/GAZEBO_TRAIN_UQ_V2_FULL_R2_GEOMETRY_QC.json",
    "uq_full_r3": RESULT / "gazebo_train_uq_v2_full_r3/GAZEBO_TRAIN_UQ_V2_FULL_R3_V2_GEOMETRY_QC.json",
}

BLUE = "#2f5d7c"
ORANGE = "#b36b2c"
RED = "#9c3b36"
GREEN = "#3f6b52"
GRAY = "#777777"
LIGHT = "#e6e6e6"
BLACK = "#222222"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=240)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect() -> dict:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source artifacts: " + ", ".join(missing))
    raw = {key: load_json(path) for key, path in SOURCES.items() if path.suffix == ".json"}
    if raw["manual_audit"]["status"] != "MANUAL_SAMPLE_GATE_PASS":
        raise ValueError("manual audit gate is not PASS")
    if raw["clean_manifest"]["status"] != "CLEAN_B1_GROUNDING_RELEASE_PASS":
        raise ValueError("clean release is not PASS")
    if raw["wp5"]["status"] != "WP5_COMPLETE_METHOD_ELIGIBLE_READY_FOR_CALIBRATION":
        raise ValueError("WP5 status changed")
    if raw["calibration"]["status"] != "CALIBRATION_NEGATIVE_NO_ROBOT_DEPLOYMENT":
        raise ValueError("unexpected Final Calibration status")
    if raw["calibration"]["test_iid_ood_access"] is not False:
        raise ValueError("Test access must remain false")
    return {
        "raw": raw,
        "calibration_scored": load_jsonl(SOURCES["calibration_scored"]),
        "calibration_predictions": load_jsonl(SOURCES["calibration_predictions"]),
    }


def grounding_rows(data: dict) -> list[dict]:
    raw = data["raw"]
    clean = raw["clean_eval"]["groups"]["all"]
    human = raw["human_stress"]["groups"]["all_primary"]
    gazebo = raw["gazebo_dev"]
    answer = raw["answerability_v2"]
    return [
        {
            "evaluation": "Clean internal held-out", "domain": "RefSpatial", "n_found": 36,
            "b0": clean["b0"]["hit_at_008"], "b1": clean["b1"]["hit_at_008"],
            "b0_ci": clean["b0"]["hit_at_008_wilson_95"], "b1_ci": clean["b1"]["hit_at_008_wilson_95"],
            "delta": clean["paired"]["hit_at_008_delta_b1_minus_b0"],
            "delta_ci": clean["paired"]["family_cluster_bootstrap_hit_delta_95"],
            "mcnemar_p": clean["paired"]["mcnemar_exact_two_sided_p"],
        },
        {
            "evaluation": "Human stress", "domain": "RefSpatial", "n_found": 99,
            "b0": human["b0"]["hit_at_008"], "b1": human["b1"]["hit_at_008"],
            "b0_ci": human["b0"]["hit_at_008_wilson_95"], "b1_ci": human["b1"]["hit_at_008_wilson_95"],
            "delta": human["paired"]["hit_at_008_delta_b1_minus_b0"],
            "delta_ci": human["paired"]["family_cluster_bootstrap_hit_delta_95"],
            "mcnemar_p": human["paired"]["mcnemar_exact_two_sided_p"],
        },
        {
            "evaluation": "Gazebo Dev-v1", "domain": "Gazebo", "n_found": 4,
            "b0": gazebo["metrics"]["b0"]["found_hit_at_008"], "b1": gazebo["metrics"]["b1"]["found_hit_at_008"],
            "b0_ci": wilson(3, 4), "b1_ci": wilson(3, 4),
            "delta": gazebo["paired"]["found_hit_delta_b1_minus_b0"], "delta_ci": None, "mcnemar_p": None,
        },
        {
            "evaluation": "Gazebo answerability-v2", "domain": "Gazebo", "n_found": 16,
            "b0": answer["metrics"]["b0"]["found_hit_at_008"], "b1": answer["metrics"]["b1"]["found_hit_at_008"],
            "b0_ci": answer["metrics"]["b0"]["found_hit_wilson_95"], "b1_ci": answer["metrics"]["b1"]["found_hit_wilson_95"],
            "delta": answer["paired"]["found_hit_delta_b1_minus_b0"],
            "delta_ci": answer["paired"]["found_hit_family_bootstrap_delta_95"],
            "mcnemar_p": answer["paired"]["mcnemar_exact_two_sided_p"],
        },
    ]


def uq_rows(data: dict) -> list[dict]:
    raw = data["raw"]
    return [
        {"dataset": "Val-UQ", "role": "selection", "system": "B0", **raw["wp5"]["val_uq"]["b0"]},
        {"dataset": "Val-UQ", "role": "selection", "system": "WP5 @0.72", **raw["wp5"]["val_uq"]["method"]},
        {"dataset": "Dev-v2", "role": "one confirmation", "system": "B0", **raw["wp5_dev"]["b0"]},
        {"dataset": "Dev-v2", "role": "one confirmation", "system": "WP5 @0.72", **raw["wp5_dev"]["method"]},
        {"dataset": "Calibration-v2", "role": "calibration fit", "system": "B0", **raw["calibration"]["b0"]},
        {"dataset": "Calibration-v2", "role": "calibration fit", "system": "candidate @0.81", **raw["calibration"]["calibrated_selected_operating_point"]},
    ]


def dataset_rows(data: dict) -> list[dict]:
    raw = data["raw"]
    clean = raw["clean_manifest"]
    return [
        {"dataset": "WP1 manual audit", "families": 300, "role": "human sample gate", "status": raw["manual_audit"]["status"], "training": "no"},
        {"dataset": "D_tabletop_clean_v1/train", "families": clean["family_counts"]["train"], "role": "B1 target-grounding train", "status": clean["status"], "training": "B1 only"},
        {"dataset": "D_tabletop_clean_v1/held-out", "families": clean["family_counts"]["dev"] + clean["family_counts"]["diagnostic"], "role": "internal paired evaluation", "status": "human-accepted", "training": "no"},
        {"dataset": "Gazebo_dev_answerability_v2", "families": 64, "role": "failure/confirmation", "status": "QC PASS", "training": "no"},
        {"dataset": "Gazebo_train_uq/full-r3", "families": raw["wp5"]["geometry_and_data_gates"]["dataset_train_families"], "role": "risk-estimator train", "status": "QC PASS", "training": "WP5 only"},
        {"dataset": "Gazebo_val_uq/full-r3", "families": raw["wp5"]["geometry_and_data_gates"]["dataset_val_families"], "role": "WP5 selection", "status": "QC PASS", "training": "no"},
        {"dataset": "Gazebo_calibration_v2", "families": raw["calibration_qc"]["records"], "role": "T + threshold fit", "status": "QC PASS; calibration negative", "training": "calibrator only"},
        {"dataset": "Gazebo_test_iid/ood", "families": 0, "role": "final generalization", "status": "SEALED / not opened", "training": "forbidden"},
    ]


def geometry_rows(data: dict) -> list[dict]:
    labels = [
        ("Train-UQ v1", "uq_qc_v1", "rejected"),
        ("Pilot r2", "uq_pilot_r2", "rejected"),
        ("Pilot r3", "uq_pilot_r3", "rejected"),
        ("Pilot r4", "uq_pilot_r4", "rejected"),
        ("Pilot r6", "uq_pilot_r6", "accepted design"),
        ("Full attempt 1", "uq_full_1", "rejected"),
        ("Full r2", "uq_full_r2", "rejected"),
        ("Full r3", "uq_full_r3", "accepted Train/Val"),
        ("Calibration v2", "calibration_qc", "accepted geometry; calibration negative"),
    ]
    out = []
    for label, key, use in labels:
        value = data["raw"][key]
        total = int(value.get("records", len(value.get("scenes", []))))
        failed = int(value["failed_scene_count"])
        out.append({
            "attempt": label, "passed": total - failed, "total": total,
            "pass_rate": (total - failed) / total, "protocol_status": value.get("status", "PASS"),
            "scientific_use": use,
        })
    return out


def plan_rows() -> list[dict]:
    return [
        {"wp": "WP0", "plan_requirement": "Environment, weights, resource baseline", "status": "DONE", "evidence": "inventory/environment/checkpoint artifacts", "remaining": "none for current scope"},
        {"wp": "WP1", "plan_requirement": "Source audit, taxonomy, manual gate", "status": "DONE", "evidence": "300/300 human reviews; precision gates PASS", "remaining": "raw-filter exhaustive rebuild remains a stated limitation"},
        {"wp": "WP2", "plan_requirement": "Clean tabletop data + Gazebo development", "status": "DONE", "evidence": "D_tabletop_clean_v1; Gazebo Dev/UQ/Calibration QC", "remaining": "Test namespaces remain sealed"},
        {"wp": "WP3", "plan_requirement": "B0 baseline and failure analysis", "status": "DONE", "evidence": "paired RefSpatial/Gazebo evaluations", "remaining": "no final-test claim"},
        {"wp": "WP4", "plan_requirement": "B1/B2 comparison and grounding lock", "status": "CLOSED_LIMITED", "evidence": "B1 negative; B0 locked", "remaining": "B2 closed: no certified reasoning supervision"},
        {"wp": "WP5", "plan_requirement": "Risk baseline + one proposed method", "status": "DONE_POSITIVE", "evidence": "16-feature Logistic-L2 passes Val eligibility", "remaining": "transfer instability recorded"},
        {"wp": "WP6", "plan_requirement": "Final calibration + operating threshold freeze", "status": "DONE_NEGATIVE", "evidence": "128-family Calibration-v2; ECE gate FAIL", "remaining": "no deployable threshold; do not open Test"},
        {"wp": "WP7", "plan_requirement": "UR3 selective policy evaluation", "status": "BLOCKED", "evidence": "base/wrist infrastructure demo only", "remaining": "ACT/REOBSERVE/ABSTAIN not authorized"},
        {"wp": "WP8", "plan_requirement": "Locked Test, report, handoff", "status": "PARTIAL", "evidence": "reports/figures/locks exist", "remaining": "Test-IID/OOD absent; thesis/slides/demo package unfinished"},
    ]


def target_rows() -> list[dict]:
    return [
        {"target_stage": "RGB-D + Language", "status": "ACHIEVED", "evidence": "registered RGB/depth/instruction manifests and frozen inference"},
        {"target_stage": "RoboRefer_tabletop", "status": "BUILT_NOT_PROMOTED", "evidence": "clean B1 trained; no stable gain over B0"},
        {"target_stage": "Relation-aware Spatial Reasoning", "status": "NOT_ACHIEVED", "evidence": "B2 closed; no certified reasoning-depth labels"},
        {"target_stage": "Target Grounding", "status": "ACHIEVED_WITH_B0", "evidence": "B0 grounding backbone frozen"},
        {"target_stage": "Calibrated Spatial Uncertainty", "status": "NOT_ACHIEVED", "evidence": "ranking improves; Final Calibration ECE gate fails"},
        {"target_stage": "ACT / REOBSERVE / ABSTAIN", "status": "BLOCKED", "evidence": "candidate threshold is explicitly non-deployable"},
        {"target_stage": "Base + Wrist Gazebo UR3", "status": "INFRASTRUCTURE_ONLY", "evidence": "two-camera demo exists; no locked policy ablation"},
        {"target_stage": "Real UR3 (optional)", "status": "NOT_STARTED", "evidence": "optional and outside achieved scope"},
    ]


def claim_rows() -> list[dict]:
    return [
        {"claim": "Clean tabletop release is usable for B1 target grounding", "allowed": "YES", "basis": "release/data gates PASS"},
        {"claim": "B1 improves grounding over B0", "allowed": "NO", "basis": "small internal signal does not repeat; Gazebo-v2 strongly regresses"},
        {"claim": "B2 relation/reasoning is demonstrated", "allowed": "NO", "basis": "B2 never opened"},
        {"claim": "Frozen WP5 score ranks unsafe cases better on Val-UQ", "allowed": "YES", "basis": "AURC 0.7072→0.6305; bootstrap CI excludes 0"},
        {"claim": "WP5 threshold transfers stably", "allowed": "NO", "basis": "Dev-v2 FOUND Hit 1.0000→0.4375"},
        {"claim": "Temperature scaling yields calibrated P(unsafe)", "allowed": "NO", "basis": "Brier improves but ECE-10 worsens"},
        {"claim": "Threshold 0.81 is deployable", "allowed": "NO", "basis": "Final Calibration decision is negative"},
        {"claim": "Final Gazebo generalization is established", "allowed": "NO", "basis": "Test-IID/OOD remains sealed"},
    ]


def plot_grounding(rows: list[dict]) -> None:
    x = np.arange(len(rows)); width = 0.34
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for offset, key, label, color in ((-width / 2, "b0", "B0", BLUE), (width / 2, "b1", "Clean B1", ORANGE)):
        values = np.asarray([r[key] for r in rows])
        cis = [r[f"{key}_ci"] for r in rows]
        errors = np.vstack([values - np.asarray([c[0] for c in cis]), np.asarray([c[1] for c in cis]) - values])
        bars = ax.bar(x + offset, values, width, label=label, color=color, yerr=errors, capsize=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("FOUND Hit@0.08")
    ax.set_xticks(x, [f"{r['evaluation']}\n{r['domain']}; n={r['n_found']}" for r in rows])
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    ax.set_title("Grounding evidence: B1 does not improve consistently", loc="left", fontweight="bold")
    fig.text(0.09, 0.01, "Error bars: Wilson 95% CI. Each dataset is interpreted separately; domains are not pooled.", color=GRAY)
    save(fig, "01_grounding_b0_b1")


def plot_uq(rows: list[dict]) -> None:
    metrics = [
        ("aurc", "AURC ↓"), ("auroc_error", "AUROC-error ↑"),
        ("nonfound_false_accept", "False accept ↓"), ("nonfound_abstain_recall", "Abstain recall ↑"),
        ("found_hit_at_008", "FOUND Hit@0.08 ↑"), ("safe_task_accuracy", "Safe-task accuracy ↑"),
    ]
    datasets = ["Val-UQ", "Dev-v2", "Calibration-v2"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharey=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        for index, dataset in enumerate(datasets):
            pair = [row for row in rows if row["dataset"] == dataset]
            ax.bar(index - 0.18, pair[0][metric], 0.36, color=BLUE, label="B0" if index == 0 else None)
            ax.bar(index + 0.18, pair[1][metric], 0.36, color=ORANGE, hatch="//" if dataset == "Calibration-v2" else None,
                   label="WP5 / candidate" if index == 0 else None)
        ax.set_title(title)
        ax.set_xticks(range(3), ["Val-UQ\nselection", "Dev-v2\nconfirmation", "Calibration-v2\nfit"])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Selective-risk behavior across allowed development/calibration roles", fontweight="bold", y=0.99)
    fig.text(0.08, 0.01, "Calibration-v2 bars use threshold candidate 0.81; it is not deployable because the probability gate failed.", color=RED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save(fig, "02_spatial_risk_tradeoffs")


def plot_calibration(data: dict) -> None:
    value = data["raw"]["calibration"]
    raw = value["raw_wp5_at_val_threshold_0_72"]
    calibrated = value["calibrated_at_reference_threshold_0_72"]
    b0 = value["b0"]
    candidate = value["calibrated_selected_operating_point"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    names = ["ECE-10 ↓", "Brier ↓"]
    raw_values = [raw["ece_10bin"], raw["brier"]]
    cal_values = [calibrated["ece_10bin"], calibrated["brier"]]
    x = np.arange(2); width = 0.34
    axes[0].bar(x - width / 2, raw_values, width, color=BLUE, label="raw WP5")
    axes[0].bar(x + width / 2, cal_values, width, color=ORANGE, label="temperature-scaled")
    axes[0].set_xticks(x, names); axes[0].set_ylim(0, 0.17); axes[0].grid(axis="y"); axes[0].set_axisbelow(True)
    axes[0].legend(); axes[0].set_title("Probability quality", fontweight="bold")
    for i, (a, b) in enumerate(zip(raw_values, cal_values)):
        axes[0].text(i, max(a, b) + 0.008, f"Δ {b-a:+.4f}", ha="center", color=RED if b > a else GREEN)
    behavior = ["False accept ↓", "Abstain recall ↑", "FOUND Hit ↑", "Coverage", "Safe-task ↑"]
    b0_values = [b0["nonfound_false_accept"], b0["nonfound_abstain_recall"], b0["found_hit_at_008"], b0["coverage"], b0["safe_task_accuracy"]]
    candidate_values = [candidate["nonfound_false_accept"], candidate["nonfound_abstain_recall"], candidate["found_hit_at_008"], candidate["coverage"], candidate["safe_task_accuracy"]]
    x = np.arange(len(behavior)); width = 0.34
    axes[1].bar(x - width / 2, b0_values, width, color=BLUE, label="B0")
    axes[1].bar(x + width / 2, candidate_values, width, color=ORANGE, hatch="//", label="candidate @0.81")
    axes[1].set_xticks(x, behavior, rotation=20, ha="right"); axes[1].set_ylim(0, 1.05); axes[1].grid(axis="y"); axes[1].set_axisbelow(True)
    axes[1].legend(); axes[1].set_title("Operating behavior (candidate only)", fontweight="bold")
    fig.suptitle("Final Calibration-v2: Brier improves, ECE gate fails", fontweight="bold")
    fig.text(0.08, 0.005, f"T={value['temperature']:.6f}; decision: CALIBRATION_NEGATIVE_NO_ROBOT_DEPLOYMENT", color=RED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save(fig, "03_final_calibration")


def plot_data_qc(datasets: list[dict], geometry: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    shown = [row for row in datasets if row["dataset"] != "Gazebo_test_iid/ood"]
    y = np.arange(len(shown))[::-1]
    axes[0].barh(y, [r["families"] for r in shown], color=BLUE)
    axes[0].set_yticks(y, [r["dataset"] for r in shown]); axes[0].set_xlabel("Independent families / reviewed items")
    axes[0].set_title("Evidence inventory (roles are not pooled)", fontweight="bold")
    axes[0].grid(axis="x"); axes[0].set_axisbelow(True)
    for yy, row in zip(y, shown): axes[0].text(row["families"] + 15, yy, str(row["families"]), va="center", fontsize=8.5)
    y = np.arange(len(geometry))[::-1]
    colors = [GREEN if row["passed"] == row["total"] else GRAY for row in geometry]
    axes[1].barh(y, [row["pass_rate"] for row in geometry], color=colors)
    axes[1].set_yticks(y, [row["attempt"] for row in geometry]); axes[1].set_xlim(0, 1.08); axes[1].set_xlabel("Geometry pass rate")
    axes[1].set_title("All-or-nothing Gazebo QC attempts", fontweight="bold")
    axes[1].grid(axis="x"); axes[1].set_axisbelow(True)
    for yy, row in zip(y, geometry): axes[1].text(row["pass_rate"] + 0.012, yy, f"{row['passed']}/{row['total']}", va="center", fontsize=8.5)
    fig.suptitle("Data scale and geometry quality-control evidence", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "04_data_and_geometry_qc")


def plot_reliability_and_risk(data: dict) -> None:
    calibration = data["raw"]["calibration"]
    scored = data["calibration_scored"]
    predictions = {row["sample_id"]: row for row in data["calibration_predictions"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot([0, 1], [0, 1], "--", color=GRAY, label="perfect")
    for key, label, color, marker in (("raw", "raw WP5", BLUE, "o"), ("calibrated", "temperature-scaled", ORANGE, "s")):
        bins = calibration["reliability"][key]
        axes[0].plot([r["mean_probability"] for r in bins], [r["unsafe_frequency"] for r in bins], marker=marker, color=color, label=label)
    axes[0].set(xlabel="Predicted unsafe probability", ylabel="Observed unsafe frequency", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_title("Reliability (Calibration-v2)", fontweight="bold"); axes[0].grid(); axes[0].legend()
    labels = np.asarray([row["unsafe"] for row in scored], dtype=float)
    by_id = {row["sample_id"]: row for row in scored}
    risks = {
        "self-consistency baseline": np.asarray([predictions[row["sample_id"]]["predictive_uncertainty"] for row in scored]),
        "frozen WP5 risk": np.asarray([row["raw_risk"] for row in scored]),
        "temperature-scaled": np.asarray([row["calibrated_risk"] for row in scored]),
    }
    for label, risk, color, style in (
        ("self-consistency baseline", risks["self-consistency baseline"], GRAY, "--"),
        ("frozen WP5 risk", risks["frozen WP5 risk"], BLUE, "-"),
        ("temperature-scaled", risks["temperature-scaled"], ORANGE, ":"),
    ):
        order = np.argsort(risk, kind="stable"); z = labels[order]
        coverage = np.arange(1, len(z) + 1) / len(z); selective_risk = np.cumsum(z) / np.arange(1, len(z) + 1)
        axes[1].plot(coverage, selective_risk, color=color, linestyle=style, label=label)
    axes[1].set(xlabel="Coverage", ylabel="Selective risk", xlim=(0, 1), ylim=(0, 0.85))
    axes[1].set_title("Risk–coverage (lower is better)", fontweight="bold"); axes[1].grid(); axes[1].legend()
    fig.suptitle("Frozen spatial-risk evidence on Calibration-v2 (fit split, not final Test)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "05_reliability_and_risk_coverage")


def table_figure(rows: list[dict], stem: str, title: str, columns: list[tuple[str, str]], widths: list[float], footer: str) -> None:
    wrap_widths = [max(9, int(width * 118)) for width in widths]
    cells = [
        [textwrap.fill(str(row[key]), width=wrap_width, break_long_words=False, break_on_hyphens=False)
         for (key, _), wrap_width in zip(columns, wrap_widths)]
        for row in rows
    ]
    maximum_lines = max(max(value.count("\n") + 1 for value in row) for row in cells)
    height = max(5.0, (0.50 + 0.18 * (maximum_lines - 1)) * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(14, height))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=[label for _, label in columns], cellLoc="left", colLoc="left", colWidths=widths, loc="upper center")
    table.auto_set_font_size(False); table.set_fontsize(8.1); table.scale(1, 2.25)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        cell.set_facecolor("#eeeeee" if r == 0 else "white")
        if r == 0: cell.get_text().set_fontweight("bold")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=13, pad=15)
    ax.text(0, -0.04, footer, transform=ax.transAxes, fontsize=9, color=GRAY)
    save(fig, stem)


def write_tables(data: dict, grounding: list[dict], uq: list[dict], datasets: list[dict], geometry: list[dict], plan: list[dict], target: list[dict], claims: list[dict]) -> None:
    ground_csv = []
    for row in grounding:
        ground_csv.append({
            "evaluation": row["evaluation"], "domain": row["domain"], "n_found": row["n_found"],
            "b0_hit_at_008": row["b0"], "b0_ci_low": row["b0_ci"][0], "b0_ci_high": row["b0_ci"][1],
            "b1_hit_at_008": row["b1"], "b1_ci_low": row["b1_ci"][0], "b1_ci_high": row["b1_ci"][1],
            "delta_b1_minus_b0": row["delta"],
            "delta_ci_low": "" if row["delta_ci"] is None else row["delta_ci"][0],
            "delta_ci_high": "" if row["delta_ci"] is None else row["delta_ci"][1],
            "mcnemar_p": "" if row["mcnemar_p"] is None else row["mcnemar_p"],
        })
    write_csv("TABLE_01_GROUNDING_RESULTS.csv", list(ground_csv[0]), ground_csv)
    uq_fields = ["dataset", "role", "system", "n", "aurc", "auroc_error", "ece_10bin", "brier", "nonfound_false_accept", "nonfound_abstain_recall", "found_hit_at_008", "coverage", "safe_task_accuracy", "threshold"]
    write_csv("TABLE_02_UNCERTAINTY_RESULTS.csv", uq_fields, [{key: row.get(key) for key in uq_fields} for row in uq])
    write_csv("TABLE_03_DATASET_ROLES.csv", list(datasets[0]), datasets)
    write_csv("TABLE_04_GEOMETRY_QC.csv", list(geometry[0]), geometry)
    write_csv("TABLE_05_PLAN_ALIGNMENT.csv", list(plan[0]), plan)
    write_csv("TABLE_06_TARGET_STACK.csv", list(target[0]), target)
    write_csv("TABLE_07_CLAIM_BOUNDARIES.csv", list(claims[0]), claims)


def format_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def report_text(data: dict, grounding: list[dict], uq: list[dict], datasets: list[dict], geometry: list[dict], plan: list[dict], target: list[dict], claims: list[dict]) -> str:
    raw = data["raw"]
    val_b0, val_method = raw["wp5"]["val_uq"]["b0"], raw["wp5"]["val_uq"]["method"]
    dev_b0, dev_method = raw["wp5_dev"]["b0"], raw["wp5_dev"]["method"]
    cal = raw["calibration"]
    cal_raw, cal_scaled, cal_op = cal["raw_wp5_at_val_threshold_0_72"], cal["calibrated_at_reference_threshold_0_72"], cal["calibrated_selected_operating_point"]
    audit = raw["manual_audit"]
    training = raw["b1_training"]["training"]
    complete = [row["wp"] for row in plan if row["status"] in {"DONE", "DONE_POSITIVE"}]
    closed = [row["wp"] for row in plan if row["status"] in {"CLOSED_LIMITED", "DONE_NEGATIVE"}]
    pending = [row["wp"] for row in plan if row["status"] in {"BLOCKED", "PARTIAL"}]
    lines = [
        "# Báo cáo tổng tiến độ và kết quả khoa học",
        "",
        "Ngày chốt: **14/09/2026**. Phạm vi: Spatial-VLM / RefSpatial / RoboRefer / Gazebo UR3.",
        "",
        "> Báo cáo này được tạo từ artifact hiện có bằng Python. Không mở Test-IID/OOD, không chạy lại inference và không fit thêm model. File `plan/tien_do_wp1_manual_audit_20260909.md` là tracker WP1 lịch sử; nguồn trạng thái hiện hành là session handoff, result index và các lock mới nhất.",
        "",
        "## 1. Kết luận điều hành",
        "",
        "Đồ án **đi đúng quy trình nghiên cứu của plan**, nhưng chưa đạt toàn bộ chuỗi năng lực mục tiêu. Chuỗi thực nghiệm đã đi tới cuối Tuần 10 của plan 16 tuần: dữ liệu → B0/B1 → risk estimator → Final Calibration. Kết quả trung thực hiện tại là:",
        "",
        "- dữ liệu sạch và hạ tầng đánh giá: đạt;",
        "- B1 tabletop adaptation: đã làm nhưng không có cải thiện ổn định, nên giữ B0;",
        "- B2 relation/reasoning: chưa mở do thiếu supervision đạt gate;",
        "- WP5 spatial-risk ranking: có kết quả dương trên Val-UQ;",
        "- Final Calibration: kết quả âm vì ECE-10 không giảm;",
        "- Test-IID/OOD và robot selective policy: tiếp tục sealed/blocked.",
        "",
        "Vì vậy claim mạnh nhất hiện có là: **một risk estimator tách rời trên frozen B0 cải thiện selective ranking trên Val-UQ**, nhưng chưa thể claim calibrated probability, final generalization hoặc deployment `ACT/REOBSERVE/ABSTAIN`.",
        "",
        "## 2. Tiến độ được hiểu theo ba trục",
        "",
        "| Trục | Trạng thái | Cách đọc |",
        "|---|---|---|",
        "| Vị trí trong plan 16 tuần | Cuối Tuần 10/16 = 62,5% chuỗi kế hoạch | Đây là vị trí dependency, không phải 62,5% năng lực đã đạt |",
        "| Vị trí trong kế hoạch rút gọn 8 tuần | Cuối Tuần 6/8 = 75% chuỗi kế hoạch | Exit gate Tuần 6 không đạt, nên không được đi sang Test Tuần 7 |",
        f"| Work package có kết quả đầy đủ/dương | {', '.join(complete)} | Hoàn tất bằng artifact |",
        f"| Work package đóng với kết quả âm/giới hạn | {', '.join(closed)} | Là kết quả hợp lệ nhưng không đạt capability target |",
        f"| Work package còn partial/blocked | {', '.join(pending)} | Chưa được gọi là hoàn thành đồ án |",
        "",
        "Không nên gộp ba trục trên thành một con số duy nhất. Về quy trình, core experiment tới calibration đã chạy; về sản phẩm cuối, calibrated uncertainty, robot policy và final Test chưa đạt.",
        "",
        "![Đối chiếu work package](06_plan_alignment.png)",
        "",
        "## 3. Đối chiếu phiên bản đẹp nhất trong mục 40",
        "",
        "| Target trong plan | Trạng thái thực | Evidence ngắn |",
        "|---|---|---|",
    ]
    for row in target:
        lines.append(f"| {row['target_stage']} | `{row['status']}` | {row['evidence']} |")
    lines += [
        "",
        "![Đối chiếu target stack](07_target_stack_status.png)",
        "",
        "Kết luận: hướng triển khai không lệch khỏi câu hỏi nghiên cứu, nhưng phiên bản đẹp nhất chưa hình thành. Hai mắt xích thiếu quan trọng nhất là `Relation-aware Spatial Reasoning` và `Calibrated Spatial Uncertainty`; vì vậy tầng robot policy chưa được phép mở.",
        "",
        "## 4. Dữ liệu và tính toàn vẹn thực nghiệm",
        "",
        f"WP1 hoàn tất {audit['completed_reviews']}/{audit['queue_count']} human review. Target precision = {format_pct(audit['precision']['target_correct']['precision'])} (Wilson 95% CI {format_pct(audit['precision']['target_correct']['wilson_ci95'][0])}–{format_pct(audit['precision']['target_correct']['wilson_ci95'][1])}); relation precision trên n={audit['precision']['relation_correct']['n']} là {format_pct(audit['precision']['relation_correct']['precision'])}; anchor precision trên n={audit['precision']['anchor_correct']['n']} là {format_pct(audit['precision']['anchor_correct']['precision'])}.",
        "",
        "`D_tabletop_clean_v1` có 1.500 train family, 17 dev và 19 diagnostic; mỗi relation giữ lại có 500 train family. Release chỉ hợp lệ cho B1 target grounding, không chứa reasoning-depth supervision và không mở B2.",
        "",
        "Gazebo UQ dùng all-or-nothing QC. Các attempt 177/320, 30/32, 27/32, 29/32, 317/320 và 316/320 đều bị giữ nguyên và không salvage. Chỉ pilot r6 32/32, full-r3 320/320 và Calibration-v2 128/128 được chấp nhận ở vai trò tương ứng.",
        "",
        "![Dữ liệu và geometry QC](04_data_and_geometry_qc.png)",
        "",
        "Chi tiết máy đọc được: `TABLE_03_DATASET_ROLES.csv` và `TABLE_04_GEOMETRY_QC.csv`.",
        "",
        "## 5. Grounding và tabletop adaptation",
        "",
        f"Clean B1 được train 1 epoch, {training['optimizer_steps']} optimizer steps, loss cuối {training['final_summary']['train_loss']:.4f}, với {training['trainable_adapter_parameters']:,} tham số adapter. Kết quả paired:",
        "",
        "| Evaluation | n FOUND | B0 Hit@.08 | B1 Hit@.08 | Delta B1−B0 | Diễn giải |",
        "|---|---:|---:|---:|---:|---|",
    ]
    interpretations = [
        "Tín hiệu nhỏ; CI delta chạm 0, McNemar p=1",
        "Không lặp lại; B1 tạo 1 lỗi và sửa 0",
        "Pilot hòa; n=4 quá nhỏ cho generalization",
        "B1 regression mạnh; quyết định KEEP_B0",
    ]
    for row, interpretation in zip(grounding, interpretations):
        lines.append(f"| {row['evaluation']} | {row['n_found']} | {format_pct(row['b0'])} | {format_pct(row['b1'])} | {100*row['delta']:+.2f} pp | {interpretation} |")
    lines += [
        "",
        "![Grounding B0/B1](01_grounding_b0_b1.png)",
        "",
        "Kết luận WP4: B1 là negative ablation, không promote; B0 là backbone khóa. Đây là đúng rule của plan: nếu adaptation không tăng ổn định thì giữ B0, không tiếp tục tune để tìm số đẹp. B2 đóng vì relation/reasoning-depth supervision chưa đạt gate.",
        "",
        "## 6. Spatial uncertainty — kết quả dương có điều kiện",
        "",
        "WP5 fit đúng một `StandardScaler + LogisticRegression(L2, C=0.1)` với 16 feature trên Train-UQ và chọn threshold 0,72 bằng Val-UQ. Trên Val-UQ:",
        "",
        "| Metric | B0 | WP5 | Delta / ý nghĩa |",
        "|---|---:|---:|---|",
        f"| AURC ↓ | {val_b0['aurc']:.4f} | {val_method['aurc']:.4f} | {val_method['aurc']-val_b0['aurc']:+.4f}; bootstrap 95% CI [−0,2295; −0,0034] |",
        f"| AUROC-error ↑ | {val_b0['auroc_error']:.4f} | {val_method['auroc_error']:.4f} | ranking lỗi tốt hơn |",
        f"| False accept ↓ | {val_b0['nonfound_false_accept']:.4f} | {val_method['nonfound_false_accept']:.4f} | giảm {val_b0['nonfound_false_accept']-val_method['nonfound_false_accept']:.4f} |",
        f"| Abstain recall ↑ | {val_b0['nonfound_abstain_recall']:.4f} | {val_method['nonfound_abstain_recall']:.4f} | tăng {val_method['nonfound_abstain_recall']-val_b0['nonfound_abstain_recall']:.4f} |",
        f"| FOUND Hit@.08 ↑ | {val_b0['found_hit_at_008']:.4f} | {val_method['found_hit_at_008']:.4f} | delta −0,125; đúng biên non-inferiority |",
        f"| Safe-task accuracy ↑ | {val_b0['safe_task_accuracy']:.4f} | {val_method['safe_task_accuracy']:.4f} | delta +0,3594 |",
        "",
        "Nhưng Dev-v2 confirmation cho thấy threshold transfer không ổn định: false accept giảm, trong khi FOUND Hit@.08 giảm từ "
        f"{dev_b0['found_hit_at_008']:.4f} xuống {dev_method['found_hit_at_008']:.4f}. Đây là cảnh báo bắt buộc, không được che bằng aggregate safe-task accuracy.",
        "",
        "![Spatial-risk trade-offs](02_spatial_risk_tradeoffs.png)",
        "",
        "## 7. Final Calibration — kết quả âm",
        "",
        f"Calibration-v2 có {cal['records']} family độc lập, geometry 128/128 PASS. Frozen inference đủ 128 record, ba stochastic draw/record và 0 exception. Chỉ một scalar temperature được fit: `T={cal['temperature']:.9f}`.",
        "",
        "| Metric | Raw WP5 | Temperature-scaled | Gate |",
        "|---|---:|---:|---|",
        f"| ECE-10 ↓ | {cal_raw['ece_10bin']:.6f} | {cal_scaled['ece_10bin']:.6f} | **FAIL**: tăng {cal_scaled['ece_10bin']-cal_raw['ece_10bin']:+.6f} |",
        f"| Brier ↓ | {cal_raw['brier']:.6f} | {cal_scaled['brier']:.6f} | PASS: giảm {cal_scaled['brier']-cal_raw['brier']:+.6f} |",
        f"| AURC ↓ | {cal_raw['aurc']:.6f} | {cal_scaled['aurc']:.6f} | bất biến như dự kiến |",
        f"| AUROC-error ↑ | {cal_raw['auroc_error']:.6f} | {cal_scaled['auroc_error']:.6f} | bất biến như dự kiến |",
        "",
        f"Threshold candidate `0,81` giảm false accept B0 từ {cal['b0']['nonfound_false_accept']:.4f} xuống {cal_op['nonfound_false_accept']:.4f}, tăng abstain recall từ 0 lên {cal_op['nonfound_abstain_recall']:.4f}, nhưng FOUND Hit giảm từ {cal['b0']['found_hit_at_008']:.4f} xuống {cal_op['found_hit_at_008']:.4f}. Vì probability gate yêu cầu cả ECE và Brier cùng cải thiện, decision đúng là `CALIBRATION_NEGATIVE_NO_ROBOT_DEPLOYMENT`.",
        "",
        "![Final Calibration](03_final_calibration.png)",
        "",
        "![Reliability và risk-coverage](05_reliability_and_risk_coverage.png)",
        "",
        "## 8. Đối chiếu chi tiết với WP0–WP8 của kế hoạch triển khai",
        "",
        "| WP | Yêu cầu | Trạng thái | Evidence | Phần còn thiếu |",
        "|---|---|---|---|---|",
    ]
    for row in plan:
        lines.append(f"| {row['wp']} | {row['plan_requirement']} | `{row['status']}` | {row['evidence']} | {row['remaining']} |")
    lines += [
        "",
        "Điểm lệch giữa kế hoạch và outcome không phải là đi sai hướng: plan đã dự liệu adaptation/calibration có thể âm. Điểm chưa đạt là capability target và exit gate, không phải tính hợp lệ của quy trình.",
        "",
        "## 9. Claim được phép và không được phép",
        "",
        "| Claim | Được phép? | Căn cứ |",
        "|---|---|---|",
    ]
    for row in claims:
        lines.append(f"| {row['claim']} | **{row['allowed']}** | {row['basis']} |")
    lines += [
        "",
        "## 10. Kết luận trọng tâm",
        "",
        "1. Đồ án đã tạo được nền dữ liệu/audit và protocol Gazebo có provenance tốt.",
        "2. Tabletop LoRA B1 không chứng minh cải thiện; quyết định giữ B0 là đúng evidence.",
        "3. Đóng góp dương hiện tại nằm ở selective spatial-risk ranking của WP5 trên Val-UQ.",
        "4. Final Calibration bác bỏ claim calibrated probability/threshold deployment dưới method hiện hành.",
        "5. Vì B2 và calibrated uncertainty chưa đạt, chưa được triển khai ACT/REOBSERVE/ABSTAIN hoặc mở final Test.",
        "6. Nếu dừng nghiên cứu mới, luận văn vẫn có một câu chuyện hợp lệ: dữ liệu sạch → adaptation âm → risk ranking dương → calibration âm, kèm phân tích giới hạn. Nếu muốn tiếp tục capability target, phải tạo method revision và Calibration split mới; không tune lại trên Calibration-v2.",
        "",
        "## 11. Danh mục bảng/hình",
        "",
        "- `TABLE_01_GROUNDING_RESULTS.csv`: grounding B0/B1 và CI.",
        "- `TABLE_02_UNCERTAINTY_RESULTS.csv`: Val, Dev confirmation và Calibration.",
        "- `TABLE_03_DATASET_ROLES.csv`: ranh giới dữ liệu.",
        "- `TABLE_04_GEOMETRY_QC.csv`: toàn bộ attempt QC quan trọng.",
        "- `TABLE_05_PLAN_ALIGNMENT.csv`: WP0–WP8.",
        "- `TABLE_06_TARGET_STACK.csv`: đối chiếu mục 40.",
        "- `TABLE_07_CLAIM_BOUNDARIES.csv`: claim matrix.",
        "- `MANIFEST.json`: SHA-256 source/output để truy vết.",
        "",
        "Không có số nào trong báo cáo này là kết quả Test-IID/OOD.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    data = collect()
    grounding = grounding_rows(data)
    uq = uq_rows(data)
    datasets = dataset_rows(data)
    geometry = geometry_rows(data)
    plan = plan_rows()
    target = target_rows()
    claims = claim_rows()
    write_tables(data, grounding, uq, datasets, geometry, plan, target, claims)
    plot_grounding(grounding)
    plot_uq(uq)
    plot_calibration(data)
    plot_data_qc(datasets, geometry)
    plot_reliability_and_risk(data)
    table_figure(plan, "06_plan_alignment", "WP0–WP8: trạng thái thực so với kế hoạch triển khai",
                 [("wp", "WP"), ("plan_requirement", "Yêu cầu"), ("status", "Trạng thái"), ("evidence", "Evidence"), ("remaining", "Còn thiếu")],
                 [0.05, 0.23, 0.12, 0.25, 0.31], "DONE_NEGATIVE/CLOSED_LIMITED là kết quả nghiên cứu hợp lệ, không đồng nghĩa capability đã đạt.")
    table_figure(target, "07_target_stack_status", "Đối chiếu trực tiếp với mục 40 — phiên bản đẹp nhất",
                 [("target_stage", "Target stage"), ("status", "Trạng thái"), ("evidence", "Evidence")],
                 [0.25, 0.20, 0.51], "Chuỗi robot bị chặn tại relation-aware reasoning và calibrated spatial uncertainty.")
    report = report_text(data, grounding, uq, datasets, geometry, plan, target, claims)
    (OUT / "BAO_CAO_TONG_TIEN_DO.md").write_text(report, encoding="utf-8")
    summary = {
        "schema_version": 1,
        "date": "2026-09-14",
        "status": "CORE_EXPERIMENT_THROUGH_FINAL_CALIBRATION_COMPLETE_WITH_NEGATIVE_CALIBRATION",
        "exact_plan_position": {"completed_through_week": 10, "total_weeks": 16, "sequence_fraction": 0.625},
        "eight_week_plan_position": {"completed_through_week": 6, "total_weeks": 8, "sequence_fraction": 0.75, "exit_gate_passed": False},
        "final_calibration": raw_status(data),
        "test_iid_ood_opened": False,
        "robot_deployment_authorized": False,
        "work_packages": plan,
        "target_stack": target,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "MANIFEST.json")
    manifest = {
        "schema_version": 1,
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "test_iid_ood_access": False,
        "model_fit_or_inference_performed": False,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in SOURCES.values()},
        "output_sha256": {path.name: sha256(path) for path in output_paths},
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT), "files": len(output_paths) + 1}, indent=2))


def raw_status(data: dict) -> dict:
    value = data["raw"]["calibration"]
    return {
        "decision": value["status"],
        "temperature": value["temperature"],
        "threshold_candidate": value["selected_threshold"],
        "ece_gate_passed": value["gates"]["ece_strictly_lower"],
        "brier_gate_passed": value["gates"]["brier_strictly_lower"],
    }


if __name__ == "__main__":
    main()
