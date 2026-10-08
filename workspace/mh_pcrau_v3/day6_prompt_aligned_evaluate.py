#!/usr/bin/env python3
"""Score the post-hoc prompt-aligned paired-view counterfactual."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import torch

from workspace.mh_pcrau_v3.day6_paired_view_evaluate import bootstrap_ci
from workspace.mh_pcrau_v3.multihead_v3 import ANSWERABILITY_CLASSES, RELATION_CLASSES, build_seeded_model


ROOT = Path(__file__).resolve().parents[2]
ALIGNED_ROOT = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_03_prompt_aligned"
SHORT_ROOT = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_02_paired_view"
MANIFEST = ALIGNED_ROOT / "PROMPT_ALIGNED_FEATURE_CACHE_MANIFEST.json"
QC = ALIGNED_ROOT / "PROMPT_ALIGNED_CACHE_QC.json"
SHORT_MANIFEST = SHORT_ROOT / "PAIRED_VIEW_FEATURE_CACHE_MANIFEST.json"
SUPERVISION = ALIGNED_ROOT / "PROMPT_ALIGNED_SUPERVISION.jsonl"
CHECKPOINT = ROOT / "workspace/mh_pcrau_v3/checkpoints/day06/s1a_best.pt"
LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY6_PROMPT_ALIGNED_EVALUATION_LOCK.json"
OLD_AUDIT = ROOT / "ketqua1/09_danh_gia/metrics/ngay_06/amendment_01/S1A_SHORTCUT_AUDIT.json"
SHORT_METRICS = ROOT / "ketqua1/09_danh_gia/metrics/ngay_06/amendment_02/PAIRED_VIEW_STRESS_METRICS.json"

OUT = ROOT / "ketqua1/07_huan_luyen/ngay_06/amendment_03_prompt_aligned"
METRIC_OUT = ROOT / "ketqua1/09_danh_gia/metrics/ngay_06/amendment_03"
TABLE_OUT = ROOT / "ketqua1/09_danh_gia/tables/ngay_06/amendment_03"
FIGURE_OUT = ROOT / "ketqua1/09_danh_gia/figures/ngay_06/amendment_03"
RELATIONS = tuple(RELATION_CLASSES); STATES = tuple(ANSWERABILITY_CLASSES)
THRESHOLDS = tuple(round(value, 3) for value in np.arange(0.01, 0.151, 0.01))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def macro_f1(target, prediction, labels) -> float:
    return float(f1_score(target, prediction, labels=list(labels), average="macro", zero_division=0))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_cache(records: dict, sample_ids: list[str]) -> torch.Tensor:
    result = []
    for sample_id in sample_ids:
        row = records[sample_id]; path = ROOT / row["feature_path"]
        if sha256(path) != row["feature_file_sha256"]:
            raise RuntimeError(f"Feature drift: {sample_id}")
        result.append(torch.load(path, map_location="cpu", weights_only=True).to(torch.float32))
    return torch.stack(result)


def main() -> None:
    if any(path.exists() for path in (OUT, METRIC_OUT, TABLE_OUT, FIGURE_OUT)):
        raise FileExistsError("Prompt-aligned evaluation is append-only")
    for path in (OUT, METRIC_OUT, TABLE_OUT, FIGURE_OUT): path.mkdir(parents=True)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "FROZEN_BEFORE_LABEL_OPEN_AND_SCORING": raise RuntimeError("Invalid evaluation lock")
    for ref in lock["bindings"]:
        if sha256(ROOT / ref["path"]) != ref["sha256"]: raise RuntimeError(f"Binding drift: {ref['path']}")
    qc = json.loads(QC.read_text(encoding="utf-8"))
    if qc["status"] != "PASS" or sha256(MANIFEST) != qc["manifest_sha256"]: raise RuntimeError("Ineligible aligned cache")

    labels = {row["sample_id"]: row for row in read_jsonl(SUPERVISION)}
    aligned_records = {row["sample_id"]: row for row in json.loads(MANIFEST.read_text())["records"]}
    short_records = {row["sample_id"]: row for row in json.loads(SHORT_MANIFEST.read_text())["records"]}
    sample_ids = sorted(labels)
    if set(sample_ids) != set(aligned_records) or set(sample_ids) != set(short_records): raise RuntimeError("Identity mismatch")
    aligned = load_cache(aligned_records, sample_ids); short = load_cache(short_records, sample_ids)
    cosine = torch.nn.functional.cosine_similarity(aligned, short, dim=1)
    feature_shift = {
        "cosine_similarity_mean": float(cosine.mean()), "cosine_similarity_min": float(cosine.min()),
        "l2_shift_mean": float((aligned - short).norm(dim=1).mean()),
        "aligned_norm_mean": float(aligned.norm(dim=1).mean()), "short_norm_mean": float(short.norm(dim=1).mean()),
    }
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_seeded_model(26092026); model.load_state_dict(checkpoint["model_state_dict"])
    model.configure_trainable("inference"); model.eval()
    with torch.no_grad(): output = model(aligned)
    relation_prediction = [RELATIONS[int(value)] for value in output.relation_logits.argmax(-1)]
    answer_prediction = [STATES[int(value)] for value in output.answerability_logits.argmax(-1)]
    point_prediction = output.mu_uv.tolist()
    relation_target = [labels[sample_id]["relation"] for sample_id in sample_ids]
    answer_target = [labels[sample_id]["answerability"] for sample_id in sample_ids]
    errors = []
    predictions = []
    for index, sample_id in enumerate(sample_ids):
        row = labels[sample_id]; error = None
        if row["answerability"] == "FOUND":
            error = math.dist(point_prediction[index], row["target_uv"]); errors.append(error)
        predictions.append({"sample_id": sample_id, "family_id": row["family_id"],
                            "relation_target": row["relation"], "relation_prediction": relation_prediction[index],
                            "answerability_target": row["answerability"], "answerability_prediction": answer_prediction[index],
                            "target_uv": row["target_uv"], "model_mu_uv": point_prediction[index], "model_l2_error": error})
    metrics_model = {
        "relation_macro_f1": macro_f1(relation_target, relation_prediction, RELATIONS),
        "answerability_macro_f1": macro_f1(answer_target, answer_prediction, STATES),
        "point_l2_mean": float(np.mean(errors)), "point_l2_median": float(np.median(errors)), "point_l2_max": float(np.max(errors)),
        "hit_at_0_03": sum(value <= 0.03 for value in errors) / 64,
        "hit_at_0_05": sum(value <= 0.05 for value in errors) / 64,
        "hit_at_0_08": sum(value <= 0.08 for value in errors) / 64,
    }
    pck = [{"threshold": threshold, "found_support": 64,
            "prompt_short_hit_rate": None,
            "prompt_aligned_hit_rate": sum(value <= threshold for value in errors) / 64}
           for threshold in THRESHOLDS]
    short_metrics = json.loads(SHORT_METRICS.read_text(encoding="utf-8"))
    short_pck_path = ROOT / "ketqua1/09_danh_gia/tables/ngay_06/amendment_02/T06D_PAIRED_VIEW_PCK.csv"
    with short_pck_path.open(encoding="utf-8") as handle:
        short_pck = {float(row["threshold"]): float(row["model_hit_rate"]) for row in csv.DictReader(handle)}
    for row in pck: row["prompt_short_hit_rate"] = short_pck[row["threshold"]]
    metrics = {
        "schema_version": "1.0", "status": "POSTHOC_DIAGNOSTIC_COMPLETE",
        "scope": "Same 256 paired RGB-D inputs; exact canonical output suffix appended after observing short-prompt failure",
        "confirmatory": False, "generalization_hold_cleared": False,
        "model": metrics_model, "ci95_family_bootstrap": bootstrap_ci(answer_target, answer_prediction, errors),
        "answerability_confusion_matrix": {"labels": list(STATES), "values": confusion_matrix(answer_target, answer_prediction, labels=list(STATES)).tolist()},
        "feature_shift_short_to_aligned": feature_shift,
        "short_prompt_reference": short_metrics["model"],
        "interpretation_rule": "Improvement diagnoses prompt sensitivity only; it cannot validate generalization or authorize S1b.",
    }
    write_json(METRIC_OUT / "PROMPT_ALIGNED_DIAGNOSTIC_METRICS.json", metrics)
    (OUT / "PROMPT_ALIGNED_PREDICTIONS.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    with (TABLE_OUT / "T06E_PROMPT_COUNTERFACTUAL.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = ("relation_macro_f1", "answerability_macro_f1", "point_l2_mean", "hit_at_0_03", "hit_at_0_05", "hit_at_0_08")
        rows = [{"metric": key, "prompt_short": short_metrics["model"][key], "prompt_aligned": metrics_model[key],
                 "delta_aligned_minus_short": metrics_model[key] - short_metrics["model"][key]} for key in keys]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (TABLE_OUT / "T06F_PROMPT_ALIGNED_PCK.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pck[0])); writer.writeheader(); writer.writerows(pck)

    old = json.loads(OLD_AUDIT.read_text(encoding="utf-8"))["model"]
    labels_chart = ["Relation F1", "Answerability F1", "Hit@0.03", "Hit@0.05"]
    old_values = [old["relation_macro_f1"], old["answerability_macro_f1"], 0.9375, 1.0]
    short_values = [short_metrics["model"][key] for key in ("relation_macro_f1", "answerability_macro_f1", "hit_at_0_03", "hit_at_0_05")]
    aligned_values = [metrics_model[key] for key in ("relation_macro_f1", "answerability_macro_f1", "hit_at_0_03", "hit_at_0_05")]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8)); x = np.arange(4); width = 0.25
    axes[0].bar(x-width, old_values, width, label="Old Val", color="#999999")
    axes[0].bar(x, short_values, width, label="Paired + short prompt", color="#D55E00")
    axes[0].bar(x+width, aligned_values, width, label="Paired + aligned prompt", color="#0072B2")
    axes[0].set_xticks(x, labels_chart, rotation=15); axes[0].set_ylim(0, 1.08); axes[0].set_ylabel("Score")
    axes[0].set_title("Prompt-format counterfactual (post-hoc diagnostic)"); axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=.25)
    axes[1].plot([row["threshold"] for row in pck], [row["prompt_short_hit_rate"] for row in pck], marker="o", label="Short prompt")
    axes[1].plot([row["threshold"] for row in pck], [row["prompt_aligned_hit_rate"] for row in pck], marker="s", label="Aligned prompt")
    axes[1].axvline(.05, color="#666666", linestyle="--"); axes[1].set_ylim(-.03, 1.03)
    axes[1].set_xlabel("Normalized L2 threshold"); axes[1].set_ylabel("Hit rate (n=64)"); axes[1].set_title("PCK on identical RGB-D")
    axes[1].legend(); axes[1].grid(alpha=.25); figure.tight_layout()
    figure.savefig(FIGURE_OUT / "F06D_PROMPT_FORMAT_COUNTERFACTUAL.png", dpi=180); plt.close(figure)
    decision = {"schema_version": "1.0", "outcome": "PROMPT_SENSITIVITY_DIAGNOSED",
                "confirmatory": False, "generalization_hold_cleared": False, "day7_oof_s1b_authorized": False,
                "model_metrics": metrics_model, "feature_shift": feature_shift,
                "required_fix": "Train with locked prompt-format augmentation and multi-view inputs, then evaluate once on fresh Anti-Shortcut Val v1."}
    write_json(OUT / "PROMPT_ALIGNED_DECISION.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__": main()
