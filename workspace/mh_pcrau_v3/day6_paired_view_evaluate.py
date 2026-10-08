#!/usr/bin/env python3
"""One-shot scoring of the frozen S1a checkpoint on paired second views."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from workspace.mh_pcrau_v3.multihead_v3 import ANSWERABILITY_CLASSES, RELATION_CLASSES, build_seeded_model


ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_02_paired_view"
INPUT = CACHE_ROOT / "PAIRED_VIEW_INPUT_MANIFEST.jsonl"
SUPERVISION = CACHE_ROOT / "PAIRED_VIEW_SUPERVISION.jsonl"
CACHE_MANIFEST = CACHE_ROOT / "PAIRED_VIEW_FEATURE_CACHE_MANIFEST.json"
CACHE_QC = CACHE_ROOT / "PAIRED_VIEW_CACHE_QC.json"
CHECKPOINT = ROOT / "workspace/mh_pcrau_v3/checkpoints/day06/s1a_best.pt"
EVAL_LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY6_PAIRED_VIEW_EVALUATION_LOCK.json"
OLD_INPUT = ROOT / "ketqua1/07_huan_luyen/ngay_06/S1A_INPUT_MANIFEST.jsonl"
OLD_SUPERVISION = ROOT / "ketqua1/07_huan_luyen/ngay_06/S1A_SUPERVISION_STORE.jsonl"
OLD_AUDIT = ROOT / "ketqua1/09_danh_gia/metrics/ngay_06/amendment_01/S1A_SHORTCUT_AUDIT.json"

OUT = ROOT / "ketqua1/07_huan_luyen/ngay_06/amendment_02_paired_view"
METRIC_OUT = ROOT / "ketqua1/09_danh_gia/metrics/ngay_06/amendment_02"
TABLE_OUT = ROOT / "ketqua1/09_danh_gia/tables/ngay_06/amendment_02"
FIGURE_OUT = ROOT / "ketqua1/09_danh_gia/figures/ngay_06/amendment_02"

RELATIONS = tuple(RELATION_CLASSES)
STATES = tuple(ANSWERABILITY_CLASSES)
THRESHOLDS = tuple(round(value, 3) for value in np.arange(0.01, 0.151, 0.01))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def macro_f1(target, prediction, labels) -> float:
    return float(f1_score(target, prediction, labels=list(labels), average="macro", zero_division=0))


def image_feature(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((16, 12))
    value = np.asarray(image, dtype=np.float32) / 255.0
    return np.concatenate((value.ravel(), value.mean((0, 1)), value.std((0, 1))))


def percentile_ci(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def bootstrap_ci(target_answer, prediction_answer, errors, seed=25092026, draws=10000):
    rng = np.random.default_rng(seed)
    answer_values = []
    hit03_values = []
    hit05_values = []
    l2_values = []
    n_answer = len(target_answer)
    n_point = len(errors)
    for _ in range(draws):
        ai = rng.integers(0, n_answer, n_answer)
        pi = rng.integers(0, n_point, n_point)
        answer_values.append(macro_f1(
            [target_answer[i] for i in ai], [prediction_answer[i] for i in ai], STATES
        ))
        sampled = [errors[i] for i in pi]
        hit03_values.append(sum(value <= 0.03 for value in sampled) / n_point)
        hit05_values.append(sum(value <= 0.05 for value in sampled) / n_point)
        l2_values.append(float(np.mean(sampled)))
    return {
        "answerability_macro_f1": percentile_ci(answer_values),
        "hit_at_0_03": percentile_ci(hit03_values),
        "hit_at_0_05": percentile_ci(hit05_values),
        "point_l2_mean": percentile_ci(l2_values),
    }


def main() -> None:
    if any(path.exists() for path in (OUT, METRIC_OUT, TABLE_OUT, FIGURE_OUT)):
        raise FileExistsError("Paired-view evaluation is append-only")
    for path in (OUT, METRIC_OUT, TABLE_OUT, FIGURE_OUT):
        path.mkdir(parents=True)
    lock = json.loads(EVAL_LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "FROZEN_BEFORE_LABEL_OPEN_AND_SCORING":
        raise RuntimeError("Invalid paired-view evaluation lock")
    for ref in lock["bindings"]:
        if sha256(ROOT / ref["path"]) != ref["sha256"]:
            raise RuntimeError(f"Evaluation binding drift: {ref['path']}")
    cache_qc = json.loads(CACHE_QC.read_text(encoding="utf-8"))
    if cache_qc["status"] != "PASS" or sha256(CACHE_MANIFEST) != cache_qc["manifest_sha256"]:
        raise RuntimeError("Paired-view cache is not eligible")

    inputs = {row["sample_id"]: row for row in read_jsonl(INPUT)}
    labels = {row["sample_id"]: row for row in read_jsonl(SUPERVISION)}
    cache = {row["sample_id"]: row for row in json.loads(CACHE_MANIFEST.read_text())["records"]}
    if set(inputs) != set(labels) or set(inputs) != set(cache) or len(inputs) != 256:
        raise RuntimeError("Paired-view identity mismatch")
    sample_ids = sorted(inputs)
    features = []
    for sample_id in sample_ids:
        record = cache[sample_id]
        path = ROOT / record["feature_path"]
        if sha256(path) != record["feature_file_sha256"]:
            raise RuntimeError(f"Cache shard drift: {sample_id}")
        features.append(torch.load(path, map_location="cpu", weights_only=True).to(torch.float32))
    features = torch.stack(features)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_seeded_model(26092026)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.configure_trainable("inference"); model.eval()
    with torch.no_grad():
        output = model(features)
    relation_prediction = [RELATIONS[int(value)] for value in output.relation_logits.argmax(-1)]
    answer_prediction = [STATES[int(value)] for value in output.answerability_logits.argmax(-1)]
    point_prediction = output.mu_uv.tolist()
    relation_target = [labels[sample_id]["relation"] for sample_id in sample_ids]
    answer_target = [labels[sample_id]["answerability"] for sample_id in sample_ids]

    old_inputs = {row["sample_id"]: row for row in read_jsonl(OLD_INPUT)}
    old_labels = {row["sample_id"]: row for row in read_jsonl(OLD_SUPERVISION)}
    old_train_ids = sorted(sample_id for sample_id, row in old_inputs.items() if row["split"] == "train_uq")
    old_instruction = [old_inputs[sample_id]["instruction"] for sample_id in old_train_ids]
    new_instruction = [inputs[sample_id]["instruction"] for sample_id in sample_ids]

    baseline_predictions = {}
    for key, target_values in (
        ("relation", [old_labels[sample_id]["relation"] for sample_id in old_train_ids]),
        ("answerability", [old_labels[sample_id]["answerability"] for sample_id in old_train_ids]),
    ):
        probe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
            LogisticRegression(max_iter=3000, C=1.0, random_state=0),
        )
        probe.fit(old_instruction, target_values)
        baseline_predictions[f"text_{key}"] = list(probe.predict(new_instruction))

    for media_key, name in (("rgb_path", "rgb_thumbnail_answerability"), ("depth_view_path", "depth_thumbnail_answerability")):
        x_train = np.stack([image_feature(ROOT / old_inputs[sample_id][media_key]) for sample_id in old_train_ids])
        x_stress = np.stack([image_feature(ROOT / inputs[sample_id][media_key]) for sample_id in sample_ids])
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, random_state=0))
        probe.fit(x_train, [old_labels[sample_id]["answerability"] for sample_id in old_train_ids])
        baseline_predictions[name] = list(probe.predict(x_stress))

    centers = {}
    for relation in RELATIONS:
        points = np.asarray([
            old_labels[sample_id]["target_uv"] for sample_id in old_train_ids
            if old_labels[sample_id]["answerability"] == "FOUND" and old_labels[sample_id]["relation"] == relation
        ])
        centers[relation] = points.mean(axis=0).tolist()

    errors = []
    centroid_errors = []
    prediction_rows = []
    for index, sample_id in enumerate(sample_ids):
        label = labels[sample_id]
        error = None
        baseline_error = None
        if label["answerability"] == "FOUND":
            error = math.dist(point_prediction[index], label["target_uv"])
            baseline_error = math.dist(centers[label["relation"]], label["target_uv"])
            errors.append(error); centroid_errors.append(baseline_error)
        prediction_rows.append({
            "sample_id": sample_id, "family_id": label["family_id"],
            "relation_target": label["relation"], "relation_prediction": relation_prediction[index],
            "answerability_target": label["answerability"], "answerability_prediction": answer_prediction[index],
            "target_uv": label["target_uv"], "model_mu_uv": point_prediction[index],
            "model_l2_error": error, "relation_centroid_l2_error": baseline_error,
        })

    relation_f1 = macro_f1(relation_target, relation_prediction, RELATIONS)
    answer_f1 = macro_f1(answer_target, answer_prediction, STATES)
    pck = [{
        "threshold": threshold, "found_support": len(errors),
        "model_hit_rate": sum(value <= threshold for value in errors) / len(errors),
        "centroid_baseline_hit_rate": sum(value <= threshold for value in centroid_errors) / len(centroid_errors),
    } for threshold in THRESHOLDS]
    metrics = {
        "schema_version": "1.0", "status": "COMPLETE",
        "scope": "Paired second-view stress diagnostic; same parent families as S1a Train-UQ; not independent Val/Test",
        "checkpoint_sha256": sha256(CHECKPOINT), "checkpoint_epoch": checkpoint["epoch"],
        "support": {"all": 256, "found": 64, "per_answerability": dict(Counter(answer_target))},
        "model": {
            "relation_macro_f1": relation_f1,
            "answerability_macro_f1": answer_f1,
            "answerability_per_class_recall": {
                state: sum(p == state and t == state for p, t in zip(answer_prediction, answer_target)) / sum(t == state for t in answer_target)
                for state in STATES
            },
            "point_l2_mean": float(np.mean(errors)), "point_l2_median": float(np.median(errors)),
            "point_l2_max": float(np.max(errors)),
            "hit_at_0_03": sum(value <= 0.03 for value in errors) / len(errors),
            "hit_at_0_05": sum(value <= 0.05 for value in errors) / len(errors),
            "hit_at_0_08": sum(value <= 0.08 for value in errors) / len(errors),
        },
        "ci95_parent_family_bootstrap": bootstrap_ci(answer_target, answer_prediction, errors),
        "baselines": {
            "text_relation_macro_f1": macro_f1(relation_target, baseline_predictions["text_relation"], RELATIONS),
            "text_answerability_macro_f1": macro_f1(answer_target, baseline_predictions["text_answerability"], STATES),
            "rgb_thumbnail_answerability_macro_f1": macro_f1(answer_target, baseline_predictions["rgb_thumbnail_answerability"], STATES),
            "depth_thumbnail_answerability_macro_f1": macro_f1(answer_target, baseline_predictions["depth_thumbnail_answerability"], STATES),
            "relation_centroid_point_l2_mean": float(np.mean(centroid_errors)),
            "relation_centroid_hit_at_0_03": sum(value <= 0.03 for value in centroid_errors) / len(centroid_errors),
            "relation_centroid_hit_at_0_05": sum(value <= 0.05 for value in centroid_errors) / len(centroid_errors),
            "relation_centroid_hit_at_0_08": sum(value <= 0.08 for value in centroid_errors) / len(centroid_errors),
        },
        "confusion_matrix": {"labels": list(STATES), "answerability": confusion_matrix(answer_target, answer_prediction, labels=list(STATES)).tolist()},
        "limitations": [
            "All 256 parent families were seen by S1a through their canonical view.",
            "This result cannot clear GENERALIZATION_HOLD or authorize confidence training.",
            "The frozen Anti-Shortcut Val v1 with fresh families remains required.",
        ],
    }
    write_json(METRIC_OUT / "PAIRED_VIEW_STRESS_METRICS.json", metrics)
    (OUT / "PAIRED_VIEW_PREDICTIONS.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows), encoding="utf-8"
    )
    with (TABLE_OUT / "T06C_PAIRED_VIEW_SUMMARY.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [
            {"metric": key, "model": value, "shortcut_baseline": metrics["baselines"].get({
                "relation_macro_f1": "text_relation_macro_f1",
                "answerability_macro_f1": "rgb_thumbnail_answerability_macro_f1",
                "point_l2_mean": "relation_centroid_point_l2_mean",
                "hit_at_0_03": "relation_centroid_hit_at_0_03",
                "hit_at_0_05": "relation_centroid_hit_at_0_05",
                "hit_at_0_08": "relation_centroid_hit_at_0_08",
            }[key]), "support": 64 if key.startswith("point") or key.startswith("hit") else 256}
            for key, value in metrics["model"].items() if key in {"relation_macro_f1", "answerability_macro_f1", "point_l2_mean", "hit_at_0_03", "hit_at_0_05", "hit_at_0_08"}
        ]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (TABLE_OUT / "T06D_PAIRED_VIEW_PCK.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pck[0])); writer.writeheader(); writer.writerows(pck)

    old = json.loads(OLD_AUDIT.read_text(encoding="utf-8"))["model"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    old_values = [old["relation_macro_f1"], old["answerability_macro_f1"], 1.0, 1.0]
    new_values = [relation_f1, answer_f1, metrics["model"]["hit_at_0_03"], metrics["model"]["hit_at_0_05"]]
    labels_chart = ["Relation F1", "Answerability F1", "Hit@0.03", "Hit@0.05"]
    x = np.arange(4); width = 0.36
    axes[0].bar(x - width / 2, old_values, width, label="Old Val", color="#999999")
    axes[0].bar(x + width / 2, new_values, width, label="Paired second view", color="#0072B2")
    axes[0].set_xticks(x, labels_chart, rotation=15); axes[0].set_ylim(0, 1.08); axes[0].set_ylabel("Score")
    axes[0].set_title("Frozen checkpoint under paired view shift"); axes[0].legend(); axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot([row["threshold"] for row in pck], [row["model_hit_rate"] for row in pck], marker="o", label="Model")
    axes[1].plot([row["threshold"] for row in pck], [row["centroid_baseline_hit_rate"] for row in pck], marker="s", label="Relation centroid")
    axes[1].axvline(0.05, color="#666666", linestyle="--"); axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("Normalized L2 threshold"); axes[1].set_ylabel("Hit rate (n=64)")
    axes[1].set_title("Paired-view PCK"); axes[1].legend(); axes[1].grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(FIGURE_OUT / "F06C_PAIRED_VIEW_ROBUSTNESS.png", dpi=180); plt.close(figure)

    result = {
        "schema_version": "1.0", "outcome": "PAIRED_VIEW_DIAGNOSTIC_COMPLETE",
        "generalization_hold_cleared": False, "day7_oof_s1b_authorized": False,
        "reason": "Paired views share S1a parent families; fresh Anti-Shortcut Val v1 is still required.",
        "metrics": metrics["model"], "baselines": metrics["baselines"],
    }
    write_json(OUT / "PAIRED_VIEW_DECISION.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
