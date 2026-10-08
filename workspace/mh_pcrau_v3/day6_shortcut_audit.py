#!/usr/bin/env python3
"""Append-only audit for the saturated Day-6 development validation.

This program does not retrain or modify the Day-6 checkpoint.  It evaluates
the frozen checkpoint, fits deliberately weak shortcut baselines on the old
Train-UQ split, and freezes the acceptance contract for a future independent
anti-shortcut validation capture before that capture is generated.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re

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

from workspace.mh_pcrau_v3.multihead_v3 import (
    ANSWERABILITY_CLASSES,
    RELATION_CLASSES,
    build_seeded_model,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/Gazebo_train_uq_v2_full_r3"
DAY6 = ROOT / "ketqua1/07_huan_luyen/ngay_06"
MANIFEST = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/S1A_FEATURE_CACHE_MANIFEST.json"
CHECKPOINT = ROOT / "workspace/mh_pcrau_v3/checkpoints/day06/s1a_best.pt"
OUT = DAY6 / "amendment_01_shortcut_audit"
METRIC_OUT = ROOT / "ketqua1/09_danh_gia/metrics/ngay_06/amendment_01"
TABLE_OUT = ROOT / "ketqua1/09_danh_gia/tables/ngay_06/amendment_01"
FIGURE_OUT = ROOT / "ketqua1/09_danh_gia/figures/ngay_06/amendment_01"
LOCK = ROOT / "protocol/MH_PCRAU_V3_ANTI_SHORTCUT_VAL_V1_LOCK.json"

STATE_ORDER = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")
RELATION_ORDER = tuple(RELATION_CLASSES)
THRESHOLDS = tuple(round(value, 3) for value in np.arange(0.01, 0.151, 0.01))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def macro_f1(prediction: list[str], target: list[str], labels: tuple[str, ...]) -> float:
    return float(f1_score(target, prediction, labels=list(labels), average="macro", zero_division=0))


def exact_all_success_ci95(sample_count: int) -> list[float]:
    # Two-sided Clopper-Pearson interval for k=n.
    return [float(0.025 ** (1.0 / sample_count)), 1.0]


def image_feature(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((16, 12))
    value = np.asarray(image, dtype=np.float32) / 255.0
    return np.concatenate((value.ravel(), value.mean((0, 1)), value.std((0, 1))))


def fit_text_probe(train: list[dict], val: list[dict], key: str) -> tuple[list[str], float, list[list[int]]]:
    probe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
        LogisticRegression(max_iter=3000, C=1.0, random_state=0),
    )
    probe.fit([row["instruction"] for row in train], [row[key] for row in train])
    prediction = list(probe.predict([row["instruction"] for row in val]))
    labels = RELATION_ORDER if key == "relation" else STATE_ORDER
    return prediction, macro_f1(prediction, [row[key] for row in val], labels), confusion_matrix(
        [row[key] for row in val], prediction, labels=list(labels)
    ).tolist()


def fit_thumbnail_probe(train: list[dict], val: list[dict], media_key: str) -> tuple[list[str], float, list[list[int]]]:
    x_train = np.stack([image_feature(ROOT / row[media_key]) for row in train])
    x_val = np.stack([image_feature(ROOT / row[media_key]) for row in val])
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=0.1, random_state=0),
    )
    probe.fit(x_train, [row["answerability"] for row in train])
    prediction = list(probe.predict(x_val))
    target = [row["answerability"] for row in val]
    return prediction, macro_f1(prediction, target, STATE_ORDER), confusion_matrix(
        target, prediction, labels=list(STATE_ORDER)
    ).tolist()


def build_rows() -> list[dict]:
    inputs = {row["sample_id"]: row for row in read_jsonl(DAY6 / "S1A_INPUT_MANIFEST.jsonl")}
    supervision = {row["sample_id"]: row for row in read_jsonl(DAY6 / "S1A_SUPERVISION_STORE.jsonl")}
    if set(inputs) != set(supervision) or len(inputs) != 320:
        raise RuntimeError("Day-6 input/supervision identity drift")
    rows = []
    for sample_id in sorted(inputs):
        source = inputs[sample_id]
        label = supervision[sample_id]
        rows.append({
            "sample_id": sample_id,
            "family_id": source["family_id"],
            "split": source["split"],
            "instruction": source["instruction"],
            "rgb_path": source["rgb_path"],
            "depth_path": source["depth_view_path"],
            "relation": label["relation"],
            "answerability": label["answerability"],
            "target_uv": label["target_uv"],
        })
    return rows


def load_features(rows: list[dict]) -> torch.Tensor:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = {row["sample_id"]: row for row in manifest["records"]}
    result = []
    for row in rows:
        record = records[row["sample_id"]]
        path = ROOT / record["feature_path"]
        if sha256(path) != record["feature_file_sha256"]:
            raise RuntimeError(f"Feature shard drift: {row['sample_id']}")
        value = torch.load(path, map_location="cpu", weights_only=True).to(torch.float32)
        if value.shape != (1536,) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Invalid feature: {row['sample_id']}")
        result.append(value)
    return torch.stack(result)


def plot_pck(rows: list[dict]) -> None:
    plt.figure(figsize=(8.4, 5.2))
    for name, color, marker in (("MH-PCRA-U-v3", "#0072B2", "o"), ("Relation-centroid baseline", "#D55E00", "s")):
        values = [row["model_hit_rate"] if name.startswith("MH") else row["centroid_baseline_hit_rate"] for row in rows]
        plt.plot([row["threshold"] for row in rows], values, label=name, color=color, marker=marker, markersize=4)
    plt.axvline(0.05, color="#666666", linestyle="--", linewidth=1, label="Primary diagnostic threshold")
    plt.axvline(0.08, color="#999999", linestyle=":", linewidth=1, label="Old saturated threshold")
    plt.xlabel("Normalized L2 threshold")
    plt.ylabel("Hit rate on FOUND Val (n=16)")
    plt.title("Day-6 development PCK: model versus no-image coordinate baseline")
    plt.xlim(0.005, 0.155)
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_OUT / "F06A_PCK_MODEL_VS_SHORTCUT_BASELINE.png", dpi=180)
    plt.close()


def plot_shortcuts(baselines: list[dict]) -> None:
    selected = [
        ("Relation\nmodel", 1.0),
        ("Relation\ntext-only", next(row["value"] for row in baselines if row["id"] == "text_relation_macro_f1")),
        ("Answerability\nmodel", 1.0),
        ("Answerability\nRGB 16x12", next(row["value"] for row in baselines if row["id"] == "rgb_thumbnail_answerability_macro_f1")),
        ("Hit@0.08\nmodel", 1.0),
        ("Hit@0.08\nrelation centroid", next(row["value"] for row in baselines if row["id"] == "relation_centroid_hit_at_0_08")),
    ]
    labels, values = zip(*selected)
    colors = ["#0072B2", "#D55E00", "#0072B2", "#D55E00", "#0072B2", "#D55E00"]
    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    bars = axis.bar(range(len(values)), values, color=colors)
    axis.set_xticks(range(len(values)), labels)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Score on old Val")
    axis.set_title("Perfect model scores are matched by deliberately weak shortcut baselines")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURE_OUT / "F06B_SHORTCUT_BASELINE_PARITY.png", dpi=180)
    plt.close(figure)


def main() -> None:
    outputs = (OUT, METRIC_OUT, TABLE_OUT, FIGURE_OUT, LOCK)
    if any(path.exists() for path in outputs):
        raise FileExistsError("Day-6 shortcut audit is append-only")
    for path in (OUT, METRIC_OUT, TABLE_OUT, FIGURE_OUT):
        path.mkdir(parents=True)

    required = (
        DAY6 / "S1A_DECISION.json",
        DAY6 / "S1A_CHECKPOINT_MANIFEST.json",
        DAY6 / "S1A_INPUT_MANIFEST.jsonl",
        DAY6 / "S1A_SUPERVISION_STORE.jsonl",
        MANIFEST,
        CHECKPOINT,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = build_rows()
    train = [row for row in rows if row["split"] == "train_uq"]
    val = [row for row in rows if row["split"] == "val_uq"]
    features = load_features(rows)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_seeded_model(26092026)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.configure_trainable("inference")
    model.eval()
    with torch.no_grad():
        output = model(features)
    relation_prediction = [RELATION_ORDER[int(value)] for value in output.relation_logits.argmax(-1)]
    answer_prediction = [STATE_ORDER[int(value)] for value in output.answerability_logits.argmax(-1)]
    point_prediction = output.mu_uv.tolist()

    val_indices = [index for index, row in enumerate(rows) if row["split"] == "val_uq"]
    val_relation_prediction = [relation_prediction[index] for index in val_indices]
    val_answer_prediction = [answer_prediction[index] for index in val_indices]
    val_relation_target = [rows[index]["relation"] for index in val_indices]
    val_answer_target = [rows[index]["answerability"] for index in val_indices]
    model_relation_f1 = macro_f1(val_relation_prediction, val_relation_target, RELATION_ORDER)
    model_answer_f1 = macro_f1(val_answer_prediction, val_answer_target, STATE_ORDER)

    text_relation_pred, text_relation_f1, text_relation_cm = fit_text_probe(train, val, "relation")
    text_answer_pred, text_answer_f1, text_answer_cm = fit_text_probe(train, val, "answerability")
    rgb_answer_pred, rgb_answer_f1, rgb_answer_cm = fit_thumbnail_probe(train, val, "rgb_path")
    depth_answer_pred, depth_answer_f1, depth_answer_cm = fit_thumbnail_probe(train, val, "depth_path")

    centers: dict[str, list[float]] = {}
    for relation in RELATION_ORDER:
        targets = np.asarray([
            row["target_uv"] for row in train
            if row["answerability"] == "FOUND" and row["relation"] == relation
        ], dtype=np.float64)
        centers[relation] = targets.mean(axis=0).tolist()

    found_val = [(index, rows[index]) for index in val_indices if rows[index]["answerability"] == "FOUND"]
    model_errors = []
    centroid_errors = []
    prediction_rows = []
    for index in val_indices:
        row = rows[index]
        point_error = None
        centroid_error = None
        if row["answerability"] == "FOUND":
            point_error = math.dist(point_prediction[index], row["target_uv"])
            centroid_error = math.dist(centers[row["relation"]], row["target_uv"])
            model_errors.append(point_error)
            centroid_errors.append(centroid_error)
        prediction_rows.append({
            "sample_id": row["sample_id"],
            "family_id": row["family_id"],
            "split": "val_uq",
            "relation_target": row["relation"],
            "relation_prediction": relation_prediction[index],
            "answerability_target": row["answerability"],
            "answerability_prediction": answer_prediction[index],
            "target_uv": row["target_uv"],
            "model_mu_uv": point_prediction[index],
            "model_l2_error": point_error,
            "relation_centroid_uv": centers[row["relation"]] if row["answerability"] == "FOUND" else None,
            "relation_centroid_l2_error": centroid_error,
        })

    pck_rows = []
    for threshold in THRESHOLDS:
        pck_rows.append({
            "threshold": threshold,
            "found_support": len(model_errors),
            "model_hit_rate": sum(error <= threshold for error in model_errors) / len(model_errors),
            "centroid_baseline_hit_rate": sum(error <= threshold for error in centroid_errors) / len(centroid_errors),
        })

    state_rule = 0
    relation_rule = 0
    for row in rows:
        suffix = int(re.search(r"(\d+)$", row["sample_id"]).group(1))
        state_rule += row["answerability"] == STATE_ORDER[(suffix % 16) // 4]
        relation_rule += row["relation"] == RELATION_ORDER[suffix % 4]

    baselines = [
        {"id": "model_relation_macro_f1", "input": "h_spatial", "metric": "macro_f1", "value": model_relation_f1, "support": 64},
        {"id": "text_relation_macro_f1", "input": "instruction_only", "metric": "macro_f1", "value": text_relation_f1, "support": 64},
        {"id": "model_answerability_macro_f1", "input": "h_spatial", "metric": "macro_f1", "value": model_answer_f1, "support": 64},
        {"id": "text_answerability_macro_f1", "input": "instruction_only", "metric": "macro_f1", "value": text_answer_f1, "support": 64},
        {"id": "rgb_thumbnail_answerability_macro_f1", "input": "RGB_16x12_linear", "metric": "macro_f1", "value": rgb_answer_f1, "support": 64},
        {"id": "depth_thumbnail_answerability_macro_f1", "input": "depth_16x12_linear", "metric": "macro_f1", "value": depth_answer_f1, "support": 64},
        {"id": "model_hit_at_0_03", "input": "h_spatial", "metric": "hit_at_0_03", "value": sum(x <= 0.03 for x in model_errors) / 16, "support": 16},
        {"id": "relation_centroid_hit_at_0_03", "input": "relation_label_only", "metric": "hit_at_0_03", "value": sum(x <= 0.03 for x in centroid_errors) / 16, "support": 16},
        {"id": "model_hit_at_0_05", "input": "h_spatial", "metric": "hit_at_0_05", "value": sum(x <= 0.05 for x in model_errors) / 16, "support": 16},
        {"id": "relation_centroid_hit_at_0_05", "input": "relation_label_only", "metric": "hit_at_0_05", "value": sum(x <= 0.05 for x in centroid_errors) / 16, "support": 16},
        {"id": "model_hit_at_0_08", "input": "h_spatial", "metric": "hit_at_0_08", "value": sum(x <= 0.08 for x in model_errors) / 16, "support": 16},
        {"id": "relation_centroid_hit_at_0_08", "input": "relation_label_only", "metric": "hit_at_0_08", "value": sum(x <= 0.08 for x in centroid_errors) / 16, "support": 16},
    ]

    with (TABLE_OUT / "T06A_SHORTCUT_BASELINE_COMPARISON.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baselines[0]))
        writer.writeheader(); writer.writerows(baselines)
    with (TABLE_OUT / "T06B_PCK_CURVE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pck_rows[0]))
        writer.writeheader(); writer.writerows(pck_rows)
    (OUT / "S1A_VAL_PREDICTIONS.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows), encoding="utf-8"
    )

    model_hit08 = next(row["value"] for row in baselines if row["id"] == "model_hit_at_0_08")
    centroid_hit08 = next(row["value"] for row in baselines if row["id"] == "relation_centroid_hit_at_0_08")
    checks = {
        "relation_text_only_matches_model": text_relation_f1 >= model_relation_f1 - 1e-12,
        "answerability_rgb_thumbnail_matches_model": rgb_answer_f1 >= model_answer_f1 - 1e-12,
        "answerability_depth_thumbnail_matches_model": depth_answer_f1 >= model_answer_f1 - 1e-12,
        "coordinate_relation_centroid_matches_model_hit08": centroid_hit08 >= model_hit08 - 1e-12,
        "sample_id_enumeration_encodes_state_all_320": state_rule == 320,
        "sample_id_enumeration_encodes_relation_all_320": relation_rule == 320,
        "found_val_support_only_16": len(model_errors) == 16,
    }
    shortcut_detected = all(checks.values())
    audit = {
        "schema_version": "1.0",
        "status": "SHORTCUT_DETECTED" if shortcut_detected else "NO_COMPLETE_SHORTCUT_PARITY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-hoc diagnostic of old Day-6 development Val; no checkpoint or label changes",
        "checkpoint": {"path": str(CHECKPOINT.relative_to(ROOT)), "sha256": sha256(CHECKPOINT), "epoch": checkpoint["epoch"]},
        "dataset": {"path": str(DATA.relative_to(ROOT)), "train": 256, "val": 64, "found_val": 16},
        "checks": checks,
        "sample_id_rule_accuracy": {"answerability": state_rule / 320, "relation": relation_rule / 320},
        "model": {
            "relation_macro_f1": model_relation_f1,
            "answerability_macro_f1": model_answer_f1,
            "point_l2_mean": float(np.mean(model_errors)),
            "point_l2_max": float(np.max(model_errors)),
            "perfect_64_exact_binomial_ci95": exact_all_success_ci95(64),
            "perfect_16_exact_binomial_ci95": exact_all_success_ci95(16),
        },
        "baselines": baselines,
        "confusion_matrices": {
            "labels_relation": list(RELATION_ORDER),
            "labels_answerability": list(STATE_ORDER),
            "text_relation": text_relation_cm,
            "text_answerability": text_answer_cm,
            "rgb_thumbnail_answerability": rgb_answer_cm,
            "depth_thumbnail_answerability": depth_answer_cm,
        },
        "interpretation": {
            "relation": "Prompt parsing metric, not evidence of visual spatial reasoning.",
            "answerability": "Old Val nuisance/template structure is linearly separable at 16x12 RGB and depth.",
            "coordinate": "Hit@0.08 is saturated by relation-conditioned coordinate priors.",
            "claim": "Engineering completion remains valid; generalization claim is placed on hold.",
        },
    }
    write_json(METRIC_OUT / "S1A_SHORTCUT_AUDIT.json", audit)

    plot_pck(pck_rows)
    plot_shortcuts(baselines)

    lock = {
        "schema_version": "1.0",
        "protocol_id": "mh_pcrau_v3_anti_shortcut_val_v1",
        "status": "FROZEN_BEFORE_CAPTURE_AND_MODEL_INFERENCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Independent one-shot development challenge for honest model selection evidence; not Test.",
        "frozen_model": {"path": str(CHECKPOINT.relative_to(ROOT)), "sha256": sha256(CHECKPOINT)},
        "required_families": 256,
        "cell_design": {"answerability_states": list(STATE_ORDER), "relations": list(RELATION_ORDER), "families_per_cell": 16},
        "generation_requirements": {
            "fresh_parent_families": True,
            "historical_layout_signature_overlap": 0,
            "historical_physical_signature_overlap": 0,
            "historical_rgb_sha256_overlap": 0,
            "sample_order": "seeded cryptographic permutation after labels are assigned; ID must not encode state/relation",
            "camera_pose_strata": 4,
            "object_triplet_strata_minimum": 8,
            "continuous_position_sampling": True,
            "matched_nuisance_across_answerability_states": ["camera_pose", "background", "object_set", "lighting", "target_x_bin", "target_y_bin"],
            "instruction_paraphrase_templates_per_relation_minimum": 4,
            "answerability_state_must_not_be_named_in_prompt": True,
            "found_target_xy_bins": {"x_bins": 4, "y_bins": 4, "minimum_per_occupied_bin": 2},
        },
        "qc_before_inference": {
            "accepted_families_exact": 256,
            "state_relation_balance_exact": True,
            "rgb_depth_registration": "PASS_ALL",
            "semantic_geometry_answerability_qc": "PASS_ALL",
            "near_duplicate_phash_distance_min": 8,
            "text_only_answerability_macro_f1_max": 0.35,
            "rgb_thumbnail_16x12_answerability_macro_f1_max": 0.70,
            "depth_thumbnail_16x12_answerability_macro_f1_max": 0.70,
            "relation_centroid_hit_at_0_05_max": 0.60,
            "all_or_nothing": True,
        },
        "evaluation": {
            "one_shot": True,
            "no_retraining_or_threshold_tuning_after_open": True,
            "primary_metrics": ["answerability_macro_f1", "point_l2_mean", "hit_at_0_03", "hit_at_0_05"],
            "secondary_metrics": ["PCK_0.01_to_0.15", "Gaussian_NLL", "per_class_recall", "false_accept_rate"],
            "relation_head_label": "instruction_relation_parser_auxiliary",
            "confidence_intervals": "95% parent-family bootstrap, 10000 draws, seed 25092026",
            "mandatory_baselines": ["text_only", "RGB_thumbnail_16x12_linear", "depth_thumbnail_16x12_linear", "relation_centroid"],
            "minimum_effect_over_matching_shortcut_baseline": 0.05,
        },
        "gate": {
            "old_val_generalization_claim": "INVALIDATED_BY_SHORTCUT_AUDIT",
            "day7_oof_s1b": "HOLD_UNTIL_ANTI_SHORTCUT_DATA_QC_PASS_AND_ONE_SHOT_EVALUATION",
            "calibration_test_robot": "SEALED",
            "permitted_claim_before_new_evaluation": "S1A_ENGINEERING_COMPLETE_ONLY",
        },
        "source_bindings": {
            "audit_source": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__).resolve())},
            "old_decision": {"path": "ketqua1/07_huan_luyen/ngay_06/S1A_DECISION.json", "sha256": sha256(DAY6 / "S1A_DECISION.json")},
            "audit_metrics": {"path": str((METRIC_OUT / "S1A_SHORTCUT_AUDIT.json").relative_to(ROOT)), "sha256": sha256(METRIC_OUT / "S1A_SHORTCUT_AUDIT.json")},
        },
    }
    write_json(LOCK, lock)

    decision = {
        "schema_version": "1.0",
        "outcome": "S1A_ENGINEERING_COMPLETE_GENERALIZATION_HOLD",
        "supersedes_scope": "Day-6 authorization to proceed directly to Day-7 OOF/S1b",
        "does_not_modify": ["Day-6 checkpoint", "Day-6 raw metrics", "Day-6 labels", "sealed Test/Calibration"],
        "shortcut_detected": shortcut_detected,
        "day7_oof_s1b_authorized": False,
        "required_next_gate": "ANTI_SHORTCUT_VAL_V1_DATA_QC_PASS_AND_ONE_SHOT_EVALUATION",
        "evidence": {
            "audit": str((METRIC_OUT / "S1A_SHORTCUT_AUDIT.json").relative_to(ROOT)),
            "table": str((TABLE_OUT / "T06A_SHORTCUT_BASELINE_COMPARISON.csv").relative_to(ROOT)),
            "pck": str((TABLE_OUT / "T06B_PCK_CURVE.csv").relative_to(ROOT)),
            "lock": str(LOCK.relative_to(ROOT)),
        },
    }
    write_json(OUT / "AMENDMENT_DECISION.json", decision)

    artifacts = []
    for path in sorted([*OUT.glob("*"), *METRIC_OUT.glob("*"), *TABLE_OUT.glob("*"), *FIGURE_OUT.glob("*"), LOCK]):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            artifacts.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_json(OUT / "ARTIFACT_MANIFEST.json", {
        "schema_version": "1.0", "status": "COMPLETE", "artifacts": artifacts,
    })
    print(json.dumps({
        "status": audit["status"],
        "decision": decision["outcome"],
        "text_relation_f1": text_relation_f1,
        "rgb_thumbnail_answerability_f1": rgb_answer_f1,
        "depth_thumbnail_answerability_f1": depth_answer_f1,
        "model_hit08": model_hit08,
        "centroid_hit08": centroid_hit08,
    }, indent=2))


if __name__ == "__main__":
    main()
