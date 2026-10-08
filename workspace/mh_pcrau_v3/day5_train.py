"""Locked gradient-scale smoke and real-feature mini-overfit for Day 5."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import torch

from workspace.mh_pcrau_v3.loss_v3 import LossWeights, SupervisionBatch, compute_multitask_loss
from workspace.mh_pcrau_v3.multihead_v3 import (
    ANSWERABILITY_CLASSES, RELATION_CLASSES, build_seeded_model,
)


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY5_RUN_LOCK.json"
CACHE_MANIFEST = ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/FEATURE_CACHE_MANIFEST.json"
CACHE_QC = ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/CACHE_QC.json"
MODEL_OUT = ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_05"
LOSS_OUT = ROOT / "ketqua1/06_ham_mat_mat/ngay_05"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def build_batch(lock: dict, cache: dict) -> tuple[torch.Tensor, SupervisionBatch, list[str]]:
    mini_ids = lock["selection"]["mini_batch_sample_ids"]
    cache_by_id = {row["sample_id"]: row for row in cache["records"]}
    supervision = read_jsonl(ROOT / lock["supervision_store"]["path"])
    supervision_by_id = {row["sample_id"]: row for row in supervision}
    if set(mini_ids) - cache_by_id.keys() or set(mini_ids) - supervision_by_id.keys():
        raise RuntimeError("Mini-batch references missing cache/supervision")
    features = torch.stack([
        torch.load(ROOT / cache_by_id[sample_id]["feature_path"], map_location="cpu", weights_only=True)
        for sample_id in mini_ids
    ])
    relation = torch.tensor([
        RELATION_CLASSES.index(supervision_by_id[sample_id]["relation"]) for sample_id in mini_ids
    ], dtype=torch.long)
    answerability = torch.tensor([
        ANSWERABILITY_CLASSES.index(supervision_by_id[sample_id]["answerability"]) for sample_id in mini_ids
    ], dtype=torch.long)
    spatial_mask = answerability == 0
    target_uv = torch.zeros(len(mini_ids), 2, dtype=torch.float32)
    for index, sample_id in enumerate(mini_ids):
        value = supervision_by_id[sample_id]["target_uv"]
        if spatial_mask[index]:
            if value is None:
                raise RuntimeError("FOUND row missing target_uv")
            target_uv[index] = torch.tensor(value)
        elif value is not None:
            raise RuntimeError("Non-FOUND row unexpectedly has target_uv")
    false = torch.zeros(len(mini_ids), dtype=torch.bool)
    target = SupervisionBatch(
        relation=relation, relation_mask=torch.ones_like(false),
        reasoning_depth=torch.full_like(relation, -1), reasoning_mask=false.clone(),
        target_uv=target_uv, spatial_mask=spatial_mask,
        uncertainty_source=torch.full_like(relation, -1), source_mask=false.clone(),
        answerability=answerability, answerability_mask=torch.ones_like(false),
    )
    return features, target, mini_ids


def grad_norm(parameters) -> float:
    total = sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None)
    return math.sqrt(total)


def gradient_scale(lock: dict, features: torch.Tensor, target: SupervisionBatch) -> tuple[list[dict], dict]:
    model = build_seeded_model(lock["seed"])
    model.eval()
    trunk = list(model.shared_trunk.parameters())
    base_norms = {}
    for task in ("relation", "spatial", "answerability"):
        model.zero_grad(set_to_none=True)
        output = model(features)
        losses = compute_multitask_loss(output, target)
        losses.components[task].backward()
        base_norms[task] = grad_norm(trunk)
        if not math.isfinite(base_norms[task]) or base_norms[task] <= 0:
            raise RuntimeError(f"Invalid base gradient norm for {task}")
    rows = []
    for order, candidate in enumerate(lock["gradient_scale"]["candidates"]):
        weighted = {task: base_norms[task] * float(candidate[task]) for task in base_norms}
        ratio = max(weighted.values()) / min(weighted.values())
        mean = sum(weighted.values()) / len(weighted)
        cv = math.sqrt(sum((value - mean) ** 2 for value in weighted.values()) / len(weighted)) / mean
        rows.append({
            "order": order, "candidate_id": candidate["id"],
            **{f"weight_{name}": candidate[name] for name in (
                "relation", "reasoning", "spatial", "source", "answerability", "confidence", "language_modeling"
            )},
            **{f"base_grad_{task}": base_norms[task] for task in base_norms},
            **{f"weighted_grad_{task}": weighted[task] for task in weighted},
            "weighted_max_min_ratio": ratio, "weighted_cv": cv,
        })
    selected = min(rows, key=lambda row: (row["weighted_max_min_ratio"], row["order"]))
    return rows, selected


def make_weights(row: dict) -> LossWeights:
    return LossWeights(
        relation=float(row["weight_relation"]), reasoning=float(row["weight_reasoning"]),
        spatial=float(row["weight_spatial"]), source=float(row["weight_source"]),
        answerability=float(row["weight_answerability"]), confidence=0.0,
        language_modeling=0.0,
    )


@torch.no_grad()
def evaluate(model, features, target, weights, step: int, grad_value: float | None = None) -> dict:
    model.eval()
    output = model(features)
    losses = compute_multitask_loss(output, target, weights)
    spatial = target.spatial_mask
    logvar = output.log_variance_uv[spatial]
    finite_tensors = [getattr(output, name) for name in (
        "relation_logits", "reasoning_logits", "mu_uv", "log_variance_uv", "source_logits",
        "answerability_logits", "confidence_logit"
    )] + [losses.total, *losses.components.values()]
    finite = all(bool(torch.isfinite(value).all()) for value in finite_tensors)
    return {
        "step": step,
        "total_loss": float(losses.total),
        "relation_loss": float(losses.components["relation"]),
        "spatial_nll": float(losses.components["spatial"]),
        "answerability_loss": float(losses.components["answerability"]),
        "reasoning_loss": float(losses.components["reasoning"]),
        "source_loss": float(losses.components["source"]),
        "confidence_loss": float(losses.components["confidence"]),
        "relation_accuracy": float((output.relation_logits.argmax(-1) == target.relation).float().mean()),
        "answerability_accuracy": float((output.answerability_logits.argmax(-1) == target.answerability).float().mean()),
        "point_mae_uv": float((output.mu_uv[spatial] - target.target_uv[spatial]).abs().mean()),
        "log_variance_min": float(logvar.min()), "log_variance_max": float(logvar.max()),
        "log_variance_mean": float(logvar.mean()),
        "gradient_norm": grad_value,
        "finite": finite,
        "valid_relation": losses.valid_counts["relation"],
        "valid_spatial": losses.valid_counts["spatial"],
        "valid_answerability": losses.valid_counts["answerability"],
        "valid_reasoning": losses.valid_counts["reasoning"],
        "valid_source": losses.valid_counts["source"],
        "valid_confidence": losses.valid_counts["confidence"],
    }


def thresholds_pass(initial: dict, current: dict, thresholds: dict) -> bool:
    return (
        current["finite"]
        and current["relation_loss"] / initial["relation_loss"] <= thresholds["relation_loss_final_over_initial_max"]
        and current["answerability_loss"] / initial["answerability_loss"] <= thresholds["answerability_loss_final_over_initial_max"]
        and current["relation_accuracy"] >= thresholds["relation_accuracy_min"]
        and current["answerability_accuracy"] >= thresholds["answerability_accuracy_min"]
        and current["point_mae_uv"] <= thresholds["point_mae_uv_max"]
        and current["spatial_nll"] < initial["spatial_nll"]
        and current["log_variance_min"] > thresholds["log_variance_min_strictly_above"]
        and current["log_variance_max"] < thresholds["log_variance_max_strictly_below"]
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if MODEL_OUT.exists() or LOSS_OUT.exists():
        raise FileExistsError("Day-5 training artifacts are append-only")
    lock = json.loads(LOCK.read_text())
    if lock["status"] != "FROZEN_BEFORE_DAY5_MODEL_FORWARD":
        raise RuntimeError("Day-5 lock invalid")
    for reference in lock["prerequisites"] + [lock["cache_input_manifest"], lock["supervision_store"]]:
        if sha256(ROOT / reference["path"]) != reference["sha256"]:
            raise RuntimeError(f"Locked prerequisite drift: {reference['path']}")
    cache_qc = json.loads(CACHE_QC.read_text())
    if cache_qc["status"] != "PASS" or sha256(CACHE_MANIFEST) != cache_qc["manifest_sha256"]:
        raise RuntimeError("Cache QC or manifest hash failed")
    cache = json.loads(CACHE_MANIFEST.read_text())
    features, target, mini_ids = build_batch(lock, cache)
    if features.shape != (16, 1536) or not torch.isfinite(features).all():
        raise RuntimeError("Mini-batch feature contract failed")
    torch.manual_seed(lock["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    MODEL_OUT.mkdir(parents=True)
    LOSS_OUT.mkdir(parents=True)
    gradient_rows, selected = gradient_scale(lock, features, target)
    gradient_csv = LOSS_OUT / "GRADIENT_SCALE.csv"
    write_csv(gradient_csv, gradient_rows)
    weights = make_weights(selected)
    weight_decision = {
        "schema_version": "1.0", "status": "FROZEN_BEFORE_S1A",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_lock_sha256": sha256(LOCK), "gradient_scale_sha256": sha256(gradient_csv),
        "candidate_count": len(gradient_rows), "selection_rule": lock["gradient_scale"]["selection_rule"],
        "selected_candidate": selected["candidate_id"],
        "selected_weights": {
            name: getattr(weights, name) for name in (
                "relation", "reasoning", "spatial", "source", "answerability", "confidence", "language_modeling"
            )
        },
        "active_gradient_tasks": ["relation", "spatial", "answerability"],
        "masked_no_support": ["reasoning", "source", "confidence"],
        "selected_weighted_max_min_ratio": selected["weighted_max_min_ratio"],
        "interpretation": "Selection uses development mini-batch gradient scale only; no Val/Calibration/Test metric.",
    }
    json_write(LOSS_OUT / "LOSS_WEIGHT_DECISION.json", weight_decision)
    model = build_seeded_model(lock["seed"])
    model.configure_trainable("s1a")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lock["mini_overfit"]["learning_rate"],
        weight_decay=lock["mini_overfit"]["weight_decay"],
    )
    initial = evaluate(model, features, target, weights, 0)
    log_rows = [initial]
    final = initial
    passed_at = None
    all_steps_finite = True
    last_grad = None
    for step in range(1, lock["mini_overfit"]["maximum_steps"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = compute_multitask_loss(model(features), target, weights)
        if not torch.isfinite(result.total):
            all_steps_finite = False
            break
        result.total.backward()
        last_grad = grad_norm(parameter for parameter in model.parameters() if parameter.requires_grad)
        if not math.isfinite(last_grad):
            all_steps_finite = False
            break
        optimizer.step()
        if step % lock["mini_overfit"]["evaluation_interval"] == 0:
            final = evaluate(model, features, target, weights, step, last_grad)
            log_rows.append(final)
            if step >= lock["mini_overfit"]["minimum_steps"] and thresholds_pass(
                initial, final, lock["mini_overfit"]["pass_thresholds"]
            ):
                passed_at = step
                break
    train_log = MODEL_OUT / "MINIBATCH_TRAIN_LOG.csv"
    write_csv(train_log, log_rows)
    checks = {
        "cache_qc_pass": cache_qc["status"] == "PASS",
        "mini_batch_16_independent_families": len(mini_ids) == len(set(mini_ids)) == 16,
        "relation_support_16": final["valid_relation"] == 16,
        "answerability_support_16": final["valid_answerability"] == 16,
        "spatial_found_support_4": final["valid_spatial"] == 4,
        "reasoning_source_confidence_masked_zero": final["valid_reasoning"] == final["valid_source"] == final["valid_confidence"] == 0,
        "all_steps_finite": all_steps_finite and all(row["finite"] for row in log_rows),
        "preregistered_thresholds_met": passed_at is not None,
        "variance_inside_locked_bounds": final["log_variance_min"] > -7.95 and final["log_variance_max"] < 1.95,
        "day4_mask_gradient_tests_passed": json.loads((ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_04/DAY4_DECISION.json").read_text())["tests"] == "16/16 PASS",
    }
    overfit = {
        "schema_version": "1.0", "status": "PASS" if all(checks.values()) else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_lock": {"path": str(LOCK.relative_to(ROOT)), "sha256": sha256(LOCK)},
        "cache_manifest_sha256": sha256(CACHE_MANIFEST),
        "mini_batch_sample_ids": mini_ids, "features_shape": list(features.shape),
        "selected_loss_weights": weight_decision["selected_weights"],
        "checks": checks, "initial": initial, "final": final,
        "passed_at_step": passed_at, "maximum_steps": lock["mini_overfit"]["maximum_steps"],
        "thresholds": lock["mini_overfit"]["pass_thresholds"],
        "train_log": {"path": str(train_log.relative_to(ROOT)), "sha256": sha256(train_log)},
        "scope_limits": {
            "reasoning": "N/A: certified support 0; loss masked",
            "source": "N/A exploratory: certified support 0; loss masked",
            "confidence": "N/A in S1a: parameters frozen; OOF gate deferred to S1b",
            "scientific_metric_claim": False,
        },
    }
    json_write(MODEL_OUT / "MINIBATCH_OVERFIT.json", overfit)
    g2_pass = overfit["status"] == "PASS"
    decision = {
        "schema_version": "1.0", "date_local": "2026-09-24",
        "outcome": "G2_PASS" if g2_pass else "G2_STOP",
        "scope": "S1a core head sanity on 16 real cached development families",
        "day6_s1a_authorized": g2_pass,
        "cache": f"{cache_qc['samples']}/{lock['cache_acceptance']['expected_samples']} PASS",
        "mini_overfit": overfit["status"], "passed_at_step": passed_at,
        "loss_weight_candidate": selected["candidate_id"],
        "active_heads": ["relation", "coordinate_logvariance", "answerability"],
        "masked_not_claimed": ["reasoning", "uncertainty_source", "confidence"],
        "confidence_gate": "DEFERRED_TO_OOF_S1B_BY_LOCKED_ARCHITECTURE",
        "calibration_test_robot": "SEALED",
        "evidence": {
            "cache_qc": "ketqua1/03_backbone_h_spatial/ngay_05/CACHE_QC.json",
            "overfit": "ketqua1/04_multihead_pcrau_v3/ngay_05/MINIBATCH_OVERFIT.json",
            "gradient_scale": "ketqua1/06_ham_mat_mat/ngay_05/GRADIENT_SCALE.csv",
            "loss_weight_decision": "ketqua1/06_ham_mat_mat/ngay_05/LOSS_WEIGHT_DECISION.json"
        }
    }
    json_write(MODEL_OUT / "G2_DECISION.json", decision)
    print(json.dumps({"outcome": decision["outcome"], "selected": selected["candidate_id"],
                      "passed_at_step": passed_at, "initial": initial, "final": final}, ensure_ascii=False))
    if not g2_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
