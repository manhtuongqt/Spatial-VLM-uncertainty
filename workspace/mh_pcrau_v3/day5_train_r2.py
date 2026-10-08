"""Day-5 attempt 02: preregistered variance-stability remediation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from workspace.mh_pcrau_v3.day5_train import (
    CACHE_MANIFEST, CACHE_QC, ROOT, build_batch, evaluate, grad_norm,
    json_write, sha256, thresholds_pass, write_csv,
)
from workspace.mh_pcrau_v3.loss_v3 import LossWeights, compute_multitask_loss
from workspace.mh_pcrau_v3.multihead_v3 import build_seeded_model


LOCK1 = ROOT / "protocol/MH_PCRAU_V3_DAY5_RUN_LOCK.json"
LOCK2 = ROOT / "protocol/MH_PCRAU_V3_DAY5_RUN_LOCK_R2.json"
WEIGHT_DECISION = ROOT / "ketqua1/06_ham_mat_mat/ngay_05/LOSS_WEIGHT_DECISION.json"
OUT = ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_05/revision_02"
FINAL_DECISION = ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_05/G2_DECISION_REVISION_V2.json"


def main() -> None:
    if OUT.exists() or FINAL_DECISION.exists():
        raise FileExistsError("Day-5 revision 02 is append-only")
    lock1 = json.loads(LOCK1.read_text())
    lock2 = json.loads(LOCK2.read_text())
    if lock2["status"] != "FROZEN_BEFORE_DAY5_ATTEMPT_02":
        raise RuntimeError("Revision lock invalid")
    if sha256(LOCK1) != lock2["inherited_lock"]["sha256"]:
        raise RuntimeError("Inherited lock drift")
    for reference in lock2["prerequisites"]:
        if sha256(ROOT / reference["path"]) != reference["sha256"]:
            raise RuntimeError(f"Revision prerequisite drift: {reference['path']}")
    cache_qc = json.loads(CACHE_QC.read_text())
    cache = json.loads(CACHE_MANIFEST.read_text())
    if cache_qc["status"] != "PASS" or sha256(CACHE_MANIFEST) != cache_qc["manifest_sha256"]:
        raise RuntimeError("Cache not eligible")
    features, target, mini_ids = build_batch(lock1, cache)
    if mini_ids != lock2["mini_batch_sample_ids"]:
        raise RuntimeError("Mini-batch identity drift")
    supervision = {
        row["sample_id"]: row for row in
        [json.loads(line) for line in (ROOT / lock1["supervision_store"]["path"]).read_text().splitlines() if line]
    }
    families = [supervision[sample_id]["family_id"] for sample_id in mini_ids]
    if len(set(families)) != 16:
        raise RuntimeError("Mini-batch families are not independent")
    weight_record = json.loads(WEIGHT_DECISION.read_text())
    if weight_record["selected_weights"] != lock2["selected_loss_weights"]:
        raise RuntimeError("Loss-weight drift")
    weights = LossWeights(**lock2["selected_loss_weights"])
    recipe = lock2["optimizer_revision"]
    torch.manual_seed(lock2["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = build_seeded_model(lock2["seed"])
    model.configure_trainable("s1a")
    variance_parameters = list(model.log_variance_head.parameters())
    variance_ids = {id(parameter) for parameter in variance_parameters}
    main_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in variance_ids
    ]
    variance_initial = [parameter.detach().clone() for parameter in variance_parameters]
    optimizer = torch.optim.AdamW([
        {"params": main_parameters, "lr": recipe["main_learning_rate"]},
        {"params": variance_parameters, "lr": recipe["log_variance_head_learning_rate"]},
    ], weight_decay=recipe["weight_decay"])
    initial = evaluate(model, features, target, weights, 0)
    log_rows = [initial]
    final = initial
    passed_at = None
    all_finite = True
    clipped_steps = 0
    for step in range(1, recipe["maximum_steps"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = compute_multitask_loss(model(features), target, weights)
        if not torch.isfinite(loss.total):
            all_finite = False
            break
        loss.total.backward()
        preclip = float(torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            recipe["maximum_gradient_norm"], error_if_nonfinite=True,
        ))
        if not math.isfinite(preclip):
            all_finite = False
            break
        clipped_steps += int(preclip > recipe["maximum_gradient_norm"])
        optimizer.step()
        if step % recipe["evaluation_interval"] == 0:
            final = evaluate(model, features, target, weights, step, preclip)
            log_rows.append(final)
            if step >= recipe["minimum_steps"] and thresholds_pass(
                initial, final, lock2["pass_thresholds_unchanged"]
            ):
                passed_at = step
                break
    OUT.mkdir(parents=True)
    train_log = OUT / "MINIBATCH_TRAIN_LOG.csv"
    write_csv(train_log, log_rows)
    variance_delta = math.sqrt(sum(
        float((after.detach() - before).double().square().sum())
        for after, before in zip(variance_parameters, variance_initial)
    ))
    checks = {
        "cache_qc_pass": cache_qc["status"] == "PASS",
        "mini_batch_16_independent_families": len(set(families)) == 16,
        "relation_support_16": final["valid_relation"] == 16,
        "answerability_support_16": final["valid_answerability"] == 16,
        "spatial_found_support_4": final["valid_spatial"] == 4,
        "reasoning_source_confidence_masked_zero": final["valid_reasoning"] == final["valid_source"] == final["valid_confidence"] == 0,
        "all_steps_finite": all_finite and all(row["finite"] for row in log_rows),
        "preregistered_thresholds_met": passed_at is not None,
        "variance_inside_locked_bounds": final["log_variance_min"] > -7.95 and final["log_variance_max"] < 1.95,
        "variance_head_received_update": variance_delta > 0,
        "attempt_01_preserved": json.loads((ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_05/G2_DECISION.json").read_text())["outcome"] == "G2_STOP",
    }
    result = {
        "schema_version": "1.0", "status": "PASS" if all(checks.values()) else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_lock": {"path": str(LOCK2.relative_to(ROOT)), "sha256": sha256(LOCK2)},
        "attempt_01_retained": "ketqua1/04_multihead_pcrau_v3/ngay_05/MINIBATCH_OVERFIT.json",
        "cache_manifest_sha256": sha256(CACHE_MANIFEST),
        "mini_batch_sample_ids": mini_ids, "mini_batch_family_ids": families,
        "features_shape": list(features.shape), "selected_loss_weights": lock2["selected_loss_weights"],
        "optimizer_recipe": recipe, "checks": checks, "initial": initial, "final": final,
        "passed_at_step": passed_at, "clipped_steps": clipped_steps,
        "variance_parameter_delta_l2": variance_delta,
        "thresholds": lock2["pass_thresholds_unchanged"],
        "train_log": {"path": str(train_log.relative_to(ROOT)), "sha256": sha256(train_log)},
        "scope_limits": {
            "reasoning": "N/A: certified support 0; loss masked",
            "source": "N/A exploratory: certified support 0; loss masked",
            "confidence": "N/A in S1a: frozen; OOF gate deferred to S1b",
            "scientific_metric_claim": False,
        }
    }
    result_path = OUT / "MINIBATCH_OVERFIT.json"
    json_write(result_path, result)
    passed = result["status"] == "PASS"
    decision = {
        "schema_version": "1.0", "date_local": "2026-09-24",
        "revision": "append_only_v2_after_variance_stability_remediation",
        "outcome": "G2_PASS" if passed else "G2_STOP",
        "scope": "S1a core head sanity on 16 real cached development families",
        "supersedes_for_next_gate_only": "G2_DECISION.json; attempt 01 remains negative evidence",
        "day6_s1a_authorized": passed,
        "cache": "32/32 PASS", "mini_overfit_revision_02": result["status"],
        "passed_at_step": passed_at, "loss_weight_candidate": "W2_SPA050",
        "active_heads_passed": ["relation", "coordinate_logvariance", "answerability"] if passed else [],
        "masked_not_claimed": ["reasoning", "uncertainty_source", "confidence"],
        "confidence_gate": "DEFERRED_TO_OOF_S1B_BY_LOCKED_ARCHITECTURE",
        "calibration_test_robot": "SEALED",
        "evidence": {
            "cache_qc": "ketqua1/03_backbone_h_spatial/ngay_05/CACHE_QC.json",
            "attempt_01": "ketqua1/04_multihead_pcrau_v3/ngay_05/MINIBATCH_OVERFIT.json",
            "attempt_02": str(result_path.relative_to(ROOT)),
            "loss_weight_decision": "ketqua1/06_ham_mat_mat/ngay_05/LOSS_WEIGHT_DECISION.json"
        }
    }
    json_write(FINAL_DECISION, decision)
    print(json.dumps({"outcome": decision["outcome"], "passed_at_step": passed_at,
                      "variance_delta_l2": variance_delta, "initial": initial,
                      "final": final}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
