#!/usr/bin/env python3
"""Full calibration fit-only comparison after train/dev acceptance head freeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pcrau.acceptance_ranker import AcceptanceRanker
from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.losses import class_weights
from pcrau.model import PCRAUTargetV2
from pcrau.selective_experiment import apply_risk, choose_policy_threshold, fit_risk, policy_result
from pcrau.utils import atomic_json, load_config, seed_everything, sha256_file, workspace_path
from experiment_answerability_head import make_loader
from experiment_hard_cases_selective import capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    root = workspace_path("new/outputs")/args.run_name
    if root.parent != workspace_path("new/outputs"):
        raise ValueError("Invalid run name")
    summary = json.loads((root/"summary.json").read_text())
    if not summary["selected"] or not summary["residual"]:
        raise ValueError("No eligible residual candidate frozen on dev")
    candidate = root/summary["selected"]/"diagnostic.pt"
    source = workspace_path("new/outputs/pcrau_target_v2_full_seed_24082026")
    checkpoint, original_config = source/"checkpoints/best/model.safetensors", source/"config.json"
    output = root/"calibration_full_fit"
    output.mkdir(exist_ok=False)
    protected = [checkpoint, original_config, source/"evaluation/calibration/calibrator.json",
                 workspace_path("new/hinhanh/19_answerability_confusion/fig19_answerability_confusion_test_iid.png")]
    hashes = {str(p):sha256_file(p) for p in protected}
    # Write freeze lock before constructing or opening the calibration dataset.
    lock = {"selected_arm": summary["selected"], "selected_epoch": summary["arms"][summary["selected"]]["epoch"],
            "ranker_checkpoint": str(candidate), "ranker_sha256": sha256_file(candidate),
            "source_v2_sha256": sha256_file(checkpoint), "selected_using": "train/dev only",
            "calibration_script_sha256":sha256_file(Path(__file__)),
            "calibration_opened_before_freeze": False, "test_accessed": False,
            "calibration_is_fit_only_not_independent_audit": True}
    atomic_json(output/"freeze_lock.json", lock)
    seed_everything(24082026)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(original_config)
    config["model"]["export_answerability_evidence"] = True
    base = PCRAUTargetV2(config).to(device).eval().requires_grad_(False)
    load_model_checkpoint(base, checkpoint, sha256_file(original_config))
    saved = torch.load(candidate, map_location=device, weights_only=True)
    assert saved["residual"] and saved["source_v2_sha256"] == lock["source_v2_sha256"]
    head = AcceptanceRanker(saved["input_dim"], saved["hidden_dim"], saved["dropout"]).to(device).eval()
    head.load_state_dict(saved["state_dict"])
    dataset = ArchivedPCRAUDataset(config, "calibration", profile="calibration")
    assert len(dataset) == 1000 and len({e["family_id"] for e in dataset.entries}) == 200
    loader, _ = make_loader(dataset, config, False)
    train = ArchivedPCRAUDataset(config, "train")
    weights, source_weights = class_weights(train, device)
    print("CALIBRATION FIT ONLY after ranker freeze", flush=True)
    _, rows, cache = capture(base, loader, dataset, device, config, weights, source_weights, output/"baseline_calibration")
    with torch.no_grad():
        base_logit = (torch.logsumexp(cache["baseline_logits"][:,1:],dim=-1)-cache["baseline_logits"][:,0]).to(device)
        logits = (base_logit+head(cache["evidence"].float().to(device))).cpu().numpy()
    for row, value in zip(rows, logits):
        row["ranker_error_logit"] = float(value)
    with (output/"ranker_predictions.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row)+"\n")
    results = {}
    for name, extra in [("reference_logistic33", False), ("ranker_logistic34", True)]:
        cal, scores = fit_risk(rows, True, .01, ranker_feature=extra)
        cal.update({"ranker_sha256":lock["ranker_sha256"] if extra else None,
                    "source_v2_sha256":lock["source_v2_sha256"]})
        atomic_json(output/f"{name}_calibrator.json", cal)
        error = np.array([row["evaluation"]["error_event"] for row in rows])
        fit_all = apply_risk(rows, cal)
        with (output/f"{name}_scores.jsonl").open("x") as handle:
            for row, score, fit in zip(rows, scores, fit_all):
                handle.write(json.dumps({"sample_id":row["sample_id"],"family_id":row["family_id"],
                            "crossfit_risk":float(score),"fit_all_risk":float(fit),"error_event":row["evaluation"]["error_event"]})+"\n")
        profiles = {}
        for policy in ("hard_found", "risk_only"):
            eligible = np.ones(len(rows), bool) if policy == "risk_only" else np.array([
                max(row["answerability_probabilities"],key=row["answerability_probabilities"].get)=="FOUND" for row in rows])
            threshold = choose_policy_threshold(scores, error, [row["family_id"] for row in rows], eligible, .05, 60)
            metrics, records = policy_result(rows, scores, policy, threshold["threshold"])
            metrics["valid_acceptance_recall"] = metrics["correct_acceptances"]/metrics["correct_found_total"]
            profiles[policy] = {"threshold_selection":threshold,"crossfit_metrics":metrics}
            with (output/f"{name}_{policy}_decisions.jsonl").open("x") as handle:
                for row in records:
                    handle.write(json.dumps(row)+"\n")
        results[name] = {"probability_metrics":cal["crossfit_metrics"],"profiles":profiles}
        if name == "reference_logistic33":
            assert abs(cal["crossfit_metrics"]["aurc"]-.2534433227558651) < 1e-8
            assert profiles["risk_only"]["crossfit_metrics"]["correct_acceptances"] == 261
            assert profiles["risk_only"]["crossfit_metrics"]["errors"] == 6
        print(json.dumps({"calibrator":name,"profiles":profiles}), flush=True)
    assert hashes == {str(p):sha256_file(p) for p in protected}
    assert lock["ranker_sha256"] == sha256_file(candidate)
    reference_metrics = results["reference_logistic33"]["profiles"]["risk_only"]["crossfit_metrics"]
    candidate_metrics = results["ranker_logistic34"]["profiles"]["risk_only"]["crossfit_metrics"]
    gain = candidate_metrics["correct_acceptances"]-reference_metrics["correct_acceptances"]
    conclusion = {"correct_acceptance_gain_at_5_percent":gain,
                  "acceptance_recall_goal_achieved_on_calibration":gain > 0,
                  "replace_main_policy":False,"independent_test_required":True}
    atomic_json(output/"summary.json", {"status":"CALIBRATION_FIT_ONLY", "results":results,"conclusion":conclusion,
                "freeze":lock,"source_hashes_unchanged":True,"test_evaluated":False,"robot_motion":False,
                "thresholds_selected_on_same_crossfit_calibration":True})
    lines = ["# Calibration fit-only của acceptance residual", "",
             "V2 và scorer freeze trước calibration. Không phải audit/test độc lập.", "",
             "| Calibrator | Policy | Nhận đúng / 411 | Recall hợp lệ | Nhận | Lỗi | Risk | Wilson upper | AURC |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        for policy, profile in result["profiles"].items():
            m = profile["crossfit_metrics"]
            risk = f"{m['empirical_risk']:.2%}" if m["empirical_risk"] is not None else "N/A"
            upper = f"{m['wilson_upper_95']:.2%}" if m["wilson_upper_95"] is not None else "N/A"
            lines.append(f"| {name} | {policy} | {m['correct_acceptances']}/411 | {m['valid_acceptance_recall']:.2%} | {m['accepted']} | {m['errors']} | {risk} | {upper} | {result['probability_metrics']['aurc']:.6f} |")
    lines += ["", "Mẫu số 411 = FOUND thật và MAP V2 đúng. Score bổ sung chỉ lấy observable evidence/context.",
              f"Tăng ròng nhận đúng ở budget 5%: {gain}. Không suy từ AURC giảm thành recall chấp nhận tăng.",
              "Ngưỡng chọn trên calibration OOF; số đo chịu lựa chọn ngưỡng, chưa là chứng minh 5% trên test.",
              "Không thay answerability/grounding hoặc calibrator/threshold V2 chính thức. Cần test độc lập mới."]
    (output/"SUMMARY.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
