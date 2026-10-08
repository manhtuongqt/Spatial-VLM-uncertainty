#!/usr/bin/env python3
"""Train/dev-only binary acceptance heads; retain original V2 unchanged."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from pcrau.acceptance_ranker import AcceptanceRanker, acceptance_loss
from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.losses import class_weights
from pcrau.model import PCRAUTargetV2
from pcrau.selective_experiment import apply_risk, choose_policy_threshold, fit_risk, policy_result, probability_metrics
from pcrau.utils import atomic_json, load_config, seed_everything, sha256_file, workspace_path
from experiment_answerability_head import make_loader
from experiment_hard_cases_selective import capture


def assess(rows, scores, reference_count=None):
    error = np.array([r["evaluation"]["error_event"] for r in rows])
    threshold = choose_policy_threshold(scores, error, [r["family_id"] for r in rows],
                                        np.ones(len(rows), bool), 0.05, 20)
    metrics, _ = policy_result(rows, scores, "risk_only", threshold["threshold"])
    metrics["valid_acceptance_recall"] = metrics["correct_acceptances"]/metrics["correct_found_total"]
    result = {"threshold_selection": threshold, "policy": metrics, "score_metrics": probability_metrics(scores, error)}
    if reference_count is not None:
        order = np.argsort(scores, kind="stable")[:reference_count]
        result["same_accepted_count"] = {"accepted": len(order), "errors": int(error[order].sum()),
                                        "correct": int((~error[order]).sum())}
    return result


def key(result):
    metrics = result["policy"]
    return metrics["correct_acceptances"], -metrics["errors"], -result["score_metrics"]["aurc"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--residual", action="store_true", help="zero-initialized correction of frozen V2 error logit")
    args = parser.parse_args()
    root = workspace_path("new/outputs")/args.run_name
    if root.parent != workspace_path("new/outputs"):
        raise ValueError("Run name must be one directory name")
    root.mkdir(exist_ok=False)
    source = workspace_path("new/outputs/pcrau_target_v2_full_seed_24082026")
    checkpoint, config_path = source/"checkpoints/best/model.safetensors", source/"config.json"
    protected = [checkpoint, config_path, source/"evaluation/calibration/calibrator.json",
                 workspace_path("new/hinhanh/19_answerability_confusion/fig19_answerability_confusion_test_iid.png")]
    hashes = {str(p): sha256_file(p) for p in protected}
    config = load_config(config_path)
    config["model"]["export_answerability_evidence"] = True
    atomic_json(root/"run.json", {"status": "DEVELOPMENT_ONLY", "source_hashes": hashes,
                "calibration_accessed": False, "test_accessed": False,
                "protocol_sha256": sha256_file(workspace_path("new/docs/EXPERIMENT_ACCEPTANCE_RESIDUAL.md" if args.residual
                                                              else "new/docs/EXPERIMENT_ACCEPTANCE_RECALL.md")),
                "residual": args.residual, "residual_penalty": .05 if args.residual else 0.,
                "script_sha256": sha256_file(Path(__file__)), "seed": 24082026})
    atomic_json(root/"config.json", config)
    seed_everything(24082026)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, dev = ArchivedPCRAUDataset(config, "train"), ArchivedPCRAUDataset(config, "dev")
    assert (len(train), len(dev)) == (1600, 400)
    assert {e["family_id"] for e in train.entries}.isdisjoint(e["family_id"] for e in dev.entries)
    tr_loader, _ = make_loader(train, config, False)
    de_loader, _ = make_loader(dev, config, False)
    _, sampler = make_loader(train, config, True)
    base = PCRAUTargetV2(config).to(device).eval().requires_grad_(False)
    load_model_checkpoint(base, checkpoint, sha256_file(config_path))
    base_state = {name: value.detach().cpu().clone() for name, value in base.state_dict().items()}
    weights, source_weights = class_weights(train, device)
    print("EXTRACT train/dev frozen observable features", flush=True)
    _, tr_rows, tr_cache = capture(base, tr_loader, train, device, config, weights, source_weights, root/"baseline_train")
    _, de_rows, de_cache = capture(base, de_loader, dev, device, config, weights, source_weights, root/"baseline_dev")
    errors = torch.tensor([r["evaluation"]["error_event"] for r in tr_rows], dtype=torch.float32, device=device)
    assert sum(not r["evaluation"]["error_event"] for r in de_rows) == 164
    raw = np.array([1-r["answerability_probabilities"]["FOUND"] for r in de_rows])
    # This stable logit is exactly log((1-p_FOUND)/p_FOUND), with no truth input.
    tr_base = (torch.logsumexp(tr_cache["baseline_logits"][:, 1:], dim=-1)
               - tr_cache["baseline_logits"][:, 0]).to(device)
    de_base = (torch.logsumexp(de_cache["baseline_logits"][:, 1:], dim=-1)
               - de_cache["baseline_logits"][:, 0]).to(device)
    baselines = {"raw_v2": assess(de_rows, raw)}
    for name, extended, l2 in [("train_logistic18", False, .001), ("train_logistic33", True, .01)]:
        calibrator, _ = fit_risk(tr_rows, extended, l2)
        atomic_json(root/f"{name}.json", calibrator)
        baselines[name] = assess(de_rows, apply_risk(de_rows, calibrator))
    reference_name = max(baselines, key=lambda n: key(baselines[n]))
    reference = baselines[reference_name]
    atomic_json(root/"baselines.json", baselines)
    print(json.dumps({"reference": reference_name, "policy": reference["policy"]}), flush=True)
    outcomes = {}
    selected = None
    for arm, context, rank_weight in [("evidence_bce", False, 0.0), ("context_bce", True, 0.0),
                                       ("context_bce_rank", True, 0.1)]:
        seed_everything(24082026)
        arm_root = root/arm
        arm_root.mkdir()
        dim = 794 if context else 26
        tr_x = tr_cache["evidence"][:, :dim].float().to(device)
        de_x = de_cache["evidence"][:, :dim].float().to(device)
        head = AcceptanceRanker(dim, zero_output=args.residual).to(device)
        head.fit_standardization(tr_x)
        if args.residual:
            with torch.no_grad():
                assert torch.equal(head(de_x), torch.zeros(len(de_rows), device=device))
                np.testing.assert_allclose((head(de_x)+de_base).sigmoid().cpu().numpy(), raw, atol=1e-6)
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-3)
        initial = deepcopy(head.state_dict())
        smoke_head = deepcopy(head)
        smoke_opt = torch.optim.AdamW(smoke_head.parameters(), lr=3e-4)
        smoke_delta = smoke_head(tr_x[:20])
        smoke_logits = smoke_delta+tr_base[:20] if args.residual else smoke_delta
        smoke_loss = acceptance_loss(smoke_logits, errors[:20], rank_weight)
        if args.residual:
            smoke_loss = smoke_loss + .05*smoke_delta.square().mean()
        smoke_loss.backward()
        assert torch.isfinite(smoke_loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in smoke_head.parameters())
        smoke_opt.step()
        assert any(not torch.equal(initial[n], v) for n, v in smoke_head.state_dict().items())
        atomic_json(arm_root/"smoke.json", {"status": "PASS", "loss": float(smoke_loss.detach()), "discarded_update": True})
        seed_everything(24082026)
        best, best_state, best_scores = None, None, None
        for epoch in range(25):
            head.train()
            sampler.set_epoch(epoch)
            total = 0.0
            for indices in sampler:
                ids = [tr_cache["lookup"][train.entries[i]["sample_id"]] for i in indices]
                optimizer.zero_grad(set_to_none=True)
                delta = head(tr_x[ids])
                logits = delta+tr_base[ids] if args.residual else delta
                loss = acceptance_loss(logits, errors[ids], rank_weight)
                if args.residual:
                    loss = loss + .05*delta.square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite acceptance loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                if not torch.isfinite(norm):
                    raise ValueError("Non-finite gradient")
                optimizer.step()
                total += float(loss.detach())*len(ids)
            record = {"epoch": epoch, "train_loss": total/1600}
            if (epoch+1) % 5 == 0:
                head.eval()
                with torch.no_grad():
                    logits = head(de_x)+(de_base if args.residual else 0)
                    scores = logits.sigmoid().cpu().numpy()
                result = assess(de_rows, scores, baselines["raw_v2"]["policy"]["accepted"])
                record["dev"] = result
                if best is None or key(result) > key(best["dev"]):
                    best, best_state, best_scores = deepcopy(record), deepcopy(head.state_dict()), scores.copy()
                print(json.dumps({"arm": arm, "epoch": epoch, "accepted": result["policy"]["accepted"],
                                  "correct": result["policy"]["correct_acceptances"], "errors": result["policy"]["errors"],
                                  "aurc": result["score_metrics"]["aurc"]}), flush=True)
            with (arm_root/"history.jsonl").open("a") as handle:
                handle.write(json.dumps(record)+"\n")
        metrics = best["dev"]["policy"]
        passed = (metrics["correct_acceptances"] > reference["policy"]["correct_acceptances"]
                  and metrics["errors"] <= reference["policy"]["errors"]
                  and metrics["accepted_absent_truth"] == 0
                  and best["dev"]["score_metrics"]["aurc"] <= baselines["raw_v2"]["score_metrics"]["aurc"])
        best["eligible"] = passed
        head.load_state_dict(best_state)
        torch.save({"state_dict": {n:v.cpu() for n,v in best_state.items()}, "input_dim": dim,
                    "hidden_dim": 64, "dropout": .3, "calibrated": False,
                    "residual": args.residual,
                    "source_v2_sha256": hashes[str(checkpoint)], "selection": best}, arm_root/"diagnostic.pt")
        head.eval()
        with torch.no_grad():
            np.testing.assert_allclose((head(de_x)+(de_base if args.residual else 0)).sigmoid().cpu().numpy(), best_scores, atol=1e-7)
        with (arm_root/"dev_scores.jsonl").open("x") as handle:
            for row, score in zip(de_rows, best_scores):
                handle.write(json.dumps({"sample_id": row["sample_id"], "family_id": row["family_id"],
                                        "raw_error_score": float(score), "evaluation": row["evaluation"]})+"\n")
        outcomes[arm] = best
        if passed and (selected is None or key(best["dev"]) > key(outcomes[selected]["dev"])):
            selected = arm
    assert all(torch.equal(v, base.state_dict()[n].cpu()) for n, v in base_state.items())
    assert hashes == {str(p): sha256_file(p) for p in protected}
    summary = {"status": "DEVELOPMENT_ONLY", "baselines": baselines, "reference": reference_name,
               "residual": args.residual,
               "arms": outcomes, "selected": selected, "source_hashes_unchanged": True,
               "calibration_fit": False, "test_evaluated": False, "robot_motion": False,
               "answerability_grounding_source_edge_unchanged": True,
               "dev_thresholds_are_selection_not_independent_evaluation": True}
    atomic_json(root/"summary.json", summary)
    lines = ["# Binary acceptance recall trên train/dev", "", "Chỉ kết quả phát triển; chưa hiệu chuẩn hoặc test độc lập.", "",
             "| Nhánh | Epoch (0-based) | Nhận đúng / 164 | Recall hợp lệ | Tổng nhận | Lỗi | AURC | Qua cổng |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name, result in baselines.items():
        m = result["policy"]
        lines.append(f"| {name} | — | {m['correct_acceptances']}/164 | {m['valid_acceptance_recall']:.2%} | {m['accepted']} | {m['errors']} | {result['score_metrics']['aurc']:.6f} | Đối chứng |")
    for name, record in outcomes.items():
        result, m = record["dev"], record["dev"]["policy"]
        lines.append(f"| {name} | {record['epoch']} | {m['correct_acceptances']}/164 | {m['valid_acceptance_recall']:.2%} | {m['accepted']} | {m['errors']} | {result['score_metrics']['aurc']:.6f} | {record['eligible']} |")
    lines += ["", f"Ứng viên được chọn: {selected or 'Không có; giữ V2' }.", "",
              "Ngưỡng quét/chọn trên dev, Wilson chưa bảo đảm family hoặc robot. Các score chưa hiệu chuẩn.",
              "Đối chứng logistic chỉ fit train; không so trực tiếp số này với calibrator fit calibration.",
              "Không dùng 411/150 calibration hoặc các ca Test-IID để train. Không tuyên bố cải thiện 261/411.",
              "Mọi tham số V2, checkpoint/config/calibrator gốc và hình 19 bất biến đã kiểm tra.",
              "Macro-F1 answerability không đổi. Scorer là nhánh riêng, chưa thay policy chính."]
    (root/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"output": str(root), "selected": selected}), flush=True)


if __name__ == "__main__":
    main()
