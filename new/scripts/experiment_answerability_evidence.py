#!/usr/bin/env python3
"""Train-only residual evidence adapter; dev selection; frozen original V2."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json

import torch
import torch.nn.functional as F

from pcrau.answerability_evidence import AnswerabilityEvidenceAdapter, EVIDENCE_NAMES
from pcrau.checkpoint import load_model_checkpoint, save_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.engine import autocast_context, evaluate
from pcrau.losses import class_weights
from pcrau.model import PCRAUTargetV2
from pcrau.utils import atomic_json, load_config, runtime_info, seed_everything, sha256_file, workspace_path
from experiment_answerability_head import answer_metrics, eligible, extract, make_loader, selection_key, write_predictions


def evidence_eligible(metrics, baseline):
    return eligible(metrics, baseline) and all(
        metrics["per_class"][index]["recall"] >= baseline["per_class"][index]["recall"] - 1e-12
        for index in range(4)
    )


def predict(adapter, cache, device, config):
    adapter.eval()
    logits = []
    with torch.no_grad():
        for start, stop in cache["batch_ranges"]:
            with autocast_context(device, config["optimization"]):
                delta = adapter(cache["evidence"][start:stop].to(device)).float()
            logits.append(cache["baseline_logits"][start:stop] + delta.cpu())
    logits = torch.cat(logits)
    metrics = answer_metrics(cache["truth"].tolist(), logits.argmax(-1).tolist())
    metrics["by_variant"] = {}
    for variant in sorted(set(cache["variants"])):
        indices = [i for i, value in enumerate(cache["variants"]) if value == variant]
        metrics["by_variant"][variant] = answer_metrics(cache["truth"][indices].tolist(), logits[indices].argmax(-1).tolist())
    return metrics, logits


def train_batch(adapter, optimizer, indices, cache, device, config, weights):
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, config["optimization"]):
        logits = cache["baseline_logits"][indices].to(device) + adapter(cache["evidence"][indices].to(device)).float()
        loss = F.cross_entropy(logits, cache["truth"][indices].to(device), weight=weights)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite adapter loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
    if not torch.isfinite(norm):
        raise RuntimeError("Non-finite adapter gradient")
    optimizer.step()
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--context", action="store_true", help="include frozen V2 global representation")
    args = parser.parse_args()
    base_root = workspace_path("new/outputs/pcrau_target_v2_full_seed_24082026")
    base_config_path = base_root / "config.json"
    checkpoint = base_root / "checkpoints/best/model.safetensors"
    root = workspace_path("new/outputs") / args.run_name
    if root.parent != workspace_path("new/outputs"):
        raise ValueError("Run name must be one directory name")
    root.mkdir(exist_ok=False)
    base_config = load_config(base_config_path)
    config = deepcopy(base_config)
    config["experiment_id"] = "answerability_evidence_dev_only"
    hidden_dim, dropout = (64, 0.3) if args.context else (32, 0.2)
    config["model"]["answerability_evidence_adapter"] = {"hidden_dim": hidden_dim, "dropout": dropout,
                                                         "include_context": args.context}
    config["experiment"] = {"epochs": 25, "learning_rate": 3e-4, "weight_decay": 1e-3,
                            "trainable_module": "answerability_adapter", "evidence_names": EVIDENCE_NAMES,
                            "initial_checkpoint": str(checkpoint), "calibration_fit": False, "test_evaluated": False}
    atomic_json(root / "config.json", config)
    config_hash = sha256_file(root / "config.json")
    protected = [checkpoint, base_config_path, base_root / "evaluation/calibration/calibrator.json",
                 workspace_path("new/hinhanh/19_answerability_confusion/fig19_answerability_confusion_test_iid.png")]
    hashes = {str(path): sha256_file(path) for path in protected}
    atomic_json(root / "run.json", {"scientific_status": "DEVELOPMENT_ONLY", "runtime": runtime_info(),
                "source_hashes": hashes, "config_sha256": config_hash,
                "protocol_sha256": sha256_file(workspace_path(
                    "new/docs/EXPERIMENT_ANSWERABILITY_CONTEXT.md" if args.context
                    else "new/docs/EXPERIMENT_ANSWERABILITY_EVIDENCE.md")),
                "script_sha256": sha256_file(workspace_path("new/scripts/experiment_answerability_evidence.py")),
                "prior_test_inspection_informed_research_question": True,
                "calibration_accessed_for_fit": False, "test_splits_accessed_for_fit_or_evaluation": False})
    seed_everything(int(config["seed"]))
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ArchivedPCRAUDataset(base_config, "train")
    dev_dataset = ArchivedPCRAUDataset(base_config, "dev")
    train_families = {e["family_id"] for e in train_dataset.entries}
    dev_families = {e["family_id"] for e in dev_dataset.entries}
    assert (len(train_dataset), len(dev_dataset), len(train_families), len(dev_families)) == (1600, 400, 320, 80)
    assert train_families.isdisjoint(dev_families)
    train_loader, _ = make_loader(train_dataset, base_config, False)
    dev_loader, _ = make_loader(dev_dataset, base_config, False)
    _, sampler = make_loader(train_dataset, base_config, True)
    base = PCRAUTargetV2(base_config).to(device)
    metadata = load_model_checkpoint(base, checkpoint, sha256_file(base_config_path))
    base.requires_grad_(False).eval()
    train_cache = extract(base, train_loader, device, base_config, "train", include_evidence=True)
    dev_cache = extract(base, dev_loader, device, base_config, "dev", include_evidence=True)
    context_dim = train_cache["features"].shape[-1] if args.context else 0
    if args.context:
        for cache in (train_cache, dev_cache):
            cache["evidence"] = torch.cat([cache["evidence"], cache["features"].float()], dim=-1)
    baseline = answer_metrics(dev_cache["truth"].tolist(), dev_cache["baseline_logits"].argmax(-1).tolist())
    assert baseline["confusion_matrix"] == metadata["metrics"]["dev"]["answerability"]["confusion_matrix"]
    assert (dev_cache["grounding_hits"], dev_cache["grounding_evaluated"]) == (326, 335)
    model = PCRAUTargetV2(config).to(device)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(name.startswith("answerability_adapter.") for name in missing)
    model.requires_grad_(False).eval()
    answer_weights, source_weights = class_weights(train_dataset, device)
    results = {}
    for arm, weights in (("balanced_evidence", answer_weights), ("unweighted_evidence", None)):
        arm_root = root / arm
        arm_root.mkdir()
        seed_everything(int(config["seed"]))
        adapter = AnswerabilityEvidenceAdapter(hidden_dim, dropout, context_dim).to(device)
        adapter.fit_standardization(train_cache["evidence"])
        initial = deepcopy(adapter.state_dict())
        before_metrics, before_logits = predict(adapter, dev_cache, device, config)
        assert torch.equal(before_logits, dev_cache["baseline_logits"])
        assert before_metrics["confusion_matrix"] == baseline["confusion_matrix"]
        trial = deepcopy(adapter).train()
        trial_optimizer = torch.optim.AdamW(trial.parameters(), lr=3e-4, weight_decay=1e-3)
        smoke_loss = train_batch(trial, trial_optimizer, list(range(20)), train_cache, device, config, weights)
        assert any(not torch.equal(initial[name], value) for name, value in trial.state_dict().items())
        atomic_json(arm_root / "smoke.json", {"status": "PASS", "loss": smoke_loss, "discarded_update": True,
                                             "zero_residual_matches_v2": True})
        # Reset randomness after discarding smoke; both arms share initialization and batch order.
        seed_everything(int(config["seed"]))
        adapter.load_state_dict(initial)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        best, best_epoch, diagnostic, records, step = baseline, -1, None, [], 0
        for epoch in range(25):
            adapter.train()
            sampler.set_epoch(epoch)
            total = 0.0
            for indices in sampler:
                cache_indices = [train_cache["lookup"][train_dataset.entries[i]["sample_id"]] for i in indices]
                total += train_batch(adapter, optimizer, cache_indices, train_cache, device, config, weights) * len(indices)
                step += 1
                scheduler.step()
            metrics, _ = predict(adapter, dev_cache, device, config)
            passed = evidence_eligible(metrics, baseline)
            record = {"epoch": epoch, "global_step": step, "train_ce": total/1600, "dev": metrics, "eligible": passed}
            records.append(record)
            with (arm_root / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            model.answerability_adapter.load_state_dict(adapter.state_dict())
            if diagnostic is None or selection_key(metrics) > selection_key(diagnostic["dev"]):
                diagnostic = record
                save_checkpoint(arm_root / "checkpoints/diagnostic", model, optimizer, scheduler, epoch, step,
                                metrics["macro_f1"], config_hash, record, replace=True)
            if passed and selection_key(metrics) > selection_key(best):
                best, best_epoch = metrics, epoch
                save_checkpoint(arm_root / "checkpoints/best", model, optimizer, scheduler, epoch, step,
                                metrics["macro_f1"], config_hash, record, replace=True)
            print(json.dumps({"arm": arm, "epoch": epoch, "macro_f1": metrics["macro_f1"],
                              "recall": [row["recall"] for row in metrics["per_class"]],
                              "false_found": metrics["false_found"], "eligible": passed}), flush=True)
        selected_name = "best" if best_epoch >= 0 else "diagnostic"
        selected_path = arm_root / "checkpoints" / selected_name / "model.safetensors"
        load_model_checkpoint(model, selected_path, config_hash)
        assert all(torch.equal(value.cpu(), model.state_dict()[name].cpu()) for name, value in base.state_dict().items())
        verified, predictions = evaluate(model, dev_loader, device, config, answer_weights, source_weights)
        selected = best if best_epoch >= 0 else diagnostic["dev"]
        assert verified["answerability"]["confusion_matrix"] == selected["confusion_matrix"]
        cached_metrics, cached_logits = predict(model.answerability_adapter, dev_cache, device, config)
        assert cached_metrics["confusion_matrix"] == verified["answerability"]["confusion_matrix"]
        assert verified["grounding_accuracy"] == 326/335
        baseline_predictions = {row["sample_id"]: row for row in
                                (json.loads(line) for line in (base_root / "evaluation/dev/predictions.jsonl").read_text().splitlines())}
        for row in predictions:
            original = baseline_predictions[row["sample_id"]]
            assert row["spatial"]["map_pixel_xy"] == original["spatial"]["map_pixel_xy"]
            assert row["source_probabilities"] == original["source_probabilities"]
            assert row["relation_edge_probabilities"] == original["relation_edge_probabilities"]
            cache_index = dev_cache["lookup"][row["sample_id"]]
            cached_probability = cached_logits[cache_index].softmax(-1)
            full_probability = cached_probability.new_tensor(list(row["answerability_probabilities"].values()))
            torch.testing.assert_close(cached_probability, full_probability, atol=1e-5, rtol=1e-5)
        atomic_json(arm_root / "full_dev_metrics.json", verified)
        with (arm_root / "full_dev_predictions.jsonl").open("x", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        train_metrics, train_logits = predict(model.answerability_adapter, train_cache, device, config)
        write_predictions(arm_root / "train_predictions.jsonl", train_cache, train_logits)
        write_predictions(arm_root / "dev_predictions.jsonl", dev_cache, cached_logits)
        results[arm] = {"best_epoch": best_epoch, "best_eligible_dev": best,
                        "highest_macro_f1_epoch": diagnostic, "verified_checkpoint": str(selected_path),
                        "verified_train": train_metrics, "verified_dev": selected,
                        "strict_improvement": best_epoch >= 0 and best["macro_f1"] > baseline["macro_f1"]+1e-12,
                        "frozen_v2_parameters_unchanged": True, "grounding_source_edge_predictions_unchanged": True}
    assert hashes == {str(path): sha256_file(path) for path in protected}
    atomic_json(root / "summary.json", {"status": "DEVELOPMENT_ONLY", "baseline_dev": baseline, "arms": results,
                                       "source_hashes_unchanged": True, "test_evaluated": False, "calibration_fit": False})
    lines = ["# Vòng 2: kết quả answerability evidence trên dev", "",
             "Chỉ học residual adapter; giữ nguyên toàn bộ V2. Hàng nhãn thật, cột dự đoán.", "",
             "| Mô hình | Epoch chọn (0-based) | Accuracy | Macro-F1 | Recall FOUND | Recall ABSENT | Recall thiếu evidence | False FOUND |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, epoch, metrics in [("V2", 13, baseline)] + [(arm, value["best_epoch"], value["best_eligible_dev"]) for arm, value in results.items()]:
        recalls = metrics["per_class"]
        lines.append(f"| {name} | {epoch} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.4f} | {recalls[0]['recall']:.2%} | {recalls[2]['recall']:.2%} | {recalls[3]['recall']:.2%} | {metrics['false_found']} |")
    for name, metrics in [("V2", baseline)] + [(arm, value["best_eligible_dev"]) for arm, value in results.items()]:
        lines += ["", f"## Ma trận {name}"]
        lines += ["", "```text", *[" ".join(f"{v:4d}" for v in row) for row in metrics["confusion_matrix"]], "```", ""]
    lines += ["", "## Checkpoint đã kiểm chứng đầy đủ, gồm cả diagnostic", "",
              "| Nhánh | Qua cổng? | Accuracy | Macro-F1 | Recall FOUND | Recall ABSENT | Recall thiếu evidence | False FOUND | ABSENT → FOUND |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, value in results.items():
        metrics = value["verified_dev"]
        recalls = metrics["per_class"]
        lines.append(f"| {arm} | {'Có' if value['best_epoch'] >= 0 else 'Không'} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.4f} | {recalls[0]['recall']:.2%} | {recalls[2]['recall']:.2%} | {recalls[3]['recall']:.2%} | {metrics['false_found']} | {metrics['confusion_matrix'][2][0]} |")
    lines += ["Epoch -1 nghĩa là chưa có checkpoint qua cổng, hàng đó giữ số baseline.",
              "Diagnostic checkpoint chỉ phục vụ phân tích, không tự thay mô hình chính.",
              "Grounding dev 326/335 = 97.31%; source/edge predictions và các tham số V2 bất biến đã kiểm chứng.",
              "Đây là một seed, chọn trên dev; chưa chứng minh cải thiện Test-IID, coverage hoặc risk.",
              "Hướng nghiên cứu chịu thông tin từ Test-IID cũ; cần tập test độc lập mới. Chưa fit calibration."]
    (root / "SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(root), "improvement": {arm:value["strict_improvement"] for arm,value in results.items()}}), flush=True)


if __name__ == "__main__":
    main()
