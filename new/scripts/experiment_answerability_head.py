#!/usr/bin/env python3
"""Controlled train/dev-only answerability head fine-tuning on frozen V2."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pcrau.checkpoint import load_model_checkpoint, save_checkpoint
from pcrau.answerability_evidence import observable_answerability_evidence
from pcrau.dataset import ANSWER_CLASSES, ArchivedPCRAUDataset, FamilyBatchSampler, collate_samples, move_model_batch
from pcrau.engine import autocast_context, evaluate
from pcrau.losses import class_weights
from pcrau.metrics import confusion_metrics
from pcrau.model import PCRAUTargetV2
from pcrau.utils import atomic_json, load_config, runtime_info, seed_everything, sha256_file, workspace_path


def make_loader(dataset, config, shuffle):
    sampler = FamilyBatchSampler(dataset, int(config["optimization"]["families_per_batch"]),
                                 int(config["seed"]), shuffle)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=collate_samples), sampler


def answer_metrics(truth, prediction):
    metrics = confusion_metrics(truth, prediction, len(ANSWER_CLASSES))
    matrix = np.asarray(metrics["confusion_matrix"])
    metrics["false_found"] = int(matrix[1:, 0].sum())
    metrics["missed_found"] = int(matrix[0, 1:].sum())
    return metrics


def eligible(metrics, baseline):
    return (
        metrics["macro_f1"] >= baseline["macro_f1"] - 1e-12
        and metrics["per_class"][0]["recall"] >= baseline["per_class"][0]["recall"] - 1e-12
        and metrics["false_found"] <= baseline["false_found"]
    )


def selection_key(metrics):
    return (metrics["macro_f1"], -metrics["false_found"], metrics["per_class"][0]["recall"])


def predict(head, cache, device, config):
    head.eval()
    outputs = []
    with torch.no_grad():
        for start, stop in cache["batch_ranges"]:
            with autocast_context(device, config["optimization"]):
                outputs.append(head(cache["features"][start:stop].to(device)).float().cpu())
    logits = torch.cat(outputs)
    metrics = answer_metrics(cache["truth"].tolist(), logits.argmax(-1).tolist())
    by_variant = {}
    for variant in sorted(set(cache["variants"])):
        indices = [i for i, value in enumerate(cache["variants"]) if value == variant]
        by_variant[variant] = answer_metrics(cache["truth"][indices].tolist(), logits[indices].argmax(-1).tolist())
    metrics["by_variant"] = by_variant
    metrics["unweighted_ce"] = float(F.cross_entropy(logits, cache["truth"]))
    return metrics, logits


def extract(model, loader, device, config, split, include_evidence=False):
    model.eval()
    inputs, logits, truth, sample_ids, families, variants = [], [], [], [], [], []
    ranges, lookup = [], {}
    count, hit, evaluated = 0, 0, 0
    evidence = []
    handle = model.answer_head.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().cpu()))
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                with autocast_context(device, config["optimization"]):
                    model_batch = move_model_batch(batch, device)
                    output = model(model_batch)
                if include_evidence:
                    evidence.append(observable_answerability_evidence(output, model_batch).cpu())
                logits.append(output["answerability_logits"].float().cpu())
                truth.append(batch["answer_target"])
                size = len(batch["sample_id"])
                ranges.append((count, count + size))
                for offset, sample_id in enumerate(batch["sample_id"]):
                    lookup[sample_id] = count + offset
                    if bool(batch["target_loss_mask"][offset]):
                        width = output["target_logits"].shape[-1]
                        cell = int(output["target_logits"][offset].flatten().argmax())
                        x, y = (cell % width) * 20 + 10, (cell // width) * 20 + 10
                        hit += int(batch["target_full"][offset, y, x])
                        evaluated += 1
                sample_ids.extend(batch["sample_id"])
                families.extend(batch["family_id"])
                variants.extend(batch["variant"])
                count += size
                if (batch_index + 1) % 10 == 0:
                    print(json.dumps({"phase": "extract", "split": split, "samples": count}), flush=True)
    finally:
        handle.remove()
    result = {
        "features": torch.cat(inputs), "baseline_logits": torch.cat(logits), "truth": torch.cat(truth),
        "sample_ids": sample_ids, "families": families, "variants": variants,
        "lookup": lookup, "batch_ranges": ranges, "grounding_hits": hit, "grounding_evaluated": evaluated,
    }
    if include_evidence:
        result["evidence"] = torch.cat(evidence)
    return result


def write_predictions(path, cache, logits):
    probability = logits.softmax(-1).tolist()
    with path.open("x", encoding="utf-8") as handle:
        for index, sample_id in enumerate(cache["sample_ids"]):
            row = {
                "sample_id": sample_id, "family_id": cache["families"][index], "variant": cache["variants"][index],
                "answerability_probabilities": dict(zip(ANSWER_CLASSES, probability[index])),
                "predicted": ANSWER_CLASSES[int(logits[index].argmax())],
                "evaluation": {"answerability_state": ANSWER_CLASSES[int(cache["truth"][index])]},
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def smoke(head, train_cache, weights, device, config):
    trial = deepcopy(head).to(device).train()
    for parameter in trial.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trial.parameters(), lr=1e-4)
    start, stop = train_cache["batch_ranges"][0]
    before = {name: value.detach().clone() for name, value in trial.state_dict().items()}
    with autocast_context(device, config["optimization"]):
        logits = trial(train_cache["features"][start:stop].to(device))
        loss = F.cross_entropy(logits.float(), train_cache["truth"][start:stop].to(device), weight=weights)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trial.parameters())
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in trial.state_dict().items())
    return {"status": "PASS", "samples": stop - start, "loss": float(loss.detach()), "discarded_update": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    base_root = workspace_path("new/outputs/pcrau_target_v2_full_seed_24082026")
    checkpoint = base_root / "checkpoints/best/model.safetensors"
    config_file = base_root / "config.json"
    root = workspace_path("new/outputs") / args.run_name
    if root.parent != workspace_path("new/outputs"):
        raise ValueError("run-name must be a single directory name")
    root.mkdir(exist_ok=False)
    config = load_config(config_file)
    config["experiment_id"] = "answerability_head_dev_only"
    config["experiment"] = {
        "initial_checkpoint": str(checkpoint), "epochs": args.epochs, "learning_rate": 1e-4,
        "trainable_module": "answer_head", "arms": ["balanced_control", "unweighted_candidate"],
        "selection": "dev macro-F1 with baseline recall FOUND and false FOUND gates",
        "calibration_accessed": False, "test_splits_accessed": False,
    }
    atomic_json(root / "config.json", config)
    config_hash = sha256_file(root / "config.json")
    # Hash preservation checks read bytes only, never use test labels or predictions for fitting.
    protected = [checkpoint, config_file, base_root / "evaluation/calibration/calibrator.json",
                 workspace_path("new/hinhanh/19_answerability_confusion/fig19_answerability_confusion_test_iid.png")]
    before = {str(path): sha256_file(path) for path in protected}
    atomic_json(root / "run.json", {
        "scientific_status": "DEVELOPMENT_ONLY", "runtime": runtime_info(), "config_sha256": config_hash,
        "source_hashes": before, "manifest_sha256": sha256_file(workspace_path(config["paths"]["development_manifest"])),
        "feature_index_sha256": sha256_file(workspace_path(config["paths"]["development_feature_index"])),
        "prior_test_inspection_informed_research_question": True,
        "calibration_accessed_for_fit": False, "test_splits_accessed_for_fit_or_evaluation": False,
    })
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    train_dataset = ArchivedPCRAUDataset(config, "train")
    dev_dataset = ArchivedPCRAUDataset(config, "dev")
    train_families = {entry["family_id"] for entry in train_dataset.entries}
    dev_families = {entry["family_id"] for entry in dev_dataset.entries}
    assert len(train_dataset) == 1600 and len(dev_dataset) == 400
    assert len(train_families) == 320 and len(dev_families) == 80 and train_families.isdisjoint(dev_families)
    audit = {}
    for split, dataset in (("train", train_dataset), ("dev", dev_dataset)):
        audit[split] = {"samples": len(dataset), "states": dict(Counter(entry["supervision"]["answerability_state"] for entry in dataset.entries))}
        by_variant = defaultdict(Counter)
        for entry in dataset.entries:
            supervision = entry["supervision"]
            assert ANSWER_CLASSES[supervision["answerability_index"]] == supervision["answerability_state"]
            by_variant[entry["variant"]][supervision["answerability_state"]] += 1
        audit[split]["states_by_variant"] = {key: dict(value) for key, value in by_variant.items()}
    atomic_json(root / "data_audit.json", audit)
    train_loader, _ = make_loader(train_dataset, config, False)
    dev_loader, _ = make_loader(dev_dataset, config, False)
    _, sampler = make_loader(train_dataset, config, True)
    model = PCRAUTargetV2(config).to(device)
    source_metadata = load_model_checkpoint(model, checkpoint, sha256_file(config_file))
    assert source_metadata["epoch"] == 13 and source_metadata["global_step"] == 1120
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    initial_head = deepcopy(model.answer_head)
    frozen = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if not name.startswith("answer_head.")}
    train_cache = extract(model, train_loader, device, config, "train")
    dev_cache = extract(model, dev_loader, device, config, "dev")
    baseline, cache_logits = predict(initial_head, dev_cache, device, config)
    torch.testing.assert_close(cache_logits, dev_cache["baseline_logits"], atol=1e-5, rtol=1e-5)
    assert baseline["confusion_matrix"] == source_metadata["metrics"]["dev"]["answerability"]["confusion_matrix"]
    assert (dev_cache["grounding_hits"], dev_cache["grounding_evaluated"]) == (326, 335)
    baseline_train, _ = predict(initial_head, train_cache, device, config)
    atomic_json(root / "baseline.json", {"train": baseline_train, "dev": baseline})
    weights, _ = class_weights(train_dataset, device)
    atomic_json(root / "smoke.json", {
        "balanced_control": smoke(initial_head, train_cache, weights, device, config),
        "unweighted_candidate": smoke(initial_head, train_cache, None, device, config),
        "cached_head_matches_full_inference": True,
    })
    results = {}
    for arm, arm_weights in (("balanced_control", weights), ("unweighted_candidate", None)):
        arm_root = root / arm
        arm_root.mkdir()
        seed_everything(int(config["seed"]))
        head = deepcopy(initial_head).to(device)
        for parameter in head.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        best, best_epoch, best_head, global_step = baseline, -1, None, 0
        records = []
        for epoch in range(args.epochs):
            head.train()
            sampler.set_epoch(epoch)
            train_loss = 0.0
            for batch_indices in sampler:
                indices = [train_cache["lookup"][train_dataset.entries[i]["sample_id"]] for i in batch_indices]
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, config["optimization"]):
                    logits = head(train_cache["features"][indices].to(device))
                    loss = F.cross_entropy(logits.float(), train_cache["truth"][indices].to(device), weight=arm_weights)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite head loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("Non-finite head gradient")
                optimizer.step()
                scheduler.step()
                global_step += 1
                train_loss += float(loss.detach()) * len(indices)
            dev_metrics, dev_logits = predict(head, dev_cache, device, config)
            improved = eligible(dev_metrics, baseline) and selection_key(dev_metrics) > selection_key(best)
            record = {"epoch": epoch, "global_step": global_step, "train_ce": train_loss / len(train_dataset),
                      "dev": dev_metrics, "eligible": eligible(dev_metrics, baseline), "improved": improved}
            records.append(record)
            with (arm_root / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if improved:
                best, best_epoch, best_head = dev_metrics, epoch, deepcopy(head.state_dict())
                model.answer_head.load_state_dict(best_head)
                save_checkpoint(arm_root / "checkpoints/best", model, optimizer, scheduler, epoch, global_step,
                                best["macro_f1"], config_hash, record, replace=True)
            print(json.dumps({"arm": arm, "epoch": epoch, "macro_f1": dev_metrics["macro_f1"],
                              "found_recall": dev_metrics["per_class"][0]["recall"],
                              "false_found": dev_metrics["false_found"], "eligible": record["eligible"]}), flush=True)
        final_metrics, _ = predict(head, dev_cache, device, config)
        results[arm] = {"best_epoch": best_epoch, "best_dev": best, "last_dev": final_metrics,
                        "highest_macro_f1_epoch": max(records, key=lambda row: selection_key(row["dev"])),
                        "highest_found_recall_epoch": max(records, key=lambda row: row["dev"]["per_class"][0]["recall"]),
                        "strict_macro_f1_improvement": best["macro_f1"] > baseline["macro_f1"] + 1e-12}
        if best_head is not None:
            head.load_state_dict(best_head)
            model.answer_head.load_state_dict(best_head)
            chosen_train, train_logits = predict(head, train_cache, device, config)
            chosen_dev, dev_logits = predict(head, dev_cache, device, config)
            results[arm]["best_train"] = chosen_train
            write_predictions(arm_root / "train_predictions.jsonl", train_cache, train_logits)
            write_predictions(arm_root / "dev_predictions.jsonl", dev_cache, dev_logits)
            answer_weights, source_weights = class_weights(train_dataset, device)
            verified, full_predictions = evaluate(model, dev_loader, device, config, answer_weights, source_weights)
            assert verified["answerability"]["confusion_matrix"] == chosen_dev["confusion_matrix"]
            assert verified["grounding_accuracy"] == 326 / 335
            atomic_json(arm_root / "full_dev_metrics.json", verified)
            with (arm_root / "full_dev_predictions.jsonl").open("x", encoding="utf-8") as handle:
                for prediction in full_predictions:
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        assert all(torch.equal(frozen[name], value.detach().cpu()) for name, value in model.state_dict().items() if name in frozen)
    after = {str(path): sha256_file(path) for path in protected}
    assert before == after
    summary = {"status": "DEVELOPMENT_ONLY", "baseline_dev": baseline, "arms": results,
               "source_hashes_unchanged": True, "non_answerability_parameters_unchanged": True,
               "dev_grounding": {"hits": 326, "evaluated": 335},
               "calibration_fit": False, "test_evaluated": False, "official_figure_updated": False}
    atomic_json(root / "summary.json", summary)
    lines = ["# Kết quả thử nghiệm answerability — chỉ train/dev", "",
             "Khởi tạo best V2 epoch 13/step 1120; chỉ fine-tune answer_head. Không thay kết quả khóa luận.", "",
             "| Mô hình | Epoch chọn (0-based) | Accuracy dev | Macro-F1 | Recall FOUND | False FOUND | Missed FOUND |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, epoch, metrics in [("V2", 13, baseline)] + [(arm, value["best_epoch"], value["best_dev"]) for arm, value in results.items()]:
        lines.append(f"| {name} | {epoch} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.4f} | {metrics['per_class'][0]['recall']:.2%} | {metrics['false_found']} | {metrics['missed_found']} |")
    lines += ["", "Epoch -1 nghĩa là không có checkpoint qua cổng tốt hơn baseline; số trong hàng đó là baseline.",
              "Grounding dev giữ 326/335 = 97.31%; tham số ngoài answer_head và hash nguồn không thay đổi.",
              "Các ma trận có hàng là nhãn thật, cột là dự đoán; thứ tự FOUND, AMBIGUOUS, ABSENT, INSUFFICIENT_EVIDENCE."]
    lines += ["", "## Các epoch tốt về macro-F1 nhưng chưa qua đủ cổng", "",
              "| Nhánh | Epoch (0-based) | Accuracy dev | Macro-F1 | Recall FOUND | False FOUND | Missed FOUND |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, value in results.items():
        record = value["highest_macro_f1_epoch"]
        metrics = record["dev"]
        lines.append(f"| {arm} | {record['epoch']} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.4f} | {metrics['per_class'][0]['recall']:.2%} | {metrics['false_found']} | {metrics['missed_found']} |")
    lines += ["", "Mỗi nhánh được thử đúng ngân sách đã định; không tự tăng epoch hoặc thay tiêu chí khi chưa đạt.",
              f"Accuracy baseline train/dev: {baseline_train['accuracy']:.2%}/{baseline['accuracy']:.2%}. Chênh lệch cho thấy giới hạn khái quát hiện tại; chưa xác lập nguyên nhân riêng cho weighting hoặc biểu diễn."]
    for name, metrics in [("V2", baseline)] + [(arm, value["best_dev"]) for arm, value in results.items()]:
        lines += ["", f"## {name}", "", "```text", *[" ".join(f"{v:4d}" for v in row) for row in metrics["confusion_matrix"]], "```"]
    lines += ["", "Đây là kết quả chọn trên dev, một seed; không chứng minh cải thiện trên test.",
              "Nghiên cứu đã xem lỗi Test-IID cũ: cần tập đánh giá độc lập mới cho vòng này.",
              "Chưa fit calibration, chưa đo risk/coverage của head mới; hình 19 và LaTeX giữ nguyên."]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(root), "arms": {name: value["strict_macro_f1_improvement"] for name, value in results.items()}}), flush=True)


if __name__ == "__main__":
    main()
