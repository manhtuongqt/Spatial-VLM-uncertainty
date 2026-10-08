#!/usr/bin/env python3
"""Explicitly authorized, locked M1/M2 pilot on development train/dev only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcrau.anchor_shadow import AnchorShadow, SHADOW_VERSION
from pcrau.anchor_shadow_experiment import PROTOCOL, ShadowPilotRunner, family_batches, observable_batch, pilot_presentations, state_digest
from pcrau.anchor_shadow_pilot import aggregate, family_bootstrap, locked_gates, pair_rows, paired_changes, score_predictions, spatial_summary
from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.engine import autocast_context
from pcrau.model import PCRAUTargetV2
from pcrau.query_parser import PARSER_VERSION
from pcrau.utils import atomic_json, read_json, runtime_info, seed_everything, sha256_file, workspace_path


def jsonl(path, rows):
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


class FeatureCache:
    """Read-only cached observable tensors; exposes no supervision loader."""
    def __init__(self, dataset, protect):
        self.dataset = dataset
        self.split, self.profile, self.model_cfg = dataset.split, dataset.profile, dataset.model_cfg
        self.cache = {}
        self.protect = protect

    def _feature(self, entry):
        key = self.dataset.sample_to_feature[entry["sample_id"]]
        desc = self.dataset.features[key]
        if desc["rgb_sha256"] != entry["feature_input"]["rgb_sha256"] or desc["depth_sha256"] != entry["feature_input"]["depth_sha256"]:
            raise ValueError("Cached visual input provenance mismatch")
        if key not in self.cache:
            self.protect(self.dataset.layout.feature_root / desc["path"], desc["sha256"])
            self.cache[key] = self.dataset._feature(entry)
        return self.cache[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="new/outputs/pcrau_s1_anchor_shadow_pilot_20261005")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    output = workspace_path(args.output)
    if workspace_path("new/outputs") not in output.parents:
        raise ValueError("Pilot outputs must stay under new/outputs")
    output.mkdir(exist_ok=False)
    protected = {}

    def protect(path, expected=None):
        path = Path(path).resolve()
        if str(path) not in protected:
            protected[str(path)] = sha256_file(path)
        if expected is not None and protected[str(path)] != expected:
            raise ValueError(f"Protected hash mismatch: {path}")
        return protected[str(path)]

    pre_root = workspace_path("new/outputs/pcrau_s1_anchor_shadow_preflight_20261005_r3")
    receipt = read_json(pre_root / "summary.json")
    pre_provenance = read_json(pre_root / "provenance.json")
    protect(pre_root / "summary.json"); protect(pre_root / "provenance.json")
    for path, digest in pre_provenance["source_sha256"].items():
        protect(path, digest)
    active_path = workspace_path("new/outputs/active_experimental_profile.json")
    active = read_json(active_path); protect(active_path)
    for key in ("config", "checkpoint", "calibrator", "profiles"):
        protect(active[key], active[key + "_sha256"])
    config = read_json(Path(active["config"]))
    protocol_path = workspace_path("plan/S1_PHRASE_CONDITIONED_ANCHOR_EXPERIMENT_20261005.md")
    protocol_hash = protect(protocol_path, receipt["protocol_sha256"])
    for path in workspace_path("new/src/pcrau").glob("*.py"):
        protect(path)
    protect(__file__)
    seed_everything(PROTOCOL.seed); torch.set_num_threads(4)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    baseline = PCRAUTargetV2(config).to(device).eval()
    load_model_checkpoint(baseline, active["checkpoint"], active["config_sha256"])
    arms = {"M1": AnchorShadow(baseline, "phrase", PROTOCOL.seed), "M2": AnchorShadow(baseline, "whole_text", PROTOCOL.seed)}
    runners = {key: ShadowPilotRunner(arm) for key, arm in arms.items()}
    frozen_digest = state_digest(baseline.state_dict())
    assert frozen_digest == receipt["frozen_model_state_sha256"]
    assert all(state_digest(arm.residual.state_dict()) == receipt["initialization_sha256"][key] for key, arm in arms.items())
    assert all(sum(p.numel() for p in arm.parameters() if p.requires_grad) == 33024 for arm in arms.values())
    hook_counts = {key: len(m._forward_hooks) for key, m in baseline.named_modules()}
    datasets = {split: ArchivedPCRAUDataset(config, split, verify_feature_hash=True) for split in ("train", "dev")}
    caches = {split: FeatureCache(dataset, protect) for split, dataset in datasets.items()}
    layout = datasets["train"].layout
    protect(layout.manifest); protect(layout.feature_index)
    manifest = read_json(layout.manifest)
    train = pilot_presentations(manifest["entries"], config["model"], "train")
    dev = pilot_presentations(manifest["entries"], config["model"], "dev")
    assert len(train) == 1024 and len(dev) == 64
    assert len({p["family_id"] for p in train}) == 64 and len({p["family_id"] for p in dev}) == 16
    assert not {p["family_id"] for p in train} & {p["family_id"] for p in dev}
    original_train = [p for p in train if p["prefix_index"] == 0]
    all_dev = [{"entry": e, "prompt": e["feature_input"]["prompt"], "sample_id": e["sample_id"],
                "family_id": e["family_id"], "prefix_index": 0} for e in datasets["dev"].entries]
    locked = {"protocol": asdict(PROTOCOL), "design_sha256": protocol_hash,
        "preflight_receipt_sha256": sha256_file(pre_root / "summary.json"), "baseline_bundle": active,
        "initialization_sha256": receipt["initialization_sha256"], "runtime": runtime_info(), "device": str(device),
        "precision": "MLP/loss FP32; frozen extractor/head config AMP bfloat16 on CUDA",
        "arm_order_each_epoch": ["M1", "M2"], "trainable_per_arm": 33024}
    atomic_json(output / "locked_protocol.json", locked)
    atomic_json(output / "split_manifest.json", {split: [{"sample_id": p["sample_id"], "family_id": p["family_id"],
        "prefix_index": p["prefix_index"], "prompt": p["prompt"]} for p in ps] for split, ps in (("train", train), ("dev", dev))})
    plans = {str(epoch): family_batches(train, epoch) for epoch in range(PROTOCOL.max_epochs)}
    assert all(len(plan) == 16 and all(len(batch) == 64 for batch in plan) for plan in plans.values())
    atomic_json(output / "family_batch_plans.json", plans)

    def batch(items):
        split = items[0]["entry"]["split"]
        return observable_batch(caches[split], [p["entry"] for p in items], [p["prompt"] for p in items], device)

    def capture(items):
        with autocast_context(device, config["optimization"]):
            return arms["M1"].capture(batch(items), [p["prompt"] for p in items])

    def predict_items(items, verify=False):
        predictions, raw = [], {}
        for start in range(0, len(items), 8):
            subset = items[start:start+8]; inputs = batch(subset)
            with torch.no_grad(), autocast_context(device, config["optimization"]):
                cap = arms["M1"].capture(inputs, [p["prompt"] for p in subset])
                maps = {key: arm.predict(cap) for key, arm in arms.items()}
                if verify:
                    plain = baseline(inputs)
                    assert all(torch.equal(v, cap.baseline[key]) for key, v in plain.items())
                    other = arms["M2"].capture(inputs, [p["prompt"] for p in subset])
                    assert all(torch.equal(v, other.baseline[key]) for key, v in plain.items())
                    inactive = ~cap.active_slots
                    assert all(torch.equal(v[inactive], plain["anchor_logits"][inactive]) for v in maps.values())
            values = {"M0": cap.baseline["anchor_logits"].float().cpu().numpy(),
                      **{key: value.float().cpu().numpy() for key, value in maps.items()}}
            target = cap.baseline["target_logits"].float().cpu().numpy()
            for index, item in enumerate(subset):
                entry = item["entry"]
                row = {"sample_id": item["sample_id"], "family_id": item["family_id"], "split": entry["split"],
                    "prompt": item["prompt"], "parsed": cap.queries[index].to_dict(), "scope_status": cap.scope_status[index],
                    "feature_key": datasets[entry["split"]].sample_to_feature[item["sample_id"]],
                    **{key: {"anchor": spatial_summary(value[index, 0])} for key, value in values.items()}}
                row["M0"]["target"] = spatial_summary(target[index])
                row["M0"]["answerability_logits"] = cap.baseline["answerability_logits"][index].float().cpu().tolist()
                predictions.append(row)
                raw[item["sample_id"]] = {**{key: value[index] for key, value in values.items()}, "target": target[index]}
        return predictions, raw

    # Local step-zero/baseline verification before optimizer creation.
    initial_predictions, initial_raw = predict_items(original_train + dev)
    for values in initial_raw.values():
        assert all(np.max(np.abs(values[key]-values["M0"])) <= 1e-6 for key in ("M1", "M2"))
    jsonl(output / "initial_inference_predictions.jsonl", initial_predictions)
    initial_prediction_hash = sha256_file(output / "initial_inference_predictions.jsonl")
    # First mask/record reads occur after the initial observable predictions exist.
    labels = {}
    for item in original_train + dev:
        e = item["entry"]; record_path = layout.dataset_path(e["record_path"])
        protect(record_path, e["record_sha256"])
        record = read_json(record_path); oracle = record["evaluator_only"]
        desc = e["supervision"]["anchor_masks"][0]
        ap = layout.dataset_path(desc["path"]); protect(ap, desc["sha256"])
        tp = layout.dataset_path(e["supervision"]["target_mask_path"]); protect(tp, e["supervision"]["target_mask_sha256"])
        anchor, target = cv2.imread(str(ap), 0), cv2.imread(str(tp), 0)
        assert anchor is not None and target is not None and anchor.shape == target.shape == (480, 640)
        masks = oracle["masks"]["anchor"]; sp = oracle["spatial_label"]
        assert len(masks) == len(sp["anchor_ids"]) == 1 and masks[0]["object_id"] == sp["anchor_ids"][0]
        assert layout.dataset_path(masks[0]["path"]) == ap and masks[0]["sha256"] == desc["sha256"]
        assert sp["relations"] == [item["entry"]["audit_only"]["relation"]] and sp["reference_frame"] == "image"
        labels[e["sample_id"]] = {"anchor_full": anchor > 0, "target_full": target > 0,
            "anchor_small": cv2.resize((anchor > 0).astype(np.float32), (32, 24), interpolation=cv2.INTER_AREA),
            "truth": e["supervision"]["answerability_state"], "variant": e["variant"], "anchor_ids": sp["anchor_ids"],
            "anchor_mask_path": str(ap), "target_mask_path": str(tp), "rgb_path": str(layout.dataset_path(e["feature_input"]["rgb_path"]))}
    initial_scored = score_predictions(initial_predictions, labels)
    initial_metrics = {split: aggregate([r for r in initial_scored if r["split"] == split]) for split in ("train", "dev")}
    assert initial_metrics["dev"]["models"]["M0"]["anchor_hits"] == 44
    assert initial_metrics["dev"]["models"]["M0"]["swap_both_hits"] == 6
    assert initial_metrics["dev"]["models"]["M0"]["found_both_hits"] == 11
    assert initial_metrics["train"]["models"]["M0"]["anchor_hits"] == 209
    atomic_json(output / "initial_baseline_metrics.json", initial_metrics)
    for runner in runners.values(): runner.start_optimizer(receipt)
    epoch_history, selected = [], {}
    start_time = time.perf_counter()
    for epoch in range(PROTOCOL.max_epochs):
        epoch_captures = {}
        # Cache each frozen batch once per epoch, share identically between arms.
        def prepare(items):
            key = tuple((i["sample_id"], i["prefix_index"]) for i in items)
            if key not in epoch_captures:
                cap = capture(items)
                supervision = torch.zeros(len(items), 3, 24, 32, device=device)
                for i, item in enumerate(items):
                    supervision[i, 0] = torch.from_numpy(labels[item["sample_id"]]["anchor_small"]).to(device)
                epoch_captures[key] = (cap, supervision)
            return epoch_captures[key]
        active_arms = [key for key, r in runners.items() if r.stale_epochs < PROTOCOL.patience]
        if not active_arms: break
        for key in active_arms:
            runner = runners[key]
            train_loss = runner.run_epoch(epoch, train, prepare)
            runner.branch.eval()
            predictions, _ = predict_items(dev)
            jsonl(output / f"dev_inference_epoch_{epoch+1:03d}_{key}.jsonl", predictions)
            scored = score_predictions(predictions, labels); metrics = aggregate(scored)
            own = metrics["models"][key]
            improved = runner.record_dev_selection(epoch, own["anchor_hits"], own["swap_both_hits"], own["empty_mean_sigmoid_max"])
            record = {"arm": key, "epoch_zero_based": epoch, "epoch": epoch+1, "optimizer_steps": runner.optimizer_steps,
                "mean_train_losses": {name: float(np.mean([v[name] for v in train_loss])) for name in ("total", "visible", "empty_bce")},
                "batch_losses": train_loss, "dev": metrics, "selection_tuple": list(runner.best_selection),
                "checkpoint_improved": improved, "stale_epochs": runner.stale_epochs, "selected_epoch": runner.best_epoch+1}
            if improved:
                directory = output / key / "checkpoints" / f"epoch_{epoch+1:03d}"
                directory.mkdir(parents=True, exist_ok=False)
                save_file({name: value.detach().cpu().contiguous() for name, value in runner.branch.residual.state_dict().items()}, str(directory / "mlp.safetensors"))
                atomic_json(directory / "metadata.json", {"arm": key, "conditioning": runner.branch.conditioning,
                    "epoch": epoch+1, "epoch_zero_based": epoch, "optimizer_steps": runner.optimizer_steps,
                    "mlp_sha256": sha256_file(directory / "mlp.safetensors"), "residual_state_sha256": state_digest(runner.branch.residual.state_dict()),
                    "baseline_checkpoint_sha256": active["checkpoint_sha256"], "protocol_sha256": protocol_hash, "dev": own})
                selected[key] = directory
            epoch_history.append(record)
            atomic_json(output / "epoch_history.json", {"records": epoch_history})
            print(f"{key} epoch {epoch+1:02d} loss={record['mean_train_losses']['total']:.6f} dev_anchor={own['anchor_hits']}/61 swap={own['swap_both_hits']}/15 found_both={own['found_both_hits']}/16 best={runner.best_epoch+1} stale={runner.stale_epochs}", flush=True)
        epoch_captures.clear()
        assert state_digest(baseline.state_dict()) == frozen_digest
    training_seconds = time.perf_counter()-start_time
    for key, directory in selected.items():
        metadata = read_json(directory / "metadata.json")
        assert sha256_file(directory / "mlp.safetensors") == metadata["mlp_sha256"]
        arms[key].residual.load_state_dict(load_file(str(directory / "mlp.safetensors"), device="cpu"), strict=True)
        arms[key].eval(); arms[key].residual.zero_grad(set_to_none=True)
        best = output / key / "checkpoints" / "best"; best.mkdir(exist_ok=False)
        save_file({name: value.detach().cpu().contiguous() for name, value in arms[key].residual.state_dict().items()}, str(best / "mlp.safetensors"))
        atomic_json(best / "metadata.json", {**metadata, "selected_from": str(directory.relative_to(output)),
            "mlp_sha256": sha256_file(best / "mlp.safetensors"), "completed_epochs": runners[key].next_epoch,
            "completed_optimizer_steps": runners[key].optimizer_steps})
    final_predictions, final_raw = predict_items(original_train + all_dev, verify=True)
    jsonl(output / "selected_inference_predictions.jsonl", final_predictions)
    horizontal = [p for p in final_predictions if p["scope_status"] == "SUPPORTED_SHADOW"]
    scored = score_predictions(horizontal, labels)
    jsonl(output / "evaluator_rows.jsonl", scored); jsonl(output / "swap_pairs.jsonl", pair_rows(scored))
    order = [p["sample_id"] for p in final_predictions]
    np.savez_compressed(output / "selected_spatial_logits.npz", sample_ids=np.asarray(order),
        **{key: np.stack([final_raw[s][key] for s in order]) for key in ("M0", "M1", "M2", "target")})
    # Reloaded checkpoints must reproduce metrics used to select them.
    groups = {split: aggregate([r for r in scored if r["split"] == split]) for split in ("train", "dev")}
    assert groups["train"]["matched_swap_pairs"] == 51, "Locked train matched subset changed"
    for key in arms:
        chosen = read_json(output / key / "checkpoints" / "best" / "metadata.json")["dev"]
        assert groups["dev"]["models"][key] == chosen, (key, "Checkpoint reload changed dev metrics")
    deltas, bootstrap = {}, {}
    for split in ("train", "dev"):
        subset = [r for r in scored if r["split"] == split]
        deltas[split] = {f"{candidate}_vs_{reference}": paired_changes(subset, candidate, reference)
            for candidate, reference in (("M1", "M0"), ("M2", "M0"), ("M1", "M2"))}
        bootstrap[split] = {f"{candidate}_vs_{reference}": family_bootstrap(subset, candidate, reference)
            for candidate, reference in (("M1", "M0"), ("M2", "M0"), ("M1", "M2"))}
    stratified = {}
    for field in ("variant", "truth"):
        stratified[field] = {f"{split}/{value}": aggregate([r for r in scored if r["split"] == split and r[field] == value])
            for split, value in sorted({(r["split"], r[field]) for r in scored})}
    phrase_groups = defaultdict(list)
    for row in scored: phrase_groups[f"{row['split']}/{row['parsed']['anchors'][0]['text']}"].append(row)
    stratified["anchor_phrase"] = {key: aggregate(rows) for key, rows in sorted(phrase_groups.items())}
    changes = []
    for r in scored:
        if r["anchor_pixels"] and any(r["models"][k]["anchor_hit"] != r["models"]["M0"]["anchor_hit"] for k in arms):
            changes.append(r)
    jsonl(output / "changed_cases.jsonl", changes)
    with (output / "changed_cases.csv").open("w", newline="", encoding="utf-8") as out:
        names = ["sample_id", "split", "variant", "truth", "anchor_phrase", "anchor_ids", "M0_hit", "M1_hit", "M2_hit", "M0_xy", "M1_xy", "M2_xy", "anchor_mask_path", "rgb_path"]
        writer = csv.DictWriter(out, names); writer.writeheader()
        for row in changes:
            writer.writerow({"sample_id": row["sample_id"], "split": row["split"], "variant": row["variant"], "truth": row["truth"],
                "anchor_phrase": row["parsed"]["anchors"][0]["text"], "anchor_ids": json.dumps(row["anchor_ids"]),
                **{key+"_hit": row["models"][key]["anchor_hit"] for key in ("M0", "M1", "M2")},
                **{key+"_xy": json.dumps(row["models"][key]["map_pixel_xy"]) for key in ("M0", "M1", "M2")},
                "anchor_mask_path": row["anchor_mask_path"], "rgb_path": row["rgb_path"]})
    # Latency on the selected checkpoints; same setup/batch/precision as preflight.
    timing_items = train[:8]; inputs = batch(timing_items); prompts = [p["prompt"] for p in timing_items]
    measurements = {key: [] for key in ("M0", "M1", "M2")}
    def sync():
        if device.type == "cuda": torch.cuda.synchronize(device)
    def timed_call(key):
        with torch.no_grad(), autocast_context(device, config["optimization"]):
            if key == "M0": arms["M1"].capture(inputs, prompts)
            else: arms[key](inputs, prompts)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for _ in range(20):
        for key in measurements: timed_call(key)
    for iteration in range(100):
        keys = list(measurements); keys = keys[iteration % 3:]+keys[:iteration % 3]
        for key in keys:
            sync(); begin = time.perf_counter(); timed_call(key); sync()
            measurements[key].append((time.perf_counter()-begin)*1000)
    latency = {key: {"median_ms": statistics.median(values), "p95_ms": float(np.percentile(values, 95))} for key, values in measurements.items()}
    for key in arms:
        latency[key]["overhead_fraction"] = latency[key]["median_ms"]/latency["M0"]["median_ms"]-1
        latency[key]["gate_pass"] = latency[key]["overhead_fraction"] <= .2
    atomic_json(output / "latency.json", {"warmup": 20, "runs_per_arm": 100, "batch_size": 8, "measurements": latency,
        "raw_ms": measurements, "scope": "same wrapper setup; cached-feature sidecar + parser; no feature loading/backbone",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
    assert state_digest(baseline.state_dict()) == frozen_digest and all(p.grad is None for p in baseline.parameters())
    assert hook_counts == {key: len(m._forward_hooks) for key, m in baseline.named_modules()}
    assert all(sha256_file(Path(path)) == digest for path, digest in protected.items())
    assert sha256_file(output / "initial_inference_predictions.jsonl") == initial_prediction_hash
    gates = locked_gates([r for r in scored if r["split"] == "dev"], True, all(latency[key]["gate_pass"] for key in arms))
    summary = {"status": "PILOT_COMPLETE", "upgrade_gates_pass": gates["pass"], "groups": groups, "gates": gates,
        "paired_changes": deltas, "family_bootstrap": bootstrap, "stratified": stratified, "latency": latency,
        "training_seconds": training_seconds, "arms": {key: {"epochs": runner.next_epoch, "optimizer_steps": runner.optimizer_steps,
            "selected_epoch": runner.best_epoch+1, "stop_reason": "patience_5" if runner.stale_epochs >= PROTOCOL.patience else "max_epochs_15",
            "selected_checkpoint": str((output / key / "checkpoints" / "best" / "mlp.safetensors").relative_to(output))} for key, runner in runners.items()},
        "protocol_sha256": protocol_hash, "baseline_state_sha256_before_after": frozen_digest,
        "baseline_weights_stats_unchanged": True, "all_400_dev_baseline_outputs_exact": True, "inactive_maps_exact": True,
        "protected_files": len(protected), "calibration_access": "bundle hash only; no samples/fit", "test_accessed": False,
        "mc_dropout": False, "verifier_implemented": False, "active_profile_changed": False,
        "shadow_version": SHADOW_VERSION, "parser_version": PARSER_VERSION, "runtime": runtime_info(),
        "limits": ["One seed, development selected checkpoints, no selection-corrected CI", "Empty dev anchors are three samples of one family",
                   "M0 training objective differs; M1/M2 control both share the new empty-mask term", "Shadow outputs do not alter answerability/risk"]}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "freeze_lock.json", {"baseline_bundle": active, "baseline_state_sha256": frozen_digest,
        "trainable_parameters_per_arm": 33024, "only_residual_updated": True, "protocol_sha256": protocol_hash,
        "checkpoints": {key: sha256_file(output / key / "checkpoints" / "best" / "mlp.safetensors") for key in arms}})
    atomic_json(output / "provenance.json", {"protected_sha256": protected, "argv": sys.argv,
        "source_sha256": {str(p): sha256_file(p) for p in (Path(__file__).resolve(), workspace_path("new/src/pcrau/anchor_shadow_pilot.py"))},
        "baseline_state_sha256_before_after": frozen_digest})
    print(json.dumps({"status": "PILOT_COMPLETE", "upgrade_gates_pass": gates["pass"], "groups": groups, "gates": gates, "arms": summary["arms"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
