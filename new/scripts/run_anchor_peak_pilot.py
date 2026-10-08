#!/usr/bin/env python3
"""Explicitly authorized, locked C1/P1/P2 peak-aware pilot on development train/dev only."""
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
from pcrau.anchor_shadow import SHADOW_VERSION
from pcrau.anchor_peak_experiment import prepare_peak_arms, PeakSupervision, supervision_from_full_masks, validate_locked_peak_config
from pcrau.anchor_peak_runner import PeakPilotRunner
from pcrau.anchor_shadow_experiment import PROTOCOL, family_batches, observable_batch, pilot_presentations, state_digest
from pcrau.anchor_peak_pilot import aggregate, family_bootstrap, locked_gates, pair_rows, paired_changes, score_predictions, spatial_summary
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
    parser.add_argument("--output", default="new/outputs/pcrau_s1_anchor_peak_pilot_v2_20261005")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--protection-manifest", type=Path, help="Optional task-start hashes for historical artifacts")
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

    pre_root = workspace_path("new/outputs/pcrau_s1_anchor_peak_v2_preflight_20261005")
    receipt = read_json(pre_root / "summary.json")
    pre_provenance = read_json(pre_root / "provenance.json")
    protect(pre_root / "summary.json"); protect(pre_root / "provenance.json")
    assert receipt["status"] == "S1_PEAK_V2_PREFLIGHT_PASS" and receipt["technical_gates_pass"]
    for path, digest in pre_provenance["protected_sha256"].items():
        protect(path, digest)
    if args.protection_manifest is not None:
        protect(args.protection_manifest)
        for path, digest in read_json(args.protection_manifest).items():
            protect(path, digest)
    candidate_path = workspace_path("new/configs/anchor_peak_pilot_v2_20261005.json")
    candidate = read_json(candidate_path); validate_locked_peak_config(candidate)
    candidate_hash = protect(candidate_path, receipt["config_sha256"])
    checks_path = workspace_path("new/outputs/pcrau_s1_anchor_peak_v2_runner_checks_20261005.json")
    checks = read_json(checks_path); protect(checks_path)
    assert checks["pass"] and len(checks["tests"]) == 5
    for path,digest in checks["source_sha256"].items(): protect(path,digest)
    active_path = workspace_path("new/outputs/active_experimental_profile.json")
    active = read_json(active_path); protect(active_path)
    for key in ("config", "checkpoint", "calibrator", "profiles"):
        protect(active[key], active[key + "_sha256"])
    config = read_json(Path(active["config"]))
    protocol_path = workspace_path("plan/S1_ANCHOR_PEAK_PROTOCOL_V2_LOCK_20261005.md")
    protocol_hash = protect(protocol_path, receipt["protocol_sha256"])
    for path in workspace_path("new/src/pcrau").glob("*.py"):
        protect(path)
    protect(__file__)
    seed_everything(PROTOCOL.seed); torch.set_num_threads(4)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    baseline = PCRAUTargetV2(config).to(device).eval()
    load_model_checkpoint(baseline, active["checkpoint"], active["config_sha256"])
    prepared_arms = prepare_peak_arms(baseline)
    arms = {key: arm.branch for key, arm in prepared_arms.items()}
    runners = {key: PeakPilotRunner(arm) for key, arm in prepared_arms.items()}
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
        "arm_order_each_epoch": ["C1", "P1", "P2"], "trainable_per_arm": 33024}
    atomic_json(output / "locked_protocol.json", locked)
    atomic_json(output / "split_manifest.json", {split: [{"sample_id": p["sample_id"], "family_id": p["family_id"],
        "prefix_index": p["prefix_index"], "prompt": p["prompt"]} for p in ps] for split, ps in (("train", train), ("dev", dev))})
    plans = {str(epoch): family_batches(train, epoch) for epoch in range(PROTOCOL.max_epochs)}
    assert all(len(plan) == 16 and all(len(batch) == 64 for batch in plan) for plan in plans.values())
    assert plans == read_json(pre_root / "family_batch_plans.json")
    assert read_json(output / "split_manifest.json") == read_json(pre_root / "split_manifest.json")
    atomic_json(output / "family_batch_plans.json", plans)
    execution_path = workspace_path("plan/S1_ANCHOR_PEAK_PILOT_V2_EXECUTION_LOCK_20261005.md")
    execution_hash = protect(execution_path)
    locked.update({"config": candidate, "config_sha256": candidate_hash, "protocol_sha256": protocol_hash,
        "execution_addendum_sha256": execution_hash, "training_authorized": True,
        "epoch_orchestration_implemented": True, "pre_optimizer_checks_pass": False})
    atomic_json(output / "locked_protocol.json", locked)

    def batch(items):
        split = items[0]["entry"]["split"]
        return observable_batch(caches[split], [p["entry"] for p in items], [p["prompt"] for p in items], device)

    def capture(items):
        with autocast_context(device, config["optimization"]):
            return arms["C1"].capture(batch(items), [p["prompt"] for p in items])

    def predict_items(items, verify=False):
        predictions, raw = [], {}
        for start in range(0, len(items), 8):
            subset = items[start:start+8]; inputs = batch(subset)
            with torch.no_grad(), autocast_context(device, config["optimization"]):
                cap = arms["C1"].capture(inputs, [p["prompt"] for p in subset])
                maps = {key: arm.predict(cap) for key, arm in arms.items()}
                if verify:
                    plain = baseline(inputs)
                    assert all(torch.equal(v, cap.baseline[key]) for key, v in plain.items())
                    other = arms["P2"].capture(inputs, [p["prompt"] for p in subset])
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
    initial_predictions, initial_raw = predict_items(original_train + all_dev, verify=True)
    for values in initial_raw.values():
        assert all(np.max(np.abs(values[key]-values["M0"])) <= 1e-6 for key in ("C1", "P1", "P2"))
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
    for label in labels.values():
        full = np.zeros((1,3,480,640),dtype=bool); full[0,0] = label["anchor_full"]
        label["peak_supervision"] = supervision_from_full_masks(full)
    support_counts = {}
    for split, items in (("train",original_train),("dev",dev)):
        ls=[labels[item["sample_id"]] for item in items]
        support_counts[split]={"visible":sum(int(x["anchor_full"].any()) for x in ls),
            "empty":sum(int(not x["anchor_full"].any()) for x in ls),
            "center_visible":sum(int(x["peak_supervision"].center_support.any()) for x in ls)}
    assert support_counts == {"train":{"visible":251,"empty":5,"center_visible":250},
        "dev":{"visible":61,"empty":3,"center_visible":61}}
    initial_scored = score_predictions([p for p in initial_predictions if p["scope_status"] == "SUPPORTED_SHADOW"], labels)
    initial_metrics = {split: aggregate([r for r in initial_scored if r["split"] == split]) for split in ("train", "dev")}
    assert initial_metrics["dev"]["models"]["M0"]["anchor_hits"] == 44
    assert initial_metrics["dev"]["models"]["M0"]["swap_both_hits"] == 6
    assert initial_metrics["dev"]["models"]["M0"]["found_both_hits"] == 11
    assert initial_metrics["train"]["models"]["M0"]["anchor_hits"] == 209
    atomic_json(output / "initial_baseline_metrics.json", initial_metrics)
    assert state_digest(baseline.state_dict()) == frozen_digest
    assert all(sha256_file(Path(path)) == digest for path,digest in protected.items())
    locked["pre_optimizer_checks_pass"] = True
    locked["support_counts"] = support_counts
    locked["unit_checks_sha256"] = sha256_file(checks_path)
    locked["source_sha256"] = {str(p):sha256_file(p) for p in workspace_path("new/src/pcrau").glob("*.py")}
    locked["source_sha256"][str(Path(__file__).resolve())] = sha256_file(Path(__file__).resolve())
    test_path = workspace_path("new/tests/test_anchor_peak_pilot.py")
    locked["source_sha256"][str(test_path)] = protect(test_path,checks["test_sha256"])
    atomic_json(output / "locked_protocol.json", locked)
    snapshot = output / "source_snapshot"; snapshot.mkdir()
    for i,(path,digest) in enumerate(locked["source_sha256"].items()):
        (snapshot / f"{i:03d}_{Path(path).name}").write_bytes(Path(path).read_bytes())
    atomic_json(snapshot / "manifest.json", {p:{"sha256":h,"snapshot":f"{i:03d}_{Path(p).name}"} for i,(p,h) in enumerate(locked["source_sha256"].items())})
    atomic_json(output / "pre_optimizer_receipt.json", {"pass":True,"optimizer_created":False,
        "execution_lock_sha256":sha256_file(output / "locked_protocol.json"),"protected_sha256":protected})
    for runner in runners.values(): runner.start_optimizer(receipt, locked)
    epoch_history, selected = [], {}
    start_time = time.perf_counter()
    for epoch in range(PROTOCOL.max_epochs):
        epoch_captures = {}
        # Cache each frozen batch once per epoch, share identically between arms.
        def prepare(items):
            key = tuple((i["sample_id"], i["prefix_index"]) for i in items)
            if key not in epoch_captures:
                cap = capture(items)
                supervision = PeakSupervision(*(torch.cat([getattr(labels[item["sample_id"]]["peak_supervision"], field) for item in items]).to(device)
                    for field in ("area_masks", "center_support", "full_visible", "full_pixel_counts")))
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
                "mean_train_losses": {name: float(np.mean([v[name] for v in train_loss])) for name in train_loss[0]},
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
    diagnostics=[]
    for row in scored:
        label=labels[row["sample_id"]]; centers=label["peak_supervision"].center_support[0,0].numpy()
        for arm in ("M0", "C1", "P1", "P2"):
            flat=final_raw[row["sample_id"]][arm][0].astype(np.float64)
            inside=float(flat[centers].max()) if centers.any() else None
            outside=float(flat[~centers].max()) if (~centers).any() else None
            prob=np.exp(flat-flat.max());prob/=prob.sum()
            diagnostics.append({"sample_id":row["sample_id"],"split":row["split"],"arm":arm,
                "full_pixels":row["anchor_pixels"],"center_count":int(centers.sum()),
                "max_inside":inside,"max_outside":outside,
                "ranking_violation":max(0.,1+outside-inside) if inside is not None and outside is not None else None,
                "softmax_mass_inside_centers":float(prob[centers].sum()),
                "max_logit":float(flat.max()),"mean_zero_bce":float(np.logaddexp(0,flat).mean())})
    jsonl(output / "peak_diagnostics.jsonl",diagnostics)
    jsonl(output / "empty_cases.jsonl",[r for r in scored if not r["anchor_pixels"]])
    order = [p["sample_id"] for p in final_predictions]
    np.savez_compressed(output / "selected_spatial_logits.npz", sample_ids=np.asarray(order),
        **{key: np.stack([final_raw[s][key] for s in order]) for key in ("M0", "C1", "P1", "P2", "target")})
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
            for candidate, reference in (("C1", "M0"), ("P1", "M0"), ("P2", "M0"), ("P1", "C1"), ("P1", "P2"))}
        bootstrap[split] = {f"{candidate}_vs_{reference}": family_bootstrap(subset, candidate, reference)
            for candidate, reference in (("C1", "M0"), ("P1", "M0"), ("P2", "M0"), ("P1", "C1"), ("P1", "P2"))}
    family_deltas=[]
    for split,family in sorted({(r["split"],r["family_id"]) for r in scored}):
        rs=[r for r in scored if r["split"]==split and r["family_id"]==family];m=aggregate(rs)
        for a,b in (("P1","C1"),("P1","P2")):
            family_deltas.append({"split":split,"family_id":family,"contrast":f"{a}_vs_{b}",
                **paired_changes(rs,a,b),"swap_delta":m["models"][a]["swap_both_hits"]-m["models"][b]["swap_both_hits"],
                "found_both_delta":m["models"][a]["found_both_hits"]-m["models"][b]["found_both_hits"]})
    jsonl(output / "family_paired_deltas.jsonl",family_deltas)
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
        names = ["sample_id", "split", "variant", "truth", "anchor_phrase", "anchor_ids", "M0_hit", "C1_hit", "P1_hit", "P2_hit", "M0_xy", "C1_xy", "P1_xy", "P2_xy", "anchor_mask_path", "rgb_path"]
        writer = csv.DictWriter(out, names); writer.writeheader()
        for row in changes:
            writer.writerow({"sample_id": row["sample_id"], "split": row["split"], "variant": row["variant"], "truth": row["truth"],
                "anchor_phrase": row["parsed"]["anchors"][0]["text"], "anchor_ids": json.dumps(row["anchor_ids"]),
                **{key+"_hit": row["models"][key]["anchor_hit"] for key in ("M0", "C1", "P1", "P2")},
                **{key+"_xy": json.dumps(row["models"][key]["map_pixel_xy"]) for key in ("M0", "C1", "P1", "P2")},
                "anchor_mask_path": row["anchor_mask_path"], "rgb_path": row["rgb_path"]})
    # Latency on the selected checkpoints; same setup/batch/precision as preflight.
    timing_items = train[:8]; inputs = batch(timing_items); prompts = [p["prompt"] for p in timing_items]
    measurements = {key: [] for key in ("M0", "C1", "P1", "P2")}
    def sync():
        if device.type == "cuda": torch.cuda.synchronize(device)
    def timed_call(key):
        with torch.no_grad(), autocast_context(device, config["optimization"]):
            if key == "M0": arms["C1"].capture(inputs, prompts)
            else: arms[key](inputs, prompts)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for _ in range(20):
        for key in measurements: timed_call(key)
    for iteration in range(100):
        keys = list(measurements); keys = keys[iteration % 4:]+keys[:iteration % 4]
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
        "execution_addendum_sha256": execution_hash, "config_sha256": candidate_hash, "epoch_orchestration_implemented": True, "shadow_version": SHADOW_VERSION, "parser_version": PARSER_VERSION, "runtime": runtime_info(),
        "limits": ["One seed, development selected checkpoints, no selection-corrected CI", "Empty dev anchors are three samples of one family",
                   "C1 exact v1 loss; P1/P2 share peak objective; all arms start from same init", "Shadow outputs do not alter answerability/risk"]}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "freeze_lock.json", {"baseline_bundle": active, "baseline_state_sha256": frozen_digest,
        "trainable_parameters_per_arm": 33024, "only_residual_updated": True, "protocol_sha256": protocol_hash,
        "checkpoints": {key: sha256_file(output / key / "checkpoints" / "best" / "mlp.safetensors") for key in arms}})
    atomic_json(output / "provenance.json", {"protected_sha256": protected, "argv": sys.argv,
        "source_sha256": locked["source_sha256"],
        "baseline_state_sha256_before_after": frozen_digest})
    print(json.dumps({"status": "PILOT_COMPLETE", "upgrade_gates_pass": gates["pass"], "groups": groups, "gates": gates, "arms": summary["arms"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
