#!/usr/bin/env python3
"""Frozen train/dev S1 shadow preflight. Backward probes only; zero optimizer steps."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcrau.anchor_shadow import AnchorShadow, SHADOW_VERSION
from pcrau.anchor_shadow_experiment import PROTOCOL, ShadowPilotRunner, anchor_shadow_loss, family_batches, observable_batch, pilot_presentations, state_digest
from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.engine import autocast_context
from pcrau.model import PCRAUTargetV2
from pcrau.query_parser import PARSER_VERSION
from pcrau.utils import atomic_json, read_json, runtime_info, seed_everything, sha256_file, workspace_path


def spatial_summary(logits):
    flat = logits.detach().float().flatten().cpu()
    index = int(flat.argmax()); width = logits.shape[-1]; height = logits.shape[-2]
    y, x = divmod(index, width)
    return {"grid_xy": [x, y], "pixel_xy": [int((x + .5) * 640 / width), int((y + .5) * 480 / height)],
            "sigmoid_max": float(flat.max().sigmoid()), "sigmoid_mean": float(flat.sigmoid().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="new/outputs/pcrau_s1_anchor_shadow_preflight_20261005_r3")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    output = workspace_path(args.output)
    if workspace_path("new/outputs") not in output.parents:
        raise ValueError("Output must be under new/outputs")
    output.mkdir(exist_ok=False)
    protected = {}

    def protect(path, expected=None):
        path = Path(path).resolve()
        if str(path) not in protected:
            protected[str(path)] = sha256_file(path)
        digest = protected[str(path)]
        if expected and digest != expected:
            raise ValueError(f"Protected hash mismatch: {path}")
        return digest

    active_path = workspace_path("new/outputs/active_experimental_profile.json")
    active = read_json(active_path); protect(active_path)
    for key in ("config", "checkpoint", "calibrator", "profiles"):
        protect(active[key], active[key + "_sha256"])
    for path in (workspace_path("new/src/pcrau")).glob("*.py"):
        protect(path)
    protocol_path = workspace_path("plan/S1_PHRASE_CONDITIONED_ANCHOR_EXPERIMENT_20261005.md")
    protocol_hash = protect(protocol_path)
    protect(__file__)
    config = read_json(Path(active["config"]))
    seed_everything(PROTOCOL.seed); torch.set_num_threads(4)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    model = PCRAUTargetV2(config).to(device).eval()
    load_model_checkpoint(model, active["checkpoint"], active["config_sha256"])
    arms = {"M1": AnchorShadow(model, "phrase", PROTOCOL.seed), "M2": AnchorShadow(model, "whole_text", PROTOCOL.seed)}
    runners = {key: ShadowPilotRunner(arm) for key, arm in arms.items()}
    initial = {key: state_digest(arm.residual.state_dict()) for key, arm in arms.items()}
    assert len(set(initial.values())) == 1
    assert all(sum(p.numel() for p in arm.parameters() if p.requires_grad) == 33024 for arm in arms.values())
    frozen_state = state_digest(model.state_dict())
    hook_counts = {name: len(m._forward_hooks) for name, m in model.named_modules()}
    datasets = {split: ArchivedPCRAUDataset(config, split, verify_feature_hash=True) for split in ("train", "dev")}
    layout = datasets["train"].layout
    protect(layout.manifest); protect(layout.feature_index)
    entries = read_json(layout.manifest)["entries"]
    train = pilot_presentations(entries, config["model"], "train")
    dev = pilot_presentations(entries, config["model"], "dev")
    assert len(train) == 1024 and len(dev) == 64
    assert not {p["family_id"] for p in train} & {p["family_id"] for p in dev}
    batches = family_batches(train, 0)
    assert len(batches) == 16 and all(len(batch) == 64 for batch in batches)
    assert sorted(i for batch in batches for i in batch) == list(range(1024))
    assert batches == family_batches(train, 0)
    atomic_json(output / "locked_protocol.json", {"protocol": asdict(PROTOCOL), "design_sha256": protocol_hash,
        "initialization_sha256": initial, "train_presentations": len(train), "dev_horizontal": len(dev),
        "trainable_names": [name for name, p in arms["M1"].named_parameters() if p.requires_grad],
        "trainable_count": 33024, "optimizer_steps": 0})
    atomic_json(output / "family_batch_plan.json", {"epoch0": [[{"sample_id": train[i]["sample_id"],
        "family_id": train[i]["family_id"], "prefix_index": train[i]["prefix_index"]} for i in batch] for batch in batches]})
    save_file({k: v.detach().cpu().contiguous() for k, v in arms["M1"].residual.state_dict().items()}, str(output / "initial_mlp.safetensors"))

    def prepare(items):
        split = items[0]["entry"]["split"]
        dataset = datasets[split]
        for item in items:
            e = item["entry"]
            desc = dataset.features[dataset.sample_to_feature[e["sample_id"]]]
            assert desc["rgb_sha256"] == e["feature_input"]["rgb_sha256"] and desc["depth_sha256"] == e["feature_input"]["depth_sha256"]
            protect(dataset.layout.feature_root / desc["path"], desc["sha256"])
        return observable_batch(dataset, [i["entry"] for i in items], [i["prompt"] for i in items], device)

    all_dev = [{"entry": e, "prompt": e["feature_input"]["prompt"], "sample_id": e["sample_id"],
                "family_id": e["family_id"], "prefix_index": 0} for e in datasets["dev"].entries]
    modes = ["fp32"] + (["amp_bfloat16"] if device.type == "cuda" else [])
    assert config["optimization"].get("amp_dtype") == "bfloat16"
    modes_summary = {}
    with (output / "inference_predictions.jsonl").open("w") as predictions, (output / "identity_checks.jsonl").open("w") as identities:
        for mode in modes:
            checked, active_count, scopes, maximum = Counter(), Counter(), Counter(), {"M1": 0., "M2": 0.}
            for split, presentations in (("train", train), ("dev", all_dev)):
                for start in range(0, len(presentations), 8):
                    items = presentations[start:start + 8]; prompts = [i["prompt"] for i in items]
                    batch = prepare(items)
                    ctx = autocast_context(device, config["optimization"]) if mode != "fp32" else torch.autocast(device.type, enabled=False)
                    with torch.no_grad(), ctx:
                        plain = model(batch)
                        capture = arms["M1"].capture(batch, prompts)
                        logits = {key: arm.predict(capture) for key, arm in arms.items()}
                    exact = {key: torch.equal(value, capture.baseline[key]) for key, value in plain.items()}
                    assert all(exact.values()), (mode, start, exact)
                    errors = {key: float((v.float() - plain["anchor_logits"].float()).abs().max()) for key, v in logits.items()}
                    assert all(v <= 1e-6 for v in errors.values()), (mode, start, errors)
                    assert hook_counts == {name: len(m._forward_hooks) for name, m in model.named_modules()}
                    for key, value in errors.items(): maximum[key] = max(maximum[key], value)
                    checked[split] += len(items); active_count[split] += int(capture.active_slots.sum())
                    identities.write(json.dumps({"mode": mode, "split": split, "start": start,
                        "samples": len(items), "baseline_output_exact": exact, "anchor_max_abs_error": errors,
                        "baseline_output_sha256": state_digest(plain), "captured_baseline_output_sha256": state_digest(capture.baseline)}) + "\n")
                    for index, item in enumerate(items):
                        scopes[(split, capture.scope_status[index])] += 1
                        predictions.write(json.dumps({"mode": mode, "sample_id": item["sample_id"], "family_id": item["family_id"],
                            "split": split, "prefix_index": item["prefix_index"], "prompt": item["prompt"],
                            "parsed": capture.queries[index].to_dict(), "scope_status": capture.scope_status[index],
                            "baseline_anchor": spatial_summary(plain["anchor_logits"][index, 0]),
                            "M1_anchor": spatial_summary(logits["M1"][index, 0]), "M2_anchor": spatial_summary(logits["M2"][index, 0]),
                            "baseline_answerability_logits": plain["answerability_logits"][index].float().cpu().tolist(),
                            "baseline_output_exact": True, "anchor_max_abs_error": errors}) + "\n")
                    if start % 160 == 0 or start + 8 >= len(presentations):
                        print(f"{mode} {split} frozen identity {min(start+8,len(presentations))}/{len(presentations)}", flush=True)
            modes_summary[mode] = {"presentations_checked": dict(checked), "active_anchor_presentations": dict(active_count),
                "scope_counts": [{"split": s, "status": status, "count": count} for (s, status), count in sorted(scopes.items())],
                "max_abs_error": maximum, "all_baseline_outputs_exact": True}
    predictions_hash = sha256_file(output / "inference_predictions.jsonl")
    # Oracle masks are first read after all prompt-only predictions are persisted.
    original_train = {p["sample_id"]: p["entry"] for p in train}
    small_masks, pixel_counts = {}, {}
    for sample, entry in sorted(original_train.items()):
        descriptor = entry["supervision"]["anchor_masks"][0]
        path = layout.dataset_path(descriptor["path"]); protect(path, descriptor["sha256"])
        mask = cv2.imread(str(path), 0)
        assert mask is not None and mask.shape == (480, 640)
        pixel_counts[sample] = int((mask > 0).sum())
        small_masks[sample] = torch.from_numpy(cv2.resize((mask > 0).astype(np.float32), (32, 24), interpolation=cv2.INTER_AREA))
    assert Counter(bool(v) for v in pixel_counts.values()) == {True: 251, False: 5}
    positive = next(k for k in sorted(pixel_counts) if pixel_counts[k])
    negative = next(k for k in sorted(pixel_counts) if not pixel_counts[k])
    probe_items = [p for p in train if p["sample_id"] in {positive, negative}]
    probe_batch = prepare(probe_items)
    masks = torch.zeros(len(probe_items), 3, 24, 32, device=device)
    for i, item in enumerate(probe_items): masks[i, 0] = small_masks[item["sample_id"]].to(device)
    with autocast_context(device, config["optimization"]):
        probe = arms["M1"].capture(probe_batch, [p["prompt"] for p in probe_items])
    gradients = {}
    for key, arm in arms.items():
        arm.train(); arm.zero_grad(set_to_none=True)
        losses = anchor_shadow_loss(arm.predict(probe), masks, probe.active_slots)
        losses["total"].backward()
        norms = {name: float(p.grad.norm()) for name, p in arm.residual.named_parameters() if p.grad is not None}
        assert len(norms) == 4 and all(torch.isfinite(p.grad).all() for p in arm.residual.parameters())
        assert norms["2.weight"] > 0 and norms["2.bias"] > 0
        assert norms["0.weight"] == norms["0.bias"] == 0  # zero W2 blocks W1 on the first backward
        assert all(p.grad is None for p in model.parameters())
        assert initial[key] == state_digest(arm.residual.state_dict())
        gradients[key] = {"losses": {k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in losses.items()},
                          "gradient_norms": norms, "baseline_gradients": 0, "optimizer_steps": 0}
        arm.zero_grad(set_to_none=True)
    atomic_json(output / "gradient_probe.json", {"samples": [positive, negative], "presentation_count": len(probe_items),
        "oracle_access": "train supervision after inference artifact persisted", "arms": gradients,
        "zero_W1_gradient_expected_at_zero_W2": True, "optimizer_created": False, "optimizer_steps": 0})

    # Same parser/capture setup for M0, shadow branch disabled; sidecar-only timing.
    timing_items = train[:8]; timing_batch = prepare(timing_items); timing_prompts = [p["prompt"] for p in timing_items]
    measurements = {k: [] for k in ("M0", "M1", "M2")}
    def synchronize():
        if device.type == "cuda": torch.cuda.synchronize(device)
    def call(key):
        with torch.no_grad(), autocast_context(device, config["optimization"]):
            if key == "M0": arms["M1"].capture(timing_batch, timing_prompts)
            else: arms[key](timing_batch, timing_prompts)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for _ in range(20):
        for key in measurements: call(key)
    for iteration in range(100):
        order = list(measurements); order = order[iteration % 3:] + order[:iteration % 3]
        for key in order:
            synchronize(); begin = time.perf_counter(); call(key); synchronize()
            measurements[key].append((time.perf_counter() - begin) * 1000)
    latency = {key: {"median_ms": statistics.median(values), "p95_ms": float(np.percentile(values, 95)), "runs": len(values)} for key, values in measurements.items()}
    for key in ("M1", "M2"):
        latency[key]["overhead_fraction"] = latency[key]["median_ms"] / latency["M0"]["median_ms"] - 1
        latency[key]["gate_20_percent_pass"] = latency[key]["overhead_fraction"] <= .20
    atomic_json(output / "latency.json", {"batch_size": 8, "warmup": 20, "measurements": latency,
        "scope": "parser + capture + cached-feature sidecar; M0 same wrapper capture with residual disabled",
        "includes_backbone_or_disk_loading": False, "samples": [p["sample_id"] for p in timing_items],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "raw_measurements_ms": measurements})
    assert all(r.optimizer is None and r.optimizer_steps == 0 for r in runners.values())
    assert frozen_state == state_digest(model.state_dict())
    assert all(initial[key] == state_digest(arm.residual.state_dict()) for key, arm in arms.items())
    assert hook_counts == {name: len(m._forward_hooks) for name, m in model.named_modules()}
    assert all(sha256_file(Path(p)) == h for p, h in protected.items())
    assert predictions_hash == sha256_file(output / "inference_predictions.jsonl")
    summary = {"status": "S1_SHADOW_PREFLIGHT_PASS", "technical_gates_pass": True, "runtime": runtime_info(),
        "device": str(device), "shadow_version": SHADOW_VERSION, "parser_version": PARSER_VERSION,
        "modes": modes_summary, "gradient_probes_pass": True, "all_baseline_weights_stats_unchanged": True,
        "protected_files": len(protected), "temporary_hooks_cleaned": True, "trainable_parameters_per_arm": 33024,
        "initialization_sha256": initial, "frozen_model_state_sha256": frozen_state, "protocol_sha256": protocol_hash,
        "optimizer_created": False, "optimizer_steps": 0, "training_executed": False,
        "calibration_access": "bundle hash only; no samples/fit", "test_accessed": False, "mc_dropout": False,
        "inference_predictions_sha256": predictions_hash, "inference_saved_before_oracle_masks": True,
        "latency": latency, "latency_gate_pass": all(latency[k]["gate_20_percent_pass"] for k in ("M1", "M2")),
        "binding_improvement_evaluated": False, "limits": ["Step-zero identity, not learned anchor improvement",
            "Backward probe only, no optimizer update", "Pilot epoch/checkpoint orchestration has not been exercised"]}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "provenance.json", {"protected_sha256": protected, "argv": sys.argv,
        "frozen_model_state_sha256_before_after": frozen_state, "source_sha256": {str(p): sha256_file(p) for p in (
            workspace_path("new/src/pcrau/anchor_shadow.py"), workspace_path("new/src/pcrau/anchor_shadow_experiment.py"),
            workspace_path("new/tests/test_anchor_shadow.py"), Path(__file__).resolve())}})
    print(json.dumps({"status": summary["status"], "output": str(output), "latency": latency,
                      "optimizer_steps": 0, "protected_files": len(protected)}), flush=True)


if __name__ == "__main__":
    main()
