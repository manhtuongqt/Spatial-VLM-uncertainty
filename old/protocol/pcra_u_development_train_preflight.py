#!/usr/bin/env python3
"""Static + one-batch/no-step gate for P-CRA-U development training."""

from __future__ import annotations

import argparse
import json
import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

if __package__:
    from .pcra_u_development_common import (
        ANSWERABILITY_CLASSES,
        SOURCE_CLASSES,
        DevelopmentTrainingError,
        PCRAUDevelopmentV1,
        compute_loss,
        global_class_weights,
        gradient_groups,
        model_inputs,
        parse_relation,
        pool_raw_feature,
        read_json,
        safe_resolve,
        seed_runtime,
        sha256_file,
        tokenize,
        write_json,
    )
    from .prepare_pcra_u_development_train import tree_digest
    from .wp3_feature_hook_smoke import extract_features, forbidden_cache_findings, load_model, model_inventory_sha256
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pcra_u_development_common import (  # type: ignore
        ANSWERABILITY_CLASSES,
        SOURCE_CLASSES,
        DevelopmentTrainingError,
        PCRAUDevelopmentV1,
        compute_loss,
        global_class_weights,
        gradient_groups,
        model_inputs,
        parse_relation,
        pool_raw_feature,
        read_json,
        safe_resolve,
        seed_runtime,
        sha256_file,
        tokenize,
        write_json,
    )
    from prepare_pcra_u_development_train import tree_digest  # type: ignore
    from wp3_feature_hook_smoke import (  # type: ignore
        extract_features,
        forbidden_cache_findings,
        load_model,
        model_inventory_sha256,
    )


WORKSPACE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "pcra_u_development_train_v1"
CONFIG_PATH = WORKSPACE / "protocol/pcra_u_development_train_config.json"
MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_development_train_manifest.json"
FEATURE_MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_development_feature_manifest.json"
SEED_LOCK_PATH = WORKSPACE / "protocol/pcra_u_development_seed_lock.json"
REPORT_JSON = WORKSPACE / "protocol/PCRA_U_DEVELOPMENT_TRAIN_PREFLIGHT_REPORT.json"
REPORT_MD = WORKSPACE / "protocol/PCRA_U_DEVELOPMENT_TRAIN_PREFLIGHT_REPORT.md"
EXECUTION_LOCK = WORKSPACE / "protocol/pcra_u_development_execution_lock.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(WORKSPACE)): sha256_file(path) for path in paths}


def protected_now(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    protected = manifest["protected_inputs"]
    model_root = safe_resolve(WORKSPACE, config["feature_input"]["model_root"])
    baseline = {
        relative: sha256_file(safe_resolve(WORKSPACE, relative))
        for relative in protected["wp3_locked_baseline_artifacts"]
    }
    dataset_root = safe_resolve(WORKSPACE, manifest["dataset_root"])
    return {
        "dataset_index_sha256": sha256_file(dataset_root / "dataset_index.json"),
        "family_manifest_sha256": sha256_file(dataset_root / "family_manifest.json"),
        "full_qc_report_sha256": sha256_file(dataset_root / "DATASET_V2_1_DEVELOPMENT_FULL_QC_REPORT.json"),
        "dataset_tree_sha256": tree_digest(dataset_root),
        "model_inventory_sha256": model_inventory_sha256(model_root),
        "wp3_locked_baseline_artifacts": baseline,
    }


def static_checks(manifest: dict[str, Any], feature_manifest: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    entries = manifest["entries"]
    train = [entry for entry in entries if entry["split"] == "train"]
    dev = [entry for entry in entries if entry["split"] == "dev"]
    train_families = {entry["family_id"] for entry in train}
    dev_families = {entry["family_id"] for entry in dev}
    sample_ids = [entry["sample_id"] for entry in entries]
    feature_sample_ids = [entry["sample_id"] for entry in feature_manifest["entries"]]
    relations_match = all(
        parse_relation(entry["feature_input"]["prompt"]) == entry["audit_only"]["relation"]
        for entry in entries
    )
    all_files_valid = True
    mask_invariants = True
    source_labels = set()
    state_counts = {split: {state: 0 for state in ANSWERABILITY_CLASSES} for split in ["train", "dev"]}
    source_counts = {split: {source: 0 for source in SOURCE_CLASSES} for split in ["train", "dev"]}
    for entry in entries:
        paths = [
            (entry["record_path"], entry["record_sha256"]),
            (entry["feature_input"]["rgb_path"], entry["feature_input"]["rgb_sha256"]),
            (entry["feature_input"]["depth_path"], entry["feature_input"]["depth_sha256"]),
            (entry["supervision"]["target_mask_path"], entry["supervision"]["target_mask_sha256"]),
            (
                entry["supervision"]["target_interior_mask_path"],
                entry["supervision"]["target_interior_mask_sha256"],
            ),
        ]
        paths.extend((row["path"], row["sha256"]) for row in entry["supervision"]["anchor_masks"])
        for relative, expected in paths:
            path = safe_resolve(WORKSPACE, relative)
            if not path.is_file() or sha256_file(path) != expected:
                all_files_valid = False
        target = cv2.imread(str(safe_resolve(WORKSPACE, entry["supervision"]["target_mask_path"])), cv2.IMREAD_GRAYSCALE)
        interior = cv2.imread(
            str(safe_resolve(WORKSPACE, entry["supervision"]["target_interior_mask_path"])), cv2.IMREAD_GRAYSCALE
        )
        if target is None or interior is None or target.shape != (480, 640) or interior.shape != (480, 640):
            mask_invariants = False
        else:
            target_binary = target > 0
            interior_binary = interior > 0
            mask_invariants &= bool(np.all(~interior_binary | target_binary))
            state = entry["supervision"]["answerability_state"]
            if state == "FOUND":
                mask_invariants &= bool(target_binary.any()) and entry["audit_only"]["valid_target_count"] == 1
            elif state == "ABSENT":
                mask_invariants &= not bool(target_binary.any()) and entry["audit_only"]["valid_target_count"] == 0
            elif state == "AMBIGUOUS":
                mask_invariants &= bool(target_binary.any()) and entry["audit_only"]["valid_target_count"] >= 2
        split = entry["split"]
        state_counts[split][entry["supervision"]["answerability_state"]] += 1
        source_labels.update(entry["supervision"]["source_labels"])
        for source in entry["supervision"]["source_labels"]:
            source_counts[split][source] += 1
    dataset_root = safe_resolve(WORKSPACE, manifest["dataset_root"])
    full_qc = read_json(dataset_root / "DATASET_V2_1_DEVELOPMENT_FULL_QC_REPORT.json")
    current = protected_now(manifest, config)
    expected = manifest["protected_inputs"]
    protected_match = (
        current["dataset_index_sha256"] == expected["dataset_index_sha256"]
        and current["family_manifest_sha256"] == expected["family_manifest_sha256"]
        and current["full_qc_report_sha256"] == expected["full_qc_report_sha256"]
        and current["dataset_tree_sha256"] == expected["dataset_tree_sha256"]
        and current["model_inventory_sha256"] == expected["model_inventory_sha256"]
        and current["wp3_locked_baseline_artifacts"] == expected["wp3_locked_baseline_artifacts"]
    )
    checks = {
        "protocol_and_full_qc_authorize_training": manifest.get("protocol_id") == PROTOCOL_ID
        and full_qc.get("passed") is True
        and full_qc.get("decision") == "GO_DEVELOPMENT_TRAIN"
        and full_qc.get("training_authorized_as_next_gate") is True,
        "calibration_and_tests_sealed": full_qc.get("calibration_or_test_created") is False,
        "exact_train_dev_counts": len(train) == 1600
        and len(dev) == 400
        and len(train_families) == 320
        and len(dev_families) == 80,
        "family_split_disjoint": not bool(train_families & dev_families),
        "unique_and_aligned_sample_ids": len(sample_ids) == len(set(sample_ids)) == 2000
        and sample_ids == feature_sample_ids,
        "all_locked_artifact_hashes_valid": all_files_valid,
        "state_mask_invariants": mask_invariants,
        "feature_manifest_oracle_free": not forbidden_cache_findings(feature_manifest),
        "prompt_only_relation_parser_matches_2000": relations_match,
        "only_four_supported_source_labels": source_labels == set(SOURCE_CLASSES),
        "spatial_source_not_fabricated": "spatial" not in source_labels
        and config["model"]["deferred_source_classes"].get("spatial")
        == "NO_POSITIVE_SUPERVISION_IN_DATASET_V2_1_1",
        "ood_heldout_assets_absent": full_qc["asset_partition_qc"]["heldout_active"] == []
        and full_qc["asset_partition_qc"]["excluded_active"] == [],
        "train_dev_leakage_qc_pass": full_qc["train_dev_leakage_qc"]["passed"] is True
        and all(value == 0 for value in full_qc["train_dev_leakage_qc"]["cross_split_overlap_counts"].values()),
        "dataset_baseline_model_protected_hashes_match": protected_match,
    }
    observed = {
        "train_families": len(train_families),
        "dev_families": len(dev_families),
        "train_samples": len(train),
        "dev_samples": len(dev),
        "record_answerability_counts": state_counts,
        "source_positive_counts": source_counts,
        "unique_feature_pairs": feature_manifest["unique_feature_pair_count"],
        "relations_matched": sum(
            parse_relation(entry["feature_input"]["prompt"]) == entry["audit_only"]["relation"] for entry in entries
        ),
        "protected_observed": current,
    }
    return checks, observed


def select_preflight_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    train = [entry for entry in entries if entry["split"] == "train"]
    for state in ANSWERABILITY_CLASSES:
        candidates = [entry for entry in train if entry["supervision"]["answerability_state"] == state]
        candidates.sort(key=lambda entry: sha256_file(safe_resolve(WORKSPACE, entry["record_path"])))
        selected.append(candidates[0])
    return selected


def one_batch_no_step(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise DevelopmentTrainingError("CUDA is required for the locked development preflight")
    device = torch.device("cuda")
    selected = select_preflight_entries(manifest["entries"])
    seed_runtime(int(config["seed"]), strict=False)
    model_root = safe_resolve(WORKSPACE, config["feature_input"]["model_root"])
    baseline = load_model(model_root)
    pooled = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for entry in selected:
        extracted = extract_features(
            baseline,
            safe_resolve(WORKSPACE, entry["feature_input"]["rgb_path"]),
            safe_resolve(WORKSPACE, entry["feature_input"]["depth_path"]),
        )
        r_grid, r_thumb = pool_raw_feature(extracted["r0"])
        d_grid, d_thumb = pool_raw_feature(extracted["d0"])
        pooled.append((r_grid.half().float(), d_grid.half().float(), r_thumb.half().float(), d_thumb.half().float()))
        del extracted
    del baseline
    torch.cuda.empty_cache()
    language = config["language"]
    masks = []
    tokens = []
    relation_ids = []
    answers = []
    sources = []
    for entry in selected:
        prompt = entry["feature_input"]["prompt"]
        tokens.append(tokenize(prompt, int(language["max_tokens"]), int(language["vocab_size"])))
        relation_ids.append(language["relations"].index(parse_relation(prompt)))
        answers.append(entry["supervision"]["answerability_index"])
        sources.append(entry["supervision"]["source_multihot"])
        mask = cv2.imread(str(safe_resolve(WORKSPACE, entry["supervision"]["target_mask_path"])), cv2.IMREAD_GRAYSCALE)
        masks.append(cv2.resize((mask > 0).astype(np.float32), (32, 24), interpolation=cv2.INTER_AREA))
    batch: dict[str, Any] = {
        "r0": torch.stack([row[0] for row in pooled]).to(device),
        "d0": torch.stack([row[1] for row in pooled]).to(device),
        "r_thumb": torch.stack([row[2] for row in pooled]).to(device),
        "d_thumb": torch.stack([row[3] for row in pooled]).to(device),
        "tokens": torch.tensor(tokens, dtype=torch.long, device=device),
        "relation_ids": torch.tensor(relation_ids, dtype=torch.long, device=device),
        "answer_targets": torch.tensor(answers, dtype=torch.long, device=device),
        "source_targets": torch.tensor(sources, dtype=torch.float32, device=device),
        "heatmap_targets": torch.tensor(np.stack(masks), dtype=torch.float32, device=device),
    }
    for name in ["r0", "d0", "r_thumb", "d_thumb"]:
        batch[name].requires_grad_(False)
    seed_runtime(int(config["seed"]), strict=True)
    sidecar = PCRAUDevelopmentV1(config).to(device)
    train_entries = [entry for entry in manifest["entries"] if entry["split"] == "train"]
    answer_weights, source_weights = global_class_weights(train_entries, device)
    strict_inputs = model_inputs(batch)
    output = sidecar(strict_inputs)
    losses = compute_loss(output, batch, answer_weights, source_weights, config)
    losses["total"].backward()
    gradients = gradient_groups(sidecar)
    finite = all(bool(torch.isfinite(value).item()) for value in losses.values()) and all(
        math.isfinite(value) for value in gradients.values()
    )
    no_feature_grad = all(batch[name].grad is None for name in ["r0", "d0", "r_thumb", "d_thumb"])
    return {
        "optimizer_created": False,
        "optimizer_step_performed": False,
        "selected_sample_ids": [entry["sample_id"] for entry in selected],
        "selected_answerability_states": [entry["supervision"]["answerability_state"] for entry in selected],
        "strict_model_input_keys": sorted(strict_inputs),
        "batch_shapes": {name: list(value.shape) for name, value in strict_inputs.items()},
        "output_shapes": {name: list(value.shape) for name, value in output.items() if torch.is_tensor(value)},
        "losses": {name: float(value.item()) for name, value in losses.items()},
        "gradient_norms": gradients,
        "all_finite": finite,
        "all_active_groups_nonzero": all(value > float(config["gates"]["min_active_gradient_norm"]) for value in gradients.values()),
        "frozen_feature_no_grad": no_feature_grad,
        "answer_weights_from_train_only": answer_weights.detach().cpu().tolist(),
        "source_pos_weights_from_train_only": source_weights.detach().cpu().tolist(),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }


def write_markdown(report: dict[str, Any]) -> None:
    checks = report["checks"]
    batch = report["one_batch_no_step"]
    lines = [
        "# P-CRA-U development training — preflight report",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Static checks: `{sum(checks.values())}/{len(checks)}`",
        f"- Dataset: `{report['observed']['train_families']} train + {report['observed']['dev_families']} dev families`",
        f"- Samples: `{report['observed']['train_samples']} train + {report['observed']['dev_samples']} dev`",
        f"- Prompt-only relation parser: `{report['observed']['relations_matched']}/2000`",
        f"- Unique RGB-depth pairs: `{report['observed']['unique_feature_pairs']}`",
        f"- One-batch loss: `{batch['losses']['total']:.6f}`; optimizer step: `NO`",
        f"- One-batch peak VRAM: `{batch['peak_vram_bytes'] / 2**30:.3f} GiB`",
        "- Calibration/Test: `SEALED`; Tables 2–6: `NOT_RUN`",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(f"- `{name}`: `{'PASS' if value else 'FAIL'}`" for name, value in checks.items())
    lines += ["", "## Active gradient norms", ""]
    lines.extend(f"- `{name}`: `{value:.9e}`" for name, value in batch["gradient_norms"].items())
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    manifest = read_json(MANIFEST_PATH)
    feature_manifest = read_json(FEATURE_MANIFEST_PATH)
    if read_json(SEED_LOCK_PATH).get("status") != "LOCKED_BEFORE_PREFLIGHT":
        raise DevelopmentTrainingError("Seed lock missing")
    before = protected_now(manifest, config)
    checkpoint_test = subprocess.run(
        [str(WORKSPACE / ".conda-roborefer/bin/python"), str(WORKSPACE / "protocol/test_training_checkpoint_manager.py")],
        cwd=WORKSPACE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    checks, observed = static_checks(manifest, feature_manifest, config)
    checks["checkpoint_manager_unit_tests_pass"] = checkpoint_test.returncode == 0
    batch = one_batch_no_step(manifest, config)
    checks["one_batch_loss_and_logits_finite"] = batch["all_finite"]
    checks["one_batch_all_active_gradients_nonzero"] = batch["all_active_groups_nonzero"]
    checks["frozen_features_no_gradient"] = batch["frozen_feature_no_grad"]
    checks["no_optimizer_step_in_preflight"] = not batch["optimizer_created"] and not batch["optimizer_step_performed"]
    after = protected_now(manifest, config)
    checks["protected_inputs_unchanged_during_preflight"] = before == after
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "passed": passed,
        "decision": "GO_DEVELOPMENT_TRAIN_CANARY" if passed else "FIX_DEVELOPMENT_TRAIN_PREFLIGHT_FIRST",
        "checks": checks,
        "observed": observed,
        "one_batch_no_step": batch,
        "checkpoint_manager_test": {
            "returncode": checkpoint_test.returncode,
            "output": checkpoint_test.stdout,
        },
        "protected_before": before,
        "protected_after": after,
        "training_performed": False,
        "calibration_fit": False,
        "test_opened": False,
        "tables_02_to_06": "NOT_RUN",
    }
    write_json(REPORT_JSON, report)
    write_markdown(report)
    if passed:
        artifact_paths = [
            WORKSPACE / "protocol/PCRA_U_DEVELOPMENT_TRAIN_CONTRACT.md",
            CONFIG_PATH,
            MANIFEST_PATH,
            FEATURE_MANIFEST_PATH,
            SEED_LOCK_PATH,
            WORKSPACE / "protocol/PCRA_U_DEVELOPMENT_CHECKPOINT_POLICY.md",
            WORKSPACE / "protocol/PCRA_U_DEVELOPMENT_PREFLIGHT_AMENDMENT_01.md",
            WORKSPACE / "protocol/pcra_u_development_common.py",
            WORKSPACE / "protocol/pcra_u_development_feature_cache.py",
            WORKSPACE / "protocol/pcra_u_development_feature_cache.schema.json",
            WORKSPACE / "protocol/prepare_pcra_u_development_train.py",
            WORKSPACE / "protocol/pcra_u_development_train.py",
            WORKSPACE / "protocol/run_pcra_u_development.sh",
            Path(__file__).resolve(),
            WORKSPACE / "protocol/training_checkpoint_manager.py",
            WORKSPACE / "protocol/test_training_checkpoint_manager.py",
            REPORT_JSON,
            REPORT_MD,
        ]
        lock = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "LOCKED_AFTER_PREFLIGHT_BEFORE_CANARY_OPTIMIZER_STEP",
            "created_at_utc": utc_now(),
            "allowed_next_action": "GO_DEVELOPMENT_TRAIN_CANARY",
            "artifacts": file_hashes(artifact_paths),
            "protected_inputs": manifest["protected_inputs"],
            "split_policy": manifest["split_policy"],
            "training_performed": False,
            "calibration_fit": False,
            "test_opened": False,
        }
        write_json(EXECUTION_LOCK, lock)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = run()
    print(json.dumps({"passed": report["passed"], "decision": report["decision"], "checks": report["checks"]}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
