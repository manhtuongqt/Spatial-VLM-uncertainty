#!/usr/bin/env python3
"""Validate and report the clean B1 grounding LoRA training artifact."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "datasets/D_tabletop_clean_v1/manifest.json"
DEFAULT_RUN_ROOT = ROOT / "results/spatial_vlm_refspatial_v1/wp7_b1_clean_grounding"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()

    release = json.loads(args.release.read_text())
    model_dir = args.run_root / "model"
    required = [
        model_dir / "adapter_config.json",
        model_dir / "adapter_model.safetensors",
        model_dir / "trainer_state.json",
    ]
    train_log = args.run_root / "train_full.log"
    missing = [str(path) for path in required if not path.is_file()]
    if not train_log.is_file():
        missing.append(str(train_log))
    if missing:
        raise FileNotFoundError(f"Incomplete B1 artifact; missing: {missing}")

    trainer = json.loads((model_dir / "trainer_state.json").read_text())
    adapter = json.loads((model_dir / "adapter_config.json").read_text())
    with safe_open(model_dir / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        adapter_keys = list(handle.keys())
        adapter_parameters = sum(handle.get_tensor(key).numel() for key in adapter_keys)
    train_count = release["data_counts"]["train"]
    gradient_accumulation = 4
    expected_steps = math.ceil(train_count / gradient_accumulation)
    completed_steps = int(trainer.get("global_step", -1))
    final_log = next(
        (row for row in reversed(trainer.get("log_history", [])) if "train_runtime" in row),
        None,
    )
    log_text = train_log.read_text(errors="replace")
    peak_match = re.search(r"\[GPU memory\] peak allocated/reserved\s+([0-9.]+)\s+([0-9.]+)", log_text)

    checks = {
        "clean_release_pass": release.get("status") == "CLEAN_B1_GROUNDING_RELEASE_PASS",
        "release_checks_passed": release.get("checks_passed") is True,
        "release_scope_b1_grounding_only": release.get("training_scope") == "B1_TARGET_GROUNDING_ONLY",
        "release_not_for_b2": release.get("not_for_b2") is True,
        "train_count_1500": train_count == 1500,
        "completed_expected_steps": completed_steps == expected_steps,
        "final_train_summary_present": final_log is not None,
        "gpu_peak_summary_present": peak_match is not None,
        "lora_rank_16": adapter.get("r") == 16,
        "lora_alpha_32": adapter.get("lora_alpha") == 32,
        "adapter_has_392_tensors": len(adapter_keys) == 392,
        "adapter_has_expected_parameter_count": adapter_parameters == 18_464_768,
        "adapter_is_llm_only": all(".llm." in key for key in adapter_keys),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"B1 artifact validation failed: {failed}")

    artifacts = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in required
    }
    artifacts["train_full.log"] = {"sha256": sha256(train_log), "bytes": train_log.stat().st_size}
    report = {
        "status": "CLEAN_B1_GROUNDING_TRAINING_PASS",
        "dataset": release["dataset"],
        "dataset_manifest_sha256": sha256(args.release),
        "dataset_file_sha256": release["files"],
        "base_model": "RoboRefer/models/RoboRefer-2B-SFT",
        "launcher_sha256": sha256(ROOT / "protocol/run_refspatial_clean_b1_lora.sh"),
        "adapter_scope": "LLM_LORA_ONLY",
        "training": {
            "train_samples": train_count,
            "epochs": 1,
            "optimizer_steps": completed_steps,
            "physical_batch_size": 1,
            "gradient_accumulation_steps": gradient_accumulation,
            "effective_batch_size": gradient_accumulation,
            "seed": 9092026,
            "data_seed": 9092026,
            "learning_rate": 2e-4,
            "quantization": "4-bit NF4, double quantization",
            "precision": "bfloat16",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "trainable_adapter_parameters": adapter_parameters,
            "adapter_tensor_count": len(adapter_keys),
            "final_summary": final_log,
            "cuda_peak_allocated_gib": float(peak_match.group(1)),
            "cuda_peak_reserved_gib": float(peak_match.group(2)),
        },
        "artifacts": artifacts,
        "checks": checks,
        "sam2_used": False,
        "b2_open": False,
        "gazebo_generalization_evidence": False,
    }
    args.run_root.mkdir(parents=True, exist_ok=True)
    json_path = args.run_root / "TRAINING_REPORT.json"
    md_path = args.run_root / "TRAINING_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    summary = final_log or {}
    md_path.write_text(
        "# Clean B1 grounding training report\n\n"
        f"Status: **{report['status']}**\n\n"
        f"- Dataset: `{report['dataset']}` ({train_count} samples / 1500 disjoint families)\n"
        f"- Optimizer steps: {completed_steps}/{expected_steps}; epoch: {trainer.get('epoch')}\n"
        f"- Train loss: {summary.get('train_loss')}; runtime: {summary.get('train_runtime')} seconds\n"
        f"- Adapter SHA-256: `{artifacts['adapter_model.safetensors']['sha256']}`\n"
        "- Scope: target-point grounding only; no rationale/relation supervision, no SAM2.\n"
        "- This artifact does not open B2 or establish Gazebo generalization.\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
