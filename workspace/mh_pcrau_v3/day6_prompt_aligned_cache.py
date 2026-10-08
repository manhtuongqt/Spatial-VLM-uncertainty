#!/usr/bin/env python3
"""Extract h_spatial for the post-hoc prompt-aligned counterfactual."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time

from workspace.mh_pcrau_v3.day6_paired_view_cache import digest, load_model, sha256, tensor_hash


ROOT = Path(__file__).resolve().parents[2]
VLM = ROOT / "RoboRefer"
OUT = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_03_prompt_aligned"
INPUT = OUT / "PROMPT_ALIGNED_INPUT_MANIFEST.jsonl"
PARENT_LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY6_PROMPT_ALIGNED_DIAGNOSTIC_LOCK.json"
EXEC_LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY6_PROMPT_ALIGNED_EXTRACTION_LOCK.json"
CACHE = OUT / "cache"
MANIFEST = OUT / "PROMPT_ALIGNED_FEATURE_CACHE_MANIFEST.json"
QC = OUT / "PROMPT_ALIGNED_CACHE_QC.json"
TIMING = OUT / "PROMPT_ALIGNED_CACHE_TIMING.csv"
CANONICAL = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/S1A_FEATURE_CACHE_MANIFEST.json"

sys.path.insert(0, str(VLM))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main() -> None:
    if any(path.exists() for path in (CACHE, MANIFEST, QC, TIMING)):
        raise FileExistsError("Prompt-aligned cache is append-only")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("Use isolated RoboRefer environment")
    import torch
    from workspace.mh_pcrau_v3.hidden_state_adapter import extract_h_spatial, prepare_rgbd_prompt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    parent = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    execution = json.loads(EXEC_LOCK.read_text(encoding="utf-8"))
    if parent["status"] != "FROZEN_BEFORE_COUNTERFACTUAL_FEATURE_EXTRACTION":
        raise RuntimeError("Invalid prompt-aligned parent lock")
    if execution["status"] != "FROZEN_BEFORE_FEATURE_EXTRACTION":
        raise RuntimeError("Invalid prompt-aligned execution lock")
    for ref in execution["bindings"]:
        if sha256(ROOT / ref["path"]) != ref["sha256"]:
            raise RuntimeError(f"Binding drift: {ref['path']}")
    if sha256(INPUT) != parent["input_manifest"]["sha256"]:
        raise RuntimeError("Input manifest drift")
    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 256:
        raise RuntimeError("Expected 256 prompt-aligned inputs")
    for row in rows:
        for name in ("rgb", "depth_view", "metric_depth"):
            path = ROOT / row[f"{name}_path"]
            if sha256(path) != row[f"{name}_sha256"]:
                raise RuntimeError(f"Media drift: {row['sample_id']}/{name}")
        if digest(row["instruction"]) != row["instruction_sha256"]:
            raise RuntimeError(f"Instruction drift: {row['sample_id']}")

    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    torch.manual_seed(26092026); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
    load_started = time.perf_counter(); model = load_model(torch); load_seconds = time.perf_counter() - load_started
    tokenizer_hash = digest(model.tokenizer.get_vocab()); template_hash = digest(model.tokenizer.chat_template)
    if tokenizer_hash != canonical["tokenizer_vocab_sha256"] or template_hash != canonical["chat_template_sha256"]:
        raise RuntimeError("Tokenizer/chat template drift")
    CACHE.mkdir(); torch.cuda.reset_peak_memory_stats()
    records = []; timings = []; seconds_all = []
    for index, row in enumerate(rows, start=1):
        prompt = prepare_rgbd_prompt(model, rgb_path=ROOT / row["rgb_path"], depth_path=ROOT / row["depth_view_path"], instruction=row["instruction"])
        mask = torch.ones_like(prompt.input_ids, dtype=torch.bool)
        torch.cuda.synchronize(); started = time.perf_counter()
        result = extract_h_spatial(model, prompt.input_ids.unsqueeze(0), prompt.media, mask.unsqueeze(0), prompt.media_config)
        torch.cuda.synchronize(); seconds = time.perf_counter() - started
        feature = result.h_spatial[0].detach().cpu().contiguous()
        if feature.shape != (1536,) or feature.dtype != torch.float32 or not bool(torch.isfinite(feature).all()):
            raise RuntimeError(f"Feature contract violation: {row['sample_id']}")
        path = CACHE / f"{row['sample_id']}.pt"; torch.save(feature, path)
        restored = torch.load(path, map_location="cpu", weights_only=True)
        records.append({
            "sample_id": row["sample_id"], "family_id": row["family_id"],
            "feature_path": str(path.relative_to(ROOT)), "feature_file_sha256": sha256(path),
            "feature_raw_sha256": tensor_hash(feature), "shape": list(feature.shape), "dtype": str(feature.dtype),
            "finite_values": int(torch.isfinite(feature).sum()), "roundtrip_max_abs": float((restored - feature).abs().max()),
            "prompt_render_sha256": digest(prompt.prompt_text), "input_token_ids_sha256": digest(prompt.input_ids.tolist()),
            "last_prompt_index": int(result.last_prompt_indices[0]),
        })
        seconds_all.append(seconds)
        timings.append({"sample_id": row["sample_id"], "seconds": f"{seconds:.9f}"})
        if index % 8 == 0:
            print(f"PROMPT_ALIGNED_CACHE_PROGRESS {index}/256", flush=True)
    with TIMING.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timings[0])); writer.writeheader(); writer.writerows(timings)
    manifest = {
        "schema_version": "1.0", "status": "COMPLETE", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_lock": {"path": str(PARENT_LOCK.relative_to(ROOT)), "sha256": sha256(PARENT_LOCK)},
        "execution_lock": {"path": str(EXEC_LOCK.relative_to(ROOT)), "sha256": sha256(EXEC_LOCK)},
        "model_checkpoint": "RoboRefer/models/RoboRefer-2B-SFT", "model_inventory_sha256": canonical["model_inventory_sha256"],
        "precision_policy": canonical["precision_policy"], "tokenizer_vocab_sha256": tokenizer_hash,
        "chat_template_sha256": template_hash, "model_load_seconds": load_seconds, "records": records,
        "labels_in_feature_shards": False, "oracle_fields_passed_to_model": False,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks = {
        "records_256": len(records) == 256, "families_256": len({row["family_id"] for row in records}) == 256,
        "unique_hashes_256": len({row["feature_raw_sha256"] for row in records}) == 256,
        "shape_dtype_finite": all(row["shape"] == [1536] and row["dtype"] == "torch.float32" and row["finite_values"] == 1536 for row in records),
        "roundtrip_exact": all(row["roundtrip_max_abs"] == 0 for row in records),
        "no_labels_or_oracle_to_model": True,
    }
    qc = {
        "schema_version": "1.0", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "seconds_median": statistics.median(seconds_all), "seconds_mean": statistics.mean(seconds_all),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "manifest_sha256": sha256(MANIFEST),
    }
    QC.write_text(json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": qc["status"], "records": len(records), "median_seconds": qc["seconds_median"], "peak_reserved_mib": qc["peak_reserved_mib"]}), flush=True)
    if qc["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
