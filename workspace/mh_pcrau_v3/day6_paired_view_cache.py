#!/usr/bin/env python3
"""Extract frozen RoboRefer h_spatial for the 256 paired second views."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
VLM = ROOT / "RoboRefer"
MODEL = VLM / "models/RoboRefer-2B-SFT"
OUT = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_02_paired_view"
INPUT = OUT / "PAIRED_VIEW_INPUT_MANIFEST.jsonl"
PARENT_LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY6_PAIRED_VIEW_STRESS_LOCK.json"
EXEC_LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY6_PAIRED_VIEW_EXTRACTION_LOCK.json"
CACHE = OUT / "cache"
MANIFEST = OUT / "PAIRED_VIEW_FEATURE_CACHE_MANIFEST.json"
QC = OUT / "PAIRED_VIEW_CACHE_QC.json"
TIMING = OUT / "PAIRED_VIEW_CACHE_TIMING.csv"
DAY6_MANIFEST = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/S1A_FEATURE_CACHE_MANIFEST.json"

sys.path.insert(0, str(VLM))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_model(torch):
    from transformers import AutoConfig
    from llava.constants import DEFAULT_DEPTH_TOKEN
    from llava.model.language_model.llava_llama import LlavaLlamaModel

    config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True)
    config.resume_path = str(MODEL)
    config.model_dtype = "torch.bfloat16"
    model = LlavaLlamaModel(
        config=config, device_map="cuda:0", attn_implementation="eager", low_cpu_mem_usage=True
    )
    model.tokenizer.media_token_ids["depth"] = model.tokenizer.convert_tokens_to_ids(DEFAULT_DEPTH_TOKEN)
    model.tokenizer.media_tokens["depth"] = DEFAULT_DEPTH_TOKEN
    model.to(device="cuda:0", dtype=torch.bfloat16)
    model.llm.to(dtype=torch.float32)
    model.eval(); model.requires_grad_(False)
    return model


def main() -> None:
    if any(path.exists() for path in (CACHE, MANIFEST, QC, TIMING)):
        raise FileExistsError("Paired-view cache is append-only")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("Use the isolated RoboRefer environment")
    import torch
    from workspace.mh_pcrau_v3.hidden_state_adapter import extract_h_spatial, prepare_rgbd_prompt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    parent_lock = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    execution_lock = json.loads(EXEC_LOCK.read_text(encoding="utf-8"))
    if parent_lock["status"] != "FROZEN_BEFORE_PAIRED_VIEW_FEATURE_EXTRACTION":
        raise RuntimeError("Invalid paired-view parent lock")
    if execution_lock["status"] != "FROZEN_BEFORE_FEATURE_EXTRACTION":
        raise RuntimeError("Invalid paired-view extraction lock")
    for ref in execution_lock["bindings"]:
        if sha256(ROOT / ref["path"]) != ref["sha256"]:
            raise RuntimeError(f"Extraction binding drift: {ref['path']}")
    if sha256(INPUT) != parent_lock["input_manifest"]["sha256"]:
        raise RuntimeError("Input manifest drift")
    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 256:
        raise RuntimeError("Expected 256 paired views")
    for row in rows:
        for name in ("rgb", "depth_view", "metric_depth"):
            path = ROOT / row[f"{name}_path"]
            if sha256(path) != row[f"{name}_sha256"]:
                raise RuntimeError(f"Input hash drift: {row['sample_id']}/{name}")
        if digest(row["instruction"]) != row["instruction_sha256"]:
            raise RuntimeError(f"Instruction drift: {row['sample_id']}")

    canonical = json.loads(DAY6_MANIFEST.read_text(encoding="utf-8"))
    torch.manual_seed(26092026)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    load_started = time.perf_counter()
    model = load_model(torch)
    load_seconds = time.perf_counter() - load_started
    tokenizer_hash = digest(model.tokenizer.get_vocab())
    template_hash = digest(model.tokenizer.chat_template)
    if tokenizer_hash != canonical["tokenizer_vocab_sha256"] or template_hash != canonical["chat_template_sha256"]:
        raise RuntimeError("Tokenizer/chat template drift from canonical Day 6")

    CACHE.mkdir()
    torch.cuda.reset_peak_memory_stats()
    records = []
    timing = []
    seconds_all = []
    finite_values = 0
    total_values = 0
    for index, row in enumerate(rows, start=1):
        prompt = prepare_rgbd_prompt(
            model,
            rgb_path=ROOT / row["rgb_path"],
            depth_path=ROOT / row["depth_view_path"],
            instruction=row["instruction"],
        )
        mask = torch.ones_like(prompt.input_ids, dtype=torch.bool)
        torch.cuda.synchronize(); started = time.perf_counter()
        result = extract_h_spatial(
            model, prompt.input_ids.unsqueeze(0), prompt.media, mask.unsqueeze(0), prompt.media_config
        )
        torch.cuda.synchronize(); seconds = time.perf_counter() - started
        feature = result.h_spatial[0].detach().cpu().contiguous()
        if feature.shape != (1536,) or feature.dtype != torch.float32 or not bool(torch.isfinite(feature).all()):
            raise RuntimeError(f"Feature contract violation: {row['sample_id']}")
        path = CACHE / f"{row['sample_id']}.pt"
        torch.save(feature, path)
        restored = torch.load(path, map_location="cpu", weights_only=True)
        difference = float((restored - feature).abs().max().item())
        record = {
            "sample_id": row["sample_id"],
            "family_id": row["family_id"],
            "feature_path": str(path.relative_to(ROOT)),
            "feature_file_sha256": sha256(path),
            "feature_raw_sha256": tensor_hash(feature),
            "shape": list(feature.shape),
            "dtype": str(feature.dtype),
            "finite_values": int(torch.isfinite(feature).sum().item()),
            "prompt_render_sha256": digest(prompt.prompt_text),
            "input_token_ids_sha256": digest(prompt.input_ids.tolist()),
            "last_prompt_index": int(result.last_prompt_indices[0]),
            "roundtrip_max_abs": difference,
        }
        records.append(record)
        seconds_all.append(seconds)
        finite_values += record["finite_values"]
        total_values += feature.numel()
        timing.append({
            "sample_id": row["sample_id"], "seconds": f"{seconds:.9f}",
            "peak_allocated_mib": f"{torch.cuda.max_memory_allocated() / 2**20:.6f}",
            "peak_reserved_mib": f"{torch.cuda.max_memory_reserved() / 2**20:.6f}",
        })
        if index % 8 == 0:
            print(f"PAIRED_VIEW_CACHE_PROGRESS {index}/256", flush=True)

    with TIMING.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timing[0]))
        writer.writeheader(); writer.writerows(timing)
    manifest = {
        "schema_version": "1.0", "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_lock": {"path": str(PARENT_LOCK.relative_to(ROOT)), "sha256": sha256(PARENT_LOCK)},
        "execution_lock": {"path": str(EXEC_LOCK.relative_to(ROOT)), "sha256": sha256(EXEC_LOCK)},
        "model_checkpoint": str(MODEL.relative_to(ROOT)),
        "model_inventory_sha256": canonical["model_inventory_sha256"],
        "precision_policy": canonical["precision_policy"],
        "tokenizer_vocab_sha256": tokenizer_hash,
        "chat_template_sha256": template_hash,
        "model_load_seconds": load_seconds,
        "records": records,
        "labels_in_feature_shards": False,
        "oracle_fields_passed_to_model": False,
        "model_input_allowlist": ["rgb_path", "depth_view_path", "instruction"],
        "timing_csv": str(TIMING.relative_to(ROOT)),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks = {
        "records_256": len(records) == 256,
        "families_256": len({row["family_id"] for row in records}) == 256,
        "unique_feature_hashes_256": len({row["feature_raw_sha256"] for row in records}) == 256,
        "shape_all_1536": all(row["shape"] == [1536] for row in records),
        "dtype_all_float32": all(row["dtype"] == "torch.float32" for row in records),
        "finite_fraction_one": finite_values == total_values,
        "roundtrip_exact": all(row["roundtrip_max_abs"] == 0 for row in records),
        "no_labels_or_oracle_to_model": True,
    }
    qc = {
        "schema_version": "1.0", "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "finite_values": finite_values, "total_values": total_values,
        "seconds_min": min(seconds_all), "seconds_median": statistics.median(seconds_all),
        "seconds_mean": statistics.mean(seconds_all), "seconds_max": max(seconds_all),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "manifest_sha256": sha256(MANIFEST),
    }
    QC.write_text(json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": qc["status"], "records": len(records), "median_seconds": qc["seconds_median"], "peak_reserved_mib": qc["peak_reserved_mib"]}), flush=True)
    if qc["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
