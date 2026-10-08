"""Extract the locked 32-family Day-5 h_spatial development cache."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VLM = ROOT / "RoboRefer"
MODEL = VLM / "models/RoboRefer-2B-SFT"
LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY5_RUN_LOCK.json"
OUT = ROOT / "ketqua1/03_backbone_h_spatial/ngay_05"
CACHE = OUT / "cache"
MANIFEST = OUT / "FEATURE_CACHE_MANIFEST.json"
QC = OUT / "CACHE_QC.json"

sys.path.insert(0, str(VLM))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_lock() -> tuple[dict, list[dict]]:
    lock = json.loads(LOCK.read_text())
    if lock["status"] != "FROZEN_BEFORE_DAY5_MODEL_FORWARD":
        raise RuntimeError("Unexpected Day-5 lock status")
    for reference in lock["prerequisites"] + [lock["cache_input_manifest"], lock["supervision_store"]]:
        if sha256(ROOT / reference["path"]) != reference["sha256"]:
            raise RuntimeError(f"Locked input drift: {reference['path']}")
    rows = read_jsonl(ROOT / lock["cache_input_manifest"]["path"])
    if len(rows) != lock["cache_acceptance"]["expected_samples"]:
        raise RuntimeError("Cache input count drift")
    for row in rows:
        for key in ("rgb", "depth_view", "metric_depth"):
            path = ROOT / row[f"{key}_path"]
            if sha256(path) != row[f"{key}_sha256"]:
                raise RuntimeError(f"Media hash drift: {path}")
        if digest(row["instruction"]) != row["instruction_sha256"]:
            raise RuntimeError("Instruction drift")
    return lock, rows


def load_model(torch):
    from transformers import AutoConfig
    from llava.constants import DEFAULT_DEPTH_TOKEN
    from llava.model.language_model.llava_llama import LlavaLlamaModel

    config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True)
    config.resume_path = str(MODEL)
    config.model_dtype = "torch.bfloat16"
    model = LlavaLlamaModel(
        config=config, device_map="cuda:0", attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.tokenizer.media_token_ids["depth"] = model.tokenizer.convert_tokens_to_ids(DEFAULT_DEPTH_TOKEN)
    model.tokenizer.media_tokens["depth"] = DEFAULT_DEPTH_TOKEN
    model.to(device="cuda:0", dtype=torch.bfloat16)
    model.llm.to(dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return model


def main() -> None:
    if MANIFEST.exists() or QC.exists() or CACHE.exists():
        raise FileExistsError("Day-5 cache artifacts are append-only")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("Use isolated environment with PYTHONNOUSERSITE=1")
    import torch
    from workspace.mh_pcrau_v3.hidden_state_adapter import prepare_rgbd_prompt, extract_h_spatial

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for canonical RoboRefer cache extraction")
    torch.manual_seed(25092026)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    lock, rows = verify_lock()
    CACHE.mkdir(parents=False)
    load_start = time.perf_counter()
    model = load_model(torch)
    load_seconds = time.perf_counter() - load_start
    tokenizer_hash = digest(model.tokenizer.get_vocab())
    chat_template_hash = digest(model.tokenizer.chat_template)
    torch.cuda.reset_peak_memory_stats()
    records = []
    timing_rows = ["sample_id,seconds,peak_allocated_mib,peak_reserved_mib\n"]
    finite_values = 0
    total_values = 0
    roundtrip_max = 0.0
    for index, row in enumerate(rows):
        prompt = prepare_rgbd_prompt(
            model, rgb_path=ROOT / row["rgb_path"],
            depth_path=ROOT / row["depth_view_path"], instruction=row["instruction"],
        )
        mask = torch.ones_like(prompt.input_ids, dtype=torch.bool)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = extract_h_spatial(
            model, prompt.input_ids.unsqueeze(0), prompt.media, mask.unsqueeze(0),
            prompt.media_config,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        feature = result.h_spatial[0].detach().cpu().contiguous()
        if feature.shape != (1536,) or feature.dtype != torch.float32:
            raise RuntimeError(f"Feature contract violation: {feature.shape}/{feature.dtype}")
        shard = CACHE / f"{row['sample_id']}.pt"
        torch.save(feature, shard)
        restored = torch.load(shard, map_location="cpu", weights_only=True)
        difference = float((restored - feature).abs().max())
        roundtrip_max = max(roundtrip_max, difference)
        finite = int(torch.isfinite(feature).sum().item())
        finite_values += finite
        total_values += feature.numel()
        peak_allocated = torch.cuda.max_memory_allocated() / 2**20
        peak_reserved = torch.cuda.max_memory_reserved() / 2**20
        records.append({
            "sample_id": row["sample_id"], "family_id": row["family_id"],
            "feature_path": str(shard.relative_to(ROOT)), "feature_file_sha256": sha256(shard),
            "feature_raw_sha256": tensor_hash(feature), "shape": list(feature.shape),
            "dtype": str(feature.dtype), "finite_values": finite,
            "prompt_render_sha256": digest(prompt.prompt_text),
            "input_token_ids_sha256": digest(prompt.input_ids.tolist()),
            "pre_insertion_tokens": int(prompt.input_ids.numel()),
            "post_insertion_tokens": int(result.post_insertion_sequence_lengths[0]),
            "last_prompt_index": int(result.last_prompt_indices[0]),
            "roundtrip_max_abs": difference,
        })
        timing_rows.append(f"{row['sample_id']},{seconds:.9f},{peak_allocated:.6f},{peak_reserved:.6f}\n")
        if (index + 1) % 4 == 0:
            print(f"CACHE_PROGRESS {index + 1}/{len(rows)}", flush=True)
    timing_path = OUT / "CACHE_EXTRACTION_TIMING.csv"
    timing_path.write_text("".join(timing_rows))
    feature_hashes = {record["feature_raw_sha256"] for record in records}
    manifest = {
        "schema_version": "1.0", "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_lock": {"path": str(LOCK.relative_to(ROOT)), "sha256": sha256(LOCK)},
        "cache_input_manifest": lock["cache_input_manifest"],
        "model_checkpoint": str(MODEL.relative_to(ROOT)),
        "model_inventory_sha256": lock["model_precision_policy"]["checkpoint_inventory_sha256"],
        "precision_policy": lock["model_precision_policy"],
        "hidden_state_adapter": {
            "path": "workspace/mh_pcrau_v3/hidden_state_adapter.py",
            "sha256": sha256(ROOT / "workspace/mh_pcrau_v3/hidden_state_adapter.py")
        },
        "tokenizer_vocab_sha256": tokenizer_hash,
        "chat_template_sha256": chat_template_hash,
        "model_load_seconds": load_seconds,
        "records": records,
        "labels_in_feature_shards": False,
        "oracle_fields_passed_to_model": False,
        "model_input_allowlist": ["rgb_path", "depth_view_path", "instruction"],
        "timing_csv": {"path": str(timing_path.relative_to(ROOT)), "sha256": sha256(timing_path)},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    qc_checks = {
        "record_count_32": len(records) == 32,
        "unique_sample_ids_32": len({record["sample_id"] for record in records}) == 32,
        "unique_family_ids_32": len({record["family_id"] for record in records}) == 32,
        "shape_all_1536": all(record["shape"] == [1536] for record in records),
        "dtype_all_float32": all(record["dtype"] == "torch.float32" for record in records),
        "finite_fraction_one": finite_values == total_values,
        "roundtrip_exact": roundtrip_max == 0,
        "feature_hashes_unique": len(feature_hashes) == 32,
        "no_labels_in_cache_payload": True,
        "no_oracle_to_model": True,
    }
    qc = {
        "schema_version": "1.0", "status": "PASS" if all(qc_checks.values()) else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": qc_checks, "samples": len(records), "families": len({r['family_id'] for r in records}),
        "total_feature_values": total_values, "finite_values": finite_values,
        "roundtrip_max_abs": roundtrip_max, "unique_raw_feature_hashes": len(feature_hashes),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "manifest_sha256": sha256(MANIFEST),
    }
    QC.write_text(json.dumps(qc, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": qc["status"], "samples": len(records),
                      "peak_reserved_mib": qc["peak_reserved_mib"]}), flush=True)
    if qc["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
