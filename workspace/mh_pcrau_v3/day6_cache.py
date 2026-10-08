"""Extract the locked 320-family S1a cache, reusing verified Day-5 shards."""

from __future__ import annotations

from collections import Counter
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
LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY6_S1A_RUN_LOCK.json"
OUT = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06"
CACHE = OUT / "cache_new"
MANIFEST = OUT / "S1A_FEATURE_CACHE_MANIFEST.json"
QC = OUT / "S1A_CACHE_QC.json"
TIMING = OUT / "S1A_CACHE_EXTRACTION_TIMING.csv"
DAY5_MANIFEST = ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/FEATURE_CACHE_MANIFEST.json"
DAY5_INPUT = ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/CACHE_INPUT_MANIFEST.jsonl"

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
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_model(torch):
    from transformers import AutoConfig
    from llava.constants import DEFAULT_DEPTH_TOKEN
    from llava.model.language_model.llava_llama import LlavaLlamaModel

    config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True)
    config.resume_path = str(MODEL)
    config.model_dtype = "torch.bfloat16"
    model = LlavaLlamaModel(config=config, device_map="cuda:0", attn_implementation="eager", low_cpu_mem_usage=True)
    model.tokenizer.media_token_ids["depth"] = model.tokenizer.convert_tokens_to_ids(DEFAULT_DEPTH_TOKEN)
    model.tokenizer.media_tokens["depth"] = DEFAULT_DEPTH_TOKEN
    model.to(device="cuda:0", dtype=torch.bfloat16)
    model.llm.to(dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return model


def main() -> None:
    if MANIFEST.exists() or QC.exists() or TIMING.exists() or CACHE.exists():
        raise FileExistsError("Day-6 cache outputs are append-only")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("Use isolated environment with PYTHONNOUSERSITE=1")
    import torch
    from workspace.mh_pcrau_v3.hidden_state_adapter import extract_h_spatial, prepare_rgbd_prompt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for canonical cache extraction")
    lock = json.loads(LOCK.read_text())
    if lock["status"] != "FROZEN_BEFORE_DAY6_CACHE_OR_TRAINING":
        raise RuntimeError("Unexpected Day-6 lock")
    for ref in lock["prerequisites"]:
        if sha256(ROOT / ref["path"]) != ref["sha256"]:
            raise RuntimeError(f"Locked prerequisite drift: {ref['path']}")
    for key in ("input_manifest", "supervision_store", "split_audit"):
        ref = lock["data"][key]
        if sha256(ROOT / ref["path"]) != ref["sha256"]:
            raise RuntimeError(f"Locked Day-6 data drift: {ref['path']}")
    rows = read_jsonl(ROOT / lock["data"]["input_manifest"]["path"])
    if len(rows) != 320:
        raise RuntimeError("Expected 320 locked inputs")
    for row in rows:
        for name in ("rgb", "depth_view", "metric_depth"):
            path = ROOT / row[f"{name}_path"]
            if sha256(path) != row[f"{name}_sha256"]:
                raise RuntimeError(f"Media hash drift: {path}")
        if digest(row["instruction"]) != row["instruction_sha256"]:
            raise RuntimeError(f"Instruction drift: {row['sample_id']}")

    day5 = json.loads(DAY5_MANIFEST.read_text())
    day5_by_id = {row["sample_id"]: row for row in day5["records"]}
    day5_input_by_id = {row["sample_id"]: row for row in read_jsonl(DAY5_INPUT)}
    if len(day5_by_id) != 32 or set(day5_by_id) != set(day5_input_by_id):
        raise RuntimeError("Day-5 reuse manifest drift")
    current_by_id = {row["sample_id"]: row for row in rows}
    for sample_id, old in day5_input_by_id.items():
        current = current_by_id.get(sample_id)
        if current is None:
            raise RuntimeError(f"Day-5 sample absent from Day-6: {sample_id}")
        for field in ("family_id", "rgb_sha256", "depth_view_sha256", "metric_depth_sha256", "instruction_sha256"):
            if current[field] != old[field]:
                raise RuntimeError(f"Day-5 reuse input drift: {sample_id}/{field}")

    torch.manual_seed(lock["seed"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    CACHE.mkdir(parents=True)
    load_started = time.perf_counter()
    model = load_model(torch)
    load_seconds = time.perf_counter() - load_started
    tokenizer_hash = digest(model.tokenizer.get_vocab())
    template_hash = digest(model.tokenizer.chat_template)
    if tokenizer_hash != day5["tokenizer_vocab_sha256"] or template_hash != day5["chat_template_sha256"]:
        raise RuntimeError("Tokenizer/chat template drift from G2 cache")
    torch.cuda.reset_peak_memory_stats()

    records: list[dict] = []
    timing_rows: list[dict] = []
    total_values = 0
    finite_values = 0
    roundtrip_max = 0.0
    new_count = 0
    reuse_count = 0
    new_seconds: list[float] = []
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in day5_by_id:
            prior = day5_by_id[sample_id]
            shard = ROOT / prior["feature_path"]
            if sha256(shard) != prior["feature_file_sha256"]:
                raise RuntimeError(f"Reused shard drift: {sample_id}")
            feature = torch.load(shard, map_location="cpu", weights_only=True).contiguous()
            if tensor_hash(feature) != prior["feature_raw_sha256"]:
                raise RuntimeError(f"Reused raw feature drift: {sample_id}")
            record = dict(prior)
            record["split"] = row["split"]
            record["cache_origin"] = "DAY5_VERIFIED_REUSE"
            seconds = 0.0
            reuse_count += 1
        else:
            prompt = prepare_rgbd_prompt(
                model,
                rgb_path=ROOT / row["rgb_path"],
                depth_path=ROOT / row["depth_view_path"],
                instruction=row["instruction"],
            )
            mask = torch.ones_like(prompt.input_ids, dtype=torch.bool)
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = extract_h_spatial(
                model, prompt.input_ids.unsqueeze(0), prompt.media, mask.unsqueeze(0), prompt.media_config
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            new_seconds.append(seconds)
            feature = result.h_spatial[0].detach().cpu().contiguous()
            shard = CACHE / f"{sample_id}.pt"
            torch.save(feature, shard)
            restored = torch.load(shard, map_location="cpu", weights_only=True)
            difference = float((restored - feature).abs().max().item())
            record = {
                "sample_id": sample_id,
                "family_id": row["family_id"],
                "split": row["split"],
                "feature_path": str(shard.relative_to(ROOT)),
                "feature_file_sha256": sha256(shard),
                "feature_raw_sha256": tensor_hash(feature),
                "shape": list(feature.shape),
                "dtype": str(feature.dtype),
                "finite_values": int(torch.isfinite(feature).sum().item()),
                "prompt_render_sha256": digest(prompt.prompt_text),
                "input_token_ids_sha256": digest(prompt.input_ids.tolist()),
                "pre_insertion_tokens": int(prompt.input_ids.numel()),
                "post_insertion_tokens": int(result.post_insertion_sequence_lengths[0]),
                "last_prompt_index": int(result.last_prompt_indices[0]),
                "roundtrip_max_abs": difference,
                "cache_origin": "DAY6_NEW_EXTRACTION",
            }
            new_count += 1
            if new_count % 8 == 0:
                print(f"DAY6_CACHE_PROGRESS new={new_count}/288 total={new_count + reuse_count}/320", flush=True)
        if feature.shape != (1536,) or feature.dtype != torch.float32:
            raise RuntimeError(f"Feature contract violation: {sample_id}")
        finite = int(torch.isfinite(feature).sum().item())
        total_values += feature.numel()
        finite_values += finite
        roundtrip_max = max(roundtrip_max, float(record["roundtrip_max_abs"]))
        records.append(record)
        timing_rows.append({
            "sample_id": sample_id,
            "split": row["split"],
            "cache_origin": record["cache_origin"],
            "seconds": f"{seconds:.9f}",
            "peak_allocated_mib": f"{torch.cuda.max_memory_allocated() / 2**20:.6f}",
            "peak_reserved_mib": f"{torch.cuda.max_memory_reserved() / 2**20:.6f}",
        })

    with TIMING.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)
    split_counts = Counter(record["split"] for record in records)
    manifest = {
        "schema_version": "1.0",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_lock": {"path": str(LOCK.relative_to(ROOT)), "sha256": sha256(LOCK)},
        "input_manifest": lock["data"]["input_manifest"],
        "model_checkpoint": str(MODEL.relative_to(ROOT)),
        "model_inventory_sha256": day5["model_inventory_sha256"],
        "precision_policy": day5["precision_policy"],
        "hidden_state_adapter": day5["hidden_state_adapter"],
        "tokenizer_vocab_sha256": tokenizer_hash,
        "chat_template_sha256": template_hash,
        "model_load_seconds": load_seconds,
        "records": records,
        "split_counts": dict(split_counts),
        "reuse_count": reuse_count,
        "new_extraction_count": new_count,
        "labels_in_feature_shards": False,
        "oracle_fields_passed_to_model": False,
        "model_input_allowlist": ["rgb_path", "depth_view_path", "instruction"],
        "timing_csv": {"path": str(TIMING.relative_to(ROOT)), "sha256": sha256(TIMING)},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    feature_hashes = {record["feature_raw_sha256"] for record in records}
    checks = {
        "records_320": len(records) == 320,
        "families_320": len({record["family_id"] for record in records}) == 320,
        "train_256_val_64": split_counts == Counter({"train_uq": 256, "val_uq": 64}),
        "reuse_32_new_288": reuse_count == 32 and new_count == 288,
        "shape_all_1536": all(record["shape"] == [1536] for record in records),
        "dtype_all_float32": all(record["dtype"] == "torch.float32" for record in records),
        "finite_fraction_one": finite_values == total_values,
        "roundtrip_exact": roundtrip_max == 0.0,
        "feature_hashes_unique_320": len(feature_hashes) == 320,
        "no_labels_in_cache_payload": True,
        "no_oracle_to_model": True,
    }
    qc = {
        "schema_version": "1.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "samples": len(records),
        "families": len({record["family_id"] for record in records}),
        "total_feature_values": total_values,
        "finite_values": finite_values,
        "roundtrip_max_abs": roundtrip_max,
        "unique_raw_feature_hashes": len(feature_hashes),
        "reuse_count": reuse_count,
        "new_extraction_count": new_count,
        "new_extraction_seconds": {
            "count": len(new_seconds),
            "minimum": min(new_seconds),
            "maximum": max(new_seconds),
            "mean": sum(new_seconds) / len(new_seconds),
            "median": statistics.median(new_seconds),
        },
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "manifest_sha256": sha256(MANIFEST),
    }
    QC.write_text(json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": qc["status"], "samples": 320, "reused": reuse_count, "new": new_count, "peak_reserved_mib": qc["peak_reserved_mib"]}), flush=True)
    if qc["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
