#!/usr/bin/env python3
"""Re-run the frozen RoboRefer-2B-SFT RGB-D baseline on the locked dev set.

The inference phase deliberately reads only the oracle-free manifest.  It is
resumable because every successful response is fsync'ed to JSONL immediately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
OLD_PROTOCOL = WORKSPACE / "old" / "protocol"
sys.path.insert(0, str(OLD_PROTOCOL))

from wp1_common import (  # noqa: E402
    EXPECTED_MODEL_INVENTORY_SHA256,
    RANDOM_SEED,
    canonical_json_bytes,
    parse_point,
    query_model,
    sha256_bytes,
    sha256_file,
)


DEFAULT_MANIFEST = OLD_PROTOCOL / "pcra_u_development_eval_manifest.json"
DEFAULT_DATASET = WORKSPACE / "old" / "roborefer_dataset_v2_1_1_development_400_20260824"
DEFAULT_OUTPUT = WORKSPACE / "new" / "outputs" / "roborefer_original_b1_rerun_seed_8132026"
FORMER_DATASET_NAME = "roborefer_dataset_v2_1_1_development_400_20260824"


def resolve_dataset_path(raw_path: str, dataset_root: Path) -> Path:
    raw = Path(raw_path)
    if FORMER_DATASET_NAME not in raw.parts:
        raise ValueError(f"Path is not inside the locked development dataset: {raw_path}")
    suffix = raw.parts[raw.parts.index(FORMER_DATASET_NAME) + 1 :]
    resolved = dataset_root.joinpath(*suffix).resolve()
    root = dataset_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path escapes dataset root: {raw_path}")
    return resolved


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSONL at {path}:{number}: {exc}") from exc
    return rows


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_json_bytes(row))
        stream.flush()
        os.fsync(stream.fileno())


def check_image_size(rgb_bytes: bytes) -> tuple[int, int]:
    image = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Cannot decode RGB input")
    height, width = image.shape[:2]
    if (width, height) != (640, 480):
        raise RuntimeError(f"Expected 640x480 RGB input, got {width}x{height}")
    return width, height


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    predictions_path = output_dir / "predictions.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["inference_entries"]
    if len(entries) != 400 or any(entry.get("split") != "dev" for entry in entries):
        raise RuntimeError("Expected exactly 400 locked dev inference entries")
    if manifest["inference_boundary"].get("oracle_or_evaluator_fields_present_per_entry") is not False:
        raise RuntimeError("Inference manifest is not oracle-free")

    existing = load_rows(predictions_path)
    by_id = {str(row["sample_id"]): row for row in existing}
    if len(by_id) != len(existing):
        raise RuntimeError("Duplicate sample_id in resumable output")
    expected_ids = {str(entry["sample_id"]) for entry in entries}
    if not set(by_id).issubset(expected_ids):
        raise RuntimeError("Output contains a sample outside the locked manifest")
    if any(
        row.get("model_inventory_sha256") != EXPECTED_MODEL_INVENTORY_SHA256
        or row.get("random_seed") != RANDOM_SEED
        or row.get("method") != "B1"
        for row in existing
    ):
        raise RuntimeError("Existing output is incompatible with this locked run")

    todo = [entry for entry in entries if entry["sample_id"] not in by_id]
    if args.max_samples is not None:
        todo = todo[: args.max_samples]
    started = time.monotonic()
    for local_index, entry in enumerate(todo, 1):
        rgb_path = resolve_dataset_path(entry["rgb_path"], dataset_root)
        depth_path = resolve_dataset_path(entry["depth_path"], dataset_root)
        rgb_bytes, depth_bytes = rgb_path.read_bytes(), depth_path.read_bytes()
        if sha256_bytes(rgb_bytes) != entry["rgb_sha256"]:
            raise RuntimeError(f"RGB hash mismatch: {entry['sample_id']}")
        if sha256_bytes(depth_bytes) != entry["depth_sha256"]:
            raise RuntimeError(f"Depth hash mismatch: {entry['sample_id']}")
        if sha256_bytes(entry["baseline_prompt"].encode("utf-8")) != entry["baseline_prompt_sha256"]:
            raise RuntimeError(f"Prompt hash mismatch: {entry['sample_id']}")
        width, height = check_image_size(rgb_bytes)
        answer, latency_ms, provenance = query_model(
            args.server_url,
            entry["baseline_prompt"],
            rgb_bytes,
            depth_bytes,
            args.timeout_sec,
            EXPECTED_MODEL_INVENTORY_SHA256,
        )
        parsed = parse_point(answer, width, height)
        row = {
            "schema_version": 1,
            "protocol_id": "roborefer_original_b1_rerun_v1",
            "scientific_status": "EXPLORATORY_DEV_ONLY",
            "sample_id": entry["sample_id"],
            "family_id": entry["family_id"],
            "variant": entry["variant"],
            "method": "B1",
            "enable_depth": True,
            "raw_answer": answer,
            **parsed,
            "latency_ms": latency_ms,
            "rgb_sha256": entry["rgb_sha256"],
            "depth_sha256": entry["depth_sha256"],
            "baseline_prompt_sha256": entry["baseline_prompt_sha256"],
            "source_prompt_sha256": entry["source_prompt_sha256"],
            "model_inventory_sha256": provenance["model_fingerprint"]["inventory_sha256"],
            "generation_mode": provenance["generation_mode"],
            "generation_config": provenance["generation_config"],
            "random_seed": provenance["random_seed"],
            "response_sha256": provenance["response_sha256"],
            "oracle_or_annotation_read_by_runner": False,
            "robot_manipulation_performed": False,
        }
        append_row(predictions_path, row)
        by_id[row["sample_id"]] = row
        if local_index == 1 or local_index % args.progress_every == 0 or local_index == len(todo):
            elapsed = time.monotonic() - started
            rate = local_index / max(elapsed, 1e-9)
            remaining = (len(todo) - local_index) / max(rate, 1e-9)
            print(
                f"B1 progress {len(by_id)}/400 | current_run={local_index}/{len(todo)} "
                f"| last={latency_ms:.0f} ms | ETA={remaining/60:.1f} min",
                flush=True,
            )

    rows = load_rows(predictions_path)
    completed = len(rows) == len(entries) and {row["sample_id"] for row in rows} == expected_ids
    summary = {
        "schema_version": 1,
        "status": "COMPLETE" if completed else "PARTIAL",
        "samples": len(rows),
        "expected_samples": len(entries),
        "method": "B1_RGB_D",
        "random_seed": RANDOM_SEED,
        "generation_mode": "greedy",
        "model_inventory_sha256": EXPECTED_MODEL_INVENTORY_SHA256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "mean_latency_ms": float(np.mean([row["latency_ms"] for row in rows])) if rows else None,
        "parse_status_counts": {
            name: sum(row["parse_status"] == name for row in rows)
            for name in sorted({row["parse_status"] for row in rows})
        },
        "oracle_or_annotation_read_by_runner": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:25552")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
