#!/usr/bin/env python3
"""Run frozen RoboRefer-2B-SFT on the oracle-free Test-IID manifest.

Every completed response is appended and fsync'ed, so an interrupted 1,000
sample run can be resumed without repeating successful inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TEST_ROOT = ROOT / "new/test_iid"
DEFAULT_MANIFEST = TEST_ROOT / "protocol/test_iid_feature_manifest.json"
DEFAULT_OUTPUT = TEST_ROOT / "evaluation/roborefer_original"
EXPECTED_SAMPLES = 1000
EXPECTED_FAMILIES = 200

sys.path.insert(0, str(ROOT / "old/protocol"))
from wp1_common import (  # noqa: E402
    EXPECTED_MODEL_INVENTORY_SHA256,
    RANDOM_SEED,
    canonical_json_bytes,
    parse_point,
    query_model,
    sha256_bytes,
    sha256_file,
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def append_fsync(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())


def resolve_input(raw_path: str, dataset_root: Path) -> Path:
    candidate = (ROOT / raw_path).resolve()
    root = dataset_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise RuntimeError(f"Input escapes frozen Test-IID dataset: {raw_path}")
    if not candidate.is_file():
        raise RuntimeError(f"Missing frozen input: {candidate}")
    return candidate


def image_size(rgb_bytes: bytes) -> tuple[int, int]:
    image = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Cannot decode RGB input")
    height, width = image.shape[:2]
    if (width, height) != (640, 480):
        raise RuntimeError(f"Expected 640x480 RGB, got {width}x{height}")
    return width, height


def validate_manifest(manifest: dict, dataset_root: Path) -> list[dict]:
    entries = manifest.get("entries", [])
    if manifest.get("status") != "ORACLE_FREE_TEST_IID_FEATURE_INPUT_ONLY":
        raise RuntimeError(f"Feature manifest is not the oracle-free locked input: {manifest.get('status')}")
    if manifest.get("split") != "test_iid":
        raise RuntimeError("Feature manifest is not Test-IID")
    if len(entries) != EXPECTED_SAMPLES or manifest.get("sample_count") != EXPECTED_SAMPLES:
        raise RuntimeError("Expected exactly 1,000 Test-IID inference entries")
    if len({row["family_id"] for row in entries}) != EXPECTED_FAMILIES:
        raise RuntimeError("Expected exactly 200 independent Test-IID families")
    if any(row.get("split") != "test_iid" for row in entries):
        raise RuntimeError("Feature manifest contains a non-Test-IID row")
    if manifest.get("model_inventory_sha256") != EXPECTED_MODEL_INVENTORY_SHA256:
        raise RuntimeError("Frozen RoboRefer checkpoint inventory hash differs")
    if (ROOT / manifest["dataset_root"]).resolve() != dataset_root.resolve():
        raise RuntimeError("Dataset root differs from the frozen feature manifest")
    # This manifest was generated before opening evaluator-only supervision.
    forbidden = {"supervision", "audit_only", "target_mask_path", "answerability_state"}
    for row in entries:
        if forbidden.intersection(row):
            raise RuntimeError(f"Oracle field found in inference row {row.get('sample_id')}")
    return entries


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    predictions_path = output_dir / "predictions.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = (ROOT / manifest["dataset_root"]).resolve()
    entries = validate_manifest(manifest, dataset_root)

    existing = read_jsonl(predictions_path)
    by_id = {str(row["sample_id"]): row for row in existing}
    expected_ids = {str(row["sample_id"]) for row in entries}
    if len(by_id) != len(existing):
        raise RuntimeError("Duplicate sample_id in resumable output")
    if not set(by_id).issubset(expected_ids):
        raise RuntimeError("Resumable output contains an unknown sample")
    if any(
        row.get("protocol_id") != "roborefer_original_test_iid_v1"
        or row.get("method") != "ROBOREFER_ORIGINAL_RGB_D"
        or row.get("model_inventory_sha256") != EXPECTED_MODEL_INVENTORY_SHA256
        or row.get("random_seed") != RANDOM_SEED
        for row in existing
    ):
        raise RuntimeError("Existing output is incompatible with this frozen run")

    todo = [row for row in entries if row["sample_id"] not in by_id]
    if args.max_samples is not None:
        todo = todo[: args.max_samples]
    started = time.monotonic()
    for local_index, entry in enumerate(todo, 1):
        rgb_path = resolve_input(entry["rgb_path"], dataset_root)
        depth_path = resolve_input(entry["depth_path"], dataset_root)
        rgb_bytes, depth_bytes = rgb_path.read_bytes(), depth_path.read_bytes()
        if sha256_bytes(rgb_bytes) != entry["rgb_sha256"]:
            raise RuntimeError(f"RGB hash mismatch: {entry['sample_id']}")
        if sha256_bytes(depth_bytes) != entry["depth_sha256"]:
            raise RuntimeError(f"Depth hash mismatch: {entry['sample_id']}")
        if sha256_bytes(entry["prompt"].encode()) != entry["prompt_sha256"]:
            raise RuntimeError(f"Prompt hash mismatch: {entry['sample_id']}")
        width, height = image_size(rgb_bytes)
        answer, latency_ms, provenance = query_model(
            args.server_url,
            entry["prompt"],
            rgb_bytes,
            depth_bytes,
            args.timeout_sec,
            EXPECTED_MODEL_INVENTORY_SHA256,
        )
        parsed = parse_point(answer, width, height)
        row = {
            "schema_version": 1,
            "protocol_id": "roborefer_original_test_iid_v1",
            "scientific_status": "OFFICIAL_FROZEN_TEST_IID_BASELINE",
            "sample_id": entry["sample_id"],
            "family_id": entry["family_id"],
            "variant": entry["variant"],
            "method": "ROBOREFER_ORIGINAL_RGB_D",
            "enable_depth": True,
            "raw_answer": answer,
            **parsed,
            "latency_ms": latency_ms,
            "rgb_sha256": entry["rgb_sha256"],
            "depth_sha256": entry["depth_sha256"],
            "prompt_sha256": entry["prompt_sha256"],
            "model_inventory_sha256": provenance["model_fingerprint"]["inventory_sha256"],
            "generation_mode": provenance["generation_mode"],
            "generation_config": provenance["generation_config"],
            "random_seed": provenance["random_seed"],
            "response_sha256": provenance["response_sha256"],
            "oracle_or_annotation_read_by_runner": False,
            "robot_manipulation_performed": False,
        }
        append_fsync(predictions_path, row)
        by_id[row["sample_id"]] = row
        if local_index == 1 or local_index % args.progress_every == 0 or local_index == len(todo):
            elapsed = time.monotonic() - started
            rate = local_index / max(elapsed, 1e-9)
            eta = (len(todo) - local_index) / max(rate, 1e-9)
            print(
                f"RoboRefer original {len(by_id)}/{EXPECTED_SAMPLES} | "
                f"last={latency_ms:.0f} ms | ETA={eta / 60:.1f} min",
                flush=True,
            )

    rows = read_jsonl(predictions_path)
    completed = len(rows) == EXPECTED_SAMPLES and {r["sample_id"] for r in rows} == expected_ids
    summary = {
        "schema_version": 1,
        "status": "COMPLETE" if completed else "PARTIAL",
        "scientific_status": "OFFICIAL_FROZEN_TEST_IID_BASELINE" if completed else "INCOMPLETE",
        "samples": len(rows),
        "expected_samples": EXPECTED_SAMPLES,
        "family_count": len({r["family_id"] for r in rows}),
        "method": "ROBOREFER_ORIGINAL_RGB_D",
        "random_seed": RANDOM_SEED,
        "generation_mode": "greedy",
        "model_inventory_sha256": EXPECTED_MODEL_INVENTORY_SHA256,
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "predictions_path": str(predictions_path.relative_to(ROOT)),
        "predictions_sha256": sha256_file(predictions_path),
        "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rows])) if rows else None,
        "parse_status_counts": {
            name: sum(r["parse_status"] == name for r in rows)
            for name in sorted({r["parse_status"] for r in rows})
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
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
