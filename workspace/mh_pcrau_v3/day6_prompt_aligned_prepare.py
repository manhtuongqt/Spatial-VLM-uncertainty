#!/usr/bin/env python3
"""Prepare a post-hoc prompt-aligned counterfactual for diagnosis only."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_02_paired_view"
SOURCE_INPUT = SOURCE_ROOT / "PAIRED_VIEW_INPUT_MANIFEST.jsonl"
SOURCE_SUPERVISION = SOURCE_ROOT / "PAIRED_VIEW_SUPERVISION.jsonl"
OLD_INPUT = ROOT / "ketqua1/07_huan_luyen/ngay_06/S1A_INPUT_MANIFEST.jsonl"
OUT = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_03_prompt_aligned"
LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY6_PROMPT_ALIGNED_DIAGNOSTIC_LOCK.json"
CHECKPOINT = ROOT / "workspace/mh_pcrau_v3/checkpoints/day06/s1a_best.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists() or LOCK.exists():
        raise FileExistsError("Prompt-aligned diagnostic is append-only")
    source = [json.loads(line) for line in SOURCE_INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    old = [json.loads(line) for line in OLD_INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    marker = "If exactly one requested target"
    suffixes = {row["instruction"][row["instruction"].index(marker):] for row in old if marker in row["instruction"]}
    if len(source) != 256 or len(suffixes) != 1:
        raise RuntimeError("Cannot establish the canonical output-contract suffix")
    suffix = next(iter(suffixes))
    OUT.mkdir(parents=True)
    aligned = []
    for row in source:
        instruction = row["instruction"].strip()
        if marker in instruction:
            raise RuntimeError("Source paired prompt unexpectedly already contains suffix")
        instruction = f"{instruction} {suffix}"
        aligned.append({**row, "instruction": instruction, "instruction_sha256": digest(instruction),
                        "prompt_variant": "POSTHOC_CANONICAL_OUTPUT_SUFFIX_ALIGNED"})
    input_path = OUT / "PROMPT_ALIGNED_INPUT_MANIFEST.jsonl"
    supervision_path = OUT / "PROMPT_ALIGNED_SUPERVISION.jsonl"
    input_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in aligned), encoding="utf-8")
    supervision_path.write_bytes(SOURCE_SUPERVISION.read_bytes())
    lock = {
        "schema_version": "1.0", "status": "FROZEN_BEFORE_COUNTERFACTUAL_FEATURE_EXTRACTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-hoc diagnostic after paired input-shift result was observed; never confirmatory evidence",
        "intervention": "Append the exact canonical Train-UQ output-contract suffix; RGB, depth and labels unchanged",
        "training_or_selection_allowed": False,
        "generalization_hold_must_remain": True,
        "checkpoint": {"path": str(CHECKPOINT.relative_to(ROOT)), "sha256": sha256(CHECKPOINT)},
        "source_input": {"path": str(SOURCE_INPUT.relative_to(ROOT)), "sha256": sha256(SOURCE_INPUT)},
        "input_manifest": {"path": str(input_path.relative_to(ROOT)), "sha256": sha256(input_path)},
        "supervision": {"path": str(supervision_path.relative_to(ROOT)), "sha256": sha256(supervision_path)},
        "source_code": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__).resolve())},
        "canonical_suffix_sha256": digest(suffix),
        "rows": 256,
    }
    write_json(LOCK, lock)
    print(json.dumps({"status": "PASS", "rows": len(aligned), "suffix_chars": len(suffix)}, indent=2))


if __name__ == "__main__":
    main()
