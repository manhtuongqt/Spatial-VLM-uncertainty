#!/usr/bin/env python3
"""Prepare the existing QC-passed second views for an honest stress test.

The 256 views share parent families with S1a Train-UQ and therefore are not an
independent validation set.  They are useful only as a paired-view robustness
diagnostic while the fresh Anti-Shortcut Val v1 capture is pending.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03/gazebo_pilot_512_revision_v2/PILOT_512_SELECTED_INDEX.jsonl"
OUT = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_02_paired_view"
LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY6_PAIRED_VIEW_STRESS_LOCK.json"
CHECKPOINT = ROOT / "workspace/mh_pcrau_v3/checkpoints/day06/s1a_best.pt"
OLD_INPUT = ROOT / "ketqua1/07_huan_luyen/ngay_06/S1A_INPUT_MANIFEST.jsonl"
OLD_SUPERVISION = ROOT / "ketqua1/07_huan_luyen/ngay_06/S1A_SUPERVISION_STORE.jsonl"

RELATIONS = ("leftmost", "rightmost", "second_from_left", "second_from_right")
STATES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")


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


def depth_view(depth: np.ndarray) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(value) & (value >= 0.10) & (value <= 2.0)
    if int(valid.sum()) < 16:
        raise ValueError("too few valid metric-depth pixels")
    near, far = np.percentile(value[valid], [2, 98])
    if far - near < 1e-6:
        raise ValueError("metric depth has no usable range")
    inverse = 1 / np.maximum(np.clip(value, near, far), 1e-6)
    gray = np.clip((inverse - 1 / far) / max(1 / near - 1 / far, 1e-6) * 255, 0, 255).astype(np.uint8)
    gray[~valid] = 0
    return np.repeat(gray[..., None], 3, axis=-1)


def main() -> None:
    if OUT.exists() or LOCK.exists():
        raise FileExistsError("Paired-view preparation is append-only")
    for path in (SOURCE, CHECKPOINT, OLD_INPUT, OLD_SUPERVISION):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in source_rows if row["view_id"] != "historical_original"]
    if len(rows) != 256 or len({row["sample_id"] for row in rows}) != 256:
        raise RuntimeError("Expected exactly 256 unique second-view rows")
    if len({row["family_id"] for row in rows}) != 256:
        raise RuntimeError("Expected one second view per parent family")

    state_counts = Counter(row["supervision"]["answerability"] for row in rows)
    relation_counts = Counter(row["supervision"]["relation"] for row in rows)
    cell_counts = Counter((row["supervision"]["answerability"], row["supervision"]["relation"]) for row in rows)
    if state_counts != Counter({state: 64 for state in STATES}):
        raise RuntimeError(f"State balance drift: {state_counts}")
    if relation_counts != Counter({relation: 64 for relation in RELATIONS}):
        raise RuntimeError(f"Relation balance drift: {relation_counts}")
    if any(cell_counts[(state, relation)] != 16 for state in STATES for relation in RELATIONS):
        raise RuntimeError("State/relation cell imbalance")

    OUT.mkdir(parents=True)
    depth_dir = OUT / "depth_view"
    depth_dir.mkdir()
    input_rows = []
    supervision_rows = []
    for row in sorted(rows, key=lambda value: value["sample_id"]):
        sample_id = row["sample_id"]
        model_input = row["model_input"]
        rgb = ROOT / model_input["rgb_path"]
        metric_depth = ROOT / model_input["depth_path"]
        if not rgb.is_file() or not metric_depth.is_file():
            raise FileNotFoundError(f"Missing paired-view media: {sample_id}")
        depth = np.load(metric_depth, allow_pickle=False)
        rendered = depth_dir / f"{sample_id}.png"
        if not cv2.imwrite(str(rendered), depth_view(depth)):
            raise RuntimeError(f"Cannot write {rendered}")
        instruction = model_input["instruction"]
        input_rows.append({
            "sample_id": sample_id,
            "family_id": row["family_id"],
            "split": "paired_view_stress",
            "origin": row["origin"],
            "rgb_path": str(rgb.relative_to(ROOT)),
            "rgb_sha256": sha256(rgb),
            "depth_view_path": str(rendered.relative_to(ROOT)),
            "depth_view_sha256": sha256(rendered),
            "metric_depth_path": str(metric_depth.relative_to(ROOT)),
            "metric_depth_sha256": sha256(metric_depth),
            "instruction": instruction,
            "instruction_sha256": digest(instruction),
        })
        supervision = row["supervision"]
        state = supervision["answerability"]
        target = supervision["target_uv"] if state == "FOUND" else None
        if state == "FOUND" and (not isinstance(target, list) or len(target) != 2):
            raise RuntimeError(f"Invalid FOUND coordinate: {sample_id}")
        supervision_rows.append({
            "sample_id": sample_id,
            "family_id": row["family_id"],
            "relation": supervision["relation"],
            "answerability": state,
            "target_uv": target,
            "head_mask": {
                "relation": True,
                "answerability": True,
                "coordinate": state == "FOUND",
                "log_variance": state == "FOUND",
                "reasoning": False,
                "source": False,
                "confidence": False,
            },
            "label_source": "Day-3 revision-v2 geometry/QC; paired-view diagnostic only",
        })

    input_path = OUT / "PAIRED_VIEW_INPUT_MANIFEST.jsonl"
    supervision_path = OUT / "PAIRED_VIEW_SUPERVISION.jsonl"
    input_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in input_rows), encoding="utf-8")
    supervision_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in supervision_rows), encoding="utf-8")
    audit = {
        "schema_version": "1.0",
        "status": "PASS",
        "samples": 256,
        "families": 256,
        "state_counts": dict(state_counts),
        "relation_counts": dict(relation_counts),
        "found_coordinate_support": state_counts["FOUND"],
        "known_limitation": "All parent families overlap S1a Train-UQ; this is paired-view stress, not independent Val/Test.",
        "training_allowed": False,
        "model_selection_allowed": False,
    }
    write_json(OUT / "PAIRED_VIEW_SPLIT_AUDIT.json", audit)
    lock = {
        "schema_version": "1.0",
        "status": "FROZEN_BEFORE_PAIRED_VIEW_FEATURE_EXTRACTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Paired second-view robustness diagnostic only",
        "boundaries": {
            "independent_validation": False,
            "test": False,
            "calibration": False,
            "training": False,
            "model_selection": False,
            "checkpoint_frozen": True,
        },
        "checkpoint": {"path": str(CHECKPOINT.relative_to(ROOT)), "sha256": sha256(CHECKPOINT)},
        "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)},
        "input_manifest": {"path": str(input_path.relative_to(ROOT)), "sha256": sha256(input_path)},
        "supervision": {"path": str(supervision_path.relative_to(ROOT)), "sha256": sha256(supervision_path)},
        "split_audit": {"path": str((OUT / "PAIRED_VIEW_SPLIT_AUDIT.json").relative_to(ROOT)), "sha256": sha256(OUT / "PAIRED_VIEW_SPLIT_AUDIT.json")},
        "old_train_reference": {
            "input": {"path": str(OLD_INPUT.relative_to(ROOT)), "sha256": sha256(OLD_INPUT)},
            "supervision": {"path": str(OLD_SUPERVISION.relative_to(ROOT)), "sha256": sha256(OLD_SUPERVISION)},
        },
        "preprocessing": {
            "depth_view": "inverse depth, valid 0.10..2.0 m, per-image percentile 2..98, identical to canonical materializer",
            "model_input_allowlist": ["rgb_path", "depth_view_path", "instruction"],
            "oracle_passed_to_model": False,
        },
        "source_code": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__).resolve())},
    }
    write_json(LOCK, lock)
    print(json.dumps({"status": "PASS", "samples": 256, "found": 64, "lock": str(LOCK.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
