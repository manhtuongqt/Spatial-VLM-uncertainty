#!/usr/bin/env python3
"""Build and validate the immutable 15-family P-CRA-U overfit manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
DATASET = WORKSPACE / "datasets/roborefer_dataset_v1_prototype_20260818_155606"
SMOKE = WORKSPACE / "protocol/wp3_smoke_manifest.json"
SELECTION = WORKSPACE / "protocol/wp3_smoke_selection_audit.json"
CACHE = WORKSPACE / "results/wp3_feature_hook_smoke_20260821/hook_a/cache"
CONTRACT = WORKSPACE / "protocol/PCRA_U_OVERFIT_SMOKE_CONTRACT.md"
CONFIG = WORKSPACE / "protocol/pcra_u_v0_overfit_config.json"
OUTPUT = WORKSPACE / "protocol/pcra_u_overfit_smoke_manifest.json"
LOCK = WORKSPACE / "protocol/pcra_u_overfit_spec_lock.json"
ANSWER_STATES = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
SOURCES = ["semantic", "relation", "spatial", "depth", "occlusion"]


class PrepareError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PrepareError(f"Expected object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    smoke = load(SMOKE)
    selection = {row["sample_id"]: row for row in load(SELECTION)["entries"]}
    config = load(CONFIG)
    entries = []
    states, source_counts = Counter(), Counter()
    families = set()
    for item in smoke["entries"]:
        sample_id = item["sample_id"]
        if item["split"] != "train":
            raise PrepareError(f"Forbidden split in smoke manifest: {sample_id}")
        record_path = DATASET / item["record_path"]
        record = load(record_path)
        evaluator = record["evaluator_only"]
        uncertainty = evaluator["uncertainty_label"]
        spatial = evaluator["spatial_label"]
        mask_ref = evaluator["masks"]["target"]
        mask_path = DATASET / mask_ref["path"]
        cache_json = CACHE / f"{sample_id}.json"
        cache_meta = load(cache_json)
        tensor_path = cache_json.parent.parent / cache_meta["features"]["r0"]["file"]
        if tensor_path != CACHE / f"{sample_id}.safetensors":
            raise PrepareError(f"Unexpected tensor path: {tensor_path}")
        if cache_meta["run_id"] != "hook_a" or cache_meta["split"] != "train":
            raise PrepareError(f"Invalid cache provenance: {sample_id}")
        tensor_hash = sha256(tensor_path)
        if tensor_hash != cache_meta["features"]["r0"]["file_sha256"]:
            raise PrepareError(f"Tensor hash mismatch: {sample_id}")
        if cache_meta["features"]["r0"]["shape"] != [13, 1024, 1152]:
            raise PrepareError(f"Unexpected R0 shape: {sample_id}")
        if cache_meta["features"]["d0"]["shape"] != [13, 1024, 1152]:
            raise PrepareError(f"Unexpected D0 shape: {sample_id}")
        family_id = item["family_id"]
        if family_id in families:
            raise PrepareError(f"Duplicate family: {family_id}")
        families.add(family_id)
        state = uncertainty["state"]
        labels = uncertainty["sources"]
        states[state] += 1
        source_counts.update(labels)
        entries.append({
            "sample_id": sample_id,
            "family_id": family_id,
            "split": "train",
            "instruction": item["instruction"],
            "instruction_sha256": cache_meta["input"]["instruction_sha256"],
            "record_path": str(record_path.relative_to(WORKSPACE)),
            "record_sha256": sha256(record_path),
            "rgb_path": str((DATASET / item["rgb_path"]).relative_to(WORKSPACE)),
            "rgb_sha256": item["rgb_sha256"],
            "depth_path": str((DATASET / item["depth_path"]).relative_to(WORKSPACE)),
            "depth_sha256": item["depth_sha256"],
            "target_mask_path": str(mask_path.relative_to(WORKSPACE)),
            "target_mask_sha256": sha256(mask_path),
            "answerability_state": state,
            "answerability_index": ANSWER_STATES.index(state),
            "source_labels": labels,
            "source_multihot": [int(source in labels) for source in SOURCES],
            "evaluator_relation_labels_for_audit_only": spatial["relations"],
            "language_relation_category": selection[sample_id]["relation"],
            "cache_metadata_path": str(cache_json.relative_to(WORKSPACE)),
            "cache_metadata_sha256": sha256(cache_json),
            "feature_tensor_path": str(tensor_path.relative_to(WORKSPACE)),
            "feature_tensor_sha256": tensor_hash,
            "feature_shape": [13, 1024, 1152],
            "feature_dtype": "torch.float16",
            "tile_order": cache_meta["tiling"]["tile_order"],
            "local_tile_count": cache_meta["tiling"]["local_tile_count"],
            "thumbnail_present": cache_meta["tiling"]["thumbnail_present"],
            "rgb_depth_aligned": cache_meta["tiling"]["rgb_depth_aligned"]
        })
    expected_states = {"FOUND": 8, "AMBIGUOUS": 2, "ABSENT": 1, "INSUFFICIENT_EVIDENCE": 4}
    gates = {
        "exactly_15_samples": len(entries) == 15,
        "exactly_15_families": len(families) == 15,
        "all_train": all(row["split"] == "train" for row in entries),
        "answerability_coverage_exact": dict(states) == expected_states,
        "all_five_sources_present": all(source_counts[source] > 0 for source in SOURCES),
        "hook_a_only": all("/hook_a/" in row["feature_tensor_path"] for row in entries),
        "feature_shapes_locked": all(row["feature_shape"] == [13, 1024, 1152] for row in entries),
        "rgb_depth_aligned": all(row["rgb_depth_aligned"] for row in entries)
    }
    if not all(gates.values()):
        raise PrepareError(f"Manifest gate failed: {gates}")
    protected = load(WORKSPACE / "results/wp3_feature_hook_smoke_20260821/comparison_summary.json")["protected_inputs"]
    manifest = {
        "schema_version": 1,
        "protocol_id": "pcra_u_overfit_smoke_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "access_policy": "TRAIN_SUPERVISION_ONLY_NEVER_INFERENCE_PAYLOAD",
        "dataset_root": str(DATASET.relative_to(WORKSPACE)),
        "feature_source": "results/wp3_feature_hook_smoke_20260821/hook_a/cache",
        "sample_count": len(entries),
        "family_count": len(families),
        "answerability_counts": dict(sorted(states.items())),
        "source_positive_counts": dict(sorted(source_counts.items())),
        "answerability_order": ANSWER_STATES,
        "source_order": SOURCES,
        "gates": gates,
        "protected_inputs": protected,
        "entries": entries
    }
    write(OUTPUT, manifest)
    lock = {
        "schema_version": 1,
        "protocol_id": "pcra_u_overfit_smoke_v1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED_BEFORE_ONE_BATCH_BACKWARD_AND_OPTIMIZER_STEP",
        "files": {
            str(CONTRACT.relative_to(WORKSPACE)): sha256(CONTRACT),
            str(CONFIG.relative_to(WORKSPACE)): sha256(CONFIG),
            str(OUTPUT.relative_to(WORKSPACE)): sha256(OUTPUT),
            str(SMOKE.relative_to(WORKSPACE)): sha256(SMOKE),
            str(SELECTION.relative_to(WORKSPACE)): sha256(SELECTION),
            "protocol/training_checkpoint_manager.py": sha256(WORKSPACE / "protocol/training_checkpoint_manager.py")
        },
        "config_digest": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "manifest_gates": gates,
        "training_performed": False,
        "dataset_scaled": False,
        "allowed_next_action": "ONE_BATCH_FORWARD_BACKWARD_NO_OPTIMIZER_STEP"
    }
    write(LOCK, lock)
    print(json.dumps({"manifest": str(OUTPUT.relative_to(WORKSPACE)), "lock": str(LOCK.relative_to(WORKSPACE)),
                      "samples": len(entries), "families": len(families), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
