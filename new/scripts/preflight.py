#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter

from safetensors.torch import load_file

from pcrau.dataset import ArchivedPCRAUDataset, VARIANT_ORDER, split_summary
from pcrau.utils import load_config, sha256_file, workspace_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate locked archived inputs without modifying them")
    parser.add_argument("--config")
    parser.add_argument("--full-hash", action="store_true", help="verify every unique feature file hash")
    args = parser.parse_args()
    config = load_config(args.config)
    summary = split_summary(config)
    locked = config["locked_counts"]
    for split in ("train", "dev"):
        assert summary[split]["families"] == locked[f"{split}_families"]
        assert summary[split]["samples"] == locked[f"{split}_samples"]
    train = ArchivedPCRAUDataset(config, "train")
    dev = ArchivedPCRAUDataset(config, "dev")
    calibration = ArchivedPCRAUDataset(config, "calibration", profile="calibration")
    assert len(calibration) == locked["calibration_samples"]
    assert len({entry["family_id"] for entry in calibration.entries}) == locked["calibration_families"]
    assert set(entry["family_id"] for entry in train.entries).isdisjoint(
        entry["family_id"] for entry in dev.entries
    )
    assert all(Counter(entry["variant"] for entry in dataset.entries if entry["family_id"] == family) == Counter(VARIANT_ORDER)
               for dataset in (train, dev) for family in {entry["family_id"] for entry in dataset.entries})
    feature_shapes = {}
    checked = 0
    unique_feature_paths = set()
    feature_records = {}
    for name, dataset in (("development", train), ("calibration", calibration)):
        feature_records[name] = len(dataset.features)
        for key, metadata in dataset.features.items():
            path = dataset.layout.feature_root / metadata["path"]
            if path in unique_feature_paths:
                continue
            unique_feature_paths.add(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if args.full_hash and sha256_file(path) != metadata["sha256"]:
                raise ValueError(f"Feature hash mismatch: {path}")
            if not feature_shapes:
                feature_shapes = {name: list(value.shape) for name, value in load_file(str(path)).items()}
            checked += 1
    mask_files = 0
    mask_hashes = 0
    for dataset in (train, dev, calibration):
        for entry in dataset.entries:
            supervision = entry["supervision"]
            descriptors = [
                (supervision["target_mask_path"], supervision["target_mask_sha256"]),
                (supervision["target_interior_mask_path"], supervision["target_interior_mask_sha256"]),
                *((anchor["path"], anchor["sha256"]) for anchor in supervision["anchor_masks"]),
            ]
            for former_path, expected_hash in descriptors:
                path = dataset.layout.dataset_path(former_path)
                if not path.is_file():
                    raise FileNotFoundError(path)
                mask_files += 1
                if args.full_hash:
                    if sha256_file(path) != expected_hash:
                        raise ValueError(f"Mask hash mismatch: {path}")
                    mask_hashes += 1
    sample = train[0]
    report = {
        "status": "PASS",
        "splits": summary,
        "feature_shapes": feature_shapes,
        "unique_feature_records": feature_records,
        "feature_hashes_verified": checked if args.full_hash else 0,
        "calibration": {"samples": len(calibration), "families": len({entry["family_id"] for entry in calibration.entries})},
        "mask_files_verified_present": mask_files,
        "mask_hashes_verified": mask_hashes,
        "model_input_keys": sorted({"r0", "d0", "r_thumb", "d_thumb", "token_ids", "token_mask", "relation_ids", "relation_mask", "anchor_mask"}),
        "evaluator_only_keys_present_but_blocked": sorted({"target_heatmap", "target_full", "answer_target", "source_target", "anchor_heatmaps"} & set(sample)),
        "sealed_test_splits_accessed": False,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
