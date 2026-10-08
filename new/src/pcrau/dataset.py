from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset, Sampler

from .text import prompt_anchor_mask, relation_ids, tokenize
from .utils import read_json, sha256_file, workspace_path


ANSWER_CLASSES = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
SOURCE_CLASSES = ["semantic", "relation", "spatial", "depth", "occlusion"]
VARIANT_ORDER = [
    "clean",
    "semantic_counterfactual",
    "relation_counterfactual",
    "depth_corruption",
    "occlusion_view_counterfactual",
]


class ArchiveLayout:
    """Resolve former workspace-relative paths without changing the archive."""

    def __init__(self, config: Mapping[str, Any], profile: str):
        paths = config["paths"]
        if profile == "development":
            self.dataset_root = workspace_path(paths["development_dataset_root"])
            self.feature_root = workspace_path(paths["development_feature_root"])
            self.feature_index = workspace_path(paths["development_feature_index"])
            self.manifest = workspace_path(paths["development_manifest"])
            self.former_dataset_name = "roborefer_dataset_v2_1_1_development_400_20260824"
        elif profile == "calibration":
            self.dataset_root = workspace_path(paths["calibration_dataset_root"])
            self.feature_root = workspace_path(paths["calibration_feature_root"])
            self.feature_index = workspace_path(paths["calibration_feature_index"])
            self.manifest = workspace_path(paths["calibration_manifest"])
            self.former_dataset_name = "roborefer_dataset_v2_1_calibration_200_20260824"
        elif profile == "test_iid":
            # Test-IID is opened only after the frozen-inference gate passes.
            # These paths are deliberately outside the training configuration,
            # so extending evaluation cannot change the checkpoint config hash.
            self.dataset_root = workspace_path("new/test_iid/dataset")
            self.feature_root = workspace_path("new/test_iid/feature_cache")
            self.feature_index = workspace_path("new/test_iid/feature_cache/indexes/index_full.json")
            self.manifest = workspace_path("new/test_iid/protocol/test_iid_eval_manifest.json")
            self.former_dataset_name = "__test_iid_paths_are_dataset_relative__"
        else:
            raise ValueError(f"Unknown profile: {profile}")

    def dataset_path(self, former_path: str) -> Path:
        raw = Path(former_path)
        parts = raw.parts
        if self.former_dataset_name in parts:
            suffix = parts[parts.index(self.former_dataset_name) + 1 :]
            result = self.dataset_root.joinpath(*suffix).resolve()
        elif raw.is_absolute():
            result = raw.resolve()
        else:
            result = (self.dataset_root / raw).resolve()
        if result != self.dataset_root and self.dataset_root not in result.parents:
            raise ValueError(f"Dataset path escapes archive root: {former_path}")
        return result


def _read_mask(path: Path, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (480, 640):
        raise ValueError(f"Invalid 640x480 mask: {path}")
    full = torch.from_numpy(image > 0)
    small = cv2.resize((image > 0).astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return torch.from_numpy(small), full


class ArchivedPCRAUDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        config: Mapping[str, Any],
        split: str,
        profile: str = "development",
        verify_feature_hash: bool = False,
    ):
        self.config = config
        self.split = split
        self.profile = profile
        self.layout = ArchiveLayout(config, profile)
        self.manifest = read_json(self.layout.manifest)
        self.index = read_json(self.layout.feature_index)
        entries = self.manifest.get("entries", [])
        if profile == "development":
            self.entries = [entry for entry in entries if entry["split"] == split]
        elif profile == "calibration":
            if split != "calibration":
                raise ValueError("Calibration profile only supports split=calibration")
            self.entries = entries
        else:
            if split != "test_iid":
                raise ValueError("Test-IID profile only supports split=test_iid")
            self.entries = entries
        self.verify_feature_hash = verify_feature_hash
        self.model_cfg = config["model"]
        self.sample_to_feature = self.index["sample_to_feature"]
        self.features = self.index["features"]

    def __len__(self) -> int:
        return len(self.entries)

    def _feature(self, entry: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        key = self.sample_to_feature[entry["sample_id"]]
        metadata = self.features[key]
        path = (self.layout.feature_root / metadata["path"]).resolve()
        if self.verify_feature_hash and sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"Feature hash mismatch: {path}")
        tensors = load_file(str(path), device="cpu")
        required = {"R0_GRID", "D0_GRID", "R0_THUMB", "D0_THUMB"}
        if set(tensors) != required:
            raise ValueError(f"Unexpected feature tensors for {entry['sample_id']}: {sorted(tensors)}")
        return tensors

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        tensors = self._feature(entry)
        prompt = entry.get("feature_input", entry).get("prompt", entry.get("prompt", ""))
        token_ids, token_mask = tokenize(
            prompt, int(self.model_cfg["max_tokens"]), int(self.model_cfg["vocab_size"])
        )
        rel_ids, rel_mask = relation_ids(prompt, int(self.model_cfg["max_relations"]))
        supervision = entry["supervision"]
        width, height = int(self.model_cfg["grid_width"]), int(self.model_cfg["grid_height"])
        target, target_full = _read_mask(
            self.layout.dataset_path(supervision["target_mask_path"]), (width, height)
        )
        interior, interior_full = _read_mask(
            self.layout.dataset_path(supervision["target_interior_mask_path"]), (width, height)
        )
        max_anchors = int(self.model_cfg["max_anchors"])
        anchor_maps = torch.zeros(max_anchors, height, width, dtype=torch.float32)
        anchor_supervision_mask = torch.zeros(max_anchors, dtype=torch.bool)
        for slot, descriptor in enumerate(supervision["anchor_masks"][:max_anchors]):
            small, _ = _read_mask(self.layout.dataset_path(descriptor["path"]), (width, height))
            anchor_maps[slot] = small
            anchor_supervision_mask[slot] = bool(small.sum() > 0)
        language_anchor_mask = torch.tensor(prompt_anchor_mask(prompt, max_anchors), dtype=torch.bool)
        source = torch.zeros(len(SOURCE_CLASSES), dtype=torch.float32)
        for name in supervision["source_labels"]:
            if name in SOURCE_CLASSES:
                source[SOURCE_CLASSES.index(name)] = 1.0
        # Explicitly declared weak label: ambiguity is observable spatial multimodality.
        source[SOURCE_CLASSES.index("spatial")] = float(supervision["answerability_state"] == "AMBIGUOUS")
        valid_target_count = int(entry.get("audit_only", {}).get("valid_target_count", 0))
        edge_target = torch.zeros(int(self.model_cfg["max_relations"]), dtype=torch.float32)
        # A direct query has a "direct" relation token but no graph edge.  Only
        # language-declared relation/anchor pairs may supervise the edge head.
        edge_mask = torch.tensor(rel_mask, dtype=torch.bool)
        edge_mask &= language_anchor_mask[: edge_mask.numel()]
        if language_anchor_mask.any() and supervision["answerability_state"] == "FOUND" and valid_target_count > 0:
            edge_target[edge_mask] = 1.0
        return {
            "sample_id": entry["sample_id"],
            "family_id": entry["family_id"],
            "variant": entry["variant"],
            "prompt": prompt,
            "r0": tensors["R0_GRID"].float(),
            "d0": tensors["D0_GRID"].float(),
            "r_thumb": tensors["R0_THUMB"].float(),
            "d_thumb": tensors["D0_THUMB"].float(),
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "token_mask": torch.tensor(token_mask, dtype=torch.bool),
            "relation_ids": torch.tensor(rel_ids, dtype=torch.long),
            "relation_mask": torch.tensor(rel_mask, dtype=torch.bool),
            "anchor_mask": language_anchor_mask,
            "target_heatmap": target.float(),
            "interior_heatmap": interior.float(),
            "anchor_heatmaps": anchor_maps,
            "anchor_supervision_mask": anchor_supervision_mask,
            "target_loss_mask": torch.tensor(bool(target.sum() > 0)),
            "interior_loss_mask": torch.tensor(bool(interior.sum() > 0)),
            "edge_target": edge_target,
            "edge_mask": edge_mask,
            "answer_target": torch.tensor(int(supervision["answerability_index"]), dtype=torch.long),
            "source_target": source,
            "target_full": target_full,
            "interior_full": interior_full,
            "weak_spatial_source": torch.tensor(True),
        }


class FamilyBatchSampler(Sampler[list[int]]):
    """Keep all five variants of sampled families in the same batch."""

    def __init__(
        self,
        dataset: ArchivedPCRAUDataset,
        families_per_batch: int,
        seed: int,
        shuffle: bool,
        sampling: Mapping[str, Any] | None = None,
    ):
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, entry in enumerate(dataset.entries):
            grouped[entry["family_id"]].append(index)
        for family, indices in grouped.items():
            variants = [dataset.entries[index]["variant"] for index in indices]
            if Counter(variants) != Counter(VARIANT_ORDER):
                raise ValueError(f"Family does not have five locked variants: {family}: {variants}")
            indices.sort(key=lambda i: VARIANT_ORDER.index(dataset.entries[i]["variant"]))
        self.grouped = grouped
        self.families = sorted(grouped)
        self.families_per_batch = families_per_batch
        self.seed = seed
        self.shuffle = shuffle
        self.sampling = dict(sampling or {})
        relation_repeats = self.sampling.get("relation_family_repeats", {})
        absent_repeat = int(self.sampling.get("absent_family_repeat", 1))
        self.family_repeats: dict[str, int] = {}
        for family, indices in grouped.items():
            entries = [dataset.entries[index] for index in indices]
            relation = entries[0].get("audit_only", {}).get("relation", "direct")
            repeat = int(relation_repeats.get(relation, 1))
            if any(entry["supervision"]["answerability_state"] == "ABSENT" for entry in entries):
                repeat = max(repeat, absent_repeat)
            self.family_repeats[family] = max(1, repeat)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        families = [family for family in self.families for _ in range(self.family_repeats[family])]
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(families), generator=generator).tolist()
            families = [families[index] for index in order]
        for start in range(0, len(families), self.families_per_batch):
            selected = families[start : start + self.families_per_batch]
            yield [index for family in selected for index in self.grouped[family]]

    def __len__(self) -> int:
        count = sum(self.family_repeats.values())
        return (count + self.families_per_batch - 1) // self.families_per_batch

    def summary(self) -> dict[str, int]:
        return {
            "unique_families": len(self.families),
            "family_presentations_per_epoch": sum(self.family_repeats.values()),
            "oversampled_family_presentations": sum(self.family_repeats.values()) - len(self.families),
            "batches_per_epoch": len(self),
        }


def collate_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [key for key, value in samples[0].items() if isinstance(value, torch.Tensor)]
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
    for key in samples[0]:
        if key not in tensor_keys:
            batch[key] = [sample[key] for sample in samples]
    return batch


def move_model_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    allowed = {
        "r0", "d0", "r_thumb", "d_thumb", "token_ids", "token_mask",
        "relation_ids", "relation_mask", "anchor_mask",
    }
    return {key: batch[key].to(device, non_blocking=True) for key in allowed}


def split_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "dev"):
        dataset = ArchivedPCRAUDataset(config, split)
        result[split] = {
            "samples": len(dataset),
            "families": len({entry["family_id"] for entry in dataset.entries}),
            "states": dict(Counter(entry["supervision"]["answerability_state"] for entry in dataset.entries)),
            "anchors": dict(Counter(len(entry["supervision"]["anchor_masks"]) for entry in dataset.entries)),
        }
    return result
