from __future__ import annotations

from collections import Counter

from pcrau.dataset import ArchivedPCRAUDataset, VARIANT_ORDER, split_summary
from pcrau.text import parse_relations, prompt_anchor_mask
from pcrau.utils import load_config


def test_locked_family_split_and_variants() -> None:
    config = load_config()
    summary = split_summary(config)
    assert summary["train"]["families"] == 320
    assert summary["train"]["samples"] == 1600
    assert summary["dev"]["families"] == 80
    assert summary["dev"]["samples"] == 400
    for split in ("train", "dev"):
        dataset = ArchivedPCRAUDataset(config, split)
        by_family = {}
        for entry in dataset.entries:
            by_family.setdefault(entry["family_id"], []).append(entry["variant"])
        assert all(Counter(variants) == Counter(VARIANT_ORDER) for variants in by_family.values())


def test_language_masks_do_not_copy_evaluator_anchor_masks() -> None:
    dataset = ArchivedPCRAUDataset(load_config(), "train")
    direct_index = next(index for index, entry in enumerate(dataset.entries) if entry["audit_only"]["relation"] == "direct")
    sample = dataset[direct_index]
    assert not sample["anchor_mask"].any()
    assert not sample["edge_mask"].any()


def test_two_anchor_relation_is_one_hyper_relation() -> None:
    prompt = "Locate the banana nearer than both the apple and mustard bottle."
    assert parse_relations(prompt) == ["nearer_than_both"]
    assert prompt_anchor_mask(prompt, 3) == [True, True, False]
