from __future__ import annotations

from pcrau.dataset import ArchivedPCRAUDataset, FamilyBatchSampler
from pcrau.utils import load_config


def test_v3_sampler_preserves_all_families_and_adds_declared_repeats() -> None:
    config = load_config("new/configs/v3_seed_24082026.json")
    dataset = ArchivedPCRAUDataset(config, "train")
    sampler = FamilyBatchSampler(
        dataset, config["optimization"]["families_per_batch"], config["seed"], True,
        sampling=config["sampling"],
    )
    assert sampler.summary() == {
        "unique_families": 320,
        "family_presentations_per_epoch": 412,
        "oversampled_family_presentations": 92,
        "batches_per_epoch": 103,
    }
    presented = []
    for batch in sampler:
        presented.extend(dataset.entries[index]["family_id"] for index in batch[::5])
    assert set(presented) == set(sampler.families)


def test_v3_regularization_and_selection_are_locked() -> None:
    config = load_config("new/configs/v3_seed_24082026.json")
    assert config["model"]["dropout"] == 0.2
    assert config["model"]["modality_dropout_probability"] == 0.15
    assert config["optimization"]["learning_rate"] == 1e-4
    assert config["optimization"]["weight_decay"] == 5e-4
    assert config["optimization"]["checkpoint_every_epochs"] == 3
    assert config["selection"]["minimum_grounding_accuracy"] == 0.97
