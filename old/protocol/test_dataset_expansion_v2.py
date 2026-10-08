"""Tests for the Dataset Expansion V2 design lock; no capture or training."""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import jsonschema
import pytest

PROTOCOL_DIR = Path(__file__).resolve().parent
WORKSPACE = PROTOCOL_DIR.parent
sys.path.insert(0, str(PROTOCOL_DIR))

from validate_dataset_expansion_v2 import (  # noqa: E402
    CATEGORIES,
    STATES,
    VARIANTS,
    assignment_commitment,
    build_development_assignments,
    build_manifest,
    build_pilot_assignments,
    run_validation,
)
from wp2_common import sha256_file  # noqa: E402


def load(name: str) -> dict:
    return json.loads((PROTOCOL_DIR / name).read_text(encoding="utf-8"))


def test_assignment_commitments_and_pilot_exclusion_are_exact():
    seed_lock = load("dataset_expansion_v2_seed_lock.json")
    partition = load("dataset_expansion_v2_asset_partition.json")
    pilot = build_pilot_assignments(seed_lock, partition)
    development = build_development_assignments(seed_lock, partition)
    assert assignment_commitment(pilot) == seed_lock["pilot_only"][
        "assignment_commitment_sha256"
    ]
    assert assignment_commitment(development) == seed_lock["development"][
        "assignment_commitment_sha256"
    ]
    assert len(pilot) == 30
    assert len(development) == 400
    assert not any(item["eligible_for_official_dataset"] for item in pilot)
    assert all(item["eligible_for_official_dataset"] for item in development)
    assert {item["family_id"] for item in pilot}.isdisjoint(
        item["family_id"] for item in development
    )


def test_development_split_state_category_and_depth_counts_are_exact():
    seed_lock = load("dataset_expansion_v2_seed_lock.json")
    partition = load("dataset_expansion_v2_asset_partition.json")
    development = build_development_assignments(seed_lock, partition)
    assert Counter(item["split"] for item in development) == {"train": 320, "dev": 80}
    assert Counter(item["primary_answerability_stratum"] for item in development) == {
        "FOUND": 160,
        "INSUFFICIENT_EVIDENCE": 100,
        "AMBIGUOUS": 70,
        "ABSENT": 70,
    }
    assert set(item["primary_answerability_stratum"] for item in development) == set(STATES)
    assert set(item["family_category"] for item in development) == set(CATEGORIES)
    assert sum(item["depth_dependent"] for item in development) == 260
    for split in ("train", "dev"):
        selected = [item for item in development if item["split"] == split]
        assert set(item["primary_answerability_stratum"] for item in selected) == set(STATES)
        assert set(item["family_category"] for item in selected) == set(CATEGORIES)
        assert all(tuple(item["variants"]) == VARIANTS for item in selected)


def test_answerability_ontology_distinguishes_absent_from_insufficient():
    ontology = load("dataset_expansion_v2_spec.json")["answerability_ontology"]
    assert ontology["ABSENT"]["denotation_cardinality_max"] == 0
    assert ontology["ABSENT"]["world_referent_exists"] is False
    assert ontology["ABSENT"]["observation_sufficient"] is True
    assert ontology["INSUFFICIENT_EVIDENCE"]["denotation_cardinality_min"] == 1
    assert ontology["INSUFFICIENT_EVIDENCE"]["world_referent_exists"] is True
    assert ontology["INSUFFICIENT_EVIDENCE"]["observation_sufficient"] is False


def test_asset_partition_and_single_axis_ood_are_disjoint():
    partition = load("dataset_expansion_v2_asset_partition.json")
    seen = set(partition["seen_pool"]["asset_ids"])
    heldout = set(partition["test_ood_asset_holdout"]["asset_ids"])
    excluded = set(partition["excluded_assets"]["asset_ids"])
    assert heldout == {"ycb_pear_01", "ycb_plum_01", "ycb_tuna_fish_can_01"}
    assert seen.isdisjoint(heldout)
    assert seen.isdisjoint(excluded)
    assert heldout.isdisjoint(excluded)
    baseline = partition["ood_axes"]["none"]
    expected = {
        "asset": "asset_pool",
        "layout": "layout_generator_id",
        "viewpoint": "camera_bin_id",
        "language": "language_template_bank_id",
        "depth_noise": "depth_noise_generator_id",
    }
    for axis, changed_key in expected.items():
        changed = [key for key in baseline if partition["ood_axes"][axis][key] != baseline[key]]
        assert changed == [changed_key]


def test_split_schema_accepts_design_manifest_and_rejects_missing_variant():
    seed_lock = load("dataset_expansion_v2_seed_lock.json")
    partition = load("dataset_expansion_v2_asset_partition.json")
    schema = load("dataset_split_v2.schema.json")
    assignments = build_pilot_assignments(seed_lock, partition)
    manifest = build_manifest(
        "dataset_expansion_v2_pilot_test",
        assignments,
        sha256_file(PROTOCOL_DIR / "dataset_expansion_v2_asset_partition.json"),
        sha256_file(PROTOCOL_DIR / "dataset_expansion_v2_seed_lock.json"),
    )
    jsonschema.validate(manifest, schema)
    invalid = copy.deepcopy(manifest)
    invalid["families"][0]["variants"].pop()
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)


def test_full_design_lock_validation_passes():
    report = run_validation(WORKSPACE)
    assert report["status"] == "PASS", {
        name: value for name, value in report["checks"].items() if not value["passed"]
    }
    assert report["decision"] == "GO_DRY_RUN_30_PILOT_ONLY"
    assert report["training_performed"] is False
    assert report["official_family_count_captured"] == 0
