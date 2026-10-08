#!/usr/bin/env python3
"""Prepare separated evaluator and oracle-free manifests for Calibration v1.

This program is intentionally the only preparation stage that opens evaluator
records.  The raw-inference runner consumes only the oracle-free feature
manifest emitted here and therefore cannot see masks or calibration labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

if __package__:
    from .pcra_u_development_common import (
        ANSWERABILITY_CLASSES,
        RELATION_CLASSES,
        SOURCE_CLASSES,
        VARIANT_ORDER,
        DevelopmentTrainingError,
        canonical_sha256,
        read_json,
        safe_resolve,
        sha256_file,
        write_json,
    )
    from .prepare_pcra_u_development_train import manifest_entry, tree_digest
    from .wp3_feature_hook_smoke import forbidden_cache_findings, model_inventory_sha256
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pcra_u_development_common import (  # type: ignore
        ANSWERABILITY_CLASSES,
        RELATION_CLASSES,
        SOURCE_CLASSES,
        VARIANT_ORDER,
        DevelopmentTrainingError,
        canonical_sha256,
        read_json,
        safe_resolve,
        sha256_file,
        write_json,
    )
    from prepare_pcra_u_development_train import manifest_entry, tree_digest  # type: ignore
    from wp3_feature_hook_smoke import forbidden_cache_findings, model_inventory_sha256  # type: ignore


WORKSPACE = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "pcra_u_calibration_v1"
CONFIG_PATH = WORKSPACE / "protocol/pcra_u_calibration_config.json"
CONTRACT_PATH = WORKSPACE / "protocol/PCRA_U_CALIBRATION_CONTRACT.md"
SEED_LOCK_PATH = WORKSPACE / "protocol/pcra_u_calibration_seed_lock.json"
ARCHITECTURE_LOCK_PATH = WORKSPACE / "protocol/pcra_u_architecture_freeze_lock.json"
EVAL_MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_calibration_eval_manifest.json"
FEATURE_MANIFEST_PATH = WORKSPACE / "protocol/pcra_u_calibration_feature_manifest.json"
DATASET_ROOT = WORKSPACE / "datasets/roborefer_dataset_v2_1_calibration_200_20260824"
DATASET_INDEX = DATASET_ROOT / "dataset_index.json"
FAMILY_MANIFEST = DATASET_ROOT / "family_manifest.json"
FULL_QC = DATASET_ROOT / "DATASET_V2_1_CALIBRATION_FULL_QC_REPORT.json"


class CalibrationPreparationError(DevelopmentTrainingError):
    pass


def verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise CalibrationPreparationError(f"{label} missing/hash mismatch: {path}")


def iter_values(value: Any, key_name: str) -> Iterable[Any]:
    """Yield values for an exact JSON key without guessing from free text."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == key_name:
                yield child
            yield from iter_values(child, key_name)
    elif isinstance(value, list):
        for child in value:
            yield from iter_values(child, key_name)


def family_identity_sets(manifest: dict[str, Any]) -> dict[str, set[str]]:
    identities = {
        "family_id": set(),
        "capture_id": set(),
        "instruction_sha256": set(),
        "language_template_family_id": set(),
        "layout_seed": set(),
        "seed_bundle_value": set(),
    }
    for family in manifest.get("families", []):
        identities["family_id"].add(str(family["family_id"]))
        if "selected_layout_seed" in family:
            identities["layout_seed"].add(str(family["selected_layout_seed"]))
        for value in family.get("seed_bundle", {}).values():
            identities["seed_bundle_value"].add(str(value))
        for variant in family.get("variant_specs", []):
            for key in ("capture_id", "instruction_sha256", "language_template_family_id"):
                if key in variant:
                    identities[key].add(str(variant[key]))
    return identities


def prior_identity_sets() -> tuple[dict[str, set[str]], list[str]]:
    combined = {
        "family_id": set(),
        "capture_id": set(),
        "instruction_sha256": set(),
        "language_template_family_id": set(),
        "layout_seed": set(),
        "seed_bundle_value": set(),
    }
    audited: list[str] = []
    for path in sorted((WORKSPACE / "datasets").glob("**/family_manifest.json")):
        if path.resolve() == FAMILY_MANIFEST.resolve():
            continue
        manifest = read_json(path)
        identities = family_identity_sets(manifest)
        for name in combined:
            combined[name].update(identities[name])
        audited.append(str(path.relative_to(WORKSPACE)))
    # Some engineering-only runs retained only a raw capture manifest.  Their
    # capture IDs remain forbidden even when no materialized family exists.
    for path in sorted((WORKSPACE / "datasets").glob("**/raw_capture_manifest.json")):
        if DATASET_ROOT.resolve() in path.resolve().parents:
            continue
        manifest = read_json(path)
        combined["capture_id"].update(str(value) for value in iter_values(manifest, "capture_id"))
        audited.append(str(path.relative_to(WORKSPACE)))
    return combined, audited


def assert_unique_current_identities(family_manifest: dict[str, Any]) -> dict[str, int]:
    owners: dict[str, dict[str, str]] = {
        "family_id": {},
        "capture_id": {},
        "instruction_sha256": {},
        "language_template_family_id": {},
        "layout_seed": {},
        "seed_bundle_value": {},
    }
    for family in family_manifest["families"]:
        family_id = str(family["family_id"])
        candidates: dict[str, list[str]] = {
            "family_id": [family_id],
            "capture_id": [str(row["capture_id"]) for row in family["variant_specs"]],
            "instruction_sha256": [str(row["instruction_sha256"]) for row in family["variant_specs"]],
            "language_template_family_id": [
                str(row["language_template_family_id"]) for row in family["variant_specs"]
            ],
            "layout_seed": [str(family["selected_layout_seed"])],
            "seed_bundle_value": [str(value) for value in family["seed_bundle"].values()],
        }
        for name, values in candidates.items():
            for value in values:
                prior_owner = owners[name].get(value)
                if prior_owner is not None and prior_owner != family_id:
                    raise CalibrationPreparationError(
                        f"Calibration {name} reused across families: {value}: {prior_owner}/{family_id}"
                    )
                owners[name][value] = family_id
        if len(set(candidates["instruction_sha256"])) != len(candidates["instruction_sha256"]):
            raise CalibrationPreparationError(f"Instruction reused within family: {family_id}")
        if len(set(candidates["language_template_family_id"])) != len(
            candidates["language_template_family_id"]
        ):
            raise CalibrationPreparationError(f"Language template reused within family: {family_id}")
        if len(set(candidates["capture_id"])) != 2:
            raise CalibrationPreparationError(f"Family does not resolve to exactly two captures: {family_id}")
    return {name: len(values) for name, values in owners.items()}


def verify_frozen_model(config: dict[str, Any]) -> dict[str, str]:
    freeze = read_json(ARCHITECTURE_LOCK_PATH)
    frozen = config["frozen_model"]
    checkpoint = safe_resolve(WORKSPACE, frozen["checkpoint"])
    model_path = checkpoint / "model.safetensors"
    metadata_path = checkpoint / "metadata.json"
    manifest_path = checkpoint / "manifest.json"
    if freeze.get("architecture_frozen") is not True or freeze.get("checkpoint_frozen") is not True:
        raise CalibrationPreparationError("Architecture/checkpoint freeze is not active")
    if freeze.get("decision") != "FREEZE_PCRA_U_ARCHITECTURE_AND_CHECKPOINT":
        raise CalibrationPreparationError("Unexpected development architecture decision")
    selected = freeze["selected_checkpoint"]
    if selected.get("path") != frozen["checkpoint"] or selected.get("epoch") != 10 or selected.get("global_step") != 2000:
        raise CalibrationPreparationError("Calibration config does not bind the frozen selected checkpoint")
    if sha256_file(model_path) != frozen["model_sha256"] or selected.get("model_sha256") != frozen["model_sha256"]:
        raise CalibrationPreparationError("Frozen checkpoint model hash mismatch")
    if sha256_file(metadata_path) != freeze["checkpoint_metadata_sha256"]:
        raise CalibrationPreparationError("Frozen checkpoint metadata hash mismatch")
    if sha256_file(manifest_path) != freeze["checkpoint_manifest_sha256"]:
        raise CalibrationPreparationError("Frozen checkpoint manifest hash mismatch")
    metadata = read_json(metadata_path)
    development_index = safe_resolve(WORKSPACE, frozen["development_feature_index"])
    if sha256_file(development_index) != metadata["identity"]["feature_cache_index_sha256"]:
        raise CalibrationPreparationError("Original development feature-index identity changed")
    return {
        "architecture_freeze_lock_sha256": sha256_file(ARCHITECTURE_LOCK_PATH),
        "checkpoint_model_sha256": sha256_file(model_path),
        "checkpoint_metadata_sha256": sha256_file(metadata_path),
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "development_feature_index_sha256": sha256_file(development_index),
    }


def write_immutable(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if read_json(path) != value:
            raise CalibrationPreparationError(f"Refusing to replace non-equivalent locked manifest: {path}")
        return
    write_json(path, value)


def prepare() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    seed_lock = read_json(SEED_LOCK_PATH)
    dataset_index = read_json(DATASET_INDEX)
    family_manifest = read_json(FAMILY_MANIFEST)
    full_qc = read_json(FULL_QC)
    if config.get("protocol_id") != PROTOCOL_ID or seed_lock.get("protocol_id") != PROTOCOL_ID:
        raise CalibrationPreparationError("Calibration config/seed lock protocol mismatch")
    if full_qc.get("passed") is not True or full_qc.get("decision") != "GO_CALIBRATION_INFERENCE_AND_FIT":
        raise CalibrationPreparationError("Calibration full-QC does not authorize inference and fit")
    if full_qc.get("training_performed") is not False or full_qc.get("test_opened") is not False:
        raise CalibrationPreparationError("Calibration full-QC safety state changed")
    capture_count = full_qc.get("raw_capture_count", full_qc.get("capture_count"))
    expected = config["dataset"]
    if (
        dataset_index.get("selection") != "official_independent_calibration"
        or dataset_index.get("eligible_for_calibration") is not True
        or dataset_index.get("family_count") != expected["family_count"]
        or dataset_index.get("record_count") != expected["sample_count"]
        or dataset_index.get("split_family_counts") != {"calibration": expected["family_count"]}
        or capture_count != expected["capture_count"]
    ):
        raise CalibrationPreparationError("Calibration dataset count/selection commitment mismatch")
    families = family_manifest.get("families", [])
    if len(families) != 200 or len({row["family_id"] for row in families}) != 200:
        raise CalibrationPreparationError("Calibration family manifest must contain 200 unique families")
    if any(row.get("split") != "calibration" for row in families):
        raise CalibrationPreparationError("Non-calibration family found in Calibration dataset")
    if any(row.get("asset_pool") != "seen" or row.get("ood_axis") != "none" for row in families):
        raise CalibrationPreparationError("Calibration contains a non-IID or non-seen family")
    forbidden_ood = {"ycb_pear", "ycb_plum", "ycb_tuna_fish_can"}
    active_assets = {
        str(asset)
        for family in families
        for asset in family.get("primary_scene", {}).get("active_scene_ids", [])
    }
    if any(any(asset == token or asset.startswith(token + "_") for token in forbidden_ood) for asset in active_assets):
        raise CalibrationPreparationError("Test-OOD-only asset leaked into Calibration")

    current_counts = assert_unique_current_identities(family_manifest)
    current = family_identity_sets(family_manifest)
    prior, audited_prior_manifests = prior_identity_sets()
    overlaps = {name: sorted(current[name] & prior[name]) for name in current}
    if any(overlaps.values()):
        first = next((name, values[:3]) for name, values in overlaps.items() if values)
        raise CalibrationPreparationError(f"Prior-data identity overlap in {first[0]}: {first[1]}")

    frozen_hashes = verify_frozen_model(config)
    model_root = safe_resolve(WORKSPACE, config["feature_input"]["model_root"])
    model_hash = model_inventory_sha256(model_root)
    if model_hash != read_json(ARCHITECTURE_LOCK_PATH)["protected_inputs"]["model_inventory_sha256"]:
        raise CalibrationPreparationError("Frozen RoboRefer model inventory changed")
    by_family = {row["family_id"]: row for row in families}
    eval_entries: list[dict[str, Any]] = []
    feature_entries: list[dict[str, Any]] = []
    for index_row in dataset_index["records"]:
        if index_row.get("split") != "calibration" or index_row["family_id"] not in by_family:
            raise CalibrationPreparationError(f"Invalid calibration index row: {index_row.get('sample_id')}")
        eval_entry, feature_entry = manifest_entry(
            DATASET_ROOT, index_row, by_family[index_row["family_id"]], config, model_hash
        )
        eval_entry["supervision"]["access_policy"] = (
            "CALIBRATION_EVALUATOR_ONLY_NEVER_RAW_INFERENCE_OR_MODEL_INPUT"
        )
        eval_entries.append(eval_entry)
        feature_entries.append(feature_entry)
    order = {name: index for index, name in enumerate(VARIANT_ORDER)}
    eval_entries.sort(key=lambda row: (row["family_id"], order[row["variant"]]))
    feature_entries.sort(key=lambda row: (row["family_id"], order[row["variant"]]))
    if len(eval_entries) != 1000 or len({row["sample_id"] for row in eval_entries}) != 1000:
        raise CalibrationPreparationError("Calibration record/sample uniqueness mismatch")
    if any(
        [row["variant"] for row in eval_entries if row["family_id"] == family_id] != VARIANT_ORDER
        for family_id in by_family
    ):
        raise CalibrationPreparationError("Calibration family variant order mismatch")

    # This direct content check catches recaptured IDs with accidentally reused
    # observable images even when metadata identifiers differ.
    development_feature_manifest_path = WORKSPACE / "protocol/pcra_u_development_feature_manifest.json"
    development_feature_manifest = read_json(development_feature_manifest_path)
    development_rgb = {row["rgb_sha256"] for row in development_feature_manifest["entries"]}
    development_depth = {row["depth_sha256"] for row in development_feature_manifest["entries"]}
    if development_rgb & {row["rgb_sha256"] for row in feature_entries}:
        raise CalibrationPreparationError("Calibration RGB content overlaps development")
    if development_depth & {row["depth_sha256"] for row in feature_entries}:
        raise CalibrationPreparationError("Calibration depth content overlaps development")
    if forbidden_cache_findings({"entries": feature_entries}):
        raise CalibrationPreparationError("Oracle/evaluator-like key leaked into feature manifest")

    answer_counts = Counter(row["supervision"]["answerability_state"] for row in eval_entries)
    family_strata = Counter(row["primary_answerability_stratum"] for row in families)
    if dict(family_strata) != expected["primary_family_strata"]:
        raise CalibrationPreparationError(f"Calibration family stratum mismatch: {dict(family_strata)}")
    protected = {
        "config_sha256": sha256_file(CONFIG_PATH),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "seed_lock_sha256": sha256_file(SEED_LOCK_PATH),
        "dataset_index_sha256": sha256_file(DATASET_INDEX),
        "family_manifest_sha256": sha256_file(FAMILY_MANIFEST),
        "full_qc_report_sha256": sha256_file(FULL_QC),
        "dataset_tree_sha256": tree_digest(DATASET_ROOT),
        "model_inventory_sha256": model_hash,
        "development_feature_manifest_sha256": sha256_file(development_feature_manifest_path),
        **frozen_hashes,
    }
    eval_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "CALIBRATION_EVALUATOR_ONLY_LABELS_SEPARATED",
        "dataset_root": str(DATASET_ROOT.relative_to(WORKSPACE)),
        "split": "calibration",
        "family_count": 200,
        "sample_count": 1000,
        "answerability_classes": ANSWERABILITY_CLASSES,
        "source_classes": SOURCE_CLASSES,
        "relation_classes": RELATION_CLASSES,
        "variant_order": VARIANT_ORDER,
        "family_ids": sorted(by_family),
        "summary": {
            "primary_family_strata": dict(sorted(family_strata.items())),
            "record_answerability_counts": dict(sorted(answer_counts.items())),
            "unique_feature_pairs": len({row["feature_key"] for row in feature_entries}),
            "cross_split_overlap_counts": {name: len(values) for name, values in overlaps.items()},
            "prior_manifests_audited": len(audited_prior_manifests),
        },
        "protected_inputs": protected,
        "entries": eval_entries,
        "safety": {
            "raw_inference_may_read_this_manifest": False,
            "model_training_performed": False,
            "checkpoint_selection_performed": False,
            "test_iid_opened": False,
            "test_ood_opened": False,
        },
    }
    feature_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "ORACLE_FREE_CALIBRATION_FEATURE_INPUT_ONLY",
        "dataset_root": str(DATASET_ROOT.relative_to(WORKSPACE)),
        "split": "calibration",
        "model_root": config["feature_input"]["model_root"],
        "model_inventory_sha256": model_hash,
        "feature_transform_sha256": canonical_sha256(config["feature_input"]),
        "sample_count": 1000,
        "family_count": 200,
        "unique_feature_pair_count": len({row["feature_key"] for row in feature_entries}),
        "entries": feature_entries,
        "safety": {
            "evaluator_artifacts_read_by_feature_extractor": False,
            "oracle_labels_present": False,
            "model_training_performed": False,
            "test_iid_opened": False,
            "test_ood_opened": False,
        },
    }
    if forbidden_cache_findings(feature_manifest):
        raise CalibrationPreparationError("Generated feature manifest contains oracle/evaluator-like keys")
    write_immutable(EVAL_MANIFEST_PATH, eval_manifest)
    write_immutable(FEATURE_MANIFEST_PATH, feature_manifest)
    return {
        "status": "CALIBRATION_MANIFESTS_PREPARED",
        "eval_manifest": str(EVAL_MANIFEST_PATH.relative_to(WORKSPACE)),
        "eval_manifest_sha256": sha256_file(EVAL_MANIFEST_PATH),
        "feature_manifest": str(FEATURE_MANIFEST_PATH.relative_to(WORKSPACE)),
        "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
        "family_count": 200,
        "sample_count": 1000,
        "unique_identity_counts": current_counts,
        "cross_split_overlap_counts": {name: len(values) for name, values in overlaps.items()},
        "model_training_performed": False,
        "test_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(prepare(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
