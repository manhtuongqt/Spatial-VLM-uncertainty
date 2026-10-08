#!/usr/bin/env python3
"""Materialize the locked 200-family Test-IID capture into 1,000 records."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TEST_ROOT = ROOT / "new/test_iid"
DATASET_ROOT = TEST_ROOT / "dataset"
PROTOCOL_ROOT = TEST_ROOT / "protocol"
PROTOCOL_ID = "roborefer_dataset_v2_1_test_iid_capture_200"
FAMILY_COUNT = 200
CAPTURE_COUNT = 400
SAMPLE_COUNT = 1000
VARIANTS = [
    "clean",
    "semantic_counterfactual",
    "relation_counterfactual",
    "depth_corruption",
    "occlusion_view_counterfactual",
]

sys.path.insert(0, str(ROOT / "old/protocol"))
from dataset_v2_pilot_materialize import (  # noqa: E402
    adaptive_erode,
    artifact_hash_map,
    binary_mask,
    corrupt_depth,
    file_ref,
    relative_depth,
    write_png,
)
from wp2_common import read_json, sha256_file, utc_now, write_json  # noqa: E402


class TestIIDMaterializeError(RuntimeError):
    pass


def verify_frozen_inputs() -> dict[str, str]:
    freeze_path = TEST_ROOT / "contracts/best_v2_freeze_lock.json"
    freeze = read_json(freeze_path)
    selected = freeze["selected_checkpoint"]
    calibrator = freeze["frozen_calibrator"]
    checkpoint = ROOT / selected["path"]
    calibrator_path = ROOT / calibrator["path"]
    metadata_path = checkpoint.parent / "metadata.json"
    checks = {
        "checkpoint": (checkpoint, selected["model_sha256"]),
        "checkpoint_metadata": (metadata_path, selected["metadata_sha256"]),
        "calibrator": (calibrator_path, calibrator["sha256"]),
    }
    for name, (path, expected) in checks.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise TestIIDMaterializeError(f"frozen {name} hash mismatch: {path}")
    return {
        "freeze_lock_sha256": sha256_file(freeze_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_metadata_sha256": sha256_file(metadata_path),
        "calibrator_sha256": sha256_file(calibrator_path),
    }


def verify_capture(capture_id: str) -> tuple[dict[str, Any], Path]:
    directory = DATASET_ROOT / "raw/captures" / capture_id
    meta_path = directory / "capture_meta.json"
    if not meta_path.is_file():
        raise TestIIDMaterializeError(f"missing capture metadata: {capture_id}")
    meta = read_json(meta_path)
    if meta.get("protocol_id") != PROTOCOL_ID or meta.get("capture_id") != capture_id:
        raise TestIIDMaterializeError(f"capture identity mismatch: {capture_id}")
    for relative, expected in meta["artifact_sha256"].items():
        path = directory / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise TestIIDMaterializeError(f"capture artifact hash mismatch: {capture_id}/{relative}")
    return meta, directory


def verify_capture_gate(plan: dict[str, Any]) -> dict[str, Any]:
    raw = read_json(DATASET_ROOT / "raw/raw_capture_manifest.json")
    if (
        raw.get("protocol_id") != PROTOCOL_ID
        or raw.get("complete") is not True
        or raw.get("capture_count") != CAPTURE_COUNT
        or raw.get("expected_capture_count") != CAPTURE_COUNT
    ):
        raise TestIIDMaterializeError("exactly 400 complete Test-IID captures are required")
    planned = {row["capture_id"] for row in plan["captures"]}
    observed = {row["capture_id"] for row in raw["captures"]}
    if len(planned) != CAPTURE_COUNT or observed != planned:
        raise TestIIDMaterializeError("planned/raw capture identity set differs")
    reports = []
    for batch_id in ("canary_000", "batch_001", "batch_002"):
        path = DATASET_ROOT / f"report_assets/checkpoints/batch_qc/{batch_id}.json"
        report = read_json(path)
        if report.get("passed") is not True or report.get("batch_id") != batch_id:
            raise TestIIDMaterializeError(f"capture QC is not PASS: {batch_id}")
        reports.append(report)
    return {
        "raw_manifest_sha256": sha256_file(DATASET_ROOT / "raw/raw_capture_manifest.json"),
        "batch_qc_sha256": {
            batch_id: sha256_file(DATASET_ROOT / f"report_assets/checkpoints/batch_qc/{batch_id}.json")
            for batch_id in ("canary_000", "batch_001", "batch_002")
        },
        "relation_checks": sum(int(row["relation_checks"]) for row in reports),
    }


def materialize() -> list[dict[str, Any]]:
    manifest_path = PROTOCOL_ROOT / "test_iid_manifest.json"
    plan_path = PROTOCOL_ROOT / "test_iid_capture_plan.json"
    lock_path = PROTOCOL_ROOT / "execution_lock.json"
    manifest = read_json(manifest_path)
    plan = read_json(plan_path)
    lock = read_json(lock_path)
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or len(manifest.get("families", [])) != FAMILY_COUNT
        or plan.get("family_count") != FAMILY_COUNT
        or plan.get("capture_count") != CAPTURE_COUNT
        or plan.get("sample_count_after_materialization") != SAMPLE_COUNT
        or lock.get("training_authorized") is not False
    ):
        raise TestIIDMaterializeError("Test-IID manifest/plan/lock contract differs")
    capture_gate = verify_capture_gate(plan)
    frozen = verify_frozen_inputs()
    registry = plan["object_registry"]
    by_id = {row["id"]: {**row, "model_name": name} for name, row in registry.items()}

    write_json(DATASET_ROOT / "family_manifest.json", manifest)
    write_json(DATASET_ROOT / "capture_plan.json", plan)
    write_json(DATASET_ROOT / "object_registry.json", registry)
    write_json(DATASET_ROOT / "capture_execution_lock.json", lock)
    provenance = {
        "manifest_sha256": sha256_file(manifest_path),
        "capture_plan_sha256": sha256_file(plan_path),
        "capture_execution_lock_sha256": sha256_file(lock_path),
        "materializer_sha256": sha256_file(Path(__file__).resolve()),
        **frozen,
        **capture_gate,
    }
    records: list[dict[str, Any]] = []
    seen_samples: set[str] = set()

    for family in manifest["families"]:
        if family.get("split") != "test_iid" or family.get("ood_axis") != "none":
            raise TestIIDMaterializeError(f"non-IID family: {family.get('family_id')}")
        specs = family.get("variant_specs", [])
        if [row.get("variant") for row in specs] != VARIANTS:
            raise TestIIDMaterializeError(f"variant order differs: {family['family_id']}")
        for spec in specs:
            variant = spec["variant"]
            sample_id = f"{family['family_id']}__{variant}"
            if sample_id in seen_samples:
                raise TestIIDMaterializeError(f"duplicate sample: {sample_id}")
            seen_samples.add(sample_id)
            capture_id = spec["capture_id"]
            meta, directory = verify_capture(capture_id)
            rgb_path = directory / "rgb_original.png"
            depth_path = directory / "depth_metric.npy"
            labels_path = directory / "semantic_instance_labels.png"
            camera_path = directory / "camera_info.json"
            tf_path = directory / "tf_snapshot.json"
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            labels = cv2.imread(str(labels_path), cv2.IMREAD_UNCHANGED)
            depth_m = np.load(depth_path, allow_pickle=False)
            if (
                rgb is None or labels is None or rgb.shape[:2] != (480, 640)
                or labels.shape != (480, 640) or depth_m.shape != (480, 640)
                or depth_m.dtype != np.float32
            ):
                raise TestIIDMaterializeError(f"RGB/depth/label decode contract failed: {capture_id}")

            relative_clean = relative_depth(depth_m)
            relative_clean_path = DATASET_ROOT / f"derived/captures/{capture_id}/depth_relative_clean.png"
            write_png(relative_clean_path, relative_clean)
            media_dir = DATASET_ROOT / "media" / sample_id
            evaluator_dir = DATASET_ROOT / "evaluator" / sample_id
            media_dir.mkdir(parents=True, exist_ok=True)
            evaluator_dir.mkdir(parents=True, exist_ok=True)
            rgb_model_path = media_dir / "rgb_model_input.jpg"
            depth_model_path = media_dir / "depth_relative_model_input.png"
            if not cv2.imwrite(str(rgb_model_path), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise TestIIDMaterializeError(f"cannot write model RGB: {sample_id}")

            perturbation = copy.deepcopy(spec["perturbation"])
            corruption_kind = None
            if variant == "depth_corruption":
                corruption_kind = str(perturbation["kind"])
            elif variant == "clean" and spec["state_submode"] == "depth_invalid_or_corrupt":
                corruption_kind = "localized_holes_edges"
            elif variant == "clean" and spec["state_submode"] == "cross_modal_conflict":
                corruption_kind = "cross_modal_shift"
            if corruption_kind:
                model_depth, realized = corrupt_depth(
                    relative_clean, int(family["seed_bundle"]["sensor"]), corruption_kind
                )
                perturbation["realized"] = realized
            else:
                model_depth = relative_clean
            write_png(depth_model_path, model_depth)

            valid_labels = [int(by_id[value]["label"]) for value in spec["valid_target_ids"]]
            anchor_labels = [int(by_id[value]["label"]) for value in spec["anchor_ids"]]
            target = binary_mask(labels, valid_labels)
            interior = adaptive_erode(target, 4)
            valid_depth = (
                np.isfinite(depth_m) & (depth_m >= 0.10) & (depth_m <= 2.0)
            ).astype(np.uint8) * 255
            graspable = cv2.bitwise_and(adaptive_erode(target, 7), valid_depth)
            if spec["answerability_state"] != "FOUND":
                graspable[:] = 0
            reachable = graspable.copy()
            if spec["valid_target_ids"]:
                model_name = by_id[spec["valid_target_ids"][0]]["model_name"]
                x_value, y_value, _ = meta["requested_layout_base_link"][model_name]
                if not (-0.55 <= float(x_value) <= -0.05 and 0.08 <= float(y_value) <= 0.62):
                    reachable[:] = 0

            mask_paths = {
                "target": evaluator_dir / "target_mask.png",
                "target_interior": evaluator_dir / "target_interior_mask.png",
                "graspable": evaluator_dir / "graspable_mask.png",
                "reachable": evaluator_dir / "reachable_mask.png",
                "valid_depth": evaluator_dir / "valid_depth_mask.png",
            }
            for name, value in (
                ("target", target), ("target_interior", interior),
                ("graspable", graspable), ("reachable", reachable),
                ("valid_depth", valid_depth),
            ):
                write_png(mask_paths[name], value)
            anchor_refs = []
            for anchor_id, label in zip(spec["anchor_ids"], anchor_labels):
                path = evaluator_dir / f"anchor_{anchor_id}_mask.png"
                write_png(path, binary_mask(labels, [label]))
                anchor_refs.append({"object_id": anchor_id, **file_ref(path, DATASET_ROOT)})

            relation_graph = [{
                "source_ids": list(spec["valid_target_ids"]),
                "predicate": relation,
                "target_ids": list(spec["anchor_ids"]),
                "clause_index": index,
                "reference_frame": spec["reference_frame"],
            } for index, relation in enumerate(spec["relations"])]
            used_ids = sorted(set(
                spec["candidate_target_ids"] + spec["valid_target_ids"] + spec["anchor_ids"]
            ))
            active_ids = sorted(
                row["id"] for model_name, row in registry.items()
                if float(meta["requested_layout_base_link"][model_name][1]) <= 0.65
            )
            record = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "sample_id": sample_id,
                "family_id": family["family_id"],
                "split": "test_iid",
                "variant": variant,
                "family_category": family["family_category"],
                "target_object_group": family.get("target_object_group"),
                "ood_axis": "none",
                "depth_dependent": bool(family["depth_dependent"]),
                "instruction": spec["instruction"],
                "inference_payload": {
                    "rgb_model_input": file_ref(rgb_model_path, DATASET_ROOT),
                    "depth_relative_model_input": file_ref(depth_model_path, DATASET_ROOT),
                    "enable_depth": True,
                    "prompt": f"{spec['instruction']} {plan['inference_payload_template']['coordinate_suffix']}",
                    "coordinate_suffix": plan["inference_payload_template"]["coordinate_suffix"],
                },
                "sensor_evidence": {
                    "rgb_original": file_ref(rgb_path, DATASET_ROOT),
                    "depth_metric": file_ref(depth_path, DATASET_ROOT),
                    "depth_relative_clean": file_ref(relative_clean_path, DATASET_ROOT),
                    "camera_info": file_ref(camera_path, DATASET_ROOT),
                    "tf_snapshot": file_ref(tf_path, DATASET_ROOT),
                    "registered_rgb_depth_labels": True,
                    "capture_timestamps": meta["capture_timestamps"],
                },
                "evaluator_only": {
                    "access_policy": "never_export_to_inference_payload",
                    "semantic_instance_labels": file_ref(labels_path, DATASET_ROOT),
                    "masks": {
                        "target": file_ref(mask_paths["target"], DATASET_ROOT),
                        "target_interior": file_ref(mask_paths["target_interior"], DATASET_ROOT),
                        "anchor": anchor_refs,
                        "graspable": file_ref(mask_paths["graspable"], DATASET_ROOT),
                        "reachable": file_ref(mask_paths["reachable"], DATASET_ROOT),
                        "valid_depth": file_ref(mask_paths["valid_depth"], DATASET_ROOT),
                    },
                    "spatial_label": {
                        "candidate_target_ids": list(spec["candidate_target_ids"]),
                        "anchor_ids": list(spec["anchor_ids"]),
                        "valid_target_ids": list(spec["valid_target_ids"]),
                        "relations": list(spec["relations"]),
                        "relation_graph": relation_graph,
                        "reference_frame": spec["reference_frame"],
                        "geometry_version": plan["observable_relation_contract"]["geometry_version"],
                    },
                    "uncertainty_label": {
                        "answerable": bool(spec["answerable"]),
                        "state": spec["answerability_state"],
                        "state_submode": spec["state_submode"],
                        "sources": list(spec["uncertainty_sources"]),
                        "severity": int(spec["severity"]),
                        "expected_intervention": spec["expected_intervention"],
                    },
                    "object_oracle": {
                        "requested_layout_base_link": meta["requested_layout_base_link"],
                        "active_scene_ids": active_ids,
                        "label_ids": {value: int(by_id[value]["label"]) for value in used_ids},
                    },
                },
                "provenance": {
                    "seed_bundle": family["seed_bundle"],
                    "generator_version": manifest["generator_version"],
                    "capture_id": capture_id,
                    "capture_backend": "Gazebo Fortress RGB-D + semantic segmentation",
                    "perturbation": perturbation,
                    "test_policy": "FROZEN_MODEL_AND_CALIBRATOR_NO_TEST_TUNING",
                    "code_config_checkpoint_sha256": provenance,
                },
                "artifact_sha256": {},
            }
            record["artifact_sha256"] = artifact_hash_map(record)
            record_path = DATASET_ROOT / f"records/test_iid/{family['family_id']}/{variant}/record.json"
            write_json(record_path, record)
            records.append(record)

    if len(records) != SAMPLE_COUNT:
        raise TestIIDMaterializeError(f"record count differs: {len(records)} != {SAMPLE_COUNT}")
    index = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "selection": "official_independent_test_iid",
        "eligible_for_training_or_dev_or_calibration": False,
        "eligible_for_test_iid": True,
        "record_count": SAMPLE_COUNT,
        "family_count": FAMILY_COUNT,
        "split_family_counts": {"test_iid": FAMILY_COUNT},
        "records": [],
    }
    for record in records:
        path = DATASET_ROOT / f"records/test_iid/{record['family_id']}/{record['variant']}/record.json"
        index["records"].append({
            "sample_id": record["sample_id"],
            "family_id": record["family_id"],
            "split": "test_iid",
            "variant": record["variant"],
            "record_path": str(path.relative_to(DATASET_ROOT)),
            "record_sha256": sha256_file(path),
        })
    write_json(DATASET_ROOT / "dataset_index.json", index)
    write_json(DATASET_ROOT / "report_assets/checkpoints/02_materialized.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "checkpoint": "TEST_IID_1000_MATERIALIZED",
        "created_at_utc": utc_now(),
        "record_count": SAMPLE_COUNT,
        "family_count": FAMILY_COUNT,
        "dataset_index_sha256": sha256_file(DATASET_ROOT / "dataset_index.json"),
        "checkpoint_frozen": True,
        "calibrator_frozen": True,
        "training_performed": False,
        "test_inference_performed": False,
    })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize-frozen-inference", action="store_true")
    args = parser.parse_args()
    if not args.authorize_frozen_inference:
        raise TestIIDMaterializeError("explicit --authorize-frozen-inference is required")
    records = materialize()
    print(f"TEST_IID_MATERIALIZE_COMPLETE records={len(records)} families={FAMILY_COUNT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
