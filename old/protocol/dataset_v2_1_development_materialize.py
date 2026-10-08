#!/usr/bin/env python3
"""Materialize 2,000 Dataset V2.1 records from 800 qualified captures."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2_1_development_generator import GENERATOR_VERSION, PROTOCOL_ID  # noqa: E402
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


EXPECTED_FAMILIES = 400
EXPECTED_CAPTURES = 800
EXPECTED_RECORDS = 2000


class V21MaterializeError(RuntimeError):
    pass


def verify_capture(dataset_root: Path, capture_id: str) -> tuple[dict[str, Any], Path]:
    directory = dataset_root / "raw" / "captures" / capture_id
    meta_path = directory / "capture_meta.json"
    if not meta_path.is_file():
        raise V21MaterializeError(f"missing raw capture metadata: {capture_id}")
    meta = read_json(meta_path)
    if meta.get("protocol_id") != PROTOCOL_ID or meta.get("capture_id") != capture_id:
        raise V21MaterializeError(f"raw capture identity mismatch: {capture_id}")
    for relative, expected in meta["artifact_sha256"].items():
        path = directory / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise V21MaterializeError(f"raw capture hash mismatch: {capture_id}/{relative}")
    return meta, directory


def source_hashes(workspace: Path, plan: dict[str, Any]) -> dict[str, str]:
    paths = {
        "contract": workspace / "protocol/DATASET_V2_1_DEVELOPMENT_CAPTURE_CONTRACT.md",
        "generator": workspace / "protocol/dataset_v2_1_development_generator.py",
        "manifest": workspace / "protocol/dataset_v2_1_development_manifest.json",
        "capture_plan": workspace / "protocol/dataset_v2_1_development_capture_plan.json",
        "execution_lock": workspace / "protocol/dataset_v2_1_development_execution_lock.json",
        "capture_runtime": workspace / "protocol/dataset_v2_1_development_capture.py",
        "batch_qc": workspace / "protocol/dataset_v2_1_development_batch_qc.py",
        "materializer": Path(__file__).resolve(),
        "full_qc": workspace / "protocol/dataset_v2_1_development_full_qc.py",
        "relation_geometry": workspace / "protocol/dataset_v2_relation_geometry_v2_1.py",
        "gazebo_world": workspace / plan["world_file"],
        "camera_xacro": workspace / "ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise V21MaterializeError(f"materialization source artifacts missing: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def materialize(source_root: Path, output_root: Path) -> list[dict[str, Any]]:
    workspace = Path(__file__).resolve().parents[1]
    protocol = workspace / "protocol"
    manifest = read_json(protocol / "dataset_v2_1_development_manifest.json")
    plan = read_json(protocol / "dataset_v2_1_development_capture_plan.json")
    lock = read_json(protocol / "dataset_v2_1_development_execution_lock.json")
    raw_manifest = read_json(source_root / "raw/raw_capture_manifest.json")
    if (
        raw_manifest.get("protocol_id") != PROTOCOL_ID
        or raw_manifest.get("capture_count") != EXPECTED_CAPTURES
        or raw_manifest.get("complete") is not True
    ):
        raise V21MaterializeError("exactly 800 complete V2.1 raw captures are required")
    if manifest.get("family_count") != EXPECTED_FAMILIES or plan.get("capture_count") != EXPECTED_CAPTURES:
        raise V21MaterializeError("locked manifest/capture plan count differs")
    if not lock.get("capture_authorized") or lock.get("training_authorized"):
        raise V21MaterializeError("execution lock is not capture-only authorized")

    registry = plan["object_registry"]
    by_id = {row["id"]: {**row, "model_name": name} for name, row in registry.items()}
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "family_manifest.json", manifest)
    write_json(output_root / "capture_plan.json", plan)
    write_json(output_root / "object_registry.json", registry)
    write_json(output_root / "execution_lock.json", lock)
    provenance_hashes = source_hashes(workspace, plan)
    records: list[dict[str, Any]] = []

    for family in manifest["families"]:
        for spec in family["variant_specs"]:
            variant = spec["variant"]
            sample_id = f"{family['family_id']}__{variant}"
            capture_id = spec["capture_id"]
            meta, directory = verify_capture(source_root, capture_id)
            rgb_path = directory / "rgb_original.png"
            depth_path = directory / "depth_metric.npy"
            labels_path = directory / "semantic_instance_labels.png"
            camera_path = directory / "camera_info.json"
            tf_path = directory / "tf_snapshot.json"
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            labels = cv2.imread(str(labels_path), cv2.IMREAD_UNCHANGED)
            depth_m = np.load(depth_path, allow_pickle=False)
            if rgb is None or labels is None or rgb.shape[:2] != labels.shape or labels.shape != depth_m.shape:
                raise V21MaterializeError(f"raw RGB-depth-label decode/shape failure: {capture_id}")

            relative_clean = relative_depth(depth_m)
            relative_clean_path = output_root / "derived/captures" / capture_id / "depth_relative_clean.png"
            write_png(relative_clean_path, relative_clean)
            media_dir = output_root / "media" / sample_id
            evaluator_dir = output_root / "evaluator" / sample_id
            media_dir.mkdir(parents=True, exist_ok=True)
            evaluator_dir.mkdir(parents=True, exist_ok=True)
            rgb_model_path = media_dir / "rgb_model_input.jpg"
            depth_model_path = media_dir / "depth_relative_model_input.png"
            if not cv2.imwrite(str(rgb_model_path), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise V21MaterializeError(f"cannot write model RGB: {sample_id}")

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
                model = by_id[spec["valid_target_ids"][0]]["model_name"]
                x_value, y_value, _ = meta["requested_layout_base_link"][model]
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
                anchor_refs.append({"object_id": anchor_id, **file_ref(path, output_root)})

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
                row["id"] for model, row in registry.items()
                if float(meta["requested_layout_base_link"][model][1]) <= 0.65
            )
            record = {
                "schema_version": 1, "protocol_id": PROTOCOL_ID,
                "sample_id": sample_id, "family_id": family["family_id"],
                "split": family["split"], "variant": variant,
                "family_category": family["family_category"],
                "ood_axis": family["ood_axis"],
                "depth_dependent": bool(family["depth_dependent"]),
                "instruction": spec["instruction"],
                "inference_payload": {
                    "rgb_model_input": file_ref(rgb_model_path, output_root),
                    "depth_relative_model_input": file_ref(depth_model_path, output_root),
                    "enable_depth": True,
                    "prompt": f"{spec['instruction']} {plan['inference_payload_template']['coordinate_suffix']}",
                    "coordinate_suffix": plan["inference_payload_template"]["coordinate_suffix"],
                },
                "sensor_evidence": {
                    "rgb_original": file_ref(rgb_path, source_root),
                    "depth_metric": file_ref(depth_path, source_root),
                    "depth_relative_clean": file_ref(relative_clean_path, output_root),
                    "camera_info": file_ref(camera_path, source_root),
                    "tf_snapshot": file_ref(tf_path, source_root),
                    "registered_rgb_depth_labels": True,
                    "capture_timestamps": meta["capture_timestamps"],
                },
                "evaluator_only": {
                    "access_policy": "never_export_to_inference_payload",
                    "semantic_instance_labels": file_ref(labels_path, source_root),
                    "masks": {
                        "target": file_ref(mask_paths["target"], output_root),
                        "target_interior": file_ref(mask_paths["target_interior"], output_root),
                        "anchor": anchor_refs,
                        "graspable": file_ref(mask_paths["graspable"], output_root),
                        "reachable": file_ref(mask_paths["reachable"], output_root),
                        "valid_depth": file_ref(mask_paths["valid_depth"], output_root),
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
                        "reachability_label_policy": "locked_base_xy_visible_valid_depth_proxy_not_MoveIt_collision_checked",
                    },
                },
                "provenance": {
                    "seed_bundle": family["seed_bundle"],
                    "generator_version": GENERATOR_VERSION,
                    "capture_id": capture_id,
                    "capture_backend": "Gazebo Fortress RGB-D + semantic segmentation",
                    "perturbation": perturbation,
                    "official_development_policy": "TRAIN_OR_DEV_ONLY_FAMILY_LOCKED",
                    "code_config_checkpoint_sha256": provenance_hashes,
                },
                "artifact_sha256": {},
            }
            record["artifact_sha256"] = artifact_hash_map(record)
            record_path = output_root / "records" / family["split"] / family["family_id"] / variant / "record.json"
            write_json(record_path, record)
            records.append(record)

    if len(records) != EXPECTED_RECORDS:
        raise V21MaterializeError(f"record count differs: {len(records)} != {EXPECTED_RECORDS}")
    index = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "selection": "official_development_train_dev",
        "eligible_for_official_development": True,
        "record_count": len(records), "family_count": EXPECTED_FAMILIES,
        "split_family_counts": {"train": 320, "dev": 80},
        "records": [],
    }
    for record in records:
        path = output_root / "records" / record["split"] / record["family_id"] / record["variant"] / "record.json"
        index["records"].append({
            "sample_id": record["sample_id"], "family_id": record["family_id"],
            "split": record["split"], "variant": record["variant"],
            "record_path": str(path.relative_to(output_root)),
            "record_sha256": sha256_file(path),
        })
    write_json(output_root / "dataset_index.json", index)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else source_root
    records = materialize(source_root, output_root)
    if not args.no_checkpoint and source_root == output_root:
        checkpoint = output_root / "report_assets/checkpoints/02_materialized.json"
        write_json(checkpoint, {
            "schema_version": 1, "protocol_id": PROTOCOL_ID,
            "checkpoint": "V2_1_DEVELOPMENT_2000_MATERIALIZED",
            "created_at_utc": utc_now(), "record_count": len(records),
            "family_count": EXPECTED_FAMILIES,
            "dataset_index_sha256": sha256_file(output_root / "dataset_index.json"),
            "materializer_source_sha256": sha256_file(Path(__file__).resolve()),
            "training_performed": False,
        })
    print(
        f"DATASET_V2_1_DEVELOPMENT_MATERIALIZE_COMPLETE "
        f"records={len(records)} families={EXPECTED_FAMILIES}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
