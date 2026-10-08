#!/usr/bin/env python3
"""Materialize Dataset v1 records from locked WP2 Gazebo raw captures."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import jsonschema
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp2_common import (  # noqa: E402
    EXPECTED_MODEL_INVENTORY_SHA256,
    GENERATOR_VERSION,
    PROTOCOL_ID,
    VARIANTS,
    WP2Error,
    artifact_entry,
    read_json,
    sha256_file,
    utc_now,
    write_json,
    workspace_root,
)


def file_ref(path: Path, dataset_root: Path) -> dict:
    return artifact_entry(path, dataset_root)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise WP2Error(f"cannot write PNG: {path}")


def relative_depth(depth_m: np.ndarray) -> np.ndarray:
    values = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(values) & (values >= 0.10) & (values <= 2.0)
    if int(np.count_nonzero(valid)) < 16:
        raise WP2Error("fewer than 16 valid metric-depth pixels")
    visible = values[valid]
    near = float(np.percentile(visible, 2.0))
    far = float(np.percentile(visible, 98.0))
    if far - near < 1e-4:
        near, far = float(visible.min()), float(visible.max())
    if far - near < 1e-6:
        raise WP2Error("metric depth has no usable range")
    clipped = np.clip(values, near, far)
    inverse = 1.0 / np.maximum(clipped, 1e-6)
    normalized = (inverse - 1.0 / far) / max(1.0 / near - 1.0 / far, 1e-6)
    gray = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    gray[~valid] = 0
    return np.repeat(gray[..., None], 3, axis=2)


def corrupt_depth(clean: np.ndarray, seed: int, perturbation: dict) -> tuple[np.ndarray, dict]:
    source = np.asarray(clean, dtype=np.uint8)
    rng = np.random.default_rng(int(seed))
    kind = str(perturbation["kind"])
    output = source.astype(np.int16)
    details: dict[str, Any] = {"kind": kind, "severity": int(perturbation["severity"])}
    if kind == "bias_noise":
        noise = rng.normal(0.0, 5.0, size=source.shape[:2])
        output = np.clip(output + 14 + noise[..., None], 0, 255).astype(np.uint8)
        details.update({"gray_bias": 14, "gaussian_sigma": 5.0})
    elif kind == "localized_holes_edges":
        output = source.copy()
        gray = source[..., 0]
        mask = cv2.dilate(
            (cv2.Canny(gray, 24, 72) > 0).astype(np.uint8),
            np.ones((7, 7), dtype=np.uint8),
        ) > 0
        height, width = gray.shape
        for nx, ny in ((0.25, 0.35), (0.50, 0.50), (0.75, 0.65)):
            cx, cy = int(round(nx * (width - 1))), int(round(ny * (height - 1)))
            half_w, half_h = max(4, width // 20), max(4, height // 20)
            mask[max(0, cy-half_h):min(height, cy+half_h), max(0, cx-half_w):min(width, cx+half_w)] = True
        output[mask] = 0
        details["corrupted_fraction"] = float(np.count_nonzero(mask) / mask.size)
    elif kind == "inversion_shift":
        output = 255 - source
        output = np.roll(output, shift=(9, -13), axis=(0, 1))
        details.update({"inverted": True, "shift_yx_pixels": [9, -13]})
    else:
        raise WP2Error(f"unknown depth perturbation: {kind}")
    return np.asarray(output, dtype=np.uint8), details


def binary_mask(labels: np.ndarray, label_ids: list[int]) -> np.ndarray:
    if not label_ids:
        return np.zeros(labels.shape, dtype=np.uint8)
    return (np.isin(labels, np.asarray(label_ids, dtype=np.uint8)).astype(np.uint8) * 255)


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    if int(np.count_nonzero(mask)) == 0:
        return np.zeros_like(mask)
    kernel_size = 2 * int(pixels) + 1
    return cv2.erode(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)


def adaptive_erode(mask: np.ndarray, max_pixels: int, minimum_pixels: int = 16) -> np.ndarray:
    """Use the largest safe erosion that preserves a usable visible interior."""
    if int(np.count_nonzero(mask)) == 0:
        return np.zeros_like(mask)
    for pixels in range(int(max_pixels), 0, -1):
        candidate = erode(mask, pixels)
        if int(np.count_nonzero(candidate)) >= int(minimum_pixels):
            return candidate
    return mask.copy()


def resolve_raw(source_root: Path, relative: str) -> Path:
    path = (source_root / relative).resolve()
    if source_root.resolve() not in path.parents:
        raise WP2Error(f"unsafe raw path: {relative}")
    return path


def verify_capture(source_root: Path, capture_id: str) -> tuple[dict, Path]:
    capture_dir = source_root / "raw" / "captures" / capture_id
    meta = read_json(capture_dir / "capture_meta.json")
    if meta.get("capture_id") != capture_id:
        raise WP2Error(f"capture metadata mismatch: {capture_id}")
    for relative, expected in meta["artifact_sha256"].items():
        path = capture_dir / relative
        if sha256_file(path) != expected:
            raise WP2Error(f"raw artifact hash mismatch: {capture_id}/{relative}")
    return meta, capture_dir


def artifact_hash_map(record: dict) -> dict[str, str]:
    values: dict[str, str] = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            if {"path", "sha256", "bytes"}.issubset(value):
                values[prefix] = str(value["sha256"])
            for key, child in value.items():
                visit(child, f"{prefix}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")

    for field in ("inference_payload", "sensor_evidence", "evaluator_only"):
        visit(record[field], field)
    return values


def materialize(
    source_root: Path,
    output_root: Path,
    selection: str,
    validate_schema: bool = True,
) -> list[dict]:
    manifest = read_json(source_root / "family_manifest.json")
    object_registry = read_json(source_root / "object_registry.json")
    capture_plan = read_json(source_root / "capture_plan.json")
    coordinate_suffix = str(capture_plan["coordinate_suffix"])
    id_to_object = {item["id"]: item for item in object_registry.values()}
    schema = read_json(Path(__file__).resolve().parent / "dataset_v1.schema.json")
    smoke_ids = set(read_json(source_root / "smoke_selection.json")["family_ids"])
    families = [
        family for family in manifest["families"]
        if selection == "all" or family["family_id"] in smoke_ids
    ]
    records = []
    workspace = workspace_root()
    locked_source_files = {
        "wp2_family_generator": Path(__file__).resolve().parent / "wp2_family_generator.py",
        "wp2_gazebo_capture": Path(__file__).resolve().parent / "wp2_gazebo_capture.py",
        "wp2_capture_launch": Path(__file__).resolve().parent / "wp2_gazebo_capture.launch.py",
        "wp2_validator": Path(__file__).resolve().parent / "wp2_validate.py",
        "family_manifest_schema": Path(__file__).resolve().parent / "family_manifest_v1.schema.json",
        "gazebo_world": workspace / "ur3/ur_simulation_gz/worlds/ur3_pick_place.sdf",
        "camera_xacro": workspace / "ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro",
        "camera_view_controller": workspace / "ur3/ur3_perception/scripts/move_camera_to_view.py",
    }
    source_hashes = {
        "family_manifest": sha256_file(source_root / "family_manifest.json"),
        "capture_plan": sha256_file(source_root / "capture_plan.json"),
        "object_registry": sha256_file(source_root / "object_registry.json"),
        "dataset_schema": sha256_file(Path(__file__).resolve().parent / "dataset_v1.schema.json"),
        "materializer": sha256_file(Path(__file__).resolve()),
        "roborefer_model_inventory": EXPECTED_MODEL_INVENTORY_SHA256,
        **{name: sha256_file(path) for name, path in locked_source_files.items()},
    }
    output_root.mkdir(parents=True, exist_ok=True)

    for family in families:
        for spec in family["variant_specs"]:
            variant = spec["variant"]
            sample_id = f"{family['family_id']}__{variant}"
            capture_id = spec["capture_id"]
            capture_meta, capture_dir = verify_capture(source_root, capture_id)
            rgb_path = capture_dir / "rgb_original.png"
            metric_path = capture_dir / "depth_metric.npy"
            labels_path = capture_dir / "semantic_instance_labels.png"
            camera_path = capture_dir / "camera_info.json"
            tf_path = capture_dir / "tf_snapshot.json"
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            labels = cv2.imread(str(labels_path), cv2.IMREAD_UNCHANGED)
            depth_m = np.load(metric_path, allow_pickle=False)
            if rgb is None or labels is None or rgb.shape[:2] != labels.shape or labels.shape != depth_m.shape:
                raise WP2Error(f"raw capture shape/decode failure: {capture_id}")

            clean_relative = relative_depth(depth_m)
            clean_relative_path = output_root / "derived" / "captures" / capture_id / "depth_relative_clean.png"
            write_png(clean_relative_path, clean_relative)

            media_dir = output_root / "media" / sample_id
            evaluator_dir = output_root / "evaluator" / sample_id
            media_dir.mkdir(parents=True, exist_ok=True)
            evaluator_dir.mkdir(parents=True, exist_ok=True)
            rgb_model_path = media_dir / "rgb_model_input.jpg"
            depth_model_path = media_dir / "depth_relative_model_input.png"
            if not cv2.imwrite(str(rgb_model_path), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise WP2Error(f"cannot write model RGB: {sample_id}")
            perturbation = dict(spec["perturbation"])
            if variant == "depth_corruption":
                model_depth, corruption_details = corrupt_depth(
                    clean_relative, family["seed"] + 404, perturbation
                )
                perturbation["realized"] = corruption_details
            else:
                model_depth = clean_relative
            write_png(depth_model_path, model_depth)

            target_labels = [int(id_to_object[item]["label"]) for item in spec["target_ids"]]
            anchor_labels = [int(id_to_object[item]["label"]) for item in spec["anchor_ids"]]
            target = binary_mask(labels, target_labels)
            interior = adaptive_erode(target, 4)
            valid_depth = (
                np.isfinite(depth_m) & (depth_m >= 0.10) & (depth_m <= 2.0)
            ).astype(np.uint8) * 255
            graspable = cv2.bitwise_and(adaptive_erode(target, 7), valid_depth)
            if spec["answerability_state"] != "FOUND":
                graspable = np.zeros_like(graspable)
            reachable = graspable.copy()
            # This is a preregistered 2D projection proxy for prototype label
            # plumbing, not a replacement for MoveIt collision/reachability.
            if spec["target_ids"]:
                first_model = next(
                    name for name, value in object_registry.items()
                    if value["id"] == spec["target_ids"][0]
                )
                x_value, y_value, _ = capture_meta["requested_layout_base_link"][first_model]
                if not (-0.55 <= float(x_value) <= -0.08 and 0.08 <= float(y_value) <= 0.55):
                    reachable[:] = 0

            mask_paths = {
                "target": evaluator_dir / "target_mask.png",
                "target_interior": evaluator_dir / "target_interior_mask.png",
                "graspable": evaluator_dir / "graspable_mask.png",
                "reachable": evaluator_dir / "reachable_mask.png",
                "valid_depth": evaluator_dir / "valid_depth_mask.png",
            }
            for key, mask in (
                ("target", target), ("target_interior", interior),
                ("graspable", graspable), ("reachable", reachable),
                ("valid_depth", valid_depth),
            ):
                write_png(mask_paths[key], mask)
            anchor_refs = []
            for anchor_id, label_id in zip(spec["anchor_ids"], anchor_labels):
                path = evaluator_dir / f"anchor_{anchor_id}_mask.png"
                write_png(path, binary_mask(labels, [label_id]))
                anchor_refs.append({"object_id": anchor_id, **file_ref(path, output_root)})

            relation_graph = []
            for relation_index, relation in enumerate(spec["relations"]):
                relation_graph.append({
                    "source_ids": list(spec["target_ids"]),
                    "predicate": relation,
                    "target_ids": list(spec["anchor_ids"]),
                    "clause_index": relation_index,
                    "reference_frame": spec["reference_frame"],
                })
            label_ids = {
                object_id: int(id_to_object[object_id]["label"])
                for object_id in sorted(set(spec["target_ids"] + spec["anchor_ids"]))
            }
            record = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "sample_id": sample_id,
                "family_id": family["family_id"],
                "split": family["split"],
                "variant": variant,
                "family_category": family["category"],
                "depth_dependent": bool(family["depth_dependent"]),
                "instruction": spec["instruction"],
                "inference_payload": {
                    "rgb_model_input": file_ref(rgb_model_path, output_root),
                    "depth_relative_model_input": file_ref(depth_model_path, output_root),
                    "enable_depth": True,
                    "prompt": f"{spec['instruction']} {coordinate_suffix}",
                    "coordinate_suffix": coordinate_suffix,
                },
                "sensor_evidence": {
                    "rgb_original": file_ref(rgb_path, source_root),
                    "depth_metric": file_ref(metric_path, source_root),
                    "depth_relative_clean": file_ref(clean_relative_path, output_root),
                    "camera_info": file_ref(camera_path, source_root),
                    "tf_snapshot": file_ref(tf_path, source_root),
                    "registered_rgb_depth": True,
                    "capture_timestamps": capture_meta["capture_timestamps"],
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
                        "target_ids": list(spec["target_ids"]),
                        "anchor_ids": list(spec["anchor_ids"]),
                        "valid_target_ids": list(spec["valid_target_ids"]),
                        "relations": list(spec["relations"]),
                        "relation_graph": relation_graph,
                        "reference_frame": spec["reference_frame"],
                    },
                    "uncertainty_label": {
                        "answerable": bool(spec["answerable"]),
                        "state": spec["answerability_state"],
                        "sources": list(spec["uncertainty_sources"]),
                        "severity": int(spec["severity"]),
                        "expected_intervention": spec["expected_intervention"],
                    },
                    "object_oracle": {
                        "requested_layout_base_link": capture_meta["requested_layout_base_link"],
                        "label_ids": label_ids,
                        "reachability_label_policy": "locked_base_xy_and_visible_valid_depth_proxy_not_MoveIt_collision_checked",
                    },
                },
                "provenance": {
                    "seed": int(family["seed"]),
                    "generator_version": GENERATOR_VERSION,
                    "capture_id": capture_id,
                    "capture_backend": "Gazebo Fortress RGB-D + semantic segmentation",
                    "perturbation": perturbation,
                    "source_dataset_ids": [],
                    "code_config_checkpoint_sha256": source_hashes,
                },
                "artifact_sha256": {},
            }
            record["artifact_sha256"] = artifact_hash_map(record)
            if validate_schema:
                jsonschema.validate(record, schema)
            record_path = (
                output_root / "records" / family["split"] / family["family_id"]
                / variant / "record.json"
            )
            write_json(record_path, record)
            records.append(record)

    index_name = "smoke_dataset_index.json" if selection == "smoke" else "dataset_index.json"
    index = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "selection": selection,
        "record_count": len(records),
        "family_count": len({item["family_id"] for item in records}),
        "records": [
            {
                "sample_id": item["sample_id"], "family_id": item["family_id"],
                "split": item["split"], "variant": item["variant"],
                "record_path": str(
                    (output_root / "records" / item["split"] / item["family_id"]
                     / item["variant"] / "record.json").relative_to(output_root)
                ),
                "record_sha256": sha256_file(
                    output_root / "records" / item["split"] / item["family_id"]
                    / item["variant"] / "record.json"
                ),
            }
            for item in records
        ],
    }
    write_json(output_root / index_name, index)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--selection", choices=("smoke", "all"), default="all")
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve() if args.output_root else source_root
    )
    records = materialize(source_root, output_root, args.selection)
    if not args.no_checkpoint and output_root == source_root:
        checkpoint_name = (
            "02_smoke_materialized.json" if args.selection == "smoke"
            else "04_full_materialized.json"
        )
        index_name = "smoke_dataset_index.json" if args.selection == "smoke" else "dataset_index.json"
        write_json(output_root / "report_assets" / "checkpoints" / checkpoint_name, {
            "schema_version": 1,
            "checkpoint": "SMOKE_MATERIALIZED" if args.selection == "smoke" else "FULL_MATERIALIZED",
            "created_at_utc": utc_now(),
            "selection": args.selection,
            "record_count": len(records),
            "family_count": len({item["family_id"] for item in records}),
            "dataset_index_sha256": sha256_file(output_root / index_name),
            "materializer_source_sha256": sha256_file(Path(__file__).resolve()),
        })
    print(f"WP2_MATERIALIZE_COMPLETE selection={args.selection} records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
