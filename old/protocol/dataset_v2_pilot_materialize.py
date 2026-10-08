#!/usr/bin/env python3
"""Materialize 150 Dataset V2 pilot-only records from 60 Gazebo captures."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import cv2
import jsonschema
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2_pilot_generator import GENERATOR_VERSION, PROTOCOL_ID, VARIANTS  # noqa: E402
from wp2_common import artifact_entry, read_json, sha256_file, utc_now, write_json  # noqa: E402


class PilotMaterializeError(RuntimeError):
    """Raised when real raw captures cannot produce a valid pilot dataset."""


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise PilotMaterializeError(f"cannot write PNG: {path}")


def file_ref(path: Path, root: Path) -> dict[str, Any]:
    return artifact_entry(path, root)


def relative_depth(depth_m: np.ndarray) -> np.ndarray:
    values = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(values) & (values >= .10) & (values <= 2.0)
    if int(np.count_nonzero(valid)) < 16:
        raise PilotMaterializeError("fewer than 16 valid metric-depth pixels")
    visible = values[valid]
    near, far = float(np.percentile(visible, 2)), float(np.percentile(visible, 98))
    if far - near < 1e-4:
        near, far = float(visible.min()), float(visible.max())
    if far - near < 1e-6:
        raise PilotMaterializeError("metric depth has no usable range")
    clipped = np.clip(values, near, far)
    inverse = 1.0 / np.maximum(clipped, 1e-6)
    normalized = (inverse - 1.0 / far) / max(1.0 / near - 1.0 / far, 1e-6)
    gray = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    gray[~valid] = 0
    return np.repeat(gray[..., None], 3, axis=2)


def corrupt_depth(source: np.ndarray, seed: int, kind: str) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    output = np.asarray(source, dtype=np.uint8).copy()
    details: dict[str, Any] = {"kind": kind}
    if kind == "bias_noise":
        noise = rng.normal(0, 5, size=source.shape[:2])
        output = np.clip(source.astype(np.int16) + 14 + noise[..., None], 0, 255).astype(np.uint8)
        details.update({"gray_bias": 14, "gaussian_sigma": 5.0})
    elif kind == "localized_holes_edges":
        gray = source[..., 0]
        mask = cv2.dilate((cv2.Canny(gray, 24, 72) > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        height, width = gray.shape
        for nx, ny in ((.25, .35), (.50, .50), (.75, .65)):
            cx, cy = int(nx * (width - 1)), int(ny * (height - 1))
            mask[max(0, cy-height//20):min(height, cy+height//20), max(0, cx-width//20):min(width, cx+width//20)] = True
        output[mask] = 0
        details["corrupted_fraction"] = float(np.count_nonzero(mask) / mask.size)
    elif kind == "inversion_shift":
        output = np.roll(255 - source, shift=(9, -13), axis=(0, 1))
        details.update({"inverted": True, "shift_yx_pixels": [9, -13]})
    elif kind == "heldout_stripe_dropout":
        height, width = source.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        phase = int(seed) % 17
        for x_value in range(phase, width, 31):
            mask[:, x_value:min(width, x_value + 9)] = True
        output[mask] = 0
        details.update({"stripe_width_px": 9, "period_px": 31, "phase_px": phase, "corrupted_fraction": float(mask.mean())})
    elif kind == "cross_modal_shift":
        output = np.roll(source, shift=(0, 37), axis=(0, 1))
        details["shift_yx_pixels"] = [0, 37]
    else:
        raise PilotMaterializeError(f"unknown depth corruption: {kind}")
    return output, details


def binary_mask(labels: np.ndarray, label_ids: list[int]) -> np.ndarray:
    if not label_ids:
        return np.zeros(labels.shape, dtype=np.uint8)
    return np.isin(labels, np.asarray(label_ids, dtype=np.uint8)).astype(np.uint8) * 255


def adaptive_erode(mask: np.ndarray, max_pixels: int, minimum_pixels: int = 16) -> np.ndarray:
    if int(np.count_nonzero(mask)) == 0:
        return np.zeros_like(mask)
    for pixels in range(int(max_pixels), 0, -1):
        kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), np.uint8)
        value = cv2.erode(mask, kernel, iterations=1)
        if int(np.count_nonzero(value)) >= minimum_pixels:
            return value
    return mask.copy()


def verify_capture(dataset_root: Path, capture_id: str) -> tuple[dict[str, Any], Path]:
    capture_dir = dataset_root / "raw" / "captures" / capture_id
    meta = read_json(capture_dir / "capture_meta.json")
    if meta.get("protocol_id") != PROTOCOL_ID or meta.get("capture_id") != capture_id:
        raise PilotMaterializeError(f"raw capture metadata mismatch: {capture_id}")
    for relative, expected in meta["artifact_sha256"].items():
        path = capture_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise PilotMaterializeError(f"raw capture hash mismatch: {capture_id}/{relative}")
    return meta, capture_dir


def artifact_hash_map(record: dict[str, Any]) -> dict[str, str]:
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


def materialize(source_root: Path, output_root: Path, validate_schema: bool = True) -> list[dict[str, Any]]:
    workspace = Path(__file__).resolve().parents[1]
    protocol = workspace / "protocol"
    manifest = read_json(protocol / "dataset_v2_pilot_manifest.json")
    capture_plan = read_json(protocol / "dataset_v2_pilot_capture_plan.json")
    raw_manifest = read_json(source_root / "raw" / "raw_capture_manifest.json")
    if not raw_manifest.get("complete") or raw_manifest.get("capture_count") != 60:
        raise PilotMaterializeError("exactly 60 complete raw captures are required")
    registry = capture_plan["object_registry"]
    by_id = {value["id"]: {**value, "model_name": name} for name, value in registry.items()}
    schema = read_json(protocol / "dataset_v2_pilot_record.schema.json")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "family_manifest.json", manifest)
    write_json(output_root / "capture_plan.json", capture_plan)
    write_json(output_root / "object_registry.json", read_json(protocol / "dataset_v2_pilot_object_registry.json"))
    write_json(output_root / "execution_lock.json", read_json(protocol / "dataset_v2_pilot_execution_lock.json"))
    source_hash_paths = {
        "pilot_contract": protocol / "DATASET_V2_PILOT_CONTRACT.md",
        "pilot_generator": protocol / "dataset_v2_pilot_generator.py",
        "pilot_manifest": protocol / "dataset_v2_pilot_manifest.json",
        "pilot_capture_plan": protocol / "dataset_v2_pilot_capture_plan.json",
        "pilot_preflight": protocol / "dataset_v2_pilot_preflight.py",
        "pilot_execution_lock": protocol / "dataset_v2_pilot_execution_lock.json",
        "pilot_capture": protocol / "dataset_v2_pilot_capture.py",
        "pilot_capture_launch": protocol / "dataset_v2_pilot_capture.launch.py",
        "pilot_schema": protocol / "dataset_v2_pilot_record.schema.json",
        "pilot_materializer": Path(__file__).resolve(),
        "gazebo_world": workspace / capture_plan["world_file"],
        "camera_xacro": workspace / "ur3/ur3_moveit_control/urdf/d435i_wrist_camera.xacro",
    }
    source_hashes = {name: sha256_file(path) for name, path in source_hash_paths.items()}
    records: list[dict[str, Any]] = []
    for family in manifest["families"]:
        for spec in family["variant_specs"]:
            variant = spec["variant"]
            sample_id = f"{family['family_id']}__{variant}"
            capture_id = spec["capture_id"]
            capture_meta, capture_dir = verify_capture(source_root, capture_id)
            rgb_path = capture_dir / "rgb_original.png"
            depth_path = capture_dir / "depth_metric.npy"
            labels_path = capture_dir / "semantic_instance_labels.png"
            camera_path = capture_dir / "camera_info.json"
            tf_path = capture_dir / "tf_snapshot.json"
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            labels = cv2.imread(str(labels_path), cv2.IMREAD_UNCHANGED)
            depth_m = np.load(depth_path, allow_pickle=False)
            if rgb is None or labels is None or rgb.shape[:2] != labels.shape or labels.shape != depth_m.shape:
                raise PilotMaterializeError(f"raw capture decode/shape failure: {capture_id}")

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
                raise PilotMaterializeError(f"cannot write model RGB: {sample_id}")
            perturbation = copy.deepcopy(spec["perturbation"])
            corruption_kind = None
            if variant == "depth_corruption":
                corruption_kind = str(perturbation["kind"])
            elif variant == "clean" and spec["state_submode"] == "depth_invalid_or_corrupt":
                corruption_kind = "localized_holes_edges"
            elif variant == "clean" and spec["state_submode"] == "cross_modal_conflict":
                corruption_kind = "cross_modal_shift"
            if corruption_kind:
                model_depth, details = corrupt_depth(clean_relative, int(family["seed_bundle"]["sensor"]), corruption_kind)
                perturbation["realized"] = details
            else:
                model_depth = clean_relative
            write_png(depth_model_path, model_depth)

            valid_labels = [int(by_id[value]["label"]) for value in spec["valid_target_ids"]]
            anchor_labels = [int(by_id[value]["label"]) for value in spec["anchor_ids"]]
            target = binary_mask(labels, valid_labels)
            interior = adaptive_erode(target, 4)
            valid_depth = (np.isfinite(depth_m) & (depth_m >= .10) & (depth_m <= 2.0)).astype(np.uint8) * 255
            graspable = cv2.bitwise_and(adaptive_erode(target, 7), valid_depth)
            if spec["answerability_state"] != "FOUND":
                graspable[:] = 0
            reachable = graspable.copy()
            if spec["valid_target_ids"]:
                model_name = by_id[spec["valid_target_ids"][0]]["model_name"]
                x_value, y_value, _ = capture_meta["requested_layout_base_link"][model_name]
                if not (-.55 <= float(x_value) <= -.05 and .08 <= float(y_value) <= .58):
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
                "source_ids": list(spec["valid_target_ids"]), "predicate": relation,
                "target_ids": list(spec["anchor_ids"]), "clause_index": index,
                "reference_frame": spec["reference_frame"],
            } for index, relation in enumerate(spec["relations"])]
            used_ids = sorted(set(spec["candidate_target_ids"] + spec["valid_target_ids"] + spec["anchor_ids"]))
            active_ids = sorted(
                value["id"] for model_name, value in registry.items()
                if float(capture_meta["requested_layout_base_link"][model_name][1]) <= .65
            )
            record = {
                "schema_version": 1, "protocol_id": PROTOCOL_ID,
                "sample_id": sample_id, "family_id": family["family_id"], "split": "pilot_only",
                "variant": variant, "family_category": family["family_category"],
                "ood_axis": family["ood_axis"], "depth_dependent": bool(family["depth_dependent"]),
                "instruction": spec["instruction"],
                "inference_payload": {
                    "rgb_model_input": file_ref(rgb_model_path, output_root),
                    "depth_relative_model_input": file_ref(depth_model_path, output_root),
                    "enable_depth": True,
                    "prompt": f"{spec['instruction']} {capture_plan['coordinate_suffix']}",
                    "coordinate_suffix": capture_plan["coordinate_suffix"],
                },
                "sensor_evidence": {
                    "rgb_original": file_ref(rgb_path, source_root),
                    "depth_metric": file_ref(depth_path, source_root),
                    "depth_relative_clean": file_ref(clean_relative_path, output_root),
                    "camera_info": file_ref(camera_path, source_root),
                    "tf_snapshot": file_ref(tf_path, source_root),
                    "registered_rgb_depth_labels": True,
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
                        "candidate_target_ids": list(spec["candidate_target_ids"]),
                        "anchor_ids": list(spec["anchor_ids"]),
                        "valid_target_ids": list(spec["valid_target_ids"]),
                        "relations": list(spec["relations"]), "relation_graph": relation_graph,
                        "reference_frame": spec["reference_frame"],
                    },
                    "uncertainty_label": {
                        "answerable": bool(spec["answerable"]), "state": spec["answerability_state"],
                        "state_submode": spec["state_submode"],
                        "sources": list(spec["uncertainty_sources"]), "severity": int(spec["severity"]),
                        "expected_intervention": spec["expected_intervention"],
                    },
                    "object_oracle": {
                        "requested_layout_base_link": capture_meta["requested_layout_base_link"],
                        "active_scene_ids": active_ids,
                        "label_ids": {value: int(by_id[value]["label"]) for value in used_ids},
                        "reachability_label_policy": "pilot_only_locked_base_xy_visible_valid_depth_proxy_not_MoveIt_collision_checked",
                    },
                },
                "provenance": {
                    "seed_bundle": family["seed_bundle"], "generator_version": GENERATOR_VERSION,
                    "capture_id": capture_id,
                    "capture_backend": "Gazebo Fortress RGB-D + semantic segmentation",
                    "perturbation": perturbation,
                    "pilot_exclusion_policy": family["pilot_exclusion_policy"],
                    "code_config_checkpoint_sha256": source_hashes,
                },
                "artifact_sha256": {},
            }
            record["artifact_sha256"] = artifact_hash_map(record)
            if validate_schema:
                jsonschema.validate(record, schema)
            record_path = output_root / "records" / "pilot_only" / family["family_id"] / variant / "record.json"
            write_json(record_path, record)
            records.append(record)

    index = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "selection": "pilot_only",
        "eligible_for_official_dataset": False, "record_count": len(records),
        "family_count": len({item["family_id"] for item in records}),
        "records": [],
    }
    for record in records:
        path = output_root / "records" / "pilot_only" / record["family_id"] / record["variant"] / "record.json"
        index["records"].append({
            "sample_id": record["sample_id"], "family_id": record["family_id"],
            "split": "pilot_only", "variant": record["variant"],
            "record_path": str(path.relative_to(output_root)), "record_sha256": sha256_file(path),
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
        write_json(output_root / "report_assets" / "checkpoints" / "02_materialized.json", {
            "schema_version": 1, "protocol_id": PROTOCOL_ID, "checkpoint": "PILOT_150_MATERIALIZED",
            "created_at_utc": utc_now(), "record_count": len(records), "family_count": 30,
            "dataset_index_sha256": sha256_file(output_root / "dataset_index.json"),
            "materializer_source_sha256": sha256_file(Path(__file__).resolve()),
        })
    print(f"DATASET_V2_PILOT_MATERIALIZE_COMPLETE records={len(records)} families=30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
