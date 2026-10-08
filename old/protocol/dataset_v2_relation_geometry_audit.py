#!/usr/bin/env python3
"""Audit quarantined V2 captures and derive engineering-only V2.1 calibration."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2_relation_geometry_v2_1 import (  # noqa: E402
    DEPTH_MARGIN_M,
    GEOMETRY_VERSION,
    asset_shape_group,
    evaluate_observed_relation,
    instance_evidence,
    project_base_point_to_camera,
)
from wp2_common import read_json, sha256_file, utc_now, write_json  # noqa: E402


PROTOCOL_ID = "roborefer_dataset_v2_1_relation_repair"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def robust_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "mad": None, "p10": None, "p90": None}
    data = np.asarray(values, dtype=np.float64)
    median = float(np.median(data))
    return {
        "count": int(data.size),
        "median": median,
        "mad": float(np.median(np.abs(data - median))),
        "p10": float(np.percentile(data, 10)),
        "p90": float(np.percentile(data, 90)),
    }


def expected_relation_targets(spec: dict[str, Any]) -> tuple[str, list[str]]:
    state = spec["answerability_state"]
    if state in {"FOUND", "AMBIGUOUS"}:
        return "SATISFIED", list(spec["valid_target_ids"])
    if state == "ABSENT" and spec["state_submode"] == "unsatisfied_relation":
        return "UNSATISFIED", list(spec["candidate_target_ids"])
    return "NOT_ASSERTED", []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="datasets/roborefer_dataset_v2_development_400_20260821",
    )
    parser.add_argument("--manifest", default="protocol/dataset_v2_development_manifest.json")
    parser.add_argument("--capture-plan", default="protocol/dataset_v2_development_capture_plan.json")
    parser.add_argument(
        "--output-root", default="results/dataset_v2_relation_repair_20260821/audit"
    )
    parser.add_argument(
        "--report", default="protocol/dataset_v2_relation_geometry_audit_230.json"
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]
    dataset_root = (workspace / args.dataset_root).resolve()
    manifest_path = (workspace / args.manifest).resolve()
    plan_path = (workspace / args.capture_plan).resolve()
    output_root = (workspace / args.output_root).resolve()
    report_path = (workspace / args.report).resolve()
    manifest = read_json(manifest_path)
    plan = read_json(plan_path)
    family_by_id = {item["family_id"]: item for item in manifest["families"]}
    capture_by_id = {item["capture_id"]: item for item in plan["captures"]}
    registry = plan["object_registry"]
    registry_by_id = {value["id"]: (name, value) for name, value in registry.items()}
    model_by_label = {int(value["label"]): (name, value) for name, value in registry.items()}

    capture_dirs = sorted(
        path for path in (dataset_root / "raw" / "captures").iterdir() if path.is_dir()
    )
    capture_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    offsets_by_model: defaultdict[str, list[float]] = defaultdict(list)
    calibration_observations: list[dict[str, Any]] = []
    errors: list[str] = []

    for directory in capture_dirs:
        capture_id = directory.name
        try:
            capture = capture_by_id[capture_id]
            family = family_by_id[capture["family_id"]]
            meta = read_json(directory / "capture_meta.json")
            tf = read_json(directory / "tf_snapshot.json")["camera_color_optical_frame"]
            labels = cv2.imread(
                str(directory / "semantic_instance_labels.png"), cv2.IMREAD_UNCHANGED
            )
            depth = np.load(directory / "depth_metric.npy", allow_pickle=False)
            if labels is None or labels.shape != depth.shape:
                raise RuntimeError("registered label/depth shape differs")
            timestamps = meta["capture_timestamps"]
            capture_rows.append({
                "capture_id": capture_id,
                "family_id": capture["family_id"],
                "condition": capture["condition"],
                "split": capture["split"],
                "answerability_state": family["primary_answerability_stratum"],
                "family_category": family["family_category"],
                "timestamp_spread_sec": timestamps["max_spread_sec"],
                "valid_depth_fraction": meta["sensor_qc"]["valid_depth_fraction"],
                "camera_position_xyz": json.dumps(tf["position"], separators=(",", ":")),
                "camera_orientation_xyzw": json.dumps(tf["orientation_xyzw"], separators=(",", ":")),
                "capture_meta_sha256": sha256_file(directory / "capture_meta.json"),
            })

            if capture["condition"] != "clean":
                continue

            visible = {int(value) for value in np.unique(labels)}
            for label in sorted(visible & set(model_by_label)):
                model_name, row = model_by_label[label]
                try:
                    evidence = instance_evidence(labels, depth, label)
                except Exception:
                    continue
                pose = capture["layout"][model_name]
                center = project_base_point_to_camera(
                    [float(pose[0]), float(pose[1]), float(row["z"])], tf
                )
                offset = float(center[2] - evidence.median_depth_m)
                offsets_by_model[model_name].append(offset)
                calibration_observations.append({
                    "capture_id": capture_id,
                    "family_id": capture["family_id"],
                    "model_name": model_name,
                    "object_id": row["id"],
                    "semantic_class": row["semantic_class"],
                    "shape_group": asset_shape_group(row),
                    "footprint_radius_m": row["footprint_radius_m"],
                    "height_m": 2.0 * float(row["z"]),
                    "center_optical_depth_m": float(center[2]),
                    "observed_median_depth_m": evidence.median_depth_m,
                    "surface_offset_m": offset,
                    "observed_depth_mad_m": evidence.depth_mad_m,
                    "mask_pixels": evidence.mask_pixels,
                })

            spec = family["variant_specs"][0]
            relation = spec["relations"][0]
            expected, target_ids = expected_relation_targets(spec)
            if relation == "direct" or expected == "NOT_ASSERTED":
                continue
            anchor_ids = list(spec["anchor_ids"])
            for target_id in target_ids:
                target_model, target_row = registry_by_id[target_id]
                result = evaluate_observed_relation(
                    relation,
                    labels,
                    depth,
                    int(target_row["label"]),
                    [int(registry_by_id[value][1]["label"]) for value in anchor_ids],
                )
                expected_pass = expected == "SATISFIED"
                correct = bool(result["passed"]) == expected_pass
                relation_rows.append({
                    "capture_id": capture_id,
                    "family_id": capture["family_id"],
                    "split": capture["split"],
                    "answerability_state": spec["answerability_state"],
                    "state_submode": spec["state_submode"],
                    "family_category": family["family_category"],
                    "relation": relation,
                    "expected": expected,
                    "observed_predicate": "SATISFIED" if result["passed"] else "UNSATISFIED",
                    "correct": correct,
                    "signed_margin": result["signed_margin"],
                    "margin_unit": result["margin_unit"],
                    "target_id": target_id,
                    "target_model": target_model,
                    "target_semantic_class": target_row["semantic_class"],
                    "target_shape_group": asset_shape_group(target_row),
                    "target_footprint_radius_m": target_row["footprint_radius_m"],
                    "target_height_m": 2.0 * float(target_row["z"]),
                    "target_centroid_x_px": result["target"]["centroid_x_px"],
                    "target_median_depth_m": result["target"]["median_depth_m"],
                    "target_depth_mad_m": result["target"]["depth_mad_m"],
                    "anchor_ids": "|".join(anchor_ids),
                    "anchor_semantic_classes": "|".join(
                        registry_by_id[value][1]["semantic_class"] for value in anchor_ids
                    ),
                    "anchor_shape_groups": "|".join(
                        asset_shape_group(registry_by_id[value][1]) for value in anchor_ids
                    ),
                    "anchor_footprint_radius_m": "|".join(
                        str(registry_by_id[value][1]["footprint_radius_m"]) for value in anchor_ids
                    ),
                    "anchor_height_m": "|".join(
                        str(2.0 * float(registry_by_id[value][1]["z"])) for value in anchor_ids
                    ),
                    "anchor_centroid_x_px": "|".join(
                        str(value["centroid_x_px"]) for value in result["anchors"]
                    ),
                    "anchor_median_depth_m": "|".join(
                        str(value["median_depth_m"]) for value in result["anchors"]
                    ),
                    "camera_position_xyz": json.dumps(tf["position"], separators=(",", ":")),
                    "camera_orientation_xyzw": json.dumps(tf["orientation_xyzw"], separators=(",", ":")),
                })
        except Exception as exc:
            errors.append(f"{capture_id}: {type(exc).__name__}: {exc}")

    calibration_rows = []
    calibration_payload = {}
    for model_name, row in sorted(registry.items()):
        values = offsets_by_model.get(model_name, [])
        summary = robust_summary(values)
        calibration_row = {
            "model_name": model_name,
            "object_id": row["id"],
            "semantic_class": row["semantic_class"],
            "asset_partition": row["asset_partition"],
            "shape_group": asset_shape_group(row),
            "footprint_radius_m": row["footprint_radius_m"],
            "height_m": 2.0 * float(row["z"]),
            "observation_count": summary["count"],
            "median_surface_offset_m": summary["median"],
            "surface_offset_mad_m": summary["mad"],
            "surface_offset_p10_m": summary["p10"],
            "surface_offset_p90_m": summary["p90"],
        }
        calibration_rows.append(calibration_row)
        if summary["count"]:
            calibration_payload[model_name] = calibration_row

    summary_by_relation = []
    for relation in sorted({row["relation"] for row in relation_rows}):
        subset = [row for row in relation_rows if row["relation"] == relation]
        summary_by_relation.append({
            "relation": relation,
            "assertion_count": len(subset),
            "correct_count": sum(bool(row["correct"]) for row in subset),
            "failure_count": sum(not bool(row["correct"]) for row in subset),
            "correct_fraction": sum(bool(row["correct"]) for row in subset) / len(subset),
            "minimum_signed_margin": min(float(row["signed_margin"]) for row in subset),
            "median_signed_margin": statistics.median(float(row["signed_margin"]) for row in subset),
            "margin_unit": subset[0]["margin_unit"],
        })

    pair_groups: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in relation_rows:
        pair_groups[(row["relation"], row["target_id"], row["anchor_ids"])].append(row)
    pair_summary = []
    for (relation, target_id, anchor_ids), subset in sorted(pair_groups.items()):
        pair_summary.append({
            "relation": relation,
            "target_id": target_id,
            "anchor_ids": anchor_ids,
            "assertion_count": len(subset),
            "correct_count": sum(bool(row["correct"]) for row in subset),
            "failure_count": sum(not bool(row["correct"]) for row in subset),
            "minimum_signed_margin": min(float(row["signed_margin"]) for row in subset),
            "median_signed_margin": statistics.median(float(row["signed_margin"]) for row in subset),
            "margin_unit": subset[0]["margin_unit"],
        })

    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "capture_inventory": output_root / "capture_inventory_230.csv",
        "relation_assertions": output_root / "relation_assertions.csv",
        "relation_summary": output_root / "relation_summary.csv",
        "asset_pair_summary": output_root / "asset_pair_summary.csv",
        "asset_depth_calibration": output_root / "asset_depth_calibration.csv",
        "calibration_observations": output_root / "asset_depth_calibration_observations.csv",
        "near_margin_cases": output_root / "near_margin_cases.csv",
        "calibration_json": output_root / "asset_depth_calibration.json",
    }
    write_csv(paths["capture_inventory"], capture_rows)
    write_csv(paths["relation_assertions"], relation_rows)
    write_csv(paths["relation_summary"], summary_by_relation)
    write_csv(paths["asset_pair_summary"], pair_summary)
    write_csv(paths["asset_depth_calibration"], calibration_rows)
    write_csv(paths["calibration_observations"], calibration_observations)
    near_margin = sorted(
        relation_rows,
        key=lambda row: (float(row["signed_margin"]), row["family_id"], row["target_id"]),
    )[:30]
    write_csv(paths["near_margin_cases"], near_margin)
    write_json(paths["calibration_json"], {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "geometry_version": GEOMETRY_VERSION,
        "source_role": "QUARANTINED_V2_ENGINEERING_DIAGNOSTIC_ONLY",
        "source_capture_count": len(capture_rows),
        "clean_capture_count": sum(row["condition"] == "clean" for row in capture_rows),
        "asset_calibration": calibration_payload,
    })

    failure_rows = [row for row in relation_rows if not bool(row["correct"])]
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "geometry_version": GEOMETRY_VERSION,
        "source_dataset_root": str(dataset_root.relative_to(workspace)),
        "source_role": "ENGINEERING_DIAGNOSTIC_ONLY_NOT_OFFICIAL_DATA",
        "capture_count": len(capture_rows),
        "clean_capture_count": sum(row["condition"] == "clean" for row in capture_rows),
        "occlusion_capture_count": sum(row["condition"] == "occlusion" for row in capture_rows),
        "family_count": len({row["family_id"] for row in capture_rows}),
        "relation_assertion_count": len(relation_rows),
        "relation_correct_count": sum(bool(row["correct"]) for row in relation_rows),
        "relation_failure_count": len(failure_rows),
        "relation_failures": failure_rows,
        "relation_summary": summary_by_relation,
        "calibrated_asset_count": len(calibration_payload),
        "seen_asset_missing_calibration": sorted(
            name for name, row in registry.items()
            if row["asset_partition"] == "seen" and name not in calibration_payload
        ),
        "capture_errors": errors,
        "depth_margin_m": DEPTH_MARGIN_M,
        "training_performed": False,
        "official_data_created": False,
        "decision": "GO_REPAIR_PILOT_STATIC_PREFLIGHT" if not errors else "FIX_AUDIT_FIRST",
        "artifacts": {
            name: {
                "path": str(path.relative_to(workspace)),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "manifest_sha256": sha256_file(manifest_path),
        "capture_plan_sha256": sha256_file(plan_path),
    }
    write_json(report_path, report)
    print(
        "DATASET_V2_RELATION_AUDIT "
        f"captures={len(capture_rows)} assertions={len(relation_rows)} "
        f"failures={len(failure_rows)} errors={len(errors)} decision={report['decision']}"
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
