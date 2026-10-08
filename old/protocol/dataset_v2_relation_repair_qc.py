#!/usr/bin/env python3
"""Observed-mask/metric-depth QC for the V2.1 relation-repair pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_v2_relation_geometry_v2_1 import (  # noqa: E402
    GEOMETRY_VERSION,
    evaluate_observed_relation,
    quaternion_rotation_matrix,
)
from dataset_v2_relation_repair_generator import PROTOCOL_ID  # noqa: E402
from wp2_common import canonical_json_sha256, read_json, sha256_file, utc_now, write_json  # noqa: E402


EXPECTED_CAPTURES = 30
DEFAULT_DATASET_ROOT = "datasets/roborefer_dataset_v2_1_relation_repair_pilot_round2_only_20260821"
DEFAULT_RESULT_ROOT = "results/dataset_v2_relation_repair_20260821/repair_pilot_round2"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["capture_id"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quaternion_angle(a: list[float], b: list[float]) -> float:
    rotation = quaternion_rotation_matrix(a).T @ quaternion_rotation_matrix(b)
    cosine = min(1.0, max(-1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
    return float(math.acos(cosine))


def save_real_qc_images(
    output: Path, capture_id: str, directory: Path, labels: np.ndarray, target_label: int
) -> None:
    target = output / capture_id
    target.mkdir(parents=True, exist_ok=True)
    rgb = cv2.imread(str(directory / "rgb_original.png"), cv2.IMREAD_COLOR)
    depth = np.load(directory / "depth_metric.npy", allow_pickle=False).astype(np.float32)
    cv2.imwrite(str(target / "rgb_original.png"), rgb)
    valid = np.isfinite(depth) & (depth >= 0.10) & (depth <= 2.0)
    depth_visual = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2, 98])
        scale = max(float(high - low), 1e-6)
        depth_visual[valid] = np.clip((depth[valid] - low) * 255.0 / scale, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(255 - depth_visual, cv2.COLORMAP_TURBO)
    depth_color[~valid] = 0
    cv2.imwrite(str(target / "depth_metric_visualization.png"), depth_color)
    palette = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in np.unique(labels):
        if int(label) == 0:
            continue
        digest = hashlib.sha256(f"semantic-label-{int(label)}".encode()).digest()
        palette[labels == label] = [digest[0], digest[1], digest[2]]
    cv2.imwrite(str(target / "semantic_instance_visualization.png"), palette)
    cv2.imwrite(str(target / "target_mask.png"), ((labels == int(target_label)) * 255).astype(np.uint8))


def run_qc(workspace: Path, dataset_root: Path, result_root: Path) -> dict[str, Any]:
    plan_path = workspace / "protocol/dataset_v2_relation_repair_capture_plan.json"
    manifest_path = workspace / "protocol/dataset_v2_relation_repair_manifest.json"
    lock_path = workspace / "protocol/dataset_v2_relation_repair_execution_lock.json"
    plan = read_json(plan_path)
    manifest = read_json(manifest_path)
    execution_lock = read_json(lock_path)
    by_family = {value["family_id"]: value for value in manifest["families"]}
    by_object_id = {value["id"]: value for value in plan["object_registry"].values()}
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    actual_ids = {
        value.name for value in (dataset_root / "raw/captures").iterdir() if value.is_dir()
    } if (dataset_root / "raw/captures").is_dir() else set()
    expected_ids = {value["capture_id"] for value in plan["captures"]}
    if actual_ids != expected_ids:
        failures.append(
            f"capture inventory mismatch missing={sorted(expected_ids - actual_ids)} extra={sorted(actual_ids - expected_ids)}"
        )

    planned_tf = plan["camera_predictor_transform"]
    selected_for_images = set(sorted(
        expected_ids,
        key=lambda value: hashlib.sha256(f"repair-qc-image|210820261|{value}".encode()).hexdigest(),
    )[:8])
    replay_hashes = []
    for capture in plan["captures"]:
        capture_id = capture["capture_id"]
        directory = dataset_root / "raw/captures" / capture_id
        if not directory.is_dir():
            continue
        local_failures = []
        try:
            meta_path = directory / "capture_meta.json"
            meta = read_json(meta_path)
            if meta.get("protocol_id") != PROTOCOL_ID or meta.get("capture_id") != capture_id:
                local_failures.append("IDENTITY")
            for relative, digest in meta.get("artifact_sha256", {}).items():
                if not (directory / relative).is_file() or sha256_file(directory / relative) != digest:
                    local_failures.append(f"HASH:{relative}")
            rgb = cv2.imread(str(directory / "rgb_original.png"), cv2.IMREAD_COLOR)
            depth = np.load(directory / "depth_metric.npy", allow_pickle=False)
            labels = cv2.imread(str(directory / "semantic_instance_labels.png"), cv2.IMREAD_UNCHANGED)
            if rgb is None or labels is None or rgb.shape[:2] != depth.shape or depth.shape != labels.shape:
                raise ValueError("registered RGB/depth/label shape mismatch")
            if list(depth.shape) != [480, 640]:
                local_failures.append("RESOLUTION")
            spread = float(meta["capture_timestamps"]["max_spread_sec"])
            if spread > float(plan["sensor_contract"]["max_timestamp_spread_sec"]):
                local_failures.append("TIMESTAMP_SYNC")
            required = set(capture["required_visible_label_ids"])
            if not required.issubset({int(value) for value in np.unique(labels)}):
                local_failures.append("REQUIRED_LABEL_VISIBILITY")

            realized = meta["realized_view_joint_pose"]
            joint_error = max(
                abs(float(a) - float(b)) for a, b in zip(realized, capture["view_joint_pose"])
            )
            if joint_error > 0.025:
                local_failures.append("JOINT_POSE")
            tf_artifact = read_json(directory / "tf_snapshot.json")
            tf = tf_artifact.get("camera_color_optical_frame", {})
            if "position" not in tf or "orientation_xyzw" not in tf:
                local_failures.append("CAMERA_TF")
                position_error = angle_error = math.inf
            else:
                position_error = float(np.linalg.norm(
                    np.asarray(tf["position"], dtype=np.float64)
                    - np.asarray(planned_tf["position"], dtype=np.float64)
                ))
                angle_error = quaternion_angle(tf["orientation_xyzw"], planned_tf["orientation_xyzw"])
                if position_error > 0.015 or angle_error > 0.050:
                    local_failures.append("CAMERA_TF_DEVIATION")

            target_label = int(by_object_id[capture["target_id"]]["label"])
            anchor_labels = [int(by_object_id[value]["label"]) for value in capture["anchor_ids"]]
            observed_a = evaluate_observed_relation(
                capture["relation"], labels, depth, target_label, anchor_labels
            )
            observed_b = evaluate_observed_relation(
                capture["relation"], labels, depth, target_label, anchor_labels
            )
            replay_a = canonical_json_sha256(observed_a)
            replay_b = canonical_json_sha256(observed_b)
            if replay_a != replay_b:
                local_failures.append("RELATION_REPLAY")
            replay_hashes.append(replay_a)
            if not observed_a["passed"]:
                local_failures.append("OBSERVED_RELATION_FALSE")
            family = by_family[capture["family_id"]]
            predictor = family["predictor_evidence"]
            predicted_target = float(predictor["target_predicted_median_depth_m"])
            observed_target = float(observed_a["target"]["median_depth_m"])
            if capture_id in selected_for_images:
                save_real_qc_images(
                    result_root / "qc_images_real", capture_id, directory, labels, target_label
                )
            rows.append({
                "capture_id": capture_id,
                "family_id": capture["family_id"],
                "relation": capture["relation"],
                "target_id": capture["target_id"],
                "anchor_ids": "|".join(capture["anchor_ids"]),
                "target_shape_group": family["target_shape_group"],
                "camera_bin_id": capture["camera_bin_id"],
                "joint_max_abs_error_rad": joint_error,
                "camera_tf_position_error_m": position_error,
                "camera_tf_angle_error_rad": angle_error,
                "timestamp_spread_sec": spread,
                "target_mask_pixels": observed_a["target"]["mask_pixels"],
                "target_interior_depth_pixels": observed_a["target"]["interior_valid_depth_pixels"],
                "target_centroid_x_px": observed_a["target"]["centroid_x_px"],
                "target_median_depth_m": observed_target,
                "target_depth_mad_m": observed_a["target"]["depth_mad_m"],
                "anchor_median_depth_m": "|".join(str(value["median_depth_m"]) for value in observed_a["anchors"]),
                "predictor_target_depth_m": predicted_target,
                "predictor_target_abs_error_m": abs(predicted_target - observed_target),
                "observed_signed_margin": observed_a["signed_margin"],
                "margin_unit": observed_a["margin_unit"],
                "observed_relation_pass": observed_a["passed"],
                "capture_pass": not local_failures,
                "failure_codes": "|".join(local_failures),
                "relation_replay_sha256": replay_a,
            })
        except Exception as exc:
            local_failures.append(f"EXCEPTION:{type(exc).__name__}:{exc}")
            rows.append({
                "capture_id": capture_id,
                "family_id": capture["family_id"],
                "relation": capture["relation"],
                "target_id": capture["target_id"],
                "anchor_ids": "|".join(capture["anchor_ids"]),
                "target_shape_group": by_family[capture["family_id"]]["target_shape_group"],
                "camera_bin_id": capture["camera_bin_id"],
                "joint_max_abs_error_rad": "",
                "camera_tf_position_error_m": "",
                "camera_tf_angle_error_rad": "",
                "timestamp_spread_sec": "",
                "target_mask_pixels": "",
                "target_interior_depth_pixels": "",
                "target_centroid_x_px": "",
                "target_median_depth_m": "",
                "target_depth_mad_m": "",
                "anchor_median_depth_m": "",
                "predictor_target_depth_m": "",
                "predictor_target_abs_error_m": "",
                "observed_signed_margin": "",
                "margin_unit": "",
                "observed_relation_pass": False,
                "capture_pass": False,
                "failure_codes": "|".join(local_failures),
                "relation_replay_sha256": "",
            })
        failures.extend(f"{capture_id}:{value}" for value in local_failures)

    relation_summary = []
    for relation in sorted({value["relation"] for value in rows}):
        subset = [value for value in rows if value["relation"] == relation]
        relation_summary.append({
            "relation": relation,
            "capture_count": len(subset),
            "pass_count": sum(str(value["capture_pass"]).lower() == "true" or value["capture_pass"] is True for value in subset),
            "fail_count": sum(not bool(value["capture_pass"]) for value in subset),
            "min_observed_signed_margin": min(
                [float(value["observed_signed_margin"]) for value in subset if value["observed_signed_margin"] != ""],
                default="",
            ),
        })
    passed = len(rows) == EXPECTED_CAPTURES and not failures and all(value["capture_pass"] for value in rows)
    result_root.mkdir(parents=True, exist_ok=True)
    measurements_path = result_root / "relation_measurements.csv"
    summary_path = result_root / "relation_summary.csv"
    write_csv(measurements_path, rows)
    write_csv(summary_path, relation_summary)
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "geometry_version": GEOMETRY_VERSION,
        "created_at_utc": utc_now(),
        "decision": "PASS_RELATION_REPAIR_PILOT" if passed else "REVISE_RELATION_PROTOCOL",
        "passed": passed,
        "capture_count": len(rows),
        "expected_capture_count": EXPECTED_CAPTURES,
        "relation_counts": dict(sorted(Counter(value["relation"] for value in rows).items())),
        "capture_pass_count": sum(bool(value["capture_pass"]) for value in rows),
        "failure_count": len(failures),
        "failures": failures,
        "deterministic_relation_replay_sha256": canonical_json_sha256(replay_hashes),
        "real_qc_image_capture_ids": sorted(selected_for_images & actual_ids),
        "measurements_csv_sha256": sha256_file(measurements_path),
        "relation_summary_csv_sha256": sha256_file(summary_path),
        "capture_plan_sha256": sha256_file(plan_path),
        "manifest_sha256": sha256_file(manifest_path),
        "execution_lock_sha256": sha256_file(lock_path),
        "training_performed": False,
        "calibration_or_test_opened": False,
    }
    write_json(result_root / "RELATION_REPAIR_QC_REPORT.json", report)
    markdown = [
        "# Dataset V2.1 relation-repair pilot report",
        "",
        f"- Decision: `{report['decision']}`.",
        f"- Real clean captures: `{len(rows)}/{EXPECTED_CAPTURES}`.",
        f"- Captures passing structural and observed-relation QC: `{report['capture_pass_count']}/{EXPECTED_CAPTURES}`.",
        f"- Geometry implementation: `{GEOMETRY_VERSION}`.",
        "- Relation labels use semantic-mask centroid or robust median metric depth; simulator centers are not final labels.",
        "- All families are engineering-only and permanently excluded from official train/dev/calibration/test.",
        "- Training performed: `False`.",
        "",
        "## Relation summary",
        "",
        "| Relation | Captures | Pass | Fail | Minimum signed margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for value in relation_summary:
        markdown.append(
            f"| {value['relation']} | {value['capture_count']} | {value['pass_count']} | "
            f"{value['fail_count']} | {value['min_observed_signed_margin']} |"
        )
    markdown.extend([
        "",
        "The CSV tables contain the per-capture target/anchor evidence, camera-pose error, predictor residual and observed signed margin. The QC image directory contains only captured RGB and direct depth/semantic/mask renderings selected by a locked seed; no planned infographic was generated.",
        "",
    ])
    (workspace / "protocol/DATASET_V2_RELATION_REPAIR_REPORT.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = workspace / dataset_root
    result_root = Path(args.result_root)
    if not result_root.is_absolute():
        result_root = workspace / result_root
    report = run_qc(workspace, dataset_root, result_root)
    print(
        "DATASET_V2_RELATION_REPAIR_QC "
        f"decision={report['decision']} captures={report['capture_count']} "
        f"pass={report['capture_pass_count']} failures={report['failure_count']}"
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
