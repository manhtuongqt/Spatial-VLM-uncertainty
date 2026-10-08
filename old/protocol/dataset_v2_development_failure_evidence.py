#!/usr/bin/env python3
"""Materialize real, non-decorative sensor evidence for a failed batch gate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from dataset_v2_development_batch_qc import atomic_json, depth_visual, semantic_visual
from wp2_common import read_json, sha256_file


def binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, 255, 0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--manifest", default="protocol/dataset_v2_development_manifest.json")
    parser.add_argument("--capture-plan", default="protocol/dataset_v2_development_capture_plan.json")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    manifest_path = (workspace / args.manifest).resolve()
    plan_path = (workspace / args.capture_plan).resolve()
    gate_path = dataset_root / "report_assets" / "checkpoints" / "batch_qc" / f"{args.batch_id}.json"
    gate = read_json(gate_path)
    if gate.get("passed") or not gate.get("relation_failures"):
        raise RuntimeError("failure evidence requires a failed gate with relation_failures")

    manifest = read_json(manifest_path)
    plan = read_json(plan_path)
    family_by_id = {item["family_id"]: item for item in manifest["families"]}
    label_by_id = {value["id"]: int(value["label"]) for value in plan["object_registry"].values()}
    output_root = dataset_root / "report_assets" / "real_capture_qc" / f"{args.batch_id}_failures"
    entries = []

    for failure in gate["relation_failures"]:
        family_id = failure["family_id"]
        capture_id = failure["capture_id"]
        family = family_by_id[family_id]
        spec = family["variant_specs"][0]
        source = dataset_root / "raw" / "captures" / capture_id
        destination = output_root / family_id
        destination.mkdir(parents=True, exist_ok=True)

        rgb_path = source / "rgb_original.png"
        labels = cv2.imread(str(source / "semantic_instance_labels.png"), cv2.IMREAD_UNCHANGED)
        depth = np.load(source / "depth_metric.npy", allow_pickle=False)
        if labels is None:
            raise RuntimeError(f"cannot read semantic labels: {capture_id}")

        shutil.copy2(rgb_path, destination / "rgb.png")
        cv2.imwrite(str(destination / "depth_metric_visualization.png"), depth_visual(depth))
        cv2.imwrite(str(destination / "semantic_instance_labels.png"), semantic_visual(labels))
        target_id = failure["target_id"]
        cv2.imwrite(
            str(destination / "target_mask.png"),
            binary_mask(labels == label_by_id[target_id]),
        )
        anchor_files = []
        for index, anchor_id in enumerate(spec["anchor_ids"], start=1):
            name = f"anchor_mask_{index:02d}.png"
            cv2.imwrite(
                str(destination / name),
                binary_mask(labels == label_by_id[anchor_id]),
            )
            anchor_files.append({"anchor_id": anchor_id, "path": name})

        files = {
            path.name: sha256_file(path)
            for path in sorted(destination.iterdir())
            if path.is_file()
        }
        entries.append({
            "family_id": family_id,
            "capture_id": capture_id,
            "relation": failure["relation"],
            "target_id": target_id,
            "anchor_masks": anchor_files,
            "target_median_depth_m": failure["target_median_depth_m"],
            "anchor_median_depth_m": failure["anchor_median_depth_m"],
            "source_capture_meta_sha256": sha256_file(source / "capture_meta.json"),
            "artifact_sha256": files,
        })

    index_path = output_root / "FAILURE_EVIDENCE_INDEX.json"
    atomic_json(index_path, {
        "schema_version": 1,
        "protocol_id": gate["protocol_id"],
        "batch_id": args.batch_id,
        "gate_decision": gate["decision"],
        "selection_policy": "all runtime metric-depth relation failures from the locked batch QC",
        "contains_planned_or_decorative_images": False,
        "source_gate_sha256": sha256_file(gate_path),
        "manifest_sha256": sha256_file(manifest_path),
        "capture_plan_sha256": sha256_file(plan_path),
        "failure_count": len(entries),
        "failures": entries,
    })
    print(f"DEVELOPMENT_FAILURE_EVIDENCE batch={args.batch_id} failures={len(entries)} path={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
