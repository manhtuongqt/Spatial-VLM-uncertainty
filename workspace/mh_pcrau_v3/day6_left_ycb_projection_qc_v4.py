#!/usr/bin/env python3
"""Evaluate the pairwise projection canary and freeze selected offsets."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_projection_canary_v4"
CAPTURE = OUT / "capture_attempt_02"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output = OUT / "PROJECTION_CALIBRATION_QC.json"
    selected_output = OUT / "SELECTED_PAIR_OFFSETS.json"
    if output.exists() or selected_output.exists():
        raise FileExistsError("append-only projection QC already exists")
    annotations = yaml.safe_load((OUT / "annotations.yaml").read_text(encoding="utf-8"))["scenes"]
    manifest_rows = [
        json.loads(line) for line in (CAPTURE / "input_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    observations = defaultdict(list)
    for row in manifest_rows:
        scene_id = row["scene_id"]
        annotation = annotations[scene_id]
        calibration = annotation["projection_calibration"]
        labels_path = CAPTURE / scene_id / "evaluator/semantic_labels.png"
        labels = cv2.imread(str(labels_path), cv2.IMREAD_UNCHANGED)
        if labels is None:
            raise RuntimeError(f"missing semantic labels: {labels_path}")
        target_pixels = int((labels == int(annotation["target_label"])).sum())
        occluder_pixels = int((labels == int(annotation["occluder_labels"][0])).sum())
        key = (calibration["box"], calibration["fruit"])
        observations[key].append({
            "scene_id": scene_id,
            "signed_lateral_offset_m": float(calibration["signed_lateral_offset_m"]),
            "target_visible_pixels": target_pixels,
            "occluder_visible_pixels": occluder_pixels,
            "pass_1_to_119": 1 <= target_pixels < 120,
            "semantic_labels_sha256": sha256(labels_path),
        })

    pair_results = []
    selected = {}
    for (box, fruit), rows in sorted(observations.items()):
        valid = [row for row in rows if row["pass_1_to_119"]]
        chosen = None
        if valid:
            chosen = min(
                valid,
                key=lambda row: (
                    abs(row["target_visible_pixels"] - 60),
                    abs(row["signed_lateral_offset_m"]),
                    row["signed_lateral_offset_m"],
                ),
            )
            selected.setdefault(box, {})[fruit] = {
                "signed_lateral_offset_m": chosen["signed_lateral_offset_m"],
                "canary_target_visible_pixels": chosen["target_visible_pixels"],
                "canary_scene_id": chosen["scene_id"],
            }
        pair_results.append({
            "box": box,
            "fruit": fruit,
            "observed_offsets": len(rows),
            "target_pixels_min": min(row["target_visible_pixels"] for row in rows),
            "target_pixels_max": max(row["target_visible_pixels"] for row in rows),
            "valid_offset_count": len(valid),
            "selected": chosen,
        })

    all_pairs_pass = len(pair_results) == 18 and all(row["valid_offset_count"] > 0 for row in pair_results)
    result = {
        "schema_version": 1,
        "status": "PROJECTION_CANARY_PASS" if all_pairs_pass else "PROJECTION_CANARY_FAIL_FINE_SCAN_REQUIRED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_inference_performed": False,
        "captured_scenes": len(manifest_rows),
        "expected_scenes": 18 * 41,
        "pair_count": len(pair_results),
        "passing_pair_count": sum(row["valid_offset_count"] > 0 for row in pair_results),
        "pass_rule": "all 18 box/fruit pairs have at least one observation with 1<=target_visible_pixels<120",
        "pair_results": pair_results,
        "source_hashes": {
            "capture_manifest": sha256(CAPTURE / "capture_manifest.json"),
            "input_manifest": sha256(CAPTURE / "input_manifest.jsonl"),
            "annotations": sha256(OUT / "annotations.yaml"),
        },
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    selected_output.write_text(json.dumps({
        "schema_version": 1,
        "status": "LOCKED_FROM_PASSING_PROJECTION_CANARY" if all_pairs_pass else "INCOMPLETE_NOT_AUTHORIZED",
        "selection_rule": "nearest visible-pixel count to 60 within inclusive 1..119",
        "pairs": selected,
        "projection_qc_sha256": sha256(output),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "captured_scenes": len(manifest_rows),
        "passing_pairs": result["passing_pair_count"],
        "pair_count": len(pair_results),
    }, indent=2))


if __name__ == "__main__":
    main()
