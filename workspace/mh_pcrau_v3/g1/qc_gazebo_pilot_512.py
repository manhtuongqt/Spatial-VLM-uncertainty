"""Audit actual Gazebo pilot captures, preserving failures and label provenance."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

from .audit_development import GAZEBO, ROOT, read_jsonl, sha256
from .generate_gazebo_pilot_512 import OUT

sys.path.insert(0, str(ROOT / "protocol"))
from gazebo_train_uq_v2_pilot_r4_pipeline import geometry, labels, project_base_point, forbidden_input_paths


def check_state(item: dict, scene: dict, label_map: np.ndarray, capture: Path,
                input_row: dict, gate: dict) -> tuple[bool, dict]:
    minimum = int(gate["min_visible_evidence_px"])
    candidates = [geometry(label_map, value) for value in item.get("candidate_labels", [])]
    target = geometry(label_map, item["target_label"]) if item.get("target_label") is not None else None
    state = item["state"]
    detail = {
        "candidate_visible_pixels": [row["visible_pixels"] for row in candidates],
        "target_visible_pixels": target["visible_pixels"] if target else None,
    }
    if state in ("FOUND", "AMBIGUOUS"):
        if not candidates or any(row["visible_pixels"] < minimum for row in candidates):
            return False, {**detail, "reason": "CANDIDATE_VISIBILITY_BELOW_MINIMUM"}
        order = sorted(candidates, key=lambda row: row["centroid_normalized_xy"][0],
                       reverse=item["rank_from"] == "right")
        gaps = [abs(a["centroid_normalized_xy"][0] - b["centroid_normalized_xy"][0])
                for a, b in zip(order, order[1:])]
        detail.update({"ordered_labels": [row["semantic_label"] for row in order], "gaps": gaps})
        if state == "FOUND":
            good = (item["rank"] <= len(order) and
                    order[item["rank"] - 1]["semantic_label"] == item["target_label"] and
                    all(gap > gate["tie_margin_normalized"] for gap in gaps))
            return good, detail
        valid = [geometry(label_map, value) for value in item["valid_target_labels"]]
        xs = [row["centroid_normalized_xy"][0] for row in valid
              if row["centroid_normalized_xy"] is not None]
        tied_positions = [index for index, row in enumerate(order)
                          if row["semantic_label"] in item["valid_target_labels"]]
        gap = max(xs) - min(xs) if len(xs) == 2 else None
        detail.update({"tie_gap": gap, "tied_positions": tied_positions})
        good = gap is not None and gap <= gate["tie_margin_normalized"] and item["rank"] - 1 in tied_positions
        return good, detail
    if state == "ABSENT":
        context = [geometry(label_map, value) for value in item.get("context_labels", [])]
        detail["context_visible_pixels"] = [row["visible_pixels"] for row in context]
        good = bool(target and target["visible_pixels"] == 0 and
                    item["target_id"] not in scene["poses"] and context and
                    all(row["visible_pixels"] >= minimum for row in context))
        return good, detail
    if state == "INSUFFICIENT_EVIDENCE":
        camera = json.loads((capture / input_row["input_files"]["camera_info"]).read_text())
        transforms = json.loads((capture / input_row["input_files"]["tf_snapshot"]).read_text())
        capture_oracle = json.loads((capture / input_row["scene_id"] / "evaluator/capture_oracle.json").read_text())
        transform = transforms.get("camera_color_optical_frame", {})
        projected = None
        if transform.get("position"):
            point = capture_oracle["requested_scene_layout_base_link"][item["target_id"]][:3]
            projected = project_base_point(point, transform, camera)
        pixel = projected.get("pixel_xy") if projected else None
        outer = gate["insufficient_evidence_rule"]["projected_center_outer_margin_fraction"]
        near = bool(pixel and -640 * outer <= pixel[0] < 640 * (1 + outer) and
                    -480 * outer <= pixel[1] < 480 * (1 + outer) and
                    0.1 <= projected["camera_xyz"][2] <= 2.0)
        others = [row for row in candidates if row["semantic_label"] != item["target_label"]]
        good = bool(target and 1 <= target["visible_pixels"] < minimum and near and
                    all(row["visible_pixels"] >= minimum for row in others))
        detail.update({"near_sensor": near, "projected": projected})
        return good, detail
    raise ValueError(state)


def main() -> None:
    summary_path = OUT / "PILOT_512_CAPTURE_QC.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    plan = json.loads((OUT / "CAPTURE_PLAN.json").read_text())
    old_rows = [row for row in read_jsonl(GAZEBO / "inference_manifest.jsonl") if row["split"] == "train_uq"]
    assert len(old_rows) == 256
    rgb_hashes = defaultdict(list)
    for row in old_rows:
        rgb_hashes[sha256(GAZEBO / row["image"])].append(row["sample_id"])
    reports = []
    all_rows = []
    for batch in plan["batches"]:
        index = batch["batch"]
        folder = OUT / f"batch_{index}"
        capture = folder / "capture_attempt_01"
        report_path = folder / "CAPTURE_QC.json"
        if report_path.exists():
            raise FileExistsError(report_path)
        scenes_path, annotations_path, gate_path = (folder / name for name in
            ("scenes.yaml", "annotations.yaml", "gate.yaml"))
        manifest_path = capture / "capture_manifest.json"
        input_path = capture / "input_manifest.jsonl"
        if not manifest_path.is_file() or not input_path.is_file():
            reports.append({"batch": index, "status": "NOT_CAPTURED", "captured": 0})
            continue
        manifest = json.loads(manifest_path.read_text())
        scene_config = yaml.safe_load(scenes_path.read_text())
        annotation = yaml.safe_load(annotations_path.read_text())
        gate = yaml.safe_load(gate_path.read_text())
        inputs = list(read_jsonl(input_path))
        integrity = (
            manifest.get("status") == "COMPLETE" and manifest.get("scene_count") == 64 and len(inputs) == 64
            and manifest.get("input_manifest_sha256") == sha256(input_path)
            and manifest.get("scene_config_sha256") == sha256(scenes_path)
            and manifest.get("annotation_file_sha256") == sha256(annotations_path)
            and manifest.get("gate_config_file_sha256") == sha256(gate_path)
            and manifest.get("pretrial_source_lock_sha256") == sha256(folder / "CAPTURE_SOURCE_LOCK.json")
            and manifest.get("view_joint_pose") == batch["view_joint_pose"]
            and manifest.get("semantic_labels_are_evaluator_only") is True
            and manifest.get("robot_manipulation_performed") is False
            and manifest.get("target_handoff_published") is False
            and manifest.get("model_inventory_sha256_preregistered") == "NO_MODEL_INFERENCE_V3_PILOT_CAPTURE"
            and manifest.get("input_only_sensor_qc", {}).get("passed") is True
        )
        scenes = {row["scene_id"]: row for row in scene_config["scenes"]}
        rows = []
        for input_row in inputs:
            sid = input_row["scene_id"]
            item = annotation["scenes"][sid]
            reasons = []
            if forbidden_input_paths(input_row):
                reasons.append("ORACLE_IN_MODEL_INPUT")
            for kind, relative in input_row["input_files"].items():
                path = capture / relative
                if not path.is_file() or sha256(path) != input_row["input_sha256"].get(kind):
                    reasons.append(f"MISSING_OR_HASH_MISMATCH:{kind}")
            rgb = capture / input_row["input_files"]["rgb"]
            rgb_hash = sha256(rgb)
            rgb_hashes[rgb_hash].append(sid)
            label_map = labels(capture / sid / "evaluator/semantic_labels.png")
            depth = np.load(capture / input_row["input_files"]["depth_m"], allow_pickle=False)
            if depth.shape != label_map.shape or depth.dtype not in (np.float32, np.float64):
                reasons.append("METRIC_DEPTH_SHAPE_OR_DTYPE")
            valid, detail = check_state(item, scenes[sid], label_map, capture, input_row, gate)
            if not valid:
                reasons.append("REQUESTED_STATE_NOT_VERIFIED")
            row = {"scene_id": sid, "family_id": item["family_id"], "batch": index,
                   "object_group": item["object_group"], "view_id": item["view_id"],
                   "requested_state": item["state"], "relation": item["relation_variant"],
                   "state_verified": valid, "reasons": reasons, "geometry": detail,
                   "rgb_sha256": rgb_hash,
                   "rgb_path": str(rgb.relative_to(ROOT))}
            rows.append(row)
            all_rows.append(row)
        status = "PASS" if integrity and all(not row["reasons"] for row in rows) else "REJECT"
        report = {
            "batch": index, "status": status, "integrity": integrity,
            "captured": len(rows), "state_verified": sum(row["state_verified"] for row in rows),
            "state_counts": dict(sorted(Counter(row["requested_state"] for row in rows).items())),
            "verified_state_counts": dict(sorted(Counter(row["requested_state"] for row in rows
                                                         if row["state_verified"]).items())),
            "failed_scene_count": sum(bool(row["reasons"]) for row in rows),
            "capture_manifest_sha256": sha256(manifest_path), "rows": rows,
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        reports.append({key: value for key, value in report.items() if key != "rows"})
    duplicates = [group for group in rgb_hashes.values() if len(group) > 1]
    new_count = len(all_rows)
    verified_count = sum(row["state_verified"] for row in all_rows)
    summary = {
        "status": "PILOT_512_CAPTURED_AND_VERIFIED" if new_count == 256 and verified_count == 256 and not duplicates
                  and all(report["status"] == "PASS" for report in reports) else "PILOT_512_QC_NOT_PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "old_gazebo_train_images": len(old_rows), "new_captured_images": new_count,
        "total_pilot_images_captured": len(old_rows) + new_count,
        "new_state_verified": verified_count,
        "new_requested_state_counts": dict(sorted(Counter(row["requested_state"] for row in all_rows).items())),
        "new_verified_state_counts": dict(sorted(Counter(row["requested_state"] for row in all_rows
                                                       if row["state_verified"]).items())),
        "exact_rgb_duplicate_groups_old_and_new": duplicates,
        "batches": reports,
        "gate_note": "Capture count and pilot QC alone do not establish G1_PASS or authorize Day 4 training.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": summary["status"], "new_captured": new_count,
                      "new_state_verified": verified_count, "duplicate_groups": len(duplicates)}))


if __name__ == "__main__":
    main()
