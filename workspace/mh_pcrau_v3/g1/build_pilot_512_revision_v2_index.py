"""Select one QC-valid revised view per historical Gazebo train family."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json

import yaml

from .audit_development import GAZEBO, ROOT, read_jsonl, sha256
from .capture_revision_v2 import OUT
from .repair_ie_revision_v2 import OUT as REPAIR


def main() -> None:
    index_path = OUT / "PILOT_512_SELECTED_INDEX.jsonl"
    summary_path = OUT / "PILOT_512_SELECTED_SUMMARY.json"
    if index_path.exists() or summary_path.exists():
        raise FileExistsError("Revision-v2 selected index is append-only")
    from protocol.gazebo_train_uq_v2_pilot_r4_pipeline import geometry, labels

    qc = json.loads((OUT / "REVISION_V2_QC.json").read_text())
    repair = json.loads((REPAIR / "REPAIR_QC.json").read_text())
    if repair["status"] != "PASS":
        raise ValueError("IE repairs not all passed")
    by_sid = {row["scene_id"]: row for batch in qc["batches"] for row in batch["rows"]}
    selected_repairs = {row["original_failed_scene"]: row for row in repair["selected_one_per_family"]}
    old = [row for row in read_jsonl(GAZEBO / "inference_manifest.jsonl")
           if row["split"] == "train_uq"]
    supervised = {row["sample_id"]: row for row in read_jsonl(GAZEBO / "train_supervision.jsonl")}
    old_gt = {row["sample_id"]: row for row in read_jsonl(GAZEBO / "evaluator_ground_truth.jsonl")}
    family_index = json.loads((OUT / "PARENT_FAMILY_INDEX.json").read_text())
    if len(old) != 256 or len(family_index) != 256:
        raise ValueError("Family inventory drift")
    rows = []
    for item in family_index:
        historical = next(row for row in old if row["sample_id"] == item["old_sample_id"])
        sup = supervised[historical["sample_id"]]["supervision"]
        gt = old_gt[historical["sample_id"]]
        if sup["answerability_state"] != item["state"] or gt["answerability_verified"] is not True:
            raise ValueError(f"Historical state drift: {item['old_sample_id']}")
        old_input = {"rgb_path": str((GAZEBO / historical["image"]).relative_to(ROOT)),
                     "depth_path": str((GAZEBO / historical["metric_depth"]).relative_to(ROOT)),
                     "instruction": historical["instruction"]}
        rows.append({"sample_id": historical["sample_id"], "family_id": item["parent_family_id"],
                     "view_id": "historical_original", "origin": "EXISTING_TRAIN_UQ",
                     "model_input": old_input,
                     "supervision": {"relation": item["relation"], "answerability": item["state"],
                                     "target_uv": sup.get("target_xy") if item["state"] == "FOUND" else None},
                     "head_mask": {"relation": 1, "answerability": 1,
                                   "coordinate": int(item["state"] == "FOUND"),
                                   "variance": int(item["state"] == "FOUND"),
                                   "reasoning": 0, "source": 0, "confidence": 0}})
        candidate = by_sid[item["scene_id"]]
        if candidate["geometry_verified"] and candidate["input_hashes_match"]:
            scene_id = item["scene_id"]
            batch = item["batch"]
            folder = OUT / f"batch_{batch}"
            capture = folder / "capture_attempt_01"
            origin = "REVISION_V2_QC_PASS"
        else:
            fix = selected_repairs.get(item["scene_id"])
            if fix is None:
                raise ValueError(f"No repair for {item['scene_id']}")
            scene_id = fix["scene_id"]
            folder = REPAIR
            capture = REPAIR / "capture_attempt_01"
            origin = "IE_REPAIR_QC_PASS"
        inputs = {row["scene_id"]: row for row in read_jsonl(capture / "input_manifest.jsonl")}
        inp = inputs[scene_id]
        ann = yaml.safe_load((folder / "annotations.yaml").read_text())["scenes"][scene_id]
        if ann["state"] != item["state"] or ann["family_id"] != item["parent_family_id"]:
            raise ValueError("Selected view state/family mismatch")
        target_uv = None
        if item["state"] == "FOUND":
            label_map = labels(capture / scene_id / "evaluator/semantic_labels.png")
            target_uv = geometry(label_map, ann["target_label"])["centroid_normalized_xy"]
            if target_uv is None:
                raise ValueError("FOUND without target point")
        model_input = {"rgb_path": str((capture / inp["input_files"]["rgb"]).relative_to(ROOT)),
                       "depth_path": str((capture / inp["input_files"]["depth_m"]).relative_to(ROOT)),
                       "instruction": inp["instruction"]}
        rows.append({"sample_id": scene_id, "family_id": item["parent_family_id"],
                     "view_id": "translated_or_repaired", "origin": origin,
                     "model_input": model_input,
                     "supervision": {"relation": item["relation"], "answerability": item["state"],
                                     "target_uv": target_uv},
                     "head_mask": {"relation": 1, "answerability": 1,
                                   "coordinate": int(item["state"] == "FOUND"),
                                   "variance": int(item["state"] == "FOUND"),
                                   "reasoning": 0, "source": 0, "confidence": 0}})
    if len(rows) != 512 or len({row["sample_id"] for row in rows}) != 512:
        raise ValueError("Pilot must contain exactly 512 unique images")
    families = Counter(row["family_id"] for row in rows)
    if len(families) != 256 or set(families.values()) != {2}:
        raise ValueError("Every old family must have exactly two views")
    counts = Counter(row["supervision"]["answerability"] for row in rows)
    if set(counts.values()) != {128}:
        raise ValueError(f"Pilot not balanced: {counts}")
    index_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    summary = {"status": "SELECTED_512_QC_VALID_NOT_G1_TRAIN_RELEASE",
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "images": 512, "independent_parent_families": 256,
               "views_per_family": 2, "answerability": dict(sorted(counts.items())),
               "origin_counts": dict(Counter(row["origin"] for row in rows)),
               "head_mask_counts": {name: sum(row["head_mask"][name] for row in rows)
                                    for name in rows[0]["head_mask"]},
               "index_sha256": sha256(index_path),
               "qc_sha256": sha256(OUT / "REVISION_V2_QC.json"),
               "repair_qc_sha256": sha256(REPAIR / "REPAIR_QC.json"),
               "model_input_allowlist": ["rgb_path", "depth_path", "instruction"],
               "note": "No v3 Test/Calibration; source/reasoning/confidence loss masked."}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": summary["status"], "images": 512,
                      "families": 256, "answerability": summary["answerability"]}))


if __name__ == "__main__":
    main()
