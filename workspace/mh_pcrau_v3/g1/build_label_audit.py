"""Build a development-only, human-editable G1 label provenance packet.

This never trains, certifies labels, changes source manifests, or opens sealed
Calibration/Test. Candidate labels and audit responses are intentionally separate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .audit_development import (
    DAY1, DAY2, GAZEBO, ROOT, STATES, TABLETOP, map_relation, read_json,
    read_jsonl, ref, sha256, valid_point,
)


DEFAULT_OUTPUT = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03/label_audit_packet_v1"
SOURCE_CLASSES = (
    "semantic_mismatch_absence", "relation_tie_conflict",
    "spatial_localization_difficulty", "depth_invalid_noisy",
    "occlusion_visibility_insufficiency",
)
REVIEW_COLUMNS = (
    "reviewer_id", "review_relation", "review_answerability",
    "review_target_correct", "review_reasoning_depth", "review_source",
    "review_single_source", "review_notes", "review_status",
)
QUEUE_COLUMNS = (
    "queue_id", "dataset", "sample_id", "family_id", "split", "stratum",
    "rgb_path", "depth_view_path", "instruction", "candidate_relation",
    "candidate_answerability", "candidate_target_uv", "candidate_source_file",
    *REVIEW_COLUMNS,
)
BLIND_COLUMNS = (
    "queue_id", "dataset", "sample_id", "family_id", "split",
    "rgb_path", "depth_view_path", "instruction", *REVIEW_COLUMNS,
)
PROVENANCE_COLUMNS = (
    "dataset", "sample_id", "family_id", "split", "rgb_path",
    "depth_view_path", "metric_depth_path", "instruction", "relation_candidate",
    "relation_source", "answerability_candidate", "answerability_source",
    "target_uv_candidate", "target_source", "reasoning_depth_source",
    "uncertainty_source_source", "safe_to_execute_source", "camera_info_path",
    "tf_snapshot_path", "camera_info_valid", "tf_snapshot_valid",
    "relation_raw_valid", "answerability_raw_valid", "coordinate_raw_valid",
    "variance_raw_valid", "reasoning_raw_valid", "source_raw_valid",
    "confidence_raw_valid", "legacy_b1_target_valid", "core_schema_candidate_valid",
)
HEADS = (
    "relation", "answerability", "coordinate", "variance", "reasoning",
    "source", "confidence",
)


def verify_prerequisites():
    decision = read_json(DAY2 / "G0_DECISION.json")
    if decision.get("outcome") != "G0_PASS":
        raise ValueError("G0_PASS is required")
    boundary = read_json(DAY1 / "DATA_ACCESS_BOUNDARY.json")
    expected = {item["dataset"]: item["manifest_sha256"]
                for item in boundary["development_data"]}
    for name, folder in (("D_tabletop_clean_v1", TABLETOP),
                         ("Gazebo_train_uq_v2_full_r3", GAZEBO)):
        if sha256(folder / "manifest.json") != expected[name]:
            raise ValueError(f"Day 1 manifest hash mismatch: {name}")
    return ref(DAY2 / "G0_DECISION.json"), ref(DAY1 / "DATA_ACCESS_BOUNDARY.json")


def project_path(path: Path) -> str:
    return str(path.relative_to(ROOT))


def camera_sidecars(row):
    base = GAZEBO / row["image"]
    camera = base.parent / "camera_info.json"
    tf = base.parent / "tf_snapshot.json"
    camera_valid = tf_valid = False
    if camera.is_file():
        try:
            info = read_json(camera)
            k = info.get("k")
            camera_valid = (isinstance(k, list) and len(k) == 9
                            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in k)
                            and k[0] > 0 and k[4] > 0
                            and info.get("width", 0) > 0 and info.get("height", 0) > 0)
        except (ValueError, TypeError, KeyError):
            pass
    if tf.is_file():
        try:
            frames = read_json(tf)
            pose = frames.get("camera_color_optical_frame", {})
            tf_valid = (pose.get("parent_frame") == "base_link"
                        and len(pose.get("position", [])) == 3
                        and len(pose.get("orientation_xyzw", [])) == 4
                        and all(isinstance(x, (int, float)) and math.isfinite(x)
                                for x in pose["position"] + pose["orientation_xyzw"]))
        except (ValueError, TypeError, KeyError):
            pass
    return camera, tf, camera_valid, tf_valid


def build_rows():
    rows = []
    for line, source in enumerate(read_jsonl(TABLETOP / "provenance.jsonl"), 1):
        relation = map_relation(source.get("relation", ""), source.get("instruction", ""))
        rgb = ROOT / source["image"]
        depth = ROOT / source["depth"]
        target = source.get("target_xy")
        media_ok = rgb.is_file() and depth.is_file() and bool(source.get("instruction", "").strip())
        legacy_target = media_ok and valid_point(target)
        record = {
            "dataset": "D_tabletop_clean_v1", "sample_id": source["sample_id"],
            "family_id": source["family_id"], "split": source["split"],
            "rgb_path": project_path(rgb), "depth_view_path": project_path(depth),
            "metric_depth_path": "", "instruction": source["instruction"],
            "relation_candidate": relation or "", "relation_source":
                f"datasets/D_tabletop_clean_v1/provenance.jsonl:{line}:relation+instruction (B1 extraction; v3 mapping candidate)",
            "answerability_candidate": "", "answerability_source": "MISSING",
            "target_uv_candidate": json.dumps(target) if valid_point(target) else "",
            "target_source": f"datasets/D_tabletop_clean_v1/provenance.jsonl:{line}:target_xy (B1 only)",
            "reasoning_depth_source": "MISSING", "uncertainty_source_source": "MISSING",
            "safe_to_execute_source": "MISSING: requires OOF correctness in S1b",
            "camera_info_path": "", "tf_snapshot_path": "",
            "camera_info_valid": False, "tf_snapshot_valid": False,
            "relation_raw_valid": media_ok and bool(relation),
            "answerability_raw_valid": False, "coordinate_raw_valid": False,
            "variance_raw_valid": False, "reasoning_raw_valid": False,
            "source_raw_valid": False, "confidence_raw_valid": False,
            "legacy_b1_target_valid": legacy_target,
            "core_schema_candidate_valid": False,
        }
        rows.append(record)
    infer_rows = list(read_jsonl(GAZEBO / "inference_manifest.jsonl"))
    supervision = list(read_jsonl(GAZEBO / "train_supervision.jsonl"))
    by_id = {row["sample_id"]: (line, row) for line, row in enumerate(supervision, 1)}
    if len(by_id) != len(supervision):
        raise ValueError("Duplicate Gazebo train supervision IDs")
    infer_ids = {row["sample_id"] for row in infer_rows}
    if not set(by_id).issubset(infer_ids):
        raise ValueError("Gazebo supervision not contained in inference manifest")
    for line, source in enumerate(infer_rows, 1):
        if "supervision" in source:
            raise ValueError("Oracle field in Gazebo inference manifest")
        sample_id = source["sample_id"]
        sup_line, sup_row = by_id.get(sample_id, (None, None))
        if sup_row and (sup_row["family_id"] != source["family_id"] or
                        sup_row["split"] != source["split"]):
            raise ValueError(f"Gazebo supervision join mismatch: {sample_id}")
        sup = sup_row["supervision"] if sup_row else {}
        rgb = GAZEBO / source["image"]
        depth = GAZEBO / source["depth"]
        metric = GAZEBO / source["metric_depth"]
        camera, tf, camera_valid, tf_valid = camera_sidecars(source)
        relation = map_relation(source.get("relation_variant", ""), source.get("instruction", ""))
        state = sup.get("answerability_state", "")
        target = sup.get("target_xy")
        media_ok = (rgb.is_file() and depth.is_file() and metric.is_file()
                    and bool(source.get("instruction", "").strip()))
        answer_valid = media_ok and state in STATES
        coord_valid = answer_valid and state == "FOUND" and valid_point(target)
        record = {
            "dataset": "Gazebo_train_uq_v2_full_r3", "sample_id": sample_id,
            "family_id": source["family_id"], "split": source["split"],
            "rgb_path": project_path(rgb), "depth_view_path": project_path(depth),
            "metric_depth_path": project_path(metric), "instruction": source["instruction"],
            "relation_candidate": relation or "", "relation_source":
                f"datasets/Gazebo_train_uq_v2_full_r3/inference_manifest.jsonl:{line}:relation_variant (prompt metadata; human v3 review pending)",
            "answerability_candidate": state, "answerability_source":
                (f"datasets/Gazebo_train_uq_v2_full_r3/train_supervision.jsonl:{sup_line}:supervision.answerability_state"
                 if sup_line else "HELD_OUT: val_uq evaluator GT not joined into train packet"),
            "target_uv_candidate": json.dumps(target) if valid_point(target) else "",
            "target_source": (f"datasets/Gazebo_train_uq_v2_full_r3/train_supervision.jsonl:{sup_line}:supervision.target_xy"
                              if sup_line else "HELD_OUT"),
            "reasoning_depth_source": "MISSING", "uncertainty_source_source": "MISSING",
            "safe_to_execute_source": "MISSING: requires OOF correctness in S1b",
            "camera_info_path": project_path(camera) if camera.is_file() else "",
            "tf_snapshot_path": project_path(tf) if tf.is_file() else "",
            "camera_info_valid": camera_valid, "tf_snapshot_valid": tf_valid,
            "relation_raw_valid": media_ok and bool(relation),
            "answerability_raw_valid": answer_valid,
            "coordinate_raw_valid": coord_valid, "variance_raw_valid": coord_valid,
            "reasoning_raw_valid": False, "source_raw_valid": False,
            "confidence_raw_valid": False, "legacy_b1_target_valid": False,
            "core_schema_candidate_valid": bool(media_ok and relation and answer_valid
                                         and (state != "FOUND" or coord_valid)
                                         and camera_valid and tf_valid),
        }
        rows.append(record)
    if len({(r["dataset"], r["sample_id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate dataset/sample ID")
    return rows


def summary(rows):
    result = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        selected = [r for r in rows if r["dataset"] == dataset]
        splits = {}
        for split in sorted({r["split"] for r in selected}):
            group = [r for r in selected if r["split"] == split]
            counts = {head: sum(bool(r[f"{head}_raw_valid"]) for r in group) for head in HEADS}
            counts["legacy_b1_target"] = sum(r["legacy_b1_target_valid"] for r in group)
            counts["core_schema_candidate"] = sum(r["core_schema_candidate_valid"] for r in group)
            splits[split] = {"records": len(group), "families": len({r["family_id"] for r in group}),
                             "raw_label_candidates_not_v3_certified": counts,
                             "relation_class_candidates": dict(sorted(Counter(
                                 r["relation_candidate"] or "UNMAPPED" for r in group).items())),
                             "answerability_train_label_candidates": dict(sorted(Counter(
                                 r["answerability_candidate"] for r in group
                                 if r["answerability_raw_valid"]).items())),
                             "camera_info_valid": sum(r["camera_info_valid"] for r in group),
                             "tf_snapshot_valid": sum(r["tf_snapshot_valid"] for r in group)}
        result[dataset] = splits
    return result


def select_queue(rows, seed: str):
    strata = defaultdict(list)
    for row in rows:
        if row["dataset"] == "D_tabletop_clean_v1" and row["split"] == "train":
            key = row["relation_candidate"] or "UNMAPPED"
            strata[("tabletop", key)].append(row)
        elif row["dataset"] == "Gazebo_train_uq_v2_full_r3" and row["split"] == "train_uq":
            key = row["answerability_candidate"]
            if key in STATES:
                strata[("gazebo", key)].append(row)
    expected = {("tabletop", label): 16 for label in
                ("leftmost", "rightmost", "second_from_left", "second_from_right", "UNMAPPED")}
    expected.update({("gazebo", state): 12 for state in STATES})
    if set(strata) != set(expected):
        raise ValueError(f"Unexpected audit strata: {sorted(set(strata) ^ set(expected))}")
    queue = []
    for key, amount in expected.items():
        pool = strata[key]
        if len(pool) < amount:
            raise ValueError(f"Insufficient audit stratum {key}: {len(pool)} < {amount}")
        ranked = sorted(pool, key=lambda r: hashlib.sha256(
            f"{seed}:{r['dataset']}:{r['sample_id']}".encode()).hexdigest())
        for row in ranked[:amount]:
            queue.append({
                "queue_id": f"G1-{len(queue)+1:03d}", "dataset": row["dataset"],
                "sample_id": row["sample_id"], "family_id": row["family_id"],
                "split": row["split"], "stratum": "/".join(key),
                "rgb_path": row["rgb_path"], "depth_view_path": row["depth_view_path"],
                "instruction": row["instruction"],
                "candidate_relation": row["relation_candidate"],
                "candidate_answerability": row["answerability_candidate"],
                "candidate_target_uv": row["target_uv_candidate"],
                "candidate_source_file": row["relation_source"],
                **{name: "" for name in REVIEW_COLUMNS},
            })
    return queue, {"seed": seed, "selection": "SHA256(seed:dataset:sample_id), lowest within each development-train stratum",
                   "per_stratum": {"/".join(k): n for k, n in expected.items()},
                   "n": len(queue), "purpose": "PILOT_HUMAN_AUDIT; not a locked precision/power sample or G1 PASS"}


def write_csv(path, columns, rows):
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", default="mh_pcrau_v3_g1_pilot_20260924")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit packet: {output}")
    g0, boundary = verify_prerequisites()
    rows = build_rows()
    queue, selection = select_queue(rows, args.seed)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "ROW_LABEL_PROVENANCE.csv", PROVENANCE_COLUMNS, rows)
    write_csv(output / "HUMAN_AUDIT_QUEUE.csv", QUEUE_COLUMNS, queue)
    write_csv(output / "HUMAN_AUDIT_BLIND.csv", BLIND_COLUMNS, queue)
    source_map = {
        "status": "RUN_COMPLETE", "gate_state": "G1_IN_PROGRESS",
        "source_script": ref(Path(__file__)), "g0_decision": g0, "day1_boundary": boundary,
        "input_manifests": [ref(TABLETOP / "manifest.json"), ref(GAZEBO / "manifest.json")],
        "input_rows": [ref(TABLETOP / "provenance.jsonl"), ref(GAZEBO / "inference_manifest.jsonl"),
                       ref(GAZEBO / "train_supervision.jsonl")],
        "provenance_rules": {
            "relation": "Tabletop provenance.relation + instruction (B1 only); Gazebo inference_manifest.relation_variant; both candidate v3 mapping pending human review",
            "answerability": "Gazebo train_supervision.supervision.answerability_state for train_uq only; val_uq evaluator not joined",
            "coordinate_and_variance": "Gazebo train_supervision.supervision.target_xy only if FOUND and normalized point valid; Tabletop target_xy retained as legacy B1, not v3 coordinate supervision",
            "reasoning_depth": "No certified 3-class label in authorized records",
            "uncertainty_source": "No certified 5-class single-source label; failure_tags and answerability must not be substituted",
            "confidence": "No OOF correctness label; S1b only after family-disjoint OOF predictions",
            "camera_intrinsics": "Gazebo record input/camera_info.json sidecar (K checked); Tabletop missing",
            "tf_provenance": "Gazebo record input/tf_snapshot.json sidecar (camera to base_link checked); Tabletop missing",
            "metric_depth": "Gazebo input/depth_m.npy path exists; Tabletop depth PNG is relative visual depth, not metric",
        },
        "source_classes_from_plan": SOURCE_CLASSES,
        "reasoning_depth_rule": "PENDING_ONTOLOGY_LOCK: numeric 0/1/2 are placeholders; human labels not training-certified",
        "source_role": "EXPLORATORY_UNTIL_HUMAN_PRECISION_GATE",
        "limitations": ["No human labels have been supplied in this packet",
                        "Raw-valid labels are not G1-certified or approved for v3 training",
                        "Sidecar existence/basic structure is not geometry alignment/depth QC",
                        "Pilot queue is not the preregistered power/precision audit sample"],
    }
    write_json(output / "LABEL_SOURCE_MAP.json", source_map)
    write_json(output / "HEAD_COVERAGE.json", {
        "status": "RUN_COMPLETE", "gate_state": "G1_IN_PROGRESS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "count_definition": "Raw label candidate + basic input/format validity; core_schema_candidate permits nullable reasoning/source but is not human-certified, geometry-QC-complete, or v3 gate/training eligible",
        "ready_for_v3_training_now": {head: 0 for head in HEADS},
        "by_dataset_split": summary(rows), "pilot_selection": selection,
        "queue_file_sha256": sha256(output / "HUMAN_AUDIT_QUEUE.csv"),
        "blind_file_sha256": sha256(output / "HUMAN_AUDIT_BLIND.csv"),
        "provenance_file_sha256": sha256(output / "ROW_LABEL_PROVENANCE.csv"),
    })
    print(json.dumps({"output": project_path(output), "records": len(rows),
                      "pilot_queue": len(queue), "coverage": summary(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
