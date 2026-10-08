"""Validate development-only v3 candidate schema; never grant G1 training access."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .audit_development import ROOT, STATES, ref
from .build_label_audit import build_rows, verify_prerequisites


DAY3 = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03"
SCHEMA_PATH = DAY3 / "SUPERVISION_SCHEMA_V3.json"
INPUT_SCHEMA_PATH = DAY3 / "MODEL_INPUT_SCHEMA_V3.json"
RELATIONS = {"leftmost", "rightmost", "second_from_left", "second_from_right"}
SOURCES = {"SEMANTIC", "RELATION", "SPATIAL", "DEPTH", "OCCLUSION"}
TARGET_SCOPES = {"NONE", "B1_LEGACY_ONLY", "V3_2D_CANDIDATE", "V3_2D_CERTIFIED"}


def candidate_record(row: dict) -> dict:
    target = json.loads(row["target_uv_candidate"]) if row["target_uv_candidate"] else None
    tabletop = row["dataset"] == "D_tabletop_clean_v1"
    answerability = row["answerability_candidate"] or None
    if tabletop:
        # B1 target is deliberately not promoted into the v3 target field.
        target_uv = None
        scope = "B1_LEGACY_ONLY" if row["legacy_b1_target_valid"] else "NONE"
    elif answerability == "FOUND" and row["coordinate_raw_valid"]:
        target_uv = target
        scope = "V3_2D_CANDIDATE"
    else:
        target_uv = None
        scope = "NONE"
    return {
        "sample_id": row["sample_id"], "family_id": row["family_id"],
        "split": row["split"], "rgb_path": row["rgb_path"],
        "depth_view_path": row["depth_view_path"],
        "metric_depth_path": row["metric_depth_path"] or None,
        "instruction": row["instruction"],
        "relation_class": row["relation_candidate"] or None,
        "reasoning_depth": None,
        "answerability_state": answerability,
        "target_uv": target_uv,
        "target_scope": scope,
        "primary_uncertainty_source": None,
        "safe_to_execute_label_source": None,
        "camera_intrinsics": row["camera_info_path"] or None,
        "tf_provenance": row["tf_snapshot_path"] or None,
        "supervision_status": "RAW_CANDIDATE_NOT_CERTIFIED",
        "training_eligible": False,
    }


def validate_record(row: dict, schema: dict) -> list[str]:
    errors = []
    expected = set(schema["required"])
    if set(row) != expected:
        errors.append(f"key mismatch: missing={sorted(expected-set(row))}, extra={sorted(set(row)-expected)}")
        return errors
    for key in ("sample_id", "family_id", "split", "rgb_path", "depth_view_path", "instruction"):
        if not isinstance(row[key], str) or not row[key].strip():
            errors.append(f"{key}: nonempty string required")
    for key in ("metric_depth_path", "camera_intrinsics", "tf_provenance", "safe_to_execute_label_source"):
        if row[key] is not None and (not isinstance(row[key], str) or not row[key].strip()):
            errors.append(f"{key}: null or nonempty string required")
    if row["relation_class"] is not None and row["relation_class"] not in RELATIONS:
        errors.append("relation_class: outside core ontology")
    if row["reasoning_depth"] is not None and (type(row["reasoning_depth"]) is not int or row["reasoning_depth"] not in (0, 1, 2)):
        errors.append("reasoning_depth: invalid")
    if row["answerability_state"] is not None and row["answerability_state"] not in STATES:
        errors.append("answerability_state: invalid")
    if row["primary_uncertainty_source"] is not None and row["primary_uncertainty_source"] not in SOURCES:
        errors.append("primary_uncertainty_source: invalid canonical value")
    if row["target_scope"] not in TARGET_SCOPES:
        errors.append("target_scope: invalid")
    target = row["target_uv"]
    if target is not None and (not isinstance(target, list) or len(target) != 2 or
                               any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in target)):
        errors.append("target_uv: expected finite normalized 2D point")
    if row["target_scope"] == "NONE" and target is not None:
        errors.append("target_scope NONE requires target_uv null")
    if row["target_scope"] == "B1_LEGACY_ONLY" and target is not None:
        errors.append("B1 target cannot be promoted into v3 target_uv")
    if row["target_scope"] in {"V3_2D_CANDIDATE", "V3_2D_CERTIFIED"} and (row["answerability_state"] != "FOUND" or target is None):
        errors.append("v3 target requires FOUND and point")
    if row["supervision_status"] not in schema["properties"]["supervision_status"]["enum"]:
        errors.append("supervision_status: invalid")
    if type(row["training_eligible"]) is not bool:
        errors.append("training_eligible: boolean required")
    if row["training_eligible"] and row["supervision_status"] != "G1_CERTIFIED":
        errors.append("uncertified supervision cannot be training eligible")
    return errors


def validate_model_input(record: dict, input_schema: dict) -> list[str]:
    required = set(input_schema["required"])
    if set(record) != required:
        return ["model input must contain only RGB, depth, instruction"]
    return [key for key in required if not isinstance(record[key], str) or not record[key].strip()]


def inspect(rows: list[dict], schema: dict) -> dict:
    by_split = defaultdict(list)
    issues = []
    seen = set()
    for line, raw in enumerate(rows, 1):
        record = candidate_record(raw)
        key = (raw["dataset"], record["sample_id"])
        if key in seen:
            issues.append(f"duplicate {key}")
        seen.add(key)
        issues.extend(f"row {line}: {item}" for item in validate_record(record, schema))
        by_split[(raw["dataset"], record["split"])].append(record)
    counts = {}
    for (dataset, split), group in sorted(by_split.items()):
        counts[f"{dataset}/{split}"] = {
            "rows": len(group), "families": len({r["family_id"] for r in group}),
            "relation_candidates": dict(sorted(Counter(r["relation_class"] or "UNMAPPED" for r in group).items())),
            "answerability_candidates": dict(sorted(Counter(r["answerability_state"] or "MISSING" for r in group).items())),
            "target_scopes": dict(sorted(Counter(r["target_scope"] for r in group).items())),
            "metric_depth_paths": sum(r["metric_depth_path"] is not None for r in group),
            "camera_sidecar_paths": sum(r["camera_intrinsics"] is not None for r in group),
            "tf_sidecar_paths": sum(r["tf_provenance"] is not None for r in group),
            "certified_reasoning": 0, "certified_source": 0,
            "v3_training_eligible": sum(r["training_eligible"] for r in group),
        }
    return {"structural_status": "PASS" if not issues else "FAIL", "issues": issues,
            "records_checked": len(rows), "by_dataset_split": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DAY3 / "SCHEMA_QC.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    verify_prerequisites()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    input_schema = json.loads(INPUT_SCHEMA_PATH.read_text(encoding="utf-8"))
    rows = build_rows()
    result = inspect(rows, schema)
    result.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_state": "G1_IN_PROGRESS",
        "schema": ref(SCHEMA_PATH), "model_input_schema": ref(INPUT_SCHEMA_PATH),
        "input_provenance": ref(ROOT / "datasets/D_tabletop_clean_v1/provenance.jsonl"),
        "input_gazebo_inference": ref(ROOT / "datasets/Gazebo_train_uq_v2_full_r3/inference_manifest.jsonl"),
        "input_gazebo_train_supervision": ref(ROOT / "datasets/Gazebo_train_uq_v2_full_r3/train_supervision.jsonl"),
        "interpretation": "Structural candidate validation only; labels and geometry not G1-certified; held-out evaluator GT not joined",
        "model_input_leakage_check": validate_model_input({"rgb_path": "x", "depth_path": "y", "instruction": "q"}, input_schema) == [],
        "model_input_leakage_check_scope": "Synthetic allowlist smoke only; runtime model-input construction requires its own audit",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"structural_status": result["structural_status"],
                      "records_checked": result["records_checked"],
                      "issues": len(result["issues"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
