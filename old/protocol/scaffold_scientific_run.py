#!/usr/bin/env python3
"""Create a claim-safe P-CRA-U run folder with NOT_RUN result templates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = WORKSPACE / "results/wp3_data_audit_20260821/tables"
DEFAULT_DATASET = WORKSPACE / "datasets/roborefer_dataset_v1_prototype_20260818_155606"
PHASE_POLICY = {
    "OVERFIT_SMOKE": {"read_splits": ["train"], "selection_split": "train", "test_opened": False},
    "DEVELOPMENT_TRAIN": {"read_splits": ["train", "dev"], "selection_split": "dev", "test_opened": False},
    "CALIBRATION_FIT": {"read_splits": ["calibration"], "selection_split": None, "test_opened": False},
    "LOCKED_TEST_IID": {"read_splits": ["test_iid"], "selection_split": None, "test_opened": True},
    "LOCKED_TEST_OOD": {"read_splits": ["test_ood"], "selection_split": None, "test_opened": True},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def blank_rows(names: list[str], first_field: str, fields: list[str], notes: str) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        row = {field: "" for field in fields}
        row.update({first_field: name, "status": "NOT_RUN", "notes": notes})
        rows.append(row)
    return rows


def dataset_tables(audit_root: Path, output: Path) -> None:
    family_rows = read_csv(audit_root / "family_audit.csv")
    record_rows = read_csv(audit_root / "record_audit.csv")
    split_states: dict[str, Counter[str]] = {}
    for row in record_rows:
        split_states.setdefault(row["split"], Counter())[row["state"]] += 1
    label_map = {"train": "Train", "dev": "Dev", "calibration": "Calibration", "test": "Prototype-Test"}
    rows = []
    for split in ["train", "dev", "calibration", "test"]:
        families = [row for row in family_rows if row["split"] == split]
        states = split_states.get(split, Counter())
        rows.append({
            "status": "PROTOTYPE_ONLY" if split == "test" else "READY_FOR_DIAGNOSTIC_ONLY",
            "split": label_map[split],
            "families": len(families),
            "samples": sum(int(row["record_count"]) for row in families),
            "FOUND": states["FOUND"],
            "AMBIGUOUS": states["AMBIGUOUS"],
            "ABSENT": states["ABSENT"],
            "INSUFFICIENT_EVIDENCE": states["INSUFFICIENT_EVIDENCE"],
            "depth_dependent_families": sum(int(row["depth_dependent"]) for row in families),
            "OOD": "NO",
            "statistical_unit": "scene-query-family",
            "notes": "Prototype test is not final locked Test-IID." if split == "test" else "WP2 prototype; not sufficient for final calibration/generalization claims.",
        })
    for split, is_ood in [("Test-IID", "NO"), ("Test-OOD", "YES")]:
        rows.append({
            "status": "MISSING_LOCKED_SPLIT", "split": split, "families": "", "samples": "",
            "FOUND": "", "AMBIGUOUS": "", "ABSENT": "", "INSUFFICIENT_EVIDENCE": "",
            "depth_dependent_families": "", "OOD": is_ood,
            "statistical_unit": "scene-query-family", "notes": "Must be created and locked after development freeze.",
        })
    fields = ["status", "split", "families", "samples", "FOUND", "AMBIGUOUS", "ABSENT",
              "INSUFFICIENT_EVIDENCE", "depth_dependent_families", "OOD", "statistical_unit", "notes"]
    write_csv(output / "table_01_dataset_independence.csv", fields, rows)

    balance = []
    for split in ["train", "dev", "calibration", "test"]:
        subset = [row for row in record_rows if row["split"] == split]
        relations, sources = Counter(), Counter()
        relation_families: dict[str, set[str]] = {}
        source_families: dict[str, set[str]] = {}
        for row in subset:
            for relation in filter(None, row["relations"].split(";")):
                relations[relation] += 1
                relation_families.setdefault(relation, set()).add(row["family_id"])
            for source in filter(None, row["sources"].split(";")):
                sources[source] += 1
                source_families.setdefault(source, set()).add(row["family_id"])
        for kind, counts, family_support in [
            ("relation", relations, relation_families),
            ("source", sources, source_families),
        ]:
            for label, count in sorted(counts.items()):
                balance.append({"split": label_map[split], "kind": kind, "label": label,
                                "sample_count": count, "family_count": len(family_support[label]),
                                "status": "DESCRIPTIVE_ONLY", "notes": "Family count is the independent support."})
    write_csv(output / "table_01b_relation_source_balance.csv",
              ["split", "kind", "label", "sample_count", "family_count", "status", "notes"], balance)


def result_templates(output: Path) -> list[dict[str, Any]]:
    tables: list[tuple[str, str, str, list[str], str, list[dict[str, Any]]]] = []

    fields2 = ["method", "status", "n_families", "n_samples", "seed_count", "instance_accuracy",
               "instance_accuracy_ci95", "point_in_target", "point_in_target_ci95", "point_in_interior",
               "point_in_interior_ci95", "mass_in_target", "mass_in_target_ci95", "top2_recall",
               "top2_recall_ci95", "normalized_2d_error", "normalized_2d_error_ci95",
               "latency_median_ms", "latency_p95_ms", "peak_vram_gib", "notes"]
    tables.append(("T02", "table_02_grounding_main.csv", "required", fields2, "Final values require frozen predictions.",
                   blank_rows(["B0_RoboRefer_RGB", "B1_RoboRefer_RGBD", "B2_RGBD_geometry_gate",
                               "U1_heatmap_no_relation", "P1_relation_aware_heatmap"], "method", fields2,
                              "Final values require frozen predictions.")))
    fields2b = ["relation", "status", "n_families", "n_samples", "B1_accuracy", "U1_accuracy",
                "P1_accuracy", "P1_minus_U1", "P1_minus_U1_ci95", "notes"]
    tables.append(("T02B", "table_02b_grounding_by_relation.csv", "supporting", fields2b,
                   "Subgroup claims require enough independent families.",
                   blank_rows(["direct", "left_right", "front_behind", "near_far", "multi_anchor"],
                              "relation", fields2b, "Subgroup claims require enough independent families.")))
    fields3 = ["method", "status", "n_families", "n_samples", "macro_f1", "macro_f1_ci95",
               "error_auprc", "error_auprc_ci95", "error_auroc", "error_auroc_ci95",
               "recall_ambiguous", "recall_absent", "recall_insufficient_evidence",
               "false_found_rate", "false_found_numerator", "false_found_denominator", "notes"]
    tables.append(("T03", "table_03_answerability.csv", "required", fields3, "False FOUND denominator is truth AMBIGUOUS or ABSENT.",
                   blank_rows(["geometry_gate", "entropy_disagreement", "pcra_answerability",
                               "pcra_multimodal_evidence"], "method", fields3,
                              "False FOUND denominator is truth AMBIGUOUS or ABSENT.")))
    fields4 = ["risk_model", "calibration", "status", "n_calibration_families", "n_test_families",
               "brier", "brier_ci95", "nll", "nll_ci95", "ece", "ece_ci95", "aurc", "aurc_ci95",
               "risk_at_80_coverage", "risk_at_80_coverage_ci95", "coverage_at_5_risk",
               "coverage_at_5_risk_ci95", "calibration_slope", "calibration_intercept",
               "frozen_threshold", "model_checkpoint_sha256", "calibrator_sha256", "notes"]
    calibration_rows = blank_rows(
        ["raw_max_score", "geometry_only", "spatial_evidence", "multimodal_evidence", "multimodal_evidence"],
        "risk_model", fields4, "NOT_RUN until model freeze and adequate calibration/Test-IID exist."
    )
    for row, calibration in zip(calibration_rows, ["none", "platt", "platt", "none", "platt"]):
        row["calibration"] = calibration
    tables.append(("T04", "table_04_calibration.csv", "primary", fields4,
                   "NOT_RUN until model freeze and adequate calibration/Test-IID exist.", calibration_rows))
    fields5 = ["perturbation", "status", "n_paired_families", "delta_semantic", "delta_semantic_ci95",
               "delta_relation", "delta_relation_ci95", "delta_depth", "delta_depth_ci95",
               "delta_occlusion", "delta_occlusion_ci95", "correct_source", "diagonal_margin",
               "diagonal_margin_ci95", "correct_source_largest_rate", "notes"]
    tables.append(("T05", "table_05_source_attribution.csv", "supporting", fields5,
                   "Paired clean/counterfactual families only.",
                   blank_rows(["shuffled_depth", "flat_depth", "relation_swap", "anchor_absent",
                               "increased_occlusion"], "perturbation", fields5,
                              "Paired clean/counterfactual families only.")))
    fields6 = ["configuration", "status", "n_families", "seed_count", "grounding_accuracy",
               "grounding_accuracy_ci95", "answerability_f1", "answerability_f1_ci95",
               "error_auprc", "error_auprc_ci95", "brier", "brier_ci95", "aurc", "aurc_ci95",
               "delta_primary_vs_full", "delta_primary_ci95", "notes"]
    tables.append(("T06", "table_06_ablation.csv", "required", fields6, "Same budget and selection rule as full model.",
                   blank_rows(["full_pcra_u", "no_relation_conditioning", "no_anchor_heatmap",
                               "no_depth_feature", "no_counterfactual_supervision",
                               "scalar_instead_of_multimodal_risk"], "configuration", fields6,
                              "Same budget and selection rule as full model.")))
    fields7 = ["run_id", "seed", "status", "best_dev_step", "selection_metric", "selection_value",
               "final_train_loss", "gradient_finite", "resume_exact", "checkpoint_verified", "notes"]
    tables.append(("T07", "table_07_training_checkpoint.csv", "diagnostic", fields7,
                   "Filled from training logs/checkpoint manifests, never manually.",
                   blank_rows(["CURRENT_RUN"], "run_id", fields7,
                              "Filled from training logs/checkpoint manifests, never manually.")))
    fields8 = ["method", "status", "trainable_params", "total_params", "feature_latency_ms",
               "head_latency_ms", "end_to_end_median_ms", "end_to_end_p95_ms", "peak_vram_gib",
               "gpu", "dtype", "batch_size", "tile_count", "notes"]
    tables.append(("T08", "table_08_efficiency.csv", "supporting", fields8,
                   "Compare only under identical hardware/config.",
                   blank_rows(["B1_RoboRefer_RGBD", "U1_heatmap_no_relation", "P1_relation_aware_heatmap"],
                              "method", fields8, "Compare only under identical hardware/config.")))
    fields9 = ["contrast", "status", "metric", "n_paired_families", "estimate_delta", "ci95_low",
               "ci95_high", "bootstrap_replicates", "p_value", "correction", "conclusion", "notes"]
    tables.append(("T09", "table_09_family_statistics.csv", "required", fields9,
                   "Primary inference uses paired cluster bootstrap by family.",
                   blank_rows(["P1_vs_U1", "P1_vs_B1", "calibrated_vs_raw", "multimodal_vs_scalar_risk"],
                              "contrast", fields9, "Primary inference uses paired cluster bootstrap by family.")))
    fields10 = ["subgroup", "status", "n_iid_families", "n_ood_families", "iid_metric", "iid_ci95",
                "ood_metric", "ood_ci95", "ood_gap", "ood_gap_ci95", "notes"]
    tables.append(("T10", "table_10_ood_subgroups.csv", "supporting", fields10,
                   "Requires separately locked Test-IID and Test-OOD.",
                   blank_rows(["overall", "relation", "depth_dependent", "occluded", "language_template"],
                              "subgroup", fields10, "Requires separately locked Test-IID and Test-OOD.")))
    fields11 = ["policy", "status", "n_families", "coverage", "coverage_ci95", "accepted_error",
                "accepted_error_ci95", "false_accept_rate", "false_accept_ci95", "wrong_object_execution",
                "wrong_object_execution_ci95", "reobserve_rate", "reobserve_recovery",
                "reobserve_recovery_ci95", "utility", "cost_definition", "notes"]
    tables.append(("T11", "table_11_selective_execution.csv", "primary", fields11,
                   "Separate perception false accept from controller/infrastructure failure.",
                   blank_rows(["always_execute", "geometry_gate", "calibrated_pcra_u",
                               "calibrated_pcra_u_reobserve"], "policy", fields11,
                              "Separate perception false accept from controller/infrastructure failure.")))
    fields12 = ["method", "status", "n_families", "empirical_coverage", "coverage_target", "coverage_gap",
                "normalized_region_area", "coverage_area_auc", "localization_3d_error_mm",
                "localization_3d_error_ci95", "covariance_calibration", "notes"]
    tables.append(("T12", "table_12_spatial_region_3d.csv", "exploratory", fields12,
                   "Optional; only after the spatial-region/metric-depth gate opens.",
                   blank_rows(["raw_spatial_region", "conformal_spatial_region", "metric_3d_covariance"],
                              "method", fields12,
                              "Optional; only after the spatial-region/metric-depth gate opens.")))

    index = [
        {"id": "T01", "path": "tables/table_01_dataset_independence.csv", "status": "PARTIAL",
         "claim_role": "required", "reason": "WP2 prototype known; final Test-IID/OOD missing."},
        {"id": "T01B", "path": "tables/table_01b_relation_source_balance.csv", "status": "PARTIAL",
         "claim_role": "supporting", "reason": "Prototype descriptive counts only."},
    ]
    for table_id, filename, role, fields, reason, rows in tables:
        write_csv(output / "tables" / filename, fields, rows)
        index.append({"id": table_id, "path": f"tables/{filename}", "status": "NOT_RUN",
                      "claim_role": role, "reason": reason})
    return index


def figure_and_image_indices(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    figures = [
        ("F01", "dataset_balance", "required", "Allowed before train; generated source data must accompany it."),
        ("F02", "training_curves", "diagnostic", "Loss components, gradient norm and LR by step."),
        ("F03", "checkpoint_selection", "required", "Dev metric with selected checkpoint marked."),
        ("F04", "grounding_qualitative", "required", "RGB/depth/GT/B1/U1/P1 panel."),
        ("F05", "answerability_confusion_pr", "required", "4x4 confusion matrix and error PR."),
        ("F06", "reliability", "primary", "Before/after calibration reliability."),
        ("F07", "risk_coverage", "primary", "Same-axis selective risk versus coverage."),
        ("F08", "source_delta", "supporting", "Paired counterfactual delta with CI."),
        ("F09", "ablation_forest", "required", "Paired ablation effect sizes."),
        ("F10", "iid_ood_gap", "supporting", "Requires locked IID/OOD."),
        ("F11", "selective_execution", "primary", "Risk/utility and reobserve outcomes at matched operating points."),
        ("F12", "spatial_region_3d", "exploratory", "Optional coverage-size and 3D covariance figure."),
    ]
    figure_rows = [{"figure_id": fid, "name": name, "status": "NOT_RUN", "claim_role": role,
                    "data_path": "", "svg_path": "", "png_path": "", "reason": reason}
                   for fid, name, role, reason in figures]
    write_csv(output / "figures/FIGURE_INDEX.csv",
              ["figure_id", "name", "status", "claim_role", "data_path", "svg_path", "png_path", "reason"],
              figure_rows)

    strata = ["FOUND_correct", "FOUND_incorrect", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE",
              "relation_dependent", "depth_corruption", "occlusion", "OOD"]
    image_rows = [{"stratum": name, "selection_status": "NOT_RUN", "family_id": "", "sample_id": "",
                   "selection_rule": "hash-sort within stratum using run seed", "image_path": "", "notes": ""}
                  for name in strata]
    write_csv(output / "images/QUALITATIVE_SELECTION.csv",
              ["stratum", "selection_status", "family_id", "sample_id", "selection_rule", "image_path", "notes"],
              image_rows)
    figure_index = [{"id": fid, "path": f"figures/{name}.svg", "status": "NOT_RUN",
                     "claim_role": role, "reason": reason} for fid, name, role, reason in figures]
    image_index = [{"id": "I01", "path": "images/QUALITATIVE_SELECTION.csv", "status": "NOT_RUN",
                    "claim_role": "required", "reason": "Selection is precommitted; rendering requires predictions."}]
    return figure_index, image_index


def rebuild_root_index(root: Path) -> None:
    records = []
    for path in sorted(root.glob("*/config/run_config.json")):
        config = read_json(path)
        records.append({"run_id": config["run_id"], "phase": config["phase"], "status": config["status"],
                        "seed": config["seed"], "created_at_utc": config["created_at_utc"],
                        "path": str(path.parent.parent.relative_to(root))})
    write_csv(root / "RUN_INDEX.csv", ["run_id", "phase", "status", "seed", "created_at_utc", "path"], records)
    write_json(root / "RUN_INDEX.json", {"schema_version": 1, "run_count": len(records), "runs": records})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=sorted(PHASE_POLICY), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", default="results/pcra_u_runs")
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT.relative_to(WORKSPACE)))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET.relative_to(WORKSPACE)))
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", args.run_id):
        raise SystemExit("run-id must match [a-z0-9][a-z0-9_.-]+")
    root = (WORKSPACE / args.output_root).resolve()
    run = root / args.run_id
    if run.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {run}")
    audit_root = (WORKSPACE / args.audit_root).resolve()
    dataset_root = (WORKSPACE / args.dataset_root).resolve()
    for child in ["config", "checkpoints", "logs", "metrics", "predictions", "tables", "figures",
                  "images", "manifests", "checks"]:
        (run / child).mkdir(parents=True, exist_ok=False)

    dataset_tables(audit_root, run / "tables")
    table_index = result_templates(run)
    figure_index, image_index = figure_and_image_indices(run)
    now = datetime.now(timezone.utc).isoformat()
    identity_paths = {
        "dataset_artifact_manifest_sha256": dataset_root / "artifact_manifest.json",
        "family_manifest_sha256": dataset_root / "family_manifest.json",
        "dataset_index_sha256": dataset_root / "dataset_index.json",
        "graph_feature_contract_sha256": WORKSPACE / "protocol/WP3_GRAPH_FEATURE_CONTRACT.md",
        "feature_cache_schema_sha256": WORKSPACE / "protocol/feature_cache_v1.schema.json",
        "feature_gate_lock_sha256": WORKSPACE / "protocol/wp3_feature_gate_lock.json",
        "scientific_contract_sha256": WORKSPACE / "protocol/WP4_SCIENTIFIC_EVIDENCE_CONTRACT.md",
        "checkpoint_schema_sha256": WORKSPACE / "protocol/training_checkpoint_v1.schema.json",
    }
    identity = {key: sha256(path) for key, path in identity_paths.items()}
    config = {
        "schema_version": 1, "run_id": args.run_id, "phase": args.phase,
        "status": "PLANNED_NOT_RUN", "seed": args.seed, "created_at_utc": now,
        "statistical_unit": "scene-query-family", "split_access": PHASE_POLICY[args.phase],
        "identity": identity,
        "model_config_status": "NOT_LOCKED",
        "training_performed": False,
        "calibration_fit_performed": False,
        "test_predictions_generated": False,
        "notes": "Scaffold only. Lock model/loss/subset contract before execution.",
    }
    write_json(run / "config/run_config.json", config)
    index = {
        "schema_version": 1, "run_id": args.run_id, "phase": args.phase, "status": "PLANNED_NOT_RUN",
        "statistical_unit": "scene-query-family", "split_access": PHASE_POLICY[args.phase],
        "tables": table_index, "figures": figure_index, "images": image_index,
        "checkpoint_policy": {"format": "safetensors+trusted-training-state", "atomic_publish": True,
                              "hash_verified": True, "auto_delete": False,
                              "best_selection_splits": ["train", "dev"]},
    }
    write_json(run / "metrics/scientific_results_index.json", index)
    (run / "checkpoints/README.md").write_text(
        "# Checkpoints\n\nUse `protocol/training_checkpoint_manager.py`; do not write ad-hoc `.pt` files here.\n",
        encoding="utf-8",
    )
    (run / "logs/metrics.jsonl").write_text("", encoding="utf-8")
    (run / "predictions/README.md").write_text(
        "# Predictions\n\nStore immutable per-sample predictions with family/sample/split/checkpoint identity.\n",
        encoding="utf-8",
    )
    card = f"""# Scientific run — `{args.run_id}`

> **PLANNED_NOT_RUN / {args.phase}**

- Statistical unit: `scene-query-family`;
- allowed read splits: `{PHASE_POLICY[args.phase]['read_splits']}`;
- checkpoint selection split: `{PHASE_POLICY[args.phase]['selection_split']}`;
- training performed: `NO`;
- results in Table 2–10: `NOT_RUN`;
- current dataset table: prototype descriptive evidence only.

No numeric cell in the result templates is a measured result yet. Update status only through a validated evaluator/report generator after predictions exist.
"""
    (run / "RUN_CARD.md").write_text(card, encoding="utf-8")

    check = {
        "schema_version": 1,
        "all_result_rows_marked_not_run": all(
            row.get("status") == "NOT_RUN"
            for table in run.glob("tables/table_*.csv")
            if not table.name.startswith("table_01")
            for row in read_csv(table)
        ),
        "test_opened": PHASE_POLICY[args.phase]["test_opened"],
        "training_performed": False,
        "calibration_fit_performed": False,
        "checkpoint_directory_empty_except_readme": len(list((run / "checkpoints").iterdir())) == 1,
    }
    check["passed"] = (check["all_result_rows_marked_not_run"]
                       and check["checkpoint_directory_empty_except_readme"])
    write_json(run / "checks/scaffold_check.json", check)
    artifacts = []
    for path in sorted(p for p in run.rglob("*") if p.is_file() and p.name != "scaffold_manifest.json"):
        artifacts.append({"path": str(path.relative_to(run)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(run / "manifests/scaffold_manifest.json", {
        "schema_version": 1, "run_id": args.run_id, "generated_at_utc": now,
        "artifact_count": len(artifacts), "artifacts": artifacts,
    })
    rebuild_root_index(root)
    print(json.dumps({"run": str(run.relative_to(WORKSPACE)), "artifacts": len(artifacts),
                      "check_passed": check["passed"]}, indent=2))


if __name__ == "__main__":
    main()
