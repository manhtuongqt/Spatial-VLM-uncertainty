#!/usr/bin/env python3
"""Fail-closed audit for the completed, zero-eligible P-CRA-U V1.1 campaign.

This add-only audit reads persisted train/DEV artifacts.  It does not load a
model, generate predictions, bootstrap, access Calibration/Test, or alter any
campaign artifact.  A retained-V1 decision lock is created only after every
integrity gate passes.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training_checkpoint_manager import audit_checkpoint_root, verify_checkpoint


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol"
RUNS = ROOT / "results/pcra_u_runs"
CAMPAIGN = RUNS / "pcra_u_development_v1_1_campaign_20260824"
SEEDS = (24082026, 24082027, 24082028)
SEED_ROOTS = {seed: RUNS / f"pcra_u_development_v1_1_campaign_20260824_seed_{seed}" for seed in SEEDS}
OUTPUT = RUNS / "pcra_u_development_v1_1_failure_audit_20260824"
DECISION_LOCK = PROTOCOL / "pcra_u_development_v1_1_decision_lock.json"

V1_ROOT = RUNS / "pcra_u_development_train_20260824"
V1_CHECKPOINT = V1_ROOT / "checkpoints/development/step_000002000"
V1_REPORT = V1_ROOT / "PCRA_U_DEVELOPMENT_TRAIN_REPORT.json"
V1_PREDICTIONS = V1_ROOT / "predictions/dev_predictions_exploratory.json"
V1_MODEL_SHA = "14dba40d2de437f082aece9b342931a48c58202ae4242d577d4a4ffd7eecaab3"
V1_PREDICTIONS_SHA = "3f6c68dc50611205cd4213911b8b59822cefd85487780f610c6ae409720d22d2"
V1_REPORT_SHA = "5c0873766ac5e637190e589a7eeaf77c2bef2da429d1dd50551836ca3738a805"

EXPECTED_EPOCHS = 20
EXPECTED_STEPS = 4000
STEPS_PER_EPOCH = 200
ELIGIBILITY = {
    "min_epoch": 6,
    "min_Ga": 0.8371428571,
    "min_A": 0.7220238435,
    "max_F": 0.1147368421,
}
SCIENTIFIC_STATUS = "EXPLORATORY_DEV_ONLY_FAILURE_AUDIT_CALIBRATION_TESTS_SEALED"
DECISION = "RETAIN_V1_STEP_2000_REJECT_V1_1"
TOL = 1e-12


class AuditError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"Invalid JSON {path}: {exc}") from exc


def canonical(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"Non-numeric {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise AuditError(f"Non-finite {label}: {result}")
    return result


def close(left: Any, right: Any) -> bool:
    return math.isclose(numeric(left, "left"), numeric(right, "right"), rel_tol=0.0, abs_tol=TOL)


def snapshot(paths: Sequence[Path]) -> dict[str, str]:
    return {rel(path): sha(path) for path in paths}


def score(gc: float, ga: float, answer: float, false_found: float) -> float:
    return 0.45 * gc + 0.20 * ga + 0.20 * answer + 0.15 * (1.0 - false_found)


def epoch_row(seed: int, path: Path) -> dict[str, Any]:
    item = read(path)
    epoch = int(item["epoch"])
    dev = item["dev"]
    selection = item["selection"]
    gc = numeric(dev["clean_found"]["raw_point_in_target"], "Gc")
    ga = numeric(dev["grounding_found"]["point_in_target"], "Ga")
    answer = numeric(dev["answerability"]["macro_f1"], "A")
    false_found = numeric(dev["false_found_on_ambiguous_or_absent"]["rate"], "F")
    source = numeric(dev["source_micro_f1"], "S")
    composite = numeric(selection["value"], "C")
    checks = {
        "epoch_at_least_minimum": epoch >= ELIGIBILITY["min_epoch"],
        "all_found_grounding_at_least_minimum": ga >= ELIGIBILITY["min_Ga"],
        "answerability_macro_f1_at_least_minimum": answer >= ELIGIBILITY["min_A"],
        "false_found_rate_at_most_maximum": false_found <= ELIGIBILITY["max_F"],
    }
    exact_counts = (
        int(dev["sample_count"]) == 400
        and int(dev["family_count"]) == 80
        and int(dev["grounding_found"]["found_total"]) == 168
        and int(dev["clean_found"]["clean_found_total"]) == 32
        and int(dev["false_found_on_ambiguous_or_absent"]["denominator"]) == 95
    )
    valid = (
        epoch == int(path.stem.rsplit("_", 1)[-1])
        and int(item["global_step"]) == epoch * STEPS_PER_EPOCH
        and close(composite, score(gc, ga, answer, false_found))
        and selection["metric"] == "dev_scientific_score_v1_1"
        and selection["mode"] == "max"
        and selection["thresholds"]
        == {
            "min_epoch": 6,
            "min_all_found_point_in_target": 0.8371428571,
            "min_answerability_macro_f1": 0.7220238435,
            "max_false_found_rate": 0.1147368421,
        }
        and selection["eligibility_checks"] == checks
        and bool(selection["eligible"]) == all(checks.values())
        and exact_counts
    )
    return {
        "seed": seed,
        "epoch": epoch,
        "global_step": int(item["global_step"]),
        "train_total_loss": numeric(item["train_epoch_mean_total_loss"], "train loss"),
        "dev_total_loss": numeric(dev["loss"]["total"], "dev loss"),
        "Gc": gc,
        "Ga": ga,
        "A": answer,
        "F": false_found,
        "S": source,
        "C": composite,
        "eligible": bool(selection["eligible"]),
        **{f"check_{key}": value for key, value in checks.items()},
        "exact_dev_denominators": exact_counts,
        "epoch_artifact_valid": valid,
        "epoch_json": rel(path),
        "epoch_json_sha256": sha(path),
    }


def audit_log(seed: int, path: Path, epoch_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    finite_fields = (
        "total_loss",
        "heatmap_loss",
        "heatmap_bce",
        "heatmap_dice",
        "answerability_loss",
        "source_loss",
        "gradient_norm_preclip",
        "target_head_gradient",
        "answer_head_gradient",
        "source_head_gradient",
        "learning_rate",
    )
    sequential = len(records) == EXPECTED_STEPS
    epoch_counts = Counter(int(row["epoch"]) for row in records)
    for index, row in enumerate(records, start=1):
        sequential &= (
            int(row["seed"]) == seed
            and int(row["step"]) == index
            and int(row["epoch"]) == (index - 1) // STEPS_PER_EPOCH + 1
            and row["scope"] == "full"
            and row["finite"] is True
            and all(math.isfinite(float(row[field])) for field in finite_fields)
        )
    means_match = all(
        close(
            statistics.fmean(float(row["total_loss"]) for row in records if int(row["epoch"]) == epoch["epoch"]),
            epoch["train_total_loss"],
        )
        for epoch in epoch_rows
    )
    return {
        "path": rel(path),
        "sha256": sha(path),
        "records": len(records),
        "sequential_finite_exact_seed_scope": sequential,
        "exact_200_steps_each_epoch": epoch_counts == Counter({epoch: 200 for epoch in range(1, 21)}),
        "epoch_means_match": means_match,
        "passed": sequential and means_match and epoch_counts == Counter({epoch: 200 for epoch in range(1, 21)}),
    }


def audit_seed(seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = SEED_ROOTS[seed]
    epoch_paths = sorted((root / "metrics").glob("development_epoch_*.json"))
    epochs = [epoch_row(seed, path) for path in epoch_paths]
    log = audit_log(seed, root / "logs/development_metrics.jsonl", epochs)
    checkpoint_root = root / "checkpoints/development"
    manager_audit = audit_checkpoint_root(checkpoint_root)
    inventory = []
    crosslinks = []
    by_epoch = {row["epoch"]: row for row in epochs}
    for checkpoint_id in manager_audit["valid_checkpoints"]:
        directory = checkpoint_root / checkpoint_id
        metadata = read(directory / "metadata.json")
        manifest = read(directory / "manifest.json")
        epoch = int(metadata["epoch"])
        row = by_epoch[epoch]
        crosslinks.append(
            metadata["global_step"] == row["global_step"]
            and metadata["runtime"]["seed"] == seed
            and metadata["selection"]["eligible_for_best"] is False
            and close(metadata["selection"]["value"], row["C"])
            and close(metadata["metrics"]["dev_clean_found_raw_point_in_target"], row["Gc"])
            and close(metadata["metrics"]["dev_all_found_raw_point_in_target"], row["Ga"])
            and close(metadata["metrics"]["dev_answerability_macro_f1"], row["A"])
            and close(metadata["metrics"]["dev_false_found_rate"], row["F"])
        )
        files = {entry["path"]: entry for entry in manifest["files"]}
        inventory.append(
            {
                "seed": seed,
                "epoch": epoch,
                "global_step": int(metadata["global_step"]),
                "checkpoint": rel(directory),
                "manifest_sha256": sha(directory / "manifest.json"),
                "model_sha256": files["model.safetensors"]["sha256"],
                "training_state_sha256": files["training_state.pt"]["sha256"],
                "metadata_sha256": files["metadata.json"]["sha256"],
                "selection_value": metadata["selection"]["value"],
                "eligible_for_best": metadata["selection"]["eligible_for_best"],
            }
        )
    report_path = root / "PCRA_U_DEVELOPMENT_V1_1_SEED_REPORT.json"
    seed_report = read(report_path)
    campaign_copy = read(CAMPAIGN / f"seed_reports/seed_{seed}.json")
    predictions = list((root / "predictions").glob("*"))
    best_pointers = list(checkpoint_root.glob("best_*.json"))
    best = min(epochs, key=lambda row: (-row["C"], row["F"], -row["Gc"], row["epoch"]))
    best_inventory = next(item for item in inventory if item["epoch"] == best["epoch"])
    best_observed = {
        **{key: best[key] for key in ("seed", "epoch", "global_step", "Gc", "Ga", "A", "F", "S", "C")},
        "checkpoint": best_inventory["checkpoint"],
        "model_sha256": best_inventory["model_sha256"],
        "eligible": False,
        "status": "BEST_OBSERVED_INELIGIBLE_DIAGNOSTIC_ONLY_NOT_PROMOTABLE",
        "failed_checks": [
            key.removeprefix("check_") for key, value in best.items() if key.startswith("check_") and not value
        ],
    }
    passed = (
        len(epochs) == EXPECTED_EPOCHS
        and [row["epoch"] for row in epochs] == list(range(1, 21))
        and all(row["epoch_artifact_valid"] for row in epochs)
        and not any(row["eligible"] for row in epochs)
        and log["passed"]
        and manager_audit["passed"]
        and len(inventory) == EXPECTED_EPOCHS
        and all(crosslinks)
        and manager_audit["pointers"] == {"last.json": {"target": "step_000004000", "valid": True}}
        and not best_pointers
        and not predictions
        and seed_report["selection"]["eligible"] is False
        and seed_report["selected_checkpoint"] is None
        and seed_report["selected_checkpoint_dev_metrics"] is None
        and seed_report["observed"]["epochs_completed"] == 20
        and seed_report["observed"]["optimizer_steps"] == 4000
        and canonical(seed_report) == canonical(campaign_copy)
    )
    return (
        {
            "seed": seed,
            "passed": passed,
            "epoch_count": len(epochs),
            "optimizer_steps": log["records"],
            "eligible_epoch_count": sum(row["eligible"] for row in epochs),
            "checkpoint_count": len(inventory),
            "checkpoint_audit": manager_audit,
            "checkpoint_epoch_crosslinks_all_pass": all(crosslinks),
            "log_audit": log,
            "predictions_present": bool(predictions),
            "best_pointer_present": bool(best_pointers),
            "seed_report_path": rel(report_path),
            "seed_report_sha256": sha(report_path),
            "best_observed_ineligible": best_observed,
        },
        epochs,
        inventory,
    )


def baseline_audit() -> dict[str, Any]:
    report = read(V1_REPORT)
    verification = verify_checkpoint(V1_CHECKPOINT)
    metrics = {
        "Gc": 17 / 32,
        "Ga": 144 / 168,
        "A": 0.7420238435082854,
        "F": 9 / 95,
        "S": 0.5876288659793815,
    }
    metrics["C"] = score(metrics["Gc"], metrics["Ga"], metrics["A"], metrics["F"])
    passed = (
        sha(V1_CHECKPOINT / "model.safetensors") == V1_MODEL_SHA
        and sha(V1_PREDICTIONS) == V1_PREDICTIONS_SHA
        and sha(V1_REPORT) == V1_REPORT_SHA
        and verification["passed"]
        and report["selected_checkpoint_model_sha256"] == V1_MODEL_SHA
        and report["selected_checkpoint"] == rel(V1_CHECKPOINT)
        and report["test_opened"] is False
        and report["calibration_fit"] is False
    )
    return {
        "passed": passed,
        "checkpoint": rel(V1_CHECKPOINT),
        "model_sha256": V1_MODEL_SHA,
        "predictions": rel(V1_PREDICTIONS),
        "predictions_sha256": V1_PREDICTIONS_SHA,
        "report": rel(V1_REPORT),
        "report_sha256": V1_REPORT_SHA,
        "checkpoint_verification": verification,
        "metrics": metrics,
    }


def table_error_audit(campaign: Mapping[str, Any]) -> dict[str, Any]:
    source_path = CAMPAIGN / "locked_inputs/pcra_u_development_v1_1_train.py"
    source = source_path.read_text(encoding="utf-8")
    report_write = source.index('write_json(campaign_root / "PCRA_U_DEVELOPMENT_V1_1_CAMPAIGN_REPORT.json"')
    metrics_assign = source.index('metrics = seed_report["selected_checkpoint_dev_metrics"]', report_write)
    failing_read = source.index('"Gc": metrics["clean_found"]["raw_point_in_target"]', metrics_assign)
    table_write = source.index('write_csv(campaign_root / "tables/three_seed_selected_checkpoints.csv"', failing_read)
    all_null = all(item["selected_checkpoint_dev_metrics"] is None for item in campaign["seed_reports"])
    table_path = CAMPAIGN / "tables/three_seed_selected_checkpoints.csv"
    passed = report_write < metrics_assign < failing_read < table_write and all_null and not table_path.exists()
    return {
        "passed": passed,
        "exception_type": "TypeError",
        "exception_message": "'NoneType' object is not subscriptable",
        "evidence_basis": "reconstructed_from_persisted_null_seed_metrics_and_locked_source_order",
        "cause": "table-only summary loop subscripts selected_checkpoint_dev_metrics after all three values are null",
        "campaign_report_written_before_failing_table_loop": report_write < metrics_assign,
        "missing_table": rel(table_path),
        "missing_table_exists": table_path.exists(),
        "effect_on_training_or_checkpoints": "NONE; loop runs after all seed training, seed reports, campaign report and gate writes",
        "locked_source": rel(source_path),
        "locked_source_sha256": sha(source_path),
    }


def svg_text(x: float, y: float, value: str, size: int = 12, anchor: str = "start", bold: bool = False) -> str:
    weight = "bold" if bold else "normal"
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="sans-serif" font-size="{size}" font-weight="{weight}" fill="#222">{html.escape(value)}</text>'


def svg_wrap(width: int, height: int, body: Sequence[str]) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n<rect width="100%" height="100%" fill="white"/>\n' + "\n".join(body) + "\n</svg>\n"


def trajectory_svg(rows: Sequence[Mapping[str, Any]], best: Sequence[Mapping[str, Any]], v1: Mapping[str, float]) -> str:
    width, height = 1120, 1030
    left, right = 90, 1080
    colors = {24082026: "#1f77b4", 24082027: "#d95f02", 24082028: "#2b8c4b"}
    panels = (("C", "Composite C", v1["C"]), ("Gc", "Clean FOUND Gc", v1["Gc"]), ("Ga", "Raw all-FOUND Ga", ELIGIBILITY["min_Ga"]), ("A", "Answerability macro-F1 A", ELIGIBILITY["min_A"]), ("F", "False-FOUND F", ELIGIBILITY["max_F"]))
    body = [svg_text(left, 30, "V1.1 measured DEV trajectories — all 60 epochs", 20, bold=True), svg_text(left, 53, "Every checkpoint was ineligible; lines connect measured epoch JSON values only", 12)]
    by_seed = {seed: sorted([row for row in rows if row["seed"] == seed], key=lambda row: row["epoch"]) for seed in SEEDS}
    best_key = {(row["seed"], row["epoch"]) for row in best}
    for panel_index, (field, title, reference) in enumerate(panels):
        top, bottom = 82 + 180 * panel_index, 222 + 180 * panel_index
        values = [float(row[field]) for row in rows] + [float(reference)]
        low, high = min(values), max(values)
        pad = max((high - low) * 0.08, 0.005)
        low, high = low - pad, high + pad
        xof = lambda epoch: left + (epoch - 1) / 19 * (right - left)
        yof = lambda value: bottom - (value - low) / (high - low) * (bottom - top)
        body += [svg_text(left, top - 8, title, 13, bold=True), f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#777"/>', f'<line x1="{left}" y1="{yof(reference):.1f}" x2="{right}" y2="{yof(reference):.1f}" stroke="#b44" stroke-dasharray="5,4"/>', svg_text(right - 4, yof(reference) - 4, f"ref {reference:.4f}", 10, "end")]
        for seed in SEEDS:
            points = " ".join(f'{xof(row["epoch"]):.1f},{yof(row[field]):.1f}' for row in by_seed[seed])
            body.append(f'<polyline points="{points}" fill="none" stroke="{colors[seed]}" stroke-width="2"/>')
            for row in by_seed[seed]:
                if (seed, row["epoch"]) in best_key:
                    body.append(f'<circle cx="{xof(row["epoch"]):.1f}" cy="{yof(row[field]):.1f}" r="5" fill="white" stroke="{colors[seed]}" stroke-width="3"/>')
    body.append(svg_text((left + right) / 2, height - 15, "Epoch 1–20; outlined points = best observed ineligible per seed", 12, "middle"))
    return svg_wrap(width, height, body)


def comparison_svg(best: Sequence[Mapping[str, Any]], v1: Mapping[str, float]) -> str:
    width, height = 1120, 620
    left, right, top, bottom = 80, 1080, 90, 525
    fields = ("Gc", "Ga", "A", "one_minus_F", "S", "C")
    labels = ("Gc", "Ga", "A", "1-F", "S", "C")
    series = [("V1", "#666", v1)] + [(str(row["seed"]), color, row) for row, color in zip(best, ("#1f77b4", "#d95f02", "#2b8c4b"))]
    body = [svg_text(left, 30, "V1 versus best observed ineligible V1.1 checkpoints", 20, bold=True), svg_text(left, 53, "Diagnostic DEV comparison only; V1.1 bars are not promotion candidates", 12)]
    body += [f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#777"/>']
    group = (right - left) / len(fields)
    bar = group / 5
    for index, field in enumerate(fields):
        for series_index, (_, color, values) in enumerate(series):
            value = 1 - float(values["F"]) if field == "one_minus_F" else float(values[field])
            x = left + index * group + (series_index + 0.5) * bar
            y = bottom - value * (bottom - top)
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar*0.8:.1f}" height="{bottom-y:.1f}" fill="{color}"/>')
        body.append(svg_text(left + (index + 0.5) * group, bottom + 20, labels[index], 11, "middle"))
    for index, (name, color, _) in enumerate(series):
        x = left + 480 + index * 125
        body += [f'<rect x="{x}" y="68" width="14" height="10" fill="{color}"/>', svg_text(x + 19, 78, name, 11)]
    return svg_wrap(width, height, body)


def tradeoff_svg(rows: Sequence[Mapping[str, Any]], best: Sequence[Mapping[str, Any]]) -> str:
    width, height = 920, 660
    left, right, top, bottom = 90, 875, 75, 575
    colors = {24082026: "#1f77b4", 24082027: "#d95f02", 24082028: "#2b8c4b"}
    xof = lambda value: left + value / 0.32 * (right - left)
    yof = lambda value: bottom - (value - 0.20) / 0.36 * (bottom - top)
    best_keys = {(row["seed"], row["epoch"]) for row in best}
    body = [svg_text(left, 30, "Clean localization versus false-FOUND trade-off", 20, bold=True), svg_text(left, 53, "60 measured epoch checkpoints; lower F and higher Gc are preferable", 12), f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#777"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#777"/>', f'<line x1="{xof(ELIGIBILITY["max_F"]):.1f}" y1="{top}" x2="{xof(ELIGIBILITY["max_F"]):.1f}" y2="{bottom}" stroke="#b44" stroke-dasharray="5,4"/>', f'<line x1="{left}" y1="{yof(0.53125):.1f}" x2="{right}" y2="{yof(0.53125):.1f}" stroke="#777" stroke-dasharray="5,4"/>']
    for row in rows:
        radius = 6 if (row["seed"], row["epoch"]) in best_keys else 3
        body.append(f'<circle cx="{xof(row["F"]):.1f}" cy="{yof(row["Gc"]):.1f}" r="{radius}" fill="{colors[row["seed"]]}" fill-opacity="0.75"/>')
    body += [svg_text((left + right) / 2, height - 25, "False-FOUND F", 13, "middle"), svg_text(18, (top + bottom) / 2, "Gc", 13), svg_text(xof(ELIGIBILITY["max_F"]) + 4, top + 15, "F max", 10), svg_text(right - 4, yof(0.53125) - 4, "V1 Gc", 10, "end")]
    return svg_wrap(width, height, body)


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P-CRA-U V1.1 failure audit",
        "",
        f"- Audit: `{'PASS' if report['passed'] else 'FAIL'}`.",
        f"- Decision: `{report['decision']}`.",
        "- Completed evidence: 3 fresh seeds × 20 epochs = 60 DEV checkpoints; 12,000 optimizer steps total.",
        "- Eligible checkpoints: `0/60`. No V1.1 prediction replay or bootstrap was authorized.",
        "- Retained model: locked V1 epoch 10 / step 2000.",
        "- Calibration and Test remained sealed.",
        "",
        "## Best observed ineligible checkpoints (diagnostic only)",
        "",
        "| Seed | Epoch | C | Gc | Ga | A | F | Failed eligibility checks |",
        "|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for item in report["best_observed_ineligible_by_seed"]:
        lines.append(f"| {item['seed']} | {item['epoch']} | {item['C']:.6f} | {item['Gc']:.6f} | {item['Ga']:.6f} | {item['A']:.6f} | {item['F']:.6f} | {', '.join(item['failed_checks'])} |")
    lines += [
        "",
        "These checkpoints are ranked with the locked rule `max C → min F → max Gc → earliest epoch`; the ranking is descriptive and creates no best pointer or promotion authority.",
        "",
        "## Post-report TypeError",
        "",
        "The persisted campaign report and locked source show that all three `selected_checkpoint_dev_metrics` values were null. After writing the campaign report and campaign gate, the table-only loop attempted `metrics[\"clean_found\"]`, yielding `TypeError: 'NoneType' object is not subscriptable`. This occurred after training, seed reports and all 60 immutable checkpoints were written. It therefore changed no weight, metric, checkpoint, eligibility result, or scientific decision; only `three_seed_selected_checkpoints.csv` was not created.",
        "",
        "The seed-report `exact_locked_dev_denominators=false` flag was mechanically caused by absent selected metrics. Direct inspection of every epoch JSON confirms exact denominators `400/80/168/32/95` for samples/families/all-FOUND/clean-FOUND/false-FOUND-risk.",
        "",
        "## Integrity gates",
        "",
    ]
    lines += [f"- `{name}`: `{'PASS' if value else 'FAIL'}`" for name, value in report["gates"].items()]
    lines += [
        "",
        "## Scientific boundary",
        "",
        "V1.1 is rejected under the precommitted development policy because no epoch jointly met Ga, macro-F1 and false-FOUND eligibility. The audit does not reinterpret an ineligible checkpoint, run post-hoc inference, or open Calibration/Test. A future V1.2 experiment requires a new precommitted contract.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if OUTPUT.exists() or DECISION_LOCK.exists():
        raise AuditError("Refusing to overwrite failure-audit output or decision lock")
    campaign_report_path = CAMPAIGN / "PCRA_U_DEVELOPMENT_V1_1_CAMPAIGN_REPORT.json"
    campaign_gate_path = CAMPAIGN / "checks/campaign_gate.json"
    campaign = read(campaign_report_path)
    critical = [
        campaign_report_path,
        campaign_gate_path,
        V1_REPORT,
        V1_PREDICTIONS,
        V1_CHECKPOINT / "model.safetensors",
        PROTOCOL / "pcra_u_development_v1_1_train.py",
        PROTOCOL / "pcra_u_development_v1_1_common.py",
        PROTOCOL / "pcra_u_development_v1_1_manifest.json",
        PROTOCOL / "pcra_u_development_v1_1_config.json",
        PROTOCOL / "pcra_u_development_v1_1_preflight_lock.json",
    ]
    before = snapshot(critical)
    seed_audits, epochs, inventory = [], [], []
    for seed in SEEDS:
        audit, seed_epochs, seed_inventory = audit_seed(seed)
        seed_audits.append(audit)
        epochs.extend(seed_epochs)
        inventory.extend(seed_inventory)
    baseline = baseline_audit()
    type_error = table_error_audit(campaign)
    best = [item["best_observed_ineligible"] for item in seed_audits]
    campaign_valid = (
        canonical(campaign) == canonical(read(campaign_gate_path))
        and campaign["passed"] is False
        and campaign["decision"] == "V1_1_CAMPAIGN_FAILED_GATES"
        and campaign["campaign_winner"] is None
        and campaign["aggregate_selected_checkpoint_metrics"] == {}
        and campaign["protected_before"] == campaign["protected_after"]
        and campaign["calibration_fit"] is False
        and campaign["test_opened"] is False
        and len(campaign["seed_reports"]) == 3
    )
    locked_files_match = all(
        sha(PROTOCOL / name) == sha(CAMPAIGN / "locked_inputs" / name)
        for name in (
            "pcra_u_development_v1_1_train.py",
            "pcra_u_development_v1_1_common.py",
            "pcra_u_development_v1_1_manifest.json",
            "pcra_u_development_v1_1_config.json",
            "pcra_u_development_v1_1_preflight_lock.json",
        )
    )
    after = snapshot(critical)
    gates = {
        "exact_three_seeds_twenty_epochs_sixty_epoch_json": len(seed_audits) == 3 and len(epochs) == 60,
        "all_sixty_epoch_artifacts_valid_exact_denominators": all(row["epoch_artifact_valid"] for row in epochs),
        "zero_of_sixty_checkpoints_eligible": sum(row["eligible"] for row in epochs) == 0,
        "all_twelve_thousand_optimizer_steps_finite_and_sequential": all(item["log_audit"]["passed"] for item in seed_audits),
        "all_sixty_checkpoint_manifests_and_hashes_manager_audit_pass": len(inventory) == 60 and all(item["checkpoint_audit"]["passed"] for item in seed_audits),
        "all_checkpoint_epoch_metric_crosslinks_pass": all(item["checkpoint_epoch_crosslinks_all_pass"] for item in seed_audits),
        "all_seed_reports_fail_only_without_eligible_selection": all(item["passed"] for item in seed_audits),
        "no_best_pointer_and_no_winner_predictions": all(not item["best_pointer_present"] and not item["predictions_present"] for item in seed_audits),
        "campaign_failure_report_and_protected_hashes_consistent": campaign_valid,
        "locked_live_inputs_match_campaign_snapshots": locked_files_match,
        "table_generation_typeerror_isolated_after_training_and_reports": type_error["passed"],
        "locked_v1_epoch10_step2000_unchanged_and_verified": baseline["passed"],
        "critical_inputs_unchanged_during_audit": before == after,
        "calibration_and_tests_sealed": True,
        "prediction_and_bootstrap_prerequisites_failed_so_not_run": True,
    }
    passed = all(gates.values())
    if not passed:
        failed = [name for name, value in gates.items() if not value]
        raise AuditError(f"Failure audit did not pass; no decision lock created: {failed}")
    inventory_digest = canonical(inventory)
    input_manifest = {
        "schema_version": 1,
        "campaign_report": {"path": rel(campaign_report_path), "sha256": sha(campaign_report_path)},
        "campaign_gate": {"path": rel(campaign_gate_path), "sha256": sha(campaign_gate_path)},
        "failure_audit_script": {"path": rel(Path(__file__)), "sha256": sha(Path(__file__))},
        "critical_input_hashes": before,
        "checkpoint_inventory_count": len(inventory),
        "checkpoint_inventory_canonical_sha256": inventory_digest,
    }
    report = {
        "schema_version": 1,
        "protocol_id": "pcra_u_development_v1_1_failure_audit",
        "generated_at_utc": now(),
        "scientific_status": SCIENTIFIC_STATUS,
        "passed": True,
        "decision": DECISION,
        "gates": gates,
        "observed": {
            "seeds": 3,
            "epochs_per_seed": 20,
            "epoch_artifacts": len(epochs),
            "optimizer_steps_total": sum(item["optimizer_steps"] for item in seed_audits),
            "checkpoint_count": len(inventory),
            "eligible_checkpoint_count": 0,
        },
        "eligibility": ELIGIBILITY,
        "score_formula": "C=0.45*Gc+0.20*Ga+0.20*A+0.15*(1-F)",
        "best_observed_ineligible_by_seed": best,
        "seed_audits": seed_audits,
        "retained_v1": baseline,
        "table_generation_typeerror": type_error,
        "prediction_generation_performed": False,
        "paired_family_bootstrap_performed": False,
        "not_run_reason": "zero eligible checkpoints; winner predictions and precommitted bootstrap prerequisites absent",
        "input_manifest": input_manifest,
        "checkpoint_inventory_canonical_sha256": inventory_digest,
        "calibration_fit": False,
        "test_iid_opened": False,
        "test_ood_opened": False,
    }
    comparison = []
    for item in best:
        comparison.append({
            **item,
            **{f"delta_{metric}_vs_v1": item[metric] - baseline["metrics"][metric] for metric in ("Gc", "Ga", "A", "F", "S", "C")},
        })
    failure_counts = []
    for seed in SEEDS:
        subset = [row for row in epochs if row["seed"] == seed]
        failure_counts.append({
            "seed": seed,
            "epochs": 20,
            "eligible": 0,
            "fail_min_epoch": sum(not row["check_epoch_at_least_minimum"] for row in subset),
            "fail_Ga": sum(not row["check_all_found_grounding_at_least_minimum"] for row in subset),
            "fail_A": sum(not row["check_answerability_macro_f1_at_least_minimum"] for row in subset),
            "fail_F": sum(not row["check_false_found_rate_at_most_maximum"] for row in subset),
        })

    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name in ("checks", "figures", "manifests", "tables"):
        (OUTPUT / name).mkdir()
    write_json(OUTPUT / "manifests/input_manifest.json", input_manifest)
    write_json(OUTPUT / "checks/failure_audit_gate.json", {"passed": True, "decision": DECISION, "gates": gates})
    write_json(OUTPUT / "PCRA_U_DEVELOPMENT_V1_1_FAILURE_AUDIT_REPORT.json", report)
    (OUTPUT / "PCRA_U_DEVELOPMENT_V1_1_FAILURE_AUDIT_REPORT.md").write_text(markdown(report), encoding="utf-8")
    write_csv(OUTPUT / "tables/epoch_trajectory_all_60.csv", list(epochs[0]), epochs)
    write_csv(OUTPUT / "tables/checkpoint_inventory_all_60.csv", list(inventory[0]), inventory)
    write_csv(OUTPUT / "tables/best_observed_ineligible_by_seed.csv", list(best[0]), best)
    write_csv(OUTPUT / "tables/v1_comparison_diagnostic.csv", list(comparison[0]), comparison)
    write_csv(OUTPUT / "tables/eligibility_failure_counts.csv", list(failure_counts[0]), failure_counts)
    (OUTPUT / "figures/F01_v1_1_all_60_epoch_trajectories.svg").write_text(trajectory_svg(epochs, best, baseline["metrics"]), encoding="utf-8")
    (OUTPUT / "figures/F02_v1_vs_best_observed_ineligible.svg").write_text(comparison_svg(best, baseline["metrics"]), encoding="utf-8")
    (OUTPUT / "figures/F03_gc_false_found_tradeoff_all_60.svg").write_text(tradeoff_svg(epochs, best), encoding="utf-8")

    report_path = OUTPUT / "PCRA_U_DEVELOPMENT_V1_1_FAILURE_AUDIT_REPORT.json"
    decision_lock = {
        "schema_version": 1,
        "protocol_id": "pcra_u_development_v1_1_decision_lock",
        "status": "LOCKED_AFTER_FAILURE_AUDIT_PASS",
        "created_at_utc": now(),
        "decision": DECISION,
        "retained_model": {
            "name": "P-CRA-U-development-v1",
            "checkpoint": rel(V1_CHECKPOINT),
            "epoch": 10,
            "global_step": 2000,
            "model_sha256": V1_MODEL_SHA,
        },
        "rejected_campaign": {
            "path": rel(CAMPAIGN),
            "report_sha256": sha(campaign_report_path),
            "checkpoint_count": 60,
            "eligible_checkpoint_count": 0,
            "checkpoint_inventory_canonical_sha256": inventory_digest,
            "best_observed_ineligible_diagnostic_only": best,
        },
        "failure_audit": {"path": rel(report_path), "sha256": sha(report_path), "passed": True},
        "prohibitions": [
            "do_not_promote_any_v1_1_ineligible_checkpoint",
            "do_not_run_posthoc_v1_1_prediction_or_bootstrap_under_this_campaign",
            "do_not_open_calibration_test_iid_or_test_ood",
        ],
        "table_generation_typeerror_scientific_effect": "NONE",
        "calibration_fit": False,
        "test_opened": False,
    }
    write_json(DECISION_LOCK, decision_lock, exclusive=True)
    write_json(OUTPUT / "checks/decision_lock_binding.json", {"path": rel(DECISION_LOCK), "sha256": sha(DECISION_LOCK), "decision": DECISION})
    print(json.dumps({"passed": True, "decision": DECISION, "output": rel(OUTPUT), "decision_lock": rel(DECISION_LOCK)}, sort_keys=True))


if __name__ == "__main__":
    main()
