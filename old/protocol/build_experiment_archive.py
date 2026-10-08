#!/usr/bin/env python3
"""Build a lightweight, auditable report bundle for one WP3 smoke run.

The bundle contains only derived tables, SVG charts, JPEG prediction overlays,
normalized metrics and hashes. Large feature tensors remain in the source run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2


WORKSPACE = Path(__file__).resolve().parents[1]
SMOKE_MANIFEST = WORKSPACE / "protocol/wp3_smoke_manifest.json"
SELECTION_AUDIT = WORKSPACE / "protocol/wp3_smoke_selection_audit.json"
ARCHIVE_SCHEMA_VERSION = 1
COLORS = {
    "green": "#16a34a",
    "red": "#dc2626",
    "amber": "#d97706",
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "slate": "#475569",
    "light": "#e2e8f0",
    "ink": "#0f172a",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def svg_text(x: float, y: float, text: Any, size: int = 13, weight: int = 400,
             anchor: str = "start", fill: str = COLORS["ink"]) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="DejaVu Sans,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{html.escape(str(text))}</text>'
    )


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def gate_chart(path: Path, gates: dict[str, bool], status: str) -> None:
    rows = list(gates.items()) or [("run_completed_before_comparison", False)]
    height = 90 + len(rows) * 28
    body = [svg_text(28, 34, "WP3 feature-gate status", 21, 700),
            svg_text(28, 58, status, 13, 600, fill=COLORS["slate"])]
    for idx, (name, passed) in enumerate(rows):
        y = 84 + idx * 28
        color = COLORS["green"] if passed else COLORS["red"]
        body += [f'<rect x="28" y="{y - 14}" width="54" height="20" rx="4" fill="{color}"/>',
                 svg_text(55, y + 1, "PASS" if passed else "FAIL", 11, 700, "middle", "#ffffff"),
                 svg_text(96, y + 1, name, 12)]
    write_svg(path, 900, height, body)


def bar_chart(path: Path, title: str, labels: list[str], values: list[float],
              unit: str, color: str = COLORS["blue"]) -> None:
    width, height = 900, max(260, 105 + 44 * len(labels))
    left, right = 245, 105
    chart_width = width - left - right
    maximum = max(values or [1.0]) or 1.0
    body = [svg_text(28, 35, title, 21, 700)]
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = 78 + idx * 44
        bar_width = chart_width * value / maximum
        body += [svg_text(left - 12, y + 15, label, 12, anchor="end"),
                 f'<rect x="{left}" y="{y}" width="{chart_width}" height="22" rx="3" fill="{COLORS["light"]}"/>',
                 f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="22" rx="3" fill="{color}"/>',
                 svg_text(left + bar_width + 8, y + 16, f"{value:.3f} {unit}", 12, 600)]
    write_svg(path, width, height, body)


def determinism_chart(path: Path, counts: dict[str, int]) -> None:
    total_samples = int(counts.get("sample_count", 0))
    tensor_total = int(counts.get("tensor_pairs", 0))
    rows = [
        ("Exact R0/D0 tensors", int(counts.get("exact_tensor_pairs", 0)), tensor_total),
        ("Equal tensor-file hashes", int(counts.get("matching_tensor_file_hashes", 0)), total_samples),
        ("Equal raw answers", int(counts.get("matching_baseline_answers", 0)), total_samples),
        ("Equal parsed points", int(counts.get("matching_parsed_predictions", 0)), total_samples),
    ]
    body = [svg_text(28, 35, "Determinism and hook non-interference", 21, 700)]
    for idx, (label, value, total) in enumerate(rows):
        y = 78 + idx * 49
        ratio = (value / total) if total else 0.0
        color = COLORS["green"] if total and value == total else COLORS["red"]
        body += [svg_text(245, y + 16, label, 12, anchor="end"),
                 f'<rect x="260" y="{y}" width="500" height="24" rx="3" fill="{COLORS["light"]}"/>',
                 f'<rect x="260" y="{y}" width="{500 * ratio:.2f}" height="24" rx="3" fill="{color}"/>',
                 svg_text(775, y + 17, f"{value}/{total}", 12, 700)]
    write_svg(path, 900, 300, body)


def read_records(source: Path, mode: str) -> dict[str, dict[str, Any]]:
    folder = source / mode / ("records" if mode == "baseline" else "cache")
    if not folder.exists():
        return {}
    return {path.stem: load_json(path) for path in sorted(folder.glob("*.json"))}


def verify_source_manifest(source: Path) -> dict[str, Any]:
    manifest_path = source / "artifact_manifest.json"
    if not manifest_path.exists():
        return {
            "manifest_present": False,
            "declared_artifact_count": 0,
            "checked_artifact_count": 0,
            "missing": [],
            "size_mismatches": [],
            "hash_mismatches": [],
            "passed": True,
            "note": "No source artifact manifest; source files are inventoried only.",
        }
    manifest = load_json(manifest_path)
    missing, size_mismatches, hash_mismatches = [], [], []
    artifacts = manifest.get("artifacts", [])
    for item in artifacts:
        candidate = source / item["path"]
        if not candidate.is_file():
            missing.append(item["path"])
            continue
        if candidate.stat().st_size != item.get("bytes"):
            size_mismatches.append(item["path"])
        if sha256(candidate) != item.get("sha256"):
            hash_mismatches.append(item["path"])
    return {
        "manifest_present": True,
        "manifest_sha256": sha256(manifest_path),
        "declared_artifact_count": len(artifacts),
        "checked_artifact_count": len(artifacts),
        "missing": missing,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "passed": not (missing or size_mismatches or hash_mismatches),
    }


def make_overlays(run_dir: Path, baseline: dict[str, dict[str, Any]],
                  manifest: dict[str, Any], selection: dict[str, Any]) -> int:
    dataset_root = WORKSPACE / manifest["dataset_root"]
    inputs = {item["sample_id"]: item for item in manifest["entries"]}
    audit = {item["sample_id"]: item for item in selection["entries"]}
    overlay_dir = run_dir / "images/overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    thumbnails = []
    for sample_id, record in sorted(baseline.items()):
        item = inputs.get(sample_id)
        if not item:
            continue
        image = cv2.imread(str(dataset_root / item["rgb_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        banner_h = 48
        canvas = cv2.copyMakeBorder(image, banner_h, 0, 0, 0, cv2.BORDER_CONSTANT,
                                    value=(20, 27, 40))
        state = audit.get(sample_id, {}).get("answerability_state", "UNKNOWN")
        label = f"{sample_id} | evaluator: {state} | baseline prediction"
        cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (245, 245, 245), 1, cv2.LINE_AA)
        for point in record.get("normalized_points_xy", []):
            if len(point) != 2:
                continue
            x = int(round(float(point[0]) * (width - 1)))
            y = int(round(float(point[1]) * (height - 1))) + banner_h
            cv2.circle(canvas, (x, y), 13, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 8, (20, 20, 230), -1, cv2.LINE_AA)
            cv2.line(canvas, (x - 18, y), (x + 18, y), (20, 20, 230), 2, cv2.LINE_AA)
            cv2.line(canvas, (x, y - 18), (x, y + 18), (20, 20, 230), 2, cv2.LINE_AA)
        output = overlay_dir / f"{sample_id}.jpg"
        cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        thumb = cv2.resize(canvas, (320, 264), interpolation=cv2.INTER_AREA)
        thumbnails.append(thumb)
    if thumbnails:
        while len(thumbnails) % 5:
            thumbnails.append(255 * thumbnails[0] * 0 + 255)
        rows = [cv2.hconcat(thumbnails[idx:idx + 5]) for idx in range(0, len(thumbnails), 5)]
        sheet = cv2.vconcat(rows)
        cv2.imwrite(str(run_dir / "images/prediction_contact_sheet.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    return len(list(overlay_dir.glob("*.jpg")))


def relative_link(source: Path, target_parent: Path) -> str:
    return str(source.resolve().relative_to(WORKSPACE)).replace(" ", "%20") if source.is_relative_to(WORKSPACE) else str(source)


def build_run(args: argparse.Namespace) -> dict[str, Any]:
    source = (WORKSPACE / args.source).resolve() if not Path(args.source).is_absolute() else Path(args.source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source run does not exist: {source}")
    archive_root = (WORKSPACE / args.archive_root).resolve()
    run_id = args.run_id or source.name
    run_dir = archive_root / "runs" / run_id
    if run_dir.exists() and not args.replace:
        raise FileExistsError(f"Archive run already exists: {run_dir}; pass --replace to rebuild")
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ["metrics", "tables", "figures", "images", "manifests", "checks"]:
        (run_dir / child).mkdir(exist_ok=True)

    comparison_path = source / "comparison_summary.json"
    comparison = load_json(comparison_path) if comparison_path.exists() else {}
    baseline = read_records(source, "baseline")
    hook_a = read_records(source, "hook_a")
    hook_b = read_records(source, "hook_b")
    smoke = load_json(SMOKE_MANIFEST)
    selection = load_json(SELECTION_AUDIT)
    selection_by_id = {item["sample_id"]: item for item in selection["entries"]}
    gates = comparison.get("gates", {})
    counts = comparison.get("counts", {})
    observed = comparison.get("observed", {})
    decision = args.decision or comparison.get("decision", "INCOMPLETE_NO_COMPARISON")

    metrics = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "run_id": run_id,
        "role": args.role,
        "status": args.status,
        "decision": decision,
        "failure_reason": args.failure_reason,
        "sample_count": len(baseline),
        "gates": gates,
        "counts": counts,
        "observed": observed,
        "safety": comparison.get("safety", {}),
        "coverage": selection.get("coverage", {}),
    }
    write_json(run_dir / "metrics/run_metrics.json", metrics)
    if comparison:
        write_json(run_dir / "metrics/source_comparison_summary.json", comparison)

    gate_rows = [{"gate": name, "passed": value, "result": "PASS" if value else "FAIL"}
                 for name, value in gates.items()]
    if not gate_rows:
        gate_rows = [{"gate": "comparison_not_reached", "passed": False, "result": "NOT_RUN"}]
    write_csv(run_dir / "tables/gates.csv", ["gate", "passed", "result"], gate_rows)

    sample_ids = sorted(set(baseline) | set(hook_a) | set(hook_b))
    runtime_rows = []
    for sample_id in sample_ids:
        row: dict[str, Any] = {"sample_id": sample_id}
        base = baseline.get(sample_id, {})
        row.update({
            "baseline_generation_ms": base.get("generation_latency_ms", ""),
            "baseline_total_ms": base.get("total_latency_ms", ""),
            "baseline_peak_vram_bytes": base.get("peak_vram_bytes", ""),
        })
        for mode, records in [("hook_a", hook_a), ("hook_b", hook_b)]:
            runtime = records.get(sample_id, {}).get("runtime", {})
            row.update({
                f"{mode}_preprocess_ms": runtime.get("preprocess_latency_ms", ""),
                f"{mode}_feature_ms": runtime.get("feature_latency_ms", ""),
                f"{mode}_generation_ms": runtime.get("generation_latency_ms", ""),
                f"{mode}_total_ms": runtime.get("total_latency_ms", ""),
                f"{mode}_peak_vram_bytes": runtime.get("peak_vram_bytes", ""),
            })
        runtime_rows.append(row)
    runtime_fields = ["sample_id", "baseline_generation_ms", "baseline_total_ms", "baseline_peak_vram_bytes",
                      "hook_a_preprocess_ms", "hook_a_feature_ms", "hook_a_generation_ms", "hook_a_total_ms",
                      "hook_a_peak_vram_bytes", "hook_b_preprocess_ms", "hook_b_feature_ms",
                      "hook_b_generation_ms", "hook_b_total_ms", "hook_b_peak_vram_bytes"]
    write_csv(run_dir / "tables/runtime_by_sample.csv", runtime_fields, runtime_rows)

    comparison_by_id = {row["sample_id"]: row for row in comparison.get("comparisons", [])}
    det_rows = []
    for sample_id in sample_ids:
        item = comparison_by_id.get(sample_id, {})
        tensor = item.get("tensor_comparison", {})
        det_rows.append({
            "sample_id": sample_id,
            "r0_exact": tensor.get("R0", {}).get("exact_equal", ""),
            "d0_exact": tensor.get("D0", {}).get("exact_equal", ""),
            "tensor_file_hash_equal": item.get("tensor_file_hash_equal", ""),
            "stable_cache_metadata_equal": item.get("stable_cache_metadata_equal", ""),
            "baseline_raw_answer_equal": item.get("baseline_raw_answer_equal", ""),
            "baseline_parsed_prediction_equal": item.get("baseline_parsed_prediction_equal", ""),
        })
    write_csv(run_dir / "tables/determinism_by_sample.csv", list(det_rows[0].keys()) if det_rows else ["sample_id"], det_rows)

    coverage_rows = []
    for sample_id in sample_ids:
        item = selection_by_id.get(sample_id, {})
        coverage_rows.append({"sample_id": sample_id, "answerability_state": item.get("answerability_state", ""),
                              "category": item.get("category", ""), "relation": item.get("relation", ""),
                              "anchor_count": item.get("anchor_count", "")})
    write_csv(run_dir / "tables/selection_coverage.csv",
              ["sample_id", "answerability_state", "category", "relation", "anchor_count"], coverage_rows)

    source_files = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        source_files.append({"path": str(path.relative_to(source)), "bytes": path.stat().st_size,
                             "sha256": "MANIFEST" if path.suffix == ".safetensors" else sha256(path)})
    write_csv(run_dir / "tables/source_artifacts.csv", ["path", "bytes", "sha256"], source_files)
    write_json(run_dir / "manifests/source_inventory.json", {
        "source_run": str(source.relative_to(WORKSPACE)), "file_count": len(source_files),
        "total_bytes": sum(item["bytes"] for item in source_files), "files": source_files,
        "note": "Large .safetensors files use MANIFEST here; authoritative hashes are checked from source artifact_manifest.json.",
    })

    gate_chart(run_dir / "figures/gate_status.svg", gates, args.status)
    runtime_labels, runtime_values = [], []
    for label, key in [("Feature (median)", "median_feature_latency_ms"),
                       ("Generation (median)", "median_generation_latency_ms")]:
        if key in observed:
            runtime_labels.append(label)
            runtime_values.append(float(observed[key]))
    if not runtime_values and baseline:
        runtime_labels = ["Baseline generation (median)"]
        runtime_values = [statistics.median(float(row.get("generation_latency_ms", 0)) for row in baseline.values())]
    bar_chart(run_dir / "figures/runtime_latency.svg", "Latency summary", runtime_labels, runtime_values, "ms")
    vram_bytes = float(observed.get("max_peak_vram_bytes", 0))
    if not vram_bytes and baseline:
        vram_bytes = max(float(row.get("peak_vram_bytes", 0)) for row in baseline.values())
    bar_chart(run_dir / "figures/peak_vram.svg", "Peak GPU memory", ["Maximum observed"],
              [vram_bytes / (1024 ** 3)], "GiB", COLORS["cyan"])
    determinism_chart(run_dir / "figures/determinism.svg", counts)
    states = selection.get("coverage", {}).get("answerability_states", {})
    bar_chart(run_dir / "figures/answerability_coverage.svg", "Smoke-set answerability coverage",
              list(states), [float(value) for value in states.values()], "samples", COLORS["amber"])

    overlay_count = make_overlays(run_dir, baseline, smoke, selection)
    source_check = verify_source_manifest(source)
    archive_check = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "source_manifest_check": source_check,
        "expected_baseline_samples": 15,
        "observed_baseline_samples": len(baseline),
        "baseline_sample_count_passed": len(baseline) == 15,
        "expected_overlays": len(baseline),
        "observed_overlays": overlay_count,
        "overlay_count_passed": overlay_count == len(baseline),
        "large_tensor_files_copied": 0,
        "passed": source_check["passed"] and overlay_count == len(baseline) and len(baseline) == 15,
    }
    write_json(run_dir / "checks/archive_check.json", archive_check)

    generated_at = datetime.now(timezone.utc).isoformat()
    run_record = {
        "schema_version": ARCHIVE_SCHEMA_VERSION, "run_id": run_id, "role": args.role,
        "status": args.status, "decision": decision, "failure_reason": args.failure_reason,
        "source_run": str(source.relative_to(WORKSPACE)), "sample_count": len(baseline),
        "generated_at_utc": generated_at, "archive_check_passed": archive_check["passed"],
        "large_tensor_files_copied": 0,
    }
    write_json(run_dir / "run_record.json", run_record)

    report = [
        f"# Run card — `{run_id}`", "", f"> **{args.role.upper()} / {args.status}**", "",
        f"- Source: [`{source.relative_to(WORKSPACE)}`](../../../../{source.relative_to(WORKSPACE)})",
        f"- Decision: `{decision}`", f"- Samples: {len(baseline)} train samples / 15 families",
        f"- Archive check: `{'PASS' if archive_check['passed'] else 'FAIL'}`",
        "- Training/dataset scaling in this report step: `NO / NO`",
        "- Tensor copied into archive: `0`", "",
    ]
    if args.failure_reason:
        report += ["## Failure diagnosis", "", args.failure_reason, ""]
    report += [
        "## Quick view", "", "![Prediction contact sheet](images/prediction_contact_sheet.jpg)", "",
        "![Gate status](figures/gate_status.svg)", "", "![Determinism](figures/determinism.svg)", "",
        "![Latency](figures/runtime_latency.svg)", "", "![Peak VRAM](figures/peak_vram.svg)", "",
        "![Coverage](figures/answerability_coverage.svg)", "", "## Data files", "",
        "- `metrics/run_metrics.json`: normalized run metrics;",
        "- `tables/runtime_by_sample.csv`: latency/VRAM for every sample and mode;",
        "- `tables/determinism_by_sample.csv`: A/B and baseline equality per sample;",
        "- `tables/gates.csv`: all gate outcomes;",
        "- `tables/selection_coverage.csv`: post-hoc evaluator coverage;",
        "- `checks/archive_check.json`: source and archive validation;",
        "- `manifests/archive_manifest.json`: SHA-256 for every derived artifact.", "",
        "## Interpretation", "",
    ]
    if args.role == "official" and args.status == "PASSED":
        report.append("This is the official feature-gate result. It permits only a tiny train-only P-CRA-U overfit smoke; it does not permit development training or dataset scaling.")
    else:
        report.append("This run is retained for diagnosis and must not be cited as the official WP3 feature-gate result.")
    (run_dir / "RUN_CARD.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest_entries = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "archive_manifest.json"):
        manifest_entries.append({"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size,
                                 "sha256": sha256(path)})
    write_json(run_dir / "manifests/archive_manifest.json", {
        "schema_version": ARCHIVE_SCHEMA_VERSION, "run_id": run_id,
        "generated_at_utc": generated_at, "artifact_count": len(manifest_entries),
        "artifacts": manifest_entries,
    })
    return run_record


def rebuild_index(archive_root: Path, latest_run: str | None) -> None:
    records = [load_json(path) for path in sorted((archive_root / "runs").glob("*/run_record.json"))]
    fields = ["run_id", "role", "status", "decision", "sample_count", "archive_check_passed",
              "large_tensor_files_copied", "generated_at_utc", "source_run", "failure_reason"]
    write_csv(archive_root / "RUN_INDEX.csv", fields, records)
    write_json(archive_root / "RUN_INDEX.json", {"schema_version": 1, "run_count": len(records), "runs": records})
    if latest_run:
        latest = next(record for record in records if record["run_id"] == latest_run)
        text = (f"# Latest official WP3 run\n\n"
                f"- Run: [`{latest_run}`](runs/{latest_run}/RUN_CARD.md)\n"
                f"- Status: `{latest['status']}`\n- Decision: `{latest['decision']}`\n"
                f"- Archive check: `{'PASS' if latest['archive_check_passed'] else 'FAIL'}`\n")
        (archive_root / "LATEST.md").write_text(text, encoding="utf-8")
    top_entries = []
    excluded = {"ARCHIVE_MANIFEST.json", "ARCHIVE_CHECK_REPORT.json"}
    for path in sorted(p for p in archive_root.rglob("*") if p.is_file() and p.name not in excluded):
        top_entries.append({"path": str(path.relative_to(archive_root)), "bytes": path.stat().st_size,
                            "sha256": sha256(path)})
    write_json(archive_root / "ARCHIVE_MANIFEST.json", {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(top_entries), "artifacts": top_entries,
    })
    top_bad = []
    for item in top_entries:
        candidate = archive_root / item["path"]
        if (not candidate.is_file() or candidate.stat().st_size != item["bytes"]
                or sha256(candidate) != item["sha256"]):
            top_bad.append(item["path"])
    run_checks = {}
    for record in records:
        check_path = archive_root / "runs" / record["run_id"] / "checks/archive_check.json"
        run_checks[record["run_id"]] = load_json(check_path).get("passed", False)
    official = [record for record in records if record["role"] == "official"]
    diagnostic = [record for record in records if record["role"] == "diagnostic"]
    check_report = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(records),
        "official_run_count": len(official),
        "diagnostic_run_count": len(diagnostic),
        "per_run_archive_checks": run_checks,
        "all_per_run_checks_passed": all(run_checks.values()),
        "top_manifest_artifact_count": len(top_entries),
        "top_manifest_mismatches": top_bad,
        "top_manifest_passed": not top_bad,
        "safetensors_copied_into_archive": len(list(archive_root.rglob("*.safetensors"))),
        "passed": (all(run_checks.values()) and not top_bad and len(official) == 1
                   and len(list(archive_root.rglob("*.safetensors"))) == 0),
        "interpretation": "Archive integrity PASS does not change a diagnostic run into a scientific gate PASS.",
    }
    write_json(archive_root / "ARCHIVE_CHECK_REPORT.json", check_report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source run relative to workspace")
    parser.add_argument("--archive-root", default="results/experiment_archive")
    parser.add_argument("--run-id")
    parser.add_argument("--role", choices=["official", "diagnostic"], required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--decision")
    parser.add_argument("--failure-reason")
    parser.add_argument("--latest", action="store_true", help="Mark this run as latest official")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build_run(args)
    archive_root = (WORKSPACE / args.archive_root).resolve()
    rebuild_index(archive_root, record["run_id"] if args.latest else None)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
