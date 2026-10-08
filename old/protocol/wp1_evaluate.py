#!/usr/bin/env python3
"""Post-lock evaluator, paired analysis, decision report and GUI for WP1."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml

from wp1_common import (
    CONDITIONS,
    CORRUPTED_DEPTH_CONDITIONS,
    PROTOCOL_ID,
    WP1Error,
    decision_from_metrics,
    pilot_tree_digest,
    read_json,
    sha256_file,
    validate_prediction_lock,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    return float(numerator) / float(denominator) if denominator else None


def quantile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def erode(mask: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return mask.astype(bool)
    kernel = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0


def task_group(task_type: str) -> str:
    if task_type == "direct_grounding":
        return "direct"
    if task_type in {
        "relative_2d",
        "depth_relation",
        "multi_reference",
        "metric_comparison",
        "shape_comparison",
        "occlusion",
    }:
        return "relation_or_comparison"
    return "selective"


def load_labels(path: Path) -> np.ndarray:
    labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise WP1Error(f"cannot read semantic labels: {path}")
    if labels.ndim == 3:
        if not np.array_equal(labels[..., 0], labels[..., 1]) or not np.array_equal(labels[..., 1], labels[..., 2]):
            raise WP1Error(f"semantic-label RGB channels disagree: {path}")
        labels = labels[..., 0]
    if labels.ndim != 2:
        raise WP1Error(f"semantic labels must be HxW: {path}")
    return labels


def evaluate_record(record: Mapping[str, Any], annotation: Mapping[str, Any], labels: np.ndarray, margin: int) -> dict:
    height, width = labels.shape
    if record["image_width"] != width or record["image_height"] != height:
        raise WP1Error(f"image/label dimension mismatch: {record['scene_id']}/{record['condition']}")
    strict = record.get("parse_status") == "EXACT_ONE_NORMALIZED_POINT"
    point = record.get("pixel_points_xy", [None])[0] if strict else None
    selected_label = None
    if point is not None:
        x_pixel, y_pixel = int(point[0]), int(point[1])
        selected_label = int(labels[y_pixel, x_pixel])
    state = str(annotation["target_state"])
    point_hit = None
    interior_hit = None
    plausible_hit = None
    if state == "single":
        target = np.isin(labels, np.asarray(annotation["target_semantic_labels"], dtype=labels.dtype))
        interior = erode(target, margin)
        point_hit = bool(point is not None and target[int(point[1]), int(point[0])])
        interior_hit = bool(point is not None and interior[int(point[1]), int(point[0])])
    elif state == "ambiguous":
        plausible = np.isin(labels, np.asarray(annotation["plausible_semantic_labels"], dtype=labels.dtype))
        plausible_hit = bool(point is not None and plausible[int(point[1]), int(point[0])])
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scene_id": record["scene_id"],
        "condition": record["condition"],
        "task_type": annotation["task_type"],
        "task_group": task_group(str(annotation["task_type"])),
        "target_state": state,
        "strict_point": strict,
        "normalized_point_xy": record.get("normalized_points_xy", [None])[0] if strict else None,
        "pixel_point_xy": point,
        "selected_semantic_label": selected_label,
        "point_hit": point_hit,
        "interior_eroded_hit": interior_hit,
        "plausible_point_hit": plausible_hit,
        "forced_point_on_absent_target": bool(state == "absent" and strict),
        "raw_answer": record.get("raw_answer"),
        "parse_status": record.get("parse_status"),
        "inference_latency_ms": record.get("inference_latency_ms"),
        "overlay_file": record.get("overlay_file"),
        "request_depth_file": record.get("request_depth_file"),
    }


def attach_pair_metrics(evaluations: list[dict]) -> None:
    by_key = {(item["scene_id"], item["condition"]): item for item in evaluations}
    for item in evaluations:
        correct = by_key[(item["scene_id"], "correct_depth")]
        rgb = by_key[(item["scene_id"], "rgb_only")]
        for name, reference in (("correct_depth", correct), ("rgb_only", rgb)):
            displacement_normalized = None
            displacement_px = None
            if item["normalized_point_xy"] is not None and reference["normalized_point_xy"] is not None:
                delta = np.asarray(item["normalized_point_xy"], dtype=np.float64) - np.asarray(reference["normalized_point_xy"], dtype=np.float64)
                displacement_normalized = float(np.linalg.norm(delta))
            if item["pixel_point_xy"] is not None and reference["pixel_point_xy"] is not None:
                delta_px = np.asarray(item["pixel_point_xy"], dtype=np.float64) - np.asarray(reference["pixel_point_xy"], dtype=np.float64)
                displacement_px = float(np.linalg.norm(delta_px))
            item[f"displacement_from_{name}_normalized"] = displacement_normalized
            item[f"displacement_from_{name}_px"] = displacement_px
        item["instance_switch_from_correct_depth"] = bool(
            item["selected_semantic_label"] is not None
            and correct["selected_semantic_label"] is not None
            and item["selected_semantic_label"] != correct["selected_semantic_label"]
        )
        item["answer_changed_from_correct_depth"] = item["raw_answer"] != correct["raw_answer"]


def summarize_subset(items: Sequence[Mapping[str, Any]]) -> dict:
    single = [item for item in items if item["target_state"] == "single"]
    latencies = [float(item["inference_latency_ms"]) for item in items if item["inference_latency_ms"] is not None]
    displacements = [
        float(item["displacement_from_correct_depth_normalized"])
        for item in items
        if item["displacement_from_correct_depth_normalized"] is not None
    ]
    strict_count = sum(item["strict_point"] for item in items)
    hit_count = sum(item["point_hit"] is True for item in single)
    interior_count = sum(item["interior_eroded_hit"] is True for item in single)
    return {
        "record_count": len(items),
        "strict_parse_count": strict_count,
        "strict_parse_rate": safe_rate(strict_count, len(items)),
        "single_target_count": len(single),
        "point_hit_count": hit_count,
        "point_hit_rate_single_targets": safe_rate(hit_count, len(single)),
        "interior_hit_count": interior_count,
        "interior_hit_rate_single_targets": safe_rate(interior_count, len(single)),
        "ambiguous_plausible_hit_count": sum(item["plausible_point_hit"] is True for item in items),
        "absent_forced_point_count": sum(item["forced_point_on_absent_target"] for item in items),
        "mean_latency_ms": float(statistics.fmean(latencies)) if latencies else None,
        "median_latency_ms": float(statistics.median(latencies)) if latencies else None,
        "mean_displacement_from_correct_normalized": float(statistics.fmean(displacements)) if displacements else None,
        "median_displacement_from_correct_normalized": float(statistics.median(displacements)) if displacements else None,
        "q90_displacement_from_correct_normalized": quantile(displacements, 0.90),
        "instance_switch_count": sum(item["instance_switch_from_correct_depth"] for item in items),
        "answer_changed_from_correct_count": sum(item["answer_changed_from_correct_depth"] for item in items),
    }


def paired_comparison(condition: str, evaluations: Sequence[Mapping[str, Any]], threshold: float) -> dict:
    by_key = {(item["scene_id"], item["condition"]): item for item in evaluations}
    items = [by_key[(scene_id, condition)] for scene_id in sorted({item["scene_id"] for item in evaluations})]
    correct_items = [by_key[(item["scene_id"], "correct_depth")] for item in items]
    displacements = [
        float(item["displacement_from_correct_depth_normalized"])
        for item in items
        if item["displacement_from_correct_depth_normalized"] is not None
    ]
    sensitive = sum(value >= threshold for value in displacements)
    single_pairs = [(item, correct) for item, correct in zip(items, correct_items) if item["target_state"] == "single"]
    return {
        "condition": condition,
        "pair_count": len(items),
        "point_pair_count": len(displacements),
        "mean_displacement_normalized": float(statistics.fmean(displacements)) if displacements else None,
        "median_displacement_normalized": float(statistics.median(displacements)) if displacements else None,
        "q90_displacement_normalized": quantile(displacements, 0.90),
        "max_displacement_normalized": max(displacements) if displacements else None,
        "displacement_sensitive_threshold": threshold,
        "displacement_sensitive_count": sensitive,
        "displacement_sensitive_rate": safe_rate(sensitive, len(displacements)),
        "instance_switch_count": sum(item["instance_switch_from_correct_depth"] for item in items),
        "instance_switch_rate": safe_rate(sum(item["instance_switch_from_correct_depth"] for item in items), len(items)),
        "answer_changed_count": sum(item["answer_changed_from_correct_depth"] for item in items),
        "correct_hit_to_miss_count": sum(correct["point_hit"] is True and item["point_hit"] is False for item, correct in single_pairs),
        "correct_miss_to_hit_count": sum(correct["point_hit"] is False and item["point_hit"] is True for item, correct in single_pairs),
        "point_hit_delta_count": sum(item["point_hit"] is True for item, _ in single_pairs) - sum(correct["point_hit"] is True for _, correct in single_pairs),
        "interior_hit_delta_count": sum(item["interior_eroded_hit"] is True for item, _ in single_pairs) - sum(correct["interior_eroded_hit"] is True for _, correct in single_pairs),
    }


def build_summary(evaluations: list[dict], protocol: Mapping[str, Any]) -> dict:
    condition_summary = {
        condition: summarize_subset([item for item in evaluations if item["condition"] == condition])
        for condition in CONDITIONS
    }
    threshold = float(protocol["decision_thresholds"]["displacement_sensitive_normalized"])
    comparisons = {
        condition: paired_comparison(condition, evaluations, threshold)
        for condition in CONDITIONS
        if condition != "correct_depth"
    }
    corrupt_items = [item for item in evaluations if item["condition"] in CORRUPTED_DEPTH_CONDITIONS]
    corrupt_displacements = [
        float(item["displacement_from_correct_depth_normalized"])
        for item in corrupt_items
        if item["displacement_from_correct_depth_normalized"] is not None
    ]
    sensitive_count = sum(value >= threshold for value in corrupt_displacements)
    switch_count = sum(item["instance_switch_from_correct_depth"] for item in corrupt_items)
    corruption_aggregate = {
        "record_count": len(corrupt_items),
        "point_pair_count": len(corrupt_displacements),
        "displacement_sensitive_threshold": threshold,
        "displacement_sensitive_count": sensitive_count,
        "displacement_sensitive_rate": safe_rate(sensitive_count, len(corrupt_displacements)),
        "instance_switch_count": switch_count,
        "instance_switch_rate": safe_rate(switch_count, len(corrupt_items)),
        "correct_hit_to_miss_count": sum(
            comparisons[condition]["correct_hit_to_miss_count"]
            for condition in CORRUPTED_DEPTH_CONDITIONS
        ),
    }
    by_task_type = {}
    for task_type_value in sorted({item["task_type"] for item in evaluations}):
        by_task_type[task_type_value] = {
            condition: summarize_subset([
                item for item in evaluations
                if item["task_type"] == task_type_value and item["condition"] == condition
            ])
            for condition in CONDITIONS
        }
    by_task_group = {}
    for group in ("direct", "relation_or_comparison", "selective"):
        by_task_group[group] = {
            condition: summarize_subset([
                item for item in evaluations
                if item["task_group"] == group and item["condition"] == condition
            ])
            for condition in CONDITIONS
        }
    metrics = {
        "condition_summary": condition_summary,
        "paired_vs_correct_depth": comparisons,
        "corruption_aggregate": corruption_aggregate,
        "task_type_summary": by_task_type,
        "task_group_summary": by_task_group,
    }
    decision, reasons = decision_from_metrics(metrics, protocol["decision_thresholds"])
    metrics["decision"] = decision
    metrics["decision_reasons"] = reasons
    metrics["decision_thresholds"] = protocol["decision_thresholds"]
    return metrics


def write_paired_csv(path: Path, comparisons: Mapping[str, Mapping[str, Any]]) -> None:
    rows = list(comparisons.values())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_gui(path: Path, evaluations: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], result_root: Path) -> None:
    by_scene = defaultdict(list)
    for item in evaluations:
        by_scene[item["scene_id"]].append(item)
    summary_rows = []
    for condition in CONDITIONS:
        item = summary["condition_summary"][condition]
        summary_rows.append(
            f"<tr><td>{condition}</td><td>{item['strict_parse_count']}/10</td>"
            f"<td>{item['point_hit_count']}/{item['single_target_count']}</td>"
            f"<td>{item['interior_hit_count']}/{item['single_target_count']}</td>"
            f"<td>{item['median_displacement_from_correct_normalized']:.4f}</td>"
            f"<td>{item['median_latency_ms']:.1f}</td></tr>"
        )
    sections = []
    for scene_id in sorted(by_scene):
        ordered = sorted(by_scene[scene_id], key=lambda item: CONDITIONS.index(item["condition"]))
        cards = []
        for item in ordered:
            overlay = "../" + str(item["overlay_file"])
            depth_html = ""
            if item["request_depth_file"]:
                depth_html = f'<img class="depth" src="../{html.escape(str(item["request_depth_file"]))}" alt="depth">'
            hit = item["point_hit"] if item["point_hit"] is not None else item["plausible_point_hit"]
            displacement = item["displacement_from_correct_depth_normalized"]
            cards.append(
                f'<article><h3>{html.escape(item["condition"])}</h3>'
                f'<img src="{html.escape(overlay)}" alt="overlay">{depth_html}'
                f'<p>point={item["normalized_point_xy"]}<br>label={item["selected_semantic_label"]}; '
                f'hit={hit}; interior={item["interior_eroded_hit"]}<br>'
                f'd(correct)={displacement:.4f}; latency={item["inference_latency_ms"]:.1f} ms</p></article>'
            )
        sections.append(
            f'<section><h2>{scene_id} — {ordered[0]["task_type"]}</h2><div class="cards">{"".join(cards)}</div></section>'
        )
    document = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>WP1 Depth Sensitivity</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#e6edf3;margin:0;padding:24px}}h1{{margin-bottom:4px}}.decision{{color:#f0b429;font-size:22px;font-weight:800}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border:1px solid #30363d;padding:8px;text-align:left}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px}}article{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px}}article img{{width:100%;border-radius:6px}}article img.depth{{width:38%;margin-top:6px;image-rendering:auto}}article p{{font-family:ui-monospace,monospace;font-size:12px;line-height:1.5}}section{{margin-top:30px}}</style></head>
<body><h1>WP1 — Depth Sensitivity Diagnostic</h1><p class="decision">{summary['decision']}</p>
<p>10 scene × 6 condition; prediction được khóa trước khi evaluator đọc semantic mask.</p>
<table><thead><tr><th>Condition</th><th>Parse</th><th>Target hit</th><th>Interior hit</th><th>Median d(correct)</th><th>Median latency ms</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table>
{''.join(sections)}</body></html>"""
    path.write_text(document, encoding="utf-8")


def markdown_report(summary: Mapping[str, Any], result_root: Path, pilot_digest: str) -> str:
    lines = [
        "# WP1 — Depth Sensitivity Diagnostic",
        "",
        f"**Quyết định chính thức: `{summary['decision']}`.**",
        "",
        f"Lý do: {', '.join(f'`{item}`' for item in summary['decision_reasons'])}.",
        "",
        "## Kết quả theo điều kiện",
        "",
        "| Điều kiện | Parse | Point hit single-target | Interior hit | Median displacement vs correct | Median latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        item = summary["condition_summary"][condition]
        lines.append(
            f"| `{condition}` | {item['strict_parse_count']}/10 | "
            f"{item['point_hit_count']}/{item['single_target_count']} | "
            f"{item['interior_hit_count']}/{item['single_target_count']} | "
            f"{item['median_displacement_from_correct_normalized']:.4f} | "
            f"{item['median_latency_ms']:.1f} ms |"
        )
    aggregate = summary["corruption_aggregate"]
    lines.extend([
        "",
        "## Paired depth-counterfactual evidence",
        "",
        f"- Sensitive displacement: {aggregate['displacement_sensitive_count']}/{aggregate['point_pair_count']} "
        f"({aggregate['displacement_sensitive_rate']:.1%}) với ngưỡng normalized Euclidean `0.02`.",
        f"- Instance switch: {aggregate['instance_switch_count']}/{aggregate['record_count']} "
        f"({aggregate['instance_switch_rate']:.1%}).",
        f"- Correct-hit → miss dưới depth corruption: {aggregate['correct_hit_to_miss_count']} paired records.",
        "",
        "## Gate và provenance",
        "",
        "- 60 prediction được query đúng một lần bằng greedy decoding, seed `8132026`.",
        "- Runner không đọc annotation, semantic labels hoặc Gazebo oracle.",
        "- Evaluator chỉ mở oracle sau khi `prediction_lock.json` và toàn bộ inference hash được xác minh.",
        f"- Pilot tree digest trước/sau: `{pilot_digest}`.",
        f"- Kết quả gốc: `{result_root.relative_to(ROOT)}`.",
        "",
        "## Diễn giải có giới hạn",
        "",
        "WP1 chỉ có 10 scene nên đây là diagnostic, không phải kiểm định ý nghĩa thống kê. "
        "Checkpoint hiện chỉ xuất một point, vì vậy không có heatmap-change metric. "
        "Quyết định trên tuân theo threshold đã khóa trước inference, không được chọn lại sau khi xem oracle.",
        "",
        "## Artifact",
        "",
        f"- GUI: `{result_root.relative_to(ROOT)}/evaluation/WP1_GUI.html`",
        f"- Summary: `{result_root.relative_to(ROOT)}/evaluation/summary.json`",
        f"- Paired CSV: `{result_root.relative_to(ROOT)}/evaluation/paired_comparisons.csv`",
        "",
    ])
    return "\n".join(lines)


def evaluate(protocol_path: Path) -> dict:
    protocol_path = protocol_path.resolve()
    protocol = read_json(protocol_path)
    locked_sources = {
        **protocol.get("runner_source_sha256", {}),
        **protocol.get("evaluator_source_sha256", {}),
    }
    for relative_path, expected_hash in locked_sources.items():
        path = ROOT / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise WP1Error(f"evaluator source differs from preregistration: {relative_path}")
    result_root = (ROOT / protocol["result_root"]).resolve()
    lock, records = validate_prediction_lock(result_root, protocol_path)

    # Oracle boundary: no annotation or evaluator image is opened before this line.
    annotations_path = ROOT / "ur3/ur3_perception/config/roborefer_pilot_v0_annotations.yaml"
    annotations_payload = yaml.safe_load(annotations_path.read_text(encoding="utf-8"))
    annotations = annotations_payload["scenes"]
    dataset_root = ROOT / protocol["dataset_root"]
    margin = int(protocol["evaluation"]["safe_eroded_margin_px"])
    evaluations = []
    label_cache = {}
    for record in records:
        scene_id = str(record["scene_id"])
        if scene_id not in label_cache:
            label_cache[scene_id] = load_labels(dataset_root / scene_id / "evaluator/semantic_labels.png")
        evaluations.append(evaluate_record(record, annotations[scene_id], label_cache[scene_id], margin))
    attach_pair_metrics(evaluations)
    summary_metrics = build_summary(evaluations, protocol)
    pilot_digest = pilot_tree_digest(ROOT, protocol["pilot_lock"]["relative_path"])
    if pilot_digest != protocol["pilot_lock"]["tree_sha256"]:
        raise WP1Error("pilot tree changed between preregistration and evaluation")
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_lock_sha256": sha256_file(result_root / "prediction_lock.json"),
        "oracle_opened_only_after_prediction_lock_verified": True,
        "pilot_tree_sha256": pilot_digest,
        **summary_metrics,
        "limitations": protocol["evaluation"]["limitations_locked_before_run"],
    }
    evaluation_root = result_root / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=False)
    evaluations_path = evaluation_root / "evaluations.jsonl"
    summary_path = evaluation_root / "summary.json"
    csv_path = evaluation_root / "paired_comparisons.csv"
    gui_path = evaluation_root / "WP1_GUI.html"
    write_jsonl(evaluations_path, evaluations)
    write_json(summary_path, summary)
    write_paired_csv(csv_path, summary["paired_vs_correct_depth"])
    create_gui(gui_path, evaluations, summary, result_root)
    report_text = markdown_report(summary, result_root, pilot_digest)
    (result_root / "RESULT.md").write_text(report_text, encoding="utf-8")
    (ROOT / "protocol/WP1_DEPTH_SENSITIVITY.md").write_text(report_text, encoding="utf-8")

    evaluation_files = []
    for path in sorted([evaluations_path, summary_path, csv_path, gui_path, result_root / "RESULT.md"]):
        evaluation_files.append({
            "path": str(path.relative_to(result_root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    evaluation_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "prediction_lock_verified_before_oracle": True,
        "prediction_lock_sha256": sha256_file(result_root / "prediction_lock.json"),
        "annotation_sha256": sha256_file(annotations_path),
        "file_count": len(evaluation_files),
        "files": evaluation_files,
    }
    evaluation_manifest_path = evaluation_root / "evaluation_manifest.json"
    write_json(evaluation_manifest_path, evaluation_manifest)

    test_log = ROOT / "protocol/logs/wp1_pytest.log"
    test_result = {"passed": False, "passed_count": None, "duration_s": None, "log": "protocol/logs/wp1_pytest.log"}
    if test_log.is_file():
        match = re.search(r"(\d+) passed(?:, \d+ warnings?)? in ([0-9.]+)s", test_log.read_text(encoding="utf-8"))
        if match:
            test_result.update(passed=True, passed_count=int(match.group(1)), duration_s=float(match.group(2)))
    exact_count = int(lock["exact_point_parse_count"])
    gates = {
        "prediction_records_60_of_60": len(records) == 60,
        "exact_point_outputs_60_of_60": exact_count == 60,
        "all_inference_artifact_hashes_verified_before_oracle": True,
        "oracle_opened_only_after_prediction_lock": True,
        "pilot_wp0_unchanged": pilot_digest == protocol["pilot_lock"]["tree_sha256"],
        "paired_comparison_complete": len(summary["paired_vs_correct_depth"]) == 5,
        "official_decision_is_valid": summary["decision"] in {"GO_PCRA_F", "NO_GO_PCRA_F", "FIX_DEPTH_PIPELINE_FIRST"},
        "software_tests_passed": bool(test_result["passed"]),
    }
    test_report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_passed": all(gates.values()),
        "gates": gates,
        "software_tests": test_result,
        "record_count": len(records),
        "exact_point_parse_count": exact_count,
        "decision": summary["decision"],
        "decision_reasons": summary["decision_reasons"],
        "pilot_tree_sha256": pilot_digest,
        "protocol_manifest_sha256": sha256_file(protocol_path),
        "prediction_lock_sha256": sha256_file(result_root / "prediction_lock.json"),
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
        "deliverables": {
            "protocol/WP1_DEPTH_SENSITIVITY.md": sha256_file(ROOT / "protocol/WP1_DEPTH_SENSITIVITY.md"),
            str(gui_path.relative_to(ROOT)): sha256_file(gui_path),
            str(summary_path.relative_to(ROOT)): sha256_file(summary_path),
        },
        "result_root": str(result_root.relative_to(ROOT)),
    }
    write_json(ROOT / "protocol/wp1_test_report.json", test_report)
    print(f"WP1_EVALUATION_COMPLETE: {summary['decision']}")
    if not test_report["overall_passed"]:
        raise WP1Error(f"WP1 gate failed: {gates}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol/wp1_manifest.json")
    arguments = parser.parse_args(argv)
    evaluate(arguments.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
