#!/usr/bin/env python3
"""Read-only WP3 audit for the locked 50-family/250-record WP2 prototype."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2


EXPECTED_VARIANTS = {
    "clean",
    "semantic_counterfactual",
    "relation_counterfactual",
    "depth_corruption",
    "occlusion_view_counterfactual",
}
FORBIDDEN_INFERENCE_TOKENS = {
    "oracle",
    "targetmask",
    "anchormask",
    "interiormask",
    "graspablemask",
    "reachablemask",
    "semanticlabel",
    "instancelabel",
    "relationgraph",
    "answerability",
    "answerable",
    "expectedintervention",
    "validtarget",
    "targetid",
    "anchorid",
    "gazebopose",
}
CORE_RELATIONS = {
    "direct",
    "left_of",
    "right_of",
    "nearer_than",
    "farther_than",
    "front_of",
    "behind",
    "between_in_depth",
    "nearer_than_both",
}


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AuditError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def normalize_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def forbidden_paths(value: Any, prefix: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = normalize_key(str(key))
            if any(token in normalized for token in FORBIDDEN_INFERENCE_TOKENS):
                findings.append(f"{prefix}.{key}")
            findings.extend(forbidden_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(forbidden_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, str):
        normalized = normalize_key(value)
        if any(token in normalized for token in FORBIDDEN_INFERENCE_TOKENS):
            findings.append(prefix)
    return findings


def counter_rows(counter: Counter[Any], key_name: str = "key") -> list[dict[str, Any]]:
    return [{key_name: str(key), "count": value} for key, value in sorted(counter.items(), key=lambda item: str(item[0]))]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["key", "count"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def crosstab_rows(mapping: dict[str, Counter[str]], row_name: str) -> tuple[list[dict[str, Any]], list[str]]:
    columns = sorted({column for counts in mapping.values() for column in counts})
    rows: list[dict[str, Any]] = []
    for row_key in sorted(mapping):
        row: dict[str, Any] = {row_name: row_key}
        row.update({column: mapping[row_key].get(column, 0) for column in columns})
        row["total"] = sum(mapping[row_key].values())
        rows.append(row)
    return rows, [row_name, *columns, "total"]


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def mask_stats(dataset_root: Path, ref: dict[str, Any]) -> tuple[int, int, int]:
    path = dataset_root / ref["path"]
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise AuditError(f"Cannot read mask: {path}")
    if image.ndim == 3:
        image = image[:, :, 0]
    height, width = image.shape[:2]
    return int((image > 0).sum()), width, height


def image_size(dataset_root: Path, ref: dict[str, Any]) -> tuple[int, int]:
    path = dataset_root / ref["path"]
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise AuditError(f"Cannot read image: {path}")
    height, width = image.shape[:2]
    return width, height


def svg_bar_chart(path: Path, title: str, rows: list[tuple[str, int]], width: int = 1000) -> None:
    rows = sorted(rows, key=lambda item: (-item[1], item[0]))
    row_height = 36
    top = 70
    label_width = 300
    right = 90
    height = top + row_height * len(rows) + 45
    maximum = max((value for _, value in rows), default=1)
    bar_width = width - label_width - right
    colors = ["#2563eb", "#0891b2", "#7c3aed", "#db2777", "#ea580c", "#16a34a"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="38" font-family="sans-serif" font-size="24" font-weight="700" fill="#0f172a">{html.escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(rows):
        y = top + index * row_height
        scaled = 0 if maximum == 0 else value / maximum * bar_width
        color = colors[index % len(colors)]
        parts.append(f'<text x="24" y="{y + 21}" font-family="sans-serif" font-size="16" fill="#334155">{html.escape(label)}</text>')
        parts.append(f'<rect x="{label_width}" y="{y + 3}" width="{scaled:.2f}" height="24" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{label_width + scaled + 8:.2f}" y="{y + 21}" font-family="sans-serif" font-size="16" fill="#0f172a">{value}</text>')
    parts.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def markdown_report(summary: dict[str, Any], result_root: Path) -> str:
    counts = summary["counts"]
    gaps = summary["data_gaps"]
    states = summary["distributions"]["states"]
    split_states = summary["distributions"]["state_by_split"]
    lines = [
        "# WP3 — Data-gap audit của Dataset WP2",
        "",
        f"> **Kết luận:** `{summary['decision']}`",
        ">",
        f"> **Thời điểm audit UTC:** {summary['generated_at_utc']}",
        ">",
        "> Audit chỉ đọc đúng dataset WP2 đã khóa. Không train model, không sinh family/sample mới và không sửa artifact WP2.",
        "",
        "## 1. Gate",
        "",
        "| Gate | Kết quả |",
        "|---|---:|",
    ]
    for key, value in summary["gates"].items():
        lines.append(f"| `{key}` | {'PASS' if value else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## 2. Quy mô và phân bố chính",
            "",
            f"- {counts['families']} family, {counts['records']} record, {counts['raw_capture_ids']} capture ID duy nhất;",
            f"- {counts['depth_dependent_families']} family phụ thuộc depth;",
            f"- anchor tối đa {counts['max_anchor_count']}, relation tối đa {counts['max_relation_count']};",
            f"- {counts['unique_instructions']} instruction duy nhất trên {counts['records']} record;",
            f"- image resolutions: {', '.join(summary['image_resolutions'])}.",
            "",
            "### Answerability state",
            "",
            "| State | Sample | Tỷ lệ |",
            "|---|---:|---:|",
        ]
    )
    for state, count in sorted(states.items()):
        lines.append(f"| `{state}` | {count} | {count / counts['records']:.1%} |")
    lines.extend(["", "### State theo split", "", "| Split | " + " | ".join(sorted(states)) + " | Total |", "|---|" + "---:|" * (len(states) + 1)])
    for split, row in sorted(split_states.items()):
        lines.append("| " + split + " | " + " | ".join(str(row.get(state, 0)) for state in sorted(states)) + f" | {sum(row.values())} |")
    lines.extend(
        [
            "",
            "## 3. Query-graph coverage",
            "",
            f"- `Kmax=3`: dataset max anchor = {counts['max_anchor_count']} → {'PASS' if summary['gates']['query_graph_anchor_capacity'] else 'FAIL'};",
            f"- `Lmax=3`: dataset max relation = {counts['max_relation_count']} → {'PASS' if summary['gates']['query_graph_relation_capacity'] else 'FAIL'};",
            f"- relation ngoài core WP3: {', '.join(summary['non_core_relations']) or 'không có'};",
            f"- reference frames: {', '.join(summary['distributions']['reference_frames'])}.",
            "",
            "`object_semantics` và các relation shape/metric không được coi là deployable spatial relation supervision nếu chưa chuyển thành observable frame/evidence tương ứng.",
            "",
            "## 4. Mask và instruction audit",
            "",
            f"- Target-mask fraction median: {summary['numeric']['target_fraction']['median']:.6f};",
            f"- Interior/target ratio median trên target khác rỗng: {summary['numeric']['interior_target_ratio']['median']:.6f};",
            f"- Instruction word count: median {summary['numeric']['instruction_words']['median']:.1f}, min {summary['numeric']['instruction_words']['min']}, max {summary['numeric']['instruction_words']['max']};",
            f"- Record hash mismatch: {counts['record_hash_mismatches']};",
            f"- Forbidden inference finding: {counts['forbidden_inference_findings']}.",
            "",
            "## 5. Data gaps bắt buộc xử lý trước calibration/locked test",
            "",
            "| Mức | Gap | Bằng chứng | Hành động |",
            "|---|---|---|---|",
        ]
    )
    for gap in gaps:
        lines.append(f"| {gap['severity']} | {gap['id']} | {gap['evidence']} | {gap['action']} |")
    lines.extend(
        [
            "",
            "## 6. Quyết định",
            "",
            "Audit cho phép bước tiếp theo là **runtime feature-hook verification và 15-sample determinism smoke**.",
            "",
            "Audit **không** cho phép:",
            "",
            "- tuyên bố dataset đủ calibration/generalization;",
            "- train P-CRA-U;",
            "- scale dataset mà chưa preregister data expansion;",
            "- mở prototype test để chọn kiến trúc;",
            "- dùng relation graph evaluator làm predicted graph.",
            "",
            "## 7. Artifact",
            "",
            f"- Machine summary: `{result_root.as_posix()}/audit_summary.json`;",
            f"- Record-level audit: `{result_root.as_posix()}/tables/record_audit.csv`;",
            f"- Family-level audit: `{result_root.as_posix()}/tables/family_audit.csv`;",
            f"- Count/crosstab tables: `{result_root.as_posix()}/tables/`;",
            f"- Figures: `{result_root.as_posix()}/figures/`;",
            f"- Artifact manifest: `{result_root.as_posix()}/audit_manifest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def audit(dataset_root: Path, output_root: Path, contract_lock: Path, report_md: Path | None) -> dict[str, Any]:
    if output_root.exists():
        raise AuditError(f"Refusing to overwrite existing output directory: {output_root}")
    output_root.mkdir(parents=True)
    tables_root = output_root / "tables"
    figures_root = output_root / "figures"
    tables_root.mkdir()
    figures_root.mkdir()

    required = [
        dataset_root / "dataset_index.json",
        dataset_root / "family_manifest.json",
        dataset_root / "artifact_manifest.json",
        contract_lock,
    ]
    for path in required:
        if not path.is_file():
            raise AuditError(f"Missing required input: {path}")

    input_hashes_before = {str(path.resolve()): sha256_file(path) for path in required}
    dataset_index = read_json(dataset_root / "dataset_index.json")
    family_manifest = read_json(dataset_root / "family_manifest.json")
    contract_lock_payload = read_json(contract_lock)
    if contract_lock_payload.get("status") != "LOCKED":
        raise AuditError("WP3 contract lock status is not LOCKED")
    workspace_root = contract_lock.parent.parent
    contract_hash_mismatches: list[str] = []
    for section in ("artifacts", "model_config_inputs", "source_inputs", "dataset_inputs"):
        for relative_path, expected_hash in contract_lock_payload.get(section, {}).items():
            locked_path = workspace_root / relative_path
            if not locked_path.is_file() or sha256_file(locked_path) != expected_hash:
                contract_hash_mismatches.append(relative_path)

    record_index = dataset_index.get("records", [])
    families = family_manifest.get("families", [])
    records: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    record_hash_mismatches: list[str] = []
    forbidden_findings: list[str] = []
    family_splits: dict[str, set[str]] = defaultdict(set)
    family_variants: dict[str, set[str]] = defaultdict(set)
    family_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    distributions: dict[str, Counter[str]] = {
        "splits": Counter(),
        "variants": Counter(),
        "categories": Counter(),
        "states": Counter(),
        "interventions": Counter(),
        "reference_frames": Counter(),
        "relations": Counter(),
        "sources": Counter(),
        "perturbations": Counter(),
        "severities": Counter(),
        "anchor_counts": Counter(),
        "relation_counts": Counter(),
    }
    cross: dict[str, dict[str, Counter[str]]] = {
        "state_by_split": defaultdict(Counter),
        "state_by_variant": defaultdict(Counter),
        "state_by_category": defaultdict(Counter),
        "source_by_variant": defaultdict(Counter),
        "relation_by_category": defaultdict(Counter),
    }
    image_resolutions: Counter[str] = Counter()
    target_fractions: list[float] = []
    target_pixels_all: list[float] = []
    interior_target_ratios: list[float] = []
    anchor_fractions: list[float] = []
    instruction_words: list[float] = []
    instruction_chars: list[float] = []
    instructions: Counter[str] = Counter()
    instruction_splits: dict[str, set[str]] = defaultdict(set)
    capture_ids: Counter[str] = Counter()
    target_object_ids: Counter[str] = Counter()
    anchor_object_ids: Counter[str] = Counter()

    index_seen: set[str] = set()
    for item in record_index:
        record_path = dataset_root / item["record_path"]
        actual_hash = sha256_file(record_path)
        if actual_hash != item["record_sha256"]:
            record_hash_mismatches.append(item["record_path"])
        record = read_json(record_path)
        sample_id = record["sample_id"]
        if sample_id in index_seen:
            raise AuditError(f"Duplicate sample_id in dataset index: {sample_id}")
        index_seen.add(sample_id)
        if sample_id != item["sample_id"]:
            raise AuditError(f"Index/record sample mismatch: {sample_id}")

        inference_findings = forbidden_paths(record["inference_payload"])
        forbidden_findings.extend(f"{sample_id}:{path}" for path in inference_findings)

        spatial = record["evaluator_only"]["spatial_label"]
        uncertainty = record["evaluator_only"]["uncertainty_label"]
        masks = record["evaluator_only"]["masks"]
        split = record["split"]
        variant = record["variant"]
        category = record["family_category"]
        state = uncertainty["state"]
        intervention = uncertainty["expected_intervention"]
        relations = list(spatial["relations"])
        sources = list(uncertainty["sources"])
        anchors = list(spatial["anchor_ids"])
        family_id = record["family_id"]

        rgb_width, rgb_height = image_size(dataset_root, record["inference_payload"]["rgb_model_input"])
        depth_width, depth_height = image_size(dataset_root, record["inference_payload"]["depth_relative_model_input"])
        if (rgb_width, rgb_height) != (depth_width, depth_height):
            raise AuditError(f"RGB/depth size mismatch: {sample_id}")
        image_resolutions[f"{rgb_width}x{rgb_height}"] += 1
        image_area = rgb_width * rgb_height

        target_pixels, target_width, target_height = mask_stats(dataset_root, masks["target"])
        interior_pixels, _, _ = mask_stats(dataset_root, masks["target_interior"])
        if (target_width, target_height) != (rgb_width, rgb_height):
            raise AuditError(f"Target mask/image size mismatch: {sample_id}")
        anchor_pixels = 0
        for anchor_ref in masks["anchor"]:
            pixels, anchor_width, anchor_height = mask_stats(dataset_root, anchor_ref)
            if (anchor_width, anchor_height) != (rgb_width, rgb_height):
                raise AuditError(f"Anchor mask/image size mismatch: {sample_id}")
            anchor_pixels += pixels

        target_fraction = target_pixels / image_area
        target_fractions.append(target_fraction)
        target_pixels_all.append(float(target_pixels))
        if target_pixels > 0:
            interior_target_ratios.append(interior_pixels / target_pixels)
        anchor_fractions.append(anchor_pixels / image_area)
        words = len(record["instruction"].split())
        instruction_words.append(float(words))
        instruction_chars.append(float(len(record["instruction"])))
        instructions[record["instruction"]] += 1
        instruction_splits[record["instruction"]].add(split)
        capture_ids[record["provenance"]["capture_id"]] += 1
        target_object_ids.update(spatial["target_ids"])
        anchor_object_ids.update(anchors)

        distributions["splits"][split] += 1
        distributions["variants"][variant] += 1
        distributions["categories"][category] += 1
        distributions["states"][state] += 1
        distributions["interventions"][intervention] += 1
        distributions["reference_frames"][spatial["reference_frame"]] += 1
        distributions["perturbations"][record["provenance"]["perturbation"]["kind"]] += 1
        distributions["severities"][str(uncertainty["severity"])] += 1
        distributions["anchor_counts"][str(len(anchors))] += 1
        distributions["relation_counts"][str(len(relations))] += 1
        distributions["relations"].update(relations)
        distributions["sources"].update(sources)
        cross["state_by_split"][split][state] += 1
        cross["state_by_variant"][variant][state] += 1
        cross["state_by_category"][category][state] += 1
        for source in sources or ["none"]:
            cross["source_by_variant"][variant][source] += 1
        for relation in relations:
            cross["relation_by_category"][category][relation] += 1

        family_splits[family_id].add(split)
        family_variants[family_id].add(variant)
        family_records[family_id].append(record)
        records.append(record)
        record_rows.append(
            {
                "sample_id": sample_id,
                "family_id": family_id,
                "split": split,
                "variant": variant,
                "category": category,
                "depth_dependent": int(record["depth_dependent"]),
                "state": state,
                "intervention": intervention,
                "relations": ";".join(relations),
                "reference_frame": spatial["reference_frame"],
                "anchor_count": len(anchors),
                "relation_count": len(relations),
                "sources": ";".join(sources),
                "severity": uncertainty["severity"],
                "perturbation": record["provenance"]["perturbation"]["kind"],
                "capture_id": record["provenance"]["capture_id"],
                "rgb_width": rgb_width,
                "rgb_height": rgb_height,
                "target_pixels": target_pixels,
                "target_fraction": f"{target_fraction:.9f}",
                "interior_pixels": interior_pixels,
                "interior_target_ratio": "" if target_pixels == 0 else f"{interior_pixels / target_pixels:.9f}",
                "anchor_pixels_total": anchor_pixels,
                "instruction_words": words,
                "instruction_chars": len(record["instruction"]),
                "record_sha256": actual_hash,
            }
        )

    family_manifest_by_id = {family["family_id"]: family for family in families}
    family_rows: list[dict[str, Any]] = []
    for family_id in sorted(family_records):
        manifest_family = family_manifest_by_id[family_id]
        family_rows.append(
            {
                "family_id": family_id,
                "split": manifest_family["split"],
                "category": manifest_family["category"],
                "depth_dependent": int(manifest_family["depth_dependent"]),
                "record_count": len(family_records[family_id]),
                "variants": ";".join(sorted(family_variants[family_id])),
                "max_anchor_count": max(len(record["evaluator_only"]["spatial_label"]["anchor_ids"]) for record in family_records[family_id]),
                "max_relation_count": max(len(record["evaluator_only"]["spatial_label"]["relations"]) for record in family_records[family_id]),
                "states": ";".join(sorted({record["evaluator_only"]["uncertainty_label"]["state"] for record in family_records[family_id]})),
                "capture_ids": ";".join(sorted({record["provenance"]["capture_id"] for record in family_records[family_id]})),
            }
        )

    max_anchor_count = max((len(record["evaluator_only"]["spatial_label"]["anchor_ids"]) for record in records), default=0)
    max_relation_count = max((len(record["evaluator_only"]["spatial_label"]["relations"]) for record in records), default=0)
    non_core_relations = sorted(set(distributions["relations"]) - CORE_RELATIONS)
    split_overlap = {family_id: sorted(splits) for family_id, splits in family_splits.items() if len(splits) != 1}
    variant_failures = {
        family_id: sorted(variants)
        for family_id, variants in family_variants.items()
        if variants != EXPECTED_VARIANTS
    }
    calibration_states = cross["state_by_split"].get("calibration", Counter())
    dev_states = cross["state_by_split"].get("dev", Counter())
    test_states = cross["state_by_split"].get("test", Counter())
    cross_split_instructions = {
        instruction: sorted(splits)
        for instruction, splits in instruction_splits.items()
        if len(splits) > 1
    }
    distributions["family_categories"] = Counter(family["category"] for family in families)
    distributions["family_splits"] = Counter(family["split"] for family in families)

    gates = {
        "contract_lock_verified": not contract_hash_mismatches,
        "exactly_250_records": len(records) == 250,
        "exactly_50_families": len(families) == 50 and len(family_records) == 50,
        "five_variants_per_family": not variant_failures,
        "family_split_no_overlap": not split_overlap,
        "record_hashes_match_index": not record_hash_mismatches,
        "inference_payload_oracle_free": not forbidden_findings,
        "query_graph_anchor_capacity": max_anchor_count <= 3,
        "query_graph_relation_capacity": max_relation_count <= 3,
        "rgb_depth_resolution_aligned": True,
        "dataset_not_scaled": len(families) == 50 and len(records) == 250,
    }

    gaps = [
        {
            "severity": "CRITICAL",
            "id": "AMBIGUOUS_UNDERREPRESENTED",
            "evidence": f"{distributions['states'].get('AMBIGUOUS', 0)}/250 sample",
            "action": "Tăng family ambiguous trước calibration/locked claims; không oversample variants như family độc lập.",
        },
        {
            "severity": "CRITICAL",
            "id": "ABSENT_UNDERREPRESENTED",
            "evidence": f"{distributions['states'].get('ABSENT', 0)}/250 sample",
            "action": "Tăng target/anchor absent families và giữ minimum negative locked evidence.",
        },
        {
            "severity": "CRITICAL",
            "id": "DEV_MISSING_AMBIGUOUS_ABSENT",
            "evidence": f"dev states={dict(sorted(dev_states.items()))}",
            "action": "Bổ sung dev families cho đủ bốn answerability states trước architecture/model selection.",
        },
        {
            "severity": "CRITICAL",
            "id": "CALIBRATION_SPLIT_TOO_SMALL",
            "evidence": f"30 sample; states={dict(sorted(calibration_states.items()))}",
            "action": "Không fit/claim multimodal calibration trên split prototype; preregister expansion theo family.",
        },
        {
            "severity": "CRITICAL",
            "id": "PROTOTYPE_TEST_NOT_FINAL_LOCKED_TEST",
            "evidence": f"30 sample; states={dict(sorted(test_states.items()))}",
            "action": "Tạo Test-IID/Test-OOD mới sau development freeze; không dùng prototype test để chọn model.",
        },
        {
            "severity": "HIGH",
            "id": "ORACLE_GRAPH_NOT_PREDICTED_GRAPH",
            "evidence": "relation_graph nằm trong evaluator_only ở toàn bộ record",
            "action": "Xây language/predicted query graph; oracle chỉ làm upper bound.",
        },
        {
            "severity": "HIGH",
            "id": "NO_MULTI_CLAUSE_RELATION_COVERAGE",
            "evidence": f"max relation count={max_relation_count}; Lmax contract=3",
            "action": "Bổ sung multi-clause families nếu muốn giữ claim Lmax>1; nếu không, thu hẹp contract/model claim về một clause.",
        },
        {
            "severity": "HIGH",
            "id": "NON_CORE_RELATION_TASKS",
            "evidence": ", ".join(non_core_relations) or "none",
            "action": "Tách shape/metric/semantic-reference khỏi core spatial relation head v1 hoặc định nghĩa head riêng.",
        },
        {
            "severity": "HIGH",
            "id": "OBJECT_SEMANTICS_REFERENCE_FRAME",
            "evidence": f"{distributions['reference_frames'].get('object_semantics', 0)} record",
            "action": "Chuyển relation deployable sang image/camera/base evidence; không dùng semantics oracle khi inference.",
        },
        {
            "severity": "HIGH",
            "id": "SIMULATOR_ONLY_DOMAIN",
            "evidence": "100 raw Gazebo captures; chưa có real calibration/test split",
            "action": "Giới hạn claim simulator và thiết kế real recalibration sau offline gate.",
        },
        {
            "severity": "MEDIUM",
            "id": "SOURCE_LABELS_ARE_CONTROLLED_PERTURBATIONS",
            "evidence": "source labels sinh từ generator/counterfactual",
            "action": "Dùng multi-label paired attribution; không tuyên bố causal diagnosis trên dữ liệu thật.",
        },
        {
            "severity": "MEDIUM",
            "id": "INSTRUCTION_TEMPLATE_REUSE",
            "evidence": f"{len(instructions)} unique/250; {len(cross_split_instructions)} exact instructions xuất hiện ở nhiều split",
            "action": "Giữ paraphrase theo family và tạo language-template OOD; không coi random family split là language generalization.",
        },
        {
            "severity": "MEDIUM",
            "id": "ASSET_AND_LAYOUT_REUSE",
            "evidence": f"{len(target_object_ids)} target object IDs, {len(capture_ids)} capture IDs trên {len(records)} record",
            "action": "Test-OOD phải hold out asset/layout/view generator, không chỉ random family split.",
        },
    ]

    distribution_json = {name: dict(sorted(counter.items())) for name, counter in distributions.items()}
    cross_json = {
        name: {row: dict(sorted(counts.items())) for row, counts in sorted(mapping.items())}
        for name, mapping in cross.items()
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "wp3_dataset_gap_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "contract_lock": str(contract_lock),
        "decision": "GO_FEATURE_HOOK_SMOKE_ONLY_NO_TRAIN_NO_SCALE" if all(gates.values()) else "AUDIT_GATE_FAILED",
        "overall_passed": all(gates.values()),
        "gates": gates,
        "counts": {
            "records": len(records),
            "families": len(families),
            "depth_dependent_families": sum(bool(family["depth_dependent"]) for family in families),
            "raw_capture_ids": len(capture_ids),
            "unique_instructions": len(instructions),
            "duplicate_instruction_instances": sum(count - 1 for count in instructions.values() if count > 1),
            "cross_split_exact_instructions": len(cross_split_instructions),
            "unique_target_object_ids": len(target_object_ids),
            "unique_anchor_object_ids": len(anchor_object_ids),
            "max_anchor_count": max_anchor_count,
            "max_relation_count": max_relation_count,
            "record_hash_mismatches": len(record_hash_mismatches),
            "forbidden_inference_findings": len(forbidden_findings),
        },
        "image_resolutions": sorted(image_resolutions),
        "image_resolution_counts": dict(sorted(image_resolutions.items())),
        "distributions": {**distribution_json, **cross_json},
        "numeric": {
            "target_pixels": summarize_numeric(target_pixels_all),
            "target_fraction": summarize_numeric(target_fractions),
            "interior_target_ratio": summarize_numeric(interior_target_ratios),
            "anchor_fraction": summarize_numeric(anchor_fractions),
            "instruction_words": summarize_numeric(instruction_words),
            "instruction_chars": summarize_numeric(instruction_chars),
        },
        "non_core_relations": non_core_relations,
        "split_overlap": split_overlap,
        "variant_failures": variant_failures,
        "cross_split_instruction_examples": dict(list(sorted(cross_split_instructions.items()))[:20]),
        "record_hash_mismatch_paths": record_hash_mismatches,
        "contract_hash_mismatch_paths": contract_hash_mismatches,
        "forbidden_inference_paths": forbidden_findings,
        "input_hashes_before": input_hashes_before,
        "data_gaps": gaps,
        "limitations": [
            "This audit measures label/layout coverage, not model performance.",
            "Variants from the same family are correlated and are not independent statistical units.",
            "Oracle masks and relation graphs were read only by this evaluator audit.",
            "No training, feature extraction, or dataset generation was performed.",
        ],
    }

    write_csv(tables_root / "record_audit.csv", record_rows)
    write_csv(tables_root / "family_audit.csv", family_rows)
    for name, counter in distributions.items():
        write_csv(tables_root / f"counts_{name}.csv", counter_rows(counter))
    for name, mapping in cross.items():
        rows, fields = crosstab_rows(mapping, name.removeprefix("state_by_").removeprefix("source_by_").removeprefix("relation_by_") or "row")
        write_csv(tables_root / f"crosstab_{name}.csv", rows, fields)

    svg_bar_chart(figures_root / "answerability_states.svg", "WP2 answerability states", list(distributions["states"].items()))
    svg_bar_chart(figures_root / "family_categories.svg", "WP2 sample counts by family category", list(distributions["categories"].items()))
    svg_bar_chart(figures_root / "relations.svg", "WP2 relation labels", list(distributions["relations"].items()))
    svg_bar_chart(figures_root / "uncertainty_sources.svg", "WP2 uncertainty-source labels", list(distributions["sources"].items()))

    input_hashes_after = {str(path.resolve()): sha256_file(path) for path in required}
    summary["input_hashes_after"] = input_hashes_after
    summary["gates"]["locked_inputs_unchanged_during_audit"] = input_hashes_before == input_hashes_after
    summary["overall_passed"] = all(summary["gates"].values())
    summary["decision"] = "GO_FEATURE_HOOK_SMOKE_ONLY_NO_TRAIN_NO_SCALE" if summary["overall_passed"] else "AUDIT_GATE_FAILED"

    write_json(output_root / "audit_summary.json", summary)
    report = markdown_report(summary, output_root)
    (output_root / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    if report_md is not None:
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text(report, encoding="utf-8")

    manifest_entries = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "audit_manifest.json":
            manifest_entries.append(
                {
                    "path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        output_root / "audit_manifest.json",
        {
            "schema_version": 1,
            "protocol_id": "wp3_dataset_gap_audit_v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_count": len(manifest_entries),
            "artifacts": manifest_entries,
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "contract_lock_sha256": sha256_file(contract_lock),
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract-lock", type=Path, required=True)
    parser.add_argument("--report-md", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = audit(
        args.dataset_root.resolve(),
        args.output_root.resolve(),
        args.contract_lock.resolve(),
        args.report_md.resolve() if args.report_md else None,
    )
    print(json.dumps({"decision": summary["decision"], "gates": summary["gates"], "counts": summary["counts"]}, indent=2, sort_keys=True))
    return 0 if summary["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
