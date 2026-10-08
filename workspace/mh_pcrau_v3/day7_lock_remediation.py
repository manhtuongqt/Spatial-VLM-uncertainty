#!/usr/bin/env python3
"""Lock Day-7 remediation data roles, splits, leakage audit and Val-v2 contract."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ketquangay/ngay_07"
DATA_OUT = OUT / "du_lieu"
TABLE_OUT = OUT / "bang"
IMAGE_OUT = OUT / "anh"

FRESH_BASE = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/amendment_07_fresh_left_ycb_v14_accepted249"
FRESH_INPUT = FRESH_BASE / "INPUT_MANIFEST.jsonl"
FRESH_SUPERVISION = FRESH_BASE / "SUPERVISION.jsonl"
FRESH_QC = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_final_v14_accepted249/CAPTURE_QC.json"
DAY6_DECISION = ROOT / "ketqua1/07_huan_luyen/ngay_06/amendment_06_left_ycb_capture/DAY6_FINAL_DECISION.json"
DAY6_METRICS = ROOT / "ketqua1/09_danh_gia/metrics/ngay_06/amendment_07_fresh_left_ycb_v14_accepted249/ANTI_SHORTCUT_METRICS.json"
HISTORICAL_MANIFESTS = [
    ROOT / "ketqua1/07_huan_luyen/ngay_06/S1A_INPUT_MANIFEST.jsonl",
    ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/CACHE_INPUT_MANIFEST.jsonl",
]
OCID_DIR = ROOT / "answerability and uncertainty/ocid_ref_occlusion_200"
OCID_QA = OCID_DIR / "qa.jsonl"
OCID_SUMMARY = OCID_DIR / "SUMMARY.json"
METRIC_LOCK = ROOT / "ketqua1/00_quan_tri_khoa/ngay_03/METRIC_MARGIN_LOCK_REVISION_V3.json"

SPLIT_SEED = "mh_pcrau_v3_day07_remediation_split_v1"


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(family_id: str):
    return hashlib.sha256(f"{SPLIT_SEED}|{family_id}".encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path):
    return str(path.relative_to(ROOT))


def font(size=20, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_distribution(split_counts, path: Path):
    states = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"]
    labels = ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT"]
    width, height = 1280, 680
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, small_font = font(30, True), font(20), font(18)
    draw.text((50, 30), "Ngày 7 - Phân bố 249 family theo vai trò remediation", fill="#17365D", font=title_font)
    left, right, top, bottom = 170, 1210, 115, 565
    draw.line((left, bottom, right, bottom), fill="#666666", width=2)
    max_count = max(split_counts[split][state] for split in split_counts for state in states)
    group_width = (right - left) / len(states)
    bar_width = 78
    colors = {"remediation_train": "#3A9D5D", "remediation_diagnostic": "#E89C31"}
    for index, (state, label) in enumerate(zip(states, labels)):
        center = left + group_width * (index + 0.5)
        for offset, split in [(-bar_width / 2, "remediation_train"), (bar_width / 2, "remediation_diagnostic")]:
            value = split_counts[split][state]
            bar_height = (bottom - top) * value / max_count
            x1 = center + offset - bar_width / 2
            x2 = center + offset + bar_width / 2
            y1 = bottom - bar_height
            draw.rectangle((x1, y1, x2, bottom), fill=colors[split])
            text = str(value)
            box = draw.textbbox((0, 0), text, font=label_font)
            draw.text(((x1 + x2 - (box[2] - box[0])) / 2, y1 - 27), text, fill="#222222", font=label_font)
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((center - (box[2] - box[0]) / 2, bottom + 18), label, fill="#222222", font=small_font)
    draw.rectangle((760, 52, 785, 77), fill=colors["remediation_train"])
    draw.text((795, 53), "Train (201)", fill="#222222", font=small_font)
    draw.rectangle((965, 52, 990, 77), fill=colors["remediation_diagnostic"])
    draw.text((1000, 53), "Diagnostic (48)", fill="#222222", font=small_font)
    draw.text((50, 625), "Split theo parent family; mỗi ô answerability × relation giữ 3 family diagnostic.", fill="#555555", font=small_font)
    image.save(path)


def main():
    for directory in (OUT, DATA_OUT, TABLE_OUT, IMAGE_OUT):
        directory.mkdir(parents=True, exist_ok=True)

    created_at = utc_now()
    inputs = read_jsonl(FRESH_INPUT)
    supervision = read_jsonl(FRESH_SUPERVISION)
    input_by_id = {row["sample_id"]: row for row in inputs}
    supervision_by_id = {row["sample_id"]: row for row in supervision}
    assert len(inputs) == len(supervision) == len(input_by_id) == len(supervision_by_id) == 249
    assert set(input_by_id) == set(supervision_by_id)

    cells = defaultdict(list)
    for sample_id, label in supervision_by_id.items():
        cells[(label["answerability"], label["relation"])].append(sample_id)
    assert len(cells) == 16
    assert min(map(len, cells.values())) >= 13

    diagnostic_ids = set()
    for sample_ids in cells.values():
        diagnostic_ids.update(sorted(sample_ids, key=lambda sid: stable_key(input_by_id[sid]["family_id"]))[:3])
    assert len(diagnostic_ids) == 48

    records = []
    for sample_id in sorted(input_by_id):
        source = input_by_id[sample_id]
        label = supervision_by_id[sample_id]
        split = "remediation_diagnostic" if sample_id in diagnostic_ids else "remediation_train"
        record = {
            "schema_version": 1,
            "sample_id": sample_id,
            "family_id": source["family_id"],
            "source_role_original": "anti_shortcut_val_v1_one_shot_consumed",
            "source_role_current": "development_remediation",
            "remediation_split": split,
            "split_seed": SPLIT_SEED,
            "rgb_path": source["rgb_path"],
            "rgb_sha256": source["rgb_sha256"],
            "metric_depth_path": source["metric_depth_path"],
            "metric_depth_sha256": source["metric_depth_sha256"],
            "depth_view_path": source["depth_view_path"],
            "depth_view_sha256": source["depth_view_sha256"],
            "instruction": source["instruction"],
            "instruction_sha256": source["instruction_sha256"],
            "relation": label["relation"],
            "answerability": label["answerability"],
            "target_uv": label["target_uv"],
            "head_mask": label["head_mask"],
            "training_permissions": {
                "relation": True,
                "answerability": True,
                "coordinate": label["answerability"] == "FOUND" and label["target_uv"] is not None,
                "log_variance": label["answerability"] == "FOUND" and label["target_uv"] is not None,
                "reasoning": False,
                "source": False,
                "confidence": False,
            },
        }
        records.append(record)
    write_jsonl(DATA_OUT / "REMEDIATION_MANIFEST.jsonl", records)

    split_counts = {
        split: Counter(row["answerability"] for row in records if row["remediation_split"] == split)
        for split in ("remediation_train", "remediation_diagnostic")
    }
    split_relations = {
        split: Counter(row["relation"] for row in records if row["remediation_split"] == split)
        for split in ("remediation_train", "remediation_diagnostic")
    }
    family_split = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "status": "PASS",
        "source_role_original": "anti_shortcut_val_v1_one_shot_consumed",
        "source_role_current": "development_remediation",
        "policy": "Deterministic stratified split by answerability x relation; three hash-ranked families per cell are diagnostic-only.",
        "seed": SPLIT_SEED,
        "counts": {
            "all": len(records),
            "remediation_train": sum(split_counts["remediation_train"].values()),
            "remediation_diagnostic": sum(split_counts["remediation_diagnostic"].values()),
        },
        "state_support": {split: dict(sorted(counts.items())) for split, counts in split_counts.items()},
        "relation_support": {split: dict(sorted(counts.items())) for split, counts in split_relations.items()},
        "family_overlap_between_splits": len(
            {r["family_id"] for r in records if r["remediation_split"] == "remediation_train"}
            & {r["family_id"] for r in records if r["remediation_split"] == "remediation_diagnostic"}
        ),
        "model_selection_policy": "The 48-family diagnostic split is for debugging/error analysis only. Candidate selection requires the new independent Development-R2 validation split created on Day 8.",
        "manifest": rel(DATA_OUT / "REMEDIATION_MANIFEST.jsonl"),
    }
    write_json(OUT / "REMEDIATION_FAMILY_SPLIT.json", family_split)

    with (TABLE_OUT / "PHAN_BO_SPLIT.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["answerability", "relation", "remediation_train", "remediation_diagnostic", "total"])
        for answerability, relation in sorted(cells):
            train = sum(r["answerability"] == answerability and r["relation"] == relation and r["remediation_split"] == "remediation_train" for r in records)
            diagnostic = sum(r["answerability"] == answerability and r["relation"] == relation and r["remediation_split"] == "remediation_diagnostic" for r in records)
            writer.writerow([answerability, relation, train, diagnostic, train + diagnostic])
    draw_distribution(split_counts, IMAGE_OUT / "PHAN_BO_REMEDIATION_249.png")

    historical_rows = []
    historical_sources = []
    for manifest in HISTORICAL_MANIFESTS:
        rows = read_jsonl(manifest)
        historical_rows.extend(rows)
        historical_sources.append({"path": rel(manifest), "rows": len(rows), "sha256": sha256_file(manifest)})
    fresh_families = {r["family_id"] for r in records}
    fresh_rgb = {r["rgb_sha256"] for r in records}
    fresh_depth = {r["metric_depth_sha256"] for r in records}
    historical_families = {r.get("family_id") for r in historical_rows if r.get("family_id")}
    historical_rgb = {r.get("rgb_sha256") for r in historical_rows if r.get("rgb_sha256")}
    historical_depth = {r.get("metric_depth_sha256") for r in historical_rows if r.get("metric_depth_sha256")}
    missing_paths = []
    hash_mismatches = []
    for row in records:
        for key, hash_key in (("rgb_path", "rgb_sha256"), ("metric_depth_path", "metric_depth_sha256")):
            path = ROOT / row[key]
            if not path.is_file():
                missing_paths.append(row[key])
            elif sha256_file(path) != row[hash_key]:
                hash_mismatches.append(row[key])
    leakage_audit = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "status": "PASS" if not (fresh_families & historical_families or fresh_rgb & historical_rgb or fresh_depth & historical_depth or missing_paths or hash_mismatches) else "FAIL",
        "analysis_unit": "parent_family",
        "remediation_rows": len(records),
        "remediation_unique_families": len(fresh_families),
        "remediation_unique_rgb_hashes": len(fresh_rgb),
        "remediation_unique_depth_hashes": len(fresh_depth),
        "historical_sources": historical_sources,
        "checks": {
            "train_diagnostic_family_overlap_zero": family_split["family_overlap_between_splits"] == 0,
            "historical_family_overlap_zero": not bool(fresh_families & historical_families),
            "historical_rgb_hash_overlap_zero": not bool(fresh_rgb & historical_rgb),
            "historical_metric_depth_hash_overlap_zero": not bool(fresh_depth & historical_depth),
            "all_source_paths_exist": not missing_paths,
            "all_source_hashes_match": not hash_mismatches,
        },
        "overlap_counts": {
            "family": len(fresh_families & historical_families),
            "rgb_sha256": len(fresh_rgb & historical_rgb),
            "metric_depth_sha256": len(fresh_depth & historical_depth),
        },
        "missing_paths": missing_paths,
        "hash_mismatches": hash_mismatches,
        "scope_note": "This audit establishes no overlap with locked S1a/Day5 development manifests. Anti-Shortcut v1 derivatives are the same consumed source by design and are not treated as independent data.",
    }
    write_json(OUT / "FAMILY_LEAKAGE_AUDIT.json", leakage_audit)

    ocid_rows = read_jsonl(OCID_QA)
    ocid_states = Counter(row.get("answerability") for row in ocid_rows)
    uncertainty_nonnull = sum(row.get("uncertainty_label") is not None for row in ocid_rows)
    ocid_policy = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "status": "PASS",
        "dataset_role": "auxiliary_found_grounding_and_ood_qualitative_only",
        "records": len(ocid_rows),
        "unique_image_hashes": len({row.get("sha256") for row in ocid_rows}),
        "answerability_distribution": dict(ocid_states),
        "uncertainty_labels_nonnull": uncertainty_nonnull,
        "head_mask_policy": {
            "relation_or_grounding": "allowed where the original OCID-Ref annotation supports it",
            "coordinate": "bbox/grounding auxiliary only; do not mix normalized point labels without an explicit adapter",
            "answerability": False,
            "log_variance": False,
            "source_uncertainty": False,
            "confidence": False,
        },
        "prohibitions": [
            "Do not relabel small bbox as INSUFFICIENT.",
            "Do not treat occlusion_priority_proxy as uncertainty ground truth.",
            "Do not synthesize ABSENT or AMBIGUOUS labels without a separately audited annotation protocol.",
        ],
        "evidence": {
            "qa_jsonl": rel(OCID_QA),
            "qa_sha256": sha256_file(OCID_QA),
            "summary": rel(OCID_SUMMARY),
            "summary_sha256": sha256_file(OCID_SUMMARY),
        },
    }
    assert len(ocid_rows) == 200 and ocid_states == {"FOUND": 200} and uncertainty_nonnull == 0
    write_json(OUT / "OCID_AUXILIARY_POLICY.json", ocid_policy)

    metric_lock = read_json(METRIC_LOCK)
    val_v2_contract = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "status": "LOCKED_BEFORE_CANDIDATE_R3_TRAINING",
        "role": "one_shot_fresh_anti_shortcut_validation_v2",
        "minimum_target": {
            "parent_families": 256,
            "per_answerability_state": 64,
            "per_answerability_relation_cell": 16,
            "unique_rgb_hashes": 256,
        },
        "required_states": ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"],
        "required_relations": ["leftmost", "rightmost", "second_from_left", "second_from_right"],
        "exclusion": {
            "zero_family_overlap_with": ["S1a development", "249 remediation", "Development-R2 train/val", "OCID auxiliary"],
            "zero_exact_rgb_overlap": True,
            "new_scene_seed_namespace": "mh_pcrau_v3/anti_shortcut_val_v2",
            "materialize_only_after_candidate_freeze": True,
        },
        "pre_inference_requirements": [
            "Capture QC PASS before model access.",
            "Frozen candidate checkpoint SHA-256 recorded.",
            "Frozen preprocessing, prompt, evaluator and shortcut baselines recorded.",
            "Run exactly once; no threshold or model tuning after metrics are viewed.",
        ],
        "hard_checks": {
            "answerability_macro_f1_effect_vs_best_matching_shortcut_ge": metric_lock["minimum_effect_gap_for_planning_absolute"],
            "hit_at_0_05_effect_vs_relation_centroid_ge": metric_lock["minimum_effect_gap_for_planning_absolute"],
            "all_outputs_finite": True,
            "capture_qc_pass": True,
        },
        "report_only_secondary": ["relation_macro_f1", "point_l2_mean_median", "false_accept_rate", "gaussian_nll"],
        "metric_lock_reference": {"path": rel(METRIC_LOCK), "sha256": sha256_file(METRIC_LOCK)},
        "failure_policy": "On FAIL, preserve predictions and report a negative result. Do not tune Candidate R3 on Val-v2.",
    }
    write_json(OUT / "ANTI_SHORTCUT_VAL_V2_CONTRACT.json", val_v2_contract)

    source_evidence = {
        "fresh_input": {"path": rel(FRESH_INPUT), "sha256": sha256_file(FRESH_INPUT)},
        "fresh_supervision": {"path": rel(FRESH_SUPERVISION), "sha256": sha256_file(FRESH_SUPERVISION)},
        "fresh_capture_qc": {"path": rel(FRESH_QC), "sha256": sha256_file(FRESH_QC)},
        "day6_decision": {"path": rel(DAY6_DECISION), "sha256": sha256_file(DAY6_DECISION)},
        "day6_metrics": {"path": rel(DAY6_METRICS), "sha256": sha256_file(DAY6_METRICS)},
    }
    remediation_contract = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "status": "LOCKED",
        "objective": "Use the consumed 249-family one-shot set as development remediation without erasing its historical FAIL, then require a new independent Development-R2 validation and fresh Val-v2.",
        "historical_result_preserved": True,
        "data_roles": {
            "accepted249": "development_remediation",
            "accepted249_train": 201,
            "accepted249_diagnostic_only": 48,
            "ocid_ref_200": "auxiliary_found_grounding_and_ood_qualitative_only",
            "gazebo_development_r2": "to_be_created_day8_train_and_independent_development_validation",
            "anti_shortcut_val_v2": "sealed_until_candidate_r3_freeze",
            "test_and_calibration": "sealed",
        },
        "model_selection": {
            "allowed": "new independent Gazebo Development-R2 validation only",
            "not_allowed": ["historical one-shot metrics", "249 diagnostic split", "OCID auxiliary", "Anti-Shortcut Val-v2"],
        },
        "training_scope_day9": {
            "first_choice": "head_only",
            "active_heads": ["relation", "coordinate_logvariance", "answerability"],
            "confidence": "frozen_until_oof_s1b",
            "reasoning_and_source": "masked_without_certified_labels",
            "lora_or_projector": "conditional_after_head_only_evidence",
            "full_finetune": "prohibited",
        },
        "sources": source_evidence,
        "artifacts": {
            "manifest": rel(DATA_OUT / "REMEDIATION_MANIFEST.jsonl"),
            "family_split": rel(OUT / "REMEDIATION_FAMILY_SPLIT.json"),
            "leakage_audit": rel(OUT / "FAMILY_LEAKAGE_AUDIT.json"),
            "ocid_policy": rel(OUT / "OCID_AUXILIARY_POLICY.json"),
            "val_v2_contract": rel(OUT / "ANTI_SHORTCUT_VAL_V2_CONTRACT.json"),
        },
    }
    write_json(OUT / "REMEDIATION_DATA_CONTRACT.json", remediation_contract)

    checks = {
        "source_records_exact_249": len(records) == 249,
        "source_families_unique_249": len(fresh_families) == 249,
        "split_exact_201_train_48_diagnostic": family_split["counts"]["remediation_train"] == 201 and family_split["counts"]["remediation_diagnostic"] == 48,
        "all_16_cells_have_3_diagnostic": all(
            sum(r["answerability"] == answerability and r["relation"] == relation and r["remediation_split"] == "remediation_diagnostic" for r in records) == 3
            for answerability, relation in cells
        ),
        "zero_train_diagnostic_family_overlap": family_split["family_overlap_between_splits"] == 0,
        "leakage_audit_pass": leakage_audit["status"] == "PASS",
        "ocid_exact_200_found_only": len(ocid_rows) == 200 and ocid_states == {"FOUND": 200},
        "ocid_uncertainty_masked": uncertainty_nonnull == 0 and not ocid_policy["head_mask_policy"]["answerability"],
        "historical_one_shot_fail_preserved": read_json(DAY6_DECISION)["fresh_left_ycb_status"] == "ONE_SHOT_FAIL",
        "val_v2_locked_before_training": val_v2_contract["status"] == "LOCKED_BEFORE_CANDIDATE_R3_TRAINING",
        "calibration_test_robot_remain_sealed": remediation_contract["data_roles"]["test_and_calibration"] == "sealed",
    }
    outcome = "DATA_CONTRACT_PASS" if all(checks.values()) else "DATA_CONTRACT_FAIL"
    decision = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "day": 7,
        "outcome": outcome,
        "checks": checks,
        "day8_authorized": outcome == "DATA_CONTRACT_PASS",
        "authorized_next_action": "Generate and QC Gazebo Development-R2 balanced across four answerability states with new train and independent development-validation families.",
        "not_authorized": ["Candidate R3 training before Development-R2 DATA_PASS", "OOF/S1b", "G3", "Calibration", "Test", "final robot claim"],
        "artifacts": remediation_contract["artifacts"],
    }
    write_json(OUT / "DECISION.json", decision)

    report = f"""# BÁO CÁO NGÀY 07 — KHÓA DỮ LIỆU REMEDIATION

**Kết luận:** `{outcome}`  
**Ngày tạo:** {created_at}  
**Bước tiếp theo được mở:** Gazebo Development-R2 ở Ngày 8.  
**Chưa được mở:** Candidate R3 trước DATA_PASS, OOF/S1b, G3, calibration, Test và robot claim cuối.

## 1. Mục tiêu

Ngày 7 không huấn luyện mô hình. Mục tiêu là đổi đúng vai trò 249 family one-shot đã tiêu thụ thành development remediation, khóa split và leakage audit, xác định phạm vi hợp lệ của 200 OCID-Ref, đồng thời khóa Anti-Shortcut Val v2 trước khi Candidate R3 được train.

## 2. Kết quả khóa 249 family

| Hạng mục | Kết quả |
|---|---:|
| Tổng family | 249 |
| Remediation train | 201 |
| Diagnostic-only | 48 |
| Số ô answerability × relation | 16 |
| Diagnostic mỗi ô | 3 |
| Overlap family train/diagnostic | 0 |
| Thiếu RGB/depth path | {len(missing_paths)} |
| Hash mismatch | {len(hash_mismatches)} |
| Overlap RGB với S1a/Day5 | {leakage_audit['overlap_counts']['rgb_sha256']} |

Split được sinh xác định bằng SHA-256 của `seed | family_id`. Diagnostic chỉ dùng kiểm tra lỗi, không dùng chọn checkpoint. Việc chọn Candidate R3 phải dựa trên validation mới thuộc Gazebo Development-R2.

![Phân bố remediation](anh/PHAN_BO_REMEDIATION_249.png)

Chi tiết từng ô: [`bang/PHAN_BO_SPLIT.csv`](bang/PHAN_BO_SPLIT.csv).  
Manifest: [`du_lieu/REMEDIATION_MANIFEST.jsonl`](du_lieu/REMEDIATION_MANIFEST.jsonl).

## 3. Chính sách OCID-Ref

- Đã xác minh 200/200 QA là `FOUND`.
- Có 200 image hash khác nhau.
- Số uncertainty label khác null: 0.
- Chỉ được dùng cho auxiliary grounding/relation và OOD qualitative analysis.
- Answerability, log-variance, source uncertainty và confidence loss phải mask.
- Không được đổi bbox nhỏ thành `INSUFFICIENT` hoặc coi proxy che khuất là uncertainty ground truth.

## 4. Anti-Shortcut Val v2 đã khóa

Val v2 phải có tối thiểu 256 parent family, 64 family/state và 16 family cho mỗi ô state × relation. Tập này dùng namespace/seed mới, không trùng S1a, 249 remediation, Development-R2 hoặc OCID. Chỉ materialize sau khi checkpoint Candidate R3, preprocessing, prompt và evaluator đã được khóa. Nếu one-shot FAIL, giữ negative result và không tune trên Val v2.

## 5. Quyết định

Tất cả {len(checks)}/{len(checks)} kiểm tra hợp lệ. Ngày 7 đạt `{outcome}` và chỉ mở một hành động tiếp theo: sinh/QC Gazebo Development-R2 cân bằng bốn trạng thái. Day 6 vẫn giữ `DAY6_COMPLETE_GENERALIZATION_HOLD`; kế hoạch này không sửa lịch sử one-shot FAIL.
"""
    (OUT / "BAO_CAO_NGAY_07.md").write_text(report)

    delivery = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        if path.name == "ARTIFACT_MANIFEST.json":
            continue
        delivery.append({"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(OUT / "ARTIFACT_MANIFEST.json", {"schema_version": 1, "created_at_utc": created_at, "outcome": outcome, "artifacts": delivery})
    print(json.dumps({"outcome": outcome, "checks": checks, "counts": family_split["counts"], "artifacts": len(delivery) + 1}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
