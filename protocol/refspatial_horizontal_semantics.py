#!/usr/bin/env python3
"""Lock a review convention for RefSpatial horizontal rankings.

This script does not relabel source data and never marks a sample training
eligible.  RefSpatial metadata exposes generated target points and reasoning
steps, but no documented object-mask, centre, or tie convention.  The output
therefore distinguishes an operational RGB review convention from source
generator semantics.
"""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/spatial_vlm_refspatial_v1"
OUT = BASE / "wp2_visual_review"
SOURCE = BASE / "wp1_source_audit"
DIRECT = {"leftmost_ranking", "rightmost_ranking", "horizontal_ordinal_ranking"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def query(text):
    return text.split("Your answer should")[0].strip()


def horizontal_query_kind(text):
    q = query(text).casefold()
    if "leftmost" in q:
        return "leftmost_ranking"
    if "rightmost" in q:
        return "rightmost_ranking"
    if "left to right" in q or "right to left" in q:
        return "horizontal_ordinal_ranking"
    return None


def source_probe(records, candidates):
    """Record only source evidence, without treating it as visual validation."""
    by_id = {r["id"]: r for r in records}
    selected = [
        "0334af62f9bfca7d26ac75a0",
        "055b6634fa34c305629f0a16",
        "01487ad64c6821f80a1317c4",
    ]
    output = []
    for sid in selected:
        row = candidates.get(sid)
        if not row:
            output.append({"sample_id": sid, "status": "not_found_in_fresh_queue"})
            continue
        record = by_id.get(row["scene_id"])
        matching_steps = []
        if record:
            for index, think in enumerate(record.get("think", [])):
                text = think.get("thinking", "")
                if any(token in text.casefold() for token in ("leftmost", "rightmost", "left to right", "right to left")):
                    matching_steps.append({"think_index": index, "thinking": text})
        output.append({
            "sample_id": sid,
            "scene_id": row["scene_id"],
            "instruction": row["instruction"],
            "source_target_xy": row["target_xy"],
            "source_horizontal_steps": matching_steps,
            "interpretation": (
                "Source provides generated position/rank statements, but metadata has no object masks, "
                "center definition, visibility policy, or tie threshold. It cannot certify an RGB review decision."
            ),
        })
    return output


def write_review_guide():
    (OUT / "HORIZONTAL_HUMAN_REVIEW_V2.md").write_text("""# Review ngang RefSpatial v2

Áp dụng duy nhất cho `leftmost_ranking`, `rightmost_ranking` và `horizontal_ordinal_ranking`.

## Quy ước đã khóa

1. Dùng ảnh RGB trong frame camera. Trục ngang là pixel `x`: trái → phải là `x` tăng.
2. Tập ứng viên là các **instance vật thể tabletop nhìn thấy được** thỏa mô tả trong câu hỏi. `object` không có bổ ngữ nghĩa là mọi vật thể tabletop; không đếm bàn, nền, robot, bóng hoặc texture. Với mô tả có màu/lớp, chỉ giữ instance cùng lớp và màu nhận diện rõ.
3. `object-center` phục vụ review là tâm hình học của bounding box 2D nhìn thấy của instance, không phải điểm đáp án nguồn. Đáp án nguồn chỉ cần nằm trên instance đã chọn.
4. `leftmost` chọn centre `x` nhỏ nhất; `rightmost` chọn centre `x` lớn nhất. Ordinal `n` từ trái sang phải chọn phần tử thứ `n` sau sort `x` tăng; từ phải sang trái sort `x` giảm.
5. Ghi `unsure` nếu bất kỳ cặp ứng viên cạnh nhau có tâm ngang cách dưới 3% chiều rộng ảnh, candidate class/màu không rõ, instance bị che khiến tâm không ước lượng đáng tin, hoặc số instance nhỏ hơn ordinal. Không tự phá tie.

## Cách điền

Mở `NEXT_AUDIT.html`; source target chỉ mở sau khi tự xác định target. Điền `reviewer`, thời điểm ISO 8601, `review_origin=human`, các cột correctness, `reference_frame=image_x_right_positive`, reasoning depth `1` cho ranking ngang một bước, và `decision`.

`accept` chỉ dùng khi mọi trường áp dụng là `yes`; bất đồng hay thiếu bằng chứng là `unsure`. Không dùng nội dung `think` hoặc triage AI để xác nhận nhãn.
""")


def update_artifact_manifest():
    """Refresh hashes for new or changed small, reviewable artifacts."""
    path = BASE / "ARTIFACT_MANIFEST.json"
    manifest = json.loads(path.read_text())
    tracked = {row["path"]: row for row in manifest.get("files", [])}
    for rel in list(tracked):
        candidate = ROOT / rel
        if candidate.exists():
            tracked[rel] = {"path": rel, "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}
    additions = [
        "protocol/refspatial_horizontal_semantics.py",
        "protocol/refspatial_ai_horizontal_review.py",
        "protocol/refspatial_ai_horizontal_finalize.py",
        "protocol/refspatial_relation_retention_gate.py",
        "protocol/refspatial_manual_review.py",
        "results/spatial_vlm_refspatial_v1/README.md",
        "results/spatial_vlm_refspatial_v1/wp2_visual_review/ontology_v1.json",
        "results/spatial_vlm_refspatial_v1/wp2_visual_review/HORIZONTAL_HUMAN_REVIEW_V2.md",
        "results/spatial_vlm_refspatial_v1/wp2_visual_review/followup_semantic_adjudication.csv",
        "results/spatial_vlm_refspatial_v1/wp2_visual_review/source_semantic_probe.json",
        "results/spatial_vlm_refspatial_v1/wp2_visual_review/horizontal_semantic_status.json",
        "results/spatial_vlm_refspatial_v1/wp2_visual_review/spotcheck_semantic_dispositions.json",
        "results/spatial_vlm_refspatial_v1/wp2_visual_review/next_manual_review_report.json",
        "results/spatial_vlm_refspatial_v1/wp2_visual_review/ai_horizontal_visual_review_report.json",
        "results/spatial_vlm_refspatial_v1/wp2_visual_review/relation_retention_gate.csv",
        "results/spatial_vlm_refspatial_v1/wp2_visual_review/relation_retention_gate.json",
    ]
    additions.extend(str(path.relative_to(ROOT)) for path in sorted((OUT / "ai_visual_review_sheets_v2").glob("*")) if path.is_file())
    for rel in additions:
        candidate = ROOT / rel
        if candidate.exists():
            tracked[rel] = {"path": rel, "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}
    manifest["status"] = "HORIZONTAL_REVIEW_CONVENTION_LOCKED_HUMAN_GATE_PENDING"
    manifest["review_origin"] = "ai_visual_plus_unreviewed_human_queue"
    manifest["training_eligible"] = False
    manifest["files"] = [tracked[key] for key in sorted(tracked)]
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    candidates = {}
    with (OUT / "next_audit_queue.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            candidates[row["sample_id"]] = row
    records = json.loads((ROOT / "datasets/RefSpatial-Tabletop-Large/metadata.json").read_text())
    source_sha = sha256(ROOT / "datasets/RefSpatial-Tabletop-Large/metadata.json")

    ontology = {
        "ontology_version": "refspatial_review_v2",
        "status": "REVIEW_CONVENTION_LOCKED_SOURCE_GENERATOR_SEMANTICS_UNVERIFIED",
        "scope": "direct horizontal rankings only; never promoted to pairwise left/right",
        "source_semantics": {
            "evidence": "metadata think fields contain generated target/rank statements",
            "not_documented": ["object instance masks", "object-center definition", "visibility inclusion", "tie threshold"],
            "consequence": "source output is not enough to auto-certify visual labels",
        },
        "review_frame": "image pixel frame; x increases left-to-right",
        "candidate_set": {
            "unqualified_object": "all visible tabletop object instances; exclude table, background, robot, shadows and texture",
            "qualified_object": "visible tabletop instances satisfying the stated class and attributes; each physical instance is counted once",
        },
        "object_center": "geometric centre of the visible 2D bounding box, used only to order candidates; source target point is not assumed to be this centre",
        "direct_horizontal": {
            "leftmost_ranking": "argmin visible bounding-box-centre x over the candidate set",
            "rightmost_ranking": "argmax visible bounding-box-centre x over the candidate set",
            "horizontal_ordinal_ranking": "sort centres by x ascending for left-to-right and descending for right-to-left; select explicit ordinal",
        },
        "tie_and_ambiguity": {
            "horizontal_gap": "mark unsure when adjacent candidate centres differ by <0.03 normalized image width",
            "other": ["ambiguous class or colour", "occluded candidate centre", "ordinal exceeds candidate count", "malformed query"],
        },
        "required_human_fields": ["candidate set", "selected instance", "target point on selected instance", "image_x_right_positive", "reasoning depth 1"],
        "retained_main_relations": [],
        "pairwise_relations_promoted_from_ranking": [],
        "training_eligible": False,
    }
    (OUT / "ontology_v1.json").write_text(json.dumps(ontology, indent=2, ensure_ascii=False) + "\n")
    write_review_guide()

    # Preserve prior AI decisions. Apply only deterministic exclusions; all
    # visually unresolved cases remain pending independent human adjudication.
    followup = list(csv.DictReader((OUT / "followup_review_required.csv").open()))
    adjudications, counts = [], Counter()
    for row in followup:
        note = row.get("notes", "")
        low = note.casefold()
        if "omits desired rank" in low or "missing rank" in low:
            disposition = "EXCLUDE_MALFORMED_QUERY"
        elif "distance" in low or "proximity" in low or "nearest" in low or "farthest" in low:
            disposition = "EXCLUDE_DISTANCE_SEMANTICS_UNRESOLVED"
        else:
            disposition = "PENDING_HUMAN_VISUAL_ADJUDICATION"
        counts[disposition] += 1
        adjudications.append({
            "sample_id": row["sample_id"],
            "previous_ai_decision": row["decision"],
            "semantic_disposition": disposition,
            "human_decision": "",
            "note": "Prior AI decision retained as provenance; no human decision has been inferred.",
        })
    with (OUT / "followup_semantic_adjudication.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adjudications[0]))
        writer.writeheader(); writer.writerows(adjudications)

    probe = {
        "metadata_sha256": source_sha,
        "samples": source_probe(records, candidates),
        "conclusion": "Operational review convention locked; compatibility with source generator remains unverified.",
    }
    (OUT / "source_semantic_probe.json").write_text(json.dumps(probe, indent=2, ensure_ascii=False) + "\n")
    spotcheck = {
        "0334af62f9bfca7d26ac75a0": {
            "semantic_disposition": "PENDING_HUMAN_VISUAL_REVIEW",
            "rule": "all visible tabletop instances; sort image-x ascending; select ordinal 7",
        },
        "055b6634fa34c305629f0a16": {
            "semantic_disposition": "PENDING_HUMAN_VISUAL_REVIEW",
            "rule": "visible green water-bottle instances only; select largest image-x centre",
        },
        "01487ad64c6821f80a1317c4": {
            "semantic_disposition": "PENDING_HUMAN_VISUAL_REVIEW_WITH_SOURCE_PROVENANCE_FLAG",
            "rule": "all visible tabletop instances; sort image-x descending; select ordinal 4",
            "reason": "Metadata does not document whether generated points equal visible bounding-box centres; no automatic acceptance.",
        },
    }
    (OUT / "spotcheck_semantic_dispositions.json").write_text(json.dumps(spotcheck, indent=2) + "\n")
    decisions_path = OUT / "next_audit_decisions.csv"
    submitted = list(csv.DictReader(decisions_path.open())) if decisions_path.exists() else []
    completed_ai = sum(row.get("review_origin") == "ai_visual" and bool(row.get("decision")) for row in submitted)
    summary = {
        "ontology": "refspatial_review_v2",
        "fresh_human_queue": len(candidates),
        "fresh_ai_review_completed": completed_ai,
        "fresh_pending_human_review": len(candidates),
        "followup_total": len(adjudications),
        "followup_semantic_dispositions": dict(counts),
        "source_generator_semantics_verified": False,
        "training_eligible": False,
    }
    (OUT / "horizontal_semantic_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    update_artifact_manifest()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
