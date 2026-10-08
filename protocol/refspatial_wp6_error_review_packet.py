#!/usr/bin/env python3
"""Create a blinded human-review packet for the WP6 B0/B1 error union.

The packet intentionally records no decisions.  A completed form is evidence
of a human review only when the reviewer identity and date are filled in by a
person.  It neither changes WP6 scores nor opens the clean-data, B2, or Gazebo
gates.
"""
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge/error_review"
CHALLENGE = ROOT / "results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge/primary_challenge.jsonl"
PREDICTIONS = {
    "b0": ROOT / "results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge/evaluation/b0/predictions.jsonl",
    "b1": ROOT / "results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge/evaluation/b1/predictions.jsonl",
}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text(value):
    return html.escape(str(value), quote=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    challenge = {row["sample_id"]: row for row in load_jsonl(CHALLENGE)}
    predictions = {model: {row["sample_id"]: row for row in load_jsonl(path)}
                   for model, path in PREDICTIONS.items()}
    errors = {model: {sid for sid, row in rows.items() if not row["hit_at_008"]}
              for model, rows in predictions.items()}
    union = sorted(errors["b0"] | errors["b1"])
    if len(errors["b0"]) != 6 or len(errors["b1"]) != 6:
        raise RuntimeError("Expected six WP6 errors per model; evaluation artifact changed.")
    # Verify the actual prediction artifacts rather than inheriting an earlier
    # narrative count.  The current files contain five shared errors, leaving
    # one model-specific error for each model (seven unique cases).
    if len(errors["b0"] & errors["b1"]) != 5 or len(union) != 7:
        raise RuntimeError("Unexpected B0/B1 WP6 error overlap; inspect prediction artifacts before review.")

    rows = []
    for sid in union:
        source = challenge[sid]
        model_data = {}
        for model in ("b0", "b1"):
            pred = predictions[model][sid]
            model_data[model] = {
                "hit_at_008": pred["hit_at_008"],
                "prediction_xy": pred["prediction_xy"],
                "normalized_point_error": pred["normalized_point_error"],
                "self_consistency_confidence": pred["self_consistency_confidence"],
            }
        category = ("shared_b0_b1_error" if sid in errors["b0"] & errors["b1"]
                    else "b0_only_error" if sid in errors["b0"] else "b1_only_error")
        rows.append({
            "sample_id": sid, "scene_id": source["scene_id"], "family_id": source["family_id"],
            "split": source["split"], "relation": source["relation"],
            "challenge_stratum": source["challenge_stratum"], "instruction": source["instruction"],
            "image": source["image"], "depth": source["depth"], "source_target_xy": source["target_xy"],
            "source_evidence": source.get("source_evidence", {}), "error_category": category,
            "model_outputs": model_data,
            "review_status": "HUMAN_REVIEW_PENDING",
            "training_eligible": False, "b2_eligible": False,
        })

    queue = OUT / "wp6_error_review_queue.jsonl"
    queue.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    fieldnames = [
        "sample_id", "scene_id", "error_category", "relation", "challenge_stratum",
        "reviewer", "reviewed_at", "review_origin", "target_correct", "object_set_correct",
        "ordinal_direction_correct", "reference_frame", "label_ambiguity", "source_target_valid",
        "b0_error_confirmed", "b1_error_confirmed", "decision", "human_target_x",
        "human_target_y", "correction_action", "completed", "notes",
    ]
    decisions = OUT / "wp6_error_review_decisions.csv"
    existing = {}
    if decisions.exists():
        with decisions.open(newline="") as handle:
            existing = {row["sample_id"]: row for row in csv.DictReader(handle)}
    with decisions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            preserved = existing.get(row["sample_id"], {})
            writer.writerow({key: preserved.get(key, row.get(key, "")) for key in fieldnames})

    parts = [
        "<!doctype html><meta charset='utf-8'><title>WP6 B0/B1 error review</title>",
        "<style>body{font:16px sans-serif;max-width:1120px;margin:auto}article{border-bottom:2px solid #777;padding:20px 0}img{max-width:960px;width:100%}pre{white-space:pre-wrap;background:#f5f5f5;padding:10px}.warn{color:#8b0000;font-weight:bold}</style>",
        f"<h1>WP6 B0/B1 error review — {len(rows)} unique scene-query families</h1>",
        "<p class='warn'>This is a human-review packet, not a completed certification. Review RGB and instruction first. Only then reveal the source target and model outputs. Record one row per case in <code>wp6_error_review_decisions.csv</code>.</p>",
        "<p>For each case verify: target/object set, ordinal direction, image reference frame, ambiguity, source target, and whether each scored model error is genuine. Do not use this packet for training; it does not by itself satisfy the 300-query G1 audit.</p>",
    ]
    for index, row in enumerate(rows, 1):
        image = "../../../../" + row["image"]
        outputs = json.dumps({"source_target_xy": row["source_target_xy"], "source_evidence": row["source_evidence"], "model_outputs": row["model_outputs"]}, indent=2)
        parts.extend([
            f"<article><h2>{index}. {text(row['sample_id'])} — {text(row['error_category'])}</h2>",
            f"<p><b>Relation:</b> {text(row['relation'])}; <b>split:</b> {text(row['split'])}; <b>stratum:</b> {text(row['challenge_stratum'])}</p>",
            f"<p><b>Instruction:</b> {text(row['instruction'])}</p>",
            f"<img loading='lazy' src='{text(image)}' alt='RGB scene {text(row['scene_id'])}'>",
            f"<details><summary>Reveal source target and B0/B1 outputs after independent scene reading</summary><pre>{text(outputs)}</pre></details></article>",
        ])
    gallery = OUT / "WP6_B0_B1_ERROR_REVIEW.html"
    gallery.write_text("\n".join(parts) + "\n")

    readme = OUT / "README.md"
    readme.write_text("""# WP6 B0/B1 error review packet

This packet freezes the seven unique scene-query families behind the six B0 and six B1 WP6 primary errors: five shared, one B0-only and one B1-only. These counts are computed from the current prediction artifacts (and supersede the inconsistent earlier handoff narrative). It is derived only from already-completed WP6 predictions and primary manifest; it performs no inference and uses no SAM2.

Open `WP6_B0_B1_ERROR_REVIEW.html`. A reviewer must inspect the RGB scene and instruction before revealing source/model coordinates, then complete `wp6_error_review_decisions.csv`. Use `review_origin=human` only for an actual independent human decision. AI-assisted review remains `ai_visual` and cannot be relabelled human-certified.

This is a targeted error-analysis review, not the full G1 data audit. It cannot by itself create `D_tabletop_clean_v1`, open B2, select B1 for Gazebo, or change the WP6 machine-source metrics.
""")
    manifest = {
        "status": "HUMAN_REVIEW_PENDING", "sam2_used": False,
        "purpose": "Verify the six B0 and six B1 WP6 primary errors before interpreting them as model failures.",
        "b0_error_count": len(errors["b0"]), "b1_error_count": len(errors["b1"]),
        "shared_error_count": len(errors["b0"] & errors["b1"]), "unique_review_count": len(rows),
        "by_error_category": dict(Counter(row["error_category"] for row in rows)),
        "not_human_certified": True, "not_training_eligible": True, "not_b2_eligible": True,
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in [CHALLENGE, *PREDICTIONS.values()]},
        "outputs": {},
    }
    for path in (queue, decisions, gallery, readme):
        manifest["outputs"][path.name] = sha256(path)
    (OUT / "error_review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
