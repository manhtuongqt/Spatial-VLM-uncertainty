#!/usr/bin/env python3
"""Generate concrete WP2 tables, figures, previews and the offline QC GUI."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp2_common import file_inventory, read_json, sha256_file, utc_now, write_json  # noqa: E402


COLORS = {
    "navy": (48, 42, 27), "blue": (214, 132, 45), "cyan": (205, 171, 61),
    "green": (95, 166, 67), "amber": (41, 164, 239), "red": (69, 72, 220),
    "gray": (105, 105, 105), "light": (247, 247, 244), "white": (255, 255, 255),
}


def put(canvas, text, xy, scale=0.65, color=COLORS["navy"], thickness=1):
    cv2.putText(canvas, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def horizontal_bars(title: str, labels: list[str], values: list[float], colors: list[tuple], output: Path, suffix=""):
    width, height = 1400, max(520, 150 + 65 * len(labels))
    canvas = np.full((height, width, 3), COLORS["light"], dtype=np.uint8)
    put(canvas, title, (55, 65), 1.05, COLORS["navy"], 2)
    max_value = max(values) if values else 1.0
    y0, bar_height = 115, 35
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = y0 + index * 65
        put(canvas, label, (55, y + 26), 0.60)
        x0, x1 = 430, 1260
        cv2.rectangle(canvas, (x0, y), (x1, y + bar_height), (225, 225, 220), -1)
        endpoint = x0 + int((x1 - x0) * value / max(max_value, 1e-9))
        cv2.rectangle(canvas, (x0, y), (endpoint, y + bar_height), color, -1)
        put(canvas, f"{value:g}{suffix}", (min(endpoint + 12, 1280), y + 27), 0.58, COLORS["navy"], 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def overlay_preview(root: Path, record: dict, output: Path) -> dict:
    rgb = cv2.imread(str(root / record["inference_payload"]["rgb_model_input"]["path"]), cv2.IMREAD_COLOR)
    masks = record["evaluator_only"]["masks"]
    target = cv2.imread(str(root / masks["target"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
    interior = cv2.imread(str(root / masks["target_interior"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
    reachable = cv2.imread(str(root / masks["reachable"]["path"]), cv2.IMREAD_GRAYSCALE) > 0
    anchor = np.zeros(target.shape, dtype=bool)
    for item in masks["anchor"]:
        anchor |= cv2.imread(str(root / item["path"]), cv2.IMREAD_GRAYSCALE) > 0
    overlay = rgb.copy()
    tint = np.zeros_like(overlay)
    tint[target] = (80, 190, 80)
    tint[anchor] = (225, 150, 40)
    overlay = cv2.addWeighted(overlay, 0.78, tint, 0.35, 0)
    for mask, color in ((interior, (70, 230, 90)), (reachable, (60, 60, 240))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
    preview = cv2.resize(overlay, (384, 288), interpolation=cv2.INTER_AREA)
    header = np.full((64, 384, 3), (35, 39, 49), dtype=np.uint8)
    put(header, record["family_id"], (12, 24), 0.48, COLORS["white"], 1)
    state = record["evaluator_only"]["uncertainty_label"]["state"]
    put(header, f"{record['variant']} | {state}", (12, 50), 0.43, (220, 226, 233), 1)
    combined = np.vstack([header, preview])
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), combined, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return {
        "target_pixels": int(np.count_nonzero(target)),
        "interior_pixels": int(np.count_nonzero(interior)),
        "reachable_pixels": int(np.count_nonzero(reachable)),
    }


def make_montage(images: list[tuple[str, Path]], output: Path, columns=3):
    tiles = []
    for label, path in images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (384, 352), interpolation=cv2.INTER_AREA)
        strip = np.full((45, 384, 3), (248, 248, 245), dtype=np.uint8)
        put(strip, label, (10, 29), 0.48, COLORS["navy"], 1)
        tiles.append(np.vstack([strip, image]))
    rows = []
    for start in range(0, len(tiles), columns):
        group = tiles[start:start + columns]
        while len(group) < columns:
            group.append(np.full_like(tiles[0], 245))
        rows.append(np.hstack(group))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))


def generate(dataset_root: Path) -> dict:
    assets = dataset_root / "report_assets"
    figures = assets / "figures"
    tables = assets / "tables"
    previews = assets / "gui_previews"
    index = read_json(dataset_root / "dataset_index.json")
    manifest = read_json(dataset_root / "family_manifest.json")
    gate = read_json(assets / "checkpoints" / "05_full_gate_report.json")
    records = [read_json(dataset_root / item["record_path"]) for item in index["records"]]
    families = manifest["families"]

    family_rows = []
    category_counts = Counter(item["category"] for item in families)
    for category in category_counts:
        selected = [item for item in families if item["category"] == category]
        split_counts = Counter(item["split"] for item in selected)
        family_rows.append({
            "category": category, "depth_dependent": selected[0]["depth_dependent"],
            "families": len(selected), "samples": len(selected) * 5,
            **{split: split_counts[split] for split in ("train", "dev", "calibration", "test")},
        })
    write_csv(
        tables / "family_distribution.csv",
        ["category", "depth_dependent", "families", "samples", "train", "dev", "calibration", "test"],
        family_rows,
    )
    split_rows = []
    for split in ("train", "dev", "calibration", "test"):
        selected = [item for item in records if item["split"] == split]
        variants = Counter(item["variant"] for item in selected)
        split_rows.append({
            "split": split, "families": len({item["family_id"] for item in selected}),
            "samples": len(selected), **{variant: variants[variant] for variant in manifest["variants"]},
        })
    write_csv(tables / "split_distribution.csv", ["split", "families", "samples", *manifest["variants"]], split_rows)

    qc_by_sample = {item["sample_id"]: item for item in gate["mask_qc"]["per_sample"]}
    mask_rows = []
    for record in records:
        item = qc_by_sample[record["sample_id"]]
        mask_rows.append({
            "sample_id": record["sample_id"], "family_id": record["family_id"],
            "split": record["split"], "category": record["family_category"],
            "variant": record["variant"], "state": record["evaluator_only"]["uncertainty_label"]["state"],
            "target_pixels": item["target_pixels"], "interior_pixels": item["interior_pixels"],
            "graspable_pixels": item["graspable_pixels"], "reachable_pixels": item["reachable_pixels"],
            "valid_depth_fraction": item["valid_depth_fraction"],
        })
    write_csv(tables / "mask_qc_per_sample.csv", list(mask_rows[0]), mask_rows)
    relation_rows = []
    for item in gate["relation_qc"]["per_sample"]:
        relation_rows.append({
            "sample_id": item["sample_id"], "status": item["status"],
            "relation": item.get("relation", ""),
            "target_median_depth_m": item.get("target_median_depth_m", ""),
            "anchor_median_depth_m": json.dumps(item.get("anchor_median_depth_m", [])),
        })
    write_csv(tables / "relation_qc_per_sample.csv", list(relation_rows[0]), relation_rows)

    capture_rows = []
    for capture_dir in sorted((dataset_root / "raw" / "captures").iterdir()):
        meta = read_json(capture_dir / "capture_meta.json")
        capture_rows.append({
            "capture_id": meta["capture_id"], "family_id": meta["family_id"],
            "condition": meta["condition"],
            "max_spread_sec": meta["capture_timestamps"]["max_spread_sec"],
            "rgb_gray_std": meta["sensor_qc"]["rgb_gray_std"],
            "valid_depth_fraction": meta["sensor_qc"]["valid_depth_fraction"],
            "valid_depth_min_m": meta["sensor_qc"]["valid_depth_min_m"],
            "valid_depth_max_m": meta["sensor_qc"]["valid_depth_max_m"],
        })
    write_csv(tables / "capture_qc.csv", list(capture_rows[0]), capture_rows)

    depth_categories = {
        "nearer_farther", "front_behind_camera", "multi_anchor_depth_order", "occlusion_depth_evidence"
    }
    horizontal_bars(
        "WP2 family distribution (50 families / 250 samples)",
        [row["category"] for row in family_rows], [row["families"] for row in family_rows],
        [COLORS["blue"] if row["category"] in depth_categories else COLORS["green"] for row in family_rows],
        figures / "family_category_distribution.png",
    )
    horizontal_bars(
        "Family-level split distribution", [row["split"] for row in split_rows],
        [row["families"] for row in split_rows],
        [COLORS["blue"], COLORS["cyan"], COLORS["amber"], COLORS["red"]],
        figures / "split_distribution.png",
    )
    intervention_counts = Counter(
        item["evaluator_only"]["uncertainty_label"]["expected_intervention"] for item in records
    )
    horizontal_bars(
        "Expected intervention labels", list(intervention_counts), list(intervention_counts.values()),
        [COLORS["green"], COLORS["blue"], COLORS["amber"], COLORS["red"], COLORS["gray"]][:len(intervention_counts)],
        figures / "intervention_distribution.png",
    )
    variant_target = defaultdict(list)
    for row in mask_rows:
        variant_target[row["variant"]].append(float(row["target_pixels"]))
    horizontal_bars(
        "Mean visible target-mask area by variant",
        list(manifest["variants"]),
        [round(float(np.mean(variant_target[item])), 1) for item in manifest["variants"]],
        [COLORS["green"], COLORS["blue"], COLORS["cyan"], COLORS["amber"], COLORS["red"]],
        figures / "mean_target_mask_area.png", suffix=" px",
    )

    gui_records = []
    preview_map = {}
    for record in records:
        preview_path = previews / f"{record['sample_id']}.jpg"
        preview_map[record["sample_id"]] = preview_path
        areas = overlay_preview(dataset_root, record, preview_path)
        label = record["evaluator_only"]["uncertainty_label"]
        gui_records.append({
            "sample_id": record["sample_id"], "family_id": record["family_id"],
            "split": record["split"], "category": record["family_category"],
            "variant": record["variant"], "depth_dependent": record["depth_dependent"],
            "instruction": record["instruction"], "state": label["state"],
            "intervention": label["expected_intervention"], "sources": label["sources"],
            "preview": str(preview_path.relative_to(dataset_root)),
            "rgb": record["inference_payload"]["rgb_model_input"]["path"],
            "depth": record["inference_payload"]["depth_relative_model_input"]["path"],
            "target_mask": record["evaluator_only"]["masks"]["target"]["path"],
            "record": next(item["record_path"] for item in index["records"] if item["sample_id"] == record["sample_id"]),
            **areas,
        })
    representative = []
    for category in category_counts:
        record = next(item for item in records if item["family_category"] == category and item["variant"] == "clean")
        representative.append((category, preview_map[record["sample_id"]]))
    make_montage(representative, figures / "representative_family_montage.png")
    corruption_examples = []
    for severity in (1, 2, 3):
        record = next(
            item for item in records
            if item["variant"] == "depth_corruption"
            and item["evaluator_only"]["uncertainty_label"]["severity"] == severity
        )
        corruption_examples.append((f"severity {severity}: {record['family_id']}", preview_map[record["sample_id"]]))
    make_montage(corruption_examples, figures / "depth_corruption_examples.png", columns=3)

    history = np.full((760, 1500, 3), COLORS["light"], dtype=np.uint8)
    put(history, "WP2 QC gate history (before training)", (55, 70), 1.05, COLORS["navy"], 2)
    events = [
        ("Attempt 1", "REJECT", "FOUND sample had empty target mask", COLORS["red"]),
        ("Attempt 2", "REJECT", "static audit found ID/pose collision risk", COLORS["red"]),
        ("Attempt 3 smoke", "PASS", "3 families x 5 variants; replay/leakage pass", COLORS["green"]),
        ("Full capture", "PASS", "100 raw Gazebo captures hashed", COLORS["green"]),
        ("Full QC", "AMEND", "anchor + relation labels corrected; split/layout unchanged", COLORS["amber"]),
        ("Final gate", "PASS", "50 families / 250 samples / 30 depth-dependent", COLORS["green"]),
    ]
    for index_event, (name, status, detail, color) in enumerate(events):
        y = 135 + index_event * 95
        cv2.circle(history, (95, y), 16, color, -1)
        if index_event + 1 < len(events):
            cv2.line(history, (95, y + 16), (95, y + 79), (170, 170, 165), 3)
        put(history, name, (135, y - 4), 0.66, COLORS["navy"], 2)
        put(history, status, (360, y - 4), 0.60, color, 2)
        put(history, detail, (510, y - 4), 0.58, COLORS["gray"], 1)
    cv2.imwrite(str(figures / "qc_gate_history.png"), history)

    summary = {
        "families": 50, "samples": 250, "variants_per_family": 5,
        "depth_dependent_families": 30, "raw_gazebo_captures": 100,
        "splits": {row["split"]: {"families": row["families"], "samples": row["samples"]} for row in split_rows},
        "family_categories": dict(category_counts),
        "answerability_states": dict(Counter(item["evaluator_only"]["uncertainty_label"]["state"] for item in records)),
        "expected_interventions": dict(intervention_counts),
        "replay_files_compared": gate["replay_validation"]["generated_files_compared"],
        "relation_records_checked": gate["relation_qc"]["projective_records_checked"],
        "mean_valid_depth_fraction": gate["mask_qc"]["mean_valid_depth_fraction"],
    }
    write_json(tables / "summary_numbers.json", summary)

    payload = json.dumps(gui_records, ensure_ascii=False).replace("</", "<\\/")
    gui = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WP2 Dataset v1 QC</title><style>
:root{{--ink:#172033;--muted:#657084;--bg:#f4f6f8;--card:#fff;--blue:#2d78b7;--green:#39915a;--amber:#d88b24}}
*{{box-sizing:border-box}}body{{margin:0;font:14px system-ui,sans-serif;color:var(--ink);background:var(--bg)}}
header{{background:#172033;color:white;padding:24px 32px}}header h1{{margin:0 0 7px;font-size:26px}}header p{{margin:0;color:#cbd4e2}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}}.badge{{padding:7px 11px;background:#293650;border-radius:999px}}
main{{padding:22px 28px;max-width:1500px;margin:auto}}.controls{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:12px;background:white;padding:16px;border-radius:12px;box-shadow:0 2px 12px #17203312}}
label{{color:var(--muted);font-size:12px}}select{{width:100%;padding:9px;margin-top:5px;border:1px solid #ccd3dc;border-radius:7px;background:white}}
.notice{{margin:16px 0;padding:12px 15px;background:#fff7e7;border-left:4px solid var(--amber);border-radius:7px}}
#cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:15px}}
.card{{background:var(--card);border-radius:12px;overflow:hidden;box-shadow:0 2px 12px #17203315}}.card img{{width:100%;display:block;background:#ddd}}
.body{{padding:13px}}.variant{{font-weight:700;color:var(--blue)}}.state{{float:right;padding:3px 7px;border-radius:10px;background:#e8f4ec;color:var(--green);font-size:11px}}
.prompt{{min-height:54px;line-height:1.42;margin:10px 0;color:#364155}}.meta{{font-size:12px;color:var(--muted);line-height:1.55}}
.links a{{display:inline-block;margin:9px 9px 0 0;color:var(--blue);text-decoration:none}}@media(max-width:800px){{.controls{{grid-template-columns:1fr 1fr}}main{{padding:14px}}}}
</style></head><body><header><h1>WP2 — RoboRefer Dataset v1 QC</h1>
<p>Evaluator view. Oracle masks shown here are never exported in inference_payload.</p>
<div class="stats"><span class="badge">50 families</span><span class="badge">250 samples</span><span class="badge">30 depth-dependent</span><span class="badge">100 Gazebo captures</span><span class="badge">Replay 100%</span></div></header>
<main><section class="controls"><div><label>Split<select id="split"></select></label></div><div><label>Category<select id="category"></select></label></div><div><label>Family<select id="family"></select></label></div><div><label>Variant<select id="variant"></select></label></div></section>
<div class="notice">Legend: target tint/outline = green; anchor tint = blue; reachable outline = red. Masks and labels below are evaluator-only.</div><section id="cards"></section></main>
<script>const DATA={payload};const order={json.dumps(manifest['variants'])};
const split=document.querySelector('#split'),category=document.querySelector('#category'),family=document.querySelector('#family'),variant=document.querySelector('#variant'),cards=document.querySelector('#cards');
function options(el,values,all){{el.innerHTML=(all?'<option value="">All</option>':'')+values.map(v=>`<option>${{v}}</option>`).join('')}}
options(split,[...new Set(DATA.map(x=>x.split))],true);options(category,[...new Set(DATA.map(x=>x.category))],true);options(variant,order,true);
function refreshFamilies(){{let d=DATA.filter(x=>(!split.value||x.split===split.value)&&(!category.value||x.category===category.value));let old=family.value;options(family,[...new Set(d.map(x=>x.family_id))],false);if([...family.options].some(o=>o.value===old))family.value=old;render()}}
function render(){{let d=DATA.filter(x=>x.family_id===family.value&&(!variant.value||x.variant===variant.value)).sort((a,b)=>order.indexOf(a.variant)-order.indexOf(b.variant));cards.innerHTML=d.map(x=>`<article class="card"><img loading="lazy" src="${{x.preview}}"><div class="body"><span class="variant">${{x.variant}}</span><span class="state">${{x.state}}</span><div class="prompt">${{x.instruction}}</div><div class="meta">split: ${{x.split}} · category: ${{x.category}}<br>intervention: ${{x.intervention}} · sources: ${{x.sources.join(', ')||'none'}}<br>target/interior/reachable: ${{x.target_pixels}} / ${{x.interior_pixels}} / ${{x.reachable_pixels}} px</div><div class="links"><a href="${{x.rgb}}">RGB input</a><a href="${{x.depth}}">Depth input</a><a href="${{x.target_mask}}">Target mask</a><a href="${{x.record}}">Record JSON</a></div></div></article>`).join('')}}
split.onchange=category.onchange=refreshFamilies;family.onchange=variant.onchange=render;refreshFamilies();
</script></body></html>"""
    (dataset_root / "WP2_QC_GUI.html").write_text(gui, encoding="utf-8")

    readme = """# WP2 report assets\n\nThis folder is deliberately separate from training/inference payloads. It contains report-ready figures, CSV/JSON tables, QC previews, and chronological checkpoints. `gui_previews/` overlays evaluator-only masks and must never be passed to RoboRefer.\n\nReplay means deterministic offline rematerialization from locked Gazebo captures; it does not claim bitwise-identical Gazebo rerendering.\n"""
    (assets / "README.md").write_text(readme, encoding="utf-8")
    write_json(assets / "checkpoints" / "06_report_assets_generated.json", {
        "schema_version": 1, "checkpoint": "REPORT_ASSETS_GENERATED",
        "created_at_utc": utc_now(), "figure_count": len(list(figures.glob("*.png"))),
        "table_count": len(list(tables.glob("*"))), "gui_preview_count": len(list(previews.glob("*.jpg"))),
        "gui_sha256": sha256_file(dataset_root / "WP2_QC_GUI.html"),
        "generator_source_sha256": sha256_file(Path(__file__).resolve()),
    })
    asset_inventory = file_inventory(assets, excluded_names={"ASSET_INDEX.json"})
    write_json(assets / "ASSET_INDEX.json", {
        "schema_version": 1, "self_hash_policy": "ASSET_INDEX.json excluded to avoid recursive hash",
        "artifact_count": len(asset_inventory), "artifacts": asset_inventory,
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    summary = generate(root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
