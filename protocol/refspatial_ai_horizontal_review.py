#!/usr/bin/env python3
"""Create visual review sheets for the fresh horizontal-ranking queue.

The sheets deliberately show the source target and a model prediction after
the question. They support an attributed AI visual audit; they are not a
replacement for independent human review and never make a clean release.
"""
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/spatial_vlm_refspatial_v1/wp2_visual_review"
QUEUE = OUT / "next_audit_queue.jsonl"
PREDICTIONS = OUT / "ai_triage_roborefer_2b_sft/predictions.jsonl"
SHEETS = OUT / "ai_visual_review_sheets_v2"


def font(size):
    for candidate in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def wrap(draw, text, width, style):
    words, lines, current = text.split(), [], ""
    for word in words:
        next_text = f"{current} {word}".strip()
        if draw.textlength(next_text, font=style) > width and current:
            lines.append(current); current = word
        else:
            current = next_text
    if current:
        lines.append(current)
    return lines


def dot(draw, xy, radius, colour, label, label_font):
    x, y = xy
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="white", width=4)
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=colour, width=2)
    draw.text((x+radius+3, y-radius-2), label, font=label_font, fill=colour, stroke_width=2, stroke_fill="black")


def main():
    rows = [json.loads(line) for line in QUEUE.read_text().splitlines()]
    predictions = {json.loads(line)["sample_id"]: json.loads(line) for line in PREDICTIONS.read_text().splitlines()}
    SHEETS.mkdir(exist_ok=True)
    tile_w, tile_h, banner_h = 960, 360, 54
    heading, body, tiny = font(21), font(16), font(14)
    manifest = []
    for page_no in range(math.ceil(len(rows) / 10)):
        page_rows = rows[page_no*10:(page_no+1)*10]
        canvas = Image.new("RGB", (tile_w * 2, banner_h + tile_h * 5), "#202124")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 14), f"AI visual review v2 — sheet {page_no+1:02d}; red=S source target, cyan=M RoboRefer prediction", font=heading, fill="white")
        metadata = []
        for i, row in enumerate(page_rows):
            col, line = i % 2, i // 2
            x0, y0 = col * tile_w, banner_h + line * tile_h
            image = Image.open(ROOT / row["image"]).convert("RGB")
            image.thumbnail((tile_w, 268), getattr(Image, "LANCZOS", Image.ANTIALIAS))
            x_img = x0 + (tile_w - image.width) // 2
            canvas.paste(image, (x_img, y0))
            tile = ImageDraw.Draw(canvas)
            target = row["target_xy"]
            dot(tile, (x_img + target[0] * image.width, y0 + target[1] * image.height), 8, "#ff4040", "S", tiny)
            pred = predictions.get(row["sample_id"], {}).get("prediction_xy")
            if pred:
                dot(tile, (x_img + pred[0] * image.width, y0 + pred[1] * image.height), 8, "#22e6e6", "M", tiny)
            text_y = y0 + 273
            title = f"{page_no*10+i+1:03d} {row['sample_id']} | {row['audit_strata'][0]} | {row['split']}"
            tile.text((x0+8, text_y), title, font=tiny, fill="#d4e3ff")
            question = row["instruction"].split("Your answer should")[0].strip()
            for j, text in enumerate(wrap(tile, question, tile_w-16, body)[:3]):
                tile.text((x0+8, text_y+18+j*18), text, font=body, fill="white")
            metadata.append({"sample_id": row["sample_id"], "source_xy": target, "prediction_xy": pred, "instruction": question})
        out = SHEETS / f"sheet_{page_no+1:02d}.jpg"
        canvas.save(out, quality=94)
        (SHEETS / f"sheet_{page_no+1:02d}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest.append({"sheet": out.name, "samples": [r["sample_id"] for r in page_rows]})
    (SHEETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sheets": len(manifest), "samples": len(rows), "output": str(SHEETS)}, indent=2))


if __name__ == "__main__":
    main()
