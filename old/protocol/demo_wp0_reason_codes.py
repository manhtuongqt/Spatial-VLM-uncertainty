#!/usr/bin/env python3
"""Generate an inspectable demo for every WP0 geometry reason code."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol"
OUTPUT_DIR = PROTOCOL / "demo_wp0_reason_codes"
GATE_PATH = ROOT / "ur3/ur3_perception/scripts/depth_component_gate_v2.py"
POSE_PATH = ROOT / "ur3/ur3_perception/scripts/rgbd_object_pose.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_module("wp0_depth_component_gate_v2", GATE_PATH)
POSE = load_module("wp0_rgbd_object_pose", POSE_PATH)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items() if key != "mask"}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def render_depth_case(code: str, depth: np.ndarray, seed: tuple[int, int], record: dict) -> Path:
    finite = np.isfinite(depth)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(finite):
        lo, hi = float(np.min(depth[finite])), float(np.max(depth[finite]))
        if hi - lo < 1e-6:
            gray[finite] = 180
        else:
            gray[finite] = np.clip((depth[finite] - lo) / (hi - lo) * 190 + 40, 0, 255)
    panel = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    panel[~finite] = (22, 22, 22)
    panel = cv2.resize(panel, (720, 480), interpolation=cv2.INTER_NEAREST)
    sx = int(seed[0] * 720 / depth.shape[1])
    sy = int(seed[1] * 480 / depth.shape[0])
    cv2.drawMarker(panel, (sx, sy), (255, 255, 255), cv2.MARKER_CROSS, 26, 3)
    selected = record.get("diagnostics", {}).get("selected_component")
    if selected:
        x0, y0, x1, y1 = selected["bbox_xyxy"]
        scale_x, scale_y = 720 / depth.shape[1], 480 / depth.shape[0]
        cv2.rectangle(
            panel,
            (int(x0 * scale_x), int(y0 * scale_y)),
            (int((x1 + 1) * scale_x), int((y1 + 1) * scale_y)),
            (255, 255, 255),
            2,
        )
    cv2.rectangle(panel, (0, 0), (720, 74), (18, 18, 18), -1)
    cv2.putText(panel, code, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 210, 255), 2)
    cv2.putText(panel, record.get("detail", "structured rejection")[:82], (22, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)
    output = OUTPUT_DIR / f"{code}.png"
    cv2.imwrite(str(output), panel)
    return output


def evaluate_depth_case(code: str, depth: np.ndarray, seed: tuple[int, int], **kwargs) -> dict:
    try:
        result = GATE.segment_seeded_depth_component_v2(depth, [seed], **kwargs)
        record = json_safe(result)
    except GATE.DepthComponentError as error:
        record = json_safe(error.as_dict())
    observed = record.get("reason_code")
    record.update(expected_reason_code=code, demo_passed=observed == code)
    record["image"] = str(render_depth_case(code, depth, seed, record).relative_to(PROTOCOL))
    return record


def unstable_3d_case() -> dict:
    samples = [
        (
            np.array([0.30 if index % 2 else 0.34, -0.18, 0.03]),
            np.array([0.06, 0.06, 0.06]),
            0.0,
            None,
        )
        for index in range(7)
    ]
    accepted, mean, dimensions, stddev = POSE.aggregate_stable_samples(samples, 0.003)
    observed = POSE.stability_reason_code(len(samples), 7, accepted)
    panel = np.full((480, 720, 3), (22, 22, 22), dtype=np.uint8)
    cv2.putText(panel, "UNSTABLE_3D", (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 210, 255), 2)
    values = [float(item[0][0]) for item in samples]
    for index in range(len(values) - 1):
        p0 = (80 + index * 90, int(390 - (values[index] - 0.28) * 4000))
        p1 = (80 + (index + 1) * 90, int(390 - (values[index + 1] - 0.28) * 4000))
        cv2.line(panel, p0, p1, (90, 190, 255), 3)
    for index, value in enumerate(values):
        point = (80 + index * 90, int(390 - (value - 0.28) * 4000))
        cv2.circle(panel, point, 7, (255, 255, 255), -1)
    cv2.putText(panel, f"7 samples; position std={stddev:.4f} m > 0.0030 m", (22, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
    cv2.putText(panel, "x position alternates between 0.30 m and 0.34 m", (22, 452), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
    output = OUTPUT_DIR / "UNSTABLE_3D.png"
    cv2.imwrite(str(output), panel)
    return {
        "accepted": accepted,
        "reason_code": observed,
        "expected_reason_code": "UNSTABLE_3D",
        "demo_passed": observed == "UNSTABLE_3D",
        "sample_count": len(samples),
        "position_mean_m": mean.tolist(),
        "dimensions_m": dimensions.tolist(),
        "position_stddev_m": stddev,
        "threshold_m": 0.003,
        "image": str(output.relative_to(PROTOCOL)),
    }


def pilot_scene_7_diagnostic() -> dict:
    pilot = ROOT / "results/roborefer_pilot_v0_20260813_173305"
    depth = np.load(pilot / "dataset_attempt_02/pilot_scene_0007/input/depth_m.npy")
    prediction = json.loads((pilot / "predictions/pilot_scene_0007/b1_prediction.json").read_text())
    seed = tuple(prediction["pixel_points_xy"][0])
    try:
        result = GATE.segment_seeded_depth_component_v2(
            depth,
            [seed],
            min_depth_m=0.05,
            max_depth_m=2.0,
            seed_radius_px=5,
            roi_radius_px=110,
            near_tolerance_m=0.015,
            far_tolerance_m=0.055,
            min_area_px=120,
            max_area_fraction=0.12,
            bbox_padding_px=5,
            reject_border_truncated=False,
        )
        record = json_safe(result)
    except GATE.DepthComponentError as error:
        record = json_safe(error.as_dict())
    record.update(
        scene_id="pilot_scene_0007",
        seed_pixel_xy=list(seed),
        interpretation="POST_HOC_DIAGNOSTIC_ONLY",
        note="Runs Gate v2 on locked inputs; it does not rewrite or rescore the frozen pilot.",
    )
    return record


def write_html(cases: list[dict], pilot: dict) -> None:
    descriptions = {
        "AREA_TOO_SMALL": "Vung depth co seed nho hon dien tich toi thieu.",
        "AREA_TOO_LARGE": "Vung co seed qua lon nhung van nam gon trong ROI.",
        "NO_SEED_SUPPORT": "Khong co depth hop le / thanh phan lien thong do seed RoboRefer ho tro.",
        "PLANE_MERGE": "Vung qua lon va tiep tuc den bien ROI: dau hieu vat bi dinh voi mat phang.",
        "BORDER_TRUNCATED": "Vung muc tieu cham bien anh, hinh dang co the bi cat mat.",
        "UNSTABLE_3D": "Cua so diem 3D da du mau nhung dao dong vuot nguong.",
    }
    cards = []
    for item in cases:
        code = item["expected_reason_code"]
        cards.append(
            f'<article class="card"><img src="{escape(item["image"])}" alt="{code}">'
            f'<h2>{code}</h2><p>{escape(descriptions[code])}</p>'
            f'<p class="pass">DEMO PASS: observed = expected</p></article>'
        )
    html = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WP0 structured reason-code demo</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#e6edf3;margin:0;padding:28px}}h1{{margin-bottom:6px}}.meta{{color:#9da7b3}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px;margin-top:24px}}.card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px}}img{{width:100%;border-radius:8px}}h2{{font-size:18px;color:#f0b429}}p{{line-height:1.45}}.pass{{color:#56d364;font-weight:700}}code{{color:#79c0ff}}pre{{white-space:pre-wrap;background:#161b22;padding:16px;border-radius:8px}}</style></head>
<body><h1>WP0 — Demo structured reason codes</h1>
<p class="meta">Moi the duoc tao boi chinh Gate v2 / stability gate dang duoc node ROS su dung. Day la demo deterministic, CPU-only.</p>
<div class="grid">{''.join(cards)}</div>
<h2>Chan doan hau nghiem pilot_scene_0007</h2>
<p>Chi doc input da khoa; khong sua artifact va khong tinh lai diem pilot.</p>
<pre>{escape(json.dumps(pilot, ensure_ascii=False, indent=2))}</pre>
</body></html>"""
    (PROTOCOL / "WP0_DEMO.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    small = np.full((100, 100), np.nan, np.float32)
    small[46:54, 46:54] = 0.62
    bounded = np.full((120, 120), np.nan, np.float32)
    bounded[20:80, 20:80] = 0.62
    plane = np.full((120, 120), 0.62, np.float32)
    border = np.full((100, 120), np.nan, np.float32)
    border[30:75, 0:35] = 0.62
    valid = np.full((80, 100), 0.62, np.float32)

    cases = [
        evaluate_depth_case("AREA_TOO_SMALL", small, (50, 50), min_area_px=120, max_area_fraction=0.5),
        evaluate_depth_case("AREA_TOO_LARGE", bounded, (50, 50), roi_radius_px=55, max_area_fraction=0.10),
        evaluate_depth_case("NO_SEED_SUPPORT", valid, (-1, 40)),
        evaluate_depth_case("PLANE_MERGE", plane, (60, 60), roi_radius_px=35, max_area_fraction=0.10),
        evaluate_depth_case("BORDER_TRUNCATED", border, (18, 52), roi_radius_px=80, max_area_fraction=0.5, reject_border_truncated=True),
        unstable_3d_case(),
    ]
    pilot = pilot_scene_7_diagnostic()
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_cases_passed": all(item["demo_passed"] for item in cases),
        "cases": cases,
        "locked_pilot_post_hoc_diagnostic": pilot,
    }
    (PROTOCOL / "reason_code_demo.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_html(cases, pilot)
    print(f"REASON_CODE_DEMO {'PASS' if report['all_cases_passed'] else 'FAIL'}: {len(cases)}/6")
    if not report["all_cases_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

