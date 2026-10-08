#!/usr/bin/env python3
"""Run one real RGB and one real RGB-D query against a loaded RoboRefer API."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = ROOT / "results/roborefer_pilot_v0_20260813_173305"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_point(answer: str):
    try:
        value = ast.literal_eval(answer.strip())
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, list) or len(value) != 1:
        return None
    point = value[0]
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in point):
        return None
    x, y = float(point[0]), float(point[1])
    return [x, y] if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 else None


def query(url: str, image: Path, depth: Path, prompt: str, enable_depth: bool, timeout: float) -> dict:
    body = {
        "image_url": [base64.b64encode(image.read_bytes()).decode("ascii")],
        "depth_url": [base64.b64encode(depth.read_bytes()).decode("ascii")] if enable_depth else [],
        "enable_depth": 1 if enable_depth else 0,
        "text": prompt,
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/query",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        status = error.code
        payload = json.loads(error.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - started) * 1000.0
    point = one_point(str(payload.get("answer", "")))
    return {
        "mode": "RGB-D" if enable_depth else "RGB",
        "http_status": status,
        "latency_ms": round(latency_ms, 3),
        "answer": payload.get("answer"),
        "normalized_point_xy": point,
        "generation_mode": payload.get("generation_mode"),
        "generation_config": payload.get("generation_config"),
        "model_fingerprint": payload.get("model_fingerprint"),
        "random_seed": payload.get("random_seed"),
        "passed": status == 200 and payload.get("result") == 1 and point is not None,
        "error": payload.get("error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:25548")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--output", type=Path, default=ROOT / "protocol/api_smoke_result.json")
    args = parser.parse_args()
    scene = DEFAULT_PILOT / "dataset_attempt_02/pilot_scene_0008"
    image = scene / "input/rgb.png"
    depth = DEFAULT_PILOT / "predictions/pilot_scene_0008/request_depth_registered_view.png"
    metadata = json.loads((scene / "input/scene_input.json").read_text(encoding="utf-8"))
    prompt = metadata["instruction"] + " " + metadata["coordinate_suffix"]
    results = [
        query(args.url, image, depth, prompt, False, args.timeout),
        query(args.url, image, depth, prompt, True, args.timeout),
    ]
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_url": args.url,
        "server_configuration_expected": {
            "skip_depth_model": True,
            "generation_mode": "greedy",
            "seed": 8132026,
        },
        "scene_id": "pilot_scene_0008",
        "inputs": {
            "rgb": str(image.relative_to(ROOT)),
            "rgb_sha256": sha256_file(image),
            "registered_depth_view": str(depth.relative_to(ROOT)),
            "registered_depth_view_sha256": sha256_file(depth),
        },
        "queries": results,
        "all_passed": all(item["passed"] for item in results),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

