#!/usr/bin/env python3
"""Compare simpler referring prompts on one locked Gazebo RGB-D capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infer_and_render import extract_frozen_features, frozen_paths, predict_v2
from pcrau.policy import decide
from pcrau.utils import read_json


PROMPTS = (
    "Locate the red apple in the current wrist-camera RGB-D image. Return its center pixel as (x, y).",
    "Locate the apple to the left of the tomato soup can. Return its center pixel as (x, y).",
    "Locate the apple to the right of the mustard bottle. Return its center pixel as (x, y).",
    "Locate the tomato soup can in the current wrist-camera RGB-D image. Return its center pixel as (x, y).",
    "Locate the yellow mustard bottle in the current wrist-camera RGB-D image. Return its center pixel as (x, y).",
    "Locate the banana above the apple. Return its center pixel as (x, y).",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    rgb_path = run_dir / "capture/rgb_model_input.jpg"
    depth_path = run_dir / "capture/depth_relative_model_input.png"
    if not rgb_path.is_file() or not depth_path.is_file():
        raise FileNotFoundError("Need an existing synchronized Gazebo capture")
    freeze, checkpoint, calibrator_path, config_path = frozen_paths()
    features, feature_runtime = extract_frozen_features(rgb_path, depth_path)
    config = read_json(config_path)
    calibrator = read_json(calibrator_path)
    rows = []
    for prompt in PROMPTS:
        prediction, runtime = predict_v2(features, prompt, config, checkpoint,
                                         config_path, "easy_prompt_probe")
        decision = decide(prediction, calibrator)
        row = {"prompt": prompt, "candidate_xy": prediction["spatial"]["map_pixel_xy"],
               "action": decision["action"],
               "risk": decision["calibrated_grounding_risk"],
               "threshold": decision["risk_threshold"],
               "answerability": prediction["answerability_probabilities"],
               "source_scores": prediction["source_probabilities"],
               "runtime_ms": runtime["v2_forward_latency_ms"]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    output = run_dir / "easy_prompt_probe.json"
    output.write_text(json.dumps({"checkpoint_sha256": freeze["selected_checkpoint"]["model_sha256"],
                                  "feature_runtime": feature_runtime, "rows": rows},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"RESULTS={output}")


if __name__ == "__main__":
    main()
