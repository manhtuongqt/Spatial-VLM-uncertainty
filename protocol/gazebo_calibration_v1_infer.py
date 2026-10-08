#!/usr/bin/env python3
"""Run the frozen B0 inference contract on sealed Calibration inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import gazebo_train_uq_v1_infer as base


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_calibration_v1"


def main():
    base.PROTOCOL_ID = PROTOCOL_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/Gazebo_calibration_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v1/b0_predictions")
    parser.add_argument("--base", type=Path, default=ROOT / "RoboRefer/models/RoboRefer-2B-SFT")
    args = parser.parse_args()
    base.run(argparse.Namespace(dataset=args.dataset, output=args.output, base=args.base,
                                adapter=None, model_id="b0", draws=3, max_new_tokens=40))


if __name__ == "__main__":
    main()
