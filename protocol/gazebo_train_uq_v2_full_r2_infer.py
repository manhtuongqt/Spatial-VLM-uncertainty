#!/usr/bin/env python3
"""Blinded frozen-B0 inference entry point for full-v2-r2."""
from __future__ import annotations

import argparse
from pathlib import Path

import gazebo_train_uq_v1_infer as base

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_train_uq_v2_full_r2"


def main():
    base.PROTOCOL_ID = PROTOCOL_ID
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,default=ROOT/"datasets/Gazebo_train_uq_v2_full_r2")
    p.add_argument("--output",type=Path,default=ROOT/"results/spatial_vlm_refspatial_v1/wp5_spatial_uncertainty/b0_full_r2")
    p.add_argument("--base",type=Path,default=ROOT/"RoboRefer/models/RoboRefer-2B-SFT")
    a=p.parse_args()
    args=argparse.Namespace(dataset=a.dataset,output=a.output,base=a.base,adapter=None,
                            model_id="b0",draws=3,max_new_tokens=40)
    base.run(args)


if __name__=="__main__":main()
