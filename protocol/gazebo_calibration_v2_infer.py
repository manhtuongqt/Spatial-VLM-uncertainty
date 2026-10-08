#!/usr/bin/env python3
"""Run frozen B0 on Calibration-v2 without oracle access."""
from __future__ import annotations
import argparse
from pathlib import Path
import gazebo_train_uq_v1_infer as base
ROOT=Path(__file__).resolve().parents[1];PROTOCOL_ID="gazebo_calibration_v2"
def main():
 base.PROTOCOL_ID=PROTOCOL_ID;p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path,default=ROOT/"datasets/Gazebo_calibration_v2");p.add_argument("--output",type=Path,default=ROOT/"results/spatial_vlm_refspatial_v1/gazebo_calibration_v2/b0_predictions");p.add_argument("--base",type=Path,default=ROOT/"RoboRefer/models/RoboRefer-2B-SFT");a=p.parse_args();base.run(argparse.Namespace(dataset=a.dataset,output=a.output,base=a.base,adapter=None,model_id="b0",draws=3,max_new_tokens=40))
if __name__=="__main__":main()
