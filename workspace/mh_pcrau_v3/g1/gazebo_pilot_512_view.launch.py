"""Gazebo pilot capture launch with a preregistered per-batch view pose.

The existing capture stack is reused; this wrapper only sets the camera
joint pose from MH_PCRAU_V3_VIEW_POSE, checked against the scenes YAML by the
capture node. It does not load a model or command manipulation.
"""

import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "ur3/ur3_perception/launch/roborefer_uq_capture.launch.py"


def generate_launch_description():
    value = json.loads(os.environ["MH_PCRAU_V3_VIEW_POSE"])
    if not isinstance(value, list) or len(value) != 6 or not all(isinstance(x, (int, float)) for x in value):
        raise ValueError("MH_PCRAU_V3_VIEW_POSE must contain six joint angles")
    spec = importlib.util.spec_from_file_location("frozen_uq_capture_launch_for_v3_pilot", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LOCKED_VIEW_JOINT_POSE = value
    return module.generate_launch_description()
