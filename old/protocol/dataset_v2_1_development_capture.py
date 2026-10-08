#!/usr/bin/python3
"""V2.1 development capture entry point with batch/resume enforcement.

The qualified V2 development persistence logic and the qualified V2.1 camera
motion are composed here without changing either historical implementation.
"""

from __future__ import annotations

import re

import dataset_v2_development_capture as batch_runtime
from dataset_v2_relation_repair_capture import RepairCapture


PROTOCOL_ID = "roborefer_dataset_v2_1_1_development_capture_400"
DECISION = "GO_DEVELOPMENT_V2_1_CAPTURE_400"
EXPECTED_CAPTURES = 800
CAPTURE_PATTERN = re.compile(r"^v211dev_family_[0-9]{6}__(?:clean|occlusion)_capture$")

# Rebind protocol policy only. The shared sensor writer, baseline launch files,
# URDF, controllers and Gazebo world remain unchanged.
batch_runtime.PROTOCOL_ID = PROTOCOL_ID
batch_runtime.DECISION = DECISION
batch_runtime.EXPECTED_CAPTURES = EXPECTED_CAPTURES
batch_runtime.CAPTURE_PATTERN = CAPTURE_PATTERN
batch_runtime.shared.PROTOCOL_ID = PROTOCOL_ID
batch_runtime.shared.CAPTURE_ID = CAPTURE_PATTERN
batch_runtime.shared.PilotCapture = RepairCapture


def main() -> int:
    return batch_runtime.main()


if __name__ == "__main__":
    raise SystemExit(main())
