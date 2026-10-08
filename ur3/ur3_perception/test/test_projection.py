import importlib.util
from pathlib import Path

import numpy as np
from sensor_msgs.msg import CameraInfo


SCRIPT = Path(__file__).parents[1] / "scripts" / "rgbd_object_pose.py"
SPEC = importlib.util.spec_from_file_location("rgbd_object_pose", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def camera_model():
    info = CameraInfo()
    info.width, info.height = 640, 480
    info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
    info.p = [500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.distortion_model = "plumb_bob"
    model = MODULE.PinholeCameraModel()
    model.fromCameraInfo(info)
    return model


def test_pixel_projection_uses_camera_intrinsics():
    x, y, z = MODULE.project_pixel(420, 290, 1.0, camera_model())
    assert np.allclose([x, y, z], [0.2, 0.1, 1.0])


def test_median_depth_rejects_invalid_values():
    depth = np.array([[np.nan, 0.0, 0.8], [0.9, 1.0, 4.0], [1.1, 1.2, np.inf]])
    value = MODULE.valid_depth_median(depth, 1, 1, 1, 0.05, 2.0)
    assert value == 1.0


def test_stability_gate_accepts_low_variance_window():
    samples = [
        (np.array([0.32 + i * 0.0001, -0.18, 0.03]),
         np.array([0.06, 0.059, 0.06]), 0.0, None)
        for i in range(7)
    ]
    accepted, mean, dimensions, stddev = MODULE.aggregate_stable_samples(
        samples, 0.003
    )
    assert accepted
    assert np.allclose(mean[1:], [-0.18, 0.03])
    assert np.allclose(dimensions, [0.06, 0.059, 0.06])
    assert stddev < 0.003


def test_stability_gate_rejects_excessive_variance():
    samples = [
        (np.array([0.30 if i % 2 else 0.34, -0.18, 0.03]),
         np.array([0.06, 0.06, 0.06]), 0.0, None)
        for i in range(7)
    ]
    accepted, _, _, stddev = MODULE.aggregate_stable_samples(samples, 0.003)
    assert not accepted
    assert stddev > 0.003
    assert MODULE.stability_reason_code(7, 7, accepted) == "UNSTABLE_3D"


def test_stability_reason_is_not_emitted_before_decision_or_after_acceptance():
    assert MODULE.stability_reason_code(6, 7, False) is None
    assert MODULE.stability_reason_code(7, 7, True) is None
