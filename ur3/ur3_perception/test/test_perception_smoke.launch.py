import json
import os
import unittest

import launch
import launch_testing
import launch_testing.actions
import pytest
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource


@pytest.mark.launch_test
def generate_test_description():
    metrics = f"/tmp/ur3_perception_smoke_{os.getpid()}.json"
    demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ur3_perception"),
                "launch", "perception_demo.launch.py",
            )
        ),
        launch_arguments={
            "gazebo_gui": "false",
            "launch_rviz": "false",
            "image_view": "false",
            "shutdown_on_completion": "true",
            "metrics_output": metrics,
        }.items(),
    )
    return (
        launch.LaunchDescription([
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(120 + os.getpid() % 80)),
            SetEnvironmentVariable("IGN_PARTITION", f"ur3_perception_smoke_{os.getpid()}"),
            demo,
            launch_testing.actions.ReadyToTest(),
        ]),
        {"metrics": metrics},
    )


class TestPerceptionRuntime(unittest.TestCase):
    def test_demo_reaches_done(self, proc_output):
        proc_output.assertWaitFor(
            "DONE: RGB-D perception demo validated against Gazebo oracle",
            timeout=160,
        )


@launch_testing.post_shutdown_test()
class TestPerceptionMetrics(unittest.TestCase):
    def test_oracle_error_is_bounded(self, metrics):
        with open(metrics, encoding="utf-8") as stream:
            report = json.load(stream)
        self.assertTrue(report["success"])
        self.assertEqual(report["samples"], 10)
        self.assertLess(report["max_error_m"], 0.02)
        self.assertEqual(report["latest"]["status"], "STABLE")
        self.assertGreaterEqual(report["latest"]["valid_frames"], 7)
        self.assertEqual(len(report["latest"]["dimensions_m"]), 3)
