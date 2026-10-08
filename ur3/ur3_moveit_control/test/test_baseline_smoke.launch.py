"""End-to-end smoke test for one repeatable UR3 baseline episode."""

import json
import os
import unittest
from pathlib import Path

import launch
import launch_testing
import launch_testing.actions
import pytest
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


@pytest.mark.launch_test
def generate_test_description():
    run_id = os.getpid()
    metrics_path = Path(f"/tmp/ur3_baseline_smoke_{run_id}.json")
    baseline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("ur3_moveit_control"),
                    "launch",
                    "baseline_fixed_pick_place.launch.py",
                ]
            )
        ),
        launch_arguments={
            "ur_type": "ur3",
            "launch_rviz": "false",
            "gazebo_gui": "false",
            "run_script": "true",
            "episode_count": "1",
            "metrics_output": str(metrics_path),
            "stop_on_failure": "true",
            "shutdown_on_completion": "true",
        }.items(),
    )
    return (
        launch.LaunchDescription(
            [
                SetEnvironmentVariable(
                    "ROS_DOMAIN_ID", str(120 + run_id % 50)
                ),
                SetEnvironmentVariable(
                    "IGN_PARTITION", f"ur3_baseline_smoke_{run_id}"
                ),
                baseline,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"metrics_path": metrics_path},
    )


class TestBaselineRuntime(unittest.TestCase):
    def test_runner_reaches_done(self, proc_output):
        proc_output.assertWaitFor(
            "DONE: repeatable fixed pick-and-place benchmark completed",
            timeout=280,
        )


@launch_testing.post_shutdown_test()
class TestBaselineMetrics(unittest.TestCase):
    def test_metrics_report_success(self, metrics_path):
        assert metrics_path.exists()
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert payload["summary"]["robot_type"] == "ur3"
        assert payload["summary"]["completed_episodes"] == 1
        assert payload["summary"]["successful_episodes"] == 1
        assert payload["summary"]["success_rate"] == 1.0
        assert payload["episodes"][0]["planning_time_sec"] >= 0.0
        assert payload["episodes"][0]["execution_time_sec"] > 0.0
