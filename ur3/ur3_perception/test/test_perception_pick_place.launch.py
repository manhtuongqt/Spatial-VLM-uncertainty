"""Dynamic scene, attach/detach and perception pick/place launch test."""

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
    metrics_path = Path(f"/tmp/ur3_perception_pick_place_test_{run_id}.json")
    scenario = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur3_perception"),
            "launch",
            "perception_pick_place.launch.py",
        ])),
        launch_arguments={
            "gazebo_gui": "false",
            "launch_rviz": "false",
            "image_view": "false",
            "episode_count": "1",
            "randomize_object": "true",
            "random_seed": "11",
            "shutdown_on_completion": "true",
            "metrics_output": str(metrics_path),
        }.items(),
    )
    return (
        launch.LaunchDescription([
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(180 + run_id % 40)),
            SetEnvironmentVariable(
                "IGN_PARTITION", f"ur3_perception_pick_place_test_{run_id}"
            ),
            scenario,
            launch_testing.actions.ReadyToTest(),
        ]),
        {"metrics_path": metrics_path},
    )


class TestPerceptionPickPlaceRuntime(unittest.TestCase):
    def test_end_to_end_reaches_done(self, proc_output):
        proc_output.assertWaitFor(
            "DONE: perception-driven pick-and-place benchmark completed",
            timeout=320,
        )


@launch_testing.post_shutdown_test()
class TestPerceptionPickPlaceMetrics(unittest.TestCase):
    def _episode(self, metrics_path):
        self.assertTrue(metrics_path.exists())
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(report["oracle_usage"], "evaluation_only")
        return report, report["episodes"][0]

    def test_perception_pose_integration(self, metrics_path):
        report, episode = self._episode(metrics_path)
        self.assertEqual(report["summary"]["object_pose_source"], "perception")
        self.assertTrue(episode["perception_success"])
        self.assertIsNotNone(episode["frozen_pose"])
        self.assertLess(episode["perception_error_m"], 0.02)

    def test_dynamic_collision_object_update(self, metrics_path):
        _, episode = self._episode(metrics_path)
        self.assertTrue(episode["scene_update_success"])
        states = [step["state"] for step in episode["steps"]]
        self.assertIn("UPDATE_PLANNING_SCENE", states)
        self.assertIn("CLEAR_OBJECT_FOR_APPROACH", states)

    def test_moveit_and_gazebo_attach_detach(self, metrics_path):
        _, episode = self._episode(metrics_path)
        self.assertTrue(episode["attach_success"])
        self.assertTrue(episode["detach_success"])
        self.assertTrue(episode["grasp_success"])

    def test_end_to_end_metrics(self, metrics_path):
        report, episode = self._episode(metrics_path)
        self.assertTrue(episode["success"])
        self.assertTrue(episode["place_success"])
        self.assertEqual(report["summary"]["end_to_end_success_rate"], 1.0)
        self.assertEqual(episode["failure_stage"], "NONE")
