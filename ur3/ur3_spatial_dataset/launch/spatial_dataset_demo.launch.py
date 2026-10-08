"""Visible perception pick/place demo with canonical spatial dataset recording."""

import time

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    dataset_root = LaunchConfiguration("dataset_root")
    run_id = LaunchConfiguration("run_id")
    random_seed = LaunchConfiguration("random_seed")
    episode_count = LaunchConfiguration("episode_count")
    default_run_id = time.strftime("run_%Y%m%d_%H%M%S")

    registry_file = PathJoinSubstitution([
        FindPackageShare("ur3_spatial_dataset"), "config", "objects.yaml"
    ])
    world_file = PathJoinSubstitution([
        FindPackageShare("ur_simulation_gz"), "worlds", "ur3_pick_place.sdf"
    ])
    control_config = PathJoinSubstitution([
        FindPackageShare("ur3_moveit_control"), "config", "moveit_custom_config.yaml"
    ])
    run_root = PathJoinSubstitution([dataset_root, run_id])

    spatial_scene = Node(
        package="ur3_spatial_dataset",
        executable="spatial_scene",
        name="spatial_scene",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "registry_file": registry_file,
        }],
    )
    recorder = Node(
        package="ur3_spatial_dataset",
        executable="dataset_recorder",
        name="dataset_recorder",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "dataset_root": dataset_root,
            "run_id": run_id,
            "random_seed": random_seed,
            "episode_count": episode_count,
            "registry_file": registry_file,
            "world_file": world_file,
            "control_config_file": control_config,
        }],
    )
    manipulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur3_perception"),
            "launch",
            "perception_pick_place.launch.py",
        ])),
        launch_arguments={
            "gazebo_gui": LaunchConfiguration("gazebo_gui"),
            "launch_rviz": LaunchConfiguration("launch_rviz"),
            "image_view": LaunchConfiguration("image_view"),
            "episode_count": episode_count,
            "metrics_output": PathJoinSubstitution([run_root, "metrics.json"]),
            "shutdown_on_completion": "false",
            "randomize_object": "true",
            "random_seed": random_seed,
            "robustness_profile": "robustness_none.yaml",
            "stop_on_failure": "true",
            "return_home": "false",
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("dataset_root", default_value="/tmp/ur3_spatial_dataset"),
        DeclareLaunchArgument("run_id", default_value=default_run_id),
        DeclareLaunchArgument("random_seed", default_value="23"),
        DeclareLaunchArgument("episode_count", default_value="1"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("image_view", default_value="true"),
        spatial_scene,
        recorder,
        manipulation,
    ])
