"""UR3 Gazebo demo: pick and place the YCB apple with RoboRefer RGB-D."""

from datetime import datetime

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    instruction = LaunchConfiguration("instruction")
    multimodal = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur3_perception"),
            "launch",
            "multimodal_pick_place.launch.py",
        ])),
        launch_arguments={
            "gazebo_gui": LaunchConfiguration("gazebo_gui"),
            "launch_rviz": "false",
            "image_view": LaunchConfiguration("image_view"),
            "metrics_output": LaunchConfiguration("metrics_output"),
            "shutdown_on_completion": LaunchConfiguration("shutdown_on_completion"),
            "randomize_object": "false",
            "random_seed": "13",
            "runner_start_delay": "45.0",
            "roborefer_start_delay": "15.0",
            "start_roborefer_server": LaunchConfiguration(
                "start_roborefer_server"
            ),
            # Only the RoboRefer RGB-D path is started by this launch.
            "grasp_object_model": "ycb_apple",
            "grasp_object_color": "red",
            "grasp_object_link": "object_link",
            "target_object": "apple",
            "target_label_hint": "apple",
            "grasp_region": "object center",
            "use_grounder_grasp_pixel": "true",
            # The RoboRefer point seeds an asymmetric metric-depth component;
            # the resulting mask provides both bbox and grasp validation.
            "roborefer_depth_seed_radius_px": "5",
            "roborefer_depth_roi_radius_px": "110",
            "roborefer_depth_near_tolerance_m": "0.015",
            "roborefer_depth_far_tolerance_m": "0.055",
            "roborefer_min_mask_area_px": "120",
            "roborefer_max_mask_area_fraction": "0.12",
            "roborefer_bbox_padding_px": "5",
            "roborefer_publish_target_selection": LaunchConfiguration(
                "roborefer_publish_target_selection"
            ),
            "roborefer_comparison_mode": LaunchConfiguration(
                "roborefer_comparison_mode"
            ),
            "instruction": instruction,
            "runner_profile": "kitchen_apple_pick_place.yaml",
        }.items(),
    )

    return LaunchDescription([
        # Every ROS process in this demo is local, so keep DDS discovery on
        # loopback.  Do not set IGN_IP=127.0.0.1: Ignition Fortress discovers
        # its GUI/server over multicast, while this machine's loopback device
        # has no multicast capability.  Forcing it there opens a black Gazebo
        # GUI which never discovers the world server.
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
        SetEnvironmentVariable("IGN_PARTITION", "ur3_roborefer_apple_local"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("image_view", default_value="true"),
        DeclareLaunchArgument("shutdown_on_completion", default_value="true"),
        DeclareLaunchArgument(
            "instruction",
            default_value=(
                "Look at the complete RGB-D tabletop scene. Point to the "
                "graspable center of only the red apple that the robot should "
                "pick up. Do not choose the mango, orange, banana, tray, red "
                "labels, or robot gripper."
            ),
            description=(
                "Spatial referring instruction passed directly to RoboRefer."
            ),
        ),
        DeclareLaunchArgument(
            "roborefer_publish_target_selection",
            default_value="true",
            description=(
                "Set false for a blind perception-only trial. The model result "
                "will not be mapped to a configured Gazebo object or sent to "
                "the manipulation pipeline."
            ),
        ),
        DeclareLaunchArgument(
            "start_roborefer_server",
            default_value="true",
            description=(
                "Start the integrated memory-saving RoboRefer server. Set false "
                "only to reuse a server already running on port 25547."
            ),
        ),
        DeclareLaunchArgument(
            "roborefer_comparison_mode",
            default_value="dimensions",
            description=(
                "Use near_far_ablation for RGB-only versus RGB-D near/far, "
                "or relative_position for camera/base_link spatial relations."
            ),
        ),
        DeclareLaunchArgument(
            "metrics_output",
            default_value=(
                "/home/dhcn/ur_ws/src/myproject/metrics/"
                f"roborefer_apple_pick_place_{datetime.now():%Y%m%d_%H%M%S}.json"
            ),
        ),
        LogInfo(msg=(
            "ROBOREFER_APPLE_DEMO: registered wrist RGB-D; point-seeded "
            "depth mask; one YCB "
            "apple pick-and-place episode"
        )),
        multimodal,
    ])
