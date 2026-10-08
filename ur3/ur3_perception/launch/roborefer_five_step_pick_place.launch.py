"""Blind five-clause RoboRefer grounding, verified before UR3 pick/place."""

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


FIVE_STEP_PROMPT = (
    "Point to the target round fruit, not the reference object, by resolving "
    "five ordered spatial stages on the complete RGB-D tabletop scene. "
    "1) Locate the tall upright yellow bottle as a reference only; never "
    "return a point on the bottle. "
    "2) Keep a round fruit to the right of that reference. "
    "3) Keep the candidate whose center is lower in the image than the "
    "bottle's center. "
    "4) Keep the candidate that is farther from the camera than the bottle. "
    "5) Point to one visible point well inside the surviving object's center "
    "for the robot to pick and place. "
    "Do not point to the tray, cubes, robot, gripper, or text labels. Return "
    "exactly one point on the final round fruit, never on the yellow bottle."
)


def generate_launch_description():
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
            "random_seed": "31",
            "runner_start_delay": "45.0",
            "roborefer_start_delay": "15.0",
            "start_roborefer_server": LaunchConfiguration("start_roborefer_server"),
            "enable_roborefer_dimension_comparison": "true",
            "roborefer_comparison_mode": "five_step_reasoning",
            # The main grounder locks a blind point and cannot release a robot
            # target. The post-inference five-clause gate is the sole publisher.
            "roborefer_publish_target_selection": "false",
            "roborefer_five_step_publish_target_selection": "true",
            "roborefer_five_step_target_object_id": "ycb_apple_01",
            # The entries below configure manipulation only after PASS. They
            # are not interpolated into FIVE_STEP_PROMPT.
            "grasp_object_model": "ycb_apple",
            "grasp_object_color": "red",
            "grasp_object_link": "object_link",
            "target_object": "",
            "target_label_hint": "",
            "grasp_region": "object center",
            "use_grounder_grasp_pixel": "true",
            "roborefer_depth_seed_radius_px": "5",
            "roborefer_depth_roi_radius_px": "110",
            "roborefer_depth_near_tolerance_m": "0.015",
            "roborefer_depth_far_tolerance_m": "0.055",
            "roborefer_min_mask_area_px": "120",
            "roborefer_max_mask_area_fraction": "0.12",
            "roborefer_bbox_padding_px": "5",
            "roborefer_compare_object_a_label": "TALL YELLOW BOTTLE",
            "roborefer_compare_object_a_description": (
                "the tall yellow mustard bottle-shaped object standing on the table"
            ),
            "roborefer_compare_object_b_label": "ROUND FRUIT TARGET",
            "roborefer_compare_object_b_description": (
                "the round red apple resting on the table"
            ),
            "instruction": FIVE_STEP_PROMPT,
            "runner_profile": "kitchen_apple_pick_place.yaml",
        }.items(),
    )

    return LaunchDescription([
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
        SetEnvironmentVariable("IGN_PARTITION", "ur3_roborefer_five_step_local"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("image_view", default_value="true"),
        DeclareLaunchArgument("shutdown_on_completion", default_value="true"),
        DeclareLaunchArgument("start_roborefer_server", default_value="true"),
        DeclareLaunchArgument(
            "metrics_output",
            default_value=(
                "/home/dhcn/ur_ws/src/myproject/metrics/"
                f"roborefer_five_step_pick_place_{datetime.now():%Y%m%d_%H%M%S}.json"
            ),
        ),
        LogInfo(msg=(
            "ROBOREFER_FIVE_STEP_DEMO: blind composite prompt -> lock point -> "
            "five observable RGB-D checks -> release target -> one UR3 "
            "pick-and-place; RoboRefer point + registered depth"
        )),
        multimodal,
    ])
