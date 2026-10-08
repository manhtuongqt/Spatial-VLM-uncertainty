"""Blind RoboRefer RGB-D reasoning demo for a randomized YCB grocery object."""

from datetime import datetime
import secrets

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RANDOM_SEED = secrets.randbelow(2_000_000_000) + 1
DEFAULT_RESULT_DIR = (
    "/home/dhcn/ur_ws/src/myproject/results/"
    f"roborefer_blind_tomato_can_{RUN_STAMP}"
)

# Deliberately excludes the semantic target name, pixel, bbox and world pose.
BLIND_FIVE_STAGE_PROMPT = (
    "Resolve five ordered visual-spatial stages on the complete RGB-D tabletop "
    "scene. 1) Locate the upright rectangular grocery packages. 2) Use the "
    "tall yellow rectangular package only as a reference and never return a "
    "point on it. 3) Among all remaining objects, keep the red grocery "
    "container with a circular top and cylindrical side. 4) Keep it only if "
    "it appears physically shorter and more compact than the yellow reference. "
    "5) Return exactly one visible point well inside the center of the surviving "
    "object for the robot to pick and place. Do not point to fruit, the tray, "
    "other boxes, robot links, gripper, or text labels. Return coordinates only."
)


def generate_launch_description():
    result_dir = LaunchConfiguration("result_dir")
    random_seed = LaunchConfiguration("random_seed")
    instruction = LaunchConfiguration("instruction")
    scenario_name = LaunchConfiguration("scenario_name")
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
            "metrics_output": PathJoinSubstitution([
                result_dir, "pick_place_metrics.json"
            ]),
            "shutdown_on_completion": LaunchConfiguration(
                "shutdown_on_completion"
            ),
            "randomize_object": "true",
            "random_seed": random_seed,
            "runner_start_delay": "50.0",
            "roborefer_start_delay": "15.0",
            "start_roborefer_server": LaunchConfiguration(
                "start_roborefer_server"
            ),
            "enable_roborefer_dimension_comparison": "true",
            "roborefer_comparison_mode": "five_step_reasoning",
            # Lock the raw VLM point first. Only the five-step referee may
            # publish a registry-backed target after every check passes.
            "roborefer_publish_target_selection": "false",
            "roborefer_five_step_publish_target_selection": "true",
            "roborefer_five_step_target_object_id": "ycb_tomato_soup_can_01",
            "roborefer_five_step_rule_profile": "compact_target",
            "roborefer_five_step_forbidden_target_terms_csv": (
                "tomato,soup,ycb_tomato_soup_can"
            ),
            "roborefer_five_step_anchor_min_aspect_ratio": "1.50",
            "roborefer_five_step_compact_target_max_aspect_ratio": "1.80",
            "roborefer_five_step_size_margin_m": "0.003",
            # Manipulation identity is not interpolated into the prompt. It is
            # attached to the handoff only after the blind point passes 5/5.
            "grasp_object_model": "ycb_tomato_soup_can",
            "grasp_object_color": "red",
            "grasp_object_link": "object_link",
            "target_object": "",
            "target_label_hint": "",
            "grasp_region": "visible object center",
            "use_grounder_grasp_pixel": "true",
            "roborefer_depth_seed_radius_px": "5",
            "roborefer_depth_roi_radius_px": "100",
            "roborefer_depth_near_tolerance_m": "0.012",
            "roborefer_depth_far_tolerance_m": "0.050",
            "roborefer_min_mask_area_px": "100",
            "roborefer_max_mask_area_fraction": "0.12",
            "roborefer_bbox_padding_px": "5",
            # These two direct queries occur only after the main point is
            # immutable and serve as an RGB-D geometric referee.
            "roborefer_compare_object_a_label": "TALL YELLOW REFERENCE PACKAGE",
            "roborefer_compare_object_a_description": (
                "the tall upright yellow rectangular sugar box on the tabletop"
            ),
            "roborefer_compare_object_b_label": "SHORT RED CYLINDRICAL TARGET",
            "roborefer_compare_object_b_description": (
                "the short upright red tomato soup can with a circular top"
            ),
            "instruction": instruction,
            "runner_profile": "kitchen_tomato_can_pick_place.yaml",
        }.items(),
    )

    recorder = Node(
        package="ur3_perception",
        executable="roborefer_demo_recorder.py",
        name="roborefer_demo_recorder",
        output="screen",
        prefix=["/usr/bin/python3"],
        additional_env={"PYTHONNOUSERSITE": "1"},
        parameters=[{
            "result_dir": result_dir,
            "main_prompt": instruction,
            "random_seed": ParameterValue(random_seed, value_type=int),
            "scenario_name": scenario_name,
        }],
    )

    return LaunchDescription([
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
        SetEnvironmentVariable(
            "IGN_PARTITION", "ur3_roborefer_blind_tomato_local"
        ),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("image_view", default_value="true"),
        DeclareLaunchArgument("shutdown_on_completion", default_value="true"),
        DeclareLaunchArgument("start_roborefer_server", default_value="true"),
        DeclareLaunchArgument("random_seed", default_value=str(RANDOM_SEED)),
        DeclareLaunchArgument("result_dir", default_value=DEFAULT_RESULT_DIR),
        DeclareLaunchArgument(
            "instruction",
            default_value=BLIND_FIVE_STAGE_PROMPT,
            description=(
                "Audited main RoboRefer prompt. It must not contain the target "
                "identity, coordinates, pixels or a bounding box."
            ),
        ),
        DeclareLaunchArgument(
            "scenario_name",
            default_value="blind_randomized_compact_red_grocery_target",
        ),
        LogInfo(msg=(
            "ROBOREFER_BLIND_TOMATO_DEMO: randomized pose hidden from VLM; "
            "RGB-D + five-stage prompt -> locked point -> 5/5 geometric gate "
            "-> UR3 pick/place; results="
        )),
        LogInfo(msg=result_dir),
        recorder,
        multimodal,
    ])
