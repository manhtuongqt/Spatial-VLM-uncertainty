"""Two-reference RoboRefer RGB-D demo from an oblique wrist-camera view."""

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
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOCKED_RANDOM_SEED = 742190531
DEFAULT_RESULT_DIR = (
    "/home/dhcn/ur_ws/src/myproject/results/"
    f"roborefer_two_reference_oblique_tomato_{RUN_STAMP}"
)

# Target identity, target pixel, bbox and world pose are deliberately absent.
TWO_REFERENCE_PROMPT = (
    "Resolve five ordered visual-spatial stages in the complete oblique RGB-D "
    "tabletop scene. 1) Locate the round orange fruit as the first reference. "
    "2) Locate the long curved yellow fruit as the second reference. 3) Keep "
    "only objects spatially between those two reference fruits. 4) Among them, "
    "select the compact upright cylindrical grocery container with a dark "
    "circular pull-tab top and red rim that is physically taller than both "
    "fruits. 5) Return exactly one point well inside the visible center of the "
    "surviving object. Do not point to either reference fruit, the tray, boxes, "
    "bottles, robot, gripper, shadows or text. Return coordinates only."
)


def generate_launch_description():
    result_dir = LaunchConfiguration("result_dir")
    random_seed = LaunchConfiguration("random_seed")
    instruction = LaunchConfiguration("instruction")
    runner_profile = LaunchConfiguration("runner_profile")
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
            "roborefer_publish_target_selection": "false",
            "roborefer_five_step_publish_target_selection": "true",
            "roborefer_five_step_target_object_id": "ycb_tomato_soup_can_01",
            "roborefer_five_step_rule_profile": (
                "two_reference_between_target"
            ),
            "roborefer_five_step_forbidden_target_terms_csv": (
                "tomato,soup,ycb_tomato_soup_can"
            ),
            # A=orange roundness, B=banana elongation, C=target corridor/height.
            "roborefer_five_step_anchor_min_aspect_ratio": "1.60",
            "roborefer_five_step_compact_target_max_aspect_ratio": "1.45",
            "roborefer_five_step_size_margin_m": "0.006",
            "roborefer_five_step_between_corridor_m": "0.080",
            "grasp_object_model": "ycb_tomato_soup_can",
            "grasp_object_color": "red",
            "grasp_object_link": "object_link",
            "target_object": "",
            "target_label_hint": "",
            "grasp_region": "visible object center",
            "use_grounder_grasp_pixel": "true",
            "roborefer_depth_seed_radius_px": "5",
            "roborefer_depth_roi_radius_px": "150",
            "roborefer_depth_near_tolerance_m": "0.015",
            "roborefer_depth_far_tolerance_m": "0.060",
            "roborefer_min_mask_area_px": "100",
            "roborefer_max_mask_area_fraction": "0.18",
            "roborefer_bbox_padding_px": "5",
            # These identity-bearing direct queries run only after the main
            # point is immutable and can never replace that point.
            "roborefer_compare_object_a_label": "ROUND ORANGE REFERENCE",
            "roborefer_compare_object_a_description": (
                "the round orange fruit resting on the tabletop"
            ),
            "roborefer_compare_object_b_label": "CURVED YELLOW REFERENCE",
            "roborefer_compare_object_b_description": (
                "the long curved yellow banana resting on the tabletop"
            ),
            "roborefer_compare_object_c_label": "CYLINDRICAL CANDIDATE",
            "roborefer_compare_object_c_description": (
                "the short upright red tomato soup can with a dark circular "
                "pull-tab top"
            ),
            "instruction": instruction,
            "runner_profile": runner_profile,
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
            "IGN_PARTITION", "ur3_roborefer_two_reference_oblique_local"
        ),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("image_view", default_value="true"),
        DeclareLaunchArgument("shutdown_on_completion", default_value="true"),
        DeclareLaunchArgument("start_roborefer_server", default_value="true"),
        DeclareLaunchArgument(
            "random_seed", default_value=str(LOCKED_RANDOM_SEED)
        ),
        DeclareLaunchArgument("result_dir", default_value=DEFAULT_RESULT_DIR),
        DeclareLaunchArgument(
            "runner_profile",
            default_value="kitchen_tomato_two_reference_oblique.yaml",
        ),
        DeclareLaunchArgument(
            "scenario_name",
            default_value="two_new_references_orange_banana_oblique_camera",
        ),
        DeclareLaunchArgument(
            "instruction", default_value=TWO_REFERENCE_PROMPT
        ),
        LogInfo(msg=(
            "ROBOREFER_TWO_REFERENCE_OBLIQUE: orange + banana references; "
            "target pose hidden; locked point -> A/B/C RGB-D gate -> UR3; results="
        )),
        LogInfo(msg=result_dir),
        recorder,
        multimodal,
    ])
