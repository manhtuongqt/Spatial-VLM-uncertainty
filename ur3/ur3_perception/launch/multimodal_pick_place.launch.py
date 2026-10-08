"""Gazebo pick/place with the RoboRefer RGB-D spatial-grounding pipeline."""

from datetime import datetime
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _is_true(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _matching_processes(marker, cwd_fragment=""):
    """Return active process descriptions matching a GPU backend."""
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
            cwd = os.readlink(entry / "cwd")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker in command and (not cwd_fragment or cwd_fragment in cwd):
            matches.append(f"pid={entry.name}: {command.strip()}")
    return matches


def _validate_roborefer_server(context):
    """Prevent two RoboRefer API processes from loading the model on one GPU."""
    start_server = _is_true(
        LaunchConfiguration("start_roborefer_server").perform(context)
    )
    existing_server = _matching_processes("api.py", "/RoboRefer/API")
    if start_server and existing_server:
        raise RuntimeError(
            "A standalone RoboRefer API is already running. Stop it or set "
            "start_roborefer_server:=false. " + " | ".join(existing_server)
        )
    return [LogInfo(msg="ROBOREFER_GPU_SAFETY: single-server check passed")]


def generate_launch_description():
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    launch_rviz = LaunchConfiguration("launch_rviz")
    image_view = LaunchConfiguration("image_view")
    multimodal_image_view = LaunchConfiguration("multimodal_image_view")
    metrics_output = LaunchConfiguration("metrics_output")
    shutdown_on_completion = LaunchConfiguration("shutdown_on_completion")
    randomize_object = LaunchConfiguration("randomize_object")
    random_seed = LaunchConfiguration("random_seed")
    runner_start_delay = LaunchConfiguration("runner_start_delay")
    roborefer_start_delay = LaunchConfiguration("roborefer_start_delay")
    ros_python = LaunchConfiguration("ros_python")
    instruction = LaunchConfiguration("instruction")
    grasp_object_model = LaunchConfiguration("grasp_object_model")
    grasp_object_color = LaunchConfiguration("grasp_object_color")
    grasp_object_link = LaunchConfiguration("grasp_object_link")
    target_object = LaunchConfiguration("target_object")
    target_label_hint = LaunchConfiguration("target_label_hint")
    grasp_region = LaunchConfiguration("grasp_region")
    use_grounder_grasp_pixel = LaunchConfiguration("use_grounder_grasp_pixel")
    runner_profile = LaunchConfiguration("runner_profile")
    roborefer_root = LaunchConfiguration("roborefer_root")
    roborefer_python = LaunchConfiguration("roborefer_python")
    roborefer_model = LaunchConfiguration("roborefer_model")
    roborefer_depth_model = LaunchConfiguration("roborefer_depth_model")
    roborefer_port = LaunchConfiguration("roborefer_port")
    start_roborefer_server = LaunchConfiguration("start_roborefer_server")
    roborefer_depth_seed_radius_px = LaunchConfiguration(
        "roborefer_depth_seed_radius_px"
    )
    roborefer_depth_roi_radius_px = LaunchConfiguration(
        "roborefer_depth_roi_radius_px"
    )
    roborefer_depth_near_tolerance_m = LaunchConfiguration(
        "roborefer_depth_near_tolerance_m"
    )
    roborefer_depth_far_tolerance_m = LaunchConfiguration(
        "roborefer_depth_far_tolerance_m"
    )
    roborefer_min_mask_area_px = LaunchConfiguration(
        "roborefer_min_mask_area_px"
    )
    roborefer_max_mask_area_fraction = LaunchConfiguration(
        "roborefer_max_mask_area_fraction"
    )
    roborefer_bbox_padding_px = LaunchConfiguration("roborefer_bbox_padding_px")
    roborefer_publish_target_selection = LaunchConfiguration(
        "roborefer_publish_target_selection"
    )
    enable_roborefer_dimension_comparison = LaunchConfiguration(
        "enable_roborefer_dimension_comparison"
    )
    roborefer_compare_object_a_label = LaunchConfiguration(
        "roborefer_compare_object_a_label"
    )
    roborefer_compare_object_a_description = LaunchConfiguration(
        "roborefer_compare_object_a_description"
    )
    roborefer_compare_object_b_label = LaunchConfiguration(
        "roborefer_compare_object_b_label"
    )
    roborefer_compare_object_b_description = LaunchConfiguration(
        "roborefer_compare_object_b_description"
    )
    roborefer_compare_object_c_label = LaunchConfiguration(
        "roborefer_compare_object_c_label"
    )
    roborefer_compare_object_c_description = LaunchConfiguration(
        "roborefer_compare_object_c_description"
    )
    roborefer_comparison_mode = LaunchConfiguration(
        "roborefer_comparison_mode"
    )
    roborefer_five_step_publish_target_selection = LaunchConfiguration(
        "roborefer_five_step_publish_target_selection"
    )
    roborefer_five_step_target_object_id = LaunchConfiguration(
        "roborefer_five_step_target_object_id"
    )
    roborefer_five_step_rule_profile = LaunchConfiguration(
        "roborefer_five_step_rule_profile"
    )
    roborefer_five_step_forbidden_target_terms_csv = LaunchConfiguration(
        "roborefer_five_step_forbidden_target_terms_csv"
    )
    roborefer_five_step_anchor_min_aspect_ratio = LaunchConfiguration(
        "roborefer_five_step_anchor_min_aspect_ratio"
    )
    roborefer_five_step_compact_target_max_aspect_ratio = LaunchConfiguration(
        "roborefer_five_step_compact_target_max_aspect_ratio"
    )
    roborefer_five_step_size_margin_m = LaunchConfiguration(
        "roborefer_five_step_size_margin_m"
    )
    roborefer_five_step_between_corridor_m = LaunchConfiguration(
        "roborefer_five_step_between_corridor_m"
    )

    pick_place = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur3_perception"),
            "launch",
            "perception_pick_place.launch.py",
        ])),
        launch_arguments={
            "gazebo_gui": gazebo_gui,
            "launch_rviz": launch_rviz,
            "image_view": "false",
            "episode_count": "1",
            "metrics_output": metrics_output,
            "shutdown_on_completion": shutdown_on_completion,
            "randomize_object": randomize_object,
            "random_seed": random_seed,
            "robustness_profile": runner_profile,
            "stop_on_failure": "true",
            "return_home": "false",
            "object_pose_source": "perception",
            "grasp_object_model": grasp_object_model,
            "grasp_object_color": grasp_object_color,
            "grasp_object_link": grasp_object_link,
            "multi_object_grasp": "false",
            "runner_start_delay": runner_start_delay,
            "require_target_selection": "true",
            "use_grounder_grasp_pixel": use_grounder_grasp_pixel,
            "ros_python": ros_python,
        }.items(),
    )

    roborefer_server = ExecuteProcess(
        cmd=[
            roborefer_python,
            "-s",
            "api.py",
            "--port",
            roborefer_port,
            "--depth_model_path",
            roborefer_depth_model,
            "--vlm_model_path",
            roborefer_model,
            # Gazebo publishes registered depth. Avoid loading the extra
            # 1.3 GB Depth Anything checkpoint and save ~2.4 GB VRAM.
            "--skip_depth_model",
        ],
        cwd=PathJoinSubstitution([roborefer_root, "API"]),
        additional_env={
            "PYTHONNOUSERSITE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        output="screen",
        condition=IfCondition(start_roborefer_server),
    )
    roborefer_grounder = Node(
        package="ur3_perception",
        executable="roborefer_grounder.py",
        name="roborefer_grounder",
        output="screen",
        prefix=[ros_python],
        additional_env={"PYTHONNOUSERSITE": "1"},
        parameters=[{
            "instruction": instruction,
            "target_color": grasp_object_color,
            "target_object": target_object,
            "target_label_hint": target_label_hint,
            "publish_target_selection": ParameterValue(
                roborefer_publish_target_selection, value_type=bool
            ),
            "grasp_region": grasp_region,
            "depth_seed_radius_px": ParameterValue(
                roborefer_depth_seed_radius_px, value_type=int
            ),
            "depth_roi_radius_px": ParameterValue(
                roborefer_depth_roi_radius_px, value_type=int
            ),
            "depth_near_tolerance_m": ParameterValue(
                roborefer_depth_near_tolerance_m, value_type=float
            ),
            "depth_far_tolerance_m": ParameterValue(
                roborefer_depth_far_tolerance_m, value_type=float
            ),
            "min_mask_area_px": ParameterValue(
                roborefer_min_mask_area_px, value_type=int
            ),
            "max_mask_area_fraction": ParameterValue(
                roborefer_max_mask_area_fraction, value_type=float
            ),
            "bbox_padding_px": ParameterValue(
                roborefer_bbox_padding_px, value_type=int
            ),
            "wait_for_manipulation_state": True,
            "trigger_state": "FREEZE_PERCEPTION_POSE",
            "server_url": PythonExpression([
                "'http://127.0.0.1:", roborefer_port, "'"
            ]),
        }],
    )
    roborefer_dimension_comparator = Node(
        package="ur3_perception",
        executable="roborefer_dimension_comparator.py",
        name="roborefer_dimension_comparator",
        output="screen",
        prefix=[ros_python],
        additional_env={"PYTHONNOUSERSITE": "1"},
        parameters=[{
            "server_url": PythonExpression([
                "'http://127.0.0.1:", roborefer_port, "'"
            ]),
            "object_a_label": roborefer_compare_object_a_label,
            "object_a_description": roborefer_compare_object_a_description,
            "object_b_label": roborefer_compare_object_b_label,
            "object_b_description": roborefer_compare_object_b_description,
            "object_c_label": roborefer_compare_object_c_label,
            "object_c_description": roborefer_compare_object_c_description,
            "comparison_mode": roborefer_comparison_mode,
            "five_step_publish_target_selection": ParameterValue(
                roborefer_five_step_publish_target_selection, value_type=bool
            ),
            "five_step_target_object_id": roborefer_five_step_target_object_id,
            "five_step_target_color": grasp_object_color,
            "five_step_target_oracle_name": grasp_object_model,
            "five_step_rule_profile": roborefer_five_step_rule_profile,
            "five_step_forbidden_target_terms_csv": (
                roborefer_five_step_forbidden_target_terms_csv
            ),
            "five_step_anchor_min_aspect_ratio": ParameterValue(
                roborefer_five_step_anchor_min_aspect_ratio, value_type=float
            ),
            "five_step_compact_target_max_aspect_ratio": ParameterValue(
                roborefer_five_step_compact_target_max_aspect_ratio,
                value_type=float,
            ),
            "five_step_size_margin_m": ParameterValue(
                roborefer_five_step_size_margin_m, value_type=float
            ),
            "five_step_between_corridor_m": ParameterValue(
                roborefer_five_step_between_corridor_m, value_type=float
            ),
            "depth_seed_radius_px": ParameterValue(
                roborefer_depth_seed_radius_px, value_type=int
            ),
            "depth_roi_radius_px": ParameterValue(
                roborefer_depth_roi_radius_px, value_type=int
            ),
            "depth_near_tolerance_m": ParameterValue(
                roborefer_depth_near_tolerance_m, value_type=float
            ),
            "depth_far_tolerance_m": ParameterValue(
                roborefer_depth_far_tolerance_m, value_type=float
            ),
            "min_mask_area_px": ParameterValue(
                roborefer_min_mask_area_px, value_type=int
            ),
            "max_mask_area_fraction": ParameterValue(
                roborefer_max_mask_area_fraction, value_type=float
            ),
            "bbox_padding_px": ParameterValue(
                roborefer_bbox_padding_px, value_type=int
            ),
        }],
        condition=IfCondition(enable_roborefer_dimension_comparison),
    )
    multimodal_viewer = Node(
        package="ur3_perception",
        executable="multimodal_image_viewer.py",
        name="multimodal_image_viewer",
        prefix=[ros_python],
        output="screen",
        additional_env={"PYTHONNOUSERSITE": "1"},
        condition=IfCondition(multimodal_image_view),
        parameters=[{
            "roborefer_comparison_mode": roborefer_comparison_mode,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("launch_rviz", default_value="false"),
        DeclareLaunchArgument("image_view", default_value="true"),
        DeclareLaunchArgument(
            "metrics_output",
            default_value=(
                "/home/dhcn/ur_ws/src/myproject/metrics/"
                f"ur3_multimodal_pick_place_{datetime.now():%Y%m%d_%H%M%S}.json"
            ),
        ),
        DeclareLaunchArgument("shutdown_on_completion", default_value="false"),
        DeclareLaunchArgument(
            "randomize_object", default_value="false",
            description="Sample the selected source object in the validated source workspace.",
        ),
        DeclareLaunchArgument(
            "random_seed", default_value="7",
            description="Seed recorded by the manipulation runner for reproducible source placement.",
        ),
        DeclareLaunchArgument("runner_start_delay", default_value="45.0"),
        DeclareLaunchArgument("grasp_object_model", default_value="red_cube"),
        DeclareLaunchArgument("grasp_object_color", default_value="red"),
        DeclareLaunchArgument("grasp_object_link", default_value="cube_link"),
        DeclareLaunchArgument("target_object", default_value="cube"),
        DeclareLaunchArgument("target_label_hint", default_value=""),
        DeclareLaunchArgument("grasp_region", default_value="object"),
        DeclareLaunchArgument("use_grounder_grasp_pixel", default_value="false"),
        DeclareLaunchArgument("runner_profile", default_value="robustness_none.yaml"),
        DeclareLaunchArgument(
            "ros_python",
            default_value="/usr/bin/python3",
        ),
        DeclareLaunchArgument(
            "instruction",
            default_value=(
                "Point to several points at the center of the red cube "
                "that the robot should pick up."
            ),
        ),
        DeclareLaunchArgument(
            "roborefer_start_delay",
            default_value="18.0",
            description=(
                "Start the lightweight ROS adapter early; its lifecycle gate "
                "still waits for the frozen wrist-camera view."
            ),
        ),
        DeclareLaunchArgument(
            "roborefer_root",
            default_value="/home/dhcn/ur_ws/src/myproject/RoboRefer",
        ),
        DeclareLaunchArgument(
            "roborefer_python",
            default_value="/home/dhcn/ur_ws/src/myproject/.conda-roborefer/bin/python",
        ),
        DeclareLaunchArgument(
            "roborefer_model",
            default_value=(
                "/home/dhcn/ur_ws/src/myproject/RoboRefer/models/"
                "RoboRefer-2B-SFT"
            ),
        ),
        DeclareLaunchArgument(
            "roborefer_depth_model",
            default_value=(
                "/home/dhcn/ur_ws/src/myproject/RoboRefer/models/"
                "Depth-Anything-V2-Large/depth_anything_v2_vitl.pth"
            ),
        ),
        DeclareLaunchArgument("roborefer_port", default_value="25547"),
        DeclareLaunchArgument("roborefer_depth_seed_radius_px", default_value="5"),
        DeclareLaunchArgument("roborefer_depth_roi_radius_px", default_value="110"),
        DeclareLaunchArgument(
            "roborefer_depth_near_tolerance_m", default_value="0.015"
        ),
        DeclareLaunchArgument(
            "roborefer_depth_far_tolerance_m", default_value="0.055"
        ),
        DeclareLaunchArgument("roborefer_min_mask_area_px", default_value="120"),
        DeclareLaunchArgument(
            "roborefer_max_mask_area_fraction", default_value="0.12"
        ),
        DeclareLaunchArgument("roborefer_bbox_padding_px", default_value="5"),
        DeclareLaunchArgument(
            "roborefer_publish_target_selection",
            default_value="true",
            description=(
                "Disable for blind grounding evaluation: publish only the raw "
                "RoboRefer result/status, never a registry-backed robot target."
            ),
        ),
        DeclareLaunchArgument(
            "enable_roborefer_dimension_comparison",
            default_value="true",
            description=(
                "Let RoboRefer select two objects, segment each with registered "
                "depth, and compare metric 3D width/height in dashboard panel four."
            ),
        ),
        DeclareLaunchArgument(
            "roborefer_comparison_mode",
            default_value="dimensions",
            description=(
                "Fourth-panel demo: dimensions, near_far_ablation, or "
                "relative_position, or five_step_reasoning."
            ),
        ),
        DeclareLaunchArgument(
            "roborefer_five_step_publish_target_selection",
            default_value="false",
            description=(
                "Release a registry-backed target only after all five observable "
                "spatial clauses pass. Intended for five_step_reasoning mode."
            ),
        ),
        DeclareLaunchArgument(
            "roborefer_five_step_target_object_id",
            default_value="ycb_apple_01",
            description=(
                "Registry identity added after five-step inference and checking; "
                "it is never inserted into the RoboRefer prompt."
            ),
        ),
        DeclareLaunchArgument(
            "roborefer_five_step_rule_profile",
            default_value="relative_round_target",
        ),
        DeclareLaunchArgument(
            "roborefer_five_step_forbidden_target_terms_csv",
            default_value="apple,táo,ycb_apple",
        ),
        DeclareLaunchArgument(
            "roborefer_five_step_anchor_min_aspect_ratio",
            default_value="1.5",
        ),
        DeclareLaunchArgument(
            "roborefer_five_step_compact_target_max_aspect_ratio",
            default_value="1.75",
        ),
        DeclareLaunchArgument(
            "roborefer_five_step_size_margin_m",
            default_value="0.005",
        ),
        DeclareLaunchArgument(
            "roborefer_five_step_between_corridor_m",
            default_value="0.10",
        ),
        DeclareLaunchArgument(
            "roborefer_compare_object_a_label",
            default_value="MUSTARD BOTTLE",
        ),
        DeclareLaunchArgument(
            "roborefer_compare_object_a_description",
            default_value=(
                "the tall yellow mustard bottle-shaped object standing on the table"
            ),
        ),
        DeclareLaunchArgument(
            "roborefer_compare_object_b_label",
            default_value="RED APPLE",
        ),
        DeclareLaunchArgument(
            "roborefer_compare_object_b_description",
            default_value="the round red apple resting on the table",
        ),
        DeclareLaunchArgument(
            "roborefer_compare_object_c_label",
            default_value="",
        ),
        DeclareLaunchArgument(
            "roborefer_compare_object_c_description",
            default_value="",
        ),
        DeclareLaunchArgument(
            "start_roborefer_server",
            default_value="true",
            description="Set false only when intentionally reusing an existing API.",
        ),
        # Preserve the top-level viewer setting before the included launch
        # locally sets its own image_view argument to false.
        SetLaunchConfiguration("multimodal_image_view", image_view),
        OpaqueFunction(function=_validate_roborefer_server),
        pick_place,
        roborefer_server,
        TimerAction(period=roborefer_start_delay, actions=[roborefer_grounder]),
        TimerAction(
            period=roborefer_start_delay,
            actions=[roborefer_dimension_comparator],
        ),
        TimerAction(period=25.0, actions=[multimodal_viewer]),
    ])
