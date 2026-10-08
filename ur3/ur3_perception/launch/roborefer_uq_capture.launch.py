"""Capture WP5 UQ scenes in an explicitly selected immutable Gazebo world."""

from datetime import datetime

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


LOCKED_VIEW_JOINT_POSE = [
    1.3815,
    -1.8273,
    1.3428,
    -1.2863,
    -1.5708,
    -1.9601,
]


def generate_launch_description():
    output_root = LaunchConfiguration("output_root")
    scene_config_file = LaunchConfiguration("scene_config_file")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    annotation_file = LaunchConfiguration("annotation_file")
    gate_config_file = LaunchConfiguration("gate_config_file")
    pretrial_lock_file = LaunchConfiguration("pretrial_lock_file")
    world_file = LaunchConfiguration("world_file")
    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    semantic_label_topic = LaunchConfiguration("semantic_label_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    camera_frame = LaunchConfiguration("camera_frame")

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur3_moveit_control"),
            "launch",
            "ur3_susgrip_sim.launch.py",
        ])),
        launch_arguments={
            "ur_type": "ur3",
            "launch_rviz": "false",
            "gazebo_gui": gazebo_gui,
            "camera_update_rate": "30",
            "world_file": world_file,
            "grasp_object_model": "red_cube",
            "grasp_object_color": "red",
            "grasp_object_link": "cube_link",
            "multi_object_grasp": "false",
            "gripper_controller_delay": "18.0",
        }.items(),
    )

    motion_action_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur3_moveit_control"),
            "launch",
            "ur3_demo_gripper.launch.py",
        ])),
        launch_arguments={
            "ur_type": "ur3",
            "use_sim_time": "true",
            "launch_rviz": "false",
            "launch_moveit": "false",
        }.items(),
    )

    camera_mover = Node(
        package="ur3_perception",
        executable="move_camera_to_view.py",
        name="pilot_move_camera_to_view",
        output="screen",
        prefix=["/usr/bin/python3"],
        additional_env={"PYTHONNOUSERSITE": "1"},
        parameters=[{
            "use_sim_time": True,
            "view_joint_pose": LOCKED_VIEW_JOINT_POSE,
            "wait_for_server_sec": 60.0,
        }],
    )

    capture = Node(
        package="ur3_perception",
        executable="roborefer_pilot_capture.py",
        name="roborefer_pilot_capture",
        output="screen",
        prefix=["/usr/bin/python3"],
        additional_env={"PYTHONNOUSERSITE": "1"},
        parameters=[{
            "use_sim_time": True,
            "scene_config_file": scene_config_file,
            "output_root": output_root,
            "annotation_file": annotation_file,
            "gate_config_file": gate_config_file,
            "pretrial_lock_file": pretrial_lock_file,
            "rgb_topic": rgb_topic,
            "depth_topic": depth_topic,
            "semantic_label_topic": semantic_label_topic,
            "camera_info_topic": camera_info_topic,
            "camera_frame": camera_frame,
            "settle_sec": 1.5,
            "sync_slop_sec": 0.02,
            "capture_timeout_sec": 45.0,
        }],
    )

    start_capture_after_view = RegisterEventHandler(
        OnProcessExit(target_action=camera_mover, on_exit=[capture])
    )
    shutdown_after_capture = RegisterEventHandler(
        OnProcessExit(
            target_action=capture,
            on_exit=[EmitEvent(event=Shutdown(reason="pilot capture completed"))],
        )
    )

    return LaunchDescription([
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
        SetEnvironmentVariable("IGN_PARTITION", "ur3_roborefer_pilot_v0_local"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument(
            "world_file",
            default_value="/home/dhcn/ur_ws/src/myproject/ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion.sdf",
            description="Absolute immutable WP5 world revision.",
        ),
        DeclareLaunchArgument(
            "scene_config_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ur3_perception"),
                "config",
                "roborefer_pilot_v0_scenes.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "annotation_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ur3_perception"),
                "config",
                "roborefer_pilot_v0_annotations.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "gate_config_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ur3_perception"),
                "config",
                "roborefer_pilot_v0_gate.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "pretrial_lock_file",
            default_value="",
            description="Absolute path to the immutable pre-capture source/hash lock",
        ),
        DeclareLaunchArgument("rgb_topic", default_value="/wrist_camera/color/image_raw"),
        DeclareLaunchArgument("depth_topic", default_value="/wrist_camera/depth/image_raw"),
        DeclareLaunchArgument(
            "semantic_label_topic",
            default_value="/wrist_camera/evaluation_labels/labels_map",
        ),
        DeclareLaunchArgument(
            "camera_info_topic", default_value="/wrist_camera/color/camera_info"
        ),
        DeclareLaunchArgument("camera_frame", default_value="camera_color_optical_frame"),
        DeclareLaunchArgument(
            "output_root",
            default_value=(
                "/tmp/roborefer_pilot_v0_"
                f"{datetime.now():%Y%m%d_%H%M%S}/dataset"
            ),
        ),
        LogInfo(msg=(
            "ROBOREFER_WP5_UQ_CAPTURE: preregistered scenes; camera motion "
            "only; no VLM, no target handoff, no pick/place"
        )),
        simulation,
        motion_action_server,
        start_capture_after_view,
        shutdown_after_capture,
        TimerAction(period=32.0, actions=[camera_mover]),
    ])
