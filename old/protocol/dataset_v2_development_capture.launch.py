"""Launch one execution-locked Dataset V2 development capture batch."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
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
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    output_root = LaunchConfiguration("output_root")
    capture_plan = LaunchConfiguration("capture_plan")
    execution_lock = LaunchConfiguration("execution_lock")
    capture_script = LaunchConfiguration("capture_script")
    world_file = LaunchConfiguration("world_file")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    batch_id = LaunchConfiguration("batch_id")

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur3_moveit_control"), "launch", "ur3_susgrip_sim.launch.py"
        ])),
        launch_arguments={
            "ur_type": "ur3", "launch_rviz": "false", "gazebo_gui": gazebo_gui,
            "world_file": world_file, "camera_update_rate": "30",
            "grasp_object_model": "red_cube", "grasp_object_color": "red",
            "grasp_object_link": "cube_link", "multi_object_grasp": "false",
            "gripper_controller_delay": "18.0",
        }.items(),
    )
    motion_action_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur3_moveit_control"), "launch", "ur3_demo_gripper.launch.py"
        ])),
        launch_arguments={
            "ur_type": "ur3", "use_sim_time": "true", "launch_rviz": "false",
            "launch_moveit": "false",
        }.items(),
    )
    capture = ExecuteProcess(
        cmd=[
            "/usr/bin/python3", capture_script,
            "--capture-plan", capture_plan,
            "--execution-lock", execution_lock,
            "--output-root", output_root,
            "--batch-id", batch_id,
            "--settle-sec", "0.8",
        ],
        output="screen", additional_env={"PYTHONNOUSERSITE": "1"},
    )
    shutdown = RegisterEventHandler(OnProcessExit(
        target_action=capture,
        on_exit=[EmitEvent(event=Shutdown(reason="Dataset V2 development batch ended"))],
    ))
    return LaunchDescription([
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
        SetEnvironmentVariable("IGN_PARTITION", "ur3_roborefer_dataset_v2_development_local"),
        DeclareLaunchArgument("gazebo_gui", default_value="false"),
        DeclareLaunchArgument("output_root"),
        DeclareLaunchArgument("capture_plan"),
        DeclareLaunchArgument("execution_lock"),
        DeclareLaunchArgument("capture_script"),
        DeclareLaunchArgument("world_file"),
        DeclareLaunchArgument("batch_id"),
        LogInfo(msg="DATASET_V2_DEVELOPMENT: one locked batch; no VLM, training or manipulation"),
        simulation,
        motion_action_server,
        TimerAction(period=75.0, actions=[capture]),
        shutdown,
    ])
