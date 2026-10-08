"""Launch the locked wrist view and WP2 Gazebo raw-capture process."""

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
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


LOCKED_VIEW_JOINT_POSE = [1.3815, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601]


def generate_launch_description():
    output_root = LaunchConfiguration("output_root")
    capture_plan = LaunchConfiguration("capture_plan")
    smoke_selection = LaunchConfiguration("smoke_selection")
    selection = LaunchConfiguration("selection")
    capture_script = LaunchConfiguration("capture_script")
    gazebo_gui = LaunchConfiguration("gazebo_gui")

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur3_moveit_control"), "launch", "ur3_susgrip_sim.launch.py"
        ])),
        launch_arguments={
            "ur_type": "ur3", "launch_rviz": "false", "gazebo_gui": gazebo_gui,
            "camera_update_rate": "30", "grasp_object_model": "red_cube",
            "grasp_object_color": "red", "grasp_object_link": "cube_link",
            "multi_object_grasp": "false", "gripper_controller_delay": "18.0",
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
    camera_mover = Node(
        package="ur3_perception", executable="move_camera_to_view.py",
        name="wp2_move_camera_to_view", output="screen", prefix=["/usr/bin/python3"],
        additional_env={"PYTHONNOUSERSITE": "1"},
        parameters=[{
            "use_sim_time": True, "view_joint_pose": LOCKED_VIEW_JOINT_POSE,
            "wait_for_server_sec": 60.0,
        }],
    )
    capture = ExecuteProcess(
        cmd=[
            "/usr/bin/python3", capture_script,
            "--capture-plan", capture_plan,
            "--output-root", output_root,
            "--selection", selection,
            "--smoke-selection", smoke_selection,
            "--settle-sec", "0.75",
        ],
        output="screen",
        additional_env={"PYTHONNOUSERSITE": "1"},
    )
    start_capture = RegisterEventHandler(OnProcessExit(target_action=camera_mover, on_exit=[capture]))
    shutdown = RegisterEventHandler(OnProcessExit(
        target_action=capture,
        on_exit=[EmitEvent(event=Shutdown(reason="WP2 Gazebo capture completed"))],
    ))
    return LaunchDescription([
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
        SetEnvironmentVariable("IGN_PARTITION", "ur3_roborefer_wp2_local"),
        DeclareLaunchArgument("gazebo_gui", default_value="false"),
        DeclareLaunchArgument("output_root"),
        DeclareLaunchArgument("capture_plan"),
        DeclareLaunchArgument("smoke_selection"),
        DeclareLaunchArgument("selection", default_value="all"),
        DeclareLaunchArgument("capture_script"),
        LogInfo(msg="WP2_CAPTURE: Gazebo RGB-D + evaluator-only labels; no VLM or manipulation"),
        simulation,
        motion_action_server,
        start_capture,
        shutdown,
        TimerAction(period=32.0, actions=[camera_mover]),
    ])
