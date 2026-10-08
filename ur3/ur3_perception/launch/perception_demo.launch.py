from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    launch_rviz = LaunchConfiguration("launch_rviz")
    image_view = LaunchConfiguration("image_view")
    run_demo = LaunchConfiguration("run_demo")
    shutdown_on_completion = LaunchConfiguration("shutdown_on_completion")
    demo_start_delay = LaunchConfiguration("demo_start_delay")
    metrics_output = LaunchConfiguration("metrics_output")

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur3_moveit_control"), "launch", "ur3_susgrip_sim.launch.py"
        ])),
        launch_arguments={
            "ur_type": "ur3",
            "launch_rviz": launch_rviz,
            "gazebo_gui": gazebo_gui,
        }.items(),
    )
    motion_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur3_moveit_control"), "launch", "ur3_demo_gripper.launch.py"
        ])),
        launch_arguments={
            "ur_type": "ur3",
            "use_sim_time": "true",
            "launch_rviz": "false",
            "launch_moveit": "false",
        }.items(),
    )
    perception = Node(
        package="ur3_perception",
        executable="rgbd_object_pose.py",
        name="rgbd_object_pose",
        output="screen",
        parameters=[PathJoinSubstitution([
            FindPackageShare("ur3_perception"), "config", "rgbd_perception.yaml"
        ])],
    )
    runner = Node(
        package="ur3_perception",
        executable="perception_demo_runner.py",
        name="perception_demo_runner",
        output="screen",
        parameters=[{"use_sim_time": True, "metrics_output": metrics_output}],
        condition=IfCondition(run_demo),
    )
    delayed_runner = TimerAction(period=demo_start_delay, actions=[runner])
    image_viewer = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="perception_image_view",
        arguments=["/ur3_perception/debug_image"],
        output="log",
        condition=IfCondition(image_view),
    )
    delayed_image_viewer = TimerAction(period=8.0, actions=[image_viewer])
    shutdown_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=runner,
            on_exit=[EmitEvent(event=Shutdown(reason="perception demo completed"))],
        ),
        condition=IfCondition(shutdown_on_completion),
    )

    return LaunchDescription([
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("image_view", default_value="true"),
        DeclareLaunchArgument("run_demo", default_value="true"),
        DeclareLaunchArgument("shutdown_on_completion", default_value="false"),
        DeclareLaunchArgument(
            "demo_start_delay",
            default_value="30.0",
            description=(
                "Wall-clock seconds to wait for Gazebo, ros2_control, MoveIt, "
                "and the RGB-D stream before starting the first episode"
            ),
        ),
        DeclareLaunchArgument("metrics_output", default_value="/tmp/ur3_perception_demo.json"),
        simulation,
        motion_server,
        perception,
        delayed_runner,
        delayed_image_viewer,
        shutdown_handler,
    ])
