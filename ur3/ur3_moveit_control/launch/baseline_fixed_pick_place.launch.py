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
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    launch_rviz = LaunchConfiguration("launch_rviz")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    run_script = LaunchConfiguration("run_script")
    world_file = LaunchConfiguration("world_file")
    episode_count = LaunchConfiguration("episode_count")
    metrics_output = LaunchConfiguration("metrics_output")
    stop_on_failure = LaunchConfiguration("stop_on_failure")
    shutdown_on_completion = LaunchConfiguration("shutdown_on_completion")
    object_pose_source = LaunchConfiguration("object_pose_source")
    randomize_object = LaunchConfiguration("randomize_object")
    random_seed = LaunchConfiguration("random_seed")
    robustness_profile = LaunchConfiguration("robustness_profile")
    return_home = LaunchConfiguration("return_home")
    grasp_object_model = LaunchConfiguration("grasp_object_model")
    grasp_object_color = LaunchConfiguration("grasp_object_color")
    grasp_object_link = LaunchConfiguration("grasp_object_link")
    distractor_object_name = LaunchConfiguration("distractor_object_name")
    multi_object_grasp = LaunchConfiguration("multi_object_grasp")
    runner_start_delay = LaunchConfiguration("runner_start_delay")
    ros_python = LaunchConfiguration("ros_python")
    attach_topic = PythonExpression(["'/ur3_sim/attach_", grasp_object_model, "'"])
    detach_topic = PythonExpression(["'/ur3_sim/detach_", grasp_object_model, "'"])
    attachment_state_topic = PythonExpression([
        "'/ur3_sim/", grasp_object_model, "_attached'"
    ])

    arguments = [
        DeclareLaunchArgument("ur_type", default_value="ur3"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("run_script", default_value="true"),
        DeclareLaunchArgument("episode_count", default_value="3"),
        DeclareLaunchArgument("metrics_output", default_value=""),
        DeclareLaunchArgument("stop_on_failure", default_value="false"),
        DeclareLaunchArgument("shutdown_on_completion", default_value="false"),
        DeclareLaunchArgument("object_pose_source", default_value="fixed"),
        DeclareLaunchArgument("randomize_object", default_value="false"),
        DeclareLaunchArgument("random_seed", default_value="7"),
        DeclareLaunchArgument("robustness_profile", default_value="robustness_none.yaml"),
        DeclareLaunchArgument("return_home", default_value="true"),
        DeclareLaunchArgument("grasp_object_model", default_value="red_cube"),
        DeclareLaunchArgument("grasp_object_color", default_value="red"),
        DeclareLaunchArgument("grasp_object_link", default_value="cube_link"),
        DeclareLaunchArgument("distractor_object_name", default_value=""),
        DeclareLaunchArgument("multi_object_grasp", default_value="false"),
        DeclareLaunchArgument(
            "runner_start_delay",
            default_value="30.0",
            description=(
                "Wall-clock startup gate before the manipulation runner; "
                "the runner still waits for a timestamped joint state"
            ),
        ),
        DeclareLaunchArgument(
            "ros_python",
            default_value="/usr/bin/python3",
        ),
        DeclareLaunchArgument(
            "world_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("ur_simulation_gz"), "worlds", "ur3_pick_place.sdf"]
            ),
        ),
    ]

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur3_moveit_control"), "launch", "ur3_susgrip_sim.launch.py"]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "launch_rviz": launch_rviz,
            "gazebo_gui": gazebo_gui,
            "world_file": world_file,
            "controllers_file": "ur3_susgrip_gazebo_controllers.yaml",
            "grasp_object_model": grasp_object_model,
            "grasp_object_color": grasp_object_color,
            "grasp_object_link": grasp_object_link,
            "multi_object_grasp": multi_object_grasp,
        }.items(),
    )

    # Reuse the existing MoveGroupInterface wrapper without launching a second move_group.
    motion_action_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur3_moveit_control"), "launch", "ur3_demo_gripper.launch.py"]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "use_sim_time": "true",
            "launch_rviz": "false",
            "launch_moveit": "false",
        }.items(),
    )

    fixed_pick_place = Node(
        package="ur3_moveit_control",
        executable="fixed_pick_place",
        name="fixed_pick_place",
        output="screen",
        prefix=[ros_python],
        additional_env={"PYTHONNOUSERSITE": "1"},
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("ur3_moveit_control"), "config", "fixed_pick_place.yaml"]
            ),
            PathJoinSubstitution(
                [FindPackageShare("ur3_moveit_control"), "config", robustness_profile]
            ),
            {"use_sim_time": True},
            {
                "episode_count": ParameterValue(episode_count, value_type=int),
                "metrics_output": metrics_output,
                "stop_on_failure": ParameterValue(stop_on_failure, value_type=bool),
                "object_pose_source": object_pose_source,
                "randomize_object": ParameterValue(randomize_object, value_type=bool),
                "random_seed": ParameterValue(random_seed, value_type=int),
                "return_home": ParameterValue(return_home, value_type=bool),
                "object_id": grasp_object_model,
                "object_color": grasp_object_color,
                "attach_topic": attach_topic,
                "detach_topic": detach_topic,
                "attachment_state_topic": attachment_state_topic,
                "distractor_object_name": distractor_object_name,
            },
        ],
        condition=IfCondition(run_script),
    )

    delayed_trial = TimerAction(period=runner_start_delay, actions=[fixed_pick_place])
    shutdown_after_trial = RegisterEventHandler(
        OnProcessExit(
            target_action=fixed_pick_place,
            on_exit=[EmitEvent(event=Shutdown(reason="baseline trial completed"))],
        ),
        condition=IfCondition(shutdown_on_completion),
    )

    return LaunchDescription(
        arguments + [simulation, motion_action_server, delayed_trial, shutdown_after_trial]
    )
