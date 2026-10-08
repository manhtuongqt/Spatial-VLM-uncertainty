"""Perception-driven UR3 pick/place while retaining the fixed fallback runner."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    launch_rviz = LaunchConfiguration("launch_rviz")
    image_view = LaunchConfiguration("image_view")
    episode_count = LaunchConfiguration("episode_count")
    metrics_output = LaunchConfiguration("metrics_output")
    shutdown_on_completion = LaunchConfiguration("shutdown_on_completion")
    randomize_object = LaunchConfiguration("randomize_object")
    random_seed = LaunchConfiguration("random_seed")
    robustness_profile = LaunchConfiguration("robustness_profile")
    stop_on_failure = LaunchConfiguration("stop_on_failure")
    return_home = LaunchConfiguration("return_home")
    object_pose_source = LaunchConfiguration("object_pose_source")
    grasp_object_model = LaunchConfiguration("grasp_object_model")
    grasp_object_color = LaunchConfiguration("grasp_object_color")
    grasp_object_link = LaunchConfiguration("grasp_object_link")
    distractor_object_name = LaunchConfiguration("distractor_object_name")
    multi_object_grasp = LaunchConfiguration("multi_object_grasp")
    runner_start_delay = LaunchConfiguration("runner_start_delay")
    require_target_selection = LaunchConfiguration("require_target_selection")
    use_grounder_grasp_pixel = LaunchConfiguration("use_grounder_grasp_pixel")
    ros_python = LaunchConfiguration("ros_python")

    baseline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur3_moveit_control"),
            "launch",
            "baseline_fixed_pick_place.launch.py",
        ])),
        launch_arguments={
            "ur_type": "ur3",
            "launch_rviz": launch_rviz,
            "gazebo_gui": gazebo_gui,
            "run_script": "true",
            "episode_count": episode_count,
            "metrics_output": metrics_output,
            "stop_on_failure": stop_on_failure,
            "shutdown_on_completion": shutdown_on_completion,
            "object_pose_source": object_pose_source,
            "randomize_object": randomize_object,
            "random_seed": random_seed,
            "robustness_profile": robustness_profile,
            "return_home": return_home,
            "grasp_object_model": grasp_object_model,
            "grasp_object_color": grasp_object_color,
            "grasp_object_link": grasp_object_link,
            "distractor_object_name": distractor_object_name,
            "multi_object_grasp": multi_object_grasp,
            "runner_start_delay": runner_start_delay,
            "ros_python": ros_python,
        }.items(),
    )
    perception = Node(
        package="ur3_perception",
        executable="rgbd_object_pose.py",
        name="rgbd_object_pose",
        output="screen",
        prefix=[ros_python],
        additional_env={"PYTHONNOUSERSITE": "1"},
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("ur3_perception"), "config", "rgbd_perception.yaml"
            ]),
            {
                "require_target_selection": ParameterValue(
                    require_target_selection, value_type=bool
                ),
                "use_grounder_grasp_pixel": ParameterValue(
                    use_grounder_grasp_pixel, value_type=bool
                ),
            },
        ],
    )
    image_viewer = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="perception_image_view",
        arguments=["/ur3_perception/debug_image"],
        output="log",
        condition=IfCondition(image_view),
    )

    return LaunchDescription([
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("image_view", default_value="true"),
        DeclareLaunchArgument("episode_count", default_value="1"),
        DeclareLaunchArgument(
            "metrics_output", default_value="/tmp/ur3_perception_pick_place.json"
        ),
        DeclareLaunchArgument("shutdown_on_completion", default_value="false"),
        DeclareLaunchArgument("randomize_object", default_value="true"),
        DeclareLaunchArgument("random_seed", default_value="7"),
        DeclareLaunchArgument("robustness_profile", default_value="robustness_none.yaml"),
        DeclareLaunchArgument("stop_on_failure", default_value="true"),
        DeclareLaunchArgument("return_home", default_value="false"),
        DeclareLaunchArgument("object_pose_source", default_value="perception"),
        DeclareLaunchArgument("grasp_object_model", default_value="red_cube"),
        DeclareLaunchArgument("grasp_object_color", default_value="red"),
        DeclareLaunchArgument("grasp_object_link", default_value="cube_link"),
        DeclareLaunchArgument("distractor_object_name", default_value=""),
        DeclareLaunchArgument("multi_object_grasp", default_value="false"),
        DeclareLaunchArgument("runner_start_delay", default_value="30.0"),
        DeclareLaunchArgument("require_target_selection", default_value="false"),
        DeclareLaunchArgument("use_grounder_grasp_pixel", default_value="false"),
        DeclareLaunchArgument(
            "ros_python",
            default_value="/usr/bin/python3",
        ),
        perception,
        baseline,
        TimerAction(period=8.0, actions=[image_viewer]),
    ])
