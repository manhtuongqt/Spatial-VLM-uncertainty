from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    declared_arguments = []
    
    # Declare arguments that we can pass to ur_sim_control
    declared_arguments.append(
        DeclareLaunchArgument("ur_type", default_value="ur3", description="UR robot type")
    )
    declared_arguments.append(
        DeclareLaunchArgument("launch_rviz", default_value="true", description="Launch RViz?")
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "controllers_file",
            default_value="ur3_susgrip_gazebo_controllers.yaml",
            description="ros2_control controller-manager YAML used by Gazebo.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "world_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("ur_simulation_gz"), "worlds", "ur3_pick_place.sdf"]
            ),
            description="Absolute path to the Gazebo pick-and-place world.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument("gazebo_gui", default_value="true", description="Start Gazebo GUI?")
    )
    declared_arguments.append(DeclareLaunchArgument("grasp_object_model", default_value="red_cube"))
    declared_arguments.append(DeclareLaunchArgument("grasp_object_color", default_value="red"))
    declared_arguments.append(DeclareLaunchArgument("grasp_object_link", default_value="cube_link"))
    declared_arguments.append(DeclareLaunchArgument("multi_object_grasp", default_value="false"))
    declared_arguments.append(DeclareLaunchArgument("camera_update_rate", default_value="60"))
    declared_arguments.append(
        DeclareLaunchArgument(
            "gripper_controller_delay",
            default_value="18.0",
            description=(
                "Delay gripper-controller activation until Gazebo and local AI "
                "model startup load has settled"
            ),
        )
    )

    ur_type = LaunchConfiguration("ur_type")
    launch_rviz = LaunchConfiguration("launch_rviz")
    controllers_file = LaunchConfiguration("controllers_file")
    world_file = LaunchConfiguration("world_file")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    grasp_object_model = LaunchConfiguration("grasp_object_model")
    grasp_object_color = LaunchConfiguration("grasp_object_color")
    grasp_object_link = LaunchConfiguration("grasp_object_link")
    multi_object_grasp = LaunchConfiguration("multi_object_grasp")
    camera_update_rate = LaunchConfiguration("camera_update_rate")
    gripper_controller_delay = LaunchConfiguration("gripper_controller_delay")

    # Include the Gazebo Simulation from UR packages
    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare("ur_simulation_gz"), "launch", "ur_sim_control.launch.py"])
        ]),
        launch_arguments={
            "ur_type": ur_type,
            "description_package": "ur3_moveit_control",
            "description_file": "ur3_with_susgrip.urdf.xacro",
            "runtime_config_package": "ur3_moveit_control",
            "controllers_file": controllers_file,
            "launch_rviz": "false",  # We launch RViz with MoveIt instead
            "gazebo_gui": gazebo_gui,
            "world_file": world_file,
            "grasp_object_model": grasp_object_model,
            "grasp_object_color": grasp_object_color,
            "grasp_object_link": grasp_object_link,
            "multi_object_grasp": multi_object_grasp,
            "camera_update_rate": camera_update_rate,
        }.items()
    )

    # Include MoveIt launch
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare("ur_moveit_config"), "launch", "ur_moveit.launch.py"])
        ]),
        launch_arguments={
            "ur_type": ur_type,
            "description_package": "ur3_moveit_control",
            "description_file": "ur3_with_susgrip.urdf.xacro",
            "moveit_config_package": "ur_moveit_config",
            "moveit_config_file": "ur3_with_susgrip.srdf.xacro",
            "use_sim_time": "true",
            "launch_rviz": launch_rviz,
            "grasp_object_model": grasp_object_model,
            "grasp_object_color": grasp_object_color,
        }.items()
    )

    # Include spawner for gripper_controller
    from launch_ros.actions import Node
    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "gripper_controller",
            "-c", "/controller_manager",
            "--controller-manager-timeout", "60",
            "--service-call-timeout", "60",
        ],
        parameters=[{"use_sim_time": True}],
    )

    delayed_gripper_controller_spawner = TimerAction(
        period=gripper_controller_delay,
        actions=[gripper_controller_spawner],
    )

    return LaunchDescription(declared_arguments + [
        gz_sim_launch,
        moveit_launch,
        delayed_gripper_controller_spawner
    ])
