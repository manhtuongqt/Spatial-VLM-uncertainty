# Copyright (c) 2021 Stogl Robotics Consulting UG (haftungsbeschränkt)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the {copyright_holder} nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Denis Stogl

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    # Initialize Arguments
    ur_type = LaunchConfiguration("ur_type")
    safety_limits = LaunchConfiguration("safety_limits")
    safety_pos_margin = LaunchConfiguration("safety_pos_margin")
    safety_k_position = LaunchConfiguration("safety_k_position")
    # General arguments
    runtime_config_package = LaunchConfiguration("runtime_config_package")
    controllers_file = LaunchConfiguration("controllers_file")
    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    prefix = LaunchConfiguration("prefix")
    start_joint_controller = LaunchConfiguration("start_joint_controller")
    initial_joint_controller = LaunchConfiguration("initial_joint_controller")
    launch_rviz = LaunchConfiguration("launch_rviz")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    world_file = LaunchConfiguration("world_file")
    grasp_object_model = LaunchConfiguration("grasp_object_model")
    grasp_object_color = LaunchConfiguration("grasp_object_color")
    grasp_object_link = LaunchConfiguration("grasp_object_link")
    multi_object_grasp = LaunchConfiguration("multi_object_grasp")
    enable_ground_truth_pose_bridge = LaunchConfiguration(
        "enable_ground_truth_pose_bridge"
    )
    camera_update_rate = LaunchConfiguration("camera_update_rate")

    initial_joint_controllers = PathJoinSubstitution(
        [FindPackageShare(runtime_config_package), "config", controllers_file]
    )

    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare(description_package), "rviz", "view_robot.rviz"]
    )

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare(description_package), "urdf", description_file]
            ),
            " ",
            "safety_limits:=",
            safety_limits,
            " ",
            "safety_pos_margin:=",
            safety_pos_margin,
            " ",
            "safety_k_position:=",
            safety_k_position,
            " ",
            "name:=",
            "ur",
            " ",
            "ur_type:=",
            ur_type,
            " ",
            "prefix:=",
            prefix,
            " ",
            "sim_ignition:=true",
            " ",
            "simulation_controllers:=",
            initial_joint_controllers,
            " ",
            "grasp_object_model:=",
            grasp_object_model,
            " ",
            "grasp_object_color:=",
            grasp_object_color,
            " ",
            "grasp_object_link:=",
            grasp_object_link,
            " ",
            "multi_object_grasp:=",
            multi_object_grasp,
            " ",
            "camera_update_rate:=",
            camera_update_rate,
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"use_sim_time": True}, robot_description],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(launch_rviz),
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "60",
            "--service-call-timeout", "60",
        ],
    )

    # Delay rviz start after `joint_state_broadcaster`
    delay_rviz_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[rviz_node],
        ),
        condition=IfCondition(launch_rviz),
    )

    # There may be other controllers of the joints, but this is the initially-started one
    initial_joint_controller_spawner_started = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            initial_joint_controller,
            "-c", "/controller_manager",
            "--controller-manager-timeout", "60",
            "--service-call-timeout", "60",
        ],
        condition=IfCondition(start_joint_controller),
    )
    initial_joint_controller_spawner_stopped = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            initial_joint_controller,
            "-c", "/controller_manager",
            "--inactive",
            "--controller-manager-timeout", "60",
            "--service-call-timeout", "60",
        ],
        condition=UnlessCondition(start_joint_controller),
    )

    # Spawn topic-based controllers in stopped state
    forward_position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "forward_position_controller",
            "-c", "/controller_manager",
            "--inactive",
            "--controller-manager-timeout", "60",
            "--service-call-timeout", "60",
        ],
    )

    forward_velocity_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "forward_velocity_controller",
            "-c", "/controller_manager",
            "--inactive",
            "--controller-manager-timeout", "60",
            "--service-call-timeout", "60",
        ],
    )

    # Loading several controllers concurrently can starve the Gazebo control
    # update loop on low-power CPUs.  A load service may then complete inside
    # controller_manager after the client timeout; the spawner retries, sees an
    # already-loaded controller and exits before activation.  Serialize the
    # startup chain so each controller reaches its requested state exactly once.
    start_initial_controller_after_joint_states = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[initial_joint_controller_spawner_started],
        ),
        condition=IfCondition(start_joint_controller),
    )
    load_initial_controller_after_joint_states = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[initial_joint_controller_spawner_stopped],
        ),
        condition=UnlessCondition(start_joint_controller),
    )
    start_position_controller_after_initial_started = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=initial_joint_controller_spawner_started,
            on_exit=[forward_position_controller_spawner],
        ),
        condition=IfCondition(start_joint_controller),
    )
    start_position_controller_after_initial_stopped = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=initial_joint_controller_spawner_stopped,
            on_exit=[forward_position_controller_spawner],
        ),
        condition=UnlessCondition(start_joint_controller),
    )
    start_velocity_controller_after_position = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=forward_position_controller_spawner,
            on_exit=[forward_velocity_controller_spawner],
        )
    )

    # GZ nodes
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-string",
            robot_description_content,
            "-name",
            "ur",
            "-allow_renaming",
            "true",
        ],
    )
    gz_launch_description_with_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={"gz_args": [" -r -v 4 ", world_file]}.items(),
        condition=IfCondition(gazebo_gui),
    )

    gz_launch_description_without_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={"gz_args": [" -s -r -v 4 ", world_file]}.items(),
        condition=UnlessCondition(gazebo_gui),
    )

    # Make the /clock topic available in ROS
    gz_sim_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
            "/ur3_sim/attach_red_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_red_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/red_cube_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_green_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_green_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/green_cube_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_blue_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_blue_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/blue_cube_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_yellow_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_yellow_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/yellow_cube_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_orange_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_orange_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/orange_cube_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_purple_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_purple_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/purple_cube_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_pink_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_pink_cube@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/pink_cube_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_mango@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_mango@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/mango_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_ycb_cracker_box@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_ycb_cracker_box@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/ycb_cracker_box_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_ycb_sugar_box@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_ycb_sugar_box@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/ycb_sugar_box_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_ycb_tomato_soup_can@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_ycb_tomato_soup_can@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/ycb_tomato_soup_can_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_ycb_mustard_bottle@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_ycb_mustard_bottle@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/ycb_mustard_bottle_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_ycb_banana@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_ycb_banana@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/ycb_banana_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_ycb_apple@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_ycb_apple@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/ycb_apple_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_ycb_orange@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_ycb_orange@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/ycb_orange_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/ur3_sim/attach_ycb_power_drill@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/detach_ycb_power_drill@std_msgs/msg/Empty]ignition.msgs.Empty",
            "/ur3_sim/ycb_power_drill_attached@std_msgs/msg/String[ignition.msgs.StringMsg",
            "/world/ur3_pick_place/set_pose@ros_gz_interfaces/srv/SetEntityPose",
            "/wrist_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/wrist_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/wrist_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
            "/wrist_camera/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
            "/top_table_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/top_table_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/top_table_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
            "/top_table_camera/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
            "/top_table_camera/evaluation_labels/labels_map@sensor_msgs/msg/Image[ignition.msgs.Image",
            # Evaluation-only semantic labels captured concurrently but opened
            # only after prediction lock. No core perception or control
            # node subscribes to this topic. The capture node writes it into a
            # separate evaluator-only directory, which the evaluator opens
            # only after B0/B1/B2 predictions are locked on disk.
            "/wrist_camera/evaluation_labels/labels_map@sensor_msgs/msg/Image[ignition.msgs.Image",
        ],
        remappings=[
            ("/wrist_camera/image", "/wrist_camera/color/image_raw"),
            ("/wrist_camera/depth_image", "/wrist_camera/depth/image_raw"),
            ("/wrist_camera/camera_info", "/wrist_camera/color/camera_info"),
            ("/top_table_camera/image", "/top_table_camera/color/image_raw"),
            ("/top_table_camera/depth_image", "/top_table_camera/depth/image_raw"),
            ("/top_table_camera/camera_info", "/top_table_camera/color/camera_info"),
        ],
        output="screen",
    )

    # Pose_V grows with every dynamic model.  On Fast-DDS used by ROS 2
    # Humble, bridging the large vector as TFMessage can corrupt CDR reads for
    # every participant ("sequence size exceeds remaining buffer").  Keep the
    # research oracle opt-in and isolate it from the camera/control bridge so
    # interactive perception demos remain reliable.
    ground_truth_pose_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ground_truth_pose_bridge",
        arguments=[
            "/world/ur3_pick_place/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
        ],
        remappings=[
            ("/world/ur3_pick_place/dynamic_pose/info", "/ur3_perception/oracle_tf"),
        ],
        output="screen",
        condition=IfCondition(enable_ground_truth_pose_bridge),
    )

    nodes_to_start = [
        robot_state_publisher_node,
        joint_state_broadcaster_spawner,
        delay_rviz_after_joint_state_broadcaster_spawner,
        start_initial_controller_after_joint_states,
        load_initial_controller_after_joint_states,
        start_position_controller_after_initial_started,
        start_position_controller_after_initial_stopped,
        start_velocity_controller_after_position,
        gz_spawn_entity,
        gz_launch_description_with_gui,
        gz_launch_description_without_gui,
        gz_sim_bridge,
        ground_truth_pose_bridge,
    ]

    return nodes_to_start


def generate_launch_description():
    declared_arguments = []
    # UR specific arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "camera_update_rate",
            default_value="60",
            description="Simulated RGB-D and evaluator-label update rate in Hz.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            description="Type/series of used UR robot.",
            choices=[
                "ur3",
                "ur5",
                "ur10",
                "ur3e",
                "ur5e",
                "ur7e",
                "ur10e",
                "ur12e",
                "ur16e",
                "ur8long",
                "ur15",
                "ur18",
                "ur20",
                "ur30",
            ],
            default_value="ur5e",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_limits",
            default_value="true",
            description="Enables the safety limits controller if true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_pos_margin",
            default_value="0.15",
            description="The margin to lower and upper limits in the safety controller.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_k_position",
            default_value="20",
            description="k-position factor in the safety controller.",
        )
    )
    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "runtime_config_package",
            default_value="ur_simulation_gz",
            description='Package with the controller\'s configuration in "config" folder. \
        Usually the argument is not set, it enables use of a custom setup.',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "controllers_file",
            default_value="ur_controllers.yaml",
            description="YAML file with the controllers configuration.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="ur3_moveit_control",
            description="Description package with robot URDF/XACRO files. Usually the argument \
        is not set, it enables use of a custom description.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value="ur3_with_susgrip.urdf.xacro",
            description="URDF/XACRO description file with the robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument("grasp_object_model", default_value="red_cube")
    )
    declared_arguments.append(
        DeclareLaunchArgument("grasp_object_color", default_value="red")
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "grasp_object_link",
            default_value="cube_link",
            description="Link of the Gazebo model used by the detachable joint.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument("multi_object_grasp", default_value="false")
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value='""',
            description="Prefix of the joint names, useful for \
        multi-robot setup. If changed than also joint names in the controllers' configuration \
        have to be updated.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "start_joint_controller",
            default_value="true",
            description="Enable headless mode for robot control",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="joint_trajectory_controller",
            description="Robot controller to start.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument("launch_rviz", default_value="false", description="Launch RViz?")
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "gazebo_gui", default_value="true", description="Start gazebo with GUI?"
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "enable_ground_truth_pose_bridge",
            default_value="false",
            description=(
                "Bridge the full Gazebo dynamic Pose_V oracle into ROS. "
                "Disabled for demos because large worlds can exceed the "
                "ROS 2 Humble Fast-CDR receive buffer."
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "world_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("ur_simulation_gz"), "worlds", "ur3_pick_place.sdf"]
            ),
            description="Gazebo world file (absolute path or filename from the gazebosim worlds collection) containing a custom world.",
        )
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
