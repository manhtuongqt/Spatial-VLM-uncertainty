#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def launch_setup(context, *args, **kwargs):
    description_package = FindPackageShare('susgrip_2f_description')
    
    # Initialize Arguments
    sus2f_port = LaunchConfiguration("port")
    slave_id = LaunchConfiguration("slave_id")
    launch_rviz = LaunchConfiguration("launch_rviz")
    
    share_dir = get_package_share_directory('susgrip_2f_description')
    urdf_file = os.path.join(share_dir, 'urdf', 'susgrip_2f.urdf')

    with open(urdf_file, 'r') as file:
        robot_description_content = file.read()
        
    robot_description = {"robot_description": robot_description_content}

    rviz_config_file = PathJoinSubstitution(
        [description_package, "config", "susgrip_2f_display.rviz"]
    )
    
    # Robot State Publisher to broadcast gripper TFs
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )
    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[
                robot_description,
                {
                    'source_list': ['/susgrip/joint_states'],
                    'rate': 20.0
                }
            ],
        output="screen",
    )

    # New Python-based Hardware Interface & Action Server
    susgrip_2f_hardware_node = Node(
        package="susgrip_2f_hardware",
        executable="hardware_interface",
        name="susgrip_2f_hardware_interface",
        output="screen",
        parameters=[
            {'serial_port': sus2f_port.perform(context)},
            {'slave_id': int(slave_id.perform(context))},
        ],
    )
    
    # Optional RViz2 for visualization
    rviz_node = Node(
        condition=IfCondition(launch_rviz),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )
    
    gui_node = Node(
        package="susgrip_2f_control",
        executable="with_gui",
        name="susgrip_2f_gui",
        output="screen",
    )

    return [
        susgrip_2f_hardware_node,
        joint_state_publisher_node,
        robot_state_publisher_node,
        rviz_node,
        gui_node
    ]

def generate_launch_description():
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "port",
            default_value="/dev/ttyUSB0",
            description="Serial port for Modbus RTU communication"
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "slave_id",
            default_value="1",
            description="Slave ID for Modbus gripper"
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz", 
            default_value="true", 
            description="Launch RViz?"
        )
    )
    
    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
