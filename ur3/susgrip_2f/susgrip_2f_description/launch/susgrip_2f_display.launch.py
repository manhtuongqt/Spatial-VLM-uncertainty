from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import UnlessCondition
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    share_dir = get_package_share_directory('susgrip_2f_description')

    urdf_file = os.path.join(share_dir, 'urdf', 'susgrip_2f.urdf')
    if not os.path.exists(urdf_file):
        raise FileNotFoundError(f"URDF file not found: {urdf_file}")

    with open(urdf_file, 'r') as file:
        robot_description_content = file.read()

    params = {'robot_description': robot_description_content}

    rviz_config_file = os.path.join(share_dir, 'config', 'susgrip_2f_display.rviz')

    gui_arg = DeclareLaunchArgument(
        name='gui',
        default_value='True'
    )

    show_gui = LaunchConfiguration('gui')

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[params],
    )

    joint_state_publisher_node = Node(
        condition=UnlessCondition(show_gui),
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[params],
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen',
    )
    visualize_node = Node(
        package='susgrip_2f_description',
        executable='susgrip_2f_visualize',
        name='susgrip_2f_visualize',
        output='screen',
        parameters=[params],
    )
    return LaunchDescription([
        gui_arg,
        robot_state_publisher_node,
        joint_state_publisher_node,
        rviz_node,
        visualize_node
    ])
