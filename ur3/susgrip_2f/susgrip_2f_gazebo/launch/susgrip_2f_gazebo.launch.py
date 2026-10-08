import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration,PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import OpaqueFunction
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit


def launch_setup(context,*arg,**kwarg):
    description_pkg = 'susgrip_2f_description'
    gazebo_pkg = 'susgrip_2f_gazebo'

    # Set GAZEBO_MODEL_PATH dynamically so Gazebo can find model:// meshes
    parent_dir = os.path.dirname(get_package_share_directory(description_pkg))
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] = parent_dir + os.pathsep + os.environ['GAZEBO_MODEL_PATH']
    else:
        os.environ['GAZEBO_MODEL_PATH'] = parent_dir

    # Disable online model fetching to prevent Gazebo from hanging during startup
    os.environ['GAZEBO_MODEL_DATABASE_URI'] = ""

    # Paths
    pkg_share = get_package_share_directory(description_pkg)
    rviz_urdf_file = os.path.join(pkg_share, 'urdf', 'susgrip_2f.urdf')
    gazebo_pkg_share = get_package_share_directory(gazebo_pkg)
    gazebo_urdf_file = os.path.join(gazebo_pkg_share, 'urdf', 'susgrip_2f_gazebo.urdf')

    from launch.substitutions import Command, FindExecutable
    robot_description_config = Command(
        [FindExecutable(name='xacro'), ' ', rviz_urdf_file]
    )
    # Find the workspace root and locate the source RViz config file
    # This allows RViz to save settings directly to the source code, avoiding write-permission issues in install/
    share_dir = get_package_share_directory(description_pkg)
    ws_root = share_dir
    for _ in range(4):
        ws_root = os.path.dirname(ws_root)
    src_rviz_config = os.path.join(ws_root, 'src', description_pkg, 'config', 'susgrip_2f_displayv2.rviz')
    
    if os.path.exists(src_rviz_config):
        rviz_config_file = src_rviz_config
    else:
        rviz_config_file = os.path.join(share_dir, 'config', 'susgrip_2f_displayv2.rviz')

    from launch_ros.parameter_descriptions import ParameterValue
    # Robot State Publisher
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(robot_description_config, value_type=str),
            'use_sim_time': True
        }],
        remappings=[
            ('/joint_states', '/rviz_joint_states')
        ]
    )
    #Rviz
    node_rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
    )

    # Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments={'verbose': 'false'}.items()
    )
    # Read and replace path in gazebo URDF
    with open(gazebo_urdf_file, 'r') as f:
        gazebo_urdf_content = f.read()
    # Resolve $(find susgrip_2f_gazebo) dynamically to its absolute share path
    gazebo_urdf_content = gazebo_urdf_content.replace(
        '$(find susgrip_2f_gazebo)', 
        gazebo_pkg_share
    )
    # Convert package:// to absolute file:// to bypass Gazebo's model loader completely
    description_pkg_share = get_package_share_directory(description_pkg)
    gazebo_urdf_content = gazebo_urdf_content.replace(
        f'package://{description_pkg}',
        f'file://{description_pkg_share}'
    )

    # Robot State Publisher for Gazebo (simple model)
    # We publish it to a different topic name to avoid conflict with the RViz one
    node_robot_state_publisher_gazebo = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher_gazebo',
        output='both',
        parameters=[{
            'robot_description': ParameterValue(gazebo_urdf_content, value_type=str),
            'use_sim_time': True,
        }],
        remappings=[
            ('/robot_description', '/robot_description_gazebo')
        ]
    )

    # Spawn Robot from topic
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', '/robot_description_gazebo', '-entity', 'susgrip'],
        output='screen'
    )

    # 1. Định nghĩa Spawner cho Joint State Broadcaster
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager']
    )

    # Đợi spawn xong + plugin gazebo_ros2_control khởi tạo controller_manager
    # Dùng TimerAction delay 5 giây sau khi toàn bộ launch khởi động
    delay_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[
                TimerAction(
                    period=5.0,  # Đợi 5 giây cho gazebo_ros2_control khởi động
                    actions=[load_joint_state_broadcaster]
                )
            ],
        )
    )

    # Đợi joint_state_broadcaster nạp xong thành công mới chạy Gripper Controller

    # 2. Định nghĩa Spawner cho Gripper Controller
    load_gripper_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
    )

    delay_gripper_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[load_gripper_controller],
        )
    )

    # Gazebo_publisher_node
    gazebo_publisher = Node(
        package='susgrip_2f_gazebo',
        executable='gazebo_joint_publisher',
        name='gazebo_joint_publisher',
        output='screen',
    )

    # Thay thế các node spawner cũ bằng các node delay đã được thiết lập sự kiện tuần tự
    node_to_start = [
        node_robot_state_publisher,
        node_robot_state_publisher_gazebo,
        node_rviz,
        gazebo,
        spawn_entity,
        delay_joint_state_broadcaster,  # Thay đổi ở đây
        delay_gripper_controller,      # Thay đổi ở đây
        gazebo_publisher
    ]
    return node_to_start

def generate_launch_description():

    return LaunchDescription([OpaqueFunction(function=launch_setup)])
