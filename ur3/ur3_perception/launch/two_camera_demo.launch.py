"""Start Gazebo and show its wrist and fixed top-table RGB views side by side."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gazebo_gui = LaunchConfiguration('gazebo_gui')
    viewer_delay = LaunchConfiguration('viewer_delay')
    world_file = LaunchConfiguration('world_file')
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('ur3_moveit_control'), 'launch', 'ur3_susgrip_sim.launch.py'
        ])),
        launch_arguments={
            'ur_type': 'ur3',
            'gazebo_gui': gazebo_gui,
            'launch_rviz': 'false',
            'world_file': world_file,
        }.items(),
    )
    wrist_view = Node(
        package='rqt_image_view', executable='rqt_image_view', name='wrist_camera_view',
        arguments=['/wrist_camera/color/image_raw'], output='log',
        prefix=['/usr/bin/python3'], additional_env={'PYTHONNOUSERSITE': '1'})
    top_view = Node(
        package='rqt_image_view', executable='rqt_image_view', name='top_table_camera_view',
        arguments=['/top_table_camera/color/image_raw'], output='log',
        prefix=['/usr/bin/python3'], additional_env={'PYTHONNOUSERSITE': '1'})
    camera_view_pose = Node(
        package='ur3_perception', executable='move_camera_to_view.py',
        name='move_wrist_camera_to_view', output='screen')
    return LaunchDescription([
        DeclareLaunchArgument('gazebo_gui', default_value='true'),
        DeclareLaunchArgument(
            'world_file',
            default_value='/home/dhcn/ur_ws/src/myproject/workspace/mh_pcrau_v3/generated/day6_top30_canary_v3/low_top30_canary.sdf',
            description='Gazebo world used for the robot plus two-camera demonstration.',
        ),
        DeclareLaunchArgument('viewer_delay', default_value='18.0',
                              description='Seconds to wait for Gazebo and wrist pose before opening viewers.'),
        simulation,
        TimerAction(period=10.0, actions=[camera_view_pose]),
        TimerAction(period=viewer_delay, actions=[wrist_view, top_view]),
    ])
