from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # 1. Khởi chạy nhận diện ArUco Marker
        Node(
            package='aruco_ros',
            executable='single',
            name='aruco_single',
            parameters=[{
                'image_is_rectified': True,
                'marker_size': 0.1,  # Đổi lại kích thước thật (mét) ở đây! Ví dụ 0.1m = 10cm
                'marker_id': 26,     # Đổi ID của Marker ở đây
                'reference_frame': 'camera_color_optical_frame',
                'camera_frame': 'camera_color_optical_frame',
                'marker_frame': 'aruco_marker_frame'
            }],
            remappings=[
                ('/camera_info', '/camera/camera/color/camera_info'),
                ('/image', '/camera/camera/color/image_raw')
            ]
        ),
        
    ])
