"""Run the physical D435i behind the same topics used by simulation."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    serial_no = LaunchConfiguration("serial_no")
    publish_tf = LaunchConfiguration("publish_tf")
    pointcloud = LaunchConfiguration("pointcloud")

    # Keep node name "camera" so frame IDs exactly match the URDF and the
    # simulation: camera_link, camera_color_optical_frame, ...
    common_interface = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace="",
        name="camera",
        output="screen",
        parameters=[{
            "serial_no": ParameterValue(serial_no, value_type=str),
            "rgb_camera.color_profile": "640,480,30",
            "depth_module.depth_profile": "640,480,30",
            "rgb_camera.color_format": "RGB8",
            "depth_module.depth_format": "Z16",
            "align_depth.enable": True,
            "enable_sync": True,
            "enable_color": True,
            "enable_depth": True,
            "enable_infra": False,
            "enable_infra1": False,
            "enable_infra2": False,
            "pointcloud.enable": ParameterValue(pointcloud, value_type=bool),
            "publish_tf": ParameterValue(publish_tf, value_type=bool),
            "enable_gyro": False,
            "enable_accel": False,
        }],
        remappings=[
            ("/camera/color/image_raw", "/wrist_camera/color/image_raw"),
            ("/camera/color/camera_info", "/wrist_camera/color/camera_info"),
            ("/camera/aligned_depth_to_color/image_raw", "/wrist_camera/depth/image_raw"),
            ("/camera/aligned_depth_to_color/camera_info", "/wrist_camera/depth/camera_info"),
            ("/camera/depth/color/points", "/wrist_camera/points"),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "serial_no",
            default_value="_243722071709",
            description="D435i serial; leading underscore keeps it a string.",
        ),
        DeclareLaunchArgument(
            "publish_tf",
            default_value="true",
            description=(
                "True for standalone camera practice; false when robot_state_publisher "
                "already publishes the D435i internal frames."
            ),
        ),
        DeclareLaunchArgument("pointcloud", default_value="false"),
        common_interface,
    ])
