import os
import subprocess
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory


def _generated_robot():
    xacro_path = os.path.join(
        get_package_share_directory("ur3_moveit_control"),
        "urdf",
        "ur3_with_susgrip.urdf.xacro",
    )
    output = subprocess.check_output(
        [
            "xacro",
            xacro_path,
            "ur_type:=ur3",
            "name:=ur",
            "sim_ignition:=true",
            "camera_update_rate:=30",
        ],
        text=True,
    )
    return ET.fromstring(output)


def test_d435i_frames_geometry_and_registered_sensor():
    robot = _generated_robot()
    links = {element.attrib["name"] for element in robot.findall("link")}
    assert {
        "camera_link",
        "camera_color_frame",
        "camera_color_optical_frame",
        "camera_depth_frame",
        "camera_depth_optical_frame",
        "camera_accel_frame",
        "camera_gyro_frame",
    }.issubset(links)

    mesh = robot.find("./link[@name='camera_link']/visual/geometry/mesh")
    assert mesh is not None
    assert mesh.attrib["filename"].endswith("realsense2_description/meshes/d435.dae")

    mount = robot.find("./joint[@name='camera_fixed_joint']/origin")
    assert mount is not None
    assert mount.attrib["xyz"] == "0.001623 -0.106306 0.013148"
    assert mount.attrib["rpy"] == "0.163640 -1.535191 1.364251"

    color = robot.find("./joint[@name='camera_color_joint']/origin")
    assert color is not None
    assert abs(float(color.attrib["xyz"].split()[1]) - 0.0149876224) < 1e-9

    sensor = robot.find("./gazebo/sensor[@name='d435i_aligned_rgbd']")
    assert sensor is not None
    assert sensor.findtext("update_rate") == "30"
    assert sensor.findtext("camera/optical_frame_id") == "camera_color_optical_frame"
    assert sensor.findtext("camera/image/width") == "640"
    assert sensor.findtext("camera/image/height") == "480"
    assert abs(float(sensor.findtext("camera/lens/intrinsics/fx")) - 606.0816650391) < 1e-6
    assert abs(float(sensor.findtext("camera/lens/projection/p_cx")) - 325.5436706543) < 1e-6
