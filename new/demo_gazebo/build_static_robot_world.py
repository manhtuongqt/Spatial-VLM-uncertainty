#!/usr/bin/env python3
"""Add a static UR3 preview to the existing Gazebo world; no controller."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_WORLD = ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf"
PREVIEW_XACRO = Path(__file__).resolve().parent / "ur3_static_preview.urdf.xacro"
JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
TEST_IID_PLAN = ROOT / "new/test_iid/protocol/test_iid_capture_plan.json"
CLOSEUP_VIEW_JOINT_POSE = {
    # Preview-only pose: extends the original camera-view posture toward the
    # YCB apple and raises the D435i while keeping its optical axis downward.
    # The base remains at (0, 0, 0); no controller or trajectory is started.
    "shoulder_pan_joint": 1.5331,
    "shoulder_lift_joint": -1.3652,
    "elbow_joint": 0.6928,
    "wrist_1_joint": -0.9024,
    "wrist_2_joint": -1.5667,
    "wrist_3_joint": -1.6067,
}

YCB_LEFT_POSES = {
    # Cluster YCB models on the robot's -X side in world coordinates; two
    # boxes flank the bin. Screen-left is camera-dependent. Preserve source
    # models' z and yaw, changing only tabletop x/y layout.
    "ycb_cracker_box": (-0.13, 0.53),
    "ycb_sugar_box": (-0.13, 0.12),
    "ycb_tomato_soup_can": (-0.22, 0.46),
    "ycb_mustard_bottle": (-0.31, 0.22),
    "ycb_banana": (-0.33, 0.36),
    "ycb_apple": (-0.21, 0.32),
    "ycb_orange": (-0.21, 0.10),
    "ycb_power_drill": (-0.42, 0.66),
}


def set_model_xy(model: ET.Element, x: float, y: float) -> None:
    pose = model.find("pose")
    if pose is None or not pose.text:
        raise RuntimeError(f"Missing pose for {model.get('name')}")
    values = pose.text.split()
    if len(values) != 6:
        raise RuntimeError(f"Unexpected pose for {model.get('name')}")
    values[:2] = [f"{x:.4f}", f"{y:.4f}"]
    pose.text = " ".join(values)


def apply_demo_layout(world: ET.Element) -> None:
    models = {model.get("name"): model for model in world.findall("model")}
    for name, (x, y) in YCB_LEFT_POSES.items():
        set_model_xy(models[name], x, y)
    set_model_xy(models["target_bin"], 0.08, 0.32)
    # Keep legacy colored cubes in a back row, clear of the YCB demonstration.
    for index, name in enumerate(("red_cube", "blue_cube", "green_cube",
                                  "yellow_cube", "orange_cube", "purple_cube",
                                  "pink_cube")):
        set_model_xy(models[name], -0.55 + 0.15 * index, 0.92)

    camera = ET.fromstring("""
      <model name="left_oblique_table_camera">
        <static>true</static>
        <pose>-0.92 0.29 0.73 0 0.61 0</pose>
        <link name="left_oblique_table_camera_link">
          <sensor name="left_oblique_table_rgbd" type="rgbd_camera">
            <camera>
              <horizontal_fov>1.10</horizontal_fov>
              <image><width>640</width><height>480</height><format>R8G8B8</format></image>
              <clip><near>0.10</near><far>3.0</far></clip>
              <depth_camera><clip><near>0.10</near><far>3.0</far></clip></depth_camera>
              <optical_frame_id>left_oblique_table_camera_optical_frame</optical_frame_id>
            </camera>
            <always_on>true</always_on><update_rate>30</update_rate>
            <visualize>true</visualize><topic>/left_oblique_table_camera</topic>
            <gz_frame_id>left_oblique_table_camera_optical_frame</gz_frame_id>
          </sensor>
        </link>
      </model>
    """)
    world.append(camera)


def selected_view_pose(name: str) -> dict[str, float]:
    if name == "closeup_legacy":
        return CLOSEUP_VIEW_JOINT_POSE
    if name != "test_iid":
        raise ValueError(f"Unknown view pose: {name}")
    plan = json.loads(TEST_IID_PLAN.read_text(encoding="utf-8"))
    angles = plan["camera_poses"]["camera_v2_1_relation"]
    if len(angles) != len(JOINT_NAMES):
        raise ValueError("Locked Test-IID view pose must have six joints")
    return dict(zip(JOINT_NAMES, map(float, angles)))


def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def _multiply_quaternions(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


def verify_test_iid_optical_pose(output_world: Path) -> dict:
    """Check the generated sensor against Test-IID's locked optical TF."""
    plan = json.loads(TEST_IID_PLAN.read_text(encoding="utf-8"))
    expected = plan["camera_predictor_transform"]
    model = ET.parse(output_world).getroot().find("./world/model[@name='ur3_static_preview']")
    if model is None:
        raise ValueError("Generated preview UR3 missing")
    observed = []
    for sensor_name in ("d435i_aligned_rgbd", "d435i_evaluation_labels"):
        sensor = model.find(f".//sensor[@name='{sensor_name}']")
        if sensor is None:
            raise ValueError(f"Generated camera sensor missing: {sensor_name}")
        pose = [float(value) for value in sensor.findtext("pose", "").split()]
        if len(pose) != 6:
            raise ValueError(f"Generated sensor pose missing: {sensor_name}")
        observed.append(pose)
    if max(abs(a - b) for a, b in zip(observed[0], observed[1])) > 1e-6:
        raise ValueError("RGB-D and evaluator-only label sensors are not co-located")
    pose = observed[0]
    optical = _multiply_quaternions(
        _quaternion_from_rpy(*pose[3:]),
        _quaternion_from_rpy(-math.pi / 2, 0.0, -math.pi / 2),
    )
    reference = [float(value) for value in expected["orientation_xyzw"]]
    dot = abs(sum(a * b for a, b in zip(optical, reference)))
    orientation_error_deg = math.degrees(2 * math.acos(min(1.0, max(-1.0, dot))))
    position_error_m = math.dist(pose[:3], map(float, expected["position"]))
    if position_error_m > 0.001 or orientation_error_deg > 0.01:
        raise ValueError(f"Generated optical pose differs from Test-IID: {position_error_m}m, {orientation_error_deg}deg")
    return {"sensor_position_xyz": pose[:3], "optical_orientation_xyzw": optical,
            "position_error_m": position_error_m,
            "orientation_error_deg": orientation_error_deg,
            "matches_locked_test_iid_optical_tf": True}


def bake_view_pose(urdf: str, joints: dict[str, float]) -> str:
    """Freeze the repo's camera-view joint angles into URDF origins."""
    robot = ET.fromstring(urdf)
    for name, angle in joints.items():
        joint = robot.find(f"./joint[@name='{name}']")
        if joint is None:
            raise RuntimeError(f"Missing UR3 joint: {name}")
        axis = joint.find("axis")
        if axis is None or axis.get("xyz") != "0 0 1":
            raise RuntimeError(f"Unexpected axis for {name}")
        origin = joint.find("origin")
        if origin is None:
            raise RuntimeError(f"Missing origin for {name}")
        rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        ca, sa = math.cos(angle), math.sin(angle)
        # URDF rpy is Rz(yaw) Ry(pitch) Rx(roll); apply the joint's local Rz(q).
        original = (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
        posed = tuple((row[0] * ca + row[1] * sa,
                       -row[0] * sa + row[1] * ca, row[2]) for row in original)
        new_pitch = math.atan2(-posed[2][0], math.hypot(posed[0][0], posed[1][0]))
        new_roll = math.atan2(posed[2][1], posed[2][2])
        new_yaw = math.atan2(posed[1][0], posed[0][0])
        origin.set("rpy", f"{new_roll:.12g} {new_pitch:.12g} {new_yaw:.12g}")
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "dynamics", "safety_controller", "mimic"):
            element = joint.find(tag)
            if element is not None:
                joint.remove(element)
    return ET.tostring(robot, encoding="unicode")


def build(output_world: Path, demo_layout: bool = False, view_pose: str = "test_iid") -> None:
    urdf = bake_view_pose(subprocess.run(["xacro", str(PREVIEW_XACRO)], check=True,
                                        capture_output=True, text=True).stdout,
                          selected_view_pose(view_pose))
    gripper_meshes = ROOT / "ur3/susgrip_2f/susgrip_2f_description/meshes"
    urdf = urdf.replace("package://susgrip_2f_description/meshes/",
                        f"file://{gripper_meshes}/")
    with tempfile.TemporaryDirectory(prefix="ur3_static_preview_") as temporary:
        urdf_path = Path(temporary) / "ur3_preview.urdf"
        urdf_path.write_text(urdf, encoding="utf-8")
        sdf = subprocess.run(["ign", "sdf", "-p", str(urdf_path)], check=True,
                             capture_output=True, text=True).stdout
    model = ET.fromstring(sdf).find("model")
    if model is None:
        raise RuntimeError("UR3 conversion produced no SDF model")
    model.set("name", "ur3_static_preview")
    ET.SubElement(model, "static").text = "true"
    # URDF base/world origin matches the simulation's robot base at (0, 0, 0).
    ET.SubElement(model, "pose").text = "0 0 0 0 0 0"
    for plugin in list(model.findall("plugin")):
        model.remove(plugin)

    world_text = SOURCE_WORLD.read_text(encoding="utf-8")
    models_dir = ROOT / "ur3/ur_simulation_gz/models/ycb"
    world_text = world_text.replace("../models/ycb/", f"{models_dir}/")
    tree = ET.ElementTree(ET.fromstring(world_text))
    world = tree.getroot().find("world")
    if world is None:
        raise RuntimeError("Source Gazebo world is missing its world element")
    if demo_layout:
        apply_demo_layout(world)
    world.append(model)
    output_world.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_world, encoding="unicode", xml_declaration=True)
    subprocess.run(["ign", "sdf", "-k", str(output_world)], check=True,
                   capture_output=True, text=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-world", type=Path, required=True)
    parser.add_argument("--demo-layout", action="store_true",
                        help="YCB left, bin right, left-oblique camera; preview world only")
    parser.add_argument("--view-pose", choices=("test_iid", "closeup_legacy"),
                        default="test_iid", help="Frozen preview arm joints; default matches Test-IID")
    args = parser.parse_args()
    build(args.output_world.resolve(), demo_layout=args.demo_layout, view_pose=args.view_pose)
