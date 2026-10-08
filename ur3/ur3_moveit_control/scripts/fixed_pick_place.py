#!/usr/bin/env python3
"""Repeatable fixed or perception-driven pick-and-place benchmark for UR3."""

import copy
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Empty, Float64, String
from ur3_perception_interfaces.msg import ObjectObservation

from ur3_moveit_control.action import UR3Control


def top_down_quaternion(yaw: float):
    """Quaternion for Rz(yaw) * Rx(pi)."""
    return [math.cos(0.5 * yaw), math.sin(0.5 * yaw), 0.0, 0.0]


def generate_manipulation_pose_values(
    object_xyz,
    object_yaw: float,
    grasp_offset_xyz,
    approach_distance: float,
    lift_distance: float,
    bin_object_pose,
    place_offset_xyz,
    place_approach_distance: float,
    tool_yaw=None,
    place_tool_yaw=None,
):
    """Generate all manipulation poses without consulting an oracle pose."""
    grasp_tool_yaw = object_yaw if tool_yaw is None else float(tool_yaw)
    placement_tool_yaw = (
        float(bin_object_pose[3]) if place_tool_yaw is None else float(place_tool_yaw)
    )
    orientation = top_down_quaternion(grasp_tool_yaw)
    grasp = [object_xyz[i] + grasp_offset_xyz[i] for i in range(3)] + orientation
    pregrasp = list(grasp)
    pregrasp[2] += approach_distance
    lift = list(grasp)
    lift[2] += lift_distance

    place_orientation = top_down_quaternion(placement_tool_yaw)
    place = [bin_object_pose[i] + place_offset_xyz[i] for i in range(3)] + place_orientation
    preplace = list(place)
    preplace[2] += place_approach_distance
    return {
        "object_pose": list(object_xyz) + [0.0, 0.0, math.sin(object_yaw / 2.0), math.cos(object_yaw / 2.0)],
        "grasp_pose": grasp,
        "pregrasp_pose": pregrasp,
        "lift_pose": lift,
        "preplace_pose": preplace,
        "place_pose": place,
        "retreat_pose": list(preplace),
    }


def make_xy_aabb(center_xy, dimensions_xy):
    """Return an axis-aligned footprint [xmin, ymin, xmax, ymax]."""
    half_x = float(dimensions_xy[0]) / 2.0
    half_y = float(dimensions_xy[1]) / 2.0
    return [
        float(center_xy[0]) - half_x,
        float(center_xy[1]) - half_y,
        float(center_xy[0]) + half_x,
        float(center_xy[1]) + half_y,
    ]


def xy_aabbs_overlap(first, second, clearance: float = 0.0) -> bool:
    """True only when two footprints overlap (touching edges is allowed)."""
    return not (
        first[2] + clearance <= second[0]
        or second[2] + clearance <= first[0]
        or first[3] + clearance <= second[1]
        or second[3] + clearance <= first[1]
    )


def source_pose_overlaps_keepout(
    center_xy,
    dimensions_xy,
    keepout_center_xy,
    keepout_size_xy,
    clearance: float = 0.0,
) -> bool:
    """True when a source object would start in the physical tray keep-out."""
    source_aabb = make_xy_aabb(center_xy, dimensions_xy)
    keepout_aabb = make_xy_aabb(keepout_center_xy, keepout_size_xy)
    return xy_aabbs_overlap(source_aabb, keepout_aabb, clearance)


def validate_bin_place_center(
    center_xy,
    dimensions_xy,
    bin_center_xy,
    bin_interior_size_xy,
    occupied,
    clearance: float = 0.0,
):
    """Build a footprint if it is inside the tray and collision-free."""
    aabb = make_xy_aabb(center_xy, dimensions_xy)
    bin_aabb = make_xy_aabb(bin_center_xy, bin_interior_size_xy)
    if (
        aabb[0] < bin_aabb[0]
        or aabb[1] < bin_aabb[1]
        or aabb[2] > bin_aabb[2]
        or aabb[3] > bin_aabb[3]
    ):
        return None
    if any(
        xy_aabbs_overlap(aabb, item["aabb_xyxy"], clearance)
        for item in occupied
    ):
        return None
    return aabb


def select_non_overlapping_bin_slot(
    bin_center_xy,
    bin_interior_size_xy,
    slot_offsets_xy,
    dimensions_xy,
    occupied,
    clearance: float = 0.0,
):
    """Choose the first configured tray slot whose 2-D box does not overlap."""
    for slot_id, offset in enumerate(slot_offsets_xy):
        center = [
            float(bin_center_xy[0]) + float(offset[0]),
            float(bin_center_xy[1]) + float(offset[1]),
        ]
        aabb = validate_bin_place_center(
            center,
            dimensions_xy,
            bin_center_xy,
            bin_interior_size_xy,
            occupied,
            clearance,
        )
        if aabb is not None:
            return {"slot_id": slot_id, "center_xy": center, "aabb_xyxy": aabb}
    return None


class FixedPickPlace(Node):
    """Run fixed-pose regression or perception-driven episodes in one world."""

    def __init__(self) -> None:
        super().__init__("fixed_pick_place")

        self.declare_parameter("pose_frame", "base_link")
        self.declare_parameter("object_pose_source", "fixed")
        self.declare_parameter("arm_action", "/ur3_control")
        self.declare_parameter("gripper_action", "/gripper_controller/gripper_cmd")
        self.declare_parameter(
            "trajectory_action", "/joint_trajectory_controller/follow_joint_trajectory"
        )
        self.declare_parameter("server_timeout", 60.0)
        self.declare_parameter("motion_timeout", 120.0)
        self.declare_parameter("gripper_timeout", 15.0)
        self.declare_parameter("attachment_timeout", 5.0)
        self.declare_parameter("scene_update_timeout", 5.0)
        self.declare_parameter("detection_timeout", 30.0)
        self.declare_parameter("max_pose_age_sec", 1.0)
        self.declare_parameter("minimum_confidence", 0.5)
        self.declare_parameter("minimum_valid_frames", 5)
        self.declare_parameter("max_position_stddev_m", 0.003)
        self.declare_parameter("workspace_min", [-0.20, 0.18, 0.0])
        self.declare_parameter("workspace_max", [-0.08, 0.30, 0.12])
        self.declare_parameter("observation_topic", "/ur3_perception/object_observation")
        self.declare_parameter("stable_pose_topic", "/ur3_perception/object_pose_stable")
        self.declare_parameter("evaluation_error_topic", "/ur3_perception/position_error_m")
        self.declare_parameter("episode_event_topic", "/ur3_dataset/episode_event")
        self.declare_parameter("joint_state_topic", "/joint_states")
        # Fixed fallback poses remain unchanged and are used verbatim.
        industrial_orientation = [0.70710678, 0.70710678, 0.0, 0.0]
        self.declare_parameter("pregrasp_pose", [-0.13, 0.24, 0.20] + industrial_orientation)
        self.declare_parameter("grasp_pose", [-0.13, 0.24, 0.03] + industrial_orientation)
        self.declare_parameter("lift_pose", [-0.13, 0.24, 0.20] + industrial_orientation)
        self.declare_parameter("preplace_pose", [0.00, 0.38, 0.20] + industrial_orientation)
        self.declare_parameter("place_pose", [0.00, 0.38, 0.11] + industrial_orientation)
        self.declare_parameter("view_pose", [-0.0234, 0.2390, 0.25] + industrial_orientation)
        self.declare_parameter(
            "view_joint_pose",
            [1.1815, -1.8273, 1.3428, -1.0863, -1.5708, -1.9601],
        )
        self.declare_parameter(
            "place_joint_pose",
            [-0.3873415, -1.0436025, 0.6870591, -1.2142529, -1.5707963, 2.7542512],
        )
        self.declare_parameter("industrial_linear_transfers", True)

        # Perception pose generation. All distances refer to gripper_tcp.
        self.declare_parameter("approach_distance", 0.17)
        self.declare_parameter("lift_distance", 0.17)
        self.declare_parameter("place_approach_distance", 0.09)
        self.declare_parameter("grasp_offset_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("minimum_grasp_tcp_z", 0.03)
        self.declare_parameter("place_offset_xyz", [0.0, 0.0, 0.07])
        self.declare_parameter("bin_object_pose", [0.00, 0.38, 0.04, 0.0])
        self.declare_parameter("bin_interior_size_xy", [0.18, 0.23])
        self.declare_parameter(
            "bin_place_slots_xy",
            [-0.055, -0.075, 0.0, -0.075, 0.055, -0.075,
             -0.055, 0.0, 0.0, 0.0, 0.055, 0.0,
             -0.055, 0.075, 0.0, 0.075, 0.055, 0.075],
        )
        self.declare_parameter("bin_place_clearance_m", 0.0)
        self.declare_parameter("bin_occupancy_topic", "/ur3_manipulation/bin_occupancy")
        # Every source object must begin outside this physical 200 x 250 mm
        # tray footprint.  This is intentionally separate from bin placement:
        # it prevents a random-scene reset from silently preloading the tray.
        self.declare_parameter("source_keepout_center_xy", [0.0, 0.38])
        self.declare_parameter("source_keepout_size_xy", [0.20, 0.25])
        self.declare_parameter("source_keepout_clearance_m", 0.005)
        self.declare_parameter("source_keepout_object_size", 0.06)
        self.declare_parameter("use_object_yaw_for_grasp", False)
        self.declare_parameter("fixed_tool_yaw", math.pi / 2.0)

        self.declare_parameter("gripper_open_position", 0.065)
        self.declare_parameter("gripper_closed_position", 0.050)
        self.declare_parameter("gripper_max_effort", 40.0)
        self.declare_parameter("gripper_staged_motion", False)
        self.declare_parameter("gripper_close_stages", [0.060, 0.055, 0.050])
        self.declare_parameter("gripper_open_stages", [0.055, 0.060, 0.065])
        self.declare_parameter("gripper_stage_pause_sec", 0.12)
        self.declare_parameter("settle_time", 0.75)
        self.declare_parameter("perception_settle_time", 2.0)
        self.declare_parameter("return_home", True)
        self.declare_parameter("attach_topic", "/ur3_sim/attach_red_cube")
        self.declare_parameter("detach_topic", "/ur3_sim/detach_red_cube")
        self.declare_parameter("attachment_state_topic", "/ur3_sim/red_cube_attached")
        self.declare_parameter("object_id", "red_cube")
        self.declare_parameter("object_color", "")
        self.declare_parameter("tcp_link", "gripper_tcp")
        self.declare_parameter("object_size", 0.055)
        self.declare_parameter("episode_count", 3)
        self.declare_parameter("inter_episode_delay", 1.0)
        self.declare_parameter("stop_on_failure", False)
        self.declare_parameter("metrics_output", "")
        self.declare_parameter("reset_service", "/world/ur3_pick_place/set_pose")
        self.declare_parameter("initial_object_pose", [-0.13, 0.24, 0.03, 0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("distractor_object_name", "")
        self.declare_parameter(
            "distractor_object_pose", [-0.30, 0.12, 0.03, 0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("randomize_object", False)
        self.declare_parameter("random_seed", 7)
        # The random source region is below and left of the physical tray.
        # A 60 mm cube (the largest source) still clears the tray by >20 mm.
        self.declare_parameter("random_x_range", [-0.16, -0.11])
        self.declare_parameter("random_y_range", [0.14, 0.20])
        self.declare_parameter("random_yaw_range", [0.0, 0.0])
        self.declare_parameter("randomize_scene_objects", True)
        self.declare_parameter("shuffle_scene_objects", True)
        self.declare_parameter(
            "scene_object_names",
            ["red_cube", "green_cube", "blue_cube", "yellow_cube", "orange_cube", "purple_cube", "pink_cube"],
        )
        # Six collision-safe slots for non-selected cubes, all outside the
        # tray keep-out. Their ownership is shuffled per episode; the selected
        # cube gets the isolated random pick slot.
        self.declare_parameter(
            "scene_object_slots_xy",
            [-0.34, 0.10, -0.34, 0.40, -0.24, 0.46,
             -0.02, 0.10, 0.15, 0.14, 0.18, 0.28],
        )
        # These arrays are parallel to scene_object_names.  Keeping physical
        # support heights and XY footprints per model lets mixed-shape scenes
        # (for example spherical fruit plus a long banana) be reset without
        # sinking objects into the table or validating only their centres.
        self.declare_parameter("scene_object_z_values", [0.03] * 7)
        self.declare_parameter("scene_object_sizes_xy", [0.06, 0.06] * 7)
        self.declare_parameter("error_injection_axis", "none")
        self.declare_parameter("error_injection_values", [0.0])
        self.declare_parameter("place_recovery_attempts", 4)
        self.declare_parameter("place_recovery_offset_m", 0.015)
        self._pose_frame = str(self.get_parameter("pose_frame").value)
        self._pose_source = str(self.get_parameter("object_pose_source").value).lower()
        if self._pose_source not in ("fixed", "perception"):
            raise ValueError("object_pose_source must be 'fixed' or 'perception'")
        self._server_timeout = float(self.get_parameter("server_timeout").value)
        self._motion_timeout = float(self.get_parameter("motion_timeout").value)
        self._gripper_timeout = float(self.get_parameter("gripper_timeout").value)
        self._settle_time = float(self.get_parameter("settle_time").value)
        self._attachment_timeout = float(self.get_parameter("attachment_timeout").value)
        self._scene_timeout = float(self.get_parameter("scene_update_timeout").value)
        self._episode_count = int(self.get_parameter("episode_count").value)
        self._inter_episode_delay = float(self.get_parameter("inter_episode_delay").value)
        self._stop_on_failure = bool(self.get_parameter("stop_on_failure").value)
        if self._episode_count < 1:
            raise ValueError("episode_count must be at least 1")

        self._arm_client = ActionClient(
            self, UR3Control, str(self.get_parameter("arm_action").value)
        )
        self._gripper_client = ActionClient(
            self, GripperCommand, str(self.get_parameter("gripper_action").value)
        )
        # The wrapper action can be ready a little before MoveIt's execution
        # controller. Waiting for the controller removes that startup race.
        self._trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("trajectory_action").value),
        )
        self._active_object_model = str(self.get_parameter("object_id").value)
        configured_color = str(self.get_parameter("object_color").value).strip()
        self._active_color = configured_color or self._active_object_model.removesuffix("_cube")
        self._active_source_id = (
            f"cube_{self._active_color}_01"
            if self._active_object_model.endswith("_cube")
            else f"{self._active_object_model}_01"
        )
        self._attach_pub = self.create_publisher(
            Empty, str(self.get_parameter("attach_topic").value), 10
        )
        self._detach_pub = self.create_publisher(
            Empty, str(self.get_parameter("detach_topic").value), 10
        )
        self.create_subscription(
            String,
            str(self.get_parameter("attachment_state_topic").value),
            self._attachment_callback,
            10,
        )
        self._attachment_state = None
        self.create_subscription(
            ObjectObservation,
            str(self.get_parameter("observation_topic").value),
            self._observation_callback,
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._joint_state_callback,
            10,
        )
        # This subscription is evaluation-only and never influences commands.
        self.create_subscription(
            Float64,
            str(self.get_parameter("evaluation_error_topic").value),
            self._evaluation_error_callback,
            10,
        )
        self._planning_scene = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )
        self._get_planning_scene = self.create_client(
            GetPlanningScene, "/get_planning_scene"
        )
        self._reset_pose = self.create_client(
            SetEntityPose, str(self.get_parameter("reset_service").value)
        )
        event_qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._episode_event_pub = self.create_publisher(
            String, str(self.get_parameter("episode_event_topic").value), event_qos
        )
        self._bin_occupancy_pub = self.create_publisher(
            String, str(self.get_parameter("bin_occupancy_topic").value), event_qos
        )

        self._latest_observation = None
        self._latest_joint_state = None
        self._latest_joint_state_wall_time = None
        self._latest_observation_wall_time = None
        self._latest_evaluation_error = None
        self._frozen_observation = None
        self._frozen_evaluation_error = None
        self._dynamic_poses = {}
        self._override_poses = {}
        self._object_dimensions = [float(self.get_parameter("object_size").value)] * 3
        self._episode_reset_pose = list(self.get_parameter("initial_object_pose").value)
        self._random = random.Random(int(self.get_parameter("random_seed").value))
        self._injection_axis = str(self.get_parameter("error_injection_axis").value).lower()
        self._injection_value = 0.0
        self._last_error = ""
        self._last_step_detail = {}
        self._episode_steps = []
        self._moveit_attached = False
        self._moveit_world_object = False
        self._milestones = {}
        self._metrics_path = self._resolve_metrics_path()
        self._current_episode = 0
        self._bin_occupancy = {}
        self._pending_place_allocation = None

        self.get_logger().info(
            f"Configured {self._episode_count} UR3 episode(s), "
            f"object_pose_source={self._pose_source}, frame={self._pose_frame}"
        )

    def _attachment_callback(self, msg: String) -> None:
        self._attachment_state = msg.data.lower()

    def _publish_bin_occupancy(self) -> None:
        entries = [self._bin_occupancy[key] for key in sorted(self._bin_occupancy)]
        self._bin_occupancy_pub.publish(String(data=json.dumps({
            "frame_id": self._pose_frame,
            "bin_id": "bin_orange_01",
            "count": len(entries),
            "objects": entries,
        }, ensure_ascii=False)))

    def _allocate_bin_place(self):
        bin_pose = [float(value) for value in self.get_parameter("bin_object_pose").value]
        flat_slots = [
            float(value) for value in self.get_parameter("bin_place_slots_xy").value
        ]
        if len(flat_slots) % 2:
            raise ValueError("bin_place_slots_xy must contain x,y pairs")
        slots = [flat_slots[index:index + 2] for index in range(0, len(flat_slots), 2)]
        # Cubes may have a slightly noisy RGB-D x/y dimension estimate. A
        # square conservative footprint keeps yaw/noise from creating overlap.
        footprint_size = max(float(self._object_dimensions[0]), float(self._object_dimensions[1]))
        allocation = select_non_overlapping_bin_slot(
            bin_pose[:2],
            [float(value) for value in self.get_parameter("bin_interior_size_xy").value],
            slots,
            [footprint_size, footprint_size],
            list(self._bin_occupancy.values()),
            float(self.get_parameter("bin_place_clearance_m").value),
        )
        if allocation is None:
            return None
        allocation.update({
            "object_id": self._active_source_id,
            "model": self._active_object_model,
            "color": self._active_color,
            "dimensions_xy": [footprint_size, footprint_size],
        })
        return allocation

    def _object_id(self) -> str:
        return self._active_object_model

    def _observation_callback(self, msg: ObjectObservation) -> None:
        self._latest_observation = msg
        self._latest_observation_wall_time = time.monotonic()

    def _joint_state_callback(self, msg: JointState) -> None:
        self._latest_joint_state = msg
        self._latest_joint_state_wall_time = time.monotonic()

    def _evaluation_error_callback(self, msg: Float64) -> None:
        if math.isfinite(msg.data):
            self._latest_evaluation_error = float(msg.data)

    def _resolve_metrics_path(self) -> Path:
        configured = str(self.get_parameter("metrics_output").value).strip()
        if configured:
            path = Path(configured).expanduser()
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path.home() / ".ros" / "ur3_baseline_metrics" / f"run_{stamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _error(self, message: str) -> bool:
        self._last_error = message
        self.get_logger().error(message)
        return False

    def _publish_episode_event(self, event: str, **payload) -> None:
        """Publish structured lifecycle data for rosbag/dataset recording."""
        stamp = self.get_clock().now().to_msg()
        message = String()
        message.data = json.dumps(
            {
                "schema_version": 1,
                "event": event,
                "episode": self._current_episode,
                "ros_stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec},
                "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                **payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._episode_event_pub.publish(message)

    def _execute_step(self, label: str, callback) -> bool:
        self._last_error = ""
        self._last_step_detail = {}
        self._publish_episode_event("step_start", state=label)
        started = time.monotonic()
        try:
            success = bool(callback())
        except Exception as exc:
            success = self._error(f"{label}: {type(exc).__name__}: {exc}")
        total_time = time.monotonic() - started
        step_result = {
            "state": label,
            "success": success,
            "total_time_sec": round(total_time, 6),
            "planning_time_sec": round(
                float(self._last_step_detail.get("planning_time_sec", 0.0)), 6
            ),
            "execution_time_sec": round(
                float(self._last_step_detail.get("execution_time_sec", total_time)), 6
            ),
            "failure_reason": "" if success else (self._last_error or "step returned false"),
        }
        self._episode_steps.append(step_result)
        self._publish_episode_event("step_end", **step_result)
        return success

    def _wait_for_servers(self) -> bool:
        self.get_logger().info("Waiting for arm, gripper, Gazebo and Planning Scene services...")
        readiness = {
            "arm action": self._arm_client.wait_for_server(timeout_sec=self._server_timeout),
            "gripper action": self._gripper_client.wait_for_server(timeout_sec=self._server_timeout),
            "trajectory controller action": self._trajectory_client.wait_for_server(
                timeout_sec=self._server_timeout
            ),
            "Gazebo reset": self._reset_pose.wait_for_service(timeout_sec=self._server_timeout),
            "apply planning scene": self._planning_scene.wait_for_service(timeout_sec=self._server_timeout),
            "get planning scene": self._get_planning_scene.wait_for_service(timeout_sec=self._server_timeout),
        }
        for name, ready in readiness.items():
            if not ready:
                self.get_logger().error(f"{name} did not become available")
        if not all(readiness.values()):
            return False
        # RGB-D rendering can make ros2_control startup slower. Use the
        # configured server timeout; motion remains forbidden until a recent,
        # non-zero timestamped state is actually observed.
        state_deadline = time.monotonic() + self._server_timeout
        while time.monotonic() < state_deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            state = self._latest_joint_state
            if state is None or len(state.position) < 6:
                continue
            stamp_is_nonzero = bool(state.header.stamp.sec or state.header.stamp.nanosec)
            state_is_recent = (
                self._latest_joint_state_wall_time is not None
                and time.monotonic() - self._latest_joint_state_wall_time < 0.5
            )
            if stamp_is_nonzero and state_is_recent:
                self.get_logger().info(
                    "JOINT_STATE_READY: controller publishes current timestamped state"
                )
                return True
        return self._error(
            "JOINT_STATE_NOT_READY: no recent timestamped state from active controller"
        )

    @staticmethod
    def _to_pose_stamped(values, frame: str, stamp) -> PoseStamped:
        if len(values) != 7:
            raise ValueError("pose must contain [x,y,z,qx,qy,qz,qw]")
        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.header.stamp = stamp
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = values[0:3]
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = values[3:7]
        return pose

    def _pose(self, pose_name: str) -> PoseStamped:
        if pose_name in self._override_poses:
            values = self._override_poses[pose_name]
        elif self._pose_source != "fixed" and pose_name in self._dynamic_poses:
            values = self._dynamic_poses[pose_name]
        else:
            values = list(self.get_parameter(pose_name).value)
        return self._to_pose_stamped(values, self._pose_frame, self.get_clock().now().to_msg())

    def _send_arm_goal(self, label: str, command: int, pose_name: str = "") -> bool:
        goal = UR3Control.Goal()
        goal.command_type = command
        if pose_name:
            goal.pose_goal = self._pose(pose_name)
        self.get_logger().info(f"STATE: {label}")
        send_future = self._arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=self._motion_timeout)
        if not send_future.done() or send_future.result() is None:
            return self._error(f"{label}: timed out while sending arm goal")
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            return self._error(f"{label}: arm goal rejected")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self._motion_timeout)
        if not result_future.done() or result_future.result() is None:
            goal_handle.cancel_goal_async()
            return self._error(f"{label}: arm execution timed out")
        wrapped_result = result_future.result()
        self._last_step_detail = {
            "planning_time_sec": wrapped_result.result.planning_time_sec,
            "execution_time_sec": wrapped_result.result.execution_time_sec,
        }
        success = (
            wrapped_result.status == GoalStatus.STATUS_SUCCEEDED
            and wrapped_result.result.success
        )
        if not success:
            return self._error(
                f"{label}: {wrapped_result.result.message} (status={wrapped_result.status})"
            )
        return True

    def _send_joint_goal(self, label: str, parameter_name: str) -> bool:
        values = [float(value) for value in self.get_parameter(parameter_name).value]
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            return self._error(f"{label}: {parameter_name} must contain six finite joints")
        goal = UR3Control.Goal()
        goal.command_type = UR3Control.Goal.MOVE_JOINT
        goal.joint_goal.position = values
        self.get_logger().info(
            f"STATE: {label} fixed industrial joint posture "
            f"({', '.join(f'{value:.3f}' for value in values)})"
        )
        send_future = self._arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=self._motion_timeout)
        if not send_future.done() or send_future.result() is None:
            return self._error(f"{label}: timed out while sending arm goal")
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            return self._error(f"{label}: arm goal rejected")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self._motion_timeout)
        if not result_future.done() or result_future.result() is None:
            goal_handle.cancel_goal_async()
            return self._error(f"{label}: arm execution timed out")
        wrapped_result = result_future.result()
        self._last_step_detail = {
            "planning_time_sec": wrapped_result.result.planning_time_sec,
            "execution_time_sec": wrapped_result.result.execution_time_sec,
        }
        if not (
            wrapped_result.status == GoalStatus.STATUS_SUCCEEDED
            and wrapped_result.result.success
        ):
            return self._error(
                f"{label}: {wrapped_result.result.message} (status={wrapped_result.status})"
            )
        return True

    def _send_gripper_goal(
        self, label: str, position: float, settle_after: bool = True
    ) -> bool:
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = float(self.get_parameter("gripper_max_effort").value)
        self.get_logger().info(f"STATE: {label} (position={position:.3f} m)")
        send_future = self._gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=self._gripper_timeout)
        if not send_future.done() or send_future.result() is None:
            return self._error(f"{label}: timed out while sending gripper goal")
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            return self._error(f"{label}: gripper goal rejected")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self._gripper_timeout)
        if not result_future.done() or result_future.result() is None:
            goal_handle.cancel_goal_async()
            return self._error(f"{label}: gripper execution timed out")
        if result_future.result().status != GoalStatus.STATUS_SUCCEEDED:
            return self._error(
                f"{label}: gripper failed with status={result_future.result().status}"
            )
        if settle_after:
            time.sleep(self._settle_time)
        return True

    def _command_gripper(self, label: str, opening: bool) -> bool:
        """Move the fingers through visible intermediate positions.

        The default single-goal behavior remains available for regression
        profiles.  Interactive object demos can enable staged motion so the
        simulated fingers close around and release an object progressively
        instead of appearing to snap between endpoints.
        """
        target_parameter = (
            "gripper_open_position" if opening else "gripper_closed_position"
        )
        target = float(self.get_parameter(target_parameter).value)
        if not bool(self.get_parameter("gripper_staged_motion").value):
            return self._send_gripper_goal(label, target)

        stage_parameter = (
            "gripper_open_stages" if opening else "gripper_close_stages"
        )
        stages = [float(value) for value in self.get_parameter(stage_parameter).value]
        if not stages or not all(math.isfinite(value) for value in stages):
            return self._error(f"{label}: {stage_parameter} must contain finite positions")
        if not math.isclose(stages[-1], target, abs_tol=1e-6):
            stages.append(target)
        pause = max(
            0.0, float(self.get_parameter("gripper_stage_pause_sec").value)
        )
        self.get_logger().info(
            f"GRIPPER_STAGED_MOTION: {label}, "
            f"positions={[round(value, 4) for value in stages]}"
        )
        for index, position in enumerate(stages, start=1):
            if not self._send_gripper_goal(
                f"{label}_STAGE_{index}/{len(stages)}",
                position,
                settle_after=False,
            ):
                return False
            if index < len(stages) and pause > 0.0:
                time.sleep(pause)
        time.sleep(self._settle_time)
        return True

    def _set_gazebo_attachment(self, attached: bool) -> bool:
        label = "ATTACH_OBJECT" if attached else "DETACH_OBJECT"
        expected_state = "attached" if attached else "detached"
        publisher = self._attach_pub if attached else self._detach_pub
        self.get_logger().info(f"STATE: {label} (Gazebo)")
        deadline = time.monotonic() + self._attachment_timeout
        while time.monotonic() < deadline:
            publisher.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._attachment_state == expected_state:
                return True
            time.sleep(0.1)
        return self._error(
            f"{label}: detachable-joint state did not become '{expected_state}'"
        )

    def _prepare_episode(self, episode_index: int) -> None:
        self._latest_observation = None
        self._latest_observation_wall_time = None
        self._latest_evaluation_error = None
        self._frozen_observation = None
        self._frozen_evaluation_error = None
        self._dynamic_poses = {}
        self._override_poses = {}
        self._pending_place_allocation = None
        self._object_dimensions = [float(self.get_parameter("object_size").value)] * 3
        self._episode_reset_pose = list(self.get_parameter("initial_object_pose").value)
        if bool(self.get_parameter("randomize_object").value):
            x_range = list(self.get_parameter("random_x_range").value)
            y_range = list(self.get_parameter("random_y_range").value)
            yaw_range = list(self.get_parameter("random_yaw_range").value)
            if any(len(values) != 2 for values in (x_range, y_range, yaw_range)):
                raise ValueError("random ranges must each contain [min,max]")
            self._episode_reset_pose[0] = self._random.uniform(*x_range)
            self._episode_reset_pose[1] = self._random.uniform(*y_range)
            yaw = self._random.uniform(*yaw_range)
            self._episode_reset_pose[3:7] = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
        self._scene_reset_poses = {}
        self._scene_reset_dimensions = {}
        if bool(self.get_parameter("randomize_scene_objects").value):
            object_names = list(self.get_parameter("scene_object_names").value)
            z_values = [
                float(value)
                for value in self.get_parameter("scene_object_z_values").value
            ]
            flat_sizes = [
                float(value)
                for value in self.get_parameter("scene_object_sizes_xy").value
            ]
            if len(z_values) != len(object_names):
                self.get_logger().warning(
                    "scene_object_z_values length does not match "
                    "scene_object_names; falling back to z=0.03"
                )
                z_values = [0.03] * len(object_names)
            if len(flat_sizes) != 2 * len(object_names):
                self.get_logger().warning(
                    "scene_object_sizes_xy length does not match "
                    "scene_object_names; falling back to source object size"
                )
                fallback_size = float(
                    self.get_parameter("source_keepout_object_size").value
                )
                flat_sizes = [fallback_size, fallback_size] * len(object_names)
            object_z = dict(zip(object_names, z_values))
            object_sizes = {
                name: flat_sizes[2 * index:2 * index + 2]
                for index, name in enumerate(object_names)
            }
            selected_name = self._object_id()
            other_names = [name for name in object_names if name != selected_name]
            flat_slots = [float(value) for value in self.get_parameter("scene_object_slots_xy").value]
            if len(flat_slots) % 2:
                raise ValueError("scene_object_slots_xy must contain x,y pairs")
            slots = [flat_slots[index:index + 2] for index in range(0, len(flat_slots), 2)]
            if len(slots) < len(other_names):
                raise ValueError("scene_object_slots_xy does not provide enough safe slots")
            if bool(self.get_parameter("shuffle_scene_objects").value):
                self._random.shuffle(slots)
                self._random.shuffle(other_names)
            else:
                self.get_logger().info(
                    "SCENE_LAYOUT_ASSIGNMENT: fixed name-to-slot order"
                )
            for name, (x_value, y_value) in zip(other_names, slots):
                self._scene_reset_poses[name] = [
                    x_value, y_value, object_z[name], 0.0, 0.0, 0.0, 1.0
                ]
                self._scene_reset_dimensions[name] = object_sizes[name]
        self._validate_source_reset_poses()
        injection_values = [float(value) for value in self.get_parameter("error_injection_values").value]
        if not injection_values:
            injection_values = [0.0]
        self._injection_value = injection_values[(episode_index - 1) % len(injection_values)]
        if self._injection_axis not in ("none", "x", "y", "z", "yaw"):
            raise ValueError("error_injection_axis must be none, x, y, z, or yaw")
        self._milestones = {
            "perception_success": self._pose_source == "fixed",
            "planning_success": False,
            "grasp_success": False,
            "place_success": False,
            "scene_update_success": self._pose_source == "fixed",
            "attach_success": False,
            "detach_success": False,
        }

    def _validate_source_reset_poses(self) -> None:
        """Fail before motion if any randomized source occupies the tray."""
        center = [float(value) for value in self.get_parameter(
            "source_keepout_center_xy"
        ).value]
        size = [float(value) for value in self.get_parameter(
            "source_keepout_size_xy"
        ).value]
        if len(center) != 2 or len(size) != 2 or any(value <= 0.0 for value in size):
            raise ValueError(
                "source_keepout_center_xy and source_keepout_size_xy must be x,y pairs"
            )
        object_size = float(self.get_parameter("source_keepout_object_size").value)
        if not math.isfinite(object_size) or object_size <= 0.0:
            raise ValueError("source_keepout_object_size must be a positive finite value")
        clearance = float(self.get_parameter("source_keepout_clearance_m").value)
        selected_name = self._object_id()
        sources = {selected_name: self._episode_reset_pose, **self._scene_reset_poses}
        dimensions = {
            selected_name: [object_size, object_size],
            **self._scene_reset_dimensions,
        }
        violations = [
            name
            for name, pose in sources.items()
            if source_pose_overlaps_keepout(
                pose[:2], dimensions[name], center, size, clearance
            )
        ]
        if violations:
            raise ValueError(
                "source object(s) would start in target-bin keep-out: "
                + ", ".join(sorted(violations))
            )

    def _set_model_pose(self, model_name: str, values) -> bool:
        request = SetEntityPose.Request()
        request.entity.name = model_name
        request.entity.type = Entity.MODEL
        request.pose.position.x, request.pose.position.y, request.pose.position.z = values[0:3]
        (
            request.pose.orientation.x,
            request.pose.orientation.y,
            request.pose.orientation.z,
            request.pose.orientation.w,
        ) = values[3:7]
        future = self._reset_pose.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._server_timeout)
        return bool(
            future.done() and future.result() is not None and future.result().success
        )

    def _clear_active_planning_scene(self) -> bool:
        if (
            not self._moveit_attached
            and self._moveit_world_object
            and self._active_source_id in self._bin_occupancy
        ):
            # Keep successfully placed boxes in MoveIt so later place motions
            # see the same occupied tray cells as the AABB slot allocator.
            self.get_logger().info(
                f"PLANNING_SCENE_KEEP_PLACED: {self._active_source_id}"
            )
            self._moveit_world_object = False
            return True
        if not self._moveit_attached and not self._moveit_world_object:
            return True
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        if self._moveit_attached:
            remove_attached = AttachedCollisionObject()
            remove_attached.link_name = str(self.get_parameter("tcp_link").value)
            remove_attached.object.id = self._object_id()
            remove_attached.object.operation = CollisionObject.REMOVE
            scene.robot_state.attached_collision_objects = [remove_attached]
        if self._moveit_world_object:
            remove_world = CollisionObject()
            remove_world.header.frame_id = self._pose_frame
            remove_world.id = self._object_id()
            remove_world.operation = CollisionObject.REMOVE
            scene.world.collision_objects = [remove_world]
        if not self._apply_planning_scene(scene, "INTERACTIVE_SCENE_CLEANUP"):
            return False
        self._moveit_attached = False
        self._moveit_world_object = False
        return True

    def _reset_object_for_episode(self) -> bool:
        self.get_logger().info(
            "STATE: RESET_OBJECT at "
            f"({self._episode_reset_pose[0]:.3f}, {self._episode_reset_pose[1]:.3f})"
        )
        # DetachableJoint starts attached in this simulation. Do not overwrite
        # the state locally: wait for Gazebo to confirm a real detach before
        # resetting the cube pose, otherwise the cube follows the wrist during
        # MOVE_VIEW/MOVE_PREGRASP and can block the vertical approach.
        # Preserve a previously confirmed detached state between episodes.
        # Ignition's DetachableJoint reports "Already detached" for repeated
        # commands but does not publish another state transition, so clearing
        # this value here made episode 2 time out despite correct physics.
        if self._attachment_state != "detached":
            detach_deadline = time.monotonic() + self._attachment_timeout
            while time.monotonic() < detach_deadline:
                self._detach_pub.publish(Empty())
                rclpy.spin_once(self, timeout_sec=0.1)
                if self._attachment_state == "detached":
                    break
                time.sleep(0.1)
        else:
            self.get_logger().info("RESET_OBJECT: detached state already confirmed by Gazebo")
        if self._attachment_state != "detached":
            return self._error("RESET_OBJECT: Gazebo did not confirm detached state")

        values = self._episode_reset_pose
        if len(values) != 7:
            return self._error("initial_object_pose must contain [x,y,z,qx,qy,qz,qw]")
        request = SetEntityPose.Request()
        request.entity.name = self._object_id()
        request.entity.type = Entity.MODEL
        request.pose.position.x, request.pose.position.y, request.pose.position.z = values[0:3]
        (
            request.pose.orientation.x,
            request.pose.orientation.y,
            request.pose.orientation.z,
            request.pose.orientation.w,
        ) = values[3:7]
        future = self._reset_pose.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._server_timeout)
        if not future.done() or future.result() is None or not future.result().success:
            return self._error("RESET_OBJECT: Gazebo SetEntityPose failed")
        for scene_name, scene_values in self._scene_reset_poses.items():
            scene_request = SetEntityPose.Request()
            scene_request.entity.name = scene_name
            scene_request.entity.type = Entity.MODEL
            (
                scene_request.pose.position.x,
                scene_request.pose.position.y,
                scene_request.pose.position.z,
            ) = scene_values[0:3]
            (
                scene_request.pose.orientation.x,
                scene_request.pose.orientation.y,
                scene_request.pose.orientation.z,
                scene_request.pose.orientation.w,
            ) = scene_values[3:7]
            scene_future = self._reset_pose.call_async(scene_request)
            rclpy.spin_until_future_complete(self, scene_future, timeout_sec=self._server_timeout)
            if (
                not scene_future.done()
                or scene_future.result() is None
                or not scene_future.result().success
            ):
                return self._error(f"RESET_OBJECT: failed to randomize {scene_name}")
        if self._scene_reset_poses:
            layout = ", ".join(
                f"{name}=({pose[0]:.2f},{pose[1]:.2f})"
                for name, pose in sorted(self._scene_reset_poses.items())
            )
            self.get_logger().info(f"SCENE_RANDOMIZED_LEFT_WORKSPACE: {layout}")
        distractor_name = str(self.get_parameter("distractor_object_name").value).strip()
        if distractor_name and not self._scene_reset_poses:
            distractor_values = list(self.get_parameter("distractor_object_pose").value)
            if len(distractor_values) != 7:
                return self._error("distractor_object_pose must contain 7 values")
            distractor_request = SetEntityPose.Request()
            distractor_request.entity.name = distractor_name
            distractor_request.entity.type = Entity.MODEL
            (
                distractor_request.pose.position.x,
                distractor_request.pose.position.y,
                distractor_request.pose.position.z,
            ) = distractor_values[0:3]
            (
                distractor_request.pose.orientation.x,
                distractor_request.pose.orientation.y,
                distractor_request.pose.orientation.z,
                distractor_request.pose.orientation.w,
            ) = distractor_values[3:7]
            distractor_future = self._reset_pose.call_async(distractor_request)
            rclpy.spin_until_future_complete(
                self, distractor_future, timeout_sec=self._server_timeout
            )
            if (
                not distractor_future.done()
                or distractor_future.result() is None
                or not distractor_future.result().success
            ):
                return self._error(
                    f"RESET_OBJECT: failed to move distractor {distractor_name}"
                )
            self.get_logger().info(
                f"DISTRACTOR_RELOCATED: {distractor_name} -> "
                f"({distractor_values[0]:.3f},{distractor_values[1]:.3f})"
            )
        if self._moveit_attached or self._moveit_world_object:
            scene = PlanningScene()
            scene.is_diff = True
            scene.robot_state.is_diff = True
            if self._moveit_attached:
                remove_attached = AttachedCollisionObject()
                remove_attached.link_name = str(self.get_parameter("tcp_link").value)
                remove_attached.object.id = self._object_id()
                remove_attached.object.operation = CollisionObject.REMOVE
                scene.robot_state.attached_collision_objects = [remove_attached]
            if self._moveit_world_object:
                remove_world = CollisionObject()
                remove_world.header.frame_id = self._pose_frame
                remove_world.id = self._object_id()
                remove_world.operation = CollisionObject.REMOVE
                scene.world.collision_objects = [remove_world]
            if not self._apply_planning_scene(scene, "RESET_OBJECT"):
                return False
            if not self._wait_for_scene_state(world_present=False, attached=False):
                return False
            self._moveit_attached = False
            self._moveit_world_object = False
        time.sleep(self._settle_time)
        return True

    def _observation_age(self, observation: ObjectObservation) -> float:
        stamp = Time.from_msg(observation.pose.header.stamp)
        return max(0.0, (self.get_clock().now() - stamp).nanoseconds / 1e9)

    def _acquire_and_freeze_observation(self) -> bool:
        self.get_logger().info("STATE: WAIT_STABLE_POSE")
        settle_deadline = time.monotonic() + float(
            self.get_parameter("perception_settle_time").value
        )
        while time.monotonic() < settle_deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        # Discard any pose accumulated while the eye-in-hand camera was moving.
        self._latest_observation = None
        deadline = time.monotonic() + float(self.get_parameter("detection_timeout").value)
        rejection = "no observation received"
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            observation = self._latest_observation
            if observation is None:
                continue
            if observation.status != "STABLE":
                rejection = f"status={observation.status}"
                continue
            age = self._observation_age(observation)
            if age > float(self.get_parameter("max_pose_age_sec").value):
                rejection = f"stale pose age={age:.3f}s"
                continue
            if observation.valid_frames < int(self.get_parameter("minimum_valid_frames").value):
                rejection = f"only {observation.valid_frames} valid frames"
                continue
            if observation.position_stddev_m > float(
                self.get_parameter("max_position_stddev_m").value
            ):
                rejection = f"position stddev={observation.position_stddev_m:.4f}m"
                continue
            if observation.confidence < float(self.get_parameter("minimum_confidence").value):
                rejection = f"confidence={observation.confidence:.3f}"
                continue
            xyz = [
                observation.pose.pose.position.x,
                observation.pose.pose.position.y,
                observation.pose.pose.position.z,
            ]
            if not all(math.isfinite(value) for value in xyz):
                rejection = "invalid/non-finite depth-derived pose"
                continue
            workspace_min = list(self.get_parameter("workspace_min").value)
            workspace_max = list(self.get_parameter("workspace_max").value)
            if any(xyz[i] < workspace_min[i] or xyz[i] > workspace_max[i] for i in range(3)):
                rejection = f"pose outside workspace: {xyz}"
                continue
            self._frozen_observation = copy.deepcopy(observation)
            self._frozen_evaluation_error = self._latest_evaluation_error
            self._milestones["perception_success"] = True
            self.get_logger().info(
                "POSE_FROZEN: "
                f"xyz=({xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f}), "
                f"frames={observation.valid_frames}, confidence={observation.confidence:.3f}"
            )
            return self._generate_dynamic_poses()
        return self._error(f"WAIT_STABLE_POSE: no valid detection ({rejection})")

    def _generate_dynamic_poses(self) -> bool:
        observation = self._frozen_observation
        object_xyz = [
            observation.pose.pose.position.x,
            observation.pose.pose.position.y,
            observation.pose.pose.position.z,
        ]
        object_yaw = observation.yaw if observation.yaw_valid else 0.0
        if bool(self.get_parameter("use_object_yaw_for_grasp").value):
            tool_yaw = object_yaw
        else:
            tool_yaw = float(self.get_parameter("fixed_tool_yaw").value)
        if self._injection_axis in ("x", "y", "z"):
            object_xyz[("x", "y", "z").index(self._injection_axis)] += self._injection_value
        elif self._injection_axis == "yaw":
            tool_yaw += self._injection_value
        self._object_dimensions = [
            observation.dimensions.x,
            observation.dimensions.y,
            observation.dimensions.z,
        ]
        if not all(self._min_dimension_valid(value) for value in self._object_dimensions):
            return self._error(f"GENERATE_POSES: invalid dimensions {self._object_dimensions}")
        self._pending_place_allocation = self._allocate_bin_place()
        if self._pending_place_allocation is None:
            return self._error("GENERATE_POSES: orange bin has no non-overlapping free slot")
        selected_bin_pose = list(self.get_parameter("bin_object_pose").value)
        selected_bin_pose[0:2] = self._pending_place_allocation["center_xy"]
        grasp_offset_xyz = list(self.get_parameter("grasp_offset_xyz").value)
        minimum_grasp_tcp_z = float(
            self.get_parameter("minimum_grasp_tcp_z").value
        )
        requested_grasp_z = object_xyz[2] + grasp_offset_xyz[2]
        if requested_grasp_z < minimum_grasp_tcp_z:
            clearance = minimum_grasp_tcp_z - requested_grasp_z
            grasp_offset_xyz[2] += clearance
            self.get_logger().info(
                "GRASP_TABLE_CLEARANCE: raised TCP %.1f mm to safe z=%.3f m"
                % (clearance * 1000.0, minimum_grasp_tcp_z)
            )
        self._dynamic_poses = generate_manipulation_pose_values(
            object_xyz=object_xyz,
            object_yaw=object_yaw,
            grasp_offset_xyz=grasp_offset_xyz,
            approach_distance=float(self.get_parameter("approach_distance").value),
            lift_distance=float(self.get_parameter("lift_distance").value),
            bin_object_pose=selected_bin_pose,
            place_offset_xyz=list(self.get_parameter("place_offset_xyz").value),
            place_approach_distance=float(
                self.get_parameter("place_approach_distance").value
            ),
            tool_yaw=tool_yaw,
            # Keep the same top-down wrist orientation during transfer and
            # placement. This mirrors a simple industrial pick/place cycle.
            place_tool_yaw=tool_yaw,
        )
        self.get_logger().info(
            "POSES_GENERATED: fixed top-down orientation for grasp/transfer/place; "
            "object, grasp, pregrasp, lift, preplace, place, retreat; "
            f"bin_slot={self._pending_place_allocation['slot_id']} "
            f"center=({selected_bin_pose[0]:.3f},{selected_bin_pose[1]:.3f}) "
            f"aabb={self._pending_place_allocation['aabb_xyxy']}"
        )
        return True

    @staticmethod
    def _min_dimension_valid(value: float) -> bool:
        return math.isfinite(value) and 0.005 <= value <= 0.25

    def _cube_primitive(self) -> SolidPrimitive:
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(self._object_dimensions)
        return primitive

    def _apply_planning_scene(self, scene: PlanningScene, label: str) -> bool:
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._planning_scene.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._server_timeout)
        if not future.done() or future.result() is None or not future.result().success:
            return self._error(f"{label}: failed to update MoveIt Planning Scene")
        return True

    def _query_scene_membership(self):
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        )
        future = self._get_planning_scene.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if not future.done() or future.result() is None:
            return None
        object_id = self._object_id()
        scene = future.result().scene
        world = any(obj.id == object_id for obj in scene.world.collision_objects)
        attached = any(
            item.object.id == object_id
            for item in scene.robot_state.attached_collision_objects
        )
        return world, attached

    def _wait_for_scene_state(self, world_present=None, attached=None) -> bool:
        deadline = time.monotonic() + self._scene_timeout
        while time.monotonic() < deadline:
            state = self._query_scene_membership()
            if state is not None:
                world_ok = world_present is None or state[0] == world_present
                attached_ok = attached is None or state[1] == attached
                if world_ok and attached_ok:
                    return True
            time.sleep(0.1)
        return self._error(
            f"PLANNING_SCENE_TIMEOUT: world={world_present}, attached={attached}"
        )

    def _update_observed_object(self) -> bool:
        observation = self._frozen_observation
        world_object = CollisionObject()
        world_object.header = observation.pose.header
        world_object.id = self._object_id()
        world_object.operation = CollisionObject.ADD
        world_object.primitives = [self._cube_primitive()]
        world_object.primitive_poses = [copy.deepcopy(observation.pose.pose)]
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = [world_object]
        self.get_logger().info("STATE: UPDATE_OBSERVED_OBJECT (MoveIt)")
        if not self._apply_planning_scene(scene, "UPDATE_OBSERVED_OBJECT"):
            return False
        if not self._wait_for_scene_state(world_present=True, attached=False):
            return False
        self._moveit_world_object = True
        self._milestones["scene_update_success"] = True
        self.get_logger().info(
            "PLANNING_SCENE_CONFIRMED: observed "
            f"{self._object_id()} present"
        )
        return True

    def _remove_world_object_for_approach(self) -> bool:
        remove = CollisionObject()
        remove.header.frame_id = self._pose_frame
        remove.id = self._object_id()
        remove.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = [remove]
        if not self._apply_planning_scene(scene, "CLEAR_APPROACH_OBJECT"):
            return False
        if not self._wait_for_scene_state(world_present=False, attached=False):
            return False
        self._moveit_world_object = False
        return True

    def _attach_moveit_object(self) -> bool:
        object_id = self._object_id()
        tcp_link = str(self.get_parameter("tcp_link").value)
        attached = AttachedCollisionObject()
        attached.link_name = tcp_link
        attached.touch_links = [tcp_link, "gripper_body", "finger_left", "finger_right"]
        attached.object.header.frame_id = tcp_link
        attached.object.id = object_id
        attached.object.operation = CollisionObject.ADD
        attached.object.primitives = [self._cube_primitive()]
        relative_pose = Pose()
        relative_pose.orientation.x = 1.0
        attached.object.primitive_poses = [relative_pose]
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        self.get_logger().info("STATE: ATTACH_OBJECT (MoveIt)")
        if not self._apply_planning_scene(scene, "ATTACH_OBJECT"):
            return False
        if not self._wait_for_scene_state(world_present=False, attached=True):
            return False
        self._moveit_attached = True
        self._moveit_world_object = False
        self._milestones["attach_success"] = True
        # If a previously placed box is picked again, it no longer occupies a
        # tray cell after the physical and MoveIt attachments are confirmed.
        if self._bin_occupancy.pop(self._active_source_id, None) is not None:
            self._publish_bin_occupancy()
        return True

    def _detach_moveit_object(self) -> bool:
        object_id = self._object_id()
        tcp_link = str(self.get_parameter("tcp_link").value)
        remove = AttachedCollisionObject()
        remove.link_name = tcp_link
        remove.object.id = object_id
        remove.object.operation = CollisionObject.REMOVE

        if self._pose_source != "fixed":
            bin_object_pose = list(self.get_parameter("bin_object_pose").value)
            if self._pending_place_allocation is not None:
                bin_object_pose[0:2] = self._pending_place_allocation["center_xy"]
            world_pose = Pose()
            world_pose.position.x, world_pose.position.y, world_pose.position.z = bin_object_pose[:3]
            world_pose.orientation.z = math.sin(bin_object_pose[3] / 2.0)
            world_pose.orientation.w = math.cos(bin_object_pose[3] / 2.0)
        else:
            place = self._pose("place_pose")
            world_pose = copy.deepcopy(place.pose)
            # Planning Scene stores the resting object center, not the TCP
            # release pose (which deliberately includes a small clearance).
            world_pose.position.z -= float(
                list(self.get_parameter("place_offset_xyz").value)[2]
            )
            world_pose.orientation.x = 0.0
            world_pose.orientation.y = 0.0
            world_pose.orientation.z = 0.0
            world_pose.orientation.w = 1.0
        world_object = CollisionObject()
        world_object.header.frame_id = self._pose_frame
        world_object.id = object_id
        world_object.operation = CollisionObject.ADD
        world_object.primitives = [self._cube_primitive()]
        world_object.primitive_poses = [world_pose]
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [remove]
        scene.world.collision_objects = [world_object]
        self.get_logger().info("STATE: DETACH_OBJECT (MoveIt)")
        if not self._apply_planning_scene(scene, "DETACH_OBJECT"):
            return False
        if not self._wait_for_scene_state(world_present=True, attached=False):
            return False
        self._moveit_attached = False
        self._moveit_world_object = True
        self._milestones["detach_success"] = True
        return True

    def _verify_lift(self) -> bool:
        rclpy.spin_once(self, timeout_sec=0.2)
        if self._attachment_state != "attached" or not self._moveit_attached:
            return self._error(
                "VERIFY_LIFT: object lost during lift; Gazebo/MoveIt attachment mismatch"
            )
        self._milestones["grasp_success"] = True
        return True

    def _move_preplace_with_fallback(self, transfer_command: int) -> bool:
        """Reach the release approach pose without sacrificing the safe lift.

        A horizontal Cartesian transfer can end at a UR3 wrist singularity in
        its final few millimetres, despite both endpoints being valid.  Do not
        execute a partial path in that case: retry the same high target with
        collision-aware point-to-point planning.  The pan remains attached and
        its later vertical release segment is still Cartesian.
        """
        if self._send_arm_goal("MOVE_PREPLACE", transfer_command, "preplace_pose"):
            return True
        if transfer_command != UR3Control.Goal.MOVE_LINEAR:
            return False
        cartesian_error = self._last_error
        self.get_logger().warning(
            "MOVE_PREPLACE_CARTESIAN_FALLBACK: "
            f"{cartesian_error}; retrying collision-aware point-to-point plan"
        )
        self._last_error = ""
        return self._send_arm_goal(
            "MOVE_PREPLACE_FALLBACK", UR3Control.Goal.MOVE_POSE, "preplace_pose"
        )

    def _descend_to_place_with_recovery(self) -> bool:
        """Retry alternate in-bin release points after Cartesian failures.

        A redundant UR3 pose can occasionally leave OMPL at an IK branch from
        which the vertical Cartesian segment self-collides. Replanning to a
        small offset that remains well inside the 200 x 180 mm bin selects a
        different branch without changing the grasp or the place destination.
        """
        if self._pose_source != "fixed":
            base_place = list(self._dynamic_poses["place_pose"])
            base_preplace = list(self._dynamic_poses["preplace_pose"])
        else:
            base_place = list(self.get_parameter("place_pose").value)
            base_preplace = list(self.get_parameter("preplace_pose").value)
        offset = float(self.get_parameter("place_recovery_offset_m").value)
        offsets = [(0.0, 0.0), (offset, 0.0), (-offset, 0.0), (0.0, offset), (0.0, -offset)]
        attempts = max(1, min(int(self.get_parameter("place_recovery_attempts").value), len(offsets)))
        for attempt, (dx, dy) in enumerate(offsets[:attempts], start=1):
            place = list(base_place)
            preplace = list(base_preplace)
            place[0] += dx
            place[1] += dy
            preplace[0] += dx
            preplace[1] += dy
            if self._pose_source != "fixed" and self._pending_place_allocation is not None:
                bin_pose = list(self.get_parameter("bin_object_pose").value)
                candidate_center = [place[0], place[1]]
                candidate_aabb = validate_bin_place_center(
                    candidate_center,
                    self._pending_place_allocation["dimensions_xy"],
                    bin_pose[:2],
                    [float(value) for value in self.get_parameter("bin_interior_size_xy").value],
                    list(self._bin_occupancy.values()),
                    float(self.get_parameter("bin_place_clearance_m").value),
                )
                if candidate_aabb is None:
                    self.get_logger().warning(
                        f"PLACE_RECOVERY_SKIPPED: occupied/outside at ({place[0]:.3f},{place[1]:.3f})"
                    )
                    continue
            self._override_poses["place_pose"] = place
            self._override_poses["preplace_pose"] = preplace
            self._override_poses["retreat_pose"] = list(preplace)
            if attempt > 1:
                self.get_logger().warning(
                    f"PLACE_RECOVERY {attempt}/{attempts}: replan at offset "
                    f"dx={dx:.3f}, dy={dy:.3f}"
                )
                if not self._send_arm_goal(
                    "RECOVER_PREPLACE", UR3Control.Goal.MOVE_POSE, "preplace_pose"
                ):
                    continue
            if self._send_arm_goal(
                "DESCEND_TO_PLACE", UR3Control.Goal.MOVE_LINEAR, "place_pose"
            ):
                if self._pose_source != "fixed" and self._pending_place_allocation is not None:
                    self._pending_place_allocation["center_xy"] = [place[0], place[1]]
                    self._pending_place_allocation["aabb_xyxy"] = candidate_aabb
                return True
        return self._error(
            f"DESCEND_TO_PLACE: exhausted {attempts} Cartesian recovery attempts"
        )

    def _mark_place_success(self) -> bool:
        if self._pending_place_allocation is not None:
            self._bin_occupancy[self._active_source_id] = copy.deepcopy(
                self._pending_place_allocation
            )
            self._publish_bin_occupancy()
            self.get_logger().info(
                f"BIN_OCCUPANCY_COMMITTED: {self._active_source_id} -> "
                f"slot={self._pending_place_allocation['slot_id']}, "
                f"aabb={self._pending_place_allocation['aabb_xyxy']}"
            )
        self._milestones["place_success"] = True
        return True

    def _run_episode(self, episode_index: int) -> dict:
        self._current_episode = episode_index
        self._episode_steps = []
        self._prepare_episode(episode_index)
        self._publish_episode_event(
            "episode_start",
            object_pose_source=self._pose_source,
            random_seed=int(self.get_parameter("random_seed").value),
            randomized=bool(self.get_parameter("randomize_object").value),
            requested_object_pose=self._episode_reset_pose,
            error_injection_axis=self._injection_axis,
            error_injection_value=self._injection_value,
        )
        episode_started = time.monotonic()
        transfer_command = (
            UR3Control.Goal.MOVE_LINEAR
            if bool(self.get_parameter("industrial_linear_transfers").value)
            else UR3Control.Goal.MOVE_POSE
        )

        reset_steps = [("RESET_OBJECT", self._reset_object_for_episode)]
        if self._pose_source == "fixed":
            steps = reset_steps + [
                ("OPEN_GRIPPER", lambda: self._command_gripper(
                    "OPEN_GRIPPER", opening=True)),
                ("MOVE_VIEW", lambda: self._send_joint_goal(
                    "MOVE_VIEW", "view_joint_pose")),
                ("MOVE_PREGRASP", lambda: self._send_arm_goal(
                    "MOVE_PREGRASP", transfer_command, "pregrasp_pose")),
            ]
        else:
            steps = reset_steps + [
                ("OPEN_GRIPPER", lambda: self._command_gripper(
                    "OPEN_GRIPPER", opening=True)),
                ("MOVE_VIEW", lambda: self._send_joint_goal(
                    "MOVE_VIEW", "view_joint_pose")),
                ("FREEZE_PERCEPTION_POSE", self._acquire_and_freeze_observation),
                ("UPDATE_PLANNING_SCENE", self._update_observed_object),
                ("MOVE_PREGRASP", lambda: self._send_arm_goal(
                    "MOVE_PREGRASP", transfer_command, "pregrasp_pose")),
                ("CLEAR_OBJECT_FOR_APPROACH", self._remove_world_object_for_approach),
            ]
        steps += [
            ("DESCEND_TO_GRASP", lambda: self._send_arm_goal(
                "DESCEND_TO_GRASP", UR3Control.Goal.MOVE_LINEAR, "grasp_pose")),
            ("CLOSE_GRIPPER", lambda: self._command_gripper(
                "CLOSE_GRIPPER", opening=False)),
            ("ATTACH_GAZEBO", lambda: self._set_gazebo_attachment(True)),
            ("ATTACH_MOVEIT", self._attach_moveit_object),
            ("LIFT", lambda: self._send_arm_goal(
                "LIFT", UR3Control.Goal.MOVE_LINEAR, "lift_pose")),
            ("VERIFY_LIFT", self._verify_lift),
            ("MOVE_PREPLACE", lambda: self._move_preplace_with_fallback(
                transfer_command)),
            ("DESCEND_TO_PLACE", self._descend_to_place_with_recovery),
            ("OPEN_GRIPPER_PLACE", lambda: self._command_gripper(
                "OPEN_GRIPPER_PLACE", opening=True)),
            ("DETACH_GAZEBO", lambda: self._set_gazebo_attachment(False)),
            ("DETACH_MOVEIT", self._detach_moveit_object),
            ("CONFIRM_PLACE", self._mark_place_success),
            ("RETREAT", lambda: self._send_arm_goal(
                "RETREAT", UR3Control.Goal.MOVE_LINEAR,
                "retreat_pose" if self._pose_source != "fixed" else "preplace_pose")),
        ]
        if bool(self.get_parameter("return_home").value):
            steps.append(("RETURN_HOME", lambda: self._send_arm_goal(
                "RETURN_HOME", UR3Control.Goal.MOVE_HOME)))

        success = True
        failed_step = ""
        failure_reason = ""
        episode_label = f"{episode_index}/{self._episode_count}"
        self.get_logger().info(f"EPISODE {episode_label}: START")
        for label, callback in steps:
            if not self._execute_step(label, callback):
                success = False
                failed_step = label
                failure_reason = self._last_error or "step returned false"
                self.get_logger().error(
                    f"EPISODE {episode_index}: FAIL at {label}: {failure_reason}"
                )
                break
        # Planning is a motion-pipeline metric, independent from failures in
        # perception, gripper I/O, attachment, or placement confirmation.
        # Count only arm-motion states that were actually attempted; otherwise
        # a late gripper timeout incorrectly turns a successful plan into a
        # planning failure.
        motion_states = {
            "MOVE_VIEW", "MOVE_PREGRASP", "DESCEND_TO_GRASP", "LIFT",
            "MOVE_PREPLACE", "DESCEND_TO_PLACE", "RETREAT", "RETURN_HOME",
        }
        attempted_motion_steps = [
            step for step in self._episode_steps if step["state"] in motion_states
        ]
        successful_motion_steps = sum(
            1 for step in attempted_motion_steps if step["success"]
        )
        self._milestones["planning_success"] = bool(attempted_motion_steps) and (
            successful_motion_steps == len(attempted_motion_steps)
        )
        total_time = time.monotonic() - episode_started
        planning_time = sum(step["planning_time_sec"] for step in self._episode_steps)
        execution_time = sum(step["execution_time_sec"] for step in self._episode_steps)
        result = {
            "episode": episode_index,
            "source_object_id": self._active_source_id,
            "source_model": self._active_object_model,
            "object_pose_source": self._pose_source,
            "success": success,
            "failed_step": failed_step,
            "failure_stage": failed_step or "NONE",
            "failure_reason": failure_reason,
            "total_time_sec": round(total_time, 6),
            "planning_time_sec": round(planning_time, 6),
            "execution_time_sec": round(execution_time, 6),
            "planning_attempt_count": len(attempted_motion_steps),
            "planning_success_count": successful_motion_steps,
            "perception_error_m": self._frozen_evaluation_error,
            "error_injection_axis": self._injection_axis,
            "error_injection_value": self._injection_value,
            "randomized_object_pose": self._episode_reset_pose,
            "observed_dimensions_m": self._object_dimensions,
            "frozen_pose": self._dynamic_poses.get("object_pose"),
            "generated_poses": self._dynamic_poses,
            "bin_place_allocation": copy.deepcopy(self._pending_place_allocation),
            "bin_occupancy_count": len(self._bin_occupancy),
            **self._milestones,
            "steps": self._episode_steps,
        }
        if success:
            self.get_logger().info(
                f"EPISODE {episode_index}: SUCCESS total={total_time:.2f}s "
                f"planning={planning_time:.2f}s execution={execution_time:.2f}s"
            )
        self._publish_episode_event("episode_end", result=result)
        return result

    @staticmethod
    def _rate(episodes, field: str) -> float:
        return round(sum(bool(ep[field]) for ep in episodes) / len(episodes), 6) if episodes else 0.0

    def _write_metrics(self, episodes: list) -> dict:
        successes = sum(1 for episode in episodes if episode["success"])
        errors = [
            float(ep["perception_error_m"])
            for ep in episodes
            if ep["perception_error_m"] is not None
        ]
        successful_injections = [
            abs(float(ep["error_injection_value"]))
            for ep in episodes if ep["success"]
        ]
        groups = {}
        for episode in episodes:
            key = f"{episode['error_injection_axis']}:{episode['error_injection_value']:.6f}"
            group = groups.setdefault(key, {"trials": 0, "successes": 0, "failure_stages": {}})
            group["trials"] += 1
            group["successes"] += int(episode["success"])
            if not episode["success"]:
                stage = episode["failure_stage"]
                group["failure_stages"][stage] = group["failure_stages"].get(stage, 0) + 1
        for group in groups.values():
            group["success_rate"] = round(group["successes"] / group["trials"], 6)

        summary = {
            "robot_type": "ur3",
            "object_pose_source": self._pose_source,
            "requested_episodes": self._episode_count,
            "completed_episodes": len(episodes),
            "successful_episodes": successes,
            "failed_episodes": len(episodes) - successes,
            "success_rate": round(successes / len(episodes), 6) if episodes else 0.0,
            "perception_success_rate": self._rate(episodes, "perception_success"),
            "planning_success_rate": self._rate(episodes, "planning_success"),
            "grasp_success_rate": self._rate(episodes, "grasp_success"),
            "place_success_rate": self._rate(episodes, "place_success"),
            "end_to_end_success_rate": round(successes / len(episodes), 6) if episodes else 0.0,
            "mean_perception_error_m": round(sum(errors) / len(errors), 6) if errors else None,
            "max_perception_error_m": round(max(errors), 6) if errors else None,
            "maximum_tolerated_pose_error": max(successful_injections) if successful_injections else None,
            "mean_total_time_sec": round(
                sum(ep["total_time_sec"] for ep in episodes) / len(episodes), 6
            ) if episodes else 0.0,
            "mean_planning_time_sec": round(
                sum(ep["planning_time_sec"] for ep in episodes) / len(episodes), 6
            ) if episodes else 0.0,
            "mean_execution_time_sec": round(
                sum(ep["execution_time_sec"] for ep in episodes) / len(episodes), 6
            ) if episodes else 0.0,
            "robustness_by_injected_error": groups,
        }
        payload = {
            "schema_version": 2,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "oracle_usage": "evaluation_only",
            "summary": summary,
            "episodes": episodes,
        }
        self._metrics_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        csv_path = self._metrics_path.with_suffix(".csv")
        fields = [
            "episode", "object_pose_source", "success", "failed_step",
            "failure_stage", "failure_reason", "perception_error_m",
            "error_injection_axis", "error_injection_value", "perception_success",
            "planning_success", "grasp_success", "place_success",
            "total_time_sec", "planning_time_sec", "execution_time_sec",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for episode in episodes:
                writer.writerow({key: episode[key] for key in fields})
        return summary

    def run(self) -> bool:
        if not self._wait_for_servers():
            return False
        episodes = []
        summary = {}
        for episode_index in range(1, self._episode_count + 1):
            result = self._run_episode(episode_index)
            episodes.append(result)
            summary = self._write_metrics(episodes)
            if not result["success"] and self._stop_on_failure:
                break
            if episode_index < self._episode_count:
                time.sleep(self._inter_episode_delay)
        rate = 100.0 * summary["success_rate"]
        self.get_logger().info(
            f"BENCHMARK SUMMARY: {summary['successful_episodes']}/"
            f"{summary['completed_episodes']} successful ({rate:.1f}%), "
            f"source={self._pose_source}, metrics={self._metrics_path}"
        )
        self._publish_episode_event("run_complete", summary=summary)
        all_successful = (
            summary["completed_episodes"] == self._episode_count
            and summary["successful_episodes"] == self._episode_count
        )
        if all_successful:
            if self._pose_source == "fixed":
                self.get_logger().info("DONE: repeatable fixed pick-and-place benchmark completed")
            else:
                self.get_logger().info("DONE: perception-driven pick-and-place benchmark completed")
        return all_successful

def main(args=None) -> None:
    rclpy.init(args=args)
    node = FixedPickPlace()
    try:
        success = node.run()
    except (KeyboardInterrupt, ValueError) as exc:
        node.get_logger().error(f"Trial interrupted: {exc}")
        success = False
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
