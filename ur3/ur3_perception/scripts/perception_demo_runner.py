#!/usr/bin/env python3
"""Move UR3 to a camera view pose and summarize oracle-validated detections."""

import json
import math
import sys
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from std_msgs.msg import Empty, String
from ur3_moveit_control.action import UR3Control


class PerceptionDemoRunner(Node):
    def __init__(self):
        super().__init__("perception_demo_runner")
        self.declare_parameter(
            "view_pose", [0.1734, 0.2790, 0.20, 0.70710678, -0.70710678, 0.0, 0.0]
        )
        self.declare_parameter(
            "view_joint_pose", [1.1815, -1.8273, 1.3428, -1.0863, -1.5708, -1.9601]
        )
        self.declare_parameter("test_object_pose", [-0.13, 0.24, 0.03])
        self.declare_parameter("metrics_output", "/tmp/ur3_perception_demo.json")
        self.declare_parameter("required_samples", 10)
        self.declare_parameter("timeout_sec", 120.0)
        self.declare_parameter("settle_time_sec", 3.0)
        self._client = ActionClient(self, UR3Control, "/ur3_control")
        self._detach_pub = self.create_publisher(
            Empty, "/ur3_sim/detach_red_cube", 10
        )
        self._reset_client = self.create_client(
            SetEntityPose, "/world/ur3_pick_place/set_pose"
        )
        self._samples = []
        self.create_subscription(String, "/ur3_perception/status", self._status, 10)

    def _status(self, msg: String):
        try:
            sample = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if sample.get("oracle_error_m") is not None:
            self._samples.append(sample)

    def run(self) -> bool:
        self.get_logger().info("Waiting for UR3 motion server...")
        if (not self._client.wait_for_server(timeout_sec=60.0)
                or not self._reset_client.wait_for_service(timeout_sec=60.0)):
            self.get_logger().error("Motion server or Gazebo reset service unavailable")
            return False
        # Fortress DetachableJoint starts attached by design. Reset it to the
        # oracle pose before positioning the camera.
        for _ in range(10):
            self._detach_pub.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.1)
        reset = SetEntityPose.Request()
        reset.entity.name = "red_cube"
        reset.entity.type = Entity.MODEL
        object_pose = [float(value) for value in self.get_parameter("test_object_pose").value]
        if len(object_pose) != 3 or not all(math.isfinite(value) for value in object_pose):
            self.get_logger().error("test_object_pose must contain three finite coordinates")
            return False
        reset.pose.position.x, reset.pose.position.y, reset.pose.position.z = object_pose
        reset.pose.orientation.w = 1.0
        reset_future = self._reset_client.call_async(reset)
        rclpy.spin_until_future_complete(self, reset_future, timeout_sec=10.0)
        if (not reset_future.done() or reset_future.result() is None
                or not reset_future.result().success):
            self.get_logger().error("Failed to reset red_cube for perception demo")
            return False
        # Enter the same deterministic, collision-checked industrial view
        # posture used by the production pick-and-place state machine.  A
        # Cartesian pose target can have several IK branches; with a wrist
        # camera fitted some branches are invalid even though another branch
        # is safe.  The calibrated joint posture removes that ambiguity while
        # preserving the downward-facing TCP and camera orientation.
        values = [float(value) for value in self.get_parameter("view_joint_pose").value]
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            self.get_logger().error("view_joint_pose must contain six finite joints")
            return False
        goal = UR3Control.Goal()
        goal.command_type = UR3Control.Goal.MOVE_JOINT
        goal.joint_goal.position = values
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=60.0)
        handle = future.result() if future.done() else None
        if handle is None or not handle.accepted:
            self.get_logger().error("View-pose goal rejected")
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)
        wrapped = result_future.result() if result_future.done() else None
        if wrapped is None or not wrapped.result.success:
            self.get_logger().error("Failed to reach RGB-D view pose")
            return False
        # Drain synchronized frames queued while the wrist was moving before
        # starting the stationary calibration window.
        settle_deadline = time.monotonic() + float(
            self.get_parameter("settle_time_sec").value
        )
        while time.monotonic() < settle_deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        self._samples.clear()
        self.get_logger().info("VIEW_READY: collecting synchronized oracle samples")
        deadline = time.monotonic() + float(self.get_parameter("timeout_sec").value)
        required = int(self.get_parameter("required_samples").value)
        while time.monotonic() < deadline and len(self._samples) < required:
            rclpy.spin_once(self, timeout_sec=0.2)
        if len(self._samples) < required:
            self.get_logger().error(f"Only received {len(self._samples)}/{required} detections")
            return False
        errors = [float(sample["oracle_error_m"]) for sample in self._samples[-required:]]
        report = {
            "success": True,
            "samples": required,
            "mean_error_m": sum(errors) / len(errors),
            "max_error_m": max(errors),
            "latest": self._samples[-1],
        }
        output = Path(str(self.get_parameter("metrics_output").value)).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        self.get_logger().info(
            f"PERCEPTION SUMMARY: {required} samples, mean error="
            f"{report['mean_error_m']*1000:.1f} mm, max="
            f"{report['max_error_m']*1000:.1f} mm, metrics={output}"
        )
        self.get_logger().info("DONE: RGB-D perception demo validated against Gazebo oracle")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionDemoRunner()
    try:
        success = node.run()
    except KeyboardInterrupt:
        success = False
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
