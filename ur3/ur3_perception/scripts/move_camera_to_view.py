#!/usr/bin/env python3
"""Move the UR3 wrist camera to its calibrated simulation view posture."""

import math
import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class MoveCameraToView(Node):
    def __init__(self) -> None:
        super().__init__("move_camera_to_view")
        self.declare_parameter(
            "view_joint_pose", [1.1815, -1.8273, 1.3428, -1.0863, -1.5708, -1.9601]
        )
        self.declare_parameter("wait_for_server_sec", 60.0)
        # Use the active Gazebo trajectory controller directly.  This keeps
        # the view helper usable in a camera-only demo, which does not need to
        # launch the heavier MoveIt action server.
        self._client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')

    def run(self) -> bool:
        values = [float(item) for item in self.get_parameter("view_joint_pose").value]
        if len(values) != 6 or not all(math.isfinite(item) for item in values):
            self.get_logger().error("view_joint_pose must contain six finite values")
            return False
        if not self._client.wait_for_server(
            timeout_sec=float(self.get_parameter("wait_for_server_sec").value)
        ):
            self.get_logger().error("UR3 motion action server is unavailable")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
        ]
        point = JointTrajectoryPoint()
        point.positions = values
        point.time_from_start.sec = 8
        goal.trajectory.points = [point]
        self.get_logger().info("Moving UR3 to the calibrated wrist-camera view")
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        handle = future.result() if future.done() else None
        if handle is None or not handle.accepted:
            self.get_logger().error("Camera-view goal was rejected")
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=90.0)
        result = result_future.result() if result_future.done() else None
        if result is None or result.result.error_code != 0:
            self.get_logger().error("Could not reach the camera-view posture")
            return False
        self.get_logger().info("CAMERA_VIEW_READY")
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MoveCameraToView()
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
