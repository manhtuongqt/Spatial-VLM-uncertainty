#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from ur3_moveit_control.action import UR3Control

class UR3ActionClient(Node):
    def __init__(self):
        super().__init__('ur3_action_client')
        self._action_client = ActionClient(self, UR3Control, 'ur3_control')

    def send_pose_goal(self, x, y, z, qx, qy, qz, qw):
        goal_msg = UR3Control.Goal()
        goal_msg.command_type = UR3Control.Goal.MOVE_POSE
        goal_msg.pose_goal.pose.position.x = float(x)
        goal_msg.pose_goal.pose.position.y = float(y)
        goal_msg.pose_goal.pose.position.z = float(z)
        goal_msg.pose_goal.pose.orientation.x = float(qx)
        goal_msg.pose_goal.pose.orientation.y = float(qy)
        goal_msg.pose_goal.pose.orientation.z = float(qz)
        goal_msg.pose_goal.pose.orientation.w = float(qw)

        self._action_client.wait_for_server()
        self.get_logger().info('Sending Pose Goal...')
        self.send_goal_future = self._action_client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def send_home_goal(self):
        goal_msg = UR3Control.Goal()
        goal_msg.command_type = UR3Control.Goal.MOVE_HOME
        self._action_client.wait_for_server()
        self.get_logger().info('Sending Home Goal...')
        self.send_goal_future = self._action_client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def send_joint_goal(self, j1, j2, j3, j4, j5, j6):
        goal_msg = UR3Control.Goal()
        goal_msg.command_type = UR3Control.Goal.MOVE_JOINT
        goal_msg.joint_goal.position = [float(j1), float(j2), float(j3), float(j4), float(j5), float(j6)]
        self._action_client.wait_for_server()
        self.get_logger().info('Sending Joint Goal...')
        self.send_goal_future = self._action_client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            import sys
            sys.exit(0)
            return
        self.get_logger().info('Goal accepted, waiting for result...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Action completed! Success: {result.success}, Message: {result.message}')
        import sys
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    action_client = UR3ActionClient()

    action_client.send_home_goal()
    # action_client.send_joint_goal(-1.57, -1.57, 1.57, -1.57, -1.57, 0.0)
    # action_client.send_pose_goal(0.3, 0.1, 0.35, 0.0, 0.0, 0.0, 1.0)

    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        pass
    finally:
        action_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
