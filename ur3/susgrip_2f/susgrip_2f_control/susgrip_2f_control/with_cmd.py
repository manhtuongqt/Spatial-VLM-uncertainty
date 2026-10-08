import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand

class GripperClient(Node):
    def __init__(self):
        super().__init__('gripper_client')
        self._action_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')

    def send_goal(self, position, max_effort=50.0):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Feedback: position={feedback.position}, effort={feedback.effort}, stalled={feedback.stalled}')

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: position={result.position}, effort={result.effort}, stalled={result.stalled}')

def main(args=None):
    rclpy.init(args=args)
    node = GripperClient()
    node.send_goal(0.0, 50.0)  # Open gripper
    while rclpy.ok():
        position = input("Enter position: ")    # 0 -> 130 milimet (for one finger)
        node.send_goal(float(position), 50.0)  # Open gripper
        rclpy.spin_once(node)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()