import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from rclpy.callback_groups import ReentrantCallbackGroup

class GazeboJointPublisher(Node):
    def __init__(self):
        super().__init__('gazebo_joint_publisher')
        
        # Use a ReentrantCallbackGroup to allow concurrent execution of subscriber and action callbacks
        self.callback_group = ReentrantCallbackGroup()
        
        self.mimic_pub = self.create_publisher(Float64MultiArray, '/mimic_joint_controller/commands', 10)
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.callback_group
        )
        self.current_gripper_pos = 0.0

        self.get_logger().info("gazebo_joint_publisher (Mimic Forwarder) started")

    def joint_state_callback(self, msg: JointState):
        try:
            # find index of gripper_joint
            idx = msg.name.index("gripper_joint")
            self.current_gripper_pos = msg.position[idx]
        except ValueError:
            return

        # Publish exactly the same position to the mimic joint controller in Gazebo
        mimic_msg = Float64MultiArray()
        mimic_msg.data = [self.current_gripper_pos]
        self.mimic_pub.publish(mimic_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GazeboJointPublisher()
    
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
