import rclpy,math   
from rclpy.node import Node
from rclpy.subscription import Subscription
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from control_msgs.action import GripperCommand
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class gazebo_joint_publisher(Node):
    def __init__(self):
        super().__init__('gazebo_joint_publisher')
        self.get_logger().info('gazebo_joint_publisher & Action Server started')
        
        # Use a ReentrantCallbackGroup to allow concurrent execution of subscriber and action callbacks
        self.callback_group = ReentrantCallbackGroup()
        
        self.joint_pub = self.create_publisher(JointState, '/rviz_joint_states', 10)
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10,
            callback_group=self.callback_group
        )
        self.current_gripper_pos = 0.0
        self.current_gripper_vel = 0.0
        self.current_gripper_force = 0.1

    def joint_states_callback(self, msg):
        try:
            idx = msg.name.index('gripper_joint')
            self.current_gripper_pos = msg.position[idx]
            self.current_gripper_vel = msg.velocity[idx]
            self.current_gripper_force = msg.effort[idx]
        except ValueError:
            return

        # Gazebo 0.0 is OPEN, 0.065 is CLOSED. The publisher math expects 0.0 to be CLOSED and 130.0 to be OPEN.
        # We map Gazebo single-finger position to total stroke gripper_dis (0.0 to 130.0 mm):
        gripper_dis = msg.position[idx]*2* 1000.0

        # Guard math domain limits
        val_to_cos = (63.45 - gripper_dis) / 108
        val_to_cos = max(-1.0, min(1.0, val_to_cos))
        radian = math.acos(val_to_cos)
        yF_val = 2916 - (31.725 - gripper_dis / 2) ** 2
        yF_val = max(0.0, yF_val)
        yF = math.sqrt(yF_val) + 3.5
        slider = (57.5 - yF) / 1000
        buff = -math.pi / 2 + radian

        joint_sus2f = [0.0] * 12
        joint_sus2f[0] = slider                    # base_slider_l_joint
        joint_sus2f[1] = -buff                     # finger_outer_l_joint
        joint_sus2f[2] = buff                      # outet_dummy_l_joint       
        joint_sus2f[3] = -buff                     # finger_inner_l_joint
        joint_sus2f[4] = 2*buff                    # pad_inner_l_joint
        joint_sus2f[5] = 2*buff                    # passive_pad_inner_l_joint
        joint_sus2f[6] = slider                    # base_slider_r_joint
        joint_sus2f[7] = buff                      # slider_outer_r_joint
        joint_sus2f[8] = buff                      # finger_outer_r_joint
        joint_sus2f[9] = buff                      # finger_inner_r_joint
        joint_sus2f[10] = -2*buff                  # pad_inner_r_joint
        joint_sus2f[11] = -2*buff                  # passive_pad_inner_r_joint

        new_msg = JointState()
        new_msg.header.stamp = self.get_clock().now().to_msg()
        new_msg.position = joint_sus2f
        new_msg.name = [
            "base_slider_l_joint",
            "slider_outer_l_joint",
            "finger_outer_l_joint",
            "finger_inner_l_joint",
            "pad_inner_l_joint",
            "passive_pad_inner_l_joint",
            "base_slider_r_joint",
            "slider_outer_r_joint",
            "finger_outer_r_joint",
            "finger_inner_r_joint",
            "pad_inner_r_joint",
            "passive_pad_inner_r_joint",
        ]
        self.joint_pub.publish(new_msg)

def main(args=None):
    rclpy.init(args=args)
    node = gazebo_joint_publisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): # Thêm dòng này để kiểm tra xem ROS 2 đã tắt chưa
            rclpy.shutdown()

if __name__ == '__main__':
    main()