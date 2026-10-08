#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class SusGripKinematicsPublisher(Node):
    def __init__(self):
        super().__init__('susgrip_kinematics_publisher')
        
        # Subscribe to internal topic (Real Hardware pipeline)
        self.subscription_real = self.create_subscription(
            JointState,
            '/susgrip_internal/joint_states',
            self.hardware_joint_state_callback,
            10)
        self.subscription_fake = self.create_subscription(
            JointState,
            '/susgrip_test/joint_states',
            self.test_joint_state_callback,
            10)
        # Publish the 12 calculated mimic joints to the source_list topic (for real hardware)
        self.publisher_real_hardware = self.create_publisher(JointState,'/susgrip/joint_states', 10)
        # Publish directly to global joint states (for fake hardware)
        self.publisher_fake_hardware = self.create_publisher(JointState, '/joint_states', 10)
        
        self.get_logger().info("SusGrip Non-linear Kinematics Node Started")
        self.last_gripper_dis = -1.0  # Track last value to prevent infinite loop

    def hardware_joint_state_callback(self, msg):
        out_msg = self.joint_calculate(msg, check_loop=False)
        if out_msg is not None:
            self.publisher_real_hardware.publish(out_msg)

    def test_joint_state_callback(self, msg):
        # Fake Hardware publish thẳng vào /joint_states, nên phải bật check_loop
        out_msg = self.joint_calculate(msg, check_loop=True)
        if out_msg is not None:
            self.publisher_fake_hardware.publish(out_msg)

    def joint_calculate(self, msg, check_loop):
        try:
            index = msg.name.index("susgrip_distance_joint")
        except ValueError:
            return None

        gripper_dis = msg.position[index] * 1000.0
        
        if check_loop:
            if abs(gripper_dis - self.last_gripper_dis) < 0.1:
                return None
            self.last_gripper_dis = gripper_dis

        # Guard math domain limits
        val_to_cos = (63.45 - gripper_dis) / 108.0
        val_to_cos = max(-1.0, min(1.0, val_to_cos))
        radian = math.acos(val_to_cos)
        yF_val = 2916.0 - (31.725 - gripper_dis / 2.0) ** 2
        yF_val = max(0.0, yF_val)
        yF = math.sqrt(yF_val) + 3.5
        slider = (57.5 - yF) / 1000.0
        buff = -math.pi / 2.0 + radian

        out_msg = JointState()
        out_msg.header.stamp = msg.header.stamp 
        
        out_msg.name = [
            "susgrip_distance_joint", # Bắt buộc phải có khớp chính để forward
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
            "passive_pad_inner_r_joint"
        ]
        out_msg.position = [0.0]*13
        out_msg.position[0] = gripper_dis / 1000.0 # Bắt buộc forward khớp chính
        out_msg.position[1] = slider     # base_slider_l_joint
        out_msg.position[2] = -buff      # slider_outer_l_joint
        out_msg.position[3] = buff       # finger_outer_l_joint
        out_msg.position[4] = -buff      # finger_inner_l_joint
        out_msg.position[5] = 2*buff     # pad_inner_l_joint
        out_msg.position[6] = 2*buff     # passive_pad_inner_l_joint
        out_msg.position[7] = slider     # base_slider_r_joint
        out_msg.position[8] = buff       # slider_outer_r_joint
        out_msg.position[9] = buff       # finger_outer_r_joint
        out_msg.position[10] = buff      # finger_inner_r_joint
        out_msg.position[11] = -2*buff   # pad_inner_r_joint
        out_msg.position[12] = -2*buff   # passive_pad_inner_r_joint
        
        return out_msg



def main(args=None):
    rclpy.init(args=args)
    node = SusGripKinematicsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()