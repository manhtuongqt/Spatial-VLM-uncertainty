#!/usr/bin/env python3
# Copyright 2026 Antigravity
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math

class TestTopicControl(Node):
    def __init__(self):
        super().__init__('test_topic_control')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        self.timer = self.create_timer(2.0, self.timer_callback)
        self.get_logger().info("TestTopicControl node has started. Publishing commands to /forward_position_controller/commands...")
        self.step = 0

    def timer_callback(self):
        msg = Float64MultiArray()
        # UR joints sequence:
        # [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
        # Let's oscillate the joints safely.
        angle = math.sin(self.step * 0.3) * 0.4
        
        # Base configuration:
        # shoulder_pan = 0.0, shoulder_lift = -1.57, elbow = 0.0, wrist_1 = -1.57, wrist_2 = 0.0, wrist_3 = 0.0
        msg.data = [
            angle,                  # shoulder_pan_joint
            -1.57 + angle * 0.5,    # shoulder_lift_joint
            angle,                  # elbow_joint
            -1.57 + angle,          # wrist_1_joint
            angle,                  # wrist_2_joint
            angle                   # wrist_3_joint
        ]
        
        self.publisher_.publish(msg)
        formatted_data = [round(x, 3) for x in msg.data]
        self.get_logger().info(f"Published joint commands: {formatted_data}")
        self.step += 1

def main(args=None):
    rclpy.init(args=args)
    node = TestTopicControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
