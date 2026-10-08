#!/usr/bin/env python3
import sys
import rclpy
import threading
import math
import time

from rclpy.node import Node
from sensor_msgs.msg import JointState
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence

class JointStateNode(Node):
    
    """ROS 2 Node that publishes JointState messages based on slider values."""
    def __init__(self):
        super().__init__('sus2f_joint_state_gui')
        
        self.publisher = self.create_publisher(
            JointState, 
            '/joint_states', 
            10)
        self.gripper_dis = 0
        self.joint_positions = [0.0] * 12  # Default joint positions (degrees)
        
    def update_distance_gripper(self, dis_mm):
        
        self.gripper_dis = dis_mm
        radian = math.acos((63.45-self.gripper_dis)/108)
        yF = math.sqrt(2916-(31.725-self.gripper_dis/2)**2)+3.5
        slider = (57.5-yF)/1000
        buff = -math.pi/2+radian
        """Update joint positions from the UI and publish."""
        self.joint_positions[0] = slider                    # base_slider_l_joint
        self.joint_positions[1] = -buff                     # finger_outer_l_joint
        self.joint_positions[2] = buff                      # outet_dummy_l_joint       
        self.joint_positions[3] = -buff                     # finger_inner_l_joint
        self.joint_positions[4] = 2*buff                    # pad_inner_l_joint
        self.joint_positions[5] = 2*buff                    # passive_pad_inner_l_joint
        self.joint_positions[6] = slider                    # sus2f_slider_r_link
        self.joint_positions[7] = buff                      # slider_outer_r_joint
        self.joint_positions[8] = buff                      # finger_outer_r_joint
        self.joint_positions[9] = buff                      # finger_inner_r_joint
        self.joint_positions[10] = -2*buff                   # pad_inner_r_joint
        self.joint_positions[11] = -2*buff                   # passive_pad_inner_r_joint
        
        """Publish the latest joint positions as a ROS 2 message."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = self.joint_positions
        msg.name = [
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
        self.publisher.publish(msg)
        self.get_logger().info(f"Published JointState: {msg.position}")
        
class JointStateUI(QWidget):
    """PyQt5 UI with sliders for controlling joint states."""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.ros_node = None  # Placeholder for the ROS node
        time.sleep(0.5)
        # Start ROS 2 in a separate thread
        self.ros_thread = threading.Thread(target=self.start_ros2,daemon =True)
        self.ros_thread.start()
        time.sleep(0.5)
        if self.ros_node:
            self.ros_node.update_distance_gripper(120)

    def init_ui(self):
        self.layout = QHBoxLayout()
        
        self.label = QLabel("POSITION")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(260)
        self.slider.setValue(240)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.slider_changed)
        # Unit Label (mm)
        self.unit_label = QLabel("120.0 mm", self)
        
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.unit_label)
        
        self.setLayout(self.layout)
        self.setWindowTitle('APICOO ROBOTICS SUSGRIP 2F')
        self.resize(500, 200)
        self.show()

    def slider_changed(self, value):
        self.unit_label.setText("{:5.1f} mm".format(value / 2))
        if self.ros_node:
            self.ros_node.update_distance_gripper(value/2)

    def start_ros2(self):
        """Initialize ROS 2 node and spin."""
        try:
            self.ros_node = JointStateNode()
            rclpy.spin(self.ros_node)
        except Exception as e:
            print(f"Error starting ROS 2 node: {e}")
        finally:
            if self.ros_node:
                self.ros_node.destroy_node()

    def keyPressEvent(self, event):
        if QKeySequence(event.key()+int(event.modifiers())) == QKeySequence("Ctrl+C"):
            self.close()
    def closeEvent(self, event):
        """Handle window close event to safely shut down ROS 2."""

        if self.ros_node:
            self.ros_node.destroy_node()
        event.accept()

def main():
    rclpy.init()
    try:
        app = QApplication(sys.argv)
        ui = JointStateUI()
        sys.exit(app.exec_())
    except SystemExit:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()