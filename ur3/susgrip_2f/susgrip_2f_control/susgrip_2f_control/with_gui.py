#!/usr/bin/env python3
from std_msgs.msg import Int8
import sys
import rclpy
import threading
import math
import time
from std_msgs.msg import Float32, Int32
from rclpy.node import Node
from sensor_msgs.msg import JointState
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from rclpy.action import ActionServer, CancelResponse, GoalResponse


class SusGrip2FControl(Node):
    
    """ROS 2 Node that publishes JointState messages based on slider values."""
    def __init__(self):
        super().__init__('susgrip_2f_control_gui_node')

        self._action_client = ActionClient(self, GripperCommand, 
        '/gripper_controller/gripper_cmd')
        self.vel_publisher = self.create_publisher(
            Float32 , 
            '/susgrip_2f/vel', 
            10)
        self.force_publisher = self.create_publisher(
            Float32, 
            '/susgrip_2f/force', 
            10)
    def send_gripper_goal(self, pos ,vel, force):
        msg = Float32()
        msg.data = float(vel)
        self.vel_publisher.publish(msg)
        msg = Float32()
        msg.data = float(force)    
        self.force_publisher.publish(msg)
        self.get_logger().info(f'Published: {vel} to /susgrip_2f/vel')
        self.get_logger().info(f'Published: {force} to /susgrip_2f/force')

        goal_msg = GripperCommand.Goal()
        # ROS 2 Action Server expects meters (0.0 to 0.130), but GUI pos is in mm
        goal_msg.command.position = float(pos) / 1000.0
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

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
        self.get_logger().info('Result: {}'.format(result))
        
class SusGrip2FControlGUI(QWidget):
    """PyQt5 UI with sliders for controlling joint states."""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.ros_node = None
        time.sleep(1)
        # Start ROS 2 in a separate thread
        self.ros_thread = threading.Thread(target=self.start_ros2)
        self.ros_thread.daemon = True
        self.ros_thread.start()
        time.sleep(0.5)

    def init_ui(self):
        self.layout = QVBoxLayout()

        box_slider_pos = QHBoxLayout()
        self.label_pos = QLabel("POSITION")
        self.slider_pos = QSlider(Qt.Horizontal)
        self.slider_pos.setMinimum(0)
        self.slider_pos.setMaximum(260)
        self.slider_pos.setValue(240)
        self.slider_pos.setTickInterval(1)
        self.slider_pos.setTickPosition(QSlider.TicksBelow)
        self.slider_pos.valueChanged.connect(self.slider_pos_changed)
        # Unit Label (mm)
        self.unit_label_pos = QLabel("120.0 mm", self)

        box_slider_vel = QHBoxLayout()
        self.label_vel = QLabel("VELOCITY")
        self.slider_vel = QSlider(Qt.Horizontal)
        self.slider_vel.setMinimum(0)
        self.slider_vel.setMaximum(100)
        self.slider_vel.setValue(50)
        self.slider_vel.setTickInterval(1)
        self.slider_vel.setTickPosition(QSlider.TicksBelow)
        self.slider_vel.valueChanged.connect(self.slider_vel_changed)
        # Unit Label (mm)
        self.unit_label_vel = QLabel("50 %", self)

        box_slider_force = QHBoxLayout()
        self.label_force = QLabel("FORCE")
        self.slider_force = QSlider(Qt.Horizontal)
        self.slider_force.setMinimum(0)
        self.slider_force.setMaximum(100)
        self.slider_force.setValue(50)
        self.slider_force.setTickInterval(1)
        self.slider_force.setTickPosition(QSlider.TicksBelow)
        self.slider_force.valueChanged.connect(self.slider_force_changed)
        # Unit Label (mm)
        self.unit_label_force = QLabel("50 %", self)

        box_slider_pos.addWidget(self.label_pos)
        box_slider_pos.addWidget(self.slider_pos)
        box_slider_pos.addWidget(self.unit_label_pos)

        box_slider_vel.addWidget(self.label_vel)
        box_slider_vel.addWidget(self.slider_vel)
        box_slider_vel.addWidget(self.unit_label_vel)

        box_slider_force.addWidget(self.label_force)
        box_slider_force.addWidget(self.slider_force)
        box_slider_force.addWidget(self.unit_label_force)


        self.load_bt = QPushButton("LOAD", self)
        self.load_bt.clicked.connect(self.load_bt_handle)

        self.layout.addLayout(box_slider_pos)
        self.layout.addLayout(box_slider_vel)
        self.layout.addLayout(box_slider_force)
        self.layout.addWidget(self.load_bt)

        self.setLayout(self.layout)
        self.setWindowTitle('APICOO ROBOTICS SUSGRIP 2F')
        self.resize(500, 200)
        self.show()

    def load_bt_handle(self):
        pos = self.slider_pos.value()/2
        vel = self.slider_vel.value()
        force = self.slider_force.value()
        if self.ros_node:
            self.ros_node.send_gripper_goal(pos, vel, force)
            self.ros_node.get_logger().info(f"send goal {pos} mm, vel= {vel}%, force = {force}% to /susgrip_2f/gripper_command")

    def slider_pos_changed(self, value):
        self.unit_label_pos.setText("{:5.1f} mm".format(value / 2))

    def slider_vel_changed(self, value):
        self.unit_label_vel.setText("{:3d} %".format(value))

    def slider_force_changed(self, value):
        self.unit_label_force.setText("{:3d} %".format(value))

    def start_ros2(self):
        """Initialize ROS 2 node and spin."""
        rclpy.init()
        try:
            self.ros_node = SusGrip2FControl()
            rclpy.spin(self.ros_node)
        except Exception as e:
            print(f"Error starting ROS 2 node: {e}")
        finally:
            if self.ros_node:
                self.ros_node.destroy_node()
            rclpy.shutdown()

    def keyPressEvent(self, event):
        if QKeySequence(event.key()+int(event.modifiers())) == QKeySequence("Ctrl+C"):
            self.close()
    def closeEvent(self, event):
        """Handle window close event to safely shut down ROS 2."""

        if self.ros_node:
            self.ros_node.destroy_node()
        rclpy.shutdown()
        event.accept()

def main():
    app = QApplication(sys.argv)
    ui = SusGrip2FControlGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
