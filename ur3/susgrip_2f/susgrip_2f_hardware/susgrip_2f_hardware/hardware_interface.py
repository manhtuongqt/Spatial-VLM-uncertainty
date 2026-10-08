#!/usr/bin/env python3
import sys,time
import os
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from control_msgs.action import GripperCommand

# Ensure SusGripLib can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from susgrip_2f_hardware.SusGripLib import SusGrip2F

class SusGrip2FHardwareInterface(Node):
    def __init__(self):
        super().__init__('susgrip_2f_hardware_interface')
        self.get_logger().info('SusGrip 2F Standalone Hardware Interface & Action Server starting...')
        # Declare parameters
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('slave_id', 1)
        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        slave_id = self.get_parameter('slave_id').get_parameter_value().integer_value
        self.is_connected = False

        try:
            self.sus_2f = SusGrip2F(port=serial_port, id=slave_id)
            self.sus_2f.set_rtu_mode()
            self.get_logger().info(f"Connected to gripper successfully on {serial_port} (ID: {slave_id})!")
            self.is_connected = True
        except Exception as e:
            self.get_logger().error(f"Failed to connect to gripper on {serial_port}: {str(e)}")
            self.is_connected = False

        # Multi-threaded execution group for concurrent action handling and telemetry
        self.callback_group = ReentrantCallbackGroup()
        # Dynamic parameter values for speed and force (default to 50%)
        self.current_vel = 80.0
        self.current_tor = 50.0

        # Subscriptions for speed and force (like the old driver)
        self.vel_sub = self.create_subscription(
            Float32,
            '/susgrip_2f/vel',
            self.vel_callback,
            10,
            callback_group=self.callback_group
        )
        self.force_sub = self.create_subscription(
            Float32,
            '/susgrip_2f/force',
            self.force_callback,
            10,
            callback_group=self.callback_group
        )

        # Joint states publisher (20Hz)
        self.joint_pub = self.create_publisher(JointState, '/susgrip/joint_states', 10)
        self.create_timer(0.05, self.publish_joint_states, callback_group=self.callback_group)

        # Standard control_msgs GripperCommand Action Server
        self._action_server = ActionServer(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd',
            execute_callback=self.execute_action_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )

        self.get_logger().info('GripperCommand Action Server is active at /gripper_controller/gripper_cmd !')

    def vel_callback(self, msg):
        self.current_vel = max(1.0, min(100.0, msg.data))
        self.get_logger().info(f"Updated gripper velocity limit: {self.current_vel}%")

    def force_callback(self, msg):
        self.current_tor = max(1.0, min(100.0, msg.data))
        self.get_logger().info(f"Updated gripper force (torque) limit: {self.current_tor}%")

    def publish_joint_states(self):
        if not self.is_connected:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position.extend([getattr(self, 'dummy_pos', 0.0), getattr(self, 'dummy_pos', 0.0)])
            msg.name.extend(["gripper_joint", "gripper_joint_mimic"])
            self.joint_pub.publish(msg)
            return
        data = self.sus_2f.reload_data()
        if data is None or not isinstance(data, list) or len(data) < 2:
            return
        # Kinematic translation from linear distance (mm) to angular joints
        gripper_dis = float(data[1])

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        # The total opening (gripper_dis) is distributed equally between the two fingers.
        # Thus, each finger's position is (gripper_dis / 2) / 1000.0 meters.
        finger_pos = (gripper_dis / 2.0) / 1000.0
        # Left finger (gripper_joint)
        msg.position.append(finger_pos)
        msg.name.append("gripper_joint")
        # Right finger (gripper_joint_mimic)
        msg.position.append(finger_pos)
        msg.name.append("gripper_joint_mimic")

        self.joint_pub.publish(msg)

    # --- Action Goal Handling ---
    def goal_callback(self, goal_request):
        self.get_logger().info(f'Received new gripper goal request: {goal_request.command.position} (meters)')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received request to cancel active gripper goal')
        return CancelResponse.ACCEPT

    def execute_action_callback(self, goal_handle):
        self.get_logger().info('Executing gripper goal...')
        
        # The action goal position represents a single finger's position in meters.
        # Total physical opening is 2x this amount, converted to mm.
        target_mm = round(goal_handle.request.command.position * 2000.0)



        # Override target torque if specified via action max_effort (if non-zero)
        target_effort = self.current_tor

        if goal_handle.request.command.max_effort > 0.0:
            target_effort = max(1.0, min(100.0, goal_handle.request.command.max_effort))

        # Guard physical boundaries
        target_mm = max(0.0, min(130.0, target_mm))
        self.get_logger().info(f'Action Target Physical opening: {target_mm:.1f} mm | Speed: {self.current_vel}% | Force: {target_effort}%')

        if self.is_connected:
            # Check for faults and auto-reset if necessary
            data = self.sus_2f.reload_data()
            if data is not None and len(data) > 0:
                status = data[0]
                fault_code = status >> 8
                if fault_code != 0:
                    self.get_logger().warn(f"Gripper is in FAULT state (Code: {fault_code}). Auto-resetting before move...")
                    self.sus_2f.set_rtu_mode()
                    time.sleep(0.1)

            self.sus_2f.rtu_set_pos_pvt(int(target_mm), int(self.current_vel), int(target_effort))
            print("moving",flush=True)
            time.sleep(0.25)
        # Loop and monitor feedback until movement completes
        feedback_msg = GripperCommand.Feedback()
        feedback_msg.reached_goal = False
        result = GripperCommand.Result()
        rate = self.create_rate(20) # 20Hz monitor loop
        timeout = 30.0

        start_time = self.get_clock().now()
        last_pos = -999.0
        last_pos_time = 0.0

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Gripper action goal canceled.')
                result.reached_goal = False
                return result

            # Read current position for feedback
            if self.is_connected:
                data = self.sus_2f.reload_data()
                if data is not None and len(data) >= 6:
                    current_mm = float(data[1])
                    
                    # Status register MOTION is data[5] (Address 0x0005). 
                    # Bit 0 = 0: Stopped, Bit 0 = 1: Moving.
                    is_stopped = 1 if (int(data[5]) & 0x01) == 0 else 0
                    
                    # Timeout check and Elapsed time
                    elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9

                    # Stall detection: if position barely changes over 1.5 seconds, force stop!
                    if abs(current_mm - last_pos) > 0.5:
                        last_pos = current_mm
                        last_pos_time = elapsed
                    elif elapsed - last_pos_time > 1.5:
                        is_stopped = 1  # Force stopped status

                    # report feedback in meters for a single finger
                    feedback_msg.position = current_mm / 2000.0
                    feedback_msg.effort = float(data[3])
                    feedback_msg.reached_goal = False
                    goal_handle.publish_feedback(feedback_msg)
                    # Timeout check
                    elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
                    
                    if int(elapsed * 10) % 10 == 0:
                        self.get_logger().info(f"Monitor - Pos: {current_mm:.1f},\n"
                                               f"Target: {target_mm:.1f},\n"
                                               f"Status(Reg5): {int(data[5]):016b},\n"
                                               f"Stopped: {is_stopped}")

                    
                    if is_stopped == 1 and elapsed > 0.5:
                        if abs(current_mm - target_mm) <= 1 :
                            self.get_logger().info(f'Target reached successfully! Current: {current_mm:.1f} mm')
                            result.reached_goal = True
                            feedback_msg.reached_goal = True
                            goal_handle.publish_feedback(feedback_msg)
                            goal_handle.succeed()
                            break
                        else:
                            # It stopped early, maybe grasped an object!
                            self.get_logger().info(f'Gripper stopped early (object grasped?). Current: {current_mm:.1f} mm')
                            result.reached_goal = True
                            feedback_msg.reached_goal = True
                            goal_handle.publish_feedback(feedback_msg)
                            goal_handle.succeed()
                            break
                    elif elapsed > timeout:
                        self.get_logger().warn('Gripper move execution timed out!')
                        result.reached_goal = False
                        feedback_msg.reached_goal = False
                        goal_handle.publish_feedback(feedback_msg)
                        goal_handle.abort()
                        break
            else:
                self.get_logger().error('============= ERROR: GRIPPER IS DISCONNECTED! =============')
                self.get_logger().error('Serial connection to /dev/ttyUSB0 failed (permission denied or unplugged).')
                self.get_logger().error('Simulating goal execution for RViz ONLY. Real gripper WILL NOT MOVE!')
                self.get_logger().error('===========================================================')
                time.sleep(1.0) # simulate movement delay
                
                self.dummy_pos = float(target_mm) / 1000.0 # update dummy state for RViz
                
                result.reached_goal = True
                result.position = float(target_mm) / 2000.0
                feedback_msg.reached_goal = True
                feedback_msg.position = float(target_mm) / 2000.0
                goal_handle.publish_feedback(feedback_msg)
                goal_handle.succeed()
                break
            rate.sleep()
        # Populate final result in mm
        if self.is_connected:
            data = self.sus_2f.reload_data()
            if data is not None and len(data) >= 2:
                result.position = float(data[1]) / 2000.0
        else:
            result.position = float(target_mm) / 2000.0
        result.stalled = False
        return result

    def close_serial(self):
        """Close serial connection when shutting down"""
        if self.is_connected:
            self.sus_2f.disconnect()
            self.get_logger().info("Closed gripper serial connection.")


def main(args=None):
    rclpy.init(args=args)
    node = SusGrip2FHardwareInterface()
    
    # Use multi-threaded executor to allow feedback updates while goal runs
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down SusGrip 2F Hardware Node...")
        node.close_serial()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
    