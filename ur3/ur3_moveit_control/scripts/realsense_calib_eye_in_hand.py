#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
import sys
import os
from datetime import datetime

class HandEyeCalibrator(Node):
    def __init__(self):
        super().__init__('hand_eye_calibrator')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.R_gripper2base_list = []
        self.t_gripper2base_list = []
        self.R_target2cam_list = []
        self.t_target2cam_list = []
        
        self.get_logger().info('=============================================')
        self.get_logger().info('  HAND-EYE CALIBRATION SCRIPT (MULTI-METHOD) ')
        self.get_logger().info('=============================================')
        self.get_logger().info('USAGE GUIDE:')
        self.get_logger().info('1. Move the robot arm to a pose where the marker is clearly visible.')
        self.get_logger().info('2. Press [Enter] to take a sample (at least 5 samples at different poses are recommended).')
        self.get_logger().info('3. Type "c" and press [Enter] to calculate the calibration offset.')
        self.get_logger().info('4. Type "q" and press [Enter] to quit.')
        self.get_logger().info('=============================================')
        
        # Start background thread for key readings
        self.input_thread = threading.Thread(target=self.input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()

    def get_transform_matrix(self, parent_frame, child_frame):
        try:
            # Wait up to 1.0s to get the latest TF
            trans = self.tf_buffer.lookup_transform(
                parent_frame, 
                child_frame, 
                rclpy.time.Time(), 
                rclpy.duration.Duration(seconds=1.0)
            )
            
            t = np.array([[trans.transform.translation.x],
                          [trans.transform.translation.y],
                          [trans.transform.translation.z]])
                          
            q = [trans.transform.rotation.x,
                 trans.transform.rotation.y,
                 trans.transform.rotation.z,
                 trans.transform.rotation.w]
                 
            rot = R.from_quat(q).as_matrix()
            
            return rot, t
        except Exception as e:
            self.get_logger().error(f"Error: Failed to find TF from {parent_frame} to {child_frame}: {e}")
            return None, None

    def take_sample(self):
        self.get_logger().info('Taking coordinate sample...')
        
        R_g2b, t_g2b = self.get_transform_matrix('base_link', 'tool0')
        if R_g2b is None:
            self.get_logger().warn('Sampling failed! Please check the robot state.')
            return
            
        R_t2c, t_t2c = self.get_transform_matrix('camera_link', 'aruco_marker_frame')
        if R_t2c is None:
            self.get_logger().warn('Sampling failed! Make sure the camera can see the ArUco marker.')
            return
            
        self.R_gripper2base_list.append(R_g2b)
        self.t_gripper2base_list.append(t_g2b)
        self.R_target2cam_list.append(R_t2c)
        self.t_target2cam_list.append(t_t2c)
        
        n = len(self.R_gripper2base_list)
        self.get_logger().info(f'Successfully saved sample {n}!')
        self.get_logger().info(f'   Tool0 pos: [{t_g2b[0][0]:.4f}, {t_g2b[1][0]:.4f}, {t_g2b[2][0]:.4f}]')
        self.get_logger().info(f'   Marker pos: [{t_t2c[0][0]:.4f}, {t_t2c[1][0]:.4f}, {t_t2c[2][0]:.4f}]')

    def build_homogeneous_matrix(self, R_mat, t_vec):
        """Construct a 4x4 homogeneous transformation matrix from R (3x3) and t (3x1)."""
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t_vec.flatten()
        return T

    def print_matrix_4x4(self, T, label=''):
        """Pretty-print a 4x4 matrix."""
        self.get_logger().info(f'--- {label} ---')
        for i in range(4):
            row = '  '.join([f'{T[i, j]:+.6f}' for j in range(4)])
            self.get_logger().info(f'  [ {row} ]')

    def compute_calibration(self):
        n = len(self.R_gripper2base_list)
        if n < 3:
            self.get_logger().error(f'At least 3 samples are required to calibrate! You only have {n} samples.')
            return
        
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('  CALCULATING HAND-EYE CALIBRATION...')
        self.get_logger().info(f'  Number of samples: {n}')
        self.get_logger().info('=' * 60)
        
        # OpenCV Hand-Eye Calibration methods
        methods = {
            'TSAI':       cv2.CALIB_HAND_EYE_TSAI,
            'PARK':       cv2.CALIB_HAND_EYE_PARK,
            'HORAUD':     cv2.CALIB_HAND_EYE_HORAUD,
            'ANDREFF':    cv2.CALIB_HAND_EYE_ANDREFF,
            'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        
        results = {}
        
        for name, method in methods.items():
            try:
                R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                    self.R_gripper2base_list,
                    self.t_gripper2base_list,
                    self.R_target2cam_list,
                    self.t_target2cam_list,
                    method=method
                )
                
                T_cam2gripper = self.build_homogeneous_matrix(R_cam2gripper, t_cam2gripper)
                rot = R.from_matrix(R_cam2gripper)
                rpy = rot.as_euler('XYZ', degrees=False)
                rpy_deg = rot.as_euler('XYZ', degrees=True)
                quat = rot.as_quat()  # x, y, z, w
                
                results[name] = {
                    'R': R_cam2gripper,
                    't': t_cam2gripper,
                    'T': T_cam2gripper,
                    'rpy': rpy,
                    'rpy_deg': rpy_deg,
                    'quat': quat,
                }
                
            except Exception as e:
                self.get_logger().error(f'  Algorithm {name} failed: {e}')
        
        if not results:
            self.get_logger().error('All algorithms failed! Please check sample data.')
            return
        
        # Print results of each algorithm
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('  ALGORITHM COMPARISON')
        self.get_logger().info('=' * 60)
        
        self.get_logger().info('')
        self.get_logger().info(f'  {"Method":<12} {"X(m)":>10} {"Y(m)":>10} {"Z(m)":>10} │ {"Roll°":>8} {"Pitch°":>8} {"Yaw°":>8}')
        self.get_logger().info('  ' + '─' * 72)
        
        for name, res in results.items():
            t = res['t']
            rpy_d = res['rpy_deg']
            self.get_logger().info(
                f'  {name:<12} {t[0][0]:>+10.5f} {t[1][0]:>+10.5f} {t[2][0]:>+10.5f} │ '
                f'{rpy_d[0]:>+8.3f} {rpy_d[1]:>+8.3f} {rpy_d[2]:>+8.3f}'
            )
        
        # Compute median translation to find the most robust result
        all_t = np.array([res['t'].flatten() for res in results.values()])
        median_t = np.median(all_t, axis=0)
        
        # Select method with translation closest to the median
        best_name = None
        best_dist = float('inf')
        for name, res in results.items():
            dist = np.linalg.norm(res['t'].flatten() - median_t)
            if dist < best_dist:
                best_dist = dist
                best_name = name
        
        best = results[best_name]
        
        # Print best method results
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'  BEST METHOD RESULT: {best_name}')
        self.get_logger().info(f'     (Closest to median, distance = {best_dist:.6f}m)')
        self.get_logger().info('=' * 60)
        
        t = best['t']
        rpy = best['rpy']
        rpy_deg = best['rpy_deg']
        quat = best['quat']
        T = best['T']
        
        # Print 4x4 homogeneous transformation matrix
        self.get_logger().info('')
        self.print_matrix_4x4(T, 'HOMOGENEOUS TRANSFORMATION MATRIX 4x4 (tool0 -> camera_link)')
        
        self.get_logger().info('')
        self.get_logger().info(f'  Translation (X, Y, Z):')
        self.get_logger().info(f'    X = {t[0][0]:+.6f} m')
        self.get_logger().info(f'    Y = {t[1][0]:+.6f} m')
        self.get_logger().info(f'    Z = {t[2][0]:+.6f} m')
        
        self.get_logger().info(f'  RPY Euler Angles (radians & degrees):')
        self.get_logger().info(f'    Roll  = {rpy[0]:+.6f} rad  ({rpy_deg[0]:+.3f}°)')
        self.get_logger().info(f'    Pitch = {rpy[1]:+.6f} rad  ({rpy_deg[1]:+.3f}°)')
        self.get_logger().info(f'    Yaw   = {rpy[2]:+.6f} rad  ({rpy_deg[2]:+.3f}°)')
        
        self.get_logger().info(f'  Quaternion (x, y, z, w):')
        self.get_logger().info(f'    [{quat[0]:+.6f}, {quat[1]:+.6f}, {quat[2]:+.6f}, {quat[3]:+.6f}]')
        
        # static_transform_publisher command
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('  TF PUBLISH COMMANDS')
        self.get_logger().info('=' * 60)
        
        self.get_logger().info('  Using Quaternion (Recommended):')
        self.get_logger().info(
            f'  ros2 run tf2_ros static_transform_publisher '
            f'--x {t[0][0]:.6f} --y {t[1][0]:.6f} --z {t[2][0]:.6f} '
            f'--qx {quat[0]:.6f} --qy {quat[1]:.6f} --qz {quat[2]:.6f} --qw {quat[3]:.6f} '
            f'--frame-id tool0 --child-frame-id camera_link'
        )
        
        # Older format (euler)
        self.get_logger().info('')
        self.get_logger().info('  Using Euler RPY (Legacy compatibility):')
        self.get_logger().info(
            f'  ros2 run tf2_ros static_transform_publisher '
            f'{t[0][0]:.6f} {t[1][0]:.6f} {t[2][0]:.6f} '
            f'{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f} '
            f'tool0 camera_link'
        )
        
        # Save results to file
        self.save_results(results, best_name)
        
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('  CALIBRATION COMPLETE!')
        self.get_logger().info('=' * 60)

    def save_results(self, results, best_name):
        """Save calibration results to .npz file and a human-readable .txt file"""
        try:
            save_dir = os.path.expanduser('~/ros_ws/calib_results')
            os.makedirs(save_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            npz_filepath = os.path.join(save_dir, f'hand_eye_calib_{timestamp}.npz')
            txt_filepath = os.path.join(save_dir, f'hand_eye_calib_{timestamp}.txt')
            
            save_data = {
                'best_method': best_name,
                'n_samples': len(self.R_gripper2base_list),
            }
            
            # Save npz data
            for name, res in results.items():
                save_data[f'{name}_T'] = res['T']
                save_data[f'{name}_R'] = res['R']
                save_data[f'{name}_t'] = res['t']
                save_data[f'{name}_quat'] = res['quat']
                save_data[f'{name}_rpy'] = res['rpy']
            
            save_data['R_gripper2base'] = np.array(self.R_gripper2base_list)
            save_data['t_gripper2base'] = np.array(self.t_gripper2base_list)
            save_data['R_target2cam'] = np.array(self.R_target2cam_list)
            save_data['t_target2cam'] = np.array(self.t_target2cam_list)
            
            np.savez(npz_filepath, **save_data)
            
            # Save txt report
            best = results[best_name]
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write("=============================================\n")
                f.write("      HAND-EYE CALIBRATION RESULTS\n")
                f.write("=============================================\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Number of samples: {len(self.R_gripper2base_list)}\n")
                f.write(f"Best method (closest to median): {best_name}\n\n")
                
                f.write("--- 4x4 HOMOGENEOUS TRANSFORMATION MATRIX (tool0 -> camera_link) ---\n")
                T = best['T']
                for i in range(4):
                    row = '  '.join([f'{T[i, j]:+.6f}' for j in range(4)])
                    f.write(f"  [ {row} ]\n")
                
                f.write("\n--- DETAILED PARAMETERS ---\n")
                t = best['t']
                rpy = best['rpy']
                rpy_deg = best['rpy_deg']
                quat = best['quat']
                
                f.write("1. Translation (X, Y, Z - meters):\n")
                f.write(f"   X = {t[0][0]:+.6f}\n")
                f.write(f"   Y = {t[1][0]:+.6f}\n")
                f.write(f"   Z = {t[2][0]:+.6f}\n\n")
                
                f.write("2. Euler Angles RPY (radians & degrees):\n")
                f.write(f"   Roll  = {rpy[0]:+.6f} rad  ({rpy_deg[0]:+.3f}°)\n")
                f.write(f"   Pitch = {rpy[1]:+.6f} rad  ({rpy_deg[1]:+.3f}°)\n")
                f.write(f"   Yaw   = {rpy[2]:+.6f} rad  ({rpy_deg[2]:+.3f}°)\n\n")
                
                f.write("3. Quaternion (x, y, z, w):\n")
                f.write(f"   [ {quat[0]:+.6f}, {quat[1]:+.6f}, {quat[2]:+.6f}, {quat[3]:+.6f} ]\n\n")
                
                f.write("--- ROS2 STATIC TF PUBLISH COMMAND (Using Quaternion) ---\n")
                f.write(f"ros2 run tf2_ros static_transform_publisher \\\n"
                        f"  --x {t[0][0]:.6f} --y {t[1][0]:.6f} --z {t[2][0]:.6f} \\\n"
                        f"  --qx {quat[0]:.6f} --qy {quat[1]:.6f} --qz {quat[2]:.6f} --qw {quat[3]:.6f} \\\n"
                        f"  --frame-id tool0 --child-frame-id camera_link\n")
            
            self.get_logger().info('')
            self.get_logger().info(f'  Saved raw data to: {npz_filepath}')
            self.get_logger().info(f'  Saved text report to: {txt_filepath}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to save files: {e}')

    def input_loop(self):
        while True:
            try:
                cmd = sys.stdin.readline().strip()
                if cmd.lower() == 'c':
                    self.compute_calibration()
                elif cmd.lower() == 'q':
                    self.get_logger().info('Exiting...')
                    rclpy.shutdown()
                    break
                else:
                    self.take_sample()
            except EOFError:
                break

def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
