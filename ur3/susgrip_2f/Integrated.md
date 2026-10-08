# SYSTEM INTEGRATION GUIDE: INDY7 ROBOT + SUSGRIP 2F GRIPPER (ROS 2 HUMBLE)
This document provides a detailed guide on integrating the SusGrip gripper into the Indy7 ecosystem, ensuring smooth simulation, collision-free operations, and stable trajectory planning.

---

## 1. WORKSPACE & ENVIRONMENT PREPARATION
To begin, you need two main repositories in your `src` directory:
* **indy-ros2**: The repository for controlling the Indy7 robot arm.
* **susgrip_2f**: The repository containing the SusGrip gripper description.


## 2. MODEL INTEGRATION (URDF & XACRO)

### 2.1. Attaching to Indy7
In the `indy.urdf.xacro` file, use conditional logic to attach the gripper:
* Attach `sus2f_base_link` to the robot's `tcp`.
* Create an additional link named `susgrip_tcp`, located 170mm (the length of the gripper) from the mounting flange, to act as the End-effector control point.

---

### 2.2. Simulation Integration
* Change the macro file calling the gripper from the description to the simulation in `indy_table.world`.
* **Drooping Arm Error (Power Loss):** In ROS 2 Humble, the `gazebo_ros2_control` plugin has a global variable bug.
    If the gripper uses PID (`effort` / `position_pid`), the plugin will automatically force the Robot Arm to also use PID, causing the arm to lose its `SetPosition()` rigidity and droop due to gravity.
* **Solution:** Separate the robot arm and the gripper into **two distinct `<ros2_control>` tags** within `indy.ros2_control.xacro`.
    The arm will use `POSITION` control (Instant Teleportation) for extreme rigidity, while the gripper will use soft PID control for physical interaction with objects.

---

## 3. DRIVER AND MOVEIT CONFIGURATION

### 3.1. Driver Configuration
* Split the `joint_state` topic into 2 distinct topics (1 topic for the robot, 1 topic for the gripper).
* Then, use a `joint_state_broadcaster` to merge these 2 topics into a single `joint_states` topic.

### 3.2. SRDF Configuration
* Declare collision exemptions (disable collision checking) between the fingers and the robot's end links within the SRDF file.

---

## 4. TRAJECTORY PLANNING CONFIGURATION (KINEMATICS & PLANNING)

#### 4.1. Fixing RViz Crash
In `kinematics.yaml`, completely remove the `timeout` and `resolution` parameters because they frequently cause type-casting errors (double vs. string) that crash RViz on Humble.
Leave only the following:
` ` `yaml
indy_manipulator:
  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver_attempts: 3
` ` `

#### 4.2. Declaring Planning Groups
* **`indy_manipulator` Group:** Modify the `tip_link` to `susgrip_tcp`.
* **`gripper` Group:** Includes the main joint `gripper_joint`.
* **`indy_susgrip` Group:** A new group encompassing all robot joints AND the gripper joints.
    This allows MoveIt to plan trajectories for both the arm and the gripper simultaneously.

---
#### 4.3. YAML Controllers Declaration
* Controller for the robot.
* Controller for the gripper.


## 5. Reference

Refer to the ROS2 system which integrates the Neuromeka robot and Susgrip_2F.
https://github.com/Teddy2209/intergrated-neuromeka-susgrip2f_apicoo