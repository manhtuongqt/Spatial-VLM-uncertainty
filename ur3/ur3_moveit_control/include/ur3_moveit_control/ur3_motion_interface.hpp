#ifndef UR3_MOVEIT_CONTROL__UR3_MOTION_INTERFACE_HPP_
#define UR3_MOVEIT_CONTROL__UR3_MOTION_INTERFACE_HPP_

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>

namespace ur3_moveit_control
{

class UR3MotionInterface
{
public:
  explicit UR3MotionInterface(
    const rclcpp::Node::SharedPtr & node,
    const std::string & planning_group = "ur_manipulator");

  bool moveToJointGoal(const std::vector<double> & joint_goal);

  bool moveToPoseGoal(
    const geometry_msgs::msg::Pose & target_pose,
    const std::string & end_effector_link = "");

  bool moveLinearToPoseGoal(const geometry_msgs::msg::Pose & target_pose);

  bool moveHome();

  double getLastPlanningTime() const;

  double getLastExecutionTime() const;

  void stop();

  void addTableObstacle();

  void calibObjectHeightEyeInHand(double depth_m, double camera_z_mount_offset);

private:
  bool trajectoryHasSafeJointTravel(
    const moveit_msgs::msg::RobotTrajectory & trajectory,
    const std::string & motion_name) const;

  rclcpp::Node::SharedPtr node_;
  moveit::planning_interface::MoveGroupInterface move_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
  double last_planning_time_sec_{0.0};
  double last_execution_time_sec_{0.0};
  double max_joint_travel_rad_{2.2};
};

}  // namespace ur3_moveit_control

#endif  // UR3_MOVEIT_CONTROL__UR3_MOTION_INTERFACE_HPP_
