#include "ur3_moveit_control/ur3_motion_interface.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

namespace ur3_moveit_control
{

  UR3MotionInterface::UR3MotionInterface(
      const rclcpp::Node::SharedPtr &node,
      const std::string &planning_group)
      : node_(node),
        move_group_(node_, planning_group)
  {
    move_group_.setPlanningTime(5.0);
    move_group_.setNumPlanningAttempts(10);

    double max_velocity_scaling = node_->get_parameter("max_velocity_scaling_factor").as_double();
    double max_acceleration_scaling = node_->get_parameter("max_acceleration_scaling_factor").as_double();
    std::string planner_id = node_->get_parameter("planner_id").as_string();
    double goal_pos_tol = node_->get_parameter("goal_position_tolerance").as_double();
    double goal_ori_tol = node_->get_parameter("goal_orientation_tolerance").as_double();
    double goal_joint_tol = node_->get_parameter("goal_joint_tolerance").as_double();
    max_joint_travel_rad_ = node_->get_parameter("max_joint_travel_rad").as_double();

    move_group_.setMaxVelocityScalingFactor(max_velocity_scaling);
    move_group_.setMaxAccelerationScalingFactor(max_acceleration_scaling);

    // Set Planning Pipeline and Planner ID
    // move_group_.setPlanningPipelineId("ompl");
    move_group_.setPlannerId(planner_id);

    // Set Goal Tolerances
    move_group_.setGoalPositionTolerance(goal_pos_tol);
    move_group_.setGoalOrientationTolerance(goal_ori_tol);
    move_group_.setGoalJointTolerance(goal_joint_tol);
    move_group_.setEndEffectorLink("gripper_tcp");

    RCLCPP_INFO(
        node_->get_logger(),
        "Planner ID: %s",
        planner_id.c_str());

    RCLCPP_INFO(
        node_->get_logger(),
        "Planning group: %s",
        planning_group.c_str());

    RCLCPP_INFO(
        node_->get_logger(),
        "Planning frame: %s",
        move_group_.getPlanningFrame().c_str());

    RCLCPP_INFO(
        node_->get_logger(),
        "End-effector link: %s",
        move_group_.getEndEffectorLink().c_str());

    RCLCPP_INFO(
        node_->get_logger(),
        "Max velocity scaling factor: %.2f",
        max_velocity_scaling);

    RCLCPP_INFO(
        node_->get_logger(),
        "Max acceleration scaling factor: %.2f",
        max_acceleration_scaling);

    RCLCPP_INFO(
        node_->get_logger(),
        "Maximum allowed travel of any joint per motion: %.2f rad",
        max_joint_travel_rad_);
  }

  bool UR3MotionInterface::trajectoryHasSafeJointTravel(
      const moveit_msgs::msg::RobotTrajectory &trajectory,
      const std::string &motion_name) const
  {
    const auto &joint_trajectory = trajectory.joint_trajectory;
    if (joint_trajectory.points.empty())
    {
      RCLCPP_ERROR(node_->get_logger(), "%s produced an empty trajectory.", motion_name.c_str());
      return false;
    }

    const std::size_t joint_count = joint_trajectory.joint_names.size();
    std::vector<double> minimum(joint_count, std::numeric_limits<double>::infinity());
    std::vector<double> maximum(joint_count, -std::numeric_limits<double>::infinity());
    for (const auto &point : joint_trajectory.points)
    {
      if (point.positions.size() != joint_count)
      {
        RCLCPP_ERROR(node_->get_logger(), "%s has malformed joint positions.", motion_name.c_str());
        return false;
      }
      for (std::size_t index = 0; index < joint_count; ++index)
      {
        minimum[index] = std::min(minimum[index], point.positions[index]);
        maximum[index] = std::max(maximum[index], point.positions[index]);
      }
    }

    for (std::size_t index = 0; index < joint_count; ++index)
    {
      const double travel = maximum[index] - minimum[index];
      if (travel > max_joint_travel_rad_)
      {
        RCLCPP_ERROR(
            node_->get_logger(),
            "%s rejected before execution: joint '%s' would travel %.3f rad (limit %.3f).",
            motion_name.c_str(), joint_trajectory.joint_names[index].c_str(),
            travel, max_joint_travel_rad_);
        return false;
      }
    }
    return true;
  }

  bool UR3MotionInterface::moveToJointGoal(
      const std::vector<double> &joint_goal)
  {
    last_planning_time_sec_ = 0.0;
    last_execution_time_sec_ = 0.0;
    if (joint_goal.size() != 6)
    {
      RCLCPP_ERROR(
          node_->get_logger(),
          "UR3 joint goal must contain exactly 6 joint values.");
      return false;
    }

    move_group_.setJointValueTarget(joint_goal);

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    const auto planning_start = std::chrono::steady_clock::now();
    bool planning_success = false;
    constexpr int max_pipeline_attempts = 3;
    for (int attempt = 1; attempt <= max_pipeline_attempts; ++attempt)
    {
      move_group_.setStartStateToCurrentState();
      planning_success = static_cast<bool>(move_group_.plan(plan));
      if (planning_success)
      {
        break;
      }
      RCLCPP_WARN(
          node_->get_logger(),
          "Joint planning pipeline attempt %d/%d failed; retrying.",
          attempt, max_pipeline_attempts);
    }
    last_planning_time_sec_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - planning_start).count();

    if (!planning_success)
    {
      RCLCPP_ERROR(node_->get_logger(), "Joint planning failed.");
      return false;
    }

    if (!trajectoryHasSafeJointTravel(plan.trajectory_, "Joint motion"))
    {
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Joint planning succeeded. Executing...");

    const auto execution_start = std::chrono::steady_clock::now();
    const auto execution_result = move_group_.execute(plan);
    last_execution_time_sec_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - execution_start).count();

    if (execution_result != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "Joint execution failed.");
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Joint execution succeeded.");
    return true;
  }

  bool UR3MotionInterface::moveToPoseGoal(
      const geometry_msgs::msg::Pose &target_pose,
      const std::string &end_effector_link)
  {
    last_planning_time_sec_ = 0.0;
    last_execution_time_sec_ = 0.0;
    if (!end_effector_link.empty())
    {
      move_group_.setEndEffectorLink(end_effector_link);
    }

    // Resolve one IK solution near the current state and plan to that concrete
    // joint target. Leaving this as a free pose target allows OMPL to choose a
    // different equivalent IK branch on each request, causing wrist flips.
    move_group_.setStartStateToCurrentState();
    if (!move_group_.setJointValueTarget(
        target_pose, move_group_.getEndEffectorLink()))
    {
      RCLCPP_ERROR(node_->get_logger(), "Seeded IK failed for pose goal.");
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    const auto planning_start = std::chrono::steady_clock::now();
    bool planning_success = false;
    constexpr int max_pipeline_attempts = 3;
    for (int attempt = 1; attempt <= max_pipeline_attempts; ++attempt)
    {
      move_group_.setStartStateToCurrentState();
      planning_success = static_cast<bool>(move_group_.plan(plan));
      if (planning_success)
      {
        break;
      }
      RCLCPP_WARN(
          node_->get_logger(),
          "Pose planning pipeline attempt %d/%d failed; retrying.",
          attempt, max_pipeline_attempts);
    }
    last_planning_time_sec_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - planning_start).count();

    if (!planning_success)
    {
      RCLCPP_ERROR(node_->get_logger(), "Seeded pose planning failed.");
      return false;
    }

    if (!trajectoryHasSafeJointTravel(plan.trajectory_, "Seeded pose motion"))
    {
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Pose planning succeeded. Executing...");

    const auto execution_start = std::chrono::steady_clock::now();
    const auto execution_result = move_group_.execute(plan);
    last_execution_time_sec_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - execution_start).count();

    if (execution_result != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "Pose execution failed.");
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Pose execution succeeded.");
    return true;
  }

  bool UR3MotionInterface::moveHome()
  {
    last_planning_time_sec_ = 0.0;
    last_execution_time_sec_ = 0.0;
    const bool target_set = move_group_.setNamedTarget("home");
    if (!target_set)
    {
      RCLCPP_ERROR(node_->get_logger(), "Failed to set named target: home");
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    const auto planning_start = std::chrono::steady_clock::now();
    const bool planning_success =
        static_cast<bool>(move_group_.plan(plan));
    last_planning_time_sec_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - planning_start).count();

    if (!planning_success)
    {
      RCLCPP_ERROR(node_->get_logger(), "Home planning failed.");
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Home planning succeeded. Executing...");

    const auto execution_start = std::chrono::steady_clock::now();
    const auto execution_result = move_group_.execute(plan);
    last_execution_time_sec_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - execution_start).count();

    if (execution_result != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "Home execution failed.");
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Home execution succeeded.");
    return true;
  }

  bool UR3MotionInterface::moveLinearToPoseGoal(
      const geometry_msgs::msg::Pose &target_pose)
  {
    last_planning_time_sec_ = 0.0;
    last_execution_time_sec_ = 0.0;
    move_group_.setStartStateToCurrentState();

    std::vector<geometry_msgs::msg::Pose> waypoints{target_pose};
    moveit_msgs::msg::RobotTrajectory trajectory;
    constexpr double eef_step = 0.01;
    constexpr double minimum_fraction = 0.98;
    const auto planning_start = std::chrono::steady_clock::now();
    const double fraction = move_group_.computeCartesianPath(
        waypoints, eef_step, 0.0, trajectory, true);
    last_planning_time_sec_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - planning_start).count();

    if (fraction < minimum_fraction)
    {
      RCLCPP_ERROR(
          node_->get_logger(),
          "Cartesian planning covered only %.1f%% of the requested path.",
          fraction * 100.0);
      return false;
    }

    if (!trajectoryHasSafeJointTravel(trajectory, "Cartesian motion"))
    {
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = trajectory;
    RCLCPP_INFO(
        node_->get_logger(),
        "Cartesian planning succeeded (%.1f%%). Executing...",
        fraction * 100.0);

    const auto execution_start = std::chrono::steady_clock::now();
    const auto execution_result = move_group_.execute(plan);
    last_execution_time_sec_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - execution_start).count();
    if (execution_result != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "Cartesian execution failed.");
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Cartesian execution succeeded.");
    return true;
  }

  double UR3MotionInterface::getLastPlanningTime() const
  {
    return last_planning_time_sec_;
  }

  double UR3MotionInterface::getLastExecutionTime() const
  {
    return last_execution_time_sec_;
  }

  void UR3MotionInterface::stop()
  {
    move_group_.stop();
  }

  void UR3MotionInterface::addTableObstacle()
  {
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = move_group_.getPlanningFrame();
    collision_object.id = "table";

    const double table_length = 1.6;
    const double table_width = 2.2;
    const double table_thickness = 0.08;

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[0] = table_length;    // dimension along X axis
    primitive.dimensions[1] = table_width;     // dimension along Y axis
    primitive.dimensions[2] = table_thickness; // thickness along Z axis

    geometry_msgs::msg::Pose box_pose;
    box_pose.orientation.w = 1.0;
    // Project convention: +Y is forward and -X is left. Match the bench that
    // extends mainly forward in ur3_pick_place.sdf.
    box_pose.position.x = 0.0;
    box_pose.position.y = 0.45;
    // Keep the table top at z=0, exactly matching ur3_pick_place.sdf.
    box_pose.position.z = -table_thickness / 2.0;

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(box_pose);
    collision_object.operation = collision_object.ADD;

    RCLCPP_INFO(node_->get_logger(), "Adding the Gazebo-matched table to the planning scene...");
    planning_scene_interface_.applyCollisionObject(collision_object);
  }

  void UR3MotionInterface::calibObjectHeightEyeInHand(double depth_m, double camera_z_mount_offset)
  {
    geometry_msgs::msg::PoseStamped current_pose = move_group_.getCurrentPose();
    double ee_z = current_pose.pose.position.z;
    double object_height = ee_z - camera_z_mount_offset - depth_m;
    RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000, "Current EE Z: %.3f, Depth: %.3f, Object Height: %.3f", ee_z, depth_m, object_height);
  }

} // namespace ur3_moveit_control
