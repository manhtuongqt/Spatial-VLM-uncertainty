#include <memory>
#include <thread>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "ur3_moveit_control/action/ur3_control.hpp"
#include "ur3_moveit_control/ur3_motion_interface.hpp"

using UR3Control = ur3_moveit_control::action::UR3Control;
using GoalHandleUR3Control = rclcpp_action::ServerGoalHandle<UR3Control>;

class UR3ControlNode : public rclcpp::Node
{
public:
  UR3ControlNode()
  : Node("ur3_control_node", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true))
  {
  }

  void init()
  {
    ur3_motion_ = std::make_shared<ur3_moveit_control::UR3MotionInterface>(shared_from_this(), "ur_manipulator");

    action_server_ = rclcpp_action::create_server<UR3Control>(
      this,
      "ur3_control",
      std::bind(&UR3ControlNode::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&UR3ControlNode::handle_cancel, this, std::placeholders::_1),
      std::bind(&UR3ControlNode::handle_accepted, this, std::placeholders::_1));

    ur3_motion_->addTableObstacle();

    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/camera/depth/image_rect_raw",
      rclcpp::SensorDataQoS(),
      std::bind(&UR3ControlNode::depth_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "UR3 Action Server initialized with Camera Calibration.");
  }

private:
  std::shared_ptr<ur3_moveit_control::UR3MotionInterface> ur3_motion_;
  rclcpp_action::Server<UR3Control>::SharedPtr action_server_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  double camera_z_mount_offset_ = 0.05; // Offset Z từ tay gắp tới camera (m)
  
  std::shared_ptr<GoalHandleUR3Control> active_goal_;
  std::mutex mutex_;
  std::thread execution_thread_;

  void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    if (msg->encoding != "16UC1") {
      RCLCPP_WARN_ONCE(this->get_logger(), "Camera encoding không phải 16UC1. Không thể đọc!");
      return;
    }

    int center_x = msg->width / 2;
    int center_y = msg->height / 2;
    int index = (center_y * msg->step) + (center_x * 2);

    if (static_cast<size_t>(index + 1) >= msg->data.size()) {
      return;
    }

    uint16_t depth_mm = msg->data[index] | (msg->data[index + 1] << 8);
    double depth_m = static_cast<double>(depth_mm) / 1000.0;

    if (depth_m > 0.0) {
      try {
        ur3_motion_->calibObjectHeightEyeInHand(depth_m, camera_z_mount_offset_);
      } catch (const std::exception &e) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Đang đợi TF tay máy để tính toán chiều cao...");
      }
    }
  }

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const UR3Control::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received goal request. Canceling active goal if any.");
    (void)uuid;
    (void)goal;
    
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_goal_ && active_goal_->is_active()) {
        RCLCPP_WARN(this->get_logger(), "Stopping current execution for new goal.");
        ur3_motion_->stop();
    }

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleUR3Control> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
    (void)goal_handle;

    std::lock_guard<std::mutex> lock(mutex_);
    ur3_motion_->stop();
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleUR3Control> goal_handle)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (execution_thread_.joinable()) {
      execution_thread_.join();
    }
    active_goal_ = goal_handle;
    execution_thread_ = std::thread(std::bind(&UR3ControlNode::execute, this, goal_handle));
  }

  void execute(const std::shared_ptr<GoalHandleUR3Control> goal_handle)
  {
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<UR3Control::Feedback>();
    auto result = std::make_shared<UR3Control::Result>();

    feedback->state = "Planning and Executing";
    goal_handle->publish_feedback(feedback);

    bool success = false;
    if (goal->command_type == UR3Control::Goal::MOVE_HOME) {
      success = ur3_motion_->moveHome();
    } else if (goal->command_type == UR3Control::Goal::MOVE_JOINT) {
      success = ur3_motion_->moveToJointGoal(goal->joint_goal.position);
    } else if (goal->command_type == UR3Control::Goal::MOVE_POSE) {
      success = ur3_motion_->moveToPoseGoal(goal->pose_goal.pose);
    } else if (goal->command_type == UR3Control::Goal::MOVE_LINEAR) {
      success = ur3_motion_->moveLinearToPoseGoal(goal->pose_goal.pose);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Invalid command type!");
    }

    result->planning_time_sec = ur3_motion_->getLastPlanningTime();
    result->execution_time_sec = ur3_motion_->getLastExecutionTime();

    if (goal_handle->is_canceling()) {
      result->success = false;
      result->message = "Goal canceled by user or new goal";
      goal_handle->canceled(result);
      RCLCPP_INFO(this->get_logger(), "Goal canceled");
      return;
    }

    if (success) {
      result->success = true;
      result->message = "Goal succeeded";
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Goal succeeded");
    } else {
      result->success = false;
      result->message = "Goal failed or was preempted";
      goal_handle->abort(result);
      RCLCPP_INFO(this->get_logger(), "Goal failed");
    }
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<UR3ControlNode>();
  
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  std::thread spinner([&executor]() {
    executor.spin();
  });

  node->init();

  if (spinner.joinable()) {
    spinner.join();
  }

  rclcpp::shutdown();
  return 0;
}
