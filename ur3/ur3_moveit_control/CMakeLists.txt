cmake_minimum_required(VERSION 3.8)
project(ur3_moveit_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/UR3Control.action"
  DEPENDENCIES geometry_msgs sensor_msgs action_msgs
)

include_directories(include)

add_library(ur3_motion_interface
  src/ur3_motion_interface.cpp
)

ament_target_dependencies(ur3_motion_interface
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
)

add_executable(ur3_control_node
  src/ur3_control_node.cpp
)

target_link_libraries(ur3_control_node
  ur3_motion_interface
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(ur3_control_node "${cpp_typesupport_target}")

ament_target_dependencies(ur3_control_node
  rclcpp
  rclcpp_action
  moveit_ros_planning_interface
  geometry_msgs
  sensor_msgs
)

install(
  TARGETS
    ur3_motion_interface
    ur3_control_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config launch urdf rviz
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

install(
  PROGRAMS scripts/fixed_pick_place.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fixed_pick_place
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  ament_add_pytest_test(test_pose_generation test/test_pose_generation.py)
  ament_add_pytest_test(test_d435i_description test/test_d435i_description.py)
  add_launch_test(test/test_baseline_smoke.launch.py TIMEOUT 300)
endif()

ament_package()
