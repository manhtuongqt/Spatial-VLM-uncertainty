cmake_minimum_required(VERSION 3.8)
project(ur3_perception)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/depth_component_gate_v2.py
  scripts/capture_two_camera_demo.py
  scripts/multimodal_image_viewer.py
  scripts/move_camera_to_view.py
  scripts/roborefer_dimension_comparator.py
  scripts/roborefer_demo_recorder.py
  scripts/roborefer_grounder.py
  scripts/roborefer_pilot_capture.py
  scripts/roborefer_pilot_runner.py
  scripts/roborefer_pilot_evaluator.py
  scripts/roborefer_pilot_validate.py
  scripts/spatial_point_utils.py
  scripts/rgbd_object_pose.py
  scripts/perception_demo_runner.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch schemas
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  ament_add_pytest_test(test_projection test/test_projection.py)
  ament_add_pytest_test(test_depth_component_gate_v2 test/test_depth_component_gate_v2.py)
  ament_add_pytest_test(test_roborefer_adapter test/test_roborefer_adapter.py)
  ament_add_pytest_test(
    test_roborefer_dimension_comparator
    test/test_roborefer_dimension_comparator.py
  )
  ament_add_pytest_test(test_roborefer_pilot test/test_roborefer_pilot.py)
  add_launch_test(test/test_perception_smoke.launch.py TIMEOUT 180)
  add_launch_test(test/test_perception_pick_place.launch.py TIMEOUT 360)
endif()

ament_package()
