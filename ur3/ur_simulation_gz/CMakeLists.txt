cmake_minimum_required(VERSION 3.5)
project(ur_simulation_gz)

find_package(ament_cmake REQUIRED)

# Default to off as starting gzsim doesn't shut down correctly at the moment
option(
  UR_SIM_INTEGRATION_TESTS
  "Run ur_simulation_gz integration tests"
  OFF
)

install(DIRECTORY config launch models worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/build_ycb_inventory_v2_gallery.py
  scripts/build_ycb_inventory_v2_main_world.py
  scripts/convert_ycb_glb_to_obj.py
  scripts/test_topic_control.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(launch_testing_ament_cmake)
  find_package(ament_cmake_pytest REQUIRED)

  if(${UR_SIM_INTEGRATION_TESTS})
    add_launch_test(test/test_gz.py
      TIMEOUT
        180
    )
  endif()
endif()

ament_package()
