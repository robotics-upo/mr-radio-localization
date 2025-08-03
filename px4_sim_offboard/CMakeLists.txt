cmake_minimum_required(VERSION 3.8)
project(px4_sim_offboard)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(eliko_messages REQUIRED)
find_package(geometry_msgs REQUIRED)


add_executable(uav_offboard_control src/uav_offboard_control.cpp)
ament_target_dependencies(uav_offboard_control rclcpp px4_msgs std_msgs nav_msgs geometry_msgs)

add_executable(agv_offboard_control src/agv_offboard_control.cpp)
ament_target_dependencies(agv_offboard_control rclcpp px4_msgs std_msgs nav_msgs geometry_msgs eliko_messages)

install(TARGETS
  uav_offboard_control
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  agv_offboard_control
  DESTINATION lib/${PROJECT_NAME}
)

# Install the launch file
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install the config (YAML param) files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
