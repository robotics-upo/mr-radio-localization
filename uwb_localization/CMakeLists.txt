cmake_minimum_required(VERSION 3.8)
project(uwb_localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(Ceres REQUIRED)
find_package(eliko_messages REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)



# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(optimizer_node src/optimizer_node.cpp)
add_executable(optimizer_node_eliko src/optimizer_node_eliko.cpp)

target_link_libraries(optimizer_node Ceres::ceres)
target_link_libraries(optimizer_node_eliko Ceres::ceres)

ament_target_dependencies(optimizer_node rclcpp std_msgs geometry_msgs tf2_ros)
ament_target_dependencies(optimizer_node_eliko rclcpp std_msgs geometry_msgs tf2_ros tf2_geometry_msgs eliko_messages)

target_include_directories(optimizer_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(optimizer_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_include_directories(optimizer_node_eliko PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(optimizer_node_eliko PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS optimizer_node optimizer_node_eliko
  DESTINATION lib/${PROJECT_NAME})

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
