cmake_minimum_required(VERSION 3.8)
project(uwb_localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(small_gicp REQUIRED)

find_package(Boost REQUIRED date_time)
find_package(Qt5OpenGL REQUIRED)

find_package(PCL REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(eliko_messages REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(rosidl_default_generators REQUIRED)

find_package(OpenMP REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/UpdatePointClouds.srv"
  "msg/PoseWithCovarianceStampedArray.msg"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs # Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)


# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(global_opt_node_eliko src/global_opt_node_eliko.cpp)
add_executable(optimizer_node_fusion src/optimizer_node_fusion.cpp)
add_executable(pcl_visualizer src/PCLVisualizer.cpp)
add_executable(uav_odometry_node src/uav_odometry_node.cpp)

include_directories(${CERES_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${pcl_conversions_INCLUDE_DIRS})
include_directories(${pcl_ros_INCLUDE_DIRS})


target_link_libraries(global_opt_node_eliko Ceres::ceres ${CERES_LIBRARIES} Sophus::Sophus)
target_link_libraries(optimizer_node_fusion Ceres::ceres ${PCL_LIBRARIES}  ${pcl_conversions_LIBRARIES} ${pcl_ros_LIBRARIES} Sophus::Sophus OpenMP::OpenMP_CXX)
target_link_libraries(uav_odometry_node Sophus::Sophus)

ament_target_dependencies(global_opt_node_eliko rclcpp std_msgs geometry_msgs nav_msgs tf2_ros tf2_geometry_msgs eliko_messages)
ament_target_dependencies(optimizer_node_fusion rclcpp std_msgs geometry_msgs nav_msgs sensor_msgs tf2_ros tf2_geometry_msgs eliko_messages pcl_ros pcl_conversions)
ament_target_dependencies(uav_odometry_node rclcpp geometry_msgs nav_msgs)


rosidl_target_interfaces(optimizer_node_fusion
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
ament_target_dependencies(pcl_visualizer rclcpp std_msgs sensor_msgs pcl_ros pcl_conversions)
rosidl_target_interfaces(pcl_visualizer
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_include_directories(global_opt_node_eliko PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(global_opt_node_eliko PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_include_directories(optimizer_node_fusion PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(optimizer_node_fusion PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_include_directories(pcl_visualizer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(pcl_visualizer PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_include_directories(uav_odometry_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(uav_odometry_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_compile_options(optimizer_node_fusion PRIVATE ${OpenMP_CXX_FLAGS})

install(TARGETS global_opt_node_eliko optimizer_node_fusion pcl_visualizer uav_odometry_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
launch
DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
config
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
