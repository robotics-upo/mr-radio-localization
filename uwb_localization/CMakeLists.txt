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

# Public include dir for headers (install + build)
set(PUBLIC_INC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

include_directories(${CERES_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${pcl_conversions_INCLUDE_DIRS})
include_directories(${pcl_ros_INCLUDE_DIRS})

# --- add pose optimization library ---
add_library(pose_optimization_lib SHARED src/poseOptimization.cpp)

target_include_directories(pose_optimization_lib PUBLIC ${PUBLIC_INC})
target_compile_features(pose_optimization_lib PUBLIC c_std_99 cxx_std_17)

target_link_libraries(pose_optimization_lib
  Ceres::ceres
  small_gicp::small_gicp
  ${PCL_LIBRARIES}
  ${pcl_conversions_LIBRARIES}
  ${pcl_ros_LIBRARIES}
  Sophus::Sophus
  OpenMP::OpenMP_CXX
)

ament_target_dependencies(pose_optimization_lib
  rclcpp std_msgs geometry_msgs nav_msgs sensor_msgs
  tf2_ros tf2_geometry_msgs eliko_messages pcl_ros pcl_conversions
)

# Ensure generated typesupport is available to the lib
rosidl_target_interfaces(pose_optimization_lib
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# Export the library so ament generates env hooks (LD_LIBRARY_PATH) and proper RPATHs.
install(TARGETS pose_optimization_lib
  EXPORT export_pose_optimization_lib
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
)
ament_export_targets(export_pose_optimization_lib HAS_LIBRARY_TARGET)

# Export headers + dependencies to downstream packages (optional but tidy)
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp std_msgs geometry_msgs nav_msgs sensor_msgs
  tf2_ros tf2_geometry_msgs eliko_messages pcl_ros pcl_conversions
  Ceres Sophus small_gicp PCL
)

# OpenMP compile options (if you want explicit flags)
target_compile_options(pose_optimization_lib PRIVATE ${OpenMP_CXX_FLAGS})


# Pose optimization executable
add_executable(pose_optimization_node
  src/pose_optimization_node.cpp
)
target_include_directories(pose_optimization_node PUBLIC ${PUBLIC_INC})
target_compile_features(pose_optimization_node PUBLIC c_std_99 cxx_std_17)
target_link_libraries(pose_optimization_node pose_optimization_lib)
ament_target_dependencies(pose_optimization_node
  rclcpp std_msgs geometry_msgs nav_msgs sensor_msgs
  tf2_ros tf2_geometry_msgs eliko_messages pcl_ros pcl_conversions
)
rosidl_target_interfaces(pose_optimization_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# Make the exe look in its own folder for the shared lib at runtime
# (install dir is .../lib/uwb_localization)
set_target_properties(pose_optimization_node PROPERTIES
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)

# Relative localization executable
add_executable(global_opt_node_eliko src/global_opt_node_eliko.cpp)
target_include_directories(global_opt_node_eliko PUBLIC ${PUBLIC_INC})
target_compile_features(global_opt_node_eliko PUBLIC c_std_99 cxx_std_17)
target_link_libraries(global_opt_node_eliko Ceres::ceres Sophus::Sophus)
ament_target_dependencies(global_opt_node_eliko
  rclcpp std_msgs geometry_msgs nav_msgs tf2_ros tf2_geometry_msgs eliko_messages
)

#PCL Visualizer executable
add_executable(pcl_visualizer src/PCLVisualizer.cpp)
target_include_directories(pcl_visualizer PUBLIC ${PUBLIC_INC})
target_compile_features(pcl_visualizer PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(pcl_visualizer rclcpp std_msgs sensor_msgs pcl_ros pcl_conversions)
rosidl_target_interfaces(pcl_visualizer ${PROJECT_NAME} "rosidl_typesupport_cpp")

#UAV and AGV odometry nodes
add_executable(uav_odometry_node src/uav_odometry_node.cpp)
target_include_directories(uav_odometry_node PUBLIC ${PUBLIC_INC})
target_compile_features(uav_odometry_node PUBLIC c_std_99 cxx_std_17)
target_link_libraries(uav_odometry_node Sophus::Sophus)
ament_target_dependencies(uav_odometry_node rclcpp geometry_msgs nav_msgs)

add_executable(agv_odometry_node src/agv_odometry_node.cpp)
target_include_directories(agv_odometry_node PUBLIC ${PUBLIC_INC})
target_compile_features(agv_odometry_node PUBLIC c_std_99 cxx_std_17)
target_link_libraries(agv_odometry_node Sophus::Sophus)
ament_target_dependencies(agv_odometry_node rclcpp geometry_msgs sensor_msgs nav_msgs)


install(TARGETS
  pose_optimization_node
  global_opt_node_eliko
  pcl_visualizer
  uav_odometry_node
  agv_odometry_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY bags DESTINATION share/${PROJECT_NAME})


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
