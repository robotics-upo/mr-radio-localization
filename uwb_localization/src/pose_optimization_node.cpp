#include <rclcpp/rclcpp.hpp>

#include "uwb_localization/poseOptimization.hpp"  
#include "uwb_localization/posegraph.hpp"

using uwb_localization::PoseOptimizationNode;

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<PoseOptimizationNode>();
  node->set_parameter(rclcpp::Parameter("use_sim_time", true));

  rclcpp::spin(node);

  // After spin, dump pose graphs and metrics (same behavior as original monolithic main)
  auto agv_pose_graph = node->getAGVPoseGraph();
  auto uav_pose_graph = node->getUAVPoseGraph();

  posegraph::writePoseGraphToCSV(agv_pose_graph, "agv_pose_graph.csv");
  posegraph::writePoseGraphToCSV(uav_pose_graph, "uav_pose_graph.csv");

  node->getMetrics();

  rclcpp::shutdown();
  return 0;
}
