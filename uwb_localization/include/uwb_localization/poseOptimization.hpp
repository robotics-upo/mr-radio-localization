#pragma once

// ROS
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "uwb_localization/msg/pose_with_covariance_stamped_array.hpp"
#include "uwb_localization/srv/update_point_clouds.hpp"

#include "uwb_localization/utils.hpp"
#include "uwb_localization/posegraph.hpp"
#include "uwb_localization/CostFunctions.hpp"
#include "uwb_localization/manifolds.hpp"

namespace uwb_localization {

/**
 * @brief Pose-graph fusion of AGV & UAV odom, radar (ICP/GICP), and UWB encounters.
 *
 * Declarations only; see optimizerNodeFusion.cpp for definitions.
 */
class PoseOptimizationNode : public rclcpp::Node {
public:
  using UpdatePointClouds = uwb_localization::srv::UpdatePointClouds;
  using PoseWithCovArray = uwb_localization::msg::PoseWithCovarianceStampedArray;

  // Pose graph/optimizer aliases
  using PriorConstraint       = posegraph::PriorConstraint;
  using MeasurementConstraint = posegraph::MeasurementConstraint;
  using MapOfStates           = posegraph::MapOfStates;
  using VectorOfConstraints   = posegraph::VectorOfConstraints;
  using Measurements          = posegraph::Measurements;
  using RadarMeasurements     = posegraph::RadarMeasurements;

  // Point cloud aliases
  using PointT       = pcl::PointXYZ;
  using Cloud        = pcl::PointCloud<PointT>;
  using CloudPtr     = Cloud::Ptr;
  using CloudConstPtr= Cloud::ConstPtr;

public:
  PoseOptimizationNode();

  /// Returns the AGV pose graph as a message array.
  PoseWithCovArray getAGVPoseGraph();

  /// Returns the UAV pose graph as a message array.
  PoseWithCovArray getUAVPoseGraph();

  /// Logs aggregate Ceres metrics (runs, avg, stddev).
  void getMetrics() const;

private:
  //---------------- Parameter management ----------------//
  void declareParams();
  void getParams();

  //---------------- ROS Callbacks ----------------//
  void agvOdomCb(const nav_msgs::msg::Odometry::SharedPtr msg);
  void uavOdomCb(const nav_msgs::msg::Odometry::SharedPtr msg);

  void pclAgvRadarCb(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void pclUavRadarCb(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  void AgvEgoVelCb(const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg);
  void UavEgoVelCb(const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg);

  void optimizedTfCb(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);

  /// Timer: runs optimization.
  void globalOptCb();

  //---------------- Constraints & ICP ----------------//
  /**
   * @brief Add ICP/GICP constraint between scans (already in sensor/body frames).
   */
  bool addPointCloudConstraint(const CloudConstPtr& source_scan,
                               const CloudConstPtr& target_scan,
                               const Sophus::SE3d& T_source_sensor,
                               const Sophus::SE3d& T_target_sensor,
                               double sigma,
                               int icp_type,
                               int id_source,
                               int id_target,
                               VectorOfConstraints& constraints_list,
                               bool send_visualization = false,
                               const Eigen::Matrix4f* initial_guess = nullptr);

  /// Send point clouds to external visualizer service (optional).
  void sendPointCloudServiceRequest(const CloudConstPtr& source_cloud,
                                    const CloudConstPtr& target_cloud,
                                    const CloudConstPtr& aligned_cloud);

  /// Downsampling / (optional) outlier removal.
  CloudPtr preprocessPointCloud(const CloudConstPtr& input_cloud,
                                float meanK = 50.0f,
                                float stdevmulthresh = 2.0f,
                                float leaf_size = 0.05f);

  /// Propagate 6x6 odom covariances to reduced 4x4 (x,y,z,yaw) relative covariance.
  Eigen::Matrix4d computeRelativeOdometryCovariance(const Sophus::SE3d& pose_target,
                                                    const Sophus::SE3d& pose_source,
                                                    const Eigen::Matrix<double,6,6>& cov_target,
                                                    const Eigen::Matrix<double,6,6>& cov_source);

  /// Approximate ICP covariance (reduced 4x4) from correspondences.
  Eigen::Matrix4d computeICPCovariance(const CloudConstPtr& source,
                                       const CloudConstPtr& target,
                                       const Eigen::Matrix4f& transformation,
                                       double sensor_variance = 0.01);

  /// Run ICP/GICP with configured settings.
  bool run_icp(const CloudConstPtr& source_cloud,
               const CloudConstPtr& target_cloud,
               CloudPtr& aligned_cloud,
               Eigen::Matrix4f& transformation,
               const double& pointcloud_sigma,
               double& fitness,
               const int& icp_type,
               Eigen::Matrix<double,6,6>& final_hessian) const;

  /// Build point+normal cloud for point-to-plane ICP.
  pcl::PointCloud<pcl::PointNormal>::Ptr computePointNormalCloud(const CloudConstPtr& cloud,
                                                                 float radius_search = 0.1f) const;

  /// Integrate radar ego-velocity (body frame) over dt to SE(3) delta.
  Sophus::SE3d integrateEgoVelIntoSE3(const Eigen::Vector3d& radar_vel_t,
                                      const Sophus::SE3d& odom_T_s,
                                      const Sophus::SE3d& odom_T_t,
                                      double dt);

  /// Add intra-robot constraints (odom/ICP) to Ceres problem.
  bool addMeasurementConstraints(ceres::Problem& problem,
                                 const VectorOfConstraints& constraints,
                                 MapOfStates& map,
                                 const int& current_id,
                                 const int& max_keyframes,
                                 ceres::LossFunction* loss);

  /// Add inter-robot (encounter) constraints to Ceres problem (single-pair).
  bool addEncounterConstraints(ceres::Problem& problem,
                               const VectorOfConstraints& constraints,
                               MapOfStates& source_map,
                               MapOfStates& target_map,
                               const int& source_current_id,
                               const int& target_current_id,
                               const int& max_keyframes,
                               ceres::LossFunction* loss);

  /// Add inter-robot (encounter) constraints to all nodes in the window.
  bool addEncounterTrajectoryConstraints(ceres::Problem& problem,
                                         MapOfStates& source_map,
                                         MapOfStates& target_map,
                                         const int& source_current_id,
                                         const int& target_current_id,
                                         const int& max_keyframes,
                                         ceres::LossFunction* loss);

  /// Build & solve the pose graph with Ceres; updates states & covariances.
  bool runPosegraphOptimization(MapOfStates& agv_map,
                                MapOfStates& uav_map,
                                const VectorOfConstraints& proprioceptive_constraints_agv,
                                const VectorOfConstraints& extraceptive_constraints_agv,
                                const VectorOfConstraints& proprioceptive_constraints_uav,
                                const VectorOfConstraints& extraceptive_constraints_uav,
                                const VectorOfConstraints& encounter_constraints_uwb,
                                const VectorOfConstraints& encounter_constraints_pointcloud);

private:
  //---------------- Subscriptions ----------------//
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr uav_odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr agv_odom_sub_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_agv_radar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_uav_radar_sub_;

  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr agv_egovel_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr uav_egovel_sub_;

  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr optimized_tf_sub_;

  //---------------- Publishers / Services / Timers ----------------//
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr anchor_agv_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr anchor_uav_publisher_;
  rclcpp::Publisher<PoseWithCovArray>::SharedPtr poses_uav_publisher_;
  rclcpp::Publisher<PoseWithCovArray>::SharedPtr poses_agv_publisher_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Client<UpdatePointClouds>::SharedPtr pcl_visualizer_client_;
  rclcpp::TimerBase::SharedPtr global_optimization_timer_;

  //---------------- Parameters ----------------//
  double opt_timer_rate_{10.0};
  Sophus::SE3d T_uav_radar_, T_agv_radar_;
  Sophus::SE3d T_uav_imu_, T_agv_imu_;
  Sophus::SE3d T_agv_anchor_prior_, T_uav_anchor_prior_;
  double pointcloud_radar_sigma_{0.1};

  std::string odom_topic_agv_, odom_topic_uav_;
  std::string pcl_topic_radar_agv_, pcl_topic_radar_uav_;
  std::string egovel_topic_radar_agv_, egovel_topic_radar_uav_;

  int icp_type_radar_{2};
  bool using_radar_{true}, using_odom_{true};
  double min_traveled_distance_{0.5}, min_traveled_angle_{0.524};
  int max_keyframes_{10}, min_keyframes_{3};
  int radar_history_size_{5};

  //---------------- Measurements & State ----------------//
  CloudPtr uav_radar_cloud_{CloudPtr(new Cloud)};
  CloudPtr agv_radar_cloud_{CloudPtr(new Cloud)};

  bool uwb_transform_available_{false};
  geometry_msgs::msg::PoseWithCovarianceStamped latest_relative_pose_;
  geometry_msgs::msg::TwistWithCovarianceStamped agv_radar_egovel_, uav_radar_egovel_;

  std::deque<Eigen::Vector3d> agv_velocity_buffer_;
  std::deque<Eigen::Vector3d> uav_velocity_buffer_;

  Sophus::SE3d latest_relative_pose_SE3_;
  Eigen::Matrix<double,6,6> latest_relative_pose_cov_;

  State init_state_uav_, init_state_agv_;
  State anchor_node_uav_, anchor_node_agv_;
  PriorConstraint prior_agv_, prior_uav_, prior_anchor_agv_, prior_anchor_uav_;

  MapOfStates uav_map_, agv_map_;
  VectorOfConstraints proprioceptive_constraints_uav_, extraceptive_constraints_uav_;
  VectorOfConstraints proprioceptive_constraints_agv_, extraceptive_constraints_agv_;
  VectorOfConstraints encounter_constraints_uwb_, encounter_constraints_pointcloud_;

  Measurements agv_measurements_, prev_agv_measurements_;
  Measurements uav_measurements_, prev_uav_measurements_;
  std::deque<RadarMeasurements> radar_history_agv_, radar_history_uav_;

  int uav_id_{0}, agv_id_{0};
  bool graph_initialized_{false};

  std::string eliko_frame_id_, uav_frame_id_, global_frame_graph_;
  std::string odom_tf_agv_t_, odom_tf_uav_t_;

  Sophus::SE3d uav_odom_pose_, last_uav_odom_pose_;
  Sophus::SE3d agv_odom_pose_, last_agv_odom_pose_;
  Sophus::SE3d agv_init_pose_, uav_init_pose_;
  Sophus::SE3d uav_radar_odom_pose_, agv_radar_odom_pose_;

  Eigen::Matrix<double,6,6> uav_odom_covariance_, agv_odom_covariance_;

  // Timing
  double measurement_sync_thr_{0.25};
  double last_agv_odom_time_sec_{0.0}, last_uav_odom_time_sec_{0.0};
  double last_agv_radar_time_sec_{0.0}, last_uav_radar_time_sec_{0.0};
  double last_agv_egovel_time_sec_{0.0}, last_uav_egovel_time_sec_{0.0};
  double last_relative_pose_time_sec_{0.0}, last_relative_pose_used_time_sec_{0.0};

  bool last_agv_odom_initialized_{false}, last_uav_odom_initialized_{false};
  bool relative_pose_initialized_{false};

  double uav_translation_{0.0}, agv_translation_{0.0};
  double uav_rotation_{0.0}, agv_rotation_{0.0};

  // Metrics
  int total_solves_{0};
  double total_solver_time_{0.0};
  std::vector<double> solver_times_;
};

} // namespace uwb_localization
