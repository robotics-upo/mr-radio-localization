#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "geometry_msgs/msg/quaternion_stamped.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include <nav_msgs/msg/odometry.hpp>

#include "eliko_messages/msg/distances_list.hpp"
#include "eliko_messages/msg/anchor_coords_list.hpp"
#include "eliko_messages/msg/tag_coords_list.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/manifold.h>

#include <sophus/se3.hpp>   // Include Sophus for SE(3) operations

#include <Eigen/Core>
#include <vector>
#include <sstream>
#include <deque>
#include <Eigen/Dense>
#include <algorithm>
#include <random>

#include <utility>
#include <unordered_map>
#include <chrono>

#include "uwb_localization/utils.hpp"
#include "uwb_localization/CostFunctions.hpp"
#include "uwb_localization/manifolds.hpp"

using namespace uwb_localization;


class ElikoGlobalOptNode : public rclcpp::Node {

public:

    ElikoGlobalOptNode() : Node("eliko_global_opt_node") {

    declareParams();

    getParams();

    //Subscribe to distances publisher
    eliko_distances_sub_ = this->create_subscription<eliko_messages::msg::DistancesList>(
                "/eliko/Distances", 10, std::bind(&ElikoGlobalOptNode::distancesCoordsCb, this, std::placeholders::_1));

    rclcpp::SensorDataQoS qos; // Use a QoS profile compatible with sensor data

    if(debug_){
        dll_agv_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/dll_node_arco/pose_estimation", qos, std::bind(&ElikoGlobalOptNode::agvDllCb, this, std::placeholders::_1));

        dll_uav_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/dll_node/pose_estimation", qos, std::bind(&ElikoGlobalOptNode::uavDllCb, this, std::placeholders::_1));

        range_pub_ = this->create_publisher<std_msgs::msg::Float32>("eliko_optimization_node/dll_odom_range", 10);
    }
    else{
        agv_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_agv_, qos, std::bind(&ElikoGlobalOptNode::agvOdomCb, this, std::placeholders::_1));
        
        uav_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_uav_, qos, std::bind(&ElikoGlobalOptNode::uavOdomCb, this, std::placeholders::_1));
    }
    
    // Create publisher/broadcaster for optimized transformation
    tf_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("eliko_optimization_node/optimized_T", 10);
    tf_ransac_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("eliko_optimization_node/ransac_optimized_T", 10);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    anchor_positions_odom_ = anchor_positions_;
    tag_positions_odom_ = tag_positions_;

    agv_odom_frame_id_ = "agv/odom"; //frame of the eliko system-> arco/eliko, for simulation use "agv_gt" for ground truth, "agv_odom" for odometry w/ errors
    uav_odom_frame_id_ = "uav/odom"; //frame of the UAV system-> uav_gt for ground truth, "uav_odom" for odometry w/ errors
    agv_body_frame_id_ = "agv/base_link";
    uav_body_frame_id_ = "uav/base_link";
    uav_opt_frame_id_ = "uav_opt"; 
    agv_opt_frame_id_ = "agv_opt"; 

    //Initial values for state
    // init_state_.state = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
    init_state_.covariance = Eigen::Matrix4d::Identity(); //
    init_state_.roll = 0.0;
    init_state_.pitch = 0.0;
    init_state_.pose = buildTransformationSE3(init_state_.roll, init_state_.pitch, init_state_.state);
    init_state_.timestamp = this->get_clock()->now();

    opt_state_ = init_state_;

    last_agv_odom_initialized_ = false;
    last_uav_odom_initialized_ = false;

    uav_delta_translation_ = agv_delta_translation_ = uav_delta_rotation_ = agv_delta_rotation_ = 0.0;
    uav_total_translation_ = agv_total_translation_ = uav_total_rotation_ = agv_total_rotation_ = 0.0;

    total_solves_ = 0;
    total_solver_time_ = 0.0;

    // Calculate the optimization timer period from opt_timer_rate_ (Hz).
    double opt_timer_period_s = 1.0 / opt_timer_rate_;
    global_optimization_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(int(opt_timer_period_s*1000)), std::bind(&ElikoGlobalOptNode::globalOptCb, this));
    
    RCLCPP_INFO(this->get_logger(), "Eliko Optimization Node initialized.");
  }

  void getMetrics() const
  {
    if (total_solves_ > 0)
    {
      double avg = total_solver_time_ / static_cast<double>(total_solves_);
      RCLCPP_INFO(this->get_logger(),
                  "===== Ceres Solver Metrics =====\n"
                  "Total runs: %d\n"
                  "Total time: %.6f s\n"
                  "Average time per solve: %.6f s",
                  total_solves_, total_solver_time_, avg);
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "No solver runs have completed yet.");
    }
  }

private:

    void declareParams() {

         // Declare parameters with their default values.
        // Topics and update rate
        this->declare_parameter<std::string>("odom_topic_agv", "/arco/idmind_motors/odom");
        this->declare_parameter<std::string>("odom_topic_uav", "/uav/odom");
        this->declare_parameter<double>("opt_timer_rate_", 10.0);  // in Hz

        // Optimization settings
        this->declare_parameter<int>("min_measurements", 50);
        this->declare_parameter<double>("min_traveled_distance", 0.5);   // in meters
        this->declare_parameter<double>("min_traveled_angle", 0.524);      // in radians (0.524 rad ~ 30 deg)
        this->declare_parameter<double>("max_traveled_distance", 10.0);    // in meters
        this->declare_parameter<double>("measurement_stdev", 0.1);         // in meters
        this->declare_parameter<bool>("use_ransac", true);

        this->declare_parameter<double>("odom_error_position", 2.0);
        this->declare_parameter<double>("odom_error_angle", 2.0);
        this->declare_parameter<bool>("debug", true);
        this->declare_parameter<bool>("use_prior", true);

        this->declare_parameter<bool>("moving_average", true);
        this->declare_parameter<int>("moving_average_max_samples", 10);

        //initial solution
        this->declare_parameter<std::vector<double>>("init_solution",
                        std::vector<double>{0.0, 0.0, 0.0, 0.0});

        //Initial local poses of robots
        this->declare_parameter<std::vector<double>>("agv_init_pose",
                        std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

        this->declare_parameter<std::vector<double>>("uav_init_pose",
                        std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

        // Declare the anchors parameters.
        this->declare_parameter<std::string>("anchors.a1.id", "0x0009D6");
        this->declare_parameter<std::vector<double>>("anchors.a1.position", std::vector<double>{-0.32, 0.3, 0.875});
        this->declare_parameter<std::string>("anchors.a2.id", "0x0009E5");
        this->declare_parameter<std::vector<double>>("anchors.a2.position", std::vector<double>{0.32, -0.3, 0.875});
        this->declare_parameter<std::string>("anchors.a3.id", "0x0016FA");
        this->declare_parameter<std::vector<double>>("anchors.a3.position", std::vector<double>{0.32, 0.3, 0.33});
        this->declare_parameter<std::string>("anchors.a4.id", "0x0016CF");
        this->declare_parameter<std::vector<double>>("anchors.a4.position", std::vector<double>{-0.32, -0.3, 0.33});

        // Declare the tags parameters.
        this->declare_parameter<std::string>("tags.t1.id", "0x001155");
        this->declare_parameter<std::vector<double>>("tags.t1.position", std::vector<double>{-0.24, -0.24, -0.06});
        this->declare_parameter<std::string>("tags.t2.id", "0x001397");
        this->declare_parameter<std::vector<double>>("tags.t2.position", std::vector<double>{0.24, 0.24, -0.06});
    }

    void getParams() {

        this->get_parameter("odom_topic_agv", odom_topic_agv_);
        this->get_parameter("odom_topic_uav", odom_topic_uav_);
        this->get_parameter("opt_timer_rate_", opt_timer_rate_);
        this->get_parameter("min_measurements", min_measurements_);
        this->get_parameter("min_traveled_distance", min_traveled_distance_);
        // Convert angle from degrees to radians.
        this->get_parameter("min_traveled_angle", min_traveled_angle_);
        this->get_parameter("max_traveled_distance", max_traveled_distance_);
        this->get_parameter("measurement_stdev", measurement_stdev_);
        this->get_parameter("use_ransac", use_ransac_);

        this->get_parameter("odom_error_position", odom_error_distance_);
        this->get_parameter("odom_error_angle", odom_error_angle_);
        this->get_parameter("debug", debug_);
        this->get_parameter("use_prior", use_prior_);
        std::vector<double> init_state, agv_init_pose, uav_init_pose;
        this->get_parameter("init_solution", init_state);
        this->get_parameter("agv_init_pose", agv_init_pose);
        this->get_parameter("uav_init_pose", uav_init_pose);

        this->get_parameter("moving_average", moving_average_);
        this->get_parameter("moving_average_max_samples", moving_average_max_samples_);

        //Set initial solution
        init_state_.state = Eigen::Vector4d(init_state[0], init_state[1], init_state[2], init_state[3]);

        //Initialize odometry
        agv_init_pose_ = buildTransformationSE3(agv_init_pose[3], agv_init_pose[4],
                                  Eigen::Vector4d(agv_init_pose[0], agv_init_pose[1], agv_init_pose[2], agv_init_pose[5]));

        uav_init_pose_ = buildTransformationSE3(uav_init_pose[3], uav_init_pose[4],
                                  Eigen::Vector4d(uav_init_pose[0], uav_init_pose[1], uav_init_pose[2], uav_init_pose[5]));

        // Log the read parameters.
        RCLCPP_INFO(this->get_logger(), "Parameters read:");
        RCLCPP_INFO(this->get_logger(), "  odom_topic_agv: %s", odom_topic_agv_.c_str());
        RCLCPP_INFO(this->get_logger(), "  odom_topic_uav: %s", odom_topic_uav_.c_str());
        RCLCPP_INFO(this->get_logger(), "  opt_timer_rate_: %f Hz", opt_timer_rate_);
        RCLCPP_INFO(this->get_logger(), "  min_measurements: %zu", min_measurements_);
        RCLCPP_INFO(this->get_logger(), "  min_traveled_distance: %f m", min_traveled_distance_);
        RCLCPP_INFO(this->get_logger(), "  min_traveled_angle: %f rad", min_traveled_angle_);
        RCLCPP_INFO(this->get_logger(), "  max_traveled_distance: %f m", max_traveled_distance_);
        RCLCPP_INFO(this->get_logger(), "  measurement_stdev: %f m", measurement_stdev_);
        RCLCPP_INFO(this->get_logger(), "  odom_error_position: %f", odom_error_distance_);
        RCLCPP_INFO(this->get_logger(), "  odom_error_angle: %f", odom_error_angle_);
        RCLCPP_INFO(this->get_logger(), "  moving_average: %s", moving_average_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  use_prior: %s", use_prior_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  use_ransac: %s", use_ransac_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  moving_average_max_samples: %zu", moving_average_max_samples_);

        RCLCPP_INFO(this->get_logger(), "  Init solution: [%f, %f, %f, %f]", init_state_.state[0], init_state_.state[1], init_state_.state[2], init_state_.state[3]);

        RCLCPP_INFO(this->get_logger(), "AGV initial pose:\n");
        logTransformationMatrix(agv_init_pose_.matrix(), this->get_logger());

        RCLCPP_INFO(this->get_logger(), "UAV initial pose:\n");
        logTransformationMatrix(uav_init_pose_.matrix(), this->get_logger());
    
        // Initialize anchors from parameters.
        std::string a1_id, a2_id, a3_id, a4_id;
        std::vector<double> a1_pos, a2_pos, a3_pos, a4_pos;
        
        {
            // Anchor A1
            this->get_parameter("anchors.a1.id", a1_id);
            this->get_parameter("anchors.a1.position", a1_pos);
            anchor_positions_[a1_id] = Eigen::Vector3d(a1_pos[0], a1_pos[1], a1_pos[2]);
    
            this->get_parameter("anchors.a2.id", a2_id);
            this->get_parameter("anchors.a2.position", a2_pos);
            anchor_positions_[a2_id] = Eigen::Vector3d(a2_pos[0], a2_pos[1], a2_pos[2]);
    
            this->get_parameter("anchors.a3.id", a3_id);
            this->get_parameter("anchors.a3.position", a3_pos);
            anchor_positions_[a3_id] = Eigen::Vector3d(a3_pos[0], a3_pos[1], a3_pos[2]);
    
            this->get_parameter("anchors.a4.id", a4_id);
            this->get_parameter("anchors.a4.position", a4_pos);
            anchor_positions_[a4_id] = Eigen::Vector3d(a4_pos[0], a4_pos[1], a4_pos[2]);
    
        }
    
        // Initialize anchors from parameters.
        std::string t1_id, t2_id;
        std::vector<double> t1_pos, t2_pos;
    
        // Initialize tags from parameters.
        {
            this->get_parameter("tags.t1.id", t1_id);
            this->get_parameter("tags.t1.position", t1_pos);
            tag_positions_[t1_id] = Eigen::Vector3d(t1_pos[0], t1_pos[1], t1_pos[2]);
    
            this->get_parameter("tags.t2.id", t2_id);
            this->get_parameter("tags.t2.position", t2_pos);
            tag_positions_[t2_id] = Eigen::Vector3d(t2_pos[0], t2_pos[1], t2_pos[2]);
        }

 
        // Log anchors.
        for (const auto& kv : anchor_positions_) {
            const auto &id = kv.first;
            const auto &pos = kv.second;
            RCLCPP_INFO(this->get_logger(), "  Anchor '%s': [%f, %f, %f]", id.c_str(), pos.x(), pos.y(), pos.z());
        }
    
        // Log tags.
        for (const auto& kv : tag_positions_) {
            const auto &id = kv.first;
            const auto &pos = kv.second;
            RCLCPP_INFO(this->get_logger(), "  Tag '%s': [%f, %f, %f]", id.c_str(), pos.x(), pos.y(), pos.z());
        }
   }

    void uavOdomCb(const nav_msgs::msg::Odometry::SharedPtr msg) {
        
        //******************Method 1: Just extract pose from odom*********************//

        Eigen::Quaterniond q(
            msg->pose.pose.orientation.w,
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z
        );
        
        // Create a 3D translation vector.
        Eigen::Vector3d t(
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z
        );
        
        Sophus::SE3d current_pose(q, t);
        uav_odom_pose_ = uav_init_pose_ * current_pose;

        // If this is the first message, simply store it and return.
        if (!last_uav_odom_initialized_) {
            last_uav_odom_time_sec_ = rclcpp::Time(msg->header.stamp).seconds();
            last_uav_odom_initialized_ = true;
            last_uav_odom_pose_ = uav_odom_pose_;

            return;
        }


        // Compute the incremental movement of the UAV tag sensors.
        for (const auto& kv : tag_positions_) {
            const std::string& tag_id = kv.first;
            const Eigen::Vector3d& tag_offset = kv.second;  // mounting offset in UAV body frame
            Eigen::Vector3d current_tag_pos = uav_odom_pose_ * tag_offset;
            tag_positions_odom_[tag_id] = current_tag_pos;
        }
        
        // Update displacement, via logarithmic map
        Eigen::Matrix<double, 6, 1> log = (last_uav_odom_pose_.inverse() * uav_odom_pose_).log();

        Eigen::Vector3d delta_translation = log.head<3>(); // first three elements
        Eigen::Vector3d delta_rotation    = log.tail<3>(); // last three elements

        uav_delta_translation_+=delta_translation.norm();
        uav_total_translation_+=delta_translation.norm();
        uav_delta_rotation_+=delta_rotation.norm();  
        uav_total_rotation_+=delta_rotation.norm(); 
            
        //Read covariance
        Eigen::Matrix<double,6,6> cov;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                cov(i, j) = msg->pose.covariance[i * 6 + j];
            }
        }
        uav_odom_covariance_ = cov;
        last_uav_odom_pose_ = uav_odom_pose_;
        last_uav_odom_time_sec_ = rclcpp::Time(msg->header.stamp).seconds();
    
    }


    void agvOdomCb(const nav_msgs::msg::Odometry::SharedPtr msg) {
        
        //******************Method 1: Just extract pose from odom*********************//

        Eigen::Quaterniond q(
            msg->pose.pose.orientation.w,
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z
        );
        
        // Create a 3D translation vector.
        Eigen::Vector3d t(
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z
        );
        
        Sophus::SE3d current_pose(q, t);
        agv_odom_pose_ = agv_init_pose_ * current_pose;

        // If this is the first message, simply store it and return.
        if (!last_agv_odom_initialized_) {
            last_agv_odom_time_sec_ = rclcpp::Time(msg->header.stamp).seconds();
            last_agv_odom_initialized_ = true;
            last_agv_odom_pose_ = agv_odom_pose_;

            return;
        }


        // Update anchor sensor positions.
        for (const auto& kv : anchor_positions_) {
            const std::string& anchor_id = kv.first;
            const Eigen::Vector3d& anchor_offset = kv.second;  // mounting offset in AGV body frame
            Eigen::Vector3d current_anchor_pos = agv_odom_pose_ * anchor_offset;
            anchor_positions_odom_[anchor_id] = current_anchor_pos;
        }
        
        // Update displacement, via logarithmic map

        Eigen::Matrix<double, 6, 1> log = (last_agv_odom_pose_.inverse() * agv_odom_pose_).log();

        Eigen::Vector3d delta_translation = log.head<3>(); // first three elements
        Eigen::Vector3d delta_rotation    = log.tail<3>(); // last three elements  
        
        agv_delta_translation_+= delta_translation.norm();
        agv_delta_rotation_ += delta_rotation.norm();
        agv_total_translation_+=delta_translation.norm();
        agv_total_rotation_+=delta_rotation.norm();

        //Read covariance
        Eigen::Matrix<double,6,6> cov;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                cov(i, j) = msg->pose.covariance[i * 6 + j];
            }
        }
        agv_odom_covariance_ = cov;
        last_agv_odom_pose_ = agv_odom_pose_;
        last_agv_odom_time_sec_ = rclcpp::Time(msg->header.stamp).seconds();
    
    }

    /*********DLL callbacks: used instead of odometry when you want accurate poses **********/

    void agvDllCb(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {

        Eigen::Quaterniond q(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z
        );
        Eigen::Vector3d t(
            msg->pose.position.x,
            msg->pose.position.y,
            msg->pose.position.z
        );
        agv_odom_pose_ = Sophus::SE3d(q, t);

        if (!last_agv_odom_initialized_) {
            last_agv_odom_initialized_ = true;
            last_agv_odom_pose_ = agv_odom_pose_;
            last_agv_odom_time_sec_ = rclcpp::Time(msg->header.stamp).seconds();
            return;
        }

        for (const auto& kv : anchor_positions_) {
            const std::string& id = kv.first;
            const Eigen::Vector3d& offset = kv.second;
            anchor_positions_odom_[id] = agv_odom_pose_ * offset;
        }

        auto log = (last_agv_odom_pose_.inverse() * agv_odom_pose_).log();
        agv_delta_translation_ += log.head<3>().norm();
        agv_total_translation_ += log.head<3>().norm();
        agv_delta_rotation_ += log.tail<3>().norm();
        agv_total_rotation_ += log.tail<3>().norm();

        // No covariance in PoseStamped
        agv_odom_covariance_ = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;  // near-zero covariance

        last_agv_odom_pose_ = agv_odom_pose_; 
        last_agv_odom_time_sec_ = rclcpp::Time(msg->header.stamp).seconds();
    }

    void uavDllCb(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        Eigen::Quaterniond q(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z
        );
        Eigen::Vector3d t(
            msg->pose.position.x,
            msg->pose.position.y,
            msg->pose.position.z
        );
        uav_odom_pose_ = Sophus::SE3d(q, t);

        if (!last_uav_odom_initialized_) {
            last_uav_odom_initialized_ = true;
            last_uav_odom_pose_ = uav_odom_pose_;
            last_uav_odom_time_sec_ = rclcpp::Time(msg->header.stamp).seconds();

            return;
        }

        for (const auto& kv : tag_positions_) {
            const std::string& id = kv.first;
            const Eigen::Vector3d& offset = kv.second;
            tag_positions_odom_[id] = uav_odom_pose_ * offset;
        }

        auto log = (last_uav_odom_pose_.inverse() * uav_odom_pose_).log();
        uav_delta_translation_ += log.head<3>().norm();
        uav_total_translation_ += log.head<3>().norm();
        uav_delta_rotation_ += log.tail<3>().norm();
        uav_total_rotation_ += log.tail<3>().norm();

        uav_odom_covariance_ = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;

        last_uav_odom_pose_ = uav_odom_pose_;
        last_uav_odom_time_sec_ = rclcpp::Time(msg->header.stamp).seconds();
    }



    void distancesCoordsCb(const eliko_messages::msg::DistancesList::SharedPtr msg) {

        // Check if both odometry messages are initialized
        if (!last_agv_odom_initialized_ || !last_uav_odom_initialized_) {
            RCLCPP_WARN(this->get_logger(), "[Eliko global_opt node] Odometry not yet initialized. Skipping UWB data.");
            return;
        }

        // Synchronization check: see if we have recent enough odometry
        double uwb_time_sec = rclcpp::Time(msg->header.stamp).seconds();
        // Time difference checks
        double dt_agv = std::abs(uwb_time_sec - last_agv_odom_time_sec_);
        double dt_uav = std::abs(uwb_time_sec - last_uav_odom_time_sec_);

        double max_age_sec = 1.0;
        if (dt_agv > max_age_sec || dt_uav > max_age_sec) {
            RCLCPP_INFO(this->get_logger(),
            "[Eliko global_opt node] Skipping UWB data due to stale odometry.\n"
            "  - UWB stamp   : %.9f\n"
            "  - AGV stamp   : %.9f (Δ = %.3fs)\n"
            "  - UAV stamp   : %.9f (Δ = %.3fs)\n"
            "  - Max allowed : %.3fs",
            uwb_time_sec,
            last_agv_odom_time_sec_, dt_agv,
            last_uav_odom_time_sec_, dt_uav,
            max_age_sec);
            return;
        }

        if (!global_measurements_.empty()) {
            const auto& last = global_measurements_.back();

            bool uav_static = std::abs(uav_total_translation_ - last.uav_cumulative_distance) < 1e-2;
            bool agv_static = std::abs(agv_total_translation_ - last.agv_cumulative_distance) < 1e-2;

            if (uav_static && agv_static) {
                RCLCPP_DEBUG(this->get_logger(), "[Eliko global_opt node] Skipping UWB data: both AGV and UAV are static.");
                return;
            }
        }

        for (const auto& distance_msg : msg->anchor_distances) {
            UWBMeasurement measurement;
            measurement.timestamp = msg->header.stamp;
            measurement.tag_id = distance_msg.tag_sn;
            measurement.anchor_id = distance_msg.anchor_sn;
            measurement.distance = distance_msg.distance / 100.0; // Convert to meters

            measurement.uav_cumulative_distance = uav_total_translation_;
            measurement.agv_cumulative_distance = agv_total_translation_;

            //Positions of tags and anchors according to odometry
            measurement.tag_odom_pose = tag_positions_odom_[measurement.tag_id];
            measurement.anchor_odom_pose = anchor_positions_odom_[measurement.anchor_id];

            global_measurements_.push_back(measurement);
        }

        if(debug_){
            // Compute the distance between AGV and UAV poses
            Eigen::Vector3d agv_pos = agv_odom_pose_.translation();
            Eigen::Vector3d uav_pos = uav_odom_pose_.translation();
            float range = 100.0 * static_cast<float>((agv_pos - uav_pos).norm());

            std_msgs::msg::Float32 range_msg;
            range_msg.data = range;

            range_pub_->publish(range_msg);
        }

        // RCLCPP_INFO(this->get_logger(), "[Eliko global node] Added %ld measurements. Total size: %ld", 
        //             msg->anchor_distances.size(), global_measurements_.size());
    }


    void globalOptCb() {

        rclcpp::Time current_time = this->get_clock()->now();

        /*Check the robots have moved enough between optimizations*/
        bool uav_enough_movement = uav_delta_translation_ >= min_traveled_distance_ || uav_delta_rotation_ >= min_traveled_angle_;
        bool agv_enough_movement = agv_delta_translation_ >= min_traveled_distance_ || agv_delta_rotation_ >= min_traveled_angle_;
        if (!uav_enough_movement || !agv_enough_movement) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "[Eliko global_opt node] Insufficient movement UAV = [%.2fm %.2fº], AGV= [%.2fm %.2fº]. Skipping optimization.", uav_delta_translation_, uav_delta_rotation_ * 180.0/M_PI, agv_delta_translation_, agv_delta_rotation_ * 180.0/M_PI);
            return;
        }

        // // Remove very old measurements
        // while (!global_measurements_.empty() &&
        //       (current_time - global_measurements_.front().timestamp).seconds() > 60.0) {
        //     global_measurements_.pop_front();
        // }

        //*****************Displace window based on distance traveled ****************************/
        while (!global_measurements_.empty()) {
            // Get the oldest measurement.
            const UWBMeasurement &oldest = global_measurements_.front();
            const UWBMeasurement &latest = global_measurements_.back();
             // Compute the net displacement over the window using the stored cumulative distances.
            double uav_disp = latest.uav_cumulative_distance - oldest.uav_cumulative_distance;
            double agv_disp = latest.agv_cumulative_distance - oldest.agv_cumulative_distance;

            if(uav_disp < 1.0 || agv_disp < 1.0){
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Insufficient displacement in window. UAV displacement = [%.2f], AGV displacement = [%.2f]",
                uav_disp, agv_disp);
                return;
            }
            // Prune the oldest measurement if the displacement exceeds the threshold.
            else if (uav_disp > max_traveled_distance_ || agv_disp > max_traveled_distance_) {
                global_measurements_.pop_front();
            } else {
                break;
            }
        }

        // Then, check if there are enough measurements remaining.
        if (global_measurements_.size() < min_measurements_) {
            RCLCPP_WARN(this->get_logger(), "[Eliko global_opt node] Got %d range measurements: not enough data to run optimization.", global_measurements_.size());
            return;
        }

        // Check for measurements from both tags
        std::unordered_set<std::string> observed_tags;
        for (const auto& measurement : global_measurements_) {
            observed_tags.insert(measurement.tag_id);
        }
        if (observed_tags.size() < 2) {
            RCLCPP_WARN(this->get_logger(), "[Eliko global_opt node] Only one tag available. YAW IS NOT RELIABLE.");
        }

        RCLCPP_WARN(this->get_logger(), "[Eliko global_opt node] Movement UAV = [%.2fm %.2fº], AGV= [%.2fm %.2fº].", uav_delta_translation_, uav_delta_rotation_ * 180.0/M_PI, agv_delta_translation_, agv_delta_rotation_ * 180.0/M_PI);

        // Compute odometry covariance based on distance increments from step to step 
        Eigen::Matrix4d predicted_motion_covariance = Eigen::Matrix4d::Zero();
        double predicted_drift_translation = (odom_error_distance_ / 100.0) * (uav_total_translation_ + agv_total_translation_);
        double predicted_drift_rotation = (odom_error_angle_ / 100.0) * (uav_total_rotation_ + agv_total_rotation_);
        predicted_motion_covariance(0,0) = std::pow(predicted_drift_translation, 2.0);
        predicted_motion_covariance(1,1) = std::pow(predicted_drift_translation, 2.0);
        predicted_motion_covariance(2,2) = std::pow(predicted_drift_translation, 2.0);
        predicted_motion_covariance(3,3) = std::pow(predicted_drift_rotation, 2.0);

        
        //*****************Calculate initial solution ****************************/

        Eigen::MatrixXd M;
        Eigen::VectorXd b;
        bool init_prior = false;
        if(use_prior_){
            init_prior = solveMartelLinearSystem(global_measurements_, M, b, init_state_, use_ransac_);
            if(init_prior){
                geometry_msgs::msg::PoseWithCovarianceStamped msg = buildPoseMsg(init_state_.pose, init_state_.covariance + predicted_motion_covariance, current_time, uav_odom_frame_id_);
                tf_ransac_publisher_->publish(msg);
            }
        }
        //*****************Nonlinear WLS refinement with all measurements ****************************/

        //Update odom for next optimization
        uav_delta_translation_ = agv_delta_translation_ = uav_delta_rotation_ = agv_delta_rotation_ = 0.0;

        //Optimize    
        RCLCPP_INFO(this->get_logger(), "[Eliko global_opt node] Optimizing trajectory of %ld measurements", global_measurements_.size());
        //Update transforms after convergence
        if(runOptimization(current_time, predicted_motion_covariance, init_prior)){
            
            if(moving_average_){
                // /*Run moving average*/
                auto smoothed_state = movingAverage(opt_state_, moving_average_max_samples_);
                //Update for initial estimation of following step
                opt_state_.state = smoothed_state;
            }

            opt_state_.pose = buildTransformationSE3(opt_state_.roll, opt_state_.pitch, opt_state_.state);
            
            geometry_msgs::msg::PoseWithCovarianceStamped msg = buildPoseMsg(opt_state_.pose, opt_state_.covariance + predicted_motion_covariance, opt_state_.timestamp, uav_odom_frame_id_);
            tf_publisher_->publish(msg);

        }

        else{
            RCLCPP_INFO(this->get_logger(), "[Eliko global_opt node] Optimizer did not converge");
        }

    }

    double predictDistanceFromMartel(const Eigen::VectorXd& x, const UWBMeasurement& meas)
    {
        double u = x[0], v = x[1], w = x[2];
        double cos_alpha = x[3], sin_alpha = x[4];

        Eigen::Matrix3d R;
        R << cos_alpha, -sin_alpha, 0,
            sin_alpha,  cos_alpha, 0,
                0,          0,    1;

        Eigen::Vector3d t(u, v, w);
        Eigen::Vector3d transformed = R * meas.tag_odom_pose + t;

        return (transformed - meas.anchor_odom_pose).norm();
    }

    bool buildMartelSystem(const std::vector<UWBMeasurement>& data,
                       Eigen::MatrixXd &M,
                       Eigen::VectorXd &b)
    {
        const int n = data.size();
        M.resize(n, 8);
        b.resize(n);

        for (int i = 0; i < n; ++i) {
            const auto& meas = data[i];

            double x = meas.anchor_odom_pose(0);
            double y = meas.anchor_odom_pose(1);
            double z = meas.anchor_odom_pose(2);

            double A = meas.tag_odom_pose(0);
            double B = meas.tag_odom_pose(1);
            double C = meas.tag_odom_pose(2);

            double d = meas.distance;
            double bi = d * d - (A * A + B * B + C * C) - (x * x + y * y + z * z) + 2.0 * C * z;

            double beta1 = -2.0 * A;
            double beta2 = -2.0 * B;
            double beta3 = 2.0 * z - 2.0 * C;
            double beta4 = -2.0 * (A * x + B * y);
            double beta5 = 2.0 * (A * y - B * x);
            double beta6 = 2.0 * x;
            double beta7 = 2.0 * y;

            M(i, 0) = beta1;
            M(i, 1) = beta2;
            M(i, 2) = beta3;
            M(i, 3) = beta4;
            M(i, 4) = beta5;
            M(i, 5) = beta6;
            M(i, 6) = beta7;
            M(i, 7) = 1.0;

            b(i) = bi;
        }

        return true;
    }


    bool solveMartelLinearSystem(const std::deque<UWBMeasurement>& measurements,
                             Eigen::MatrixXd &M,
                             Eigen::VectorXd &b,
                             State &init_state,
                             bool useRANSAC = true)
    {
        if (measurements.size() < 10) {
            RCLCPP_WARN(rclcpp::get_logger("eliko_global_opt_node"), "Not enough measurements for linear system.");
            return false;
        }

        if (useRANSAC) {
            size_t num_iters = 100;
            size_t min_samples = 10;
            double inlier_threshold = 0.5;  // meters
            size_t best_inliers = 0;
            Eigen::Vector4d best_state;
            double inlier_ratio = 0.0;
            double min_inlier_ratio = 0.7;
            bool found_valid = false;

            for (size_t iter = 0; iter < num_iters; ++iter) {
                std::vector<UWBMeasurement> sample = getRandomSubset(measurements, min_samples);

                Eigen::MatrixXd M_local;
                Eigen::VectorXd b_local;
                if (!buildMartelSystem(sample, M_local, b_local)) continue;

                Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_local, Eigen::ComputeThinU | Eigen::ComputeThinV);
                Eigen::VectorXd solution = svd.solve(b_local);

                size_t inliers = 0;
                std::vector<UWBMeasurement> inlier_set;

                for (const auto& meas : measurements) {
                    double pred = predictDistanceFromMartel(solution, meas);
                    double err = std::abs(pred - meas.distance);
                    if (err < inlier_threshold) {
                        inliers++;
                        inlier_set.push_back(meas);
                    }
                }

                if (inliers > best_inliers) {

                    best_inliers = inliers;
                    inlier_ratio = static_cast<double>(best_inliers) / measurements.size();

                    if (inlier_ratio > min_inlier_ratio) {

                        found_valid = true;

                        Eigen::MatrixXd M_final;
                        Eigen::VectorXd b_final;
                        buildMartelSystem(inlier_set, M_final, b_final);
                        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_final, Eigen::ComputeThinU | Eigen::ComputeThinV);
                        Eigen::VectorXd refined_solution = svd.solve(b_final);
                        init_state.state[0] = refined_solution[0];
                        init_state.state[1] = refined_solution[1];
                        init_state.state[2] = refined_solution[2];
                        init_state.state[3] = std::atan2(refined_solution[4], refined_solution[3]);
                        init_state.pose = buildTransformationSE3(init_state.roll, init_state.pitch, init_state.state);
                        init_state.covariance = Eigen::Matrix4d::Identity() * 0.1;

                        RCLCPP_INFO_STREAM(rclcpp::get_logger("eliko_global_opt_node"),
                            "Refined RANSAC solution with " << inlier_set.size() << " inliers:\n"
                            << "State: [x=" << init_state.state[0] << ", y=" << init_state.state[1]
                            << ", z=" << init_state.state[2] << ", yaw=" << init_state.state[3] * 180.0 / M_PI << " deg]");
                        break;
                    }
                }
            }

            if (!found_valid) {
                RCLCPP_WARN(rclcpp::get_logger("eliko_global_opt_node"), "RANSAC failed to find a valid model. Inlier ratio: %f", inlier_ratio);
                return false;
            }
            
            return true;
        }

        // Else: normal least squares with full data
        std::vector<UWBMeasurement> finalSet(measurements.begin(), measurements.end());
        if (!buildMartelSystem(finalSet, M, b)) return false;

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd solution = svd.solve(b);

        init_state.state[0] = solution[0];
        init_state.state[1] = solution[1];
        init_state.state[2] = solution[2];
        init_state.state[3] = std::atan2(solution[4], solution[3]);

        init_state.covariance = Eigen::Matrix4d::Identity();  // or something more meaningful

        RCLCPP_INFO_STREAM(rclcpp::get_logger("eliko_global_opt_node"), "Initial solution: " << init_state.state.transpose());

        return true;
    }

    Eigen::Vector4d movingAverage(const State &sample, const size_t &max_samples) {
        
            // Add the new sample with timestamp
            moving_average_states_.push_back(sample);

            // Remove samples that are too old or if we exceed the window size
            while (!moving_average_states_.empty() &&
                moving_average_states_.size() > max_samples) {
                moving_average_states_.pop_front();
            }

            // Initialize smoothed values
            Eigen::Vector4d smoothed_state = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);

            // Initialize accumulators for circular mean of yaw
            double sum_sin = 0.0;
            double sum_cos = 0.0;

            // Initialize total weight
            double total_weight = 0.0;
            // Define a decay factor (e.g., between 0 and 1) for exponential weighting
            double decay_factor = 0.9;  // Adjust as needed for desired weighting effect
            double weight = 1.0;  // Start with a weight of 1.0 for the most recent sample

            // Iterate over samples in reverse order (from newest to oldest)
            for (auto it = moving_average_states_.rbegin(); it != moving_average_states_.rend(); ++it) {
                // Apply the weight to each sample's translation and accumulate
                smoothed_state[0] += weight * it->state[0];
                smoothed_state[1] += weight * it->state[1];
                smoothed_state[2] += weight * it->state[2];
                
                // Convert yaw to sine and cosine, apply the weight, then accumulate
                sum_sin += weight * std::sin(it->state[3]);
                sum_cos += weight * std::cos(it->state[3]);

                // Accumulate total weight for averaging
                total_weight += weight;

                // Decay the weight for the next (older) sample
                weight *= decay_factor;
            }

            // Normalize to get the weighted average
            if (total_weight > 0.0) {
                smoothed_state[0] /= total_weight;
                smoothed_state[1] /= total_weight;
                smoothed_state[2] /= total_weight;

                // Calculate weighted average yaw using the weighted sine and cosine
                smoothed_state[3] = std::atan2(sum_sin / total_weight, sum_cos / total_weight);
                smoothed_state[3] = normalizeAngle(smoothed_state[3]);
            }

            return smoothed_state;
    }

    // Run the optimization once all measurements are received
    bool runOptimization(const rclcpp::Time &current_time, const Eigen::Matrix4d& motion_covariance, const bool& init_prior) {

            ceres::Problem problem;

            State opt_state;

            if(use_prior_) opt_state = init_state_;
            else opt_state = opt_state_;
            
            //Update the timestamp
            opt_state.timestamp = current_time;

            // Create an instance of our custom manifold.
            ceres::Manifold* state_manifold = new StateManifold4D();
            // Attach it to the parameter block.
            problem.AddParameterBlock(opt_state.state.data(), 4);

            //Tell the optimizer to perform updates in the manifold space
            problem.SetManifold(opt_state.state.data(), state_manifold);

            // Define a robust kernel
            double loss_threshold = 2.5; // = residuals higher than 2.5 times sigma are outliers
            // ceres::LossFunction* robust_loss = new ceres::HuberLoss(loss_threshold);
            ceres::LossFunction* robust_loss = new ceres::CauchyLoss(loss_threshold);

            //Set prior using linear initial solution
            if(init_prior){
                Eigen::Matrix4d prior_covariance = motion_covariance + Eigen::Matrix4d::Identity() * std::pow(measurement_stdev_,2.0);
                Sophus::SE3d prior_T = buildTransformationSE3(init_state_.roll, init_state_.pitch, init_state_.state);
                //Add the prior residual with the full covariance
                ceres::CostFunction* prior_cost = PriorCostFunction::Create(prior_T, opt_state.roll, opt_state.pitch, prior_covariance);
                problem.AddResidualBlock(prior_cost, robust_loss, opt_state.state.data());
            }


            for (const auto& measurement : global_measurements_) {                   
                
                // //Use positions of anchors and tags in each robot odometry frame
                Eigen::Vector3d anchor_pos = measurement.anchor_odom_pose;
                Eigen::Vector3d tag_pos = measurement.tag_odom_pose;

                ceres::CostFunction* cost_function = UWBCostFunction::Create(
                    anchor_pos, tag_pos, measurement.distance, opt_state.roll, opt_state.pitch, measurement_stdev_);
                problem.AddResidualBlock(cost_function, robust_loss, opt_state.state.data());
            }


            // Configure solver options
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR; // ceres::SPARSE_NORMAL_CHOLESKY,  ceres::DENSE_QR
            options.num_threads = 4;
            options.use_nonmonotonic_steps = true;  // Help escape plateaus
            options.max_consecutive_nonmonotonic_steps = 10; //default is 5
            options.max_num_iterations = 100;
            // Logging
            // options.minimizer_progress_to_stdout = true;

            // Solve
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            RCLCPP_INFO(this->get_logger(), summary.BriefReport().c_str());

            // Notify and update values if optimization converged
            if (summary.termination_type == ceres::CONVERGENCE){
                
                total_solves_++;
                total_solver_time_ += summary.total_time_in_seconds;
                // Compute Covariance
                ceres::Covariance::Options cov_options;
                ceres::Covariance covariance(cov_options);

                // Specify the parameter blocks for which to compute the covariance
                std::vector<std::pair<const double*, const double*>> covariance_blocks;
                covariance_blocks.emplace_back(opt_state.state.data(), opt_state.state.data());  // Full parameter block

                // Compute the covariance
                Eigen::Matrix4d opt_covariance = Eigen::Matrix4d::Zero();

                if (covariance.Compute(covariance_blocks, &problem)) {
                    // Extract the full covariance matrix
                    covariance.GetCovarianceBlock(opt_state.state.data(), opt_state.state.data(), opt_covariance.data());
                    
                    // Optionally, store covariance for further use
                    opt_state.covariance = opt_covariance;  // Add this to your State struct if needed

                    //Update global state variable
                    opt_state_ = opt_state;

                    return true;

                } else {
                    RCLCPP_WARN(this->get_logger(), "Failed to compute covariance.");
                    
                    return false;
                }
                                            
            }

            return false; 
            
    }
  
    // Subscriptions
    rclcpp::Subscription<eliko_messages::msg::DistancesList>::SharedPtr eliko_distances_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr agv_odom_sub_, uav_odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr dll_agv_sub_, dll_uav_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr range_pub_;

    // Timers
    rclcpp::TimerBase::SharedPtr global_optimization_timer_;

    std::deque<UWBMeasurement> global_measurements_;
    
    // Publishers/Broadcasters
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr tf_publisher_, tf_ransac_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    //Parameters
    std::string odom_topic_agv_, odom_topic_uav_;
    double opt_timer_rate_;
    size_t min_measurements_;
    double min_traveled_distance_, min_traveled_angle_, max_traveled_distance_;
    double measurement_stdev_;
    size_t moving_average_max_samples_;
    bool moving_average_;
    bool use_prior_;
    bool use_ransac_;
    bool debug_;
    std::unordered_map<std::string, Eigen::Vector3d> anchor_positions_;
    std::unordered_map<std::string, Eigen::Vector3d> tag_positions_;


    std::string agv_odom_frame_id_, uav_odom_frame_id_;
    std::string agv_body_frame_id_, uav_body_frame_id_;
    std::string uav_opt_frame_id_, agv_opt_frame_id_;

    State opt_state_;
    State init_state_;
    std::deque<State> moving_average_states_;

    Sophus::SE3d uav_odom_pose_, last_uav_odom_pose_;         // Current UAV odometry position and last used for optimization
    Sophus::SE3d agv_odom_pose_, last_agv_odom_pose_;        // Current AGV odometry position and last used for optimization
    Sophus::SE3d agv_init_pose_, uav_init_pose_;
    Eigen::Matrix<double, 6, 6> uav_odom_covariance_, agv_odom_covariance_;  // UAV odometry covariance
    double last_agv_odom_time_sec_, last_uav_odom_time_sec_;
    bool last_agv_odom_initialized_, last_uav_odom_initialized_;

    double uav_delta_translation_, agv_delta_translation_, uav_delta_rotation_, agv_delta_rotation_;
    double uav_total_translation_, agv_total_translation_, uav_total_rotation_, agv_total_rotation_;

    std::unordered_map<std::string, Eigen::Vector3d> anchor_positions_odom_, tag_positions_odom_;

    double odom_error_distance_, odom_error_angle_;

    //Timing metrics
    int total_solves_;
    double total_solver_time_;

};
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ElikoGlobalOptNode>();
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));
    rclcpp::spin(node);

    // Once spin() returns (e.g. on SIGINT), dump out metrics:
    node->getMetrics();

    rclcpp::shutdown();
    return 0;
}
