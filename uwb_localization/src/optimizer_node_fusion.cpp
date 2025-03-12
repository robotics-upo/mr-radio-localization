#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "geometry_msgs/msg/quaternion_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"


#include <nav_msgs/msg/odometry.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>

#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>

/// @brief Basic point cloud registration example with PCL interfaces
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/benchmark/read_points.hpp>


#include "eliko_messages/msg/distances_list.hpp"
#include "eliko_messages/msg/anchor_coords_list.hpp"
#include "eliko_messages/msg/tag_coords_list.hpp"


#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/manifold.h>

#include <sophus/se3.hpp>   // Include Sophus for SE(3) operations

#include <Eigen/Core>
#include <vector>
#include <sstream>
#include <deque>
#include <Eigen/Dense>
#include <utility>
#include <unordered_map>
#include <chrono>

#include "uwb_localization/msg/pose_with_covariance_stamped_array.hpp"  

// Include the service header (adjust the package name accordingly)
#include "uwb_localization/srv/update_point_clouds.hpp"  
using UpdatePointClouds = uwb_localization::srv::UpdatePointClouds;
using namespace small_gicp;


struct PriorConstraint {
    Sophus::SE3d pose;  // measured relative transform
    Eigen::Matrix<double, 4, 4> covariance;
};

struct MeasurementConstraint {
    int id_begin;
    int id_end;
    Sophus::SE3d t_T_s;  // measured relative transform
    Eigen::Matrix<double, 4, 4> covariance;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  };

using VectorOfConstraints =
    std::vector<MeasurementConstraint, Eigen::aligned_allocator<MeasurementConstraint>>;

struct State {
    
    rclcpp::Time timestamp; 
    Eigen::Vector4d state; // [x,y,z,yaw]
    double roll;
    double pitch;
    Sophus::SE3d pose; //full pose, with roll and pitch read from imu
    Eigen::Matrix4d covariance;
    bool is_anchor;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

using MapOfStates =
    std::map<int,
             State,
             std::less<int>,
             Eigen::aligned_allocator<std::pair<const int, State>>>;


struct Measurements {
    rclcpp::Time timestamp;      // The time at which the measurement was taken.
    Sophus::SE3d odom_pose;
    
    Eigen::Matrix<double, 6, 6> odom_covariance;
  
    // Pointer to an associated point cloud (keyframe scan).
    pcl::PointCloud<pcl::PointXYZ>::Ptr scan;

     // Constructor to initialize the pointer.
     Measurements() : scan(new pcl::PointCloud<pcl::PointXYZ>) {}
};

// Custom manifold for a state in R^3 x S^1.
class StateManifold3D : public ceres::Manifold {
    public:
      // Ambient space dimension is 4.
      virtual int AmbientSize() const override { return 4; }
      // Tangent space dimension is also 4.
      virtual int TangentSize() const override { return 4; }
    
      // Plus(x, delta) = [x0+delta0, x1+delta1, x2+delta2, wrap(x3+delta3)]
      virtual bool Plus(const double* x,
                        const double* delta,
                        double* x_plus_delta) const override {
        // Translation is standard addition.
        x_plus_delta[0] = x[0] + delta[0];
        x_plus_delta[1] = x[1] + delta[1];
        x_plus_delta[2] = x[2] + delta[2];
        // For yaw, perform addition with wrapping to [-pi, pi].
        double new_yaw = x[3] + delta[3];
        while(new_yaw > M_PI)  new_yaw -= 2.0 * M_PI;
        while(new_yaw < -M_PI) new_yaw += 2.0 * M_PI;
        x_plus_delta[3] = new_yaw;
        return true;
      }
    
      // The Jacobian of Plus with respect to delta at delta = 0 is the identity.
      virtual bool PlusJacobian(const double* /*x*/, double* jacobian) const override {
        // Fill a 4x4 identity matrix (row-major ordering).
        for (int i = 0; i < 16; ++i) {
          jacobian[i] = 0.0;
        }
        jacobian[0]  = 1.0;
        jacobian[5]  = 1.0;
        jacobian[10] = 1.0;
        jacobian[15] = 1.0;
        return true;
      }
    
      // Minus(y, x) computes the tangent vector delta such that x ⊕ delta = y.
      virtual bool Minus(const double* y,
                         const double* x,
                         double* y_minus_x) const override {
        // For translation, simple subtraction.
        y_minus_x[0] = y[0] - x[0];
        y_minus_x[1] = y[1] - x[1];
        y_minus_x[2] = y[2] - x[2];
        // For yaw, compute the difference and wrap it.
        double dtheta = y[3] - x[3];
        while (dtheta > M_PI)  dtheta -= 2.0 * M_PI;
        while (dtheta < -M_PI) dtheta += 2.0 * M_PI;
        y_minus_x[3] = dtheta;
        return true;
      }
    
      // The Jacobian of Minus with respect to y at y = x is the identity.
      virtual bool MinusJacobian(const double* /*x*/, double* jacobian) const override {
        for (int i = 0; i < 16; ++i) {
          jacobian[i] = 0.0;
        }
        jacobian[0]  = 1.0;
        jacobian[5]  = 1.0;
        jacobian[10] = 1.0;
        jacobian[15] = 1.0;
        return true;
      }
    };



class FusionOptimizationNode : public rclcpp::Node {

public:

    FusionOptimizationNode() : Node("fusion_optimization_node") {

    //Option 1: get odometry through topics -> includes covariance
    std::string odom_topic_agv = "/arco/idmind_motors/odom"; //or "/agv/odom"
    std::string odom_topic_uav = "/uav/odom"; //or "/uav/odom"

    rclcpp::SensorDataQoS qos; // Use a QoS profile compatible with sensor data

    last_agv_odom_initialized_ = false;
    last_uav_odom_initialized_ = false;

    min_traveled_distance_ = 0.25;
    min_traveled_angle_ = 25.0 * M_PI / 180.0;
    uav_translation_ = agv_translation_ = uav_rotation_ = agv_rotation_ = 0.0;

    agv_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    odom_topic_agv, qos, std::bind(&FusionOptimizationNode::agv_odom_cb_, this, std::placeholders::_1));

    uav_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_uav, qos, std::bind(&FusionOptimizationNode::uav_odom_cb_, this, std::placeholders::_1));

    //Option 2: get odometry through tf readings -> only transform
    odom_tf_agv_s_ = "arco/base_link"; //source
    odom_tf_agv_t_ = "arco/odom"; //target
    odom_tf_uav_s_ = "odom"; //source
    odom_tf_uav_t_ = "world"; //target

    //Point cloud topics (RADAR or LIDAR)
    std::string pcl_topic_agv = "/arco/ouster/points"; //LIDAR: "/arco/ouster/points", RADAR: "/arco/radar/PointCloudObject"
    std::string pcl_topic_uav = "/os1_cloud_node/points_non_dense"; //LIDAR: "/os1_cloud_node/points_non_dense", RADAR: "/drone/radar/PointCloudObject"

    pcl_source_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                pcl_topic_agv, qos, std::bind(&FusionOptimizationNode::pcl_source_cb_, this, std::placeholders::_1));

    pcl_target_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                pcl_topic_uav, qos, std::bind(&FusionOptimizationNode::pcl_target_cb_, this, std::placeholders::_1));

    pcl_visualizer_client_ = this->create_client<uwb_localization::srv::UpdatePointClouds>("eliko_optimization_node/pcl_visualizer_service");

    optimized_tf_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/eliko_optimization_node/optimized_T", 10,
        std::bind(&FusionOptimizationNode::optimized_tf_cb_, this, std::placeholders::_1));

    //Pose publishers
    anchor_agv_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("pose_graph_node/agv_anchor", 10);
    anchor_uav_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("pose_graph_node/uav_anchor", 10);

    poses_uav_publisher_ = this->create_publisher<uwb_localization::msg::PoseWithCovarianceStampedArray>("pose_graph_node/uav_poses", 10);
    poses_agv_publisher_ = this->create_publisher<uwb_localization::msg::PoseWithCovarianceStampedArray>("pose_graph_node/agv_poses", 10);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    global_opt_window_s_ = 5.0; //size of the sliding window in seconds
    global_opt_rate_s_ = 0.2 * global_opt_window_s_; //rate of the optimization
    min_keyframes_ = 3.0; //number of nodes to run optimization

    T_uav_lidar_ = build_transformation_SE3(0.0,0.0, Eigen::Vector4d(0.21,0.0,0.25,0.0));
    T_agv_lidar_ = build_transformation_SE3(3.14,0.0, Eigen::Vector4d(0.3,0.0,0.45,0.0));
    pointcloud_sigma_ = 0.05;  //5cm for lidar, 10 cm for radar

    global_optimization_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(int(global_opt_rate_s_*1000)), std::bind(&FusionOptimizationNode::global_opt_cb_, this));

    global_frame_graph_ = "graph_odom";
    eliko_frame_id_ = "agv_opt"; //frame of the eliko system-> arco/eliko, for simulation use "agv_gt" for ground truth, "agv_odom" for odometry w/ errors
    uav_frame_id_ = "uav_opt"; //frame of the uav -> "base_link", for simulation use "uav_opt"

    //Start node counter
    agv_id_ = uav_id_ = 0;

    graph_initialized_ = false;
    first_optimization_done_ = false;

    //Set ICP algorithm variant: 2->Generalized ICP, 1->Point to Plane ICP, else -> basic ICP
    icp_type_ = 1;

    uwb_encounter_available_ = false;
    icp_encounter_available_ = false;

    RCLCPP_INFO(this->get_logger(), "Eliko Optimization Node initialized.");
  }


private:


    void agv_odom_cb_(const nav_msgs::msg::Odometry::SharedPtr msg) {
        
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
        
        agv_odom_pose_ = Sophus::SE3d(q, t);


        // If this is the first message, simply store it and return.
        if (!last_agv_odom_initialized_) {
            last_agv_odom_msg_ = *msg;
            last_agv_odom_initialized_ = true;
            last_agv_odom_pose_ = agv_odom_pose_;
            return;
        }
        
        // Update displacement, via logarithmic map

        Eigen::Matrix<double, 6, 1> log = (last_agv_odom_pose_.inverse() * agv_odom_pose_).log();

        Eigen::Vector3d delta_translation = log.head<3>(); // first three elements
        Eigen::Vector3d delta_rotation    = log.tail<3>(); // last three elements

        agv_translation_+=delta_translation.norm();
        agv_rotation_+=delta_rotation.norm();       
        
    
        //Read covariance
        Eigen::Matrix<double,6,6> cov;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                cov(i, j) = msg->pose.covariance[i * 6 + j];
            }
        }
        agv_odom_covariance_ = cov;
        
        last_agv_odom_msg_ = *msg;
    
        // RCLCPP_INFO(this->get_logger(), "Updated AGV odometry");
        // log_transformation_matrix(agv_odom_pose_.matrix());
    }

    void uav_odom_cb_(const nav_msgs::msg::Odometry::SharedPtr msg) {
        
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
        
        uav_odom_pose_ = Sophus::SE3d(q, t);

        // If this is the first message, simply store it and return.
        if (!last_uav_odom_initialized_) {
            last_uav_odom_msg_ = *msg;
            last_uav_odom_initialized_ = true;
            last_uav_odom_pose_ = uav_odom_pose_;
            return;
        }
        
        // Update displacement, via logarithmic map

        Eigen::Matrix<double, 6, 1> log = (last_uav_odom_pose_.inverse() * uav_odom_pose_).log();

        Eigen::Vector3d delta_translation = log.head<3>(); // first three elements
        Eigen::Vector3d delta_rotation    = log.tail<3>(); // last three elements

        uav_translation_+=delta_translation.norm();
        uav_rotation_+=delta_rotation.norm();       
    
        //Read covariance
        Eigen::Matrix<double,6,6> cov;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                cov(i, j) = msg->pose.covariance[i * 6 + j];
            }
        }
        uav_odom_covariance_ = cov;
        
        last_uav_odom_msg_ = *msg;
    
        // RCLCPP_INFO(this->get_logger(), "Updated AGV odometry from velocities");
        // log_transformation_matrix(agv_odom_pose_.matrix());
    }


    void pcl_source_cb_(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
        
        if (!msg->data.empty()) {  // Ensure the incoming message has data
            pcl::fromROSMsg(*msg, *agv_cloud_);
            RCLCPP_DEBUG(this->get_logger(), "AGV cloud received with %zu points", agv_cloud_->points.size());
        } 

        else {
                RCLCPP_WARN(this->get_logger(), "Empty source point cloud received!");
        }
            return;
    }

    void pcl_target_cb_(const sensor_msgs::msg::PointCloud2::SharedPtr msg){

        if (!msg->data.empty()) {  // Ensure the incoming message has data
            pcl::fromROSMsg(*msg, *uav_cloud_);
            RCLCPP_DEBUG(this->get_logger(), "UAV cloud received with %zu points", uav_cloud_->points.size());
        } 
        
        else {            
            RCLCPP_WARN(this->get_logger(), "Empty target point cloud received!");
        }
        
        return;
    }


     // Callback for receiving the optimized relative transform from the fast node.
    void optimized_tf_cb_(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {

        latest_relative_pose_ = *msg;
        RCLCPP_DEBUG(this->get_logger(), "Received optimized relative transform.");   
    }


    void global_opt_cb_() {

        rclcpp::Time current_time = this->get_clock()->now();

        //********************INITIALIZATIONS*********************** */
        if(!graph_initialized_){

            RCLCPP_INFO(this->get_logger(), "Initializing graph!");

            //Get first measurements
            agv_measurements_.timestamp = current_time;
            *(agv_measurements_.scan) = *(downsamplePointCloud(agv_cloud_, pointcloud_sigma_));
            agv_measurements_.odom_pose = agv_odom_pose_;
            agv_measurements_.odom_covariance = agv_odom_covariance_;

            Eigen::Vector3d t_odom_agv = agv_measurements_.odom_pose.translation();
            Eigen::Matrix3d R_odom_agv = agv_measurements_.odom_pose.rotationMatrix();  // or T.so3().matrix()
            // Compute Euler angles in ZYX order: [yaw, pitch, roll]
            Eigen::Vector3d euler_agv = R_odom_agv.eulerAngles(2, 1, 0);
            //Initial values for state AGV
            init_state_agv_.timestamp = current_time;
            init_state_agv_.state = Eigen::Vector4d(t_odom_agv[0], t_odom_agv[1], t_odom_agv[2], euler_agv[0]);
            init_state_agv_.roll = euler_agv[2];
            init_state_agv_.pitch = euler_agv[1];
            init_state_agv_.pose = build_transformation_SE3(init_state_agv_.roll, init_state_agv_.pitch, init_state_agv_.state);
            init_state_agv_.covariance = Eigen::Matrix4d::Identity(); //
            init_state_agv_.is_anchor = false;

            RCLCPP_INFO(this->get_logger(), "Adding initial AGV node at timestamp %.2f: [%f, %f, %f, %f]", current_time.seconds(),
            init_state_agv_.state[0], init_state_agv_.state[1], init_state_agv_.state[2], init_state_agv_.state[3]);

            agv_map_[agv_id_] = init_state_agv_;

            //Anchor node for AGV
            anchor_node_agv_.timestamp = current_time;
            anchor_node_agv_.state = init_state_agv_.state;
            anchor_node_agv_.roll = 0.0;
            anchor_node_agv_.pitch = 0.0;
            anchor_node_agv_.pose = init_state_agv_.pose;
            anchor_node_agv_.covariance = Eigen::Matrix4d::Identity(); //
            anchor_node_agv_.is_anchor = true;

            agv_translation_ = agv_rotation_ = 0.0;
            prev_agv_measurements_ = agv_measurements_;

            uav_measurements_.timestamp = current_time;
            *(uav_measurements_.scan) = *(downsamplePointCloud(uav_cloud_, pointcloud_sigma_));
            uav_measurements_.odom_pose = uav_odom_pose_;
            uav_measurements_.odom_covariance = uav_odom_covariance_;

            Eigen::Vector3d t_odom_uav = uav_measurements_.odom_pose.translation();
            Eigen::Matrix3d R_odom_uav = uav_measurements_.odom_pose.rotationMatrix();  // or T.so3().matrix()
            Eigen::Vector3d euler_uav = R_odom_uav.eulerAngles(2, 1, 0);

            //Initial values for state UAV
            init_state_uav_.timestamp = current_time;
            init_state_uav_.state = Eigen::Vector4d(t_odom_uav[0], t_odom_uav[1], t_odom_uav[2], euler_uav[0]);
            init_state_uav_.roll = euler_uav[2];
            init_state_uav_.pitch = euler_uav[1];
            init_state_uav_.pose = build_transformation_SE3(init_state_uav_.roll, init_state_uav_.pitch, init_state_uav_.state);
            init_state_uav_.covariance = Eigen::Matrix4d::Identity(); //
            init_state_uav_.is_anchor = false;

            RCLCPP_INFO(this->get_logger(), "Adding initial UAV node at timestamp %.2f: [%f, %f, %f, %f]", current_time.seconds(),
            init_state_uav_.state[0], init_state_uav_.state[1], init_state_uav_.state[2], init_state_uav_.state[3]);

            uav_map_[uav_id_] = init_state_uav_;

            //Create anchor node for UAV
            anchor_node_uav_.timestamp = current_time;
            anchor_node_uav_.state = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
            anchor_node_uav_.roll = 0.0;
            anchor_node_uav_.pitch = 0.0;
            anchor_node_uav_.pose = build_transformation_SE3(anchor_node_uav_.roll, anchor_node_uav_.pitch, anchor_node_uav_.state);
            anchor_node_uav_.covariance = Eigen::Matrix4d::Identity(); //
            anchor_node_uav_.is_anchor = true;

            uav_translation_ = uav_rotation_ = 0.0;
            prev_uav_measurements_ = uav_measurements_;

            //Initialize prior constraints
            
            //AGV local trajectory prior
            prior_agv_.pose = agv_measurements_.odom_pose;
            prior_agv_.covariance = Eigen::Matrix4d::Identity() * 1e-6;

            //Anchor AGV prior
            prior_anchor_.pose = agv_measurements_.odom_pose;
            prior_anchor_.covariance = Eigen::Matrix4d::Identity() * 1e-6;

            //UAV local trajectory prior
            prior_uav_.pose = uav_measurements_.odom_pose;
            prior_uav_.covariance = Eigen::Matrix4d::Identity() * 1e-6;

            graph_initialized_ = true;

            RCLCPP_INFO(this->get_logger(), "Initialization done!");

            return;
        }

        //********************ADDING NEW NODES*********************** */

        bool new_agv_node = (agv_translation_ >= min_traveled_distance_ || agv_rotation_ >= min_traveled_angle_);
        bool new_uav_node = (uav_translation_ >= min_traveled_distance_ || uav_rotation_ >= min_traveled_angle_);

        if(!new_agv_node && !new_uav_node){

            RCLCPP_INFO(this->get_logger(), "Neither robot moved enough, skipping optimization...");
            return;
        }
        
        if (new_agv_node) {

            agv_id_++;
            
            agv_measurements_.timestamp = current_time;
            *(agv_measurements_.scan) = *(downsamplePointCloud(agv_cloud_, pointcloud_sigma_));
            agv_measurements_.odom_pose = agv_odom_pose_;
            agv_measurements_.odom_covariance = agv_odom_covariance_;

            // Create a new AGV node from the current odometry.
            State new_agv;
            new_agv.timestamp = current_time;
            new_agv.is_anchor = false;

            Eigen::Matrix3d R_odom_agv = agv_measurements_.odom_pose.rotationMatrix();  // or T.so3().matrix()
            // Compute Euler angles in ZYX order: [yaw, pitch, roll]
            Eigen::Vector3d euler_agv = R_odom_agv.eulerAngles(2, 1, 0);
            new_agv.pitch = euler_agv[1];  // rotation around Y-axis
            new_agv.roll = euler_agv[2];  // rotation around X-axis
            new_agv.state = agv_map_[agv_id_ - 1].state;
            new_agv.pose = agv_map_[agv_id_ - 1].pose;
            new_agv.covariance = agv_map_[agv_id_ - 1].covariance;
            
            agv_map_[agv_id_] = new_agv;

            RCLCPP_INFO(this->get_logger(), "Adding AGV node %d at timestamp %.2f: [%f, %f, %f, %f]", agv_id_, current_time.seconds(),
                            new_agv.state[0], new_agv.state[1], new_agv.state[2], new_agv.state[3]);


            //ADD AGV Proprioceptive constraints
            
            //AGV odom constraints
            MeasurementConstraint constraint_odom_agv;
            constraint_odom_agv.id_begin = agv_id_ - 1;
            constraint_odom_agv.id_end = agv_id_;

            Sophus::SE3d odom_T_s_agv = prev_agv_measurements_.odom_pose;
            Sophus::SE3d odom_T_t_agv = agv_measurements_.odom_pose;
            constraint_odom_agv.t_T_s = odom_T_t_agv.inverse()*odom_T_s_agv;

            constraint_odom_agv.covariance = computeRelativeOdometryCovariance(agv_measurements_.odom_pose, prev_agv_measurements_.odom_pose,
                                                                                agv_measurements_.odom_covariance, prev_agv_measurements_.odom_covariance);

            proprioceptive_constraints_agv_.push_back(constraint_odom_agv);

            //AGV ICP constraints

            if (!agv_measurements_.scan->points.empty() &&
                !prev_agv_measurements_.scan->points.empty()) {

                    RCLCPP_WARN(this->get_logger(), "Computing ICP for AGV nodes %d and %d.", agv_id_ - 1, agv_id_);

                    Eigen::Matrix4f T_icp = constraint_odom_agv.t_T_s.cast<float>().matrix();
                    //Eigen::Matrix4f T_icp = Eigen::Matrix4f::Identity();

                    double fitness = 0.0;
        
                    if(run_icp(prev_agv_measurements_.scan, agv_measurements_.scan, T_icp, fitness, icp_type_)){
                        MeasurementConstraint constraint_icp_agv;
                        constraint_icp_agv.id_begin = agv_id_ - 1;
                        constraint_icp_agv.id_end = agv_id_;
                        //Transform to the robots body frame
                        Sophus::SE3d T_icp_agv = T_agv_lidar_ * Sophus::SE3f(T_icp).cast<double>() * T_agv_lidar_.inverse();
                        constraint_icp_agv.t_T_s = T_icp_agv;
                        
                        //log_transformation_matrix(constraint_icp_agv.t_T_s.matrix());

                        // Compute the ICP covariance from the scan pair.
                        constraint_icp_agv.covariance = computeICPCovariance(prev_agv_measurements_.scan,
                            agv_measurements_.scan, T_icp, pointcloud_sigma_);

                        extraceptive_constraints_agv_.push_back(constraint_icp_agv);
                    };
            }

            prev_agv_measurements_ = agv_measurements_;
            agv_translation_ = agv_rotation_ = 0.0;

        }

        if (new_uav_node) {

            uav_id_++;

            uav_measurements_.timestamp = current_time;

            *(uav_measurements_.scan) = *(downsamplePointCloud(uav_cloud_, pointcloud_sigma_));
            uav_measurements_.odom_pose = uav_odom_pose_;
            uav_measurements_.odom_covariance = uav_odom_covariance_;

            // Similarly, create a new UAV node.
            State new_uav;  
            new_uav.timestamp = current_time;
            new_uav.is_anchor = false;

            Eigen::Matrix3d R_odom_uav = uav_measurements_.odom_pose.rotationMatrix();  // or T.so3().matrix()
            // Compute Euler angles in ZYX order: [yaw, pitch, roll]
            Eigen::Vector3d euler_uav = R_odom_uav.eulerAngles(2, 1, 0);
            new_uav.pitch = euler_uav[1];  // rotation around Y-axis
            new_uav.roll = euler_uav[2];  // rotation around X-axis
            new_uav.state = uav_map_[uav_id_ - 1].state;
            new_uav.pose = uav_map_[uav_id_ - 1].pose;
            new_uav.covariance = uav_map_[uav_id_ - 1].covariance;

            uav_map_[uav_id_] = new_uav;

            RCLCPP_INFO(this->get_logger(), "Adding new UAV node %d at timestamp %.2f: [%f, %f, %f, %f]", uav_id_, current_time.seconds(),
                            new_uav.state[0], new_uav.state[1], new_uav.state[2], new_uav.state[3]);

            //ADD UAV Proprioceptive constraints

            //UAV odom constraints
            MeasurementConstraint constraint_odom_uav;
            constraint_odom_uav.id_begin = uav_id_ - 1;
            constraint_odom_uav.id_end = uav_id_;

            Sophus::SE3d odom_T_s_uav = prev_uav_measurements_.odom_pose;
            Sophus::SE3d odom_T_t_uav = uav_measurements_.odom_pose;
            constraint_odom_uav.t_T_s = odom_T_t_uav.inverse()*odom_T_s_uav;
            constraint_odom_uav.covariance = computeRelativeOdometryCovariance(uav_measurements_.odom_pose, prev_uav_measurements_.odom_pose,
                                                                                uav_measurements_.odom_covariance, prev_uav_measurements_.odom_covariance);
            proprioceptive_constraints_uav_.push_back(constraint_odom_uav);

            //UAV ICP constraints
            if (!uav_measurements_.scan->points.empty() &&
                !prev_uav_measurements_.scan->points.empty()) {

                    RCLCPP_WARN(this->get_logger(), "Computing ICP for UAV nodes %d and %d.", uav_id_ - 1, uav_id_);
                    
                    Eigen::Matrix4f T_icp = constraint_odom_uav.t_T_s.cast<float>().matrix();
                    //Eigen::Matrix4f T_icp = Eigen::Matrix4f::Identity();

                    double fitness = 0.0;
                    if(run_icp(prev_uav_measurements_.scan, uav_measurements_.scan, T_icp, fitness, icp_type_)){
                        MeasurementConstraint constraint_icp_uav;
                        constraint_icp_uav.id_begin = uav_id_ - 1;
                        constraint_icp_uav.id_end = uav_id_;
                        //Transform to the robots body frame
                        Sophus::SE3d T_icp_uav = T_uav_lidar_ * Sophus::SE3f(T_icp).cast<double>() * T_uav_lidar_.inverse();
                        constraint_icp_uav.t_T_s = T_icp_uav;
                        
                        //log_transformation_matrix(constraint_icp_uav.t_T_s.matrix());

                        // Compute the ICP covariance from the scan pair.
                        constraint_icp_uav.covariance = computeICPCovariance(prev_uav_measurements_.scan,
                            uav_measurements_.scan, T_icp, pointcloud_sigma_);

                        extraceptive_constraints_uav_.push_back(constraint_icp_uav);
                    };

            }

            prev_uav_measurements_ = uav_measurements_;
            uav_translation_ = uav_rotation_ = 0.0;
        }

        //////////////// MANAGE ENCOUNTERS ///////////////

        //Check if there is a new relative position available
        uwb_encounter_available_ = isRelativeTransformAvailable(current_time, latest_relative_pose_.header.stamp, global_opt_window_s_);

        if(uwb_encounter_available_){

            latest_relative_pose_SE3_ = transformSE3FromPoseMsg(latest_relative_pose_.pose.pose);
            //Unflatten matrix to extract the covariance
            for (size_t i = 0; i < 6; ++i) {
                for (size_t j = 0; j < 6; ++j) {
                    latest_relative_pose_cov_(i,j) = latest_relative_pose_.pose.covariance[i * 6 + j];
                }
            }
            
            MeasurementConstraint uwb_constraint;
            uwb_constraint.id_begin = agv_id_; //relates latest UAV and AGV nodes
            uwb_constraint.id_end = uav_id_;
            uwb_constraint.t_T_s = latest_relative_pose_SE3_;
            double covariance_multiplier = 1;
            uwb_constraint.covariance = reduce_covariance_matrix(latest_relative_pose_cov_);
            uwb_constraint.covariance*=covariance_multiplier;

            encounter_constraints_uwb_.push_back(uwb_constraint);

        }

        //Inter-robot ICP constraints

        icp_encounter_available_ = false;
        if (!uav_measurements_.scan->points.empty() &&
            !agv_measurements_.scan->points.empty()) {
                
                Eigen::Matrix4f T_icp = Eigen::Matrix4f::Identity();
                if(uwb_encounter_available_) T_icp = latest_relative_pose_SE3_.cast<float>().matrix();

                double fitness = 0.0;

                RCLCPP_WARN(this->get_logger(), "Computing ICP for inter-robot pairs index %d and %d", agv_id_, uav_id_);

                icp_encounter_available_ = run_icp(agv_measurements_.scan, uav_measurements_.scan, T_icp, fitness, icp_type_);

                if(icp_encounter_available_){
                    MeasurementConstraint inter_icp_constraint;
                    inter_icp_constraint.id_begin = agv_id_; //same id, relates node i of uav map, and node i of uav map
                    inter_icp_constraint.id_end = uav_id_;

                    //Transform to the robots body frame, using the corrected orthogonal rotation matrix
                    Sophus::SE3d T_icp_uav_agv = T_uav_lidar_ * Sophus::SE3f(T_icp).cast<double>() * T_agv_lidar_.inverse();

                    inter_icp_constraint.t_T_s = T_icp_uav_agv;

                    //log_transformation_matrix(inter_icp_constraint.t_T_s.matrix());

                    // Compute the ICP covariance from the scan pair.
                    inter_icp_constraint.covariance = computeICPCovariance(agv_measurements_.scan,
                        uav_measurements_.scan, T_icp, pointcloud_sigma_);
                    
                    encounter_constraints_pointcloud_.push_back(inter_icp_constraint);
                };

            // Convert the PCL point cloud to a ROS message
            sensor_msgs::msg::PointCloud2 source_cloud_msg, target_cloud_msg;
            pcl::toROSMsg(*agv_measurements_.scan, source_cloud_msg);
            pcl::toROSMsg(*uav_measurements_.scan, target_cloud_msg);

            // Create and populate the service request.
            auto request = std::make_shared<UpdatePointClouds::Request>();
            request->source_cloud = source_cloud_msg;
            request->target_cloud = target_cloud_msg;

            // Send the request asynchronously.
            auto future_result = pcl_visualizer_client_->async_send_request(
                request,
                [this](rclcpp::Client<UpdatePointClouds>::SharedFuture result) {
                    if (result.get()->success) {
                    RCLCPP_INFO(this->get_logger(), "Visualizer updated successfully.");
                    } else {
                    RCLCPP_WARN(this->get_logger(), "Visualizer update failed: %s", result.get()->message.c_str());
                    }
                });
    
        }
        
        bool window_initialized = agv_id_ >= min_keyframes_ && uav_id_ >= min_keyframes_;

        if(!window_initialized){
            RCLCPP_INFO(this->get_logger(), "Sliding window not yet full!");
            return;
        }

        //Update transforms after convergence
        if(run_posegraph_optimization(current_time, agv_map_, uav_map_, 
                                      proprioceptive_constraints_agv_, extraceptive_constraints_agv_,
                                      proprioceptive_constraints_uav_, extraceptive_constraints_uav_,
                                      encounter_constraints_uwb_, encounter_constraints_pointcloud_)){
            
            // ---------------- Publish All Optimized Poses ----------------
            RCLCPP_INFO(this->get_logger(), "Anchor node AGV:\n"
                        "[%f, %f, %f, %f]", anchor_node_agv_.state[0], anchor_node_agv_.state[1], anchor_node_agv_.state[2], anchor_node_agv_.state[3]);

            RCLCPP_INFO(this->get_logger(), "Anchor node UAV:\n"
                        "[%f, %f, %f, %f]", anchor_node_uav_.state[0], anchor_node_uav_.state[1], anchor_node_uav_.state[2], anchor_node_uav_.state[3]);

            anchor_node_agv_.pose = build_transformation_SE3(anchor_node_agv_.roll, anchor_node_agv_.pitch, anchor_node_agv_.state);
            anchor_node_uav_.pose = build_transformation_SE3(anchor_node_uav_.roll, anchor_node_uav_.pitch, anchor_node_uav_.state);

            // Create PoseWithCovarianceArray messages for AGV and UAV nodes.
            uwb_localization::msg::PoseWithCovarianceStampedArray agv_poses;
            uwb_localization::msg::PoseWithCovarianceStampedArray uav_poses;

            agv_poses.header.stamp = uav_poses.header.stamp = current_time;
            
            agv_poses.header.frame_id = odom_tf_agv_t_;
            uav_poses.header.frame_id = odom_tf_uav_t_;

            // Iterate over the global map and convert each state's optimized pose.
            for (const auto &kv : agv_map_) {
                
                const State &state = kv.second;
                Sophus::SE3d T = build_transformation_SE3(state.roll, state.pitch, state.state);
                geometry_msgs::msg::PoseWithCovarianceStamped pose = build_pose_msg(T, state.covariance, state.timestamp, odom_tf_agv_t_);

                if(kv.first == agv_id_){
                    publish_transform(T, current_time, odom_tf_agv_t_, eliko_frame_id_);
                    RCLCPP_INFO(this->get_logger(), "AGV Optimized pose (local):\n"
                    "[%f, %f, %f, %f]", state.state[0], state.state[1], state.state[2], state.state[3]);

                    Sophus::SE3d global_pose = anchor_node_agv_.pose * T;
                    publish_transform(global_pose, current_time, global_frame_graph_, eliko_frame_id_);
                }

                agv_poses.array.push_back(pose);
            }


            // Iterate over the global map and convert each state's optimized pose.
            for (const auto &kv : uav_map_) {             
                const State &state = kv.second;
                Sophus::SE3d T = build_transformation_SE3(state.roll, state.pitch, state.state);
                geometry_msgs::msg::PoseWithCovarianceStamped pose = build_pose_msg(T, state.covariance, state.timestamp, odom_tf_uav_t_);

                if(kv.first == uav_id_){
                    publish_transform(T, current_time, odom_tf_uav_t_, uav_frame_id_);
                    RCLCPP_INFO(this->get_logger(), "UAV Optimized pose (local):\n"
                    "[%f, %f, %f, %f]", state.state[0], state.state[1], state.state[2], state.state[3]);

                    Sophus::SE3d global_pose = anchor_node_uav_.pose * T;
                    publish_transform(global_pose, current_time, global_frame_graph_, uav_frame_id_);
                }

                uav_poses.array.push_back(pose);
            }

            geometry_msgs::msg::PoseWithCovarianceStamped anchor_agv = build_pose_msg(anchor_node_agv_.pose, anchor_node_agv_.covariance, anchor_node_agv_.timestamp, global_frame_graph_);
            geometry_msgs::msg::PoseWithCovarianceStamped anchor_uav = build_pose_msg(anchor_node_uav_.pose, anchor_node_uav_.covariance, anchor_node_uav_.timestamp, global_frame_graph_);

            poses_agv_publisher_->publish(agv_poses);
            poses_uav_publisher_->publish(uav_poses);

            anchor_agv_publisher_->publish(anchor_agv);
            anchor_uav_publisher_->publish(anchor_uav);
            
            first_optimization_done_ = true;
        }

        else{
            RCLCPP_INFO(this->get_logger(), "[Eliko global_opt node] Local optimizer did not converge");
        }


    }


    // Helper function to downsample a point cloud using a voxel grid filter.
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplePointCloud(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &input_cloud,
        float leaf_size = 0.05f) // default leaf size of 5cm
    {
        // // Create a VoxelGrid filter and set the input cloud.
        // pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        // voxel_filter.setInputCloud(input_cloud);
        // voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);

        // // Create a new point cloud to hold the filtered data.
        // pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        // voxel_filter.filter(*output_cloud);

        //Use small_gicp library
        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud = voxelgrid_sampling_omp(*input_cloud, leaf_size);

        RCLCPP_DEBUG(this->get_logger(), "Pointcloud downsampled to %zu points", output_cloud->points.size());

        return output_cloud;
    }


    // Helper function to compute the covariance for the relative odometry transform.
    // state_target: new measurement (target), state_source: previous measurement (source)
    // cov_target and cov_source are the 4x4 covariance matrices of the corresponding states.
    Eigen::Matrix4d computeRelativeOdometryCovariance(
        const Sophus::SE3d &pose_target,
        const Sophus::SE3d &pose_source,
        const Eigen::Matrix<double, 6, 6> &cov_target,
        const Eigen::Matrix<double, 6, 6> &cov_source)
    {
        
        Eigen::Matrix3d R_target = pose_target.rotationMatrix();  // or T.so3().matrix()
        Eigen::Vector3d t_target = pose_target.translation();
        Eigen::Vector3d t_source = pose_source.translation();
        // Compute Euler angles in ZYX order: [yaw, pitch, roll]
        Eigen::Vector3d euler_target = R_target.eulerAngles(2, 1, 0);
        // Extract yaw from the target state.
        double theta_t = euler_target[0];

        double dx = t_source[0] - t_target[0];
        double dy = t_source[1] - t_target[1];

        // Jacobian with respect to the source (old) state.
        Eigen::Matrix4d J_s = Eigen::Matrix4d::Zero();
        J_s(0, 0) = std::cos(theta_t);
        J_s(0, 1) = std::sin(theta_t);
        J_s(1, 0) = -std::sin(theta_t);
        J_s(1, 1) = std::cos(theta_t);
        J_s(2, 2) = 1.0;
        J_s(3, 3) = 1.0;

        // Jacobian with respect to the target (new) state.
        Eigen::Matrix4d J_t = Eigen::Matrix4d::Zero();
        J_t(0, 0) = -std::cos(theta_t);
        J_t(0, 1) = -std::sin(theta_t);
        J_t(1, 0) = std::sin(theta_t);
        J_t(1, 1) = -std::cos(theta_t);
        J_t(2, 2) = -1.0;
        J_t(3, 3) = -1.0;
        // Derivatives with respect to theta_t:
        J_t(0, 3) = -(-std::sin(theta_t) * dx + std::cos(theta_t) * dy);
        J_t(1, 3) = -(-std::cos(theta_t) * dx - std::sin(theta_t) * dy);


        //Reduce the covariances to 4x4 -- remove rows and columns 3,4
        Eigen::Matrix4d cov_source_reduced = reduce_covariance_matrix(cov_source);
        Eigen::Matrix4d cov_target_reduced = reduce_covariance_matrix(cov_target);

        // Propagate the covariance.
        Eigen::Matrix4d cov_rel = J_s * cov_source_reduced * J_s.transpose() +
                                J_t * cov_target_reduced * J_t.transpose();
        return cov_rel;
    }



        // Helper function to approximate the ICP covariance.
    // 'source' and 'target' are the downsampled point clouds used for ICP.
    // 'transformation' is the final 4x4 transform from ICP.
    // sensor_variance is a tuning parameter (e.g., 0.01).
    Eigen::Matrix4d computeICPCovariance(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &source,
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &target,
        const Eigen::Matrix4f &transformation,
        double sensor_variance = 0.01)
    {
        pcl::search::KdTree<pcl::PointXYZ> tree;
        tree.setInputCloud(target);
        
        Eigen::Matrix4d H_total = Eigen::Matrix4d::Zero();
        int count = 0;

        // Extract an approximate yaw from the transformation.
        double yaw = std::atan2(transformation(1, 0), transformation(0, 0));

        for (size_t i = 0; i < source->points.size(); ++i)
        {
            const pcl::PointXYZ &p = source->points[i];
            Eigen::Vector4f p_h(p.x, p.y, p.z, 1.0f);
            Eigen::Vector4f p_trans = transformation * p_h;

            pcl::PointXYZ p_trans_p;
            p_trans_p.x = p_trans[0];
            p_trans_p.y = p_trans[1];
            p_trans_p.z = p_trans[2];

            std::vector<int> indices(1);
            std::vector<float> sqr_dists(1);
            if (tree.nearestKSearch(p_trans_p, 1, indices, sqr_dists) > 0)
            {
                const pcl::PointXYZ &q = target->points[indices[0]];
                // Compute the error (not used further here, but could be logged if desired)
                Eigen::Vector3d e(q.x - p_trans[0], q.y - p_trans[1], q.z - p_trans[2]);

                // Build the Jacobian J_i (3x4) for the point.
                Eigen::Matrix<double, 3, 4> J = Eigen::Matrix<double, 3, 4>::Zero();
                // Derivative with respect to translation (first 3 columns): identity.
                J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                // Derivative with respect to yaw (4th column):
                double dxdtheta = -std::sin(yaw) * p.x - std::cos(yaw) * p.y;
                double dydtheta =  std::cos(yaw) * p.x - std::sin(yaw) * p.y;
                J(0, 3) = dxdtheta;
                J(1, 3) = dydtheta;
                // For the z component, yaw has no effect (J(2,3)=0).

                H_total += J.transpose() * J;
                count++;
            }
        }
        if (count > 0)
        {
            H_total /= static_cast<double>(count);
        }
        else
        {
            H_total = Eigen::Matrix4d::Identity();
        }
        // Approximate covariance from the (averaged) Hessian.
        Eigen::Matrix4d cov_icp = sensor_variance * H_total.inverse();
        return cov_icp;
    }


    bool run_icp(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &source_cloud,
             const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &target_cloud, Eigen::Matrix4f &transformation, double &fitness, const int &icp_type) const {
            
        if(icp_type == 2) {

            // RegistrationPCL is derived from pcl::Registration and has mostly the same interface as pcl::GeneralizedIterativeClosestPoint.
            RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> reg;
            reg.setNumThreads(4);
            
            reg.setCorrespondenceRandomness(20);
            reg.setMaxCorrespondenceDistance(2.5*pointcloud_sigma_);
            reg.setVoxelResolution(pointcloud_sigma_);
            // Set the maximum number of iterations (criterion 1)
            reg.setMaximumIterations (50);
            // Set the transformation epsilon (criterion 2)
            reg.setTransformationEpsilon (1e-4);
            // Set the euclidean distance difference epsilon (criterion 3)
            reg.setEuclideanFitnessEpsilon (2.5*pointcloud_sigma_);

            // Set input point clouds.
            reg.setInputSource(source_cloud);
            reg.setInputTarget(target_cloud);

            // Align point clouds.
            auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            reg.align(*aligned, transformation);

            // // Swap source and target and align again.
            // // This is useful when you want to re-use preprocessed point clouds for successive registrations (e.g., odometry estimation).
            // reg.swapSourceAndTarget();
            // reg.align(*aligned, transformation);
            
            if (!reg.hasConverged()) {
                RCLCPP_WARN(this->get_logger(), "GICP did not converge.");
                return false;
            }
            
            transformation = reg.getFinalTransformation();
            fitness = reg.getFitnessScore();

            //RCLCPP_INFO(this->get_logger(), "ICP converged with score: %f", fitness);

            return true;
        }
        
        else if(icp_type == 1){

            // Compute normals and build PointNormal clouds.
            pcl::PointCloud<pcl::PointNormal>::Ptr source_with_normals = computePointNormalCloud(source_cloud, 0.1f);
            pcl::PointCloud<pcl::PointNormal>::Ptr target_with_normals = computePointNormalCloud(target_cloud, 0.1f);

            // Set up the ICP object using point-to-plane error metric.
            pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
            icp.setTransformationEstimation(
                typename pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal>::Ptr(
                    new pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal>));
            
            icp.setInputSource(source_with_normals);
            icp.setInputTarget(target_with_normals);
    
            // Optionally, you can tune ICP parameters (e.g., maximum iterations, convergence criteria, etc.)
            // Set the max correspondence distance to 5cm (e.g., correspondences with higher
            // distances will be ignored)
            icp.setMaxCorrespondenceDistance (2.5*pointcloud_sigma_);
            // Set the maximum number of iterations (criterion 1)
            icp.setMaximumIterations (50);
            // Set the transformation epsilon (criterion 2)
            icp.setTransformationEpsilon (1e-4);
            // Set the euclidean distance difference epsilon (criterion 3)
            icp.setEuclideanFitnessEpsilon (2.5*pointcloud_sigma_);
            
            pcl::PointCloud<pcl::PointNormal> aligned_cloud;
            icp.align(aligned_cloud, transformation);

            if (!icp.hasConverged()) {
                //RCLCPP_WARN(this->get_logger(), "ICP did not converge.");
                return false;
            }
                   
            // Get the transformation matrix
            transformation = icp.getFinalTransformation();
            fitness = icp.getFitnessScore();

            //RCLCPP_INFO(this->get_logger(), "ICP converged with score: %f", fitness);

            return true;
        }
        
        else {
            // Perform Point-to-Point ICP
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setInputSource(source_cloud);
            icp.setInputTarget(target_cloud);

            // Set the max correspondence distance to 5cm (e.g., correspondences with higher
            // distances will be ignored)
            icp.setMaxCorrespondenceDistance (2.5*pointcloud_sigma_);
            // Set the maximum number of iterations (criterion 1)
            icp.setMaximumIterations (50);
            // Set the transformation epsilon (criterion 2)
            icp.setTransformationEpsilon (1e-4);
            // Set the euclidean distance difference epsilon (criterion 3)
            icp.setEuclideanFitnessEpsilon (2.5*pointcloud_sigma_);

            pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            icp.align(*aligned_cloud, transformation);

            if (!icp.hasConverged()) {
                //RCLCPP_WARN(this->get_logger(), "ICP did not converge.");
                return false;
            }

            // Get the transformation matrix
            transformation = icp.getFinalTransformation();
            fitness = icp.getFitnessScore();

            //RCLCPP_INFO(this->get_logger(), "ICP converged with score: %f", fitness);

            return true;
        }

    }


    pcl::PointCloud<pcl::PointNormal>::Ptr computePointNormalCloud(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
        float radius_search = 0.1f) const // you may adjust the search radius
    {
        // Estimate normals.
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        ne.setRadiusSearch(radius_search);
        ne.compute(*normals);
        
        // Combine the XYZ data and the normals into a PointNormal cloud.
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
        pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
        
        return cloud_with_normals;
    }


    //Write to display a 4x4 transformation matrix
    void log_transformation_matrix(const Eigen::Matrix4d &T)   {

        RCLCPP_INFO(this->get_logger(), "T:\n"
            "[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]",
            T(0, 0), T(0, 1), T(0, 2), T(0, 3),
            T(1, 0), T(1, 1), T(1, 2), T(1, 3),
            T(2, 0), T(2, 1), T(2, 2), T(2, 3),
            T(3, 0), T(3, 1), T(3, 2), T(3, 3));                 

    }


    // Helper function to add measurement (proprioceptive or extraceptive) constraints.
    bool addMeasurementConstraints(ceres::Problem &problem, 
        const VectorOfConstraints &constraints,
        MapOfStates &map, 
        const rclcpp::Time &current_time, 
        double window_size,
        ceres::LossFunction *loss){
        
        bool constraint_available = false;

        for (const auto &constraint : constraints) {
            bool source_fixed = first_optimization_done_ && isNodeFixed(current_time, constraint.id_begin, map, window_size);
            bool target_fixed = first_optimization_done_ && isNodeFixed(current_time, constraint.id_end, map, window_size);
            if (source_fixed && target_fixed) continue;

            if(!constraint_available) constraint_available = true;

            State &state_i = map[constraint.id_begin];
            State &state_j = map[constraint.id_end];
            ceres::CostFunction *cost = MeasurementResidual::Create(constraint.t_T_s, constraint.covariance, 
                                            state_i.roll, state_i.pitch, 
                                            state_j.roll, state_j.pitch);
            problem.AddResidualBlock(cost, loss, state_i.state.data(), state_j.state.data());
            RCLCPP_WARN(this->get_logger(), "Adding measurement constraint between nodes %d and %d", 
            constraint.id_begin, constraint.id_end);
        }

        return constraint_available;
    }


    // Helper function to add encounter (anchor) constraints.
    bool addEncounterConstraints(ceres::Problem &problem, 
        const VectorOfConstraints &constraints,
        MapOfStates &source_map, MapOfStates &target_map,
        const rclcpp::Time &current_time, 
        double window_size,
        ceres::LossFunction *loss){

        bool constraint_available = false;

        for (const auto &constraint : constraints) {
            
            bool source_fixed = first_optimization_done_ && isNodeFixed(current_time, constraint.id_begin, source_map, window_size);
            bool target_fixed = first_optimization_done_ && isNodeFixed(current_time, constraint.id_end, target_map, window_size);
            
            if (source_fixed && target_fixed) continue;
            
            if(!constraint_available) constraint_available = true;

            State &state_i = source_map[constraint.id_begin];
            State &state_j = target_map[constraint.id_end];
            ceres::CostFunction *cost = AnchorResidual::Create(constraint.t_T_s, constraint.covariance, 
                                        state_i.roll, state_i.pitch, 
                                        state_j.roll, state_j.pitch,
                                        anchor_node_uav_.roll, anchor_node_uav_.pitch,
                                        anchor_node_agv_.roll, anchor_node_agv_.pitch);
            problem.AddResidualBlock(cost, loss, state_i.state.data(), state_j.state.data(), 
                anchor_node_uav_.state.data(), anchor_node_agv_.state.data());
            // Optionally, add a prior residual for the anchor node.
            ceres::CostFunction *prior_cost_anchor = PriorResidual::Create(prior_anchor_.pose, 
                                                    anchor_node_agv_.roll, anchor_node_agv_.pitch, 
                                                    prior_anchor_.covariance);
            problem.AddResidualBlock(prior_cost_anchor, nullptr, anchor_node_agv_.state.data());
            RCLCPP_WARN(this->get_logger(), "Adding encounter constraint between nodes %d and %d", 
            constraint.id_begin, constraint.id_end);
        }

        return constraint_available;
    }


    void publish_transform(const Sophus::SE3d& T, const rclcpp::Time &current_time,
                           const std::string &frame_id, const std::string &child_frame_id) {


        geometry_msgs::msg::TransformStamped T_msg;
        T_msg.header.stamp = current_time;
        T_msg.header.frame_id = frame_id;  // Adjust frame_id as needed
        T_msg.child_frame_id = child_frame_id;            // Adjust child_frame_id as needed

         // Extract translation and rotation from the Sophus SE3 object.
        Eigen::Vector3d t = T.translation();

        // Extract translation
        T_msg.transform.translation.x = t.x();
        T_msg.transform.translation.y = t.y();
        T_msg.transform.translation.z = t.z();

        Eigen::Matrix3d R = T.rotationMatrix();
        Eigen::Quaterniond q(R);

        T_msg.transform.rotation.x = q.x();
        T_msg.transform.rotation.y = q.y();
        T_msg.transform.rotation.z = q.z();
        T_msg.transform.rotation.w = q.w();


        // Broadcast the inverse transform
        tf_broadcaster_->sendTransform(T_msg);
    }

    geometry_msgs::msg::PoseWithCovarianceStamped build_pose_msg(const Sophus::SE3d &T, const Eigen::Matrix4d& cov4, 
        const rclcpp::Time &current_time, const std::string &frame_id) {

        geometry_msgs::msg::PoseWithCovarianceStamped p_msg;
        p_msg.header.stamp = current_time;
        p_msg.header.frame_id = frame_id;
        
        // Extract translation and rotation from the Sophus SE3 object.
        Eigen::Vector3d t = T.translation();
        p_msg.pose.pose.position.x = t.x();
        p_msg.pose.pose.position.y = t.y();
        p_msg.pose.pose.position.z = t.z();

        Eigen::Matrix3d R = T.rotationMatrix();
        Eigen::Quaterniond q(R);
        p_msg.pose.pose.orientation.x = q.x();
        p_msg.pose.pose.orientation.y = q.y();
        p_msg.pose.pose.orientation.z = q.z();
        p_msg.pose.pose.orientation.w = q.w();

        // Build a 6x6 covariance matrix.
        Eigen::Matrix<double, 6, 6> cov6 = Eigen::Matrix<double, 6, 6>::Zero();

        // Copy translation covariance.
        cov6.block<3, 3>(0, 0) = cov4.block<3, 3>(0, 0);

        // Copy cross-covariance between translation and yaw.
        cov6.block<3, 1>(0, 5) = cov4.block<3, 1>(0, 3);
        cov6.block<1, 3>(5, 0) = cov4.block<1, 3>(3, 0);

        // For roll and pitch (which are not estimated), set very small variances.
        cov6(3, 3) = 1e-6;  // roll
        cov6(4, 4) = 1e-6;  // pitch

        // Set yaw variance.
        cov6(5, 5) = cov4(3, 3);

        // (Optionally, if cov4 contains nonzero cross-covariances between translation and yaw,
        // they are already copied above. The remaining cross terms (between roll/pitch and others)
        // remain zero.)

        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 6; ++j) {
            p_msg.pose.covariance[i * 6 + j] = cov6(i, j);
            }
        }

        return p_msg;
    }

    Sophus::SE3d build_transformation_SE3(double roll, double pitch, const Eigen::Vector4d& s) {
        Eigen::Vector3d t(s[0], s[1], s[2]);  // Use Vector3d instead of an incorrect Matrix type.
        Eigen::Matrix3d R = (Eigen::AngleAxisd(s[3], Eigen::Vector3d::UnitZ()) *
                             Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX())).toRotationMatrix();
        return Sophus::SE3d(R, t);
    }


    Eigen::Vector4d state_from_transformation_SE3(const Sophus::SE3d &T) {
        
        Eigen::Matrix3d R = T.rotationMatrix();  // or T.so3().matrix()
        Eigen::Vector3d t = T.translation();
        // Compute Euler angles in ZYX order: [yaw, pitch, roll]
        Eigen::Vector3d euler_zyx = R.eulerAngles(2, 1, 0);

        return Eigen::Vector4d(t[0], t[1], t[2], euler_zyx[0]);
    }

    // Convert Pose from ROS msg to SE3
    Sophus::SE3d transformSE3FromPoseMsg(const geometry_msgs::msg::Pose& T_msg) {
        
        // Extract the translation vector.
        Eigen::Vector3d t(
            T_msg.position.x,
            T_msg.position.y,
            T_msg.position.z);

        // Build the quaternion (make sure the order is correct: w, x, y, z).
        Eigen::Quaterniond q(
            T_msg.orientation.w,
            T_msg.orientation.x,
            T_msg.orientation.y,
            T_msg.orientation.z);

        q.normalize();  // Ensure the quaternion is normalized

        // Construct the Sophus::SE3d object.
        Sophus::SE3d T(q, t);

        return T;
    }

    // Convert Pose from ROS msg to SE3
    Sophus::SE3d transformSE3FromMsg(const geometry_msgs::msg::TransformStamped& T_msg) {
        
        // Extract the translation vector.
        Eigen::Vector3d t(
            T_msg.transform.translation.x,
            T_msg.transform.translation.y,
            T_msg.transform.translation.z);

        // Build the quaternion (make sure the order is correct: w, x, y, z).
        Eigen::Quaterniond q(
            T_msg.transform.rotation.w,
            T_msg.transform.rotation.x,
            T_msg.transform.rotation.y,
            T_msg.transform.rotation.z);

        q.normalize();  // Ensure the quaternion is normalized

        // Construct the Sophus::SE3d object.
        Sophus::SE3d T(q, t);

        return T;
    }
    

    Eigen::Vector4d transformSE3ToState(const Sophus::SE3d& T) {
        // Extract translation.
        Eigen::Vector3d t = T.translation();
        
        // Extract rotation matrix.
        Eigen::Matrix3d R = T.rotationMatrix();
        
        // Compute yaw from the rotation matrix.
        double yaw = std::atan2(R(1,0), R(0,0));
        
        // Pack x, y, z and yaw into a 4D state vector.
        Eigen::Vector4d state;
        state << t[0], t[1], t[2], yaw;
        
        return state;
    }


    //Reduce a 6x6 covariance matrix to a 4x4 one, removing roll and pitch elements
    Eigen::Matrix4d reduce_covariance_matrix(const Eigen::Matrix<double, 6, 6> &cov){

        std::vector<int> indices = {0, 1, 2, 5};
        Eigen::Matrix4d cov_reduced;
        for (size_t i = 0; i < indices.size(); ++i) {
            for (size_t j = 0; j < indices.size(); ++j) {
                cov_reduced(i, j) = cov(indices[i], indices[j]);
            }
        }
        
        return cov_reduced;
    }

    bool isNodeFixed(const rclcpp::Time &current_time, const int node_id, MapOfStates &map, const double &window_size){
        return (current_time - map[node_id].timestamp).seconds() > window_size;
    }

    bool isRelativeTransformAvailable(const rclcpp::Time &current_time, const rclcpp::Time &latest_relative_time, const double &window_size){
        return ((current_time - latest_relative_time).seconds() <= window_size);
    }

    // ---------------- Pose Graph Optimization -------------------
    //
    // In this function we build a Ceres problem that fuses:
    // - A prior on each node and the previous.
    // - Inter-robot UWB factor linking nodes based on relative transform estimation input
    // - Intra-robot ICP and odometry factors linking consecutive nodes.
    // - Inter-robot ICP factors linking nodes from the two robots.
    
    bool run_posegraph_optimization(const rclcpp::Time &current_time,
                                    MapOfStates &agv_map, MapOfStates &uav_map,
                                    const VectorOfConstraints &proprioceptive_constraints_agv, const VectorOfConstraints &extraceptive_constraints_agv,
                                    const VectorOfConstraints &proprioceptive_constraints_uav, const VectorOfConstraints &extraceptive_constraints_uav,
                                    const VectorOfConstraints &encounter_constraints_uwb, const VectorOfConstraints &encounter_constraints_pointcloud) {

        ceres::Problem problem;

        ceres::Manifold* state_manifold_3d = new StateManifold3D;

        // Define a robust kernel
        double huber_threshold = 2.5; // = residuals higher than 2.5 times sigma are outliers
        ceres::LossFunction* robust_loss = new ceres::HuberLoss(huber_threshold);

        //Add the anchor nodes
        problem.AddParameterBlock(anchor_node_uav_.state.data(), 4);
        problem.AddParameterBlock(anchor_node_agv_.state.data(), 4);
        problem.SetManifold(anchor_node_uav_.state.data(), state_manifold_3d);
        problem.SetManifold(anchor_node_agv_.state.data(), state_manifold_3d);

        //remove the gauge freedom (i.e. the fact that an overall rigid body transform can be added to all poses without changing the relative errors).
        //anchor -freeze- the first node, and freeze the part of the node outside the sliding window

        for (auto& kv : agv_map) {
            State& state = kv.second;
            problem.AddParameterBlock(state.state.data(), 4); 
            problem.SetManifold(state.state.data(), state_manifold_3d);
            // For a starting node: only fix if it has been optimized before AND is outside the window.
            if (first_optimization_done_ && isNodeFixed(current_time, kv.first, agv_map, global_opt_window_s_)) {
                problem.SetParameterBlockConstant(state.state.data());
            } else {
                RCLCPP_WARN(this->get_logger(), "Optimizing for AGV node %d", kv.first);
            }
   
        }

        for (auto& kv : uav_map) {
            State& state = kv.second;
            problem.AddParameterBlock(state.state.data(), 4); 
            problem.SetManifold(state.state.data(), state_manifold_3d);
            // For a starting node: only fix if it has been optimized before AND is outside the window.
            if (first_optimization_done_ && isNodeFixed(current_time, kv.first, uav_map, global_opt_window_s_)) {
                problem.SetParameterBlockConstant(state.state.data());
            } else {
                RCLCPP_WARN(this->get_logger(), "Optimizing for UAV node %d", kv.first);
            }
   
        }

       // For the starting nodes, add the prior residual blocks if they are not yet optimized and fixed.
        if (!first_optimization_done_ || !isNodeFixed(current_time, 0, agv_map, global_opt_window_s_)) {
            auto first_node_agv = agv_map[0];
            ceres::CostFunction* prior_cost_agv = PriorResidual::Create(prior_agv_.pose, first_node_agv.roll, first_node_agv.pitch, prior_agv_.covariance);
            problem.AddResidualBlock(prior_cost_agv, nullptr, first_node_agv.state.data());
        }

        if (!first_optimization_done_ || !isNodeFixed(current_time, 0, uav_map, global_opt_window_s_)) {    
            auto first_node_uav = uav_map[0];
            ceres::CostFunction* prior_cost_uav = PriorResidual::Create(prior_uav_.pose, first_node_uav.roll, first_node_uav.pitch, prior_uav_.covariance);
            problem.AddResidualBlock(prior_cost_uav, nullptr, first_node_uav.state.data());
        }

        // Add measurement constraints for each set.
        bool proprioceptive_constraints_agv_available = addMeasurementConstraints(problem, proprioceptive_constraints_agv, agv_map, current_time, global_opt_window_s_, robust_loss);
        bool proprioceptive_constraints_uav_available = addMeasurementConstraints(problem, proprioceptive_constraints_uav, uav_map, current_time, global_opt_window_s_, robust_loss);
        bool extraceptive_constraints_agv_available = addMeasurementConstraints(problem, extraceptive_constraints_agv, agv_map, current_time, global_opt_window_s_, robust_loss);
        bool extraceptive_constraints_uav_available = addMeasurementConstraints(problem, extraceptive_constraints_uav, uav_map, current_time, global_opt_window_s_, robust_loss);

        // Add encounter constraints.
        bool encounter_uwb_available = addEncounterConstraints(problem, encounter_constraints_uwb, agv_map, uav_map, current_time, global_opt_window_s_, robust_loss);
        bool encounter_pointcloud_available = addEncounterConstraints(problem, encounter_constraints_pointcloud, agv_map, uav_map, current_time, global_opt_window_s_, robust_loss);
        // (You may want to update a flag if any encounter constraint was added.)

        if(!encounter_uwb_available && !encounter_pointcloud_available){
            // Freeze the anchor nodes when the constraints available do not involve encounters
            problem.SetParameterBlockConstant(anchor_node_uav_.state.data());
            problem.SetParameterBlockConstant(anchor_node_agv_.state.data());
        }
        
        // Configure solver options
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // ceres::SPARSE_NORMAL_CHOLESKY,  ceres::DENSE_QR
        options.num_threads = 4;
        // Logging
        options.minimizer_progress_to_stdout = false;

        // Solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        RCLCPP_INFO(this->get_logger(), summary.BriefReport().c_str());

        // Notify and update values if optimization converged
        if (summary.termination_type == ceres::CONVERGENCE){

            // --- Compute Covariances for the Last Nodes ---
            // Here we compute the covariance for the last AGV and UAV states.
            ceres::Covariance::Options cov_options;
            ceres::Covariance covariance(cov_options);

            std::vector<std::pair<const double*, const double*>> covariance_blocks;
            for (auto& kv : agv_map) {
                // kv.first is the node's unique ID and kv.second is the State.
                covariance_blocks.emplace_back(kv.second.state.data(), kv.second.state.data());
            }

            for (auto& kv : uav_map) {
                // kv.first is the node's unique ID and kv.second is the State.
                covariance_blocks.emplace_back(kv.second.state.data(), kv.second.state.data());
            }

            covariance_blocks.emplace_back(anchor_node_agv_.state.data(), anchor_node_agv_.state.data());
            covariance_blocks.emplace_back(anchor_node_uav_.state.data(), anchor_node_uav_.state.data());

            if (covariance.Compute(covariance_blocks, &problem)) {
                // Update each node's covariance in the unified map.
                for (auto& kv : agv_map) {
                    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
                    covariance.GetCovarianceBlock(kv.second.state.data(), kv.second.state.data(), cov.data());
                    kv.second.covariance = cov;
                }

                for (auto& kv : uav_map) {
                    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
                    covariance.GetCovarianceBlock(kv.second.state.data(), kv.second.state.data(), cov.data());
                    kv.second.covariance = cov;
                }

                Eigen::Matrix4d cov_anchor_agv = Eigen::Matrix4d::Zero();
                covariance.GetCovarianceBlock(anchor_node_agv_.state.data(), anchor_node_agv_.state.data(), cov_anchor_agv.data());
                anchor_node_agv_.covariance = cov_anchor_agv;

                Eigen::Matrix4d cov_anchor_uav = Eigen::Matrix4d::Zero();
                covariance.GetCovarianceBlock(anchor_node_uav_.state.data(), anchor_node_uav_.state.data(), cov_anchor_uav.data());
                anchor_node_uav_.covariance = cov_anchor_uav;

            } else {
                RCLCPP_WARN(this->get_logger(), "Failed to compute covariances.");
                // Set default covariances if needed.
                for (auto& kv : agv_map) {
                    kv.second.covariance = Eigen::Matrix4d::Identity();
                }

                for (auto& kv : uav_map) {
                    kv.second.covariance = Eigen::Matrix4d::Identity();
                }

                anchor_node_agv_.covariance = Eigen::Matrix4d::Identity();
                anchor_node_uav_.covariance = Eigen::Matrix4d::Identity();
            }

            return true;

        } else {

                RCLCPP_WARN(this->get_logger(), "Failed to converge.");
                
                return false;
        }
                                    
        return false; 
            
    }




    //Prior Residual

    struct PriorResidual {
        PriorResidual(const Sophus::SE3d& prior_T, const double &roll, const double &pitch, const Eigen::Matrix4d& prior_cov)
            : prior_T_(prior_T), roll_(roll), pitch_(pitch), prior_cov_(prior_cov) {}
    
        template <typename T>
        bool operator()(const T* const state, T* residual) const {
            
            Sophus::SE3<T> SE3_pred = buildTransformationSE3(state, roll_, pitch_);
            Sophus::SE3<T> T_meas = prior_T_.template cast<T>();
    
            // Compute the error transformation: T_err = T_meas^{-1} * T_pred.
            Sophus::SE3<T> T_err = T_meas.inverse() * SE3_pred;
            
            // Compute the full 6-vector logarithm (xi = [rho; phi]),
            // where phi is the rotation vector.
            Eigen::Matrix<T,6,1> xi = T_err.log();
    
            // Project the 6-vector error onto the 4-DOF space:
            // Keep the three translation components and only the z component of the rotation.
            Eigen::Matrix<T,4,1> error_vec;
            error_vec.template head<3>() = xi.template head<3>();
            error_vec[3] = xi[5];  // use the z-component (yaw) of the rotation error
    
            // Scale by the square root of the inverse covariance matrix
            Eigen::LLT<Eigen::Matrix4d> chol(prior_cov_);
            Eigen::Matrix4d sqrt_inv_covariance = Eigen::Matrix4d(chol.matrixL().transpose()).inverse();
            //Eigen::Matrix4d sqrt_covariance = Eigen::Matrix4d(chol.matrixL());
    
            Eigen::Matrix<T, 4, 1> weighted_residual = sqrt_inv_covariance.cast<T>() * error_vec;
    
            // Assign to residual
            residual[0] = weighted_residual[0];
            residual[1] = weighted_residual[1];
            residual[2] = weighted_residual[2];
            residual[3] = weighted_residual[3];
    
            return true;
        }
    
        static ceres::CostFunction* Create(const Sophus::SE3d& prior_T, const double& roll, const double &pitch, const Eigen::Matrix4d& prior_cov) {
            return new ceres::AutoDiffCostFunction<PriorResidual, 4, 4>(
                new PriorResidual(prior_T, roll, pitch, prior_cov));
        }
    
    private:
    
        template <typename T>
        Sophus::SE3<T> buildTransformationSE3(const T* state, double roll, double pitch) const {
            // Extract translation
            Eigen::Matrix<T, 3, 1> t;
            t << state[0], state[1], state[2];
            // Build rotation from yaw, with fixed roll and pitch
            Eigen::Matrix<T, 3, 3> R = (Eigen::AngleAxis<T>(state[3], Eigen::Matrix<T, 3, 1>::UnitZ()) *
                                        Eigen::AngleAxis<T>(T(pitch), Eigen::Matrix<T, 3, 1>::UnitY()) *
                                        Eigen::AngleAxis<T>(T(roll),  Eigen::Matrix<T, 3, 1>::UnitX())).toRotationMatrix();
            // Return the Sophus SE3 object
            return Sophus::SE3<T>(R, t);
        }
    
        const Sophus::SE3d prior_T_;
        const double roll_, pitch_;
        const Eigen::Matrix4d prior_cov_;
    };

    /////////////////////////////////



    struct MeasurementResidual {
        MeasurementResidual(const Sophus::SE3d& T_meas, const Eigen::Matrix4d& cov, double source_roll, double source_pitch, double target_roll, double target_pitch)
            : T_meas_(T_meas), cov_(cov), source_roll_(source_roll), source_pitch_(source_pitch), target_roll_(target_roll), target_pitch_(target_pitch) {}
    
        template <typename T>
        bool operator()(const T* const source_state, const T* const target_state, T* residual) const {
            // Build homogeneous transforms from the state vectors.
            Sophus::SE3<T> T_s = buildTransformationSE3(source_state, source_roll_, source_pitch_);
            Sophus::SE3<T> T_t = buildTransformationSE3(target_state, target_roll_, target_pitch_);
            
            // Compute the relative transform from AGV to UAV:
            Sophus::SE3<T> SE3_pred = T_t.inverse() * T_s;
            Sophus::SE3<T> SE3_meas = T_meas_.template cast<T>();
    
            // Compute the error transformation: T_err = T_meas^{-1} * T_pred.
            Sophus::SE3<T> T_err = SE3_meas.inverse() * SE3_pred;
            
            // Compute the full 6-vector logarithm (xi = [rho; phi]),
            // where phi is the rotation vector.
            Eigen::Matrix<T,6,1> xi = T_err.log();
    
            // Project the 6-vector error onto the 4-DOF space:
            // Keep the three translation components and only the z component of the rotation.
            Eigen::Matrix<T,4,1> error_vec;
            error_vec.template head<3>() = xi.template head<3>();
            error_vec[3] = xi[5];  // use the z-component (yaw) of the rotation error
    
            // Scale by the square root of the inverse covariance matrix
            Eigen::LLT<Eigen::Matrix4d> chol(cov_);
            Eigen::Matrix4d sqrt_inv_covariance = Eigen::Matrix4d(chol.matrixL().transpose()).inverse();
            
            Eigen::Matrix<T, 4, 1> weighted_residual = sqrt_inv_covariance.cast<T>() * error_vec;
    
            // Assign to residual
            residual[0] = weighted_residual[0];
            residual[1] = weighted_residual[1];
            residual[2] = weighted_residual[2];
            residual[3] = weighted_residual[3];
    
            return true;
        }
    
        static ceres::CostFunction* Create(const Sophus::SE3d& T_meas, const Eigen::Matrix4d& cov, double source_roll, double source_pitch, double target_roll, double target_pitch) {
            return new ceres::AutoDiffCostFunction<MeasurementResidual, 4, 4, 4>(
                new MeasurementResidual(T_meas, cov, source_roll, source_pitch, target_roll, target_pitch));
        }
    
        private:
            template <typename T>
            Sophus::SE3<T> buildTransformationSE3(const T* state, double roll, double pitch) const {
                // Extract translation
                Eigen::Matrix<T, 3, 1> t;
                t << state[0], state[1], state[2];
                // Build rotation from yaw, with fixed roll and pitch
                Eigen::Matrix<T, 3, 3> R = (Eigen::AngleAxis<T>(state[3], Eigen::Matrix<T, 3, 1>::UnitZ()) *
                                            Eigen::AngleAxis<T>(T(pitch), Eigen::Matrix<T, 3, 1>::UnitY()) *
                                            Eigen::AngleAxis<T>(T(roll),  Eigen::Matrix<T, 3, 1>::UnitX())).toRotationMatrix();
                // Return the Sophus SE3 object
                return Sophus::SE3<T>(R, t);
            }
    
        const Sophus::SE3d T_meas_;
        const Eigen::Matrix4d cov_;
        const double source_roll_, source_pitch_;
        const double target_roll_, target_pitch_;
    
    };


    struct AnchorResidual {
        AnchorResidual(const Sophus::SE3d &T_meas, const Eigen::Matrix4d &cov, 
                        double source_roll, double source_pitch, 
                        double target_roll, double target_pitch,
                        double anchor_roll_uav, double anchor_pitch_uav,
                        double anchor_roll_agv, double anchor_pitch_agv)
          : T_meas_(T_meas), cov_(cov), source_roll_(source_roll), source_pitch_(source_pitch), 
            target_roll_(target_roll), target_pitch_(target_pitch), 
            anchor_roll_uav_(anchor_roll_uav), anchor_pitch_uav_(anchor_pitch_uav),
            anchor_roll_agv_(anchor_roll_agv), anchor_pitch_agv_(anchor_pitch_agv) {}
      
        template <typename T>
        bool operator()(const T* const source_state, const T* const target_state, const T* const T_anchor_uav, const T* const T_anchor_agv, T* residual) const {
        
          // Build homogeneous transforms from the state vectors.
          Sophus::SE3<T> T_s = buildTransformationSE3(source_state, source_roll_, source_pitch_);
          Sophus::SE3<T> T_t = buildTransformationSE3(target_state, target_roll_, target_pitch_);

          // Build the extra transformation from UAV local frame into AGV frame.
          Sophus::SE3<T> anchor_T_t = buildTransformationSE3(T_anchor_uav, anchor_roll_uav_, anchor_pitch_uav_);
          Sophus::SE3<T> anchor_T_s = buildTransformationSE3(T_anchor_agv, anchor_roll_agv_, anchor_pitch_agv_);

          Sophus::SE3<T> w_T_s = anchor_T_s * T_s;
          Sophus::SE3<T> w_T_t = anchor_T_t * T_t;

          Sophus::SE3<T> T_pred = w_T_t.inverse() * w_T_s;

          Sophus::SE3<T> T_err = T_meas_.template cast<T>().inverse() * T_pred;
          Eigen::Matrix<T, 6, 1> xi = T_err.log();
      
          // Project the 6-vector error onto the 4-DOF space (translation and yaw).
          Eigen::Matrix<T, 4, 1> error_vec;
          error_vec.template head<3>() = xi.template head<3>();
          error_vec[3] = xi[5];
      
          // Scale by the square root of the inverse covariance matrix
          Eigen::LLT<Eigen::Matrix4d> chol(cov_);
          Eigen::Matrix4d sqrt_inv_cov = Eigen::Matrix4d(chol.matrixL().transpose()).inverse();
          Eigen::Matrix<T, 4, 1> weighted_residual = sqrt_inv_cov.cast<T>() * error_vec;

          // Assign to residual
          residual[0] = weighted_residual[0];
          residual[1] = weighted_residual[1];
          residual[2] = weighted_residual[2];
          residual[3] = weighted_residual[3];
      
          return true;
        }
      
        static ceres::CostFunction* Create(const Sophus::SE3d &T_meas, const Eigen::Matrix4d &cov, 
                                            double source_roll, double source_pitch, 
                                            double target_roll, double target_pitch,
                                            double anchor_roll_uav, double anchor_pitch_uav,
                                            double anchor_roll_agv, double anchor_pitch_agv) {
          return new ceres::AutoDiffCostFunction<AnchorResidual, 4, 4, 4, 4, 4>(
            new AnchorResidual(T_meas, cov, source_roll, source_pitch, target_roll, target_pitch, anchor_roll_uav, anchor_pitch_uav, anchor_roll_agv, anchor_pitch_agv));
        }
      
      private:
        // Helper function to build a 4-DOF SE3 transformation given a state and fixed roll, pitch.
        template <typename T>
        Sophus::SE3<T> buildTransformationSE3(const T* state, double roll, double pitch) const {
            // Extract translation
            Eigen::Matrix<T, 3, 1> t;
            t << state[0], state[1], state[2];
            // Build rotation from yaw, with fixed roll and pitch
            Eigen::Matrix<T, 3, 3> R = (Eigen::AngleAxis<T>(state[3], Eigen::Matrix<T, 3, 1>::UnitZ()) *
                                        Eigen::AngleAxis<T>(T(pitch), Eigen::Matrix<T, 3, 1>::UnitY()) *
                                        Eigen::AngleAxis<T>(T(roll),  Eigen::Matrix<T, 3, 1>::UnitX())).toRotationMatrix();
            // Return the Sophus SE3 object
            return Sophus::SE3<T>(R, t);
        }
      
        const Sophus::SE3d T_meas_;
        const Eigen::Matrix4d cov_;
        const double source_roll_, source_pitch_;
        const double target_roll_, target_pitch_;
        const double anchor_roll_uav_, anchor_pitch_uav_;
        const double anchor_roll_agv_, anchor_pitch_agv_;
      };

    
    // Subscriptions
    rclcpp::Subscription<eliko_messages::msg::DistancesList>::SharedPtr eliko_distances_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr uav_odom_sub_, agv_odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr uav_linear_vel_sub_, uav_angular_vel_sub_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_source_sub_, pcl_target_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr optimized_tf_sub_;

    // Timers
    rclcpp::TimerBase::SharedPtr global_optimization_timer_;
    //Service client for visualization
    rclcpp::Client<uwb_localization::srv::UpdatePointClouds>::SharedPtr pcl_visualizer_client_;

    //Pose publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr anchor_agv_publisher_, anchor_uav_publisher_;
    rclcpp::Publisher<uwb_localization::msg::PoseWithCovarianceStampedArray>::SharedPtr poses_uav_publisher_, poses_agv_publisher_;

    //Lidar and radar positions
    Sophus::SE3d T_uav_lidar_, T_agv_lidar_;
    Sophus::SE3d T_uav_radar_, T_agv_radar_;
    double pointcloud_sigma_;

    //Measurements
    pcl::PointCloud<pcl::PointXYZ>::Ptr uav_cloud_{new pcl::PointCloud<pcl::PointXYZ>};
    pcl::PointCloud<pcl::PointXYZ>::Ptr agv_cloud_{new pcl::PointCloud<pcl::PointXYZ>};
    int icp_type_;
    bool uwb_encounter_available_, icp_encounter_available_; 
    geometry_msgs::msg::PoseWithCovarianceStamped latest_relative_pose_;
    Sophus::SE3d latest_relative_pose_SE3_;
    Eigen::Matrix<double, 6, 6> latest_relative_pose_cov_;
    double min_keyframes_;

    State init_state_uav_, init_state_agv_; 
    State anchor_node_uav_, anchor_node_agv_;
    PriorConstraint prior_agv_, prior_uav_, prior_anchor_;

    MapOfStates uav_map_, agv_map_;
    VectorOfConstraints proprioceptive_constraints_uav_, extraceptive_constraints_uav_; 
    VectorOfConstraints proprioceptive_constraints_agv_, extraceptive_constraints_agv_;
    VectorOfConstraints encounter_constraints_uwb_, encounter_constraints_pointcloud_;
    Measurements agv_measurements_, prev_agv_measurements_, uav_measurements_, prev_uav_measurements_;
    int uav_id_, agv_id_;
    bool graph_initialized_, first_optimization_done_;
    // Publishers/Broadcasters
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::string eliko_frame_id_, uav_frame_id_, global_frame_graph_;
    std::string odom_tf_agv_s_, odom_tf_agv_t_;
    std::string odom_tf_uav_s_, odom_tf_uav_t_;

    double global_opt_window_s_, global_opt_rate_s_;

    Sophus::SE3d uav_odom_pose_, last_uav_odom_pose_;         // Current UAV odometry position and last used for optimization
    Sophus::SE3d agv_odom_pose_, last_agv_odom_pose_;        // Current AGV odometry position and last used for optimization
    Eigen::Matrix<double, 6, 6> uav_odom_covariance_, agv_odom_covariance_ ;  // UAV and AGV odometry covariance
    Eigen::Quaterniond uav_quaternion_;

    nav_msgs::msg::Odometry last_agv_odom_msg_, last_uav_odom_msg_;
    bool last_agv_odom_initialized_, last_uav_odom_initialized_;
    
    double min_traveled_distance_, min_traveled_angle_;
    double uav_translation_, agv_translation_;
    double uav_rotation_, agv_rotation_;

};
int main(int argc, char** argv) {

    rclcpp::init(argc, argv);
    auto node = std::make_shared<FusionOptimizationNode>();
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
