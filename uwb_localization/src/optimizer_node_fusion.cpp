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


// Include the service header (adjust the package name accordingly)
#include "uwb_localization/srv/update_point_clouds.hpp"  
using UpdatePointClouds = uwb_localization::srv::UpdatePointClouds;
using namespace small_gicp;




struct Constraint3d {
    int id_begin;
    int id_end;
    Sophus::SE3d t_T_s;  // measured relative transform
    Eigen::Matrix<double, 4, 4> covariance;
    bool is_uwb;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  };

using VectorOfConstraints =
    std::vector<Constraint3d, Eigen::aligned_allocator<Constraint3d>>;

struct State {
    Eigen::Vector4d state; // [x,y,z,yaw]
    double roll;
    double pitch;
    Eigen::Matrix4d covariance;
    rclcpp::Time timestamp;  // e.g., seconds since epoch
    int robot_id; //0 for AGV, 1 for UAV

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

using MapOfStates =
    std::map<int,
             State,
             std::less<int>,
             Eigen::aligned_allocator<std::pair<const int, State>>>;


struct Measurements {
    rclcpp::Time timestamp;      // The time at which the measurement was taken.
    Sophus::SE3d relative_transform;       // The measured relative transform.
    
    Eigen::Matrix4d relative_transform_covariance;
    Eigen::Vector4d agv_odom; // [x,y,z,yaw]
    Eigen::Matrix4d agv_odom_covariance;
    Eigen::Vector4d uav_odom; // [x,y,z,yaw]
    Eigen::Matrix4d uav_odom_covariance;
  
    // Pointer to an associated point cloud (keyframe scan).
    pcl::PointCloud<pcl::PointXYZ>::Ptr agv_scan;
    pcl::PointCloud<pcl::PointXYZ>::Ptr uav_scan;

    double uav_roll;
    double uav_pitch;
    double agv_roll;
    double agv_pitch;

     // Constructor to initialize the pointer.
     Measurements() : agv_scan(new pcl::PointCloud<pcl::PointXYZ>), uav_scan(new pcl::PointCloud<pcl::PointXYZ>) {}
};


class StateManifoldAGV : public ceres::Manifold {
    public:
     // The ambient (parameter) space is 4: [x, y, z, yaw].
     virtual int AmbientSize() const override { return 4; }
     // The tangent (update) space is 3: we update only x, y, and yaw.
     virtual int TangentSize() const override { return 3; }
     
     // Plus operation: x ⊕ delta
     // x: [x, y, z, yaw], delta: [dx, dy, dyaw]
     virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
       // Update x and y normally.
       x_plus_delta[0] = x[0] + delta[0];
       x_plus_delta[1] = x[1] + delta[1];
       // Leave z unchanged.
       x_plus_delta[2] = x[2];
       // Update yaw and wrap it.
       double new_yaw = x[3] + delta[2];
       while (new_yaw > M_PI)  new_yaw -= 2.0 * M_PI;
       while (new_yaw < -M_PI) new_yaw += 2.0 * M_PI;
       x_plus_delta[3] = new_yaw;
       return true;
     }
     
     // PlusJacobian: The Jacobian of Plus with respect to delta, evaluated at delta = 0.
     // It is a 4x3 matrix (in row-major order) that should look like:
     // [1, 0, 0;
     //  0, 1, 0;
     //  0, 0, 0;
     //  0, 0, 1]
     virtual bool PlusJacobian(const double* /*x*/, double* jacobian) const override {
       // There are 4*3 = 12 entries.
       for (int i = 0; i < 12; ++i) jacobian[i] = 0.0;
       jacobian[0]  = 1.0;  // d(x)/d(dx)
       jacobian[4]  = 1.0;  // d(y)/d(dy)  (row 1, col 1)
       // Row 2 (z) remains zero.
       jacobian[11] = 1.0;  // d(yaw)/d(dyaw)  (row 3, col 3)
       return true;
     }
     
     // Minus: Given two parameter vectors x and y, compute the tangent vector delta
     // such that x ⊕ delta ≈ y. Since z is not updated, we ignore its difference.
     virtual bool Minus(const double* y, const double* x, double* y_minus_x) const override {
       // Difference in x and y.
       y_minus_x[0] = y[0] - x[0];
       y_minus_x[1] = y[1] - x[1];
       // Do not compute a difference for z (the tangent space is 3D).
       // Compute difference for yaw with wrapping.
       double dtheta = y[3] - x[3];
       while (dtheta > M_PI)  dtheta -= 2.0 * M_PI;
       while (dtheta < -M_PI) dtheta += 2.0 * M_PI;
       y_minus_x[2] = dtheta;
       return true;
     }
     
     // MinusJacobian: The Jacobian of Minus with respect to y.
     // For our 3x4 matrix (3-dimensional output, 4-dimensional input), we have:
     // [1, 0, 0, 0;
     //  0, 1, 0, 0;
     //  0, 0, 0, 1]
     virtual bool MinusJacobian(const double* /*x*/, double* jacobian) const override {
       // 3 rows x 4 columns = 12 elements, row-major order.
       for (int i = 0; i < 12; ++i) jacobian[i] = 0.0;
       jacobian[0]  = 1.0;  // Row0: d((y-x)_0)/d(y0)
       // Row0: columns 1,2,3 remain 0.
       jacobian[5]  = 1.0;  // Row1: d((y-x)_1)/d(y1)
       // Row1: other entries 0.
       jacobian[11] = 1.0;  // Row2: d((yaw difference))/d(y3)
       return true;
     }
   };

// Custom manifold for a state in R^3 x S^1.
class StateManifoldUAV : public ceres::Manifold {
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
    std::string odom_topic_uav = "/odom"; //or "/uav/odom"
    std::string vel_topic_uav = "/dji_sdk/velocity";
    

    rclcpp::SensorDataQoS qos; // Use a QoS profile compatible with sensor data

    dji_attitude_sub_ = this->create_subscription<geometry_msgs::msg::QuaternionStamped>(
        "/dji_sdk/attitude", qos, std::bind(&FusionOptimizationNode::attitude_cb_, this, std::placeholders::_1));

    // uav_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    // odom_topic_uav, 10, std::bind(&FusionOptimizationNode::uav_odom_cb_, this, std::placeholders::_1));

    // Initialize odometry positions and errors
    uav_odom_pos_ = Eigen::Vector4d::Zero();
    uav_last_odom_pos_ = Eigen::Vector4d::Zero();
    last_uav_vel_initialized_ = false;
    agv_odom_pos_ = Eigen::Vector4d::Zero();
    agv_last_odom_pos_ = Eigen::Vector4d::Zero();

    min_traveled_distance_ = 0.25;
    min_traveled_angle_ = 25.0 * M_PI / 180.0;
    uav_distance_ = agv_distance_ = uav_angle_ = agv_angle_ = 0.0;

    agv_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    odom_topic_agv, qos, std::bind(&FusionOptimizationNode::agv_odom_cb_, this, std::placeholders::_1));

    uav_vel_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
        vel_topic_uav, qos, std::bind(&FusionOptimizationNode::uav_vel_cb_, this, std::placeholders::_1));

    //Option 2: get odometry through tf readings -> only transform
    odom_tf_agv_s_ = "arco/eliko"; //source
    odom_tf_agv_t_ = "arco/odom"; //target
    odom_tf_uav_s_ = "odom"; //source
    odom_tf_uav_t_ = "world"; //target

    // Set up transform listener for UAV odometry.
    tf_buffer_uav_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_uav_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_uav_);

    // Set up transform listener for AGV odometry.
    tf_buffer_agv_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_agv_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_agv_);

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
    pose_uav_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("pose_graph_node/uav_pose", 10);
    pose_agv_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("pose_graph_node/agv_pose", 10);

    global_poses_uav_publisher_ = this->create_publisher<geometry_msgs::msg::PoseArray>("pose_graph_node/global_uav_poses", 10);
    global_poses_agv_publisher_ = this->create_publisher<geometry_msgs::msg::PoseArray>("pose_graph_node/global_agv_poses", 10);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    global_opt_window_s_ = 5.0; //size of the sliding window in seconds
    global_opt_rate_s_ = 0.2 * global_opt_window_s_; //rate of the optimization
    min_keyframes_ = 3.0; //number of nodes to run optimization

    T_uav_lidar_ = build_transformation_SE3(0.0,0.0, Eigen::Vector4d(0.21,0.0,0.25,0.0));
    T_agv_lidar_ = build_transformation_SE3(3.14,0.0, Eigen::Vector4d(0.3,0.0,0.45,0.0));

    global_optimization_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(int(global_opt_rate_s_*1000)), std::bind(&FusionOptimizationNode::global_opt_cb_, this));

    eliko_frame_id_ = "agv_opt"; //frame of the eliko system-> arco/eliko, for simulation use "agv_gt" for ground truth, "agv_odom" for odometry w/ errors
    uav_frame_id_ = "uav_opt"; //frame of the uav -> "base_link", for simulation use "uav_opt"

    //Initial values for state AGV
    init_state_agv_.state = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);//Eigen::Vector4d(-2.805, -1.951, 0.0, 0.102);
    init_state_agv_.roll = 0.0;
    init_state_agv_.pitch = 0.0;

    init_state_agv_.covariance = Eigen::Matrix4d::Identity(); //
    init_state_agv_.timestamp = this->get_clock()->now();
    init_state_agv_.robot_id = 0;

    //Initial values for state UAV
    init_state_uav_.state = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
    init_state_uav_.roll = 0.0;
    init_state_uav_.pitch = 0.0;

    init_state_uav_.covariance = Eigen::Matrix4d::Identity(); //
    init_state_uav_.timestamp = this->get_clock()->now();
    init_state_uav_.robot_id = 1;

    //Start node counter
    new_id_ = 0;
    new_id_offset_ = 1000;

    global_map_[new_id_] = init_state_agv_;
    global_map_[new_id_ + new_id_offset_] = init_state_uav_;

    moving_average_ = true;
    moving_average_window_s_ = global_opt_window_s_ / 2.0;

    //Set ICP algorithm variant: 2->Generalized ICP, 1->Point to Plane ICP, else -> basic ICP
    icp_type_ = 0;

    relative_pose_available_ = false;
    global_optimization_ = false;            

    RCLCPP_INFO(this->get_logger(), "Eliko Optimization Node initialized.");
  }


private:

    void uav_odom_cb_(const nav_msgs::msg::Odometry::SharedPtr msg) {
        
        // Extract quaternion components.
        double qx = msg->pose.pose.orientation.x;
        double qy = msg->pose.pose.orientation.y;
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;

        // Compute yaw.
        double siny_cosp = 2.0 * (qw * qz + qx * qy);
        double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
        double yaw = std::atan2(siny_cosp, cosy_cosp);

        // Assign the position and yaw to a 4D vector.
        uav_odom_pos_ = Eigen::Vector4d(
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z,
            yaw
        );

        // Extract covariance from the odometry message and store it in Eigen::Matrix4d
        uav_odom_covariance_ = Eigen::Matrix4d::Zero();
        uav_odom_covariance_(0, 0) = msg->pose.covariance[0];  // x variance
        uav_odom_covariance_(1, 1) = msg->pose.covariance[7];  // y variance
        uav_odom_covariance_(2, 2) = msg->pose.covariance[14]; // z variance
        uav_odom_covariance_(3, 3) = msg->pose.covariance[35]; // yaw variance
    }

    void uav_vel_cb_(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {
        
        // If this is the first message, simply store it and return.
       if (!last_uav_vel_initialized_) {
           last_uav_vel_msg_ = *msg;
           last_uav_vel_initialized_ = true;
           return;
       }
       
       // Compute the time difference between the current and last velocity messages.
       rclcpp::Time current_time(msg->header.stamp);
       rclcpp::Time last_time(last_uav_vel_msg_.header.stamp);
       double dt = (current_time - last_time).seconds();
       
       // For a basic integration, assume the velocity is constant over dt.
       Eigen::Vector3d current_vel(msg->vector.x, msg->vector.y, msg->vector.z);
       // Update your integrated UAV odometry position (only x, y, z)
       uav_odom_pos_.head<3>() += current_vel * dt;
       
       // Optionally, you can update the yaw separately from your attitude callback.
       uav_odom_pos_[3] = uav_yaw_;

       // Compute UAV translational distance using only [x, y, z]
       uav_distance_ += (uav_odom_pos_.head<3>() - uav_last_odom_pos_.head<3>()).norm();
       // Compute UAV yaw difference and normalize it to [-pi, pi]
       uav_angle_ += normalize_angle(uav_odom_pos_[3] - uav_last_odom_pos_[3]);

       // Update last_uav_vel_msg_ for the next iteration.
       last_uav_vel_msg_ = *msg;
       uav_last_odom_pos_ = uav_odom_pos_;

   }

    void agv_odom_cb_(const nav_msgs::msg::Odometry::SharedPtr msg) {
        
        // Extract quaternion components.
        double qx = msg->pose.pose.orientation.x;
        double qy = msg->pose.pose.orientation.y;
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;

        // Compute yaw.
        double siny_cosp = 2.0 * (qw * qz + qx * qy);
        double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
        double yaw = std::atan2(siny_cosp, cosy_cosp);

        // Assign the position and yaw to a 4D vector.
        agv_odom_pos_ = Eigen::Vector4d(
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z,
            yaw
        );

         // Similarly, compute AGV translation and yaw
         agv_distance_ += (agv_odom_pos_.head<3>() - agv_last_odom_pos_.head<3>()).norm();
         agv_angle_ += normalize_angle(agv_odom_pos_[3] - agv_last_odom_pos_[3]); 

        // Extract covariance from the odometry message and store it in Eigen::Matrix4d
        agv_odom_covariance_ = Eigen::Matrix4d::Zero();
        agv_odom_covariance_(0, 0) = msg->pose.covariance[0];  // x variance
        agv_odom_covariance_(1, 1) = msg->pose.covariance[7];  // y variance
        agv_odom_covariance_(2, 2) = msg->pose.covariance[14]; // z variance
        agv_odom_covariance_(3, 3) = msg->pose.covariance[35]; // yaw variance

        agv_last_odom_pos_ = agv_odom_pos_;

        //RCLCPP_WARN(this->get_logger(), "Received AGV odometry.");

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
        RCLCPP_INFO(this->get_logger(), "Received optimized relative transform.");   
    }

    geometry_msgs::msg::PoseWithCovarianceStamped constructPoseFromState(const State &state, const std::string &frame_id, const double &roll, const double &pitch) {
        geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
        pose_msg.header.stamp = state.timestamp;
        pose_msg.header.frame_id = frame_id;
        
        // Build a transformation from the state.
        // Here we assume that roll and pitch are either zero or provided in the state.
        Sophus::SE3d T = build_transformation_SE3(roll, pitch, state.state);
        Eigen::Vector3d t = T.translation();
        pose_msg.pose.pose.position.x = t.x();
        pose_msg.pose.pose.position.y = t.y();
        pose_msg.pose.pose.position.z = t.z();
        
        Eigen::Quaterniond q(T.rotationMatrix());
        pose_msg.pose.pose.orientation.x = q.x();
        pose_msg.pose.pose.orientation.y = q.y();
        pose_msg.pose.pose.orientation.z = q.z();
        pose_msg.pose.pose.orientation.w = q.w();
        
        // Map the 4x4 covariance (translation and yaw) to a 6x6 covariance for the pose.
        // Here, we assume that roll and pitch have very low uncertainty.
        Eigen::Matrix<double, 6, 6> cov6 = Eigen::Matrix<double, 6, 6>::Zero();
        // Copy the translation covariance (first 3x3 block):
        cov6.block<3,3>(0,0) = state.covariance.block<3,3>(0,0);
        // Set a small covariance for roll and pitch:
        cov6(3,3) = 1e-6;
        cov6(4,4) = 1e-6;
        // Copy the yaw variance:
        cov6(5,5) = state.covariance(3,3);
        // You can also copy the cross terms if needed.
        
        // Flatten the 6x6 covariance into the 36-element array required by the message.
        for (size_t i = 0; i < 6; ++i) {
          for (size_t j = 0; j < 6; ++j) {
            pose_msg.pose.covariance[i * 6 + j] = cov6(i, j);
          }
        }
        
        return pose_msg;
      }



    void global_opt_cb_() {

        rclcpp::Time current_time = this->get_clock()->now();

        // //Read the transform (if odom topics not available)
        // try {
        //     auto transform_agv = tf_buffer_agv_->lookupTransform(odom_tf_agv_t_, odom_tf_agv_s_, rclcpp::Time(0));
        //     Sophus::SE3d T_agv_odom = transformSE3FromMsg(transform_agv);
        //     agv_odom_pos_ = transformSE3ToState(T_agv_odom);
        //     } catch (const tf2::TransformException &ex) {
        //     RCLCPP_WARN(this->get_logger(), "Could not get transform for AGV: %s", ex.what());
        //     return;
        //     }

        // try {
        //     auto transform_uav = tf_buffer_uav_->lookupTransform(odom_tf_uav_t_, odom_tf_uav_s_, rclcpp::Time(0));
        //     Sophus::SE3d T_uav_odom = transformSE3FromMsg(transform_uav);
        //     uav_odom_pos_ = transformSE3ToState(T_uav_odom);
        //     } catch (const tf2::TransformException &ex) {
        //     RCLCPP_WARN(this->get_logger(), "Could not get transform for UAV: %s", ex.what());
        //     }

        if ((uav_distance_ < min_traveled_distance_ && uav_angle_ < min_traveled_angle_) && (agv_distance_ < min_traveled_distance_ && agv_angle_ < min_traveled_angle_)) {
            RCLCPP_WARN(this->get_logger(), "[Fusion optimization node] Insufficient movement UAV = [%.2fm %.2fº], AGV= [%.2fm %.2fº]. Skipping optimization.", uav_distance_, uav_angle_ * 180.0/M_PI, agv_distance_, agv_angle_ * 180.0/M_PI);
            return;
        }

        relative_pose_available_ = isRelativeTransformAvailable(current_time, latest_relative_pose_.header.stamp, global_opt_window_s_);

        Measurements new_measurements;
        new_measurements.timestamp = current_time;

        new_measurements.agv_roll = 0.0;
        new_measurements.agv_pitch = 0.0;
        new_measurements.uav_roll = uav_roll_;
        new_measurements.uav_pitch = uav_pitch_;
        *(new_measurements.agv_scan) = *(downsamplePointCloud(agv_cloud_, 0.05f));
        *(new_measurements.uav_scan) = *(downsamplePointCloud(uav_cloud_, 0.05f));
        new_measurements.agv_odom = agv_odom_pos_;
        new_measurements.agv_odom_covariance = agv_odom_covariance_;
        new_measurements.uav_odom = uav_odom_pos_;
        new_measurements.uav_odom_covariance = agv_odom_covariance_;

        //Check if there is a new relative position available
        if(relative_pose_available_){
            //Unflatten matrix to extract the covariance
            Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    cov(i,j) = latest_relative_pose_.pose.covariance[i * 6 + j];
                }
            }
            new_measurements.relative_transform = transformSE3FromPoseMsg(latest_relative_pose_.pose.pose);
            new_measurements.relative_transform_covariance = cov; 
        }
        else{
            new_measurements.relative_transform = Sophus::SE3d(Eigen::Matrix4d::Identity());
            new_measurements.relative_transform_covariance = Eigen::Matrix4d::Identity(); 
        }

        //Increase ID
        new_id_++;

        // Create a new AGV node from the current odometry.
        State new_agv;
        int agv_id = new_id_;
        new_agv.timestamp = current_time;
        new_agv.roll = new_measurements.agv_roll;
        new_agv.pitch = new_measurements.agv_pitch;
        // if(new_id_ < min_keyframes_){
        //     new_agv.state = new_measurements.agv_odom;
        //     new_agv.covariance = new_measurements.agv_odom_covariance;
        // }
        // else{

            new_agv.state = global_map_[agv_id-1].state;
            new_agv.covariance = global_map_[agv_id-1].covariance;
        // }
        new_agv.robot_id = 0;
        
        global_map_[agv_id] = new_agv;

        RCLCPP_INFO(this->get_logger(), "Adding new AGV node at timestamp %.2f: [%f, %f, %f, %f]", current_time.seconds(),
                        new_agv.state[0], new_agv.state[1], new_agv.state[2], new_agv.state[3]);


        // Similarly, create a new UAV node.
        State new_uav;  
        int uav_id = new_id_ + new_id_offset_;
        new_uav.timestamp = current_time;
        new_uav.roll = new_measurements.uav_roll;
        new_uav.pitch = new_measurements.uav_pitch;

        // if(new_id_ < min_keyframes_ && relative_pose_available_){
        //     Sophus::SE3d pred_That_uav_w = new_measurements.relative_transform*(build_transformation_SE3(0.0,0.0,agv_odom_pos_).inverse());
        //     new_uav.state = transformSE3ToState(pred_That_uav_w.inverse());
        //     new_uav.covariance = new_measurements.uav_odom_covariance;
        // }
        // else{

            new_uav.state = global_map_[uav_id - 1].state;
            new_uav.covariance = global_map_[uav_id-1].covariance;
        // }

        new_uav.robot_id = 1;

        global_map_[uav_id] = new_uav;


        // Now, if this is not the first node, add a sequential constraint:
        if (new_id_ > 0) {
            
            //AGV odom constraints
            Constraint3d constraint_odom_agv;
            constraint_odom_agv.id_begin = agv_id - 1;
            constraint_odom_agv.id_end = agv_id;
            constraint_odom_agv.is_uwb = false;

            Sophus::SE3d odom_T_s_agv = build_transformation_SE3(prev_measurements_.agv_roll, prev_measurements_.agv_pitch, prev_measurements_.agv_odom);
            Sophus::SE3d odom_T_t_agv = build_transformation_SE3(new_measurements.agv_roll, new_measurements.agv_pitch, new_measurements.agv_odom);
            constraint_odom_agv.t_T_s = odom_T_t_agv.inverse()*odom_T_s_agv;
            constraint_odom_agv.covariance = new_measurements.agv_odom_covariance;

            global_constraints_.push_back(constraint_odom_agv);

            //AGV ICP constraints

            if (!new_measurements.agv_scan->points.empty() &&
                !prev_measurements_.agv_scan->points.empty()) {

                    RCLCPP_WARN(this->get_logger(), "Computing ICP for AGV nodes %d and %d.", new_id_ - 1, new_id_);

                    Eigen::Matrix4f T_icp = constraint_odom_agv.t_T_s.cast<float>().matrix();
                    double fitness = 0.0;
        
                    if(run_icp(prev_measurements_.agv_scan, new_measurements.agv_scan, T_icp, fitness, icp_type_)){
                        Constraint3d constraint_icp_agv;
                        constraint_icp_agv.id_begin = agv_id - 1;
                        constraint_icp_agv.id_end = agv_id;
                        constraint_icp_agv.is_uwb = false;
                        //Transform to the robots body frame
                        Sophus::SE3d T_icp_agv = T_agv_lidar_ * Sophus::SE3f(T_icp).cast<double>() * T_agv_lidar_.inverse();
                        constraint_icp_agv.t_T_s = T_icp_agv;
                        log_transformation_matrix(constraint_icp_agv.t_T_s.matrix());
                        constraint_icp_agv.covariance = Eigen::Matrix4d::Identity() * 0.01;
                        global_constraints_.push_back(constraint_icp_agv);
                    };
            }

            //UAV odom constraints
            Constraint3d constraint_odom_uav;
            constraint_odom_uav.id_begin = uav_id - 1;
            constraint_odom_uav.id_end = uav_id;
            constraint_odom_uav.is_uwb = false;

            Sophus::SE3d odom_T_s_uav = build_transformation_SE3(prev_measurements_.uav_roll, prev_measurements_.uav_pitch, prev_measurements_.uav_odom);
            Sophus::SE3d odom_T_t_uav = build_transformation_SE3(new_measurements.uav_roll, new_measurements.uav_pitch, new_measurements.uav_odom);
            constraint_odom_uav.t_T_s = odom_T_t_uav.inverse()*odom_T_s_uav;
            constraint_odom_uav.covariance = new_measurements.uav_odom_covariance;
            global_constraints_.push_back(constraint_odom_uav);

            //UAV ICP constraints
            if (!new_measurements.uav_scan->points.empty() &&
                !prev_measurements_.uav_scan->points.empty()) {

                    RCLCPP_WARN(this->get_logger(), "Computing ICP for UAV nodes %d and %d.", new_id_ - 1, new_id_);
                    
                    Eigen::Matrix4f T_icp = constraint_odom_uav.t_T_s.cast<float>().matrix();
                    double fitness = 0.0;

                    if(run_icp(prev_measurements_.uav_scan, new_measurements.uav_scan, T_icp, fitness, icp_type_)){
                        Constraint3d constraint_icp_uav;
                        constraint_icp_uav.id_begin = uav_id - 1;
                        constraint_icp_uav.id_end = uav_id;
                        constraint_icp_uav.is_uwb = false;
                        //Transform to the robots body frame
                        Sophus::SE3d T_icp_uav = T_uav_lidar_ * Sophus::SE3f(T_icp).cast<double>() * T_uav_lidar_.inverse();
                        constraint_icp_uav.t_T_s = T_icp_uav;
                        log_transformation_matrix(constraint_icp_uav.t_T_s.matrix());
                        constraint_icp_uav.covariance = Eigen::Matrix4d::Identity() * 0.01;
                        global_constraints_.push_back(constraint_icp_uav);
                    };
        
            }

            //Inter-robot UWB constraints
            if(relative_pose_available_){

                Constraint3d uwb_constraint;
                uwb_constraint.id_begin = agv_id; //same id, relates node i of agv map, and node i of uav map
                uwb_constraint.id_end = uav_id;
                uwb_constraint.is_uwb = true;
                uwb_constraint.t_T_s = new_measurements.relative_transform;
                uwb_constraint.covariance = new_measurements.relative_transform_covariance;
                global_constraints_.push_back(uwb_constraint);

            }

            //Update other constraints in the window
            for(auto& constraint : global_constraints_){
                if (constraint.is_uwb){
                    bool source_fixed = isNodeFixed(current_time, constraint.id_begin, global_map_, global_opt_window_s_);
                    bool target_fixed = isNodeFixed(current_time, constraint.id_end, global_map_, global_opt_window_s_);
                    if (source_fixed && target_fixed) {
                        // Both nodes are fixed; skip adding this constraint.
                        continue;
                    }
                    constraint.t_T_s = new_measurements.relative_transform;
                    constraint.covariance = new_measurements.relative_transform_covariance;
                }
            }

            //Inter-robot ICP constraints

            if (!new_measurements.uav_scan->points.empty() &&
                !new_measurements.agv_scan->points.empty()) {
                    
                    Eigen::Matrix4f T_icp = Eigen::Matrix4f::Identity();
                    if(relative_pose_available_) T_icp = new_measurements.relative_transform.cast<float>().matrix();

                    double fitness = 0.0;

                    RCLCPP_WARN(this->get_logger(), "Computing ICP for inter-robot pairs index %d and %d", agv_id, uav_id);

                    if(run_icp(new_measurements.agv_scan, new_measurements.uav_scan, T_icp, fitness, icp_type_)){
                        Constraint3d inter_icp_constraint;
                        inter_icp_constraint.id_begin = agv_id; //same id, relates node i of uav map, and node i of uav map
                        inter_icp_constraint.id_end = uav_id;
                        inter_icp_constraint.is_uwb = false;

                        //Transform to the robots body frame, using the corrected orthogonal rotation matrix
                        Sophus::SE3d T_icp_uav_agv = T_uav_lidar_ * Sophus::SE3f(T_icp).cast<double>() * T_agv_lidar_.inverse();

                        inter_icp_constraint.t_T_s = T_icp_uav_agv;

                        log_transformation_matrix(inter_icp_constraint.t_T_s.matrix());

                        inter_icp_constraint.covariance = Eigen::Matrix4d::Identity() * 0.01;
                        global_constraints_.push_back(inter_icp_constraint);
                    };
        
            }

        }
   
        RCLCPP_INFO(this->get_logger(), "Adding new UAV node at timestamp %.2f: [%f, %f, %f, %f]", current_time.seconds(),
                        new_uav.state[0], new_uav.state[1], new_uav.state[2], new_uav.state[3]);


        // Convert the PCL point cloud to a ROS message
        sensor_msgs::msg::PointCloud2 source_cloud_msg, target_cloud_msg;
        pcl::toROSMsg(*new_measurements.agv_scan, source_cloud_msg);
        pcl::toROSMsg(*new_measurements.uav_scan, target_cloud_msg);

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

        
        if(new_id_ >= min_keyframes_){

            RCLCPP_INFO(this->get_logger(), "Optimizing trajectory of %ld nodes", global_map_.size());


            //Update transforms after convergence
            if(run_posegraph_optimization(current_time, global_map_, global_constraints_)){

                State& agv_node = global_map_[agv_id];
                State& uav_node = global_map_[uav_id];

                Sophus::SE3d That_ws = build_transformation_SE3(agv_node.roll, agv_node.pitch, agv_node.state);
                Sophus::SE3d That_wt = build_transformation_SE3(uav_node.roll, uav_node.pitch, uav_node.state);

                RCLCPP_INFO(this->get_logger(), "AGV Optimized pose:\n"
                            "[%f, %f, %f, %f]", agv_node.state[0], agv_node.state[1],agv_node.state[2], agv_node.state[3]);

                RCLCPP_INFO(this->get_logger(), "UAV Optimized pose:\n"
                            "[%f, %f, %f, %f]", uav_node.state[0], uav_node.state[1], uav_node.state[2], uav_node.state[3]);

                //AGV odom frame to be common reference frame
                publish_transform(That_ws, current_time, odom_tf_agv_t_, eliko_frame_id_);
                publish_pose(That_ws, agv_node.covariance, current_time, odom_tf_agv_t_, pose_agv_publisher_);

                publish_transform(That_wt, current_time, odom_tf_agv_t_, uav_frame_id_);
                publish_pose(That_wt, uav_node.covariance, current_time, odom_tf_agv_t_, pose_uav_publisher_);

            }

            else{
                RCLCPP_INFO(this->get_logger(), "[Eliko global_opt node] Local optimizer did not converge");
            }

        }

        //Update odom for next optimization
        uav_distance_ = agv_distance_ = uav_angle_ = agv_angle_ = 0.0;

        prev_measurements_ = new_measurements;
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


    bool run_icp(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &source_cloud,
             const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &target_cloud, Eigen::Matrix4f &transformation, double &fitness, const int &icp_type) const {
            
        if(icp_type == 2) {

            // RegistrationPCL is derived from pcl::Registration and has mostly the same interface as pcl::GeneralizedIterativeClosestPoint.
            RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> reg;
            reg.setNumThreads(4);
            reg.setCorrespondenceRandomness(20);
            reg.setMaxCorrespondenceDistance(1.0);
            reg.setVoxelResolution(1.0);

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
            icp.setMaxCorrespondenceDistance (0.1);
            // Set the maximum number of iterations (criterion 1)
            icp.setMaximumIterations (50);
            // Set the transformation epsilon (criterion 2)
            icp.setTransformationEpsilon (1e-8);
            // Set the euclidean distance difference epsilon (criterion 3)
            icp.setEuclideanFitnessEpsilon (1);
            
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
            icp.setMaxCorrespondenceDistance (0.1);
            // Set the maximum number of iterations (criterion 1)
            icp.setMaximumIterations (50);
            // Set the transformation epsilon (criterion 2)
            icp.setTransformationEpsilon (1e-8);
            // Set the euclidean distance difference epsilon (criterion 3)
            icp.setEuclideanFitnessEpsilon (1);

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


    void attitude_cb_(const geometry_msgs::msg::QuaternionStamped::SharedPtr msg) {

            const auto& q = msg->quaternion;
            // Convert the quaternion to roll, pitch, yaw
            tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
            tf2::Matrix3x3 m(tf_q);
            m.getRPY(uav_roll_, uav_pitch_, uav_yaw_);
        }

    
    // Normalize an angle to the range [-pi, pi]
    double normalize_angle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
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

    // Computes the weighted moving average for UAV states.
    Eigen::Vector4d moving_average_uav(const State &new_sample) {

        // Append the new sample to the UAV moving average deque.
        moving_average_states_uav_.push_back(new_sample);

        // Remove UAV samples that are too old.
        while (!moving_average_states_uav_.empty() &&
                (new_sample.timestamp - moving_average_states_uav_.front().timestamp) >
                    rclcpp::Duration::from_seconds(global_opt_window_s_)) {
            moving_average_states_uav_.pop_front();
        }

        // Initialize accumulators.
        Eigen::Vector4d smoothed_state = Eigen::Vector4d::Zero();
        double sum_sin = 0.0;
        double sum_cos = 0.0;
        double total_weight = 0.0;
        double decay_factor = 0.9;  // Adjust as needed.
        double weight = 1.0;        // Start with a weight of 1.0 for the most recent sample.

        // Iterate over UAV samples from newest to oldest.
        for (auto it = moving_average_states_uav_.rbegin();
            it != moving_average_states_uav_.rend(); ++it) {
            smoothed_state[0] += weight * it->state[0];
            smoothed_state[1] += weight * it->state[1];
            smoothed_state[2] += weight * it->state[2];
            sum_sin += weight * std::sin(it->state[3]);
            sum_cos += weight * std::cos(it->state[3]);
            total_weight += weight;
            weight *= decay_factor;
        }

        if (total_weight > 0.0) {
            smoothed_state[0] /= total_weight;
            smoothed_state[1] /= total_weight;
            smoothed_state[2] /= total_weight;
            smoothed_state[3] = std::atan2(sum_sin / total_weight, sum_cos / total_weight);
            smoothed_state[3] = normalize_angle(smoothed_state[3]);
        }

    return smoothed_state;
    
    }

    // Computes the weighted moving average for AGV states.
    Eigen::Vector4d moving_average_agv(const State &new_sample) {

        // Append the new sample to the AGV moving average deque.
        moving_average_states_agv_.push_back(new_sample);

        // Remove AGV samples that are too old.
        while (!moving_average_states_agv_.empty() &&
                (new_sample.timestamp - moving_average_states_agv_.front().timestamp) >
                    rclcpp::Duration::from_seconds(global_opt_window_s_)) {
            moving_average_states_agv_.pop_front();
        }

        // Initialize accumulators.
        Eigen::Vector4d smoothed_state = Eigen::Vector4d::Zero();
        double sum_sin = 0.0;
        double sum_cos = 0.0;
        double total_weight = 0.0;
        double decay_factor = 0.9;  // Adjust as needed.
        double weight = 1.0;        // Start with a weight of 1.0 for the most recent sample.

        // Iterate over AGV samples from newest to oldest.
        for (auto it = moving_average_states_agv_.rbegin();
            it != moving_average_states_agv_.rend(); ++it) {
            smoothed_state[0] += weight * it->state[0];
            smoothed_state[1] += weight * it->state[1];
            smoothed_state[2] += weight * it->state[2];
            sum_sin += weight * std::sin(it->state[3]);
            sum_cos += weight * std::cos(it->state[3]);
            total_weight += weight;
            weight *= decay_factor;
        }

        if (total_weight > 0.0) {
            smoothed_state[0] /= total_weight;
            smoothed_state[1] /= total_weight;
            smoothed_state[2] /= total_weight;
            smoothed_state[3] = std::atan2(sum_sin / total_weight, sum_cos / total_weight);
            smoothed_state[3] = normalize_angle(smoothed_state[3]);
        }

  return smoothed_state;
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

    void publish_pose(const Sophus::SE3d &T, const Eigen::Matrix4d& cov4, 
        const rclcpp::Time &current_time, const std::string &frame_id, 
        const rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr &pub) {


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


        // Publish transform
        pub->publish(p_msg);
    }


    void publishGlobalOptimizedPoses(const std::deque<State> &states, std::deque<Measurements> &measurements, const rclcpp::Time &current_time, const std::string &frame_id, 
                                    const rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr &pub) {
        // Create a PoseArray message
        geometry_msgs::msg::PoseArray pose_array_msg;
        pose_array_msg.header.stamp = current_time;
        pose_array_msg.header.frame_id = frame_id;  // Use your common reference frame
      
        // Loop over each state in your global pose graph (for example, AGV states)
        for (size_t i = 0; i < states.size(); ++i) {
          geometry_msgs::msg::Pose pose;
      
          Sophus::SE3d T;
          if (pub == global_poses_uav_publisher_) T = build_transformation_SE3(measurements[i].uav_roll, measurements[i].uav_pitch, states[i].state);
          else T = build_transformation_SE3(measurements[i].agv_roll, measurements[i].agv_pitch, states[i].state);
          
          Eigen::Vector3d t = T.translation();
          pose.position.x = t.x();
          pose.position.y = t.y();
          pose.position.z = t.z();
      
          // Convert the rotation matrix into a quaternion.
          Eigen::Quaterniond q(T.rotationMatrix());
          pose.orientation.x = q.x();
          pose.orientation.y = q.y();
          pose.orientation.z = q.z();
          pose.orientation.w = q.w();
      
          // Append this pose to the PoseArray
          pose_array_msg.poses.push_back(pose);
        }
      
        // Publish the PoseArray message
        pub->publish(pose_array_msg);
      
        RCLCPP_INFO(this->get_logger(), "Published %zu optimized poses.", pose_array_msg.poses.size());

    }

    Sophus::SE3d build_transformation_SE3(double roll, double pitch, const Eigen::Vector4d& s) {
        Eigen::Vector3d t(s[0], s[1], s[2]);  // Use Vector3d instead of an incorrect Matrix type.
        Eigen::Matrix3d R = (Eigen::AngleAxisd(s[3], Eigen::Vector3d::UnitZ()) *
                             Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX())).toRotationMatrix();
        return Sophus::SE3d(R, t);
    }

    Eigen::Matrix4d re_orthogonalize(const Eigen::Matrix4d &T){
        // Re-orthogonalize the rotation part of T_icp_d.
        Eigen::Matrix3d R = T.block<3,3>(0,0);
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d R_corrected = svd.matrixU() * svd.matrixV().transpose();

        Eigen::Matrix4d corrected_T = T;
        // Replace the rotation block with the corrected matrix -> translation remains the same
        corrected_T.block<3,3>(0,0) = R_corrected;

        return corrected_T;
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
                                    MapOfStates &global_map, VectorOfConstraints &global_constraints) {

        ceres::Problem problem;

        ceres::Manifold* agv_state_manifold = new StateManifoldAGV;
        ceres::Manifold* uav_state_manifold = new StateManifoldUAV;

        //remove the gauge freedom (i.e. the fact that an overall rigid body transform can be added to all poses without changing the relative errors).
        //anchor -freeze- the first node, and freeze the part of the node outside the sliding window
        for (auto& kv : global_map) {
            State& state = kv.second;
            problem.AddParameterBlock(state.state.data(), 4);
            if(state.robot_id == 0) problem.SetManifold(state.state.data(), agv_state_manifold);
            else if(state.robot_id == 1) problem.SetManifold(state.state.data(), uav_state_manifold);
            //Fix old nodes
            if (kv.first == 0 || isNodeFixed(current_time, kv.first, global_map, global_opt_window_s_)) {
                problem.SetParameterBlockConstant(kv.second.state.data());
            }
          }

        for(auto& constraint : global_constraints){

            bool source_fixed = isNodeFixed(current_time, constraint.id_begin, global_map, global_opt_window_s_); // your function to check
            bool target_fixed = isNodeFixed(current_time, constraint.id_end, global_map, global_opt_window_s_);
            if (source_fixed && target_fixed) {
                // Both nodes are fixed; skip adding this constraint.
                continue;
            }
            // Retrieve the states associated with the constraint.
            State& state_i = global_map[constraint.id_begin];
            State& state_j = global_map[constraint.id_end];

            ceres::CostFunction* rel_cost = Pose3DResidual::Create(constraint.t_T_s, constraint.covariance, state_i.roll, state_i.pitch, state_j.roll, state_j.pitch);
            problem.AddResidualBlock(rel_cost, nullptr, state_i.state.data(), state_j.state.data());
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
            for (auto& kv : global_map_) {
                // kv.first is the node's unique ID and kv.second is the State.
                covariance_blocks.emplace_back(kv.second.state.data(), kv.second.state.data());
            }

            if (covariance.Compute(covariance_blocks, &problem)) {
                // Update each node's covariance in the unified map.
                for (auto& kv : global_map_) {
                    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
                    covariance.GetCovarianceBlock(kv.second.state.data(), kv.second.state.data(), cov.data());
                    kv.second.covariance = cov;
                }
            } else {
                RCLCPP_WARN(this->get_logger(), "Failed to compute covariances.");
                // Set default covariances if needed.
                for (auto& kv : global_map_) {
                    kv.second.covariance = Eigen::Matrix4d::Identity();
                }
            }

            return true;

        } else {

                RCLCPP_WARN(this->get_logger(), "Failed to converge.");
                
                return false;
        }
                                    
        return false; 
            
    }



    struct Pose3DResidual {
        Pose3DResidual(const Sophus::SE3d& T_meas, const Eigen::Matrix4d& cov, double source_roll, double source_pitch, double target_roll, double target_pitch)
            : T_meas_(T_meas), cov_(cov), source_roll_(source_roll), source_pitch_(source_pitch), target_roll_(target_roll), target_pitch_(target_pitch) {}
    
        template <typename T>
        bool operator()(const T* const source_state, const T* const target_state, T* residual) const {
            // Build homogeneous transforms from the state vectors.
            Sophus::SE3<T> w_T_s = buildTransformationSE3(source_state, source_roll_, source_pitch_);
            Sophus::SE3<T> w_T_t = buildTransformationSE3(target_state, target_roll_, target_pitch_);
            
            // Compute the relative transform from AGV to UAV:
            Sophus::SE3<T> SE3_pred = w_T_t.inverse() * w_T_s;
            Sophus::SE3<T> SE3_meas = T_meas_.template cast<T>();
    
            // Compute the error transformation: T_err = T_meas^{-1} * T_pred.
            Sophus::SE3<T> T_err = SE3_meas.inverse() * SE3_pred;
            
            // Compute the full 6-vector logarithm (xi = [rho; phi]),
            // where phi is the rotation vector.
            Eigen::Matrix<T,6,1> xi = T_err.log();
    
            // Project the 6-vector error onto the 4-DOF space:
            // Keep the three translation components and only the z component of the rotation.
            Eigen::Matrix<T,4,1> error_vec;
            error_vec.template segment<3>(0) = xi.template segment<3>(0); // translation error
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
    
        static ceres::CostFunction* Create(const Sophus::SE3d& T_opt, const Eigen::Matrix4d& cov, double source_roll, double source_pitch, double target_roll, double target_pitch) {
            return new ceres::AutoDiffCostFunction<Pose3DResidual, 4, 4, 4>(
                new Pose3DResidual(T_opt, cov, source_roll, source_pitch, target_roll, target_pitch));
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

    
    // Subscriptions
    rclcpp::Subscription<eliko_messages::msg::DistancesList>::SharedPtr eliko_distances_sub_;
    rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr dji_attitude_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr uav_odom_sub_, agv_odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr uav_vel_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_source_sub_, pcl_target_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr optimized_tf_sub_;

    // Timers
    rclcpp::TimerBase::SharedPtr global_optimization_timer_, window_opt_timer_;
    //Service client for visualization
    rclcpp::Client<uwb_localization::srv::UpdatePointClouds>::SharedPtr pcl_visualizer_client_;

    //Pose publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_uav_publisher_, pose_agv_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr global_poses_uav_publisher_, global_poses_agv_publisher_;


    // Transform buffers and listeners for each odometry source.
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_agv_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_agv_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_uav_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_uav_;

    //Lidar and radar positions
    Sophus::SE3d T_uav_lidar_, T_agv_lidar_;
    Sophus::SE3d T_uav_radar_, T_agv_radar_;

    //Measurements
    pcl::PointCloud<pcl::PointXYZ>::Ptr uav_cloud_{new pcl::PointCloud<pcl::PointXYZ>};
    pcl::PointCloud<pcl::PointXYZ>::Ptr agv_cloud_{new pcl::PointCloud<pcl::PointXYZ>};
    int icp_type_;
    bool relative_pose_available_; 
    geometry_msgs::msg::PoseWithCovarianceStamped latest_relative_pose_;
    double min_keyframes_;
    bool global_optimization_;

    State init_state_uav_, init_state_agv_;

    MapOfStates global_map_;
    VectorOfConstraints inter_uwb_constraints_;
    VectorOfConstraints global_constraints_;
    Measurements prev_measurements_;
    int new_id_, new_id_offset_;
    
    // Publishers/Broadcasters
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::string eliko_frame_id_, uav_frame_id_;
    std::string odom_tf_agv_s_, odom_tf_agv_t_;
    std::string odom_tf_uav_s_, odom_tf_uav_t_;

    std::deque<State> moving_average_states_uav_, moving_average_states_agv_;

    double global_opt_window_s_, global_opt_rate_s_;
    double moving_average_window_s_;
    double moving_average_;
    double uav_roll_, uav_pitch_, uav_yaw_;

    Eigen::Vector4d uav_odom_pos_, uav_last_odom_pos_;         // Current UAV odometry position and last used for optimization
    Eigen::Vector4d agv_odom_pos_, agv_last_odom_pos_;        // Current AGV odometry position and last used for optimization
    Eigen::Matrix4d uav_odom_covariance_;  // UAV odometry covariance
    Eigen::Matrix4d agv_odom_covariance_;  // AGV odometry covariance
    geometry_msgs::msg::Vector3Stamped last_uav_vel_msg_;
    bool last_uav_vel_initialized_;
    double min_traveled_distance_, min_traveled_angle_;
    double uav_distance_, agv_distance_, uav_angle_, agv_angle_;

};
int main(int argc, char** argv) {

    rclcpp::init(argc, argv);
    auto node = std::make_shared<FusionOptimizationNode>();
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
