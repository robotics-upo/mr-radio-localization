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


struct State {
    Eigen::Vector4d state; // [x,y,z,yaw]
    Eigen::Matrix4d covariance;
    rclcpp::Time timestamp;  // e.g., seconds since epoch
};

struct Measurements {
    rclcpp::Time timestamp;      // The time at which the measurement was taken.
    Sophus::SE3d relative_transform;       // The measured relative transform.
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


// In addition, we create a structure for a pose graph which holds separate trajectories for each robot.
struct PoseGraph {
    std::deque<State> agv_states;
    std::deque<State> uav_states;
    std::deque<Measurements> measurements;
};


// Custom manifold for a state in R^3 x S^1.
class StateManifold : public ceres::Manifold {
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

    //Initial values for state
    init_state_.state = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
    init_state_.covariance = Eigen::Matrix4d::Identity(); //
    init_state_.timestamp = this->get_clock()->now();

    moving_average_ = true;
    moving_average_window_s_ = global_opt_window_s_ / 2.0;

    // Initialize odometry positions and errors
    uav_odom_pos_ = Eigen::Vector4d::Zero();
    last_uav_vel_initialized_ = false;
    agv_odom_pos_ = Eigen::Vector4d::Zero();

    //Set ICP algorithm variant: 2->Generalized ICP, 1->Point to Plane ICP, else -> basic ICP
    icp_type_ = 2;

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

       // Update last_uav_vel_msg_ for the next iteration.
       last_uav_vel_msg_ = *msg;
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

        // Extract covariance from the odometry message and store it in Eigen::Matrix4d
        agv_odom_covariance_ = Eigen::Matrix4d::Zero();
        agv_odom_covariance_(0, 0) = msg->pose.covariance[0];  // x variance
        agv_odom_covariance_(1, 1) = msg->pose.covariance[7];  // y variance
        agv_odom_covariance_(2, 2) = msg->pose.covariance[14]; // z variance
        agv_odom_covariance_(3, 3) = msg->pose.covariance[35]; // yaw variance

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

        if(!relative_pose_available_) relative_pose_available_ = true;

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

         //Check if there is a new relative position available
        if((!relative_pose_available_ || (current_time - latest_relative_pose_.header.stamp).seconds() > global_opt_window_s_)){
            RCLCPP_WARN(this->get_logger(), "Relative transform not available. Skipping optimization");
            return;
        }

        Measurements new_measurements;
        new_measurements.timestamp = current_time;
        
        opt_relative_pose_ = latest_relative_pose_; //Use a fixed relative tf for optimization step -the latest
        //Unflatten matrix to extract the covariance
        Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                cov(i,j) = opt_relative_pose_.pose.covariance[i * 6 + j];
            }
        }

        new_measurements.relative_transform = transformSE3FromPoseMsg(opt_relative_pose_.pose.pose);
        new_measurements.agv_roll = 0.0;
        new_measurements.agv_pitch = 0.0;
        new_measurements.uav_roll = uav_roll_;
        new_measurements.uav_pitch = uav_pitch_;
        *(new_measurements.agv_scan) = *(downsamplePointCloud(agv_cloud_, 0.1f));
        *(new_measurements.uav_scan) = *(downsamplePointCloud(uav_cloud_, 0.1f));
        new_measurements.agv_odom = agv_odom_pos_;
        new_measurements.agv_odom_covariance = agv_odom_covariance_;
        new_measurements.uav_odom = uav_odom_pos_;
        new_measurements.uav_odom_covariance = agv_odom_covariance_ * cov;

        local_pose_graph_.measurements.push_back(new_measurements);

        // Create a new AGV node from the current odometry.
        State new_agv;
        new_agv.timestamp = current_time;
        if(local_pose_graph_.agv_states.size() < min_keyframes_){
            new_agv.state = new_measurements.agv_odom;
            new_agv.covariance = new_measurements.agv_odom_covariance;
        }
        else{

            new_agv.state = local_pose_graph_.agv_states.back().state;
            new_agv.covariance = local_pose_graph_.agv_states.back().covariance;
        }
        
        local_pose_graph_.agv_states.push_back(new_agv);

        RCLCPP_INFO(this->get_logger(), "Adding new AGV node at timestamp %.2f: [%f, %f, %f, %f]", current_time.seconds(),
                        new_agv.state[0], new_agv.state[1], new_agv.state[2], new_agv.state[3]);


        // Similarly, create a new UAV node.
        State new_uav;  
        new_uav.timestamp = current_time;
        if(local_pose_graph_.uav_states.size() < min_keyframes_){
            Sophus::SE3d pred_That_uav_w = transformSE3FromPoseMsg(opt_relative_pose_.pose.pose)*(build_transformation_SE3(0.0,0.0,agv_odom_pos_).inverse());
            new_uav.state = transformSE3ToState(pred_That_uav_w.inverse());
            new_uav.covariance = new_measurements.uav_odom_covariance;
        }
        else{

            new_uav.state = local_pose_graph_.uav_states.back().state;
            new_uav.covariance = local_pose_graph_.uav_states.back().covariance;
        }
        
        local_pose_graph_.uav_states.push_back(new_uav);
   
        RCLCPP_INFO(this->get_logger(), "Adding new UAV node at timestamp %.2f: [%f, %f, %f, %f]", current_time.seconds(),
                        new_uav.state[0], new_uav.state[1], new_uav.state[2], new_uav.state[3]);


        // Convert the PCL point cloud to a ROS message
        sensor_msgs::msg::PointCloud2 source_cloud_msg, target_cloud_msg;
        pcl::toROSMsg(*local_pose_graph_.measurements.back().agv_scan, source_cloud_msg);
        pcl::toROSMsg(*local_pose_graph_.measurements.back().uav_scan, target_cloud_msg);

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

        // Remove old nodes outside the sliding window in the local graph. There is the same number for uav, agv and transform, as they come in triplets
        while (!local_pose_graph_.agv_states.empty() &&
               (current_time - local_pose_graph_.agv_states.front().timestamp).seconds() > global_opt_window_s_) {
            
            //Add these nodes to the global graph
            global_pose_graph_.agv_states.push_back(local_pose_graph_.agv_states.front());
            global_pose_graph_.uav_states.push_back(local_pose_graph_.uav_states.front());
            global_pose_graph_.measurements.push_back(local_pose_graph_.measurements.front());

            //Remove them from the local graph
            local_pose_graph_.agv_states.pop_front();
            local_pose_graph_.uav_states.pop_front();
            local_pose_graph_.measurements.pop_front();

        }

        //Check if we have enough KFs for each robot
        size_t local_graph_nodes = std::min(local_pose_graph_.uav_states.size(), local_pose_graph_.agv_states.size());

        //Update the new relative transform for all the previous nodes in the window
        for(size_t i = 0; i < local_graph_nodes - 1; ++i){
            local_pose_graph_.measurements[i].relative_transform = local_pose_graph_.measurements.back().relative_transform;
        }

        if(local_graph_nodes >= min_keyframes_){

            RCLCPP_INFO(this->get_logger(), "Optimizing trajectory of %ld AGV nodes and %ld UAV nodes",
                            local_pose_graph_.agv_states.size(), local_pose_graph_.uav_states.size());


            //Update transforms after convergence
            if(run_posegraph_optimization(local_pose_graph_)){
            
                // if(moving_average_){
                //     // /*Run moving average*/
                //     auto smoothed_state = moving_average(opt_state_);
                //     //Update for initial estimation of following step
                //     opt_state_.state = smoothed_state;
                // }

                State& agv_node = local_pose_graph_.agv_states.back();
                State& uav_node = local_pose_graph_.uav_states.back();
                Measurements& m = local_pose_graph_.measurements.back();

                Sophus::SE3d That_ws = build_transformation_SE3(m.agv_roll, m.agv_pitch, agv_node.state);
                Sophus::SE3d That_wt = build_transformation_SE3(m.uav_roll, m.uav_pitch, uav_node.state);

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
        
        //Check if we have enough KFs for global optimization
        int global_graph_nodes = std::min(global_pose_graph_.uav_states.size(),  global_pose_graph_.agv_states.size());
        if(global_optimization_ && global_graph_nodes >= min_keyframes_*3){
            
            RCLCPP_INFO(this->get_logger(), "[Eliko global_opt node] Performing global optimization with %ld Keyframes", global_pose_graph_.uav_states.size());

            if(run_posegraph_optimization(global_pose_graph_)){

                publishGlobalOptimizedPoses(global_pose_graph_.agv_states, global_pose_graph_.measurements, current_time, odom_tf_agv_t_, global_poses_agv_publisher_);
                publishGlobalOptimizedPoses(global_pose_graph_.uav_states, global_pose_graph_.measurements, current_time, odom_tf_agv_t_, global_poses_uav_publisher_);

            }

            else{
                RCLCPP_INFO(this->get_logger(), "[Eliko global_opt node] Global optimizer did not converge");
            }

            // Finally, reset the global pose graph:
            global_pose_graph_.agv_states.clear();
            global_pose_graph_.uav_states.clear();
            global_pose_graph_.measurements.clear();
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
            reg.setInputTarget(target_cloud);
            reg.setInputSource(source_cloud);

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

    // ---------------- Pose Graph Optimization -------------------
    //
    // In this function we build a Ceres problem that fuses:
    // - A prior on each node and the previous.
    // - Inter-robot UWB factor linking nodes based on relative transform estimation input
    // - Intra-robot ICP and odometry factors linking consecutive nodes.
    // - Inter-robot ICP factors linking nodes from the two robots.
    
    bool run_posegraph_optimization(PoseGraph &graph) {

        ceres::Problem problem;

        // For convenience, alias the vectors.
        std::deque<State>& agv = graph.agv_states;
        std::deque<State>& uav = graph.uav_states;
        std::deque<Measurements>& m = graph.measurements;


        //Add parameter blocks and manifold parametrization
        for (size_t i = 0; i < agv.size(); ++i) {
            problem.AddParameterBlock(agv[i].state.data(), 4);
            problem.SetManifold(agv[i].state.data(), new StateManifold());
        }

        //Add parameter blocks and manifold parameterization
        for (size_t i = 0; i < uav.size(); ++i) {
            problem.AddParameterBlock(uav[i].state.data(), 4);
            problem.SetManifold(uav[i].state.data(), new StateManifold());
        }
    

        // Add residual based on UWB relative transform

        // Ensure both trajectories have the same number of nodes.
        size_t num_nodes = std::min(agv.size(), uav.size());
        for (size_t i = 0; i < num_nodes; ++i) {
            State& agv_node = agv[i];
            State& uav_node = uav[i];
        
            Sophus::SE3d T = m[i].relative_transform; //use the latest updated for each node

            //Unflatten matric to extract the covariance
            Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    cov(i,j) = opt_relative_pose_.pose.covariance[i * 6 + j];
                }
            }

            RCLCPP_WARN(this->get_logger(), "Adding UWB residuals for inter-robot pairs index %ld", i);


            ceres::CostFunction* rel_cost = RelativeTransformResidual::Create(T, cov, m[i].uav_roll, m[i].uav_pitch);
            problem.AddResidualBlock(rel_cost, nullptr, agv_node.state.data(), uav_node.state.data());
        }

        // Add inter-robot ICP factors.
        for (size_t i = 0; i < agv.size(); ++i) {

            if (m[i].agv_scan->points.empty() ||
                m[i].uav_scan->points.empty()) {
                RCLCPP_WARN(this->get_logger(), "Skipping ICP for inter-robot pairs index %ld. One of them is empty", i);
                continue;
            }

            RCLCPP_WARN(this->get_logger(), "Computing ICP for inter-robot pairs index %ld", i);

            //Get initial solution
            Sophus::SE3d T_icp_init = m[i].relative_transform; //use the latest updated for each node

            Eigen::Matrix4f T_icp = T_icp_init.cast<float>().matrix(); //Here, we lose precision
            double fitness = 0.0;

            if(!run_icp(m[i].agv_scan, m[i].uav_scan, T_icp, fitness, icp_type_)){
                continue;
            };


            double roll_t = m[i].uav_roll;
            double roll_s = m[i].agv_roll;
            double pitch_t = m[i].uav_pitch;
            double pitch_s = m[i].agv_pitch;

            
            //Transform to the robots body frame, using the corrected orthogonal rotation matrix
            Sophus::SE3f T_icp_uav_agv = T_uav_lidar_.cast<float>() * Sophus::SE3f(T_icp) * T_agv_lidar_.inverse().cast<float>();

            log_transformation_matrix(T_icp_uav_agv.cast<double>().matrix());

            ceres::CostFunction* icp_cost = ICPResidual::Create(T_icp_uav_agv, fitness, roll_t, roll_s, pitch_t, pitch_s);
            problem.AddResidualBlock(icp_cost, nullptr,
                                    agv[i].state.data(), uav[i].state.data());
        }


        // Add intra-robot ICP factors for AGV.
        if (agv.size() >= 2) {
            for (size_t i = 0; i + 1 < agv.size(); ++i) {

                double roll_t = m[i+1].agv_roll;
                double roll_s = m[i].agv_roll;
                double pitch_t = m[i+1].agv_pitch;
                double pitch_s = m[i].agv_pitch;

                Sophus::SE3d odom_T_s = build_transformation_SE3(roll_s, pitch_s, m[i].agv_odom);
                Sophus::SE3d odom_T_t = build_transformation_SE3(roll_t, pitch_t, m[i+1].agv_odom);
                Sophus::SE3d odom_T_ts = odom_T_t.inverse()*odom_T_s;

                RCLCPP_WARN(this->get_logger(), "Adding odometry residuals for for AGV pairs %ld and %ld", i, i+1);


                ceres::CostFunction* cost_odom_agv = OdometryResidual::Create(odom_T_ts, roll_t, roll_s, pitch_t, pitch_s, m[i].agv_odom_covariance);
                problem.AddResidualBlock(cost_odom_agv, nullptr, agv[i].state.data(), agv[i+1].state.data());


                if (m[i].agv_scan->points.empty() ||
                    m[i+1].agv_scan->points.empty()) {
                    RCLCPP_WARN(this->get_logger(), "Skipping ICP for AGV pairs %ld and %ld. One of them is empty", i, i+1);
                    continue;
                }

                RCLCPP_WARN(this->get_logger(), "Computing ICP for AGV nodes %ld and %ld.", i, i+1);

                Eigen::Matrix4f T_icp = odom_T_ts.cast<float>().matrix();
                double fitness = 0.0;

                if(!run_icp(m[i].agv_scan, m[i+1].agv_scan, T_icp, fitness, icp_type_)){
                    continue;
                };

                //Transform to the robots body frame
                Sophus::SE3f T_icp_agv = T_agv_lidar_.cast<float>() * Sophus::SE3f(T_icp) * T_agv_lidar_.inverse().cast<float>();

                log_transformation_matrix(T_icp_agv.cast<double>().matrix());


                ceres::CostFunction* icp_cost = ICPResidual::Create(T_icp_agv, fitness, roll_t, roll_s, pitch_t, pitch_s);
                problem.AddResidualBlock(icp_cost, nullptr,
                                        agv[i].state.data(), agv[i+1].state.data());
            }
        }
        
        // Add intra-robot ICP factors for UAV.
        if (uav.size() >= 2) {
            for (size_t i = 0; i + 1 < uav.size(); ++i) {

                double roll_t = m[i+1].uav_roll;
                double roll_s = m[i].uav_roll;
                double pitch_t = m[i+1].uav_pitch;
                double pitch_s = m[i].uav_pitch;

                Sophus::SE3d odom_T_s = build_transformation_SE3(roll_s, pitch_s, m[i].uav_odom);
                Sophus::SE3d odom_T_t = build_transformation_SE3(roll_t, pitch_t, m[i+1].uav_odom);
                Sophus::SE3d odom_T_ts = odom_T_t.inverse()*odom_T_s;

                RCLCPP_WARN(this->get_logger(), "Adding odometry residuals for for UAV pairs %ld and %ld", i, i+1);

                ceres::CostFunction* cost_odom_uav = OdometryResidual::Create(odom_T_ts, roll_t, roll_s, pitch_t, pitch_s, m[i].uav_odom_covariance);
                problem.AddResidualBlock(cost_odom_uav, nullptr, uav[i].state.data(), uav[i+1].state.data());

                if (m[i].uav_scan->points.empty() ||
                    m[i+1].uav_scan->points.empty()) {
                    RCLCPP_WARN(this->get_logger(), "Skipping ICP for UAV pairs %ld and %ld. One of them is empty", i, i+1);
                    continue;
                }


                Eigen::Matrix4f T_icp = odom_T_ts.cast<float>().matrix();
                double fitness = 0.0;

                if(!run_icp(m[i].uav_scan, m[i+1].uav_scan, T_icp, fitness, icp_type_)){
                    continue;
                };

                RCLCPP_WARN(this->get_logger(), "Computing ICP for UAV nodes %ld and %ld.", i, i+1);

                //Transform to the robots body frame
                Sophus::SE3f T_icp_uav = T_uav_lidar_.cast<float>() * Sophus::SE3f(T_icp) * T_uav_lidar_.inverse().cast<float>();

                log_transformation_matrix(T_icp_uav.cast<double>().matrix());

                ceres::CostFunction* icp_cost = ICPResidual::Create(T_icp_uav, fitness, roll_t, roll_s, pitch_t, pitch_s);

                problem.AddResidualBlock(icp_cost, nullptr,
                                        uav[i].state.data(), uav[i+1].state.data());
            }
        }

        // Configure solver options
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR; // ceres::SPARSE_NORMAL_CHOLESKY,  ceres::DENSE_QR

        // Logging
        options.minimizer_progress_to_stdout = true;

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

            size_t node_size = std::min(uav.size(), agv.size());

            for (size_t i = 0; i < node_size; ++i) {
                // Only add if this state was involved in any residual.
                covariance_blocks.emplace_back(uav[i].state.data(),
                                            uav[i].state.data());
                covariance_blocks.emplace_back(agv[i].state.data(),
                                            agv[i].state.data());
                }

            if (covariance.Compute(covariance_blocks, &problem)) {
                
                for (size_t i = 0; i < node_size; ++i) {
                    Eigen::Matrix4d cov_uav = Eigen::Matrix4d::Zero();
                    covariance.GetCovarianceBlock(uav[i].state.data(),
                                                uav[i].state.data(), cov_uav.data());
                    uav[i].covariance = cov_uav;

                    Eigen::Matrix4d cov_agv = Eigen::Matrix4d::Zero();
                    covariance.GetCovarianceBlock(agv[i].state.data(),
                                                agv[i].state.data(), cov_agv.data());
                    agv[i].covariance = cov_agv;
                }
                
                } else {
                
                RCLCPP_WARN(this->get_logger(), "Failed to compute covariance in pose graph optimization.");
                
                for (size_t i = 0; i < node_size; ++i) {
                    agv[i].covariance = uav[i].covariance = Eigen::Matrix4d::Identity();
                }
                
                }

            return true;

            } else {

                RCLCPP_WARN(this->get_logger(), "Failed to converge.");
                
                return false;
            }
                                    
        return false; 
            
    }

 
    // ---------- Residual to enforce relative transform constraint ----------
    //
    // This residual forces the relative transform computed from the current AGV and UAV
    // nodes to be consistent with the optimized transform that is computed at a higher frequency, using UWB measurements.
    //

struct RelativeTransformResidual {
    RelativeTransformResidual(const Sophus::SE3d& T_opt, const Eigen::Matrix4d& cov, double roll_uav, double pitch_uav)
        : T_opt_(T_opt), cov_(cov), uav_roll_(roll_uav), uav_pitch_(pitch_uav) {}

    template <typename T>
    bool operator()(const T* const agv_state, const T* const uav_state, T* residual) const {
        // Build homogeneous transforms from the state vectors.
        Sophus::SE3<T> w_T_s = buildTransformationSE3(agv_state, 0.0, 0.0);
        Sophus::SE3<T> w_T_t = buildTransformationSE3(uav_state, uav_roll_, uav_pitch_);
        
        // Compute the relative transform from AGV to UAV:
        Sophus::SE3<T> SE3_pred = w_T_t.inverse() * w_T_s;
        Sophus::SE3<T> SE3_meas = T_opt_.template cast<T>();

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

    static ceres::CostFunction* Create(const Sophus::SE3d& T_opt, const Eigen::Matrix4d& cov, double roll_uav, double pitch_uav) {
        return new ceres::AutoDiffCostFunction<RelativeTransformResidual, 4, 4, 4>(
            new RelativeTransformResidual(T_opt, cov, roll_uav, pitch_uav));
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

    const Sophus::SE3d T_opt_;
    const Eigen::Matrix4d cov_;
    const double uav_roll_;
    const double uav_pitch_;

};


struct OdometryResidual {
    OdometryResidual(const Sophus::SE3d& odom_T, double roll_t, double roll_s, double pitch_t, double pitch_s, const Eigen::Matrix4d& odom_covariance)
        : odom_T_(odom_T), roll_t_(roll_t), roll_s_(roll_s), pitch_t_(pitch_t), pitch_s_(pitch_s), odom_covariance_(odom_covariance) {}

    template <typename T>
    bool operator()(const T* const state_source, const T* const state_target, T* residual) const {
        
        // Build homogeneous transforms from state_i and state_j.
        // Here we assume states are 4-vectors: [x,y,z,yaw]. (Roll and pitch are fixed.)
        Sophus::SE3<T> w_T_s = buildTransformationSE3(state_source, roll_s_, pitch_s_);
        Sophus::SE3<T> w_T_t = buildTransformationSE3(state_target, roll_t_, pitch_t_);
        // Compute the predicted relative transform from node i to node j.
        Sophus::SE3<T> SE3_pred = w_T_t.inverse() * w_T_s;

        // Create Sophus SE3 objects from the 4x4 matrices.
        Sophus::SE3<T> SE3_meas = odom_T_.template cast<T>();

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
        Eigen::LLT<Eigen::Matrix4d> chol(odom_covariance_);
        Eigen::Matrix4d sqrt_inv_covariance = Eigen::Matrix4d(chol.matrixL().transpose()).inverse();
        
        Eigen::Matrix<T, 4, 1> weighted_residual = sqrt_inv_covariance.cast<T>() * error_vec;

        // Assign to residual
        residual[0] = weighted_residual[0];
        residual[1] = weighted_residual[1];
        residual[2] = weighted_residual[2];
        residual[3] = weighted_residual[3];

        return true;
    }

    static ceres::CostFunction* Create(const Sophus::SE3d& odom_T, double roll_t, double roll_s, double pitch_t, double pitch_s, const Eigen::Matrix4d& odom_covariance) {
        return new ceres::AutoDiffCostFunction<OdometryResidual, 4, 4, 4>(
            new OdometryResidual(odom_T, roll_t, roll_s, pitch_t, pitch_s, odom_covariance));
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

    const Sophus::SE3d odom_T_;
    const double roll_t_, roll_s_;
    const double pitch_t_, pitch_s_;
    const Eigen::Matrix4d odom_covariance_;

};


// Residual to enforce that the relative transform between nodes 
// (computed from the states) is close to the measured relative transform from scan matching.
struct ICPResidual {
    ICPResidual(const Sophus::SE3f& T_icp, const double& fitness, double roll_t, double roll_s, double pitch_t, double pitch_s)
        : T_icp_(T_icp), fitness_(fitness), roll_t_(roll_t), roll_s_(roll_s), pitch_t_(pitch_t), pitch_s_(pitch_s) {}

    template <typename T>
    bool operator()(const T* const state_source, const T* const state_target, T* residual) const {
        // Build homogeneous transforms from state_i and state_j.
        // Here we assume states are 4-vectors: [x,y,z,yaw]. (Roll and pitch are fixed.)
        Sophus::SE3<T> w_T_s = buildTransformationSE3(state_source, roll_s_, pitch_s_);
        Sophus::SE3<T> w_T_t = buildTransformationSE3(state_target, roll_t_, pitch_t_);
        // Compute the predicted relative transform from node i to node j.
        Sophus::SE3<T> SE3_pred = w_T_t.inverse() * w_T_s;

        // Create Sophus SE3 objects from the 4x4 matrices.
        Sophus::SE3<T> SE3_meas = T_icp_.template cast<T>();

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

        // Scale by the fitness factor.
        Eigen::Matrix<T, 4, 1> scaled_error = error_vec / T(fitness_);
        
        residual[0] = scaled_error[0];
        residual[1] = scaled_error[1];
        residual[2] = scaled_error[2];
        residual[3] = scaled_error[3];

        return true;
    }

    static ceres::CostFunction* Create(const Sophus::SE3f& T_icp, 
                                        const double& fitness,
                                        double roll_t, double roll_s,
                                        double pitch_t, double pitch_s) {
        return new ceres::AutoDiffCostFunction<ICPResidual, 4, 4, 4>(
            new ICPResidual(T_icp, fitness, roll_t, roll_s, pitch_t, pitch_s));
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
        
        const Sophus::SE3f T_icp_;
        const double fitness_;
        const double roll_t_, roll_s_;
        const double pitch_t_, pitch_s_;

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
    geometry_msgs::msg::PoseWithCovarianceStamped latest_relative_pose_, opt_relative_pose_;
    double min_keyframes_;
    bool global_optimization_;

    // For pose graph (multi-robot) optimization:
    PoseGraph local_pose_graph_, global_pose_graph_;
    State init_state_;
    
    // Publishers/Broadcasters
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::string eliko_frame_id_, uav_frame_id_;
    std::string odom_tf_agv_s_, odom_tf_agv_t_;
    std::string odom_tf_uav_s_, odom_tf_uav_t_;

    std::deque<State> moving_average_states_uav_, moving_average_states_agv_;

    double global_opt_window_s_, global_opt_rate_s_;
    double measurement_stdev_, measurement_covariance_;
    double moving_average_window_s_;
    double moving_average_;
    double uav_roll_, uav_pitch_, uav_yaw_;

    Eigen::Vector4d uav_odom_pos_;         // Current UAV odometry position 
    Eigen::Vector4d agv_odom_pos_;        // Current AGV odometry position
    Eigen::Matrix4d uav_odom_covariance_;  // UAV odometry covariance
    Eigen::Matrix4d agv_odom_covariance_;  // AGV odometry covariance
    geometry_msgs::msg::Vector3Stamped last_uav_vel_msg_;
    bool last_uav_vel_initialized_;

};
int main(int argc, char** argv) {

    rclcpp::init(argc, argv);
    auto node = std::make_shared<FusionOptimizationNode>();
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
