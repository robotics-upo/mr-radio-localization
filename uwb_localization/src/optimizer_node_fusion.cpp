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
#include <nav_msgs/msg/odometry.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>

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



struct State {
    Eigen::Vector4d state; // [x,y,z,yaw]
    Eigen::Matrix4d covariance;
    double roll;
    double pitch;
    rclcpp::Time timestamp;  // e.g., seconds since epoch

    // Pointer to an associated point cloud (keyframe scan).
    pcl::PointCloud<pcl::PointXYZ>::Ptr scan;

    // Constructor to initialize the pointer.
    State() : scan(new pcl::PointCloud<pcl::PointXYZ>) {}
};

// In addition, we create a structure for a pose graph which holds separate trajectories for each robot.
struct PoseGraph {
    std::deque<State> agv_states;
    std::deque<State> uav_states;
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
    
    
    dji_attitude_sub_ = this->create_subscription<geometry_msgs::msg::QuaternionStamped>(
                "/dji_sdk/attitude", 10, std::bind(&FusionOptimizationNode::attitude_cb_, this, std::placeholders::_1));


    //Option 1: get odometry through topics -> includes covariance
    std::string odom_topic_agv = "/arco/idmind_motors/odom"; //or "/agv/odom"
    std::string odom_topic_uav = "/odom"; //or "/uav/odom"

    
    // uav_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    // odom_topic_uav, 10, std::bind(&FusionOptimizationNode::uav_odom_cb_, this, std::placeholders::_1));

    // agv_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    // odom_topic_agv, 10, std::bind(&FusionOptimizationNode::agv_odom_cb_, this, std::placeholders::_1));


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

    rclcpp::SensorDataQoS qos; // Use a QoS profile compatible with sensor data
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
    init_state_.roll = 0.0;
    init_state_.pitch = 0.0;
    init_state_.timestamp = this->get_clock()->now();

    moving_average_ = true;
    moving_average_window_s_ = global_opt_window_s_ / 2.0;

    // Initialize odometry positions and errors
    uav_odom_pos_ = Eigen::Vector4d::Zero();
    agv_odom_pos_ = Eigen::Vector4d::Zero();

    latest_relative_available_ = false;
    point_to_plane_ = true;
            

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

        RCLCPP_WARN(this->get_logger(), "Received AGV odometry.");

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



    void global_opt_cb_() {

        rclcpp::Time current_time = this->get_clock()->now();

        //Read the transform (if odom topics not available)
        try {
            auto transform_agv = tf_buffer_agv_->lookupTransform(odom_tf_agv_t_, odom_tf_agv_s_, rclcpp::Time(0));
            Sophus::SE3d T_agv_odom = transformSE3FromMsg(transform_agv);
            agv_odom_pos_ = transformSE3ToState(T_agv_odom);
            } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "Could not get transform for AGV: %s", ex.what());
            return;
            }

        try {
            auto transform_uav = tf_buffer_uav_->lookupTransform(odom_tf_uav_t_, odom_tf_uav_s_, rclcpp::Time(0));
            Sophus::SE3d T_uav_odom = transformSE3FromMsg(transform_uav);
            uav_odom_pos_ = transformSE3ToState(T_uav_odom);
            } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "Could not get transform for UAV: %s", ex.what());
            }

        // Create a new AGV node from the current odometry.
        State new_agv;

        // if(pose_graph_.agv_states.empty()){
        //     new_agv.state = init_state_.state;
        //     new_agv.covariance = init_state_.covariance;
        // }
        // else {
        //     new_agv.state = pose_graph_.agv_states.back().state;
        //     new_agv.covariance = pose_graph_.agv_states.back().covariance;
        //}

        new_agv.state = agv_odom_pos_;
        new_agv.covariance = Eigen::Matrix4d::Identity() * 1e-6;

        new_agv.timestamp = current_time;
        new_agv.roll = 0.0;   // or update with additional sensors
        new_agv.pitch = 0.0;

        //Downsample the latest scan and store as keyframe
        *(new_agv.scan) = *(downsamplePointCloud(agv_cloud_, 0.05f));

        pose_graph_.agv_states.push_back(new_agv);

        RCLCPP_INFO(this->get_logger(), "Adding new AGV node at timestamp %.2f: [%f, %f, %f, %f]", current_time.seconds(),
                        new_agv.state[0], new_agv.state[1], new_agv.state[2], new_agv.state[3]);


        //Check if there is something we can use for initial relative solution
        if((current_time - latest_relative_pose_.header.stamp).seconds() > global_opt_window_s_){
            latest_relative_available_ = false;
        }
        else {
            latest_relative_available_ = true;
        }
        
        // Similarly, create a new UAV node.
        State new_uav;
        new_uav.timestamp = current_time;
        new_uav.roll = roll_uav_;  // from attitude callback
        new_uav.pitch = pitch_uav_;

        if(pose_graph_.uav_states.empty()){
            new_uav.state = init_state_.state;
            new_uav.covariance = init_state_.covariance;
        }
        
        else if(latest_relative_available_){
            Sophus::SE3d pred_That_uav_w = transformSE3FromPoseMsg(latest_relative_pose_.pose.pose)*(build_transformation_SE3(0.0,0.0,agv_odom_pos_).inverse());
            new_uav.state = transformSE3ToState(pred_That_uav_w.inverse());
            //Unflatten matrix to extract the covariance
            Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    cov(i,j) = latest_relative_pose_.pose.covariance[i * 6 + j];
                }
            }
            new_uav.covariance = cov;
        }
        else {
            new_uav.state = pose_graph_.uav_states.back().state;
            new_uav.covariance = pose_graph_.uav_states.back().covariance;
        }
        
        //Downsample the latest scan and store as keyframe
        *(new_uav.scan) = *(downsamplePointCloud(uav_cloud_, 0.05f));

        pose_graph_.uav_states.push_back(new_uav);
        
        RCLCPP_INFO(this->get_logger(), "Adding new UAV node at timestamp %.2f: [%f, %f, %f, %f]", current_time.seconds(),
                        new_uav.state[0], new_uav.state[1], new_uav.state[2], new_uav.state[3]);


        // Convert the PCL point cloud to a ROS message
        sensor_msgs::msg::PointCloud2 source_cloud_msg, target_cloud_msg;
        pcl::toROSMsg(*pose_graph_.agv_states.back().scan, source_cloud_msg);
        pcl::toROSMsg(*pose_graph_.uav_states.back().scan, target_cloud_msg);

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

        // Remove old nodes outside the sliding window.
        while (!pose_graph_.agv_states.empty() &&
               (current_time - pose_graph_.agv_states.front().timestamp).seconds() > global_opt_window_s_) {
            pose_graph_.agv_states.pop_front();
        }
        while (!pose_graph_.uav_states.empty() &&
               (current_time - pose_graph_.uav_states.front().timestamp).seconds() > global_opt_window_s_) {
            pose_graph_.uav_states.pop_front();
        }

        //Check if we have enough KFs for each robot
        if(pose_graph_.uav_states.size() < min_keyframes_ || pose_graph_.agv_states.size() < min_keyframes_){
            return;
        }


        RCLCPP_INFO(this->get_logger(), "Optimizing trajectory of %ld AGV nodes and %ld UAV nodes",
                        pose_graph_.agv_states.size(), pose_graph_.uav_states.size());


        //Update transforms after convergence
        if(run_posegraph_optimization()){
        
            // if(moving_average_){
            //     // /*Run moving average*/
            //     auto smoothed_state = moving_average(opt_state_);
            //     //Update for initial estimation of following step
            //     opt_state_.state = smoothed_state;
            // }

            State& agv_node = pose_graph_.agv_states.back();
            State& uav_node = pose_graph_.uav_states.back();

            Sophus::SE3d That_ws = build_transformation_SE3(0.0, 0.0, agv_node.state);
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
            RCLCPP_INFO(this->get_logger(), "[Eliko global_opt node] Optimizer did not converge");
        }
        

    }


    // Helper function to downsample a point cloud using a voxel grid filter.
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplePointCloud(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &input_cloud,
        float leaf_size = 0.05f) // default leaf size of 5cm
    {
        // Create a VoxelGrid filter and set the input cloud.
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(input_cloud);
        voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);

        // Create a new point cloud to hold the filtered data.
        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        voxel_filter.filter(*output_cloud);

        RCLCPP_DEBUG(this->get_logger(), "Pointcloud downsampled to %zu points", output_cloud->points.size());

        return output_cloud;
    }


    bool run_icp(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &source_cloud,
             const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &target_cloud, Eigen::Matrix4f &transformation, double &fitness, const bool &point_to_plane) const {
               
        
        if(point_to_plane){

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
            //icp.setMaximumIterations(50);
            icp.setMaxCorrespondenceDistance(0.25);
            
            pcl::PointCloud<pcl::PointNormal> aligned_cloud;
            icp.align(aligned_cloud);

            if (!icp.hasConverged()) {
                //RCLCPP_INFO(this->get_logger(), "ICP converged with score: %f", icp.getFitnessScore());
                //RCLCPP_WARN(this->get_logger(), "ICP did not converge.");
                return false;
            }
       
            // Get the transformation matrix
            transformation = icp.getFinalTransformation();
            fitness = icp.getFitnessScore();

            return true;
        }
        
        else {
            // Perform Point-to-Point ICP
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setInputSource(source_cloud);
            icp.setInputTarget(target_cloud);

            // Optionally, you can tune ICP parameters (e.g., maximum iterations, convergence criteria, etc.)
            //icp.setMaximumIterations(50);
            icp.setMaxCorrespondenceDistance(0.25);

            pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            icp.align(*aligned_cloud);

            if (!icp.hasConverged()) {
                //RCLCPP_INFO(this->get_logger(), "ICP converged with score: %f", icp.getFitnessScore());
                //RCLCPP_WARN(this->get_logger(), "ICP did not converge.");
                return false;
            }
       
            // Get the transformation matrix
            transformation = icp.getFinalTransformation();
            fitness = icp.getFitnessScore();

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
            double yaw;
            m.getRPY(roll_uav_, pitch_uav_, yaw);
        }

    
    // Normalize an angle to the range [-pi, pi]
    double normalize_angle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
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



    Sophus::SE3d build_transformation_SE3(double roll, double pitch, const Eigen::Vector4d& s) {
        Eigen::Vector3d t(s[0], s[1], s[2]);  // Use Vector3d instead of an incorrect Matrix type.
        Eigen::Matrix3d R = (Eigen::AngleAxisd(s[3], Eigen::Vector3d::UnitZ()) *
                             Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX())).toRotationMatrix();
        return Sophus::SE3d(R, t);
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
    // - Intra-robot ICP factors linking consecutive nodes.
    // - Inter-robot ICP factors linking nodes from the two robots.
    
    bool run_posegraph_optimization() {

        ceres::Problem problem;

        // For convenience, alias the vectors.
        std::deque<State>& agv = pose_graph_.agv_states;
        std::deque<State>& uav = pose_graph_.uav_states;

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

        
        // Add prior factors (anchor the graph)
        for (size_t i = 0; i + 1 < agv.size(); ++i) {
            Sophus::SE3d prior_T = build_transformation_SE3(agv[i].roll, agv[i].pitch, agv[i].state);
            ceres::CostFunction* prior_cost_agv = PriorResidual::Create(prior_T, agv[i+1].roll, agv[i+1].pitch, agv[i].covariance);
            problem.AddResidualBlock(prior_cost_agv, nullptr, agv[i+1].state.data());
        }

        for (size_t i = 0; i + 1 < uav.size(); ++i) {
            Sophus::SE3d prior_T = build_transformation_SE3(uav[i].roll, uav[i].pitch, uav[i].state);
            ceres::CostFunction* prior_cost_uav = PriorResidual::Create(prior_T, uav[i+1].roll, uav[i+1].pitch, uav[i].covariance);
            problem.AddResidualBlock(prior_cost_uav, nullptr, uav[i+1].state.data());
        }
    

        // Add residual based on UWB relative transform, if available
        if (latest_relative_available_) {
            // Ensure both trajectories have the same number of nodes.
            size_t num_nodes = std::min(agv.size(), uav.size());
            for (size_t i = 0; i < num_nodes; ++i) {
                State& agv_node = agv[i];
                State& uav_node = uav[i];

                Sophus::SE3d T = transformSE3FromPoseMsg(latest_relative_pose_.pose.pose);

                //Unflatten matric to extract the covariance
                Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
                for (size_t i = 0; i < 4; ++i) {
                    for (size_t j = 0; j < 4; ++j) {
                        cov(i,j) = latest_relative_pose_.pose.covariance[i * 6 + j];
                    }
                }

                ceres::CostFunction* rel_cost = RelativeTransformResidual::Create(T, cov, uav_node.roll, uav_node.pitch);
                problem.AddResidualBlock(rel_cost, nullptr, agv_node.state.data(), uav_node.state.data());
            }
        }

        // Add inter-robot ICP factors.
        for (size_t i = 0; i < agv.size(); ++i) {

            if (agv[i].scan->points.empty() ||
                uav[i].scan->points.empty()) {
                RCLCPP_WARN(this->get_logger(), "Skipping ICP for inter-robot pairs index %ld. One of them is empty", i);
                continue;
            }

            Eigen::Matrix4f T_icp = Eigen::Matrix4f::Zero();
            double fitness = 0.0;

            if(!run_icp(agv[i].scan, uav[i].scan, T_icp, fitness, point_to_plane_)){
                continue;
            };

            double roll_t = uav[i].roll;
            double roll_s = agv[i].roll;
            double pitch_t = uav[i].pitch;
            double pitch_s = agv[i].pitch;

            //Transform to the robots body frame
            Sophus::SE3d T_icp_uav_agv = T_uav_lidar_ * Sophus::SE3d(T_icp.cast<double>()) * T_agv_lidar_.inverse();

            ceres::CostFunction* icp_cost = ICPResidual::Create(T_icp_uav_agv, fitness, roll_t, roll_s, pitch_t, pitch_s);
            problem.AddResidualBlock(icp_cost, nullptr,
                                    agv[i].state.data(), uav[i].state.data());
        }


        // Add intra-robot ICP factors for AGV.
        if (agv.size() >= 2) {
            for (size_t i = 0; i + 1 < agv.size(); ++i) {

                if (agv[i].scan->points.empty() ||
                    agv[i+1].scan->points.empty()) {
                    RCLCPP_WARN(this->get_logger(), "Skipping ICP for AGV pairs %ld and %ld. One of them is empty", i, i+1);
                    continue;
                }

                //RCLCPP_WARN(this->get_logger(), "Computing ICP for AGV nodes %ld and %ld.", i, i+1);

                Eigen::Matrix4f T_icp = Eigen::Matrix4f::Zero();
                double fitness = 0.0;

                if(!run_icp(agv[i].scan, agv[i+1].scan, T_icp, fitness, point_to_plane_)){
                    continue;
                };

                //Transform to the robots body frame
                Sophus::SE3d T_icp_agv = T_agv_lidar_ * Sophus::SE3d(T_icp.cast<double>()) * T_agv_lidar_.inverse();

                ceres::CostFunction* icp_cost = ICPResidual::Create(T_icp_agv, fitness, 0.0, 0.0, 0.0, 0.0);
                problem.AddResidualBlock(icp_cost, nullptr,
                                        agv[i].state.data(), agv[i+1].state.data());
            }
        }
        
        // Add intra-robot ICP factors for UAV.
        if (uav.size() >= 2) {
            for (size_t i = 0; i + 1 < uav.size(); ++i) {

                if (uav[i].scan->points.empty() ||
                    uav[i+1].scan->points.empty()) {
                    RCLCPP_WARN(this->get_logger(), "Skipping ICP for UAV pairs %ld and %ld. One of them is empty", i, i+1);
                    continue;
                }

                //RCLCPP_WARN(this->get_logger(), "Computing ICP for UAV nodes %ld and %ld.", i, i+1);

                Eigen::Matrix4f T_icp = Eigen::Matrix4f::Zero();
                double fitness = 0.0;

                if(!run_icp(uav[i].scan, uav[i+1].scan, T_icp, fitness, point_to_plane_)){
                    continue;
                };

                double roll_t = uav[i+1].roll;
                double roll_s = uav[i].roll;
                double pitch_t = uav[i+1].pitch;
                double pitch_s = uav[i+1].pitch;

                //Transform to the robots body frame
                Sophus::SE3d T_icp_uav = T_uav_lidar_ * Sophus::SE3d(T_icp.cast<double>()) * T_uav_lidar_.inverse();

                ceres::CostFunction* icp_cost = ICPResidual::Create(T_icp_uav, fitness, roll_t, roll_s, pitch_t, pitch_s);

                problem.AddResidualBlock(icp_cost, nullptr,
                                        uav[i].state.data(), uav[i+1].state.data());
            }
        }

        // Configure solver options
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR; // ceres::SPARSE_NORMAL_CHOLESKY,  ceres::DENSE_QR

        // Logging
        //options.minimizer_progress_to_stdout = true;

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

            for (size_t i = 0; i < uav.size(); ++i) {
                // Only add if this state was involved in any residual.
                covariance_blocks.emplace_back(uav[i].state.data(),
                                            uav[i].state.data());
                }
            for (size_t i = 0; i < agv.size(); ++i) {
                covariance_blocks.emplace_back(agv[i].state.data(),
                                            agv[i].state.data());
            }


            if (covariance.Compute(covariance_blocks, &problem)) {
                
                for (size_t i = 0; i < uav.size(); ++i) {
                    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
                    covariance.GetCovarianceBlock(uav[i].state.data(),
                                                uav[i].state.data(), cov.data());
                    uav[i].covariance = cov;
                }

                for (size_t i = 0; i < agv.size(); ++i) {
                    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
                    covariance.GetCovarianceBlock(agv[i].state.data(),
                                                agv[i].state.data(), cov.data());
                    agv[i].covariance = cov;
                }
                
                } else {
                
                RCLCPP_WARN(this->get_logger(), "Failed to compute covariance in pose graph optimization.");
                
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
        : T_opt_(T_opt), cov_(cov), roll_uav_(roll_uav), pitch_uav_(pitch_uav) {}

    template <typename T>
    bool operator()(const T* const agv_state, const T* const uav_state, T* residual) const {
        // Build homogeneous transforms from the state vectors.
        Sophus::SE3<T> w_T_s = buildTransformationSE3(agv_state, 0.0, 0.0);
        Sophus::SE3<T> w_T_t = buildTransformationSE3(uav_state, roll_uav_, pitch_uav_);
        
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
    const double roll_uav_;
    const double pitch_uav_;

};


struct PriorResidual {
    PriorResidual(const Sophus::SE3d& prior_T, const double &roll, const double &pitch, const Eigen::Matrix4d& state_covariance)
        : prior_T_(prior_T), roll_(roll), pitch_(pitch), state_covariance_(state_covariance) {}

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
        error_vec.template segment<3>(0) = xi.template segment<3>(0); // translation error
        error_vec[3] = xi[5];  // use the z-component (yaw) of the rotation error

        // Scale by the square root of the inverse covariance matrix
        Eigen::LLT<Eigen::Matrix4d> chol(state_covariance_);
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

    static ceres::CostFunction* Create(const Sophus::SE3d& prior_T, const double& roll, const double &pitch, const Eigen::Matrix4d& state_covariance) {
        return new ceres::AutoDiffCostFunction<PriorResidual, 4, 4>(
            new PriorResidual(prior_T, roll, pitch, state_covariance));
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
    const Eigen::Matrix4d state_covariance_;
};


// Residual to enforce that the relative transform between nodes 
// (computed from the states) is close to the measured relative transform from scan matching.
struct ICPResidual {
    ICPResidual(const Sophus::SE3d& T_icp, const double& fitness, double roll_t, double roll_s, double pitch_t, double pitch_s)
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

    static ceres::CostFunction* Create(const Sophus::SE3d& T_icp, 
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
        
        const Sophus::SE3d T_icp_;
        const double fitness_;
        const double roll_t_, roll_s_;
        const double pitch_t_, pitch_s_;

    };

    
    // Subscriptions
    rclcpp::Subscription<eliko_messages::msg::DistancesList>::SharedPtr eliko_distances_sub_;
    rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr dji_attitude_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr uav_odom_sub_, agv_odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_source_sub_, pcl_target_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr optimized_tf_sub_;

    // Timers
    rclcpp::TimerBase::SharedPtr global_optimization_timer_, window_opt_timer_;
    //Service client for visualization
    rclcpp::Client<uwb_localization::srv::UpdatePointClouds>::SharedPtr pcl_visualizer_client_;

    //Pose publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_uav_publisher_, pose_agv_publisher_;


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
    bool point_to_plane_; 
    geometry_msgs::msg::PoseWithCovarianceStamped latest_relative_pose_;
    bool latest_relative_available_;
    double min_keyframes_;

    // For pose graph (multi-robot) optimization:
    PoseGraph pose_graph_;
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
    double roll_uav_, pitch_uav_;

    Eigen::Vector4d uav_odom_pos_;         // Current UAV odometry position 
    Eigen::Vector4d agv_odom_pos_;        // Current AGV odometry position
    Eigen::Matrix4d uav_odom_covariance_;  // UAV odometry covariance
    Eigen::Matrix4d agv_odom_covariance_;  // AGV odometry covariance

};
int main(int argc, char** argv) {

    rclcpp::init(argc, argv);
    auto node = std::make_shared<FusionOptimizationNode>();
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
