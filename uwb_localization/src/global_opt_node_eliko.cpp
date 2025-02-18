#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
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
#include <utility>
#include <unordered_map>
#include <chrono>


struct Measurement {
    rclcpp::Time timestamp;   // Time of the measurement
    std::string tag_id;       // ID of the tag
    std::string anchor_id;    // ID of the anchor
    double distance;          // Measured distance (in meters)
};


struct State {
    Eigen::Vector4d state; // [x,y,z,yaw]
    Eigen::Matrix4d covariance;
    double roll;
    double pitch;
    rclcpp::Time timestamp;  // e.g., seconds since epoch
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


class ElikoGlobalOptNode : public rclcpp::Node {

public:

    ElikoGlobalOptNode() : Node("eliko_global_opt_node") {
    

    //Subscribe to distances publisher
    eliko_distances_sub_ = this->create_subscription<eliko_messages::msg::DistancesList>(
                "/eliko/Distances", 10, std::bind(&ElikoGlobalOptNode::distances_coords_cb_, this, std::placeholders::_1));

    
    //Option 1: get odometry through topics -> includes covariance
    std::string odom_topic_agv = "/arco/idmind_motors/odom"; //or "/agv/odom" "/arco/idmind_motors/odom"
    std::string odom_topic_uav = "/uav/odom"; //or "/uav/odom"
    std::string vel_topic_uav = "/dji_sdk/velocity";

    rclcpp::SensorDataQoS qos; // Use a QoS profile compatible with sensor data
    dji_attitude_sub_ = this->create_subscription<geometry_msgs::msg::QuaternionStamped>(
                "/dji_sdk/attitude", qos, std::bind(&ElikoGlobalOptNode::attitude_cb_, this, std::placeholders::_1));

    // uav_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    // odom_topic_uav, qos, std::bind(&ElikoGlobalOptNode::uav_odom_cb_, this, std::placeholders::_1));

    uav_vel_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
        vel_topic_uav, qos, std::bind(&ElikoGlobalOptNode::uav_vel_cb_, this, std::placeholders::_1));

    agv_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    odom_topic_agv, qos, std::bind(&ElikoGlobalOptNode::agv_odom_cb_, this, std::placeholders::_1));
    
    //Option 2: get odometry through tf readings -> only transform
    odom_tf_agv_s_ = "agv_odom"; //source "agv_odom"
    odom_tf_agv_t_ = "world"; //target "world"
    odom_tf_uav_s_ = "uav_odom"; // "uav_odom"
    odom_tf_uav_t_ = "world"; //"world"

    // Set up transform listener for UAV odometry.
    tf_buffer_uav_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_uav_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_uav_);

    // Set up transform listener for AGV odometry.
    tf_buffer_agv_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_agv_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_agv_);
    
    // Create publisher/broadcaster for optimized transformation
    tf_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("eliko_optimization_node/optimized_T", 10);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    global_opt_window_s_ = 5.0; //size of the sliding window in seconds
    global_opt_rate_s_ = 0.1 * global_opt_window_s_; //rate of the optimization -> displace the window 10% of its size
    min_measurements_ = 50.0; //min number of measurements for running optimizer
    measurement_stdev_ = 0.1; //10 cm measurement noise
    measurement_covariance_ = measurement_stdev_ * measurement_stdev_;

    global_optimization_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(int(global_opt_rate_s_*1000)), std::bind(&ElikoGlobalOptNode::global_opt_cb_, this));

    anchor_positions_ = {
        {"0x0009D6", {-0.32, 0.3, 0.875}}, {"0x0009E5", {0.32, -0.3, 0.875}},
        {"0x0016FA", {0.32, 0.3, 0.33}}, {"0x0016CF", {-0.32, -0.3, 0.33}}
    };
    
    tag_positions_ = {
        {"0x001155", {-0.24, -0.24, -0.06}}, {"0x001397", {0.24, 0.24, -0.06}}  //{"0x001155", {-0.24, -0.24, -0.06}}, {"0x001397", {0.24, 0.24, -0.06}}
    };

    agv_odom_frame_id_ = "arco/eliko"; //frame of the eliko system-> arco/eliko, for simulation use "agv_gt" for ground truth, "agv_odom" for odometry w/ errors
    uav_odom_frame_id_ = "odom"; //frame of the UAV system-> uav_gt for ground truth, "uav_odom" for odometry w/ errors
    uav_opt_frame_id_ = "uav_opt"; 
    agv_opt_frame_id_ = "agv_opt"; 

    //Initial values for state
    init_state_.state = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
    init_state_.covariance = Eigen::Matrix4d::Identity(); //
    init_state_.roll = 0.0;
    init_state_.pitch = 0.0;
    init_state_.timestamp = this->get_clock()->now();

    opt_state_ = init_state_;

    moving_average_ = true;
    moving_average_window_s_ = global_opt_window_s_ / 2.0;

    // Initialize odometry positions and errors
    uav_odom_pos_ = Eigen::Vector4d::Zero();
    uav_last_odom_pos_ = Eigen::Vector4d::Zero();
    last_uav_vel_initialized_ = false;
    agv_odom_pos_ = Eigen::Vector4d::Zero();
    agv_last_odom_pos_ = Eigen::Vector4d::Zero();
    min_traveled_distance_ = 0.1;
    min_traveled_angle_ = 10.0 * M_PI / 180.0;
    uav_distance_ = agv_distance_ = uav_angle_ = agv_angle_ = 0.0;

    odom_error_distance_ = 2.0;
    odom_error_angle_ = 2.0;
    
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

        // Similarly, compute AGV translation and yaw
        agv_distance_ += (agv_odom_pos_.head<3>() - agv_last_odom_pos_.head<3>()).norm();
        agv_angle_ += normalize_angle(agv_odom_pos_[3] - agv_last_odom_pos_[3]); 

        // Extract covariance from the odometry message and store it in Eigen::Matrix4d
        uav_odom_covariance_ = Eigen::Matrix4d::Zero();
        uav_odom_covariance_(0, 0) = msg->pose.covariance[0];  // x variance
        uav_odom_covariance_(1, 1) = msg->pose.covariance[7];  // y variance
        uav_odom_covariance_(2, 2) = msg->pose.covariance[14]; // z variance
        uav_odom_covariance_(3, 3) = msg->pose.covariance[35]; // yaw variance

        uav_last_odom_pos_ = uav_odom_pos_;
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
    }

    void distances_coords_cb_(const eliko_messages::msg::DistancesList::SharedPtr msg) {
    

        for (const auto& distance_msg : msg->anchor_distances) {
            Measurement measurement;
            measurement.timestamp = msg->header.stamp;
            measurement.tag_id = distance_msg.tag_sn;
            measurement.anchor_id = distance_msg.anchor_sn;
            measurement.distance = distance_msg.distance / 100.0; // Convert to meters

            global_measurements_.push_back(measurement);
        }

        RCLCPP_INFO(this->get_logger(), "[Eliko global node] Added %ld measurements. Total size: %ld", 
                    msg->anchor_distances.size(), global_measurements_.size());
    }


    void global_opt_cb_() {

        rclcpp::Time current_time = this->get_clock()->now();

        // Remove measurements older than the size of the sliding window
        while (!global_measurements_.empty() && (current_time - global_measurements_.front().timestamp).seconds() > global_opt_window_s_) {
            global_measurements_.pop_front();
        }

        // Ensure we have enough measurements
        if (global_measurements_.size() < min_measurements_) {
            RCLCPP_WARN(this->get_logger(), "[Eliko global_opt node] Not enough data to run optimization.");
            return;
        }


        // Check for measurements from both tags
        std::unordered_set<std::string> observed_tags;
        for (const auto& measurement : global_measurements_) {
            observed_tags.insert(measurement.tag_id);
        }
        if (observed_tags.size() < 2) {
            RCLCPP_WARN(this->get_logger(), "[Eliko global_opt node] Only one tag available. YAW IS NOT RELIABLE.");
            //return;
        }


        // //Read the transform (if odom topics not available)
        // try {
        //     auto transform_agv = tf_buffer_agv_->lookupTransform(odom_tf_agv_t_, odom_tf_agv_s_, rclcpp::Time(0));
        //     Sophus::SE3d T_agv_odom = transformSE3FromMsg(transform_agv);
        //     agv_odom_pos_ = transformSE3ToState(T_agv_odom);
        //     } catch (const tf2::TransformException &ex) {
        //     RCLCPP_WARN(this->get_logger(), "Could not get transform for AGV: %s", ex.what());
        //     }

        // try {
        //     auto transform_uav = tf_buffer_uav_->lookupTransform(odom_tf_uav_t_, odom_tf_uav_s_, rclcpp::Time(0));
        //     Sophus::SE3d T_uav_odom = transformSE3FromMsg(transform_uav);
        //     uav_odom_pos_ = transformSE3ToState(T_uav_odom);
        //     } catch (const tf2::TransformException &ex) {
        //     RCLCPP_WARN(this->get_logger(), "Could not get transform for UAV: %s", ex.what());
        //     }


        /*Check enough translation in the window*/

        if ((uav_distance_ < min_traveled_distance_ && uav_angle_ < min_traveled_angle_) && (agv_distance_ < min_traveled_distance_ && agv_angle_ < min_traveled_angle_)) {
            RCLCPP_WARN(this->get_logger(), "[Eliko global_opt node] Insufficient movement UAV = [%.2fm %.2fº], AGV= [%.2fm %.2fº]. Skipping optimization.", uav_distance_, uav_angle_ * 180.0/M_PI, agv_distance_, agv_angle_ * 180.0/M_PI);
            return;
        }

        //RCLCPP_WARN(this->get_logger(), "[Eliko global_opt node] Movement UAV = [%.2fm %.2fº], AGV= [%.2fm %.2fº].", uav_distance_, uav_angle_ * 180.0/M_PI, agv_distance_, agv_angle_ * 180.0/M_PI);


        // Compute odometry covariance based on distance increments from step to step 
        Eigen::Matrix4d predicted_motion_covariance = Eigen::Matrix4d::Zero();
        predicted_motion_covariance(0,0) = std::pow((2.0 * odom_error_distance_ / 100.0) * uav_distance_, 2.0);
        predicted_motion_covariance(1,1) = std::pow((2.0 * odom_error_distance_ / 100.0) * uav_distance_, 2.0);
        predicted_motion_covariance(2,2) = std::pow((2.0 * odom_error_distance_ / 100.0) * uav_distance_, 2.0);
        predicted_motion_covariance(3,3) = std::pow(((2.0 * odom_error_angle_ / 100.0) * M_PI / 180.0) * uav_angle_, 2.0);


        //Update odom for next optimization
        uav_distance_ = agv_distance_ = uav_angle_ = agv_angle_ = 0.0;

        //Optimize    
        RCLCPP_INFO(this->get_logger(), "[Eliko global_opt node] Optimizing trajectory of %ld measurements", global_measurements_.size());
        //Update transforms after convergence
        if(run_optimization(current_time, predicted_motion_covariance)){
        
            if(moving_average_){
                // /*Run moving average*/
                auto smoothed_state = moving_average(opt_state_);
                //Update for initial estimation of following step
                opt_state_.state = smoothed_state;
            }

            Sophus::SE3d That_ts = build_transformation_SE3(opt_state_.roll, opt_state_.pitch, opt_state_.state);

            // publish_transform(That_ts.inverse(), agv_odom_frame_id_, uav_opt_frame_id_, opt_state_.timestamp);
            // publish_transform(That_ts, uav_odom_frame_id_, agv_opt_frame_id_, opt_state_.timestamp);
            
            publish_pose(That_ts, opt_state_.covariance + predicted_motion_covariance, opt_state_.timestamp, uav_odom_frame_id_, tf_publisher_);

        }

        else{
            RCLCPP_INFO(this->get_logger(), "[Eliko global_opt node] Optimizer did not converge");
        }

    }



    void attitude_cb_(const geometry_msgs::msg::QuaternionStamped::SharedPtr msg) {

            const auto& q = msg->quaternion;
            // Convert the quaternion to roll, pitch, yaw
            tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
            tf2::Matrix3x3 m(tf_q);
            m.getRPY(uav_roll_, uav_pitch_, uav_yaw_);  // Get roll and pitch, but ignore yaw as we optimize it separately
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

    Eigen::Vector4d moving_average(const State &sample) {
        
            // Add the new sample with timestamp
            moving_average_states_.push_back(sample);

            // Remove samples that are too old or if we exceed the window size
            while (!moving_average_states_.empty() &&
                (sample.timestamp - moving_average_states_.front().timestamp > rclcpp::Duration::from_seconds(global_opt_window_s_))) {
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
                smoothed_state[3] = normalize_angle(smoothed_state[3]);
            }

            return smoothed_state;
    }

    void publish_transform(const Sophus::SE3d& T, const std::string &frame_id, const std::string &child_frame_id, const rclcpp::Time &current_time) {
        
        
        geometry_msgs::msg::TransformStamped that_st_msg;
        that_st_msg.header.stamp = current_time;
        that_st_msg.header.frame_id = frame_id;  // Adjust frame_id as needed
        that_st_msg.child_frame_id = child_frame_id;            // Adjust child_frame_id as needed

        // Extract translation
        Eigen::Vector3d translation_that_st = T.translation();

        that_st_msg.transform.translation.x = translation_that_st.x();
        that_st_msg.transform.translation.y = translation_that_st.y();
        that_st_msg.transform.translation.z = translation_that_st.z();

        // Convert Eigen rotation matrix of inverse transform to tf2 Matrix3x3
        Eigen::Matrix3d rotation_matrix_that_st = T.rotationMatrix();

        Eigen::Quaterniond q_that_st(rotation_matrix_that_st);

        that_st_msg.transform.rotation.x = q_that_st.x();
        that_st_msg.transform.rotation.y = q_that_st.y();
        that_st_msg.transform.rotation.z = q_that_st.z();
        that_st_msg.transform.rotation.w = q_that_st.w();

        // Broadcast the inverse transform
        tf_broadcaster_->sendTransform(that_st_msg);
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

    // Convert transform matrix from ROS msg to SE3
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

    Sophus::SE3d build_transformation_SE3(double roll, double pitch, const Eigen::Vector4d& s) {
        Eigen::Vector3d t(s[0], s[1], s[2]);  // Use Vector3d instead of an incorrect Matrix type.
        Eigen::Matrix3d R = (Eigen::AngleAxisd(s[3], Eigen::Vector3d::UnitZ()) *
                             Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX())).toRotationMatrix();
        return Sophus::SE3d(R, t);
    }


    // Run the optimization once all measurements are received
    bool run_optimization(const rclcpp::Time &current_time, const Eigen::Matrix4d& motion_covariance) {

            ceres::Problem problem;

            //Initial conditions
            State opt_state = opt_state_;
            opt_state.roll = uav_roll_;
            opt_state.pitch = uav_pitch_;
            
            //Update the timestamp
            opt_state.timestamp = current_time;

            // Create an instance of our custom manifold.
            ceres::Manifold* state_manifold = new StateManifold();
            // Attach it to the parameter block.
            problem.AddParameterBlock(opt_state.state.data(), 4);

            //Tell the optimizer to perform updates in the manifold space
            problem.SetManifold(opt_state.state.data(), state_manifold);

            // Define a robust kernel
            double huber_threshold = 2.5; // = residuals higher than 2.5 times sigma are outliers
            ceres::LossFunction* robust_loss = new ceres::HuberLoss(huber_threshold);

            Eigen::Matrix4d prior_covariance = opt_state_.covariance + motion_covariance;

            Sophus::SE3d prior_T = build_transformation_SE3(opt_state_.roll, opt_state_.pitch, opt_state_.state);
            // Add the prior residual with the full covariance
            ceres::CostFunction* prior_cost = PriorResidual::Create(prior_T, opt_state.roll, opt_state.pitch, prior_covariance);

            problem.AddResidualBlock(prior_cost, nullptr, opt_state.state.data());

            for (const auto& measurement : global_measurements_) {
                if (anchor_positions_.count(measurement.anchor_id) > 0 &&
                    tag_positions_.count(measurement.tag_id) > 0) {
                    Eigen::Vector3d anchor_pos = anchor_positions_[measurement.anchor_id];
                    Eigen::Vector3d tag_pos = tag_positions_[measurement.tag_id];

                    ceres::CostFunction* cost_function = UWBResidual::Create(
                        anchor_pos, tag_pos, measurement.distance, opt_state.roll, opt_state.pitch, measurement_stdev_);
                    problem.AddResidualBlock(cost_function, robust_loss, opt_state.state.data());
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
            //RCLCPP_INFO(this->get_logger(), summary.BriefReport().c_str());

            // Notify and update values if optimization converged
            if (summary.termination_type == ceres::CONVERGENCE){
                
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

  // Residual struct for Ceres
  struct UWBResidual {
        UWBResidual(const Eigen::Vector3d& anchor, const Eigen::Vector3d& tag, double measured_distance, double roll, double pitch, double measurement_stdev)
            : anchor_(anchor), tag_(tag), measured_distance_(measured_distance), roll_(roll), pitch_(pitch), measurement_stdev_inv_(1.0 / measurement_stdev) {}

        template <typename T>
        bool operator()(const T* const state, T* residual) const {

                // Build the 4x4 transformation matrix
                Sophus::SE3<T> SE3_rel = buildTransformationSE3(state, roll_, pitch_);

                // Transform the anchor point using the robot pose.
                Eigen::Matrix<T,3,1> anchor_vec;
                anchor_vec << T(anchor_(0)), T(anchor_(1)), T(anchor_(2));
                Eigen::Matrix<T,3,1> anchor_transformed = SE3_rel * anchor_vec;
                
                // Compute the predicted distance from the transformed anchor to the tag.
                Eigen::Matrix<T,3,1> tag_vec;
                tag_vec << T(tag_(0)), T(tag_(1)), T(tag_(2));
                T predicted_distance = (tag_vec - anchor_transformed).norm();

                // Residual as (measured - predicted)
                residual[0] = T(measured_distance_) - predicted_distance;
                residual[0] *= T(measurement_stdev_inv_);  // Scale residual

                //std::cout << "Measurement residual: " << residual[0] << std::endl;

                return true;
        }

      static ceres::CostFunction* Create(const Eigen::Vector3d& anchor, const Eigen::Vector3d& tag, double measured_distance, double roll, double pitch, double measurement_stdev) {
            return (new ceres::AutoDiffCostFunction<UWBResidual, 1, 4>(
                new UWBResidual(anchor, tag, measured_distance, roll, pitch, measurement_stdev)));
        }

        template <typename T>
        Sophus::SE3<T> buildTransformationSE3(const T* state, double roll, double pitch) const{
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

        const Eigen::Vector3d anchor_;
        const Eigen::Vector3d tag_;
        const double measured_distance_;
        const double roll_;
        const double pitch_;
        const double measurement_stdev_inv_;
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

    
    // Subscriptions
    rclcpp::Subscription<eliko_messages::msg::DistancesList>::SharedPtr eliko_distances_sub_;
    rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr dji_attitude_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr uav_odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr uav_vel_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr agv_odom_sub_;
    
    // Timers
    rclcpp::TimerBase::SharedPtr global_optimization_timer_;

    std::deque<Measurement> global_measurements_;
    
    // Publishers/Broadcasters
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr tf_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;


    // Transform buffers and listeners for each odometry source.
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_agv_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_agv_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_uav_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_uav_;

    std::unordered_map<std::string, Eigen::Vector3d> anchor_positions_;
    std::unordered_map<std::string, Eigen::Vector3d> tag_positions_;

    std::string agv_odom_frame_id_, uav_odom_frame_id_, uav_opt_frame_id_, agv_opt_frame_id_;
    std::string odom_tf_agv_s_, odom_tf_agv_t_;
    std::string odom_tf_uav_s_, odom_tf_uav_t_;

    State opt_state_;
    State init_state_;
    std::deque<State> moving_average_states_;

    bool moving_average_;
    size_t min_measurements_;
    double global_opt_window_s_, global_opt_rate_s_;
    double measurement_stdev_, measurement_covariance_;
    double moving_average_window_s_;

    Eigen::Vector4d uav_odom_pos_, uav_last_odom_pos_;         // Current UAV odometry position and last used for optimization
    Eigen::Vector4d agv_odom_pos_, agv_last_odom_pos_;        // Current AGV odometry position and last used for optimization
    Eigen::Matrix4d uav_odom_covariance_;  // UAV odometry covariance
    Eigen::Matrix4d agv_odom_covariance_;  // AGV odometry covariance
    geometry_msgs::msg::Vector3Stamped last_uav_vel_msg_;
    bool last_uav_vel_initialized_;

    double min_traveled_distance_, min_traveled_angle_;
    double uav_distance_, agv_distance_, uav_angle_, agv_angle_;
    double odom_error_distance_, odom_error_angle_;

    double uav_roll_, uav_pitch_, uav_yaw_;

};
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ElikoGlobalOptNode>();
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
