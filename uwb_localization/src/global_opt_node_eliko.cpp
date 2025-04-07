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

//#include "uwb_localization/algebraic_initialization.h"

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


struct Measurement {
    rclcpp::Time timestamp;   // Time of the measurement
    std::string tag_id;       // ID of the tag
    std::string anchor_id;    // ID of the anchor
    double distance;          // Measured distance (in meters)

    // Position of the tag in the UAV odometry frame.
    Eigen::Vector3d tag_odom_pose;
    // Position of the anchor in the AGV odometry frame.
    Eigen::Vector3d anchor_odom_pose;

    //Store the cumulative displacement (e.g., total distance traveled) for each sensor at the measurement time.
    double tag_cumulative_distance;
    double anchor_cumulative_distance;
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
    
    std::string odom_topic_agv = "/arco/idmind_motors/odom"; //or "/agv/odom" "/arco/idmind_motors/odom"
    std::string odom_topic_uav = "/uav/odom"; //or "/uav/odom"

    //Subscribe to distances publisher
    eliko_distances_sub_ = this->create_subscription<eliko_messages::msg::DistancesList>(
                "/eliko/Distances", 10, std::bind(&ElikoGlobalOptNode::distancesCoordsCb, this, std::placeholders::_1));

    rclcpp::SensorDataQoS qos; // Use a QoS profile compatible with sensor data

    agv_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    odom_topic_agv, qos, std::bind(&ElikoGlobalOptNode::agvOdomCb, this, std::placeholders::_1));
    
    uav_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_uav, qos, std::bind(&ElikoGlobalOptNode::uavOdomCb, this, std::placeholders::_1));
    
    // Create publisher/broadcaster for optimized transformation
    tf_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("eliko_optimization_node/optimized_T", 10);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    global_opt_rate_s_ = 0.1; //rate of the optimization
    min_measurements_ = 50.0; //min number of measurements for running optimizer
    measurement_stdev_ = 0.1; //10 cm measurement noise
    measurement_covariance_ = measurement_stdev_ * measurement_stdev_;

    anchor_positions_ = {
        {"0x0009D6", {-0.32, 0.3, 0.875}}, {"0x0009E5", {0.32, -0.3, 0.875}},
        {"0x0016FA", {0.32, 0.3, 0.33}}, {"0x0016CF", {-0.32, -0.3, 0.33}}
    };
    
    tag_positions_ = {
        {"0x001155", {-0.24, -0.24, -0.06}}, {"0x001397", {0.24, 0.24, -0.06}}  //{"0x001155", {-0.24, -0.24, -0.06}}, {"0x001397", {0.24, 0.24, -0.06}}
    };

    anchor_delta_translation_ = {
        {"0x0009D6", 0.0}, {"0x0009E5", 0.0},
        {"0x0016FA", 0.0}, {"0x0016CF", 0.0}
    };

    tag_delta_translation_ = {
        {"0x001155", 0.0}, {"0x001397", 0.0} 
    };

    anchor_positions_odom_ = last_anchor_positions_odom_ = anchor_positions_;
    tag_positions_odom_ = last_tag_positions_odom_ = tag_positions_;

    agv_odom_frame_id_ = "agv/odom"; //frame of the eliko system-> arco/eliko, for simulation use "agv_gt" for ground truth, "agv_odom" for odometry w/ errors
    uav_odom_frame_id_ = "uav/odom"; //frame of the UAV system-> uav_gt for ground truth, "uav_odom" for odometry w/ errors
    agv_body_frame_id_ = "agv/base_link";
    uav_body_frame_id_ = "uav/base_link";
    uav_opt_frame_id_ = "uav_opt"; 
    agv_opt_frame_id_ = "agv_opt"; 

    odom_error_distance_ = 2.0;
    odom_error_angle_ = 2.0;

    //Initial values for state
    init_state_.state = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
    init_state_.covariance = Eigen::Matrix4d::Identity(); //
    init_state_.roll = 0.0;
    init_state_.pitch = 0.0;
    init_state_.timestamp = this->get_clock()->now();

    opt_state_ = init_state_;

    moving_average_ = true;
    moving_average_max_samples_ = 10;

    last_agv_odom_initialized_ = false;
    last_uav_odom_initialized_ = false;

    min_traveled_distance_ = 0.5;
    min_traveled_angle_ = 30.0 * M_PI / 180.0;
    max_traveled_distance_ = min_traveled_distance_ * 10.0;
    uav_delta_translation_ = agv_delta_translation_ = uav_delta_rotation_ = agv_delta_rotation_ = 0.0;
    uav_total_translation_ = agv_total_translation_ = uav_total_rotation_ = agv_total_rotation_ = 0.0;

    global_optimization_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(int(global_opt_rate_s_*1000)), std::bind(&ElikoGlobalOptNode::globalOptCb, this));
    
    RCLCPP_INFO(this->get_logger(), "Eliko Optimization Node initialized.");
  }

private:

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
        
        uav_odom_pose_ = Sophus::SE3d(q, t);

        // If this is the first message, simply store it and return.
        if (!last_uav_odom_initialized_) {
            last_uav_odom_msg_ = *msg;
            last_uav_odom_initialized_ = true;
            last_uav_odom_pose_ = uav_odom_pose_;

            for (const auto& kv : tag_positions_) {
                last_tag_positions_odom_[kv.first] = uav_odom_pose_ * kv.second;
            }

            return;
        }


        // Compute the incremental movement of the UAV tag sensors.
        for (const auto& kv : tag_positions_) {
            const std::string& tag_id = kv.first;
            const Eigen::Vector3d& tag_offset = kv.second;  // mounting offset in UAV body frame
            Eigen::Vector3d current_tag_pos = uav_odom_pose_ * tag_offset;
            tag_positions_odom_[tag_id] = current_tag_pos;
            // If a last position exists, accumulate delta translation.
            double delta = (current_tag_pos - last_tag_positions_odom_[tag_id]).norm();
            //Accumulate delta for this tag
            tag_delta_translation_[tag_id] += delta;
            // RCLCPP_INFO(this->get_logger(), "Tag %s delta translation = %.2f", tag_id.c_str(), tag_delta_translation_[tag_id]);
            // Update the last position.
            last_tag_positions_odom_[tag_id] = current_tag_pos;
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
        last_uav_odom_msg_ = *msg;
    
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
        
        agv_odom_pose_ = Sophus::SE3d(q, t);


        // If this is the first message, simply store it and return.
        if (!last_agv_odom_initialized_) {
            last_agv_odom_msg_ = *msg;
            last_agv_odom_initialized_ = true;
            last_agv_odom_pose_ = agv_odom_pose_;

            // Initialize last anchor positions using known mounting offsets.
            for (const auto& kv : anchor_positions_) {
                last_anchor_positions_odom_[kv.first] = agv_odom_pose_ * kv.second;
            }

            return;
        }


        // Update anchor sensor positions.
        for (const auto& kv : anchor_positions_) {
            const std::string& anchor_id = kv.first;
            const Eigen::Vector3d& anchor_offset = kv.second;  // mounting offset in AGV body frame
            Eigen::Vector3d current_anchor_pos = agv_odom_pose_ * anchor_offset;
            anchor_positions_odom_[anchor_id] = current_anchor_pos;
            double delta = (current_anchor_pos - last_anchor_positions_odom_[anchor_id]).norm();
            // Accumulate delta for this anchor if needed.
            anchor_delta_translation_[anchor_id] += delta;
            // RCLCPP_INFO(this->get_logger(), "Anchor %s delta translation = %.2f", anchor_id.c_str(), anchor_delta_translation_[anchor_id]);
            last_anchor_positions_odom_[anchor_id] = current_anchor_pos;
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
        last_agv_odom_msg_ = *msg;
    
    }



    void distancesCoordsCb(const eliko_messages::msg::DistancesList::SharedPtr msg) {
    

        for (const auto& distance_msg : msg->anchor_distances) {
            Measurement measurement;
            measurement.timestamp = msg->header.stamp;
            measurement.tag_id = distance_msg.tag_sn;
            measurement.anchor_id = distance_msg.anchor_sn;
            measurement.distance = distance_msg.distance / 100.0; // Convert to meters

            //Positions of tags and anchors according to odometry
            measurement.tag_odom_pose = tag_positions_odom_[measurement.tag_id];
            measurement.anchor_odom_pose = anchor_positions_odom_[measurement.anchor_id];

            // Save the current cumulative displacement for each sensor.
            measurement.tag_cumulative_distance = tag_delta_translation_[measurement.tag_id];
            measurement.anchor_cumulative_distance = anchor_delta_translation_[measurement.anchor_id];

            global_measurements_.push_back(measurement);
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

        // Remove very old measurements
        while (!global_measurements_.empty() &&
              (current_time - global_measurements_.front().timestamp).seconds() > 30.0) {
            global_measurements_.pop_front();
        }

        //*****************Displace window based on distance traveled ****************************/
        while (!global_measurements_.empty()) {
            // Get the oldest measurement.
            const Measurement &oldest = global_measurements_.front();
            const Measurement &latest = global_measurements_.back();
             // Compute the net displacement over the window using the stored cumulative distances.
            double tag_disp = latest.tag_cumulative_distance - oldest.tag_cumulative_distance;
            double anchor_disp = latest.anchor_cumulative_distance - oldest.anchor_cumulative_distance;

            if(tag_disp < 2.0 || anchor_disp < 2.0){
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Insufficient displacement in window. Tag displacement = [%.2f], Anchor displacement = [%.2f]",
                tag_disp, anchor_disp);
                return;
            }
            // Prune the oldest measurement if the displacement exceeds the threshold.
            else if (tag_disp > max_traveled_distance_ || anchor_disp > max_traveled_distance_) {
                global_measurements_.pop_front();
            } else {
                break;
            }
        }

        // Then, check if there are enough measurements remaining.
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
        }

        RCLCPP_WARN(this->get_logger(), "[Eliko global_opt node] Movement UAV = [%.2fm %.2fº], AGV= [%.2fm %.2fº].", uav_delta_translation_, uav_delta_rotation_ * 180.0/M_PI, agv_delta_translation_, agv_delta_rotation_ * 180.0/M_PI);

        //*****************Calculate initial solution ****************************/

        Eigen::MatrixXd M;
        Eigen::VectorXd b;
        bool init_prior = solveMartelLinearSystem(global_measurements_, M, b, init_state_, true); //false: use full stack of measurements true: subset of 50 measurements

        //*****************Nonlinear WLS refinement with all measurements ****************************/

        // Compute odometry covariance based on distance increments from step to step 
        Eigen::Matrix4d predicted_motion_covariance = Eigen::Matrix4d::Zero();
        double predicted_drift_translation = (odom_error_distance_ / 100.0) * (uav_total_translation_ + agv_total_translation_);
        double predicted_drift_rotation = (odom_error_angle_ / 100.0) * (uav_total_rotation_ + agv_total_rotation_);
        predicted_motion_covariance(0,0) = std::pow(predicted_drift_translation, 2.0);
        predicted_motion_covariance(1,1) = std::pow(predicted_drift_translation, 2.0);
        predicted_motion_covariance(2,2) = std::pow(predicted_drift_translation, 2.0);
        predicted_motion_covariance(3,3) = std::pow(predicted_drift_rotation, 2.0);


        //Update odom for next optimization
        uav_delta_translation_ = agv_delta_translation_ = uav_delta_rotation_ = agv_delta_rotation_ = 0.0;

        //Optimize    
        RCLCPP_INFO(this->get_logger(), "[Eliko global_opt node] Optimizing trajectory of %ld measurements", global_measurements_.size());
        //Update transforms after convergence
        if(runOptimization(current_time, predicted_motion_covariance, init_prior)){
        
            if(moving_average_){
                // /*Run moving average*/
                auto smoothed_state = movingAverage(opt_state_);
                //Update for initial estimation of following step
                opt_state_.state = smoothed_state;
            }

            Sophus::SE3d That_ts = buildTransformationSE3(opt_state_.roll, opt_state_.pitch, opt_state_.state);
            
            geometry_msgs::msg::PoseWithCovarianceStamped msg = buildPoseMsg(That_ts, opt_state_.covariance + predicted_motion_covariance, opt_state_.timestamp, uav_odom_frame_id_);
            tf_publisher_->publish(msg);

        }

        else{
            RCLCPP_INFO(this->get_logger(), "[Eliko global_opt node] Optimizer did not converge");
        }

    }

    // This function builds the linear system M*x = b based on the current measurements window.
    bool solveMartelLinearSystem(const std::deque<Measurement>& measurements,
            Eigen::MatrixXd &M,
            Eigen::VectorXd &b, State &init_state, bool useSubset = false)
    {

        std::vector<Measurement> finalSet;
        if (useSubset) {
            // Define a desired subset size (30% of the total set)
            size_t subsetSize = 0.3 * measurements.size();
            finalSet = getRandomSubset(measurements, subsetSize);
        }
        else {
            // Use all measurements by copying the deque into a vector.
            finalSet = std::vector<Measurement>(measurements.begin(), measurements.end());
        }
        // Number of measurements in the window
        const int numMeasurements = finalSet.size();
        // Our unknown vector x has 8 elements:
        // [ u, v, w, cos(α), sin(α), L1, L2, L3 ]
        M.resize(numMeasurements, 8);
        b.resize(numMeasurements);

        for (int i = 0; i < numMeasurements; ++i) {
            const Measurement &meas = finalSet[i];

            // Anchor coordinates from AGV odometry frame
            double x = meas.anchor_odom_pose(0);
            double y = meas.anchor_odom_pose(1);
            double z = meas.anchor_odom_pose(2);

            // Tag coordinates from UAV odometry frame
            double A = meas.tag_odom_pose(0);
            double B = meas.tag_odom_pose(1);
            double C = meas.tag_odom_pose(2);

            // Measured distance (already converted to meters)
            double d = meas.distance;

            // Compute the right-hand side term b_i (following Equation (14) in the paper)
            double bi = d*d - (A*A + B*B + C*C) - (x*x + y*y + z*z) + 2.0 * C * z;

            // Compute the coefficients β_i1 through β_i7
            double beta1 = -2.0 * A;
            double beta2 = -2.0 * B;
            double beta3 = 2.0 * z - 2.0 * C;
            double beta4 = -2.0 * (A * x + B * y);
            double beta5 = 2.0 * (A * y - B * x);
            double beta6 = 2.0 * x;
            double beta7 = 2.0 * y;

            // Form the i-th row of the matrix M:
            // [ β_i1, β_i2, β_i3, β_i4, β_i5, β_i6, β_i7, 1 ]
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

        // Compute the SVD of M
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        auto singularValues = svd.singularValues();

        // Determine effective rank: a common approach is to compare each singular value with tol:
        int rank = 0;
        double tol = std::max(M.rows(), M.cols()) * singularValues(0) * std::numeric_limits<double>::epsilon();
        for (int i = 0; i < singularValues.size(); ++i) {
            if (singularValues(i) > tol)
                ++rank;
        }

        if (rank < 7) {
            RCLCPP_WARN(this->get_logger(), "Unfavourable configuration: rank(%d) < 7", rank);
            // Here you may choose to return a default state or handle the situation accordingly.
            init_state.state = Eigen::Vector4d::Zero();
            init_state.covariance = Eigen::Matrix4d::Identity();
            return false;
        }

        // Solve for x using the computed SVD (least-squares solution)
        Eigen::VectorXd solution = svd.solve(b);
        // The solution vector is:
        // x = [ u, v, w, cos(α), sin(α), L1, L2, L3 ]ᵀ

        init_state.state[0] = solution[0];
        init_state.state[1] = solution[1];
        init_state.state[2] = solution[2];
        init_state.state[3] = std::atan2(solution[4], solution[3]);

        RCLCPP_INFO_STREAM(this->get_logger(), "Initial solution: " << init_state.state.transpose());

        return true;
    }


    std::vector<Measurement> getRandomSubset(const std::deque<Measurement>& measurements, size_t N)
    {
        // Copy measurements into a vector
        std::vector<Measurement> measurementVec(measurements.begin(), measurements.end());
        
        // Create a random generator
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Shuffle the vector randomly
        std::shuffle(measurementVec.begin(), measurementVec.end(), gen);
        
        // If N is greater than the available measurements, use the full vector
        if (N > measurementVec.size()) {
            N = measurementVec.size();
        }
        
        // Create a subset vector with the first N elements
        std::vector<Measurement> subset(measurementVec.begin(), measurementVec.begin() + N);
        return subset;
    }


        // Normalize an angle to the range [-pi, pi]
    double normalizeAngle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
        }

    //Write to display a 4x4 transformation matrix
    void logTransformationMatrix(const Eigen::Matrix4d &T)   {

        RCLCPP_INFO(this->get_logger(), "T:\n"
            "[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]",
            T(0, 0), T(0, 1), T(0, 2), T(0, 3),
            T(1, 0), T(1, 1), T(1, 2), T(1, 3),
            T(2, 0), T(2, 1), T(2, 2), T(2, 3),
            T(3, 0), T(3, 1), T(3, 2), T(3, 3));                 

    }

    Eigen::Vector4d movingAverage(const State &sample) {
        
            // Add the new sample with timestamp
            moving_average_states_.push_back(sample);

            // Remove samples that are too old or if we exceed the window size
            while (!moving_average_states_.empty() &&
                moving_average_states_.size() > moving_average_max_samples_) {
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

    void publishTransform(const Sophus::SE3d& T, const std::string &frame_id, const std::string &child_frame_id, const rclcpp::Time &current_time) {
        
        
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


    geometry_msgs::msg::PoseWithCovarianceStamped buildPoseMsg(const Sophus::SE3d &T, const Eigen::Matrix4d& cov4, 
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

    Sophus::SE3d transformSE3FromPoseMsg(const geometry_msgs::msg::Pose& pose_msg) {
        Eigen::Vector3d t(pose_msg.position.x, pose_msg.position.y, pose_msg.position.z);
        Eigen::Quaterniond q(pose_msg.orientation.w, pose_msg.orientation.x,
                             pose_msg.orientation.y, pose_msg.orientation.z);
        q.normalize();
        return Sophus::SE3d(q, t);
      }

    Eigen::Matrix4d reduceCovarianceMatrix(const Eigen::Matrix<double,6,6> &cov6) {
        // Choose indices for [x, y, z, yaw]. Here, we assume that the yaw covariance
        // is stored at row/column 5 in the 6x6 matrix (with the 6-vector being [x,y,z,roll,pitch,yaw]).
        std::vector<int> indices = {0, 1, 2, 5};
        Eigen::Matrix4d cov4;
        for (size_t i = 0; i < indices.size(); ++i) {
            for (size_t j = 0; j < indices.size(); ++j) {
            cov4(i, j) = cov6(indices[i], indices[j]);
            }
        }
        return cov4;
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

    Sophus::SE3d buildTransformationSE3(double roll, double pitch, const Eigen::Vector4d& s) {
        Eigen::Vector3d t(s[0], s[1], s[2]);  // Use Vector3d instead of an incorrect Matrix type.
        Eigen::Matrix3d R = (Eigen::AngleAxisd(s[3], Eigen::Vector3d::UnitZ()) *
                             Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX())).toRotationMatrix();
        return Sophus::SE3d(R, t);
    }

    // Run the optimization once all measurements are received
    bool runOptimization(const rclcpp::Time &current_time, const Eigen::Matrix4d& motion_covariance, const bool& init_prior) {

            ceres::Problem problem;

            //Initial conditions
            State opt_state = opt_state_;
  
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

            //Set prior using linear initial solution
            if(init_prior){
                Eigen::Matrix4d prior_covariance = motion_covariance + Eigen::Matrix4d::Identity() * std::pow(2.0*measurement_stdev_,2.0);
                Sophus::SE3d prior_T = buildTransformationSE3(init_state_.roll, init_state_.pitch, init_state_.state);
                //Add the prior residual with the full covariance
                ceres::CostFunction* prior_cost = PriorResidual::Create(prior_T, opt_state.roll, opt_state.pitch, prior_covariance);
                problem.AddResidualBlock(prior_cost, robust_loss, opt_state.state.data());
            }


            for (const auto& measurement : global_measurements_) {                   
                
                // //Use positions of anchors and tags in each robot odometry frame
                Eigen::Vector3d anchor_pos = measurement.anchor_odom_pose;
                Eigen::Vector3d tag_pos = measurement.tag_odom_pose;

                ceres::CostFunction* cost_function = UWBResidual::Create(
                    anchor_pos, tag_pos, measurement.distance, opt_state.roll, opt_state.pitch, measurement_stdev_);
                problem.AddResidualBlock(cost_function, robust_loss, opt_state.state.data());
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
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr agv_odom_sub_, uav_odom_sub_;

    // Timers
    rclcpp::TimerBase::SharedPtr global_optimization_timer_;

    std::deque<Measurement> global_measurements_;
    
    // Publishers/Broadcasters
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr tf_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::unordered_map<std::string, Eigen::Vector3d> anchor_positions_;
    std::unordered_map<std::string, Eigen::Vector3d> tag_positions_;

    std::string agv_odom_frame_id_, uav_odom_frame_id_;
    std::string agv_body_frame_id_, uav_body_frame_id_;
    std::string uav_opt_frame_id_, agv_opt_frame_id_;

    State opt_state_;
    State init_state_;
    std::deque<State> moving_average_states_;
    size_t moving_average_max_samples_;

    bool moving_average_;
    size_t min_measurements_;
    double global_opt_rate_s_;
    double measurement_stdev_, measurement_covariance_;

    Sophus::SE3d uav_odom_pose_, last_uav_odom_pose_;         // Current UAV odometry position and last used for optimization
    Sophus::SE3d agv_odom_pose_, last_agv_odom_pose_;        // Current AGV odometry position and last used for optimization
    Eigen::Matrix<double, 6, 6> uav_odom_covariance_, agv_odom_covariance_;  // UAV odometry covariance
    nav_msgs::msg::Odometry last_agv_odom_msg_, last_uav_odom_msg_;
    bool last_agv_odom_initialized_, last_uav_odom_initialized_;

    double min_traveled_distance_, min_traveled_angle_, max_traveled_distance_;
    double uav_delta_translation_, agv_delta_translation_, uav_delta_rotation_, agv_delta_rotation_;
    double uav_total_translation_, agv_total_translation_, uav_total_rotation_, agv_total_rotation_;

    std::unordered_map<std::string, Eigen::Vector3d> anchor_positions_odom_, last_anchor_positions_odom_, tag_positions_odom_, last_tag_positions_odom_;
    std::unordered_map<std::string, double> anchor_delta_translation_;
    std::unordered_map<std::string, double> tag_delta_translation_;

    double odom_error_distance_, odom_error_angle_;

};
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ElikoGlobalOptNode>();
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
