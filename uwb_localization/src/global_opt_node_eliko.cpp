#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "geometry_msgs/msg/quaternion_stamped.hpp"

#include "eliko_messages/msg/distances_list.hpp"
#include "eliko_messages/msg/anchor_coords_list.hpp"
#include "eliko_messages/msg/tag_coords_list.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
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


class ElikoGlobalOptNode : public rclcpp::Node {

public:

    ElikoGlobalOptNode() : Node("eliko_global_opt_node") {
    

    //Subscribe to distances publisher
    eliko_distances_sub_ = this->create_subscription<eliko_messages::msg::DistancesList>(
                "/eliko/Distances", 10, std::bind(&ElikoGlobalOptNode::distances_coords_cb_, this, std::placeholders::_1));
    
    dji_attitude_sub_ = this->create_subscription<geometry_msgs::msg::QuaternionStamped>(
                "/dji_sdk/attitude", 10, std::bind(&ElikoGlobalOptNode::attitude_cb_, this, std::placeholders::_1));

    // Create publisher/broadcaster for optimized transformation
    tf_publisher_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("eliko_optimization_node/optimized_T", 10);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    global_opt_window_s_ = 10.0; //size of the sliding window in seconds
    global_opt_rate_s_ = 0.2; //rate of the optimization
    min_measurements_ = 100; //min number of measurements for running optimizer
    measurement_stdev_ = 0.1; //10 cm measurement noise
    measurement_covariance_ = measurement_stdev_ * measurement_stdev_;

    accumulated_covariance_ = Eigen::Matrix4d::Identity() * 1e-6;

    global_optimization_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(int(global_opt_rate_s_*1000)), std::bind(&ElikoGlobalOptNode::global_opt_cb_, this));


    anchor_positions_ = {
        {"0x0009D6", {-0.32, 0.3, 0.875}}, {"0x0009E5", {0.32, -0.3, 0.875}},
        {"0x0016FA", {0.32, 0.3, 0.33}}, {"0x0016CF", {-0.32, -0.3, 0.33}}
    };
    
    tag_positions_ = {
        {"0x001155", {-0.24, -0.24, -0.06}}, {"0x001397", {0.24, 0.24, -0.06}}
    };

    eliko_frame_id_ = "ground_vehicle"; //frame of the eliko system-> arco/eliko, for simulation use "ground_vehicle"
    uav_frame_id_ = "uav_opt"; //frame of the uav -> "base_link", for simulation use "uav_opt"

   
    //Initial values for state
    init_state_.state = Eigen::Vector4d(0.0, 0.0, -2.0, 0.0);
    init_state_.covariance = accumulated_covariance_; // Add epsilon to diagonal
    init_state_.roll = 0.0;
    init_state_.pitch = 0.0;
    init_state_.timestamp = this->get_clock()->now();

    opt_state_ = init_state_;

    moving_average_window_s_ = 1.0;
    moving_average_window_size_ = 10; //moving average sample window (N)
    
    RCLCPP_INFO(this->get_logger(), "Eliko Optimization Node initialized.");
  }

private:

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
            RCLCPP_WARN(this->get_logger(), "[Eliko global_opt node] Only one tag available. Optimization skipped.");
            return;
        }


        /*TODO: check enough translation in the window*/

        
        RCLCPP_INFO(this->get_logger(), "[Eliko global_opt node] Optimizing trajectory of %ld measurements", global_measurements_.size());
        //Update transforms after convergence
        if(run_optimization(current_time)){
        
            /*Run moving average*/
            auto smoothed_state = moving_average(opt_state_);
            //Update for initial estimation of following step
            opt_state_.state = smoothed_state;
            opt_state_.state[3] = normalize_angle(smoothed_state[3]);

            Eigen::Matrix4d That_ts = build_transformation_matrix(opt_state_.roll, opt_state_.pitch, opt_state_.state);

            publish_transform(That_ts, opt_state_.timestamp);

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
            double yaw;
            m.getRPY(opt_state_.roll, opt_state_.pitch, yaw);  // Get roll and pitch, but ignore yaw as we optimize it separately
        }

        // Normalize an angle to the range [-pi, pi]
    double normalize_angle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
        }


    Eigen::Vector4d moving_average(const State &sample) {
        
            // Add the new sample with timestamp
            moving_average_states_.push_back(sample);

            // Remove samples that are too old or if we exceed the window size
            while (!moving_average_states_.empty() &&
                (moving_average_states_.size() > moving_average_window_size_ || 
                    sample.timestamp - moving_average_states_.front().timestamp > rclcpp::Duration::from_seconds(global_opt_window_s_*moving_average_window_size_))) {
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
            }

            return smoothed_state;
    }

    void publish_transform(const Eigen::Matrix4d& T, const rclcpp::Time &current_time) {


        geometry_msgs::msg::TransformStamped that_ts_msg;
        that_ts_msg.header.stamp = current_time;

        // Extract translation
        that_ts_msg.transform.translation.x = T(0, 3);
        that_ts_msg.transform.translation.y = T(1, 3);
        that_ts_msg.transform.translation.z = T(2, 3);

        // Extract rotation (Convert Eigen rotation matrix to quaternion)
        Eigen::Matrix3d rotation_matrix_that_ts = T.block<3, 3>(0, 0);

        Eigen::Quaterniond q_that_ts(rotation_matrix_that_ts);

        that_ts_msg.transform.rotation.x = q_that_ts.x();
        that_ts_msg.transform.rotation.y = q_that_ts.y();
        that_ts_msg.transform.rotation.z = q_that_ts.z();
        that_ts_msg.transform.rotation.w = q_that_ts.w();
        
        /*Create the inverse to publish to tf tree*/
        
        Eigen::Matrix4d That_st = T.inverse();

        geometry_msgs::msg::TransformStamped that_st_msg;
        that_st_msg.header.stamp = that_ts_msg.header.stamp;
        that_st_msg.header.frame_id = eliko_frame_id_;  // Adjust frame_id as needed
        that_st_msg.child_frame_id = uav_frame_id_;            // Adjust child_frame_id as needed

        // Extract translation
        that_st_msg.transform.translation.x = That_st(0, 3);
        that_st_msg.transform.translation.y = That_st(1, 3);
        that_st_msg.transform.translation.z = That_st(2, 3);

        // Convert Eigen rotation matrix of inverse transform to tf2 Matrix3x3
        Eigen::Matrix3d rotation_matrix_that_st = That_st.block<3, 3>(0, 0);

        Eigen::Quaterniond q_that_st(rotation_matrix_that_st);

        that_st_msg.transform.rotation.x = q_that_st.x();
        that_st_msg.transform.rotation.y = q_that_st.y();
        that_st_msg.transform.rotation.z = q_that_st.z();
        that_st_msg.transform.rotation.w = q_that_st.w();


        // Publish the estimated transform
        tf_publisher_->publish(that_ts_msg);
        // Broadcast the inverse transform
        tf_broadcaster_->sendTransform(that_st_msg);
    }


  // Build transformation matrix from roll, pitch, optimized yaw, and translation vector
    Eigen::Matrix4d build_transformation_matrix(double roll, double pitch, const Eigen::Vector4d& s) {
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(s[3], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

        R = Eigen::Quaterniond(R).normalized().toRotationMatrix();
        
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = R;
        T(0, 3) = s[0];
        T(1, 3) = s[1];
        T(2, 3) = s[2];

        return T;
    }

    // Run the optimization once all measurements are received
    bool run_optimization(const rclcpp::Time &current_time) {

            ceres::Problem problem;

            //Initial conditions
            //State opt_state = opt_state_;
            State opt_state = init_state_;
            
            //Update the timestamp
            opt_state.timestamp = current_time;

            // Define a robust kernel
            double huber_threshold = 2.5; // = residuals higher than 2.5 times sigma are outliers
            ceres::LossFunction* robust_loss = new ceres::HuberLoss(huber_threshold);

            // Add the prior residual with the full covariance
            ceres::CostFunction* prior_cost = PriorResidual::Create(init_state_.state, accumulated_covariance_);
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
            options.linear_solver_type = ceres::DENSE_QR;

            // // Use Levenberg-Marquardt (default):
            // options.minimizer_type = ceres::TRUST_REGION;
            // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

            options.minimizer_progress_to_stdout = true;

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
                Eigen::Matrix4d full_covariance = Eigen::Matrix4d::Zero();
                if (covariance.Compute(covariance_blocks, &problem)) {
                    // Extract the full covariance matrix
                    covariance.GetCovarianceBlock(opt_state.state.data(), opt_state.state.data(), full_covariance.data());

                    // RCLCPP_INFO(this->get_logger(), "Full covariance matrix:\n"
                    //     "[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]",
                    //     full_covariance(0, 0), full_covariance(0, 1), full_covariance(0, 2), full_covariance(0, 3),
                    //     full_covariance(1, 0), full_covariance(1, 1), full_covariance(1, 2), full_covariance(1, 3),
                    //     full_covariance(2, 0), full_covariance(2, 1), full_covariance(2, 2), full_covariance(2, 3),
                    //     full_covariance(3, 0), full_covariance(3, 1), full_covariance(3, 2), full_covariance(3, 3));

                    //Accumulate covariances
                    accumulated_covariance_ += full_covariance;

                    RCLCPP_INFO(this->get_logger(), "Full covariance matrix:\n"
                        "[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]",
                        accumulated_covariance_(0, 0), accumulated_covariance_(0, 1), accumulated_covariance_(0, 2), accumulated_covariance_(0, 3),
                        accumulated_covariance_(1, 0), accumulated_covariance_(1, 1), accumulated_covariance_(1, 2), accumulated_covariance_(1, 3),
                        accumulated_covariance_(2, 0), accumulated_covariance_(2, 1), accumulated_covariance_(2, 2), accumulated_covariance_(2, 3),
                        accumulated_covariance_(3, 0), accumulated_covariance_(3, 1), accumulated_covariance_(3, 2), accumulated_covariance_(3, 3));

                    // Optionally, store covariance for further use
                    opt_state.covariance = accumulated_covariance_;  // Add this to your State struct if needed

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

                // Extract state variables
                T x = state[0];
                T y = state[1];
                T z = state[2];
                T yaw = state[3];

                // Build rotation matrix using roll, pitch, and variable yaw
                Eigen::Matrix<T, 3, 3> R;
                R = Eigen::AngleAxis<T>(yaw, Eigen::Matrix<T, 3, 1>::UnitZ()) *
                    Eigen::AngleAxis<T>(T(pitch_), Eigen::Matrix<T, 3, 1>::UnitY()) *
                    Eigen::AngleAxis<T>(T(roll_), Eigen::Matrix<T, 3, 1>::UnitX());

                // Transform the anchor point
                Eigen::Matrix<T, 3, 1> anchor_transformed = R * anchor_.cast<T>() + Eigen::Matrix<T, 3, 1>(x, y, z);

                // Calculate the predicted distance
                T dx = T(tag_(0)) - anchor_transformed[0];
                T dy = T(tag_(1)) - anchor_transformed[1];
                T dz = T(tag_(2)) - anchor_transformed[2];
                T predicted_distance = ceres::sqrt(dx * dx + dy * dy + dz * dz);

                // Residual as (measured - predicted)
                residual[0] = T(measured_distance_) - predicted_distance;
                residual[0] *= T(measurement_stdev_inv_);  // Scale residual

                // std::cout << "Measurement residual: " << residual[0] << std::endl;

                return true;
        }

      static ceres::CostFunction* Create(const Eigen::Vector3d& anchor, const Eigen::Vector3d& tag, double measured_distance, double roll, double pitch, double measurement_stdev) {
            return (new ceres::AutoDiffCostFunction<UWBResidual, 1, 4>(
                new UWBResidual(anchor, tag, measured_distance, roll, pitch, measurement_stdev)));
        }

        const Eigen::Vector3d anchor_;
        const Eigen::Vector3d tag_;
        const double measured_distance_;
        const double roll_;
        const double pitch_;
        const double measurement_stdev_inv_;
  };

  struct PriorResidual {
    PriorResidual(const Eigen::Vector4d& prior_state, const Eigen::Matrix4d& full_covariance)
        : prior_state_(prior_state), full_covariance_(full_covariance) {}

    template <typename T>
    bool operator()(const T* const state, T* residual) const {
        // Combine translation and yaw into a single residual vector
        Eigen::Matrix<T, 4, 1> delta_state;
        delta_state << state[0] - T(prior_state_(0)),
                       state[1] - T(prior_state_(1)),
                       state[2] - T(prior_state_(2)),
                       normalize_angle(state[3] - T(prior_state_(3)));

        // Scale by the square root of the inverse covariance matrix
        Eigen::LLT<Eigen::Matrix4d> chol(full_covariance_);
        Eigen::Matrix4d sqrt_inv_covariance = Eigen::Matrix4d(chol.matrixL().transpose()).inverse();

        Eigen::Matrix<T, 4, 1> weighted_residual = sqrt_inv_covariance.cast<T>() * delta_state;

        // Assign to residual
        residual[0] = weighted_residual[0];
        residual[1] = weighted_residual[1];
        residual[2] = weighted_residual[2];
        residual[3] = weighted_residual[3];

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector4d& prior_state, const Eigen::Matrix4d& full_covariance) {
        return new ceres::AutoDiffCostFunction<PriorResidual, 4, 4>(
            new PriorResidual(prior_state, full_covariance));
    }

private:
    template <typename T>
    T normalize_angle(const T& angle) const {
        T normalized_angle = angle;
        while (normalized_angle > T(M_PI)) normalized_angle -= T(2.0 * M_PI);
        while (normalized_angle < T(-M_PI)) normalized_angle += T(2.0 * M_PI);
        return normalized_angle;
    }

    const Eigen::Vector4d prior_state_;
    const Eigen::Matrix4d full_covariance_;
};

    
    rclcpp::Subscription<eliko_messages::msg::DistancesList>::SharedPtr eliko_distances_sub_;
    rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr dji_attitude_sub_;
    rclcpp::TimerBase::SharedPtr global_optimization_timer_, global_optimization_oneoff_timer_;

    std::deque<Measurement> global_measurements_;
    
    rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr tf_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::unordered_map<std::string, Eigen::Vector3d> anchor_positions_;
    std::unordered_map<std::string, Eigen::Vector3d> tag_positions_;

    Eigen::Matrix4d accumulated_covariance_;

    std::string eliko_frame_id_, uav_frame_id_;

    State opt_state_;
    State init_state_;
    std::deque<State> moving_average_states_;

    size_t moving_average_window_size_;
    size_t min_measurements_;
    double global_opt_window_s_, global_opt_rate_s_;
    double measurement_stdev_, measurement_covariance_;
    double moving_average_window_s_;

};
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ElikoGlobalOptNode>();
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
