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

using MeasurementKey = std::pair<std::string, std::string>; // (anchor_sn, tag_sn)


// Define a hash function for pair of strings
struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ (hash2 << 1); // Combine the two hashes
    }
};


struct State {
    Eigen::Vector3d translation;
    double yaw;
    double roll;
    double pitch;
    rclcpp::Time timestamp;  // e.g., seconds since epoch
};


class ElikoBatchOptNode : public rclcpp::Node {

public:

    ElikoBatchOptNode() : Node("eliko_batchopt_node"), roll_(0.0), pitch_(0.0), yaw_(0.0) {
    

    //Subscribe to distances publisher
    eliko_distances_sub_ = this->create_subscription<eliko_messages::msg::DistancesList>(
                "/eliko/Distances", 10, std::bind(&ElikoBatchOptNode::distances_coords_cb_, this, std::placeholders::_1));
    
    // eliko_anchors_coords_sub_ = this->create_subscription<eliko_messages::msg::AnchorCoordsList>(
    //             "/eliko/AnchorCoords", 10, std::bind(&ElikoBatchOptNode::anchors_coords_cb_, this, std::placeholders::_1));
    
    dji_attitude_sub_ = this->create_subscription<geometry_msgs::msg::QuaternionStamped>(
                "/dji_sdk/attitude", 10, std::bind(&ElikoBatchOptNode::attitude_cb_, this, std::placeholders::_1));

    // Create publisher/broadcaster for optimized transformation
    tf_publisher_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("eliko_optimization_node/optimized_T", 10);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    sliding_window_s_ = 0.1; //use only messages in this window for optimization
    batch_opt_window_s_ = 1.0;
    min_measurements_ = 1; //min number of measurements for running optimizer

    sliding_window_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(int(sliding_window_s_*1000)), std::bind(&ElikoBatchOptNode::sliding_window_cb_, this));

    batch_optimization_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(int(batch_opt_window_s_*1000)), std::bind(&ElikoBatchOptNode::batchopt_cb_, this));

    anchor_positions_ = {
        {"0x0009D6", {-0.32, 0.3, 0.875}}, {"0x0009E5", {0.32, -0.3, 0.875}},
        {"0x0016FA", {0.32, 0.3, 0.33}}, {"0x0016CF", {-0.32, -0.3, 0.33}}
    };
    
    tag_positions_ = {
        {"0x001155", {-0.24, -0.24, -0.06}}, {"0x001397", {0.24, 0.24, -0.06}}
    };

    eliko_frame_id_ = "ground_vehicle"; //frame of the eliko system-> arco/eliko, for simulation use "ground_vehicle"
    uav_frame_id_ = "uav_opt"; //frame of the uav -> "base_link", for simulation use "uav_opt"

    // Known roll and pitch values
    roll_ = 0.0;   // Example roll in radians
    pitch_ = 0.0;  // Example pitch in radians
    yaw_ = 0.0;    // Initial guess for yaw

    t_ = Eigen::Vector3d(0.0, 0.0, 0.0);       // Translation That_ts (x, y, z)

    moving_average_window_size_ = 10; //moving average sample window (N)
    
    RCLCPP_INFO(this->get_logger(), "Eliko Optimization Node initialized.");
  }

private:

    // Callback for each measurement
    void distances_coords_cb_(const eliko_messages::msg::DistancesList::SharedPtr msg) {
            
            if(msg->anchor_distances[0].tag_sn == "0x001155") distances_t1_ = *msg;
            else if(msg->anchor_distances[0].tag_sn == "0x001397") distances_t2_ = *msg;

            for (const auto& distance_msg : msg->anchor_distances) {
                RCLCPP_INFO(this->get_logger(), "[Eliko batchopt node] Distance received from Anchor %s to tag %s: %.2f cm", distance_msg.anchor_sn.c_str(), distance_msg.tag_sn.c_str(), distance_msg.distance);
            }

    }

    void sliding_window_cb_() {

        rclcpp::Time current_time = this->get_clock()->now();

        std::unordered_map<MeasurementKey, float, PairHash> measurement_map;

        if(!distances_t1_.anchor_distances.empty()){
            for (const auto& distance_msg : distances_t1_.anchor_distances) {
                    // Store distance measurements using (anchor_sn, tag_sn) as key
                    MeasurementKey key(distance_msg.anchor_sn, distance_msg.tag_sn);
                    measurement_map[key] = distance_msg.distance / 100.0; // Convert to meters
                }
            RCLCPP_DEBUG(this->get_logger(), "[Eliko batchopt node] Using %ld measurements from t1", distances_t1_.anchor_distances.size());
        }

        if(!distances_t2_.anchor_distances.empty()){
            for (const auto& distance_msg : distances_t2_.anchor_distances) {
                    // Store distance measurements using (anchor_sn, tag_sn) as key
                    MeasurementKey key(distance_msg.anchor_sn, distance_msg.tag_sn);
                    measurement_map[key] = distance_msg.distance / 100.0; // Convert to meters
                }
            RCLCPP_DEBUG(this->get_logger(), "[Eliko batchopt node] Using %ld measurements from t2", distances_t2_.anchor_distances.size());
        }


        //Ensure at least one measurement from each of the tags
        if(measurement_map.size() >= min_measurements_){
            RCLCPP_INFO(this->get_logger(), "[Eliko batchopt node] Creating new step with %ld measurements", measurement_map.size());
            // Add the new measurement map to the vector
            batch_measurements_.push_back(measurement_map);

            //Add state to optimization with initial guesses
            State new_state;
            new_state.translation = t_;
            new_state.yaw = yaw_;
            new_state.roll = roll_;
            new_state.pitch = pitch_;
            new_state.timestamp = current_time;

            batch_states_.push_back(new_state);

        }

        distances_t1_.anchor_distances.clear();
        distances_t2_.anchor_distances.clear();

    }


    void batchopt_cb_() {

        rclcpp::Time current_time = this->get_clock()->now();

        if(!batch_measurements_.empty()){

            RCLCPP_INFO(this->get_logger(), "[Eliko batchopt node] Optimizing batch of %ld steps", batch_measurements_.size());

            //Update transforms after convergence
            if(run_optimization()){


                for (auto& state : batch_states_) {        
                    // Construct transformation matrix T for all batch with optimized yaw, roll, pitch, and translation
                    Eigen::Matrix4d That_ts = build_transformation_matrix(state.roll, state.pitch, state.yaw, state.translation);
                    batchopt_T_.push_back(That_ts);
                }

                publish_transform(batchopt_T_.back(), current_time);
                
                // /*Run moving average*/
                // State sample {t_, yaw_, current_time};
                // auto [smoothed_translation, smoothed_yaw] = moving_average(sample);
                // //Update for initial estimation of following step
                // t_ = smoothed_translation;
                // yaw_ = normalize_angle(smoothed_yaw);

            }

            else{
                RCLCPP_INFO(this->get_logger(), "[Eliko batchopt node] Optimizer did not converge");
            }


        }

        else{
            RCLCPP_WARN(this->get_logger(), "[Eliko batchopt node] Not enough data to run optimization");
        }

        batch_measurements_.clear();
        batch_states_.clear();
        batchopt_T_.clear();

    }


//    void anchors_coords_cb_(const eliko_messages::msg::AnchorCoordsList::SharedPtr msg) {
//      // Update anchor coordinates based on received data
//         for (const auto& anchor_coord : msg->anchor_coords) {
//             Eigen::Vector3d position(anchor_coord.x_coord, anchor_coord.y_coord, anchor_coord.z_coord);
//             anchor_positions_[anchor_coord.anchor_sn] = position;

//             RCLCPP_DEBUG(this->get_logger(), "[Eliko batchopt node] Anchor %s coordinates: [%.2f, %.2f, %.2f]", anchor_coord.anchor_sn.c_str(), position[0], position[1], position[2]);
//         }
//   }

    void attitude_cb_(const geometry_msgs::msg::QuaternionStamped::SharedPtr msg) {

            const auto& q = msg->quaternion;
            // Convert the quaternion to roll, pitch, yaw
            tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
            tf2::Matrix3x3 m(tf_q);
            double yaw;
            m.getRPY(roll_, pitch_, yaw);  // Get roll and pitch, but ignore yaw as we optimize it separately
        }

        // Normalize an angle to the range [-pi, pi]
    double normalize_angle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
        }


    std::pair<Eigen::Vector3d, double> moving_average(const State &sample) {
        
            // Add the new sample with timestamp
            batch_states_.push_back(sample);

            // Remove samples that are too old or if we exceed the window size
            while (!batch_states_.empty() &&
                (batch_states_.size() > moving_average_window_size_ || 
                    sample.timestamp - batch_states_.front().timestamp > rclcpp::Duration::from_seconds(sliding_window_s_*moving_average_window_size_))) {
                batch_states_.pop_front();
            }

            // Initialize smoothed values
            Eigen::Vector3d smoothed_translation = Eigen::Vector3d(0.0, 0.0, 0.0);
            double smoothed_yaw = 0.0;

            // Initialize accumulators for circular mean of yaw
            double sum_sin = 0.0;
            double sum_cos = 0.0;


            // Initialize total weight
            double total_weight = 0.0;
            // Define a decay factor (e.g., between 0 and 1) for exponential weighting
            double decay_factor = 0.9;  // Adjust as needed for desired weighting effect
            double weight = 1.0;  // Start with a weight of 1.0 for the most recent sample

            // Iterate over samples in reverse order (from newest to oldest)
            for (auto it = batch_states_.rbegin(); it != batch_states_.rend(); ++it) {
                // Apply the weight to each sample's translation and accumulate
                smoothed_translation[0] += weight * it->translation[0];
                smoothed_translation[1] += weight * it->translation[1];
                smoothed_translation[2] += weight * it->translation[2];
                
                // Convert yaw to sine and cosine, apply the weight, then accumulate
                sum_sin += weight * std::sin(it->yaw);
                sum_cos += weight * std::cos(it->yaw);

                // Accumulate total weight for averaging
                total_weight += weight;

                // Decay the weight for the next (older) sample
                weight *= decay_factor;
            }

            // Normalize to get the weighted average
            if (total_weight > 0.0) {
                smoothed_translation[0] /= total_weight;
                smoothed_translation[1] /= total_weight;
                smoothed_translation[2] /= total_weight;

                // Calculate weighted average yaw using the weighted sine and cosine
                smoothed_yaw = std::atan2(sum_sin / total_weight, sum_cos / total_weight);
            }

            return {smoothed_translation, smoothed_yaw};
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
    Eigen::Matrix4d build_transformation_matrix(double roll, double pitch, double yaw, const Eigen::Vector3d& t) {
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

        R = Eigen::Quaterniond(R).normalized().toRotationMatrix();
        
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = R;
        T(0, 3) = t[0];
        T(1, 3) = t[1];
        T(2, 3) = t[2];

        return T;
    }

    // Run the optimization once all measurements are received
    bool run_optimization() {

            ceres::Problem problem;

            // Temporary variables to hold optimized results

            // Eigen::Vector3d t = t_;
            // double yaw = yaw_;
            std::deque<State> batch_states = batch_states_;


            // Ensure state window matches current measurement window size
            // if (state_window_.size() != batch_measurements_.size()) {
            //     state_window_.resize(batch_measurements_.size());
                // for (auto& state : state_window_) {
                //     state.translation = t_;  // Initial guess for position
                //     state.yaw = yaw;  // Initial guess for yaw
                //     }
            //}


            for(size_t i=0; i < batch_measurements_.size(); ++i){

                const auto& measurement_map = batch_measurements_[i];
                auto& state = batch_states[i];

                // Dynamically add residuals for available measurements
                for (const auto& [key, distance] : measurement_map) {
                    const auto& [anchor_sn, tag_sn] = key;
                    if (anchor_positions_.count(anchor_sn) > 0 && tag_positions_.count(tag_sn) > 0) {
                        Eigen::Vector3d anchor_pos = anchor_positions_[anchor_sn];
                        Eigen::Vector3d tag_pos = tag_positions_[tag_sn];
                        
                        // Create cost function with the received distance
                        ceres::CostFunction* cost_function = UWBResidual::Create(anchor_pos, tag_pos, distance, roll_, pitch_);
                        problem.AddResidualBlock(cost_function, nullptr, &state.yaw, state.translation.data());
                    }
                }
            }

            // Smoothness Constraints between consecutive states
            for (size_t i = 1; i < batch_states.size(); ++i) {
                State& prev_state = batch_states[i - 1];
                State& curr_state = batch_states[i];

                ceres::CostFunction* smoothness_cost = SmoothnessResidual::Create();
                problem.AddResidualBlock(smoothness_cost, nullptr, prev_state.translation.data(), &prev_state.yaw,
                                        curr_state.translation.data(), &curr_state.yaw);
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
            RCLCPP_INFO(this->get_logger(), summary.BriefReport().c_str());

            // Notify and update values if optimization converged
            if (summary.termination_type == ceres::CONVERGENCE){
                
                batch_states_ = batch_states;
                State last_optimized_state = batch_states_.back();
                t_ = last_optimized_state.translation;
                yaw_ = last_optimized_state.yaw;
                
                return true;
            } 
            
            return false;

    }


  struct SmoothnessResidual {
        SmoothnessResidual(double weight = 1.0) : weight_(weight) {}

        template <typename T>
        bool operator()(const T* const prev_position, const T* const prev_yaw,
                        const T* const curr_position, const T* const curr_yaw, T* residual) const {
            // Smoothness penalty on position
            residual[0] = T(weight_) * (curr_position[0] - prev_position[0]);
            residual[1] = T(weight_) * (curr_position[1] - prev_position[1]);
            residual[2] = T(weight_) * (curr_position[2] - prev_position[2]);

            // Smoothness penalty on yaw (angle wrap-around handled separately if needed)
            residual[3] = T(weight_) * (curr_yaw[0] - prev_yaw[0]);
            
            return true;
        }

        static ceres::CostFunction* Create(double weight = 1.0) {
            return (new ceres::AutoDiffCostFunction<SmoothnessResidual, 4, 3, 1, 3, 1>(
                new SmoothnessResidual(weight)));
        }

        double weight_;
    };

  // Residual struct for Ceres
  struct UWBResidual {
        UWBResidual(const Eigen::Vector3d& anchor, const Eigen::Vector3d& tag, double measured_distance, double roll, double pitch)
            : anchor_(anchor), tag_(tag), measured_distance_(measured_distance), roll_(roll), pitch_(pitch) {}

        template <typename T>
        bool operator()(const T* const yaw, const T* const t, T* residual) const {
                // Build rotation matrix using roll, pitch, and variable yaw
                Eigen::Matrix<T, 3, 3> R;
                R = Eigen::AngleAxis<T>(yaw[0], Eigen::Matrix<T, 3, 1>::UnitZ()) *
                    Eigen::AngleAxis<T>(T(pitch_), Eigen::Matrix<T, 3, 1>::UnitY()) *
                    Eigen::AngleAxis<T>(T(roll_), Eigen::Matrix<T, 3, 1>::UnitX());

                // Transform the anchor point
                Eigen::Matrix<T, 3, 1> anchor_transformed = R * anchor_.cast<T>() + Eigen::Matrix<T, 3, 1>(t[0], t[1], t[2]);

                // Calculate the predicted distance
                T dx = T(tag_(0)) - anchor_transformed[0];
                T dy = T(tag_(1)) - anchor_transformed[1];
                T dz = T(tag_(2)) - anchor_transformed[2];
                T predicted_distance = ceres::sqrt(dx * dx + dy * dy + dz * dz);

                // Residual as (measured - predicted)
                residual[0] = T(measured_distance_) - predicted_distance;
                return true;
        }

      static ceres::CostFunction* Create(const Eigen::Vector3d& anchor, const Eigen::Vector3d& tag, double measured_distance, double roll, double pitch) {
            return (new ceres::AutoDiffCostFunction<UWBResidual, 1, 1, 3>(
                new UWBResidual(anchor, tag, measured_distance, roll, pitch)));
        }

        const Eigen::Vector3d anchor_;
        const Eigen::Vector3d tag_;
        const double measured_distance_;
        const double roll_;
        const double pitch_;
  };

    
    rclcpp::Subscription<eliko_messages::msg::DistancesList>::SharedPtr eliko_distances_sub_;
    // rclcpp::Subscription<eliko_messages::msg::AnchorCoordsList>::SharedPtr eliko_anchors_coords_sub_;
    rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr dji_attitude_sub_;
    rclcpp::TimerBase::SharedPtr sliding_window_timer_, batch_optimization_timer_;

    eliko_messages::msg::DistancesList distances_t1_, distances_t2_;
    std::vector<std::unordered_map<MeasurementKey, float, PairHash>> batch_measurements_;
    
    std::vector<Eigen::Matrix4d> batchopt_T_;

    rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr tf_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::unordered_map<std::string, Eigen::Vector3d> anchor_positions_;
    std::unordered_map<std::string, Eigen::Vector3d> tag_positions_;

    std::string eliko_frame_id_, uav_frame_id_;

    // Known roll and pitch, and yaw to be optimized
    double roll_, pitch_;
    double yaw_;
    Eigen::Vector3d t_;  // Translation vector (x, y, z)

    std::deque<State> batch_states_;

    size_t moving_average_window_size_;
    size_t min_measurements_;
    double sliding_window_s_;
    double batch_opt_window_s_;

};
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ElikoBatchOptNode>();
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
