#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <tf2/LinearMath/Quaternion.h>
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

class ElikoOptimizationNode : public rclcpp::Node {

public:

    ElikoOptimizationNode() : Node("eliko_optimization_node"), roll_(0.0), pitch_(0.0), yaw_(0.0) {
    

    //Subscribe to distances publisher
    eliko_distances_sub_ = this->create_subscription<eliko_messages::msg::DistancesList>(
                "/eliko/Distances", 10, std::bind(&ElikoOptimizationNode::distances_coords_cb_, this, std::placeholders::_1));
    
    // eliko_tags_coords_sub_ = this->create_subscription<eliko_messages::msg::TagCoordsList>(
    //             "/eliko/TagCoords", 10, std::bind(&ElikoOptimizationNode::tags_coords_cb_, this, std::placeholders::_1));

    // eliko_anchors_coords_sub_ = this->create_subscription<eliko_messages::msg::AnchorCoordsList>(
    //             "/eliko/AnchorCoords", 10, std::bind(&ElikoOptimizationNode::anchors_coords_cb_, this, std::placeholders::_1));
    
    dji_attitude_sub_ = this->create_subscription<geometry_msgs::msg::QuaternionStamped>(
                "/dji_sdk/attitude", 10, std::bind(&ElikoOptimizationNode::attitude_cb_, this, std::placeholders::_1));

    // Create publisher/broadcaster for optimized transformation
    tf_publisher_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("eliko_optimization_node/optimized_T", 10);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    optimizer_time_window_s_ = 0.2; //use only messages in this window for optimization
    min_measurements_ = 3; //min number of measurements for running optimizer

    optimization_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(int(optimizer_time_window_s_*1000)), std::bind(&ElikoOptimizationNode::optimizer_cb_, this));

    anchor_positions_ = {
        {"0x0009D6", {-0.32, 0.3, 0.875}}, {"0x0009E5", {0.32, -0.3, 0.875}},
        {"0x0016FA", {0.32, 0.3, 0.33}}, {"0x0016CF", {-0.32, -0.3, 0.33}}
    };
    
    tag_positions_ = {
        {"0x001155", {-0.24, -0.24, -0.06}}, {"0x001397", {0.24, 0.24, -0.06}}
    };

    eliko_frame_id_ = "arco/eliko"; //frame of the eliko system-> same as robot frame
    uav_frame_id_ = "base_link";

    // Known roll and pitch values
    roll_ = 0.0;   // Example roll in radians
    pitch_ = 0.0;  // Example pitch in radians
    yaw_ = 0.0;    // Initial guess for yaw

    t_ = {0.0, 0.0, 0.0};       // Translation That_ts (x, y, z)

    moving_average_window_size_ = 5; //moving average sample window (N)

    RCLCPP_INFO(this->get_logger(), "Eliko Optimization Node initialized.");
  }

private:

    // Callback for each measurement
    void distances_coords_cb_(const eliko_messages::msg::DistancesList::SharedPtr msg) {
            
            if(msg->anchor_distances[0].tag_sn == "0x001155") distances_t1_ = *msg;
            else if(msg->anchor_distances[0].tag_sn == "0x001397") distances_t2_ = *msg;

            for (const auto& distance_msg : msg->anchor_distances) {
                RCLCPP_INFO(this->get_logger(), "[Eliko optimizer node] Distance received from Anchor %s to tag %s: %.2f cm", distance_msg.anchor_sn.c_str(), distance_msg.tag_sn.c_str(), distance_msg.distance);
            }

    }

    void optimizer_cb_() {

        rclcpp::Time current_time = this->get_clock()->now();

        if(!distances_t1_.anchor_distances.empty() && current_time - distances_t1_.header.stamp < rclcpp::Duration::from_seconds(optimizer_time_window_s_)){
            for (const auto& distance_msg : distances_t1_.anchor_distances) {
                    // Store distance measurements using (anchor_sn, tag_sn) as key
                    MeasurementKey key(distance_msg.anchor_sn, distance_msg.tag_sn);
                    current_measurements_[key] = distance_msg.distance / 100.0; // Convert to meters
                }
            RCLCPP_DEBUG(this->get_logger(), "[Eliko optimizer node] Using %ld measurements from t1", distances_t1_.anchor_distances.size());
        }

        if(!distances_t2_.anchor_distances.empty() && current_time - distances_t2_.header.stamp < rclcpp::Duration::from_seconds(optimizer_time_window_s_)){
            for (const auto& distance_msg : distances_t2_.anchor_distances) {
                    // Store distance measurements using (anchor_sn, tag_sn) as key
                    MeasurementKey key(distance_msg.anchor_sn, distance_msg.tag_sn);
                    current_measurements_[key] = distance_msg.distance / 100.0; // Convert to meters
                }
            RCLCPP_DEBUG(this->get_logger(), "[Eliko optimizer node] Using %ld measurements from t2", distances_t2_.anchor_distances.size());
        }
 
        //Ensure at least one measurement from each of the tags
        if(current_measurements_.size() >= min_measurements_){
            RCLCPP_INFO(this->get_logger(), "[Eliko optimizer node] Running optimizer with %ld measurements", current_measurements_.size());
            run_optimization();
        }
        else{
            RCLCPP_WARN(this->get_logger(), "[Eliko optimizer node] Not enough data to run optimization");
        }

        current_measurements_.clear();
        distances_t1_.anchor_distances.clear();
        distances_t2_.anchor_distances.clear();

    }



//   void tags_coords_cb_(const eliko_messages::msg::TagCoordsList::SharedPtr msg) {
//         // Update tag coordinates based on received data
//         for (const auto& tag_coord : msg->tag_coords) {
//             Eigen::Vector3d position(tag_coord.x_coord, tag_coord.y_coord, tag_coord.z_coord);
//             tag_positions_[tag_coord.tag_sn] = position;

//             RCLCPP_DEBUG(this->get_logger(), "[Eliko optimizer node] Tag %s coordinates: [%.2f, %.2f, %.2f]", tag_coord.tag_sn.c_str(), position[0], position[1], position[2]);
//         }
//   }

//    void anchors_coords_cb_(const eliko_messages::msg::AnchorCoordsList::SharedPtr msg) {
//      // Update anchor coordinates based on received data
//         for (const auto& anchor_coord : msg->anchor_coords) {
//             Eigen::Vector3d position(anchor_coord.x_coord, anchor_coord.y_coord, anchor_coord.z_coord);
//             anchor_positions_[anchor_coord.anchor_sn] = position;

//             RCLCPP_DEBUG(this->get_logger(), "[Eliko optimizer node] Anchor %s coordinates: [%.2f, %.2f, %.2f]", anchor_coord.anchor_sn.c_str(), position[0], position[1], position[2]);
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


  std::pair<std::array<double, 3>, double> moving_average(const double yaw, const std::array<double, 3>& t) {
    
        // Add new values to history
        translation_history_.push_back(t);
        yaw_history_.push_back(yaw);

        // Remove oldest if history exceeds desired size
        if (translation_history_.size() > moving_average_window_size_) {
            translation_history_.pop_front();
            yaw_history_.pop_front();
        }
        
        // Compute average translation
        std::array<double, 3> smoothed_translation = {0.0, 0.0, 0.0};
        for (const auto& t : translation_history_) {
            smoothed_translation[0] += t[0];
            smoothed_translation[1] += t[1];
            smoothed_translation[2] += t[2];
        }
        smoothed_translation[0] /= translation_history_.size();
        smoothed_translation[1] /= translation_history_.size();
        smoothed_translation[2] /= translation_history_.size();

        double smoothed_yaw = 0.0;
        for (const auto& theta : yaw_history_) {
            smoothed_yaw += theta;
        }
        smoothed_yaw /= yaw_history_.size();

        return {smoothed_translation, smoothed_yaw};
}

    void publish_transform(const Eigen::Matrix4d& T) {


        geometry_msgs::msg::TransformStamped that_ts_msg;
        that_ts_msg.header.stamp = this->now();

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

        // Extract rotation (Convert Eigen rotation matrix to quaternion)
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
    Eigen::Matrix4d build_transformation_matrix(double roll, double pitch, double yaw, const std::array<double, 3>& t) {
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = R;
        T(0, 3) = t[0];
        T(1, 3) = t[1];
        T(2, 3) = t[2];

        return T;
    }

    // Run the optimization once all measurements are received
    void run_optimization() {

            ceres::Problem problem;

            // Dynamically add residuals for available measurements
            for (const auto& [key, distance] : current_measurements_) {
                const auto& [anchor_sn, tag_sn] = key;
                if (anchor_positions_.count(anchor_sn) > 0 && tag_positions_.count(tag_sn) > 0) {
                    Eigen::Vector3d anchor_pos = anchor_positions_[anchor_sn];
                    Eigen::Vector3d tag_pos = tag_positions_[tag_sn];
                    
                    // Create cost function with the received distance
                    ceres::CostFunction* cost_function = UWBResidual::Create(anchor_pos, tag_pos, distance, roll_, pitch_);
                    problem.AddResidualBlock(cost_function, nullptr, &yaw_, t_.data());
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
            RCLCPP_INFO(this->get_logger(), summary.BriefReport().c_str());

            // Update if optimization converged
            if (summary.termination_type == ceres::CONVERGENCE){

                /*Run moving average*/
                auto [t_avg, yaw_avg] = moving_average(yaw_, t_);
                t_ = t_avg;
                yaw_ = yaw_avg;

                // Construct transformation matrix T with optimized yaw, roll, pitch, and translation
                Eigen::Matrix4d That_ts = build_transformation_matrix(roll_, pitch_, yaw_, t_);

                publish_transform(That_ts);
            }

    }

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
    // rclcpp::Subscription<eliko_messages::msg::TagCoordsList>::SharedPtr eliko_tags_coords_sub_;
    rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr dji_attitude_sub_;
    rclcpp::TimerBase::SharedPtr optimization_timer_;

    eliko_messages::msg::DistancesList distances_t1_, distances_t2_;
    std::unordered_map<MeasurementKey, float, PairHash> current_measurements_;

    rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr tf_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::array<double, 8> measurements_;
    std::array<bool, 8> received_measurements_ = {false};

    std::unordered_map<std::string, Eigen::Vector3d> anchor_positions_;
    std::unordered_map<std::string, Eigen::Vector3d> tag_positions_;

    std::string eliko_frame_id_, uav_frame_id_;

    // Known roll and pitch, and yaw to be optimized
    double roll_, pitch_;
    double yaw_;
    std::array<double, 3> t_;  // Translation vector (x, y, z)

    std::deque<std::array<double, 3>> translation_history_;
    std::deque<double> yaw_history_;
    size_t moving_average_window_size_;
    size_t min_measurements_;
    double optimizer_time_window_s_;

};
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ElikoOptimizationNode>();
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
