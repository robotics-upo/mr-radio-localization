#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <vector>
#include <sstream>
#include <deque>
#include <Eigen/Dense>
#include <utility>


class UWBOptimizationNode : public rclcpp::Node {

public:

    UWBOptimizationNode() : Node("uwb_optimization_node"), roll_(0.0), pitch_(0.0), yaw_(0.0) {
    
    // Initialize subscriptions for 8 distance measurement topics
    int count = 0;
    for (int i = 0; i < 2; ++i) {
        for(int j = 0; j < 4; ++j){
            std::string topic_name = "/range/t" + std::to_string(i+1) + 'a' + std::to_string(j+1); /*TODO: parametrize topic names*/
            auto callback = [this, count](const std_msgs::msg::Float32::SharedPtr msg) {
                this->measurement_callback(count, msg);
            };
            distance_subs_.emplace_back(this->create_subscription<std_msgs::msg::Float32>(
                topic_name, 10, callback));
            
            count++;
        }
    }

    // Create publisher/broadcaster for optimized transformation
    tf_publisher_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("uwb_optimization_node/optimized_T", 10);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);


    // Known roll and pitch values
    roll_ = 0.0;   // Example roll in radians
    pitch_ = 0.0;  // Example pitch in radians
    yaw_ = 0.0;    // Initial guess for yaw

    t_ = {0.0, 0.0, -2.0};       // Translation That_ts (x, y, z)

    window_size_ = 5; //moving average sample window (N)

    RCLCPP_INFO(this->get_logger(), "UWB Optimization Node initialized.");
  }

private:

  // Callback for each measurement
  void measurement_callback(int index, const std_msgs::msg::Float32::SharedPtr msg) {
      measurements_[index] = msg->data;
      received_measurements_[index] = true;

      // Check if all measurements have been received
      if (std::all_of(received_measurements_.begin(), received_measurements_.end(), [](bool v) { return v; })) {
          RCLCPP_INFO(this->get_logger(), "All measurements received. Running optimization...");
          run_optimization();
          std::fill(received_measurements_.begin(), received_measurements_.end(), false);  // Reset for next batch
      }
  }

  std::pair<std::array<double, 3>, double> moving_average(const double yaw, const std::array<double, 3>& t) {
    
        // Add new values to history
        translation_history_.push_back(t);
        yaw_history_.push_back(yaw);

        // Remove oldest if history exceeds desired size
        if (translation_history_.size() > window_size_) {
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
        that_st_msg.header.frame_id = "ground_vehicle";  // Adjust frame_id as needed
        that_st_msg.child_frame_id = "uav_opt";            // Adjust child_frame_id as needed

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

        // Define anchor and tag positions (replace with actual positions)
        std::vector<Eigen::Vector3d> anchors = { {0.5, 0.5, 0.5}, {-0.5, -0.5, 0.5}, {0.5, -0.5, -0.5}, {-0.5, 0.5, -0.5} };
        std::vector<Eigen::Vector3d> tags = { {0.5, 0.5, 0.25}, {-0.5, -0.5, 0.25} };

      // Add residual blocks for each measurement
        for (size_t i = 0; i < tags.size(); ++i) {
            for (size_t j = 0; j < anchors.size(); ++j) {
                ceres::CostFunction* cost_function =
                    UWBResidual::Create(anchors[j], tags[i], measurements_[i * 4 + j], roll_, pitch_);
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
        RCLCPP_INFO(this->get_logger(), summary.FullReport().c_str());

        /*Run moving average*/
        auto [t_avg, yaw_avg] = moving_average(yaw_, t_);
        t_ = t_avg;
        yaw_ = yaw_avg;

        // Construct transformation matrix T with optimized yaw, roll, pitch, and translation
        Eigen::Matrix4d That_ts = build_transformation_matrix(roll_, pitch_, yaw_, t_);

        // Convert Eigen::Matrix4d to a string
        std::stringstream ss;
        ss << That_ts;
        RCLCPP_INFO(this->get_logger(), "Optimized Transformation Matrix:\n%s", ss.str().c_str());

        publish_transform(That_ts);

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

    std::vector<rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr> distance_subs_;
    rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr tf_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::array<double, 8> measurements_;
    std::array<bool, 8> received_measurements_ = {false};

    // Known roll and pitch, and yaw to be optimized
    double roll_, pitch_;
    double yaw_;
    std::array<double, 3> t_;  // Translation vector (x, y, z)

    std::deque<std::array<double, 3>> translation_history_;
    std::deque<double> yaw_history_;
    size_t window_size_;

};
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UWBOptimizationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
