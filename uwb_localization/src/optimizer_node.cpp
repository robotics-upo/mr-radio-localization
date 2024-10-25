#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <vector>



class UWBOptimizationNode : public rclcpp::Node {

public:

  UWBOptimizationNode() : Node("optimizer_node") {
    
      // Initialize subscriptions for 8 distance measurement topics
      for (int i = 0; i < 2; ++i) {
        for(int j = 0; j < 4; ++j){
          std::string topic_name = "/range/t" + std::to_string(i+1) + 'a' + std::to_string(j+1); /*TODO: parametrize topic names*/
          auto callback = [this, i](const std_msgs::msg::Float32::SharedPtr msg) {
              this->measurement_callback(i, msg);
          };
          distance_subs_.emplace_back(this->create_subscription<std_msgs::msg::Float32>(
              topic_name, 10, callback));
          
          count_++;

        }
      }

      // Initial guesses for optimization
      q_ = {1.0, 0.0, 0.0, 0.0};  // Quaternion (w, x, y, z)
      t_ = {0.0, 0.0, 0.0};       // Translation (x, y, z)

      RCLCPP_INFO(this->get_logger(), "UWB Optimization Node initialized.");
  }

private:

  int count_ = 0;
  std::vector<rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr> distance_subs_;
  std::array<double, 8> measurements_;
  std::array<bool, 8> received_measurements_ = {false};
  std::array<double, 4> q_;  // Quaternion (w, x, y, z)
  std::array<double, 3> t_;  // Translation (x, y, z)

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

   // Run the optimization once all measurements are received
  void run_optimization() {
      ceres::Problem problem;

      // Define anchor and tag positions (replace with actual positions)
      std::vector<Eigen::Vector3d> anchors = { /* anchor positions */ };
      std::vector<Eigen::Vector3d> tags = { /* tag positions */ };

      // Add residual blocks for each measurement
      for (size_t i = 0; i < tags.size(); ++i) {
          for (size_t j = 0; j < anchors.size(); ++j) {
              ceres::CostFunction* cost_function =
                  UWBResidual::Create(anchors[j], tags[i], measurements_[i * 4 + j]);
              problem.AddResidualBlock(cost_function, nullptr, q_.data(), t_.data());
          }
      }

      // Set the quaternion parameterization
      problem.SetParameterization(q_.data(), new ceres::QuaternionParameterization());

      // Configure solver options
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::DENSE_QR;
      options.minimizer_progress_to_stdout = true;

      // Solve
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      RCLCPP_INFO(this->get_logger(), summary.FullReport().c_str());

      // Output the optimized values
      RCLCPP_INFO(this->get_logger(), "Optimized quaternion: [%f, %f, %f, %f]", q_[0], q_[1], q_[2], q_[3]);
      RCLCPP_INFO(this->get_logger(), "Optimized translation: [%f, %f, %f]", t_[0], t_[1], t_[2]);
  }

  // Residual struct for Ceres
  struct UWBResidual {
      UWBResidual(const Eigen::Vector3d& anchor, const Eigen::Vector3d& tag, double measured_distance)
          : anchor_(anchor), tag_(tag), measured_distance_(measured_distance) {}

      template <typename T>
      bool operator()(const T* const q, const T* const t, T* residual) const {
          T anchor_transformed[3];
          ceres::QuaternionRotatePoint(q, anchor_.data(), anchor_transformed);

          anchor_transformed[0] += t[0];
          anchor_transformed[1] += t[1];
          anchor_transformed[2] += t[2];

          T dx = T(tag_(0)) - anchor_transformed[0];
          T dy = T(tag_(1)) - anchor_transformed[1];
          T dz = T(tag_(2)) - anchor_transformed[2];
          T predicted_distance = ceres::sqrt(dx * dx + dy * dy + dz * dz);

          residual[0] = T(measured_distance_) - predicted_distance;
          return true;
      }

      static ceres::CostFunction* Create(const Eigen::Vector3d& anchor, const Eigen::Vector3d& tag, double measured_distance) {
          return (new ceres::AutoDiffCostFunction<UWBResidual, 1, 4, 3>(
              new UWBResidual(anchor, tag, measured_distance)));
      }

      const Eigen::Vector3d anchor_;
      const Eigen::Vector3d tag_;
      const double measured_distance_;
  };

};
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UWBOptimizationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
