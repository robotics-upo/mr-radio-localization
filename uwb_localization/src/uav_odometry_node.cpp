#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

class UavOdometryNode : public rclcpp::Node
{
public:
  UavOdometryNode() : Node("uav_odometry_node")
  {
    // Initialize the UAV pose as identity.
    uav_odom_pose_ = Sophus::SE3d();

    rclcpp::SensorDataQoS qos; // Use a QoS profile compatible with sensor data

    // Create subscriptions for linear and angular velocities.
    linear_vel_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
      "/dji_sdk/velocity", qos,
      std::bind(&UavOdometryNode::uav_linear_vel_cb_, this, std::placeholders::_1));

    angular_vel_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
      "/dji_sdk/angular_velocity_fused", qos,
      std::bind(&UavOdometryNode::uav_angular_vel_cb_, this, std::placeholders::_1));

    // Create a publisher for the computed odometry.
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("uav/odom", 10);

    RCLCPP_INFO(this->get_logger(), "UAV Odometry integration node initialized");

  }

private:

    // Callback for linear velocity updates.
    void uav_linear_vel_cb_(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
    {
        // If this is the first message, store it and return.
        if (!last_uav_linear_vel_initialized_) {
        last_uav_linear_vel_msg_ = *msg;
        last_uav_linear_vel_initialized_ = true;
        return;
        }

        // Compute time difference.
        rclcpp::Time current_time(msg->header.stamp);
        rclcpp::Time last_time(last_uav_linear_vel_msg_.header.stamp);
        double dt = (current_time - last_time).seconds();

        // Get the linear velocity vector.
        Eigen::Vector3d linear_vel(msg->vector.x, msg->vector.y, msg->vector.z);
        // No angular contribution in this callback.
        Eigen::Vector3d angular_vel(0.0, 0.0, 0.0);

        last_uav_linear_vel_msg_ = *msg;

        // Update odometry.
        update_uav_odometry(linear_vel, angular_vel, dt);
        publish_odometry(msg->header.stamp);
    }

    // Callback for angular velocity updates.
    void uav_angular_vel_cb_(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
    {
        // If this is the first message, store it and return.
        if (!last_uav_angular_vel_initialized_) {
        last_uav_angular_vel_msg_ = *msg;
        last_uav_angular_vel_initialized_ = true;
        return;
        }

        // Compute time difference.
        rclcpp::Time current_time(msg->header.stamp);
        rclcpp::Time last_time(last_uav_angular_vel_msg_.header.stamp);
        double dt = (current_time - last_time).seconds();

        // Get the angular velocity vector.
        Eigen::Vector3d angular_vel(msg->vector.x, msg->vector.y, msg->vector.z);
        // No linear contribution in this callback.
        Eigen::Vector3d linear_vel(0.0, 0.0, 0.0);

        last_uav_angular_vel_msg_ = *msg;

        // Update odometry.
        update_uav_odometry(linear_vel, angular_vel, dt);
        publish_odometry(msg->header.stamp);
    }

    // Function that updates the UAV odometry.
    void update_uav_odometry(const Eigen::Vector3d linear_vel,
                            const Eigen::Vector3d angular_vel,
                            const double &dt)
    {
        // Create the 6D twist vector (xi) for SE(3) integration.
        Eigen::Matrix<double, 6, 1> xi;
        xi.head<3>() = linear_vel * dt;   // Translational component.
        xi.tail<3>() = angular_vel * dt;  // Rotational component.

        // Compute the incremental transformation using the exponential map.
        Sophus::SE3d delta = Sophus::SE3d::exp(xi);

        // Update the current pose by composing with the incremental transformation.
        uav_odom_pose_ = uav_odom_pose_ * delta;

        // Optionally update accumulated metrics.
        uav_translation_ += linear_vel.norm() * dt;
        uav_rotation_   += angular_vel.norm() * dt;

        // Build a 6x6 covariance matrix.
        // Here we assume translation and rotation uncertainties are uncorrelated.
        uav_odom_covariance_ = Eigen::Matrix<double, 6, 6>::Zero();
        double pos_variance = 0.1;  // Example variance for x, y, z (m^2)
        double rot_variance = 0.1;  // Example variance for roll, pitch, yaw (rad^2)
        uav_odom_covariance_.block<3,3>(0,0) = pos_variance * Eigen::Matrix3d::Identity();
        uav_odom_covariance_.block<3,3>(3,3) = rot_variance * Eigen::Matrix3d::Identity();
    }

    // Function to publish the current UAV odometry.
    void publish_odometry(const rclcpp::Time &stamp)
    {
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = stamp;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_link";

        // Extract the translation and orientation from the pose.
        Eigen::Vector3d t = uav_odom_pose_.translation();
        Eigen::Quaterniond q(uav_odom_pose_.rotationMatrix());

        odom_msg.pose.pose.position.x = t.x();
        odom_msg.pose.pose.position.y = t.y();
        odom_msg.pose.pose.position.z = t.z();
        odom_msg.pose.pose.orientation.x = q.x();
        odom_msg.pose.pose.orientation.y = q.y();
        odom_msg.pose.pose.orientation.z = q.z();
        odom_msg.pose.pose.orientation.w = q.w();

        // Flatten the 6x6 covariance matrix (row-major order).
        for (size_t i = 0; i < 6; i++) {
        for (size_t j = 0; j < 6; j++) {
            odom_msg.pose.covariance[i * 6 + j] = uav_odom_covariance_(i, j);
        }
        }

        odom_pub_->publish(odom_msg);
    }


    // Subscribers.
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr linear_vel_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr angular_vel_sub_;

    // Publisher.
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

    // Storage for the last messages to compute time differences.
    geometry_msgs::msg::Vector3Stamped last_uav_linear_vel_msg_;
    geometry_msgs::msg::Vector3Stamped last_uav_angular_vel_msg_;
    bool last_uav_linear_vel_initialized_ = false;
    bool last_uav_angular_vel_initialized_ = false;

    // UAV odometry state.
    Sophus::SE3d uav_odom_pose_;
    double uav_translation_ = 0.0;
    double uav_rotation_ = 0.0;
    Eigen::Matrix<double, 6, 6> uav_odom_covariance_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<UavOdometryNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
