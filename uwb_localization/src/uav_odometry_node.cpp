#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

class UavOdometryNode : public rclcpp::Node
{
public:
  UavOdometryNode() : Node("uav_odometry_node")
  {
    this->declare_parameter<bool>("use_attitude", true);
    this->get_parameter("use_attitude", use_attitude_);

    uav_odom_pose_ = Sophus::SE3d();
    rclcpp::SensorDataQoS qos;

    linear_vel_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
      "/dji_sdk/velocity", qos,
      std::bind(&UavOdometryNode::uav_linear_vel_cb_, this, std::placeholders::_1));

    if (use_attitude_) {
      attitude_sub_ = this->create_subscription<geometry_msgs::msg::QuaternionStamped>(
        "/dji_sdk/attitude", qos,
        std::bind(&UavOdometryNode::attitude_cb_, this, std::placeholders::_1));
    } else {
      angular_vel_sub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
        "/dji_sdk/angular_velocity_fused", qos,
        std::bind(&UavOdometryNode::angular_vel_cb_, this, std::placeholders::_1));
    }

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("uav/odom", 10);
    RCLCPP_INFO(this->get_logger(), "UAV Odometry node initialized. Using %s for orientation.",
                use_attitude_ ? "attitude topic" : "angular velocity integration");
  }

private:
  void uav_linear_vel_cb_(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
  {
    if (!linear_initialized_) {
      last_linear_vel_msg_ = *msg;
      linear_initialized_ = true;
      return;
    }

    rclcpp::Time t_now = msg->header.stamp;
    rclcpp::Time t_prev = last_linear_vel_msg_.header.stamp;
    double dt = (t_now - t_prev).seconds();

    Eigen::Vector3d linear_vel_enu(msg->vector.x, msg->vector.y, msg->vector.z);
    last_linear_vel_msg_ = *msg;

    Eigen::Vector3d angular_vel = last_angular_vel_;

    update_odometry(linear_vel_enu, angular_vel, dt);
    publish_odometry(t_now);
  }

  void attitude_cb_(const geometry_msgs::msg::QuaternionStamped::SharedPtr msg)
  {
    uav_orientation_ = msg->quaternion;
    attitude_received_ = true;

    if (!initial_orientation_received_) {
      Eigen::Quaterniond q(msg->quaternion.w, msg->quaternion.x, msg->quaternion.y, msg->quaternion.z);
      initial_orientation_inv_ = Sophus::SE3d(Sophus::SO3d(q), Eigen::Vector3d::Zero()).inverse();
      initial_orientation_received_ = true;
    }
  }

  void angular_vel_cb_(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
  {
    last_angular_vel_ = Eigen::Vector3d(msg->vector.x, msg->vector.y, msg->vector.z);
    angular_initialized_ = true;
  }

  void update_odometry(const Eigen::Vector3d& linear_vel_enu,
                       const Eigen::Vector3d& angular_vel,
                       const double& dt)
  {
    Eigen::Vector3d linear_vel_local;
    if (use_attitude_) {
      linear_vel_local = linear_vel_enu;
    } else {
      linear_vel_local = uav_odom_pose_.rotationMatrix().inverse() * linear_vel_enu;
    }

    Eigen::Matrix<double, 6, 1> xi;
    xi.head<3>() = linear_vel_local * dt;
    if(use_attitude_) xi.tail<3>() = Eigen::Vector3d::Zero();
    else xi.tail<3>() = angular_vel * dt;

    Sophus::SE3d delta = Sophus::SE3d::exp(xi);
    uav_odom_pose_ = uav_odom_pose_ * delta;

    uav_translation_ += linear_vel_local.norm() * dt;

    uav_odom_covariance_ = Eigen::Matrix<double, 6, 6>::Zero();
    double pos_var = 0.1;
    double rot_var = 0.1;
    uav_odom_covariance_.block<3, 3>(0, 0) = pos_var * Eigen::Matrix3d::Identity();
    uav_odom_covariance_.block<3, 3>(3, 3) = rot_var * Eigen::Matrix3d::Identity();
  }

  void publish_odometry(const rclcpp::Time& stamp)
  {
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = stamp;
    odom_msg.header.frame_id = "odom";
    odom_msg.child_frame_id = "base_link";

    Sophus::SE3d local_pose;
    if (use_attitude_) {
      if (!attitude_received_ || !initial_orientation_received_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Waiting for /dji_sdk/attitude message...");
        return;
      }
      Eigen::Quaterniond q_curr(uav_orientation_.w, uav_orientation_.x, uav_orientation_.y, uav_orientation_.z);
      Sophus::SE3d current_pose(Sophus::SO3d(q_curr), uav_odom_pose_.translation());
      local_pose = initial_orientation_inv_ * current_pose;
    } else {
      local_pose = uav_odom_pose_;
    }

    Eigen::Vector3d t = local_pose.translation();
    Eigen::Quaterniond q(local_pose.rotationMatrix());

    odom_msg.pose.pose.position.x = t.x();
    odom_msg.pose.pose.position.y = t.y();
    odom_msg.pose.pose.position.z = t.z();
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    odom_msg.pose.pose.orientation.w = q.w();

    for (size_t i = 0; i < 6; ++i)
      for (size_t j = 0; j < 6; ++j)
        odom_msg.pose.covariance[i * 6 + j] = uav_odom_covariance_(i, j);

    odom_pub_->publish(odom_msg);
  }

  // Parameters
  bool use_attitude_;

  // Subscribers
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr linear_vel_sub_;
  rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr attitude_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr angular_vel_sub_;

  // Publisher
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

  // Internal state
  geometry_msgs::msg::Vector3Stamped last_linear_vel_msg_;
  Eigen::Vector3d last_angular_vel_ = Eigen::Vector3d::Zero();
  geometry_msgs::msg::Quaternion uav_orientation_;
  Sophus::SE3d initial_orientation_inv_;

  bool linear_initialized_ = false;
  bool angular_initialized_ = false;
  bool attitude_received_ = false;
  bool initial_orientation_received_ = false;

  Sophus::SE3d uav_odom_pose_;
  double uav_translation_ = 0.0;
  Eigen::Matrix<double, 6, 6> uav_odom_covariance_;
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<UavOdometryNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
