#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

class AgvOdometryNode : public rclcpp::Node
{
public:
  AgvOdometryNode() : Node("agv_odometry_node")
  {
    rclcpp::SensorDataQoS qos;

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/arco/idmind_motors/odom", qos,
      std::bind(&AgvOdometryNode::odom_callback_, this, std::placeholders::_1));

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "/arco/idmind_imu/imu", qos,
      std::bind(&AgvOdometryNode::imu_callback_, this, std::placeholders::_1));

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("agv/odom", 10);

    RCLCPP_INFO(this->get_logger(), "AGV Odometry node initialized with local frame adjustment.");
  }

private:
  void odom_callback_(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    last_position_ = msg->pose.pose.position;
    last_twist_ = msg->twist.twist;
    last_stamp_ = msg->header.stamp;
    position_received_ = true;

    if (orientation_received_)
      publish_odometry();
  }

  void imu_callback_(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    last_orientation_ = msg->orientation;
    orientation_received_ = true;

    if (!initial_pose_received_ && position_received_) {
      Eigen::Quaterniond q(last_orientation_.w, last_orientation_.x, last_orientation_.y, last_orientation_.z);
      Eigen::Vector3d t(last_position_.x, last_position_.y, last_position_.z);
      initial_pose_inv_ = Sophus::SE3d(Sophus::SO3d(q), t).inverse();
      initial_pose_received_ = true;
    }

    if (position_received_)
      publish_odometry();
  }

  void publish_odometry()
  {
    if (!position_received_ || !orientation_received_ || !initial_pose_received_)
      return;

    Eigen::Quaterniond q(last_orientation_.w, last_orientation_.x, last_orientation_.y, last_orientation_.z);
    Eigen::Vector3d t(last_position_.x, last_position_.y, last_position_.z);
    Sophus::SE3d current_pose(Sophus::SO3d(q), t);

    Sophus::SE3d local_pose = initial_pose_inv_ * current_pose;

    Eigen::Vector3d t_local = local_pose.translation();
    Eigen::Quaterniond q_local = local_pose.unit_quaternion();

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = last_stamp_;
    odom_msg.header.frame_id = "arco/odom";
    odom_msg.child_frame_id = "arco/base_link";

    odom_msg.pose.pose.position.x = t_local.x();
    odom_msg.pose.pose.position.y = t_local.y();
    odom_msg.pose.pose.position.z = t_local.z();
    odom_msg.pose.pose.orientation.x = q_local.x();
    odom_msg.pose.pose.orientation.y = q_local.y();
    odom_msg.pose.pose.orientation.z = q_local.z();
    odom_msg.pose.pose.orientation.w = q_local.w();

    odom_msg.twist.twist = last_twist_;

    for (int i = 0; i < 36; ++i)
      odom_msg.pose.covariance[i] = 0.0;
    odom_msg.pose.covariance[0] = 0.1;
    odom_msg.pose.covariance[7] = 0.1;
    odom_msg.pose.covariance[14] = 0.1;
    odom_msg.pose.covariance[21] = 1e-3;
    odom_msg.pose.covariance[28] = 1e-3;
    odom_msg.pose.covariance[35] = 0.1;

    odom_pub_->publish(odom_msg);
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

  geometry_msgs::msg::Point last_position_;
  geometry_msgs::msg::Quaternion last_orientation_;
  geometry_msgs::msg::Twist last_twist_;
  rclcpp::Time last_stamp_;

  bool position_received_ = false;
  bool orientation_received_ = false;
  bool initial_pose_received_ = false;

  Sophus::SE3d initial_pose_inv_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AgvOdometryNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
