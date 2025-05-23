#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>

class AgvOdometryNode : public rclcpp::Node
{
public:
  AgvOdometryNode() : Node("agv_odometry_node")
  {
    rclcpp::SensorDataQoS qos;

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/arco/idmind_motors/odom", qos,
      std::bind(&AgvOdometryNode::agv_odom_cb_, this, std::placeholders::_1));

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "/arco/idmind_imu/imu", qos,
      std::bind(&AgvOdometryNode::agv_imu_cb_, this, std::placeholders::_1));

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("agv/odom", 10);

    RCLCPP_INFO(this->get_logger(), "AGV Odometry fusion node initialized.");
  }

private:
  void agv_odom_cb_(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    last_position_ = msg->pose.pose.position;
    last_twist_ = msg->twist.twist;
    last_stamp_ = msg->header.stamp;
    position_received_ = true;

    publish_odometry();
  }

  void agv_imu_cb_(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    last_orientation_ = msg->orientation;
    orientation_received_ = true;
  }

  void publish_odometry()
  {
    if (!position_received_ || !orientation_received_)
      return;

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = last_stamp_;
    odom_msg.header.frame_id = "arco/odom";
    odom_msg.child_frame_id = "arco/base_link";

    odom_msg.pose.pose.position = last_position_;
    odom_msg.pose.pose.orientation = last_orientation_;
    odom_msg.twist.twist = last_twist_;  // optional: carry forward velocity from wheel odom

    // Covariance (example values)
    for (int i = 0; i < 36; ++i)
      odom_msg.pose.covariance[i] = 0.0;
    odom_msg.pose.covariance[0] = 0.1;   // x
    odom_msg.pose.covariance[7] = 0.1;   // y
    odom_msg.pose.covariance[14] = 0.1;  // z
    odom_msg.pose.covariance[21] = 1e-3; // roll
    odom_msg.pose.covariance[28] = 1e-3; // pitch
    odom_msg.pose.covariance[35] = 0.1;  // yaw

    odom_pub_->publish(odom_msg);
  }

  // Subscribers and publisher
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

  // Latest data from each topic
  geometry_msgs::msg::Point last_position_;
  geometry_msgs::msg::Quaternion last_orientation_;
  geometry_msgs::msg::Twist last_twist_;
  rclcpp::Time last_stamp_;

  bool position_received_ = false;
  bool orientation_received_ = false;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AgvOdometryNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
