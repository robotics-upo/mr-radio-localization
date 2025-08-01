#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>

#include <rclcpp/rclcpp.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <mutex>
#include <cmath>
#include <limits>

using namespace std::chrono_literals;
using namespace px4_msgs::msg;

class AGVOffboardControl : public rclcpp::Node
{
public:
    AGVOffboardControl();

private:
    void timer_callback();
    void load_trajectory(const std::string &filename);
    void publish_offboard_control_mode();
    void publish_trajectory_setpoint();
    void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0);

    // Retrieve parameter values
    std::string offboard_control_mode_topic_;
    std::string trajectory_setpoint_topic_;
    std::string vehicle_command_topic_;
    std::string vehicle_odometry_topic_;
    std::string ros_odometry_topic_;
    std::string traj_file_;
    
    double lookahead_distance_, kp_v_;
    size_t current_index_ = 0;
    size_t offboard_counter_ = 0;

    std::vector<std::array<double, 3>> trajectory_;
    std::array<double, 3> current_pose_ = {0.0, 0.0, 0.0};
    std::mutex pose_mutex_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr ros_odometry_publisher_;

    rclcpp::Subscription<VehicleOdometry>::SharedPtr vehicle_odometry_subscriber_;
};

AGVOffboardControl::AGVOffboardControl() : Node("agv_offboard_control")
{
    this->declare_parameter<std::string>("offboard_control_mode_topic", "/px4_2/fmu/in/offboard_control_mode");
    this->declare_parameter<std::string>("trajectory_setpoint_topic", "/px4_2/fmu/in/trajectory_setpoint");
    this->declare_parameter<std::string>("vehicle_command_topic", "/px4_2/fmu/in/vehicle_command");
    this->declare_parameter<std::string>("vehicle_odometry_topic", "/px4_2/fmu/out/vehicle_odometry");
    this->declare_parameter<std::string>("trajectory_csv_file", "trajectory_agv.csv");
    this->declare_parameter<std::string>("ros_odometry_topic", "/agv/odom");

    this->declare_parameter<double>("lookahead_distance", 1.0);
    this->declare_parameter<double>("kp_v", 1.0);

    this->get_parameter("offboard_control_mode_topic", offboard_control_mode_topic_);
    this->get_parameter("trajectory_setpoint_topic", trajectory_setpoint_topic_);
    this->get_parameter("vehicle_command_topic", vehicle_command_topic_);
    this->get_parameter("vehicle_odometry_topic", vehicle_odometry_topic_);
    this->get_parameter("trajectory_csv_file", traj_file_);
    this->get_parameter("ros_odometry_topic", ros_odometry_topic_);

    this->get_parameter("lookahead_distance", lookahead_distance_);
    this->get_parameter("kp_v", kp_v_);

    load_trajectory(traj_file_);

    offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>(offboard_control_mode_topic_, 10);
    trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>(trajectory_setpoint_topic_, 10);
    vehicle_command_publisher_ = this->create_publisher<VehicleCommand>(vehicle_command_topic_, 10);

    ros_odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(ros_odometry_topic_, 10);


    vehicle_odometry_subscriber_ = this->create_subscription<VehicleOdometry>(
        vehicle_odometry_topic_, rclcpp::SensorDataQoS(),
        [this](VehicleOdometry::SharedPtr msg) {
            double qw = msg->q[0];
            double qx = msg->q[1];
            double qy = msg->q[2];
            double qz = msg->q[3];
            double siny_cosp = 2.0 * (qw * qz + qx * qy);
            double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
            double yaw = std::atan2(siny_cosp, cosy_cosp);

            std::lock_guard<std::mutex> lock(pose_mutex_);
            current_pose_ = {msg->position[0], msg->position[1], yaw};

            // Construct a nav_msgs/Odometry message
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = this->get_clock()->now();
            odom_msg.header.frame_id = "/uav/odom";  // or "odom" or "world"
            odom_msg.child_frame_id = "/uav/base_link";  // or "uav_base"

            // Position
            odom_msg.pose.pose.position.x = msg->position[0];
            odom_msg.pose.pose.position.y = msg->position[1];
            odom_msg.pose.pose.position.z = msg->position[2];

            // Orientation
            odom_msg.pose.pose.orientation.x = msg->q[0];
            odom_msg.pose.pose.orientation.y = msg->q[1];
            odom_msg.pose.pose.orientation.z = msg->q[2];
            odom_msg.pose.pose.orientation.w = msg->q[3];

            // Linear velocity
            odom_msg.twist.twist.linear.x = msg->velocity[0];
            odom_msg.twist.twist.linear.y = msg->velocity[1];
            odom_msg.twist.twist.linear.z = msg->velocity[2];

            // Angular velocity
            odom_msg.twist.twist.angular.x = msg->angular_velocity[0];
            odom_msg.twist.twist.angular.y = msg->angular_velocity[1];
            odom_msg.twist.twist.angular.z = msg->angular_velocity[2];

            // Fill pose covariance (position + orientation)
            odom_msg.pose.covariance = {
                msg->position_variance[0], 0, 0, 0, 0, 0,
                0, msg->position_variance[1], 0, 0, 0, 0,
                0, 0, msg->position_variance[2], 0, 0, 0,
                0, 0, 0, msg->orientation_variance[0], 0, 0,
                0, 0, 0, 0, msg->orientation_variance[1], 0,
                0, 0, 0, 0, 0, msg->orientation_variance[2]
            };

            // Fill twist covariance (linear velocity)
            odom_msg.twist.covariance = {
                msg->velocity_variance[0], 0, 0, 0, 0, 0,
                0, msg->velocity_variance[1], 0, 0, 0, 0,
                0, 0, msg->velocity_variance[2], 0, 0, 0,
                0, 0, 0, 0.0, 0, 0,
                0, 0, 0, 0, 0.0, 0,
                0, 0, 0, 0, 0, 0.0
            };

            // Publish the converted message
            ros_odometry_publisher_->publish(odom_msg);
        });

    timer_ = this->create_wall_timer(100ms, std::bind(&AGVOffboardControl::timer_callback, this));
}

void AGVOffboardControl::timer_callback()
{
    if (++offboard_counter_ == 10) {
        publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);
        publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1);
    }

    publish_offboard_control_mode();
    publish_trajectory_setpoint();
}

void AGVOffboardControl::load_trajectory(const std::string &filename)
{
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::array<double, 3> pose;
        for (int i = 0; i < 3; ++i) {
            std::getline(ss, line, ',');
            pose[i] = std::stod(line);
        }
        trajectory_.push_back(pose);
    }
}

void AGVOffboardControl::publish_offboard_control_mode()
{
    OffboardControlMode msg{};
    msg.velocity = true;
    msg.position = false;
    msg.acceleration = false;
    msg.attitude = false;
    msg.body_rate = false;
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    offboard_control_mode_publisher_->publish(msg);
}

void AGVOffboardControl::publish_trajectory_setpoint()
{
    std::array<double, 3> pose;
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        pose = current_pose_;
    }

    if (trajectory_.empty()) return;

    const auto &target = trajectory_[std::min(current_index_, trajectory_.size() - 1)];
    double dx = target[0] - pose[0];
    double dy = target[1] - pose[1];
    double distance = std::sqrt(dx*dx + dy*dy);

    if (distance < lookahead_distance_ && current_index_ < trajectory_.size() - 1)
        ++current_index_;

    double ux = dx / distance;
    double uy = dy / distance;
    double vx = kp_v_ * distance * ux;
    double vy = kp_v_ * distance * uy;
    double yaw = std::atan2(dy, dx);

    TrajectorySetpoint msg{};
    auto nan = std::numeric_limits<float>::quiet_NaN();
    msg.position = {nan, nan, nan};
    msg.velocity = {static_cast<float>(vx), static_cast<float>(vy), 0.0f};
    msg.yaw = static_cast<float>(yaw);
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    trajectory_setpoint_publisher_->publish(msg);
}

void AGVOffboardControl::publish_vehicle_command(uint16_t command, float param1, float param2)
{
    VehicleCommand msg{};
    msg.param1 = param1;
    msg.param2 = param2;
    msg.command = command;
    msg.target_system = 3; // PX4 instance 2 = MAV_SYS_ID 3
    msg.target_component = 1;
    msg.source_system = 1;
    msg.source_component = 1;
    msg.from_external = true;
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    vehicle_command_publisher_->publish(msg);
}

int main(int argc, char *argv[])
{
    std::cout << "Starting AGV offboard control node..." << std::endl;
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AGVOffboardControl>());
    rclcpp::shutdown();
    return 0;
}
