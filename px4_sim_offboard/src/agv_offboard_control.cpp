#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/rover_velocity_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>

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
    AGVOffboardControl() : Node("agv_offboard_control")
    {
        this->declare_parameter<std::string>("offboard_control_mode_topic", "/px4_2/fmu/in/offboard_control_mode");
        this->declare_parameter<std::string>("velocity_setpoint_topic", "/px4_2/fmu/in/rover_velocity_setpoint");
        this->declare_parameter<std::string>("vehicle_command_topic", "/px4_2/fmu/in/vehicle_command");
        this->declare_parameter<std::string>("vehicle_odometry_topic", "/px4_2/fmu/out/vehicle_odometry");
        this->declare_parameter<std::string>("trajectory_csv_file", "trajectory_agv.csv");
        this->declare_parameter<double>("lookahead_distance", 1.0);
        this->declare_parameter<double>("kp_v", 1.0);
        this->declare_parameter<double>("kp_w", 1.0);

        this->get_parameter("offboard_control_mode_topic", offboard_control_mode_topic_);
        this->get_parameter("velocity_setpoint_topic", velocity_setpoint_topic_);
        this->get_parameter("vehicle_command_topic", vehicle_command_topic_);
        this->get_parameter("vehicle_odometry_topic", vehicle_odometry_topic_);
        this->get_parameter("trajectory_csv_file", traj_file_);
        this->get_parameter("lookahead_distance", lookahead_distance_);
        this->get_parameter("kp_v", kp_v_);
        this->get_parameter("kp_w", kp_w_);

        load_trajectory(traj_file_);

        offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>(offboard_control_mode_topic_, 10);
        rover_velocity_setpoint_publisher_ = this->create_publisher<RoverVelocitySetpoint>(velocity_setpoint_topic_, 10);
        vehicle_command_publisher_ = this->create_publisher<VehicleCommand>(vehicle_command_topic_, 10);

        vehicle_odometry_subscriber_ = this->create_subscription<VehicleOdometry>(
            vehicle_odometry_topic_, rclcpp::SensorDataQoS(),
            [this](VehicleOdometry::SharedPtr msg) {
                double qx = msg->q[0];
                double qy = msg->q[1];
                double qz = msg->q[2];
                double qw = msg->q[3];
                double siny_cosp = 2.0 * (qw * qz + qx * qy);
                double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
                double yaw = std::atan2(siny_cosp, cosy_cosp);

                std::lock_guard<std::mutex> lock(pose_mutex_);
                current_pose_ = {msg->position[0], msg->position[1], yaw};
            });

        timer_ = this->create_wall_timer(100ms, std::bind(&AGVOffboardControl::timer_callback, this));
    }

private:
    void timer_callback()
    {
        if (++offboard_counter_ == 10) {
            publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);
            publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1);
        }

        publish_offboard_control_mode();
        publish_rover_velocity_setpoint();
    }

    void load_trajectory(const std::string &filename)
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

    void publish_offboard_control_mode()
    {
        OffboardControlMode msg{};
        msg.velocity = true;
        msg.position = false;
        msg.attitude = false;
        msg.body_rate = false;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        offboard_control_mode_publisher_->publish(msg);
    }

    void publish_rover_velocity_setpoint()
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

        double bearing = std::atan2(dy, dx);
        double speed = kp_v_ * distance;

        RoverVelocitySetpoint msg{};
        msg.speed = static_cast<float>(speed);
        msg.bearing = static_cast<float>(bearing);
        msg.yaw = std::numeric_limits<float>::quiet_NaN();  // default to vehicle yaw
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        rover_velocity_setpoint_publisher_->publish(msg);
    }

    void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0)
    {
        VehicleCommand msg{};
        msg.param1 = param1;
        msg.param2 = param2;
        msg.command = command;
        msg.target_system = 3;
        msg.target_component = 1;
        msg.source_system = 1;
        msg.source_component = 1;
        msg.from_external = true;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        vehicle_command_publisher_->publish(msg);
    }

    std::string offboard_control_mode_topic_, velocity_setpoint_topic_, vehicle_command_topic_, vehicle_odometry_topic_, traj_file_;
    double lookahead_distance_, kp_v_, kp_w_;
    size_t current_index_ = 0;
    size_t offboard_counter_ = 0;

    std::vector<std::array<double, 3>> trajectory_;
    std::array<double, 3> current_pose_ = {0.0, 0.0, 0.0};
    std::mutex pose_mutex_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<RoverVelocitySetpoint>::SharedPtr rover_velocity_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Subscription<VehicleOdometry>::SharedPtr vehicle_odometry_subscriber_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AGVOffboardControl>());
    rclcpp::shutdown();
    return 0;
}
