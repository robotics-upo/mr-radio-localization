#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>

#include <px4_ros_com/frame_transforms.h>

#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <Eigen/Geometry>


#include <std_msgs/msg/float64.hpp>
#include <unordered_map>

#include <eliko_messages/msg/distances.hpp>
#include <eliko_messages/msg/distances_list.hpp>

#include <rclcpp/rclcpp.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <mutex>
#include <cmath>
#include <limits>

#include "ament_index_cpp/get_package_share_directory.hpp"

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
    void publish_distances();

    // Retrieve parameter values
    std::string offboard_control_mode_topic_;
    std::string trajectory_setpoint_topic_;
    std::string vehicle_command_topic_;
    std::string vehicle_odometry_topic_;
    std::string vehicle_local_position_topic_;
    std::string ros_odometry_topic_, ros_gt_topic_;
    std::string traj_file_;
    
    double lookahead_distance_, kp_v_;
    size_t current_index_ = 0;
    size_t offboard_counter_ = 0;

    std::vector<std::array<double, 4>> trajectory_;
    std::array<double, 4> current_pose_ = {0.0, 0.0, 0.0, 0.0}; // x, y, z, yaw
    std::mutex pose_mutex_;

    rclcpp::TimerBase::SharedPtr timer_, distances_timer_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr ros_odometry_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ros_gt_publisher_;

    rclcpp::Publisher<eliko_messages::msg::DistancesList>::SharedPtr distances_list_pub_;

    rclcpp::Subscription<VehicleOdometry>::SharedPtr vehicle_odometry_subscriber_;


    std::unordered_map<std::string, double> uwb_distances_;
    std::mutex uwb_mutex_;
    std::vector<rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr> uwb_subscribers_;
    const std::vector<std::string> uwb_topic_names_ = {
        "/uwb_gz_simulator/distances/a1t1",
        "/uwb_gz_simulator/distances/a1t2",
        "/uwb_gz_simulator/distances/a2t1",
        "/uwb_gz_simulator/distances/a2t2",
        "/uwb_gz_simulator/distances/a3t1",
        "/uwb_gz_simulator/distances/a3t2",
        "/uwb_gz_simulator/distances/a4t1",
        "/uwb_gz_simulator/distances/a4t2"
    };

    std::unordered_map<std::string, std::string> anchor_ids_;
    std::unordered_map<std::string, std::string> tag_ids_;

};

AGVOffboardControl::AGVOffboardControl() : Node("agv_offboard_control")
{
    //Declare parameters
    this->declare_parameter<std::string>("offboard_control_mode_topic", "/px4_2/fmu/in/offboard_control_mode");
    this->declare_parameter<std::string>("trajectory_setpoint_topic", "/px4_2/fmu/in/trajectory_setpoint");
    this->declare_parameter<std::string>("vehicle_command_topic", "/px4_2/fmu/in/vehicle_command");
    this->declare_parameter<std::string>("vehicle_odometry_topic", "/px4_2/fmu/out/vehicle_odometry");
    this->declare_parameter<std::string>("vehicle_local_position_topic", "/px4_2/fmu/out/vehicle_local_position");

    this->declare_parameter<std::string>("trajectory_csv_file", "trajectory_agv.csv");
    this->declare_parameter<std::string>("ros_odometry_topic", "/agv/odom");
    this->declare_parameter<std::string>("ros_gt_topic", "/agv/gt");

    this->declare_parameter<double>("lookahead_distance", 2.0);
    this->declare_parameter<double>("kp_v", 0.5);

    // Declare anchor IDs
    this->declare_parameter<std::string>("anchors.a1.id", "0x0009D6");
    this->declare_parameter<std::string>("anchors.a2.id", "0x0009E5");
    this->declare_parameter<std::string>("anchors.a3.id", "0x0016FA");
    this->declare_parameter<std::string>("anchors.a4.id", "0x0016CF");

    // Declare tag IDs
    this->declare_parameter<std::string>("tags.t1.id", "0x001155");
    this->declare_parameter<std::string>("tags.t2.id", "0x001397");

    // Load parameters
    this->get_parameter("offboard_control_mode_topic", offboard_control_mode_topic_);
    this->get_parameter("trajectory_setpoint_topic", trajectory_setpoint_topic_);
    this->get_parameter("vehicle_command_topic", vehicle_command_topic_);
    this->get_parameter("vehicle_odometry_topic", vehicle_odometry_topic_);
    this->get_parameter("vehicle_local_position_topic", vehicle_local_position_topic_);
    this->get_parameter("trajectory_csv_file", traj_file_);
    this->get_parameter("ros_odometry_topic", ros_odometry_topic_);
    this->get_parameter("ros_gt_topic", ros_gt_topic_);

    this->get_parameter("lookahead_distance", lookahead_distance_);
    this->get_parameter("kp_v", kp_v_);

    // Get anchor IDs
    this->get_parameter("anchors.a1.id", anchor_ids_["a1"]);
    this->get_parameter("anchors.a2.id", anchor_ids_["a2"]);
    this->get_parameter("anchors.a3.id", anchor_ids_["a3"]);
    this->get_parameter("anchors.a4.id", anchor_ids_["a4"]);

    // Get tag IDs
    this->get_parameter("tags.t1.id", tag_ids_["t1"]);
    this->get_parameter("tags.t2.id", tag_ids_["t2"]);

    std::string package_path = ament_index_cpp::get_package_share_directory("px4_sim_offboard");
    std::string full_csv_path = package_path + "/trajectories_positions/" + traj_file_;
    load_trajectory(traj_file_);

    offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>(offboard_control_mode_topic_, 10);
    trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>(trajectory_setpoint_topic_, 10);
    vehicle_command_publisher_ = this->create_publisher<VehicleCommand>(vehicle_command_topic_, 10);

    ros_odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(ros_odometry_topic_, 10);
    ros_gt_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(ros_gt_topic_, 10);
    distances_list_pub_ = this->create_publisher<eliko_messages::msg::DistancesList>("eliko/Distances", 10);

    vehicle_odometry_subscriber_ = this->create_subscription<VehicleOdometry>(
        vehicle_odometry_topic_, rclcpp::SensorDataQoS(),
        [this](VehicleOdometry::SharedPtr msg) {
            
            // Convert orientation
            Eigen::Quaterniond q_ned(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
            Eigen::Quaterniond q_enu = px4_ros_com::frame_transforms::px4_to_ros_orientation(q_ned);

            // Convert position and velocity
            Eigen::Vector3d pos_ned(msg->position[0], msg->position[1], msg->position[2]);
            Eigen::Vector3d pos_enu = px4_ros_com::frame_transforms::ned_to_enu_local_frame(pos_ned);

            Eigen::Vector3d vel_ned(msg->velocity[0], msg->velocity[1], msg->velocity[2]);
            Eigen::Vector3d vel_enu = px4_ros_com::frame_transforms::ned_to_enu_local_frame(vel_ned);

           
           {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                current_pose_[0] = msg->position[0];
                current_pose_[1] = msg->position[1];
                current_pose_[2] = msg->position[2];
                current_pose_[3] = px4_ros_com::frame_transforms::utils::quaternion::quaternion_get_yaw(q_ned);
            }

            // Construct a nav_msgs/Odometry message
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = this->get_clock()->now();
            odom_msg.header.frame_id = "/agv/odom";  // or "odom" or "world"
            odom_msg.child_frame_id = "/uav/base_link";  // or "uav_base"

            odom_msg.pose.pose.position.x = pos_enu.x();
            odom_msg.pose.pose.position.y = pos_enu.y();
            odom_msg.pose.pose.position.z = pos_enu.z();

            odom_msg.pose.pose.orientation.x = q_enu.x();
            odom_msg.pose.pose.orientation.y = q_enu.y();
            odom_msg.pose.pose.orientation.z = q_enu.z();
            odom_msg.pose.pose.orientation.w = q_enu.w();

            odom_msg.twist.twist.linear.x = vel_enu.x();
            odom_msg.twist.twist.linear.y = vel_enu.y();
            odom_msg.twist.twist.linear.z = vel_enu.z();

            odom_msg.twist.twist.angular.x = msg->angular_velocity[1];
            odom_msg.twist.twist.angular.y = msg->angular_velocity[0];
            odom_msg.twist.twist.angular.z = -msg->angular_velocity[2];

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

    this->create_subscription<px4_msgs::msg::VehicleLocalPosition>(
        vehicle_local_position_topic_, rclcpp::SensorDataQoS(),
        [this](const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg) {
            geometry_msgs::msg::PoseStamped pose_msg;

            // Header
            pose_msg.header.stamp = this->get_clock()->now();
            pose_msg.header.frame_id = "map";  // or "world", depending on your convention

            Eigen::Vector3d pos_ned(msg->x, msg->y, msg->z);
            Eigen::Vector3d pos_enu = px4_ros_com::frame_transforms::ned_to_enu_local_frame(pos_ned);

            pose_msg.pose.position.x = pos_enu.x();
            pose_msg.pose.position.y = pos_enu.y();
            pose_msg.pose.position.z = pos_enu.z();

            Eigen::Quaterniond q_heading(Eigen::AngleAxisd(msg->heading, Eigen::Vector3d::UnitZ()));
            Eigen::Quaterniond q_enu = px4_ros_com::frame_transforms::px4_to_ros_orientation(q_heading);

            pose_msg.pose.orientation.x = q_enu.x();
            pose_msg.pose.orientation.y = q_enu.y();
            pose_msg.pose.orientation.z = q_enu.z();
            pose_msg.pose.orientation.w = q_enu.w();
            
            ros_gt_publisher_->publish(pose_msg);
        });

    timer_ = this->create_wall_timer(100ms, std::bind(&AGVOffboardControl::timer_callback, this));

    distances_timer_ = this->create_wall_timer(100ms, std::bind(&AGVOffboardControl::publish_distances, this));

    for (const auto& topic : uwb_topic_names_) {
        auto sub = this->create_subscription<std_msgs::msg::Float64>(
            topic, 10,
            [this, topic = topic](const std_msgs::msg::Float64::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(this->uwb_mutex_);
                this->uwb_distances_[topic] = msg->data;
            });
        uwb_subscribers_.push_back(sub);
    }

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
	if (!file.is_open()) {
		RCLCPP_ERROR(this->get_logger(), "Failed to open trajectory file: %s", filename.c_str());
		return;
	}

	std::string line;
	size_t line_num = 0;

	// Skip the header row
	std::getline(file, line);  // <- Skip line 1

	while (std::getline(file, line)) {
		line_num++;
		std::stringstream ss(line);
		std::string val;
		std::array<double, 4> pose;
		int i = 0;
		while (std::getline(ss, val, ',')) {
			try {
				pose[i++] = std::stod(val);
			} catch (const std::exception &e) {
				RCLCPP_WARN(this->get_logger(), "Skipping invalid line %zu: %s", line_num + 1, line.c_str());
				i = -1;
				break;
			}
		}
		if (i == 4) {
			double x_enu = pose[0];
			double y_enu = pose[1];
			double z_enu = pose[2];
			double yaw_enu = pose[3];

			// Convert to NED
			pose[0] = y_enu;
			pose[1] = x_enu;
			pose[2] = -z_enu;
			pose[3] = -yaw_enu;

			trajectory_.push_back(pose);
		}
	}
	file.close();
	RCLCPP_INFO(this->get_logger(), "Loaded %zu trajectory points.", trajectory_.size());
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
    std::array<double, 4> pose;
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


void AGVOffboardControl::publish_distances()
{
    eliko_messages::msg::DistancesList list_msg;
    list_msg.header.stamp = this->get_clock()->now();
    list_msg.header.frame_id = "arco/eliko";

    {
        std::lock_guard<std::mutex> lock(uwb_mutex_);
        for (const auto& [topic, distance] : uwb_distances_) {
            // Parse topic like "/uwb_gz_simulator/distances/a1t2"
            std::string key = topic.substr(topic.find_last_of('/') + 1); // "a1t2"
            if (key.size() != 4) continue;

            std::string anchor_key = key.substr(0, 2); // "a1"
            std::string tag_key = key.substr(2, 2);    // "t2"

            if (anchor_ids_.count(anchor_key) && tag_ids_.count(tag_key)) {
                eliko_messages::msg::Distances d;
                d.anchor_sn = anchor_ids_[anchor_key];
                d.tag_sn = tag_ids_[tag_key];
                d.distance = static_cast<float>(distance);
                list_msg.anchor_distances.push_back(d);
            }
        }

        // 🔴 Clear old values after publishing
        uwb_distances_.clear();
    }

    if (!list_msg.anchor_distances.empty()) {
        distances_list_pub_->publish(list_msg);
    }

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
