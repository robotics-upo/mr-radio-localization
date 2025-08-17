/****************************************************************************
 *
 * Copyright 2020 PX4 Development Team. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/**
 * @brief Offboard control example
 * @file offboard_control.cpp
 * @addtogroup examples
 * @author Mickey Cowden <info@cowden.tech>
 * @author Nuno Marques <nuno.marques@dronesolutions.io>
 */

#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_control_mode.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>

#include <px4_ros_com/frame_transforms.h>

#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>
#include <stdint.h>

#include <chrono>
#include <iostream>

#include <rclcpp/qos.hpp>

#include <fstream>
#include <sstream>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"


using namespace std::chrono;
using namespace std::chrono_literals;
using namespace px4_msgs::msg;

class UAVOffboardControl : public rclcpp::Node
{
public:
	UAVOffboardControl() : Node("uav_offboard_control")
	{

		// Declare parameters with default values
		this->declare_parameter<std::string>("offboard_control_mode_topic", "/px4_1/fmu/in/offboard_control_mode");
		this->declare_parameter<std::string>("trajectory_setpoint_topic", "/px4_1/fmu/in/trajectory_setpoint");
		this->declare_parameter<std::string>("vehicle_command_topic", "/px4_1/fmu/in/vehicle_command");
		this->declare_parameter<std::string>("vehicle_odometry_topic", "/px4_1/fmu/out/vehicle_odometry");
		this->declare_parameter<std::string>("vehicle_local_position_topic", "/px4_1/fmu/out/vehicle_local_position");
		this->declare_parameter<std::string>("trajectory_csv_file", "trajectory_lemniscate_uav.csv");
		this->declare_parameter<std::string>("ros_odometry_topic", "/uav/odom");
		this->declare_parameter<std::string>("ros_gt_topic", "/uav/gt");

		this->declare_parameter<double>("lookahead_distance", 2.0);
		this->declare_parameter<double>("kp_v", 1.0);
		this->declare_parameter<double>("kp_w", 0.1);

		this->declare_parameter<std::vector<double>>("uav_origin", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

		// Retrieve parameter values
		std::string offboard_control_mode_topic_;
		std::string trajectory_setpoint_topic_;
		std::string vehicle_command_topic_;
		std::string vehicle_odometry_topic_;
		std::string vehicle_local_position_topic_;
		std::string ros_odometry_topic_, ros_gt_topic_;
		std::string traj_file_;

		this->get_parameter("offboard_control_mode_topic", offboard_control_mode_topic_);
		this->get_parameter("trajectory_setpoint_topic", trajectory_setpoint_topic_);
		this->get_parameter("vehicle_command_topic", vehicle_command_topic_);
		this->get_parameter("vehicle_odometry_topic", vehicle_odometry_topic_);
		this->get_parameter("vehicle_local_position_topic", vehicle_local_position_topic_);
		this->get_parameter("trajectory_csv_file", traj_file_);
		this->get_parameter("lookahead_distance", lookahead_distance_);
		this->get_parameter("ros_odometry_topic", ros_odometry_topic_);
		this->get_parameter("ros_gt_topic", ros_gt_topic_);

		this->get_parameter("kp_v", kp_v_);
		this->get_parameter("kp_w", kp_w_);

		std::vector<double> uav_origin_vec;
		this->get_parameter("uav_origin", uav_origin_vec);
		if (uav_origin_vec.size() != 6) {
			RCLCPP_WARN(this->get_logger(),
						"uav_origin must have 6 elements [x,y,z,roll,pitch,yaw]; using zeros");
		} else {
			origin_pos_enu_ = Eigen::Vector3d(uav_origin_vec[0], uav_origin_vec[1], uav_origin_vec[2]);

			// RPY are given in ENU. Build quaternion qz(yaw)*qy(pitch)*qx(roll).
			const double roll  = uav_origin_vec[3];
			const double pitch = uav_origin_vec[4];
			const double yaw   = uav_origin_vec[5];

			origin_q_enu_ =
				Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitZ()) *
				Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
				Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX());
		}
		
		std::string package_path = ament_index_cpp::get_package_share_directory("px4_sim_offboard");
		std::string full_csv_path = package_path + "/trajectories_positions/" + traj_file_;
		load_trajectory(full_csv_path);
		initial_hover_target_ = trajectory_.front(); 

		// Create publishers with the topics from params
		offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>(offboard_control_mode_topic_, 10);
		trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>(trajectory_setpoint_topic_, 10);
		vehicle_command_publisher_ = this->create_publisher<VehicleCommand>(vehicle_command_topic_, 10);

		ros_odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(ros_odometry_topic_, 10);
		ros_gt_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(ros_gt_topic_, 10);
		// Set QoS to match PX4 publisher
		rclcpp::QoS qos_profile = rclcpp::SensorDataQoS();

		vehicle_odometry_subscriber_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
			vehicle_odometry_topic_, qos_profile,
			[this](const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {


				// Convert orientation
                Eigen::Quaterniond q_ned(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
                Eigen::Quaterniond q_enu = px4_ros_com::frame_transforms::px4_to_ros_orientation(q_ned);

                // Convert position and velocity
                Eigen::Vector3d pos_ned(msg->position[0], msg->position[1], msg->position[2]);
                Eigen::Vector3d pos_enu = px4_ros_com::frame_transforms::ned_to_enu_local_frame(pos_ned);

                Eigen::Vector3d vel_ned(msg->velocity[0], msg->velocity[1], msg->velocity[2]);
                Eigen::Vector3d vel_enu = px4_ros_com::frame_transforms::ned_to_enu_local_frame(vel_ned);

				// Construct a nav_msgs/Odometry message
				nav_msgs::msg::Odometry odom_msg;
				odom_msg.header.stamp = this->get_clock()->now();
				odom_msg.header.frame_id = "/uav/odom";  // or "odom" or "world"
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

		vehicle_local_position_subscriber_ = this->create_subscription<px4_msgs::msg::VehicleLocalPosition>(
			vehicle_local_position_topic_, qos_profile,
			[this](const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg) {
				geometry_msgs::msg::PoseStamped pose_msg;

				// Header
				pose_msg.header.stamp = this->get_clock()->now();
				pose_msg.header.frame_id = "map";  // or "world", depending on your convention

				// PX4 local position is NED, origin at arming/start.
      			Eigen::Vector3d pos_ned(msg->x, msg->y, msg->z);
				
				// Convert to ENU (local)
				Eigen::Vector3d pos_enu_local =
					px4_ros_com::frame_transforms::ned_to_enu_local_frame(pos_ned);

				// Rotate by origin orientation, then translate by origin position to get world/map
				Eigen::Vector3d pos_enu_world = origin_q_enu_ * pos_enu_local + origin_pos_enu_;

				pose_msg.pose.position.x = pos_enu_world.x();
				pose_msg.pose.position.y = pos_enu_world.y();
				pose_msg.pose.position.z = pos_enu_world.z();

				// Orientation: PX4 gives heading (yaw in NED). Build NED yaw quaternion,
				// convert to ENU, then compose with origin orientation.
				Eigen::Quaterniond q_heading_ned(Eigen::AngleAxisd(msg->heading, Eigen::Vector3d::UnitZ()));
				Eigen::Quaterniond q_heading_enu =
					px4_ros_com::frame_transforms::px4_to_ros_orientation(q_heading_ned);

				Eigen::Quaterniond q_world_enu = origin_q_enu_ * q_heading_enu;

				// Extract ENU yaw from q_world_enu
				Eigen::Vector3d rpy = q_world_enu.toRotationMatrix().eulerAngles(2, 1, 0); // ZYX
				double yaw_world_enu = rpy[0];
				 // UPDATE current pose in WORLD ENU
				{
					std::lock_guard<std::mutex> lock(position_mutex_);
					current_pose_[0] = pos_enu_world.x();
					current_pose_[1] = pos_enu_world.y();
					current_pose_[2] = pos_enu_world.z();
					current_pose_[3] = yaw_world_enu;
				}

				pose_msg.pose.orientation.x = q_world_enu.x();
				pose_msg.pose.orientation.y = q_world_enu.y();
				pose_msg.pose.orientation.z = q_world_enu.z();
				pose_msg.pose.orientation.w = q_world_enu.w();

				ros_gt_publisher_->publish(pose_msg);
			});


		offboard_setpoint_counter_ = 0;

		auto timer_callback = [this]() -> void {

			if (offboard_setpoint_counter_ == 10) {
				// Change to Offboard mode after 10 setpoints
				this->publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);

				// Arm the vehicle
				this->arm();
			}

			// offboard_control_mode needs to be paired with trajectory_setpoint
			publish_offboard_control_mode();
			publish_trajectory_setpoint();

			// stop the counter after reaching 11
			if (offboard_setpoint_counter_ < 11) {
				offboard_setpoint_counter_++;
			}
		};
		timer_ = this->create_wall_timer(100ms, timer_callback);
	}

	void arm();
	void disarm();
	void land();

private:
	rclcpp::TimerBase::SharedPtr timer_;

	rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
	rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
	rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
	rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr ros_odometry_publisher_;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ros_gt_publisher_;

	rclcpp::Subscription<VehicleOdometry>::SharedPtr vehicle_odometry_subscriber_;
	rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr vehicle_local_position_subscriber_;

	std::atomic<uint64_t> timestamp_;   //!< common synced timestamped

	uint64_t offboard_setpoint_counter_;   //!< counter for the number of setpoints sent

	std::array<double, 4> current_pose_{0.0, 0.0, 0.0, 0.0};	//[x,y,z,yaw]
	std::mutex position_mutex_;  // to make access thread-safe

	void publish_offboard_control_mode();
	void publish_trajectory_setpoint();
	void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0);
	void load_trajectory(const std::string &filename);

	std::vector<std::array<double, 4>> trajectory_;
	size_t current_target_idx_ = 0;
	size_t closest_idx_ = 0;
	double lookahead_distance_ = 2.0;  // meters
	double kp_v_ = 1.0;         // proportional gain
	double kp_w_ = 0.1;

	bool hover_reached_ = false;
	bool land_sent_ = false;  // flag to ensure LAND command is sent only once
	std::array<double, 4> initial_hover_target_{0.0, 0.0, -2.0, 0.0};
	double hover_tolerance_ = 0.3;  // meters

	Eigen::Vector3d origin_pos_enu_{0.0, 0.0, 0.0};
	Eigen::Quaterniond origin_q_enu_{1.0, 0.0, 0.0, 0.0}; // w,x,y,z

};

void UAVOffboardControl::load_trajectory(const std::string &filename)
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
			trajectory_.push_back(pose);
		}
	}
	file.close();

	RCLCPP_INFO(this->get_logger(), "Loaded %zu trajectory points.", trajectory_.size());
}


/**
 * @brief Send a command to Arm the vehicle
 */
void UAVOffboardControl::arm()
{
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);

	RCLCPP_INFO(this->get_logger(), "Arm command send");
}

/**
 * @brief Send a command to Disarm the vehicle
 */
void UAVOffboardControl::disarm()
{
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0);

	RCLCPP_INFO(this->get_logger(), "Disarm command send");
}

void UAVOffboardControl::land()
{
    // PX4: NAV_LAND starts an auto land at current position for MC
    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);
	land_sent_ = true;  // Set flag to prevent multiple LAND commands
    RCLCPP_INFO(this->get_logger(), "LAND command sent");
}

/**
 * @brief Publish the offboard control mode.
 *        For this example, only position and altitude controls are active.
 */
void UAVOffboardControl::publish_offboard_control_mode()
{
	OffboardControlMode msg{};
	if(!hover_reached_){
		msg.position = true;
		msg.velocity = false;
	}
	else{
		msg.position = false;
		msg.velocity = true;
	}
	msg.acceleration = false;
	msg.attitude = false;
	msg.body_rate = false;
	msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
	offboard_control_mode_publisher_->publish(msg);
}

/**
 * @brief Publish a trajectory setpoint
 */
void UAVOffboardControl::publish_trajectory_setpoint()
{
	std::array<double, 4> pose;
	{
		std::lock_guard<std::mutex> lock(position_mutex_);
		pose = current_pose_;
	}

	if (!hover_reached_) {
		// Check distance to hover point
		double dx = pose[0] - initial_hover_target_[0];
		double dy = pose[1] - initial_hover_target_[1];
		double dz = pose[2] - initial_hover_target_[2];
		double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

		// Convert hover target (world ENU) to local NED for PX4
		Eigen::Vector3d p_world_enu(initial_hover_target_[0], initial_hover_target_[1], initial_hover_target_[2]);
		Eigen::Vector3d p_local_enu = origin_q_enu_.inverse() * (p_world_enu - origin_pos_enu_);
		Eigen::Vector3d p_local_ned = px4_ros_com::frame_transforms::enu_to_ned_local_frame(p_local_enu);

		// Yaw: build world ENU quaternion, move to local ENU, then to PX4(NED)
		Eigen::Quaterniond q_hover_world_enu(
			Eigen::AngleAxisd(initial_hover_target_[3], Eigen::Vector3d::UnitZ()));
		Eigen::Quaterniond q_hover_local_enu = origin_q_enu_.inverse() * q_hover_world_enu;
		Eigen::Quaterniond q_hover_ned = px4_ros_com::frame_transforms::ros_to_px4_orientation(q_hover_local_enu);
		double yaw_ned = px4_ros_com::frame_transforms::utils::quaternion::quaternion_get_yaw(q_hover_ned);

		TrajectorySetpoint msg{};
		msg.position = {static_cast<float>(p_local_ned.x()),
						static_cast<float>(p_local_ned.y()),
						static_cast<float>(p_local_ned.z())};
		msg.yaw = static_cast<float>(yaw_ned);
		msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
		trajectory_setpoint_publisher_->publish(msg);

		if (dist < hover_tolerance_) {
			RCLCPP_INFO(this->get_logger(), "Initial hover position reached.");
			hover_reached_ = true;
		}
		return;
	}

	// ---- Begin normal trajectory tracking ----

	if (trajectory_.empty()) {
		RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Trajectory is empty");
		return;
	}

	if(land_sent_){
		RCLCPP_INFO(this->get_logger(), "Landing already sent, not sending further setpoints.");
		return;
	}

	// Step 1: Find the closest point in a forward-looking window
	size_t window_size = 50;
	size_t search_start = closest_idx_;
	size_t search_end = std::min(closest_idx_ + window_size, trajectory_.size() - 1);

	double min_dist = std::numeric_limits<double>::max();
	size_t new_closest_idx = closest_idx_;  // start from last known

	for (size_t i = search_start; i < search_end; ++i) {
		double dx = trajectory_[i][0] - pose[0];
		double dy = trajectory_[i][1] - pose[1];
		double dz = trajectory_[i][2] - pose[2];
		double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
		if (dist < min_dist) {
			min_dist = dist;
			new_closest_idx = i;
		}
	}

	double accum_dist = 0.0;
	size_t new_target_idx = closest_idx_;

	for (size_t i = closest_idx_ + 1; i < trajectory_.size(); ++i) {
		double dx = trajectory_[i][0] - trajectory_[i-1][0];
		double dy = trajectory_[i][1] - trajectory_[i-1][1];
		double dz = trajectory_[i][2] - trajectory_[i-1][2];
		accum_dist += std::sqrt(dx*dx + dy*dy + dz*dz);
		if (accum_dist >= lookahead_distance_) {
			new_target_idx = i;
			break;
		}
	}


	// Update indices
	closest_idx_ = new_closest_idx;
	current_target_idx_ = new_target_idx;

	const auto &target = trajectory_[current_target_idx_];

	// Compute position error
	double ex = target[0] - pose[0];
	double ey = target[1] - pose[1];
	double ez = target[2] - pose[2];
	double dist = std::sqrt(ex*ex + ey*ey + ez*ez);

	double dist_to_end = std::sqrt(
		(trajectory_.back()[0] - pose[0]) * (trajectory_.back()[0] - pose[0]) +
		(trajectory_.back()[1] - pose[1]) * (trajectory_.back()[1] - pose[1]) +
		(trajectory_.back()[2] - pose[2]) * (trajectory_.back()[2] - pose[2])
	);

	// If we're very near the end of the path, trigger LAND (once)
	if ((closest_idx_ >= 0.9*trajectory_.size()) && dist_to_end <= 0.25) {
		land();
		return;
	}

	// Compute unit direction vector
	double ux = ex / dist;
	double uy = ey / dist;
	double uz = ez / dist;

	// Compute commanded velocity magnitude
	double max_vel = 1.5;
	double vel_mag = std::min(kp_v_ * dist, max_vel);	// double vel_mag = 0.5; //track at constant velocity
	// Apply velocity vector
	double vx = vel_mag * ux;
	double vy = vel_mag * uy;
	double vz = vel_mag * uz;

	// v_world_enu from your controller (vx, vy, vz in world ENU)
	Eigen::Vector3d v_world_enu(vx, vy, vz);

	// Transform velocity to local ENU then to local NED
	Eigen::Vector3d v_local_enu = origin_q_enu_.inverse() * v_world_enu;
	Eigen::Vector3d v_local_ned = px4_ros_com::frame_transforms::enu_to_ned_local_frame(v_local_enu);

	// Yaw to face the target in world ENU
	double yaw_world_enu = std::atan2(ey, ex);

	// Convert that yaw to PX4 NED heading
	Eigen::Quaterniond q_world_enu(Eigen::AngleAxisd(yaw_world_enu, Eigen::Vector3d::UnitZ()));
	Eigen::Quaterniond q_local_enu = origin_q_enu_.inverse() * q_world_enu;
	Eigen::Quaterniond q_ned = px4_ros_com::frame_transforms::ros_to_px4_orientation(q_local_enu);
	double yaw_ned = px4_ros_com::frame_transforms::utils::quaternion::quaternion_get_yaw(q_ned);

	// Build setpoint
	TrajectorySetpoint msg{};
	auto nan = std::numeric_limits<float>::quiet_NaN();
	msg.position = {nan, nan, nan};  // velocity mode
	msg.velocity = {static_cast<float>(v_local_ned.x()),
					static_cast<float>(v_local_ned.y()),
					static_cast<float>(v_local_ned.z())};
	msg.yaw = static_cast<float>(yaw_ned);
	msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;

	trajectory_setpoint_publisher_->publish(msg);
}

/**
 * @brief Publish vehicle commands
 * @param command   Command code (matches VehicleCommand and MAVLink MAV_CMD codes)
 * @param param1    Command parameter 1
 * @param param2    Command parameter 2
 */
void UAVOffboardControl::publish_vehicle_command(uint16_t command, float param1, float param2)
{
	VehicleCommand msg{};
	msg.param1 = param1;
	msg.param2 = param2;
	msg.command = command;
	msg.target_system = 2; //2 //CAREFUL WITH THIS!!!
	msg.target_component = 1;
	msg.source_system = 1;
	msg.source_component = 1;
	msg.from_external = true;
	msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
	vehicle_command_publisher_->publish(msg);
}

int main(int argc, char *argv[])
{
	std::cout << "Starting offboard control node..." << std::endl;
	setvbuf(stdout, NULL, _IONBF, BUFSIZ);
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<UAVOffboardControl>());

	rclcpp::shutdown();
	return 0;
}