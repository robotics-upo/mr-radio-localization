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

#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>

#include <rclcpp/rclcpp.hpp>
#include <stdint.h>

#include <chrono>
#include <iostream>

#include <rclcpp/qos.hpp>

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

		// Retrieve parameter values
		std::string offboard_control_mode_topic;
		std::string trajectory_setpoint_topic;
		std::string vehicle_command_topic;
		std::string vehicle_odometry_topic;

		this->get_parameter("offboard_control_mode_topic", offboard_control_mode_topic);
		this->get_parameter("trajectory_setpoint_topic", trajectory_setpoint_topic);
		this->get_parameter("vehicle_command_topic", vehicle_command_topic);
		this->get_parameter("vehicle_odometry_topic", vehicle_odometry_topic);

		// Create publishers with the topics from params
		offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>(offboard_control_mode_topic, 10);
		trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>(trajectory_setpoint_topic, 10);
		vehicle_command_publisher_ = this->create_publisher<VehicleCommand>(vehicle_command_topic, 10);

		ros_odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/uav/odom", 10);

		// Set QoS to match PX4 publisher
		rclcpp::QoS qos_profile = rclcpp::SensorDataQoS();

		vehicle_odometry_subscriber_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
			vehicle_odometry_topic, qos_profile,
			[this](const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {
				
				double qx = msg->q[0];
				double qy = msg->q[1];
				double qz = msg->q[2];
				double qw = msg->q[3];

				// Convert quaternion to yaw
				double siny_cosp = 2.0 * (qw * qz + qx * qy);
				double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
				double yaw = std::atan2(siny_cosp, cosy_cosp);

				{
					std::lock_guard<std::mutex> lock(position_mutex_);
					current_pose_[0] = msg->position[0];
					current_pose_[1] = msg->position[1];
					current_pose_[2] = msg->position[2];
					current_pose_[3] = yaw;
				}

				// Construct a nav_msgs/Odometry message
				nav_msgs::msg::Odometry odom_msg;
				odom_msg.header.stamp = this->get_clock()->now();
				odom_msg.header.frame_id = "map";  // or "odom" or "world"
				odom_msg.child_frame_id = "base_link";  // or "uav_base"

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

private:
	rclcpp::TimerBase::SharedPtr timer_;

	rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
	rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
	rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
	rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr ros_odometry_publisher_;

	rclcpp::Subscription<VehicleOdometry>::SharedPtr vehicle_odometry_subscriber_;

	std::atomic<uint64_t> timestamp_;   //!< common synced timestamped

	uint64_t offboard_setpoint_counter_;   //!< counter for the number of setpoints sent

	std::array<double, 4> current_pose_{0.0, 0.0, 0.0, 0.0};	//[x,y,z,yaw]
	std::mutex position_mutex_;  // to make access thread-safe

	void publish_offboard_control_mode();
	void publish_trajectory_setpoint();
	void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0);
};

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

/**
 * @brief Publish the offboard control mode.
 *        For this example, only position and altitude controls are active.
 */
void UAVOffboardControl::publish_offboard_control_mode()
{
	OffboardControlMode msg{};
	msg.position = true;
	msg.velocity = false;
	msg.acceleration = false;
	msg.attitude = false;
	msg.body_rate = false;
	msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
	offboard_control_mode_publisher_->publish(msg);
}

/**
 * @brief Publish a trajectory setpoint
 *        For this example, it sends a trajectory setpoint to make the
 *        vehicle hover at 5 meters with a yaw angle of 180 degrees.
 */
void UAVOffboardControl::publish_trajectory_setpoint()
{
	TrajectorySetpoint msg{};
	msg.position = {0.0, 0.0, -5.0};
	msg.yaw = -3.14; // [-PI:PI]
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
	msg.target_system = 0; //CAREFUL WITH THIS!!!
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