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


static inline double wrapPi(double a) {
    while (a >  M_PI) a -= 2.0*M_PI;
    while (a < -M_PI) a += 2.0*M_PI;
    return a;
}

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
    
    double lookahead_distance_, cruise_speed_;
    size_t current_index_ = 0;
    size_t offboard_counter_ = 0;

    std::vector<std::array<double, 4>> trajectory_;
    std::array<double, 4> current_pose_ = {0.0, 0.0, 0.0, 0.0}; // x, y, z, yaw
    std::mutex pose_mutex_;

    Eigen::Vector3d origin_pos_enu_{0.0, 0.0, 0.0};
	Eigen::Quaterniond origin_q_enu_{1.0, 0.0, 0.0, 0.0}; // w,x,y,z

    rclcpp::TimerBase::SharedPtr timer_, distances_timer_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr ros_odometry_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ros_gt_publisher_;

    rclcpp::Publisher<eliko_messages::msg::DistancesList>::SharedPtr distances_list_pub_;

    rclcpp::Subscription<VehicleOdometry>::SharedPtr vehicle_odometry_subscriber_;
	rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr vehicle_local_position_subscriber_;


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

    rclcpp::Time start_time_;
    bool started_ = false;
};

AGVOffboardControl::AGVOffboardControl() : Node("agv_offboard_control")
{
    //Declare parameters
    this->declare_parameter<std::string>("offboard_control_mode_topic", "/px4_2/fmu/in/offboard_control_mode");
    this->declare_parameter<std::string>("trajectory_setpoint_topic", "/px4_2/fmu/in/trajectory_setpoint");
    this->declare_parameter<std::string>("vehicle_command_topic", "/px4_2/fmu/in/vehicle_command");
    this->declare_parameter<std::string>("vehicle_odometry_topic", "/px4_2/fmu/out/vehicle_odometry");
    this->declare_parameter<std::string>("vehicle_local_position_topic", "/px4_2/fmu/out/vehicle_local_position");

    this->declare_parameter<std::string>("trajectory_csv_file", "trajectory_lemniscate_agv.csv");
    this->declare_parameter<std::string>("ros_odometry_topic", "/agv/odom");
    this->declare_parameter<std::string>("ros_gt_topic", "/agv/gt");

    this->declare_parameter<double>("lookahead_distance", 2.0);
    this->declare_parameter<double>("cruise_speed", 0.5);

    this->declare_parameter<std::vector<double>>("agv_origin", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

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
    this->get_parameter("cruise_speed", cruise_speed_);


    std::vector<double> agv_origin_vec;
		this->get_parameter("agv_origin", agv_origin_vec);
		if (agv_origin_vec.size() != 6) {
			RCLCPP_WARN(this->get_logger(),
						"agv_origin must have 6 elements [x,y,z,roll,pitch,yaw]; using zeros");
		} else {
			origin_pos_enu_ = Eigen::Vector3d(agv_origin_vec[0], agv_origin_vec[1], agv_origin_vec[2]);

			// RPY are given in ENU. Build quaternion qz(yaw)*qy(pitch)*qx(roll).
			const double roll  = agv_origin_vec[3];
			const double pitch = agv_origin_vec[4];
			const double yaw   = agv_origin_vec[5];

			origin_q_enu_ =
				Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitZ()) *
				Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
				Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX());
		}


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
    load_trajectory(full_csv_path);

    offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>(offboard_control_mode_topic_, 10);
    trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>(trajectory_setpoint_topic_, 10);
    vehicle_command_publisher_ = this->create_publisher<VehicleCommand>(vehicle_command_topic_, 10);

    ros_odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(ros_odometry_topic_, 10);
    ros_gt_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(ros_gt_topic_, 10);
    distances_list_pub_ = this->create_publisher<eliko_messages::msg::DistancesList>("eliko/Distances", 10);

    start_time_ = this->get_clock()->now();

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

            // Construct a nav_msgs/Odometry message
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = this->get_clock()->now();
            odom_msg.header.frame_id = "/agv/odom";  // or "odom" or "world"
            odom_msg.child_frame_id = "/agv/base_link";  // or "uav_base"

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
        vehicle_local_position_topic_, rclcpp::SensorDataQoS(),
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

            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
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
        
        start_time_ = this->get_clock()->now();
    }

    else if (!started_ && offboard_counter_ > 10){
        //Wait a bit for the drone to take off
        auto now = this->get_clock()->now();
        if ((now - start_time_).seconds() >= 5.0) {
            started_ = true;
            RCLCPP_INFO(this->get_logger(), "AGV trajectory setpoint publishing started after 5s delay.");
        }
    }

    publish_offboard_control_mode();

    //Start publishing setpoints
    if(started_) publish_trajectory_setpoint();
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

     // Unit direction (safe when distance==0 -> ux,uy=0)
    const double ux = (distance > 0.0) ? dx / distance : 0.0;
    const double uy = (distance > 0.0) ? dy / distance : 0.0;

    // Controller in WORLD ENU
    const double vx_world_enu = cruise_speed_ * ux;
    const double vy_world_enu = cruise_speed_ * uy;
    const double vz_world_enu = 0.0;

    // Desired facing yaw in WORLD ENU
    const double yaw_world_enu = std::atan2(dy, dx);

    // --- Convert command to PX4 LOCAL NED ---
    // 1) Velocity: world ENU -> local ENU -> local NED
    Eigen::Vector3d v_world_enu(vx_world_enu, vy_world_enu, vz_world_enu);
    Eigen::Vector3d v_local_enu = origin_q_enu_.inverse() * v_world_enu;
    Eigen::Vector3d v_local_ned = px4_ros_com::frame_transforms::enu_to_ned_local_frame(v_local_enu);

    // 2) Yaw: world ENU -> local ENU -> PX4(NED) heading
    Eigen::Quaterniond q_world_enu(Eigen::AngleAxisd(yaw_world_enu, Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond q_local_enu = origin_q_enu_.inverse() * q_world_enu;
    Eigen::Quaterniond q_ned = px4_ros_com::frame_transforms::ros_to_px4_orientation(q_local_enu);
    const double yaw_ned =
        px4_ros_com::frame_transforms::utils::quaternion::quaternion_get_yaw(q_ned);

    // --- Publish TrajectorySetpoint in PX4 local NED ---
    TrajectorySetpoint msg{};
    const auto nan = std::numeric_limits<float>::quiet_NaN();

    // Velocity mode
    msg.position = {nan, nan, nan};
    msg.velocity = {
        static_cast<float>(v_local_ned.x()),
        static_cast<float>(v_local_ned.y()),
        static_cast<float>(v_local_ned.z())
    };

    msg.yaw = static_cast<float>(wrapPi(yaw_ned));
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
