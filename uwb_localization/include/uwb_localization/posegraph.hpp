#ifndef UWB_LOCALIZATION_POSEGRAPH_H_
#define UWB_LOCALIZATION_POSEGRAPH_H_

#include <cmath>
#include <deque>
#include <vector>
#include <random>
#include <algorithm>
#include <map>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <sophus/se3.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "uwb_localization/msg/pose_with_covariance_stamped_array.hpp" 
#include "uwb_localization/utils.hpp"

using namespace uwb_localization;

namespace posegraph
{
    struct PriorConstraint {
        Sophus::SE3d pose;  // measured relative transform
        Eigen::Matrix<double, 4, 4> covariance;
    };
    
    struct MeasurementConstraint {
        int id_begin;
        int id_end;
        Sophus::SE3d t_T_s;  // measured relative transform
        Eigen::Matrix<double, 4, 4> covariance;
    
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
      };
    
    using VectorOfConstraints =
        std::vector<MeasurementConstraint, Eigen::aligned_allocator<MeasurementConstraint>>;
    
    using MapOfStates =
        std::map<int,
                 State,
                 std::less<int>,
                 Eigen::aligned_allocator<std::pair<const int, State>>>;
    
    
    struct Measurements {
        rclcpp::Time timestamp;      // The time at which the measurement was taken.
        Sophus::SE3d odom_pose;
        
        Eigen::Matrix<double, 6, 6> odom_covariance;
      
        // Pointer to an associated point cloud (keyframe scan).
        pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_scan;
        pcl::PointCloud<pcl::PointXYZ>::Ptr radar_scan;
    
         // Constructor to initialize the pointer.
         Measurements() : lidar_scan(new pcl::PointCloud<pcl::PointXYZ>), radar_scan(new pcl::PointCloud<pcl::PointXYZ>) {}
    };
    
    
    struct RadarMeasurements {
        int KF_id;
        pcl::PointCloud<pcl::PointXYZ>::Ptr radar_scan;
        Sophus::SE3d odom_pose;
    };


    inline void getPoseGraph(const MapOfStates &map, uwb_localization::msg::PoseWithCovarianceStampedArray &msg){

        // Iterate over the global map and convert each state's optimized pose.
        for (const auto &kv : map) {             
            const State &state = kv.second;
            Sophus::SE3d T = buildTransformationSE3(state.roll, state.pitch, state.state);
            geometry_msgs::msg::PoseWithCovarianceStamped pose = buildPoseMsg(T, state.covariance, state.timestamp, msg.header.frame_id);
            msg.array.push_back(pose);
        }

        return;

    }

    // Function to write the pose graph to a CSV file.
    inline void writePoseGraphToCSV(const uwb_localization::msg::PoseWithCovarianceStampedArray& poseGraph,
        const std::string &filename)
    {
            std::ofstream file(filename);
            if (!file.is_open()) {
                // Handle error: unable to open file.
                return;
            }

            // Write CSV header.
            file << "id,timestamp,frame_id,position_x,position_y,position_z,"
            << "orientation_x,orientation_y,orientation_z,orientation_w";
            // Write covariance headers: cov_0, cov_1, ..., cov_35.
            for (int i = 0; i < 36; ++i) {
                file << ",cov_" << i;
            }
            file << "\n";

            int id = 0;
            for (const auto &poseMsg : poseGraph.array) {
                // Here we combine seconds and nanoseconds into one timestamp string.
                // Adjust this formatting as needed.
                std::stringstream timestampStream;
                timestampStream << poseMsg.header.stamp.sec << "." << std::setw(9) << std::setfill('0') 
                << poseMsg.header.stamp.nanosec;

                file << id << ","
                << timestampStream.str() << ","
                << poseMsg.header.frame_id << ","
                << std::fixed << std::setprecision(6)
                << poseMsg.pose.pose.position.x << ","
                << poseMsg.pose.pose.position.y << ","
                << poseMsg.pose.pose.position.z << ","
                << poseMsg.pose.pose.orientation.x << ","
                << poseMsg.pose.pose.orientation.y << ","
                << poseMsg.pose.pose.orientation.z << ","
                << poseMsg.pose.pose.orientation.w;

                // Write all 36 covariance elements.
                for (size_t i = 0; i < 36; ++i) {
                    file << "," << poseMsg.pose.covariance[i];
                }
                file << "\n";
                ++id;
            }
            file.close();
    }

    inline bool isRelativeTransformAvailable(const rclcpp::Time &current_time, const rclcpp::Time &latest_relative_time, const double threshold){

        bool is_recent = (current_time - latest_relative_time).seconds() < threshold;

        return is_recent;
    }

    inline bool isNodeFixedKF(const int current_node_id, const int node_id, const double &max_keyframes, const double &min_keyframes){
        return (current_node_id - node_id > max_keyframes || current_node_id < min_keyframes);
    }

    inline bool isNodeFixedTime(const rclcpp::Time &current_time, const int node_id, MapOfStates &map, const double &window_size){
        return (current_time - map[node_id].timestamp).seconds() > window_size;
    }
}

#endif