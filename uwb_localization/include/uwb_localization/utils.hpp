#ifndef UWB_LOCALIZATION_UTILS_H_
#define UWB_LOCALIZATION_UTILS_H_

#include <cmath>
#include <deque>
#include <vector>
#include <random>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <sophus/se3.hpp>


namespace uwb_localization
{
    struct UWBMeasurement {
        rclcpp::Time timestamp;   // Time of the measurement
        std::string tag_id;       // ID of the tag
        std::string anchor_id;    // ID of the anchor
        double distance;          // Measured distance (in meters)
    
        // Position of the tag in the UAV odometry frame.
        Eigen::Vector3d tag_odom_pose;
        // Position of the anchor in the AGV odometry frame.
        Eigen::Vector3d anchor_odom_pose;
    
        //Store the cumulative displacement (e.g., total distance traveled) for each robot at the measurement time.
        double uav_cumulative_distance;
        double agv_cumulative_distance;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
    
    
    struct State {
        
        rclcpp::Time timestamp; 
        Eigen::Vector4d state; // [x,y,z,yaw]
        double roll;
        double pitch;
        Sophus::SE3d pose; //full pose, with roll and pitch read from imu
        Eigen::Matrix4d covariance;
        bool planar; //to indicate the optimizer whether the motion is planar or not
    
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    };

    // Normalize an angle to the range [-pi, pi]
    inline double normalizeAngle(double angle) {
        while (angle > M_PI)
            angle -= 2.0 * M_PI;
        while (angle < -M_PI)
            angle += 2.0 * M_PI;
        return angle;
    }

    // Build a ROS PoseWithCovarianceStamped message from a Sophus SE3 transformation and a 4x4 covariance.
    inline geometry_msgs::msg::PoseWithCovarianceStamped buildPoseMsg(const Sophus::SE3d &T,
                                                                        const Eigen::Matrix4d& cov4,
                                                                        const rclcpp::Time &current_time,
                                                                        const std::string &frame_id) 
    {
        geometry_msgs::msg::PoseWithCovarianceStamped p_msg;
        p_msg.header.stamp = current_time;
        p_msg.header.frame_id = frame_id;

        // Extract translation and rotation
        Eigen::Vector3d t = T.translation();
        p_msg.pose.pose.position.x = t.x();
        p_msg.pose.pose.position.y = t.y();
        p_msg.pose.pose.position.z = t.z();

        Eigen::Matrix3d R = T.rotationMatrix();
        Eigen::Quaterniond q(R);
        p_msg.pose.pose.orientation.x = q.x();
        p_msg.pose.pose.orientation.y = q.y();
        p_msg.pose.pose.orientation.z = q.z();
        p_msg.pose.pose.orientation.w = q.w();

        // Build a 6x6 covariance matrix from the provided 4x4.
        Eigen::Matrix<double, 6, 6> cov6 = Eigen::Matrix<double, 6, 6>::Zero();
        // Translation covariance
        cov6.block<3, 3>(0, 0) = cov4.block<3, 3>(0, 0);
        // Cross-covariance between translation and yaw.
        cov6.block<3, 1>(0, 5) = cov4.block<3, 1>(0, 3);
        cov6.block<1, 3>(5, 0) = cov4.block<1, 3>(3, 0);
        // Set small variances for roll and pitch.
        cov6(3, 3) = 1e-6;
        cov6(4, 4) = 1e-6;
        // Yaw variance.
        cov6(5, 5) = cov4(3, 3);

        // Flatten cov6 into the message.
        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                p_msg.pose.covariance[i * 6 + j] = cov6(i, j);
            }
        }
        return p_msg;
    }

     // Convert a TransformStamped message into a Sophus SE3 transformation.
    inline Sophus::SE3d transformSE3FromTfMsg(const geometry_msgs::msg::TransformStamped& T_msg) {

        Eigen::Vector3d t(T_msg.transform.translation.x,
                        T_msg.transform.translation.y,
                        T_msg.transform.translation.z);

        Eigen::Quaterniond q(T_msg.transform.rotation.w,
                            T_msg.transform.rotation.x,
                            T_msg.transform.rotation.y,
                            T_msg.transform.rotation.z);

        q.normalize();

        return Sophus::SE3d(q, t);
    }

    inline Sophus::SE3d transformSE3FromPoseMsg(const geometry_msgs::msg::Pose& pose_msg) {

        Eigen::Vector3d t(pose_msg.position.x, 
                        pose_msg.position.y, 
                        pose_msg.position.z);

        Eigen::Quaterniond q(pose_msg.orientation.w, 
                            pose_msg.orientation.x,
                            pose_msg.orientation.y, 
                            pose_msg.orientation.z);
        q.normalize();

        return Sophus::SE3d(q, t);
      }

      // Convert a Sophus SE3 transformation to a 4D state vector [x, y, z, yaw].
    inline Eigen::Vector4d transformSE3ToState(const Sophus::SE3d& T) {
        Eigen::Vector3d t = T.translation();
        Eigen::Matrix3d R = T.rotationMatrix();
        double yaw = std::atan2(R(1,0), R(0,0));
        Eigen::Vector4d state;
        state << t.x(), t.y(), t.z(), yaw;
        return state;
    }

    // Build a Sophus SE3 transformation from roll, pitch and a 4D state vector.
    inline Sophus::SE3d buildTransformationSE3(double roll, double pitch, const Eigen::Vector4d& s) {
        Eigen::Vector3d t(s[0], s[1], s[2]);
        Eigen::Matrix3d R = (Eigen::AngleAxisd(s[3], Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX())).toRotationMatrix();
        return Sophus::SE3d(R, t);
    }

    inline Eigen::Matrix4d reduceCovarianceMatrix(const Eigen::Matrix<double,6,6> &cov6) {
        // Choose indices for [x, y, z, yaw]. Here, we assume that the yaw covariance
        // is stored at row/column 5 in the 6x6 matrix (with the 6-vector being [x,y,z,roll,pitch,yaw]).
        std::vector<int> indices = {0, 1, 2, 5};
        Eigen::Matrix4d cov4;
        for (size_t i = 0; i < indices.size(); ++i) {
            for (size_t j = 0; j < indices.size(); ++j) {
            cov4(i, j) = cov6(indices[i], indices[j]);
            }
        }
        return cov4;
    }

    inline std::vector<UWBMeasurement> getRandomSubset(const std::deque<UWBMeasurement>& measurements, size_t N)
    {
        // Copy measurements into a vector
        std::vector<UWBMeasurement> measurementVec(measurements.begin(), measurements.end());
        
        // Create a random generator
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Shuffle the vector randomly
        std::shuffle(measurementVec.begin(), measurementVec.end(), gen);
        
        // If N is greater than the available measurements, use the full vector
        if (N > measurementVec.size()) {
            N = measurementVec.size();
        }
        
        // Create a subset vector with the first N elements
        std::vector<UWBMeasurement> subset(measurementVec.begin(), measurementVec.begin() + N);
        return subset;
    }


      // Publish a transform using a provided tf2_ros::TransformBroadcaster.
    inline void publishTransform(const Sophus::SE3d& T,
        const std::string &frame_id,
        const std::string &child_frame_id,
        const rclcpp::Time &current_time,
        std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster) 
    {
        geometry_msgs::msg::TransformStamped ts;
        ts.header.stamp = current_time;
        ts.header.frame_id = frame_id;
        ts.child_frame_id = child_frame_id;

        Eigen::Vector3d translation = T.translation();
        ts.transform.translation.x = translation.x();
        ts.transform.translation.y = translation.y();
        ts.transform.translation.z = translation.z();

        Eigen::Matrix3d R = T.rotationMatrix();
        Eigen::Quaterniond q(R);
        ts.transform.rotation.x = q.x();
        ts.transform.rotation.y = q.y();
        ts.transform.rotation.z = q.z();
        ts.transform.rotation.w = q.w();

        tf_broadcaster->sendTransform(ts);
    }

    // Log a 4x4 transformation matrix to the ROS logger.
    inline void logTransformationMatrix(const Eigen::Matrix4d &T, rclcpp::Logger logger) {
        RCLCPP_INFO(logger, "T:\n"
                    "[%f, %f, %f, %f]\n"
                    "[%f, %f, %f, %f]\n"
                    "[%f, %f, %f, %f]\n"
                    "[%f, %f, %f, %f]",
                    T(0, 0), T(0, 1), T(0, 2), T(0, 3),
                    T(1, 0), T(1, 1), T(1, 2), T(1, 3),
                    T(2, 0), T(2, 1), T(2, 2), T(2, 3),
                    T(3, 0), T(3, 1), T(3, 2), T(3, 3));
    }

}

#endif