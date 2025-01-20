#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

import tf_transformations
import tf2_ros
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import time
import os

class OdometrySimulator(Node):

    def __init__(self):
        super().__init__('odometry_simulator')
        self.get_logger().info("Odometry Simulator Node Started")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('load_trajectory', False),
                ('same_trajectory', False),
                ('trajectory_name', "trajectory1"),
                ('total_distance', 100),
                ('pub_rate', 10.0),
                ('uav_origin', [0.0, 0.0, 0.0, 0.0]),
                ('ground_vehicle_origin', [0.0, 0.0, 0.0, 0.0]),
                ('linear_velocity_range', [0.0, 0.0]),
                ('angular_velocity_range', [0.0, 0.0]),
                ('odom_error', 2.0),
                ('anchors.a1.position', [0.0, 0.0, 0.0]),
                ('anchors.a2.position', [0.0, 0.0, 0.0]),
                ('anchors.a3.position', [0.0, 0.0, 0.0]),
                ('anchors.a4.position', [0.0, 0.0, 0.0]),
                ('tags.t1.location', [0.0, 0.0, 0.0]),
                ('tags.t2.location', [0.0, 0.0, 0.0])
            ])

        # Parameters
    
        self.total_distance = self.get_parameter('total_distance').value

        self.traveled_distance_uav = self.traveled_distance_agv = 0.0

        self.publish_rate = self.get_parameter('pub_rate').value

        # Initialize odometry to starting poses
        self.uav_pose = np.array(self.get_parameter('uav_origin').value)
        self.agv_pose = np.array(self.get_parameter('ground_vehicle_origin').value)

        self.odom_error = self.get_parameter('odom_error').value

        self.uav_odom_pose = self.uav_pose
        self.agv_odom_pose = self.agv_pose

        self.linear_velocity_range = self.get_parameter('linear_velocity_range').value
        self.angular_velocity_range = self.get_parameter('angular_velocity_range').value

        # Retrieve anchor and tag locations and convert them to numpy arrays
        anchors_location = {
            "a1": np.array(self.get_parameter('anchors.a1.position').value),
            "a2": np.array(self.get_parameter('anchors.a2.position').value),
            "a3": np.array(self.get_parameter('anchors.a3.position').value),
            "a4": np.array(self.get_parameter('anchors.a4.position').value),
        }

        tags_location = {
            "t1": np.array(self.get_parameter('tags.t1.location').value),
            "t2": np.array(self.get_parameter('tags.t2.location').value),
        }

        self.load_trajectory = self.get_parameter('load_trajectory').value
        self.same_trajectory = self.get_parameter('same_trajectory').value
        self.trajectory_name = self.get_parameter('trajectory_name').value


        self.get_logger().info("Load trajectory is: " +  str(self.load_trajectory))

        if self.load_trajectory:
            self.get_logger().info(f"Loading trajectories from {self.trajectory_name}")
            try:
                data = np.load(self.trajectory_name + '.npy', allow_pickle=True).item()
                self.uav_velocity_commands = data['uav']
                self.agv_velocity_commands = data['agv']
            except Exception as e:
                self.get_logger().error(f"Failed to load trajectories: {e}")
                return

        else:
            self.get_logger().info("Generating new trajectories")
            self.uav_velocity_commands, self.agv_velocity_commands = self.generate_velocity_commands()
            self.save_trajectories()

        
        # Plot the trajectories
        self.plot_trajectories()

        self.num_points = min(len(self.uav_velocity_commands), len(self.agv_velocity_commands))
        self.get_logger().info('Trajectory length: ' + str(self.num_points))

        # Create a TransformBroadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Create a StaticTransformBroadcaster
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Create tags in UAV frame
        self.create_tag(tags_location['t1'], 't1')
        self.create_tag(tags_location['t2'], 't2')

        # Create anchors in ground vehicle frame
        self.create_anchor(anchors_location['a1'], 'a1')
        self.create_anchor(anchors_location['a2'], 'a2')
        self.create_anchor(anchors_location['a3'], 'a3')
        self.create_anchor(anchors_location['a4'], 'a4')

        self.create_timer(1./self.publish_rate, self.update_odometry)


    def generate_velocity_commands(self):
        """Generate random linear and angular velocity commands."""
        uav_commands = []
        agv_commands = []
        distance_covered = 0.0
        
        uav_last_v = uav_last_w = 0.0
        agv_last_v = agv_last_w = 0.0

        temp_uav_pose = self.uav_pose.copy()
        temp_uav_odom_pose = self.uav_odom_pose.copy()
        temp_agv_pose = self.agv_pose.copy()
        temp_agv_odom_pose = self.agv_odom_pose.copy()

        temp_traveled_distance_uav = temp_traveled_distance_agv = 0.0

        while distance_covered < self.total_distance:


            agv_v = np.clip(agv_last_v + np.random.uniform(-0.05, 0.05), *self.linear_velocity_range)
            agv_w = np.clip(agv_last_w + np.random.uniform(-0.01, 0.01), *self.angular_velocity_range)

            #Uncomment this to make them follow similar trajectories
            uav_last_v = agv_last_v
            uav_last_w = agv_last_w

            # Smooth velocity changes
            uav_v = np.clip(uav_last_v + np.random.uniform(-0.05, 0.05), *self.linear_velocity_range)
            uav_w = np.clip(uav_last_w + np.random.uniform(-0.01, 0.01), *self.angular_velocity_range)

            dt = 1.0 / self.publish_rate
            
            # Ensure UAV and AGV stay within a reasonable distance
            if len(uav_commands) > 0 and len(agv_commands) > 0:
                temp_traveled_distance_uav += uav_v * dt
                temp_traveled_distance_agv += agv_v * dt

                temp_uav_pose, temp_uav_odom_pose = self.integrate_odometry(temp_uav_pose, temp_uav_odom_pose, uav_commands[-1][0], uav_commands[-1][1], dt, temp_traveled_distance_uav)
                temp_agv_pose, temp_agv_odom_pose = self.integrate_odometry(temp_agv_pose, temp_agv_odom_pose, agv_commands[-1][0], agv_commands[-1][1], dt, temp_traveled_distance_agv)
                if np.linalg.norm(temp_uav_pose[:2] - temp_agv_pose[:2]) > 8.0:
                    agv_v = uav_v  # Align AGV velocity with UAV
                    agv_w = uav_w

            
            #self.get_logger().warning(f'[Vel commands]Distance traveled AGV: {temp_traveled_distance_agv}')
            
            uav_commands.append((uav_v, uav_w))
            agv_commands.append((agv_v, agv_w))

            uav_last_v, uav_last_w = uav_v, uav_w
            agv_last_v, agv_last_w = agv_v, agv_w

            distance_increment = (uav_v + agv_v) / 2 / self.publish_rate

            distance_covered += distance_increment
            
        return uav_commands, agv_commands

    

    def update_odometry(self):
        """Update UAV and AGV poses based on velocity commands."""
        if len(self.uav_velocity_commands) == 0 or len(self.agv_velocity_commands) == 0:
            self.get_logger().warning("Trajectory completed")
            return

        dt = 1.0 / self.publish_rate 

        # Update UAV pose - GT and Odometry
        v_uav, w_uav = self.uav_velocity_commands.pop(0)
        self.traveled_distance_uav += v_uav * dt
        self.uav_pose, self.uav_odom_pose = self.integrate_odometry(self.uav_pose, self.uav_odom_pose, v_uav, w_uav, dt, self.traveled_distance_uav)

        # Update AGV pose - GT and Odometry
        v_agv, w_agv = self.agv_velocity_commands.pop(0)
        self.traveled_distance_agv += v_agv * dt
        self.agv_pose, self.agv_odom_pose = self.integrate_odometry(self.agv_pose, self.agv_odom_pose, v_agv, w_agv, dt, self.traveled_distance_agv)

        self.get_logger().info(f'Traveled distance AGV: {self.traveled_distance_agv:.2f}')

        # Publish transforms
        self.transform_publisher(self.uav_pose, 'world', 'uav_gt')
        self.transform_publisher(self.agv_pose, 'world', 'agv_gt')

        self.transform_publisher(self.uav_odom_pose, 'world', 'uav_odom')
        self.transform_publisher(self.agv_odom_pose, 'world', 'agv_odom')

    def integrate_odometry(self, pose_gt, pose_odom, v, w, dt, traveled_distance):
        """Integrate odometry using simple kinematic equations."""
        x, y, z, theta = pose_gt
        xp, yp, zp, thetap = pose_odom
        
        # Error proportional to the traveled distance
        error_distance = traveled_distance * self.odom_error / 100.0

        if abs(w) > 1e-6:
            # Arc-based motion model
            r = v / w
            dx = r * (np.sin(theta + w * dt) - np.sin(theta))
            dy = r * (np.cos(theta) - np.cos(theta + w * dt))

        else:
            # Straight-line motion
            dx = v * dt * np.cos(theta)
            dy = v * dt * np.sin(theta)

        dxp = dx + error_distance * np.cos(theta)
        dyp = dy + error_distance * np.sin(theta)

        dtheta = w * dt
        
        dz = 0.0
        dzp = 0.0

        pose_gt = np.array([x + dx, y + dy, z + dz, theta + dtheta])
        pose_odom = np.array([x + dxp, y + dyp, z + dzp, theta + dtheta])

        return pose_gt, pose_odom


    def plot_trajectories(self):
        """Plot UAV and AGV trajectories based on velocity commands."""
        uav_positions_gt = [self.uav_pose[:3]]
        agv_positions_gt = [self.agv_pose[:3]]
        uav_positions_odom = [self.uav_odom_pose[:3]]
        agv_positions_odom = [self.agv_odom_pose[:3]]

        temp_uav_pose = self.uav_pose.copy()
        temp_uav_odom_pose = self.uav_odom_pose.copy()
        temp_agv_pose = self.agv_pose.copy()
        temp_agv_odom_pose = self.agv_odom_pose.copy()

        dt = 1.0 / self.publish_rate
        temp_traveled_distance_uav = temp_traveled_distance_agv = 0.0


        for v_uav, w_uav in self.uav_velocity_commands:
            temp_traveled_distance_uav += v_uav * dt
            temp_uav_pose, temp_uav_odom_pose = self.integrate_odometry(temp_uav_pose, temp_uav_odom_pose, v_uav, w_uav, dt, temp_traveled_distance_uav)
            uav_positions_gt.append(temp_uav_pose[:3])
            uav_positions_odom.append(temp_uav_odom_pose[:3])

        for v_agv, w_agv in self.agv_velocity_commands:
            temp_traveled_distance_agv += v_agv * dt
            temp_agv_pose, temp_agv_odom_pose = self.integrate_odometry(temp_agv_pose, temp_agv_odom_pose, v_agv, w_agv, dt, temp_traveled_distance_agv)
            agv_positions_gt.append(temp_agv_pose[:3])
            agv_positions_odom.append(temp_agv_odom_pose[:3])

        uav_positions_gt = np.array(uav_positions_gt)
        uav_positions_odom = np.array(uav_positions_odom)
        agv_positions_gt = np.array(agv_positions_gt)
        agv_positions_odom = np.array(agv_positions_odom)


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(uav_positions_gt[:, 0], uav_positions_gt[:, 1], uav_positions_gt[:, 2], label='UAV GT', color='blue')
        ax.plot(uav_positions_odom[:, 0], uav_positions_odom[:, 1], uav_positions_odom[:, 2], label='UAV Odom 2%', color='blue', linestyle='--')

        ax.plot(agv_positions_gt[:, 0], agv_positions_gt[:, 1], agv_positions_gt[:, 2], label='AGV GT', color='red')
        ax.plot(agv_positions_odom[:, 0], agv_positions_odom[:, 1], agv_positions_odom[:, 2], label='AGV Odom 2%', color='red', linestyle='--')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')

        ax.set_title('UAV and AGV Trajectories')
        ax.legend()

        plt.show()

    def transform_publisher(self, pose, parent_frame, child_frame):
        """Publish the transform for a given pose."""
        x, y, z, theta = pose
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame

        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z

        quat = R.from_euler('z', theta).as_quat()
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(transform)

    def create_tag(self, position, label):

        # Tag transform

        tag_transform = TransformStamped()
        tag_transform.header.stamp = self.get_clock().now().to_msg()
        tag_transform.header.frame_id = 'uav_gt'
        tag_transform.child_frame_id = label

        tag_transform.transform.translation.x = position[0]
        tag_transform.transform.translation.y = position[1]
        tag_transform.transform.translation.z = position[2]
        tag_transform.transform.rotation.w = 1.0  # No rotation for simplicity

        self.tf_static_broadcaster.sendTransform(tag_transform)

    def create_anchor(self, position, label):

        # Tag transform

        anchor_transform = TransformStamped()
        anchor_transform.header.stamp = self.get_clock().now().to_msg()
        anchor_transform.header.frame_id = 'agv_gt'
        anchor_transform.child_frame_id = label

        anchor_transform.transform.translation.x = position[0]
        anchor_transform.transform.translation.y = position[1]
        anchor_transform.transform.translation.z = position[2]
        anchor_transform.transform.rotation.w = 1.0  # No rotation for simplicity

        self.tf_static_broadcaster.sendTransform(anchor_transform)


    def save_trajectories(self):
            """Save generated trajectories to a file."""
            data = {
                'uav': self.uav_velocity_commands,
                'agv': self.agv_velocity_commands
            }
            np.save(self.trajectory_name, data)
            self.get_logger().info(f"Trajectories saved to {self.trajectory_name}")



def main(args=None):
    rclpy.init(args=args)

    odometry_simulator = OdometrySimulator()

    rclpy.spin(odometry_simulator)

    odometry_simulator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
