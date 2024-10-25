#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

import tf2_ros
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import time

class TrajectorySimulator(Node):

    def __init__(self):
        super().__init__('trajectory_simulator')
        self.get_logger().info("Trajectory Simulator Node Started")

        # Parameters
        self.uav_start = np.array([0, 0, 2])  # UAV starts at (0, 0, 2)
        self.ground_start = np.array([0, 0, 0])  # Ground vehicle starts at (0, 0, 0)
        self.steps = 100  # Number of trajectory steps to simulate
        self.smoothness = 10  # Number of interpolation points between key points

        self.publish_rate = 10  # Hz

        # Generate random (smoothed) trajectories
        self.uav_trajectory = self.generate_trajectory(self.uav_start, self.steps, is_uav=True, smooth = True)
        self.get_logger().info('Generated UAV trajectory with %d points' % (len(self.uav_trajectory)))

        self.ground_trajectory = self.generate_trajectory(self.ground_start, self.steps, smooth = True)
        self.get_logger().info('Generated ground vehicle trajectory with %d points' % (len(self.ground_trajectory)))


        # Plot the trajectories
        self.plot_trajectories()

        self.num_points = min(len(self.uav_trajectory), len(self.ground_trajectory))
        self.current_point = 0 #initialize iterator

        # Create a TransformBroadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Create a StaticTransformBroadcaster
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Create tags in UAV frame
        self.create_tag(np.array([0.5, 0.5, 0.25]), 't1')
        self.create_tag(np.array([-0.5, -0.5, 0.25]), 't2')

        # Create anchors in ground vehicle frame
        self.create_anchor(np.array([0.5, 0.5, 0.5]), 'a1')
        self.create_anchor(np.array([-0.5, -0.5, 0.5]), 'a2')
        self.create_anchor(np.array([0.5, -0.5, -0.5]), 'a3')
        self.create_anchor(np.array([-0.5, 0.5, -0.5]), 'a4')


        self.create_timer(1./self.publish_rate, self.transform_publisher)


    def random_curvature(self):
        return random.uniform(-0.5, 0.5)

    def generate_trajectory(self, start, steps, is_uav=False, smooth = False):
        key_points = [start]
        direction = np.array([1, 0, 0])  # Start heading along the x-axis
        curvature = self.random_curvature()

        few_steps = random.randint(2,10)
        j = 0

        for i in range(steps):

            #Change curvature every random number of steps
            if(j > few_steps):
                curvature = self.random_curvature()
                few_steps = random.randint(2,10)
                j = 0
            # Apply curvature to direction (2D rotation in the x-y plane)
            direction = np.dot(np.array([[np.cos(curvature), -np.sin(curvature), 0],
                                         [np.sin(curvature), np.cos(curvature), 0],
                                         [0, 0, 1]]), direction)

            next_point = key_points[-1] + direction

            # if is_uav:
            #     # Add small random change in z for the UAV's altitude
            #     next_point[2] += random.uniform(-0.2, 0.2)

            key_points.append(next_point)
            j += 1

        key_points = np.array(key_points)

        if smooth: 
            key_points = self.smooth_trajectory(key_points)
        
        return key_points
    
    def smooth_trajectory(self, key_points):

        time = np.linspace(0, 1, len(key_points))
        interp_time = np.linspace(0, 1, len(key_points) * self.smoothness)

        x_spline = CubicSpline(time, key_points[:, 0])
        y_spline = CubicSpline(time, key_points[:, 1])
        z_spline = CubicSpline(time, key_points[:, 2])

        smooth_trajectory = np.vstack((x_spline(interp_time), y_spline(interp_time), z_spline(interp_time))).T
        
        return smooth_trajectory


    def plot_trajectories(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot UAV trajectory
        ax.plot(self.uav_trajectory[:, 0], self.uav_trajectory[:, 1], self.uav_trajectory[:, 2], label="UAV", color='blue')

        # Plot Ground Vehicle trajectory (fixed z = 0)
        ax.plot(self.ground_trajectory[:, 0], self.ground_trajectory[:, 1], self.ground_trajectory[:, 2], label="Ground Vehicle", color='red')

        # Axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("UAV and Ground Vehicle Trajectories")
        ax.legend()

        plt.show()

    def transform_publisher(self):

        if(self.current_point < self.num_points):
            
            # Create and send UAV transform
            uav_transform = TransformStamped()
            uav_transform.header.stamp = self.get_clock().now().to_msg()
            uav_transform.header.frame_id = 'world'
            uav_transform.child_frame_id = 'uav'

            uav_transform.transform.translation.x = float(self.uav_trajectory[self.current_point][0])
            uav_transform.transform.translation.y = float(self.uav_trajectory[self.current_point][1])
            uav_transform.transform.translation.z = float(self.uav_trajectory[self.current_point][2])
            uav_transform.transform.rotation.w = 1.0  # No rotation for simplicity

            self.tf_broadcaster.sendTransform(uav_transform)


            # Create and send Ground Vehicle transform
            ground_transform = TransformStamped()
            ground_transform.header.stamp = self.get_clock().now().to_msg()
            ground_transform.header.frame_id = 'world'
            ground_transform.child_frame_id = 'ground_vehicle'

            ground_transform.transform.translation.x = float(self.ground_trajectory[self.current_point][0])
            ground_transform.transform.translation.y = float(self.ground_trajectory[self.current_point][1])
            ground_transform.transform.translation.z = float(self.ground_trajectory[self.current_point][2])  # This should be 0 for ground vehicle
            ground_transform.transform.rotation.w = 1.0  # No rotation for simplicity

            self.tf_broadcaster.sendTransform(ground_transform)

            self.current_point += 1

    def create_tag(self, position, label):

        # Tag transform

        tag_transform = TransformStamped()
        tag_transform.header.stamp = self.get_clock().now().to_msg()
        tag_transform.header.frame_id = 'uav'
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
        anchor_transform.header.frame_id = 'ground_vehicle'
        anchor_transform.child_frame_id = label

        anchor_transform.transform.translation.x = position[0]
        anchor_transform.transform.translation.y = position[1]
        anchor_transform.transform.translation.z = position[2]
        anchor_transform.transform.rotation.w = 1.0  # No rotation for simplicity

        self.tf_static_broadcaster.sendTransform(anchor_transform)




def main(args=None):
    rclpy.init(args=args)

    trajectory_simulator = TrajectorySimulator()

    rclpy.spin(trajectory_simulator)

    trajectory_simulator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
