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

class TrajectorySimulator(Node):

    def __init__(self):
        super().__init__('trajectory_simulator')
        self.get_logger().info("Trajectory Simulator Node Started")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('load_trajectory', False),
                ('same_trajectory', False),
                ('trajectory_name', "trajectory1"),
                ('total_steps', 100),
                ('pub_rate', 10.0),
                ('uav_origin', [0.0, 0.0, 0.0]),
                ('ground_vehicle_origin', [0.0, 0.0, 0.0]),
                ('anchors.a1.position', [0.0, 0.0, 0.0]),
                ('anchors.a2.position', [0.0, 0.0, 0.0]),
                ('anchors.a3.position', [0.0, 0.0, 0.0]),
                ('anchors.a4.position', [0.0, 0.0, 0.0]),
                ('tags.t1.location', [0.0, 0.0, 0.0]),
                ('tags.t2.location', [0.0, 0.0, 0.0])
            ])

        # Parameters

    
        self.steps = self.get_parameter('total_steps').value
        self.meters_per_step = 0.1
        self.smoothness = 10  # Number of interpolation points between key points

        self.publish_rate = self.get_parameter('pub_rate').value

        self.uav_origin = np.array(self.get_parameter('uav_origin').value)
        self.ground_vehicle_origin = np.array(self.get_parameter('ground_vehicle_origin').value)

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


        if(self.load_trajectory == False):

            self.get_logger().info("Generating trajectory" +  self.trajectory_name)

            #Generate similar trajectories
            if self.same_trajectory == True:
                self.ground_trajectory, self.uav_trajectory = self.generate_similar_trajectories(self.ground_vehicle_origin, self.uav_origin, self.steps, True) #True: identical trajectory, False: add some variation

            #Generate different trajectories
            else:
                self.ground_trajectory = self.generate_trajectory(self.ground_vehicle_origin, self.steps)
                self.uav_trajectory = self.generate_trajectory(self.uav_origin, self.steps)

            #Apply cubic spline to trajectories
            self.ground_trajectory = self.smooth_trajectory(self.ground_trajectory, self.smoothness)
            self.get_logger().info('Generated ground vehicle trajectory with %d points' % (len(self.ground_trajectory)))

            self.uav_trajectory = self.smooth_trajectory(self.uav_trajectory, self.smoothness)
            self.get_logger().info('Generated UAV trajectory with %d points' % (len(self.uav_trajectory)))

            #Store and save trajectory
            self.trajectory = np.zeros((len(self.uav_trajectory), 3, 2))
            self.trajectory[:,:,0] = self.uav_trajectory
            self.trajectory[:,:,1] = self.ground_trajectory

            np.save(self.trajectory_name + '.npy', self.trajectory)
        
        else:

            self.get_logger().info("Loading trajectory: " + self.trajectory_name)

            try: 
                self.trajectory = np.load(self.trajectory_name + '.npy')
                self.uav_trajectory = self.trajectory[:,:,0]
                self.ground_trajectory = self.trajectory[:,:,1]
               
                self.get_logger().info('Read ground vehicle trajectory with size ' + str(np.shape(self.ground_trajectory)))
                self.get_logger().info('Read UAV trajectory trajectory with size' + str(np.shape(self.uav_trajectory)))

            except OSError as ex:
                self.get_logger().info(f'Error reading trajectory: {ex}')
                return

        # Plot the trajectories
        self.plot_trajectories()

        self.num_points = min(len(self.uav_trajectory), len(self.ground_trajectory))
        self.get_logger().info('Trajectory length: ' + str(self.num_points))

        self.current_point = 0 #initialize iterator

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


        self.create_timer(1./self.publish_rate, self.transform_publisher)


    def random_curvature(self):

        curvature = random.uniform(-0.25, 0.25)
        steps = random.randint(5,15)
        
        return curvature, steps

    def generate_trajectory(self, start, steps):
        key_points = [start]
           
        direction = np.array([self.meters_per_step, 0, 0])  # Start heading along the x-axis, 10 cm each step
        curvature, few_steps = self.random_curvature()
        j = 0

        for i in range(steps):

            #Change curvature every random number of steps
            if(j > few_steps):
                curvature, few_steps = self.random_curvature()
                j = 0
            # Apply curvature to direction (2D rotation in the x-y plane)
            direction = np.dot(np.array([[np.cos(curvature), -np.sin(curvature), 0],
                                         [np.sin(curvature), np.cos(curvature), 0],
                                         [0, 0, 1]]), direction)

            next_point = key_points[-1] + direction


            key_points.append(next_point)
            j += 1

        key_points = np.array(key_points)
        
        return key_points
    
    def generate_similar_trajectories(self, start_ground, start_uav, steps, identical = False):
        
        key_points = [start_ground]
        key_points_uav = [start_uav]
           
        direction = np.array([self.meters_per_step, 0, 0])  # Start heading along the x-axis, 10 cm each step
        direction_uav = direction
        curvature_ground, few_steps_ground = self.random_curvature()
        
        if identical is False:
            curvature_uav = curvature_ground + np.random.uniform(-0.1, 0.1)
            few_steps_uav = max(1, few_steps_ground + random.randint(-5,5))
            j_uav = 0

        j_ground = 0

        # Initialize drift values
        cumulative_drift = 0.0  # Starts with no drift
        
        for i in range(steps):

            #Change curvature every random number of steps
            if j_ground > few_steps_ground:
                curvature_ground, few_steps_ground = self.random_curvature()
                j_ground = 0

            # Apply curvature to direction (2D rotation in the x-y plane)
            direction = np.dot(np.array([[np.cos(curvature_ground), -np.sin(curvature_ground), 0],
                                         [np.sin(curvature_ground), np.cos(curvature_ground), 0],
                                         [0, 0, 1]]), direction)

            if identical is False:   
                #Slightly change the other trajectory
                if j_uav > few_steps_uav:
                    curvature_uav = curvature_ground + np.random.uniform(-0.1, 0.1)
                    few_steps_uav = max(1, few_steps_ground + random.randint(-5,5))
                    j_uav = 0

                direction_uav = np.dot(np.array([[np.cos(curvature_uav), -np.sin(curvature_uav), 0],
                                            [np.sin(curvature_uav), np.cos(curvature_uav), 0],
                                            [0, 0, 1]]), direction)
                
                j_uav +=1
        
            else:
         
                # Apply incremental drift
                cumulative_drift += 0.002  # Incrementally increase the drift
                yaw_rotation = np.array([
                    [np.cos(cumulative_drift), -np.sin(cumulative_drift), 0],
                    [np.sin(cumulative_drift),  np.cos(cumulative_drift), 0],
                    [0, 0, 1]
                ])

                # Start identical and apply drift gradually
                direction_uav = np.dot(yaw_rotation, direction)

            j_ground += 1

            next_point = key_points[-1] + direction
            next_point_uav = key_points_uav[-1] + direction_uav

            key_points.append(next_point)
            key_points_uav.append(next_point_uav)

        key_points = np.array(key_points)
        key_points_uav = np.array(key_points_uav)
        
        return key_points, key_points_uav
    
    def smooth_trajectory(self, key_points, smoothness):

        time = np.linspace(0, 1, len(key_points))
        interp_time = np.linspace(0, 1, len(key_points) * smoothness)

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

            self.get_logger().info(f'Point {self.current_point}/{self.num_points}', throttle_duration_sec=1)
            
            ### Create and send UAV transform

            uav_transform = TransformStamped()
            uav_transform.header.stamp = self.get_clock().now().to_msg()
            uav_transform.header.frame_id = 'world'
            uav_transform.child_frame_id = 'uav_gt'

            uav_transform.transform.translation.x = float(self.uav_trajectory[self.current_point][0])
            uav_transform.transform.translation.y = float(self.uav_trajectory[self.current_point][1])
            uav_transform.transform.translation.z = float(self.uav_trajectory[self.current_point][2])

            # Add small noise to roll and pitch, keep yaw at 0 for simplicity
            roll = np.deg2rad(np.random.normal(0.0, 1.0))  # ±0.5 degrees of noise
            pitch = np.deg2rad(np.random.normal(0.0, 1.0)) # ±0.5 degrees of noise
            if self.current_point + 10 < len(self.uav_trajectory):
                deltax = float(self.uav_trajectory[self.current_point + 10][0]) - uav_transform.transform.translation.x
                deltay = float(self.uav_trajectory[self.current_point + 10][1]) - uav_transform.transform.translation.y
            else:
                # If not enough points, default yaw to zero or handle gracefully
                deltax, deltay = 1.0, 0.0  # Some default direction
            yaw = np.arctan2(deltay, deltax)  # Yaw

            # Assuming `matrix` is a 3x3 rotation matrix
            rotation = R.from_euler('zyx', [yaw, pitch, roll])
            quat = rotation.as_quat()  # Returns [qx, qy, qz, qw]

            # Assign the quaternion to the transform's rotation
            uav_transform.transform.rotation.x = quat[0]
            uav_transform.transform.rotation.y = quat[1]
            uav_transform.transform.rotation.z = quat[2]
            uav_transform.transform.rotation.w = quat[3]

            self.tf_broadcaster.sendTransform(uav_transform)


            ####  Create and send Ground Vehicle transform
            ground_transform = TransformStamped()
            ground_transform.header.stamp = self.get_clock().now().to_msg()
            ground_transform.header.frame_id = 'world'
            ground_transform.child_frame_id = 'ground_vehicle'

            ground_transform.transform.translation.x = float(self.ground_trajectory[self.current_point][0])
            ground_transform.transform.translation.y = float(self.ground_trajectory[self.current_point][1])
            ground_transform.transform.translation.z = float(self.ground_trajectory[self.current_point][2])  # This should be 0 for ground vehicle


            # Add small noise to roll and pitch, keep yaw at 0 for simplicity
            roll = 0.0  
            pitch = 0.0
            if self.current_point + 10 < len(self.uav_trajectory):
                deltax = float(self.ground_trajectory[self.current_point + 10][0]) - ground_transform.transform.translation.x
                deltay = float(self.ground_trajectory[self.current_point + 10][1]) - ground_transform.transform.translation.y
            else:
                # If not enough points, default yaw to zero or handle gracefully
                deltax, deltay = 1.0, 0.0  # Some default direction
            yaw = np.arctan2(deltay, deltax)  # Yaw

            # Assuming `matrix` is a 3x3 rotation matrix
            rotation = R.from_euler('zyx', [yaw, pitch, roll])
            quat = rotation.as_quat()  # Returns [qx, qy, qz, qw]

            # Assign the quaternion to the transform's rotation
            ground_transform.transform.rotation.x = quat[0]
            ground_transform.transform.rotation.y = quat[1]
            ground_transform.transform.rotation.z = quat[2]
            ground_transform.transform.rotation.w = quat[3]

            self.tf_broadcaster.sendTransform(ground_transform)

            self.current_point += 1
        
        else:
            
            self.get_logger().warning('Trajectory Finished')


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
