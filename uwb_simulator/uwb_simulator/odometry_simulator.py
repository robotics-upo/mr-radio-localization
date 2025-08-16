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
from nav_msgs.msg import Odometry
import time
import os

class OdometrySimulator(Node):

    def __init__(self):
        super().__init__('odometry_simulator')
        self.get_logger().info("Odometry Simulator Node Started")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('use_mission', False),
                ('holonomic_xy', False),
                ('trajectory_name', "trajectory1"),
                ('total_distance', 100),
                ('pub_rate', 10.0),
                ('uav_origin', [0.0, 0.0, 0.0, 0.0]),
                ('ground_vehicle_origin', [0.0, 0.0, 0.0, 0.0]),
                ('linear_velocity_range', [0.0, 0.0]),
                ('max_linear_acceleration', 0.0),
                ('angular_velocity_range', [0.0, 0.0]),
                ('max_angular_acceleration', 0.0),
                ('odom_error_position', 0.0),
                ('odom_error_angle', 0.0),
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
        self.traveled_angle_uav = self.traveled_angle_agv = 0.0

        self.publish_rate = self.get_parameter('pub_rate').value

        # Initialize odometry to starting poses
        self.uav_origin = np.array(self.get_parameter('uav_origin').value)
        self.agv_origin = np.array(self.get_parameter('ground_vehicle_origin').value)

        self.odom_error_position = self.get_parameter('odom_error_position').value
        self.odom_error_angle = self.get_parameter('odom_error_angle').value

        self.uav_pose = self.uav_origin
        self.agv_pose = self.agv_origin
        self.uav_odom_pose = np.array([0.0, 0.0, 0.0, 0.0])
        self.agv_odom_pose = np.array([0.0, 0.0, 0.0, 0.0])

        self.linear_velocity_range = self.get_parameter('linear_velocity_range').value
        self.max_linear_acceleration = self.get_parameter('max_linear_acceleration').value
        self.angular_velocity_range = self.get_parameter('angular_velocity_range').value
        self.max_angular_acceleration = self.get_parameter('max_angular_acceleration').value

        self.holonomic_xy = self.get_parameter('holonomic_xy').value

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

        self.trajectory_name = self.get_parameter('trajectory_name').value
        self.mission = self.get_parameter('use_mission').value

        self.get_logger().info(f"Loading trajectory: {self.trajectory_name}")

        try:
            data = np.load(self.trajectory_name + '.npy', allow_pickle=True).item()
            self.uav_velocity_commands = data['uav']
            self.agv_velocity_commands = data['agv']
        except Exception as e:
            self.get_logger().error(f"Trajectory not found: {e}")
            self.get_logger().info("Generating new trajectories")
            if self.mission: self.uav_velocity_commands, self.agv_velocity_commands = self.generate_mission_velocity_commands()
            else: self.uav_velocity_commands, self.agv_velocity_commands = self.generate_random_velocity_commands()
            
            self.save_trajectories()
            # self.save_trajectory_positions()
        
        self.save_trajectory_positions()

        # Plot the trajectories
        self.plot_trajectories()

        self.num_points = min(len(self.uav_velocity_commands), len(self.agv_velocity_commands))
        self.get_logger().info('Trajectory length: ' + str(self.num_points))

        # Create a TransformBroadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Create a StaticTransformBroadcaster
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Initialize publishers for UAV and AGV odometry
        self.uav_odom_publisher = self.create_publisher(Odometry, '/uav/odom', 10)
        self.agv_odom_publisher = self.create_publisher(Odometry, '/agv/odom', 10)

        self.uav_cumulative_covariance = [0.0] * 36  # 6x6 matrix in row-major order
        self.agv_cumulative_covariance = [0.0] * 36  # 6x6 matrix in row-major order

        #create odom frames
        self.create_odom_frame(self.agv_origin, 'agv/odom')
        self.create_odom_frame(self.uav_origin, 'uav/odom')

        # Create tags in UAV frame
        self.create_tag(tags_location['t1'], 't1')
        self.create_tag(tags_location['t2'], 't2')

        # Create anchors in ground vehicle frame
        self.create_anchor(anchors_location['a1'], 'a1')
        self.create_anchor(anchors_location['a2'], 'a2')
        self.create_anchor(anchors_location['a3'], 'a3')
        self.create_anchor(anchors_location['a4'], 'a4')

        self.create_timer(1./self.publish_rate, self.update_odometry)

    
    def generate_mission_velocity_commands(self):
        """Generate velocity commands for simple predefined missions."""
        uav_commands = []
        agv_commands = []

        # === Select the mission here ===
        use_custom_mission = True 
        use_circular_trajectory    = False
        use_lemniscate             = False

        dt = 1.0 / self.publish_rate

        # ---------- Helpers to hit exact distances/angles ----------
        def append_straight(cmds, distance, v):
            """Append forward straight motion of 'distance' at speed v, exact to last step."""
            if distance <= 0.0 or v <= 0.0:
                return
            steps_full = int(np.floor(distance / (v * dt)))
            for _ in range(steps_full):
                cmds.append((v, 0.0, 0.0))
            rem = distance - steps_full * v * dt
            if rem > 1e-9:
                cmds.append((rem / dt, 0.0, 0.0))  # scale last step to finish exactly

        def append_turn(cmds, angle, w_abs):
            """Append pure rotation by 'angle' radians (CCW positive), exact to last step."""
            if w_abs <= 0.0 or abs(angle) <= 0.0:
                return
            sgn = 1.0 if angle >= 0.0 else -1.0
            w = sgn * w_abs
            steps_full = int(np.floor(abs(angle) / (w_abs * dt)))
            for _ in range(steps_full):
                cmds.append((0.0, 0.0, w))
            rem = abs(angle) - steps_full * w_abs * dt
            if rem > 1e-9:
                cmds.append((0.0, 0.0, sgn * rem / dt))

        def append_circle_laps(cmds, radius, v, laps, ccw=True):
            """Append constant-curvature motion for 'laps' of a circle with given radius."""
            if radius <= 0.0 or v <= 0.0 or laps <= 0:
                return
            sgn = 1.0 if ccw else -1.0
            w = sgn * (v / radius)                     # keep curvature r = v / w = radius
            total_angle = laps * 2.0 * np.pi
            steps_full = int(np.floor(total_angle / (abs(w) * dt)))
            for _ in range(steps_full):
                cmds.append((v, 0.0, w))
            rem = total_angle - steps_full * abs(w) * dt
            if rem > 1e-9:
                alpha = rem / (abs(w) * dt)            # scale v,w for the last partial step
                cmds.append((v * alpha, 0.0, w * alpha))

        if use_custom_mission:
            # --- UAV: rectangular loop (5m x 20m) ---
            rect_width  = 5.0     # meters
            rect_length = 20.0    # meters
            v_lin       = 0.5     # m/s straight speed
            w_turn_abs  = np.pi/2 / 2.0   # 90° turns in ~2 s (pure rotation)

            half_W = rect_width * 0.5
            half_L = rect_length * 0.5

            # A) center -> lateral (move +Y by half width)
            append_turn(uav_commands,  np.pi/2, w_turn_abs)   # face +Y
            append_straight(uav_commands, half_W, v_lin)      # go to mid of +Y side
            append_turn(uav_commands, -np.pi/2, w_turn_abs)   # face +X again

            # B) perimeter sequence (clockwise), starting at mid +Y side, heading +X
            # 1)  half length  (+X)
            append_straight(uav_commands, half_L, v_lin)

            # 2)  full width   (-Y)
            append_turn(uav_commands, -np.pi/2, w_turn_abs)   # face -Y
            append_straight(uav_commands, rect_width, v_lin)

            # 3)  full length  (-X)
            append_turn(uav_commands, -np.pi/2, w_turn_abs)   # face -X
            append_straight(uav_commands, rect_length, v_lin)

            # 4)  full width   (+Y)
            append_turn(uav_commands, -np.pi/2, w_turn_abs)   # face +Y
            append_straight(uav_commands, rect_width, v_lin)

            # 5)  half length  (+X) back to mid +Y side
            append_turn(uav_commands, -np.pi/2, w_turn_abs)   # face +X
            append_straight(uav_commands, half_L, v_lin)

            # C) lateral -> center (move -Y by half width)
            append_turn(uav_commands, -np.pi/2, w_turn_abs)   # face -Y
            append_straight(uav_commands, half_W, v_lin)      # back to center
            append_turn(uav_commands,  np.pi/2, w_turn_abs)   # restore heading +X

            # --- AGV: from center -> radius 4m -> 3 laps -> back to center ---
            R         = 3.0       # meters
            v_lin = 0.5  # m/s straight speed
            laps      = 3

            # Start at center, heading = 0. Go to (R, 0), turn to +90°, do 3 CCW laps, undo turn, return.
            append_straight(agv_commands, R, v_lin)              # center -> (R,0), heading 0
            append_turn(agv_commands, np.pi/2, np.pi/2/2.0)      # ~2 s, face +Y so ICC = origin
            append_circle_laps(agv_commands, R, v_lin, laps, ccw=True)
            append_turn(agv_commands, -np.pi/2, np.pi/2/2.0)     # back to heading 0 at (R,0)
            append_straight(agv_commands, R, v_lin)              # (R,0) -> center

            self.get_logger().info(f"Generated {len(uav_commands)} commands for UAV and {len(agv_commands)} for AGV.")
            return uav_commands, agv_commands

        if use_circular_trajectory:
            self.get_logger().info(f"Generating circular trajectory!.")
            radius = 2.0
            angular_velocity = 0.5
            average_vel = radius * angular_velocity
            duration = self.total_distance / average_vel
            total_steps = int(duration * self.publish_rate)
            for _ in range(total_steps):
                v = radius * angular_velocity
                w = angular_velocity
                agv_commands.append((v, 0.0, w))
                uav_commands.append((v, 0.0, w))
            self.get_logger().info(f"Generated {len(uav_commands)} commands for UAV and AGV.")
            return uav_commands, agv_commands

        if use_lemniscate:
            R = 5.0
            thetas = np.linspace(0, 2*np.pi, 4_000)
            dx_dθ =  R * np.cos(thetas)
            dy_dθ =  R * np.cos(2*thetas)
            L = np.trapz(np.hypot(dx_dθ, dy_dθ), thetas)
            v_avg = 0.5
            T_loop = L / v_avg
            N_steps = int(np.ceil(T_loop * self.publish_rate))
            N_loops = int(np.ceil(self.total_distance / L))
            for _ in range(N_loops):
                for i in range(N_steps):
                    theta = 2.0 * np.pi * (i / N_steps)
                    dx =  R * np.cos(theta) * (2.0*np.pi/T_loop)
                    dy =  R * np.cos(2*theta)* (2.0*np.pi/T_loop)
                    ddx = -R * np.sin(theta) * (2.0*np.pi/T_loop)**2.0
                    ddy = -2*R*np.sin(2*theta)*(2.0*np.pi/T_loop)**2.0
                    v = np.hypot(dx, dy)
                    kappa = (dx*ddy - dy*ddx) / (v**3.0 + 1e-8)
                    angular_vel = kappa * v
                    agv_commands.append((v, 0.0, angular_vel))
                    uav_commands.append((v, 0.0, angular_vel))
            self.get_logger().info(f"Generated {len(uav_commands)} commands for UAV and AGV.")
            return uav_commands, agv_commands

        # Fallback in case nothing selected
        self.get_logger().warn("No mission selected; defaulting to empty command lists.")
        return uav_commands, agv_commands


    def generate_random_velocity_commands(self):
        """Generate random linear and angular velocity commands."""
        uav_commands = []
        agv_commands = []
        distance_covered = 0.0
        
        uav_last_v = np.array([0.0, 0.0])  # Linear velocity [vx, vy]
        uav_last_w = 0.0  # Angular velocity
        agv_last_v = np.array([0.0, 0.0])  # Linear velocity [vx, vy]
        agv_last_w = 0.0  # Angular velocity

        max_dacc_v = self.max_linear_acceleration / self.publish_rate
        max_dacc_w = self.max_angular_acceleration / self.publish_rate

        while distance_covered < self.total_distance:
            
            agv_v = np.clip(agv_last_v + np.random.uniform(-max_dacc_v, max_dacc_v, size=2), *self.linear_velocity_range)

            if self.holonomic_xy is False:

                agv_w = np.clip(agv_last_w + np.random.uniform(-max_dacc_w, max_dacc_w), *self.angular_velocity_range)

            
            else: #independent velocity commands
                
                #Change angular motion independently
                if distance_covered < 10.0: 
                    agv_w = np.clip(0.1, *self.angular_velocity_range)
                elif distance_covered < 25.0: 
                    agv_w = np.clip(-0.1, *self.angular_velocity_range)
                elif distance_covered < 50.0: 
                    agv_w = np.clip(0.0, *self.angular_velocity_range)
                elif distance_covered < 75.0: 
                    agv_w = np.clip(-0.25, *self.angular_velocity_range)
                else: 
                    agv_w = np.clip(0.25, *self.angular_velocity_range)
                
            #Uncomment this to make them follow similar trajectories
            uav_last_v = agv_last_v
            uav_last_w = agv_last_w

            # Smooth velocity changes
            uav_v = np.clip(uav_last_v + np.random.uniform(-max_dacc_v, max_dacc_v, size=2), *self.linear_velocity_range)
            uav_w = np.clip(uav_last_w + np.random.uniform(-max_dacc_w, max_dacc_w), *self.angular_velocity_range)

            # uav_v = agv_v
            # uav_w = agv_w
            
            uav_commands.append((*uav_v, uav_w))
            agv_commands.append((*agv_v, agv_w))

            self.get_logger().info(f"UAV_V: {uav_v}, UAV_W: {uav_w}.")

            uav_last_v = uav_v
            uav_last_w = uav_w
            agv_last_v = agv_v
            agv_last_w = agv_w

            distance_increment = (np.linalg.norm(uav_v) + np.linalg.norm(agv_v)) / 2 / self.publish_rate

            distance_covered += distance_increment
            
        return uav_commands, agv_commands

    

    def update_odometry(self):
        """Update UAV and AGV poses based on velocity commands."""
        if len(self.uav_velocity_commands) == 0 or len(self.agv_velocity_commands) == 0:
            self.get_logger().warning("Trajectory completed")
            return

        dt = 1.0 / self.publish_rate 

        # Update UAV pose - GT and Odometry
        uav_command = self.uav_velocity_commands.pop(0)
        v_uav = np.array(uav_command[:2])  # Extract [v_x, v_y]
        w_uav = uav_command[2]  # Extract angular velocity
    
        incremental_distance_uav = np.linalg.norm(v_uav) * dt
        incremental_angle_uav = abs(w_uav) * dt
        self.traveled_distance_uav += incremental_distance_uav
        self.traveled_angle_uav += np.rad2deg(incremental_angle_uav)
        self.uav_pose, self.uav_odom_pose = self.integrate_odometry(self.uav_pose, v_uav, w_uav, dt, self.traveled_distance_uav, self.traveled_angle_uav, self.uav_origin, self.holonomic_xy)

        # Update AGV pose - GT and Odometry
        agv_command = self.agv_velocity_commands.pop(0)
        v_agv = np.array(agv_command[:2])  # Extract [v_x, v_y]
        w_agv = agv_command[2]  # Extract angular velocity

        incremental_distance_agv = np.linalg.norm(v_agv) * dt
        incremental_angle_agv = abs(w_agv) * dt
        self.traveled_distance_agv += incremental_distance_agv
        self.traveled_angle_agv += np.rad2deg(incremental_angle_agv)
        self.agv_pose, self.agv_odom_pose = self.integrate_odometry(self.agv_pose, v_agv, w_agv, dt, self.traveled_distance_agv, self.traveled_angle_agv, self.agv_origin, self.holonomic_xy)
        
        #self.get_logger().info(f'Traveled distance AGV: {self.traveled_distance_agv:.2f}', throttle_duration_sec=1)
        # self.get_logger().info(f'AGV Odom Pose: {self.agv_odom_pose}', throttle_duration_sec=1)
        # self.get_logger().info(f'UAV Odom Pose: {self.uav_odom_pose}', throttle_duration_sec=1)
        # self.get_logger().info(f'AGV GT Pose: {self.agv_pose}', throttle_duration_sec=1)
        # self.get_logger().info(f'UAV GT Pose: {self.uav_pose}', throttle_duration_sec=1)

        # Publish transforms (ground truth)
        self.transform_publisher(self.uav_pose, 'world', 'uav_gt')
        self.transform_publisher(self.agv_pose, 'world', 'agv_gt')

        # Publish odometry
        self.transform_publisher(self.uav_odom_pose, 'uav/odom', 'uav/base_link')
        self.transform_publisher(self.agv_odom_pose, 'agv/odom', 'agv/base_link')

        # Publish odometry messages for UAV and AGV
        self.publish_odometry(
            self.uav_odom_pose, v_uav, w_uav, dt, self.traveled_distance_uav, self.traveled_angle_uav, 'uav/odom', 'uav/base_link', '/uav/odom',
            cumulative_covariance=self.uav_cumulative_covariance
        )
        self.publish_odometry(
            self.agv_odom_pose, v_agv, w_agv, dt, self.traveled_distance_agv, self.traveled_angle_agv, 'agv/odom', 'agv/base_link', '/agv/odom',
            cumulative_covariance=self.agv_cumulative_covariance
        )

    def integrate_odometry(self, pose, v, w, dt, traveled_distance, traveled_angle, origin, holonomic = True):
        """Integrate odometry using simple kinematic equations."""
        x, y, z, theta = pose
    
        if holonomic is True:
            # Independent motion in x, y, and yaw (holonomic)
            dx = v[0] * dt
            dy = v[1] * dt
            dtheta = w * dt
        else:
            # Unicycle model: v[0] is forward speed, ignore v[1]
            v_fwd = float(v[0])
            if abs(w) < 1e-6:
                # Straight line
                dx = v_fwd * dt * np.cos(theta)
                dy = v_fwd * dt * np.sin(theta)
            else:
                # Constant curvature arc
                dx = (v_fwd / w) * (np.sin(theta + w * dt) - np.sin(theta))
                dy = (v_fwd / w) * (np.cos(theta) - np.cos(theta + w * dt))
            
            dtheta = w * dt
        
        dz = 0.0

        updated_pose = np.array([x + dx, y + dy, z + dz, self.normalize_angle(theta + dtheta)])

        # Error delta based on increments (independent, non-cumulative)
        delta_distance = np.linalg.norm(v)*dt
        delta_angle = abs(w) * dt

        #Systematic error based on total distance traveled (cumulative)
        error_distance_sys = traveled_distance * self.odom_error_position / 100.0
        error_angle_sys = np.deg2rad(traveled_angle * self.odom_error_angle / 100.0)
        error_distance_delta = delta_distance * self.odom_error_position / 100.0
        error_angle_delta = np.deg2rad(delta_angle * self.odom_error_angle / 100.0)  # Convert to radians
        
        # Odometry error
        error_distance = np.random.normal(error_distance_sys, error_distance_delta)
        error_angle = np.random.normal(error_angle_sys, error_angle_delta)

        dx = updated_pose[0] - origin[0]
        dy = updated_pose[1] - origin[1]
        x_ideal = np.cos(origin[3]) * dx + np.sin(origin[3]) * dy
        y_ideal = -np.sin(origin[3]) * dx + np.cos(origin[3]) * dy
        theta_ideal = self.normalize_angle(updated_pose[3] - origin[3])
        z_ideal = updated_pose[2] - origin[2]

        x_noisy = x_ideal + error_distance * np.cos(theta_ideal)
        y_noisy = y_ideal + error_distance * np.sin(theta_ideal)
        theta_noisy = self.normalize_angle(theta_ideal + error_angle)
        z_noisy = z_ideal

        updated_pose_odom = np.array([x_noisy, y_noisy, z_noisy, theta_noisy])

        return updated_pose, updated_pose_odom
    
    def normalize_angle(self,theta):
        return np.arctan2(np.sin(theta), np.cos(theta))

    def publish_odometry(self, pose, linear_velocity, angular_velocity, dt, traveled_distance, traveled_angle, frame_id, child_frame_id, topic, cumulative_covariance):
        """Publish an odometry message for the given pose."""
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = frame_id
        odom_msg.child_frame_id = child_frame_id

        # Set position and orientation
        odom_msg.pose.pose.position.x = pose[0]
        odom_msg.pose.pose.position.y = pose[1]
        odom_msg.pose.pose.position.z = pose[2]

        # sample noise
        roll_noise  = random.gauss(0.0, 0.01)
        pitch_noise = random.gauss(0.0, 0.01)

         # orientation w/ noise on roll & pitch
        noisy_quat = R.from_euler('xyz', [roll_noise, pitch_noise, pose[3]]).as_quat()
        odom_msg.pose.pose.orientation.x = noisy_quat[0]
        odom_msg.pose.pose.orientation.y = noisy_quat[1]
        odom_msg.pose.pose.orientation.z = noisy_quat[2]
        odom_msg.pose.pose.orientation.w = noisy_quat[3]

        # Set linear and angular velocity
        odom_msg.twist.twist.linear.x = linear_velocity[0]
        odom_msg.twist.twist.linear.y = linear_velocity[1]
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.z = angular_velocity

        # Calculate systematic error covariance (cumulative drift)
        systematic_position_variance = (traveled_distance * self.odom_error_position / 100.0) ** 2
        systematic_angle_variance = np.deg2rad(traveled_angle * self.odom_error_angle / 100.0) ** 2

        # Calculate incremental covariance based on traveled distance
        delta_distance = np.linalg.norm(linear_velocity) * dt
        delta_angle = angular_velocity * dt
        incremental_position_variance = (delta_distance * self.odom_error_position / 100.0) ** 2
        incremental_angle_variance = np.deg2rad(delta_angle * self.odom_error_angle / 100.0) ** 2

        # Update cumulative covariance incrementally
        cumulative_covariance[0] += incremental_position_variance  # x
        cumulative_covariance[7] += incremental_position_variance  # y
        cumulative_covariance[14] += incremental_position_variance  # z
        cumulative_covariance[35] += incremental_angle_variance  # yaw

        # Populate pose covariance (6x6 matrix in row-major order)
        odom_msg.pose.covariance = cumulative_covariance.copy()
        odom_msg.pose.covariance[0] = max(1e-6, cumulative_covariance[0] + systematic_position_variance)
        odom_msg.pose.covariance[7] = max(1e-6,cumulative_covariance[7] + systematic_position_variance)
        odom_msg.pose.covariance[14] = max(1e-6,cumulative_covariance[14] + systematic_position_variance)
        odom_msg.pose.covariance[35] = max(1e-6,cumulative_covariance[35] + systematic_angle_variance)


        # Publish the odometry message
        if topic == '/uav/odom':
            self.uav_odom_publisher.publish(odom_msg)
        elif topic == '/agv/odom':
            self.agv_odom_publisher.publish(odom_msg)


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

        temp_traveled_distance_uav = temp_traveled_distance_agv = 0
        temp_traveled_angle_uav = temp_traveled_angle_agv = 0

        dt = 1.0 / self.publish_rate

        for uav_command in self.uav_velocity_commands:
            v_uav = np.array(uav_command[:2])  # Extract [v_x, v_y]
            w_uav = uav_command[2]              # Extract angular velocity
            temp_traveled_distance_uav += np.linalg.norm(v_uav) * dt
            temp_traveled_angle_uav += np.rad2deg(abs(w_uav) * dt)

            temp_uav_pose, temp_uav_odom_pose = self.integrate_odometry(
                temp_uav_pose, v_uav, w_uav, dt,
                temp_traveled_distance_uav, temp_traveled_angle_uav,
                self.uav_origin, self.holonomic_xy
            )

            uav_positions_gt.append(temp_uav_pose[:3])
            uav_positions_odom.append(temp_uav_odom_pose[:3])
        
        for agv_command in self.agv_velocity_commands:
            v_agv = np.array(agv_command[:2])  # Extract [v_x, v_y]
            w_agv = agv_command[2]              # Extract angular velocity
            temp_traveled_distance_agv += np.linalg.norm(v_agv) * dt
            temp_traveled_angle_agv += np.rad2deg(abs(w_agv) * dt)

            temp_agv_pose, temp_agv_odom_pose = self.integrate_odometry(
                temp_agv_pose, v_agv, w_agv, dt,
                temp_traveled_distance_agv, temp_traveled_angle_agv,
                self.agv_origin, self.holonomic_xy
            )

            agv_positions_gt.append(temp_agv_pose[:3])
            agv_positions_odom.append(temp_agv_odom_pose[:3])
        
        uav_positions_gt = np.array(uav_positions_gt)
        uav_positions_odom = np.array(uav_positions_odom)
        agv_positions_gt = np.array(agv_positions_gt)
        agv_positions_odom = np.array(agv_positions_odom)

        # Transform UAV odom points from odom frame to world frame using uav_origin
        theta_origin = self.uav_origin[3]
        rot_matrix = np.array([[np.cos(theta_origin), -np.sin(theta_origin)],
                            [np.sin(theta_origin),  np.cos(theta_origin)]])
        uav_positions_odom_world = []

        for p in uav_positions_odom:
            xy_transformed = rot_matrix @ p[:2] + self.uav_origin[:2]
            z_transformed = p[2] + self.uav_origin[2]
            uav_positions_odom_world.append([xy_transformed[0], xy_transformed[1], z_transformed])
        uav_positions_odom_world = np.array(uav_positions_odom_world)


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(uav_positions_gt[:, 0], uav_positions_gt[:, 1], uav_positions_gt[:, 2],
                label='UAV GT', color='blue')
        ax.plot(uav_positions_odom_world[:, 0], uav_positions_odom_world[:, 1], uav_positions_odom_world[:, 2],
                label='UAV Odom', color='blue', linestyle='--')

        ax.plot(agv_positions_gt[:, 0], agv_positions_gt[:, 1], agv_positions_gt[:, 2],
                label='AGV GT', color='red')
        ax.plot(agv_positions_odom[:, 0], agv_positions_odom[:, 1], agv_positions_odom[:, 2],
                label='AGV Odom', color='red', linestyle='--')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        ax.set_title('UAV and AGV Trajectories')
        ax.legend()

        plt.show()

    def transform_publisher(self, pose, parent_frame, child_frame):
        """Publish the transform for a given pose."""
        x, y, z, theta = pose

        # sample noise
        roll_noise  = random.gauss(0.0, 0.01)
        pitch_noise = random.gauss(0.0, 0.01)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame

        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z

        # rotation: apply noise on roll & pitch, keep yaw = theta
        quat = R.from_euler('xyz', [roll_noise, pitch_noise, theta]).as_quat()
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

    def create_odom_frame(self, position, label):

        # Tag transform
        odom_transform = TransformStamped()
        odom_transform.header.stamp = self.get_clock().now().to_msg()
        odom_transform.header.frame_id = 'world'
        odom_transform.child_frame_id = label

        odom_transform.transform.translation.x = position[0]
        odom_transform.transform.translation.y = position[1]
        odom_transform.transform.translation.z = position[2]

        quat = R.from_euler('z', position[3]).as_quat()
        odom_transform.transform.rotation.x = quat[0]
        odom_transform.transform.rotation.y = quat[1]
        odom_transform.transform.rotation.z = quat[2]
        odom_transform.transform.rotation.w = quat[3]

        self.tf_static_broadcaster.sendTransform(odom_transform)


    def save_trajectories(self):
            """Save generated trajectories to a file."""
            data = {
                'uav': self.uav_velocity_commands,
                'agv': self.agv_velocity_commands
            }
            np.save(self.trajectory_name, data)
            self.get_logger().info(f"Trajectories saved to {self.trajectory_name}")

    def save_trajectory_positions(self):

        """Integrate velocity commands and save trajectories as x, y, z, yaw coordinates."""
        dt = 1.0 / self.publish_rate

        # UAV trajectory integration
        uav_positions = [self.uav_pose.copy()]
        agv_positions = [self.agv_pose.copy()]

        temp_uav_pose = self.uav_pose.copy()
        temp_agv_pose = self.agv_pose.copy()

        temp_traveled_distance_uav = temp_traveled_distance_agv = 0
        temp_traveled_angle_uav = temp_traveled_angle_agv = 0

        for uav_command in self.uav_velocity_commands:
            v_uav = np.array(uav_command[:2])
            w_uav = uav_command[2]
            temp_traveled_distance_uav += np.linalg.norm(v_uav) * dt
            temp_traveled_angle_uav += np.rad2deg(abs(w_uav) * dt)
            temp_uav_pose, _ = self.integrate_odometry(
                temp_uav_pose, v_uav, w_uav, dt,
                temp_traveled_distance_uav, temp_traveled_angle_uav,
                self.uav_origin, self.holonomic_xy
            )
            uav_positions.append(temp_uav_pose.copy())

        for agv_command in self.agv_velocity_commands:
            v_agv = np.array(agv_command[:2])
            w_agv = agv_command[2]
            temp_traveled_distance_agv += np.linalg.norm(v_agv) * dt
            temp_traveled_angle_agv += np.rad2deg(abs(w_agv) * dt)
            temp_agv_pose, _ = self.integrate_odometry(
                temp_agv_pose, v_agv, w_agv, dt,
                temp_traveled_distance_agv, temp_traveled_angle_agv,
                self.agv_origin, self.holonomic_xy
            )
            agv_positions.append(temp_agv_pose.copy())

        data = {
            'uav_positions': np.array(uav_positions),  # Shape: [N, 4] (x, y, z, yaw)
            'agv_positions': np.array(agv_positions)
        }
        
        np.save(self.trajectory_name + "_positions.npy", data)
        np.savetxt(self.trajectory_name + "_uav.csv", data['uav_positions'], delimiter=',', header="x,y,z,yaw", comments='')
        np.savetxt(self.trajectory_name + "_agv.csv", data['agv_positions'], delimiter=',', header="x,y,z,yaw", comments='')
        
        self.get_logger().info(f"Trajectory positions saved to {self.trajectory_name}_position")



def main(args=None):
    rclpy.init(args=args)

    odometry_simulator = OdometrySimulator()

    rclpy.spin(odometry_simulator)

    odometry_simulator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
