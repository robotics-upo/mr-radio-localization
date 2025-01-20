#!/usr/bin/env python3

import rclpy
import rclpy.duration
from rclpy.node import Node
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

import tf_transformations
import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped, QuaternionStamped
from std_msgs.msg import Float32, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray  # Import Marker messages
from eliko_messages.msg import Distances, DistancesList
import time


class MeasurementSimulatorEliko(Node):

    def __init__(self):

        super().__init__('measurement_simulator_eliko')
        self.get_logger().info("Measurement Simulator Node Started")

        #Declare params
        self.declare_parameters(
            namespace='',
            parameters=[
                ('measurement_noise_std', 0.1),
                ('pub_rate', 10.0),
                ('anchors.a1.id', "0x0009D6"),
                ('anchors.a2.id', "0x0009E5"),
                ('anchors.a3.id', "0x0016FA"),
                ('anchors.a4.id', "0x0016CF"),
                ('tags.t1.id', "0x001155"),
                ('tags.t2.id', "0x001397")
            ])

        self.measurement_noise_std = self.get_parameter("measurement_noise_std").value
        self.rate = self.get_parameter("pub_rate").value

        self.anchors_ids =	{
            'a1': self.get_parameter("anchors.a1.id").value,
            'a2': self.get_parameter("anchors.a2.id").value,
            'a3': self.get_parameter("anchors.a3.id").value,
            'a4': self.get_parameter("anchors.a4.id").value
        } 

        self.tags_ids =	{
            't1': self.get_parameter("tags.t1.id").value,
            't2': self.get_parameter("tags.t2.id").value
        } 

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.uav_gt_markers = MarkerArray()
        self.arco_gt_markers = MarkerArray()
        self.uav_opt_markers = MarkerArray()

        self.translation_errors = []
        self.rotation_errors = []

        #Create publisher to simulate orientation changes
        self.attitude_pub = self.create_publisher(QuaternionStamped, 'dji_sdk/attitude', 1)

        # Create measurement publisher tags
        self.distances_publisher = self.create_publisher(DistancesList, 'eliko/Distances', 1)

        #Publish ground truth errors
        self.error_publisher = self.create_publisher(Float32MultiArray, 'optimization/metrics', 1)

        # Publishers for RViz markers
        #apply transform to most recent point
        self.opt_target_marker_pub = self.create_publisher(MarkerArray, 'visualization/opt_target_marker', 1)
        #to apply transform globally to all points in source window
        self.global_opt_target_marker_pub = self.create_publisher(MarkerArray, 'visualization/global_opt_target_marker', 1)

        self.gt_source_marker_pub = self.create_publisher(MarkerArray, 'visualization/gt_source_marker', 1)
        self.gt_target_marker_pub = self.create_publisher(MarkerArray, 'visualization/gt_target_marker', 1)

        self.timer1 = self.create_timer(1./self.rate, self.on_timer_t1)
        self.timer2 = self.create_timer(1./self.rate, self.on_timer_t2)
        self.timer_att = self.create_timer(1./self.rate, self.on_timer_att)
        
        self.timer_gt = self.create_timer(1./self.rate, self.on_timer_gt)
        self.agv_transforms = []  # List to store AGV transforms (w_T_s)
        self.sliding_window_duration = 5.0  # Sliding window duration in seconds
   

        self.optimized_tf_sub = self.create_subscription(
            TransformStamped,
            'eliko_optimization_node/optimized_T',
            self.optimized_tf_cb,
            10)
        self.optimized_tf_sub  # prevent unused variable warning



    def transform_stamped_to_matrix(self, transform_stamped):
        # Extract translation
        tx = transform_stamped.transform.translation.x
        ty = transform_stamped.transform.translation.y
        tz = transform_stamped.transform.translation.z
        
        # Extract quaternion
        qx = transform_stamped.transform.rotation.x
        qy = transform_stamped.transform.rotation.y
        qz = transform_stamped.transform.rotation.z
        qw = transform_stamped.transform.rotation.w
        
        transformation_matrix = np.eye(4)
        # Convert quaternion to a rotation matrix
        rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
        
        # Add translation to the rotation matrix
        transformation_matrix[:3,:3] = rotation_matrix
        transformation_matrix[0, 3] = tx
        transformation_matrix[1, 3] = ty
        transformation_matrix[2, 3] = tz
        
        return transformation_matrix

    def matrix_to_transform_stamped(self, matrix, frame_id="world", child_frame_id="uav_opt", stamp=None):
        
        # Create a TransformStamped message
        transform_stamped = TransformStamped()
        
        # Set header information
        transform_stamped.header.frame_id = frame_id
        transform_stamped.child_frame_id = child_frame_id
        if stamp is None:
            transform_stamped.header.stamp = self.get_clock().now().to_msg()
        else:
            transform_stamped.header.stamp = stamp

        # Extract translation from the transformation matrix
        transform_stamped.transform.translation.x = matrix[0, 3]
        transform_stamped.transform.translation.y = matrix[1, 3]
        transform_stamped.transform.translation.z = matrix[2, 3]

        # Assuming `matrix` is a 3x3 rotation matrix
        rotation = R.from_matrix(matrix[:3,:3])
        quaternion = rotation.as_quat()  # Returns [qx, qy, qz, qw]

        # Assign quaternion to the TransformStamped message
        transform_stamped.transform.rotation.x = quaternion[0]
        transform_stamped.transform.rotation.y = quaternion[1]
        transform_stamped.transform.rotation.z = quaternion[2]
        transform_stamped.transform.rotation.w = quaternion[3]

        return transform_stamped

    def compute_transformation_errors(self, T_w_s, T_w_t, That_t_s):
    
        T_ts = np.linalg.inv(T_w_t) @ T_w_s
        # # Example converting a rotation matrix to Euler angles
        Te = np.linalg.inv(That_t_s) @ T_ts

        Re = R.from_matrix(Te[:3,:3])
        Re_rpy = Re.as_euler('zyx', degrees=True)
        te = Te[:3,3]
        dett = np.linalg.norm(te)
        detR = np.linalg.norm(Re_rpy)

        self.translation_errors.append(dett)
        self.rotation_errors.append(detR)

        # Compute RMSE for translation
        rmse_translation = np.sqrt(np.mean(np.square(np.array(self.translation_errors))))

        # Compute RMSE for rotation
        rmse_rotation = np.sqrt(np.mean(np.square(np.array(self.rotation_errors))))

        return detR, dett, rmse_rotation, rmse_translation
    
    
    def create_marker(self, transform, marker_list, frame_id, marker_ns, color):
        
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = transform.header.stamp
        marker.ns = marker_ns
        marker.id = len(marker_list.markers)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = transform.transform.translation.x
        marker.pose.position.y = transform.transform.translation.y
        marker.pose.position.z = transform.transform.translation.z
        marker.pose.orientation = transform.transform.rotation
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.8

        marker_list.markers.append(marker)


    def optimized_tf_cb(self, msg):

        that_ts_msg = msg
        That_t_s = self.transform_stamped_to_matrix(that_ts_msg)

        try:
                                         
                gt_target = self.tf_buffer.lookup_transform(
                    'world',
                    'uav_gt',
                    rclpy.time.Time())
                               
                gt_source = self.tf_buffer.lookup_transform(
                    'world',
                    'agv_gt',
                    rclpy.time.Time())
                
                odom_source = self.tf_buffer.lookup_transform(
                    'world',
                    'agv_odom',
                    rclpy.time.Time())
                
                T_w_s = self.transform_stamped_to_matrix(gt_source)
                T_w_t = self.transform_stamped_to_matrix(gt_target)
                T_w_s_odom = self.transform_stamped_to_matrix(odom_source)

                That_w_t_odom = T_w_s_odom  @ np.linalg.inv(That_t_s) #I am applying the transform to the odometry source, not gt -for visualization
                         
                detR, dett, rmse_R, rmse_t = self.compute_transformation_errors(T_w_s, #errors are computed wrt gt
                                                                T_w_t, 
                                                                That_t_s)
                
                opt_target = self.matrix_to_transform_stamped(That_w_t_odom, "world", "uav_opt", that_ts_msg.header.stamp)   
                self.create_marker(opt_target, self.uav_opt_markers, "world", "uav_opt_marker", [0.0, 0.0, 1.0])
                
                self.create_marker(gt_target, self.uav_gt_markers, "world", "uav_gt_marker", [0.0, 1.0, 0.0])
                self.create_marker(gt_source, self.arco_gt_markers, "world", "arco_gt_marker", [1.0, 0.0, 0.0])

                metrics = Float32MultiArray()
                metrics.data = [detR, dett, rmse_R, rmse_t]  # Set both values at once

                # Publish the MarkerArrays
                self.gt_source_marker_pub.publish(self.arco_gt_markers)
                self.gt_target_marker_pub.publish(self.uav_gt_markers)
                self.opt_target_marker_pub.publish(self.uav_opt_markers)

                #Publish metrics
                self.error_publisher.publish(metrics)

        
        except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform: {ex}')
                return

        ##Apply transform to all points in sliding window

        self.global_transform_trajectory(That_t_s)
    

    def global_transform_trajectory(self, That_t_s):

        # Compute w_That_t for each w_T_s in the sliding window
        transformed_points = []
        for agv_transform in self.agv_transforms:
            w_T_s_odom = self.transform_stamped_to_matrix(agv_transform)

            # Compute w_That_t = w_T_s * np.linalg.inv(t_That_s)
            w_That_t = w_T_s_odom @ np.linalg.inv(That_t_s)

            # Store the result for visualization
            transformed_points.append({
                'timestamp': agv_transform.header.stamp,
                'transform': w_That_t
            })

        uav_opt_markers = MarkerArray()
        # Publish transformed points as markers
        for point in transformed_points:
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = point['timestamp']
            marker.ns = "transformed_points"
            marker.id = len(uav_opt_markers.markers)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = point['transform'][0][3]
            marker.pose.position.y = point['transform'][1][3]
            marker.pose.position.z = point['transform'][2][3]
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05

            marker.color.r = 0.25
            marker.color.g = 0.5
            marker.color.b = 1.0
            marker.color.a = 0.8

            uav_opt_markers.markers.append(marker)
        
        self.global_opt_target_marker_pub.publish(uav_opt_markers)
                                            
        
    def on_timer_att(self):

        try:
                t = self.tf_buffer.lookup_transform(
                    'world',
                    'uav_gt',
                    rclpy.time.Time())
                
                quaternion_msg = QuaternionStamped()
                quaternion_msg.header.stamp = t.header.stamp  # Use the same timestamp as the transform
                quaternion_msg.header.frame_id = t.header.frame_id

                # Copy the orientation (quaternion) from the transform
                quaternion_msg.quaternion = t.transform.rotation

                # Publish the QuaternionStamped message
                self.attitude_pub.publish(quaternion_msg)

                
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform world to uav_gt: {ex}')
            return


    def on_timer_gt(self):
        try:
            # Get the AGV transform (w_T_s) --odometry, not gt
            agv_transform = self.tf_buffer.lookup_transform(
                'world', 
                'agv_odom', 
                rclpy.time.Time())

            # Store the transform with a timestamp
            self.agv_transforms.append(agv_transform)

            # Remove old transforms that are outside the sliding window duration
            self.agv_transforms = [
                t for t in self.agv_transforms
                if (rclpy.time.Time.from_msg(agv_transform.header.stamp).nanoseconds - rclpy.time.Time.from_msg(t.header.stamp).nanoseconds) / 1e9 <= self.sliding_window_duration
            ]

        except TransformException as ex:
            self.get_logger().info(f"Could not transform world to ground_vehicle: {ex}")

    #Publish simulated measures from tag1 to anchors using simulated ground truth

    def on_timer_t1(self):

        distances_list = DistancesList()

        for anchor, id in self.anchors_ids.items(): 
            try:
                t = self.tf_buffer.lookup_transform(
                    't1',
                    anchor,
                    rclpy.time.Time())
                
                distance = Distances()
                distance.anchor_sn = id
                distance.tag_sn = self.tags_ids['t1']

                 # Add outlier with 5% probability
                if random.random() < 0.05:
                    gaussian_noise = np.random.normal(0, self.measurement_noise_std*5.0)
                else:
                    gaussian_noise = np.random.normal(0, self.measurement_noise_std)

                distance.distance = np.sqrt(t.transform.translation.x**2 + t.transform.translation.y**2 + t.transform.translation.z**2) + gaussian_noise
                distance.distance = distance.distance * 100.0 #cm
                distances_list.anchor_distances.append(distance)


            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform t1 to {anchor}: {ex}')
                return
            
        distances_list.header.stamp = self.get_clock().now().to_msg()
        distances_list.header.frame_id = 'arco/eliko'
        self.distances_publisher.publish(distances_list)
        

    #Publish simulated measures from tag2 to anchors

    def on_timer_t2(self):

        distances_list = DistancesList()

        for anchor, id in self.anchors_ids.items(): 
            try:
                t = self.tf_buffer.lookup_transform(
                    't2',
                    anchor,
                    rclpy.time.Time())
                
                distance = Distances()
                distance.anchor_sn = id
                distance.tag_sn = self.tags_ids['t2']
                 # Add outlier with 5% probability
                if random.random() < 0.05:
                    gaussian_noise = np.random.normal(0, self.measurement_noise_std*5.0)
                else:
                    gaussian_noise = np.random.normal(0, self.measurement_noise_std)

                distance.distance = np.sqrt(t.transform.translation.x**2 + t.transform.translation.y**2 + t.transform.translation.z**2) + gaussian_noise
                distance.distance = distance.distance * 100.0 #cm

                distances_list.anchor_distances.append(distance)


            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform t1 to {anchor}: {ex}')
                return
            
        distances_list.header.stamp = self.get_clock().now().to_msg()
        distances_list.header.frame_id = 'arco/eliko'
        self.distances_publisher.publish(distances_list)
        


def main(args=None):
    rclpy.init(args=args)

    measurement_simulator = MeasurementSimulatorEliko()

    try:
        rclpy.spin(measurement_simulator)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()
