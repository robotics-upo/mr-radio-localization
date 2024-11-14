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
        self.uav_opt_markers = MarkerArray()
        self.arco_gt_markers = MarkerArray()

        #Create publisher to simulate orientation changes
        self.attitude_pub = self.create_publisher(QuaternionStamped, 'dji_sdk/attitude', 1)

        # Create measurement publisher tags
        self.distances_publisher = self.create_publisher(DistancesList, 'eliko/Distances', 1)

        #Publish ground truth errors
        self.error_publisher = self.create_publisher(Float32MultiArray, 'optimization/metrics', 1)

        # Publishers for RViz markers
        self.opt_target_marker_pub = self.create_publisher(MarkerArray, 'visualization/opt_target_marker', 1)
        self.gt_source_marker_pub = self.create_publisher(MarkerArray, 'visualization/gt_source_marker', 1)
        self.gt_target_marker_pub = self.create_publisher(MarkerArray, 'visualization/gt_target_marker', 1)

        self.timer1 = self.create_timer(1./self.rate, self.on_timer_t1)
        self.timer2 = self.create_timer(1./self.rate, self.on_timer_t2)
        self.timer_att = self.create_timer(1./self.rate, self.on_timer_att)
        

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
        Re_rpy = Re.as_euler('zxy', degrees=True)
        te = Te[:3,3]
        dett = np.linalg.norm(te)
        detR = np.linalg.norm(Re_rpy)

        return detR, dett
    
    def compute_transformation_errors_alt(self, T_w_s, T_w_t):
    
        Te = np.linalg.inv(T_w_t) @ T_w_s

        Re = R.from_matrix(Te[:3,:3])
        Re_rpy = Re.as_euler('zxy', degrees=True)
        te = Te[:3,3]
        dett = np.linalg.norm(te)
        detR = np.linalg.norm(Re_rpy)

        return detR, dett
    
    
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

        try:
                that_ts_msg = msg

                gt_target = self.tf_buffer.lookup_transform(
                    'world',
                    'uav_gt',
                    that_ts_msg.header.stamp)
                               
                gt_source = self.tf_buffer.lookup_transform(
                    'world',
                    'ground_vehicle',
                    that_ts_msg.header.stamp)

                T_w_s = self.transform_stamped_to_matrix(gt_source)
                T_w_t = self.transform_stamped_to_matrix(gt_target)
                That_t_s = self.transform_stamped_to_matrix(that_ts_msg)
                That_w_t = T_w_s  @ np.linalg.inv(That_t_s)
                         
                detR1, dett1 = self.compute_transformation_errors(T_w_s, 
                                                                T_w_t, 
                                                                That_t_s)
                
                detR2, dett2 = self.compute_transformation_errors_alt(T_w_t, That_w_t)

                #Select minimum error from the two methods
                dett = min(dett1, dett2)
                detR = min(detR1,detR2)

                self.get_logger().info(
                f'Alt rotation error (deg): {detR}, translation error (m): {dett}', throttle_duration_sec=1)
                
                opt_target = self.matrix_to_transform_stamped(That_w_t, "world", "uav_opt", that_ts_msg.header.stamp)

                
                self.create_marker(opt_target, self.uav_opt_markers, "world", "uav_opt_marker", [0.0, 0.0, 1.0])
                self.create_marker(gt_target, self.uav_gt_markers, "world", "uav_gt_marker", [0.0, 1.0, 0.0])
                self.create_marker(gt_source, self.arco_gt_markers, "world", "arco_gt_marker", [1.0, 0.0, 0.0])

                metrics = Float32MultiArray()
                metrics.data = [detR, dett]  # Set both values at once
                self.error_publisher.publish(metrics)

                # Publish the MarkerArrays
                self.gt_source_marker_pub.publish(self.arco_gt_markers)
                self.gt_target_marker_pub.publish(self.uav_gt_markers)
                self.opt_target_marker_pub.publish(self.uav_opt_markers)

                                            
        except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform: {ex}')
                return
        
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
                distance.distance = np.sqrt(t.transform.translation.x**2 + t.transform.translation.y**2 + t.transform.translation.z**2) + np.random.normal(0, self.measurement_noise_std)
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
                distance.distance = np.sqrt(t.transform.translation.x**2 + t.transform.translation.y**2 + t.transform.translation.z**2) + np.random.normal(0, self.measurement_noise_std)
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
