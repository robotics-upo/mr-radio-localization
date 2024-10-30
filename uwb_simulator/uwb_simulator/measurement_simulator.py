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
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32, Float32MultiArray
import time

class MeasurementSimulator(Node):

    def __init__(self):

        super().__init__('measurement_simulator')
        self.get_logger().info("Measurement Simulator Node Started")

        #Declare params
        self.declare_parameters(
            namespace='',
            parameters=[
                ('measurement_noise_std', 0.1),
                ('pub_rate', 10.0)
            ])

        self.measurement_noise_std = self.get_parameter("measurement_noise_std").value
        self.rate = self.get_parameter("pub_rate").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create measurement publisher tag1
        self.t1a1_publisher = self.create_publisher(Float32, 'range/t1a1', 1)
        self.t1a2_publisher = self.create_publisher(Float32, 'range/t1a2', 1)
        self.t1a3_publisher = self.create_publisher(Float32, 'range/t1a3', 1)
        self.t1a4_publisher = self.create_publisher(Float32, 'range/t1a4', 1)
        # Create measurement publisher tag2
        self.t2a1_publisher = self.create_publisher(Float32, 'range/t2a1', 1)
        self.t2a2_publisher = self.create_publisher(Float32, 'range/t2a2', 1)
        self.t2a3_publisher = self.create_publisher(Float32, 'range/t2a3', 1)
        self.t2a4_publisher = self.create_publisher(Float32, 'range/t2a4', 1)

        #Publish ground truth errors
        self.error_publisher = self.create_publisher(Float32MultiArray, 'optimization/metrics', 1)

        self.timer1 = self.create_timer(1./self.rate, self.on_timer_t1)
        self.timer2 = self.create_timer(1./self.rate, self.on_timer_t2)

        # Calculate error using ground truth and optimized T
        self.optimized_tf_sub = self.create_subscription(
            TransformStamped,
            'uwb_optimization_node/optimized_T',
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
        
        # Convert quaternion to a rotation matrix
        rotation_matrix = tf_transformations.quaternion_matrix([qx, qy, qz, qw])
        
        # Add translation to the rotation matrix
        transformation_matrix = rotation_matrix
        transformation_matrix[0, 3] = tx
        transformation_matrix[1, 3] = ty
        transformation_matrix[2, 3] = tz
        
        return transformation_matrix


    def compute_transformation_errors(self, gt_source, gt_target, That_ts):
    
        T_ts = np.linalg.inv(gt_target) @ gt_source
        # # Example converting a rotation matrix to Euler angles
        Te = np.linalg.inv(That_ts) @ T_ts
        Re = R.from_matrix(Te[:3,:3])
        te = Te[:3,3]
        dett = np.linalg.norm(te)
        detR = np.linalg.norm(Re.as_euler('zxy', degrees=True))

        return detR, dett


    def optimized_tf_cb(self, msg):

        try:
                gt_target = self.tf_buffer.lookup_transform(
                    'world',
                    'uav_gt',
                    rclpy.time.Time())
                
                gt_source = self.tf_buffer.lookup_transform(
                    'world',
                    'ground_vehicle',
                    rclpy.time.Time())
                
                # that_ts = self.tf_buffer.lookup_transform(
                #     'uav_opt',
                #     'ground_vehicle',
                #     rclpy.time.Time())
                
                that_ts = msg

                        
                detR, dett = self.compute_transformation_errors(self.transform_stamped_to_matrix(gt_source), 
                                                                self.transform_stamped_to_matrix(gt_target), 
                                                                self.transform_stamped_to_matrix(that_ts))


                metrics = Float32MultiArray()
                metrics.data = [detR, dett]  # Set both values at once


                self.error_publisher.publish(metrics)


                # self.get_logger().info(
                # f'rotation error (deg): {detR}, translation error (m): {dett}', throttle_duration_sec=1)
                                                        
                                            
        except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform: {ex}')
                return
        

    #Publish simulated measures from tag1 to anchors using simulated ground truth

    def on_timer_t1(self):
   
        anchors_publishers =	{
        'a1': self.t1a1_publisher,
        'a2': self.t1a2_publisher,
        'a3': self.t1a3_publisher,
        'a4': self.t1a4_publisher
        } 

        for anchor, publisher in anchors_publishers.items(): 
            try:
                t = self.tf_buffer.lookup_transform(
                    't1',
                    anchor,
                    rclpy.time.Time())
                
                distance = Float32()
                distance.data = np.sqrt(t.transform.translation.x**2 + t.transform.translation.y**2 + t.transform.translation.z**2) + np.random.normal(0, self.measurement_noise_std)
                
            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform t1 to {anchor}: {ex}')
                return
        
            publisher.publish(distance)


    #Publish simulated measures from tag2 to anchors

    def on_timer_t2(self):
   
        anchors_publishers =	{
        'a1': self.t2a1_publisher,
        'a2': self.t2a2_publisher,
        'a3': self.t2a3_publisher,
        'a4': self.t2a4_publisher
        } 

        for anchor, publisher in anchors_publishers.items(): 
            try:
                t = self.tf_buffer.lookup_transform(
                    't2',
                    anchor,
                    rclpy.time.Time())
                
                distance = Float32()
                distance.data = np.sqrt(t.transform.translation.x**2 + t.transform.translation.y**2 + t.transform.translation.z**2) + np.random.normal(0, self.measurement_noise_std)
                
            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform t2 to {anchor}: {ex}')
                return
        
            publisher.publish(distance)



def main(args=None):
    rclpy.init(args=args)

    measurement_simulator = MeasurementSimulator()

    try:
        rclpy.spin(measurement_simulator)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()
