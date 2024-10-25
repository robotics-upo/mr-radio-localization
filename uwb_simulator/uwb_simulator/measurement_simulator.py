#!/usr/bin/env python3

import rclpy
import rclpy.duration
from rclpy.node import Node
import numpy as np
import random

import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32
import time

class MeasurementSimulator(Node):

    def __init__(self):

        super().__init__('measurement_simulator')
        self.get_logger().info("Measurement Simulator Node Started")

        self.rate = 10.0

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

        self.distance_publisher = self.create_publisher(Float32, 'range/uav_to_groundvehicle', 1)

        self.timer1 = self.create_timer(1./self.rate, self.on_timer_t1)
        self.timer2 = self.create_timer(1./self.rate, self.on_timer_t2)

        #self.timer = self.create_timer(1./self.rate, self.on_timer_debug)
        #self.timer = self.create_timer(1./self.rate, self.on_timer('uav', 'ground_vehicle', self.distance_publisher))

    def on_timer_debug(self):

        try:
            t = self.tf_buffer.lookup_transform(
                "uav",
                "ground_vehicle",
                rclpy.time.Time(), rclpy.duration.Duration(seconds=2.0))
            
            distance = Float32()
            distance.data = np.sqrt(t.transform.translation.x**2 + t.transform.translation.y**2 + t.transform.translation.z**2) + np.random.normal(0, 0.2)
            self.get_logger().info("Distance: %f" % distance.data)
            
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform "ground vehicle" to "uav": {ex}')
            return
        
        self.distance_publisher.publish(distance)
    
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
                distance.data = np.sqrt(t.transform.translation.x**2 + t.transform.translation.y**2 + t.transform.translation.z**2) + np.random.normal(0, 0.2)
                
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
                distance.data = np.sqrt(t.transform.translation.x**2 + t.transform.translation.y**2 + t.transform.translation.z**2) + np.random.normal(0, 0.2)
                
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
