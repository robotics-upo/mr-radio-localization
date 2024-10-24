#!/usr/bin/env python3

import rclpy
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

        # Declare and acquire `target_frame` parameter
        self.target_frame = self.declare_parameter(
          'target_frame', 'turtle1').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create measurement publisher tag2
        self.t1a1_publisher = self.create_publisher(Float32, 'range/t1a1', 1)
        self.t1a2_publisher = self.create_publisher(Float32, 'range/t1a2', 1)
        self.t1a3_publisher = self.create_publisher(Float32, 'range/t1a3', 1)
        self.t1a4_publisher = self.create_publisher(Float32, 'range/t1a4', 1)
        # Create measurement publisher
        self.t2a1_publisher = self.create_publisher(Float32, 'range/t2a1', 1)
        self.t2a2_publisher = self.create_publisher(Float32, 'range/t2a2', 1)
        self.t2a3_publisher = self.create_publisher(Float32, 'range/t2a3', 1)
        self.t2a4_publisher = self.create_publisher(Float32, 'range/t2a4', 1)


        # Call on_timer function every second
        self.timer1 = self.create_timer(1.0, self.on_timer('t1','a1', self.t1a1_publisher))
        self.timer2 = self.create_timer(1.0, self.on_timer('t1','a2', self.t1a2_publisher))
        self.timer3 = self.create_timer(1.0, self.on_timer('t1','a3', self.t1a3_publisher))
        self.timer4 = self.create_timer(1.0, self.on_timer('t1','a4', self.t1a4_publisher))

        # # Call on_timer function every second
        self.timer5 = self.create_timer(1.0, self.on_timer('t2','a1', self.t2a1_publisher))
        self.timer6 = self.create_timer(1.0, self.on_timer('t2','a2', self.t2a2_publisher))
        self.timer7 = self.create_timer(1.0, self.on_timer('t2','a3', self.t2a3_publisher))
        self.timer8 = self.create_timer(1.0, self.on_timer('t2','a4', self.t2a4_publisher))

        #self.timer = self.create_timer(1.0, self.on_timer('uav','ground_vehicle', self.distance_publisher))


    def on_timer(self, to_frame_rel, from_frame_rel, publisher):

        try:
            t = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return
        
        distance = Float32()

        distance.data = np.sqrt(t.transform.translation.x**2 + t.transform.translation.y**2 + t.transform.translation.z**2) + np.random.normal(0, 0.2)
        
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
