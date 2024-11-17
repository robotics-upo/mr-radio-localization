import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from rosgraph_msgs.msg import Clock

class ClockPublisher(Node):
    def __init__(self):
        super().__init__('clock_publisher')
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.timer = self.create_timer(0.05, self.publish_time)  # 10 Hz clock

        # Start with a simulated time of 0
        self.sim_time = 0.0  # seconds
        self.time_increment = 0.05  # seconds per timer callback

    def publish_time(self):
        # Simulate time advancing
        self.sim_time += self.time_increment

        # Create a Clock message
        clock_msg = Clock()
        clock_msg.clock = Time()
        clock_msg.clock.sec = int(self.sim_time)
        clock_msg.clock.nanosec = int((self.sim_time - int(self.sim_time)) * 1e9)

        # Publish the clock
        self.clock_pub.publish(clock_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ClockPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()