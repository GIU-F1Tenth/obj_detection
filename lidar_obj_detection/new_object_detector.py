import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import math
import numpy as np

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        self.map = None
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def map_cb(self, msg):
        self.map = msg

    def scan_cb(self, msg):
        if self.map is None:
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.map.header.frame_id,
                msg.header.frame_id,
                rclpy.time.Time())
        except:
            self.get_logger().warn("TF unavailable")
            return

        resolution = self.map.info.resolution
        origin = self.map.info.origin.position
        width = self.map.info.width
        height = self.map.info.height
        data = np.array(self.map.data).reshape((height, width))

        obstacle_found = False
        angle = msg.angle_min

        for r in msg.ranges:
            if r < msg.range_min or r > msg.range_max:
                angle += msg.angle_increment
                continue

            # Convert to base_link coordinates
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            angle += msg.angle_increment

            # Transform to map frame
            map_x = tf.transform.translation.x + x
            map_y = tf.transform.translation.y + y

            mx = int((map_x - origin.x) / resolution)
            my = int((map_y - origin.y) / resolution)

            if 0 <= mx < width and 0 <= my < height:
                val = data[my][mx]
                if val == 0:
                    # free space in map, but laser hit → unexpected obstacle
                    self.get_logger().warn("Unexpected obstacle detected!")
                    obstacle_found = True
                    break

        if not obstacle_found:
            self.get_logger().info("No unexpected obstacle.")

def main():
    rclpy.init()
    node = ObstacleDetector()
    rclpy.spin(node)
    rclpy.shutdown()
