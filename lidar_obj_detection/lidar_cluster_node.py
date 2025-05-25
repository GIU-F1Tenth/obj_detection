import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import numpy as np
from sklearn.cluster import DBSCAN
import math
import random

class LidarClusterNode(Node):
    def __init__(self):
        super().__init__('lidar_cluster_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.marker_pub = self.create_publisher(Marker, '/clusters', 10)
        self.marker_id = 0

    def scan_callback(self, msg):
        self.marker_id = 0  # reset marker ID for each scan

        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        valid = (ranges > msg.range_min) & (ranges < msg.range_max)
        angles = angles[valid]
        ranges = ranges[valid]

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        points = np.stack((xs, ys), axis=1)

        if len(points) == 0:
            return

        clustering = DBSCAN(eps=0.15, min_samples=10).fit(points)
        labels = clustering.labels_

        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # noise

            cluster = points[labels == label]
            size = np.linalg.norm(cluster.max(axis=0) - cluster.min(axis=0))
            center = cluster.mean(axis=0)
            distance = np.linalg.norm(center)

            color = (0.0, 1.0, 0.0)  # green by default
            if 0.3 < size < 1.5 and distance < 2.0:
                self.get_logger().warn(">> Likely another robot nearby!")
                color = (1.0, 0.0, 0.0)  # red if suspicious
                self.get_logger().info(f"x: {center[0]}, y: {center[1]}")
                self.publish_marker(center[0], center[1], size, color) # publish the marker if there is a car
            else:
                self.publish_marker(center[0], center[1], 0.01, color) # publish the marker if there is a car

    def publish_marker(self, x, y, size, color):
        marker = Marker()
        marker.header.frame_id = "/laser"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "clusters"
        marker.id = self.marker_id
        self.marker_id += 1

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = 0.1
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.8
        self.marker_pub.publish(marker)

def main():
    rclpy.init()
    node = LidarClusterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
