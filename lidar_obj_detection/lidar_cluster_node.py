import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import numpy as np
from sklearn.cluster import DBSCAN
import math

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
        self.angle_thresh = 2.0

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

        # DBSCAN tuned for detecting ~12x12 cm robot
        clustering = DBSCAN(eps=0.07, min_samples=5).fit(points)
        labels = clustering.labels_

        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # noise

            cluster = points[labels == label]
            size = np.linalg.norm(cluster.max(axis=0) - cluster.min(axis=0))
            center = cluster.mean(axis=0)
            distance = np.linalg.norm(center)

            # Debug info
            self.get_logger().info(f"Cluster size: {size:.2f} m, Distance: {distance:.2f} m")

            color = (0.0, 1.0, 0.0)  # green by default
            angle = np.arctan2(center[1], center[0])
            if 0.2 < size < 0.25 and 1.0 < distance < 4.0 and center[0] > 0 and abs(angle) < np.radians(self.angle_thresh):
                self.get_logger().warn(">> Likely another robot nearby!")
                self.get_logger().info(f"Position -> x: {center[0]:.2f}, y: {center[1]:.2f}")
                color = (1.0, 0.0, 0.0)  # red
                self.publish_marker(center[0], center[1], size, color)
            else:
                # self.publish_marker(center[0], center[1], 0.01, color)
                pass

    def publish_marker(self, x, y, size, color):
        marker = Marker()
        marker.header.frame_id = "ego_racecar/laser"
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
