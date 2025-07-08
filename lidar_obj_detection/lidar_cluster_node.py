"""
F1TENTH LiDAR Clustering Node for Object Detection.

This module implements a ROS2 node that processes LiDAR scan data to detect and track
objects in F1TENTH racing environments. It uses DBSCAN clustering to identify potential
racing opponents and publishes visualization markers and detection flags.

Author: F1TENTH Object Detection Team
License: MIT
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import numpy as np
from sklearn.cluster import DBSCAN
from std_msgs.msg import Bool
import math
import time


class LidarClusterNode(Node):
    """
    ROS2 node for LiDAR-based object detection using clustering algorithms.

    This node processes LiDAR scan data to detect objects that match the size and 
    characteristics of F1TENTH racing robots. It uses DBSCAN clustering followed
    by geometric filtering to identify potential racing opponents.

    The node subscribes to LiDAR data, processes it to find clusters, filters
    those clusters based on expected robot dimensions, and publishes both
    visualization markers and boolean detection flags.

    Attributes:
        marker_id (int): Counter for unique marker IDs
        angle_thresh (float): Angular threshold for forward-facing object detection (degrees)
        pub_true_timer (Timer): Timer for publishing detection signals
        counter (int): Counter for timed detection publishing
    """

    def __init__(self):
        """
        Initialize the LiDAR clustering node.

        Sets up ROS2 subscriptions, publishers, and configures detection parameters
        optimized for F1TENTH robot detection in racing environments.
        """
        super().__init__('lidar_cluster_node')

        # Subscribe to LiDAR scan data
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publishers for visualization and detection signals
        self.marker_pub = self.create_publisher(Marker, '/clusters', 10)
        self.obj_detected_pub = self.create_publisher(
            Bool, '/tmp/obj_detected', 10)

        # Detection parameters optimized for F1TENTH racing
        self.marker_id = 0
        self.angle_thresh = 15.0  # Forward detection cone angle in degrees
        self.pub_true_timer = None  # Timer to publish True periodically
        self.counter = 0

    def scan_callback(self, msg):
        """
        Process incoming LiDAR scan data to detect objects.

        This callback function is the core of the object detection system. It:
        1. Filters and validates LiDAR range data
        2. Converts polar coordinates to Cartesian
        3. Applies DBSCAN clustering to group related points
        4. Filters clusters based on size and position criteria
        5. Publishes visualization markers for detected objects

        Args:
            msg (LaserScan): ROS2 LaserScan message containing LiDAR data
                - ranges: Array of distance measurements
                - angle_min/max: Angular range of the scan
                - angle_increment: Angular resolution
                - range_min/max: Valid distance measurement range
        """
        self.marker_id = 0  # Reset marker ID for each scan

        # Generate angle array corresponding to each range measurement
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # Filter out invalid range measurements (inf, nan, out of bounds)
        valid = (ranges > msg.range_min) & (ranges < msg.range_max)
        angles = angles[valid]
        ranges = ranges[valid]

        # Convert from polar (range, angle) to Cartesian (x, y) coordinates
        # x: forward/backward axis, y: left/right axis
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        points = np.stack((xs, ys), axis=1)

        if len(points) == 0:
            return  # No valid points to process

        # Apply DBSCAN clustering to group nearby points
        # Parameters tuned for detecting ~12x12 cm F1TENTH robots
        clustering = DBSCAN(eps=0.07, min_samples=5).fit(points)
        labels = clustering.labels_

        # Process each detected cluster
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points (unclustered)

            # Extract points belonging to this cluster
            cluster = points[labels == label]

            # Calculate cluster characteristics
            size = np.linalg.norm(cluster.max(axis=0) - cluster.min(axis=0))
            center = cluster.mean(axis=0)
            distance = np.linalg.norm(center)
            angle = np.arctan2(center[1], center[0])

            # Apply geometric filters for F1TENTH robot detection
            # Size filter: 20-25 cm (typical F1TENTH robot dimensions)
            # Distance filter: within 3 meters (racing proximity)
            # Position filter: in front of the robot (center[0] > 0)
            # Angle filter: within forward-facing cone
            color = (0.0, 1.0, 0.0)  # Default green color

            if (0.20 < size < 0.25 and distance < 3.0 and
                    center[0] > 0 and abs(angle) < np.radians(self.angle_thresh)):

                # Potential racing opponent detected
                self.get_logger().warn(">> Likely another robot nearby!")
                self.get_logger().info(
                    f"Position -> x: {center[0]:.2f}, y: {center[1]:.2f}")

                color = (1.0, 0.0, 0.0)  # Red color for detected robots
                self.publish_marker(center[0], center[1], size, color)

                # Start timed detection signal publishing
                if self.pub_true_timer is None:
                    self.pub_true_timer = self.create_timer(
                        0.01, self.publish_true)

                self.get_logger().info(
                    f"Cluster size: {size:.2f} m, Distance: {distance:.2f} m")
            else:
                # Object doesn't match robot criteria
                if not self.pub_true_timer:
                    self.obj_detected_pub.publish(Bool(data=False))

    def publish_true(self):
        """
        Publish object detection signal for a limited duration.

        This method is called by a timer to publish True detection signals
        for a brief period when a robot is detected. This helps ensure that
        downstream nodes receive the detection signal reliably.
        """
        if self.pub_true_timer:
            self.counter += 1
            self.obj_detected_pub.publish(Bool(data=True))

            # Publish True for a short duration (2 timer cycles)
            if self.counter >= 2:
                self.pub_true_timer.cancel()
                self.pub_true_timer = None
                self.counter = 0

    def publish_marker(self, x, y, size, color):
        """
        Publish a visualization marker for a detected object.

        Creates and publishes an RViz marker to visualize detected objects
        in the robot's coordinate frame. The marker appears as a colored
        sphere at the object's estimated position.

        Args:
            x (float): Object's x-coordinate in the laser frame (meters)
            y (float): Object's y-coordinate in the laser frame (meters)  
            size (float): Estimated size/diameter of the object (meters)
            color (tuple): RGB color values (r, g, b) each in range [0, 1]
        """
        marker = Marker()

        # Header information
        marker.header.frame_id = "laser"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "clusters"
        marker.id = self.marker_id
        self.marker_id += 1

        # Marker properties
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position (object center)
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.1  # Slightly above ground plane
        marker.pose.orientation.w = 1.0  # No rotation

        # Size (sphere dimensions)
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = 0.1  # Flat disk appearance

        # Color and transparency
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.8  # Semi-transparent

        self.marker_pub.publish(marker)


def main():
    """
    Main entry point for the LiDAR clustering node.

    Initializes the ROS2 system, creates the clustering node instance,
    and runs the node until shutdown. Handles cleanup on exit.
    """
    rclpy.init()
    node = LidarClusterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
