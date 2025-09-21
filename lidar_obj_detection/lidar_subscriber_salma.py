"""
Basic LiDAR Object Detection and Classification.

This module provides a foundational implementation of LiDAR-based object detection
for F1TENTH racing applications. It demonstrates core concepts including FOV filtering,
coordinate transformation, clustering, and spatial object classification.

Author: Salma Tarek
License: MIT
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
from sklearn.cluster import DBSCAN


class LidarProcessor(Node):
    """
    Basic LiDAR processing node for object detection and spatial classification.

    This node implements fundamental LiDAR processing techniques including:
    - Field of view (FOV) filtering for 270-degree LiDAR sensors
    - Polar to Cartesian coordinate conversion
    - Distance-based filtering
    - DBSCAN clustering for object grouping
    - Spatial classification (front vs. side objects)

    This implementation serves as an educational example and foundation for
    more advanced object detection algorithms in the package.
    """

    def __init__(self):
        """
        Initialize the basic LiDAR processor node.

        Sets up the ROS2 subscriber for LiDAR data and configures basic
        processing parameters suitable for F1TENTH racing environments.
        """
        super().__init__('lidar_processor')

        # Subscribe to the LiDAR topic
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',  # Change to '/echoes' if using multi-echo LiDAR
            self.listener_callback,
            10
        )
        self.subscription  # Prevent unused variable warning

    def listener_callback(self, msg):
        """
        Process incoming LiDAR scan data to detect and classify objects.

        This method implements a complete LiDAR processing pipeline:
        1. Extract and validate LiDAR measurements
        2. Apply field-of-view filtering (270-degree coverage)
        3. Convert polar coordinates to Cartesian coordinates
        4. Filter objects by distance thresholds
        5. Apply clustering to group related measurements
        6. Filter clusters by size to identify significant objects
        7. Classify objects by spatial position (front vs. side)
        8. Log detected objects with position information

        Args:
            msg (LaserScan): ROS2 LaserScan message containing:
                - ranges: Array of distance measurements (meters)
                - angle_min: Minimum scan angle (radians)
                - angle_increment: Angular resolution between measurements (radians)
                - range_min/max: Valid measurement range bounds
        """
        # Extract LiDAR data from the message
        ranges = np.array(msg.ranges)  # Distance measurements array
        angle_min = msg.angle_min      # Starting angle of the scan (radians)
        angle_increment = msg.angle_increment  # Angular step between measurements

        # Step 1: Filter by the LiDAR's 270° field of view
        # Generate angle array corresponding to each range measurement
        angles = np.arange(angle_min, angle_min + len(ranges)
                           * angle_increment, angle_increment)

        # Define 270-degree FOV bounds (-135° to +135°)
        fov_min = -np.deg2rad(135)  # Left-most angle (-135°)
        fov_max = np.deg2rad(135)   # Right-most angle (+135°)

        # Filter measurements within the desired FOV
        fov_filtered_indices = np.where(
            (angles >= fov_min) & (angles <= fov_max))[0]
        fov_filtered_ranges = ranges[fov_filtered_indices]
        fov_filtered_angles = angles[fov_filtered_indices]

        # Step 2: Convert filtered polar coordinates to Cartesian coordinates
        # Transform from (range, angle) to (x, y) coordinates
        # Forward/backward axis
        x = fov_filtered_ranges * np.cos(fov_filtered_angles)
        y = fov_filtered_ranges * \
            np.sin(fov_filtered_angles)  # Left/right axis
        points = np.column_stack((x, y))

        # Step 3: Filter by distance (within 5 meters for racing relevance)
        distance_threshold = 5.0  # Maximum detection range (meters)
        distance_filtered_indices = np.where(
            (np.abs(x) <= distance_threshold) & (
                np.abs(y) <= distance_threshold)
        )[0]
        distance_filtered_points = points[distance_filtered_indices]

        # Step 4: Cluster points to detect objects using DBSCAN
        # Group nearby points that likely belong to the same object
        clustering = DBSCAN(eps=0.2, min_samples=3).fit(
            distance_filtered_points)
        labels = clustering.labels_

        # Step 5: Filter clusters by size (ignore small/insignificant clusters)
        # Only consider clusters with more than 10 points as significant objects
        large_cluster_indices = [
            i for i, label in enumerate(labels)
            if label != -1 and np.sum(labels == label) > 10
        ]
        large_cluster_points = distance_filtered_points[large_cluster_indices]

        # Step 6: Classify objects by spatial position
        # Define thresholds for spatial classification
        side_threshold = 1.0   # Objects with |x| < 1.0 m are considered "on the side"
        front_threshold = 1.0  # Objects with |y| < 1.0 m are considered "in front"

        objects_in_front = []
        objects_on_side = []

        # Classify each significant point based on its position
        for point in large_cluster_points:
            x, y = point
            if abs(y) < front_threshold:
                # Object is directly ahead/behind
                objects_in_front.append(point)
            elif abs(x) < side_threshold:
                objects_on_side.append(point)   # Object is to the left/right

        # Step 7: Log detected objects with position information
        # Report objects detected in front of the robot
        for i, point in enumerate(objects_in_front):
            x, y = point
            self.get_logger().info(
                f"Object {i + 1} in front at: (x={x:.2f}m, y={y:.2f}m)"
            )

        # Report objects detected on the sides of the robot
        for i, point in enumerate(objects_on_side):
            x, y = point
            self.get_logger().info(
                f"Object {i + 1} on the side at: (x={x:.2f}m, y={y:.2f}m)"
            )


def main(args=None):
    """
    Main entry point for the basic LiDAR processor node.

    Initializes the ROS2 system, creates and runs the LiDAR processor node,
    and handles graceful shutdown when the program is terminated.

    Args:
        args: Command line arguments passed to ROS2 initialization
    """
    rclpy.init(args=args)
    lidar_processor = LidarProcessor()

    try:
        rclpy.spin(lidar_processor)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
