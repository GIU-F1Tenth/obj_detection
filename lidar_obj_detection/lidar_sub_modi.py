"""
Advanced LiDAR Clustering with Map Integration for F1TENTH Racing.

This module implements a high-performance LiDAR clustering system that integrates
occupancy grid maps to filter static obstacles and focuses on dynamic object
detection. It uses HDBSCAN clustering with temporal consistency checking and
coordinate frame transformations for robust racing opponent detection.

Dependencies:
    - numpy: Numerical computations
    - hdbscan: Hierarchical density-based clustering  
    - ROS2 packages: tf2-geometry-msgs, sensor-msgs, nav-msgs, visualization-msgs, tf2-ros

Author: F1TENTH Object Detection Team  
License: MIT
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid
import numpy as np
import hdbscan
import math
import time
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import tf2_geometry_msgs  # For transforming points


class LidarClusterNode(Node):
    """
    Advanced LiDAR clustering node with map integration and temporal filtering.

    This node provides state-of-the-art object detection for F1TENTH racing by:
    - Integrating occupancy grid maps to filter static track features
    - Using HDBSCAN for superior clustering performance
    - Implementing temporal consistency checking across frames
    - Applying coordinate transformations between reference frames
    - Optimizing performance for real-time racing applications

    The node is specifically tuned for detecting F1TENTH racing opponents
    while filtering out static track elements like walls and barriers.

    Attributes:
        marker_id (int): Counter for unique visualization marker IDs
        angle_thresh (float): Angular threshold for forward object detection (degrees)
        prev_centers (list): Previous detection centers for temporal consistency
        map_data (numpy.ndarray): Occupancy grid map data for static filtering
        map_info (nav_msgs.MapMetaData): Map metadata including resolution and origin
        tf_buffer (tf2_ros.Buffer): Transform buffer for coordinate conversions
        tf_listener (tf2_ros.TransformListener): Transform listener for frame updates
    """

    def __init__(self):
        """
        Initialize the advanced LiDAR clustering node.

        Sets up all necessary subscriptions, publishers, and internal data structures
        for high-performance object detection with map integration.
        """
        super().__init__('lidar_cluster_node')

        # Primary LiDAR data subscription
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            5  # Reduced queue size for lower latency
        )

        # Occupancy grid map subscription for static obstacle filtering
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # Visualization marker publisher
        self.marker_pub = self.create_publisher(Marker, '/clusters', 5)

        # Detection and tracking parameters
        self.marker_id = 0
        self.angle_thresh = 2.0  # Narrow forward detection cone (degrees)
        self.prev_centers = []   # Previous frame detections for consistency

        # Map integration components
        self.map_data = None  # Occupancy grid data array
        self.map_info = None  # Map metadata (resolution, origin, etc.)

        # Transform handling for coordinate frame conversions
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Advanced filtering parameters (remove duplicate)
        self.map_occupancy_threshold = 60  # Occupancy threshold for static objects
        self.downsample_factor = 4         # Data downsampling for performance
        self.outlier_diff_threshold = 0.1  # Range difference threshold for outliers

    def map_callback(self, msg):
        """
        Process incoming occupancy grid map data.

        Stores the occupancy grid map for use in static obstacle filtering.
        The map is used to distinguish between static track features (walls, barriers)
        and dynamic objects (racing opponents) during LiDAR processing.

        Args:
            msg (OccupancyGrid): ROS2 occupancy grid message containing:
                - info: Map metadata (resolution, origin, dimensions)
                - data: Flattened array of occupancy values (0=free, 100=occupied, -1=unknown)
        """
        self.map_data = np.array(msg.data).reshape(
            msg.info.height, msg.info.width)
        self.map_info = msg.info
        self.get_logger().info(
            f"Occupancy grid map loaded: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution}")

    def transform_point(self, point, source_frame, target_frame):
        """
        Transform a 2D point between coordinate frames using TF2.

        Converts points from one coordinate frame to another using the ROS2
        transform system. This is essential for comparing LiDAR measurements
        (in laser frame) with map data (in map frame).

        Args:
            point (array-like): 2D point [x, y] in source frame
            source_frame (str): Source coordinate frame name
            target_frame (str): Target coordinate frame name

        Returns:
            numpy.ndarray or None: Transformed point [x, y] in target frame,
                                 or None if transformation fails
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time())
            point_stamped = tf2_geometry_msgs.PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point.x, point_stamped.point.y, point_stamped.point.z = point[
                0], point[1], 0.0
            transformed = tf2_geometry_msgs.do_transform_point(
                point_stamped, transform)
            return np.array([transformed.point.x, transformed.point.y])
        except Exception as e:
            self.get_logger().warn(f"Transform failed: {e}")
            return None

    def scan_callback(self, msg):
        """
        Advanced LiDAR scan processing with map integration and temporal filtering.

        This is the main processing pipeline that implements:
        1. Data preprocessing and outlier filtering
        2. Coordinate transformation and downsampling
        3. Map-based static obstacle filtering
        4. HDBSCAN clustering for object detection
        5. Geometric filtering for F1TENTH robot detection
        6. Temporal consistency checking
        7. Performance monitoring and optimization

        The method is optimized for real-time performance in racing environments
        and focuses specifically on detecting dynamic racing opponents.

        Args:
            msg (LaserScan): Input LiDAR scan data with range measurements
        """
        start_time = time.time()
        self.marker_id = 0

        # Preprocess: Extract and filter ranges
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)
        valid = (ranges > msg.range_min) & (ranges < msg.range_max)

        # Outlier filter
        ranges_diff = np.abs(np.diff(ranges))
        valid[:-1] &= ranges_diff < self.outlier_diff_threshold
        angles = angles[valid]
        ranges = ranges[valid]

        # Downsample
        angles = angles[::self.downsample_factor]
        ranges = ranges[::self.downsample_factor]

        # Convert to Cartesian
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        points = np.stack((xs, ys), axis=1)

        if len(points) == 0:
            self.get_logger().debug("No valid points after filtering")
            return

        # Filter static map points using occupancy grid
        if self.map_data is not None and self.map_info is not None:
            # Transform points to map frame (if needed)
            points_map = np.array([
                self.transform_point(
                    p, "/laser", self.map_info.header.frame_id)
                for p in points
            ])
            # Remove failed transforms
            points_map = points_map[points_map != None]
            if len(points_map) == 0:
                self.get_logger().debug("No points after transform")
                return

            # Convert to grid indices
            origin = np.array([self.map_info.origin.position.x,
                              self.map_info.origin.position.y])
            resolution = self.map_info.resolution
            indices = ((points_map - origin) / resolution).astype(int)

            # Check valid indices and occupancy
            valid_indices = (
                (0 <= indices[:, 0]) & (indices[:, 0] < self.map_info.width) &
                (0 <= indices[:, 1]) & (indices[:, 1] < self.map_info.height)
            )
            dynamic_mask = np.ones(len(points), dtype=bool)
            valid_idx = np.where(valid_indices)[0]
            dynamic_mask[valid_idx] = self.map_data[indices[valid_idx, 1],
                                                    indices[valid_idx, 0]] < self.map_occupancy_threshold

            points = points[dynamic_mask]
            if len(points) == 0:
                self.get_logger().debug("No dynamic points after map filtering")
                return

        # Cluster with HDBSCAN
        clustering = hdbscan.HDBSCAN(
            min_cluster_size=8, min_samples=5).fit(points)
        labels = clustering.labels_

        # Process clusters
        unique_labels = np.unique(labels[labels != -1])
        if len(unique_labels) == 0:
            self.get_logger().debug("No clusters found")
            return

        # Vectorized cluster metrics
        centers = np.array([points[labels == lbl].mean(axis=0)
                           for lbl in unique_labels])
        sizes = np.array([np.linalg.norm(points[labels == lbl].max(
            axis=0) - points[labels == lbl].min(axis=0)) for lbl in unique_labels])
        distances = np.linalg.norm(centers, axis=1)
        angles = np.arctan2(centers[:, 1], centers[:, 0])

        # Filter clusters for opponent car
        valid_clusters = (
            (0.2 < sizes) & (sizes < 0.25) &
            (1.0 < distances) & (distances < 4.0) &
            (centers[:, 0] > 0) &
            (np.abs(angles) < np.radians(self.angle_thresh))
        )

        # Publish markers for valid clusters
        for idx in np.where(valid_clusters)[0]:
            center = centers[idx]
            size = sizes[idx]
            distance = distances[idx]

            # Temporal consistency check
            if not self.prev_centers or any(np.linalg.norm(center - prev) < 0.1 for prev in self.prev_centers):
                self.get_logger().warn(">> Likely another robot nearby!")
                self.get_logger().debug(
                    f"Position -> x: {center[0]:.2f}, y: {center[1]:.2f}, Size: {size:.2f} m, Distance: {distance:.2f} m")
                self.publish_marker(
                    center[0], center[1], size, (1.0, 0.0, 0.0))

        # Update previous centers
        self.prev_centers = centers[valid_clusters].tolist(
        ) if np.any(valid_clusters) else []

        # Log performance
        elapsed = (time.time() - start_time) * 1000
        self.get_logger().debug(
            f"Callback time: {elapsed:.2f} ms, Clusters: {len(unique_labels)}, Noise points: {np.sum(labels == -1)}, Dynamic points: {len(points)}")

    def publish_marker(self, x, y, size, color):
        """
        Publish a visualization marker for a detected racing opponent.

        Creates and publishes an RViz marker to visualize detected racing opponents.
        The marker appears as a colored sphere in the ego vehicle's laser frame,
        providing real-time feedback for debugging and monitoring.

        Args:
            x (float): Object x-coordinate in laser frame (meters, forward/backward)
            y (float): Object y-coordinate in laser frame (meters, left/right)
            size (float): Estimated object size/diameter (meters)
            color (tuple): RGB color values (r, g, b) each in range [0, 1]
        """
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
    """
    Main entry point for the advanced LiDAR clustering node.

    Initializes the ROS2 system, creates the advanced clustering node with
    map integration capabilities, and runs the detection system until shutdown.
    Handles graceful cleanup and performance monitoring.
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
