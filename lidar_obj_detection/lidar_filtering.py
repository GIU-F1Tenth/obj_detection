#!/usr/bin/env python3
"""
LiDAR Data Filtering and Preprocessing for F1TENTH Racing.

This module provides advanced filtering techniques for LiDAR scan data to improve
object detection accuracy in racing environments. It implements multi-stage
filtering including smoothing, outlier removal, clustering, and adaptive bubble
filtering for path planning applications.

Author: F1TENTH Object Detection Team
License: MIT
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarFiltering(Node):
    """
    ROS2 node for comprehensive LiDAR data filtering and preprocessing.

    This node applies a multi-stage filtering pipeline to raw LiDAR data to
    improve data quality for downstream processing. The filtering stages include:

    1. Moving average smoothing to reduce measurement noise
    2. Outlier detection and removal to eliminate spurious readings
    3. Obstacle clustering to group related measurements
    4. Dynamic bubble filtering for path planning integration

    The filtered data maintains the same message structure as input LiDAR
    scans but with improved signal quality and reduced noise artifacts.

    Typical applications:
    - Preprocessing for object detection algorithms
    - Path planning data preparation
    - Racing environment perception enhancement
    """

    def __init__(self):
        """
        Initialize the LiDAR filtering node.

        Sets up ROS2 subscriber for raw LiDAR data and publisher for
        filtered results. Configures the filtering pipeline parameters
        optimized for F1TENTH racing applications.
        """
        super().__init__('lidar_filtering_node')

        # Subscribe to raw LiDAR scans
        self.sub_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publish filtered scans
        self.pub_filtered_scan = self.create_publisher(
            LaserScan,
            '/filtered_scan',
            10
        )

        # Initialize data storage
        self.ranges = []

    def scan_callback(self, msg):
        """
        Process incoming LiDAR scan data through the filtering pipeline.

        Applies the complete filtering pipeline to incoming scan data and
        publishes the filtered result. The pipeline preserves all original
        message metadata while improving the quality of range measurements.

        Args:
            msg (LaserScan): Input ROS2 LaserScan message containing:
                - ranges: Array of distance measurements
                - Header and timing information
                - Angular parameters (min, max, increment)
                - Range limits and other metadata
        """
        self.ranges = msg.ranges

        # Apply the filtering pipeline
        filtered_ranges = self.filter_scan(self.ranges)

        # Create a new LaserScan message with filtered ranges
        filtered_msg = LaserScan()

        # Copy all original message metadata
        filtered_msg.header = msg.header
        filtered_msg.angle_min = msg.angle_min
        filtered_msg.angle_max = msg.angle_max
        filtered_msg.angle_increment = msg.angle_increment
        filtered_msg.time_increment = msg.time_increment
        filtered_msg.scan_time = msg.scan_time
        filtered_msg.range_min = msg.range_min
        filtered_msg.range_max = msg.range_max
        filtered_msg.intensities = msg.intensities

        # Apply filtered range data
        filtered_msg.ranges = filtered_ranges

        # Publish the filtered scan
        self.pub_filtered_scan.publish(filtered_msg)

    def filter_scan(self, ranges):
        """
        Apply comprehensive filtering pipeline to LiDAR range data.

        Implements a four-stage filtering process designed for F1TENTH racing:

        Stage 1 - Moving Average Smoothing:
        Applies a 5-point moving average filter to reduce measurement noise
        while preserving important geometric features.

        Stage 2 - Outlier Removal:
        Identifies and removes spurious measurements that differ significantly
        from neighboring readings, replacing them with interpolated values.

        Stage 3 - Obstacle Clustering:
        Groups consecutive measurements that likely belong to the same physical
        object, enabling object-level processing in later stages.

        Stage 4 - Dynamic Bubble Filtering:
        Applies distance-dependent safety bubbles around detected obstacles,
        useful for path planning and collision avoidance applications.

        Args:
            ranges (list): Input array of LiDAR range measurements (meters)

        Returns:
            list: Filtered range measurements with same length as input
        """
        if not ranges:
            return []  # Handle empty input gracefully

        # Stage 1: Moving Average Smoothing (5-ray window)
        # Reduces measurement noise while preserving object boundaries
        smoothed = []
        for i in range(len(ranges)):
            # Define sliding window boundaries
            # Window start (clamped to array bounds)
            start = max(0, i - 2)
            # Window end (clamped to array bounds)
            end = min(len(ranges), i + 3)

            # Calculate average within the window
            window = ranges[start:end]
            smoothed.append(sum(window) / len(window))

        # Stage 2: Outlier Removal
        # Detect and correct measurements that deviate significantly from neighbors
        filtered = smoothed.copy()
        # Maximum allowed difference from neighbors (meters)
        outlier_threshold = 5.0

        for i in range(1, len(smoothed) - 1):
            current_reading = smoothed[i]
            prev_reading = smoothed[i-1]
            next_reading = smoothed[i+1]

            # Check if current reading is an outlier compared to neighbors
            if (abs(current_reading - prev_reading) > outlier_threshold and
                    abs(current_reading - next_reading) > outlier_threshold):
                # Replace outlier with interpolated value
                filtered[i] = (prev_reading + next_reading) / 2

        # Stage 3: Obstacle Clustering
        # Group consecutive measurements that belong to the same physical object
        clusters = []
        current_cluster = []
        # Maximum distance between points in same cluster (meters)
        proximity_threshold = 0.2

        for i in range(len(filtered)):
            if not current_cluster:
                # Start new cluster
                current_cluster.append((i, filtered[i]))
            elif abs(filtered[i] - current_cluster[-1][1]) < proximity_threshold:
                # Add to existing cluster
                current_cluster.append((i, filtered[i]))
            else:
                # End current cluster and start new one
                clusters.append(current_cluster)
                current_cluster = [(i, filtered[i])]

        # Don't forget the last cluster
        if current_cluster:
            clusters.append(current_cluster)

        # Stage 4: Dynamic Bubble Filtering
        # Apply distance-dependent safety margins around obstacles
        final_ranges = filtered.copy()

        for cluster in clusters:
            if not cluster:
                continue  # Skip empty clusters

            # Calculate cluster characteristics
            avg_range = sum(r[1] for r in cluster) / len(cluster)

            # Apply bubble filtering only to nearby obstacles
            if avg_range < 3.0:
                # Calculate bubble size based on distance (closer = larger bubble)
                base_bubble_angle = 0.3  # Base bubble size (radians)
                # Distance-dependent adjustment
                distance_factor = (avg_range - 1) * 0.1
                bubble_angle = max(0.3, base_bubble_angle + distance_factor)

                # Convert bubble angle to number of LiDAR rays
                # Typical LiDAR angular resolution (radians)
                angular_resolution = 0.0087
                bubble_rays = int(bubble_angle / angular_resolution)

                # Apply bubble around each point in the cluster
                for i in range(len(cluster)):
                    idx = cluster[i][0]  # Original index in ranges array

                    # Apply bubble to surrounding rays
                    bubble_start = max(0, idx - bubble_rays)
                    bubble_end = min(len(final_ranges), idx + bubble_rays + 1)

                    for j in range(bubble_start, bubble_end):
                        # Reduce range to create safety margin
                        final_ranges[j] = min(final_ranges[j], avg_range)

        return final_ranges


def main(args=None):
    """
    Main entry point for the LiDAR filtering node.

    Initializes ROS2, creates the filtering node, and runs the filtering
    pipeline until shutdown. Handles graceful termination and cleanup.

    Args:
        args: Command line arguments for ROS2 initialization
    """
    rclpy.init(args=args)
    node = LidarFiltering()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
