"""
Map-based Dynamic Obstacle Detection for F1TENTH Racing.

This module implements an advanced obstacle detection system that uses occupancy
grid maps to distinguish between static track features and dynamic obstacles.
It's particularly useful for detecting unexpected objects or opponents that 
don't appear in the pre-built track map.

Author: F1TENTH Object Detection Team
License: MIT
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import math
import numpy as np


class ObstacleDetector(Node):
    """
    ROS2 node for detecting dynamic obstacles using map-based filtering.

    This detector compares live LiDAR measurements against a known occupancy
    grid map to identify unexpected obstacles. It transforms LiDAR points
    to the map coordinate frame and checks if they fall in areas that should
    be free space according to the map.

    Key capabilities:
    - Map-based static obstacle filtering
    - Transform handling between coordinate frames
    - Dynamic obstacle identification
    - Real-time processing of LiDAR data

    Typical use cases:
    - Detecting racing opponents on track
    - Identifying debris or unexpected obstacles
    - Monitoring track conditions during autonomous racing
    """

    def __init__(self):
        """
        Initialize the map-based obstacle detector.

        Sets up subscriptions for map and LiDAR data, initializes the
        transform listener for coordinate frame conversions, and prepares
        the node for obstacle detection operations.
        """
        super().__init__('obstacle_detector')

        # Initialize data storage
        self.map = None  # Will store the occupancy grid map

        # Set up subscriptions
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_cb,
            10
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_cb,
            10
        )

        # Initialize transform handling for coordinate conversions
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def map_cb(self, msg):
        """
        Callback for receiving occupancy grid map data.

        Stores the incoming map for use in obstacle detection. The map
        represents the static environment with known obstacles and free space.

        Args:
            msg (OccupancyGrid): ROS2 occupancy grid message containing:
                - info: Map metadata (resolution, origin, dimensions)
                - data: Grid cell occupancy values (0=free, 100=occupied, -1=unknown)
        """
        self.map = msg

    def scan_cb(self, msg):
        """
        Process LiDAR scan data to detect dynamic obstacles using map comparison.

        This method implements the core obstacle detection algorithm:
        1. Verify that map data is available
        2. Obtain coordinate frame transformation (laser -> map)
        3. For each LiDAR measurement:
           - Convert to Cartesian coordinates in laser frame
           - Transform to map coordinate frame
           - Check corresponding map cell occupancy
           - Flag unexpected obstacles in free space areas

        Args:
            msg (LaserScan): ROS2 LaserScan message with LiDAR measurements
        """
        # Ensure we have map data before processing
        if self.map is None:
            return  # No map available yet

        # Attempt to get transform from laser frame to map frame
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map.header.frame_id,  # Target frame (map)
                msg.header.frame_id,       # Source frame (laser)
                rclpy.time.Time()          # Use latest available transform
            )
        except Exception:
            self.get_logger().warn("TF unavailable")
            return  # Cannot proceed without valid transform

        # Extract map parameters for coordinate calculations
        resolution = self.map.info.resolution        # Meters per grid cell
        origin = self.map.info.origin.position       # Map origin in world coordinates
        width = self.map.info.width                  # Map width in cells
        height = self.map.info.height                # Map height in cells

        # Convert map data to 2D array for efficient access
        data = np.array(self.map.data).reshape((height, width))

        # Process each LiDAR measurement
        obstacle_found = False
        angle = msg.angle_min  # Current beam angle

        for r in msg.ranges:
            # Skip invalid measurements
            if r < msg.range_min or r > msg.range_max:
                angle += msg.angle_increment
                continue

            # Convert polar coordinates (range, angle) to Cartesian (x, y)
            # in the laser coordinate frame
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            angle += msg.angle_increment

            # Transform point from laser frame to map frame
            map_x = tf.transform.translation.x + x
            map_y = tf.transform.translation.y + y

            # Convert world coordinates to grid cell indices
            mx = int((map_x - origin.x) / resolution)
            my = int((map_y - origin.y) / resolution)

            # Check if the grid indices are within map bounds
            if 0 <= mx < width and 0 <= my < height:
                val = data[my][mx]  # Get occupancy value at this location

                # Check for unexpected obstacle
                # val == 0 means free space in map, but LiDAR detected something
                if val == 0:
                    self.get_logger().warn("Unexpected obstacle detected!")
                    obstacle_found = True
                    break  # Stop processing once obstacle is found

        # Report results
        if not obstacle_found:
            self.get_logger().info("No unexpected obstacle.")


def main():
    """
    Main entry point for the map-based obstacle detector.

    Initializes ROS2, creates the detector node, and runs until shutdown.
    Provides clean exit handling for graceful termination.
    """
    rclpy.init()
    node = ObstacleDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
