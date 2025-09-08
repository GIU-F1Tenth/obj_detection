#!/usr/bin/env python3
"""
Test data publisher for LiDAR Object Detection testing.

This node publishes synthetic LiDAR scan data with known objects
to test the detection and tracking algorithms.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math
from typing import List, Tuple


class TestDataPublisher(Node):
    """
    Node that publishes synthetic LiDAR data for testing.
    """

    def __init__(self):
        super().__init__('test_data_publisher')
        
        # Declare parameters
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('test_scenario', 'racing')
        self.declare_parameter('noise_level', 0.01)
        self.declare_parameter('num_rays', 360)
        
        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').value
        self.test_scenario = self.get_parameter('test_scenario').value
        self.noise_level = self.get_parameter('noise_level').value
        self.num_rays = self.get_parameter('num_rays').value
        
        # Create publisher
        self.scan_publisher = self.create_publisher(
            LaserScan,
            '/scan',
            10
        )
        
        # Create timer
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_scan)
        
        # Test state
        self.scan_count = 0
        self.objects = []  # List of (x, y, vx, vy) for moving objects
        
        # Initialize test scenario
        self.setup_test_scenario()
        
        self.get_logger().info(f'Test data publisher started with scenario: {self.test_scenario}')

    def setup_test_scenario(self):
        """Initialize the test scenario."""
        if self.test_scenario == 'racing':
            # F1TENTH racing scenario with opponents and walls
            self.objects = [
                (3.0, 0.0, 0.1, 0.0),    # Stationary opponent ahead
                (2.0, 1.5, 0.05, -0.02), # Moving opponent to the right
                (-1.0, 2.0, -0.08, 0.0), # Opponent behind and to the side
            ]
        elif self.test_scenario == 'obstacles':
            # Static obstacles scenario
            self.objects = [
                (1.5, 0.5, 0.0, 0.0),    # Static obstacle
                (2.5, -1.0, 0.0, 0.0),   # Another static obstacle
                (4.0, 0.0, 0.0, 0.0),    # Distant obstacle
            ]
        elif self.test_scenario == 'moving':
            # Multiple moving objects
            self.objects = [
                (2.0, 0.0, 0.1, 0.0),    # Moving forward
                (1.5, 1.5, 0.0, -0.1),   # Moving left
                (3.0, -1.0, -0.05, 0.05), # Moving diagonally
            ]
        elif self.test_scenario == 'noise':
            # Scenario with noise and false positives
            self.objects = [
                (2.0, 0.0, 0.0, 0.0),    # One real object
            ]
        else:
            # Default: single stationary object
            self.objects = [
                (2.0, 0.0, 0.0, 0.0),
            ]

    def update_objects(self, dt: float):
        """Update object positions based on their velocities."""
        for i, (x, y, vx, vy) in enumerate(self.objects):
            new_x = x + vx * dt
            new_y = y + vy * dt
            self.objects[i] = (new_x, new_y, vx, vy)

    def create_object_points(self, obj_x: float, obj_y: float, 
                           obj_width: float = 0.3) -> List[Tuple[float, float]]:
        """
        Create LiDAR points for a rectangular object.
        
        Args:
            obj_x, obj_y: Object center position
            obj_width: Object width (typical F1TENTH car width)
            
        Returns:
            List of (x, y) points representing the object
        """
        points = []
        
        # Create a simple rectangular object
        half_width = obj_width / 2
        
        # Front and back of the object
        for offset in [-half_width, 0, half_width]:
            points.append((obj_x + 0.15, obj_y + offset))  # Front
            points.append((obj_x - 0.15, obj_y + offset))  # Back
        
        return points

    def create_wall_points(self, wall_distance: float, 
                          start_angle: float, end_angle: float) -> List[Tuple[float, float]]:
        """Create points for a wall at given distance and angle range."""
        points = []
        angles = np.linspace(start_angle, end_angle, int((end_angle - start_angle) * 180 / np.pi))
        
        for angle in angles:
            # Add some variation to make wall more realistic
            distance = wall_distance + 0.1 * np.sin(angle * 5) + np.random.normal(0, 0.02)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            points.append((x, y))
        
        return points

    def add_noise_points(self, num_noise_points: int) -> List[Tuple[float, float]]:
        """Add random noise points to simulate measurement errors."""
        noise_points = []
        
        for _ in range(num_noise_points):
            # Random angle and distance
            angle = np.random.uniform(-np.pi, np.pi)
            distance = np.random.uniform(0.5, 8.0)
            
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            noise_points.append((x, y))
        
        return noise_points

    def points_to_ranges(self, points: List[Tuple[float, float]]) -> List[float]:
        """Convert (x, y) points to range measurements."""
        ranges = [float('inf')] * self.num_rays
        
        angle_min = -np.pi
        angle_max = np.pi
        angle_increment = (angle_max - angle_min) / self.num_rays
        
        for x, y in points:
            # Convert to polar coordinates
            range_val = math.sqrt(x*x + y*y)
            angle = math.atan2(y, x)
            
            # Find corresponding ray index
            if angle_min <= angle <= angle_max:
                ray_index = int((angle - angle_min) / angle_increment)
                if 0 <= ray_index < self.num_rays:
                    # Keep the closest measurement for each ray
                    if range_val < ranges[ray_index]:
                        ranges[ray_index] = range_val
        
        return ranges

    def publish_scan(self):
        """Publish a synthetic LiDAR scan."""
        # Update object positions
        dt = 1.0 / self.publish_rate
        self.update_objects(dt)
        
        # Collect all points
        all_points = []
        
        # Add object points
        for obj_x, obj_y, _, _ in self.objects:
            obj_points = self.create_object_points(obj_x, obj_y)
            all_points.extend(obj_points)
        
        # Add walls for racing scenario
        if self.test_scenario == 'racing':
            # Left wall
            left_wall = self.create_wall_points(1.2, np.pi/3, 2*np.pi/3)
            all_points.extend(left_wall)
            
            # Right wall
            right_wall = self.create_wall_points(1.0, -2*np.pi/3, -np.pi/3)
            all_points.extend(right_wall)
        
        # Add noise for noise scenario
        if self.test_scenario == 'noise':
            noise_points = self.add_noise_points(20)
            all_points.extend(noise_points)
        
        # Convert points to ranges
        ranges = self.points_to_ranges(all_points)
        
        # Add measurement noise
        for i in range(len(ranges)):
            if ranges[i] != float('inf'):
                ranges[i] += np.random.normal(0, self.noise_level)
                ranges[i] = max(0.1, ranges[i])  # Ensure positive range
        
        # Create and publish scan message
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'base_scan'
        
        scan.angle_min = -np.pi
        scan.angle_max = np.pi
        scan.angle_increment = 2 * np.pi / self.num_rays
        scan.time_increment = 0.0
        scan.scan_time = 1.0 / self.publish_rate
        
        scan.range_min = 0.1
        scan.range_max = 10.0
        
        scan.ranges = ranges
        scan.intensities = [1000.0 if r != float('inf') else 0.0 for r in ranges]
        
        self.scan_publisher.publish(scan)
        
        self.scan_count += 1
        if self.scan_count % 50 == 0:  # Log every 5 seconds at 10Hz
            self.get_logger().info(f'Published {self.scan_count} scans, objects at: {[(round(x,2), round(y,2)) for x,y,_,_ in self.objects]}')


def main(args=None):
    """Main entry point for the test data publisher."""
    rclpy.init(args=args)
    
    try:
        node = TestDataPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
