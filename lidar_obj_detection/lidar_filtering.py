#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class lidarFiltering(Node):
    def __init__(self):
        super().__init__('lidar_filtering_node')
        # Subscribe to raw LiDAR scans
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        # Publish filtered scans
        self.pub_filtered_scan = self.create_publisher(
            LaserScan, '/filtered_scan', 10)
        self.ranges = []

    def scan_callback(self, msg):
        self.ranges = msg.ranges
        # Filter the scans
        filtered_ranges = self.filter_scan(self.ranges)
        # Create a new LaserScan message with filtered ranges
        filtered_msg = LaserScan()
        filtered_msg.header = msg.header
        filtered_msg.angle_min = msg.angle_min
        filtered_msg.angle_max = msg.angle_max
        filtered_msg.angle_increment = msg.angle_increment
        filtered_msg.time_increment = msg.time_increment
        filtered_msg.scan_time = msg.scan_time
        filtered_msg.range_min = msg.range_min
        filtered_msg.range_max = msg.range_max
        filtered_msg.ranges = filtered_ranges
        filtered_msg.intensities = msg.intensities
        # Publish the filtered scan
        self.pub_filtered_scan.publish(filtered_msg)

    def filter_scan(self, ranges):
        if not ranges:
            return []
        
        # Step 1: Moving Average Smoothing (5-ray window)
        smoothed = []
        for i in range(len(ranges)):
            start = max(0, i - 2)
            end = min(len(ranges), i + 3)
            window = ranges[start:end]
            smoothed.append(sum(window) / len(window))
        
        # Step 2: Outlier Removal
        filtered = smoothed.copy()
        for i in range(1, len(smoothed) - 1):
            if abs(smoothed[i] - smoothed[i-1]) > 5.0 and abs(smoothed[i] - smoothed[i+1]) > 5.0:
                filtered[i] = (smoothed[i-1] + smoothed[i+1]) / 2
        
        # Step 3: Obstacle Clustering
        clusters = []
        current_cluster = []
        for i in range(len(filtered)):
            if not current_cluster:
                current_cluster.append((i, filtered[i]))
            elif abs(filtered[i] - current_cluster[-1][1]) < 0.2:
                current_cluster.append((i, filtered[i]))
            else:
                clusters.append(current_cluster)
                current_cluster = [(i, filtered[i])]
        if current_cluster:
            clusters.append(current_cluster)
        
        # Step 4: Dynamic Bubble Adjustment
        final_ranges = filtered.copy()
        for cluster in clusters:
            if not cluster:
                continue
            avg_range = sum(r[1] for r in cluster) / len(cluster)
            if avg_range < 3.0:
                bubble_angle = max(0.3, 0.3 + (avg_range - 1) * 0.1)
                bubble_rays = int(bubble_angle / 0.0087)
                for i in range(len(cluster)):
                    idx = cluster[i][0]
                    for j in range(max(0, idx - bubble_rays), min(len(final_ranges), idx + bubble_rays + 1)):
                        final_ranges[j] = min(final_ranges[j], avg_range)
        
        return final_ranges

def main(args=None):
    rclpy.init(args=args)
    node = lidarFiltering()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()