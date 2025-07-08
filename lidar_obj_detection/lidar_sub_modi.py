# Dependencies that need to be installed:
# pip3 install numpy
# pip3 install hdbscan
# sudo apt install ros-humble-tf2-geometry-msgs
# sudo apt install ros-humble-sensor-msgs
# sudo apt install ros-humble-nav-msgs
# sudo apt install ros-humble-visualization-msgs
# sudo apt install ros-humble-tf2-ros
# sudo apt install ros-humble-tf2-ros
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
    def __init__(self):
        super().__init__('lidar_cluster_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            5
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        self.marker_pub = self.create_publisher(Marker, '/clusters', 5)
        self.marker_id = 0
        self.angle_thresh = 2.0
        self.prev_centers = []

        # Map data
        self.map_data = None
        self.map_info = None

        # TF2 for transforming points
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Parameters
        self.map_occupancy_threshold = 60  # Occupancy value for static cells
        self.downsample_factor = 4
        self.outlier_diff_threshold = 0.1  # For range outlier filtering

    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_info = msg.info
        self.get_logger().info(f"Occupancy grid map loaded: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution}")
    
    def transform_point(self, point, source_frame, target_frame):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            point_stamped = tf2_geometry_msgs.PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point.x, point_stamped.point.y, point_stamped.point.z = point[0], point[1], 0.0
            transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return np.array([transformed.point.x, transformed.point.y])
        except Exception as e:
            self.get_logger().warn(f"Transform failed: {e}")
            return None

    def scan_callback(self, msg):
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
                self.transform_point(p, "/laser", self.map_info.header.frame_id) 
                for p in points
            ])
            points_map = points_map[points_map != None]  # Remove failed transforms
            if len(points_map) == 0:
                self.get_logger().debug("No points after transform")
                return

            # Convert to grid indices
            origin = np.array([self.map_info.origin.position.x, self.map_info.origin.position.y])
            resolution = self.map_info.resolution
            indices = ((points_map - origin) / resolution).astype(int)

            # Check valid indices and occupancy
            valid_indices = (
                (0 <= indices[:, 0]) & (indices[:, 0] < self.map_info.width) &
                (0 <= indices[:, 1]) & (indices[:, 1] < self.map_info.height)
            )
            dynamic_mask = np.ones(len(points), dtype=bool)
            valid_idx = np.where(valid_indices)[0]
            dynamic_mask[valid_idx] = self.map_data[indices[valid_idx, 1], indices[valid_idx, 0]] < self.map_occupancy_threshold

            points = points[dynamic_mask]
            if len(points) == 0:
                self.get_logger().debug("No dynamic points after map filtering")
                return

        # Cluster with HDBSCAN
        clustering = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=5).fit(points)
        labels = clustering.labels_

        # Process clusters
        unique_labels = np.unique(labels[labels != -1])
        if len(unique_labels) == 0:
            self.get_logger().debug("No clusters found")
            return

        # Vectorized cluster metrics
        centers = np.array([points[labels == lbl].mean(axis=0) for lbl in unique_labels])
        sizes = np.array([np.linalg.norm(points[labels == lbl].max(axis=0) - points[labels == lbl].min(axis=0)) for lbl in unique_labels])
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
                self.get_logger().debug(f"Position -> x: {center[0]:.2f}, y: {center[1]:.2f}, Size: {size:.2f} m, Distance: {distance:.2f} m")
                self.publish_marker(center[0], center[1], size, (1.0, 0.0, 0.0))

        # Update previous centers
        self.prev_centers = centers[valid_clusters].tolist() if np.any(valid_clusters) else []

        # Log performance
        elapsed = (time.time() - start_time) * 1000
        self.get_logger().debug(f"Callback time: {elapsed:.2f} ms, Clusters: {len(unique_labels)}, Noise points: {np.sum(labels == -1)}, Dynamic points: {len(points)}")

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