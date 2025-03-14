import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
from sklearn.cluster import DBSCAN
from collections import deque
import math

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        # Subscribe to the LiDAR topic
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        self.subscription  # Prevent unused variable warning
        
        # Parameters for filtering
        self.distance_threshold = 5.0  # Max detection range in meters
        self.min_distance = 0.05      # Min detection range (ignore very close points)
        
        # Improved DBSCAN parameters
        self.eps = 0.15               # Reduced cluster distance for better separation
        self.min_samples = 5          # Increased min samples for more robust clusters
        self.min_cluster_size = 8     # Minimum points to consider a valid object
        
        # Object tracking
        self.max_tracked_objects = 10
        self.object_history = []      # List to store object history
        self.history_length = 3       # Number of frames to track objects
        
        # Classification thresholds
        self.front_threshold = 1.2    # Objects with |y| < threshold are in front
        self.side_threshold = 1.2     # Objects with |x| < threshold are on the side
        
        # Create publishers for detected objects
        self.front_objects_pub = self.create_publisher(
            LaserScan, 
            '/front_objects', 
            10
        )
        self.side_objects_pub = self.create_publisher(
            LaserScan, 
            '/side_objects', 
            10
        )
        
        self.get_logger().info('LiDAR Object Detection Node Initialized')

    def filter_invalid_readings(self, ranges, angles):
        """Filter out invalid readings (inf, NaN, too close)"""
        valid_indices = np.where(
            (~np.isinf(ranges)) & 
            (~np.isnan(ranges)) & 
            (ranges > self.min_distance)
        )[0]
        return ranges[valid_indices], angles[valid_indices]

    def get_cluster_centroids(self, clusters, labels):
        """Calculate centroids for each cluster"""
        unique_labels = np.unique(labels)
        centroids = []
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            # Get points in this cluster
            cluster_points = clusters[labels == label]
            
            # Calculate centroid
            centroid = np.mean(cluster_points, axis=0)
            
            # Calculate cluster size (number of points)
            cluster_size = len(cluster_points)
            
            # Store centroid and size
            centroids.append({
                'position': centroid,
                'size': cluster_size,
                'points': cluster_points
            })
            
        return centroids

    def track_objects(self, current_objects):
        """Track objects across frames for stability"""
        if not self.object_history:
            # First frame, initialize history
            self.object_history = [current_objects]
            return current_objects
            
        # Add current objects to history
        self.object_history.append(current_objects)
        
        # Keep only recent history
        if len(self.object_history) > self.history_length:
            self.object_history.pop(0)
            
        # If we have enough history, filter objects for stability
        if len(self.object_history) >= 2:
            stable_objects = []
            
            # Check if current objects appear in previous frames
            for obj in current_objects:
                position = obj['position']
                
                # Object is considered stable if it appears in most frames
                stability_count = 1  # Current frame
                
                # Check previous frames (excluding the current one)
                for prev_frame in self.object_history[:-1]:
                    for prev_obj in prev_frame:
                        prev_position = prev_obj['position']
                        
                        # Calculate distance between current and previous object
                        distance = np.linalg.norm(position - prev_position)
                        
                        # If close enough, consider it the same object
                        if distance < 0.3:  # Threshold for same object
                            stability_count += 1
                            break
                
                # If object appears in enough frames, consider it stable
                stability_threshold = max(1, len(self.object_history) // 2)
                if stability_count >= stability_threshold:
                    stable_objects.append(obj)
                    
            return stable_objects
        
        return current_objects

    def classify_objects(self, objects):
        """Classify objects as front or side with improved logic"""
        objects_in_front = []
        objects_on_side = []
        
        for obj in objects:
            x, y = obj['position']
            size = obj['size']
            points = obj['points']
            
            # Calculate object width and length
            if len(points) >= 3:
                # Find min/max x and y coordinates
                min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
                min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
                
                width = max_x - min_x
                length = max_y - min_y
                
                # Calculate distance from origin
                distance = math.sqrt(x**2 + y**2)
                
                # Improved classification logic
                if abs(y) < self.front_threshold:
                    # Object is in front
                    objects_in_front.append({
                        'position': (x, y),
                        'size': size,
                        'width': width,
                        'length': length,
                        'distance': distance
                    })
                elif abs(x) < self.side_threshold:
                    # Object is on the side
                    objects_on_side.append({
                        'position': (x, y),
                        'size': size,
                        'width': width,
                        'length': length,
                        'distance': distance
                    })
        
        return objects_in_front, objects_on_side

    def listener_callback(self, msg):
        # Extract LiDAR data
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        
        # Step 1: Generate angle array and filter by FOV
        angles = np.arange(angle_min, angle_min + len(ranges) * angle_increment, angle_increment)
        fov_min = -np.deg2rad(135)  # -135° (leftmost angle)
        fov_max = np.deg2rad(135)   # +135° (rightmost angle)
        
        fov_filtered_indices = np.where((angles >= fov_min) & (angles <= fov_max))[0]
        fov_filtered_ranges = ranges[fov_filtered_indices]
        fov_filtered_angles = angles[fov_filtered_indices]
        
        # Step 2: Filter invalid readings
        filtered_ranges, filtered_angles = self.filter_invalid_readings(
            fov_filtered_ranges, fov_filtered_angles
        )
        
        # Skip processing if not enough valid points
        if len(filtered_ranges) < self.min_samples:
            self.get_logger().warn('Not enough valid LiDAR points for processing')
            return
            
        # Step 3: Convert filtered polar coordinates to Cartesian
        x = filtered_ranges * np.cos(filtered_angles)
        y = filtered_ranges * np.sin(filtered_angles)
        points = np.column_stack((x, y))
        
        # Step 4: Filter by distance
        distance_filtered_indices = np.where(
            (np.abs(x) <= self.distance_threshold) & 
            (np.abs(y) <= self.distance_threshold)
        )[0]
        distance_filtered_points = points[distance_filtered_indices]
        
        # Skip if no points after filtering
        if len(distance_filtered_points) < self.min_samples:
            self.get_logger().warn('Not enough points after distance filtering')
            return
            
        # Step 5: Cluster points with improved parameters
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        ).fit(distance_filtered_points)
        labels = clustering.labels_
        
        # Step 6: Get cluster centroids and filter small clusters
        centroids = self.get_cluster_centroids(distance_filtered_points, labels)
        valid_centroids = [c for c in centroids if c['size'] >= self.min_cluster_size]
        
        # Step 7: Track objects across frames
        tracked_objects = self.track_objects(valid_centroids)
        
        # Step 8: Classify objects
        objects_in_front, objects_on_side = self.classify_objects(tracked_objects)
        
        # Log the detected objects with improved information
        for i, obj in enumerate(objects_in_front):
            x, y = obj['position']
            width = obj.get('width', 0)
            length = obj.get('length', 0)
            distance = obj.get('distance', 0)
            
            self.get_logger().info(
                f"Object {i+1} in front: position=({x:.2f}m, {y:.2f}m), "
                f"size={obj['size']} points, width={width:.2f}m, "
                f"length={length:.2f}m, distance={distance:.2f}m"
            )
        
        for i, obj in enumerate(objects_on_side):
            x, y = obj['position']
            width = obj.get('width', 0)
            length = obj.get('length', 0)
            distance = obj.get('distance', 0)
            
            self.get_logger().info(
                f"Object {i+1} on side: position=({x:.2f}m, {y:.2f}m), "
                f"size={obj['size']} points, width={width:.2f}m, "
                f"length={length:.2f}m, distance={distance:.2f}m"
            )

def main(args=None):
    rclpy.init(args=args)
    lidar_processor = LidarProcessor()
    rclpy.spin(lidar_processor)
    lidar_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 