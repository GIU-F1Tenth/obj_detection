# F1TENTH LiDAR Processor (Collaborative Edition)  
# Contributors: Salma & Hatem  
# Combines LiDAR processing with obstacle classification for racing

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
import numpy as np
from sklearn.cluster import DBSCAN

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        
        # ---------------------------
        # Original Subscription Setup 🛠️ (Salma)
        # ---------------------------

        # Subscribe to the LiDAR topic (ORIGINAL)
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',   # Change to '/echoes' if using multi-echo (ORIGINAL)
            self.listener_callback,
            10)
        self.subscription  # Prevent unused variable warning (ORIGINAL)
        
        # ---------------------------
        # New ROS Publishers 🚀 (Hatem)
        # ---------------------------
        self.obstacle_pub = self.create_publisher(Detection3DArray, '/racing_obstacles', 10)
        self.wall_pub = self.create_publisher(Detection3DArray, '/track_walls', 10)
        
        # ---------------------------
        # Tuned F1TENTH Parameters 📊 (Hatem's scaling + Salma's base)
        # ---------------------------
        self.cluster_eps = 0.15  # Scaled for 1/10th cars (Original: 0.2)
        self.cluster_min_samples = 3  # Common ground value
        self.car_length = 0.4  # Scaled target size (Original: 1m concept)
        self.track_width = 2.0  # Track-specific constraint

    def listener_callback(self, msg):
        try:
            # ---------------------------
            # Core Coordinate Conversion 🔄 (Salma)
            # ---------------------------

            # Extract LiDAR data (ORIGINAL)
            ranges = np.array(msg.ranges)  # List of distances (ORIGINAL)
            angle_min = msg.angle_min      # Minimum angle (radians) (ORIGINAL)
            angle_increment = msg.angle_increment  # Angle between measurements (radians) (ORIGINAL)
            angles = np.arange(angle_min, angle_min + len(ranges)*angle_increment, angle_increment)
            
            # ---------------------------
            # Enhanced Validation Check ✅ (Hatem)
            # ---------------------------
            valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
            ranges = ranges[valid_mask]
            angles = angles[valid_mask]

            # ---------------------------
            # FOV Filtering Magic 🎯 (Salma)
            # ---------------------------

            # Step 1: Filter by the LiDAR's 270° FOV (ORIGINAL)
            # Original Salma's implementation:
            # fov_min = -np.deg2rad(135)  # -135° (leftmost angle) (ORIGINAL)
            # fov_max = np.deg2rad(135)   # +135° (rightmost angle) (ORIGINAL)
            # fov_filtered_indices = np.where((angles >= fov_min) & (angles <= fov_max))[0]
            # fov_filtered_ranges = ranges[fov_filtered_indices]
            # fov_filtered_angles = angles[fov_filtered_indices]
            fov_filter = (angles >= -np.deg2rad(135)) & (angles <= np.deg2rad(135))
            ranges = ranges[fov_filter]
            angles = angles[fov_filter]

            # ---------------------------
            # Optimized Cartesian Conversion 🤖 (Salma + Hatem)
            # ---------------------------
            
            # Step 2: Convert filtered polar coordinates to Cartesian coordinates (ORIGINAL)
            # Original Salma's implementation:
            # x = fov_filtered_ranges * np.cos(fov_filtered_angles)
            # y = fov_filtered_ranges * np.sin(fov_filtered_angles)
            # points = np.column_stack((x, y))
            x = ranges * np.cos(angles)  # Forward/backward (x-axis) (ORIGINAL)
            y = ranges * np.sin(angles)  # Left/right (y-axis) (ORIGINAL)
            points = np.column_stack((x, y))

            # ---------------------------
            # Track-Aware Filtering 🏁 (Hatem)
            # ---------------------------
            
            # Step 3: Filter by distance (e.g., within 5 meters along x and y axes) (ORIGINAL)
            # Original Salma's implementation:
            # distance_threshold = 5.0  # Adjust as needed
            # distance_filtered_indices = np.where(
            #     (np.abs(x) <= distance_threshold) & (np.abs(y) <= distance_threshold)
            # )[0]
            # distance_filtered_points = points[distance_filtered_indices]
            track_filter = (np.abs(y) < self.track_width/2)
            points = points[track_filter]

            # ---------------------------
            # Collaborative Clustering 🤝 (Salma's method + Hatem's params)
            # ---------------------------

            # Step 4: Cluster points to detect objects (ORIGINAL)
            # Original Salma's implementation:
            # clustering = DBSCAN(eps=0.2, min_samples=3).fit(distance_filtered_points)
            # labels = clustering.labels_
            clustering = DBSCAN(eps=self.cluster_eps, 
                            min_samples=self.cluster_min_samples).fit(points)
            labels = clustering.labels_

            # ---------------------------
            # Structured Message Creation 📦 (Hatem)
            # ---------------------------

            # Step 5: Filter by size (e.g., ignore small clusters) (ORIGINAL)
            # Original Salma's implementation:
            # large_cluster_indices = [i for i, label in enumerate(labels) if label != -1 and np.sum(labels == label) > 10]
            # large_cluster_points = distance_filtered_points[large_cluster_indices]
            obstacles = Detection3DArray()
            walls = Detection3DArray()
            obstacles.header.stamp = self.get_clock().now().to_msg()
            obstacles.header.frame_id = 'lidar_frame'
            walls.header = obstacles.header

            # Initialize empty lists for detections
            obstacles.detections = []
            walls.detections = []

            for cluster_id in np.unique(labels):
                if cluster_id == -1:
                    continue  # Mutual noise exclusion
                
                cluster_pts = points[labels == cluster_id]
                min_pt = np.min(cluster_pts, axis=0)
                max_pt = np.max(cluster_pts, axis=0)
                
                # ---------------------------
                # Size Analysis Duo 📏 (Hatem's metrics + Salma's positions)
                # ---------------------------
                width = max_pt[0] - min_pt[0]
                aspect_ratio = width / (max_pt[1] - min_pt[1] + 1e-5)

                det = Detection3D()
                det.bbox.center.position.x = np.mean(cluster_pts[:, 0])  # Salma's position logic
                det.bbox.center.position.y = np.mean(cluster_pts[:, 1])
                det.bbox.size.x = width  # Hatem's dimension analysis
                det.bbox.size.y = max_pt[1] - min_pt[1]
                det.bbox.size.z = 0.05  # 2D assumption

                # ---------------------------
                # Classification Teamwork 🧠 (Hatem's types + Salma's direction)
                # ---------------------------
                if (aspect_ratio > 5) or (np.max(cluster_pts[:,0]) - np.min(cluster_pts[:,0]) > self.track_width*0.8):
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = "wall"
                    hypothesis.hypothesis.score = 0.9
                    walls.detections.append(det)
                else:
                    # ---------------------------
                    # Directional Awareness ➡️⬅️ (Salma's original concept)
                    # ---------------------------

                    # Step 6: Separate objects on the sides from those in front (ORIGINAL)
                    is_front = abs(np.mean(cluster_pts[:,1])) < 1.0  # Salma's directional threshold
                    class_id = "car" if (abs(width - self.car_length) < 0.1) else "unknown"
                    
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = f"{class_id}_{'front' if is_front else 'side'}"
                    hypothesis.hypothesis.score = 0.7
                    obstacles.detections.append(det)

            # Original Salma's implementation for Step 6:
            # side_threshold = 1.0  # Objects with |x| < 1.0 m are on the side
            # front_threshold = 1.0  # Objects with |y| < 1.0 m are in front
            # objects_in_front = []
            # objects_on_side = []
            # for point in large_cluster_points:
            #     x, y = point
            #     if abs(y) < front_threshold:
            #         objects_in_front.append(point)  # Object is in front
            #     elif abs(x) < side_threshold:
            #         objects_on_side.append(point)  # Object is on the side

            # ---------------------------
            # Dual Output System 📤 (Hatem's ROS + Salma's logging)
            # ---------------------------

            # Log the detected objects (ORIGINAL)
            self.obstacle_pub.publish(obstacles)
            self.wall_pub.publish(walls)
            self.get_logger().info(f'Published {len(obstacles.detections)} obstacles | {len(walls.detections)} walls')

            # Original Salma's implementation for logging:
            # for i, point in enumerate(objects_in_front):
            #     x, y = point
            #     self.get_logger().info(f"Object {i + 1} in front at: (x={x:.2f}m, y={y:.2f}m)")
            # for i, point in enumerate(objects_on_side):
            #     x, y = point
            #     self.get_logger().info(f"Object {i + 1} on the side at: (x={x:.2f}m, y={y:.2f}m)")

        except Exception as e:
            # ---------------------------
            # Error Handling Upgrade 🚨 (Hatem)
            # ---------------------------
            self.get_logger().error(f"Processing failed: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()