#!/usr/bin/env python3
"""
LiDAR Object Detection Node

A comprehensive ROS2 node for detecting and tracking opponents using LiDAR data.
This node implements advanced filtering techniques, clustering-based detection,
and Kalman filter-based tracking for F1TENTH autonomous racing applications.

Author: F1TENTH Team
License: MIT
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, Pose, PoseStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header, ColorRGBA

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean


@dataclass
class DetectedObject:
    """Data class for storing detected object information."""
    position: Tuple[float, float]  # (x, y) in meters
    velocity: Tuple[float, float]  # (vx, vy) in m/s
    size: float  # Estimated radius in meters
    confidence: float  # Detection confidence [0-1]
    last_seen: float  # Timestamp of last detection
    track_id: int  # Unique tracking ID


@dataclass
class LiDARFilterConfig:
    """Configuration parameters for LiDAR filtering."""
    min_range: float = 0.1  # Minimum valid range (m)
    max_range: float = 10.0  # Maximum detection range (m)
    min_angle: float = -np.pi  # Minimum scan angle (rad)
    max_angle: float = np.pi  # Maximum scan angle (rad)
    intensity_threshold: float = 100.0  # Minimum intensity for valid points
    noise_filter_window: int = 3  # Moving average window size
    outlier_threshold: float = 2.0  # Standard deviations for outlier removal


@dataclass
class DetectionConfig:
    """Configuration parameters for object detection."""
    dbscan_eps: float = 0.3  # DBSCAN clustering distance threshold (m)
    dbscan_min_samples: int = 3  # Minimum points per cluster
    min_cluster_size: int = 3  # Minimum cluster size for valid detection
    max_cluster_size: int = 50  # Maximum cluster size for valid detection
    min_object_width: float = 0.2  # Minimum object width (m)
    max_object_width: float = 2.0  # Maximum object width (m)
    confidence_threshold: float = 0.5  # Minimum confidence for publishing


@dataclass
class TrackingConfig:
    """Configuration parameters for object tracking."""
    max_tracking_distance: float = 1.0  # Maximum distance for track association (m)
    track_timeout: float = 2.0  # Time before removing lost tracks (s)
    velocity_smoothing_factor: float = 0.7  # Velocity smoothing coefficient
    position_smoothing_factor: float = 0.8  # Position smoothing coefficient
    max_velocity: float = 15.0  # Maximum believable velocity (m/s)


class KalmanFilter:
    """
    Simple 2D Kalman filter for object tracking.
    State vector: [x, y, vx, vy]
    """
    
    def __init__(self, initial_position: Tuple[float, float], dt: float = 0.1):
        """Initialize Kalman filter with initial position."""
        self.dt = dt
        
        # State vector [x, y, vx, vy]
        self.state = np.array([initial_position[0], initial_position[1], 0.0, 0.0])
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        q = 0.1  # Process noise
        self.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ])
        
        # Measurement noise covariance
        r = 0.1  # Measurement noise
        self.R = np.array([
            [r, 0],
            [0, r]
        ])
        
        # Error covariance matrix
        self.P = np.eye(4) * 1.0
    
    def predict(self):
        """Predict the next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: Tuple[float, float]):
        """Update the filter with a new measurement."""
        z = np.array([measurement[0], measurement[1]])
        
        # Innovation
        y = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position estimate."""
        return (self.state[0], self.state[1])
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        return (self.state[2], self.state[3])


class ObjectTracker:
    """
    Multi-object tracker using Kalman filters.
    """
    
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.tracks = {}  # track_id -> KalmanFilter
        self.track_info = {}  # track_id -> DetectedObject
        self.next_track_id = 1
        self.last_update_time = 0.0
    
    def update(self, detections: List[Tuple[float, float, float]], current_time: float):
        """
        Update tracks with new detections.
        
        Args:
            detections: List of (x, y, confidence) tuples
            current_time: Current timestamp
        """
        dt = current_time - self.last_update_time if self.last_update_time > 0 else 0.1
        self.last_update_time = current_time
        
        # Predict all existing tracks
        for track_id, kalman_filter in self.tracks.items():
            kalman_filter.predict()
        
        # Associate detections with existing tracks
        track_assignments = self._associate_detections(detections)
        
        # Update assigned tracks
        for track_id, detection_idx in track_assignments.items():
            if detection_idx is not None:
                detection = detections[detection_idx]
                self.tracks[track_id].update((detection[0], detection[1]))
                
                # Update track info
                pos = self.tracks[track_id].get_position()
                vel = self.tracks[track_id].get_velocity()
                
                self.track_info[track_id].position = pos
                self.track_info[track_id].velocity = vel
                self.track_info[track_id].confidence = detection[2]
                self.track_info[track_id].last_seen = current_time
        
        # Create new tracks for unassigned detections
        assigned_detections = set(track_assignments.values())
        for i, detection in enumerate(detections):
            if i not in assigned_detections and detection[2] > self.config.confidence_threshold:
                self._create_new_track(detection, current_time)
        
        # Remove old tracks
        self._remove_old_tracks(current_time)
    
    def _associate_detections(self, detections: List[Tuple[float, float, float]]) -> dict:
        """Associate detections with existing tracks using nearest neighbor."""
        assignments = {}
        
        if not self.tracks or not detections:
            return assignments
        
        # Calculate distance matrix
        track_ids = list(self.tracks.keys())
        distances = np.full((len(track_ids), len(detections)), np.inf)
        
        for i, track_id in enumerate(track_ids):
            track_pos = self.tracks[track_id].get_position()
            for j, detection in enumerate(detections):
                dist = euclidean(track_pos, (detection[0], detection[1]))
                if dist < self.config.max_tracking_distance:
                    distances[i, j] = dist
        
        # Simple greedy assignment (could be improved with Hungarian algorithm)
        assigned_detections = set()
        for i, track_id in enumerate(track_ids):
            min_dist_idx = np.argmin(distances[i, :])
            min_dist = distances[i, min_dist_idx]
            
            if min_dist < self.config.max_tracking_distance and min_dist_idx not in assigned_detections:
                assignments[track_id] = min_dist_idx
                assigned_detections.add(min_dist_idx)
            else:
                assignments[track_id] = None
        
        return assignments
    
    def _create_new_track(self, detection: Tuple[float, float, float], current_time: float):
        """Create a new track for an unassigned detection."""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        # Create Kalman filter
        kalman_filter = KalmanFilter((detection[0], detection[1]))
        self.tracks[track_id] = kalman_filter
        
        # Create track info
        detected_object = DetectedObject(
            position=(detection[0], detection[1]),
            velocity=(0.0, 0.0),
            size=0.3,  # Default size
            confidence=detection[2],
            last_seen=current_time,
            track_id=track_id
        )
        self.track_info[track_id] = detected_object
    
    def _remove_old_tracks(self, current_time: float):
        """Remove tracks that haven't been updated recently."""
        tracks_to_remove = []
        
        for track_id, track_info in self.track_info.items():
            if current_time - track_info.last_seen > self.config.track_timeout:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            del self.track_info[track_id]
    
    def get_tracked_objects(self) -> List[DetectedObject]:
        """Get all currently tracked objects."""
        return list(self.track_info.values())


class LiDARObjectDetectionNode(Node):
    """
    ROS2 node for LiDAR-based object detection and tracking.
    """
    
    def __init__(self):
        super().__init__('lidar_object_detection_node')
        
        # Initialize configurations
        self.filter_config = LiDARFilterConfig()
        self.detection_config = DetectionConfig()
        self.tracking_config = TrackingConfig()
        
        # Initialize object tracker
        self.tracker = ObjectTracker(self.tracking_config)
        
        # Initialize data buffers
        self.scan_buffer = deque(maxlen=5)  # For temporal filtering
        
        # Declare parameters
        self._declare_parameters()
        
        # Set up QoS profile
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create subscribers
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile
        )
        
        # Create publishers
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/detected_objects_markers',
            10
        )
        
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/opponent_pose',
            10
        )
        
        # Create timer for periodic processing
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        
        self.get_logger().info("LiDAR Object Detection Node initialized successfully")
    
    def _declare_parameters(self):
        """Declare ROS2 parameters with default values."""
        # Filter parameters
        self.declare_parameter('filter.min_range', self.filter_config.min_range)
        self.declare_parameter('filter.max_range', self.filter_config.max_range)
        self.declare_parameter('filter.noise_filter_window', self.filter_config.noise_filter_window)
        
        # Detection parameters
        self.declare_parameter('detection.dbscan_eps', self.detection_config.dbscan_eps)
        self.declare_parameter('detection.dbscan_min_samples', self.detection_config.dbscan_min_samples)
        self.declare_parameter('detection.confidence_threshold', self.detection_config.confidence_threshold)
        
        # Tracking parameters
        self.declare_parameter('tracking.max_tracking_distance', self.tracking_config.max_tracking_distance)
        self.declare_parameter('tracking.track_timeout', self.tracking_config.track_timeout)
    
    def _update_parameters(self):
        """Update configuration from ROS2 parameters."""
        self.filter_config.min_range = self.get_parameter('filter.min_range').value
        self.filter_config.max_range = self.get_parameter('filter.max_range').value
        self.filter_config.noise_filter_window = self.get_parameter('filter.noise_filter_window').value
        
        self.detection_config.dbscan_eps = self.get_parameter('detection.dbscan_eps').value
        self.detection_config.dbscan_min_samples = self.get_parameter('detection.dbscan_min_samples').value
        self.detection_config.confidence_threshold = self.get_parameter('detection.confidence_threshold').value
        
        self.tracking_config.max_tracking_distance = self.get_parameter('tracking.max_tracking_distance').value
        self.tracking_config.track_timeout = self.get_parameter('tracking.track_timeout').value
    
    def scan_callback(self, msg: LaserScan):
        """Process incoming LiDAR scan messages."""
        # Store scan in buffer for temporal filtering
        self.scan_buffer.append(msg)
        
        # Process the scan
        try:
            filtered_points = self._filter_scan(msg)
            detections = self._detect_objects(filtered_points)
            
            # Update tracker
            current_time = self.get_clock().now().nanoseconds / 1e9
            self.tracker.update(detections, current_time)
            
        except Exception as e:
            self.get_logger().error(f"Error processing scan: {str(e)}")
    
    def _filter_scan(self, scan: LaserScan) -> List[Tuple[float, float]]:
        """
        Apply comprehensive filtering to LiDAR scan data.
        
        Args:
            scan: LaserScan message
            
        Returns:
            List of filtered (x, y) points in the robot frame
        """
        ranges = np.array(scan.ranges)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
        
        # 1. Range filtering
        valid_mask = (ranges >= self.filter_config.min_range) & \
                     (ranges <= self.filter_config.max_range) & \
                     np.isfinite(ranges)
        
        # 2. Angle filtering
        angle_mask = (angles >= self.filter_config.min_angle) & \
                     (angles <= self.filter_config.max_angle)
        
        # Combine masks
        combined_mask = valid_mask & angle_mask
        
        if not np.any(combined_mask):
            return []
        
        # 3. Remove isolated points (noise filtering)
        filtered_ranges = self._remove_noise(ranges, combined_mask)
        
        # 4. Outlier removal using statistical filtering
        filtered_ranges = self._remove_outliers(filtered_ranges, combined_mask)
        
        # 5. Convert to Cartesian coordinates
        valid_indices = np.where(combined_mask)[0]
        points = []
        
        for i in valid_indices:
            if np.isfinite(filtered_ranges[i]) and filtered_ranges[i] > 0:
                x = filtered_ranges[i] * np.cos(angles[i])
                y = filtered_ranges[i] * np.sin(angles[i])
                points.append((x, y))
        
        return points
    
    def _remove_noise(self, ranges: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove noise using a moving average filter."""
        filtered_ranges = ranges.copy()
        window_size = self.filter_config.noise_filter_window
        
        for i in range(len(ranges)):
            if mask[i]:
                # Get window around current point
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(ranges), i + window_size // 2 + 1)
                
                window_values = ranges[start_idx:end_idx]
                window_mask = mask[start_idx:end_idx]
                
                if np.any(window_mask):
                    valid_values = window_values[window_mask]
                    if len(valid_values) >= 2:
                        filtered_ranges[i] = np.median(valid_values)
        
        return filtered_ranges
    
    def _remove_outliers(self, ranges: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove outliers using statistical filtering."""
        if not np.any(mask):
            return ranges
        
        valid_ranges = ranges[mask]
        
        if len(valid_ranges) < 3:
            return ranges
        
        # Calculate statistics
        mean_range = np.mean(valid_ranges)
        std_range = np.std(valid_ranges)
        threshold = self.filter_config.outlier_threshold
        
        # Create outlier mask
        outlier_mask = np.abs(ranges - mean_range) > (threshold * std_range)
        
        # Set outliers to NaN
        filtered_ranges = ranges.copy()
        filtered_ranges[outlier_mask] = np.nan
        
        return filtered_ranges
    
    def _detect_objects(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
        """
        Detect objects using DBSCAN clustering.
        
        Args:
            points: List of filtered (x, y) points
            
        Returns:
            List of (x, y, confidence) detections
        """
        if len(points) < self.detection_config.dbscan_min_samples:
            return []
        
        # Convert to numpy array
        point_array = np.array(points)
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(
            eps=self.detection_config.dbscan_eps,
            min_samples=self.detection_config.dbscan_min_samples
        )
        
        cluster_labels = dbscan.fit_predict(point_array)
        
        detections = []
        
        # Process each cluster
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_points = point_array[cluster_labels == cluster_id]
            
            # Filter clusters by size
            if len(cluster_points) < self.detection_config.min_cluster_size or \
               len(cluster_points) > self.detection_config.max_cluster_size:
                continue
            
            # Calculate cluster properties
            centroid = np.mean(cluster_points, axis=0)
            cluster_width = self._calculate_cluster_width(cluster_points)
            
            # Filter by object size
            if cluster_width < self.detection_config.min_object_width or \
               cluster_width > self.detection_config.max_object_width:
                continue
            
            # Calculate confidence based on cluster properties
            confidence = self._calculate_detection_confidence(cluster_points, cluster_width)
            
            if confidence >= self.detection_config.confidence_threshold:
                detections.append((centroid[0], centroid[1], confidence))
        
        return detections
    
    def _calculate_cluster_width(self, cluster_points: np.ndarray) -> float:
        """Calculate the width of a point cluster."""
        if len(cluster_points) < 2:
            return 0.0
        
        # Use the maximum distance between any two points in the cluster
        max_distance = 0.0
        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                distance = euclidean(cluster_points[i], cluster_points[j])
                max_distance = max(max_distance, distance)
        
        return max_distance
    
    def _calculate_detection_confidence(self, cluster_points: np.ndarray, width: float) -> float:
        """Calculate detection confidence based on cluster properties."""
        # Base confidence on cluster density and size
        num_points = len(cluster_points)
        density = num_points / (width + 0.01)  # Avoid division by zero
        
        # Normalize confidence to [0, 1]
        confidence = min(1.0, density / 10.0)  # Adjust scaling as needed
        
        # Bonus for clusters with reasonable size
        if 0.3 <= width <= 1.5:  # Typical car/opponent size
            confidence *= 1.2
        
        return min(1.0, confidence)
    
    def timer_callback(self):
        """Periodic processing and publishing."""
        self._update_parameters()
        self._publish_detections()
        self._publish_closest_opponent()
    
    def _publish_detections(self):
        """Publish visualization markers for detected objects."""
        tracked_objects = self.tracker.get_tracked_objects()
        
        if not tracked_objects:
            # Publish empty marker array to clear previous markers
            marker_array = MarkerArray()
            self.marker_publisher.publish(marker_array)
            return
        
        marker_array = MarkerArray()
        
        for obj in tracked_objects:
            # Create position marker
            marker = Marker()
            marker.header.frame_id = "laser"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "detected_objects"
            marker.id = obj.track_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = obj.position[0]
            marker.pose.position.y = obj.position[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = obj.size * 2
            marker.scale.y = obj.size * 2
            marker.scale.z = 0.5
            
            # Color based on confidence
            marker.color = ColorRGBA()
            marker.color.r = 1.0 - obj.confidence
            marker.color.g = obj.confidence
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker.lifetime.sec = 1
            marker_array.markers.append(marker)
            
            # Create velocity arrow
            if np.linalg.norm(obj.velocity) > 0.1:  # Only show if moving
                arrow_marker = Marker()
                arrow_marker.header = marker.header
                arrow_marker.ns = "velocity_arrows"
                arrow_marker.id = obj.track_id
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD
                
                arrow_marker.pose.position.x = obj.position[0]
                arrow_marker.pose.position.y = obj.position[1]
                arrow_marker.pose.position.z = 0.25
                
                # Calculate arrow orientation from velocity
                velocity_angle = math.atan2(obj.velocity[1], obj.velocity[0])
                arrow_marker.pose.orientation.z = math.sin(velocity_angle / 2)
                arrow_marker.pose.orientation.w = math.cos(velocity_angle / 2)
                
                velocity_magnitude = np.linalg.norm(obj.velocity)
                arrow_marker.scale.x = velocity_magnitude * 0.2
                arrow_marker.scale.y = 0.1
                arrow_marker.scale.z = 0.1
                
                arrow_marker.color = ColorRGBA()
                arrow_marker.color.r = 0.0
                arrow_marker.color.g = 0.0
                arrow_marker.color.b = 1.0
                arrow_marker.color.a = 0.8
                
                arrow_marker.lifetime.sec = 1
                marker_array.markers.append(arrow_marker)
        
        self.marker_publisher.publish(marker_array)
    
    def _publish_closest_opponent(self):
        """Publish the pose of the closest opponent for tracking."""
        tracked_objects = self.tracker.get_tracked_objects()
        
        if not tracked_objects:
            return
        
        # Find the closest object
        closest_object = min(tracked_objects, 
                           key=lambda obj: np.linalg.norm(obj.position))
        
        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "laser"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        
        pose_msg.pose.position.x = closest_object.position[0]
        pose_msg.pose.position.y = closest_object.position[1]
        pose_msg.pose.position.z = 0.0
        
        # Orientation based on velocity direction
        if np.linalg.norm(closest_object.velocity) > 0.1:
            velocity_angle = math.atan2(closest_object.velocity[1], closest_object.velocity[0])
            pose_msg.pose.orientation.z = math.sin(velocity_angle / 2)
            pose_msg.pose.orientation.w = math.cos(velocity_angle / 2)
        else:
            pose_msg.pose.orientation.w = 1.0
        
        self.pose_publisher.publish(pose_msg)
        
        # Log information about the closest opponent
        distance = np.linalg.norm(closest_object.position)
        speed = np.linalg.norm(closest_object.velocity)
        
        self.get_logger().info(
            f"Closest opponent: ID={closest_object.track_id}, "
            f"Distance={distance:.2f}m, Speed={speed:.2f}m/s, "
            f"Confidence={closest_object.confidence:.2f}"
        )


def main(args=None):
    """Main function to run the LiDAR object detection node."""
    rclpy.init(args=args)
    
    try:
        node = LiDARObjectDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()