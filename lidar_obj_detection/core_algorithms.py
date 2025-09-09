#!/usr/bin/env python3
"""
Standalone implementations of core detection algorithms for testing.
This module contains ROS2-independent versions of the key classes.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
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
    max_cluster_size: int = 120  # Maximum cluster size for valid detection
    min_object_width: float = 0.1  # Minimum object width (m) along minor axis - more lenient
    max_object_width: float = 0.8  # Maximum object width (m) along minor axis - more lenient
    confidence_threshold: float = 0.3  # Minimum confidence for publishing - more lenient


@dataclass
class TrackingConfig:
    """Configuration parameters for object tracking."""
    max_tracking_distance: float = 1.0  # Maximum distance for track association (m)
    track_timeout: float = 2.0  # Time before removing lost tracks (s)
    velocity_smoothing_factor: float = 0.7  # Velocity smoothing coefficient
    position_smoothing_factor: float = 0.8  # Position smoothing coefficient
    max_velocity: float = 15.0  # Maximum believable velocity (m/s)
    confidence_threshold: float = 0.5  # Minimum confidence for track creation


class KalmanFilter:
    """
    Simple 2D Kalman filter for object tracking.
    State vector: [x, y, vx, vy]
    """

    def __init__(self, initial_position: Tuple[float, float], dt: float = 0.1):
        """Initialize Kalman filter with initial position."""
        self.dt = dt

        # State vector [x, y, vx, vy]
        self.state = np.array(
            [initial_position[0], initial_position[1], 0.0, 0.0])

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


class GeometryDetector:
    """
    Standalone geometry-based detection algorithms.
    """
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def cluster_extent_pca(self, cluster_xy: np.ndarray) -> Tuple[float, float]:
        """Compute major and minor extents using PCA for robustness to outliers."""
        if cluster_xy.shape[0] < 2:
            return 0.0, 0.0
        
        # Ensure we have at least 2D data
        if cluster_xy.shape[1] < 2:
            cluster_xy = np.column_stack([cluster_xy.flatten(), np.zeros(cluster_xy.shape[0])])
        
        C = cluster_xy - cluster_xy.mean(axis=0)
        
        # Handle degenerate cases
        if np.allclose(C, 0):
            return 0.0, 0.0
        
        try:
            # SVD of centered points - ensure proper dimensions
            U, S, Vt = np.linalg.svd(C, full_matrices=False)
            
            # Project onto principal axes
            proj = C @ Vt.T
            
            # Calculate extents along each principal axis
            extents = np.ptp(proj, axis=0)  # peak-to-peak (max - min)
            
            # Ensure we have at least 2 extents
            if len(extents) >= 2:
                major = float(extents[0])
                minor = float(extents[1])
            elif len(extents) == 1:
                major = float(extents[0])
                minor = 0.0
            else:
                major = 0.0
                minor = 0.0
                
            return major, minor
            
        except (np.linalg.LinAlgError, ValueError) as e:
            # Fallback to simple bounding box calculation
            if cluster_xy.shape[0] > 1:
                extents = np.ptp(cluster_xy, axis=0)
                major = float(extents[0]) if len(extents) > 0 else 0.0
                minor = float(extents[1]) if len(extents) > 1 else 0.0
                return major, minor
            else:
                return 0.0, 0.0
    
    def segment_ring(self, angles: np.ndarray, ranges: np.ndarray, angle_increment: float) -> List[np.ndarray]:
        """Segment the scan into clusters using consecutive-beam gap thresholding."""
        pts = []
        for a, r in zip(angles, ranges):
            if np.isfinite(r) and 0.1 <= r <= 10.0:  # Basic range filtering
                pts.append((r * np.cos(a), r * np.sin(a), r))
        if not pts:
            return []

        # Geometry-aware threshold: gap <= k * r * dtheta with additive noise cushion
        k = 1.8
        sigma = 0.03
        clusters = []
        cur = [pts[0]]

        for (x, y, r), (xp, yp, rp) in zip(pts[1:], pts[:-1]):
            dr = math.hypot(x - xp, y - yp)
            thr = max(k * max(r, rp) * angle_increment, sigma)
            if dr <= thr:
                cur.append((x, y, r))
            else:
                if len(cur) >= self.config.min_cluster_size:
                    clusters.append(np.array([(cx, cy) for cx, cy, _ in cur]))
                cur = [(x, y, r)]

        if len(cur) >= self.config.min_cluster_size:
            clusters.append(np.array([(cx, cy) for cx, cy, _ in cur]))

        return clusters
    
    def clusters_to_detections(self, clusters: List[np.ndarray]) -> List[Tuple[float, float, float]]:
        """Convert clusters to detections with robust extent checks and confidence."""
        detections: List[Tuple[float, float, float]] = []
        for cluster_points in clusters:
            if cluster_points.shape[0] < self.config.min_cluster_size:
                continue
            if cluster_points.shape[0] > self.config.max_cluster_size:
                continue

            centroid = np.mean(cluster_points, axis=0)
            major_len, minor_w = self.cluster_extent_pca(cluster_points)

            # Filter by minor axis width to match cart width
            if minor_w < self.config.min_object_width or minor_w > self.config.max_object_width:
                continue

            # Confidence based on density along major extent and width reasonableness
            num_points = cluster_points.shape[0]
            arc_len = max(major_len, 0.05)
            density = num_points / arc_len
            confidence = min(1.0, density / 8.0)
            if 0.25 <= minor_w <= 0.45:
                confidence = min(1.0, confidence * 1.2)

            if confidence >= self.config.confidence_threshold:
                detections.append(
                    (float(centroid[0]), float(centroid[1]), float(confidence)))

        return detections
