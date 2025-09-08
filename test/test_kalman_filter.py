#!/usr/bin/env python3
"""
Unit tests for the Kalman Filter implementation in LiDAR Object Detection.
Tests the 2D tracking functionality without ROS2 dependencies.
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import only the classes we can test without ROS2
from lidar_obj_detection.core_algorithms import (
    KalmanFilter, 
    DetectedObject, 
    LiDARFilterConfig, 
    DetectionConfig, 
    TrackingConfig
)


class TestKalmanFilter:
    """Test suite for the Kalman Filter implementation."""
    
    def test_initialization(self):
        """Test Kalman filter initialization."""
        initial_pos = (1.0, 2.0)
        kf = KalmanFilter(initial_pos, dt=0.1)
        
        # Check initial state
        assert abs(kf.state[0] - 1.0) < 1e-10  # x position
        assert abs(kf.state[1] - 2.0) < 1e-10  # y position
        assert abs(kf.state[2] - 0.0) < 1e-10  # x velocity
        assert abs(kf.state[3] - 0.0) < 1e-10  # y velocity
        
        # Check matrix dimensions
        assert kf.F.shape == (4, 4)
        assert kf.H.shape == (2, 4)
        assert kf.Q.shape == (4, 4)
        assert kf.R.shape == (2, 2)
        assert kf.P.shape == (4, 4)
    
    def test_prediction_step(self):
        """Test the prediction step of Kalman filter."""
        initial_pos = (0.0, 0.0)
        dt = 0.1
        kf = KalmanFilter(initial_pos, dt=dt)
        
        # Set initial velocity
        kf.state[2] = 1.0  # vx = 1 m/s
        kf.state[3] = 0.5  # vy = 0.5 m/s
        
        # Predict one step
        kf.predict()
        
        # Check predicted position (constant velocity model)
        expected_x = 0.0 + 1.0 * dt  # x + vx * dt
        expected_y = 0.0 + 0.5 * dt  # y + vy * dt
        
        assert abs(kf.state[0] - expected_x) < 1e-10
        assert abs(kf.state[1] - expected_y) < 1e-10
        assert abs(kf.state[2] - 1.0) < 1e-10  # velocity unchanged
        assert abs(kf.state[3] - 0.5) < 1e-10  # velocity unchanged
    
    def test_update_step(self):
        """Test the update step of Kalman filter."""
        initial_pos = (0.0, 0.0)
        kf = KalmanFilter(initial_pos, dt=0.1)
        
        # Perform update with measurement
        measurement = (1.0, 2.0)
        kf.update(measurement)
        
        # State should move towards measurement
        assert kf.state[0] > 0.0  # x should be positive
        assert kf.state[1] > 0.0  # y should be positive
        assert abs(kf.state[0] - 1.0) < 1.0  # but not exactly the measurement
        assert abs(kf.state[1] - 2.0) < 1.0  # due to uncertainty
    
    def test_get_position(self):
        """Test position getter."""
        initial_pos = (3.0, 4.0)
        kf = KalmanFilter(initial_pos)
        
        pos = kf.get_position()
        assert abs(pos[0] - 3.0) < 1e-10
        assert abs(pos[1] - 4.0) < 1e-10
    
    def test_get_velocity(self):
        """Test velocity getter."""
        initial_pos = (0.0, 0.0)
        kf = KalmanFilter(initial_pos)
        
        # Set velocity
        kf.state[2] = 2.5
        kf.state[3] = -1.5
        
        vel = kf.get_velocity()
        assert abs(vel[0] - 2.5) < 1e-10
        assert abs(vel[1] - (-1.5)) < 1e-10


class TestDetectedObject:
    """Test suite for DetectedObject data class."""
    
    def test_creation(self):
        """Test DetectedObject creation."""
        obj = DetectedObject(
            position=(1.0, 2.0),
            velocity=(0.5, -0.3),
            size=0.4,
            confidence=0.8,
            last_seen=123.45,
            track_id=7
        )
        
        assert obj.position == (1.0, 2.0)
        assert obj.velocity == (0.5, -0.3)
        assert obj.size == 0.4
        assert obj.confidence == 0.8
        assert obj.last_seen == 123.45
        assert obj.track_id == 7


class TestConfigDataClasses:
    """Test suite for configuration data classes."""
    
    def test_lidar_filter_config_defaults(self):
        """Test LiDARFilterConfig default values."""
        config = LiDARFilterConfig()
        
        assert config.min_range == 0.1
        assert config.max_range == 10.0
        assert config.min_angle == -np.pi
        assert config.max_angle == np.pi
        assert config.intensity_threshold == 100.0
        assert config.noise_filter_window == 3
        assert config.outlier_threshold == 2.0
    
    def test_detection_config_defaults(self):
        """Test DetectionConfig default values."""
        config = DetectionConfig()
        
        assert config.dbscan_eps == 0.3
        assert config.dbscan_min_samples == 3
        assert config.min_cluster_size == 3
        assert config.max_cluster_size == 120
        assert config.min_object_width == 0.2
        assert config.max_object_width == 0.6
        assert config.confidence_threshold == 0.5
    
    def test_tracking_config_defaults(self):
        """Test TrackingConfig default values."""
        config = TrackingConfig()
        
        assert config.max_tracking_distance == 1.0
        assert config.track_timeout == 2.0
        assert config.velocity_smoothing_factor == 0.7
        assert config.position_smoothing_factor == 0.8
        assert config.max_velocity == 15.0


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""
    
    def test_tracking_moving_object(self):
        """Test tracking a moving object over multiple frames."""
        initial_pos = (0.0, 0.0)
        kf = KalmanFilter(initial_pos, dt=0.1)
        
        # Simulate object moving in straight line
        true_positions = [(i * 0.1, i * 0.05) for i in range(10)]  # vx=1, vy=0.5 m/s
        
        estimated_positions = []
        for pos in true_positions:
            kf.predict()
            kf.update(pos)
            estimated_positions.append(kf.get_position())
        
        # Check that filter converges to correct velocity
        final_velocity = kf.get_velocity()
        assert abs(final_velocity[0] - 1.0) < 0.1  # Should be close to true vx
        assert abs(final_velocity[1] - 0.5) < 0.1  # Should be close to true vy
    
    def test_realistic_detection_parameters(self):
        """Test that detection parameters are reasonable for F1TENTH."""
        config = DetectionConfig()
        
        # Object width should match F1TENTH car dimensions
        assert 0.1 <= config.min_object_width <= 0.3  # At least 10cm wide
        assert 0.4 <= config.max_object_width <= 1.0  # At most 100cm wide
        
        # DBSCAN parameters should be reasonable for LiDAR
        assert 0.1 <= config.dbscan_eps <= 0.5  # Reasonable clustering distance
        assert 2 <= config.dbscan_min_samples <= 5  # Reasonable cluster size
        
        # Confidence threshold should be balanced
        assert 0.3 <= config.confidence_threshold <= 0.7


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
