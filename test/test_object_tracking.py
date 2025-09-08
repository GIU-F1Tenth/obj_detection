#!/usr/bin/env python3
"""
Integration tests for the ObjectTracker class.
Tests multi-object tracking functionality without ROS2 dependencies.
"""

import pytest
import numpy as np
import sys
import os
import time

# Add the project root to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lidar_obj_detection.core_algorithms import (
    ObjectTracker,
    TrackingConfig,
    DetectedObject
)


class TestObjectTracker:
    """Test suite for the ObjectTracker class."""
    
    def test_tracker_initialization(self):
        """Test ObjectTracker initialization."""
        config = TrackingConfig()
        tracker = ObjectTracker(config)
        
        assert len(tracker.tracks) == 0
        assert len(tracker.track_info) == 0
        assert tracker.next_track_id == 1
        assert tracker.last_update_time == 0.0
    
    def test_single_object_tracking(self):
        """Test tracking a single object over time."""
        config = TrackingConfig()
        tracker = ObjectTracker(config)
        
        # First detection
        detections1 = [(1.0, 2.0, 0.8)]  # (x, y, confidence)
        current_time = 1.0
        tracker.update(detections1, current_time)
        
        # Should create one track
        assert len(tracker.tracks) == 1
        assert len(tracker.track_info) == 1
        
        tracked_objects = tracker.get_tracked_objects()
        assert len(tracked_objects) == 1
        assert tracked_objects[0].confidence == 0.8
        assert tracked_objects[0].track_id == 1
    
    def test_object_movement_tracking(self):
        """Test tracking an object that moves over time."""
        config = TrackingConfig()
        tracker = ObjectTracker(config)
        
        # Simulate object moving in straight line
        detections = [
            [(0.0, 0.0, 0.9)],    # t=0: starting position
            [(0.1, 0.0, 0.9)],    # t=1: moved right
            [(0.2, 0.0, 0.9)],    # t=2: moved right again
            [(0.3, 0.0, 0.9)]     # t=3: moved right again
        ]
        
        for i, detection_list in enumerate(detections):
            tracker.update(detection_list, float(i))
        
        # Should maintain single track
        assert len(tracker.tracks) == 1
        
        # Check final velocity estimate
        tracked_objects = tracker.get_tracked_objects()
        velocity = tracked_objects[0].velocity
        assert velocity[0] > 0.05  # Should detect rightward motion
        assert abs(velocity[1]) < 0.05  # Should have minimal y velocity
    
    def test_multiple_object_tracking(self):
        """Test tracking multiple objects simultaneously."""
        config = TrackingConfig()
        tracker = ObjectTracker(config)
        
        # Two objects at different positions
        detections = [
            (1.0, 1.0, 0.8),   # Object 1
            (5.0, 5.0, 0.9)    # Object 2
        ]
        
        tracker.update(detections, 1.0)
        
        # Should create two tracks
        assert len(tracker.tracks) == 2
        assert len(tracker.track_info) == 2
        
        tracked_objects = tracker.get_tracked_objects()
        assert len(tracked_objects) == 2
        
        # Objects should have different track IDs
        track_ids = [obj.track_id for obj in tracked_objects]
        assert len(set(track_ids)) == 2  # Should be unique
    
    def test_track_association(self):
        """Test that detections are correctly associated with existing tracks."""
        config = TrackingConfig()
        config.max_tracking_distance = 0.5  # Small association distance
        tracker = ObjectTracker(config)
        
        # Initial detection
        tracker.update([(1.0, 1.0, 0.8)], 1.0)
        initial_track_id = tracker.get_tracked_objects()[0].track_id
        
        # Close detection should be associated with existing track
        tracker.update([(1.1, 1.1, 0.9)], 2.0)
        
        # Should still have only one track
        assert len(tracker.tracks) == 1
        tracked_objects = tracker.get_tracked_objects()
        assert tracked_objects[0].track_id == initial_track_id
        assert tracked_objects[0].confidence == 0.9  # Updated confidence
    
    def test_track_creation_for_distant_objects(self):
        """Test that distant detections create new tracks."""
        config = TrackingConfig()
        config.max_tracking_distance = 0.5  # Small association distance
        tracker = ObjectTracker(config)
        
        # Initial detection
        tracker.update([(1.0, 1.0, 0.8)], 1.0)
        
        # Distant detection should create new track
        tracker.update([(1.0, 1.0, 0.8), (5.0, 5.0, 0.9)], 2.0)
        
        # Should have two tracks now
        assert len(tracker.tracks) == 2
        tracked_objects = tracker.get_tracked_objects()
        assert len(tracked_objects) == 2
    
    def test_track_timeout(self):
        """Test that tracks are removed after timeout."""
        config = TrackingConfig()
        config.track_timeout = 1.0  # 1 second timeout
        tracker = ObjectTracker(config)
        
        # Create track
        tracker.update([(1.0, 1.0, 0.8)], 1.0)
        assert len(tracker.tracks) == 1
        
        # Update without detections after timeout period
        tracker.update([], 3.0)  # 2 seconds later
        
        # Track should be removed
        assert len(tracker.tracks) == 0
        assert len(tracker.track_info) == 0
    
    def test_confidence_threshold_filtering(self):
        """Test that low-confidence detections are filtered out."""
        config = TrackingConfig()
        config.confidence_threshold = 0.6
        tracker = ObjectTracker(config)
        
        # Low confidence detection should not create track
        low_confidence_detections = [(1.0, 1.0, 0.4)]  # Below threshold
        tracker.update(low_confidence_detections, 1.0)
        
        assert len(tracker.tracks) == 0  # No tracks created
        
        # High confidence detection should create track
        high_confidence_detections = [(1.0, 1.0, 0.8)]  # Above threshold
        tracker.update(high_confidence_detections, 2.0)
        
        assert len(tracker.tracks) == 1  # Track created


class TestTrackingScenarios:
    """Integration tests for realistic tracking scenarios."""
    
    def test_racing_scenario_two_cars(self):
        """Test tracking two cars in a racing scenario."""
        config = TrackingConfig()
        config.max_tracking_distance = 1.0
        config.track_timeout = 2.0
        tracker = ObjectTracker(config)
        
        # Simulate two cars moving in parallel
        time_steps = 10
        dt = 0.1
        
        for t in range(time_steps):
            current_time = t * dt
            
            # Car 1: moving right
            car1_x = t * 0.1
            car1_y = 0.0
            
            # Car 2: moving diagonally
            car2_x = t * 0.05
            car2_y = t * 0.08
            
            detections = [
                (car1_x, car1_y, 0.9),
                (car2_x, car2_y, 0.8)
            ]
            
            tracker.update(detections, current_time)
        
        # Should maintain two separate tracks
        assert len(tracker.tracks) == 2
        
        tracked_objects = tracker.get_tracked_objects()
        velocities = [obj.velocity for obj in tracked_objects]
        
        # Both cars should have positive x velocity
        for vx, vy in velocities:
            assert vx > 0.0  # Moving forward
    
    def test_object_disappearance_and_reappearance(self):
        """Test handling object temporarily disappearing from view."""
        config = TrackingConfig()
        config.track_timeout = 0.5  # Short timeout for this test
        tracker = ObjectTracker(config)
        
        # Object appears
        tracker.update([(1.0, 1.0, 0.8)], 0.0)
        assert len(tracker.tracks) == 1
        original_track_id = tracker.get_tracked_objects()[0].track_id
        
        # Object disappears (no detections)
        tracker.update([], 0.2)
        assert len(tracker.tracks) == 1  # Track still exists
        
        # Object reappears nearby
        tracker.update([(1.1, 1.1, 0.9)], 0.3)
        assert len(tracker.tracks) == 1  # Should associate with existing track
        assert tracker.get_tracked_objects()[0].track_id == original_track_id
        
        # Wait too long, track should be removed
        tracker.update([], 1.0)
        assert len(tracker.tracks) == 0
    
    def test_crossing_trajectories(self):
        """Test tracking objects with crossing trajectories."""
        config = TrackingConfig()
        config.max_tracking_distance = 0.8
        tracker = ObjectTracker(config)
        
        # Two objects starting far apart
        tracker.update([(0.0, 0.0, 0.9), (2.0, 2.0, 0.9)], 0.0)
        assert len(tracker.tracks) == 2
        
        # Objects move towards each other
        tracker.update([(0.5, 0.5, 0.9), (1.5, 1.5, 0.9)], 1.0)
        assert len(tracker.tracks) == 2
        
        # Objects cross paths (close to each other)
        tracker.update([(1.0, 1.0, 0.9), (1.0, 1.0, 0.9)], 2.0)
        # This is a challenging case - tracker might lose one or merge tracks
        # The exact behavior depends on association algorithm
        assert len(tracker.tracks) >= 1  # At least one track should survive


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
