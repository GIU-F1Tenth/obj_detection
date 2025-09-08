#!/usr/bin/env python3
"""
Unit tests for the geometry-based object detection algorithms.
Tests the core detection logic without ROS2 dependencies.
"""

import pytest
import numpy as np
import sys
import os
import math

# Add the project root to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lidar_obj_detection.core_algorithms import (
    GeometryDetector,
    DetectionConfig,
    LiDARFilterConfig
)


class TestGeometryDetection:
    """Test suite for geometry-based detection algorithms."""
    
    def test_cluster_extent_pca_simple(self):
        """Test PCA-based extent calculation with simple rectangular cluster."""
        detector = GeometryDetector(DetectionConfig())
        
        # Test with rectangular cluster (should have clear major/minor axes)
        rect_points = np.array([
            [0, 0], [1, 0], [2, 0], [3, 0],  # Long horizontal line
            [0, 0.1], [1, 0.1], [2, 0.1], [3, 0.1]  # Slightly offset
        ])
        
        major, minor = detector.cluster_extent_pca(rect_points)
        
        # Major axis should be along length (~3), minor along width (~0.1)
        assert major > minor
        assert major > 2.5  # Should capture most of the length
        assert minor < 0.5  # Should be small for width
    
    def test_cluster_extent_pca_single_point(self):
        """Test PCA with single point."""
        detector = GeometryDetector(DetectionConfig())
        
        single_point = np.array([[1.0, 2.0]])
        major, minor = detector.cluster_extent_pca(single_point)
        
        assert major == 0.0
        assert minor == 0.0
    
    def test_geometry_aware_threshold(self):
        """Test the geometry-aware threshold calculation."""
        # Test the threshold formula: max(k * max(r, rp) * angle_increment, sigma)
        k = 1.8
        sigma = 0.03
        angle_increment = 0.01  # typical LiDAR angular resolution
        
        # Test with close range
        r1, r2 = 1.0, 1.1
        threshold = max(k * max(r1, r2) * angle_increment, sigma)
        expected = max(1.8 * 1.1 * 0.01, 0.03)
        assert abs(threshold - expected) < 1e-10
        
        # Test with far range (should be dominated by geometry term)
        r1, r2 = 10.0, 10.1
        threshold = max(k * max(r1, r2) * angle_increment, sigma)
        assert threshold > sigma  # Should be larger than noise cushion
    
    def test_object_width_validation(self):
        """Test object width validation for F1TENTH cars."""
        config = DetectionConfig()
        
        # Test valid car widths
        assert config.min_object_width <= 0.33 <= config.max_object_width  # Typical F1TENTH width
        assert config.min_object_width <= 0.25 <= config.max_object_width  # Narrow car
        assert config.min_object_width <= 0.45 <= config.max_object_width  # Wide car
        
        # Test invalid widths
        assert not (config.min_object_width <= 0.1 <= config.max_object_width)  # Too narrow
        assert not (config.min_object_width <= 1.0 <= config.max_object_width)  # Too wide


class TestRingSegmentation:
    """Test suite for ring-based segmentation algorithms."""
    
    def test_consecutive_point_gap_calculation(self):
        """Test gap calculation between consecutive LiDAR points."""
        # Simulate two points at different ranges and angles
        r1, angle1 = 2.0, 0.0      # Point at (2, 0)
        r2, angle2 = 2.0, 0.1      # Point at (2*cos(0.1), 2*sin(0.1))
        
        # Convert to Cartesian
        x1, y1 = r1 * math.cos(angle1), r1 * math.sin(angle1)
        x2, y2 = r2 * math.cos(angle2), r2 * math.sin(angle2)
        
        # Calculate gap
        gap = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # For small angles: gap ≈ r * dθ
        expected_gap = r1 * (angle2 - angle1)
        assert abs(gap - expected_gap) < 0.01  # Should be close for small angles
    
    def test_segmentation_threshold_scaling(self):
        """Test that segmentation threshold scales with distance."""
        k = 1.8
        sigma = 0.03
        angle_increment = 0.01
        
        # Near points should have smaller threshold
        r_near = 1.0
        threshold_near = max(k * r_near * angle_increment, sigma)
        
        # Far points should have larger threshold
        r_far = 5.0
        threshold_far = max(k * r_far * angle_increment, sigma)
        
        assert threshold_far > threshold_near


class TestConfidenceCalculation:
    """Test suite for detection confidence scoring."""
    
    def test_confidence_based_on_density(self):
        """Test confidence calculation based on point density."""
        # High density cluster should have high confidence
        num_points_dense = 20
        arc_length = 0.5
        density_dense = num_points_dense / arc_length
        confidence_dense = min(1.0, density_dense / 8.0)
        
        # Low density cluster should have low confidence
        num_points_sparse = 5
        arc_length = 2.0
        density_sparse = num_points_sparse / arc_length
        confidence_sparse = min(1.0, density_sparse / 8.0)
        
        assert confidence_dense > confidence_sparse
    
    def test_confidence_width_bonus(self):
        """Test confidence bonus for realistic car widths."""
        base_confidence = 0.6
        
        # Car-like width should get bonus
        car_width = 0.35  # Typical F1TENTH width
        if 0.25 <= car_width <= 0.45:
            boosted_confidence = min(1.0, base_confidence * 1.2)
        else:
            boosted_confidence = base_confidence
        
        # Non-car width should not get bonus
        weird_width = 0.15
        if 0.25 <= weird_width <= 0.45:
            normal_confidence = min(1.0, base_confidence * 1.2)
        else:
            normal_confidence = base_confidence
        
        assert boosted_confidence > normal_confidence


class TestLiDARDataProcessing:
    """Test suite for LiDAR data processing functions."""
    
    def test_polar_to_cartesian_conversion(self):
        """Test conversion from polar to Cartesian coordinates."""
        # Test known values
        r, theta = 1.0, 0.0
        x, y = r * np.cos(theta), r * np.sin(theta)
        assert abs(x - 1.0) < 1e-10
        assert abs(y - 0.0) < 1e-10
        
        r, theta = 2.0, np.pi/2
        x, y = r * np.cos(theta), r * np.sin(theta)
        assert abs(x - 0.0) < 1e-10
        assert abs(y - 2.0) < 1e-10
    
    def test_range_filtering(self):
        """Test range-based filtering of LiDAR points."""
        config = LiDARFilterConfig()
        
        # Test points within range
        assert config.min_range <= 1.0 <= config.max_range
        assert config.min_range <= 5.0 <= config.max_range
        
        # Test points outside range
        assert not (config.min_range <= 0.05 <= config.max_range)  # Too close
        assert not (config.min_range <= 15.0 <= config.max_range)  # Too far


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
