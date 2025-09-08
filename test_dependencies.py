#!/usr/bin/env python3
"""
Test script to verify all dependencies are properly installed and core functionality works.
"""

import sys
import numpy as np
import scipy
import sklearn
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def test_numpy():
    """Test NumPy functionality."""
    print("Testing NumPy...")
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    print("✓ NumPy working correctly")

def test_scipy():
    """Test SciPy functionality."""
    print("Testing SciPy...")
    from scipy.spatial.distance import euclidean
    dist = euclidean([0, 0], [3, 4])
    assert abs(dist - 5.0) < 1e-10
    print("✓ SciPy working correctly")

def test_sklearn():
    """Test scikit-learn functionality."""
    print("Testing scikit-learn...")
    # Create sample data for clustering
    X = np.array([[1, 1], [1, 2], [2, 1], [10, 10], [10, 11], [11, 10]])
    clustering = DBSCAN(eps=2, min_samples=2).fit(X)
    assert len(set(clustering.labels_)) >= 2  # Should find at least 2 clusters
    print("✓ scikit-learn working correctly")

def test_matplotlib():
    """Test Matplotlib functionality."""
    print("Testing Matplotlib...")
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot([1, 2, 3], [1, 4, 2])
    ax.set_title("Test Plot")
    plt.close(fig)  # Close to avoid display issues
    print("✓ Matplotlib working correctly")

def test_kalman_filter_math():
    """Test core mathematical operations for Kalman filter."""
    print("Testing Kalman Filter math...")
    
    # Test matrix operations
    F = np.array([[1, 0, 0.1, 0],
                  [0, 1, 0, 0.1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    
    state = np.array([1.0, 2.0, 0.5, -0.3])
    new_state = F @ state
    
    # Verify prediction step
    assert abs(new_state[0] - 1.05) < 1e-10  # x + vx*dt
    assert abs(new_state[1] - 1.97) < 1e-10  # y + vy*dt
    
    print("✓ Kalman Filter math working correctly")

def test_geometry_calculations():
    """Test geometry-based detection calculations."""
    print("Testing geometry calculations...")
    
    # Test PCA-like calculations
    points = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
    center = points.mean(axis=0)
    centered = points - center
    
    # Simple SVD test
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    assert len(S) == 2  # Should have 2 singular values for 2D data
    
    print("✓ Geometry calculations working correctly")

def main():
    """Run all tests."""
    print("=== Dependency Testing for LiDAR Object Detection ===\n")
    
    try:
        test_numpy()
        test_scipy()
        test_sklearn()
        test_matplotlib()
        test_kalman_filter_math()
        test_geometry_calculations()
        
        print("\n=== All Tests Passed! ===")
        print("🎉 Environment is ready for LiDAR Object Detection development!")
        
        # Print versions
        print(f"\nInstalled Versions:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  NumPy: {np.__version__}")
        print(f"  SciPy: {scipy.__version__}")
        print(f"  scikit-learn: {sklearn.__version__}")
        print(f"  Matplotlib: {plt.matplotlib.__version__}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
