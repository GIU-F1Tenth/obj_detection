import numpy as np
from typing import List


class AdaptiveBreakpoint:
    def __init__(self, base_threshold: float = 0.3, adaptive_coeff: float = 0.05,
                 min_cluster_points: int = 3):
        self.base_threshold = base_threshold
        self.adaptive_coeff = adaptive_coeff
        self.min_cluster_points = min_cluster_points
        
    def _compute_threshold(self, r1: float, r2: float) -> float:
        return self.base_threshold + self.adaptive_coeff * (r1 + r2)
        
    def cluster(self, ranges: np.ndarray, angles: np.ndarray) -> List[np.ndarray]:
        if len(ranges) == 0:
            return []
            
        clusters = []
        current_cluster_indices = [0]
        
        for i in range(1, len(ranges)):
            r1 = ranges[i - 1]
            r2 = ranges[i]
            theta_diff = abs(angles[i] - angles[i - 1])
            
            point_distance = np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(theta_diff))
            
            threshold = self._compute_threshold(r1, r2)
            
            if point_distance > threshold:
                if len(current_cluster_indices) >= self.min_cluster_points:
                    cluster_ranges = ranges[current_cluster_indices]
                    cluster_angles = angles[current_cluster_indices]
                    
                    x = cluster_ranges * np.cos(cluster_angles)
                    y = cluster_ranges * np.sin(cluster_angles)
                    points = np.column_stack((x, y))
                    clusters.append(points)
                    
                current_cluster_indices = [i]
            else:
                current_cluster_indices.append(i)
                
        if len(current_cluster_indices) >= self.min_cluster_points:
            cluster_ranges = ranges[current_cluster_indices]
            cluster_angles = angles[current_cluster_indices]
            
            x = cluster_ranges * np.cos(cluster_angles)
            y = cluster_ranges * np.sin(cluster_angles)
            points = np.column_stack((x, y))
            clusters.append(points)
            
        return clusters
