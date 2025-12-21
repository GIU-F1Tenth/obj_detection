import numpy as np
from typing import Optional
from nav_msgs.msg import OccupancyGrid
from scipy.ndimage import binary_dilation


class BoundaryFilter:
    def __init__(self, inflation_radius: float = 0.5, occupancy_threshold: int = 50,
                 check_radius: int = 2):
        self.inflation_radius = inflation_radius
        self.occupancy_threshold = occupancy_threshold
        self.check_radius = check_radius
        self.inflated_map: Optional[np.ndarray] = None
        self.raw_map: Optional[np.ndarray] = None
        self.map_resolution: float = 0.0
        self.map_origin: tuple = (0.0, 0.0)
        self.map_width: int = 0
        self.map_height: int = 0
        
    def update_map(self, occupancy_grid: OccupancyGrid):
        self.map_width = occupancy_grid.info.width
        self.map_height = occupancy_grid.info.height
        self.map_resolution = occupancy_grid.info.resolution
        self.map_origin = (
            occupancy_grid.info.origin.position.x,
            occupancy_grid.info.origin.position.y
        )
        
        grid_data = np.array(occupancy_grid.data).reshape((self.map_height, self.map_width))
        self.raw_map = grid_data
        
        occupied = grid_data > self.occupancy_threshold
        unknown = grid_data == -1
        obstacles = occupied | unknown
        
        inflation_cells = max(1, int(self.inflation_radius / self.map_resolution))
        self.inflated_map = binary_dilation(obstacles, iterations=inflation_cells)
        
    def is_on_boundary(self, point: np.ndarray) -> bool:
        if self.inflated_map is None:
            return False
            
        grid_x = int((point[0] - self.map_origin[0]) / self.map_resolution)
        grid_y = int((point[1] - self.map_origin[1]) / self.map_resolution)
        
        if not (0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height):
            return True
        
        for dx in range(-self.check_radius, self.check_radius + 1):
            for dy in range(-self.check_radius, self.check_radius + 1):
                check_x = grid_x + dx
                check_y = grid_y + dy
                
                if 0 <= check_x < self.map_width and 0 <= check_y < self.map_height:
                    if self.inflated_map[check_y, check_x]:
                        return True
                        
        return False
    
    def filter_points(self, points: np.ndarray) -> np.ndarray:
        if self.inflated_map is None or len(points) == 0:
            return points
            
        valid_mask = np.array([not self.is_on_boundary(point) for point in points])
        return points[valid_mask] if np.any(valid_mask) else np.empty((0, 2))
    
    def filter_cluster_centroids(self, centroids: np.ndarray, 
                                  cluster_points: list) -> tuple:
        if self.inflated_map is None or len(centroids) == 0:
            return centroids, cluster_points
            
        valid_indices = []
        for i, centroid in enumerate(centroids):
            if len(cluster_points[i]) == 0:
                continue
                
            points_on_boundary = sum(1 for p in cluster_points[i] 
                                    if self.is_on_boundary(p))
            boundary_ratio = points_on_boundary / len(cluster_points[i])
            
            if boundary_ratio < 0.5 and not self.is_on_boundary(centroid):
                valid_indices.append(i)
                
        if not valid_indices:
            return np.empty((0, 2)), []
            
        return centroids[valid_indices], [cluster_points[i] for i in valid_indices]


class AdaptiveLidarFilter:
    def __init__(self, window_size: int = 5, threshold: float = 0.3):
        self.window_size = window_size
        self.threshold = threshold
        self.history = []
        
    def filter_scan(self, ranges: np.ndarray) -> np.ndarray:
        filtered_ranges = ranges.copy()
        n = len(ranges)
        
        for i in range(n):
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(n, i + self.window_size // 2 + 1)
            window = ranges[start_idx:end_idx]
            
            valid_window = window[~np.isnan(window) & ~np.isinf(window)]
            if len(valid_window) > 0:
                median = np.median(valid_window)
                if abs(ranges[i] - median) > self.threshold * median:
                    filtered_ranges[i] = median
                    
        return filtered_ranges
