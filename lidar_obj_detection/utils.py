import numpy as np
from typing import List
from geometry_msgs.msg import Point


def lidar_to_cartesian(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return np.column_stack((x, y))


def filter_by_range(points: np.ndarray, min_range: float, max_range: float) -> np.ndarray:
    distances = np.linalg.norm(points, axis=1)
    mask = (distances >= min_range) & (distances <= max_range)
    return points[mask]


def points_to_point_list(points: np.ndarray) -> List[Point]:
    return [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in points]
