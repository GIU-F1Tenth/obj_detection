import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class BoundingBox:
    center_x: float
    center_y: float
    width: float
    height: float
    rotation: float
    points: np.ndarray
    
    def get_corners(self) -> np.ndarray:
        cos_r = np.cos(self.rotation)
        sin_r = np.sin(self.rotation)
        
        dx = self.width / 2
        dy = self.height / 2
        
        corners = np.array([
            [-dx, -dy],
            [dx, -dy],
            [dx, dy],
            [-dx, dy]
        ])
        
        rotation_matrix = np.array([
            [cos_r, -sin_r],
            [sin_r, cos_r]
        ])
        
        rotated_corners = corners @ rotation_matrix.T
        translated_corners = rotated_corners + np.array([self.center_x, self.center_y])
        
        return translated_corners


class RectangleFitter:
    def __init__(self, min_bbox_width: float = 0.1, 
                 max_bbox_width: float = 2.0,
                 min_bbox_height: float = 0.1,
                 max_bbox_height: float = 2.0):
        self.min_width = min_bbox_width
        self.max_width = max_bbox_width
        self.min_height = min_bbox_height
        self.max_height = max_bbox_height
        
    def fit_rectangle(self, points: np.ndarray) -> BoundingBox:
        if len(points) < 3:
            center = np.mean(points, axis=0)
            return BoundingBox(
                center_x=center[0],
                center_y=center[1],
                width=0.3,
                height=0.3,
                rotation=0.0,
                points=points
            )
            
        center = np.mean(points, axis=0)
        centered_points = points - center
        
        cov = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        rotation = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        cos_r = np.cos(-rotation)
        sin_r = np.sin(-rotation)
        rotation_matrix = np.array([
            [cos_r, -sin_r],
            [sin_r, cos_r]
        ])
        
        rotated_points = centered_points @ rotation_matrix.T
        
        min_x, max_x = np.min(rotated_points[:, 0]), np.max(rotated_points[:, 0])
        min_y, max_y = np.min(rotated_points[:, 1]), np.max(rotated_points[:, 1])
        
        width = max_x - min_x
        height = max_y - min_y
        
        return BoundingBox(
            center_x=center[0],
            center_y=center[1],
            width=width,
            height=height,
            rotation=rotation,
            points=points
        )
        
    def is_valid_size(self, bbox: BoundingBox) -> bool:
        return (self.min_width <= bbox.width <= self.max_width and
                self.min_height <= bbox.height <= self.max_height)
                
    def fit_and_filter(self, clusters: list) -> list:
        valid_boxes = []
        for cluster in clusters:
            if len(cluster) < 3:
                continue
                
            bbox = self.fit_rectangle(cluster)
            if self.is_valid_size(bbox):
                valid_boxes.append(bbox)
                
        return valid_boxes
