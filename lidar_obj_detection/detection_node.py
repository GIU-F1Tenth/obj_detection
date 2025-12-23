#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np

from lidar_obj_detection.adaptive_breakpoint import AdaptiveBreakpoint
from lidar_obj_detection.filters import AdaptiveLidarFilter, BoundaryFilter
from lidar_obj_detection.frenet_utils import FrenetConverter, FrenetBoundaryFilter
from lidar_obj_detection.rectangle_fitting import RectangleFitter
from lidar_obj_detection.frenet_tracking import FrenetObjectTracker
from lidar_obj_detection.utils import filter_by_range


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('use_ekf_filter', True),
                ('use_boundary_filter', True),
                ('use_adaptive_breakpoint', True),
                ('use_frenet_filtering', False),
                ('min_range', 0.2),
                ('max_range', 10.0),
                ('breakpoint_base_threshold', 0.3),
                ('breakpoint_adaptive_coeff', 0.05),
                ('min_cluster_points', 5),
                ('inflation_radius', 4.0),
                ('occupancy_threshold', 50),
                ('boundary_check_radius', 2),
                ('min_bbox_width', 0.15),
                ('max_bbox_width', 1.5),
                ('min_bbox_height', 0.15),
                ('max_bbox_height', 1.5),
                ('track_width', 2.0),
                ('left_boundary_inflation', 0.5),
                ('right_boundary_inflation', 0.5),
                ('tracker_dt', 0.1),
                ('tracker_max_distance', 2.0),
                ('tracker_max_age', 5),
                ('track_length', 100.0),
                ('vs_target', 5.0),
                ('los_distance', 10.0),
                ('p_vs', 0.5),
                ('p_d', 0.8),
                ('p_vd', 0.8),
                ('process_noise_s', 0.1),
                ('process_noise_vs', 0.5),
                ('process_noise_d', 0.1),
                ('process_noise_vd', 0.5),
                ('meas_noise_s', 0.2),
                ('meas_noise_vs', 0.5),
                ('meas_noise_d', 0.2),
                ('meas_noise_vd', 0.5),
            ]
        )
        
        self.use_ekf_filter = self.get_parameter('use_ekf_filter').value
        self.use_boundary_filter = self.get_parameter('use_boundary_filter').value
        self.use_adaptive_breakpoint = self.get_parameter('use_adaptive_breakpoint').value
        self.use_frenet_filtering = self.get_parameter('use_frenet_filtering').value
        self.min_range = self.get_parameter('min_range').value
        self.max_range = self.get_parameter('max_range').value
        
        self.lidar_filter = AdaptiveLidarFilter()
        
        self.boundary_filter = BoundaryFilter(
            inflation_radius=self.get_parameter('inflation_radius').value,
            occupancy_threshold=self.get_parameter('occupancy_threshold').value,
            check_radius=self.get_parameter('boundary_check_radius').value
        )
        
        self.breakpoint_clusterer = AdaptiveBreakpoint(
            base_threshold=self.get_parameter('breakpoint_base_threshold').value,
            adaptive_coeff=self.get_parameter('breakpoint_adaptive_coeff').value,
            min_cluster_points=self.get_parameter('min_cluster_points').value
        )
        
        self.rectangle_fitter = RectangleFitter(
            min_bbox_width=self.get_parameter('min_bbox_width').value,
            max_bbox_width=self.get_parameter('max_bbox_width').value,
            min_bbox_height=self.get_parameter('min_bbox_height').value,
            max_bbox_height=self.get_parameter('max_bbox_height').value
        )
        
        self.frenet_converter = FrenetConverter()
        
        self.frenet_boundary_filter = FrenetBoundaryFilter(
            frenet_converter=self.frenet_converter,
            left_boundary_inflation=self.get_parameter('left_boundary_inflation').value,
            right_boundary_inflation=self.get_parameter('right_boundary_inflation').value,
            track_width=self.get_parameter('track_width').value
        )
        
        ekf_params = {
            'p_vs': self.get_parameter('p_vs').value,
            'p_d': self.get_parameter('p_d').value,
            'p_vd': self.get_parameter('p_vd').value,
            'process_noise_s': self.get_parameter('process_noise_s').value,
            'process_noise_vs': self.get_parameter('process_noise_vs').value,
            'process_noise_d': self.get_parameter('process_noise_d').value,
            'process_noise_vd': self.get_parameter('process_noise_vd').value,
            'meas_noise_s': self.get_parameter('meas_noise_s').value,
            'meas_noise_vs': self.get_parameter('meas_noise_vs').value,
            'meas_noise_d': self.get_parameter('meas_noise_d').value,
            'meas_noise_vd': self.get_parameter('meas_noise_vd').value,
        }
        
        self.ego_s = 0.0
        self.ego_vs = 0.0 
        self.ego_vd = 0.0 
        self.ego_position_updated = False
        self.is_circular_track = True
        
        self.tracker = FrenetObjectTracker(
            dt=self.get_parameter('tracker_dt').value,
            max_distance=self.get_parameter('tracker_max_distance').value,
            max_age=self.get_parameter('tracker_max_age').value,
            track_length=self.get_parameter('track_length').value,
            vs_target=self.get_parameter('vs_target').value,
            los_distance=self.get_parameter('los_distance').value,
            is_circular_track=self.is_circular_track,
            **ekf_params
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )
        
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10
        )
        
        self.raceline_sub = self.create_subscription(
            Path, 'pp_path', self.raceline_callback, 10
        )
        
        self.ego_odom_sub = self.create_subscription(
            Odometry, 'ego_racecar/odom', self.ego_odom_callback, 10
        )
        
        self.marker_pub = self.create_publisher(MarkerArray, 'detected_objects', 10)
        self.bbox_pub = self.create_publisher(MarkerArray, 'object_bboxes', 10)
        self.velocity_marker_pub = self.create_publisher(MarkerArray, 'object_velocities', 10)
        
        self.get_logger().info('Object Detection Node initialized')
        
    def map_callback(self, msg: OccupancyGrid):
        if self.use_boundary_filter:
            self.boundary_filter.update_map(msg)
            self.get_logger().info('Map updated for boundary filtering')
    
    def ego_odom_callback(self, msg: Odometry):
        position = msg.pose.pose.position
        velocity = msg.twist.twist.linear
        
        if self.frenet_converter.raceline_s is not None:
            try:
                s, d = self.frenet_converter.cartesian_to_frenet(position.x, position.y)
                self.ego_s = s
                
                self.ego_vs, self.ego_vd = self.frenet_converter.cartesian_velocity_to_frenet(
                    s, d, velocity.x, velocity.y
                )
                
                if not self.ego_position_updated:
                    self.get_logger().info(f'Ego position initialized: s={s:.2f}, d={d:.2f}, vs={self.ego_vs:.2f}m/s')
                    self.ego_position_updated = True
            except Exception as e:
                self.get_logger().error(f'Failed to convert ego position to Frenet: {e}')
        else:
            if not self.ego_position_updated:
                self.get_logger().warn('Raceline not yet available for ego position conversion', throttle_duration_sec=5.0)
        
    def raceline_callback(self, msg: Path):
        if len(msg.poses) < 2:
            return
            
        s_coords = []
        x_coords = []
        y_coords = []
        
        s = 0.0
        s_coords.append(s)
        x_coords.append(msg.poses[0].pose.position.x)
        y_coords.append(msg.poses[0].pose.position.y)
        
        for i in range(1, len(msg.poses)):
            dx = msg.poses[i].pose.position.x - msg.poses[i-1].pose.position.x
            dy = msg.poses[i].pose.position.y - msg.poses[i-1].pose.position.y
            ds = np.sqrt(dx**2 + dy**2)
            s += ds
            s_coords.append(s)
            x_coords.append(msg.poses[i].pose.position.x)
            y_coords.append(msg.poses[i].pose.position.y)
            
        self.frenet_converter.update_raceline(
            np.array(s_coords), np.array(x_coords), np.array(y_coords)
        )
        
        # Update tracker's track length for proper s-coordinate wrapping
        if self.is_circular_track:
            self.tracker.update_track_length(s)
            self.get_logger().info(f'Raceline updated with {len(msg.poses)} points, circular track length: {s:.2f}m')
        else:
            self.get_logger().info(f'Raceline updated with {len(msg.poses)} points, total length: {s:.2f}m')
            
    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        
        if self.use_ekf_filter:
            ranges = self.lidar_filter.filter_scan(ranges)
        
        valid_indices = ~np.isnan(ranges) & ~np.isinf(ranges)
        valid_indices &= (ranges >= self.min_range) & (ranges <= self.max_range)
        
        ranges = ranges[valid_indices]
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        angles = angles[valid_indices]
        
        if len(ranges) == 0:
            self.publish_markers([], [], msg.header.frame_id)
            return
        
        if self.use_adaptive_breakpoint:
            clusters = self.breakpoint_clusterer.cluster(ranges, angles)
        else:
            from lidar_obj_detection.utils import lidar_to_cartesian
            points = lidar_to_cartesian(ranges, angles)
            points = filter_by_range(points, self.min_range, self.max_range)
            clusters = [points]
        
        if self.use_boundary_filter:
            filtered_clusters = []
            for cluster in clusters:
                filtered_points = self.boundary_filter.filter_points(cluster)
                if len(filtered_points) > 0:
                    filtered_clusters.append(filtered_points)
            clusters = filtered_clusters
            
        if self.use_frenet_filtering and self.frenet_converter.raceline_s is not None:
            clusters = self.frenet_boundary_filter.filter_clusters(clusters)
        
        if len(clusters) == 0:
            self.publish_markers([], [], msg.header.frame_id)
            return
        
        bounding_boxes = self.rectangle_fitter.fit_and_filter(clusters)
        
        if len(bounding_boxes) == 0:
            self.publish_markers([], [], msg.header.frame_id)
            return
        
        detections_frenet = []
        for bbox in bounding_boxes:
            if self.frenet_converter.raceline_s is not None:
                try:
                    s, d = self.frenet_converter.cartesian_to_frenet(bbox.center_x, bbox.center_y)
                    detections_frenet.append((s, d, bbox.center_x, bbox.center_y))
                except Exception as e:
                    self.get_logger().warn(f'Failed to convert detection to Frenet: {e}', throttle_duration_sec=5.0)
                    detections_frenet.append((0.0, 0.0, bbox.center_x, bbox.center_y))
            else:
                detections_frenet.append((0.0, 0.0, bbox.center_x, bbox.center_y))
        
        # Only update tracker if ego position has been properly initialized
        if self.ego_position_updated or self.frenet_converter.raceline_s is None:
            tracked_objects = self.tracker.update(
                detections_frenet, self.ego_s, self.frenet_converter,
                ego_vs=self.ego_vs, ego_vd=self.ego_vd
            )
        else:
            self.get_logger().warn('Skipping tracking update: ego position not yet initialized', throttle_duration_sec=2.0)
            tracked_objects = []
        
        self.publish_markers(bounding_boxes, tracked_objects, msg.header.frame_id)
        
    def publish_markers(self, bounding_boxes, tracked_objects, frame_id):
        cluster_markers = MarkerArray()
        bbox_markers = MarkerArray()
        velocity_markers = MarkerArray()
        
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        cluster_markers.markers.append(delete_marker)
        bbox_markers.markers.append(Marker(action=Marker.DELETEALL))
        velocity_markers.markers.append(Marker(action=Marker.DELETEALL))
        
        for i, bbox in enumerate(bounding_boxes):
            points_marker = Marker()
            points_marker.header.frame_id = frame_id
            points_marker.header.stamp = self.get_clock().now().to_msg()
            points_marker.ns = 'cluster_points'
            points_marker.id = i
            points_marker.type = Marker.POINTS
            points_marker.action = Marker.ADD
            
            points_marker.scale.x = 0.05
            points_marker.scale.y = 0.05
            
            points_marker.color.r = 0.0
            points_marker.color.g = 1.0
            points_marker.color.b = 0.0
            points_marker.color.a = 0.6
            
            from lidar_obj_detection.utils import points_to_point_list
            points_marker.points = points_to_point_list(bbox.points)
            cluster_markers.markers.append(points_marker)
            
            bbox_marker = Marker()
            bbox_marker.header.frame_id = frame_id
            bbox_marker.header.stamp = self.get_clock().now().to_msg()
            bbox_marker.ns = 'bounding_boxes'
            bbox_marker.id = i
            bbox_marker.type = Marker.LINE_STRIP
            bbox_marker.action = Marker.ADD
            
            corners = bbox.get_corners()
            for corner in corners:
                p = Point()
                p.x = float(corner[0])
                p.y = float(corner[1])
                p.z = 0.0
                bbox_marker.points.append(p)
            bbox_marker.points.append(bbox_marker.points[0])
            
            bbox_marker.scale.x = 0.05
            
            bbox_marker.color.r = 1.0
            bbox_marker.color.g = 0.0
            bbox_marker.color.b = 0.0
            bbox_marker.color.a = 1.0
            
            bbox_markers.markers.append(bbox_marker)
        
        for obj in tracked_objects:
            if obj.age > 1:
                continue
            
            # Compute velocity magnitude for display
            speed = obj.get_speed()
            
            # Skip visualization for very slow objects (likely static)
            if speed < 0.1:
                continue
                
            arrow_marker = Marker()
            arrow_marker.header.frame_id = frame_id
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.ns = 'velocities'
            arrow_marker.id = obj.id
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            
            start = Point()
            start.x = float(obj.x)
            start.y = float(obj.y)
            start.z = 0.0
            
            # Use Cartesian velocities for accurate visualization
            # Scale by 0.5 seconds for visualization length
            end = Point()
            end.x = float(obj.x + obj.vx * 0.5)
            end.y = float(obj.y + obj.vy * 0.5)
            end.z = 0.0
            
            arrow_marker.points = [start, end]
            
            arrow_marker.scale.x = 0.1   # Shaft diameter
            arrow_marker.scale.y = 0.15  # Head diameter
            arrow_marker.scale.z = 0.2   # Head length
            
            if obj.is_static:
                arrow_marker.color.r = 1.0
                arrow_marker.color.g = 1.0
                arrow_marker.color.b = 0.0
            else:
                arrow_marker.color.r = 0.0
                arrow_marker.color.g = 0.0
                arrow_marker.color.b = 1.0
            arrow_marker.color.a = 1.0
            
            # Add text marker with velocity info
            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = 'velocity_text'
            text_marker.id = obj.id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = float(obj.x)
            text_marker.pose.position.y = float(obj.y)
            text_marker.pose.position.z = 0.5
            
            text_marker.text = f"ID:{obj.id} {speed:.2f}m/s"
            
            text_marker.scale.z = 0.3
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            velocity_markers.markers.append(arrow_marker)
            velocity_markers.markers.append(text_marker)
            
        self.marker_pub.publish(cluster_markers)
        self.bbox_pub.publish(bbox_markers)
        self.velocity_marker_pub.publish(velocity_markers)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
