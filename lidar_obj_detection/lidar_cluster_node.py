import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
import numpy as np
from sklearn.cluster import DBSCAN
from std_msgs.msg import Bool, Header
import math
import time
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_ros import Buffer, TransformListener
from tf2_ros.transform_broadcaster import TransformBroadcaster
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, Pose
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class TrackedObject:
    """Class to represent a tracked object with position, velocity, and metadata."""

    def __init__(self, track_id, position, timestamp):
        self.track_id = track_id
        self.position = np.array(position)  # [x, y] in map frame
        self.velocity = np.array([0.0, 0.0])  # [vx, vy] in map frame
        self.last_seen = timestamp
        self.lost_frames = 0
        self.creation_time = timestamp
        self.last_position = self.position.copy()
        self.size = 0.0

    def predict_position(self, dt):
        """Predict position based on current velocity."""
        return self.position + self.velocity * dt

    def update(self, new_position, timestamp, size):
        """Update object with new detection."""
        dt = timestamp - self.last_seen
        if dt > 0:
            # Update velocity using simple finite difference
            self.velocity = (np.array(new_position) - self.position) / dt

        self.last_position = self.position.copy()
        self.position = np.array(new_position)
        self.last_seen = timestamp
        self.lost_frames = 0
        self.size = size


class LidarClusterNode(Node):
    def __init__(self):
        super().__init__('lidar_cluster_node')

        # Declare parameters
        self.declare_parameter('cluster_pub_topic', '/clusters')
        self.declare_parameter('scan_sub_topic', '/scan')
        self.declare_parameter('laser_frame', 'ego_racecar/laser')
        self.declare_parameter('base_frame', 'ego_racecar/base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('map_sub_topic', '/map')
        self.declare_parameter('outlier_diff_threshold', 0.1)
        self.declare_parameter('angle_thresh', 60.0)
        self.declare_parameter('min_cluster_size', 0.20)
        self.declare_parameter('max_cluster_size', 0.25)
        self.declare_parameter('eps', 0.07)  # DBSCAN parameter
        self.declare_parameter('min_samples', 10)  # DBSCAN parameter
        # Max distance for clustering
        self.declare_parameter('max_distance', 3.0)
        # Safety radius around detected objects
        self.declare_parameter('safety_radius', 0.1)
        # Occupied cell threshold in the map
        self.declare_parameter('occupied_threshold', 60)

        # Tracking parameters
        self.declare_parameter('tracking_enabled', True)
        # Max distance to associate detection with tracked object
        self.declare_parameter('max_tracking_distance', 0.5)
        # Max frames before removing lost track
        self.declare_parameter('max_lost_frames', 10)
        self.declare_parameter('tracked_objects_topic', '/tracked_objects')

        # Get parameters
        cluster_pub_topic = self.get_parameter(
            'cluster_pub_topic').get_parameter_value().string_value
        scan_sub_topic = self.get_parameter(
            'scan_sub_topic').get_parameter_value().string_value
        laser_frame = self.get_parameter(
            'laser_frame').get_parameter_value().string_value
        base_frame = self.get_parameter(
            'base_frame').get_parameter_value().string_value
        map_sub_topic = self.get_parameter(
            'map_sub_topic').get_parameter_value().string_value
        self.map_frame = self.get_parameter(
            'map_frame').get_parameter_value().string_value
        self.outlier_diff_threshold = self.get_parameter(
            'outlier_diff_threshold').get_parameter_value().double_value
        self.angle_thresh = self.get_parameter(
            'angle_thresh').get_parameter_value().double_value
        self.min_cluster_size = self.get_parameter(
            'min_cluster_size').get_parameter_value().double_value
        self.max_cluster_size = self.get_parameter(
            'max_cluster_size').get_parameter_value().double_value
        self.eps = self.get_parameter('eps').get_parameter_value().double_value
        self.min_samples = self.get_parameter(
            'min_samples').get_parameter_value().integer_value
        self.max_distance = self.get_parameter(
            'max_distance').get_parameter_value().double_value
        self.safety_radius = self.get_parameter(
            'safety_radius').get_parameter_value().double_value
        self.occupied_threshold = self.get_parameter(
            'occupied_threshold').get_parameter_value().integer_value

        # Tracking parameters
        self.tracking_enabled = self.get_parameter(
            'tracking_enabled').get_parameter_value().bool_value
        self.max_tracking_distance = self.get_parameter(
            'max_tracking_distance').get_parameter_value().double_value
        self.max_lost_frames = self.get_parameter(
            'max_lost_frames').get_parameter_value().integer_value
        tracked_objects_topic = self.get_parameter(
            'tracked_objects_topic').get_parameter_value().string_value

        self.laser_frame = laser_frame
        self.base_link_frame = base_frame

        self.subscription = self.create_subscription(
            LaserScan,
            scan_sub_topic,
            self.scan_callback,
            10
        )
        # QoS for map subscription
        map_qos = QoSProfile(depth=10)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            map_sub_topic,
            self.map_callback,
            map_qos
        )

        self.marker_pub = self.create_publisher(Marker, cluster_pub_topic, 10)
        self.tracked_objects_pub = self.create_publisher(
            MarkerArray, tracked_objects_topic, 10)

        self.marker_id = 0
        self.counter = 0

        self.map = None

        self.car_pose_on_map = Pose()

        # Tracking data structures
        self.tracked_objects = {}  # Dictionary to store tracked objects
        self.next_track_id = 0  # Counter for assigning unique track IDs
        self.current_frame_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose_timer = self.create_timer(0.01, self.get_pose)

    def map_callback(self, msg: OccupancyGrid):
        self.map = msg

    def scan_callback(self, msg: LaserScan):

        if self.map is None:
            self.get_logger().warn("No map received yet.")
            return

        self.marker_id = 0  # reset marker ID for each scan
        self.current_frame_time = time.time()

        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        valid = (ranges > msg.range_min) & (ranges < msg.range_max)
        angles = angles[valid]
        ranges = ranges[valid]

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        points = np.stack((xs, ys), axis=1)

        if len(points) == 0:
            if self.tracking_enabled:
                self.update_tracking([])
            return

        # DBSCAN tuned for detecting ~12x12 cm robot
        clustering = DBSCAN(
            eps=self.eps, min_samples=self.min_samples).fit(points)
        labels = clustering.labels_

        # Collect valid detections
        valid_detections = []

        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # noise

            cluster = points[labels == label]
            size = np.linalg.norm(cluster.max(axis=0) - cluster.min(axis=0))
            center = cluster.mean(axis=0)
            distance = np.linalg.norm(center)

            # Debug info
            color = (0.0, 1.0, 0.0)  # green by default
            angle = np.arctan2(center[1], center[0])
            if self.min_cluster_size < size < self.max_cluster_size and distance < self.max_distance and center[0] > 0 and abs(angle) < np.radians(self.angle_thresh):
                # Transform cluster center to map frame
                point_in_laser = Pose()
                point_in_laser.position.x = center[0]
                point_in_laser.position.y = center[1]
                point_in_map = self.transform_to_map_frame(
                    point_in_laser, self.car_pose_on_map)
                grid_coords = self.world_to_grid(point_in_map)

                if self.is_occupied(grid_coords, radius=int(self.safety_radius / self.map.info.resolution)):
                    self.get_logger().warn(">> ignoring cluster, it is the map !")
                    color = (0.0, 0.0, 1.0)  # blue
                    self.publish_marker(
                        point_in_map.position.x, point_in_map.position.y, size, color)
                    continue

                self.get_logger().warn(">> Likely another robot nearby!")
                self.get_logger().info(
                    f"Position -> x: {center[0]:.2f}, y: {center[1]:.2f} in laser frame")

                # Store valid detection for tracking
                detection = {
                    'position': [point_in_map.position.x, point_in_map.position.y],
                    'size': size,
                    'laser_position': [center[0], center[1]]
                }
                valid_detections.append(detection)

                color = (1.0, 0.0, 0.0)  # red
                self.publish_marker(point_in_map.position.x,
                                    point_in_map.position.y, size, color)
                self.get_logger().info(
                    f"Cluster size: {size:.2f} m, Distance: {distance:.2f} m")

        # Update tracking with current detections
        if self.tracking_enabled:
            self.update_tracking(valid_detections)

    def get_pose(self):
        """
        Main control loop that gets robot pose and executes pure pursuit control.

        This method is called at the configured control frequency. It:
        1. Gets the current robot pose from TF2
        2. Calculates lookahead distance based on current velocity
        3. Finds the appropriate lookahead point on the path
        4. Executes pure pursuit control to track that point
        5. Publishes visualization markers for debugging
        """
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,          # target_frame
                self.base_link_frame,    # source_frame
                now,
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            trans = transform.transform.translation
            rot = transform.transform.rotation

            self.car_pose_on_map.position.x = trans.x
            self.car_pose_on_map.position.y = trans.y
            self.car_pose_on_map.orientation = rot

        except Exception as e:
            self.get_logger().warn(f"Transform not available: {e}")

    def publish_marker(self, x, y, size, color):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "clusters"
        marker.id = self.marker_id
        self.marker_id += 1

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = 0.1
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.8
        self.marker_pub.publish(marker)

    def transform_to_vehicle_frame(self, point_on_map: Pose, car_pose: Pose):
        # Convert quaternion to yaw angle
        orientation_list = [car_pose.orientation.x, car_pose.orientation.y,
                            car_pose.orientation.z, car_pose.orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        dx = point_on_map.position.x - car_pose.position.x
        dy = point_on_map.position.y - car_pose.position.y
        transformed_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        transformed_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        return transformed_x, transformed_y

    def transform_to_map_frame(self, point: Pose, car_pose: Pose):
        # Convert quaternion to yaw angle
        orientation_list = [car_pose.orientation.x, car_pose.orientation.y,
                            car_pose.orientation.z, car_pose.orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        transformed_x = (math.cos(yaw) * point.position.x -
                         math.sin(yaw) * point.position.y) + car_pose.position.x
        transformed_y = (math.sin(yaw) * point.position.x +
                         math.cos(yaw) * point.position.y) + car_pose.position.y
        transformed_pose = Pose()
        transformed_pose.position.x = transformed_x
        transformed_pose.position.y = transformed_y
        return transformed_pose

    def world_to_grid(self, pose: Pose) -> tuple:
        grid_x = int(
            (pose.position.x - self.map.info.origin.position.x) / self.map.info.resolution)
        grid_y = int(
            (pose.position.y - self.map.info.origin.position.y) / self.map.info.resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, node: tuple) -> Pose:
        pose = Pose()
        pose.position.x = node[0] * self.map.info.resolution + \
            self.map.info.origin.position.x
        pose.position.y = node[1] * self.map.info.resolution + \
            self.map.info.origin.position.y
        return pose

    def pose_to_cell(self, node: tuple):
        return node[1] * self.map.info.width + node[0]

    def pose_on_map(self, node: tuple):
        return 0 <= node[0] < self.map.info.width and 0 <= node[1] < self.map.info.height

    def is_occupied(self, node: tuple, radius: int = 20):
        """
        Check if any cell within a square radius around the given node is occupied.
        :param node: (grid_x, grid_y) tuple
        :param radius: number of cells to check around the node (default: 20)
        :return: True if any cell in the radius is occupied, False otherwise
        """
        if not self.pose_on_map(node):
            return False
        x, y = node
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if self.pose_on_map((nx, ny)):
                    cell = self.pose_to_cell((nx, ny))
                    if self.map.data[cell] > 60:  # Occupied cell threshold
                        return True
        return False

    def update_tracking(self, detections):
        """
        Update object tracking with current detections.

        Args:
            detections: List of detection dictionaries with 'position', 'size', and 'laser_position'
        """
        # Increment lost frames for all tracked objects
        for track_id in self.tracked_objects:
            self.tracked_objects[track_id].lost_frames += 1

        # Associate detections with existing tracks
        matched_tracks = set()
        matched_detections = set()

        for i, detection in enumerate(detections):
            best_track_id = None
            best_distance = float('inf')

            detection_pos = np.array(detection['position'])

            for track_id, tracked_obj in self.tracked_objects.items():
                if track_id in matched_tracks:
                    continue

                # Predict where the tracked object should be
                dt = self.current_frame_time - tracked_obj.last_seen
                predicted_pos = tracked_obj.predict_position(dt)

                distance = np.linalg.norm(detection_pos - predicted_pos)

                if distance < self.max_tracking_distance and distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                self.tracked_objects[best_track_id].update(
                    detection['position'], self.current_frame_time, detection['size']
                )
                matched_tracks.add(best_track_id)
                matched_detections.add(i)

                self.get_logger().info(
                    f"Updated track {best_track_id} at position ({detection['position'][0]:.2f}, {detection['position'][1]:.2f})"
                )

        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                new_track = TrackedObject(
                    self.next_track_id,
                    detection['position'],
                    self.current_frame_time
                )
                new_track.size = detection['size']
                self.tracked_objects[self.next_track_id] = new_track

                self.get_logger().info(
                    f"Created new track {self.next_track_id} at position ({detection['position'][0]:.2f}, {detection['position'][1]:.2f})"
                )

                self.next_track_id += 1

        # Remove lost tracks
        tracks_to_remove = []
        for track_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.lost_frames > self.max_lost_frames:
                tracks_to_remove.append(track_id)
                self.get_logger().info(f"Removing lost track {track_id}")

        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]

        # Publish tracked objects
        self.publish_tracked_objects()

    def publish_tracked_objects(self):
        """Publish tracked objects as MarkerArray."""
        marker_array = MarkerArray()

        for track_id, tracked_obj in self.tracked_objects.items():
            # Main object marker
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "tracked_objects"
            marker.id = track_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = float(tracked_obj.position[0])
            marker.pose.position.y = float(tracked_obj.position[1])
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1.0

            marker.scale.x = max(tracked_obj.size, 0.1)
            marker.scale.y = max(tracked_obj.size, 0.1)
            marker.scale.z = 0.2

            # Color based on track age and status
            if tracked_obj.lost_frames > 0:
                # Orange for lost tracks
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
                marker.color.a = 0.6
            else:
                # Green for active tracks
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.8

            marker_array.markers.append(marker)

            # Velocity arrow marker
            velocity_magnitude = np.linalg.norm(tracked_obj.velocity)
            if velocity_magnitude > 0.01:  # Only show if moving
                arrow_marker = Marker()
                arrow_marker.header = marker.header
                arrow_marker.ns = "tracked_velocities"
                arrow_marker.id = track_id
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD

                arrow_marker.pose.position.x = float(tracked_obj.position[0])
                arrow_marker.pose.position.y = float(tracked_obj.position[1])
                arrow_marker.pose.position.z = 0.15

                # Calculate arrow orientation from velocity
                arrow_yaw = math.atan2(
                    tracked_obj.velocity[1], tracked_obj.velocity[0])
                quat = quaternion_from_euler(0, 0, arrow_yaw)
                arrow_marker.pose.orientation.x = quat[0]
                arrow_marker.pose.orientation.y = quat[1]
                arrow_marker.pose.orientation.z = quat[2]
                arrow_marker.pose.orientation.w = quat[3]

                # Scale arrow based on velocity
                arrow_length = min(velocity_magnitude * 2.0, 1.0)  # Cap at 1m
                arrow_marker.scale.x = arrow_length
                arrow_marker.scale.y = 0.05
                arrow_marker.scale.z = 0.05

                arrow_marker.color.r = 0.0
                arrow_marker.color.g = 0.0
                arrow_marker.color.b = 1.0
                arrow_marker.color.a = 0.7

                marker_array.markers.append(arrow_marker)

            # Text marker with track ID
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "track_ids"
            text_marker.id = track_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = float(tracked_obj.position[0])
            text_marker.pose.position.y = float(tracked_obj.position[1])
            text_marker.pose.position.z = 0.3
            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.2
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            text_marker.text = f"ID: {track_id}\nV: {np.linalg.norm(tracked_obj.velocity):.2f} m/s"

            marker_array.markers.append(text_marker)

        # Clear old markers by sending empty markers for removed IDs
        for i in range(len(marker_array.markers), 100):  # Clear up to 100 old markers
            clear_marker = Marker()
            clear_marker.header.frame_id = self.map_frame
            clear_marker.header.stamp = self.get_clock().now().to_msg()
            clear_marker.ns = "tracked_objects"
            clear_marker.id = i
            clear_marker.action = Marker.DELETE
            marker_array.markers.append(clear_marker)

            clear_marker2 = Marker()
            clear_marker2.header = clear_marker.header
            clear_marker2.ns = "tracked_velocities"
            clear_marker2.id = i
            clear_marker2.action = Marker.DELETE
            marker_array.markers.append(clear_marker2)

            clear_marker3 = Marker()
            clear_marker3.header = clear_marker.header
            clear_marker3.ns = "track_ids"
            clear_marker3.id = i
            clear_marker3.action = Marker.DELETE
            marker_array.markers.append(clear_marker3)

        self.tracked_objects_pub.publish(marker_array)

        # Log tracking status
        if len(self.tracked_objects) > 0:
            active_tracks = sum(
                1 for obj in self.tracked_objects.values() if obj.lost_frames == 0)
            lost_tracks = len(self.tracked_objects) - active_tracks
            self.get_logger().info(
                f"Tracking: {active_tracks} active, {lost_tracks} lost tracks"
            )


def main():
    rclpy.init()
    node = LidarClusterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
