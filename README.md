# Advanced LiDAR Object Detection Node (v4)

## Overview

The Advanced LiDAR Object Detection Node is a comprehensive ROS2 Python implementation designed for F1TENTH autonomous racing applications. This node provides sophisticated opponent detection and tracking capabilities using LiDAR sensor data.

## Features

### 🔍 **Advanced Filtering**

-   **Multi-stage noise reduction**: Moving average and statistical outlier removal
-   **Range and angular filtering**: Configurable detection zones
-   **Temporal filtering**: Multi-frame data fusion for stability
-   **Intensity-based filtering**: Leverages LiDAR intensity data when available

### 🎯 **Robust Object Detection**

-   **DBSCAN clustering**: Density-based spatial clustering for object segmentation
-   **Size-based validation**: Filters objects based on realistic opponent dimensions
-   **Confidence scoring**: Dynamic confidence calculation based on cluster properties
-   **Multi-object detection**: Simultaneous tracking of multiple opponents

### 📡 **Kalman Filter Tracking**

-   **State estimation**: 2D position and velocity tracking
-   **Prediction capabilities**: Anticipates object movement between scans
-   **Track association**: Intelligent matching of detections across frames
-   **Track lifecycle management**: Automatic creation and deletion of tracks

### 📊 **Comprehensive Visualization**

-   **Real-time markers**: Color-coded confidence visualization in RViz
-   **Velocity arrows**: Visual representation of object movement
-   **Track persistence**: Maintains visual continuity across frames
-   **Closest opponent highlighting**: Special visualization for the nearest threat

## Architecture

### Core Components

1. **LiDARObjectDetectionNode**: Main ROS2 node class
2. **KalmanFilter**: 2D tracking filter implementation
3. **ObjectTracker**: Multi-object tracking manager
4. **DetectedObject**: Data structure for object information

### Data Flow

```
LiDAR Scan → Filtering → Clustering → Detection → Tracking → Publishing
     ↓            ↓           ↓           ↓          ↓          ↓
Raw Points → Clean Points → Clusters → Objects → Tracks → Markers/Poses
```

## Usage

### Basic Launch

```bash
# Launch with default configuration
ros2 launch obj_detection lidar_detection.launch.py

# Launch with custom config
ros2 launch obj_detection lidar_detection.launch.py config_file:=/path/to/config.yaml

# Launch without RViz
ros2 launch obj_detection lidar_detection.launch.py use_rviz:=false

# Debug mode with verbose logging
ros2 launch obj_detection lidar_detection.launch.py debug:=true
```

### Direct Node Execution

```bash
# Run the node directly
ros2 run obj_detection lidar_obj_detection_node_v4

# With custom parameters
ros2 run obj_detection lidar_obj_detection_node_v4 --ros-args -p detection.dbscan_eps:=0.4
```

## Configuration

The node is highly configurable through ROS2 parameters. Key configuration sections include:

### Filter Parameters

```yaml
filter:
    min_range: 0.1 # Minimum valid range (m)
    max_range: 10.0 # Maximum detection range (m)
    noise_filter_window: 5 # Noise reduction window size
    outlier_threshold: 2.0 # Outlier removal threshold
```

### Detection Parameters

```yaml
detection:
    dbscan_eps: 0.3 # Clustering distance threshold (m)
    dbscan_min_samples: 3 # Minimum cluster size
    confidence_threshold: 0.5 # Minimum detection confidence
```

### Tracking Parameters

```yaml
tracking:
    max_tracking_distance: 1.0 # Track association distance (m)
    track_timeout: 2.0 # Track expiration time (s)
    max_velocity: 15.0 # Maximum believable speed (m/s)
```

## Topics

### Subscribed Topics

-   `/scan` (sensor_msgs/LaserScan): LiDAR scan data

### Published Topics

-   `/detected_objects_markers` (visualization_msgs/MarkerArray): Visualization markers
-   `/opponent_pose` (geometry_msgs/PoseStamped): Closest opponent pose

## Performance Characteristics

### Processing Speed

-   **Target Rate**: 10 Hz (100ms processing time)
-   **Typical Performance**: 20-50ms per scan on modern hardware
-   **Scalability**: Handles up to 10 simultaneous objects efficiently

### Detection Accuracy

-   **Range**: Effective from 0.1m to 10m
-   **Angular Coverage**: Full 360° coverage
-   **Object Size**: Optimized for 0.2m to 2.0m objects (racing cars)
-   **Velocity Range**: Tracks objects up to 15 m/s

## Algorithm Details

### Filtering Pipeline

1. **Range Validation**: Removes invalid distance measurements
2. **Angular Filtering**: Applies field-of-view constraints
3. **Noise Reduction**: Moving average filter with configurable window
4. **Outlier Removal**: Statistical filtering using standard deviation
5. **Temporal Smoothing**: Multi-frame consistency checking

### Detection Algorithm

1. **Point Cloud Generation**: Convert polar to Cartesian coordinates
2. **DBSCAN Clustering**: Group nearby points into potential objects
3. **Cluster Validation**: Filter by size, density, and geometry
4. **Confidence Calculation**: Score based on cluster properties
5. **Object Extraction**: Generate final detection list

### Tracking System

1. **Prediction Step**: Kalman filter time update
2. **Association**: Match detections to existing tracks
3. **Update Step**: Kalman filter measurement update
4. **Track Management**: Create new tracks, remove old ones
5. **State Estimation**: Smooth position and velocity estimates

## Tuning Guide

### For Racing Environments

-   Reduce `max_range` to 8m for better performance
-   Increase `dbscan_eps` to 0.4 for faster cars
-   Set `max_velocity` to 12 m/s for F1TENTH speeds

### For Crowded Environments

-   Decrease `dbscan_eps` to 0.2 for better separation
-   Increase `max_tracking_distance` to 1.5m
-   Reduce `track_timeout` to 1.5s for responsiveness

### For Noisy Sensors

-   Increase `noise_filter_window` to 7
-   Set `outlier_threshold` to 1.5
-   Enable `temporal_filtering` for stability

## Troubleshooting

### Common Issues

1. **No detections**: Check LiDAR topic and range parameters
2. **False positives**: Adjust confidence threshold and cluster validation
3. **Tracking instability**: Tune Kalman filter noise parameters
4. **Performance issues**: Reduce processing rate or detection range

### Debug Tools

-   Enable debug logging: `debug.enable_debug_logging: true`
-   Monitor performance: `debug.monitor_processing_time: true`
-   Visualize in RViz: Use provided configuration file

## Dependencies

### Required ROS2 Packages

-   `sensor_msgs`: LiDAR message types
-   `geometry_msgs`: Pose and transform messages
-   `visualization_msgs`: RViz marker messages
-   `std_msgs`: Standard message types

### Python Dependencies

-   `numpy`: Numerical computing
-   `scikit-learn`: DBSCAN clustering
-   `scipy`: Distance calculations
-   `dataclasses`: Data structures

## Future Enhancements

-   [ ] Deep learning integration for object classification
-   [ ] Multi-sensor fusion (camera + LiDAR)
-   [ ] Predictive path planning integration
-   [ ] Advanced association algorithms (Hungarian method)
-   [ ] Real-time parameter adaptation
-   [ ] Support for 3D LiDAR sensors

## Authors

-   **F1TENTH Team**: Core development and racing optimizations
-   **Contributors**: Various optimization and feature additions

## License

MIT License - See LICENSE file for details

---

For technical support or feature requests, please open an issue in the repository.
