# F1TENTH Object Detection Package

A comprehensive ROS2 package for LiDAR-based object detection and clustering designed for F1TENTH autonomous racing applications.

## Overview

This package provides multiple LiDAR processing algorithms for detecting and tracking objects in F1TENTH racing environments. It includes clustering algorithms, filtering techniques, and visualization tools to identify other robots, obstacles, and track boundaries.

## Features

-   **Multi-algorithm Support**: Multiple LiDAR processing implementations (DBSCAN, HDBSCAN)
-   **Real-time Object Detection**: Detects and tracks racing opponents in real-time
-   **Map Integration**: Uses occupancy grids to filter static obstacles
-   **Visualization**: RViz markers for detected objects
-   **Configurable Parameters**: Easy-to-tune detection parameters
-   **Track-aware Filtering**: Specialized for racing track environments

## Package Structure

```
obj_detection/
├── lidar_obj_detection/          # Main Python package
│   ├── lidar_cluster_node.py     # Primary clustering node
│   ├── lidar_subscriber_salma.py # Basic LiDAR processing
│   ├── lidar_sub_modi.py         # Advanced clustering with map integration
│   ├── lidar_filtering.py        # LiDAR data filtering utilities
│   ├── new_object_detector.py    # Map-based obstacle detection
│   └── ...
├── launch/                       # Launch files
│   └── clustering.launch.py      # Main launch configuration
├── config/                       # Configuration files
│   └── cluster_config.yaml       # Clustering parameters
├── util/                         # Utility scripts
└── test/                        # Test files
```

## Dependencies

### ROS2 Dependencies

-   `sensor_msgs` - LiDAR data messages
-   `visualization_msgs` - RViz markers
-   `nav_msgs` - Occupancy grid maps
-   `tf2_ros` - Transform handling
-   `vision_msgs` - 3D detection messages

### Python Dependencies

-   `numpy` - Numerical computations
-   `scikit-learn` - DBSCAN clustering
-   `hdbscan` - Hierarchical clustering (optional)

## Installation

1. Clone this repository into your ROS2 workspace:

```bash
cd ~/ros2_ws/src
git clone <repository_url> obj_detection
```

2. Install Python dependencies:

```bash
pip3 install -r requirements.txt
```

3. Install ROS2 dependencies:

```bash
sudo apt install ros-humble-sensor-msgs ros-humble-visualization-msgs ros-humble-nav-msgs ros-humble-tf2-ros
```

4. Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select obj_detection
source install/setup.bash
```

## Usage

### Basic Object Detection

Launch the main clustering node:

```bash
ros2 launch obj_detection clustering.launch.py
```

### Individual Nodes

Run specific detection algorithms:

```bash
# Basic LiDAR clustering
ros2 run obj_detection lidar_cluster_exe

# Advanced clustering with map integration
ros2 run obj_detection lidar_obj_detection_node_v3

# Simple object processing
ros2 run obj_detection lidar_obj_detection_node_v2
```

### Configuration

Modify parameters in `config/cluster_config.yaml`:

```yaml
lidar_cluster_node:
    ros__parameters:
        cluster_pub_topic: "/clusters"
        scan_sub_topic: "/scan"
        laser_frame: "laser"
        base_frame: "base_link"
        map_frame: "map"
        map_sub_topic: "/map"
        outlier_diff_threshold: 0.1
        angle_thresh: 60.0 
        min_cluster_size: 0.2
        max_cluster_size: 0.25
        eps: 0.07
        min_samples: 10
        max_distance: 3.0
        safety_radius: 0.5
        occupied_threshold: 60
```

## Topics

### Subscribed Topics

-   `/scan` (sensor_msgs/LaserScan) - LiDAR scan data
-   `/map` (nav_msgs/OccupancyGrid) - Occupancy grid map (for map-based filtering)

### Published Topics

-   `/clusters` (visualization_msgs/Marker) - Detected object markers
-   `/racing_obstacles` (vision_msgs/Detection3DArray) - 3D obstacle detections
-   `/track_walls` (vision_msgs/Detection3DArray) - Track wall detections

## Algorithm Details

### Primary Clustering (lidar_cluster_node.py)

-   Uses DBSCAN clustering optimized for F1TENTH robot detection
-   Filters objects by size (20-25cm) and distance (<3m)
-   Publishes visualization markers
-   Filters map walls from clustering
-   Note that the clusters are published w.r.to the map frame

### Advanced Clustering (lidar_sub_modi.py)

-   Integrates occupancy grid maps to filter static obstacles
-   Uses HDBSCAN for improved clustering performance
-   Includes temporal consistency checking
-   Optimized for real-time performance

### Basic Processing (lidar_subscriber_salma.py)

-   Simple FOV filtering and distance thresholding
-   Separates objects into "front" and "side" categories
-   Good starting point for custom implementations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive docstrings to new functions
4. Test your changes
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

-   Salma Tarek - Initial LiDAR processing implementation
-   Hatem - Advanced clustering and optimization
-   Fam Shihata
-   George Halim
-   F1TENTH Community - Continued development

## Acknowledgments

-   F1TENTH Autonomous Racing Community
-   ROS2 Development Team
-   scikit-learn and HDBSCAN developers
