"""
F1TENTH Object Detection Launch Configuration.

This launch file configures and starts the LiDAR clustering node for object
detection in F1TENTH racing environments. It handles parameter loading,
node configuration, and output management for the clustering system.

Author: F1TENTH Object Detection Team
License: MIT
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os


def generate_launch_description():
    """
    Generate the launch description for the object detection system.

    Creates a ROS2 launch configuration that:
    1. Loads clustering parameters from the configuration file
    2. Starts the LiDAR clustering node with proper configuration
    3. Configures output display for monitoring and debugging

    The launch system handles node lifecycle management and ensures
    proper parameter propagation to the clustering algorithm.

    Returns:
        LaunchDescription: Complete launch configuration for the object detection system
    """
    ld = LaunchDescription()

    # Locate the parameter configuration file
    # This file contains tuning parameters for the clustering algorithm
    param_path = os.path.join(
        get_package_share_directory("obj_detection"),
        "config",
        "cluster_config.yaml"
    )

    # Configure the main clustering node
    clustering_node = Node(
        package='obj_detection',              # Package containing the executable
        executable='lidar_cluster_exe',       # Main clustering executable
        name='lidar_cluster_node',            # Node name for ROS2 graph
        # Load parameters from config file
        parameters=[param_path],
        output='screen'                       # Display node output in terminal
    )

    # Add the clustering node to the launch description
    ld.add_action(clustering_node)

    return ld
