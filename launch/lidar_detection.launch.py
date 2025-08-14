#!/usr/bin/env python3
"""
Launch file for the advanced LiDAR Object Detection Node.

This launch file starts the LiDAR object detection node with proper configuration
and visualization tools for F1TENTH opponent detection and tracking.

Usage:
    ros2 launch obj_detection lidar_detection.launch.py
    
Optional arguments:
    config_file:=path/to/config.yaml    # Custom configuration file
    use_rviz:=true                      # Launch RViz for visualization
    laser_topic:=/scan                  # LiDAR topic name
    debug:=false                        # Enable debug mode
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate the launch description for LiDAR object detection."""

    # Package directory
    pkg_dir = FindPackageShare('obj_detection')

    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            pkg_dir, 'config', 'lidar_detection_config.yaml'
        ]),
        description='Path to the configuration file'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz for visualization'
    )

    laser_topic_arg = DeclareLaunchArgument(
        'laser_topic',
        default_value='/scan',
        description='LiDAR topic name'
    )

    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug mode with verbose logging'
    )

    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='laser',
        description='Frame ID for the LiDAR sensor'
    )

    # LiDAR Object Detection Node
    lidar_detection_node = Node(
        package='obj_detection',
        executable='lidar_obj_detection_node',
        name='lidar_object_detection_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
        remappings=[
            ('/scan', LaunchConfiguration('laser_topic')),
        ],
        arguments=['--ros-args', '--log-level',
                   'DEBUG' if LaunchConfiguration('debug') else 'INFO']
    )

    # RViz configuration for visualization
    rviz_config_file = PathJoinSubstitution([
        pkg_dir, 'config', 'lidar_detection_rviz.rviz'
    ])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen'
    )

    # Static transform publisher for visualization (if needed)
    static_transform_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_laser_tf',
        arguments=['0', '0', '0', '0', '0', '0',
                   'base_link', LaunchConfiguration('frame_id')],
        output='screen'
    )

    return LaunchDescription([
        # Launch arguments
        config_file_arg,
        use_rviz_arg,
        laser_topic_arg,
        debug_arg,
        frame_id_arg,

        # Nodes
        lidar_detection_node,
        static_transform_node,
        rviz_node,
    ])


if __name__ == '__main__':
    generate_launch_description()
