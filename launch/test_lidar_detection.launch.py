#!/usr/bin/env python3
"""
ROS2 Launch file for LiDAR Object Detection System Tests.

This launch file sets up a complete ROS2 test environment for validating
the LiDAR object detection and tracking system.

Usage:
    ros2 launch obj_detection test_lidar_detection.launch.py
    
Optional arguments:
    use_synthetic_data:=true   # Use synthetic test data publisher
    verbose:=true             # Enable verbose output
    test_scenario:=racing     # Test scenario (racing, obstacles, moving, noise)
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate the launch description for ROS2 system tests."""

    # Package directory
    pkg_dir = FindPackageShare('obj_detection')

    # Declare launch arguments
    use_synthetic_data_arg = DeclareLaunchArgument(
        'use_synthetic_data',
        default_value='true',
        description='Use synthetic test data publisher'
    )

    verbose_arg = DeclareLaunchArgument(
        'verbose',
        default_value='true',
        description='Enable verbose output'
    )

    test_scenario_arg = DeclareLaunchArgument(
        'test_scenario',
        default_value='racing',
        choices=['racing', 'obstacles', 'moving', 'noise'],
        description='Test scenario to run'
    )

    # Test configuration file
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            pkg_dir, 'config', 'test_config.yaml'
        ]),
        description='Path to the test configuration file'
    )

    # Main LiDAR detection node with test configuration
    lidar_detection_node = Node(
        package='obj_detection',
        executable='lidar_obj_detection_node',
        name='lidar_object_detection_node',
        parameters=[LaunchConfiguration('config_file')],
        output='screen',
        emulate_tty=True,
    )

    # Synthetic test data publisher
    test_data_publisher = Node(
        condition=IfCondition(LaunchConfiguration('use_synthetic_data')),
        package='obj_detection',
        executable='test_data_publisher',
        name='test_data_publisher',
        parameters=[{
            'publish_rate': 10.0,
            'test_scenario': LaunchConfiguration('test_scenario'),
            'noise_level': 0.01,
            'num_rays': 360,
        }],
        output='screen'
    )

    # RViz for visualization (optional)
    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('verbose')),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            pkg_dir, 'config', 'lidar_detection_rviz.rviz'
        ])],
        output='screen'
    )

    # Test execution with delay to allow nodes to start
    run_tests = ExecuteProcess(
        cmd=[
            'python3', '-m', 'pytest',
            PathJoinSubstitution([pkg_dir, '..', '..', 'test', 'test_ros2_integration.py']),
            '-v', '--tb=short', '-x'  # Stop on first failure
        ],
        output='screen',
        cwd=PathJoinSubstitution([pkg_dir, '..', '..']),
    )

    delayed_tests = TimerAction(
        period=3.0,  # Wait 3 seconds for nodes to initialize
        actions=[run_tests]
    )

    return LaunchDescription([
        # Arguments
        use_synthetic_data_arg,
        verbose_arg,
        test_scenario_arg,
        config_file_arg,
        
        # Core nodes
        lidar_detection_node,
        test_data_publisher,
        
        # Optional visualization
        # rviz_node,
        
        # Delayed test execution
        delayed_tests,
    ])
