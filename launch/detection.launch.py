from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os


def generate_launch_description():
    ld = LaunchDescription()

    config_path = os.path.join(
        get_package_share_directory("obj_detection"),
        "config",
        "detection_config.yaml"
    )

    detection_node = Node(
        package='obj_detection',
        executable='detection_node',
        name='object_detection',
        parameters=[config_path],
        output='screen'
    )

    ld.add_action(detection_node)

    return ld
