from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os

def generate_launch_description():
    ld = LaunchDescription()
    
    param_path = os.path.join(get_package_share_directory("obj_detection"), "config", "cluster_config.yaml")
    clustering_node = Node(
            package='obj_detection',
            executable='lidar_cluster_exe',
            name='lidar_cluster_node',
            parameters=[param_path],
            output='screen'
        )
    
    ld.add_action(clustering_node)

    return ld