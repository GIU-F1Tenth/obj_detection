from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'obj_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include ALL launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),

        # Include ALL config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Salma Tarek',
    maintainer_email='sosototo427@gmail.com',
    description='LiDAR-based object detection and clustering for F1TENTH autonomous racing',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_obj_detection_node_v1 = lidar_obj_detection.lidar_sub:main',
            'lidar_obj_detection_node_v2 = lidar_obj_detection.lidar_subscriber_salma:main',
            "lidar_cluster_exe = lidar_obj_detection.lidar_cluster_node:main",
            "lidar_obj_detection_node = lidar_obj_detection.new_object_detector:main",
            "lidar_obj_detection_node_v3 = lidar_obj_detection.lidar_sub_modi:main",
        ],
    },
)
