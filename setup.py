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
    maintainer='salmatarek, George Halim',
    maintainer_email='sosototo427@gmail.com, georgehany064@gmail.com',
    description='Lidar object detection Package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "lidar_cluster_exe = lidar_obj_detection.lidar_cluster_node:main",
        ],
    },
)
