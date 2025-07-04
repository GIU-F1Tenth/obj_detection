from setuptools import find_packages, setup

package_name = 'obj_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='salmatarek',
    maintainer_email='sosototo427@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_obj_detection_node_v1 = lidar_obj_detection.lidar_sub:main',
            'lidar_obj_detection_node_v2 = lidar_obj_detection.lidar_subscriber_salma:main',
            "lidar_cluster_exe = lidar_obj_detection.lidar_cluster_node:main"
        ],
    },
)
