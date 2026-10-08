#!/usr/bin/env python3
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'susgrip_2f_gazebo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
    ],
    zip_safe=True,
    maintainer='buiminhgiang',
    maintainer_email='[EMAIL_ADDRESS]',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'integrated_gazebo_joint_publisher = susgrip_2f_gazebo.integrated_gazebo_joint_publisher:main',
            'gazebo_joint_publisher = susgrip_2f_gazebo.gazebo_joint_publisher:main',
        ],
    },
)