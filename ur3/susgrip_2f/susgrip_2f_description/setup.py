#!/usr/bin/env python3
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'susgrip_2f_description'

current_dir = os.path.dirname(os.path.realpath(__file__))
mesh_sus_2f_rel_dir = os.path.relpath(os.path.join(current_dir, 'meshes', 'susgrip_2f'))

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes/susgrip_2f_full'), glob('meshes/susgrip_2f_full/*')),
        (os.path.join('share', package_name, 'meshes/susgrip_2f_simple'), glob('meshes/susgrip_2f_simple/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*')),
    ],
    zip_safe=True,
    maintainer='hongxiem',
    maintainer_email='nghia.nguyenkhac@apicoorobotic.com',
    description='Apicoo Robotics SusGrip 2F Gripper ROS2 package',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'susgrip_2f_visualize = susgrip_2f_description.susgrip_2f_visualize:main',
        ],
    },
)
