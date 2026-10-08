from setuptools import setup
import os
from glob import glob

package_name = 'susgrip_2f_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    zip_safe=True,
    maintainer='hongxiem',
    maintainer_email='nghia.nguyenkhac@apicoorobotic.com',
    description='Control utilities for SusGrip 2F',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'with_cmd = susgrip_2f_control.with_cmd:main',
            'with_gui = susgrip_2f_control.with_gui:main',
        ],
    },
)
