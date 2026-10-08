from setuptools import setup

package_name = 'susgrip_2f_hardware'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    zip_safe=True,
    maintainer='apicoo-ai',
    maintainer_email='apicoo-ai@todo.todo',
    description='Python based ros2_control mimic for SusGrip 2F',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'hardware_interface = susgrip_2f_hardware.hardware_interface:main',
            'hardware_joint_publisher = susgrip_2f_hardware.hardware_joint_publisher:main'
        ],
    },
)
