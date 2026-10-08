from glob import glob
from setuptools import find_packages, setup


package_name = "ur3_spatial_dataset"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/schemas", glob("schemas/*.json")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="uet_ur3",
    maintainer_email="qbao1607@gmail.com",
    description="Canonical spatial scene graph and reproducible UR3 dataset recorder.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "spatial_scene = ur3_spatial_dataset.spatial_scene_node:main",
            "dataset_recorder = ur3_spatial_dataset.dataset_recorder_node:main",
            "validate_dataset = ur3_spatial_dataset.validate_dataset:main",
        ],
    },
)
