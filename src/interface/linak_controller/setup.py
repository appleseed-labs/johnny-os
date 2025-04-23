from setuptools import setup
import os
from glob import glob

package_name = "linak_controller"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # # Install launch files
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.py")),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fyandun",
    maintainer_email="fyandun@andrew.cmu.edu",
    description="TODO: Package for controlling the Linak LA36 linear actuator",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "linak_control = linak_controller.linak_control:main",
        ],
    },
)
