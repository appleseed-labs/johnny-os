from setuptools import find_packages, setup
import os
from glob import glob

package_name = "forest_planning"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "data"), glob("data/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TODO",  # See package.xml
    maintainer_email="todo@todo.todo",  # See package.xml
    description="TODO: Package description",  # See package.xml
    license="Apache-2.0",  # See package.xml
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [f"forest_planner = {package_name}.forest_planner:main"],
    },
)
