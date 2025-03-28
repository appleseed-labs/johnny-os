from setuptools import find_packages, setup

package_name = "xarm_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="main",
    maintainer_email="will@heit.mn",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pick_and_place_node = xarm_control.pick_and_place:main",
            "xarm_control_node = xarm_control.xarm_control:main",
        ],
    },
)
