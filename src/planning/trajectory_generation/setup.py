from setuptools import find_packages, setup

package_name = "trajectory_generation"

setup(
    name=package_name,
    version="0.0.0",  # See package.xml
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Joyce",
    maintainer_email="junyuzhu@andrew.cmu.edu",  # See package.xml
    description="TODO: Package description",  # See package.xml
    license="TODO: License declaration",  # See package.xml
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"planner = {package_name}.obstacle_detection:main"
            # ,
            # f"twist_tester = {package_name}.path_planning:main",
        ],
    },
)
