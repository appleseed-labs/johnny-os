from setuptools import find_packages, setup

package_name = "sensor_processing"

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
    maintainer="Will Heitman",
    maintainer_email="will@heit.mn",
    description="See package.xml",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"gnss_processor = {package_name}.gnss_processor:main",
        ],
    },
)
