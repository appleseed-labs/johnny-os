from setuptools import find_packages, setup

package_name = "yolov8_detecter"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/detect.launch.py"]),
        ("share/" + package_name, ["resource/" + package_name]),
        ("share/" + package_name + "/models", ["yolov8_detecter/best.pt"]),  # optional
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ericz",
    maintainer_email="ericz@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_detect_node = yolov8_detecter.detector_node:main",
        ],
    },
)
