from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="yolov8_detecter",
                executable="yolo_detect_node",
                name="yolo_detect_node",
                output="screen",
            )
        ]
    )
