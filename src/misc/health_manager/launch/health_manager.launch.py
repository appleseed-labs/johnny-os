# launch/health_monitor.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory("health_manager"), "config")
    config_file = os.path.join(config_dir, "params.yaml")

    return LaunchDescription(
        [
            Node(
                package="health_manager",
                executable="health_manager",
                name="health_manager",
                parameters=[config_file],
                output="screen",
            )
        ]
    )
