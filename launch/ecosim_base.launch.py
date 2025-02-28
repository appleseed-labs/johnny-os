import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
import yaml


def generate_launch_description():

    # Load our robot description, which handles the joint states and transforms
    with open("description/xarm6/xarm6_with_gripper.urdf", "r") as f:
        robot_description = f.read()

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
        remappings=[
            ("/tf", "tf"),
            ("/tf_static", "tf_static"),
        ],
    )

    # Connects ROS to EcoSim
    unity_endpoint = Node(
        package="ros_tcp_endpoint", executable="default_server_endpoint"
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", "param/ecosim_base.rviz"],
    )

    return LaunchDescription(
        [
            # INFRASTRUCTURE
            robot_state_publisher_node,
            # INTERFACES
            unity_endpoint,
            # PERCEPTION
            # PLANNING
            # VISUALIZATION
            rviz2,
        ]
        # + controller_nodes
    )
