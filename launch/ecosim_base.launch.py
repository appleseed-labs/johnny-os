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

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "{}/controller_manager".format(""),
        ],
    )

    add_gripper = True

    controllers = ["xarm6_traj_controller"]
    if add_gripper:
        controllers.append("xarm_gripper_traj_controller")

    # Load controllers
    controller_nodes = []
    for controller in controllers:
        controller_nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                output="screen",
                arguments=[controller, "--controller-manager", "/controller_manager"],
            )
        )

    return LaunchDescription(
        [
            # joint_state_broadcaster,
            # robot_moveit_common_launch,
            robot_state_publisher_node,
            # ros2_control_node,
            unity_endpoint,
        ]
        # + controller_nodes
    )
