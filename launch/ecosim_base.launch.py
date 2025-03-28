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

MAP_ORIGIN_LAT_LON_ALT_DEGREES = [40.443166012335624, -79.940285695498559, 288.0961589]


def generate_launch_description():

    # Load our robot description, which handles the joint states and transforms
    with open("description/xarm6/xarm6_with_gripper.urdf", "r") as f:
        robot_description = f.read()

        # robot_state_publisher_node = Node(
        #     package="robot_state_publisher",
        #     executable="robot_state_publisher",
        #     output="screen",
        #     parameters=[{"robot_description": robot_description}],
        #     remappings=[
        #         ("/tf", "tf"),
        #         ("/tf_static", "tf_static"),
        #     ],
        # )

    description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("johnny_description"),
                    "launch",
                    "description.launch.py",
                ]
            )
        ),
        launch_arguments={}.items(),
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

    motion_controller = Node(
        package="ros_motion_controller",
        executable="motion_controller",
    )

    waypoint_publisher = Node(
        package="ros_waypoint",
        executable="waypoint_node",
    )

    trajectory_planner = Node(package="trajectory_planning", executable="planner")

    rosbridge_server = Node(
        package="rosbridge_server",
        executable="rosbridge_websocket",
        parameters=[{"ros_port": 9090}],
    )

    xarm_controller = Node(
        package="xarm_control",
        executable="xarm_control",
        parameters=[{"sim_only": False}],
    )

    fix_to_transform_node = Node(
        package="gnss_interface",
        executable="fix_to_transform_node",
        parameters=[{"map_origin_lat_lon_alt_degrees": MAP_ORIGIN_LAT_LON_ALT_DEGREES}],
    )

    return LaunchDescription(
        [
            # INFRASTRUCTURE
            description_launch,
            # robot_state_publisher_node,
            # INTERFACES
            # rosbridge_server,
            unity_endpoint,
            # PERCEPTION
            # PLANNING
            fix_to_transform_node,
            # motion_controller,
            # trajectory_planner,
            # waypoint_publisher,
            # VISUALIZATION
            # rviz2,
        ]
        # + controller_nodes
    )
