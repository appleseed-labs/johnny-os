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

    use_sim_time = LaunchConfiguration("use_sim_time", default=False)

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

    swiftnav_interface = Node(
        package="swiftnav_ros2_driver",
        executable="sbp-to-ros",
        parameters=[
            os.path.join(
                get_package_share_directory("swiftnav_ros2_driver"),
                "config",
                "settings.yaml",
            )
        ],
        remappings=[
            ("imu", "/gnss/imu"),
            ("gpsfix", "/gnss/gpsfix"),
            ("navsatfix", "/gnss/navsatfix"),
            ("baseline", "/gnss/baseline"),
            ("timereference", "/gnss/timereference"),
            ("twistwithcovariancestamped", "/gnss/twist"),
        ],
    )

    joy_node = Node(
        package="joy",
        executable="joy_node",
    )

    joystick_interface = Node(
        package="joystick_interface",
        executable="joystick_interface",
    )

    velodyne_driver_node = Node(
        package="velodyne_driver",
        executable="velodyne_driver_node",
        output="screen",
        parameters=[
            {"frame_id": "lidar_link"},
            {"device_ip": "192.168.1.201"},  # Replace with your Velodyne's IP address
            {"port": 2368},  # Default port for Velodyne
        ],
    )

    velodyne_cloud_node = Node(
        package="velodyne_pointcloud",
        executable="velodyne_transform_node",
        parameters=[
            {
                "calibration": os.path.join(
                    get_package_share_directory("velodyne_pointcloud"),
                    "params",
                    "VLP16db.yaml",
                )
            },
            {"fixed_frame": "lidar_link"},
            {"target_frame": "base_link"},
            {"use_sim_time": use_sim_time},
            {"model": "VLP16"},
        ],
    )

    lidar_filter = Node(
        package="sensor_processing",
        executable="lidar_filter",
    )

    return LaunchDescription(
        [
            # INFRASTRUCTURE
            description_launch,
            # robot_state_publisher_node,
            # INTERFACES
            # rosbridge_server,
            # unity_endpoint,
            joy_node,
            joystick_interface,
            swiftnav_interface,
            velodyne_driver_node,
            velodyne_cloud_node,
            # PERCEPTION
            lidar_filter,
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
