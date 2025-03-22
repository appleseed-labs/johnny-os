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
    hw_ns = LaunchConfiguration("hw_ns", default="xarm")
    use_sim_time = LaunchConfiguration("use_sim_time", default=False)
    no_gui_ctrl = LaunchConfiguration("no_gui_ctrl", default=False)
    show_rviz = LaunchConfiguration("show_rviz", default=True)
    attach_xyz = LaunchConfiguration("attach_xyz", default='"0 0 0"')
    attach_rpy = LaunchConfiguration("attach_rpy", default='"0 0 0"')

    with open("param/moveit_config_dict.yaml", "r") as f:
        moveit_config_dict = yaml.load(f, Loader=yaml.FullLoader)

    unity_endpoint = Node(
        package="ros_tcp_endpoint", executable="default_server_endpoint"
    )

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

    robot_moveit_common_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("xarm_moveit_config"),
                    "launch",
                    "_robot_moveit_common2.launch.py",
                ]
            )
        ),
        launch_arguments={
            "prefix": "",
            "attach_to": "world",
            "attach_xyz": attach_xyz,
            "attach_rpy": attach_rpy,
            "no_gui_ctrl": no_gui_ctrl,
            "use_sim_time": "false",
            "moveit_config_dump": yaml.dump(moveit_config_dict),
        }.items(),
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            "description/xarm6/xarm6_with_gripper.urdf",
            "param/xarm6_controllers.yaml",
            # robot_params,
        ],
        output="screen",
        remappings=[("/controller_manager/robot_description", "robot_description")],
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
        ],
    )

    return LaunchDescription(
        [
            description_launch,
            # ros2_control_node,
            # unity_endpoint,
            velodyne_driver_node,  # Add the Velodyne driver node here
            velodyne_cloud_node,  # Add the Velodyne cloud node here
        ]
    )
