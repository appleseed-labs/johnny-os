import numpy as np
import rclpy
from array import array as Array
from rclpy.node import Node, ParameterDescriptor, ParameterType
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from time import time

# from tqdm import tqdm, trange
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import norm
from matplotlib import pyplot as plt
from matplotlib import patches

# import cv2
from enum import IntEnum
import json
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
import utm
import math
from scipy.spatial.transform.rotation import Rotation as R
from scipy import stats

# from skimage.draw import disk


# ROS2 message definitions
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

# from geographic_msgs.msg import GeoPoint
from geometry_msgs.msg import Pose, Point, Twist, PoseStamped, PointStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path

from johnny_msgs.msg import (
    FailedChecks,
    HealthCheck,
    SystemwideStatus,
    TrajectoryCandidates,
    TrajectoryCandidate,
    Mode,
    PlantingPlan,
    Seedling,
)
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Header, String, Float32, Bool, Empty


class SimpleGoalFollower(Node):
    def __init__(self):
        super().__init__("simple_goal_follower")

        # Create transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Controller parameters
        self.declare_parameter("linear_gain", 0.1)
        self.declare_parameter("angular_gain", 0.10)
        self.declare_parameter("distance_tolerance", 0.3)
        self.declare_parameter("control_frequency", 10.0)

        self.linear_gain = self.get_parameter("linear_gain").value
        self.angular_gain = self.get_parameter("angular_gain").value
        self.distance_tolerance = self.get_parameter("distance_tolerance").value
        self.control_frequency = self.get_parameter("control_frequency").value

        # Goal pose
        self.goal_pose = None
        self.goal_frame_id = None

        # Subscriptions and publishers
        self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 1)
        self.twist_pub = self.create_publisher(Twist, "/cmd_vel", 1)

        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, self.spin_controller
        )

    def goal_cb(self, msg: PoseStamped):
        """Callback for new goal poses"""
        self.get_logger().info(
            f"Received goal pose: {msg.pose.position.x}, {msg.pose.position.y}"
        )
        self.goal_pose = msg.pose
        self.goal_frame_id = msg.header.frame_id

    def get_robot_pose(self):
        """Get the current robot pose from tf"""
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )

            robot_pose = Pose()
            robot_pose.position.x = transform.transform.translation.x
            robot_pose.position.y = transform.transform.translation.y
            robot_pose.position.z = transform.transform.translation.z
            robot_pose.orientation = transform.transform.rotation

            return robot_pose, True
        except TransformException as ex:
            self.get_logger().warning(f"Could not get robot pose: {ex}")
            return None, False

    def transform_pose_to_map_frame(self, pose, source_frame):
        """Transform a pose from source_frame to map frame"""
        if source_frame == "map":
            return pose, True

        try:
            # Create a PoseStamped to transform
            ps = PoseStamped()
            ps.header.frame_id = source_frame
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose = pose

            # Look up the transform
            transform = self.tf_buffer.lookup_transform(
                "map", source_frame, rclpy.time.Time()
            )

            # Calculate the transformed pose
            transformed_pose = Pose()

            # Transform position
            transformed_pose.position.x = (
                pose.position.x + transform.transform.translation.x
            )
            transformed_pose.position.y = (
                pose.position.y + transform.transform.translation.y
            )
            transformed_pose.position.z = (
                pose.position.z + transform.transform.translation.z
            )

            # Transform orientation (simplified - just take the transformation orientation)
            transformed_pose.orientation = transform.transform.rotation

            return transformed_pose, True
        except TransformException as ex:
            self.get_logger().warning(f"Could not transform pose: {ex}")
            return None, False

    def get_yaw_from_quaternion(self, q):
        """Extract yaw from quaternion"""
        # Convert to euler angles
        x, y, z, w = q.x, q.y, q.z, q.w
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return yaw

    def spin_controller(self):
        """Main control loop that calculates and publishes velocity commands"""
        if self.goal_pose is None:
            return

        # Get current robot pose
        robot_pose, success = self.get_robot_pose()
        if not success:
            return

        # Transform goal to map frame if needed
        goal_in_map, success = self.transform_pose_to_map_frame(
            self.goal_pose, self.goal_frame_id
        )
        if not success:
            return

        # Calculate distance to goal
        dx = goal_in_map.position.x - robot_pose.position.x
        dy = goal_in_map.position.y - robot_pose.position.y
        distance = math.sqrt(dx * dx + dy * dy)

        print(f"robot pos: {robot_pose.position.x:.2f}, {robot_pose.position.y:.2f}")
        print(f"goal  pos: {goal_in_map.position.x:.2f}, {goal_in_map.position.y:.2f}")

        # Calculate heading error
        desired_heading = math.atan2(dy, dx)
        robot_yaw = self.get_yaw_from_quaternion(robot_pose.orientation)

        heading_error = desired_heading - robot_yaw

        # Normalize heading error to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # Create and publish Twist message
        twist = Twist()
        print(
            f"d: {distance}, robot_yaw: {robot_yaw}. desired_heading: {desired_heading}"
        )

        # Check if we've reached the goal
        if distance < self.distance_tolerance:
            # We've reached the goal, stop
            self.get_logger().info("Goal reached!")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            # Apply proportional control
            # Scale linear velocity based on heading error - slow down when turning more

            # Range of angular_factor: [0, 1]
            angular_factor = max(0.0, 1.0 - 0.5 * abs(heading_error) / math.pi)
            twist.linear.x = self.linear_gain * distance * angular_factor
            twist.angular.z = self.angular_gain * heading_error

        self.get_logger().info(
            f"Twist: linear.x={twist.linear.x}, angular.z={twist.angular.z}"
        )
        self.twist_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)

    node = SimpleGoalFollower()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
