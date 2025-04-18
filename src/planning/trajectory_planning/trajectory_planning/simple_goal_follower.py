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

        self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 1)

        self.tf_listen

    def goal_cb(self, msg: PoseStamped):
        self.get_logger().info(
            f"Received goal pose: {msg.pose.position.x}, {msg.pose.position.y}"
        )


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
