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


class Candidate:
    def __init__(
        self, speed: float, omega: float, trajectory: np.ndarray, cost: float = -1
    ):
        self.speed = speed
        self.omega = omega
        self.trajectory = trajectory
        self.cost = cost


class PlannerNode(Node):
    def __init__(self):
        super().__init__("trajectory_planner")

        self.setUpParameters()

        lat0, lon0, alt0 = self.get_parameter("map_origin_lat_lon_alt_degrees").value
        self.origin_utm_x, self.origin_utm_y, _, __ = utm.from_latlon(lat0, lon0)

        self.create_subscription(String, "planning/plan_json", self.planCb, 1)
        self.create_subscription(Twist, "/cmd_vel/teleop", self.teleopTwistCb, 1)
        self.create_subscription(OccupancyGrid, "/cost/total", self.totalCostCb, 1)
        self.create_subscription(Float32, "/gnss/yaw", self.egoYawCb, 1)
        self.create_subscription(Mode, "/planning/current_mode", self.currentModeCb, 1)
        self.create_subscription(Bool, "/behavior/is_planting", self.isPlantingCb, 1)
        self.create_subscription(
            Bool, "/behavior/is_turning_downhill", self.isTurningDownhillCb, 1
        )

        self.create_subscription(NavSatFix, "/gnss/fix", self.gnssFixCb, 1)

        self.create_subscription(
            PlantingPlan, "/planning/remaining_plan", self.planCb, 1
        )

        self.create_subscription(String, "/waypoints_string", self.waypointsStringCb, 1)

        self.create_subscription(Imu, "/imu", self.imuCb, 1)

        self.create_subscription(PoseStamped, "/ground_truth", self.groundTruthCb, 1)

        self.create_subscription(
            Empty, "/behavior/start_driving", self.startDrivingCb, 1
        )

        self.twist_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.facing_downhill_pub = self.create_publisher(
            Empty, "/behavior/facing_downhill", 1
        )
        self.twist_path_pub = self.create_publisher(Path, "/cmd_vel/path", 1)
        self.candidates_pub = self.create_publisher(
            TrajectoryCandidates, "/planning/candidates", 1
        )
        self.status_pub = self.create_publisher(DiagnosticStatus, "/diagnostics", 1)

        self.start_planting_pub = self.create_publisher(
            Empty, "/behavior/start_planting", 1
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info("Hello, world!")

        self.goal_point = None
        self.ego_pos = None
        self.ego_yaw = None
        self.seedling_points = []
        self.total_cost_map = None
        self.grid_info = None
        self.cached_teleop = Twist()
        self.current_mode = Mode.STOPPED
        self.is_planting = False
        self.is_turning_downhill = False
        self.closest_point_bl = None
        self.remaining_seedling_count = 0

        self.previous_twist = Twist()

        self.create_timer(0.1, self.updateTrajectorySimply)
        # self.create_timer(0.1, self.updateTrajectory)

    def groundTruthCb(self, msg: PoseStamped):
        q = msg.pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        self.ego_yaw = r.as_euler("xyz")[2] + (math.pi / 2)

        if (self.ego_yaw) > 2 * math.pi:
            self.ego_yaw -= 2 * math.pi
        if (self.ego_yaw) < 0:
            self.ego_yaw += 2 * math.pi

        # print(self.ego_yaw)

    def imuCb(self, msg: Imu):
        pass

    def gnssFixCb(self, msg: NavSatFix):
        lat = msg.latitude
        lon = msg.longitude
        x, y = self.latLonToMap(lat, lon)
        self.ego_pos = [x, y]

    def waypointsStringCb(self, msg: String):
        seedling_points_latlon = json.loads(msg.data)

        self.seedling_points = []

        # Convert each from latlon to meters
        for seedling_point_latlon in seedling_points_latlon:
            lat, lon = seedling_point_latlon
            x, y = self.latLonToMap(lat, lon)
            self.seedling_points.append([x, y])

        self.get_logger().info(f"Got {self.seedling_points} seedling points")

        self.current_mode = Mode.AUTO

    def isPlantingCb(self, msg: Bool):
        self.is_planting = msg.data

    def isTurningDownhillCb(self, msg: Bool):
        return
        self.is_turning_downhill = msg.data

    def teleopTwistCb(self, msg: Twist):
        self.cached_teleop = msg

    def currentModeCb(self, msg: Mode):
        self.current_mode = msg.level

    def egoYawCb(self, msg: Float32):
        # self.get_logger().info("Updated ego yaw")
        self.ego_yaw = msg.data

    def totalCostCb(self, msg: OccupancyGrid):
        arr = np.asarray(msg.data).reshape(msg.info.height, msg.info.width)

        self.total_cost_map = arr
        self.grid_info = msg.info

        # plt.imshow(mega_mask, extent=[-8, 12, -10, 10])
        # plt.show()

    def latLonToMap(self, lat: float, lon: float):
        lat0, lon0, _ = self.get_parameter("map_origin_lat_lon_alt_degrees").value
        origin_x, origin_y, _, __ = utm.from_latlon(lat0, lon0)

        x, y, _, __ = utm.from_latlon(lat, lon)

        x = x - origin_x
        y = y - origin_y

        return (x, y)

    def transformToBaselink(self, points):

        if len(points) < 1:
            return None

        # Form a 2D homogeneous transform matrix
        t = -self.ego_yaw
        u = -self.ego_pos[0]
        v = -self.ego_pos[1]

        H = np.identity(3)
        H[0, 2] = u
        H[1, 2] = v

        R = np.asarray(
            [
                [math.cos(t), -math.sin(t)],
                [math.sin(t), math.cos(t)],
            ]
        )

        points = np.asarray(points)

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1.scatter(points[:, 0], points[:, 1])
        # ax1.scatter(*self.ego_pos, c="red")
        # ax1.set_title(f"Yaw: {self.ego_yaw}")

        # Transform to base_link
        try:
            pts_homog = np.vstack((points.T, np.ones(len(points))))
            # self.get_logger().info(f"{pts_homog}")
            pts_tfed = H @ pts_homog
            pts_tfed = pts_tfed.T
            pts_tfed = pts_tfed[:, :-1]
            # self.get_logger().info(f"{self.ego_yaw}")
            # self.get_logger().info(f"{pts_tfed}")

        except ValueError as e:
            # There were no points nearby
            return None

        # ax2.scatter(pts_tfed[:, 0], pts_tfed[:, 1])
        # ax2.scatter(0, 0, c="red")
        # ax2.set_xlim((-20, 20))
        # ax2.set_ylim((-20, 20))

        pts_tfed = (R @ pts_tfed.T).T
        # ax3.scatter(pts_tfed[:, 0], pts_tfed[:, 1])
        # ax3.scatter(0, 0, c="red")
        # ax3.set_xlim((-20, 20))
        # ax3.set_ylim((-20, 20))

        # plt.show()

        return pts_tfed

    def getClosestSeedlingInBaselink(self):
        if self.seedling_points is None or len(self.seedling_points) < 1:
            self.get_logger().warning(
                f"Could not find closest seedling point. Seedling points unknown."
            )
            return None

        if self.ego_pos is None:
            self.get_logger().error(f"Ego position unknown.")

        closest_distance = 999999.9

        closest_seedling = None

        for seedling_pt in self.seedling_points:
            seedling_x, seedling_y = seedling_pt

            dist = pdist([self.ego_pos, [seedling_x, seedling_y]])[0]

            if dist < closest_distance:
                closest_distance = dist
                closest_seedling = seedling_pt

        if closest_seedling is None:
            self.get_logger().error("Could not find closest seedling point.")
            return None
        closest_seedling_bl = self.transformToBaselink([closest_seedling])

        print(f"Ego Yaw: {self.ego_yaw}")

        if len(closest_seedling_bl) == 1:
            closest_seedling_bl = closest_seedling_bl[0]

        return closest_seedling_bl

        # if self.total_cost_map is None:
        #     self.get_logger().warning(
        #         f"Could not find closest seedling point. Seedling points unknown."
        #     )
        #     return None

        # x = self.total_cost_map
        # pixel_coords = np.unravel_index(x.argmin(), x.shape)
        # return [pixel_coords[1] - 40, pixel_coords[0] - 50]

    def planCb(self, msg: PlantingPlan):
        print(f"Got plan with {len(msg.seedlings)} seedlings")
        self.remaining_seedling_count = len(msg.seedlings)
        # for seedling in msg.seedlings:
        #     seedling: Seedling
        #     print(seedling.species_id)

    def getYawError(self, goal_point_bl):
        print(goal_point_bl)
        yaw_error = math.atan2(goal_point_bl[0], goal_point_bl[1]) - np.pi / 2
        yaw_error *= -1

        if yaw_error < -np.pi:
            yaw_error += 2 * np.pi
        if yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        # print(f"{goal_point_bl} -> {yaw_error / np.pi * 180}")

        return yaw_error

    def pointTurnFromYawError(self, yaw_error, omega=0.8, linear=0.2):
        cmd_msg = Twist()

        if yaw_error < 0:
            omega *= -1

        cmd_msg.angular.z = omega
        cmd_msg.linear.x = linear

        self.twist_pub.publish(self.getSmoothed(cmd_msg))

    def publishStatus(self, desc: str, level=DiagnosticStatus.OK):
        self.status_pub.publish(
            DiagnosticStatus(message=desc, level=level, name=self.get_name())
        )

    def getSmoothed(
        self, in_twist: Twist, linear_max_delta=0.2, angular_max_delta=0.3
    ) -> Twist:
        previous_linear_speed = self.previous_twist.linear.x
        previous_angular_speed = self.previous_twist.angular.z

        requested_linear_speed = in_twist.linear.x
        requested_angular_speed = in_twist.angular.z

        if requested_linear_speed > previous_linear_speed + linear_max_delta:
            smoothed_linear_speed = previous_linear_speed + linear_max_delta
        elif requested_linear_speed < previous_linear_speed - linear_max_delta:
            smoothed_linear_speed = previous_linear_speed - linear_max_delta
        else:
            smoothed_linear_speed = requested_linear_speed

        if requested_angular_speed > previous_angular_speed + angular_max_delta:
            smoothed_angular_speed = previous_angular_speed + angular_max_delta
        elif requested_angular_speed < previous_angular_speed - angular_max_delta:
            smoothed_angular_speed = previous_angular_speed - angular_max_delta
        else:
            smoothed_angular_speed = requested_angular_speed

        # print(f"{requested_linear_speed} -> {smoothed_linear_speed}")

        smoothed_twist = Twist()
        smoothed_twist.linear.x = smoothed_linear_speed
        smoothed_twist.angular.z = smoothed_angular_speed

        self.previous_twist = smoothed_twist
        return smoothed_twist

    def switchToPlantingMode(self):
        self.current_mode = Mode.STOPPED
        self.is_planting = True
        self.start_planting_pub.publish(Empty())

    def startDrivingCb(self, msg: Empty):
        self.current_mode = Mode.AUTO
        self.is_planting = False

        if len(self.seedling_points) > 0:
            self.seedling_points.pop(0)

        # self.start_planting_pub.publish(Empty())

    def updateTrajectorySimply(self):

        # if self.candidates_msg is not None:
        #     self.candidates_pub.publish(self.candidates_msg)
        # else:
        #     self.get_logger().warning("No candidates message available.")

        goal_point = self.getClosestSeedlingInBaselink()

        if self.ego_pos is None:
            self.get_logger().warning("Ego position unknown. Stopping.")
            self.publishStatus("Paused")
            self.twist_pub.publish(self.getSmoothed(Twist()))
            return

        elif self.ego_yaw is None:
            self.get_logger().warning("Could not plan trajectory. Ego yaw unavailable.")
            self.publishStatus("Paused")
            self.twist_pub.publish(self.getSmoothed(Twist()))
            return

        if self.current_mode == Mode.STOPPED:
            self.publishStatus("Paused")
            self.twist_pub.publish(self.getSmoothed(Twist()))
            return

        if self.is_planting:
            self.publishStatus("Planting a seedling")
            self.twist_pub.publish(self.getSmoothed(Twist()))
            return

        if goal_point is None:
            self.get_logger().error("Seedling waypoint unknown. Stopping.")
            self.twist_pub.publish(self.getSmoothed(Twist()))
            return

        # Check yaw error. If |yaw err| > pi/4 (45 deg), point turn.
        # goal_point = self.closest_point_bl

        distance_remaining = np.linalg.norm(goal_point)

        DISTANCE_THRESHOLD = 0.5  # meters

        if distance_remaining < DISTANCE_THRESHOLD:
            self.get_logger().info("Reached seedling")
            self.switchToPlantingMode()

        yaw_error = self.getYawError(goal_point)

        print(f"Yaw err: {yaw_error:.1f}, dist {distance_remaining:.1f}")

        POINT_TURN_YAW_ERROR_THRESHOLD = np.pi / 8  # 22.5 degrees
        if abs(yaw_error) > POINT_TURN_YAW_ERROR_THRESHOLD:

            direction_string = "left" if yaw_error > 0 else "right"
            self.publishStatus(f"Turning {direction_string} toward seedling")

            self.pointTurnFromYawError(yaw_error, omega=1.2, linear=0.4)
            return

        Kp_linear = 1.0
        Kp_angular = 1.0
        target_speed = distance_remaining * Kp_linear
        target_angular = yaw_error * Kp_angular
        SPEED_LIMIT = 1.2  # m/s
        target_speed = min(target_speed, SPEED_LIMIT)

        # cmd_msg = Twist()
        # self.twist_pub.publish(cmd_msg)
        # return

        # print(self.candidates)
        # exit()
        cmd_msg = Twist()
        cmd_msg.linear.x = target_speed
        cmd_msg.angular.z = target_angular
        self.twist_pub.publish(self.getSmoothed(cmd_msg))
        self.publishStatus(
            f"Driving {target_speed:.2} m/s, {distance_remaining:.2}m away"
        )

        return

    def setUpParameters(self):
        param_desc = ParameterDescriptor()
        param_desc.type = ParameterType.PARAMETER_DOUBLE_ARRAY
        self.declare_parameter(
            "map_origin_lat_lon_alt_degrees",
            [40.443166012335624, -79.9402856954985594, 288.0961589],
        )


def main(args=None):
    rclpy.init(args=args)

    node = PlannerNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
