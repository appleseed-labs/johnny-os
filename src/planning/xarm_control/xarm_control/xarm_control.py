#!/usr/bin/env python3
"""
Example of moving to a pose goal.
- ros2 run pymoveit2 ex_pose_goal.py --ros-args -p position:="[0.25, 0.0, 1.0]" -p quat_xyzw:="[0.0, 0.0, 0.0, 1.0]" -p cartesian:=False
- ros2 run pymoveit2 ex_pose_goal.py --ros-args -p position:="[0.25, 0.0, 1.0]" -p quat_xyzw:="[0.0, 0.0, 0.0, 1.0]" -p cartesian:=False -p synchronous:=False -p cancel_after_secs:=1.0
- ros2 run pymoveit2 ex_pose_goal.py --ros-args -p position:="[0.25, 0.0, 1.0]" -p quat_xyzw:="[0.0, 0.0, 0.0, 1.0]" -p cartesian:=False -p synchronous:=False -p cancel_after_secs:=0.0
"""

from ast import Not
import math
from os import error
from threading import Thread
import time


import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time as BuiltinTime
from rclpy.time import Time as RclpyTime

from rcl_interfaces.msg import ParameterDescriptor
from rclpy.parameter import Parameter

from matplotlib import pyplot as plt

from random import random
from sensor_msgs.msg import JointState
import numpy as np

# from scipy.spatial.transform import Rotation

# from matplotlib import pyplot as plt

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from std_msgs.msg import Bool, Empty
from time import sleep

from xarm.wrapper import XArmAPI
import serial
from cv_bridge import CvBridge


# from ikpy.chain import Chain
# from ikpy.link import Link
# from ikpy.utils import plot

from enum import Enum

pos_dictionary = {
    "home": [
        -0.32572651,
        -80.93894022,
        -14.3142046,
        -188.32544039,
        -7.37580029,
        -170.15998538,
        0.0,
    ],
    "ready": [
        130.09470834,
        33.48371084,
        -139.9335396,
        -68.80289198,
        129.62064306,
        -175.63041452,
        0.0,
    ],
    "seedling_1_pre": [
        117.616547,
        -12.89321197,
        -55.8271168,
        -53.73828456,
        99.10324932,
        -147.33598243,
        0.0,
    ],
    "seedling_1_grab": [115.6613, -10.2879, -56.2885, -50.8607, 97.578, -147.3331, 0.0],
    "seedling_1_lift": [
        138.18154925,
        11.32691724,
        -77.62907763,
        -43.34248103,
        115.49729071,
        -129.13036308,
        0.0,
    ],
    "over_hole": [-41.4897, 12.6384, -30.3916, -201.077, 73.6514, -170.3068, 0.0],
    "camera_over_hole": [-58.3486, 11.259, -27.4902, -210.6698, 73.6543, -170.307, 0.0],
    "to_hole_1": [83.0299, -6.7996, -91.7994, -64.717, 77.3725, -187.4991, 0.0],
    "to_hole_2": [5.979, -73.9998, -18.5266, -96.1193, 9.008, -196.6823, 0.0],
    "into_hole": [-30.8998, 32.878, -57.2936, -194.921, 72.2279, -170.3049, 0.0],
    "sweep_1_start": [
        -47.4052,
        40.4668,
        -111.8521,
        -149.4101,
        -51.2849,
        -158.9844,
        0.0,
    ],
    "sweep_1_end": [-46.9914, 67.4331, -120.2814, -153.5416, -47.5788, -158.9937, 0.0],
    "sweep_2_start": [
        -21.4479,
        30.1746,
        -52.0257,
        -153.4955,
        47.0232,
        -158.9836,
        0.0,
    ],
    "sweep_2_end": [
        -32.4677,
        48.2822,
        -67.9598,
        -148.6514,
        51.1192,
        -170.1973,
        0.0,
    ],
    "sweep_3_start": [
        -49.7081,
        102.6205,
        -123.6335,
        -223.697,
        111.2129,
        -247.0071,
        0.0,
    ],
    "sweep_3_end": [-49.4795, 80.2942, -97.1986, -219.7065, 127.7964, -246.9978, 0.0],
}


class JointTrajectory:
    """Stores a sequence of joint positions, with times associated with each point in the sequence"""

    Q: np.ndarray  # List of joint positions
    T: np.ndarray  # Times for each joint position group (sec), starting from zero
    names: list[str]

    def __init__(
        self,
        Q: np.ndarray,
        T: np.ndarray,
        names: list[str] = ["link1", "link2", "link3", "link4", "link5", "link6"],
    ):
        assert len(Q) == len(
            T
        ), f"Tried to create JointTrajectory with {len(T)} times and {len(Q)} joint frames"

        self.Q = Q
        self.T = T
        self.names = names

    def toJointStateMsgs(self, t0=0.0) -> list[JointState]:
        assert len(self.Q) == len(self.T)
        assert isinstance(self.Q, np.ndarray)
        assert isinstance(self.T, np.ndarray)

        joint_state_msgs = []

        for i, frame in enumerate(self.Q[:]):
            msg = JointState()
            msg.position = frame.tolist()
            msg.name = self.names
            msg.header.frame_id = "link_base"
            msg.header.stamp = getStamp(self.T[i] + t0)

            joint_state_msgs.append(msg)

        return joint_state_msgs


def getStraightLineTrajectoryInJointSpace(
    q_start: np.ndarray, q_end: np.ndarray, duration, dt, slice_joints=False
) -> JointTrajectory:
    assert len(q_start) == len(q_end)
    dq = q_end - q_start

    n_steps = int(duration / dt)
    dq_t = dq / n_steps  # The change in ejoint positions to make at each step

    qs = []
    ts = []

    for i in range(1, n_steps + 1):
        qs.append(q_start + i * dq_t)
        ts.append(i * dt)

    if slice_joints:
        return JointTrajectory(np.asarray(qs)[:, 1:7], np.asarray(ts))
    else:
        return JointTrajectory(np.asarray(qs), np.asarray(ts))


class XarmControlNode(Node):
    def __init__(self):
        super().__init__("xarm_control_node")

        self.declare_parameter(
            "xarm_ip",
            "192.168.1.196",
            ParameterDescriptor(description="IP address of the xArm"),
        )
        ip = self.get_parameter("xarm_ip").get_parameter_value().string_value

        self.declare_parameter(
            "sim_only",
            False,
            ParameterDescriptor(
                description="Skip xArm connection, only send EcoSim signals"
            ),
        )
        self.sim_only = self.get_parameter("sim_only").get_parameter_value().bool_value

        self.declare_parameter(
            "planting_time_secs",
            10.0,
            ParameterDescriptor(
                description="Total time it takes for the arm to plant a seedling, in seconds."
            ),
        )
        self.planting_time_secs = (
            self.get_parameter("planting_time_secs").get_parameter_value().double_value
        )

        self.is_correcting_with_camera = True
        self.correction_start_time = time.time()

        if self.sim_only:
            self.get_logger().info("Sim only mode enabled. Skipping xArm setup.")
            return

        self.FULL_SPEED = 20
        self.CAREFUL_SPEED = self.FULL_SPEED / 2
        self.CORRECTION_DURATION = 10.0  # s

        self.arm = XArmAPI(ip)

        # This adds a gripper, modelled as a cylinder with
        # radius of 90mm and height of 100mm
        self.arm.set_collision_tool_model(21, radius=90.0, height=100.0)
        self.arm.set_self_collision_detection(True)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        self.set_joints(
            q=pos_dictionary["camera_over_hole"], speed=self.CAREFUL_SPEED, wait=True
        )

        self.max_angular_speed = 80

        self.q = self.arm.angles

        self.rolling_back_started = False
        self.total_distance_so_far = 0.0
        self.target_roll_distance = -0.15  # meters

        # SUBSCRIPTIONS
        self.create_subscription(
            Empty, "/behavior/start_planting", self.start_planting_cb, 1
        )
        # From the Amiga control node
        # self.create_subscription(Odometry, "/odom", self.odometry_cb, 1)

        # For hole distance correction
        self.create_subscription(
            Image, "/camera/camera/depth/image_rect_raw", self.depth_image_cb, 1
        )

        # PUBLISHERS
        self.on_plant_complete_pub = self.create_publisher(
            Empty, "/behavior/on_plant_complete", 1
        )
        self.twist_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 1)

        self.bridge = CvBridge()

        # self.pick_and_place_seedling()
        # self.sweep_soil()

        self.start_x = None

        self.x_error_history = np.zeros(10)
        self.n_samples_collected = 0

    def depth_image_cb(self, msg):

        if not self.is_correcting_with_camera:
            return

        if time.time() - self.correction_start_time > self.CORRECTION_DURATION:
            self.get_logger().info("Stopping camera correction")
            self.is_correcting_with_camera = False
            self.plant_seedling()
            return
        else:
            self.get_logger().info(
                f"Camera correction in progress, remaining time: {self.CORRECTION_DURATION - (time.time() - self.correction_start_time):.2f} seconds"
            )

        try:
            # Convert ROS2 Image message to OpenCV image (numpy array)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Rotate 90 deg clockwise
            cv_image = np.rot90(cv_image, k=3)

        except Exception as e:
            self.get_logger().error(f"Error converting image: {str(e)}")
            return

        min_depth = np.min(cv_image)
        max_depth = np.max(cv_image)
        top_90 = min_depth + (max_depth - min_depth) * 0.9

        plt.figure()
        plt.imshow(cv_image)
        plt.savefig("depth_image.png")
        plt.close()

        # Get the indices of the pixels that are more than the top 90% depth
        mask = cv_image > top_90
        indices = np.argwhere(mask)

        # Get the average index of the pixels that are more than the top 90% depth
        if len(indices) > 0:
            avg_index = np.median(indices, axis=0)
            print(f"Average index of pixels above top 90% depth: {avg_index}")
        else:
            print("No pixels above top 90% depth found.")

        x_error = int(avg_index[0] - (cv_image.shape[0] / 2))
        y_error = int(avg_index[1] - (cv_image.shape[1] / 2))

        # # Normalize the x_error and y_error to be between -1 and 1
        x_error = x_error / (cv_image.shape[0] / 2)
        y_error = y_error / (cv_image.shape[1] / 2)

        # Invert x error so that "forward" is positive
        x_error = -x_error

        # Make a weighted moving average of the x_error
        self.x_error_history = np.roll(self.x_error_history, -1)
        self.x_error_history[-1] = x_error
        x_error = np.mean(self.x_error_history)

        print(f"x_error: {x_error:.2f}")

        # if abs(x_error) < 5.0 and self.n_samples_collected >= 10:
        #     self.get_logger().info("Arrived over hole. Starting placement")
        #     self.is_planting = False
        #     # self.pick_up_seedling()
        #     self.plant_seedling()
        #     return

        self.n_samples_collected += 1

        # Move forward according to the x error
        twist = Twist()
        twist.linear.x = x_error * 0.35  # m/s

        if twist.linear.x > 0.10:
            twist.linear.x = 0.1
        elif twist.linear.x < -0.1:
            twist.linear.x = -0.1

        print(f"twist.linear.x: {twist.linear.x:.2f}")

        self.twist_pub.publish(twist)

        # plt.imshow(cv_image)
        # plt.savefig("depth_image.png")

    # def odometry_cb(self, msg):

    #     if not self.rolling_back_started:
    #         self.start_x = msg.pose.pose.position.x
    #         return

    #     target_x = self.start_x + self.target_roll_distance

    #     current_x = msg.pose.pose.position.x

    #     if current_x <= target_x:
    #         print(f"REACHED TARGET, error = {target_x - current_x}")
    #         self.rolling_back_started = False
    #         self.pick_and_place_seedling()
    #         return

    #     twist = Twist()
    #     twist.linear.x = -0.15  # m/s

    #     self.twist_pub.publish(twist)

    #     # if self.total_distance_so_far < self.target_roll_distance:
    #     #     print("DONE TURNING")
    #     #     self.rolling_back_started = False
    #     #     self.total_distance_so_far = 0.0
    #     #     self.pick_and_place_seedling()

    def start_planting_cb(self, msg):
        self.get_logger().info("Received start planting signal")

        self.pick_up_seedling()

        # Enable this to start the camera correction
        self.get_logger().info("Starting camera correction")
        self.is_correcting_with_camera = True
        self.correction_start_time = time.time()

    def pick_up_seedling(self):

        self.open_gripper()

        self.set_joints(q=pos_dictionary["home"], speed=self.FULL_SPEED, wait=True)
        self.set_joints(q=pos_dictionary["ready"], speed=self.FULL_SPEED, wait=True)
        self.set_joints(
            q=pos_dictionary["seedling_1_pre"], speed=self.FULL_SPEED, wait=True
        )
        self.set_joints(
            q=pos_dictionary["seedling_1_grab"], speed=self.FULL_SPEED, wait=True
        )
        self.close_gripper()
        self.set_joints(
            q=pos_dictionary["seedling_1_lift"], speed=self.CAREFUL_SPEED, wait=True
        )
        self.set_joints(q=pos_dictionary["ready"], speed=self.FULL_SPEED, wait=True)
        self.set_joints(q=pos_dictionary["to_hole_1"], speed=self.FULL_SPEED, wait=True)
        self.set_joints(q=pos_dictionary["to_hole_2"], speed=self.FULL_SPEED, wait=True)
        self.set_joints(
            q=pos_dictionary["camera_over_hole"], speed=self.CAREFUL_SPEED, wait=True
        )

    def plant_seedling(self):

        self.set_joints(
            q=pos_dictionary["over_hole"], speed=self.CAREFUL_SPEED, wait=True
        )
        self.set_joints(
            q=pos_dictionary["into_hole"], speed=self.CAREFUL_SPEED, wait=True
        )

    def sweep_soil(self):
        self.set_joints(
            q=pos_dictionary["sweep_1_start"], speed=self.CAREFUL_SPEED, wait=True
        )
        self.set_joints(
            q=pos_dictionary["sweep_1_end"], speed=self.CAREFUL_SPEED, wait=True
        )
        self.set_joints(
            q=pos_dictionary["over_hole"], speed=self.CAREFUL_SPEED, wait=True
        )
        self.set_joints(
            q=pos_dictionary["sweep_2_start"], speed=self.CAREFUL_SPEED, wait=True
        )
        self.set_joints(
            q=pos_dictionary["sweep_2_end"], speed=self.CAREFUL_SPEED, wait=True
        )
        self.set_joints(
            q=pos_dictionary["over_hole"], speed=self.CAREFUL_SPEED, wait=True
        )
        self.set_joints(
            q=pos_dictionary["sweep_3_start"], speed=self.CAREFUL_SPEED, wait=True
        )
        self.set_joints(
            q=pos_dictionary["sweep_3_end"], speed=self.CAREFUL_SPEED, wait=True
        )
        self.set_joints(
            q=pos_dictionary["over_hole"], speed=self.CAREFUL_SPEED, wait=True
        )

    def set_joints(self, q: list[float], speed=25, wait=True) -> int:
        return self.arm.set_servo_angle(angle=q, speed=speed, wait=wait)

    def open_gripper(self):
        with serial.Serial("/dev/ttyACM0", 57600, timeout=1) as ser:
            ser.write(b"O\r\n")
            time.sleep(0.5)

    def close_gripper(self):
        with serial.Serial("/dev/ttyACM0", 57600, timeout=1) as ser:
            ser.write(b"C\r\n")
            time.sleep(1.0)

    def publish_joint_state(self):

        names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]

        angles = self.arm.angles[:6]

        msg = JointState()
        msg.position = angles
        msg.name = names
        msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_state_pub.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = XarmControlNode()

    if not node.sim_only:
        arm = node.arm

    try:
        rclpy.spin(node)

    finally:
        node.destroy_node()

        # Clean up the xArm connection
        node.get_logger().warning("Disconnecting from xArm...")
        if not node.sim_only:
            arm.set_state(state=4)
            # arm.motion_enable(enable=False)
            arm.disconnect()

        # Shutdown rclpy
        rclpy.shutdown()


if __name__ == "__main__":
    main()
