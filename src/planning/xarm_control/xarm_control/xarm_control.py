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

from random import random
from sensor_msgs.msg import JointState
import numpy as np

# from scipy.spatial.transform import Rotation

# from matplotlib import pyplot as plt

from geometry_msgs.msg import Point
from time import sleep
from std_msgs.msg import Bool, Empty

from xarm.wrapper import XArmAPI
import serial

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
    "seedling_1_grab": [
        116.8671755,
        -9.62150837,
        -51.20936345,
        -53.80446119,
        96.2770204,
        -138.36598437,
        0.0,
    ],
    "seedling_1_lift": [
        138.18154925,
        11.32691724,
        -77.62907763,
        -43.34248103,
        115.49729071,
        -129.13036308,
        0.0,
    ],
    "over_hole": [-29.1542, 16.5677, -28.8071, -176.041, 62.1542, -158.9843, 0.0],
    "to_hole_1": [73.9455, -4.1437, -91.7979, -64.7164, 76.854, -187.4988, 0.0],
    "to_hole_2": [7.449, -51.7186, -27.6111, -84.6549, 8.9232, -187.504, 0.0],
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

        self.create_subscription(
            Empty, "/behavior/start_planting", self.start_planting_cb, 1
        )

        self.start_driving_pub = self.create_publisher(
            Empty, "/behavior/start_driving", 1
        )

        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 1)

        self.is_planting = False
        self.planting_stop_time = time.time()

        if self.sim_only:
            self.get_logger().info("Sim only mode enabled. Skipping xArm setup.")
            return

        self.FULL_SPEED = 40
        self.CAREFUL_SPEED = self.FULL_SPEED / 2

        self.arm = XArmAPI(ip)

        # This adds a gripper, modelled as a cylinder with
        # radius of 90mm and height of 100mm
        self.arm.set_collision_tool_model(21, radius=90.0, height=100.0)
        self.arm.set_self_collision_detection(True)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        self.max_angular_speed = 80

        self.q = self.arm.angles

    def start_planting_cb(self, msg):
        self.get_logger().info("Received start planting signal")

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
            q=pos_dictionary["over_hole"], speed=self.CAREFUL_SPEED, wait=True
        )
        time.sleep(5)
        self.open_gripper()

        self.publish_joint_state()
        self.get_logger().info("Finished planting")

    def set_joints(self, q: list[float], speed=25) -> int:
        return self.arm.set_servo_angle(angle=q, speed=speed, wait=True)

    def open_gripper(self):
        with serial.Serial("/dev/ttyACM0", 57600, timeout=1) as ser:
            ser.write(b"O\r\n")
            time.sleep(0.5)

    def close_gripper(self):
        with serial.Serial("/dev/ttyACM0", 57600, timeout=1) as ser:
            ser.write(b"C\r\n")
            time.sleep(0.5)

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
