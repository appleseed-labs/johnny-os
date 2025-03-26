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

# from ikpy.chain import Chain
# from ikpy.link import Link
# from ikpy.utils import plot

from enum import Enum

# Create an enum that maps the error code to a string
ERROR_CODE_MAP = {
    2: "ESTOP_ACTIVE",
    22: "SELF_COLLISION",
}

HOME_Q = [0, 0, 0, 0, 0, 0]
OVERHEAD_Q = [0, -56.1, -34.9, 0, 0, 0]
READY_Q = [95.1, -40.6, -17.9, 34.2, -11.3, -23.5]


# Create an enum for the pose FSM
class PoseState(Enum):
    HOME = 0
    OVERHEAD = 1
    READY = 2


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


# def getStraightLineTrajectoryInTaskSpace(
#     chain: Chain,
#     q_start: np.ndarray,
#     q_end: np.ndarray,
#     duration,
#     dt,
#     slice_joints=False,
# ) -> JointTrajectory:
#     assert len(q_start) == len(q_end)

#     pos_start = chain.forward_kinematics(q_start)[:3, 3]
#     pos_end = chain.forward_kinematics(q_end)[:3, 3]

#     d_pos = pos_end - pos_start

#     n_steps = int(duration / dt)
#     d_pos_t = (
#         d_pos / n_steps
#     )  # The change in end effector position to make at each step

#     qs = []
#     ts = []

#     # fig, (ax) = plt.subplots(1, 1)

#     initial_guess = q_start

#     for i in range(1, n_steps + 1):
#         pos_i = pos_start + d_pos_t * i
#         # ax.scatter(pos_i[1], pos_i[2])

#         ik = chain.inverse_kinematics(pos_i, initial_position=initial_guess)

#         initial_guess = ik

#         qs.append(ik)
#         ts.append(i * dt)

#     # plt.show()

#     if slice_joints:
#         return JointTrajectory(np.asarray(qs)[:, 1:7], np.asarray(ts))
#     else:
#         return JointTrajectory(np.asarray(qs), np.asarray(ts))


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
            True,
            ParameterDescriptor(
                description="Skip xArm connection, only send EcoSim signals"
            ),
        )
        self.sim_only = self.get_parameter("sim_only").get_parameter_value().bool_value

        self.declare_parameter(
            "planting_time_secs",
            3.0,
            ParameterDescriptor(
                description="Total time it takes for the arm to plant a seedling, in seconds."
            ),
        )
        self.planting_time_secs = (
            self.get_parameter("planting_time_secs").get_parameter_value().double_value
        )

        self.create_subscription(
            Empty, "/behavior/start_planting", self.startPlantingCb, 1
        )

        self.start_driving_pub = self.create_publisher(
            Empty, "/behavior/start_driving", 1
        )

        # self.chain = Chain.from_urdf_file(
        #     "description/xarm6/xarm6.urdf",
        #     base_elements=["link_base"],
        #     active_links_mask=[
        #         False,
        #         True,
        #         True,
        #         True,
        #         True,
        #         True,
        #         True,
        #         False,
        #     ],
        # )

        self.is_planting = False
        self.planting_stop_time = time.time()
        self.create_timer(0.1, self.checkIfPlantingComplete)

        if self.sim_only:
            self.get_logger().info("Sim only mode enabled. Skipping xArm setup.")
            return

        self.arm = XArmAPI(ip)

        # This adds a gripper, modelled as a cylinder with
        # radius of 90mm and height of 100mm
        self.arm.set_collision_tool_model(21, radius=90.0, height=100.0)
        self.arm.set_self_collision_detection(True)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        speed = 35

        code, current_joint_angles = self.arm.get_servo_angle()

        current_joint_angles = current_joint_angles[
            :6
        ]  # Ignore the last joint (gripper)

        # self.arm.move_gohome(wait=True, speed=speed)
        print(current_joint_angles)
        print(self.arm.angles)
        # self.goOverhead()

        if np.allclose(current_joint_angles, HOME_Q, atol=0.1):
            self.get_logger().info("Already at home")
            self.fsm_state = PoseState.HOME

        elif np.allclose(current_joint_angles, OVERHEAD_Q, atol=0.1):
            self.get_logger().info("Starting at overhead")
            self.fsm_state = PoseState.OVERHEAD
            self.goHome(speed=speed)

        elif np.allclose(current_joint_angles, READY_Q, atol=0.1):
            self.get_logger().info("Starting at ready")
            self.fsm_state = PoseState.READY
            self.goHome(speed=speed)

        self.goHome(speed=speed)
        self.goOverhead(speed=speed)
        self.goReady()

    def checkIfPlantingComplete(self):
        if self.is_planting and time.time() > self.planting_stop_time:
            self.get_logger().info("Planting complete")
            self.is_planting = False
            self.start_driving_pub.publish(Empty())

        elif self.is_planting:
            remaining_time = self.planting_stop_time - time.time()
            self.get_logger().info(
                f"Planting in progress. {remaining_time:.2f} seconds remaining"
            )

    def startPlantingCb(self, msg):
        self.get_logger().info("Received start planting signal")
        self.is_planting = True
        self.planting_stop_time = time.time() + self.planting_time_secs
        self.goHome()

    def setJointPosition(self, q: list[float], speed=25) -> int:
        self.arm.set_servo_angle(angle=HOME_Q, speed=speed, wait=True)

    def goHome(self, speed=25):
        if self.sim_only:
            return

        if self.fsm_state == PoseState.READY:
            self.get_logger().info("Going to overhead from ready")
            self.goOverhead(speed=speed)

        if self.fsm_state == PoseState.OVERHEAD:
            self.get_logger().info("Going to home from overhead")

            self.arm.set_servo_angle(angle=HOME_Q, speed=speed, wait=True)
            self.fsm_state = PoseState.HOME

        elif self.fsm_state == PoseState.HOME:
            self.get_logger().info("Already at home")
            return

        else:
            assert NotImplementedError(
                "Cannot go home from state {}".format(self.fsm_state)
            )

    def goOverhead(self, speed=25):
        if self.sim_only:
            return

        self.arm.set_servo_angle(angle=OVERHEAD_Q, speed=speed, wait=True)
        self.get_logger().info("Now at overhead position")
        self.fsm_state = PoseState.OVERHEAD

    def goReady(self, speed=25):
        if self.sim_only:
            return
        self.arm.set_servo_angle(angle=READY_Q, speed=speed, wait=True)
        self.get_logger().info("Now at ready position")
        self.fsm_state = PoseState.READY

    def handleCode(self, code: int):
        """
        Handle the API code returned by the xArm.
        """

        if code == 0:
            return  # Everything is ok

        elif code == 9:
            self.get_logger().error(f"Code 9: xArm is not ready to move.")

        code, [error_code, warn_code] = self.arm.get_err_warn_code()

        if error_code in ERROR_CODE_MAP:
            self.get_logger().error(
                f"Could not move to home position: {ERROR_CODE_MAP[error_code]}"
            )
        else:
            self.get_logger().error(f"Unknown error code {error_code} returned by xArm")
            self.arm.get_err_warn_code(show=True)


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
