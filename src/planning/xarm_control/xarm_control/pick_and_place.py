#!/usr/bin/env python3
"""
Example of moving to a pose goal.
- ros2 run pymoveit2 ex_pose_goal.py --ros-args -p position:="[0.25, 0.0, 1.0]" -p quat_xyzw:="[0.0, 0.0, 0.0, 1.0]" -p cartesian:=False
- ros2 run pymoveit2 ex_pose_goal.py --ros-args -p position:="[0.25, 0.0, 1.0]" -p quat_xyzw:="[0.0, 0.0, 0.0, 1.0]" -p cartesian:=False -p synchronous:=False -p cancel_after_secs:=1.0
- ros2 run pymoveit2 ex_pose_goal.py --ros-args -p position:="[0.25, 0.0, 1.0]" -p quat_xyzw:="[0.0, 0.0, 0.0, 1.0]" -p cartesian:=False -p synchronous:=False -p cancel_after_secs:=0.0
"""

from threading import Thread

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time as BuiltinTime
from rclpy.time import Time as RclpyTime

from random import random
from sensor_msgs.msg import JointState
import numpy as np
from scipy.spatial.transform import Rotation

from ikpy.chain import Chain
from ikpy.link import Link
from ikpy.utils import plot
from matplotlib import pyplot as plt

from geometry_msgs.msg import Point
from time import sleep
from std_msgs.msg import Bool


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

    print(n_steps)

    for i in range(1, n_steps + 1):
        qs.append(q_start + i * dq_t)
        ts.append(i * dt)

    if slice_joints:
        return JointTrajectory(np.asarray(qs)[:, 1:7], np.asarray(ts))
    else:
        return JointTrajectory(np.asarray(qs), np.asarray(ts))


def getStraightLineTrajectoryInTaskSpace(
    chain: Chain,
    q_start: np.ndarray,
    q_end: np.ndarray,
    duration,
    dt,
    slice_joints=False,
) -> JointTrajectory:
    assert len(q_start) == len(q_end)

    pos_start = chain.forward_kinematics(q_start)[:3, 3]
    pos_end = chain.forward_kinematics(q_end)[:3, 3]

    d_pos = pos_end - pos_start

    n_steps = int(duration / dt)
    d_pos_t = (
        d_pos / n_steps
    )  # The change in end effector position to make at each step

    qs = []
    ts = []

    print(n_steps)

    # fig, (ax) = plt.subplots(1, 1)

    initial_guess = q_start

    for i in range(1, n_steps + 1):
        pos_i = pos_start + d_pos_t * i
        print(pos_i)
        # ax.scatter(pos_i[1], pos_i[2])

        ik = chain.inverse_kinematics(pos_i, initial_position=initial_guess)

        initial_guess = ik

        qs.append(ik)
        ts.append(i * dt)

    # plt.show()

    if slice_joints:
        return JointTrajectory(np.asarray(qs)[:, 1:7], np.asarray(ts))
    else:
        return JointTrajectory(np.asarray(qs), np.asarray(ts))


def plotTrajectory(traj: JointTrajectory):
    fig, (ax) = plt.subplots(1, 1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("q (rad)")

    for joint in traj.Q[:].T:
        # "joint" is the vector of joint angles for a single joint across all timesteps
        print(joint)
        ax.plot(traj.T, joint)

    print(traj.T)

    plt.show()


def getSecs(rostime):
    print(type(rostime))
    print(rostime)
    if isinstance(rostime, RclpyTime):
        return rostime.nanoseconds / 1e9
    elif isinstance(rostime, BuiltinTime):
        return rostime.sec + rostime.nanosec * 1e-9
    else:
        raise NotImplementedError


def getStamp(secs: float) -> BuiltinTime:
    rostime = BuiltinTime()
    rostime.sec = int(secs)
    rostime.nanosec = (int)((secs - rostime.sec) * 1e9)
    return rostime


class PickAndPlaceNode(Node):
    def __init__(self):
        super().__init__("pick_and_place_node")

        self.create_subscription(JointState, "/joint_states", self.jointStateCb, 10)

        self.joint_command_pub = self.create_publisher(
            JointState, "/joint_commands", 10
        )

        self.close_gripper_pub = self.create_publisher(Bool, "/close_gripper", 1)

        self.chain = Chain.from_urdf_file(
            "description/xarm6/xarm6.urdf", base_elements=["link_base"]
        )

        self.current_joints = np.zeros(8)  # Serves as initial guess for IK optimization

        self.doPickAndPlace()
        exit()

    def doTrajectory(self, traj: JointTrajectory):

        t0 = getSecs(self.get_clock().now())
        msgs = traj.toJointStateMsgs(t0=t0)

        t = 0

        for i, msg in enumerate(msgs[:-1]):
            self.joint_command_pub.publish(msg)

            next_msg = msgs[i + 1]
            dt = getSecs(next_msg.header.stamp) - getSecs(msg.header.stamp)
            assert dt > 0

            t += dt

            print(f"t = {t} | q = {msg.position}")

            sleep(dt)  # TODO: Make async

        self.joint_command_pub.publish(msgs[-1])

    def doPickAndPlace(self):
        EUCLIDEAN_TOLERANCE = 0.03  # Allowed distance error in meters
        # target_pos = [msg.x, msg.y, msg.z + vertical_offset]
        # target_pos = [msg.x, msg.y, vertical_offset]

        # Move from home position to just above seedling
        target_pos = [0.0, 0.0, 0.8]

        target_orientation = Rotation.from_euler(
            "xyz", [-np.pi / 2, np.pi / 2, 0.0]
        ).as_matrix()
        ik = self.chain.inverse_kinematics(
            target_pos, initial_position=self.current_joints
        )
        ik = self.chain.inverse_kinematics(
            target_pos,
            target_orientation,
            initial_position=ik,
            orientation_mode="all",
        )

        traj = getStraightLineTrajectoryInJointSpace(
            self.current_joints, ik, 3.0, 0.1, slice_joints=True
        )
        traj = getStraightLineTrajectoryInTaskSpace(
            self.chain, self.current_joints, ik, 3.0, 0.1, slice_joints=True
        )
        plotTrajectory(traj)

        self.doTrajectory(traj)

        exit()

        for z in np.arange(0.0, 0.45, 0.05):
            target_pos = [0.0, 0.63, 0.5 - z]

            # Assume we want EEF pointed straight down
            target_orientation = Rotation.from_euler(
                "xyz", [-np.pi / 2, np.pi / 2, 0.0]
            ).as_matrix()

            ik = self.chain.inverse_kinematics(
                target_pos, initial_position=self.current_joints
            )
            ik = self.chain.inverse_kinematics(
                target_pos,
                target_orientation,
                initial_position=ik,
                orientation_mode="all",
            )

            self.current_joints = ik

            print(type(self.current_joints))
            print(f"RESULT: {ik}")

            # print(f"Target: {[msg.x, msg.y, msg.z]}")
            reached_pos = self.chain.forward_kinematics(ik)[:3, 3]
            # print(f"Result: {reached_pos}")

            euclidean_error = np.linalg.norm(target_pos - reached_pos)
            # print(f"Error: {euclidean_error}")

            if euclidean_error > EUCLIDEAN_TOLERANCE:
                self.get_logger().warning(f"Euclidean error {euclidean_error} > 1 cm")
            else:
                self.joint_command_pub.publish(self.toJointCommandMsg(ik[1:7]))

            sleep(0.1)

        print("DONE")

        self.close_gripper_pub.publish(Bool(data=True))

        self.pointToSky()
        sleep(1.0)

        for z in np.arange(0.0, 0.45, 0.05):
            target_pos = [0.0, -0.63, 0.5 - z]

            # Assume we want EEF pointed straight down
            target_orientation = Rotation.from_euler(
                "xyz", [np.pi / 2, np.pi / 2, 0.0]
            ).as_matrix()

            ik = self.chain.inverse_kinematics(
                target_pos, initial_position=self.current_joints
            )

            if z > 0.25:
                ik = self.chain.inverse_kinematics(
                    target_pos,
                    target_orientation,
                    initial_position=ik,
                    orientation_mode="all",
                )

            self.current_joints = ik

            print(type(self.current_joints))
            print(f"RESULT: {ik}")

            # print(f"Target: {[msg.x, msg.y, msg.z]}")
            reached_pos = self.chain.forward_kinematics(ik)[:3, 3]
            # print(f"Result: {reached_pos}")

            euclidean_error = np.linalg.norm(target_pos - reached_pos)
            # print(f"Error: {euclidean_error}")

            self.joint_command_pub.publish(self.toJointCommandMsg(ik[1:7]))

            sleep(0.1)

        self.close_gripper_pub.publish(Bool(data=False))

    def pointToSky(self):
        target_pos = [0.0, 0.0, 1.0]
        ik = self.chain.inverse_kinematics(target_pos)
        self.joint_command_pub.publish(self.toJointCommandMsg(ik[1:7]))

    def moveToHome(self):
        self.joint_command_pub.publish(self.toJointCommandMsg(np.zeros(6)))

    def toJointCommandMsg(self, q: list[float]):

        if isinstance(q, np.ndarray):
            q = q.tolist()

        msg = JointState()
        msg.name = ["link1", "link2", "link3", "link4", "link5", "link6"]
        print(q)
        msg.position = q

        assert len(msg.position) == len(msg.name)

        return msg

    def jointStateCb(self, msg: JointState):
        """Called when new JointState received.

        Args:
            msg (JointState): Current joint positions
        """
        pass


def main(args=None):

    rclpy.init(args=args)

    node = PickAndPlaceNode()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
