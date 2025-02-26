#!/usr/bin/env python3
"""
Example of moving to a pose goal.
- ros2 run pymoveit2 ex_pose_goal.py --ros-args -p position:="[0.25, 0.0, 1.0]" -p quat_xyzw:="[0.0, 0.0, 0.0, 1.0]" -p cartesian:=False
- ros2 run pymoveit2 ex_pose_goal.py --ros-args -p position:="[0.25, 0.0, 1.0]" -p quat_xyzw:="[0.0, 0.0, 0.0, 1.0]" -p cartesian:=False -p synchronous:=False -p cancel_after_secs:=1.0
- ros2 run pymoveit2 ex_pose_goal.py --ros-args -p position:="[0.25, 0.0, 1.0]" -p quat_xyzw:="[0.0, 0.0, 0.0, 1.0]" -p cartesian:=False -p synchronous:=False -p cancel_after_secs:=0.0
"""

from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from pymoveit2 import MoveIt2, MoveIt2State
from pymoveit2.robots import panda as robot
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


class PickAndPlaceNode(Node):
    def __init__(self):
        super().__init__("pick_and_place_node")

        self.create_subscription(JointState, "/joint_states", self.jointStateCb, 10)
        # self.create_subscription(Point, "/ecosim/clicked_point", self.clickedPointCb, 1)
        self.close_gripper_pub = self.create_publisher(Bool, "/close_gripper", 1)

        self.joint_command_pub = self.create_publisher(
            JointState, "/joint_commands", 10
        )

        # # a, alpha, d
        # self.dh_params = [
        #     [0, -np.pi / 2, 0.267],
        #     [0.28948866, 0, 0],
        #     [0.0775, -np.pi / 2, 0],
        #     [0, np.pi / 2, 0.3425],
        #     [0.076, -np.pi / 2, 0],
        #     [0, 0, 0.097],
        # ]

        # self.target_pose = [0.2, 0.2, 0.2, 0.0, 1.5, 0.0]  # xyzrpy

        # joint_angles = self.getInverseKinematics(
        #     self.dh_params, self.target_pose
        # ).tolist()
        # print(joint_angles)

        self.chain = Chain.from_urdf_file(
            "description/xarm6/xarm6.urdf", base_elements=["link_base"]
        )

        self.current_joints = np.zeros(8)

        self.doPickAndPlace()

    def doPickAndPlace(self):

        vertical_offset = 0.1  # meters
        EUCLIDEAN_TOLERANCE = 0.03  # Allowed distance error in meters
        # target_pos = [msg.x, msg.y, msg.z + vertical_offset]
        # target_pos = [msg.x, msg.y, vertical_offset]

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
        # print(msg)
        pass

    def getHomogTf(self, a, alpha, d, theta):
        T = np.array(
            [
                [
                    np.cos(theta),
                    -np.sin(theta) * np.cos(alpha),
                    np.sin(theta) * np.sin(alpha),
                    a * np.cos(theta),
                ],
                [
                    np.sin(theta),
                    np.cos(theta) * np.cos(alpha),
                    -np.cos(theta) * np.sin(alpha),
                    a * np.sin(theta),
                ],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )

        return T

    # Translated from https://github.com/vla-gove/InverseKinematics/blob/main/InverseKinematics/InverseKinematics/InverseKinematics.cpp
    def getInverseKinematics(self, dh_params: list[float], target_pose: list[float]):
        x, y, z, roll, pitch, yaw = target_pose

        # Rotation matrix of end effector
        R: np.ndarray = Rotation.from_euler(
            "xyz", [roll, pitch, yaw], degrees=False
        ).as_matrix()

        joint_angles = np.zeros(6)

        p_e = R.T @ [x, y, z]  # position of end effector in base frame
        p_e *= -1

        print(p_e)

        # joint angles using closed-form solution method

        # joint_angles(0) = atan2(p_e(1), p_e(0)) - atan2(dh_params[0].d, sqrt(p_e(0)*p_e(0) + p_e(1)*p_e(1) - dh_params[0].d*dh_params[0].d));
        # joint_angles(1) = atan2(sqrt(p_e(0)*p_e(0) + p_e(1)*p_e(1) - dh_params[0].d*dh_params[0].d), -dh_params[0].d) + atan2(p_e(2) - dh_params[0].a, sqrt(p_e(0)*p_e(0) + p_e(1)*p_e(1) - dh_params[0].d*dh_params[0].d));
        # joint_angles(2) = atan2(dh_params[2].d, dh_params[1].a) - atan2(sqrt(p_e(0)*p_e(0) + p_e(1)*p_e(1) - dh_params[0].d*dh_params[0].d), p_e(2) - dh_params[0].a);

        joint_angles[0] = np.arctan2(p_e[1], p_e[0]) - np.arctan2(
            dh_params[0][2], np.sqrt(p_e[0] ** 2 + p_e[1] ** 2 - dh_params[0][2] ** 2)
        )
        joint_angles[1] = np.arctan2(
            np.sqrt(p_e[0] ** 2 + p_e[1] ** 2 - dh_params[0][2] ** 2), -dh_params[0][2]
        ) + np.arctan2(
            p_e[2] - dh_params[0][0],
            np.sqrt(p_e[0] ** 2 + p_e[1] ** 2 - dh_params[0][2] ** 2),
        )
        joint_angles[2] = np.arctan2(dh_params[2][2], dh_params[1][0]) - np.arctan2(
            np.sqrt(p_e[0] ** 2 + p_e[1] ** 2 - dh_params[0][2] ** 2),
            p_e[2] - dh_params[0][0],
        )

        # rotation matrix of the end effector with respect to the base frame
        R_b_e = R.copy()  # TODO: Check this
        # R_b_e.col(0) = R.col(0);
        # R_b_e.col(1) = R.col(1);
        # R_b_e.col(2) = R.col(2);

        # joint angles for the remaining joints

        for i in range(3, 6):
            R_b_j = self.getHomogTf(
                dh_params[i - 1][0],
                dh_params[i - 1][2],
                dh_params[i - 1][1],
                joint_angles[i - 1],
            )[:3, :3]
            R_j_e = R_b_j.T * R_b_e
            joint_angles[i] = np.arctan2(
                np.sqrt(R_j_e[0, 2] ** 2 + R_j_e[2, 2] ** 2), R_j_e[1, 2]
            )

            if R_j_e[2, 2] < 0:
                joint_angles[i] = -joint_angles[i]
            if R_j_e[0, 2] < 0:
                joint_angles[i] = -joint_angles[i]
            joint_angles[i] += joint_angles[i - 1]

        # for (int i = 3; i < 6; i++)
        # {
        #     Matrix3d R_b_j = HomogeneousTransformation(dh_params[i - 1].a, dh_params[i - 1].d, dh_params[i - 1].alpha, joint_angles(i - 1)).block<3, 3>(0, 0);
        #     Matrix3d R_j_e = R_b_j.transpose() * R_b_e;
        #     joint_angles(i) = atan2(sqrt(R_j_e(0, 2)*R_j_e(0, 2) + R_j_e(2, 2)*R_j_e(2, 2)), R_j_e(1, 2));
        #     if (R_j_e(2, 2) < 0) joint_angles(i) = -joint_angles(i);
        #     if (R_j_e(0, 2) < 0) joint_angles(i) = -joint_angles(i);
        #     joint_angles(i) += joint_angles(i - 1);
        # }

        return joint_angles


def main(args=None):

    rclpy.init(args=args)

    minimal_publisher = PickAndPlaceNode()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
