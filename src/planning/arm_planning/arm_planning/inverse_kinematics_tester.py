#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import String
import xml.etree.ElementTree as ET
import time
import random
import math
from threading import Lock
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R
import pytorch_kinematics as pk
from ikpy.chain import Chain
from ikpy.link import URDFLink
from ikpy.utils import plot
import torch

from geometry_msgs.msg import Point


class InverseKinematicsTester(Node):
    def __init__(self):
        super().__init__("ik_tester")

        # QoS profile for getting the latest robot_description (transient local)
        robot_description_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.robot_description_sub = self.create_subscription(
            String,
            "/robot_description",
            self.robot_description_cb,
            qos_profile=robot_description_qos,
        )

        # self.create_timer(1.0, self.send_random_joint_angle)
        # self.create_timer(0.1, self.draw_spiral)

        self.joint_command_pub = self.create_publisher(
            JointState,
            "/joint_commands",
            10,
        )

        self.clicked_point_sub = self.create_subscription(
            Point, "/ecosim/clicked_point", self.clicked_point_cb, 10
        )

        self.initial_ik = [0.0] * 10

    def clicked_point_cb(self, msg: Point):
        if not hasattr(self, "chain"):
            self.get_logger().warn(
                "Chain not initialized. Waiting for robot_description."
            )
            return

        links: list[URDFLink] = self.chain.links

        q = [0.0] * len(links)
        names = []
        limits = []
        for i, link in enumerate(links):
            if not isinstance(link, URDFLink) or link.joint_type == "fixed":
                continue

            limits.append(link.bounds)
            q[i] = random.uniform(link.bounds[0], link.bounds[1])
            names.append(link.name)

        target_pos = [msg.x, msg.y, msg.z]
        target_rot = [1.0, 0.0, 0.0]

        ik = self.chain.inverse_kinematics(target_pos, target_rot, orientation_mode="X")

        positions = [float(x) for x in ik[1:7]]
        print(positions)
        print(names)

        msg = JointState()
        msg.name = names
        msg.position = positions

        self.joint_command_pub.publish(msg)

    def robot_description_cb(self, msg: String):
        with open("/tmp/johnny.urdf", "w") as f:
            f.write(msg.data)

        self.chain = Chain.from_urdf_file(
            "/tmp/johnny.urdf",
            base_elements=["link_base"],
            active_links_mask=[
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
            ],
        )

    def send_random_joint_angle(self):
        if not hasattr(self, "chain"):
            self.get_logger().warn(
                "Chain not initialized. Waiting for robot_description."
            )
            return

        links: list[URDFLink] = self.chain.links

        q = [0.0] * len(links)
        names = []
        for i, link in enumerate(links):
            if not isinstance(link, URDFLink) or link.joint_type == "fixed":
                continue

            q[i] = random.uniform(link.bounds[0], link.bounds[1])
            names.append(link.name)

        T = self.chain.forward_kinematics(q)

        target_pos = T[:3, 3]
        target_rot = T[:3, :3]
        ik = self.chain.inverse_kinematics(
            target_pos, target_rot, orientation_mode="all"
        )

        print(T)

        positions = [float(x) for x in ik[1:7]]
        print(positions)
        print(names)

        msg = JointState()
        msg.name = names
        msg.position = positions

        self.joint_command_pub.publish(msg)

    def draw_spiral(self, radius=0.5, num_points=100):
        if not hasattr(self, "chain"):
            self.get_logger().warn(
                "Chain not initialized. Waiting for robot_description."
            )
            return

        links: list[URDFLink] = self.chain.links

        q = [0.0] * len(links)
        names = []
        limits = []
        for i, link in enumerate(links):
            if not isinstance(link, URDFLink) or link.joint_type == "fixed":
                continue

            limits.append(link.bounds)
            q[i] = random.uniform(link.bounds[0], link.bounds[1])
            names.append(link.name)

        T = self.chain.forward_kinematics(q)

        if not hasattr(self, "dtheta"):
            self.dtheta = 2 * math.pi / num_points
            self.theta = 0.0

            self.dz = 0.01
            self.target_z = 0.2

        x = radius * math.cos(self.theta)
        y = radius * math.sin(self.theta)

        target_pos = [x, y, self.target_z]

        target_rot = R.from_euler(
            "xyz", [-np.pi / 2, -np.pi / 2, self.theta - np.pi / 2], degrees=False
        ).as_matrix()

        T = np.eye(4)
        T[:3, :3] = target_rot
        T[:3, 3] = target_pos

        # IK can break if the robot is at its limits
        if self.at_limits(self.initial_ik[1:7], limits):
            print("At limits, setting initial_ik to zero")
            ik = self.chain.inverse_kinematics_frame(
                T,
                initial_position=[0.0] * 10,
                orientation_mode="all",
            )

        else:
            ik = self.chain.inverse_kinematics_frame(
                T,
                initial_position=self.initial_ik,
                orientation_mode="all",
            )

        self.initial_ik = ik

        print(T)

        positions = [float(x) for x in ik[1:7]]
        print(positions)
        print(limits)

        msg = JointState()
        msg.name = names
        msg.position = positions

        self.joint_command_pub.publish(msg)

        self.theta += self.dtheta
        if self.theta >= 2 * math.pi:
            self.theta = 0.0

        self.target_z += self.dz
        if self.target_z >= 0.8:
            self.dz = -self.dz

        elif self.target_z <= 0.2:
            self.dz = -self.dz

        print(self.theta)

    def at_limits(self, q, limits, tolerance=0.1):
        for i in range(len(q)):
            if q[i] < limits[i][0] + tolerance or q[i] > limits[i][1] - tolerance:
                return True
        return False


def main(args=None):
    rclpy.init(args=args)

    node = InverseKinematicsTester()

    # Use a multithreaded executor to handle callbacks concurrently

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
