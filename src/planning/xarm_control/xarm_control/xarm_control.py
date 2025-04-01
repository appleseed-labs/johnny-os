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
from turtle import pos

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
READY_Q = [95.1, -55.4, -10.1, 33.7, -11.3, -118.1]
SEEDLING_1_Q = [93.6, -33.9, -21.5, 33.7, -11.3, -118.1]


# Create an enum for the pose FSM
class PoseState(Enum):
    HOME = 0
    OVERHEAD = 1
    READY = 2
    UNKNOWN = 3
    SEEDLING = 4


from collections import defaultdict


# This class represents a directed graph
# using adjacency list representation
class Graph:

    # Constructor
    def __init__(self):

        # Default dictionary to store graph
        self.graph = defaultdict(list)

    # Function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)  # Add the reverse edge to make the graph undirected

    # Function to print a BFS of graph
    def BFS(self, start, end):
        # Mark all the vertices as not visited
        visited = {key: False for key in self.graph}

        # Create a queue for BFS
        queue = []

        # Create a dictionary to store the path
        parent = {key: None for key in self.graph}

        # Mark the start node as visited and enqueue it
        visited[start] = True
        queue.append(start)

        while queue:
            # Dequeue a vertex from the queue
            current = queue.pop(0)

            # If the end node is reached, construct the path
            if current == end:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]  # Reverse the path to get the correct order

            # Get all adjacent vertices of the dequeued vertex
            # If an adjacent has not been visited, mark it visited and enqueue it
            for neighbor in self.graph[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = current
                    queue.append(neighbor)

        # If the end node is not reachable, return an empty list
        return []


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


class Pose:
    def __init__(self, name: str, Q: list[float]):
        self.name = name
        self.Q = Q


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
            6.0,
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

        self.is_planting = False
        self.planting_stop_time = time.time()
        self.create_timer(0.1, self.checkIfPlantingComplete)

        # Create a graph given in
        # the above diagram
        self.g = Graph()

        self.home_pose = Pose("home", HOME_Q)
        self.overhead_pose = Pose("overhead", OVERHEAD_Q)
        self.ready_pose = Pose("ready", READY_Q)
        self.seedling_pose = Pose("seedling", SEEDLING_1_Q)

        self.g.addEdge(self.home_pose, self.overhead_pose)
        self.g.addEdge(self.overhead_pose, self.ready_pose)
        self.g.addEdge(self.ready_pose, self.seedling_pose)

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

        self.max_angular_speed = 60

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
            self.current_pose = self.home_pose

        elif np.allclose(current_joint_angles, OVERHEAD_Q, atol=0.1):
            self.get_logger().info("Starting at overhead")
            self.fsm_state = PoseState.OVERHEAD
            self.current_pose = self.overhead_pose
            self.goReady(speed=self.max_angular_speed)

        elif np.allclose(current_joint_angles, READY_Q, atol=0.1):
            self.get_logger().info("Starting at ready")
            self.fsm_state = PoseState.READY
            self.current_pose = self.ready_pose
            self.goReady(speed=self.max_angular_speed)

        # TODO: CHECK FOR ALL SEEDLING POSITIONS
        elif np.allclose(current_joint_angles, SEEDLING_1_Q, atol=0.1):
            self.get_logger().info("Starting at seedling")
            self.fsm_state = PoseState.SEEDLING
            self.current_pose = self.seedling_pose
            self.goReady(speed=self.max_angular_speed)

        else:
            self.get_logger().error("Starting at unknown position")
            self.fsm_state = PoseState.UNKNOWN
            self.current_pose = None
            self.goReady(speed=self.max_angular_speed, force=True)

        # self.moveToPose(self.home_pose, speed=self.max_angular_speed)
        # self.moveToPose(self.seedling_pose, speed=self.max_angular_speed)
        # self.moveToPose(self.home_pose, speed=self.max_angular_speed)

    def moveToPose(self, target_pose: Pose, speed=None):
        if self.sim_only:
            return

        if speed is None:
            speed = self.max_angular_speed

        print(f"Moving from {self.current_pose.name} to {target_pose.name}")
        # Print the dict of the graph, formatted from a defaultdict
        print("Graph dict:")
        for key, value in self.g.graph.items():
            print(f"{key.name}: {[v.name for v in value]}")
        poses = self.g.BFS(self.current_pose, target_pose)
        print([pose.name for pose in poses])

        if len(poses) == 0:
            self.get_logger().error("No path found")
            return
        for i in range(len(poses) - 1):
            start = poses[i]
            end = poses[i + 1]
            print(f"Moving from {start.name} to {end.name}")
            self.arm.set_servo_angle(angle=end.Q, speed=speed, wait=True)

        self.current_pose = target_pose

    def checkIfPlantingComplete(self):
        if self.is_planting and time.time() > self.planting_stop_time:
            self.get_logger().info("Planting complete.")
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
        self.moveToPose(self.seedling_pose, speed=self.max_angular_speed)
        time.sleep(1.0)
        self.moveToPose(self.overhead_pose, speed=self.max_angular_speed)

    def setJointPosition(self, q: list[float], speed=25) -> int:
        self.arm.set_servo_angle(angle=HOME_Q, speed=speed, wait=True)

    def goHome(self, speed=None, force=False):
        if self.sim_only:
            return

        print("GO HOME")

        if speed is None:
            speed = self.max_angular_speed

        if self.fsm_state == PoseState.SEEDLING:
            self.get_logger().info("Going to ready from seedling")
            self.goReady()

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

        elif force:
            self.get_logger().warning("Going to home from unknown")

            self.arm.set_servo_angle(angle=HOME_Q, speed=speed, wait=True)
            self.fsm_state = PoseState.HOME

        else:
            assert NotImplementedError(
                "Cannot go home from state {}".format(self.fsm_state)
            )

    def goOverhead(self, speed=None):
        if self.sim_only:
            return

        if speed is None:
            speed = self.max_angular_speed

        if self.fsm_state == PoseState.SEEDLING:
            self.goReady(speed=speed)

        elif self.fsm_state == PoseState.READY:
            self.get_logger().info("Going to overhead from ready")
            self.arm.set_servo_angle(angle=OVERHEAD_Q, speed=speed, wait=True)
            self.get_logger().info("Now at overhead position")
            self.fsm_state = PoseState.OVERHEAD
            return

        elif self.fsm_state == PoseState.HOME:
            self.get_logger().info("Going to overhead from home")
            self.arm.set_servo_angle(angle=OVERHEAD_Q, speed=speed, wait=True)
            self.get_logger().info("Now at overhead position")
            self.fsm_state = PoseState.OVERHEAD
            return

        else:
            self.get_logger().error(f"Unsupported state {self.fsm_state}")
            return

    def goReady(self, speed=None):
        if self.sim_only:
            return

        if speed is None:
            speed = self.max_angular_speed

        if self.fsm_state == PoseState.SEEDLING:
            self.get_logger().info("Going to ready from seedling")
            self.arm.set_servo_angle(angle=READY_Q, speed=speed, wait=True)
            self.get_logger().info("Now at ready position")
            self.fsm_state = PoseState.READY
            return

        if self.fsm_state == PoseState.OVERHEAD:
            self.get_logger().info("Going to ready from overhead")
            self.arm.set_servo_angle(angle=READY_Q, speed=speed, wait=True)
            self.get_logger().info("Now at ready position")
            self.fsm_state = PoseState.READY
            return

        if self.fsm_state == PoseState.HOME:
            self.get_logger().info("Going to overhead from home")
            self.goOverhead(speed=speed)
            self.goReady(speed=speed)
            return

        elif self.fsm_state == PoseState.READY:
            self.get_logger().info("Already at ready")
            return

        else:
            self.get_logger().error(f"Unsupported state {self.fsm_state}")
            return

    def goToFirstSeedling(self, speed=None):
        if self.sim_only:
            return

        if self.fsm_state == PoseState.READY:
            self.arm.set_servo_angle(angle=SEEDLING_1_Q, speed=speed, wait=True)

        else:
            self.goReady(speed=speed)
            self.goToFirstSeedling(speed=speed)

        self.get_logger().info("Now at seedling position")
        self.fsm_state = PoseState.SEEDLING

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
