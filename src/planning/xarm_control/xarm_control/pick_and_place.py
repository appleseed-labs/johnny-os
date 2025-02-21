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


class PickAndPlaceNode(Node):
    def __init__(self):
        super().__init__("pick_and_place_node")

        # Declare parameters for position and orientation
        self.declare_parameter("position", [random() / 3, random() / 3, random() / 3])
        self.declare_parameter("quat_xyzw", [1.0, 0.0, 0.0, 0.0])
        self.declare_parameter("synchronous", True)
        # If non-positive, don't cancel. Only used if synchronous is False
        self.declare_parameter("cancel_after_secs", 0.0)
        # Planner ID
        self.declare_parameter("planner_id", "")
        # Declare parameters for cartesian planning
        self.declare_parameter("cartesian", False)
        self.declare_parameter("cartesian_max_step", 0.0025)
        self.declare_parameter("cartesian_fraction_threshold", 0.0)
        self.declare_parameter("cartesian_jump_threshold", 0.0)
        self.declare_parameter("cartesian_avoid_collisions", False)

        # Create callback group that allows execution of callbacks in parallel without restrictions
        callback_group = ReentrantCallbackGroup()

        # Create MoveIt 2 interface
        moveit2 = MoveIt2(
            node=self,
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            base_link_name="link_base",
            end_effector_name="link6",
            group_name="xarm6",
            callback_group=callback_group,
        )
        moveit2.planner_id = (
            self.get_parameter("planner_id").get_parameter_value().string_value
        )

        # Spin the node in background thread(s) and wait a bit for initialization
        executor = rclpy.executors.MultiThreadedExecutor(2)
        executor.add_node(self)
        executor_thread = Thread(target=executor.spin, daemon=True, args=())
        executor_thread.start()
        self.create_rate(1.0).sleep()

        # Scale down velocity and acceleration of joints (percentage of maximum)
        moveit2.max_velocity = 0.5
        moveit2.max_acceleration = 0.5

        # Get parameters
        position = (
            self.get_parameter("position").get_parameter_value().double_array_value
        )
        quat_xyzw = (
            self.get_parameter("quat_xyzw").get_parameter_value().double_array_value
        )
        synchronous = self.get_parameter("synchronous").get_parameter_value().bool_value
        cancel_after_secs = (
            self.get_parameter("cancel_after_secs").get_parameter_value().double_value
        )
        cartesian = self.get_parameter("cartesian").get_parameter_value().bool_value
        cartesian_max_step = (
            self.get_parameter("cartesian_max_step").get_parameter_value().double_value
        )
        cartesian_fraction_threshold = (
            self.get_parameter("cartesian_fraction_threshold")
            .get_parameter_value()
            .double_value
        )
        cartesian_jump_threshold = (
            self.get_parameter("cartesian_jump_threshold")
            .get_parameter_value()
            .double_value
        )
        cartesian_avoid_collisions = (
            self.get_parameter("cartesian_avoid_collisions")
            .get_parameter_value()
            .bool_value
        )

        # Set parameters for cartesian planning
        moveit2.cartesian_avoid_collisions = cartesian_avoid_collisions
        moveit2.cartesian_jump_threshold = cartesian_jump_threshold

        # Move to pose
        self.get_logger().info(
            f"Moving to {{position: {list(position)}, quat_xyzw: {list(quat_xyzw)}}}"
        )

        moveit2.move_to_pose(
            position=position,
            quat_xyzw=quat_xyzw,
            cartesian=cartesian,
            cartesian_max_step=cartesian_max_step,
            cartesian_fraction_threshold=cartesian_fraction_threshold,
        )
        if synchronous:
            # Note: the same functionality can be achieved by setting
            # `synchronous:=false` and `cancel_after_secs` to a negative value.
            moveit2.wait_until_executed()
        else:
            # Wait for the request to get accepted (i.e., for execution to start)
            print("Current State: " + str(moveit2.query_state()))
            rate = self.create_rate(10)
            while moveit2.query_state() != MoveIt2State.EXECUTING:
                rate.sleep()

            # Get the future
            print("Current State: " + str(moveit2.query_state()))
            future = moveit2.get_execution_future()

            # Cancel the goal
            if cancel_after_secs > 0.0:
                # Sleep for the specified time
                sleep_time = self.create_rate(cancel_after_secs)
                sleep_time.sleep()
                # Cancel the goal
                print("Cancelling goal")
                moveit2.cancel_execution()

            # Wait until the future is done
            while not future.done():
                rate.sleep()

            # Print the result
            print("Result status: " + str(future.result().status))
            print("Result error code: " + str(future.result().result.error_code))

        self.get_logger().info(f"DONE")


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
