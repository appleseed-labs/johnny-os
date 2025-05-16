# -----------------------------------------------------------------------------
# Description: Converts Joy messages to Johnny control signals
# Author: Will Heitman
# (c) 2025 Appleseed Labs. CMU Robotics Institute
# -----------------------------------------------------------------------------

import numpy as np
import math
import rclpy
from rclpy.node import Node, ParameterDescriptor, ParameterType

# Messages
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Empty
from time import time


class JoystickInterfaceNode(Node):
    """
    Converts a NavSatFix and Imu message to a transform and odometry message.
    """

    def __init__(self):
        super().__init__("joystick_interface_node")

        self.set_up_parameters()

        # Published by "joy_node" ("joy" package)
        self.create_subscription(Joy, "/joy", self.joy_cb, 1)

        # Subscribed to by "xarm_control" ("xarm_control" package)
        self.start_planting_pub = self.create_publisher(
            Empty, "/behavior/start_planting", 1
        )

        # Subscribed to by "linak_control" ("linak_control" package)
        self.start_drilling_pub = self.create_publisher(
            Empty, "/behavior/start_drilling", 1
        )

        # Subscribed to by the Behavior Manager
        self.on_arrived_pub = self.create_publisher(Empty, "/behavior/on_arrived", 1)

        # For sending movement commands
        self.twist_pub = self.create_publisher(Twist, "/cmd_vel", 1)

        self.time_since_sending_plant = 0.0
        self.time_since_sending_plant_cooloff = 3.0  # seconds
        self.time_since_sending_drill = 0.0
        self.time_since_sending_drill_cooloff = 3.0  # seconds

        self.time_since_sending_on_arrived = 0.0
        self.time_since_sending_on_arrived_cooloff = 3.0  # seconds

    def joy_cb(self, msg):

        if msg.buttons[0] == 1:  # "A" button
            if (
                time() - self.time_since_sending_plant
                < self.time_since_sending_plant_cooloff
            ):
                return
            # Start planting
            self.get_logger().info("Start planting")
            self.start_planting_pub.publish(Empty())
            self.time_since_sending_plant = time()

        if msg.buttons[1] == 1:  # "B" button
            if (
                time() - self.time_since_sending_drill
                < self.time_since_sending_drill_cooloff
            ):
                return
            # Start drilling
            self.get_logger().info("Start drilling")
            self.start_drilling_pub.publish(Empty())
            self.time_since_sending_drill = time()

        if msg.buttons[3] == 1:  # "X" button
            if (
                time() - self.time_since_sending_on_arrived
                < self.time_since_sending_on_arrived_cooloff
            ):
                return
            # On arrived
            self.get_logger().info("On arrived")
            self.on_arrived_pub.publish(Empty())
            self.time_since_sending_on_arrived = time()

        # Convert joystick input to movement commands
        # Right stick controls forward/backward and left/right
        forward = msg.axes[3] * 0.7
        left_right = msg.axes[2]
        twist = Twist()
        twist.linear.x = forward
        twist.angular.z = left_right
        self.twist_pub.publish(twist)

    def set_up_parameters(self):
        pass


def main(args=None):
    rclpy.init(args=args)

    node = JoystickInterfaceNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
