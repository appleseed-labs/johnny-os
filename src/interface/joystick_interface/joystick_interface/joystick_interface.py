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


class JoystickInterfaceNode(Node):
    """
    Converts a NavSatFix and Imu message to a transform and odometry message.
    """

    def __init__(self):
        super().__init__("joystick_interface_node")

        self.set_up_parameters()

        # Published by "joy_node" ("joy" package)
        self.create_subscription(Joy, "/joy", self.joy_cb, 1)

        # Published by "xarm_control" ("xarm_control" package)
        self.start_planting_pub = self.create_publisher(
            Empty, "/behavior/start_planting", 1
        )

        # For sending movement commands
        self.twist_pub = self.create_publisher(Twist, "/cmd_vel", 1)

    def joy_cb(self, msg):

        if msg.buttons[0] == 1:
            # Start planting
            self.get_logger().info("Start planting")
            self.start_planting_pub.publish(Empty())

        # Convert joystick input to movement commands
        # Right stick controls forward/backward and left/right
        forward = msg.axes[3]
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
