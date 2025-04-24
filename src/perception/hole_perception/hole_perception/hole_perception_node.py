# -----------------------------------------------------------------------------
# Description: Converts Joy messages to Johnny control signals
# Author: Will Heitman
# (c) 2025 Appleseed Labs. CMU Robotics Institute
# -----------------------------------------------------------------------------

from cairo import ImageSurface
import numpy as np
import math

from sympy import Point
import rclpy
from rclpy.node import Node, ParameterDescriptor, ParameterType


# Messages
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy, PointCloud2, Image
from std_msgs.msg import Empty
from time import time

from matplotlib import pyplot as plt

import numpy as np
from numpy.lib.stride_tricks import as_strided
from cv_bridge import CvBridge

name_to_dtypes = {}


def image_to_numpy(msg):
    if not msg.encoding in name_to_dtypes:
        raise TypeError("Unrecognized encoding {}".format(msg.encoding))

    dtype_class, channels = name_to_dtypes[msg.encoding]
    dtype = np.dtype(dtype_class)
    dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
    shape = (msg.height, msg.width, channels)

    data = np.frombuffer(msg.data, dtype=dtype).reshape(shape)
    data.strides = (msg.step, dtype.itemsize * channels, dtype.itemsize)

    if channels == 1:
        data = data[..., 0]
    return data


class HolePerceptionNode(Node):
    """
    Converts a NavSatFix and Imu message to a transform and odometry message.
    """

    def __init__(self):
        super().__init__("hole_perception_node")

        self.set_up_parameters()

        self.create_subscription(
            Image, "/camera/camera/depth/image_rect_raw", self.depth_image_cb, 1
        )

        self.get_logger().info("Hole Perception Node Initialized")
        self.bridge = CvBridge()

    def depth_image_cb(self, msg):
        try:
            # Convert ROS2 Image message to OpenCV image (numpy array)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Now cv_image is a numpy array
            print(type(cv_image))  # Should print: <class 'numpy.ndarray'>
            print(cv_image.shape)  # Prints the dimensions of the image

        except Exception as e:
            self.get_logger().error(f"Error converting image: {str(e)}")
            return

        min_depth = np.min(cv_image)
        max_depth = np.max(cv_image)
        top_90 = min_depth + (max_depth - min_depth) * 0.9

        print(cv_image)
        print(f"Top 90% depth: {top_90}")

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
        print(f"x_error: {x_error}, y_error: {y_error}")

        plt.imshow(cv_image)
        plt.savefig("depth_image.png")

    def set_up_parameters(self):
        pass


def main(args=None):
    rclpy.init(args=args)

    node = HolePerceptionNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
