# -----------------------------------------------------------------------------
# Description: Converts a NavSatFix and Imu message to a transform and odometry message.
# Author: Will Heitman
# (c) 2025 Appleseed Labs. CMU Robotics Institute
# -----------------------------------------------------------------------------

import numpy as np
import math
import rclpy
from rclpy.node import Node, ParameterDescriptor, ParameterType
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R
import utm

# Messages
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Header


class FixToTransformNode(Node):
    """
    Converts a NavSatFix and Imu message to a transform and odometry message.
    """

    def __init__(self):
        super().__init__("fix_to_transform_node")

        self.setUpParameters()

        self.create_subscription(NavSatFix, "/gnss/fix", self.fixCb, 1)  # For position
        self.create_subscription(Imu, "/imu", self.imuCb, 1)  # For orientation

        self.odom_pub = self.create_publisher(Odometry, "/gnss/odom", 1)

        # Broadcast a map -> base_link transform
        self.tf_broadcaster = TransformBroadcaster(self)

        self.yaw_enu = None

        # Calculate our map origin
        lat0, lon0, alt0 = self.get_parameter("map_origin_lat_lon_alt_degrees").value
        # self.origin_utm_x, self.origin_utm_y, _, __ = utm.from_latlon(lat0, lon0)
        self.origin_z = alt0

        # NOTE: Rohan fix
        self.origin_utm_x = None
        self.origin_utm_y = None

    def imuCb(self, msg: Imu):
        # Check for valid quaternion
        if (
            msg.orientation.x == 0
            and msg.orientation.y == 0
            and msg.orientation.z == 0
            and msg.orientation.w == 0
        ):
            return

        # Convert quaternion to yaw using SciPy
        q = msg.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        self.yaw_enu = r.as_euler("xyz")[2] + math.pi / 2

        if (self.yaw_enu) > 2 * math.pi:
            self.yaw_enu -= 2 * math.pi
        if (self.yaw_enu) < 0:
            self.yaw_enu += 2 * math.pi

        print(self.yaw_enu)

    def fixCb(self, msg: NavSatFix):
        # Convert to UTM
        utm_x, utm_y, _, __ = utm.from_latlon(msg.latitude, msg.longitude)
        # NOTE: Rohan fix
        if self.origin_utm_x is None and self.origin_utm_y is None:
            self.origin_utm_x = utm_x
            self.origin_utm_y = utm_y
            self.get_logger().info(f"{msg.latitude}, {msg.longitude}")
            self.get_logger().info(f"We got origin: {utm_x}, {utm_y}")
        # Calculate the position relative to the origin
        x = utm_x - self.origin_utm_x
        y = utm_y - self.origin_utm_y

        # Create a transform message
        t = TransformStamped()
        t.header = self.getHeader()
        t.child_frame_id = "base_link"
        t.transform.translation.x = x
        t.transform.translation.y = y

        # NOTE: We assume that z is zero, which makes downstream algorithms much easier
        # This is an okay assumption for a UGV when considering global features.
        # We can leverage local height data when processing relative sensor data, e.g.
        # lidar, camera, etc. WSH.
        # t.transform.translation.z = msg.altitude - self.origin_z
        t.transform.translation.z = 0.0

        # Set the orientation (yaw) from the IMU data
        if self.yaw_enu is not None:
            q = R.from_euler("z", self.yaw_enu, degrees=False).as_quat()
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            # Publish the transform
            self.tf_broadcaster.sendTransform(t)

        # Create and publish the odometry message
        odom = Odometry()
        odom.header = self.getHeader()
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = msg.altitude - self.origin_z
        odom.pose.pose.orientation = t.transform.rotation
        self.odom_pub.publish(odom)

    def getHeader(self):
        msg = Header()
        msg.frame_id = "map"  # routes are in the map frame
        msg.stamp = self.get_clock().now().to_msg()
        return msg

    def setUpParameters(self):
        param_desc = ParameterDescriptor()
        param_desc.type = ParameterType.PARAMETER_DOUBLE_ARRAY
        self.declare_parameter(
            "map_origin_lat_lon_alt_degrees",
            [40.4431653, -79.9402844, 288.0961589],
        )


def main(args=None):
    rclpy.init(args=args)

    node = FixToTransformNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
