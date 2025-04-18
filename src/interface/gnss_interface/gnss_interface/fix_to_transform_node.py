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
from gps_msgs.msg import GPSFix
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Header, Float32


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

        self.create_subscription(GPSFix, "/gps/gpsfix", self.swiftFixCb, 1)

        # Broadcast a map -> base_link transform
        self.tf_broadcaster = TransformBroadcaster(self)

        self.yaw_enu = None

        # Calculate our map origin
        lat0, lon0, alt0 = self.get_parameter("map_origin_lat_lon_alt_degrees").value
        self.origin_utm_x, self.origin_utm_y, _, __ = utm.from_latlon(lat0, lon0)
        self.origin_z = alt0

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

    def swiftFixCb(self, swift_msg: GPSFix):
        # publish as Odometry message

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_link"

        ego_utm_x, ego_utm_y, _, __ = utm.from_latlon(
            swift_msg.latitude, swift_msg.longitude
        )

        ego_x = ego_utm_x - self.origin_utm_x
        ego_y = ego_utm_y - self.origin_utm_y

        odom_msg.pose.pose.position.x = ego_x
        odom_msg.pose.pose.position.y = ego_y
        odom_msg.pose.pose.position.z = swift_msg.altitude

        # This calculates the orientation for the entire robot.
        # Yes, this assumes that we're on a flat plane.
        yaw = self.trueTrackToEnuRads(swift_msg.track)
        q = R.from_euler("xyz", [0.0, 0.0, yaw]).as_quat()

        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        t = TransformStamped()
        t.transform.translation.x = ego_x
        t.transform.translation.y = ego_y
        t.transform.translation.z = swift_msg.altitude
        t.transform.rotation = odom_msg.pose.pose.orientation
        t.header = odom_msg.header
        t.child_frame_id = odom_msg.child_frame_id
        self.tf_broadcaster.sendTransform(t)

        self.odom_pub.publish(odom_msg)
        self.yaw_pub.publish(Float32(data=yaw))

    def fixCb(self, msg: NavSatFix):
        # Convert to UTM
        utm_x, utm_y, _, __ = utm.from_latlon(msg.latitude, msg.longitude)

        # Calculate the position relative to the origin
        x = utm_x - self.origin_utm_x
        y = utm_y - self.origin_utm_y

        # Create a transform message
        t = TransformStamped()
        t.header = self.getHeader()
        t.child_frame_id = "base_link"
        t.transform.translation.x = x
        t.transform.translation.y = y
        # t.transform.translation.z = msg.altitude - self.origin_z

        # Here we use the planar assumption, that z is zero. Big assumption,
        # but it simplifies things greatly!
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
