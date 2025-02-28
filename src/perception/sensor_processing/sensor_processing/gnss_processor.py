import numpy as np
import math
import rclpy
from rclpy.node import Node, ParameterDescriptor, ParameterType
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

# Messages
from geometry_msgs.msg import TransformStamped
from gps_msgs.msg import GPSFix
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Header, Float32

from tf2_ros import TransformException, TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R

import utm


class GnssProcessor(Node):
    def __init__(self):
        super().__init__("gnss_processor")

        self.setUpParameters()

        self.create_subscription(NavSatFix, "/gnss/fix", self.fixCb, 1)
        self.create_subscription(Imu, "/imu", self.imuCb, 1)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.yaw_enu_rads = None

    def imuCb(self, msg: Imu):
        """Check if IMU message contains orientation data. If it does, convert this to yaw and cache it.

        We assume here that the IMU's yaw is zero at north but is otherwise ENU.
        0 degrees = North

        Args:
            msg (Imu): IMU message
        """

        if msg.orientation.x == 0 and msg.orientation.y == 0 and msg.orientation.z == 0:
            return  # No orientation data available

        # Convert quaternion to yaw
        r = R.from_quat(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        )
        yaw = r.as_euler("xyz", degrees=False)[2] + math.pi / 2

        # Wrap in interval [0, 2pi]
        if yaw < 0:
            yaw += 2 * math.pi
        elif yaw > 2 * math.pi:
            yaw -= 2 * math.pi

        self.yaw_enu_rads = yaw

    def fixCb(self, msg: NavSatFix):
        # print(msg.latitude, msg.longitude)

        map_origin_lat_lon_alt_degrees = (
            self.get_parameter("map_origin_lat_lon_alt_degrees")
            .get_parameter_value()
            .double_array_value
        )

        assert (
            len(map_origin_lat_lon_alt_degrees) == 3
        ), "map_origin_lat_lon_alt_degrees must be a list of 3 values"
        origin_lat, origin_lon, origin_alt = map_origin_lat_lon_alt_degrees

        origin_utm_x, origin_utm_y, _, __ = utm.from_latlon(origin_lat, origin_lon)

        ego_utm_x, ego_utm_y, _, __ = utm.from_latlon(msg.latitude, msg.longitude)

        # Calculate the ego position, relative to the map frame origin, in meters
        # Recall that we use ENU coordinates (East, North, Up) for the map frame
        ego_x = ego_utm_x - origin_utm_x
        ego_y = ego_utm_y - origin_utm_y

        # Publish the transform from the map frame to the base_link frame
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = ego_x
        t.transform.translation.y = ego_y
        t.transform.translation.z = msg.altitude - origin_alt

        q = R.from_euler("z", self.yaw_enu_rads, degrees=False).as_quat()
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

        # print(f"{ego_x} {ego_y}")

    def trueTrackToEnuRads(self, track_deg: float):
        enu_yaw = track_deg

        enu_yaw -= 90

        enu_yaw = 360 - enu_yaw

        if enu_yaw < 0:
            enu_yaw += 360
        elif enu_yaw > 360:
            enu_yaw -= 360

        enu_yaw *= math.pi / 180.0
        return enu_yaw

    def setUpParameters(self):
        param_desc = ParameterDescriptor()
        param_desc.type = ParameterType.PARAMETER_DOUBLE_ARRAY
        self.declare_parameter(
            "map_origin_lat_lon_alt_degrees",
            [35.710202065753009, 139.81070039691542, 3.0],
        )


def main(args=None):
    rclpy.init(args=args)

    node = GnssProcessor()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.ser.close()
    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
