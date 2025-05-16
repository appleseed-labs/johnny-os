# Copyright (c) farm-ng, inc.
#
# Licensed under the Amiga Development Kit License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/farm-ng/amiga-dev-kit/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python

import can
import rclpy
from rclpy.node import Node
import math

from threading import Thread

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from transforms3d.euler import euler2quat

import time
from enum import IntEnum
from struct import pack
from struct import unpack
from typing import Optional

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

DASHBOARD_NODE_ID = 0xE
PENDANT_NODE_ID = 0xF
BRAIN_NODE_ID = 0x1F
SDK_NODE_ID = 0x2A


class AmigaControlState(IntEnum):
    """State of the Amiga vehicle control unit (VCU)"""

    STATE_BOOT = 0
    STATE_MANUAL_READY = 1
    STATE_MANUAL_ACTIVE = 2
    STATE_CC_ACTIVE = 3
    STATE_AUTO_READY = 4
    STATE_AUTO_ACTIVE = 5
    STATE_ESTOPPED = 6


class Packet:
    """Base class inherited by all CAN message data structures."""

    @classmethod
    def from_can_data(cls, data, stamp: float):
        """Unpack CAN data directly into CAN message data structure."""
        obj = cls()  # Does not call __init__
        obj.decode(data)
        obj.stamp_packet(stamp)
        return obj

    def stamp_packet(self, stamp: float):
        """Time most recent message was received."""
        pass
        # self.stamp: Timestamp = timestamp_from_monotonic("canbus/packet", stamp)

    def fresh(self, thresh_s: float = 0.5):
        """Returns False if the most recent message is older than ``thresh_s`` in seconds."""
        return self.age() < thresh_s

    def age(self):
        """Age of the most recent message."""
        return time.monotonic() - self.stamp.stamp


class AmigaRpdo1(Packet):
    """State, speed, and angular rate command (request) sent to the Amiga vehicle control unit (VCU)"""

    cob_id = 0x200

    def __init__(
        self,
        state_req: AmigaControlState = AmigaControlState.STATE_ESTOPPED,
        cmd_speed: float = 0.0,
        cmd_ang_rate: float = 0.0,
    ):
        self.format = "<Bhh"
        self.state_req = state_req
        self.cmd_speed = cmd_speed
        self.cmd_ang_rate = cmd_ang_rate

        self.stamp_packet(time.monotonic())

    def encode(self):
        """Returns the data contained by the class encoded as CAN message data."""
        return pack(
            self.format,
            self.state_req,
            int(self.cmd_speed * 1000.0),
            int(self.cmd_ang_rate * 1000.0),
        )

    def decode(self, data):
        """Decodes CAN message data and populates the values of the class."""
        (self.state_req, cmd_speed, cmd_ang_rate) = unpack(self.format, data)
        self.cmd_speed = cmd_speed / 1000.0
        self.cmd_ang_rate = cmd_ang_rate / 1000.0

    def __str__(self):
        return "AMIGA RPDO1 Request state {} Command speed {:0.3f} Command angular rate {:0.3f}".format(
            self.state_req, self.cmd_speed, self.cmd_ang_rate
        )


class AmigaTpdo1(Packet):
    """State, speed, and angular rate of the Amiga vehicle control unit (VCU)"""

    cob_id = 0x180

    def __init__(
        self,
        state: AmigaControlState = AmigaControlState.STATE_ESTOPPED,
        meas_speed: float = 0.0,
        meas_ang_rate: float = 0.0,
    ):
        self.format = "<Bhh"
        self.state = state
        self.meas_speed = meas_speed
        self.meas_ang_rate = meas_ang_rate

        self.stamp_packet(time.monotonic())

    def encode(self):
        """Returns the data contained by the class encoded as CAN message data."""
        return pack(
            self.format,
            self.state,
            int(self.meas_speed * 1000.0),
            int(self.meas_ang_rate * 1000.0),
        )

    def decode(self, data):
        """Decodes CAN message data and populates the values of the class."""
        (self.state, meas_speed, meas_ang_rate) = unpack(self.format, data)
        self.meas_speed = meas_speed / 1000.0
        self.meas_ang_rate = meas_ang_rate / 1000.0

    def __str__(self):
        return "AMIGA TPDO1 Amiga state {} Measured speed {:0.3f} Measured angular rate {:0.3f} @ time {}".format(
            self.state, self.meas_speed, self.meas_ang_rate, self.stamp.stamp
        )


class AmigaControl(Node):
    def __init__(self):
        super().__init__("amiga_control_node")
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.can_filters_ = [{"can_id": 0x18E, "can_mask": 0x7FF, "extended": False}]
        self.can_bus_ = can.interface.Bus(
            bustype="socketcan",
            channel="can0",
            bitrate=500000,
            can_filters=self.can_filters_,
        )

        self.amiga_state_ = AmigaTpdo1()
        self.amiga_cmd_state_ = AmigaRpdo1()

        self.prev_time = None
        self.x = 0
        self.y = 0
        self.th = 0

        self.odom_pub_ = self.create_publisher(Odometry, "odom", 1)
        self.cmd_vel_sub_ = self.create_subscription(
            Twist, "cmd_vel", self.cmd_vel_cb, qos_profile
        )

        self.monitor_thread_ = Thread(target=self.can_loop)
        self.monitor_thread_.start()

        self.get_logger().info("Amiga control node started")

    def __del__(self):
        self.monitor_thread_.join()

    def can_loop(self):
        for msg in self.can_bus_:
            if not rclpy.ok():
                break
            self.amiga_state_.decode(msg.data)

            curr_time = msg.timestamp
            if self.prev_time is None:
                self.prev_time = curr_time
                continue
            dT = curr_time - self.prev_time
            self.prev_time = curr_time

            dist = self.amiga_state_.meas_speed * dT

            self.th = (self.th + self.amiga_state_.meas_ang_rate * dT) % (2 * math.pi)
            self.x = self.x + dist * math.cos(self.th)
            self.y = self.y + dist * math.sin(self.th)

            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = "odom"
            odom_msg.pose.pose.position.x = self.x
            odom_msg.pose.pose.position.y = self.y
            odom_msg.pose.pose.position.z = 0.0
            quaternion = euler2quat(0.0, 0.0, self.th)
            odom_msg.pose.pose.orientation.w = quaternion[0]
            odom_msg.pose.pose.orientation.x = quaternion[1]
            odom_msg.pose.pose.orientation.y = quaternion[2]
            odom_msg.pose.pose.orientation.z = quaternion[3]

            odom_msg.pose.covariance = [
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.01,
                0.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.001,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.001,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.001,
            ]

            odom_msg.twist.twist.linear.x = self.amiga_state_.meas_speed
            odom_msg.twist.twist.angular.z = self.amiga_state_.meas_ang_rate
            self.odom_pub_.publish(odom_msg)

    def cmd_vel_cb(self, msg):
        cmd_vel_msg = msg
        self.amiga_cmd_state_.state_req = AmigaControlState.STATE_AUTO_ACTIVE
        self.amiga_cmd_state_.cmd_speed = cmd_vel_msg.linear.x
        self.amiga_cmd_state_.cmd_ang_rate = cmd_vel_msg.angular.z

        cmd_msg = can.Message(
            arbitration_id=0x20E,
            data=self.amiga_cmd_state_.encode(),
            is_extended_id=True,
        )
        try:
            self.can_bus_.send(cmd_msg)
        except can.CanError:
            self.get_logger().info("Error sending CAN msg to Amiga")


def main(args=None):
    rclpy.init(args=args)
    control = AmigaControl()

    try:
        rclpy.spin(control)
    finally:
        control.can_bus_.shutdown()
        control.can_bus_.stop()
        control.can_bus_.reset()
        control.can_bus_.close()
        control.can_bus_ = None
        control.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
