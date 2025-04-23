#!/usr/bin/env python
import canopen
import time

from pathlib import Path
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

import os
import serial
from tqdm import tqdm
import numpy as np

# === Configuration ===
CHANNEL = "can1"  # change this to can1 for the Amiga computer
NODE_ID = 0x20  # 32 decimal
COB_ID_RPDO1 = 0x200 + NODE_ID
BOOTUP_COB_ID = 0x700 + NODE_ID
HEARTBEAT_PRODUCER_ID = 0x701  # Master heartbeat ID
HEARTBEAT_TIME_MS = 100

LINAK_DRILL_POS = 2400


class actuation_subscriber(Node):
    def __init__(self, canbus):
        super().__init__("linak_actuator_node")
        self.subscription = self.create_subscription(
            Empty, "/behavior/start_drilling", self.start_drilling_cb, 10
        )
        self.canbus = canbus

    def increment_linak_position(self, initial, final, duration, n_steps):
        step_size = (final - initial) / n_steps
        step_duration = duration / n_steps

        positions = np.linspace(initial, final, n_steps + 1)

        pbar = tqdm(total=n_steps, desc="Moving LINAK", unit="step")
        for i, pos in enumerate(positions):
            self.canbus.send_actuator_command(int(pos))
            time.sleep(step_duration)
            pbar.update(1)
            pbar.set_description(f"Moving LINAK to {pos}")
        pbar.close()

        # Ensure the final position is set
        self.canbus.send_actuator_command(final)

    def start_drilling_cb(self, msg):
        self.get_logger().info("Drilling now...")
        self.canbus.send_actuator_command(1800)
        self.start_auger()
        time.sleep(15)
        self.increment_linak_position(initial=1800, final=2400, duration=15, n_steps=50)

        self.canbus.send_actuator_command(64258)  # === RUN IN ===
        time.sleep(15)
        # print("Final STOP...")
        self.canbus.send_actuator_command(64259)  # === STOP ===
        self.stop_auger()
        time.sleep(0.5)

    def start_auger(self):
        with serial.Serial("/dev/ttyACM1", 9600, timeout=1) as ser:
            print("Starting auger...")
            ser.write(b"1\r\n")
            # time.sleep(5.0)

    def stop_auger(self):

        with serial.Serial("/dev/ttyACM1", 9600, timeout=1) as ser:
            print("Stopping auger...")
            ser.write(b"0\r\n")
            time.sleep(0.5)


class canbus_comms:
    def __init__(self):
        # === Connect to CAN ===
        self.network = canopen.Network()
        self.network.connect(channel=CHANNEL, bustype="socketcan")
        pkg_share = Path(get_package_share_directory("linak_controller"))
        # eds_path  = pkg_share + 'config' + 'LINAK-actuator-v3-1.eds'
        eds_path = os.path.join(pkg_share, "config", "LINAK-actuator-v3-1.eds")
        # print("eds_pathxxxxxxxxxxx", eds_path)
        self.node = canopen.RemoteNode(NODE_ID, eds_path)
        self.network.add_node(self.node)
        self.setup_bus()

    def setup_bus(self):
        self.network.send_message(HEARTBEAT_PRODUCER_ID, [0x05])
        # === Set heartbeat expectation (consumer heartbeat time) ===
        print(f"Setting actuator consumer heartbeat time to {HEARTBEAT_TIME_MS} ms...")
        self.node.sdo[0x1016][1].raw = (0x01 << 16) + 100  # 0x00010064
        time.sleep(0.1)

        rpdo = self.node.rpdo[1]
        rpdo.clear()
        rpdo.add_variable("Actuator Command.Position")
        rpdo.enabled = True
        rpdo.save()
        time.sleep(0.1)

        # === Set actuator to OPERATIONAL ===
        print("Setting actuator to OPERATIONAL...")
        self.node.nmt.state = "OPERATIONAL"
        time.sleep(0.1)

        # === Initialize with STOP ===
        print("Initial STOP...")
        self.send_actuator_command(64259)
        time.sleep(1)

        # === Clear errors, if any ===
        self.send_actuator_command(64256)
        time.sleep(1)

    def send_actuator_command(self, position_code):
        """Send 8-byte RPDO message."""
        msg = [
            position_code & 0xFF,
            (position_code >> 8) & 0xFF,  # Position field (little endian)
            0xFB,
            0xFB,
            0xFB,
            0xFB,  # Default current, speed, ramps
            0x00,
            0x00,  # Padding
        ]
        print(f"RPDO Command → {msg}")
        self.network.send_message(COB_ID_RPDO1, msg)

    def shut_down(self):
        self.network.disconnect()


def main(args=None):
    rclpy.init(args=args)
    canbus = canbus_comms()
    act_subscriber = actuation_subscriber(canbus)

    rclpy.spin(act_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    canbus.shut_down()
    act_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
