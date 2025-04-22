#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: Move Joint
"""

import os
import sys
import time
import math
import serial


# sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
from xarm.wrapper import XArmAPI

pos_dictionary = {
    "home": [
        -0.32572651,
        -80.93894022,
        -14.3142046,
        -188.32544039,
        -7.37580029,
        -170.15998538,
        0.0,
    ],
    "ready": [
        130.09470834,
        33.48371084,
        -139.9335396,
        -68.80289198,
        129.62064306,
        -175.63041452,
        0.0,
    ],
    "seedling_1_pre": [
        117.616547,
        -12.89321197,
        -55.8271168,
        -53.73828456,
        99.10324932,
        -147.33598243,
        0.0,
    ],
    "seedling_1_grab": [
        116.8671755,
        -9.62150837,
        -51.20936345,
        -53.80446119,
        96.2770204,
        -138.36598437,
        0.0,
    ],
    "seedling_1_lift": [
        138.18154925,
        11.32691724,
        -77.62907763,
        -43.34248103,
        115.49729071,
        -129.13036308,
        0.0,
    ],
    "over_hole": [-29.1542, 16.5677, -28.8071, -176.041, 62.1542, -158.9843, 0.0],
    "to_hole_1": [73.9455, -4.1437, -91.7979, -64.7164, 76.854, -187.4988, 0.0],
    "to_hole_2": [7.449, -51.7186, -27.6111, -84.6549, 8.9232, -187.504, 0.0],
}

ip = "192.168.1.196"

# Setup up the xArm
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

angles = np.zeros_like(arm.angles)


def open_gripper():
    with serial.Serial("/dev/ttyACM0", 57600, timeout=1) as ser:
        ser.write(b"O\r\n")
        time.sleep(0.5)


def close_gripper():
    with serial.Serial("/dev/ttyACM0", 57600, timeout=1) as ser:
        ser.write(b"C\r\n")
        time.sleep(0.5)


FULL_SPEED = 40
CAREFUL_SPEED = FULL_SPEED / 2

open_gripper()
arm.set_servo_angle(angle=pos_dictionary["home"], speed=FULL_SPEED, wait=True)
arm.set_servo_angle(angle=pos_dictionary["ready"], speed=FULL_SPEED, wait=True)
arm.set_servo_angle(angle=pos_dictionary["seedling_1_pre"], speed=FULL_SPEED, wait=True)
arm.set_servo_angle(
    angle=pos_dictionary["seedling_1_grab"], speed=FULL_SPEED, wait=True
)
close_gripper()
arm.set_servo_angle(
    angle=pos_dictionary["seedling_1_lift"], speed=CAREFUL_SPEED, wait=True
)
arm.set_servo_angle(angle=pos_dictionary["ready"], speed=FULL_SPEED, wait=True)
arm.set_servo_angle(angle=pos_dictionary["to_hole_1"], speed=FULL_SPEED, wait=True)
arm.set_servo_angle(angle=pos_dictionary["to_hole_2"], speed=FULL_SPEED, wait=True)
arm.set_servo_angle(angle=pos_dictionary["over_hole"], speed=CAREFUL_SPEED, wait=True)
time.sleep(5)
open_gripper()

# close_gripper()
# arm.set_servo_angle(angle=pos_dictionary["home"], speed=10, wait=True)
# arm.set_servo_angle(angle=pos_dictionary["seedling_1_grab"], speed=10, wait=True)

while True:
    new_angles = np.asarray(arm.angles)

    if np.linalg.norm(new_angles - angles) > 0.01:
        print(f"{[float(round(angle, 4)) for angle in new_angles]}")
        angles = new_angles


# arm.move_gohome(wait=True)
arm.disconnect()
