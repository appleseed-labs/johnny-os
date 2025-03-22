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

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI


ip = "192.168.1.196"

# Setup up the xArm
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

arm.move_gohome(wait=True, speed=50)

# speed = 50
# arm.set_servo_angle(angle=[90, 0, 0, 0, 0, 0], speed=speed, wait=True)
# print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=True))
# arm.set_servo_angle(angle=[90, 0, -60, 0, 0, 0], speed=speed, wait=True)
# print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=True))
# arm.set_servo_angle(angle=[90, -30, -60, 0, 0, 0], speed=speed, wait=True)
# print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=True))
# arm.set_servo_angle(angle=[0, -30, -60, 0, 0, 0], speed=speed, wait=True)
# print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=True))
# arm.set_servo_angle(angle=[0, 0, -60, 0, 0, 0], speed=speed, wait=True)
# print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=True))
# arm.set_servo_angle(angle=[0, 0, 0, 0, 0, 0], speed=speed, wait=True)
# print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=True))

# arm.set_mode(0)
# arm.set_state(state=0)

# arm.move_gohome(wait=True)

# def go

READY_POSE_XYZ = [-7, 263, 523, -71, 67, 69]  # Last three in degrees
READY_POSE_Q = [94.8, -34.4, -53.7, -2.4, -3.7, 0.2]
OVERHEAD_POSE_Q = [0, -56.1, -34.9, 0, 0, 0]

speed = 50
# arm.set_servo_angle(angle=OVERHEAD_POSE_Q, speed=speed, wait=True)
# arm.set_servo_angle(angle=READY_POSE_Q, speed=speed, wait=True)

# arm.set_servo_angle()
# arm.set_servo_angle(angle=[90, 0, 0, 0, 0, 0], speed=speed, wait=True)
# print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=True))
# arm.set_servo_angle(angle=[90, 0, -60, 0, 0, 0], speed=speed, wait=True)
# print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=True))


# arm.move_gohome(wait=True)
arm.disconnect()
