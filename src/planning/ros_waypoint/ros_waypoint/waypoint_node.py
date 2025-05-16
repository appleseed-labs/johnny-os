import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from collections import deque
import random
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose
import casadi
from scipy.spatial.transform import Rotation as R
import time
import numpy as np

# CODE CREDIT: https://uuvsimulator.github.io/packages/uuv_simulator/docs/features/jupyter_notebooks/2d_dubins_path/


def get2dRotationMatrix(angle):
    """
    Generates a 2D rotation matrix for a given angle.
    The rotation matrix is used to rotate a point or vector in a 2D plane
    counterclockwise by the specified angle.
    Parameters:
        angle (float): The angle of rotation in radians.
    Returns:
        numpy.ndarray: A 2x2 rotation matrix.
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


u = casadi.SX.sym("u")


def getCirclePoint(pos, radius, offset=0.0):
    """
    Generates points on a circle given its center position, radius, and an optional angular offset.
    Args:
        pos (tuple): A tuple (x, y) representing the center position of the circle.
        radius (float): The radius of the circle.
        offset (float, optional): An angular offset in radians to rotate the circle points. Defaults to 0.0.
    Returns:
        tuple: Two numpy arrays (x, y) representing the x and y coordinates of the points on the circle.
    """

    u = np.linspace(0, 1, 50)
    x = pos[0] + radius * np.cos(2 * np.pi * u + offset)
    y = pos[1] + radius * np.sin(2 * np.pi * u + offset)
    return x, y


def getArcCenters(wp, radius, heading):
    """
    Calculate the center points of an arc (left and right) based on a waypoint, radius, and heading.
    Args:
        wp (numpy.ndarray): The waypoint position as a 2D vector (e.g., [x, y]).
        radius (float): The radius of the arc.
        heading (float): The heading angle in radians.
    Returns:
        dict: A dictionary containing the center points of the arc:
            - 'R': The center of the arc to the right of the waypoint.
            - 'L': The center of the arc to the left of the waypoint.
    """

    frame = get2dRotationMatrix(heading)
    return dict(
        R=wp - radius * frame[:, 1].flatten(), L=wp + radius * frame[:, 1].flatten()
    )


def getTangents(
    center_1,
    radius_1,
    heading_1,
    delta_1,
    center_2,
    radius_2,
    end_yaw,
    delta_2,
    resolution_meters=0.1,
):
    """
    Compute the tangents between two circles and optionally plot the results.
    This function calculates the tangents between two circles defined by their
    centers, radii, headings, and deltas. It also provides an option to visualize
    the tangents, circles, and other relevant geometric elements.
    Parameters:
    ----------
    center_1 : array-like
        Coordinates of the center of the first circle [x, y].
    radius_1 : float
        Radius of the first circle.
    heading_1 : float
        Heading angle of the first circle in radians.
    delta_1 : float
        Direction multiplier for the first circle (+1 for counterclockwise, -1 for clockwise).
    center_2 : array-like
        Coordinates of the center of the second circle [x, y].
    radius_2 : float
        Radius of the second circle.
    end_yaw : float
        Heading angle of the second circle in radians.
    delta_2 : float
        Direction multiplier for the second circle (+1 for counterclockwise, -1 for clockwise).
    plot : bool, optional
        If True, plots the circles, tangents, and other geometric elements (default is False).
    Returns:
    -------
    output : dict
        A dictionary containing the following keys:
        - "C1": List of points along the first circle up to the tangent point.
        - "S": List of points representing the straight-line segment between the tangent points.
        - "C2": List of points along the second circle from the tangent point onward.
    Raises:
    ------
    ValueError
        If no valid path is found due to incorrect tangent calculations.
    Notes:
    ------
    - The function uses CasADi for symbolic computation and NumPy for numerical operations.
    - The plotting functionality requires a compatible plotting library (e.g., Matplotlib).
    Example:
    -------
    >>> center_1 = [0, 0]
    >>> radius_1 = 5
    >>> heading_1 = 0
    >>> delta_1 = 1
    >>> center_2 = [10, 0]
    >>> radius_2 = 5
    >>> end_yaw = 0
    >>> delta_2 = -1
    >>> result = get_tangents(center_1, radius_1, heading_1, delta_1, center_2, radius_2, end_yaw, delta_2, plot=True)
    """

    output = dict()

    u1 = casadi.SX.sym("u1")
    u2 = casadi.SX.sym("u2")

    phi_1 = 2 * np.pi * u1 * delta_1 + heading_1 - delta_1 * np.pi / 2
    phi_2 = 2 * np.pi * u2 * delta_2 + end_yaw - delta_2 * np.pi / 2

    u1_func = lambda angle: (angle - heading_1 + delta_1 * np.pi / 2) / (
        delta_1 * 2 * np.pi
    )
    u2_func = lambda angle: (angle - end_yaw + delta_2 * np.pi / 2) / (
        delta_2 * 2 * np.pi
    )
    # Make tangents vector functions
    tan_1 = casadi.cross(
        np.array([0, 0, 1]),
        np.array(
            [delta_1 * radius_1 * np.cos(phi_1), delta_1 * radius_2 * np.sin(phi_1), 0]
        ),
    )[0:2]
    tan_2 = casadi.cross(
        np.array([0, 0, 1]),
        np.array(
            [delta_2 * radius_1 * np.cos(phi_2), delta_2 * radius_2 * np.sin(phi_2), 0]
        ),
    )[0:2]

    # Make circle functions
    circle_1_func = center_1 + radius_1 * np.array([np.cos(phi_1), np.sin(phi_1)])
    circle_2_func = center_2 + radius_2 * np.array([np.cos(phi_2), np.sin(phi_2)])

    # Plot the circles

    # Compute the line connecting the circle's centers
    d = center_2 - center_1
    # Calculate normal vector to the connecting line
    n = np.dot(get2dRotationMatrix(np.pi / 2), d / np.linalg.norm(d))

    ##########################################################
    # Compute the first tangent
    ## Compute the normal vector's angle
    n_angle = np.arctan2(n[1], n[0])
    ## Compute the parameter for the tangent points on both circles
    u1_opt = u1_func(n_angle)
    if u1_opt < 0:
        u1_opt = u1_opt + 1
    u2_opt = u2_func(n_angle)
    if u2_opt < 0:
        u2_opt = u2_opt + 1

    ## Compute the points on the circles for the first tangent
    c1 = casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt])
    c2 = casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt])

    tangent_1 = c2 - c1
    tangent_1 /= casadi.norm_2(tangent_1)

    ## Compute the tangent vectors on the circles
    t1 = casadi.substitute(tan_1, casadi.vertcat(*[u1]), [u1_opt])
    t1 /= casadi.norm_2(t1)
    t2 = casadi.substitute(tan_2, casadi.vertcat(*[u2]), [u2_opt])
    t2 /= casadi.norm_2(t2)

    diff = float(casadi.norm_2(tangent_1 - t1) + casadi.norm_2(tangent_1 - t2))

    if np.isclose(diff, 0):
        u = np.arange(0, u1_opt, resolution_meters)
        output["C1"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [ui]) for ui in u
        ]
        output["S"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt]),
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt]),
        ]
        u = np.arange(u2_opt, 1, resolution_meters)
        output["C2"] = [
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [ui]) for ui in u
        ]

    ##########################################################
    # Compute the second tangent
    n_angle = np.arctan2(-n[1], -n[0])
    u1_opt = u1_func(n_angle)
    if u1_opt < 0:
        u1_opt = u1_opt + 1
    u2_opt = u2_func(n_angle)
    if u2_opt < 0:
        u2_opt = u2_opt + 1

    c1 = casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt])
    c2 = casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt])

    tangent_2 = c2 - c1
    tangent_2 /= casadi.norm_2(tangent_2)

    ## Compute the tangent vectors on the circles
    t1 = casadi.substitute(tan_1, casadi.vertcat(*[u1]), [u1_opt])
    t1 /= casadi.norm_2(t1)
    t2 = casadi.substitute(tan_2, casadi.vertcat(*[u2]), [u2_opt])
    t2 /= casadi.norm_2(t2)

    diff = float(casadi.norm_2(tangent_2 - t1) + casadi.norm_2(tangent_2 - t2))

    if np.isclose(diff, 0):
        u = np.arange(0, u1_opt, resolution_meters)
        output["C1"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [ui]) for ui in u
        ]
        output["S"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt]),
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt]),
        ]
        u = np.arange(u2_opt, 1, resolution_meters)
        output["C2"] = [
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [ui]) for ui in u
        ]

    ##########################################################
    # Computing inner tangents
    # Calculate the intersection point of the two tangent lines
    xp = (center_1[0] * radius_1 + center_2[0] * radius_2) / (radius_1 + radius_2)
    yp = (center_1[1] * radius_1 + center_2[1] * radius_2) / (radius_1 + radius_2)

    # Third and fourth tangents
    xt1 = (
        radius_1**2 * (xp - center_1[0])
        + radius_1
        * (yp - center_1[1])
        * np.sqrt((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2 - radius_1**2)
    ) / ((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2) + center_1[0]
    xt2 = (
        radius_1**2 * (xp - center_1[0])
        - radius_1
        * (yp - center_1[1])
        * np.sqrt((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2 - radius_1**2)
    ) / ((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2) + center_1[0]

    yt1 = (
        (radius_1**2 * (yp - center_1[1]))
        - radius_1
        * (xp - center_1[0])
        * np.sqrt((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2 - radius_1**2)
    ) / ((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2) + center_1[1]
    yt2 = (
        (radius_1**2 * (yp - center_1[1]))
        + radius_1
        * (xp - center_1[0])
        * np.sqrt((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2 - radius_1**2)
    ) / ((xp - center_1[0]) ** 2 + (yp - center_1[1]) ** 2) + center_1[1]

    xt3 = (
        radius_2**2 * (xp - center_2[0])
        + radius_2
        * (yp - center_2[1])
        * np.sqrt((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2 - radius_2**2)
    ) / ((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2) + center_2[0]
    xt4 = (
        radius_2**2 * (xp - center_2[0])
        - radius_2
        * (yp - center_2[1])
        * np.sqrt((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2 - radius_2**2)
    ) / ((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2) + center_2[0]

    yt3 = (
        (radius_2**2 * (yp - center_2[1]))
        - radius_2
        * (xp - center_2[0])
        * np.sqrt((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2 - radius_2**2)
    ) / ((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2) + center_2[1]
    yt4 = (
        (radius_2**2 * (yp - center_2[1]))
        + radius_2
        * (xp - center_2[0])
        * np.sqrt((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2 - radius_2**2)
    ) / ((xp - center_2[0]) ** 2 + (yp - center_2[1]) ** 2) + center_2[1]

    # Third tangent
    u1_opt = u1_func(np.arctan2(yt1 - center_1[1], xt1 - center_1[0]))
    if u1_opt < 0:
        u1_opt = u1_opt + 1
    u2_opt = u2_func(np.arctan2(yt3 - center_2[1], xt3 - center_2[0]))
    if u2_opt < 0:
        u2_opt = u2_opt + 1

    c1 = casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt])
    c2 = casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt])

    t1 = casadi.substitute(tan_1, casadi.vertcat(*[u1]), [u1_opt])
    t1 /= casadi.norm_2(t1)
    t2 = casadi.substitute(tan_2, casadi.vertcat(*[u2]), [u2_opt])
    t2 /= casadi.norm_2(t2)

    tangent_3 = np.array([xt3 - xt1, yt3 - yt1])
    tangent_3 /= np.linalg.norm(tangent_3)

    diff = float(casadi.norm_2(tangent_3 - t1) + casadi.norm_2(tangent_3 - t2))

    if np.isclose(diff, 0):
        u = np.arange(0, u1_opt, resolution_meters)
        output["C1"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [ui]) for ui in u
        ]
        output["S"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt]),
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt]),
        ]
        u = np.arange(u2_opt, 1, resolution_meters)
        output["C2"] = [
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [ui]) for ui in u
        ]

    # Fourth tangent
    u1_opt = u1_func(np.arctan2(yt2 - center_1[1], xt2 - center_1[0]))
    if u1_opt < 0:
        u1_opt = u1_opt + 1
    u2_opt = u2_func(np.arctan2(yt4 - center_2[1], xt4 - center_2[0]))
    if u2_opt < 0:
        u2_opt = u2_opt + 1

    c1 = casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt])
    c2 = casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt])

    t1 = casadi.substitute(tan_1, casadi.vertcat(*[u1]), [u1_opt])
    t1 /= casadi.norm_2(t1)
    t2 = casadi.substitute(tan_2, casadi.vertcat(*[u2]), [u2_opt])
    t2 /= casadi.norm_2(t2)

    tangent_4 = np.array([xt4 - xt2, yt4 - yt2])
    tangent_4 /= np.linalg.norm(tangent_4)

    diff = float(casadi.norm_2(tangent_4 - t1) + casadi.norm_2(tangent_4 - t2))

    if np.isclose(diff, 0):
        u = np.arange(0, u1_opt, resolution_meters)
        output["C1"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [ui]) for ui in u
        ]
        output["S"] = [
            casadi.substitute(circle_1_func, casadi.vertcat(*[u1]), [u1_opt]),
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [u2_opt]),
        ]
        u = np.arange(u2_opt, 1, resolution_meters)
        output["C2"] = [
            casadi.substitute(circle_2_func, casadi.vertcat(*[u2]), [ui]) for ui in u
        ]

    return output


def getPathLength(path):
    """Calculate the total length of the path"""

    length = 0.0

    for key in path:
        # Convert to numpy if necessary
        if isinstance(path[key], list):
            path[key] = [np.array(casadi.DM(c)).flatten() for c in path[key]]
        else:
            path[key] = np.array(casadi.DM(path[key])).flatten()

        segment = path[key]
        if len(segment) < 2:
            continue

        for i in range(1, len(segment) - 1):
            p1 = segment[i - 1]
            p2 = segment[i]
            length += np.linalg.norm(np.array(p2) - np.array(p1))

    return length


def pathIsValid(path):
    """
    Check if the path is valid.
    A path is considered valid if it has at least one segment in C1, S, and C2.
    """
    if "C1" not in path or "S" not in path or "C2" not in path:
        return False

    if len(path["C1"]) == 0 or len(path["S"]) == 0 or len(path["C2"]) == 0:
        return False

    return True


def trimPath(path, goal_x, goal_y, distance_threshold=0.1):
    # Combine all points from C1, S, C2 into a single list
    full_path = []
    assert "C1" in path, "Paths must contain C1"
    assert "S" in path, "Paths must contain S"
    assert "C2" in path, "Paths must contain C2"

    full_path.extend(path["C1"])
    full_path.extend(path["S"])
    full_path.extend(path["C2"])

    full_length = len(full_path)

    # Now trim any parts that extend beyond the goal point
    trimmed_path = []
    for idx, point in enumerate(full_path):
        trimmed_path.append(point)
        # Check if the point is close to the goal
        if (
            np.linalg.norm(np.array(point) - np.array([goal_x, goal_y]))
            < distance_threshold
        ):
            print(f"Stopping at index {idx}/{full_length}")
            break

    return np.asarray(trimmed_path)


def getShortestDubbinsPath(
    radius, start_x, start_y, start_yaw, end_x, end_y, end_yaw, resolution_meters
):

    start = time.time()

    start_center = getArcCenters([start_x, start_y], radius, start_yaw)
    end_center = getArcCenters([end_x, end_y], radius, end_yaw)
    min_length = float("inf")
    shortest_path = None

    # RSR
    u1 = casadi.SX.sym("u1")
    u2 = casadi.SX.sym("u2")
    path_rsr = getTangents(
        start_center["R"],
        radius,
        start_yaw,
        -1,
        end_center["R"],
        radius,
        end_yaw,
        -1,
        resolution_meters=resolution_meters,
    )
    length = getPathLength(path_rsr)

    if length < min_length and pathIsValid(path_rsr):
        min_length = length
        shortest_path = path_rsr

    # LSR
    u1 = casadi.SX.sym("u1")
    u2 = casadi.SX.sym("u2")
    path_lsr = getTangents(
        start_center["L"],
        radius,
        start_yaw,
        1,
        end_center["R"],
        radius,
        end_yaw,
        -1,
        resolution_meters=resolution_meters,
    )
    length = getPathLength(path_lsr)

    if length < min_length and pathIsValid(path_lsr):
        min_length = length
        shortest_path = path_lsr

    # RSL
    u1 = casadi.SX.sym("u1")
    u2 = casadi.SX.sym("u2")
    path_rsl = getTangents(
        start_center["R"],
        radius,
        start_yaw,
        -1,
        end_center["L"],
        radius,
        end_yaw,
        1,
        resolution_meters=resolution_meters,
    )
    length = getPathLength(path_rsl)

    if length < min_length and pathIsValid(path_rsl):
        min_length = length
        shortest_path = path_rsl

    # LSL
    u1 = casadi.SX.sym("u1")
    u2 = casadi.SX.sym("u2")
    path_lsl = getTangents(
        start_center["L"],
        radius,
        start_yaw,
        1,
        end_center["L"],
        radius,
        end_yaw,
        1,
        resolution_meters=resolution_meters,
    )
    length = getPathLength(path_lsl)

    if length < min_length and pathIsValid(path_lsl):
        min_length = length
        shortest_path = path_lsl

    print(f"Found shortest path in {time.time() - start:.4f} seconds")

    return shortest_path, min_length


class WayPointController(Node):
    """Class to send waypoint signals to the motion controller"""

    def __init__(self):
        super().__init__("WayPointController")

        # Publisher
        qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.path_publisher = self.create_publisher(Path, "/planning/path", qos_profile)
        # Subscriber
        self.create_subscription(PoseStamped, "/goal_pose", self.goalPoseCb, 1)

        # For looking up the robot's position on the map
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Boolean needed
        self.mc_bool = True

        # Waypoint queue (example)
        self.waypoint_queue = deque(
            [
                [[-66.5, -338.1]],  # Planting Area 1
                # [(0, 0), (-2.75, -2), (-3.5, -4.25), (-4, -6.5)],  # Planting Area 2
                # [(0, 0), (1.0, 3.0), (4.0, 9.75), (10.0, 12.0)],  # planting Area 3
            ]
        )

        # Initial pose of robot
        self.init_robot_x = None
        self.init_robot_y = None

        self.latest_path = None

        self.goal_pose = (-1.2, -343.0)  # in m

    def transformPose(self, pose_msg: PoseStamped, target_frame: str):
        """Transform a PoseStamped message to a target frame
        Args:
            pose_msg (PoseStamped): The pose to transform
            target_frame (str): The target frame to transform to
        Returns:
            PoseStamped: The transformed pose
        """
        if pose_msg.header.frame_id == target_frame:
            return pose_msg

        try:
            # Get the latest transform from map to base_link
            t = self.tf_buffer.lookup_transform(
                target_frame, pose_msg.header.frame_id, rclpy.time.Time()
            )
            # Make shallow copy of pose_msg
            transformed_pose = PoseStamped()
            transformed_pose.header.stamp = pose_msg.header.stamp
            transformed_pose.header.frame_id = target_frame
            transformed_pose.pose = do_transform_pose(pose_msg.pose, t)
            return transformed_pose
        except TransformException as ex:
            self.get_logger().warning(f"Could not transform pose: {ex}")
            return None

    def forward_random_path(start, end, steps=10, noise_scale=0.1):
        """A helper function to generate random path from the start to end with random side deviations"""
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        path = [start]

        for i in range(1, steps - 1):
            t = i / steps  # progress from 0 to 1

            # Compute target point at this step on straight line
            target = start + t * (end - start)

            # Add noise around the target
            random_offset = np.random.randn(2) * noise_scale
            point = target + random_offset

            path.append(point)

        path.append(end)
        return np.array(path)

    # def goalPoseCB(self, msg: PoseStamped):
    #     """
    #     Construct a random part to the goal pose
    #     """
    #     try:
    #         # Get the latest transform from map to base_link
    #         t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())

    #         self.ego_x = t.transform.translation.x
    #         self.ego_y = t.transform.translation.y
    #         self.ego_yaw = R.from_quat(
    #             [
    #                 t.transform.rotation.x,
    #                 t.transform.rotation.y,
    #                 t.transform.rotation.z,
    #                 t.transform.rotation.w,
    #             ]
    #         ).as_euler("xyz")[2]

    #     except TransformException as ex:
    #         self.get_logger().warning(
    #             f"Could not find ego transform. Skipping path generation: {ex}"
    #         )
    #         return
    #     #Randomly selects points from the start to the end
    #     final_path = self.forward_random_path((self.ego_x, self.ego_y), self.goal_pose, 10)

    #     path_msg = Path()
    #     path_msg.header.stamp = self.get_clock().now().to_msg()
    #     path_msg.header.frame_id = "map"
    #     path_msg.poses = []
    #     for point in final_path:
    #         pose = PoseStamped()
    #         pose.header.stamp = self.get_clock().now().to_msg()
    #         pose.header.frame_id = "map"
    #         pose.pose.position.x = float(point[0])
    #         pose.pose.position.y = float(point[1])
    #         pose.pose.position.z = 0.0  # Assume 2D navigation
    #         path_msg.poses.append(pose)

    #     self.path_publisher.publish(path_msg)

    #     # Cache this for later
    #     self.latest_path = path_msg

    #     self.get_logger().info(f"Published path with {len(final_path)} points")

    def goalPoseCb(self, msg: PoseStamped):
        """Contruct a Dubbins path from the current ego pose to the goal pose

        Args:
            msg (PoseStamped): The goal pose
        """

        try:
            # Get the latest transform from map to base_link
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            # if self.init_robot_x is None or self.init_robot_y is None:
            #     # To help get the robot starting position as 0,0
            #     self.init_robot_x = t.transform.translation.x
            #     self.init_robot_y = t.transform.translation.y

            self.ego_x = t.transform.translation.x  # Try this or 0.0
            self.ego_y = t.transform.translation.y  # Loook right above
            self.ego_yaw = R.from_quat(
                [
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w,
                ]
            ).as_euler("xyz")[2]

        except TransformException as ex:
            self.get_logger().warning(
                f"Could not find ego transform. Skipping path generation: {ex}"
            )
            return

        # Transform goal pose to map frame if necessary
        msg = self.transformPose(msg, "map")
        if msg is None:
            self.get_logger().warning("Could not transform goal pose")
            return

        goal_yaw = R.from_quat(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        ).as_euler("xyz")[2]

        print(f"Ego Pose: ({self.ego_x}, {self.ego_y}), Yaw: {self.ego_yaw}")
        print(
            f"Goal Pose: ({msg.pose.position.x}, {msg.pose.position.y}), Yaw: {goal_yaw}"
        )

        assert msg.header.frame_id == "map", "Goal pose must be in the map frame"

        shortest_path, min_length = getShortestDubbinsPath(
            radius=2.0,
            start_x=self.ego_x,
            start_y=self.ego_y,
            start_yaw=self.ego_yaw,
            end_x=msg.pose.position.x,
            end_y=msg.pose.position.y,
            end_yaw=goal_yaw,
            resolution_meters=0.01,
        )

        final_path = trimPath(
            shortest_path, goal_x=-5, goal_y=-10, distance_threshold=0.1
        )

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        path_msg.poses = []
        for point in final_path:
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0  # Assume 2D navigation
            path_msg.poses.append(pose)

        self.path_publisher.publish(path_msg)

        # Cache this for later
        self.latest_path = path_msg

        self.get_logger().info(f"Published path with {len(final_path)} points")

    def mc_callback(self, msg):
        """Callback to check if the motion controller is ready for waypoints"""
        # self.mc_bool = msg.data
        self.get_logger().info("We trying to send the waypoints")
        # self.publishPath()

    def signal_callback(self, msg):
        """Get signal to send or not send waypoints"""
        pass
        if msg.data and self.mc_bool:
            # Send the waypoints
            self.publishPath()
            self.mc_bool = False

    def publishPath(self):
        """Send waypoints to the motion controller"""

        if self.latest_path is None:
            self.get_logger().warning("No path to publish")
            return
        self.path_publisher.publish(self.latest_path)


def main(args=None):
    rclpy.init(args=args)
    waypoints_controller_cls = WayPointController()
    rclpy.spin(waypoints_controller_cls)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
