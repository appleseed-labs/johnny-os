# -----------------------------------------------------------------------------
# Description: Generate twist commands using a Pure Pursuit controller
# Author: Rohan Walia
# (c) 2025 Appleseed Labs. CMU Robotics Institute
# -----------------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool
import math, time
import numpy as np
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose


class MotionController(Node):
    """Class to help with autonomous motion control using Pure Pursuit"""

    def __init__(self):
        super().__init__("motion_controller")

        # Publishers
        self.twist_publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.controller_signal_publisher = self.create_publisher(
            Bool, "/controller_signal", 10
        )

        # Subscribers
        self.create_subscription(Path, "/planning/path", self.pathCb, 10)

        # For looking up the robot's position on the map
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.waypoints = []

        # Current pose (initialize to None until received)
        self.ego_x = None
        self.ego_y = None
        self.ego_yaw = None

        # Counter to help us track the amount of lookahead_points not found
        self.lookahead_not_found_counter = 0
        # Initilize timer for control loop
        self.timer = self.create_timer(0.1, self.spinController)

        # NOTE: This is for testing purposes
        # self.controller_signal_publisher.publish(Bool(data=True))

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
            # Get the latest transform from map to robot_position
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

    def pathCb(self, msg: Path):
        """Callback to get waypoints from path planning"""
        poses = msg.poses
        # Get the needed x, y tuples for waypoints
        for pose_stamped in poses:
            if pose_stamped.header.frame_id != "robot_position":
                # Transform the pose to the ego frame
                pose_stamped = self.transformPose(pose_stamped, "robot_position")
                if pose_stamped is None:
                    continue
            pose = pose_stamped.pose
            loc_tuple = (pose.position.x, pose.position.y)
            self.waypoints.append(loc_tuple)
        # self.get_logger().info(f"Got waypoints: {self.waypoints}")

    def findLookaheadPoint(self, lookahead_distance):
        """Find a point on the path at lookahead distance ahead of the robot.

        Returns:
            Lookahead_point: Point that the robot should move towards
        """

        if self.ego_x is None or self.ego_y is None:
            self.get_logger().warning("No robot position data available.")
            return None

        # print(
        #     f"Searching along the path for lookahead point with distance: {lookahead_distance}"
        # )
        # print(self.waypoints)

        for i in range(len(self.waypoints) - 1):
            p1 = np.array(self.waypoints[i])
            p2 = np.array(self.waypoints[i + 1])

            # Vector between waypoints
            path_vector = p2 - p1
            path_length = np.linalg.norm(path_vector)  # Length of the path segment

            # Vector from robot to the first waypoint
            to_p1 = p1 - np.array([self.ego_x, self.ego_y])

            # Quadratic coefficients for solving for the lookahead distance
            a = np.dot(path_vector, path_vector)
            b = 2 * np.dot(path_vector, to_p1)
            c = np.dot(to_p1, to_p1) - np.square(lookahead_distance)

            # Solve the quadratic equation: at^2 + bt + c = 0
            t = np.roots([a, b, c])

            # Filter out complex roots and get only real values
            t_real = t[np.isreal(t)].real

            # If no real roots exist, continue to the next segment
            if len(t_real) == 0:
                continue

            # Choose the valid solution (if real and within the path segment)
            t = np.max(t_real)  # We want the largest solution in case of two roots

            if 0 <= t <= 1:  # Ensure the solution is within the path segment [0, 1]
                lookahead_point = p1 + t * path_vector  # Calculate the lookahead point
                return lookahead_point  # Return the lookahead point

        return None  # No valid lookahead point found

    def findCurvature(
        self, lookahead_point, robot_heading, robot_loc, lookahead_distance
    ):
        """Function used to find curvature to guide robot towards path

        Args:
            lookahead_point: Array of lookahead points (in [x, y])

        Returns:
            Curvature
        """
        # Unpack from the args
        x_pos = robot_loc[0]
        y_pos = robot_loc[1]

        lookahead_x = lookahead_point[0]
        lookahead_y = lookahead_point[1]

        a = -np.tan(
            robot_heading
        )  # The slope of the line at the current robot orientation
        b = 1  # The coefficient for the y-axis in the line equation
        c = x_pos * np.tan(robot_heading) - y_pos  # The line offset

        # Distance (N) from the robot's current position to the closest point on the path
        N = abs(a * lookahead_x + b * lookahead_y + c) / np.sqrt(
            np.square(a) + np.square(b)
        )

        # Determine the sign of the curvature (left or right turn)
        sign_pos = (
            lookahead_point - robot_loc
        )  # Vector from robot position to the pursuit point
        side = sign_pos[0] * np.sin(robot_heading) - sign_pos[1] * np.cos(robot_heading)
        side = np.sign(
            side
        )  # Sign is +1 or -1 depending on the direction (left or right)

        # The curvature is inversely proportional to the distance N and also depends on the path orientation
        curvature = 2 * N / np.square(lookahead_distance) * side * -1

        return curvature

    def stop(self):
        """Stop the robot and reset values"""
        # Stop if close to goal
        self.get_logger().info("Goal reached!")
        self.get_logger().info(f"Currently at: ({self.ego_x}, {self.ego_y})")
        self.reset()
        # Send signal to let waypoint node that we can accept more waypoints
        self.controller_signal_publisher.publish(Bool(data=True))

    def reset(self):
        """Resets key values and the robot"""
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.twist_publisher.publish(twist_msg)
        self.waypoints = []
        # self.timer.cancel()

    def findAngLinSpeeds(self, lookahead_point, remaining_distance, adaptive_lookahead):
        """Find the angular and linear speed

        Args:
            lookahead_point: Point closest to the curve we want to get to
            remaining_distance: The remaining distance to the end goal
            adaptive_lookahead: Lookahead Distance

        Returns:
            Angular and Linear speeds
        """
        # Compute curvature
        loc_cur = np.array([self.ego_x, self.ego_y])
        curvature = self.findCurvature(
            lookahead_point, self.ego_yaw, loc_cur, adaptive_lookahead
        )
        angular_speed = math.atan(curvature)

        # Compute linear speed using proportional controller
        # NOTE: Let's adjust lin. speed first
        linear_speed = min(0.5 * remaining_distance, 1.0)  # Cap max speed to be 1m/s
        if remaining_distance < 0.75:  # If within 0.75m of goal, slow down
            linear_speed = min(linear_speed, 0.4)

        return angular_speed, linear_speed

    def findAdaptiveLookahead(self):
        """Dynamically calculate lookahead distance based on the closest waypoint segment."""
        if self.ego_x is None or self.ego_y is None:
            return 1.5  # Default lookahead if no position data

        min_dist = float("inf")
        best_lookahead = 1.5  # Default lookahead

        for i in range(len(self.waypoints) - 1):
            # Get the current and next waypoint
            x1, y1 = self.waypoints[i]
            x2, y2 = self.waypoints[i + 1]

            # Project the robot's position onto the path segment
            segment_vec = np.array([x2 - x1, y2 - y1])
            robot_vec = np.array([self.ego_x - x1, self.ego_y - y1])
            segment_length = np.linalg.norm(segment_vec)

            if segment_length == 0:
                continue  # Skip if waypoints are identical

            # Projection formula to find t (parametric representation)
            t = np.dot(robot_vec, segment_vec) / (segment_length**2)
            t = max(0, min(1, t))  # Clamp t between 0 and 1

            # Closest point on the segment to the robot
            closest_x = x1 + t * (x2 - x1)
            closest_y = y1 + t * (y2 - y1)

            # Distance from robot to closest point
            distance = np.linalg.norm([self.ego_x - closest_x, self.ego_y - closest_y])

            # Adaptive lookahead logic
            if distance < min_dist:
                min_dist = distance
                # self.get_logger().info(f"Distance: {distance}\n")
                # self.get_logger().info(f"Formula: {max(0.1, min(2.0, distance))}\n")
                best_lookahead = max(0.01, min(1.5, distance))  # Adjust dynamically

        self.get_logger().info(f"Adaptive lookahead distance: {best_lookahead}")
        return best_lookahead

    def spinController(self):
        """Compute and publish velocity commands using Pure Pursuit."""
        # No waypoints recieved
        if len(self.waypoints) == 0:
            # self.get_logger().warning(
            #     "No waypoints received yet. Skipping control loop."
            # )
            return

        try:
            # Get the latest transform from map to base_link
            t = self.tf_buffer.lookup_transform(
                "robot_position", "map", rclpy.time.Time()
            )
            self.ego_x = t.transform.translation.x
            self.ego_y = t.transform.translation.y
            self.ego_yaw = R.from_quat(
                [
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w,
                ]
            ).as_euler("xyz")[2]
            # self.get_logger().info(f"Loc: {self.ego_x}, {self.ego_y}")
        except TransformException as ex:
            self.get_logger().warning(
                f"Could not find ego transform. Skipping control loop: {ex}"
            )
            return

        # Calculate the remaining distance
        remaining_distance = math.sqrt(
            (self.waypoints[-1][0] - self.ego_x) ** 2
            + (self.waypoints[-1][1] - self.ego_y) ** 2
        )

        # Stop if close to goal

        # TODO: Parameterize this distance threshold
        if remaining_distance < 0.25:
            self.get_logger().info("We are close to the waypoint!")
            # Stop the robot
            self.stop()
            return

        # Get the lookhead distance adaptively
        adaptive_lookahead_distance = self.findAdaptiveLookahead()
        if adaptive_lookahead_distance is None:
            self.get_logger().warning("Couldn't get a adaptive lookahead distance!")
            return

        # Find lookahead point
        lookahead_point = self.findLookaheadPoint(adaptive_lookahead_distance)
        # No lookahead point so stop
        if lookahead_point is None:
            self.get_logger().warn("No valid lookahead point found!")
            self.lookahead_not_found_counter += 1

            # If we have seen the no valid lookahead point then lets stop
            if self.lookahead_not_found_counter == 20:
                # Stop the robot
                self.stop()
                self.lookahead_not_found_counter = 0
            return

        # Compute angular and linear speed
        angular_speed, linear_speed = self.findAngLinSpeeds(
            lookahead_point, remaining_distance, adaptive_lookahead_distance
        )

        # Publish Twist command
        twist_msg = Twist()
        twist_msg.linear.x = linear_speed

        # TODO: Parameterize this speed limit
        twist_msg.angular.z = max(min(angular_speed, 1.5), -1.5)

        # self.get_logger().info(f'Lin speed and ang speed: ({linear_speed}, {angular_speed})')

        self.twist_publisher.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    motion_controller_cls = MotionController()
    rclpy.spin(motion_controller_cls)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
